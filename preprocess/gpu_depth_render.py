"""GPU-based depth map rendering from point cloud using PyTorch.

Projects all points onto each camera and z-buffers to get per-pixel depth.
No mesh reconstruction needed. Handles 100M+ points on GPU.

Usage:
  python preprocess/gpu_depth_render.py \
    --colmap_dir .../aligned/sparse/0 \
    --pointcloud /path/to/colorized.ply \
    --output_dir /path/to/output \
    --gpu 0 --splat_radius 3
"""
import argparse, os, sys, time, cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from read_write_model import read_model, qvec2rotmat


def load_pointcloud_gpu(ply_path, device):
    """Load PLY to GPU tensor."""
    from plyfile import PlyData
    print(f"Loading {ply_path}...")
    t0 = time.time()
    ply = PlyData.read(ply_path)
    v = ply['vertex']
    xyz = np.vstack([v['x'], v['y'], v['z']]).T.astype(np.float32)
    print(f"  {len(xyz):,} points loaded in {time.time()-t0:.1f}s")

    print(f"  Uploading to GPU...")
    xyz_gpu = torch.from_numpy(xyz).to(device)
    print(f"  GPU memory: {xyz_gpu.element_size() * xyz_gpu.nelement() / 1e9:.1f}GB")
    return xyz_gpu


@torch.no_grad()
def render_depth_gpu(xyz_gpu, R, t_vec, fx, fy, cx, cy, W, H, splat_radius=2, max_depth=100.0, chunk_size=10_000_000):
    """Render depth map by projecting points and z-buffering on GPU.

    Processes points in chunks to avoid OOM on large point clouds.
    """
    device = xyz_gpu.device
    N = xyz_gpu.shape[0]

    R_t = torch.from_numpy(R.astype(np.float32)).to(device)
    t_t = torch.from_numpy(t_vec.astype(np.float32)).to(device)

    depth_buf = torch.full((H * W,), float('inf'), device=device, dtype=torch.float32)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        pts = xyz_gpu[start:end]

        p_cam = pts @ R_t.T + t_t.unsqueeze(0)
        z = p_cam[:, 2]
        valid = (z > 0.01) & (z < max_depth)
        p_cam = p_cam[valid]
        z = z[valid]

        if len(z) == 0:
            continue

        u = (fx * p_cam[:, 0] / z + cx).long()
        v = (fy * p_cam[:, 1] / z + cy).long()

        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u = u[in_bounds]
        v = v[in_bounds]
        z = z[in_bounds]

        if len(z) == 0:
            continue

        if splat_radius <= 1:
            idx = v * W + u
            depth_buf.scatter_reduce_(0, idx, z, reduce='amin', include_self=True)
        else:
            for dy in range(-splat_radius + 1, splat_radius):
                for dx in range(-splat_radius + 1, splat_radius):
                    if dx * dx + dy * dy >= splat_radius * splat_radius:
                        continue
                    vv = v + dy
                    uu = u + dx
                    mask = (vv >= 0) & (vv < H) & (uu >= 0) & (uu < W)
                    idx = vv[mask] * W + uu[mask]
                    depth_buf.scatter_reduce_(0, idx, z[mask], reduce='amin', include_self=True)

        del p_cam, z, u, v, pts
        torch.cuda.empty_cache()

    depth_np = depth_buf.reshape(H, W).cpu().numpy()
    depth_np[depth_np == float('inf')] = 0
    return depth_np


def save_depth_16bit(path, depth, as_inverse=True):
    """Save depth as 16-bit PNG (inverse depth normalized, compatible with Depth Anything format)."""
    valid = depth > 0
    if not valid.any():
        cv2.imwrite(path, np.zeros_like(depth, dtype=np.uint16))
        return

    if as_inverse:
        inv = np.zeros_like(depth)
        inv[valid] = 1.0 / depth[valid]
        d_min, d_max = inv[valid].min(), inv[valid].max()
        if d_max - d_min > 1e-6:
            norm = (inv - d_min) / (d_max - d_min)
        else:
            norm = np.zeros_like(inv)
    else:
        d_min, d_max = depth[valid].min(), depth[valid].max()
        if d_max - d_min > 1e-6:
            norm = (depth - d_min) / (d_max - d_min)
        else:
            norm = np.zeros_like(depth)

    norm = np.clip(norm, 0, 1)
    cv2.imwrite(path, (norm * 65535).astype(np.uint16))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--colmap_dir', required=True)
    parser.add_argument('--pointcloud', required=True)
    parser.add_argument('--images_dir', default='', help='For preview')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--splat_radius', type=int, default=3,
                        help='Pixel radius for point splatting (fills gaps)')
    parser.add_argument('--max_depth', type=float, default=100.0)
    parser.add_argument('--max_dim', type=int, default=0, help='Max render dimension (0=full)')
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--num_preview', type=int, default=5)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    xyz_gpu = load_pointcloud_gpu(args.pointcloud, device)

    cams, imgs, _ = read_model(args.colmap_dir, ext='.bin')
    print(f"{len(imgs)} cameras")

    depth_dir = os.path.join(args.output_dir, 'depths')
    preview_dir = os.path.join(args.output_dir, 'preview')
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)

    img_keys = sorted(imgs.keys())
    preview_step = max(1, len(img_keys) // args.num_preview)
    preview_set = set(img_keys[::preview_step][:args.num_preview])

    t0 = time.time()
    for idx, img_id in enumerate(img_keys):
        im = imgs[img_id]
        cam = cams[im.camera_id]
        R = qvec2rotmat(im.qvec)
        t_vec = im.tvec

        if cam.model == 'PINHOLE':
            fx, fy, cx, cy = cam.params
        else:
            f = cam.params[0]
            fx = fy = f
            cx, cy = cam.params[1], cam.params[2]

        W, H = cam.width, cam.height
        scale = 1.0
        if args.max_dim > 0 and max(W, H) > args.max_dim:
            scale = args.max_dim / max(W, H)

        rW, rH = int(W * scale), int(H * scale)
        rfx, rfy, rcx, rcy = fx * scale, fy * scale, cx * scale, cy * scale

        stem = os.path.splitext(im.name)[0]
        out_path = os.path.join(depth_dir, f"{stem}.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if args.skip_existing and os.path.exists(out_path):
            continue

        depth = render_depth_gpu(xyz_gpu, R, t_vec, rfx, rfy, rcx, rcy, rW, rH,
                                  splat_radius=args.splat_radius, max_depth=args.max_depth)
        save_depth_16bit(out_path, depth)

        # Preview
        if img_id in preview_set and args.images_dir:
            img_path = os.path.join(args.images_dir, im.name)
            image = cv2.imread(img_path)
            if image is not None:
                if scale != 1.0:
                    image = cv2.resize(image, (rW, rH))
                valid = depth > 0
                if valid.any():
                    d_norm = np.clip((depth - depth[valid].min()) /
                                    (depth[valid].max() - depth[valid].min() + 1e-6), 0, 1)
                    depth_color = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                    depth_color[~valid] = 0
                    blend = cv2.addWeighted(image, 0.4, depth_color, 0.6, 0)
                    cv2.imwrite(os.path.join(preview_dir, f"depth_{os.path.basename(stem)}.jpg"), blend)
                    coverage = valid.sum() / valid.size * 100
                    print(f"    Preview saved, coverage: {coverage:.1f}%")

        if (idx + 1) % 100 == 0 or idx < 3:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(img_keys) - idx - 1) / rate / 60
            print(f"  [{idx+1}/{len(img_keys)}] {rate:.1f} img/s, ETA {eta:.0f}min")

    elapsed = time.time() - t0
    print(f"\nDone! {len(img_keys)} depth maps in {elapsed/60:.1f}min ({len(img_keys)/elapsed:.1f} img/s)")


if __name__ == '__main__':
    main()

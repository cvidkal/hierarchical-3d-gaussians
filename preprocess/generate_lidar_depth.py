"""
Project a full LiDAR point-cloud map onto each camera image.
All 300M+ points stay on GPU. Per image: distance filter + projection on GPU,
occlusion on CPU. Eliminates CPU->GPU transfer bottleneck.
"""

import argparse
import numpy as np
from pathlib import Path
from plyfile import PlyData
from scipy.ndimage import minimum_filter
import time
import torch

def parse_intrinsic_txt(filepath):
    lines = Path(filepath).read_text().strip().splitlines()
    row0 = lines[0].split()
    row1 = lines[1].split()
    return float(row0[0]), float(row1[1]), float(row0[2]), float(row1[2])

def parse_intrinsic_from_opt(filepath):
    """Extract fx, fy, cx, cy (in pixels) from .opt XML when no intrinsic txt exists."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(filepath)
    root = tree.getroot()
    dims = root.find("ImageDimensions")
    W = int(dims.find("Width").text)
    sensor = float(root.find("SensorSize").text)
    focal_mm = float(root.find("FocalLength").text)
    pp = root.find("PrincipalPoint")
    cx = float(pp.find("X").text)
    cy = float(pp.find("Y").text)
    fx = focal_mm / sensor * W
    fy = fx  # AspectRatio = 1
    return fx, fy, cx, cy

def parse_imgpose(filepath):
    poses = []
    with open(filepath) as f:
        f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) < 12:
                continue
            poses.append({
                "name": parts[0],
                "x": float(parts[1]), "y": float(parts[2]), "z": float(parts[3]),
                "qx": float(parts[7]), "qy": float(parts[8]),
                "qz": float(parts[9]), "qw": float(parts[10]),
            })
    return poses

def quat_to_rotation_matrix(qw, qx, qy, qz):
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ], dtype=np.float32)

def parse_opt(filepath):
    import xml.etree.ElementTree as ET
    tree = ET.parse(filepath)
    root = tree.getroot()
    dims = root.find("ImageDimensions")
    return int(dims.find("Width").text), int(dims.find("Height").text)

@torch.no_grad()
def project_one_image(pts_gpu, cam_pos_gpu, R_gpu, t_gpu,
                      fx, fy, cx, cy, W, H, max_depth,
                      pool_k=11, margin_ratio=0.02, occ_scale=4):
    """All heavy compute on GPU, only small result to CPU."""

    # Distance filter on GPU (squared distance, no sqrt needed)
    diff = pts_gpu - cam_pos_gpu
    dist_sq = (diff * diff).sum(dim=1)
    nearby_mask = dist_sq < (max_depth * max_depth)
    nearby = pts_gpu[nearby_mask]

    if nearby.shape[0] == 0:
        return np.array([]), np.array([]), np.array([]), 0

    n_nearby = nearby.shape[0]

    # Transform to camera space on GPU
    pts_cam = nearby @ R_gpu.T + t_gpu

    # Filter z > 0.1
    z = pts_cam[:, 2]
    front = z > 0.1
    pts_cam = pts_cam[front]
    z = pts_cam[:, 2]

    if z.shape[0] == 0:
        return np.array([]), np.array([]), np.array([]), n_nearby

    # Project
    u = fx * pts_cam[:, 0] / z + cx
    v = fy * pts_cam[:, 1] / z + cy

    # In-frame filter
    in_frame = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[in_frame]
    v = v[in_frame]
    d = z[in_frame]

    if d.shape[0] == 0:
        return np.array([]), np.array([]), np.array([]), n_nearby

    # Transfer only in-frame points to CPU (much smaller than input)
    u_np = u.cpu().numpy()
    v_np = v.cpu().numpy()
    d_np = d.cpu().numpy()

    # Occlusion on CPU
    s = occ_scale
    h_lo, w_lo = H // s, W // s
    u_lo = np.clip((u_np / s).astype(np.int32), 0, w_lo - 1)
    v_lo = np.clip((v_np / s).astype(np.int32), 0, h_lo - 1)

    order = np.argsort(-d_np)
    zbuf = np.full((h_lo, w_lo), np.inf, dtype=np.float32)
    zbuf[v_lo[order], u_lo[order]] = d_np[order]
    near_surface = minimum_filter(zbuf, size=pool_k, mode='constant', cval=np.inf)

    margin = margin_ratio * d_np
    ns_at_pt = near_surface[v_lo, u_lo]
    visible = (d_np <= (ns_at_pt + margin)) & (ns_at_pt < np.inf)

    return u_np[visible], v_np[visible], d_np[visible], n_nearby


def save_visualisation(image_path, u, v, depth, out_path, max_side=1600):
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        return
    h_img, w_img = img.shape[:2]
    scale = min(max_side / max(h_img, w_img), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w_img * scale), int(h_img * scale)))
        u_s = (u * scale).astype(np.int32)
        v_s = (v * scale).astype(np.int32)
    else:
        u_s, v_s = np.round(u).astype(np.int32), np.round(v).astype(np.int32)
    if len(depth) == 0:
        return
    d_min, d_max = np.percentile(depth, [2, 98])
    norm = np.clip((depth - d_min) / (d_max - d_min + 1e-6), 0, 1)
    colors = (norm * 255).astype(np.uint8)
    cmap = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(1, -1), cv2.COLORMAP_TURBO)[0]
    h_out, w_out = img.shape[:2]
    keep = (u_s >= 0) & (u_s < w_out) & (v_s >= 0) & (v_s < h_out)
    u_s, v_s, colors = u_s[keep], v_s[keep], colors[keep]
    for x, y, c_idx in zip(u_s, v_s, colors):
        cv2.circle(img, (x, y), 2, tuple(int(vv) for vv in cmap[c_idx]), -1)
    cv2.imwrite(str(out_path), img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--undistort_dir", required=True)
    parser.add_argument("--pointcloud", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pool_k", type=int, default=11)
    parser.add_argument("--margin_ratio", type=float, default=0.02)
    parser.add_argument("--occ_scale", type=int, default=4)
    parser.add_argument("--max_depth", type=float, default=30.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--visualise", action="store_true")
    parser.add_argument("--vis_count", type=int, default=10)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    undistort_dir = Path(args.undistort_dir)
    output_dir = Path(args.output_dir)

    # Intrinsics (try .txt first, fall back to .opt XML)
    left_intrinsic_txt = undistort_dir / "left_undistort_intrinsic.txt"
    right_intrinsic_txt = undistort_dir / "right_undistort_intrinsic.txt"
    if left_intrinsic_txt.exists():
        left_fx, left_fy, left_cx, left_cy = parse_intrinsic_txt(left_intrinsic_txt)
    else:
        left_fx, left_fy, left_cx, left_cy = parse_intrinsic_from_opt(undistort_dir / "Left_undistort.opt")
    if right_intrinsic_txt.exists():
        right_fx, right_fy, right_cx, right_cy = parse_intrinsic_txt(right_intrinsic_txt)
    else:
        right_fx, right_fy, right_cx, right_cy = parse_intrinsic_from_opt(undistort_dir / "Right_undistort.opt")
    left_w, left_h = parse_opt(undistort_dir / "Left_undistort.opt")
    right_w, right_h = parse_opt(undistort_dir / "Right_undistort.opt")
    intrinsics = {
        "left": (left_fx, left_fy, left_cx, left_cy, left_w, left_h),
        "right": (right_fx, right_fy, right_cx, right_cy, right_w, right_h),
    }

    poses = parse_imgpose(undistort_dir / "ImgPose.txt")
    print(f"Loaded {len(poses)} image poses")

    # Load point cloud to GPU
    print(f"Loading point cloud: {args.pointcloud}")
    t0 = time.time()
    plydata = PlyData.read(args.pointcloud)
    verts = plydata['vertex']
    pts_np = np.vstack([verts['x'], verts['y'], verts['z']]).T.astype(np.float32)
    del plydata, verts
    print(f"  {pts_np.shape[0]:,} points loaded in {time.time()-t0:.1f}s")

    print(f"  Uploading {pts_np.nbytes/1e9:.1f}GB to GPU...")
    t0 = time.time()
    pts_gpu = torch.from_numpy(pts_np).to(device)
    del pts_np
    print(f"  Uploaded in {time.time()-t0:.1f}s")
    print(f"  GPU memory: {torch.cuda.memory_allocated(device)/1e9:.1f}GB")

    # Output dirs
    depths_dir = output_dir / "lidar_depths"
    depths_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = vis_set = None
    if args.visualise:
        vis_dir = output_dir / "depths_vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        vis_set = set(np.linspace(0, len(poses)-1, args.vis_count, dtype=int))

    # Filter done
    poses_todo = []
    skipped = 0
    for i, pose in enumerate(poses):
        side = "left" if pose["name"].startswith("left/") else "right"
        stem = Path(pose["name"]).stem
        if args.skip_existing and (depths_dir / side / f"{stem}.npz").exists():
            skipped += 1
        else:
            poses_todo.append((i, pose))

    if skipped:
        print(f"  Skipping {skipped} already-processed images")
    print(f"  Processing {len(poses_todo)} images (max_depth={args.max_depth}m)")

    t_start = time.time()
    processed = 0

    for i_global, pose in poses_todo:
        name = pose["name"]
        side = "left" if name.startswith("left/") else "right"
        fx, fy, cx, cy, W, H = intrinsics[side]
        stem = Path(name).stem
        side_dir = depths_dir / side
        side_dir.mkdir(parents=True, exist_ok=True)
        out_file = side_dir / f"{stem}.npz"

        cam_pos = np.array([pose["x"], pose["y"], pose["z"]], dtype=np.float32)
        R_w2c = quat_to_rotation_matrix(pose["qw"], -pose["qx"], -pose["qy"], -pose["qz"])
        t_w2c = -R_w2c @ cam_pos

        cam_pos_gpu = torch.from_numpy(cam_pos).to(device)
        R_gpu = torch.from_numpy(R_w2c).to(device)
        t_gpu = torch.from_numpy(t_w2c).to(device)

        u, v, d, n_nearby = project_one_image(
            pts_gpu, cam_pos_gpu, R_gpu, t_gpu,
            fx, fy, cx, cy, W, H, args.max_depth,
            pool_k=args.pool_k, margin_ratio=args.margin_ratio,
            occ_scale=args.occ_scale)

        np.savez_compressed(out_file, u=u, v=v, depth=d)
        processed += 1

        if processed % 50 == 0 or processed <= 5:
            elapsed = time.time() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            coverage = len(d) / (W * H) * 100 if len(d) > 0 else 0
            eta_min = (len(poses_todo) - processed) / rate / 60 if rate > 0 else 0
            print(f"  [{i_global+1}/{len(poses)}] {name}: {len(d):,} pts ({coverage:.1f}%) "
                  f"nearby={n_nearby:,} [{rate:.1f} img/s, ETA {eta_min:.0f}min]")

        if vis_dir and i_global in vis_set:
            save_visualisation(undistort_dir / name, u, v, d, vis_dir / f"{side}_{stem}.png")

    total = time.time() - t_start
    print(f"\nDone! {processed} processed, {skipped} skipped in {total/60:.1f}min")


if __name__ == "__main__":
    main()

"""Verify COLMAP camera parameters by projecting 3D points onto images.

Renders point cloud projections overlaid on actual images to visually check alignment.
"""
import argparse, cv2, os, sys, numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from read_write_model import read_model, qvec2rotmat


def project_points(pts_xyz, pts_rgb, R, t, fx, fy, cx, cy, W, H):
    """Project 3D points to image plane, return u, v, depth, colors for valid points."""
    p_cam = (R @ pts_xyz.T).T + t  # Nx3
    mask = p_cam[:, 2] > 0.1
    u = fx * p_cam[:, 0] / p_cam[:, 2] + cx
    v = fy * p_cam[:, 1] / p_cam[:, 2] + cy
    mask &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u[mask].astype(int), v[mask].astype(int), p_cam[mask, 2], pts_rgb[mask]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--colmap_dir', required=True, help='sparse/0 dir')
    parser.add_argument('--images_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_images', type=int, default=5)
    parser.add_argument('--max_points', type=int, default=200000)
    parser.add_argument('--external_ply', default='', help='Optional external PLY to project')
    args = parser.parse_args()

    cams, imgs, pts = read_model(args.colmap_dir, ext='.bin')

    # Gather 3D points
    if args.external_ply:
        from plyfile import PlyData
        ply = PlyData.read(args.external_ply)
        v = ply['vertex']
        xyz = np.vstack([v['x'], v['y'], v['z']]).T.astype(np.float64)
        if 'red' in v:
            rgb = np.vstack([v['red'], v['green'], v['blue']]).T.astype(np.uint8)
        else:
            rgb = np.full((len(xyz), 3), 180, dtype=np.uint8)
        print(f"External PLY: {len(xyz)} points")
    else:
        xyz = np.array([pts[k].xyz for k in pts])
        rgb = np.array([pts[k].rgb for k in pts]).astype(np.uint8)
        print(f"COLMAP points: {len(xyz)}")

    if len(xyz) > args.max_points:
        idx = np.random.choice(len(xyz), args.max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]

    os.makedirs(args.output_dir, exist_ok=True)

    img_keys = sorted(imgs.keys())
    step = max(1, len(img_keys) // args.num_images)
    selected = img_keys[::step][:args.num_images]

    for img_id in selected:
        im = imgs[img_id]
        cam = cams[im.camera_id]
        R = qvec2rotmat(im.qvec)
        t = im.tvec

        if cam.model == 'PINHOLE':
            fx, fy, cx, cy = cam.params
        elif cam.model == 'SIMPLE_PINHOLE':
            f, cx, cy = cam.params
            fx = fy = f
        W, H = cam.width, cam.height

        # Load image
        img_path = os.path.join(args.images_dir, im.name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"  Skip {im.name}: not found")
            continue

        # Resize for display if too large
        scale = 1.0
        if max(W, H) > 2000:
            scale = 2000.0 / max(W, H)
            image = cv2.resize(image, (int(W * scale), int(H * scale)))

        u, v, depth, colors = project_points(xyz, rgb, R, t, fx, fy, cx, cy, W, H)

        # Scale projected coords
        u_s = (u * scale).astype(int)
        v_s = (v * scale).astype(int)

        # Sort by depth (far first) so close points draw on top
        order = np.argsort(-depth)
        u_s, v_s, colors = u_s[order], v_s[order], colors[order]

        # Draw points
        for i in range(len(u_s)):
            cv2.circle(image, (u_s[i], v_s[i]), 2, (int(colors[i, 2]), int(colors[i, 1]), int(colors[i, 0])), -1)

        out_path = os.path.join(args.output_dir, f"proj_{Path(im.name).stem}.jpg")
        cv2.imwrite(out_path, image)
        print(f"  {im.name}: {len(u)} projected points, saved to {out_path}")

    print("Done!")


if __name__ == '__main__':
    main()

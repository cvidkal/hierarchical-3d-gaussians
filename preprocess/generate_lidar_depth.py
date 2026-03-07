"""
Project a full LiDAR point-cloud map onto each camera image to produce
sparse-but-accurate metric depth maps for 3DGS training supervision.

Occlusion handling (done at reduced resolution for speed):
  1. Project all points -> per-pixel z-buffer (keep nearest)
  2. scipy minimum_filter to fill surface gaps -> "near surface depth"
  3. Reject any point whose depth >> near surface depth (occluded)

Outputs per image:
  - lidar_depths/<side>/<stem>.npz : sparse depth (u, v, depth arrays)
  - depths_vis/<name>.png          : (optional) visualisation overlay
"""

import argparse
import numpy as np
from pathlib import Path
from plyfile import PlyData
from scipy.ndimage import minimum_filter
import time

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_intrinsic_txt(filepath):
    lines = Path(filepath).read_text().strip().splitlines()
    row0 = lines[0].split()
    row1 = lines[1].split()
    return float(row0[0]), float(row1[1]), float(row0[2]), float(row1[2])


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
    ], dtype=np.float64)


def parse_opt(filepath):
    import xml.etree.ElementTree as ET
    tree = ET.parse(filepath)
    root = tree.getroot()
    dims = root.find("ImageDimensions")
    return int(dims.find("Width").text), int(dims.find("Height").text)

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def project_and_filter(pts_world, R_w2c, t_w2c, fx, fy, cx, cy, W, H,
                       pool_k=11, margin_ratio=0.02, occ_scale=4, min_depth=0.1):
    """
    Project points, do occlusion filtering, return visible (u, v, depth).

    Occlusion is computed on a (H/occ_scale x W/occ_scale) grid for speed.
    Final u, v are in full-resolution pixel coordinates.
    """
    # --- project all points ---
    pts_cam = (R_w2c @ pts_world.T).T + t_w2c  # (N,3)
    z = pts_cam[:, 2].astype(np.float32)
    valid = z > min_depth

    u_full = (fx * pts_cam[:, 0] / pts_cam[:, 2] + cx).astype(np.float32)
    v_full = (fy * pts_cam[:, 1] / pts_cam[:, 2] + cy).astype(np.float32)
    valid &= (u_full >= 0) & (u_full < W) & (v_full >= 0) & (v_full < H)

    u_f = u_full[valid]
    v_f = v_full[valid]
    d = z[valid]

    if len(d) == 0:
        return np.array([]), np.array([]), np.array([])

    # --- occlusion on low-res grid ---
    s = occ_scale
    h_lo, w_lo = H // s, W // s
    u_lo = np.clip((u_f / s).astype(np.int32), 0, w_lo - 1)
    v_lo = np.clip((v_f / s).astype(np.int32), 0, h_lo - 1)

    # z-buffer: sort far-to-near, write (nearer overwrites)
    order = np.argsort(-d)
    zbuf = np.full((h_lo, w_lo), np.inf, dtype=np.float32)
    zbuf[v_lo[order], u_lo[order]] = d[order]

    # min-filter to fill gaps
    near_surface = minimum_filter(zbuf, size=pool_k, mode='constant', cval=np.inf)

    # occlusion test
    margin = margin_ratio * d
    ns_at_pt = near_surface[v_lo, u_lo]
    visible = d <= (ns_at_pt + margin)

    # also drop points where near_surface is inf (no data in neighbourhood)
    visible &= ns_at_pt < np.inf

    return u_f[visible], v_f[visible], d[visible]

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

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
        u_s = np.round(u).astype(np.int32)
        v_s = np.round(v).astype(np.int32)

    if len(depth) == 0:
        return

    d_min, d_max = np.percentile(depth, [2, 98])
    norm = np.clip((depth - d_min) / (d_max - d_min + 1e-6), 0, 1)
    colors = (norm * 255).astype(np.uint8)

    # Build colour map lookup
    cmap = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(1, -1), cv2.COLORMAP_TURBO)[0]

    h_out, w_out = img.shape[:2]
    keep = (u_s >= 0) & (u_s < w_out) & (v_s >= 0) & (v_s < h_out)
    u_s, v_s, colors = u_s[keep], v_s[keep], colors[keep]

    for x, y, c_idx in zip(u_s, v_s, colors):
        color = tuple(int(v) for v in cmap[c_idx])
        cv2.circle(img, (x, y), 2, color, -1)

    cv2.imwrite(str(out_path), img)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate LiDAR depth maps for GS training")
    parser.add_argument("--undistort_dir", required=True)
    parser.add_argument("--pointcloud", required=True,
                        help="Full map PLY (e.g. colorized_ds5cm.ply)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pool_k", type=int, default=11,
                        help="Min-filter kernel for occlusion (in low-res pixels)")
    parser.add_argument("--margin_ratio", type=float, default=0.02)
    parser.add_argument("--occ_scale", type=int, default=4,
                        help="Downscale factor for occlusion grid")
    parser.add_argument("--visualise", action="store_true")
    parser.add_argument("--vis_count", type=int, default=10)
    args = parser.parse_args()

    undistort_dir = Path(args.undistort_dir)
    output_dir = Path(args.output_dir)

    # --- intrinsics & dimensions ---
    left_fx, left_fy, left_cx, left_cy = parse_intrinsic_txt(
        undistort_dir / "left_undistort_intrinsic.txt")
    right_fx, right_fy, right_cx, right_cy = parse_intrinsic_txt(
        undistort_dir / "right_undistort_intrinsic.txt")
    left_w, left_h = parse_opt(undistort_dir / "Left_undistort.opt")
    right_w, right_h = parse_opt(undistort_dir / "Right_undistort.opt")

    intrinsics = {
        "left": (left_fx, left_fy, left_cx, left_cy, left_w, left_h),
        "right": (right_fx, right_fy, right_cx, right_cy, right_w, right_h),
    }

    # --- poses ---
    poses = parse_imgpose(undistort_dir / "ImgPose.txt")
    print(f"Loaded {len(poses)} image poses")

    # --- point cloud ---
    print(f"Loading point cloud: {args.pointcloud}")
    t0 = time.time()
    plydata = PlyData.read(args.pointcloud)
    verts = plydata['vertex']
    pts_world = np.vstack([verts['x'], verts['y'], verts['z']]).T.astype(np.float32)
    print(f"  {pts_world.shape[0]:,} points loaded in {time.time()-t0:.1f}s")

    # --- output dirs ---
    depths_dir = output_dir / "lidar_depths"
    depths_dir.mkdir(parents=True, exist_ok=True)

    if args.visualise:
        vis_dir = output_dir / "depths_vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        vis_indices = set(np.linspace(0, len(poses)-1, args.vis_count, dtype=int))

    t_start = time.time()

    for i, pose in enumerate(poses):
        name = pose["name"]
        side = "left" if name.startswith("left/") else "right"
        fx, fy, cx, cy, W, H = intrinsics[side]

        # c2w -> w2c
        R_w2c = quat_to_rotation_matrix(pose["qw"], -pose["qx"], -pose["qy"], -pose["qz"])
        cam_pos = np.array([pose["x"], pose["y"], pose["z"]])
        t_w2c = -R_w2c @ cam_pos

        u, v, d = project_and_filter(
            pts_world, R_w2c, t_w2c, fx, fy, cx, cy, W, H,
            pool_k=args.pool_k, margin_ratio=args.margin_ratio,
            occ_scale=args.occ_scale)

        # save sparse depth as compressed npz
        stem = Path(name).stem
        side_dir = depths_dir / side
        side_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(side_dir / f"{stem}.npz", u=u, v=v, depth=d)

        if i % 200 == 0 or i == len(poses) - 1:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            coverage = len(d) / (W * H) * 100 if len(d) > 0 else 0
            print(f"  [{i+1}/{len(poses)}] {name}: {len(d):,} pts ({coverage:.1f}%) "
                  f"[{rate:.1f} img/s]")

        if args.visualise and i in vis_indices:
            img_path = undistort_dir / name
            vis_out = vis_dir / f"{side}_{stem}.png"
            save_visualisation(img_path, u, v, d, vis_out)
            print(f"    -> vis: {vis_out}")

    print(f"\nDone! {len(poses)} images processed in {time.time()-t_start:.0f}s")
    print(f"Output: {depths_dir}")


if __name__ == "__main__":
    main()

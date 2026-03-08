"""Generate depth maps from LiDAR point cloud via mesh reconstruction + raycasting.

Steps:
  1. Load LiDAR point cloud (in aligned COLMAP coordinates)
  2. Reconstruct mesh using Poisson surface reconstruction
  3. For each camera, raycast to get clean depth maps without point cloud gaps

Usage:
  python preprocess/mesh_depth.py \
    --colmap_dir .../aligned/sparse/0 \
    --pointcloud /path/to/aligned_points.ply \
    --images_dir .../rectified/images \
    --output_dir /path/to/mesh_depth \
    --poisson_depth 10 \
    --num_preview 5
"""
import argparse, os, sys, time
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(__file__))
from read_write_model import read_model, qvec2rotmat


def build_mesh(pcd_path, poisson_depth=10, voxel_size=0.05):
    """Load point cloud, estimate normals, run Poisson reconstruction."""
    print(f"Loading point cloud: {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"  {len(pcd.points):,} points")

    if voxel_size > 0:
        print(f"  Voxel downsampling ({voxel_size}m)...")
        pcd = pcd.voxel_down_sample(voxel_size)
        print(f"  {len(pcd.points):,} points after downsample")

    print("  Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)

    print(f"  Poisson reconstruction (depth={poisson_depth})...")
    t0 = time.time()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, n_threads=-1)
    print(f"  Mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles ({time.time()-t0:.1f}s)")

    # Remove low-density vertices (artifacts at boundaries)
    densities = np.asarray(densities)
    threshold = np.quantile(densities, 0.05)
    vertices_to_remove = densities < threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(f"  After density filter: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    return mesh


def render_depth_raycast(mesh, R, t, fx, fy, cx, cy, W, H):
    """Render depth map using Open3D raycasting."""
    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh_t)

    # Build rays for each pixel
    # Camera looks down -z in camera frame, but raycasting needs world-space rays
    # For pixel (u,v): ray_dir_cam = [(u-cx)/fx, (v-cy)/fy, 1]
    # ray_dir_world = R^T @ ray_dir_cam
    # ray_origin = -R^T @ t (camera center in world)

    cam_center = -R.T @ t  # world coords

    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    dirs_cam = np.stack([
        (uu - cx) / fx,
        (vv - cy) / fy,
        np.ones_like(uu)
    ], axis=-1)  # HxWx3

    # Transform to world frame
    dirs_world = dirs_cam @ R  # HxWx3 (R^T applied via right multiply since dirs_cam is row vectors)

    # Normalize
    norms = np.linalg.norm(dirs_world, axis=-1, keepdims=True)
    dirs_world = dirs_world / norms

    # Reshape for raycasting
    origins = np.broadcast_to(cam_center.astype(np.float32), dirs_world.shape).copy()
    rays = np.concatenate([origins, dirs_world.astype(np.float32)], axis=-1)  # HxWx6

    rays_t = o3d.core.Tensor(rays.reshape(-1, 6))
    result = scene.cast_rays(rays_t)
    t_hit = result['t_hit'].numpy().reshape(H, W)

    # Convert ray distance to z-depth: z = t_hit * cos(angle) = t_hit * dirs_cam_z / |dirs_cam|
    # Actually t_hit is distance along ray, and z-depth = t_hit * (1 / |dir_cam|) * 1
    # Since dir_cam = [(u-cx)/fx, (v-cy)/fy, 1], |dir_cam| = sqrt(dx^2+dy^2+1)
    # z = t_hit / |dir_cam|... no.
    # Actually: point_cam = R @ (origin + t*dir_world) + t_vec
    # Simpler: z-depth = dot(point_world - cam_center, forward_axis)
    # where forward_axis = R^T @ [0,0,1] = R[2,:] (third row of R)
    # z = t_hit * dot(dir_world, R[2,:])... but dir_world is normalized
    # Actually for pinhole: z = t_hit * cos(angle_from_optical_axis)
    # cos = dir_cam_z / |dir_cam| = 1 / sqrt(dx^2+dy^2+1)
    # So z_depth = t_hit * 1 / |dir_cam_unnormalized|

    # But for depth maps used in training, we typically want z-buffer depth (not ray distance)
    # z_depth = t_hit * (z_component of normalized direction in camera frame)
    # = t_hit * (1.0 / norms)  since dir_cam_z = 1 before normalization, and norms = |dir_cam|
    z_depth = t_hit / norms.squeeze(-1)

    # Mark invalid (inf) as 0
    z_depth[~np.isfinite(z_depth)] = 0

    return z_depth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--colmap_dir', required=True)
    parser.add_argument('--pointcloud', required=True, help='PLY in aligned COLMAP coordinates')
    parser.add_argument('--images_dir', default='', help='For preview overlays')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--poisson_depth', type=int, default=10)
    parser.add_argument('--voxel_size', type=float, default=0.05)
    parser.add_argument('--num_preview', type=int, default=5, help='Number of preview images')
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--max_dim', type=int, default=0, help='Max image dimension (0=full res)')
    args = parser.parse_args()

    # Build mesh
    mesh = build_mesh(args.pointcloud, args.poisson_depth, args.voxel_size)

    # Save mesh for inspection
    mesh_path = os.path.join(args.output_dir, 'mesh.ply')
    os.makedirs(args.output_dir, exist_ok=True)
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"Mesh saved to {mesh_path}")

    # Load COLMAP
    cams, imgs, _ = read_model(args.colmap_dir, ext='.bin')
    print(f"\n{len(imgs)} cameras")

    depth_dir = os.path.join(args.output_dir, 'depths')
    preview_dir = os.path.join(args.output_dir, 'preview')
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)

    img_keys = sorted(imgs.keys())
    preview_step = max(1, len(img_keys) // args.num_preview)
    preview_set = set(img_keys[::preview_step][:args.num_preview])

    import cv2

    t0 = time.time()
    for idx, img_id in enumerate(img_keys):
        im = imgs[img_id]
        cam = cams[im.camera_id]
        R = qvec2rotmat(im.qvec)
        t_vec = im.tvec

        if cam.model == 'PINHOLE':
            fx, fy, cx, cy = cam.params
        else:
            f, cx, cy = cam.params[:3]
            fx = fy = f

        W, H = cam.width, cam.height

        # Optionally reduce resolution
        scale = 1.0
        if args.max_dim > 0 and max(W, H) > args.max_dim:
            scale = args.max_dim / max(W, H)

        render_W = int(W * scale)
        render_H = int(H * scale)
        render_fx = fx * scale
        render_fy = fy * scale
        render_cx = cx * scale
        render_cy = cy * scale

        # Output path preserving subdirectory structure
        stem = os.path.splitext(im.name)[0]
        out_path = os.path.join(depth_dir, f"{stem}.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if args.skip_existing and os.path.exists(out_path):
            continue

        z_depth = render_depth_raycast(mesh, R, t_vec, render_fx, render_fy, render_cx, render_cy, render_W, render_H)

        # Save as 16-bit PNG (inverse depth normalized, same as Depth Anything)
        valid = z_depth > 0
        if valid.any():
            inv_depth = np.zeros_like(z_depth)
            inv_depth[valid] = 1.0 / z_depth[valid]
            d_min, d_max = inv_depth[valid].min(), inv_depth[valid].max()
            if d_max - d_min > 1e-6:
                inv_norm = (inv_depth - d_min) / (d_max - d_min)
            else:
                inv_norm = np.zeros_like(inv_depth)
            cv2.imwrite(out_path, (inv_norm * 65535).astype(np.uint16))

        # Preview
        if img_id in preview_set and args.images_dir:
            img_path = os.path.join(args.images_dir, im.name)
            image = cv2.imread(img_path)
            if image is not None:
                if scale != 1.0:
                    image = cv2.resize(image, (render_W, render_H))
                # Color depth overlay
                depth_vis = np.zeros_like(image)
                if valid.any():
                    d_norm = np.clip((z_depth - z_depth[valid].min()) / (z_depth[valid].max() - z_depth[valid].min() + 1e-6), 0, 1)
                    depth_color = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                    depth_color[~valid] = 0
                    blend = cv2.addWeighted(image, 0.4, depth_color, 0.6, 0)
                    cv2.imwrite(os.path.join(preview_dir, f"depth_{os.path.basename(stem)}.jpg"), blend)

        if (idx + 1) % 50 == 0 or idx < 3:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(img_keys) - idx - 1) / rate / 60
            print(f"  [{idx+1}/{len(img_keys)}] {rate:.2f} img/s, ETA {eta:.0f}min")

    print(f"\nDone! {len(img_keys)} depth maps in {(time.time()-t0)/60:.1f}min")


if __name__ == '__main__':
    main()

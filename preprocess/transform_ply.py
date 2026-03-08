"""
Transform an external PLY point cloud to match the auto_reorient coordinate system.

Computes the rotation + scale transform by comparing pre_aligned and aligned COLMAP models,
then applies it to the input PLY. Optionally downsamples via voxel grid.

Usage:
  python preprocess/transform_ply.py \
    --pre_aligned_dir .../pre_aligned/sparse/0 \
    --aligned_dir .../aligned/sparse/0 \
    --input_ply /path/to/colorized.ply \
    --output_ply /path/to/output.ply \
    --voxel_size 0.1
"""
import argparse
import numpy as np
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from read_write_model import read_model, qvec2rotmat


def compute_transform(pre_dir, ali_dir):
    """Recover rotation_matrix and upscale from pre_aligned vs aligned models."""
    _, imgs_pre, pts_pre = read_model(pre_dir, ext='.bin')
    _, imgs_ali, pts_ali = read_model(ali_dir, ext='.bin')

    # Get matching points (same keys)
    common_keys = sorted(set(pts_pre.keys()) & set(pts_ali.keys()))
    if len(common_keys) < 10:
        print(f"Only {len(common_keys)} common points, using camera centers instead")
        common_img_keys = sorted(set(imgs_pre.keys()) & set(imgs_ali.keys()))
        src = np.array([-qvec2rotmat(imgs_pre[k].qvec).T @ imgs_pre[k].tvec for k in common_img_keys])
        dst = np.array([-qvec2rotmat(imgs_ali[k].qvec).T @ imgs_ali[k].tvec for k in common_img_keys])
    else:
        n = min(len(common_keys), 50000)
        sel = common_keys[:n]
        src = np.array([pts_pre[k].xyz for k in sel])
        dst = np.array([pts_ali[k].xyz for k in sel])

    # Solve: dst = upscale * src @ R
    # Use Procrustes analysis
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    src_centered = src - src_c
    dst_centered = dst - dst_c

    # Scale
    scale = np.linalg.norm(dst_centered) / np.linalg.norm(src_centered)

    # Rotation via SVD
    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Verify: dst ≈ scale * (src - src_c) @ R + dst_c
    # But auto_reorient does: dst = upscale * src @ rotation_matrix (no centering)
    # So let's fit: dst = scale * src @ R + t
    t = dst_c - scale * src_c @ R

    # Verify error
    reconstructed = scale * src @ R + t
    error = np.linalg.norm(reconstructed - dst, axis=1).mean()
    print(f"Transform: scale={scale:.6f}, mean error={error:.6f}")

    return scale, R, t


def voxel_downsample(xyz, rgb, voxel_size):
    """Simple voxel grid downsampling."""
    if voxel_size <= 0:
        return xyz, rgb
    voxel_indices = np.floor(xyz / voxel_size).astype(np.int64)
    # Use unique voxels, keep first point per voxel
    _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    return xyz[unique_idx], rgb[unique_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_aligned_dir', required=True)
    parser.add_argument('--aligned_dir', required=True)
    parser.add_argument('--input_ply', required=True)
    parser.add_argument('--output_ply', required=True)
    parser.add_argument('--voxel_size', type=float, default=0.1,
                        help='Voxel size for downsampling (0=no downsample)')
    args = parser.parse_args()

    # Compute transform
    print("Computing transform from pre_aligned -> aligned...")
    scale, R, t = compute_transform(args.pre_aligned_dir, args.aligned_dir)

    # Load input PLY
    print(f"Loading {args.input_ply}...")
    from plyfile import PlyData
    plydata = PlyData.read(args.input_ply)
    verts = plydata['vertex']
    xyz = np.vstack([verts['x'], verts['y'], verts['z']]).T.astype(np.float64)
    if 'red' in verts:
        rgb = np.vstack([verts['red'], verts['green'], verts['blue']]).T.astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), 128, dtype=np.uint8)
    print(f"  {len(xyz):,} points loaded")

    # Voxel downsample before transform (faster)
    if args.voxel_size > 0:
        print(f"Voxel downsampling (size={args.voxel_size})...")
        xyz, rgb = voxel_downsample(xyz, rgb, args.voxel_size)
        print(f"  {len(xyz):,} points after downsample")

    # Apply transform
    print("Applying transform...")
    xyz_transformed = scale * xyz @ R + t
    xyz_transformed = xyz_transformed.astype(np.float32)

    # Save PLY
    print(f"Saving to {args.output_ply}...")
    from plyfile import PlyElement
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    normals = np.zeros_like(xyz_transformed)
    elements = np.empty(len(xyz_transformed), dtype=dtype)
    elements[:] = list(map(tuple, np.concatenate([
        xyz_transformed, normals, rgb.astype(np.float32)
    ], axis=1)))
    ply_out = PlyData([PlyElement.describe(elements, 'vertex')])
    ply_out.write(args.output_ply)
    print(f"Done! {len(xyz_transformed):,} points saved")


if __name__ == "__main__":
    main()

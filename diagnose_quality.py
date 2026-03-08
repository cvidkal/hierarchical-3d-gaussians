#!/usr/bin/env python3
"""
Diagnostic script for Gaussian Splatting training quality issues.
Systematically checks camera calibration, image quality, point cloud alignment,
training pipeline, per-image PSNR breakdown, and spatial analysis.

Usage:
    python diagnose_quality.py \
        --colmap_path D:/hengdian_gs/cropped_project/camera_calibration/aligned/sparse/0/ \
        --images_path D:/hengdian_gs/cropped_project/camera_calibration/rectified/images/ \
        --trained_model D:/hengdian_gs/cropped_output/trained_chunks/0_0/ \
        --num_sample_images 20
"""

import os
import sys
import argparse
import math
import struct
import collections
import json
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# COLMAP binary readers (self-contained, from scene/colmap_loader.py)
# ---------------------------------------------------------------------------
CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
ColmapCamera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

CAMERA_MODEL_IDS = {
    0: CameraModel(0, "SIMPLE_PINHOLE", 3),
    1: CameraModel(1, "PINHOLE", 4),
    2: CameraModel(2, "SIMPLE_RADIAL", 4),
    3: CameraModel(3, "RADIAL", 5),
    4: CameraModel(4, "OPENCV", 8),
    5: CameraModel(5, "OPENCV_FISHEYE", 8),
    6: CameraModel(6, "FULL_OPENCV", 12),
    7: CameraModel(7, "FOV", 5),
    8: CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4),
    9: CameraModel(9, "RADIAL_FISHEYE", 5),
    10: CameraModel(10, "THIN_PRISM_FISHEYE", 12),
}


def _read_next_bytes(fid, num_bytes, fmt, endian="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian + fmt, data)


def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            props = _read_next_bytes(fid, 24, "iiQQ")
            camera_id = props[0]
            model_id = props[1]
            model_name = CAMERA_MODEL_IDS[model_id].model_name
            width = props[2]
            height = props[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = _read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = ColmapCamera(
                id=camera_id, model=model_name,
                width=width, height=height, params=np.array(params))
    return cameras


def read_cameras_text(path):
    cameras = {}
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(list(map(float, elems[4:])))
                cameras[camera_id] = ColmapCamera(
                    id=camera_id, model=model,
                    width=width, height=height, params=params)
    return cameras


class ColmapImage(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def read_images_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num_images = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            props = _read_next_bytes(fid, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            camera_id = props[8]
            name = ""
            ch = _read_next_bytes(fid, 1, "c")[0]
            while ch != b"\x00":
                name += ch.decode("utf-8")
                ch = _read_next_bytes(fid, 1, "c")[0]
            num_points2D = _read_next_bytes(fid, 8, "Q")[0]
            xys_ids = _read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
            xys = np.column_stack([
                list(map(float, xys_ids[0::3])),
                list(map(float, xys_ids[1::3]))]) if num_points2D > 0 else np.zeros((0, 2))
            point3D_ids = np.array(list(map(int, xys_ids[2::3]))) if num_points2D > 0 else np.array([], dtype=np.int64)
            images[image_id] = ColmapImage(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_images_text(path):
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(list(map(float, elems[1:5])))
                tvec = np.array(list(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([
                    list(map(float, elems[0::3])),
                    list(map(float, elems[1::3]))]) if len(elems) > 0 else np.zeros((0, 2))
                point3D_ids = np.array(list(map(int, elems[2::3]))) if len(elems) > 0 else np.array([], dtype=np.int64)
                images[image_id] = ColmapImage(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_points3D_binary(path):
    pts = {}
    with open(path, "rb") as fid:
        num_points = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            props = _read_next_bytes(fid, 43, "QdddBBBd")
            point_id = props[0]
            xyz = np.array(props[1:4])
            rgb = np.array(props[4:7])
            error = props[7]
            track_length = _read_next_bytes(fid, 8, "Q")[0]
            track = _read_next_bytes(fid, 8 * track_length, "ii" * track_length)
            image_ids = np.array(track[0::2])
            point2D_idxs = np.array(track[1::2])
            pts[point_id] = Point3D(id=point_id, xyz=xyz, rgb=rgb, error=error,
                                    image_ids=image_ids, point2D_idxs=point2D_idxs)
    return pts


def read_points3D_text(path):
    pts = {}
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point_id = int(elems[0])
                xyz = np.array(list(map(float, elems[1:4])))
                rgb = np.array(list(map(int, elems[4:7])))
                error = float(elems[7])
                # track: pairs of (image_id, point2D_idx)
                track = list(map(int, elems[8:]))
                image_ids = np.array(track[0::2]) if len(track) > 0 else np.array([], dtype=np.int64)
                point2D_idxs = np.array(track[1::2]) if len(track) > 0 else np.array([], dtype=np.int64)
                pts[point_id] = Point3D(id=point_id, xyz=xyz, rgb=rgb, error=error,
                                        image_ids=image_ids, point2D_idxs=point2D_idxs)
    return pts


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def load_colmap_model(sparse_path):
    """Load cameras, images, points3D from a COLMAP sparse directory."""
    # Try binary first, fallback to text
    cam_bin = os.path.join(sparse_path, "cameras.bin")
    cam_txt = os.path.join(sparse_path, "cameras.txt")
    img_bin = os.path.join(sparse_path, "images.bin")
    img_txt = os.path.join(sparse_path, "images.txt")
    pts_bin = os.path.join(sparse_path, "points3D.bin")
    pts_txt = os.path.join(sparse_path, "points3D.txt")

    if os.path.exists(cam_bin):
        cameras = read_cameras_binary(cam_bin)
    else:
        cameras = read_cameras_text(cam_txt)

    if os.path.exists(img_bin):
        images = read_images_binary(img_bin)
    else:
        images = read_images_text(img_txt)

    if os.path.exists(pts_bin):
        points3D = read_points3D_binary(pts_bin)
    elif os.path.exists(pts_txt):
        points3D = read_points3D_text(pts_txt)
    else:
        points3D = {}

    return cameras, images, points3D


def get_intrinsic_matrix(cam):
    """Return 3x3 intrinsic matrix from a ColmapCamera."""
    if cam.model == "SIMPLE_PINHOLE":
        f = cam.params[0]
        cx, cy = cam.params[1], cam.params[2]
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    elif cam.model == "PINHOLE":
        fx, fy = cam.params[0], cam.params[1]
        cx, cy = cam.params[2], cam.params[3]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    else:
        # Fallback: treat first two as fx, fy
        fx = cam.params[0]
        fy = cam.params[1] if len(cam.params) > 1 else fx
        cx = cam.width / 2
        cy = cam.height / 2
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def project_points(pts3D, R, t, K):
    """Project Nx3 world points to pixel coords. Returns Nx2 pixel, Nx1 depth."""
    pts_cam = (R @ pts3D.T + t.reshape(3, 1))  # 3xN
    depth = pts_cam[2, :]
    pts_proj = K @ pts_cam  # 3xN
    px = pts_proj[0, :] / pts_proj[2, :]
    py = pts_proj[1, :] / pts_proj[2, :]
    return np.stack([px, py], axis=1), depth


def compute_psnr(img1, img2):
    """Compute PSNR between two float images in [0,1] range."""
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def separator(title):
    width = 80
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


# ---------------------------------------------------------------------------
# Diagnostic checks
# ---------------------------------------------------------------------------
def check_1_camera_calibration(cameras, images, points3D):
    """Check camera model types, parameter sanity, and COLMAP reprojection errors."""
    separator("CHECK 1: Camera Calibration Quality")

    print(f"\nNumber of cameras (intrinsic models): {len(cameras)}")
    print(f"Number of images (extrinsics): {len(images)}")
    print(f"Number of 3D points: {len(points3D)}")

    # Camera models
    for cam_id, cam in cameras.items():
        print(f"\n  Camera {cam_id}: model={cam.model}, {cam.width}x{cam.height}")
        K = get_intrinsic_matrix(cam)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        print(f"    fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        print(f"    Principal point offset from center: "
              f"dx={cx - cam.width/2:.2f} px, dy={cy - cam.height/2:.2f} px")

        # Sanity: focal length should be reasonable
        diag = math.sqrt(cam.width**2 + cam.height**2)
        fov_h = focal2fov(fx, cam.width)
        fov_v = focal2fov(fy, cam.height)
        print(f"    FoV: horizontal={math.degrees(fov_h):.1f} deg, vertical={math.degrees(fov_v):.1f} deg")

        if fov_h > 150 or fov_v > 150:
            print("    [WARNING] Very wide FoV -- possibly wrong focal length or uncropped fisheye!")
        if fov_h < 10 or fov_v < 10:
            print("    [WARNING] Very narrow FoV -- check if focal length is correct for this resolution!")

        # Check model type compatibility with the pipeline
        if cam.model not in ("PINHOLE", "SIMPLE_PINHOLE"):
            print(f"    [CRITICAL] Camera model {cam.model} is NOT supported by the GS pipeline!")
            print(f"               The code asserts PINHOLE or SIMPLE_PINHOLE only.")
            print(f"               This will cause assertion errors or incorrect rendering.")

    # COLMAP reprojection errors (from points3D)
    if len(points3D) > 0:
        errors = np.array([p.error for p in points3D.values()])
        print(f"\n  COLMAP reprojection error (from points3D):")
        print(f"    Mean:   {errors.mean():.4f} px")
        print(f"    Median: {np.median(errors):.4f} px")
        print(f"    Std:    {errors.std():.4f} px")
        print(f"    Max:    {errors.max():.4f} px")
        print(f"    P95:    {np.percentile(errors, 95):.4f} px")
        print(f"    P99:    {np.percentile(errors, 99):.4f} px")

        if errors.mean() > 2.0:
            print("    [WARNING] Mean reprojection error > 2px -- calibration may be poor!")
        elif errors.mean() > 1.0:
            print("    [CAUTION] Mean reprojection error > 1px -- could affect quality.")
        else:
            print("    [OK] Reprojection error looks reasonable.")

        # Distribution of track lengths
        track_lengths = np.array([len(p.image_ids) for p in points3D.values()])
        print(f"\n  Track lengths (num images per 3D point):")
        print(f"    Mean: {track_lengths.mean():.1f}, Median: {np.median(track_lengths):.0f}, "
              f"Max: {track_lengths.max()}")
    else:
        print("\n  [WARNING] No 3D points found in COLMAP model!")

    # Check number of 2D observations per image
    obs_counts = {}
    for img_id, img in images.items():
        valid = np.sum(img.point3D_ids >= 0) if len(img.point3D_ids) > 0 else 0
        obs_counts[img.name] = valid

    vals = list(obs_counts.values())
    if len(vals) > 0:
        print(f"\n  2D observations per image (matched to 3D points):")
        print(f"    Mean: {np.mean(vals):.0f}, Min: {np.min(vals)}, Max: {np.max(vals)}")
        low_obs = [name for name, v in obs_counts.items() if v < 50]
        if low_obs:
            print(f"    [WARNING] {len(low_obs)} images have < 50 matched points!")
            for name in low_obs[:5]:
                print(f"      {name}: {obs_counts[name]} points")
            if len(low_obs) > 5:
                print(f"      ... and {len(low_obs) - 5} more")


def check_2_image_quality(images, images_path, num_samples=10):
    """Check image files: existence, size, pixel range, artifacts."""
    separator("CHECK 2: Image Quality")

    all_image_names = sorted([img.name for img in images.values()])
    print(f"\nTotal images in COLMAP model: {len(all_image_names)}")

    # Check how many actually exist on disk
    found = 0
    not_found = []
    for name in all_image_names:
        full_path = os.path.join(images_path, name)
        if os.path.exists(full_path):
            found += 1
        else:
            # Try alternate extensions
            stem = Path(name).stem
            parent = os.path.dirname(name)
            alt_jpg = os.path.join(images_path, parent, stem + ".jpg")
            alt_png = os.path.join(images_path, parent, stem + ".png")
            if os.path.exists(alt_jpg) or os.path.exists(alt_png):
                found += 1
            else:
                not_found.append(name)

    print(f"  Found on disk: {found}/{len(all_image_names)}")
    if not_found:
        print(f"  [WARNING] {len(not_found)} images NOT found!")
        for n in not_found[:5]:
            print(f"    Missing: {n}")

    # Sample some images for detailed check
    sample_names = all_image_names[:min(num_samples, len(all_image_names))]
    print(f"\n  Checking {len(sample_names)} sample images:")

    sizes = []
    for name in sample_names:
        full_path = os.path.join(images_path, name)
        if not os.path.exists(full_path):
            stem = Path(name).stem
            parent = os.path.dirname(name)
            for ext in [".jpg", ".png", ".JPG", ".PNG"]:
                alt = os.path.join(images_path, parent, stem + ext)
                if os.path.exists(alt):
                    full_path = alt
                    break

        if not os.path.exists(full_path):
            print(f"    {name}: NOT FOUND")
            continue

        try:
            img = Image.open(full_path)
            w, h = img.size
            sizes.append((w, h))
            arr = np.array(img).astype(np.float32)

            # Check for mostly black/white
            mean_val = arr.mean()
            std_val = arr.std()
            min_val = arr.min()
            max_val = arr.max()

            issues = []
            if mean_val < 10:
                issues.append("VERY DARK (mean<10)")
            if mean_val > 245:
                issues.append("VERY BRIGHT (mean>245)")
            if std_val < 5:
                issues.append("LOW CONTRAST (std<5)")
            if max_val - min_val < 20:
                issues.append("NEARLY UNIFORM")

            # Check for black borders (common with fisheye crops)
            border_top = arr[:10, :].mean()
            border_bot = arr[-10:, :].mean()
            border_left = arr[:, :10].mean()
            border_right = arr[:, -10:].mean()
            if min(border_top, border_bot, border_left, border_right) < 5:
                issues.append("BLACK BORDERS detected")

            status = " | ".join(issues) if issues else "OK"
            print(f"    {name}: {w}x{h}, mode={img.mode}, "
                  f"mean={mean_val:.1f}, std={std_val:.1f}, [{min_val:.0f}-{max_val:.0f}] -- {status}")

        except Exception as e:
            print(f"    {name}: ERROR loading -- {e}")

    if sizes:
        unique_sizes = list(set(sizes))
        print(f"\n  Unique image sizes found: {unique_sizes}")
        if len(unique_sizes) > 1:
            print("  [WARNING] Multiple image sizes detected! This can cause issues.")


def check_3_point_cloud_alignment(cameras, images, points3D, images_path, num_samples=20):
    """Project 3D points onto sample images and measure alignment error."""
    separator("CHECK 3: Point Cloud Alignment (3D->2D projection)")

    if len(points3D) == 0:
        print("\n  [SKIP] No 3D points available.")
        return

    # Collect per-image reprojection stats
    image_list = sorted(images.values(), key=lambda x: x.name)
    sample_images = image_list[:min(num_samples, len(image_list))]

    print(f"\n  Checking {len(sample_images)} sample images:")

    all_errors = []
    per_image_errors = {}

    for img in sample_images:
        cam = cameras[img.camera_id]
        K = get_intrinsic_matrix(cam)
        R = qvec2rotmat(img.qvec)
        t = img.tvec

        # Get 2D-3D correspondences from this image
        valid_mask = img.point3D_ids >= 0
        if valid_mask.sum() == 0:
            print(f"    {img.name}: no matched 3D points")
            continue

        valid_indices = np.where(valid_mask)[0]
        obs_2d = img.xys[valid_indices]
        pt3d_ids = img.point3D_ids[valid_indices]

        # Gather 3D points
        pts3d = []
        obs2d_filtered = []
        for i, pid in enumerate(pt3d_ids):
            if pid in points3D:
                pts3d.append(points3D[pid].xyz)
                obs2d_filtered.append(obs_2d[i])

        if len(pts3d) == 0:
            continue

        pts3d = np.array(pts3d)
        obs2d_filtered = np.array(obs2d_filtered)

        # Project
        proj2d, depths = project_points(pts3d, R, t, K)

        # Only keep points in front of camera
        in_front = depths > 0
        if in_front.sum() == 0:
            print(f"    {img.name}: all points behind camera!")
            continue

        proj2d = proj2d[in_front]
        obs2d_filtered = obs2d_filtered[in_front]
        depths_valid = depths[in_front]

        # Compute reprojection error
        errors = np.sqrt(np.sum((proj2d - obs2d_filtered) ** 2, axis=1))

        behind_pct = 100.0 * (1 - in_front.mean())
        mean_err = errors.mean()
        median_err = np.median(errors)
        max_err = errors.max()

        per_image_errors[img.name] = mean_err
        all_errors.extend(errors.tolist())

        status = ""
        if mean_err > 5:
            status = " [POOR]"
        elif mean_err > 2:
            status = " [MEDIOCRE]"

        print(f"    {img.name}: {len(errors)} pts, mean={mean_err:.2f}px, "
              f"med={median_err:.2f}px, max={max_err:.1f}px, "
              f"behind={behind_pct:.0f}%{status}")

    if all_errors:
        all_errors = np.array(all_errors)
        print(f"\n  Overall projection alignment:")
        print(f"    Mean error:   {all_errors.mean():.3f} px")
        print(f"    Median error: {np.median(all_errors):.3f} px")
        print(f"    P95 error:    {np.percentile(all_errors, 95):.3f} px")

        if all_errors.mean() > 3.0:
            print("    [CRITICAL] Large projection errors! Camera calibration may be wrong.")
        elif all_errors.mean() > 1.5:
            print("    [WARNING] Moderate projection errors.")
        else:
            print("    [OK] Projection alignment looks reasonable.")


def check_4_training_pipeline(cameras, images, images_path, trained_model_path, resolution_arg=-1):
    """Check resolution scaling behavior and training configuration."""
    separator("CHECK 4: Training Data Pipeline")

    # Determine what resolution the pipeline would use
    print(f"\n  Resolution argument: {resolution_arg}")

    sample_images = sorted(images.values(), key=lambda x: x.name)
    if len(sample_images) == 0:
        print("  [SKIP] No images to check.")
        return

    # Check a few images
    checked = 0
    for img in sample_images[:5]:
        full_path = os.path.join(images_path, img.name)
        if not os.path.exists(full_path):
            stem = Path(img.name).stem
            parent = os.path.dirname(img.name)
            for ext in [".jpg", ".png", ".JPG", ".PNG"]:
                alt = os.path.join(images_path, parent, stem + ext)
                if os.path.exists(alt):
                    full_path = alt
                    break
        if not os.path.exists(full_path):
            continue

        pil_img = Image.open(full_path)
        orig_w, orig_h = pil_img.size
        cam = cameras[img.camera_id]

        # Replicate the resolution logic from camera_utils.py loadCam
        if resolution_arg in [1, 2, 4, 8]:
            train_w = round(orig_w / resolution_arg)
            train_h = round(orig_h / resolution_arg)
        else:
            if resolution_arg == -1:
                if orig_w > 1600:
                    global_down = orig_w / 1600
                else:
                    global_down = 1
            else:
                global_down = orig_w / resolution_arg
            train_w = int(orig_w / global_down)
            train_h = int(orig_h / global_down)

        scale_factor = orig_w / train_w

        print(f"\n  Image: {img.name}")
        print(f"    COLMAP registered size: {cam.width}x{cam.height}")
        print(f"    Actual image on disk:   {orig_w}x{orig_h}")
        print(f"    Training resolution:    {train_w}x{train_h} (scale={scale_factor:.2f}x)")

        if cam.width != orig_w or cam.height != orig_h:
            print(f"    [WARNING] COLMAP size ({cam.width}x{cam.height}) != disk size ({orig_w}x{orig_h})!")
            print(f"    This means intrinsics (fx, fy, cx, cy) are for the WRONG resolution!")
            print(f"    The pipeline uses FoV (scale-invariant) + primx/primy (normalized), so this")
            print(f"    may still work IF the aspect ratio is the same.")
            aspect_colmap = cam.width / cam.height
            aspect_disk = orig_w / orig_h
            if abs(aspect_colmap - aspect_disk) > 0.01:
                print(f"    [CRITICAL] Aspect ratios differ! COLMAP={aspect_colmap:.4f}, disk={aspect_disk:.4f}")
            else:
                print(f"    Aspect ratios match ({aspect_colmap:.4f}). FoV-based approach should be OK.")

        # Check pixel values
        arr = np.array(pil_img)
        print(f"    Pixel dtype: {arr.dtype}, range: [{arr.min()}, {arr.max()}]")
        print(f"    Mode: {pil_img.mode} (pipeline expects RGB, converts /255 to [0,1])")

        if pil_img.mode not in ("RGB", "RGBA"):
            print(f"    [WARNING] Image mode is {pil_img.mode}, not RGB! May cause color issues.")

        checked += 1
        if checked >= 3:
            break

    # Check for cfg_args in trained model
    cfg_path = os.path.join(trained_model_path, "cfg_args") if trained_model_path else None
    if cfg_path and os.path.exists(cfg_path):
        print(f"\n  Training config (from {cfg_path}):")
        try:
            with open(cfg_path, "r") as f:
                cfg_str = f.read()
            print(f"    {cfg_str[:500]}")
            # Parse key values
            ns = eval(cfg_str)
            if hasattr(ns, 'resolution'):
                print(f"\n    Training resolution arg: {ns.resolution}")
            if hasattr(ns, 'source_path'):
                print(f"    Source path: {ns.source_path}")
            if hasattr(ns, 'images'):
                print(f"    Images subdir: {ns.images}")
            if hasattr(ns, 'white_background'):
                print(f"    White background: {ns.white_background}")
        except Exception as e:
            print(f"    Could not parse: {e}")
    else:
        print(f"\n  [INFO] No cfg_args found at trained model path.")

    # Check exposure.json
    exp_path = os.path.join(trained_model_path, "exposure.json") if trained_model_path else None
    if exp_path and os.path.exists(exp_path):
        try:
            with open(exp_path, "r") as f:
                exposure = json.load(f)
            vals = [np.array(v) for v in exposure.values()]
            if vals:
                all_exp = np.stack(vals)
                print(f"\n  Exposure compensation (from exposure.json):")
                print(f"    Num images: {len(exposure)}")
                print(f"    Shape per entry: {all_exp.shape[1:]}")
                # Check if exposure is identity (3x3 matrix should be close to I)
                if all_exp.ndim == 3 and all_exp.shape[1] == 3 and all_exp.shape[2] == 3:
                    identities = np.eye(3)[None, :, :]
                    diffs = np.abs(all_exp - identities).mean(axis=(1, 2))
                    print(f"    Mean deviation from identity: {diffs.mean():.6f}")
                    print(f"    Max deviation from identity:  {diffs.max():.6f}")
                    if diffs.max() > 0.1:
                        print(f"    [INFO] Significant exposure correction applied.")
                        worst_idx = np.argmax(diffs)
                        worst_name = list(exposure.keys())[worst_idx]
                        print(f"    Largest correction: {worst_name} (dev={diffs[worst_idx]:.4f})")
        except Exception as e:
            print(f"    Could not read exposure.json: {e}")


def check_5_per_image_psnr(trained_model_path, images, cameras, images_path, resolution_arg=-1):
    """If rendered test images exist, compute per-image PSNR. Otherwise check training logs."""
    separator("CHECK 5: Per-Image Quality Breakdown")

    # Look for rendered images in the trained model directory
    render_dir = None
    if trained_model_path:
        # Common locations for rendered images
        for subdir in ["test", "train", "renders", "ours_30000"]:
            candidate = os.path.join(trained_model_path, subdir)
            if os.path.exists(candidate):
                render_dir = candidate
                break

        # Also check point_cloud iterations for renders
        pc_dir = os.path.join(trained_model_path, "point_cloud")
        if os.path.exists(pc_dir):
            iters = []
            try:
                for d in os.listdir(pc_dir):
                    if d.startswith("iteration_"):
                        iters.append(int(d.split("_")[1]))
            except:
                pass
            if iters:
                print(f"  Available iterations: {sorted(iters)}")

    # Instead of relying on pre-rendered images, compute PSNR using the existing model
    # We'll look for any existing evaluation output
    eval_results = {}
    if trained_model_path:
        results_file = os.path.join(trained_model_path, "results.json")
        if os.path.exists(results_file):
            try:
                with open(results_file) as f:
                    eval_results = json.load(f)
                print(f"\n  Found results.json:")
                for key, val in eval_results.items():
                    print(f"    {key}: {val}")
            except:
                pass

        # Check per_view results
        per_view_file = os.path.join(trained_model_path, "per_view_count.json")
        if os.path.exists(per_view_file):
            try:
                with open(per_view_file) as f:
                    per_view = json.load(f)
                print(f"\n  Found per_view_count.json with {len(per_view)} entries")
            except:
                pass

    # Manual PSNR computation: look for renders alongside GT
    # Typical hierarchy-gs eval output structure: test/ours_ITER/renders/ and test/ours_ITER/gt/
    psnr_results = []
    for search_root in [trained_model_path, os.path.join(trained_model_path, "test") if trained_model_path else ""]:
        if not search_root or not os.path.exists(search_root):
            continue
        for dirpath, dirnames, filenames in os.walk(search_root):
            if "renders" in dirnames and "gt" in dirnames:
                render_subdir = os.path.join(dirpath, "renders")
                gt_subdir = os.path.join(dirpath, "gt")
                render_files = sorted([f for f in os.listdir(render_subdir) if f.endswith(('.png', '.jpg'))])
                gt_files = sorted([f for f in os.listdir(gt_subdir) if f.endswith(('.png', '.jpg'))])

                if len(render_files) > 0 and len(gt_files) > 0:
                    print(f"\n  Found render/gt pair at: {dirpath}")
                    print(f"    Renders: {len(render_files)}, GT: {len(gt_files)}")

                    for rf in render_files:
                        # Find matching GT
                        gf = rf  # usually same name
                        render_path = os.path.join(render_subdir, rf)
                        gt_path = os.path.join(gt_subdir, gf)
                        if not os.path.exists(gt_path):
                            continue

                        try:
                            r_img = np.array(Image.open(render_path)).astype(np.float32) / 255.0
                            g_img = np.array(Image.open(gt_path)).astype(np.float32) / 255.0

                            if r_img.shape != g_img.shape:
                                continue

                            psnr = compute_psnr(r_img, g_img)
                            psnr_results.append((rf, psnr))
                        except:
                            continue

    if psnr_results:
        psnr_results.sort(key=lambda x: x[1])
        psnr_values = [p for _, p in psnr_results]

        print(f"\n  Per-image PSNR ({len(psnr_results)} images):")
        print(f"    Mean PSNR:   {np.mean(psnr_values):.2f} dB")
        print(f"    Median PSNR: {np.median(psnr_values):.2f} dB")
        print(f"    Min PSNR:    {np.min(psnr_values):.2f} dB")
        print(f"    Max PSNR:    {np.max(psnr_values):.2f} dB")

        print(f"\n  Worst 10 images:")
        for name, psnr in psnr_results[:10]:
            print(f"    {name}: {psnr:.2f} dB")

        print(f"\n  Best 10 images:")
        for name, psnr in psnr_results[-10:]:
            print(f"    {name}: {psnr:.2f} dB")

        if np.mean(psnr_values) < 20:
            print("\n  [CRITICAL] Mean PSNR < 20 dB -- very poor quality!")
        elif np.mean(psnr_values) < 25:
            print("\n  [WARNING] Mean PSNR < 25 dB -- below typical GS quality.")
    else:
        print("\n  [INFO] No render/gt pairs found for PSNR computation.")
        print("         Run evaluation first, or check training logs for loss values.")

        # Try to parse training loss from log files
        if trained_model_path:
            log_candidates = [
                os.path.join(trained_model_path, "log.txt"),
                os.path.join(trained_model_path, "training.log"),
            ]
            for lp in log_candidates:
                if os.path.exists(lp):
                    print(f"\n  Found log file: {lp}")
                    # Read last few lines
                    try:
                        with open(lp, "r") as f:
                            lines = f.readlines()
                        print(f"  Last 5 lines:")
                        for line in lines[-5:]:
                            print(f"    {line.rstrip()}")
                    except:
                        pass


def check_6_spatial_analysis(images, cameras, points3D, images_path):
    """Analyze quality by spatial grouping: left vs right cameras, by position."""
    separator("CHECK 6: Spatial Analysis (Left vs Right)")

    left_images = []
    right_images = []
    other_images = []

    for img_id, img in images.items():
        name_lower = img.name.lower()
        if "left" in name_lower:
            left_images.append(img)
        elif "right" in name_lower:
            right_images.append(img)
        else:
            other_images.append(img)

    print(f"\n  Left images:  {len(left_images)}")
    print(f"  Right images: {len(right_images)}")
    print(f"  Other images: {len(other_images)}")

    # Analyze camera positions
    def analyze_group(group_images, label):
        if len(group_images) == 0:
            return
        positions = []
        obs_counts = []
        for img in group_images:
            R = qvec2rotmat(img.qvec)
            t = img.tvec
            # Camera center = -R^T @ t
            C = -R.T @ t
            positions.append(C)
            valid = np.sum(img.point3D_ids >= 0) if len(img.point3D_ids) > 0 else 0
            obs_counts.append(valid)

        positions = np.array(positions)
        obs_counts = np.array(obs_counts)

        print(f"\n  {label} cameras ({len(group_images)}):")
        print(f"    Position range X: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}]")
        print(f"    Position range Y: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]")
        print(f"    Position range Z: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}]")
        print(f"    Mean 2D observations: {obs_counts.mean():.0f}")
        print(f"    Min  2D observations: {obs_counts.min()}")

        # Which camera intrinsics are used
        cam_ids_used = set(img.camera_id for img in group_images)
        print(f"    Camera intrinsic IDs used: {cam_ids_used}")
        for cid in cam_ids_used:
            cam = cameras[cid]
            K = get_intrinsic_matrix(cam)
            print(f"      Camera {cid}: {cam.model} {cam.width}x{cam.height}, "
                  f"fx={K[0,0]:.1f} fy={K[1,1]:.1f}")

    analyze_group(left_images, "LEFT")
    analyze_group(right_images, "RIGHT")
    if other_images:
        analyze_group(other_images, "OTHER")

    # Check if left and right have different intrinsics
    if left_images and right_images:
        left_cams = set(img.camera_id for img in left_images)
        right_cams = set(img.camera_id for img in right_images)
        shared = left_cams & right_cams
        if shared:
            print(f"\n  [INFO] Left and right share camera intrinsics: {shared}")
        else:
            print(f"\n  [INFO] Left and right use DIFFERENT camera intrinsics")
            print(f"    Left uses:  {left_cams}")
            print(f"    Right uses: {right_cams}")
            # Compare focal lengths
            for lid in left_cams:
                for rid in right_cams:
                    lc = cameras[lid]
                    rc = cameras[rid]
                    lK = get_intrinsic_matrix(lc)
                    rK = get_intrinsic_matrix(rc)
                    print(f"    Left cam {lid}: fx={lK[0,0]:.1f}, fy={lK[1,1]:.1f}")
                    print(f"    Right cam {rid}: fx={rK[0,0]:.1f}, fy={rK[1,1]:.1f}")
                    fx_diff = abs(lK[0,0] - rK[0,0]) / max(lK[0,0], rK[0,0]) * 100
                    if fx_diff > 5:
                        print(f"    [WARNING] Focal length differs by {fx_diff:.1f}%!")

    # Check baseline between left/right pairs
    if left_images and right_images:
        left_positions = []
        right_positions = []
        for img in left_images:
            R = qvec2rotmat(img.qvec)
            C = -R.T @ img.tvec
            left_positions.append(C)
        for img in right_images:
            R = qvec2rotmat(img.qvec)
            C = -R.T @ img.tvec
            right_positions.append(C)

        left_center = np.mean(left_positions, axis=0)
        right_center = np.mean(right_positions, axis=0)
        baseline = np.linalg.norm(left_center - right_center)
        print(f"\n  Mean left camera center:  [{left_center[0]:.2f}, {left_center[1]:.2f}, {left_center[2]:.2f}]")
        print(f"  Mean right camera center: [{right_center[0]:.2f}, {right_center[1]:.2f}, {right_center[2]:.2f}]")
        print(f"  Mean baseline: {baseline:.4f} (scene units)")


def check_7_colmap_image_size_consistency(cameras, images, images_path):
    """Critical check: does COLMAP think images are the same size as what's on disk?"""
    separator("CHECK 7: COLMAP vs Disk Image Size Consistency")

    mismatches = []
    checked = 0
    for img_id, img in sorted(images.items(), key=lambda x: x[1].name):
        cam = cameras[img.camera_id]
        full_path = os.path.join(images_path, img.name)
        if not os.path.exists(full_path):
            stem = Path(img.name).stem
            parent = os.path.dirname(img.name)
            for ext in [".jpg", ".png", ".JPG", ".PNG"]:
                alt = os.path.join(images_path, parent, stem + ext)
                if os.path.exists(alt):
                    full_path = alt
                    break

        if not os.path.exists(full_path):
            continue

        try:
            pil_img = Image.open(full_path)
            disk_w, disk_h = pil_img.size
            pil_img.close()
        except:
            continue

        if cam.width != disk_w or cam.height != disk_h:
            mismatches.append({
                "name": img.name,
                "colmap": (cam.width, cam.height),
                "disk": (disk_w, disk_h),
                "cam_id": img.camera_id
            })

        checked += 1

    print(f"\n  Checked {checked} images")
    if mismatches:
        print(f"  [CRITICAL] {len(mismatches)} images have COLMAP/disk size mismatch!")
        for m in mismatches[:10]:
            print(f"    {m['name']}: COLMAP={m['colmap'][0]}x{m['colmap'][1]}, "
                  f"disk={m['disk'][0]}x{m['disk'][1]} (cam_id={m['cam_id']})")
        if len(mismatches) > 10:
            print(f"    ... and {len(mismatches) - 10} more")

        print(f"\n  [EXPLANATION] The hierarchy-gs pipeline converts focal length to FoV using")
        print(f"  the COLMAP-registered resolution, then at training time computes focal from")
        print(f"  FoV using the training resolution. If COLMAP width != disk width:")
        print(f"    - FoV is computed from COLMAP's (fx, colmap_width)")
        print(f"    - Training focal = fov2focal(FoV, train_width)")
        print(f"  This is CORRECT as long as the crop preserves the same optical center and")
        print(f"  field of view. But if you cropped the image WITHOUT updating COLMAP intrinsics,")
        print(f"  the focal/principal point will be WRONG.")
    else:
        print(f"  [OK] All checked images match COLMAP registered size.")


def check_8_principal_point(cameras):
    """Check if principal point is close to image center (critical for the pipeline)."""
    separator("CHECK 8: Principal Point Analysis")

    print("\n  The pipeline normalizes principal point as primx = cx/width, primy = cy/height.")
    print("  The projection matrix shifts based on primx, primy (0.5 = centered).\n")

    for cam_id, cam in cameras.items():
        K = get_intrinsic_matrix(cam)
        cx, cy = K[0, 2], K[1, 2]
        primx = cx / cam.width
        primy = cy / cam.height

        offset_x_pct = abs(primx - 0.5) * 100
        offset_y_pct = abs(primy - 0.5) * 100

        print(f"  Camera {cam_id} ({cam.model} {cam.width}x{cam.height}):")
        print(f"    cx={cx:.2f}, cy={cy:.2f}")
        print(f"    primx={primx:.4f}, primy={primy:.4f} (0.5 = centered)")
        print(f"    Offset from center: {offset_x_pct:.1f}% horizontal, {offset_y_pct:.1f}% vertical")

        if offset_x_pct > 5 or offset_y_pct > 5:
            print(f"    [WARNING] Principal point is significantly off-center!")
            print(f"    For center-cropped fisheye images, this could indicate that the crop")
            print(f"    center does not match the optical center, or COLMAP intrinsics were")
            print(f"    not updated after cropping.")


def generate_summary(cameras, images, points3D):
    """Print a final summary with actionable recommendations."""
    separator("DIAGNOSTIC SUMMARY")

    issues = []

    # Camera model check
    for cam_id, cam in cameras.items():
        if cam.model not in ("PINHOLE", "SIMPLE_PINHOLE"):
            issues.append(f"Camera {cam_id} uses {cam.model} -- NOT supported by the pipeline!")

    # Reprojection error
    if points3D:
        errors = np.array([p.error for p in points3D.values()])
        if errors.mean() > 2.0:
            issues.append(f"High COLMAP reprojection error (mean={errors.mean():.2f}px)")

    # Few observations
    low_obs_count = 0
    for img_id, img in images.items():
        valid = np.sum(img.point3D_ids >= 0) if len(img.point3D_ids) > 0 else 0
        if valid < 20:
            low_obs_count += 1
    if low_obs_count > 0:
        issues.append(f"{low_obs_count} images have < 20 matched 3D points")

    if issues:
        print("\n  ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
    else:
        print("\n  No critical issues detected from COLMAP data alone.")

    print("\n  COMMON CAUSES OF LOW PSNR (~15 dB):")
    print("    1. COLMAP intrinsics are for original fisheye images, not the cropped ones")
    print("       -> FoV is computed wrong, everything renders at wrong scale/position")
    print("    2. Image resolution mismatch: COLMAP says WxH but actual files are W'xH'")
    print("       -> Check 7 above diagnoses this")
    print("    3. Images were center-cropped but principal point was not updated")
    print("       -> Check 8 above diagnoses this")
    print("    4. Training resolution is too low (auto-downscale to 1600px width)")
    print("       -> Check 4 above shows actual training resolution")
    print("    5. Wrong images directory -- training on different images than COLMAP used")
    print("    6. Color space issue (e.g., BGR vs RGB if using OpenCV somewhere)")
    print("    7. Exposure compensation diverged or was not used")
    print("    8. Too few Gaussians / not enough densification for this scene scale")
    print("    9. Scaffold point cloud does not cover the scene well")

    print("\n  RECOMMENDED NEXT STEPS:")
    print("    1. Verify COLMAP was run on the EXACT same images used for training")
    print("    2. If images were cropped, ensure COLMAP intrinsics match the cropped size")
    print("    3. Visually inspect: project sparse points onto a few images using this data")
    print("    4. Try training at full resolution (--resolution 1)")
    print("    5. Check if scaffold covers the scene (visualize in a point cloud viewer)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Diagnostic tool for Gaussian Splatting training quality")
    parser.add_argument("--colmap_path", type=str, required=True,
                        help="Path to COLMAP sparse model directory (containing cameras.bin, images.bin, points3D.bin)")
    parser.add_argument("--images_path", type=str, required=True,
                        help="Path to images directory (with left/ and right/ subdirs)")
    parser.add_argument("--trained_model", type=str, default="",
                        help="Path to trained model output directory")
    parser.add_argument("--resolution", type=int, default=-1,
                        help="Resolution argument used during training (default: -1 = auto)")
    parser.add_argument("--num_sample_images", type=int, default=20,
                        help="Number of sample images for detailed checks")
    parser.add_argument("--skip_render_check", action="store_true",
                        help="Skip render/GT PSNR comparison")
    args = parser.parse_args()

    print("=" * 80)
    print("  GAUSSIAN SPLATTING QUALITY DIAGNOSTIC")
    print("=" * 80)
    print(f"\n  COLMAP path:    {args.colmap_path}")
    print(f"  Images path:    {args.images_path}")
    print(f"  Trained model:  {args.trained_model}")
    print(f"  Resolution arg: {args.resolution}")

    # Load COLMAP model
    print(f"\n  Loading COLMAP model from {args.colmap_path} ...")
    cameras, images, points3D = load_colmap_model(args.colmap_path)
    print(f"  Loaded {len(cameras)} cameras, {len(images)} images, {len(points3D)} 3D points")

    # Run all checks
    check_1_camera_calibration(cameras, images, points3D)
    check_2_image_quality(images, args.images_path, num_samples=args.num_sample_images)
    check_3_point_cloud_alignment(cameras, images, points3D, args.images_path, num_samples=args.num_sample_images)
    check_4_training_pipeline(cameras, images, args.images_path, args.trained_model, args.resolution)
    if not args.skip_render_check:
        check_5_per_image_psnr(args.trained_model, images, cameras, args.images_path, args.resolution)
    check_6_spatial_analysis(images, cameras, points3D, args.images_path)
    check_7_colmap_image_size_consistency(cameras, images, args.images_path)
    check_8_principal_point(cameras)
    generate_summary(cameras, images, points3D)

    print("\n" + "=" * 80)
    print("  DIAGNOSTIC COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

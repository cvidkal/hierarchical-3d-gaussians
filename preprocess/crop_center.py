"""Crop center region of undistorted images and update COLMAP cameras.

Reduces extreme FOV from fisheye-undistorted images to improve GS training.

Usage:
  python preprocess/crop_center.py \
    --input_dir /path/to/rectified \
    --output_dir /path/to/cropped \
    --colmap_dir /path/to/aligned/sparse/0 \
    --crop_ratio 0.35
"""
import argparse, cv2, os, sys, struct, shutil
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from read_write_model import read_model, write_model, Camera


def crop_image(args):
    src, dst, x0, y0, new_w, new_h = args
    img = cv2.imread(src)
    if img is None:
        return False
    cropped = img[y0:y0+new_h, x0:x0+new_w]
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cv2.imwrite(dst, cropped)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='rectified dir with images/ subdir')
    parser.add_argument('--output_dir', required=True, help='output cropped dir')
    parser.add_argument('--colmap_dir', required=True, help='aligned/sparse/0 with cameras.bin etc')
    parser.add_argument('--output_colmap_dir', default='', help='output colmap dir (default: output_dir/sparse/0)')
    parser.add_argument('--crop_ratio', type=float, default=0.35, help='keep center this fraction')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    # Read COLMAP model
    cams, imgs, pts = read_model(args.colmap_dir, ext='.bin')

    # Compute crop for each camera
    cam_crops = {}
    new_cams = {}
    for cam_id, cam in cams.items():
        W, H = cam.width, cam.height
        new_W = int(W * args.crop_ratio)
        new_H = int(H * args.crop_ratio)
        # Make even
        new_W = new_W - (new_W % 2)
        new_H = new_H - (new_H % 2)
        x0 = (W - new_W) // 2
        y0 = (H - new_H) // 2

        # Update intrinsics: fx, fy stay same, cx/cy shift
        params = list(cam.params)
        if cam.model == 'PINHOLE':
            # params = [fx, fy, cx, cy]
            params[2] -= x0  # cx
            params[3] -= y0  # cy
        elif cam.model == 'SIMPLE_PINHOLE':
            # params = [f, cx, cy]
            params[1] -= x0
            params[2] -= y0

        new_cams[cam_id] = Camera(
            id=cam.id, model=cam.model,
            width=new_W, height=new_H,
            params=np.array(params))
        cam_crops[cam_id] = (x0, y0, new_W, new_H)

        fov_x = 2 * np.arctan(new_W / (2 * params[0])) * 180 / np.pi
        fov_y = 2 * np.arctan(new_H / (2 * (params[1] if cam.model == 'PINHOLE' else params[0]))) * 180 / np.pi
        print(f"Camera {cam_id}: {W}x{H} -> {new_W}x{new_H}, FOV {fov_x:.1f}x{fov_y:.1f} deg")

    # Write updated COLMAP model
    out_colmap = args.output_colmap_dir if args.output_colmap_dir else os.path.join(args.output_dir, 'sparse', '0')
    os.makedirs(out_colmap, exist_ok=True)
    write_model(new_cams, imgs, pts, out_colmap, ext='.bin')

    # Copy depth_params.json if exists
    dp = os.path.join(args.colmap_dir, 'depth_params.json')
    if os.path.exists(dp):
        shutil.copy2(dp, os.path.join(out_colmap, 'depth_params.json'))
        print(f"Copied depth_params.json")

    # Crop images
    input_images = os.path.join(args.input_dir, 'images')
    output_images = os.path.join(args.output_dir, 'images')

    tasks = []
    for img_id, img in imgs.items():
        cam_id = img.camera_id
        x0, y0, new_w, new_h = cam_crops[cam_id]
        src = os.path.join(input_images, img.name)
        dst = os.path.join(output_images, img.name)
        tasks.append((src, dst, x0, y0, new_w, new_h))

    print(f"\nCropping {len(tasks)} images with {args.workers} workers...")
    done = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for ok in pool.map(crop_image, tasks):
            if ok:
                done += 1
            else:
                failed += 1
            if done % 200 == 0:
                print(f"  {done}/{len(tasks)} done")

    print(f"\nDone! {done} cropped, {failed} failed")


if __name__ == '__main__':
    main()

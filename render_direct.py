"""Render images directly from trained Gaussians (no hierarchy LOD)."""
import os, sys, torch, torchvision
from argparse import ArgumentParser, Namespace
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import CameraDataset, loadCam
from gaussian_renderer import render
from arguments import PipelineParams

@torch.no_grad()
def main():
    parser = ArgumentParser()
    parser.add_argument('-s', '--source_path', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('-i', '--images', default='images')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--num_images', type=int, default=5)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])
    pipe = pp.extract(args)

    # Load scene info
    scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, "", "", False, False)

    # Load gaussians
    gaussians = GaussianModel(3)
    # Find max iteration
    pc_dir = os.path.join(args.model_path, "point_cloud")
    iters = [int(d.split("_")[-1]) for d in os.listdir(pc_dir)]
    max_iter = max(iters)
    ply_path = os.path.join(pc_dir, f"iteration_{max_iter}", "point_cloud.ply")
    print(f"Loading model from {ply_path}")
    gaussians.load_ply(ply_path)

    # Load exposure
    exp_path = os.path.join(args.model_path, "exposure.json")
    if os.path.exists(exp_path):
        import json
        with open(exp_path) as f:
            exp_data = json.load(f)
        gaussians.pretrained_exposures = {k: torch.FloatTensor(v).cuda() for k, v in exp_data.items()}

    # Load cameras
    cam_args = Namespace(
        resolution=-1, data_device='cpu', train_test_exp=False,
        white_background=False
    )
    cam_infos = scene_info.train_cameras
    n = len(cam_infos)
    step = max(1, n // args.num_images)
    indices = list(range(0, n, step))[:args.num_images]

    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Rendering {len(indices)} images from {n} cameras ({gaussians._xyz.shape[0]} gaussians)")

    for i, idx in enumerate(indices):
        cam = loadCam(cam_args, idx, cam_infos[idx], 1.0, False)
        cam.world_view_transform = cam.world_view_transform.cuda()
        cam.projection_matrix = cam.projection_matrix.cuda()
        cam.full_proj_transform = cam.full_proj_transform.cuda()
        cam.camera_center = cam.camera_center.cuda()

        result = render(cam, gaussians, pipe, bg, use_trained_exp=False)
        image = torch.clamp(result["render"], 0.0, 1.0)

        gt = cam.original_image.cuda()
        if gt.max() > 1.0:
            gt = gt / 255.0

        out_path = os.path.join(args.out_dir, f"render_{idx:04d}.png")
        gt_path = os.path.join(args.out_dir, f"gt_{idx:04d}.png")
        torchvision.utils.save_image(image, out_path)
        torchvision.utils.save_image(gt, gt_path)
        print(f"  [{i+1}/{len(indices)}] {cam_infos[idx].image_name} -> {out_path}")
        torch.cuda.empty_cache()

    print("Done!")

if __name__ == "__main__":
    main()

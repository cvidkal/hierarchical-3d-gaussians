"""Render a few sample images from the merged hierarchy."""
import math, os, sys, torch
import torchvision
from argparse import ArgumentParser
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render_post
from gaussian_hierarchy._C import expand_to_size, get_interpolation_weights
from utils.loss_utils import ssim
from utils.image_utils import psnr

@torch.no_grad()
def render_one(viewpoint, scene, pipe, tau):
    viewpoint.world_view_transform = viewpoint.world_view_transform.cuda()
    viewpoint.projection_matrix = viewpoint.projection_matrix.cuda()
    viewpoint.full_proj_transform = viewpoint.full_proj_transform.cuda()
    viewpoint.camera_center = viewpoint.camera_center.cuda()

    render_indices = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()
    parent_indices = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()
    nodes_for_render_indices = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()
    interpolation_weights = torch.zeros(scene.gaussians._xyz.size(0)).float().cuda()
    num_siblings = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()

    tanfovx = math.tan(viewpoint.FoVx * 0.5)
    threshold = (2 * (tau + 0.5)) * tanfovx / (0.5 * viewpoint.image_width)

    to_render = expand_to_size(
        scene.gaussians.nodes, scene.gaussians.boxes, threshold,
        viewpoint.camera_center, torch.zeros((3)),
        render_indices, parent_indices, nodes_for_render_indices)

    indices = render_indices[:to_render].int().contiguous()
    node_indices = nodes_for_render_indices[:to_render].contiguous()

    get_interpolation_weights(
        node_indices, threshold, scene.gaussians.nodes, scene.gaussians.boxes,
        viewpoint.camera_center.cpu(), torch.zeros((3)),
        interpolation_weights, num_siblings)

    image = torch.clamp(render_post(
        viewpoint, scene.gaussians, pipe,
        torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda"),
        render_indices=indices, parent_indices=parent_indices,
        interpolation_weights=interpolation_weights,
        num_node_kids=num_siblings, use_trained_exp=False
    )["render"], 0.0, 1.0)
    return image

if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--tau', type=float, default=0.0)
    parser.add_argument('--num_images', type=int, default=10)
    args = parser.parse_args(sys.argv[1:])

    dataset, pipe = lp.extract(args), pp.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.active_sh_degree = dataset.sh_degree
    scene = Scene(dataset, gaussians, resolution_scales=[1], create_from_hier=True)

    cameras = scene.getTrainCameras()
    n = len(cameras)
    step = max(1, n // args.num_images)
    indices = list(range(0, n, step))[:args.num_images]

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Rendering {len(indices)} images from {n} cameras (tau={args.tau})")

    psnr_sum = 0.0
    ssim_sum = 0.0
    count = 0

    for i, idx in enumerate(indices):
        cam = cameras[idx]
        print(f"  [{i+1}/{len(indices)}] Rendering camera {idx}: {cam.image_name}")
        image = render_one(cam, scene, pipe, args.tau)

        gt_image = torch.clamp(cam.original_image.to("cuda"), 0.0, 1.0)
        p = psnr(image, gt_image).mean().item()
        s = ssim(image, gt_image).mean().item()
        psnr_sum += p
        ssim_sum += s
        count += 1
        print(f"    PSNR: {p:.2f}, SSIM: {s:.4f}")

        out_path = os.path.join(args.out_dir, f"render_{idx:04d}.png")
        torchvision.utils.save_image(image, out_path)
        gt_path = os.path.join(args.out_dir, f"gt_{idx:04d}.png")
        torchvision.utils.save_image(gt_image, gt_path)
        torch.cuda.empty_cache()

    print(f"\n===== Average over {count} images =====")
    print(f"  PSNR: {psnr_sum/count:.2f}")
    print(f"  SSIM: {ssim_sum/count:.4f}")

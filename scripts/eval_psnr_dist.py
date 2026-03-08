"""Evaluate PSNR distribution for a trained chunk model."""
import sys, os, torch, numpy as np
from argparse import Namespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.image_utils import psnr
from arguments import PipelineParams

model_path = sys.argv[1] if len(sys.argv) > 1 else 'D:/project2_gs/output/trained_chunks/0_0'
sample_n = int(sys.argv[2]) if len(sys.argv) > 2 else 100

with open(os.path.join(model_path, 'cfg_args')) as f:
    cfg = eval(f.read())

# Force CPU data loading to avoid OOM
cfg.data_device = 'cpu'
cfg.resolution = 2  # downsample to avoid large image issues

gaussians = GaussianModel(cfg.sh_degree)
scene = Scene(cfg, gaussians, load_iteration=30000, shuffle=False)
bg = torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda')

# Pipeline params - simple object with required attributes
class Pipe:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
pipe = Pipe()

cameras = scene.getTrainCameras()
step = max(1, len(cameras) // sample_n)
indices = list(range(0, len(cameras), step))[:sample_n]

print(f"Evaluating {len(indices)}/{len(cameras)} views...")

psnrs = []
for i, idx in enumerate(indices):
    cam = cameras[idx]
    with torch.no_grad():
        rendering = render(cam, gaussians, pipe, bg)['render']
        gt = cam.original_image[:3].cuda()
        p = psnr(rendering, gt).item()
    psnrs.append(p)
    del rendering, gt
    torch.cuda.empty_cache()

psnrs = np.array(psnrs)
print(f'\n=== PSNR Distribution ({len(psnrs)} views) ===')
print(f'Mean: {psnrs.mean():.2f}  |  Median: {np.median(psnrs):.2f}')

for pct in [10, 25, 50, 75, 90]:
    print(f'  P{pct}: {np.percentile(psnrs, pct):.2f}')

ranges = [(0,10), (10,15), (15,20), (20,25), (25,30), (30,100)]
print(f'\nHistogram:')
for lo, hi in ranges:
    count = ((psnrs >= lo) & (psnrs < hi)).sum()
    bar = '#' * int(count / len(psnrs) * 50)
    print(f'  [{lo:>2},{hi:>2}): {count:>4} ({count/len(psnrs)*100:5.1f}%) {bar}')

sorted_p = np.sort(psnrs)
n = len(sorted_p)
print(f'\nTop 25% mean:    {sorted_p[int(n*0.75):].mean():.2f}')
print(f'Middle 50% mean: {sorted_p[int(n*0.25):int(n*0.75)].mean():.2f}')
print(f'Bottom 25% mean: {sorted_p[:int(n*0.25)].mean():.2f}')
print(f'去掉最差10%后:   {sorted_p[int(n*0.1):].mean():.2f}')
print(f'去掉最差25%后:   {sorted_p[int(n*0.25):].mean():.2f}')

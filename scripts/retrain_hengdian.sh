#!/bin/bash
# Hengdian retrain with ALL fixes:
# 1. LiDAR initialization (3.78M points instead of 200K COLMAP sparse)
# 2. Depth Anything monocular depth supervision
# 3. Explicit --depths and --init_ply args (bypass full_train.py auto-detection)
# 4. GPU LiDAR depth maps as lidar_depths constraint

set -e

PYTHON=/c/Users/Administrator/miniconda3/envs/gaussian_splatting/python.exe
GSDIR="C:/Users/Administrator/gs_test/hierachy-gs-shire/hierarchy-gs"
cd "$GSDIR"

# Paths
PROJECT=D:/hengdian_gs/cropped_project
COLMAP=$PROJECT/camera_calibration/aligned
CHUNKS=$PROJECT/camera_calibration/chunks
IMAGES=../../rectified/images  # relative to chunk source dir
DEPTHS=../../rectified/depths  # relative to chunk source dir (becomes ../../../rectified/depths after prepend)
LIDAR_DEPTHS=D:/hengdian_gs/gpu_depth_full/depths
INIT_PLY=D:/hengdian_gs/init_points.ply
OUTPUT=D:/hengdian_gs/v2_output

GPU=${1:-1}  # default GPU 1
export CUDA_VISIBLE_DEVICES=$GPU

echo "============================================"
echo "  Hengdian v2 retrain - GPU $GPU"
echo "  Output: $OUTPUT"
echo "============================================"

mkdir -p "$OUTPUT/scaffold" "$OUTPUT/trained_chunks"

#############################
# Step 1: Scaffold training
#############################
SCAFFOLD_PLY="$OUTPUT/scaffold/point_cloud/iteration_30000/point_cloud.ply"
if [ -f "$SCAFFOLD_PLY" ]; then
    echo "[SKIP] Scaffold already done: $SCAFFOLD_PLY"
else
    echo "[STEP 1] Training scaffold with LiDAR init..."
    $PYTHON train_coarse.py \
        -s "$COLMAP" \
        -i ../rectified/images \
        --save_iterations -1 \
        --skybox_num 100000 \
        --init_ply "$INIT_PLY" \
        --model_path "$OUTPUT/scaffold"

    echo "[STEP 1] Scaffold done."
fi

# Verify scaffold PLY exists
if [ ! -f "$SCAFFOLD_PLY" ]; then
    echo "[ERROR] Scaffold PLY not found at $SCAFFOLD_PLY"
    echo "  Checking for .bin/.pt format..."
    if [ -f "$OUTPUT/scaffold/point_cloud/iteration_30000/point_cloud.bin" ]; then
        echo "  Found .bin format. Converting to .ply..."
        $PYTHON -c "
import torch, numpy as np, os
from plyfile import PlyData, PlyElement

path = '$OUTPUT/scaffold/point_cloud/iteration_30000'
xyz = torch.load(os.path.join(path, 'done_xyz.pt')).detach().cpu().numpy()
features_dc = torch.load(os.path.join(path, 'done_dc.pt')).detach().cpu()
features_rest = torch.load(os.path.join(path, 'done_rest.pt')).detach().cpu()
opacity = torch.load(os.path.join(path, 'done_opacity.pt')).detach().cpu().numpy()
scaling = torch.load(os.path.join(path, 'done_scaling.pt')).detach().cpu().numpy()
rotation = torch.load(os.path.join(path, 'done_rotation.pt')).detach().cpu().numpy()

normals = np.zeros_like(xyz)
f_dc = features_dc.transpose(1, 2).flatten(start_dim=1).contiguous().numpy()
f_rest = features_rest.transpose(1, 2).flatten(start_dim=1).contiguous().numpy()

attrs = ['x', 'y', 'z', 'nx', 'ny', 'nz']
for i in range(f_dc.shape[1]): attrs.append(f'f_dc_{i}')
for i in range(f_rest.shape[1]): attrs.append(f'f_rest_{i}')
attrs.append('opacity')
for i in range(scaling.shape[1]): attrs.append(f'scale_{i}')
for i in range(rotation.shape[1]): attrs.append(f'rot_{i}')

dtype_full = [(a, 'f4') for a in attrs]
elements = np.empty(xyz.shape[0], dtype=dtype_full)
all_attrs = np.concatenate((xyz, normals, f_dc, f_rest, opacity, scaling, rotation), axis=1)
elements[:] = list(map(tuple, all_attrs))
el = PlyElement.describe(elements, 'vertex')
PlyData([el]).write(os.path.join(path, 'point_cloud.ply'))
print(f'Converted {len(xyz)} points to PLY')
"
    else
        echo "[FATAL] No scaffold point cloud found!"
        exit 1
    fi
fi

#############################
# Step 2: Train each chunk
#############################
SCAFFOLD_DIR="$OUTPUT/scaffold/point_cloud/iteration_30000"

for CHUNK in 0_0 1_0; do
    SRC="$CHUNKS/$CHUNK"
    DST="$OUTPUT/trained_chunks/$CHUNK"
    HIER="$DST/hierarchy.hier"
    HIER_OPT="$DST/hierarchy.hier_opt"

    if [ -f "$HIER_OPT" ]; then
        echo "[SKIP] Chunk $CHUNK fully done"
        continue
    fi

    # Step 2a: Train chunk
    CHUNK_PLY="$DST/point_cloud/iteration_30000/point_cloud.ply"
    if [ -f "$CHUNK_PLY" ]; then
        echo "[SKIP] Chunk $CHUNK training done"
    else
        echo "[STEP 2a] Training chunk $CHUNK with depth supervision + LiDAR init..."
        $PYTHON -u train_single.py \
            -s "$SRC" \
            -i "$IMAGES" \
            -d "$DEPTHS" \
            --init_ply "$INIT_PLY" \
            --scaffold_file "$SCAFFOLD_DIR" \
            --skybox_locked \
            --bounds_file "$SRC" \
            --model_path "$DST" \
            --save_iterations -1 \
            --port 6029

        # Check for .bin conversion needed
        if [ ! -f "$CHUNK_PLY" ] && [ -f "$DST/point_cloud/iteration_30000/point_cloud.bin" ]; then
            echo "  Converting chunk .bin to .ply..."
            $PYTHON -c "
import torch, numpy as np, os
from plyfile import PlyData, PlyElement

path = '$DST/point_cloud/iteration_30000'
xyz = torch.load(os.path.join(path, 'done_xyz.pt')).detach().cpu().numpy()
features_dc = torch.load(os.path.join(path, 'done_dc.pt')).detach().cpu()
features_rest = torch.load(os.path.join(path, 'done_rest.pt')).detach().cpu()
opacity = torch.load(os.path.join(path, 'done_opacity.pt')).detach().cpu().numpy()
scaling = torch.load(os.path.join(path, 'done_scaling.pt')).detach().cpu().numpy()
rotation = torch.load(os.path.join(path, 'done_rotation.pt')).detach().cpu().numpy()

normals = np.zeros_like(xyz)
f_dc = features_dc.transpose(1, 2).flatten(start_dim=1).contiguous().numpy()
f_rest = features_rest.transpose(1, 2).flatten(start_dim=1).contiguous().numpy()

attrs = ['x', 'y', 'z', 'nx', 'ny', 'nz']
for i in range(f_dc.shape[1]): attrs.append(f'f_dc_{i}')
for i in range(f_rest.shape[1]): attrs.append(f'f_rest_{i}')
attrs.append('opacity')
for i in range(scaling.shape[1]): attrs.append(f'scale_{i}')
for i in range(rotation.shape[1]): attrs.append(f'rot_{i}')

dtype_full = [(a, 'f4') for a in attrs]
elements = np.empty(xyz.shape[0], dtype=dtype_full)
all_attrs = np.concatenate((xyz, normals, f_dc, f_rest, opacity, scaling, rotation), axis=1)
elements[:] = list(map(tuple, all_attrs))
el = PlyElement.describe(elements, 'vertex')
PlyData([el]).write(os.path.join(path, 'point_cloud.ply'))
print(f'Converted {len(xyz)} points to PLY')
"
        fi
    fi

    # Step 2b: Generate hierarchy
    if [ -f "$HIER" ]; then
        echo "[SKIP] Chunk $CHUNK hierarchy exists"
    else
        echo "[STEP 2b] Generating hierarchy for $CHUNK..."
        "$GSDIR/submodules/gaussianhierarchy/build/Release/GaussianHierarchyCreator.exe" \
            "$CHUNK_PLY" "$SRC" "$DST" "$SCAFFOLD_DIR"
    fi

    # Step 2c: Post-optimization
    if [ -f "$HIER_OPT" ]; then
        echo "[SKIP] Chunk $CHUNK post-opt done"
    else
        echo "[STEP 2c] Post-optimizing chunk $CHUNK..."
        $PYTHON -u train_post.py \
            -s "$SRC" \
            -i "$IMAGES" \
            --scaffold_file "$SCAFFOLD_DIR" \
            --hierarchy "$HIER" \
            --model_path "$DST" \
            --iterations 15000 \
            --feature_lr 0.0005 \
            --opacity_lr 0.01 \
            --scaling_lr 0.001 \
            --save_iterations -1 \
            --port 6029
    fi
done

#############################
# Step 3: Consolidation
#############################
MERGED="$OUTPUT/merged.hier"
if [ -f "$MERGED" ]; then
    echo "[SKIP] Consolidation already done"
else
    echo "[STEP 3] Merging hierarchy..."
    "$GSDIR/submodules/gaussianhierarchy/build/Release/GaussianHierarchyMerger.exe" \
        "$OUTPUT/trained_chunks" 0 "$CHUNKS" "$MERGED" 0_0 1_0
fi

echo ""
echo "============================================"
echo "  DONE! Output: $OUTPUT"
echo "============================================"

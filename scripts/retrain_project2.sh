#!/bin/bash
# Project 2 retrain with fix for chunk boundary coverage:
# Root cause: chunk splitting gives only 190K/8.3M points to each chunk.
# Scaffold ring points are LOCKED (--skybox_locked), can't densify.
# Fix: use --init_ply with full LiDAR pointcloud so chunks get dense initialization everywhere.

set -e

PYTHON=/c/Users/Administrator/miniconda3/envs/gaussian_splatting/python.exe
GSDIR="C:/Users/Administrator/gs_test/hierachy-gs-shire/hierarchy-gs"
cd "$GSDIR"

# Paths
PROJECT=D:/project2_gs/project
COLMAP=$PROJECT/camera_calibration/aligned
CHUNKS=$PROJECT/camera_calibration/chunks
IMAGES=../../rectified/images
DEPTHS=../../rectified/depths
INIT_PLY=$COLMAP/sparse/0/points3D.ply    # Full 8.3M LiDAR-sourced points
OUTPUT=D:/project2_gs/v2_output

GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU

echo "============================================"
echo "  Project 2 v2 retrain - GPU $GPU"
echo "  Fix: full LiDAR init for chunks"
echo "  Output: $OUTPUT"
echo "============================================"

mkdir -p "$OUTPUT/scaffold" "$OUTPUT/trained_chunks"

#############################
# Step 0: Verify init_ply has more points than chunk sparse
#############################
echo "[CHECK] Verifying init_ply coverage..."
$PYTHON -c "
from plyfile import PlyData
ply = PlyData.read('$INIT_PLY')
print(f'  init_ply: {len(ply[\"vertex\"]):,} points (full LiDAR)')
for chunk in ['0_0', '1_0']:
    ply2 = PlyData.read(f'$CHUNKS/{chunk}/sparse/0/points3D.ply')
    print(f'  chunk {chunk} sparse: {len(ply2[\"vertex\"]):,} points')
"

#############################
# Step 1: Reuse existing scaffold (already trained with 8.3M LiDAR points)
#############################
SCAFFOLD_PLY="D:/project2_gs/output/scaffold/point_cloud/iteration_30000/point_cloud.ply"
SCAFFOLD_DIR="D:/project2_gs/output/scaffold/point_cloud/iteration_30000"

if [ -f "$SCAFFOLD_PLY" ]; then
    echo "[REUSE] Scaffold from v1: $SCAFFOLD_PLY"
    $PYTHON -c "from plyfile import PlyData; ply=PlyData.read('$SCAFFOLD_PLY'); print(f'  {len(ply[\"vertex\"]):,} Gaussians')"
else
    echo "[ERROR] Scaffold not found at $SCAFFOLD_PLY"
    echo "  Need to train scaffold first or point to correct path."
    exit 1
fi

#############################
# Step 2: Train each chunk with full LiDAR init
#############################
for CHUNK in 0_0 1_0; do
    SRC="$CHUNKS/$CHUNK"
    DST="$OUTPUT/trained_chunks/$CHUNK"
    HIER="$DST/hierarchy.hier"
    HIER_OPT="$DST/hierarchy.hier_opt"

    if [ -f "$HIER_OPT" ]; then
        echo "[SKIP] Chunk $CHUNK fully done"
        continue
    fi

    # Step 2a: Train chunk with full init_ply
    CHUNK_PLY="$DST/point_cloud/iteration_30000/point_cloud.ply"
    if [ -f "$CHUNK_PLY" ]; then
        echo "[SKIP] Chunk $CHUNK training done"
    else
        echo "[STEP 2a] Training chunk $CHUNK with FULL LiDAR init + depth supervision..."
        echo "  init_ply: $INIT_PLY (8.3M points, not chunk's 190K)"
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
            --port 6039
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
            --port 6039
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

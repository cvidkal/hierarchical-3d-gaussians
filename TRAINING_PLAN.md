# Hierarchy-GS 训练计划 (5090)

## 代码仓库
```
git clone https://github.com/cvidkal/hierarchical-3d-gaussians.git
cd hierarchical-3d-gaussians
git checkout feat/lidar-init-fix
```

## 环境依赖
- PyTorch 2.x + CUDA (需要重新编译 submodules)
- `pip install plyfile opencv-python tqdm`
- 编译 submodules:
  ```bash
  cd submodules/hierarchy-rasterizer && pip install . && cd ../..
  cd submodules/gaussianhierarchy && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j && cd ../../..
  ```

---

## 核心问题与修复

### 已诊断的根本原因
Hierarchy-GS 的 chunk 分割会把完整的 LiDAR 点云（如 8.3M）只分配约 190K 给每个 chunk。
边缘相机看到的区域只有 scaffold ring 的稀疏点（被 `--skybox_locked` 锁死无法 densify），
导致渲染出大片黑色，PSNR 低至 6-11，拖垮整体均值到 18。

### 修复方案
使用 `--init_ply` 参数将完整 LiDAR 点云传给每个 chunk 训练，
绕过 chunk splitting 的点云子集限制，让每个视角都有密集的可优化高斯初始化。

### 已验证
- 覆盖好的视角 PSNR 已达 24-25（说明相机参数、FOV 都没问题）
- FOV **不需要裁剪**，直接用去畸变后的完整图像
- Depth Anything 深度监督有效

---

## ⚠️ 关键注意事项（常见误区）

### 1. points3D 不是空的
`prepare_hierarchy_gs.py` / `share_to_colmap.py` 生成的 `points3D.bin`/`points3D.ply`
应该包含下采样的 LiDAR 点（如 ds5cm 的 8.3M 点），不是空的。
这些点是训练初始化的基础。如果 points3D 是空的，训练会失败。

### 2. `--init_ply` 是最关键的参数（必须加！）
**这是我们诊断出的核心修复。** chunk splitting 会把百万级点云砍到只有 ~190K/chunk。
不加 `--init_ply` 的后果：
- 边缘视角只有 5K-12K 个高斯（正常应该 50万+）
- 渲染出大片黑色，PSNR 低至 6-11
- 整体均值被拖到 18（本应 24+）

**scaffold 训练和 chunk 训练都要传 `--init_ply`！**

### 3. LiDAR 深度图格式区别
代码里有两种深度渲染工具，格式不同：

| 工具 | 输出格式 | 用途 | 训练参数 |
|------|---------|------|---------|
| `preprocess/generate_lidar_depth.py` | `.npz` (含 u, v, depth 数组) | LiDAR 稀疏深度约束 | `--lidar_depths` |
| `preprocess/gpu_depth_render.py` | 16-bit PNG (全图深度) | 仅供可视化/调试 | **不兼容训练代码** |

训练代码 (`utils/camera_utils.py`) 只认 `.npz` 格式（含 u, v, depth 键）。
**不要把 gpu_depth_render.py 的 PNG 输出传给 `--lidar_depths`，会被静默忽略。**

### 4. Depth Anything 深度 vs LiDAR 深度
- Depth Anything (`-d` 参数): 16-bit PNG 逆深度图，每个像素都有值，但是**相对深度**（单目估计）
- LiDAR 深度 (`--lidar_depths`): .npz 稀疏深度，只有 LiDAR 能打到的像素有值，但是**绝对精确深度**
- 两者互补：Depth Anything 提供全局几何引导，LiDAR 提供局部精确约束

### 5. FOV 不需要裁剪
之前怀疑过鱼眼裁剪后的宽 FOV 可能影响质量。已验证否定：
- 横店 fx=885 (FOV 103°) 和 Project 2 fx=1475 (FOV 74°) PSNR 问题一样
- 覆盖好的区域已达 24-25 PSNR
- 直接用去畸变后的完整图像训练即可

---

## SHARE SLAM 数据 Pipeline

```
SHARE SLAM 设备输出
  ├── undistort/ImgPose.txt        (相机 c2w pose: position + quaternion)
  ├── undistort/left/              (去畸变左相机图像)
  ├── undistort/left_undistort_intrinsic.txt  (相机内参)
  └── colorized.ply                (LiDAR 全局点云, 1-3亿点)
        │
        ▼
[share_to_colmap.py]
  - ImgPose.txt 的 c2w → COLMAP 的 w2c (四元数共轭 + T = -R_w2c @ pos)
  - 内参 → cameras.bin (PINHOLE 模型)
  - LiDAR 下采样 → points3D.bin (注意: 不是空的!)
  - 输出: aligned/sparse/0/{cameras,images,points3D}.bin
        │
        ▼
[prepare_hierarchy_gs.py]
  - 采样 20万点建伪 2D-3D 对应 (让 auto_reorient/make_chunk 工作)
  - auto_reorient: 对齐坐标系
  - make_chunk: 按空间分块 (100m×100m)
  - 输出: chunks/0_0, chunks/1_0, ...
        │
        ▼
[Depth Anything V2]  →  depths/ 目录 (16-bit PNG 逆深度)
[generate_lidar_depth.py]  →  lidar_depths/ 目录 (.npz 稀疏精确深度)
        │
        ▼
[train_coarse.py]  ← scaffold (--init_ply 传完整 LiDAR 下采样)
[train_single.py]  ← chunk训练 (--init_ply 传完整点云, -d 深度监督)
[GaussianHierarchyCreator]  ← 层级生成
[train_post.py]  ← 后优化
[GaussianHierarchyMerger]  ← 合并所有 chunk
```

---

## 两个数据集

### 1. 横店 (Hengdian) - 传统建筑外景
| 项目 | 值 |
|------|-----|
| 图像数 | 2838 |
| 分辨率 | 2214×2952 (PINHOLE) |
| 焦距 | fx=fy=885 |
| Chunks | 0_0, 1_0 |
| LiDAR (full) | 137M 点 (原始 colorized.ply) |
| LiDAR (下采样) | 3.78M 点 (init_points.ply) |

**数据路径 (需要从旧机器拷贝):**
- COLMAP 项目: `D:/hengdian_gs/cropped_project/`
  - aligned: `cropped_project/camera_calibration/aligned/`
  - chunks: `cropped_project/camera_calibration/chunks/0_0`, `1_0`
  - 图像 (相对路径): `../rectified/images`
  - Depth Anything 深度 (相对路径): `../rectified/depths`
- LiDAR init: `D:/hengdian_gs/init_points.ply` (3.78M, 已对齐到 COLMAP 坐标系)
- 原始 LiDAR: `share-pointclouds-studio/横店/2026-02-09_14-04-12/output/2026-02-09_14-04-12_colorized.ply` (137M)

**5090 建议:** 用原始 137M LiDAR 下采样到 3cm (约10-15M点) 作为 init_ply，比当前 3.78M 更密。
```python
import open3d as o3d
pcd = o3d.io.read_point_cloud("colorized.ply")
pcd_ds = pcd.voxel_down_sample(0.03)  # 3cm
o3d.io.write_point_cloud("init_points_ds3cm.ply", pcd_ds)
```

### 2. Project 2 - 工业厂房/仓库室内
| 项目 | 值 |
|------|-----|
| 图像数 | 5854 (chunk 0_0: 1500, chunk 1_0: ~4354) |
| 分辨率 | 2214×2952 (PINHOLE) |
| 焦距 | fx=1475, fy=1474 |
| Chunks | 0_0, 1_0 |
| LiDAR | 311M 点 (原始), 8.3M (ds5cm, = COLMAP sparse) |

**数据路径 (需要从旧机器拷贝):**
- COLMAP 项目: `D:/project2_gs/project/`
  - aligned: `project/camera_calibration/aligned/`
  - chunks: `project/camera_calibration/chunks/0_0`, `1_0`
  - 图像 (相对路径): `../../rectified/images`
  - Depth Anything 深度 (相对路径): `../../rectified/depths`
- LiDAR init (已对齐): `project/camera_calibration/aligned/sparse/0/points3D.ply` (8.3M)
- 原始 LiDAR: `share-pointclouds-studio/Project_2026-03-04_11-05-22/2026-02-02_15-55-36/output/2026-02-02_15-55-36_colorized.ply` (311M)

**5090 建议:** 311M 下采样到 3cm (约15-20M点) 作为 init_ply。

---

## 训练步骤

### Step 1: Scaffold 训练 (每个数据集一次)
```bash
python train_coarse.py \
    -s <COLMAP_ALIGNED_DIR> \
    -i ../rectified/images \
    --save_iterations -1 \
    --skybox_num 100000 \
    --init_ply <LIDAR_INIT_PLY> \
    --model_path <OUTPUT>/scaffold
```

### Step 2: 每个 Chunk 训练
```bash
SCAFFOLD_DIR=<OUTPUT>/scaffold/point_cloud/iteration_30000

python -u train_single.py \
    -s <CHUNK_DIR> \
    -i <IMAGES_RELATIVE> \
    -d <DEPTHS_RELATIVE> \
    --init_ply <LIDAR_INIT_PLY> \
    --scaffold_file $SCAFFOLD_DIR \
    --skybox_locked \
    --bounds_file <CHUNK_DIR> \
    --model_path <OUTPUT>/trained_chunks/<CHUNK_NAME> \
    --save_iterations -1 \
    --resolution 1 \
    --port 6039
```

**关键参数：**
- `--init_ply`: **必须传完整 LiDAR 点云**，不要用 chunk 自带的 sparse (只有 ~190K)
- `-d`: Depth Anything 深度图路径 (相对于 chunk source dir)
- `--resolution 1`: 5090 有 32GB，可以全分辨率训练
- `--skybox_locked`: 保持 scaffold 一致性
- 可选: `--lidar_depths <LIDAR_DEPTHS_DIR>` (如果已生成 .npz 格式的 LiDAR 深度图)

### Step 3: Hierarchy 生成
```bash
GaussianHierarchyCreator <CHUNK_PLY> <CHUNK_SRC> <CHUNK_DST> $SCAFFOLD_DIR
```

### Step 4: Post-optimization
```bash
python -u train_post.py \
    -s <CHUNK_DIR> \
    -i <IMAGES_RELATIVE> \
    --scaffold_file $SCAFFOLD_DIR \
    --hierarchy <CHUNK_DST>/hierarchy.hier \
    --model_path <CHUNK_DST> \
    --iterations 15000 \
    --feature_lr 0.0005 \
    --opacity_lr 0.01 \
    --scaling_lr 0.001 \
    --save_iterations -1 \
    --port 6039
```

### Step 5: 合并
```bash
GaussianHierarchyMerger <OUTPUT>/trained_chunks 0 <CHUNKS_DIR> <OUTPUT>/merged.hier 0_0 1_0
```

---

## 5090 优化建议

| 项目 | 4090 (当前) | 5090 (建议) |
|------|------------|------------|
| VRAM | 24GB | 32GB |
| init_ply 密度 | ds5cm (8.3M) | **ds3cm (15-20M)** |
| 训练分辨率 | 自动降到 1.6K | **--resolution 1 (全分辨率 2952)** |
| 迭代次数 | 30000 | **45000-60000** (更密的点需要更多迭代) |
| Depth Anything | 有 | 有 |
| LiDAR 深度 | 未使用 | 建议生成 .npz 配合 --lidar_depths |
| FOV 裁剪 | 不需要 | **不需要** |

---

## 评估

训练完成后用以下脚本评估:
```bash
# PSNR 分布 (采样100张)
python scripts/eval_psnr_dist.py <MODEL_PATH> 100

# 直接渲染 (不经过 hierarchy LOD)
python render_direct.py --model_path <CHUNK_MODEL_PATH> --skip_video --gpu <GPU_ID>
```

**目标 PSNR:** 全分辨率 + 密集 LiDAR 初始化，预期均值 **24-27**。

---

## 参考脚本
- `scripts/retrain_hengdian.sh` - 横店完整训练脚本
- `scripts/retrain_project2.sh` - Project 2 完整训练脚本
- 路径需要根据新机器调整

# SimBEV 数据迁移与排障改动记录

本文档记录当前 `thirdparty/bevfusion` 仓库里，为了适配 SimBEV 数据迁移与新环境运行所做的改动。

## 1. 配置改动

### 1.1 `dataset_root` 改为本地相对路径

- 文件: `configs/simbev/default.yaml`
- 改动:
  - 从 `dataset_root: /dataset/simbev/`
  - 改为 `dataset_root: data/simbev/original/`
- 目的:
  - 避免依赖机器上的固定绝对路径 `/dataset/...`。

## 2. 数据集路径解析改动（保留）

### 2.1 `SimBEVDataset` 增强路径解析

- 文件: `mmdet3d/datasets/simbev_dataset.py`
- 主要改动:
  - 规范化 `dataset_root` 与 `ann_file`（支持相对/绝对路径）。
  - 增加 `ann_root`（由 `ann_file` 推导）。
  - 新增 `_resolve_path()` 与 `_resolve_info_paths()`，在加载注释后统一修正路径。
  - 支持常见路径别名映射:
    - `ground-truth` <-> `ground_truth`
    - `sweeps` <-> `samples`
  - 支持将 infos 中的绝对路径前缀映射回本地根目录（例如 `/dataset/...`、`/data/simbev/...`）。

### 2.2 当前已确认的回退策略

- 已回退: 为兼容 `frame mismatch` 增加的特殊逻辑（例如 frame 编号重映射）已移除。
- 保留: 上述相对路径/根路径映射能力继续保留。

## 3. PyTorch 2.7 / CUDA 新环境适配改动

这部分是运行时/编译兼容性改动，与 SimBEV 标注路径问题是两条线。

### 3.1 `setup.py` 构建参数调整

- 文件: `setup.py`
- 改动:
  - 去掉固定 `sm_70/sm_75/sm_80/sm_86` 的 `-gencode`。
  - `bev_pool_ext` 编译标准由 `-std=c++14` 改为 `-std=c++17`。
- 目的:
  - 减少旧架构写死导致的新卡不兼容概率。
  - 兼容新版本编译链与 PyTorch C++ 扩展要求。

### 3.2 点云相关 CUDA 算子头文件兼容修改

- 文件:
  - `mmdet3d/ops/ball_query/src/ball_query.cpp`
  - `mmdet3d/ops/furthest_point_sample/src/furthest_point_sample.cpp`
  - `mmdet3d/ops/gather_points/src/gather_points.cpp`
  - `mmdet3d/ops/group_points/src/group_points.cpp`
  - `mmdet3d/ops/interpolate/src/interpolate.cpp`
  - `mmdet3d/ops/knn/src/knn.cpp`
- 改动:
  - 移除旧 `THC/THC.h` 依赖。
  - 改为 `ATen/cuda/...` 相关头文件。
  - 注释掉 `extern THCState *state;`。
- 目的:
  - 适配新版本 PyTorch 对 THC 接口移除后的编译环境。

### 3.3 `spconv` 注册冲突兼容

- 文件: `mmdet3d/ops/spconv/conv.py`
- 改动:
  - 多处 `@CONV_LAYERS.register_module()` 改为 `@CONV_LAYERS.register_module(force=True)`。
- 目的:
  - 避免重复注册时直接报错，提升运行兼容性。

## 4. 数据 mismatch 问题结论（当前）

### 4.1 现象

- 评估时报错:
  - `FileNotFoundError: ... /dataset/nuscenes_val/worker_0/simbev/ground-truth/det/... does not exist`

### 4.2 判断

- 该错误的直接原因是:
  - infos 中记录的路径前缀与本地目录结构不一致。
- 与 frame mismatch 已做拆分:
  - frame mismatch 兼容代码已回退，不再在 bevfusion 侧“猜测修复”帧号。

## 5. 建议的评估命令写法

建议显式指定 `dataset_root` 与 `ann_file`，避免隐式读到旧 infos:

```bash
torchpack dist-run -np 1 python tools/test.py \
  configs/simbev/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  checkpoints/simbev-bevfusion-det.pth --eval bbox \
  --cfg-options \
  dataset_root=data/simbev/setup/nuscenes_val/ \
  data.test.ann_file=data/simbev/setup/nuscenes_val/infos/simbev_infos_val.json
```

## 6. 仍需关注的问题

- 若仍出现 `CUDA error: no kernel image is available for execution on the device`，说明至少一个自定义 CUDA 扩展未正确支持当前 GPU 架构（如 `sm_120`），需要继续核查该扩展的实际编译产物与架构目标。

## 7. Waymo 多相机与侧视 Pad 适配（新增）

### 7.1 动态相机数量/名称支持

- 文件: `mmdet3d/datasets/simbev_dataset.py`
- 改动:
  - 不再使用固定 6 相机列表。
  - 每个 sample 运行时根据 infos 的 `RGB-*` 键与 metadata 相机外参键交集，动态确定相机列表。
  - 相机内参优先使用 `metadata['camera_intrinsics_by_name'][camera]`，若缺失则回退到 `metadata['camera_intrinsics']`。
- 结果:
  - 支持 `waymo_val` 的 5 相机命名（含 `CAM_SIDE_LEFT` / `CAM_SIDE_RIGHT`）。

### 7.2 图像加载阶段统一 pad 与主点修正

- 文件: `mmdet3d/datasets/pipelines/loading.py`
- 改动:
  - `LoadMultiViewImageFromFiles` 新增参数:
    - `pad_to_max_shape`（是否 pad 到同一分辨率）
    - `pad_value`（补边像素值）
    - `pad_align`（当前支持 `center`）
  - 对多相机图像按样本内最大尺寸执行 pad（waymo 侧视图上下居中补边）。
  - 记录 `img_pad_offsets` 与 `ori_shapes`。
  - 同步修正每个相机内参主点（`cx/cy`）并重算 `lidar2image`。
- 结果:
  - 进入后续 pipeline 前，多相机图像尺寸一致，几何关系保持一致。

### 7.3 配置开关与 waymo 专用配置

- 文件:
  - `configs/simbev/default.yaml`
  - `configs/simbev/det/transfusion/secfpn/camera+lidar/swint_v0p075/waymo_val.yaml`
- 改动:
  - 在 `configs/simbev/default.yaml` 新增通用开关:
    - `img_pad_to_max_shape`（默认 `false`）
    - `img_pad_value`（默认 `0`）
    - `img_pad_align`（默认 `center`）
  - train/test pipeline 的 `LoadMultiViewImageFromFiles` 读取上述开关。
  - 新增 `waymo_val.yaml`，仅对 waymo 数据启用 pad（`img_pad_to_max_shape: true`）并设置 `dataset_root`。
- 结果:
  - 现有 nuscenes/simbev 流程默认不受影响；waymo 评估可通过专用配置启用 pad。

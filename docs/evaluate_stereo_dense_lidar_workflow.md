# `evaluate_stereo_dense_lidar.py` 工作流说明

这个脚本用于在 Middlebury 风格数据上评估 DEFOM-Stereo，并把外部的稠密 LiDAR 视差图作为初始输入注入到迭代优化流程里。

## 1. 这个脚本做什么

它和 `evaluate_stereo.py` 的区别是：

- `evaluate_stereo.py` 走模型内部的 DAV2 初始化
- `evaluate_stereo_dense_lidar.py` 可以改成读取 `disp0_lidar_dense.png` 这类外部稠密 LiDAR 视差图

因此它适合对比两种初始化方式：

1. LiDAR init
2. DAV2 init

## 2. 数据要求

### 2.1 官方 Middlebury 目录

直接使用仓库里现有的 `datasets.Middlebury` 读取逻辑，常见 split 如：

- `2014`
- `2021`
- `F`
- `H`
- `Q`

### 2.2 自定义平铺目录

当使用 `--split custom` 时，目录形状应类似：

```text
middlebury_root/
  SceneA/
    im0.png
    im1.png
    disp0GT.pfm or disp0_gt.png or disp0.pfm
```

LiDAR 视差图可以放在同一个场景目录里，也可以放到单独的根目录里。

## 3. 路径参数结构

这几个参数的分工不同：

- `--middlebury_root`：读左右图和 GT 的根目录
- `--lidar_root`：读 LiDAR disparity 的根目录
- `--lidar_disp_name`：每个场景里 LiDAR 文件名
- `--gt_disp_name`：`custom` 模式下 GT 文件名

### 3.1 不传 `--lidar_root`

脚本会直接在左图所在场景目录里找 LiDAR 文件：

```text
middlebury_root/
  <scene>/
    im0.png
    im1.png
    disp0GT.pfm
    disp0_lidar_dense.png
```

### 3.2 传了 `--lidar_root`

脚本会按“场景名”去另一套目录里找 LiDAR 文件：

```text
middlebury_root/
  <scene>/
    im0.png
    im1.png
    disp0GT.pfm

lidar_root/
  <scene>/
    disp0_lidar_dense.png
```

这里还保留一层 `<scene>`，是为了让每个场景的 `im0/im1/GT/LiDAR` 一一对应。脚本不是按文件顺序配对，而是按场景名配对。

### 3.3 常见组织方式

```text
middlebury_root/
  Adirondack-perfect/
    im0.png
    im1.png
    disp0_gt.png

lidar_root/
  Adirondack-perfect/
    disp0_lidar_dense.png
```

## 4. 参数含义

- `--restore_ckpt`：模型权重路径
- `--middlebury_root`：数据根目录
- `--split`：评估拆分方式
- `--lidar_disp_name`：每个场景里的 LiDAR 视差文件名
- `--gt_disp_name`：`custom` 模式下的 GT 文件名
- `--lidar_fill`：LiDAR 无效像素的填充策略
- `--compare_dav2`：是否同时跑一遍 DAV2 init

## 5. 运行流程

1. 解析命令行参数
2. 按参数构建 `DEFOMStereo(args)`
3. 加载 checkpoint
4. 构建评估数据集
5. 对每个场景：
   - 读取左右图
   - 读取 GT disparity
   - 可选读取 LiDAR disparity
   - 对输入做 padding
   - 调用模型推理
   - 统计 EPE、bad pixel、FPS
6. 汇总整套 split 的平均指标

## 6. 代码主入口

### 6.1 `_read_gt_disp(path)`

读取 GT disparity。

- `.pfm` 走 Middlebury 原始读取
- `.png` 走 `read_lidar_disp_png`

这样就能兼容官方 GT 和转换后的 PNG GT。

### 6.2 `MiddleburyDenseLidarEval`

这是给自定义平铺目录用的数据集适配器。

它只负责：

- 扫描 `root/*/im0.png`
- 检查 `im1.png`
- 找 GT disparity
- 记录路径

真正的图像读取仍然交给父类 `StereoDataset`。

### 6.3 `build_eval_dataset(args)`

根据 `--split` 选择数据集：

- `custom` -> `MiddleburyDenseLidarEval`
- 其他 split -> `datasets.Middlebury`

### 6.4 `load_lidar_batch(...)`

读取单个场景的 LiDAR disparity，并对齐到左图尺寸。

返回：

- `disp`
- `valid`
- `lidar_path`

### 6.5 `run_inference(...)`

统一封装一次前向推理。

它会：

1. pad 左右图
2. 若启用 LiDAR，再 pad LiDAR init
3. 调用 `model(..., test_mode=True)`
4. 去掉 padding

### 6.6 `validate_middlebury_lidar(...)`

这是主评估循环。

它会逐场景计算：

- EPE
- bad pixel rate
- runtime

最后输出 split 级别平均值。

## 7. 典型命令

### 7.1 评估 Middlebury 2014

```bash
python evaluate_stereo_dense_lidar.py \
  --restore_ckpt checkpoints/defomstereo_vitl_middlebury.pth \
  --middlebury_root ./datasets/Middlebury \
  --split 2014 \
  --lidar_disp_name disp0_lidar_dense.png
```

### 7.2 自定义平铺数据

```bash
python evaluate_stereo_dense_lidar.py \
  --restore_ckpt checkpoints/defomstereo_vitl_middlebury.pth \
  --middlebury_root ./dataset \
  --split custom \
  --gt_disp_name disp0_gt.png \
  --lidar_disp_name disp0_lidar_dense.png
```

### 7.3 同时对比 DAV2 init

```bash
python evaluate_stereo_dense_lidar.py \
  --restore_ckpt checkpoints/defomstereo_vitl_middlebury.pth \
  --middlebury_root ./datasets/Middlebury \
  --split 2014 \
  --lidar_disp_name disp0_lidar_dense.png \
  --compare_dav2
```

## 8. 代码阅读顺序

如果要快速理解脚本，建议按这个顺序看：

1. 顶部模块说明
2. `validate_middlebury_lidar(...)`
3. `run_inference(...)`
4. `load_lidar_batch(...)`
5. `build_eval_dataset(args)`
6. `MiddleburyDenseLidarEval`

这个顺序基本就是实际运行顺序。

# Tube DEFOM-Stereo Workflow

## 目标

对 `tube/<日期>/left` 和 `tube/<日期>/right` 下的双目图像逐对处理：

1. 使用 `DEFOM-Stereo` 预测视差。
2. 使用视差和双目标定参数生成点云。
3. 将视差结果和点云统一保存到 `tube/<日期>/result`。

这个流程面向服务器推理，不要求在当前机器上直接执行。

## DEFOM-Stereo 基本原理

结合仓库根目录 [README.md](/D:/project/DEFOM-Stereo/README.md:11) 和实现文件 [core/defom_stereo.py](/D:/project/DEFOM-Stereo/core/defom_stereo.py:1)，DEFOM-Stereo 的核心逻辑可以概括为：

1. 左右图分别进入特征编码器。
2. 编码器不仅提取常规 stereo matching 特征，也融合 depth foundation model 的单目几何先验。
3. 单目分支提供初始深度/逆深度，再换算为初始视差。
4. 主干采用 RAFT-style 的迭代更新方式，不断查询相关性体并细化视差。
5. 额外的 scale update 模块用于修正单目先验的尺度误差，得到最终视差图。

工程上要注意一点：`DEFOM-Stereo` 只适合在极线对齐后的双目图上工作。因此 `tube` 的原始左右图若还带有畸变、相机姿态差异，就需要先做双目矫正，再送入网络。

## tube 数据的处理思路

以 `tube/260625` 为例，目录结构大致是：

```text
tube/
  260625/
    left/
    right/
    stereo_cam/
      cam0.txt
      cam1.txt
      camrt.txt
```

建议流程如下：

1. 遍历 `left` 与 `right`，按同名文件配对。
2. 读取 `stereo_cam` 中的相机内参、畸变参数和双目标定外参。
3. 对每对图像做去畸变和双目极线矫正。
4. 根据命令行参数选择 checkpoint。
5. 根据命令行参数选择是否对原图缩放后再推理。
6. 将预测出的低分辨率视差恢复到矫正后的原图分辨率。
7. 用恢复后的视差和矫正得到的 `Q` 矩阵反投影点云。
8. 保存视差和点云到 `tube/<日期>/result`。

## 标定文件的使用方式

当前 `tube/260625/stereo_cam` 下的文件可解释为：

- `cam0.txt`
  - 第 1 行：左相机 `3x3` 内参矩阵
  - 第 2 行：左相机畸变参数
- `cam1.txt`
  - 第 1 行：右相机 `3x3` 内参矩阵
  - 第 2 行：右相机畸变参数
- `camrt.txt`
  - 第 1 行：右相机相对左相机的旋转矩阵 `R`
  - 第 2 行：平移向量 `T`

脚本里可以直接用 OpenCV 完成几何预处理：

1. `cv2.stereoRectify()` 生成 `R1/R2/P1/P2/Q`
2. `cv2.initUndistortRectifyMap()` 生成 remap 表
3. `cv2.remap()` 得到矫正后的左右图
4. `cv2.reprojectImageTo3D()` 根据 `Q` 和视差得到点云

## 为什么要暴露缩放接口

`tube` 图像目前是 `5120x5120`。这类大图直接跑 `DEFOM-Stereo`，显存和耗时压力都很大，尤其在 `vitl` checkpoint 下更明显。

因此脚本应该暴露两个层面的控制：

1. `--restore-ckpt`
   - 允许自由切换 `vits` / `vitl` 等 checkpoint
2. `--resize-width` 与 `--resize-height`
   - 明确指定送入网络的推理尺寸
   - 若不指定，则默认使用矫正后的原始分辨率

缩放后的视差恢复规则要注意：

- 图像从 `(W0, H0)` 缩放到 `(W1, H1)` 后，网络输出的是缩放图上的视差。
- 将视差插值回原分辨率时，视差值本身还要乘以 `W0 / W1`。
- 因为视差是水平像素位移，本质上跟宽度尺度绑定，而不是和高度单独绑定。

## 输出设计

建议输出目录：

```text
tube/<date>/result/
  disparity/
  pointcloud/
```

每张图建议至少输出：

- `disparity/<stem>_disp.npy`
  - 原始 `float32` 视差
- `disparity/<stem>_disp_u16.png`
  - 按 `disp * 256` 编码后的 `uint16` 视差图
- `disparity/<stem>_disp_vis.png`
  - 伪彩可视化图
- `pointcloud/<stem>.ply`
  - 彩色点云

这样既能保留可视化结果，也能保留后续可计算的原始视差。

## 命令行接口建议

脚本建议支持这些参数：

```text
--tube-root
--tube-date
--restore-ckpt
--dinov2-encoder
--resize-width
--resize-height
--valid-iters
--scale-iters
--mixed-precision
--min-disp
```

其中：

- `--restore-ckpt`：切换不同训练权重
- `--dinov2-encoder`：与 checkpoint 对应，如 `vits` / `vitl`
- `--resize-width` / `--resize-height`：显式指定推理分辨率
- `--min-disp`：生成点云时过滤近似无效的视差

## 建议的服务器运行方式

```powershell
python -m tools.run_tube_defom `
  --tube-date 260625 `
  --restore-ckpt checkpoints/defomstereo_vits_sceneflow.pth `
  --dinov2-encoder vits `
  --resize-width 1024 `
  --resize-height 1024 `
  --extractor-module extractor_defom
```

如果服务器显存更大，可以直接提高推理分辨率，或者切到 `vitl` checkpoint。

## 实现约束

当前机器不实际运行推理，原因很明确：

- 当前 `tube` 图像分辨率过高
- 本机显卡带不动完整批处理
- 本次目标是整理思路并把服务器可执行脚本准备好

因此当前交付应当是：

1. 一份说明思路的 markdown
2. 一个可在服务器上执行的批处理脚本

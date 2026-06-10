# RAFT-Stereo 模型架构详解

## 模型概述

RAFT-Stereo 是一个用于立体视觉匹配的深度学习模型，通过迭代的光流估计方法来计算视差(disparity)。模型主要包括：
- **特征提取器 (Feature Extractors)**：使用 ResNet 编码器
- **相关性计算 (Correlation Block)**：计算左右图像特征间的相关性
- **迭代更新 (GRU Update)**：通过多次迭代优化视差估计

---

## 完整数据流架构

```
输入图像对 (Stereo Image Pair)
       ↓
┌──────────────────────────────────────────────┐
│  图像预处理 (Image Preprocessing)            │
│  左右图像: (B, 3, H, W) → [-1, 1]范围         │
└────────────────┬─────────────────────────────┘
                 ↓
         ┌───────────────────┐
         │  特征提取模块      │
         │ (Feature Extractor)│
         └────────┬──────────┘
                  ↓
    ┌─────────────────────────────┐
    ↓                             ↓
┌──────────────────┐      ┌──────────────────┐
│ Context Network  │      │  Feature Network │
│   (cnet)         │      │     (fnet)       │
│ MultiBasicEncoder│      │  BasicEncoder    │
└────────┬─────────┘      └────────┬─────────┘
         │                        │
    出力│(B,128,H/8,W/8)   出力│(B,256,H/8,W/8)
       └─────────┬────────────────┘
                 ↓
      ┌──────────────────────┐
      │ Correlation Block    │
      │  (相关性体积构建)     │
      │ 1D Correlation       │
      └──────────┬───────────┘
                 ↓
┌─────────────────────────────────────────────────┐
│       迭代更新循环 (Iterative Refinement)       │
│         Iterations (默认: 12 次迭代)            │
├─────────────────────────────────────────────────┤
│ 每次迭代步骤:                                   │
│                                                 │
│  ① 采样相关性特征                              │
│     corr_sampler(corr_volume, coords, radius)  │
│                                                 │
│  ② 编码运动信息                                │
│     MotionEncoder(flow, corr) → motion_feat    │
│                                                 │
│  ③ ConvGRU 更新隐状态                          │
│     h = ConvGRU(h, context, motion_feat)       │
│                                                 │
│  ④ 预测流增量                                  │
│     delta_flow = FlowHead(h)                    │
│                                                 │
│  ⑤ 更新坐标                                    │
│     coords = coords + delta_flow                │
│     (注: 立体模式下仅更新水平方向)              │
│                                                 │
│  ⑥ 上采样预测 (仅最后一次迭代)                │
│     flow_up = upsample_flow(coords, mask)      │
│                                                 │
└─────────────────────────────────────────────────┘
                 ↓
            最终输出
        (B, 1, H, W)
        视差估计图
```

---

## 各模块详细说明

### 1. 输入预处理

```
原始输入图像                    处理后
━━━━━━━━━━━━━━                 ━━━━━━━━
左图: (B, 3, H, W)             (B, 3, H, W)
右图:   uint8 [0,255]     →    float32 [-1, 1]

操作: image = 2 * (image / 255.0) - 1.0
```

### 2. 特征提取网络

#### Context Network (cnet) - MultiBasicEncoder

```
┌─────────────────────────────────────────┐
│  输入: image1 或 [image1, image2]       │
│        (B, 3, H, W)                     │
└──────────────┬──────────────────────────┘
               ↓
    ┌──────────────────────────┐
    │ Conv2d(3, 64, 7×7, s=2)  │  stride=2 (if downsample > 2)
    │ + GroupNorm(8, 64)       │  否则 stride=1
    │ + ReLU                   │
    └──────────────┬───────────┘
          ↓ (B, 64, H/2, W/2) 或 (B, 64, H, W)
    ┌──────────────────────────┐
    │ Layer1: ResidualBlock    │  [64→64]
    │ × 2 (stride=1)           │
    └──────────────┬───────────┘
          ↓ (B, 64, H/2, W/2)
    ┌──────────────────────────┐
    │ Layer2: ResidualBlock    │  [64→96]
    │ × 2 (stride=1 or 2)      │  stride=2 (if downsample > 1)
    └──────────────┬───────────┘
          ↓ (B, 96, H/4, W/4) 或 (B, 96, H/2, W/2)
    ┌──────────────────────────┐
    │ Layer3: ResidualBlock    │  [96→128]
    │ × 2 (stride=1 or 2)      │  stride=2 (if downsample > 0)
    └──────────────┬───────────┘
          ↓ (B, 128, H/8, W/8)
    ┌──────────────────────────────────────┐
    │ MultiBasicEncoder 多尺度输出         │
    │                                      │
    │ outputs08: Conv2d(128, hidden_dim, 3)│ (H/8)
    │ outputs16: Conv2d(128, hidden_dim, 3)│ (H/16)
    │ outputs32: Conv2d(128, hidden_dim, 3)│ (H/32)
    │                                      │
    │ 每层输出形式: (net, inp) 对           │
    └──────────────┬──────────────────────┘
    
输出: [(h, c), (h, c), ...]  × n_gru_layers
其中 h: (B, hidden_dim[i], H/2^k, W/2^k)  [net初始隐状态]
     c: (B, hidden_dim[i], H/2^k, W/2^k)  [inp初始输入]

说明: MultiBasicEncoder 输出多个尺度特征，用于支持多层GRU
      最后再通过 context_zqr_convs 处理成 Z,Q,R 三个门控信号
```

#### Feature Network (fnet) - BasicEncoder

```
┌──────────────────────────────────────┐
│  输入: image1 和 image2               │
│        (B, 3, H, W) × 2              │
└──────────────┬───────────────────────┘
               ↓ 分别处理
     ┌─────────────────────┐
     │ Conv2d(3, 64, 7×7)  │  stride=2
     │ + InstanceNorm      │
     │ + ReLU              │
     └──────────┬──────────┘
           ↓ (B, 64, H/2, W/2)
     ┌─────────────────────┐
     │ ResidualBlock       │
     │ × 2: 64→64         │
     └──────────┬──────────┘
           ↓ (B, 64, H/2, W/2)
     ┌─────────────────────┐
     │ ResidualBlock       │
     │ × 2: 64→96         │ stride=2
     └──────────┬──────────┘
           ↓ (B, 96, H/4, W/4)
     ┌─────────────────────┐
     │ ResidualBlock       │
     │ × 2: 96→128        │ stride=2
     └──────────┬──────────┘
           ↓ (B, 128, H/8, W/8)
     ┌─────────────────────┐
     │ Conv2d(128, 256, 1) │
     └──────────┬──────────┘

输出: fmap1, fmap2  (B, 256, H/8, W/8) × 2
```

### 3. 相关性体积构建 (Correlation Volume)

```
┌────────────────────────────────────────┐
│ 输入特征:                              │
│ fmap1: (B, 256, H/8, W/8)  左图特征   │
│ fmap2: (B, 256, H/8, W/8)  右图特征   │
└──────────────┬─────────────────────────┘
               ↓
    ┌──────────────────────────────────┐
    │ 全对相关性计算 (All-Pairs Corr)  │
    │ corr = einsum('BIRC,BIRC→BRRC')  │
    │ 其中 I=H/8, R=W/8                │
    └──────────────┬───────────────────┘
        ↓ (B, H/8, W/8, 1, W/8)
    ┌──────────────────────────────────┐
    │ 归一化                           │
    │ corr = corr / sqrt(256)          │
    └──────────────┬───────────────────┘
        ↓ (B, H/8, W/8, 1, W/8)
    ┌──────────────────────────────────┐
    │ 创建金字塔结构 (num_levels=4)    │
    │ 通过2倍池化逐级下采样             │
    └──────────────┬───────────────────┘

金字塔输出:
Level 0: (B, H/8, W/8, 1, W/8)
Level 1: (B, H/8, W/8, 1, W/16)
Level 2: (B, H/8, W/8, 1, W/32)
Level 3: (B, H/8, W/8, 1, W/64)
```

### 4. 坐标初始化

```
┌───────────────────────────────┐
│ 初始化网格坐标                │
│ coords0, coords1             │
└──────────────┬────────────────┘
               ↓
    ┌──────────────────────┐
    │ 创建网格 (H/8, W/8)  │
    │ 坐标范围: [0, W/8-1] │
    └──────────────┬───────┘
        ↓ (B, 2, H/8, W/8)
        
初始值: coords1 = coords0
后续用 flow = coords1 - coords0 表示视差
```

### 5. 迭代更新循环

#### 第 i 次迭代结构

```
┌─────────────────────────────────────────────────┐
│          迭代 i (前置条件已准备好)              │
├─────────────────────────────────────────────────┤
│                                                 │
│  1️⃣  相关性采样                                 │
│     ├─ 输入: corr_pyramid, coords1, radius=4  │
│     │                                         │
│     └─ 对每个金字塔层级:                       │
│        ├─ 在当前金字塔层级采样                 │
│        │  采样大小: [2×radius+1] × levels     │
│        │                                     │
│        └─ 级联所有层级特征                     │
│           输出: (B, 288, H/8, W/8)            │
│           其中 288 = 4 levels × (2×4+1)      │
│                                                 │
│  2️⃣  运动编码 (MotionEncoder)                  │
│     ├─ 输入: flow (B, 2, H/8, W/8)           │
│     │         corr (B, 288, H/8, W/8)        │
│     │                                         │
│     ├─ 处理相关特征:                          │
│     │  Conv2d(288, 64, 1)                     │
│     │  Conv2d(64, 64, 3)                      │
│     │  输出: (B, 64, H/8, W/8)                │
│     │                                         │
│     ├─ 处理流:                                │
│     │  Conv2d(2, 64, 7)                       │
│     │  Conv2d(64, 64, 3)                      │
│     │  输出: (B, 64, H/8, W/8)                │
│     │                                         │
│     └─ 融合特征:                              │
│        Conv2d(128, 128, 3)                    │
│        输出: motion_features (B, 128, H/8, W/8)│
│                                                 │
│  3️⃣  GRU 隐状态更新 (ConvGRU)                  │
│     ├─ 输入: h_prev: (B, 128, H/8, W/8)     │
│     │         context: (B, 128, H/8, W/8)    │
│     │         motion_feat: (B, 128, H/8, W/8)│
│     │                                         │
│     ├─ ConvGRU(h, c_z, c_r, c_q, x):        │
│     │  cat([h, x]) → (B, 256, H/8, W/8)      │
│     │  z = sigmoid(Conv(hx) + c_z)           │
│     │  r = sigmoid(Conv(hx) + c_r)           │
│     │  q = tanh(Conv([r*h, x]) + c_q)        │
│     │  h_new = (1-z)*h + z*q                 │
│     │                                         │
│     └─ 输出: h (B, 128, H/8, W/8)            │
│                                                 │
│  4️⃣  流增量预测 (FlowHead)                     │
│     ├─ 输入: h (B, 128, H/8, W/8)            │
│     │                                         │
│     ├─ Conv2d(128, 256, 3)                    │
│     │ + ReLU                                 │
│     │ Conv2d(256, 2, 3)                      │
│     │                                         │
│     └─ 输出: delta_flow (B, 2, H/8, W/8)    │
│                                                 │
│  5️⃣  立体约束与坐标更新                        │
│     ├─ 应用立体约束:                          │
│     │  delta_flow[:, 1] = 0  (垂直方向为0)   │
│     │                                         │
│     └─ 更新坐标:                              │
│        coords1 = coords1 + delta_flow         │
│        输出: (B, 2, H/8, W/8)                │
│                                                 │
│  6️⃣  上采样 (仅最后一次迭代)                  │
│     ├─ 输入: 视差 (B, 2, H/8, W/8)           │
│     │         上采样掩码 (B, 9, H/8, W/8)    │
│     │                                         │
│     ├─ 使用凸组合上采样:                      │
│     │  factor = 2^n_downsample = 8            │
│     │  掩码: (B, 1, 9, 8, 8, H/8, W/8)       │
│     │  流unfold: (B, 2, 9, 1, 1, H/8, W/8)   │
│     │                                         │
│     └─ 输出: flow_up (B, 1, H, W)            │
│             [只输出视差的第一个分量]          │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 6. 最终输出

```
┌──────────────────────────────┐
│  测试模式 (test_mode=True)    │
├──────────────────────────────┤
│ 返回:                        │
│ - coords1 - coords0          │
│   (B, 2, H/8, W/8)          │
│ - flow_up (B, 1, H, W)      │
│   上采样到原始分辨率          │
│   值为视差(disparity)         │
└──────────────────────────────┘

┌──────────────────────────────┐
│  训练模式                    │
├──────────────────────────────┤
│ 返回:                        │
│ flow_predictions =           │
│   [flow_up_iter1,            │
│    flow_up_iter2,            │
│    ...,                      │
│    flow_up_iterN]            │
│                              │
│ 其中每个元素:                │
│ (B, 1, H, W)                │
└──────────────────────────────┘
```

---

## 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_downsample` | **2** | 下采样倍数，控制特征空间分辨率 (2→H/4, 3→H/8) |
| `n_gru_layers` | 3 | GRU层数 |
| `hidden_dims` | [32, 64, 128] | 各GRU层的隐维度 |
| `corr_radius` | 4 | 相关采样半径 |
| `corr_levels` | 4 | 相关金字塔层级数 |
| `iters` | 12 | 迭代次数 |
| `context_norm` | 'batch' | 上下文网络归一化方式 |
| `shared_backbone` | False | 是否共享左右图特征提取 |

**⚠️ `n_downsample` 参数说明**：
- `n_downsample=2`：特征空间为原图的 **1/4** (H/4, W/4) - 训练时默认值，内存占用较少
- `n_downsample=3`：特征空间为原图的 **1/8** (H/8, W/8) - 精度更高，但显存占用约2倍

---

## 内存和计算复杂度估计

### 假设输入: 420×540 (典型KITTI分辨率)

```
特征空间尺寸: 105×135 (约H/4, W/4，当n_downsample=2时)
或 52×67 (约H/8, W/8，当n_downsample=3时)

1️⃣  特征提取
   ├─ Context特征: 3 × (B×128×52×67×4) ≈ 35 MB
   └─ Feature特征: 2 × (B×256×52×67×4) ≈ 36 MB

2️⃣  相关性体积
   ├─ 全对相关: (B×52×67×1×67×4) ≈ 18 MB
   └─ 金字塔: ~18 MB (共4层)

3️⃣  单次迭代激活
   ├─ 相关特征: (B×288×52×67×4) ≈ 40 MB
   ├─ 隐状态: 3 × (B×128×52×67×4) ≈ 35 MB
   └─ 流: (B×2×52×67×4) ≈ 1.8 MB

4️⃣  迭代12次总计
   └─ 中间激活: ~150-200 MB
   
总内存占用: ~300-400 MB (B=1时)
```

---

## 输入输出规范

### 输入
- **类型**: `torch.Tensor` (uint8 图像)
- **形状**: `(B, 3, H, W)`
- **值域**: `[0, 255]`
- **说明**: 左右立体图像对

### 输出
- **类型**: `torch.Tensor` (float32)
- **形状**: `(B, 1, H, W)` 或 `(B, 2, H, W)`
- **值域**: 通常 `[0-100]` (视差值)
- **说明**: 视差估计图

---

## 核心创新点

1. **1D 相关性** - 利用立体约束，将2D相关简化为1D，大幅降低计算
2. **迭代更新** - 通过多次GRU迭代逐步细化视差估计
3. **多尺度相关** - 使用金字塔结构在多个尺度上匹配
4. **立体约束** - 强制垂直方向流为0 (`delta_flow[:, 1] = 0`)
5. **凸组合上采样** - 学习软掩码进行柔和上采样

---

## 关于两个 Encoder 最后的卷积层

### 为什么要加最后的卷积？

#### 1. `cnet` (MultiBasicEncoder) 的多尺度输出
```
Layer3 输出: (B, 128, H/d, W/d)

分支到三个不同尺度:
├─ outputs08: Conv(128, hidden_dim[i], 3) → (B, hidden_dim[i], H/8, W/8)
├─ outputs16: Conv(128, hidden_dim[i], 3) → (B, hidden_dim[i], H/16, W/16)
└─ outputs32: Conv(128, hidden_dim[i], 3) → (B, hidden_dim[i], H/32, W/32)

作用: 为不同分辨率的GRU层提供适配的特征
      根据 n_gru_layers 选择 1/2/3 层输出
```

这里用 **3×3 卷积** 而非 1×1 的原因：
- 需要整合多尺度信息
- 3×3 卷积有更大感受野
- 保留空间上下文信息

#### 2. `fnet` (BasicEncoder) 的单一输出
```
Layer3 输出: (B, 128, H/d, W/d)

最后: Conv2d(128, 256, kernel_size=1)

输出: (B, 256, H/d, W/d) → 用于相关性计算
```

这里用 **1×1 卷积** 的原因：
- 仅做通道映射（128 → 256）
- 不需要感受野扩大
- 1×1 卷积是通道维度的"全连接"
- 计算量小、参数少

### 总结对比

| Encoder | 最后卷积 | 作用 | 输出数量 |
|---------|---------|------|---------|
| `cnet` | 3×3 (多个) | 多尺度特征分支 | 1-3 个 (按 n_gru_layers) |
| `fnet` | **1×1** | 通道映射 | 1 个 (左右共用) |

**关键差异**：
- `cnet` 需要 **多尺度、多分支** 输出 → 用 3×3 卷积
- `fnet` 需要 **通道压缩** → 用 1×1 卷积

---


# DEFOM Stereo 模型架构详解

## 模型概述

DEFOM Stereo 是一个结合 **Depth-Anything-V2 / DINOv2 视觉表征** 与 **RAFT 风格迭代更新** 的立体匹配模型。和传统 RAFT-Stereo 相比，它额外引入了更强的预训练视觉编码器来提取深层语义特征，并通过专门的 `DefomEncoder` 生成初始视差与多尺度特征，再由相关性体积和多层 GRU 进行逐步优化。

核心组件包括：
- **预训练视觉编码器 (DefomEncoder)**：从双目图像中提取高层语义特征，并输出初始视差
- **特征网络 (fnet)**：对左右图像特征做进一步映射，用于构建相关性体积
- **上下文网络 (cnet)**：提取多尺度上下文状态，初始化 GRU 隐状态与输入门控特征
- **相关性匹配 (Correlation Block)**：基于 1D 视差假设构建相关体积
- **迭代更新模块 (Update Block)**：通过多层 GRU 逐步细化视差
- **尺度更新分支 (Scale Update Block)**：在前几次迭代中先做乘性尺度修正，再做增量修正

---

## 完整数据流架构

```
                           输入图像对 (Stereo Image Pair)
                                      ↓
        ┌────────────────────────────────────────────────────┐
        │  图像预处理 (Image Preprocessing)                  │
        │  左右图像: (B, 3, H, W) → 标准化                    │
        └────────────────────────────────────────────────────┘
                                      ↓
        ┌────────────────────────────────────────────────────┐
        │  DefomEncoder (DINOv2 / DAV2)                      │
        └────────────────────────────────────────────────────┘
                                      ↓
        ┌────────────────────────────────────────────────────┐
        │  输出: d_features, dfeat1, dfeat2, disp            │
        │  - d_features: 上下文输入特征                       │
        │  - dfeat1/dfeat2: 左右图像特征                      │
        │  - disp: 初始视差                                   │
        └────────────────────────────────────────────────────┘
                                      ↓
                ┌──────────────────────────────┐   ┌──────────────────────────────┐
                │  Context Network (cnet)      │   │  Feature Network (fnet)      │
                │  MultiBasicEncoder           │   │  BasicEncoder                │
                └──────────────┬───────────────┘   └──────────────┬───────────────┘
                               │                                   │
                               └──────────────┬────────────────────┘
                                              ↓
        ┌────────────────────────────────────────────────────┐
        │  Correlation Block                                 │
        │  1D Correlation                                     │
        └────────────────────────────────────────────────────┘
                                      ↓
        ┌────────────────────────────────────────────────────┐
        │  Iterative Refinement Loop                          │
        │  前几次迭代: ScaleBasicMultiUpdateBlock             │
        │  后续迭代:  BasicMultiUpdateBlock                   │
        └────────────────────────────────────────────────────┘
                                      ↓
                          最终视差输出 (B, 1, H, W)
```

---

## 各模块详细说明

### 1. 输入预处理

DEFOM Stereo 在 `forward()` 中先对左右图像做标准化：

```python
image = ((image - mean) / std).contiguous()
```

其中 `mean/std` 按 ImageNet 统计量设置，并乘以 255 对齐输入为像素值范围的情况。

---

### 2. `DefomEncoder`

这是 DEFOM Stereo 的关键区别。它不是直接使用普通卷积编码器，而是利用预训练视觉 backbone 提取更强的语义表示。

输入：
- `image1`, `image2`
- `danv2_io_sizes = get_danv2_io_size(h, w, n_downsample)`

输出：
- `d_features`：用于上下文网络的深层特征
- `dfeat1`, `dfeat2`：左右图像对应特征，用于后续 `fnet`
- `disp`：初始视差估计

可以理解为：

```
双目图像
   ↓
预训练视觉编码器
   ↓
语义特征 + 初始视差
```

这个模块让模型不必从零学习所有视觉表征，因此通常比纯卷积特征更稳健。

---

### 3. 上下文网络 `cnet`

```python
self.cnet = MultiBasicEncoder(self.defomencoder.out_dim, output_dim=[args.hidden_dims, context_dims], ...)
```

`cnet` 负责生成每层 GRU 所需的初始状态和输入特征。

输出形式为：

```python
cnet_list = self.cnet(image1, d_features)
net_list = [torch.tanh(x[0]) for x in cnet_list]
inp_list = [torch.relu(x[1]) for x in cnet_list]
```

含义：
- `net_list`：GRU 的隐状态初始化 `h`
- `inp_list`：GRU 的上下文输入特征 `x`

随后会通过 `context_zqr_convs` 一次性把 `inp_list` 映射成 Z/Q/R 三组门控输入：

```python
inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i, conv in zip(inp_list, self.context_zqr_convs)]
```

这一步的作用是：
- 避免在每次迭代中重复卷积
- 为多层 GRU 提供门控所需的预处理特征

---

### 4. 特征网络 `fnet`

```python
self.fnet = BasicEncoder(self.defomencoder.out_dim, output_dim=256, ...)
```

`fnet` 用于将 `dfeat1/dfeat2` 转换成适合做相关性的特征图：

```python
fmap1, fmap2 = self.fnet([image1, image2], [dfeat1, dfeat2])
```

输出：
- `fmap1`, `fmap2`：左右特征图，通常为 `(B, 256, H/d, W/d)`

这些特征会被送入 1D 相关性模块，用来在视差方向搜索匹配。

---

### 5. 相关性体积 `CorrBlock1D`

```python
corr_fn = CorrBlock1D(fmap1, fmap2, coords, radius=..., num_levels=...)
```

DEFOM Stereo 仍然遵循立体匹配的核心假设：
- 视差主要沿水平方向变化
- 因此只构建 1D 相关性搜索

输入：
- 左右特征 `fmap1`, `fmap2`
- 初始坐标 `coords`

输出：
- 在当前视差附近采样得到的相关特征 `corr`

它会根据 `corr_radius` 和 `corr_levels` 构建多尺度相关金字塔，从而支持粗到细的匹配。

---

### 6. 坐标初始化

```python
coords = self.initialize_coords(net_list[0])
```

这里只初始化一个坐标平面：

- `coords = coords_grid(N, H, W)[:, :1]`

也就是说，模型主要把问题简化为 **单通道视差估计**，而不是完整 2D 光流。

---

### 7. 迭代更新循环

DEFOM Stereo 的核心仍然是迭代式 refinement。循环逻辑为：

```python
for itr in range(iters):
      disp = disp.detach()
      if itr < scale_iters:
            ... scale_update_block ...
            disp = scale_disp * disp
      else:
            ... update_block ...
            disp = disp + delta_disp
```

#### 7.1 前期：尺度更新 `ScaleBasicMultiUpdateBlock`

前 `scale_iters` 次迭代先做乘性更新：

$$
disp_{t+1} = scale_t \cdot disp_t
$$

这类更新适合快速修正初始视差的整体偏差。

#### 7.2 后期：增量更新 `BasicMultiUpdateBlock`

后续迭代采用加性修正：

$$
disp_{t+1} = disp_t + \Delta disp_t
$$

并且会对增量做裁剪，避免越界：

```python
delta_disp = torch.clip(delta_disp, min=..., max=...)
```

这种“先乘后加”的策略通常能让模型先快速收敛到大致范围，再细化局部误差。

---

### 8. 上采样 `upsample_flow`

如果更新块输出了 `up_mask`，则使用 convex combination 上采样：

```python
disp_up = self.upsample_flow(disp, up_mask)
```

否则使用普通上采样：

```python
disp_up = upflow(disp, factor=2 ** self.n_downsample)
```

作用：
- 将低分辨率视差恢复到原图分辨率
- 通过学习到的 mask 保留边界细节

---

## 关键参数说明

| 参数 | 说明 |
|------|------|
| `dinov2_encoder` | DefomEncoder 所使用的预训练 backbone 类型 |
| `idepth_scale` | 初始深度/视差的缩放系数 |
| `hidden_dims` | 多层 GRU 的隐状态维度 |
| `n_gru_layers` | GRU 层数 |
| `n_downsample` | 特征图下采样倍率 |
| `corr_radius` | 相关性采样半径 |
| `corr_levels` | 相关性金字塔层数 |
| `scale_iters` | 前多少次迭代使用尺度更新 |
| `mixed_precision` | 是否启用混合精度推理/训练 |

---

## 输入输出规范

### 输入
- **类型**：`torch.Tensor`
- **形状**：`(B, 3, H, W)`
- **说明**：双目图像对 `image1`, `image2`

### 输出
- **测试模式**：返回最终上采样视差图 `disp_up`
- **训练模式**：返回每次迭代的预测列表 `disp_predictions`
- **形状**：通常为 `(B, 1, H, W)`

---

## 与 RAFT-Stereo 的主要差异

| 模块 | RAFT-Stereo | DEFOM Stereo |
|------|-------------|--------------|
| 特征来源 | 纯卷积编码器 | 预训练视觉编码器 + 卷积特征头 |
| 初始视差 | 通常从零初始化 | `DefomEncoder` 直接预测初始视差 |
| 更新策略 | 纯增量更新 | 前期乘性尺度更新 + 后期增量更新 |
| 语义表征 | 较弱 | 更强，依赖 DINOv2 / DAV2 预训练表征 |
| 目标 | 传统立体匹配 | 更强泛化与更稳健的初始匹配 |

---

## 总结

DEFOM Stereo 可以理解为：

> 以预训练视觉 backbone 提供强语义特征和初始视差，再结合 RAFT 风格的 1D 相关体积与多层 GRU 进行迭代优化。

它的优势主要体现在：
- 初始匹配更强
- 对纹理稀少区域更稳健
- 通过尺度更新提升大位移场景的收敛速度
- 保留了 RAFT 式迭代细化的高精度优势



# DepthAnythingV2 引导的可变形特征编码器

## 目标

修改 DEFOM 的 feature encoder，让 `DepthAnythingV2` 输出的特征不再只是通过后续残差加和的方式参与融合，而是能够主动引导底层双目特征提取。具体方向是将 `BasicEncoder` 第一层的 `7x7` 图像卷积替换为可变形卷积，并由 `DepthAnythingV2` 的输出预测这层可变形卷积的参数。

这份说明主要聚焦于 feature 分支：

- 当前路径：`DefomEncoder -> dfeat1/dfeat2 -> BasicEncoder`
- 目标路径：`DefomEncoder -> guide head -> BasicEncoder 中的 deformable stem`


## 当前基线结构

在当前实现中：

- `DefomEncoder` 返回 `d_features`、`dfeat1`、`dfeat2` 和初始视差
- `BasicEncoder` 的处理流程是：
  - 对 RGB 做 `conv1: 7x7`
  - 经过若干残差块
  - 再执行 `x = x + self.convd(dfeats)`

也就是说，`DepthAnythingV2` 特征确实影响了 `fnet`，但它的作用方式是：

- 在第一层图像卷积之后才介入
- 仅通过加法残差注入
- 不会改变图像分支实际“从哪里采样”


## 核心想法

让 `DepthAnythingV2` 特征预测 `BasicEncoder` 第一层 stem 的可变形卷积参数，使 feature encoder 在构建 correlation 特征之前，就能够根据场景几何自适应调整采样位置。

不是：

```text
RGB --7x7 conv--> early feature
```

而是：

```text
DepthAnythingV2 feature --guide head--> offsets / modulation
RGB --deformable 7x7 conv--> geometry-aware early feature
```


## 为什么这个方向合理

第一层 stem 对后续纹理特征提取影响很大，因为它：

- 直接作用在原始图像梯度上
- 决定后续残差层使用的初始空间支持范围
- stereo matching 对边界、斜面、弱纹理区域非常敏感

如果采样网格能根据单目几何先验进行偏移，feature encoder 理论上可以：

- 更贴合斜视差结构进行采样
- 在低纹理区域扩大支持范围
- 减少跨深度边界的混合
- 从一开始就构建更几何感知的相关特征


## 推荐设计

### 方案 A：仅预测 Offset 的 Deformable Stem

将 `BasicEncoder.conv1` 替换成 deformable conv，仅由 `dfeat` 预测 offsets。

结构如下：

```text
dfeat -> 3x3 conv -> ReLU -> 3x3 conv -> offsets
RGB -> DeformConv2d(7x7, offsets) -> norm -> relu
```

对于一个 `7x7` 的 deformable convolution：

- 采样点个数是 `49`
- offset 通道数是 `2 * 49 = 98`

如果使用 DCNv2 风格的 modulation：

- mask 通道数是 `49`

因此参数输出维度为：

- 仅 offset：`98`
- offset + modulation mask：`147`

这是最适合的第一版实现，结构简单，也最容易稳定训练和做消融比较。

建议：先从这个版本开始。


### 方案 B：Offset + Modulation Stem

同时从 `dfeat` 预测 offsets 和 modulation mask。

结构如下：

```text
dfeat -> guide head -> [offset_x, offset_y, mask]
RGB -> ModulatedDeformConv2d(7x7, offsets, mask)
```

潜在优点：

- mask 可以抑制不可靠的偏移采样点
- 表达能力比 offset-only 更强

潜在代价：

- 更容易训练不稳定
- 前期可能退化出无意义 mask

建议：在 offset-only 版本跑通后再尝试。


### 方案 C：残差式 Deformable Stem

同时保留标准卷积分支和 deformable 分支：

```text
stem_std = Conv7x7(RGB)
stem_def = DeformConv7x7(RGB, guide(dfeat))
alpha = sigmoid(head(dfeat))
stem = stem_std + alpha * stem_def
```

优点：

- 保留原始基线行为
- 优化更稳定
- 当 `DepthAnythingV2` 引导有噪声时风险更低

如果你想要一个更适合正式实验、更低风险的结构，这是我更偏好的方案。


## 强推荐结构

如果目标是“尽可能大概率有效，同时不过度增加不稳定性”，建议采用下面这个结构：

```text
DepthAnythingV2 dfeat
  -> guide_reduce: 3x3 conv 到 64 通道
  -> guide_refine: residual block
  -> offset_head: 3x3 conv -> 98 channels
  -> mask_head: 3x3 conv -> 49 channels
  -> alpha_head: 3x3 conv -> 1 channel

RGB
  -> std_stem: 7x7 conv, 输出 64
  -> def_stem: modulated deformable 7x7 conv, 输出 64

stem = std_stem + sigmoid(alpha) * def_stem
```

然后继续沿用现有的 normalization 和 residual tower。

这个结构同时保留了三点优势：

- baseline 路径仍然存在
- 几何引导采样能力被引入
- 模型可以自己学习对 deformable 分支信任多少


## 应该用哪一种 DepthAnythingV2 特征做引导

对于 `BasicEncoder`，直接引导应当使用 `dfeat1` 和 `dfeat2`，而不是 `d_features`。

原因：

- `BasicEncoder` 分别处理左图和右图分支
- `dfeat1` / `dfeat2` 已经与左右图一一对应
- `d_features` 当前主要是给 `cnet` 的多尺度 context 分支使用，角色不同

因此更合理的对应关系是：

- 左图的 feature stem 由 `dfeat1` 引导
- 右图的 feature stem 由 `dfeat2` 引导


## 分辨率对齐

用于预测 deformable 参数的 guide feature，必须与第一层 stem 输出所需的空间分辨率一致。

当前 `BasicEncoder.conv1` 的 stride 取决于 `downsample`：

- 当 `downsample > 2` 时，stride 为 `2`
- 否则 stride 为 `1`

这意味着 guide head 预测 offset 的分辨率，应该与 stem 输出分辨率一致，而不一定是 RGB 原图分辨率。

推荐做法：

1. 先在 `dfeat` 上做 guide head
2. 如果需要，对 guide feature 插值
3. 在与 `conv1` 输出一致的空间分辨率上预测 offset/mask

不要先在全分辨率预测 offset，再粗暴地下采样。


## 如何注入深度先验

用 `DepthAnythingV2` 引导 deformable 参数，主要有三种合理做法。

### 1. 仅使用 Feature Guidance

只使用 `dfeat`。

优点：

- 最简单
- 不会过度依赖单目深度尺度

缺点：

- 几何信号不够显式

这应该是你的默认第一组实验。


### 2. 使用 Feature + Initial Disparity Guidance

把 `dfeat` 和 `disp_init` 拼接起来：

```text
guide_input = concat(dfeat, disp_init)
```

优点：

- 可以显式使用深度结构信息
- 对斜平面和深度边界可能更有帮助

缺点：

- 可能过拟合到单目视差初始化的偏差

这是一个很值得做的第二组实验。


### 3. 使用 Feature + Edge Guidance

额外构造一个边界或不连续性先验，例如：

- 图像梯度
- 深度梯度
- 从 `disp_init` 计算出的边缘强度

然后用它控制 offset 的幅度，例如：

```text
offset = raw_offset * edge_gate
```

这种方式比较适合防止低纹理平坦区域里出现大幅度、无意义的 offset 扰动。


## Offset 约束

这一部分对训练稳定性非常关键。

不要让 offset head 从一开始就输出无约束的大偏移。

推荐约束形式：

```text
offset = offset_range * tanh(raw_offset)
```

建议初值：

- 对 `7x7` 卷积，`offset_range = 2.0`
- 如果训练不稳定，先试 `1.0`

对于 modulation mask：

```text
mask = sigmoid(raw_mask)
```

对于残差融合权重：

```text
alpha = sigmoid(raw_alpha)
```


## 初始化策略

为了保证第一版训练稳定，建议：

- deformable offset head 的权重和 bias 初始化为 0
- modulation mask 的 bias 初始化为使 mask 接近 `1`
- alpha 的 bias 初始化为让 deformable 分支初始权重较小

这样做的效果是：

- 初始行为尽量接近普通卷积
- deformable 采样逐渐学出来，而不是一开始就强介入

如果你是从现有 DEFOM checkpoint 继续训练，这一点尤其重要。


## 代码应该改哪里

主要涉及：

- `core/extractor.py`
- `core/defom_stereo.py`

核心插入点如下：

1. 在 `BasicEncoder` 中
   - 替换 `self.conv1`
   - 增加用于预测 offset / mask / alpha 的 guide head
   - 修改 `forward(self, x, dfeats)`

2. 在 `DefomEncoder` 中
   - 保持继续返回 `dfeat1`、`dfeat2`
   - 如果后续需要，也可以额外返回更专门的 guide tensor

3. 在 `DEFOMStereo.forward` 中
   - 如果 `BasicEncoder` 的接口仍然保持 `([image1, image2], [dfeat1, dfeat2])`
   - 则整体逻辑不需要大改

如果你后面还想把同样思想扩展到 `cnet`，再单独修改 `MultiBasicEncoder`。第一轮实验不建议把两者同时改掉。


## 推荐实现顺序

### 阶段 1

先只在 `BasicEncoder` 中实现 offset-only deformable stem。

保持其他部分不变：

- `cnet` 不动
- update blocks 不动
- 输出不动

这样就把影响严格限制在 `fnet` 上。


### 阶段 2

加入残差融合：

```text
stem = std + alpha * def
```

对比：

- 标准 stem
- 纯 deformable stem
- 残差式 deformable stem


### 阶段 3

测试不同引导输入：

- `dfeat`
- `concat(dfeat, disp_init)`


### 阶段 4

只有在前面确实有效的前提下，再把同样的 guided deformable stem 思路扩展到 `MultiBasicEncoder`，也就是 `cnet`。


## 推荐做的消融实验

至少比较下面这些版本：

1. 原始 DEFOM baseline
2. offset-only deformable `fnet` stem
3. offset+mask deformable `fnet` stem
4. 残差式 deformable `fnet` stem
5. 使用 `dfeat + disp_init` 引导的残差式 deformable `fnet` stem

建议评估：

- SceneFlow validation
- Middlebury H/F
- 如果关注跨域鲁棒性，再看 KITTI
- 运行时间和显存开销


## 主要风险

### 风险 1：单目先验压过双目证据

如果 guide 分支在前期过强，`fnet` 可能更依赖单目几何猜测，而不是 stereo consistency。

缓解方式：

- 保留标准 stem 分支
- alpha 初值设小
- 先用 offset-only 或残差混合方案


### 风险 2：Offset 噪声破坏匹配精度

stereo 对空间对齐很敏感。偏移预测不好时，反而会模糊 correspondence cue。

缓解方式：

- 用 `tanh` 约束 offset
- offset 初始化为 0
- 如有需要，对 offset 幅度加正则


### 风险 3：左右分支不对称过强

如果左右两侧 guide 分支偏移行为差异过大，可能反而损害 correlation 质量。

缓解方式：

- 左右两侧共享 guide-head 结构
- 保持参数化形式完全对称
- 检查左右 offset 的统计分布


## 我建议的第一版

如果目标是先落地一个较实用、风险较低的版本，我建议：

1. 只修改 `BasicEncoder`
2. 用 `dfeat1` / `dfeat2` 作为 guide 输入
3. 使用残差式 deformable stem
4. 预测 offset + mask + alpha
5. offset 初始化为 0，alpha 初始较小

这样在创新性、可控性和与现有代码兼容性之间，是一个比较好的平衡点。


## 最小伪代码

```python
guide = guide_reduce(dfeat)
guide = guide_refine(guide)

offset = offset_range * torch.tanh(offset_head(guide))
mask = torch.sigmoid(mask_head(guide))
alpha = torch.sigmoid(alpha_head(guide))

stem_std = conv1_std(x)
stem_def = deform_conv1(x, offset, mask)
stem = stem_std + alpha * stem_def

stem = norm1(stem)
stem = relu1(stem)
```


## 仅使用 Feature Guidance 时的分辨率对齐与形状流

这一节严格按上面的“强推荐结构”展开，并固定采用下面这组约束：

- 只改 `BasicEncoder`
- 只使用 `Feature Guidance`
- guide 输入只使用 `dfeat1` / `dfeat2`
- 不引入 `disp_init`
- 使用残差式 deformable stem：
  - `stem_std = conv1_std(x)`
  - `stem_def = deform_conv1(x, offset, mask)`
  - `stem = stem_std + alpha * stem_def`
- offset / mask / alpha 按前文的保守初始化与约束方式实现


### 1. 先统一记号

定义：

- 输入左右图：`image1`, `image2`
- 原始分辨率：`(B, 3, H, W)`
- `n_downsample = K`
- 工作尺度因子：`s = 2^K`

则 `DefomEncoder` 里会先通过 `get_danv2_io_size(h, w, K)` 得到：

- `oh = H / s`
- `ow = W / s`

这里的 `oh, ow` 是 `DepthAnythingV2` 相关输出最终对齐到的工作分辨率。

在当前默认配置下，通常：

- `K = 2`
- 因此 `s = 4`
- 即 `oh = H/4`, `ow = W/4`

这一点很关键，因为它说明：

- `dfeat1/dfeat2` 不是原图分辨率
- 它们已经被对齐到 stereo 主干使用的工作尺度
- 这个尺度恰好也是当前 `BasicEncoder` 最终输出 `fmap1/fmap2` 的尺度


### 2. `DefomEncoder` 输出的实际形状

在 [core/extractor.py](D:\project\DEFOM-Stereo\core\extractor.py:381) 和 [depth_anything_v2/dpt.py](D:\project\DEFOM-Stereo\depth_anything_v2\dpt.py:222) 里，`DefomEncoder.forward()` 的输出可以理解为：

- `d_features`
- `dfeat1`
- `dfeat2`
- `disp`

其中在只看 shape 时：

#### 2.1 `d_features`

`d_features` 是一个三层 list，对应给 `cnet` 的多尺度特征：

- `d_features[0]`: `(B, C_d, oh, ow)`
- `d_features[1]`: `(B, C_d, oh/2, ow/2)`
- `d_features[2]`: `(B, C_d, oh/4, ow/4)`

这里：

- `C_d = self.defomencoder.out_dim`
- 与 backbone 选择有关：
  - `vits -> 64`
  - `vitb -> 128`
  - `vitl -> 256`
  - `vitg -> 384`

#### 2.2 `dfeat1`, `dfeat2`

这两个是给 `fnet` 用的左右图引导特征，来自 `DPTFeat` 里的 `path_1`：

- `dfeat1`: `(B, C_d, oh, ow)`
- `dfeat2`: `(B, C_d, oh, ow)`

注意它们不是多尺度 list，而是单尺度特征图。

在默认 `n_downsample=2` 下，就是：

- `dfeat1`: `(B, C_d, H/4, W/4)`
- `dfeat2`: `(B, C_d, H/4, W/4)`

#### 2.3 `disp`

当前 `DefomEncoder` 输出的初始视差 `disp` 也是工作尺度：

- `disp`: `(B, 1, oh, ow)`

但在当前这版设计中，我们先不使用它。


### 3. 当前 `BasicEncoder` 的尺度关系

现有 `BasicEncoder` 定义在 [core/extractor.py](D:\project\DEFOM-Stereo\core\extractor.py:150)。

它的主干尺度由 `downsample=args.n_downsample` 控制。

对默认 `K=2` 的情况：

- `conv1` 的 stride = `1`
- `layer1` 的 stride = `1`
- `layer2` 的 stride = `2`
- `layer3` 的 stride = `2`

因此当前 `BasicEncoder` 的空间分辨率流是：

- 输入 RGB：`(B, 3, H, W)`
- `conv1` 后：`(B, 64, H, W)`
- `layer1` 后：`(B, 64, H, W)`
- `layer2` 后：`(B, 96, H/2, W/2)`
- `layer3` 后：`(B, 128, H/4, W/4)`
- `conv2` 后：`(B, 256, H/4, W/4)`

而当前它融合 `dfeat` 的位置是在 `layer3` 之后：

```python
x = x + self.convd(dfeats)
```

所以 `self.convd(dfeats)` 的输出必须是 `(B, 128, H/4, W/4)`，这也反过来说明：

- `dfeat1/dfeat2` 已经与 `layer3` 输出同尺度
- 默认情况下它们不需要再额外缩放就能和 `layer3` 对齐


### 4. 改成“强推荐结构”后的关键问题

现在你想把 `dfeat1/dfeat2` 用来指导第一层 `7x7 stem` 的 deformable conv。

但 `conv1` 处理的是原图分辨率：

- RGB 输入是 `(B, 3, H, W)`
- `conv1` 输出是 `(B, 64, H, W)`，在 `K=2` 时仍是全分辨率

而 guide 特征 `dfeat1/dfeat2` 是：

- `(B, C_d, H/4, W/4)`

所以这里存在一个明确的尺度不匹配：

- `dfeat` 是工作尺度
- 第一层 stem 是更高分辨率

因此我们必须显式设计一条“guide 对齐到 stem 分辨率”的路径。


### 5. 推荐的对齐方式

在“只使用 Feature Guidance”的前提下，最稳妥的方式是：

1. 先在工作尺度上对 `dfeat` 做轻量 guide 编码
2. 再上采样到 `conv1` 输出尺度
3. 最后在 `conv1` 尺度上预测 offset / mask / alpha

也就是：

```text
dfeat (H/4, W/4)
  -> guide_reduce / guide_refine
  -> upsample
  -> offset_head / mask_head / alpha_head
```

而不是直接把 `dfeat` 生硬插值到原图尺寸后再做所有卷积。

原因：

- guide 编码本身更适合在较低分辨率做，开销更小
- 先抽象再上采样，通常比直接高分辨率预测更稳
- 更容易控制参数量和显存


### 6. 强推荐结构下的完整形状流

下面以默认 `n_downsample = 2` 为例，给出完整 shape 流。

#### 6.1 `DefomEncoder` 输出

左图分支：

- `image1`: `(B, 3, H, W)`
- `dfeat1`: `(B, C_d, H/4, W/4)`

右图分支：

- `image2`: `(B, 3, H, W)`
- `dfeat2`: `(B, C_d, H/4, W/4)`

其中 `C_d` 由 backbone 决定，例如 `vitl` 时 `C_d = 256`。


#### 6.2 Guide 编码阶段

对 `dfeat1` 先做引导特征压缩和细化：

```python
guide_1 = guide_reduce(dfeat1)
guide_1 = guide_refine(guide_1)
```

建议 shape 为：

- `guide_reduce`: `Conv3x3(C_d -> 64)`
- `guide_refine`: `ResidualBlock(64 -> 64)`

于是：

- `guide_1`: `(B, 64, H/4, W/4)`

右图同理：

- `guide_2`: `(B, 64, H/4, W/4)`


#### 6.3 Guide 上采样到 stem 分辨率

因为 `conv1` 输出是全分辨率 `(H, W)`，所以 guide 要先被上采样：

```python
guide_1_up = F.interpolate(guide_1, size=(H, W), mode="bilinear", align_corners=True)
guide_2_up = F.interpolate(guide_2, size=(H, W), mode="bilinear", align_corners=True)
```

得到：

- `guide_1_up`: `(B, 64, H, W)`
- `guide_2_up`: `(B, 64, H, W)`

这里的逻辑是：

- `dfeat` 提供低分辨率几何引导
- `guide_up` 将这种几何引导广播到 stem 采样分辨率


### 6.4 从上采样 guide 预测 DCN 参数

对 `guide_1_up` 分别预测：

- `offset`
- `mask`
- `alpha`

如果使用 `7x7` deformable conv：

- offset 通道数：`98`
- mask 通道数：`49`
- alpha 通道数：`1`

则：

```python
raw_offset_1 = offset_head(guide_1_up)
raw_mask_1 = mask_head(guide_1_up)
raw_alpha_1 = alpha_head(guide_1_up)
```

shape 分别为：

- `raw_offset_1`: `(B, 98, H, W)`
- `raw_mask_1`: `(B, 49, H, W)`
- `raw_alpha_1`: `(B, 1, H, W)`

然后做约束：

```python
offset_1 = offset_range * torch.tanh(raw_offset_1)
mask_1 = torch.sigmoid(raw_mask_1)
alpha_1 = torch.sigmoid(raw_alpha_1)
```

约束后的 shape 不变：

- `offset_1`: `(B, 98, H, W)`
- `mask_1`: `(B, 49, H, W)`
- `alpha_1`: `(B, 1, H, W)`

右图同理：

- `offset_2`: `(B, 98, H, W)`
- `mask_2`: `(B, 49, H, W)`
- `alpha_2`: `(B, 1, H, W)`


### 6.5 RGB stem 与 deformable stem

对左图 `image1`：

标准分支：

```python
stem_std_1 = conv1_std(image1)
```

输出：

- `stem_std_1`: `(B, 64, H, W)`

deformable 分支：

```python
stem_def_1 = deform_conv1(image1, offset_1, mask_1)
```

输出：

- `stem_def_1`: `(B, 64, H, W)`

残差融合：

```python
stem_1 = stem_std_1 + alpha_1 * stem_def_1
stem_1 = norm1(stem_1)
stem_1 = relu1(stem_1)
```

最终：

- `stem_1`: `(B, 64, H, W)`

右图完全对称：

- `stem_2`: `(B, 64, H, W)`


### 6.6 stem 之后回到原始 `BasicEncoder` 主干

左图分支继续走：

- `layer1(stem_1)` -> `(B, 64, H, W)`
- `layer2(...)` -> `(B, 96, H/2, W/2)`
- `layer3(...)` -> `(B, 128, H/4, W/4)`

然后这里有两种选择：

#### 方案 1：保留原来的后融合

继续保留：

```python
x = x + self.convd(dfeat1)
```

则：

- `self.convd(dfeat1)` -> `(B, 128, H/4, W/4)`
- 融合后 `x` -> `(B, 128, H/4, W/4)`

最后：

- `conv2(x)` -> `fmap1: (B, 256, H/4, W/4)`

右图同理得到：

- `fmap2: (B, 256, H/4, W/4)`

这是我更推荐的第一版，因为：

- `dfeat` 同时在 stem 前和高层融合处发挥作用
- DCN 负责“引导早期采样”
- `convd(dfeat)` 仍保留原 baseline 的中高层注入路径

#### 方案 2：去掉原来的后融合

也可以把：

```python
x = x + self.convd(dfeats)
```

删掉，只让 `dfeat` 通过 deformable stem 起作用。

但这会改变更大，第一版不建议这样做，因为你会同时丢掉 baseline 已验证有效的融合路径。


### 7. 推荐的第一版完整 shape 总表

以左图为例：

- `image1`: `(B, 3, H, W)`
- `dfeat1`: `(B, C_d, H/4, W/4)`
- `guide_reduce(dfeat1)`: `(B, 64, H/4, W/4)`
- `guide_refine(...)`: `(B, 64, H/4, W/4)`
- `guide_up`: `(B, 64, H, W)`
- `offset_head(guide_up)`: `(B, 98, H, W)`
- `mask_head(guide_up)`: `(B, 49, H, W)`
- `alpha_head(guide_up)`: `(B, 1, H, W)`
- `conv1_std(image1)`: `(B, 64, H, W)`
- `deform_conv1(image1, offset, mask)`: `(B, 64, H, W)`
- `stem`: `(B, 64, H, W)`
- `layer1(stem)`: `(B, 64, H, W)`
- `layer2(...)`: `(B, 96, H/2, W/2)`
- `layer3(...)`: `(B, 128, H/4, W/4)`
- `convd(dfeat1)`: `(B, 128, H/4, W/4)`
- `x + convd(dfeat1)`: `(B, 128, H/4, W/4)`
- `conv2(...)`: `(B, 256, H/4, W/4)`

右图完全对称。

最后输出：

- `fmap1`: `(B, 256, H/4, W/4)`
- `fmap2`: `(B, 256, H/4, W/4)`

这与当前相关性模块的输入接口保持一致，因此：

- `CorrBlock1D`
- `update_block`
- `scale_update_block`

都不需要因为这次改动而改 shape 接口。


### 8. 为什么这一版不引入 `disp`

在这个阶段不引入 `disp`，主要是为了把问题拆干净。

当前只使用 `Feature Guidance` 时：

- guide 只来自 `dfeat1/dfeat2`
- 不需要额外处理 `disp` 与 stem 分辨率不一致的问题
- 不需要考虑 monocular disparity bias 会不会直接主导 offset
- 更容易判断收益究竟来自哪里

如果后面要引入 `disp`，最自然的做法是：

- 先把 `disp` 从 `(B, 1, H/4, W/4)` 投影到较小通道数
- 与 `guide_1` 在 `(H/4, W/4)` 上拼接
- 再统一上采样到 `(H, W)` 预测 DCN 参数

而不是把 `disp` 单独上采样到原图后直接拼进去。


### 9. 对实现的直接建议

按当前这版文档，我建议第一版代码结构是：

1. 在 `BasicEncoder` 里新增：
   - `conv1_std`
   - `conv1_def`
   - `guide_reduce`
   - `guide_refine`
   - `offset_head`
   - `mask_head`
   - `alpha_head`

2. `forward(self, x, dfeats)` 中：
   - 左右图仍然先按 batch 维拼起来
   - 左右 `dfeat` 也先按 batch 维拼起来
   - 对拼接后的 `dfeat` 统一走 guide 分支
   - guide 上采样到 `x` 的 stem 分辨率
   - 预测 offset/mask/alpha
   - 同时计算 `conv1_std(x)` 和 `conv1_def(x, offset, mask)`
   - 残差融合后继续走原有 layer1/layer2/layer3
   - 保留现有 `x = x + self.convd(dfeats)` 的后融合

3. 这样做的好处是：
   - 不需要动 `DEFOMStereo.forward()` 的接口
   - 不需要动 `CorrBlock1D`
   - 不需要动 `update_block`
   - 风险集中在 `BasicEncoder` 内部，可控性最好


## 总结建议

不要一开始就同时把 `feature encoder` 和 `context encoder` 都改成 deformable。先只改 `BasicEncoder` 的第一层 `7x7 stem`，并用 `dfeat1/dfeat2` 做引导，采用保守初始化的残差式 deformable 分支。这是当前代码框架下最干净、风险最低、最容易验证收益的一条实验路径。

## 验证记录

下面记录当前这版实现的两项直接验证结果。验证环境使用：

- `conda run -n raftstereo`
- 随机输入，不加载真实数据集
- `dinov2_encoder='vits'`
- `n_downsample=2`
- `iters=2`
- `scale_iters=1`


### 1. 完整 `DEFOMStereo` 随机前向验证

验证目标：

- 确认从 `DefomEncoder -> cnet/fnet -> CorrBlock1D -> scale_update_block/update_block -> upsample` 整条前向链路可以跑通
- 确认这次只改 `BasicEncoder` 没有破坏外部接口

测试输入：

- `image1`: `(1, 3, 64, 96)`
- `image2`: `(1, 3, 64, 96)`

测试结果：

- 前向成功完成
- `disp_predictions` 长度为 `2`
- 两次预测输出的 shape 都是：
  - `(1, 1, 64, 96)`

对应的随机前向统计值如下：

- `pred[0]`
  - shape: `(1, 1, 64, 96)`
  - min: `17.8046`
  - max: `45.0619`
  - mean: `39.5405`
  - std: `5.7961`
- `pred[1]`
  - shape: `(1, 1, 64, 96)`
  - min: `17.5568`
  - max: `44.2886`
  - mean: `38.5319`
  - std: `5.6058`

结论：

- 当前这版 `BasicEncoder` 修改没有破坏 `DEFOMStereo` 的整体前向
- `fnet` 输出 shape 仍然与 `CorrBlock1D` 和后续更新模块兼容
- 在随机输入下，模型可以完整地产生有效的 disparity prediction


### 2. `BasicEncoder` 中 `offset / mask / alpha` 的初始化验证

验证目标：

- 检查初始化是否与设计文档一致
- 确认训练开始时 deformable stem 的行为接近“弱扰动、近似标准卷积”

验证方式：

- 对 `model.fnet.offset_head`
- `model.fnet.mask_head`
- `model.fnet.alpha_head`

注册 forward hook，抓取完整 `DEFOMStereo` 前向中的原始输出

由于 `fnet` 内部会把左右图拼成一个 batch，因此 hook 抓到的 batch 维是 `2`，也就是：

- 左图 1 份
- 右图 1 份


#### 2.1 原始 head 输出

`raw_offset`：

- shape: `(2, 98, 64, 96)`
- min: `0.0`
- max: `0.0`
- mean: `0.0`
- std: `0.0`

`raw_mask`：

- shape: `(2, 49, 64, 96)`
- min: `2.0`
- max: `2.0`
- mean: `2.0`
- std: `0.0`

`raw_alpha`：

- shape: `(2, 1, 64, 96)`
- min: `-2.0`
- max: `-2.0`
- mean: `-2.0`
- std: `0.0`

这与当前初始化完全一致：

- `offset_head.weight = 0`
- `offset_head.bias = 0`
- `mask_head.weight = 0`
- `mask_head.bias = 2`
- `alpha_head.weight = 0`
- `alpha_head.bias = -2`


#### 2.2 激活和约束后的参数

按当前实现：

- `offset = 2.0 * tanh(raw_offset)`
- `mask = sigmoid(raw_mask)`
- `alpha = sigmoid(raw_alpha)`

得到：

`offset`：

- shape: `(2, 98, 64, 96)`
- min: `0.0`
- max: `0.0`
- mean: `0.0`
- std: `0.0`

`mask`：

- shape: `(2, 49, 64, 96)`
- min: `0.8807970`
- max: `0.8807970`
- mean: `0.8807972`
- std: `1.79e-07`

`alpha`：

- shape: `(2, 1, 64, 96)`
- min: `0.1192029`
- max: `0.1192029`
- mean: `0.1192029`
- std: `2.98e-08`


### 3. 对初始化结果的解释

这组结果说明当前实现确实满足了“保守初始化”的设计目标：

- `offset = 0`
  - deformable 采样点初始不发生偏移
  - 初始采样位置与标准卷积完全一致

- `mask ≈ 0.8808`
  - deformable 分支初始并没有被强行压到很小
  - 它会提供一个接近 1 的有效采样权重

- `alpha ≈ 0.1192`
  - deformable 分支虽然存在，但初始融合权重较小
  - 整体 stem 仍主要由标准卷积分支主导

因此，初始 stem 的行为可以理解为：

- 标准卷积分支是主路径
- deformable 分支已经连通
- 但它只以较弱权重参与
- 并且采样位置起点与标准卷积一致

这正是当前方案希望得到的训练起点。


### 4. 当前验证结论

截至目前，这版实现已经满足下面两点：

1. `BasicEncoder` 的 guided residual deformable stem 可以与现有 `DEFOMStereo` 完整主干兼容
2. `offset / mask / alpha` 的初始化数值范围符合设计预期，不会在训练起始阶段引入大幅采样扰动

因此，从结构连通性和初始化稳定性两方面看，这一版已经具备进入小规模训练验证的条件。

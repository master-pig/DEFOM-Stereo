# 使用 TensorBoard 绘制当前 DEFOM-Stereo 网络结构图

本文给出一份**基于当前仓库代码**的 TensorBoard 结构图绘制教程，并针对现在这版已经修改过的网络结构，给出一个可直接参考的示例。

当前说明对应的主模型实现位于：

- `core/defom_stereo.py`
- `core/extractor.py`
- `core/update.py`

## 1. 当前模型结构先看清楚

当前 `DEFOMStereo` 不是一个单纯的 CNN 顺序堆叠结构，而是一个“**DepthAnythingV2 引导特征编码 + 相关体构建 + 多层 GRU 迭代更新**”的立体匹配网络。

按前向流程看，主干可以概括为：

1. 输入左右图 `image1`、`image2`
2. 做均值方差归一化
3. 进入 `DefomEncoder`
4. `DefomEncoder` 内部调用 `DepthAnythingV2`
5. 输出：
   - 多尺度深度引导特征 `d_features`
   - 左右分支引导特征 `dfeat1`、`dfeat2`
   - 初始视差 `disp`
6. `cnet = MultiBasicEncoder(...)`
   - 生成 GRU 所需的多尺度 context 特征
7. `fnet = BasicEncoder(...)`
   - 生成左右图匹配特征 `fmap1`、`fmap2`
8. `CorrBlock1D(...)`
   - 构建 1D correlation 查询模块
9. 前 `scale_iters` 次进入 `scale_update_block`
   - 对视差做乘性缩放更新
10. 后续迭代进入 `update_block`
   - 对视差做加性残差更新
11. `upsample_flow(...)`
   - 输出上采样后的最终视差

### 1.1 当前修改后的关键点

这次最值得在结构图里重点观察的是 `BasicEncoder` 的输入 stem 已经不是单一路径卷积，而是变成了：

- 标准分支：`conv1_std`
- 可变形分支：`conv1_def`
- 引导分支：
  - `guide_reduce`
  - `guide_refine`
  - `offset_head`
  - `mask_head`
  - `alpha_head`

融合逻辑在 `core/extractor.py` 的 `BasicEncoder.forward()` 里是：

1. 用 `dfeats` 先生成 guide feature
2. 用 guide feature 预测 deformable conv 的 `offset`
3. 同时预测 `mask`
4. 同时预测融合权重 `alpha`
5. 得到：
   - `stem_std = conv1_std(x)`
   - `stem_def = conv1_def(x, offset, mask)`
6. 融合：

```python
x = stem_std + alpha * stem_def
```

也就是说，**当前结构图里最关键的新结构就是“DepthAnything 引导的可变形卷积 stem”**。

## 2. 为什么不能直接对原模型粗暴 `add_graph`

理论上可以直接：

```python
writer.add_graph(model, (image1, image2))
```

但当前这个模型有几个特点，会让直接画图不够稳定：

- `forward()` 有额外参数：`iters`、`scale_iters`、`test_mode`
- 训练模式下返回的是 `disp_predictions` 列表
- 内部存在迭代循环，不同 `iters` 会展开成不同大小的 traced graph
- 有 `external_disp` / `external_valid` 这样的可选分支

所以更稳妥的做法是：

1. 额外包一层 `Wrapper`
2. 在 `Wrapper.forward()` 里固定：
   - `iters`
   - `scale_iters`
   - `test_mode=True`
3. 让模型只返回最终一张视差图

这样 TensorBoard 更容易把图追踪出来。

## 3. 推荐示例

下面这个示例适合直接作为“画当前网络结构图”的参考模板。

你可以把它临时保存成任意脚本，例如 `tools/visualize_graph_example.py`，或者直接按这个逻辑改成你自己的版本。

```python
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from core.defom_stereo import DEFOMStereo


class TensorBoardGraphWrapper(nn.Module):
    def __init__(self, model, iters=4, scale_iters=2):
        super().__init__()
        self.model = model
        self.iters = iters
        self.scale_iters = scale_iters

    def forward(self, image1, image2):
        return self.model(
            image1,
            image2,
            iters=self.iters,
            scale_iters=self.scale_iters,
            test_mode=True,
        )


def build_args():
    return argparse.Namespace(
        dinov2_encoder="vitl",
        idepth_scale=0.5,
        mixed_precision=False,
        n_downsample=2,
        context_norm="batch",
        n_gru_layers=3,
        hidden_dims=[128, 128, 128],
        corr_radius=4,
        corr_levels=2,
        scale_list=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        scale_corr_radius=2,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = build_args()
    base_model = DEFOMStereo(args).to(device)
    base_model.eval()

    model = TensorBoardGraphWrapper(
        base_model,
        iters=4,
        scale_iters=2,
    ).to(device)
    model.eval()

    h, w = 320, 736
    image1 = torch.randint(0, 256, (1, 3, h, w), dtype=torch.float32, device=device)
    image2 = torch.randint(0, 256, (1, 3, h, w), dtype=torch.float32, device=device)

    logdir = "runs/tensorboard_graph/defom_stereo_current"
    writer = SummaryWriter(logdir=logdir)

    with torch.no_grad():
        writer.add_graph(model, (image1, image2))

    writer.close()
    print(f"TensorBoard graph saved to: {logdir}")


if __name__ == "__main__":
    main()
```

## 4. 运行方式

先运行生成图：

```bash
python tools/visualize_graph_example.py
```

然后启动 TensorBoard：

```bash
tensorboard --logdir runs/tensorboard_graph
```

浏览器打开后，在 `Graphs` 页签里看结构图。

## 5. 这个示例画出来的图，对应当前模型哪些部分

如果你用的是上面的 `Wrapper`，图里通常能看到下面几块主结构。

### 5.1 `defomencoder`

这是 `DEFOMStereo` 的深度引导编码入口，内部会进一步进入 `DepthAnythingV2`。

你可以把它理解为：

- 左右图先拼 batch
- 重采样到 `DepthAnythingV2` 需要的尺寸
- 输出深度相关特征
- 生成初始视差

如果没有传入 `external_disp`，初始视差来自：

```python
dav2_disp = self._dav2_init_disp(idepth, ow)
```

### 5.2 `fnet`

这部分对应当前已经修改过的 `BasicEncoder`，它不再只是普通卷积 stem，而是：

- `conv1_std`
- `conv1_def`
- `guide_reduce`
- `guide_refine`
- `offset_head`
- `mask_head`
- `alpha_head`
- `layer1 / layer2 / layer3`
- `convd`
- `conv2`

在 TensorBoard 图里，和原始版本最明显的区别就是：

- 会出现一条由 `dfeats` 引导出的 offset / mask / alpha 分支
- 输入图像会同时经过标准卷积与可变形卷积分支
- 两条 stem 分支最后融合

### 5.3 `cnet`

`MultiBasicEncoder` 是 context encoder，用来生成多尺度 GRU 所需特征。

它的典型结构是：

- `conv1`
- `layer1`
- `layer2`
- `layer3`
- `layer4`
- `layer5`
- `conv08 / conv16 / conv32`
- `outputs08 / outputs16 / outputs32`

这里可以理解成：

- 图像 backbone 抽特征
- DepthAnything 提供的多尺度引导特征在 `/8`、`/16`、`/32` 三个尺度注入
- 输出给后面的 update block

### 5.4 `CorrBlock1D`

这是左右特征之间的 1D 相关性查询模块。  
它不是单纯一个卷积层，更像“相关体构建 + 查询”。

### 5.5 `scale_update_block` 和 `update_block`

这两个块负责迭代更新视差：

- `scale_update_block`
  - 前 `scale_iters` 次使用
  - 更新形式近似为 `disp = scale_disp * disp`
- `update_block`
  - 后续迭代使用
  - 更新形式近似为 `disp = disp + delta_disp`

因此在 TensorBoard 里，你会看到**迭代展开后的重复子图**。  
如果你把 `iters=4` 改成 `iters=12`，图会明显更大。

## 6. 建议的画图参数

为了让图不要过大，建议第一次先用：

```python
iters = 4
scale_iters = 2
```

原因很简单：

- 当前模型有循环迭代
- `add_graph()` 追踪的是一次固定前向
- 迭代次数越大，Graph 越臃肿

如果你的目标只是看主干结构，`iters=2~4` 基本就够了。

## 7. 如果你想专门突出“修改后的新结构”

如果你的重点不是整网，而是想把“修改后的 `BasicEncoder`”看清楚，更推荐单独对 `BasicEncoder` 画图。

因为整网图里：

- `DepthAnythingV2`
- correlation
- GRU 迭代

这些部分会把图撑得很复杂，导致新加的 deformable stem 不够显眼。

单独画 `BasicEncoder` 时，可以只喂：

- 一张 RGB 特征输入
- 一张对应的 `dfeats`

这样更容易观察下面这条关键路径：

```text
dfeats
  -> guide_reduce
  -> guide_refine
  -> offset_head / mask_head / alpha_head
  -> conv1_def
RGB
  -> conv1_std
融合
  -> residual layers
  -> conv2
```

## 8. 常见问题

### 8.1 为什么 Graph 页面很大

因为当前模型不是一次性前向到底，而是包含：

- 多尺度编码
- correlation 查询
- 多轮迭代更新

这类模型本来就会比普通分类网络复杂很多。

### 8.2 为什么训练脚本里没直接看到 `add_graph`

当前 `train_stereo.py` 已经有 TensorBoard 的：

- `add_scalar`
- `add_image`

但没有单独写入 `add_graph`。  
也就是说，**训练日志和结构图日志目前是两回事**，结构图更适合单独跑一次示例脚本来导出。

### 8.3 没有 DepthAnythingV2 预训练权重能不能画图

可以。  
当前 `DefomEncoder` 只有在本地存在：

```text
./checkpoints/depth_anything_v2_{dinov2_encoder}.pth
```

时才会加载权重。没有这个文件时，仍然可以实例化模型并导出 graph，只是权重不是预训练值。

## 9. 一句话总结

对于当前这版 DEFOM-Stereo，最推荐的 TensorBoard 画图方法是：

1. 用 `Wrapper` 固定 `iters`、`scale_iters`、`test_mode=True`
2. 用随机假输入调用 `writer.add_graph(...)`
3. 优先先用较小 `iters` 导图
4. 如果想突出本次结构修改，单独对 `BasicEncoder` 画图比整网更清楚

如果你后面希望，我也可以继续把上面的示例直接落成一个真实可运行的 `tools/*.py` 脚本，而不只是文档示例。

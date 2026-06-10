# DEFOM-Stereo 的 Middlebury 训练初步方案

这份文档面向当前仓库的实际实现，目标是先把“如何用 Middlebury 训练这个项目”梳理清楚，尤其是数据集是怎么从磁盘读进来、经过哪些变换、最后怎么送进模型和 loss 的。

重点不是泛泛讲一个 Middlebury 训练流程，而是把这个项目当前代码里的真实路径讲明白，后面你再改训练策略或者数据集混合配比时，就知道应该改哪里。


## 1. 目标

当前的直接目标是：

- 基于这个仓库现有训练框架
- 用 Middlebury 相关数据对 DEFOM-Stereo 做训练或微调
- 后续用于验证你新加的 `DepthAnythingV2` 引导 deformable stem 是否有效

第一轮建议尽量保守，优先复用项目已有的 Middlebury 训练 recipe，不要一开始同时改：

- 网络结构
- 数据混合方式
- 训练轮次
- 增强策略

这样更容易判断性能变化到底来自你的 deformable stem，还是来自训练配方变化。


## 2. 训练入口在哪里

训练入口是 [train_stereo.py](D:/project/DEFOM-Stereo/train_stereo.py:41)。

主流程是：

1. 解析命令行参数
2. 构建 `DEFOMStereo(args)`
3. 如果传了 `--resume_ckpt`，先加载已有 checkpoint
4. 调用 `datasets.fetch_dataset(args)` 构建训练集
5. 再用 `torch.utils.data.DataLoader` 包一层
6. 进入训练循环
7. 前向得到 `disp_predictions`
8. 用 `sequence_loss(...)` 计算损失
9. 定期保存 checkpoint 和做验证

这里有一个细节要特别记住：

- `train_stereo.py` 实际用的是 `fetch_dataset(args)`
- 不是 `fetch_dataloader(args)`

也就是说，真正的 `DataLoader` 是在 [train_stereo.py](D:/project/DEFOM-Stereo/train_stereo.py:117) 里创建的，而不是在 `core/stereo_datasets.py` 里直接返回的。


## 3. Middlebury 是怎么接进训练流程的

Middlebury 相关代码主要在两个位置：

- 数据集定义：[core/stereo_datasets.py](D:/project/DEFOM-Stereo/core/stereo_datasets.py:237)
- 官方训练脚本：[scripts/train_middlebury.sh](D:/project/DEFOM-Stereo/scripts/train_middlebury.sh:1)

在 `fetch_dataset(args)` 里，只要数据集名字满足：

```python
dataset_name.startswith("middlebury_")
```

就会走到：

```python
Middlebury(aug_params, split=dataset_name.replace('middlebury_', '')) * fold
```

所以这个仓库当前已经支持下面这些训练集名字：

- `middlebury_F`
- `middlebury_H`
- `middlebury_Q`
- `middlebury_2005`
- `middlebury_2006`
- `middlebury_2014`
- `middlebury_2021`


## 4. 数据目录要求

代码默认要求 Middlebury 放在：

```text
./datasets/Middlebury
```

训练时当前代码没有提供单独的 CLI 参数去改这个路径，所以最省事的做法是直接按仓库默认目录组织。

目录结构应当至少满足：

```text
datasets/Middlebury/
  2005/
  2006/
  2014/
  2021/
  MiddEval3/
    trainingF/
    trainingH/
    trainingQ/
    testF/
    testH/
    testQ/
    official_train.txt
```

如果你的数据不在这里，有两种办法：

- 在本地做一个同结构目录或软链接
- 修改 [core/stereo_datasets.py](D:/project/DEFOM-Stereo/core/stereo_datasets.py:237) 给 `Middlebury` 增加自定义 root

第一轮建议先不要改代码，先把目录按默认约定摆好。


## 5. `Middlebury` 数据集类的基本行为

`Middlebury` 类定义在 [core/stereo_datasets.py](D:/project/DEFOM-Stereo/core/stereo_datasets.py:237)。

它继承自 `StereoDataset`，初始化时传入了两个很关键的设置：

- `sparse=True`
- `reader=frame_utils.readDispMiddlebury`

这意味着两件事：

1. Middlebury 在这个仓库里走的是稀疏视差增强路径，也就是 `SparseDispAugmentor`
2. 视差读取使用的是 `readDispMiddlebury()`，不是别的 reader

虽然 Middlebury 某些版本本身并不算严格意义上的稀疏标注，但这个仓库统一把它按 `sparse=True` 处理，所以你看训练行为时要以代码为准，不要只凭数据集印象判断。


## 6. 不同 Middlebury split 是怎么枚举样本的

### 6.1 `middlebury_F` / `middlebury_H` / `middlebury_Q`

这三个 split 走的是 MiddEval3 路径，代码在 [core/stereo_datasets.py](D:/project/DEFOM-Stereo/core/stereo_datasets.py:277)。

逻辑是：

- 如果 `image_set='training'`
  - 先列出 `MiddEval3/trainingF/*` 下的所有场景名
- 如果 `is_eval=True`
  - 再用 `MiddEval3/official_train.txt` 过滤场景
- 然后根据 split 去实际拼接路径：
  - 左图：`MiddEval3/training{split}/{scene}/im0.png`
  - 右图：`MiddEval3/training{split}/{scene}/im1.png`
  - GT：`MiddEval3/training{split}/{scene}/disp0GT.pfm`

这个实现有个值得注意的点：

- 场景名是先从 `trainingF/*` 取出来的
- 然后复用于 `trainingH` 和 `trainingQ`

所以默认假设：

- `trainingF`
- `trainingH`
- `trainingQ`

这三个目录下场景名是一致的。

换句话说：

- `middlebury_F` 训练时读 `trainingF`
- `middlebury_H` 训练时读 `trainingH`
- `middlebury_Q` 训练时读 `trainingQ`


### 6.2 `middlebury_2014`

`2014` split 会扫描：

```text
datasets/Middlebury/2014/<scene>/
```

然后为每个场景构造 3 组图像对：

- `im0.png` 和 `im1.png`
- `im0.png` 和 `im1E.png`
- `im0.png` 和 `im1L.png`

对应视差：

- 左视差：`disp0.pfm`
- 右视差：`disp1.pfm`

所以 `2014` 这里相当于把同一场景的不同成像条件都利用起来了。


### 6.3 `middlebury_2021`

`2021` split 会扫描：

```text
datasets/Middlebury/2021/data/<scene>/
```

首先加入一组基础样本：

- `im0.png`
- `im1.png`

然后再看是否存在 ambient 变体：

- `ambient/L0/im0e0.png` ... `im0e3.png`
- `ambient/L0/im1e0.png` ... `im1e3.png`

只要这些文件存在，就继续加入对应 pair。

对应视差还是：

- `disp0.pfm`
- `disp1.pfm`


### 6.4 `middlebury_2005` / `middlebury_2006`

这两个老 split 的逻辑类似：

- 默认加入 `view1.png` 和 `view5.png`
- 再遍历 `Illum*/Exp*/` 下的变体图像

视差文件是：

- `disp1.png`
- `disp5.png`

所以这两个版本也不是每个场景只对应一个样本，而是会把照明和曝光变化都展开。


## 7. 一个样本是怎么从磁盘读进来的

通用读取逻辑在 [StereoDataset.__getitem__](D:/project/DEFOM-Stereo/core/stereo_datasets.py:37)。

训练态下，一个样本大致经过下面这些步骤：

1. 根据 index 取出 `image_list[index]` 和 `disparity_list[index]`
2. 调用 `self.disparity_reader(...)` 读取视差
3. 调用 `frame_utils.read_gen(...)` 读取左右图
4. 如果这个数据集提供左右双向视差，并且满足随机条件，可能会做左右交换增强
5. 如果图像是灰度图，扩成 3 通道
6. 调用 augmentor 做数据增强
7. 转成 PyTorch tensor
8. 返回一个字典

返回格式是：

```python
{
    "img1": img1,
    "img2": img2,
    "disp": disp,
    "valid": valid,
    "imageL_file": self.image_list[index][0],
    "disp_file": self.disparity_list[index][0],
}
```

后面 `train_stereo.py` 就是直接从这个字典里拿：

- `img1`
- `img2`
- `disp`
- `valid`


## 8. Middlebury 的视差是怎么读的

Middlebury 的 reader 是 [readDispMiddlebury](D:/project/DEFOM-Stereo/core/utils/frame_utils.py:239)。

它支持：

- `disp0GT.pfm`
- `disp1GT.pfm`
- `disp0.pfm`
- `disp1.pfm`
- `.png`

对于 PFM 文件，这个函数当前使用的有效像素规则是：

```python
valid = disp < 1e3
```

也就是说，在这个仓库里：

- 只要视差值不是特别大的无效哨兵值
- 就认为该像素有效

这里要特别注意，它没有使用更严格的 non-occlusion mask。

同一个文件里其实还有一个 `readDispMiddlebury0()`，它会去读：

- `mask0nocc.png`
- `mask1nocc.png`

但当前训练代码并没有走这个版本。

这意味着当前 Middlebury 训练监督并不是“只用官方 non-occluded 区域”，而是使用了更宽的有效区域判定。


## 9. Middlebury 在这个项目里为什么走 `sparse=True`

这和增强方式直接相关。

因为 `Middlebury` 初始化时设了 `sparse=True`，所以如果传入了 `aug_params`，在 `StereoDataset` 里就会构建：

- `SparseDispAugmentor`

而不是：

- `DispAugmentor`

这部分逻辑在 [core/stereo_datasets.py](D:/project/DEFOM-Stereo/core/stereo_datasets.py:22)。

对训练来说，这个区别很重要，因为稀疏增强在 resize disparity 时不会像 dense map 那样直接插值整张图，而是只对有效点做坐标缩放，再重新散射回新图。

这样能避免在无效区域“插值出假视差”。


## 10. `SparseDispAugmentor` 到底做了什么

对应实现是 [core/utils/augmentor.py](D:/project/DEFOM-Stereo/core/utils/augmentor.py:155)。

它主要分 3 步：

1. 颜色增强
2. 右图随机 eraser
3. 空间增强 + crop

### 10.1 颜色增强

对左右图拼接后的结果做 `ColorJitter + Gamma` 变换。

### 10.2 Eraser 增强

随机在右图抹掉一些矩形区域，用平均颜色填充，模拟遮挡。

### 10.3 空间增强

这里是最关键的部分：

- 随机采样缩放比例
- resize 左右图
- 调用 `resize_sparse_flow_map()` 重建稀疏视差图和 valid mask
- 最后随机 crop 到 `args.image_size`

`resize_sparse_flow_map()` 的思路是：

- 先把所有有效视差点取出来
- 只缩放这些有效点的坐标和值
- 再把它们写回一张新的 disparity/valid 图

所以 Middlebury 在当前仓库中的训练数据流是：

- 原始图像和视差
- 稀疏增强
- crop
- tensor 化
- 送入模型


## 11. fold 倍数是怎么工作的

训练脚本里经常会看到：

- `middlebury_H * 200`
- `middlebury_2021 * 200`

这不是把文件真实复制 200 份，而是利用了 `StereoDataset.__mul__()`。

对应代码在 [core/stereo_datasets.py](D:/project/DEFOM-Stereo/core/stereo_datasets.py:116)。

本质做法是：

- 记录一个倍数 `self.v`
- `__len__()` 返回 `len(self.image_list) * self.v`
- `__getitem__()` 再通过取模映射回原始样本列表

所以它的效果是“过采样”，不是“复制数据”。

这对 Middlebury 很重要，因为 Middlebury 数据量本身不大，训练时必须靠 fold 提高采样频率，让它在混合训练里占更高比重。


## 12. 项目里现成的 Middlebury 两阶段训练策略

官方脚本在 [scripts/train_middlebury.sh](D:/project/DEFOM-Stereo/scripts/train_middlebury.sh:1)。

它实际上分两个阶段。


### 阶段 A：Middlebury pretrain

从这个 checkpoint 恢复：

```text
checkpoints/defomstereo_vitl_sceneflow.pth
```

训练数据混合为：

- `tartan_air`
- `sceneflow`
- `falling_things`
- `instereo2k`
- `carla_highres`
- `crestereo`
- `middlebury_2014`
- `middlebury_2021`
- `middlebury_H`

对应 fold：

- `1 1 1 50 50 1 200 200 200`

主要超参：

- `num_steps=200000`
- `image_size=384 512`
- `dinov2_encoder=vitl`
- `n_downsample=2`
- `train_iters=18`
- `scale_iters=8`


### 阶段 B：最终 Middlebury fine-tune

从这个 checkpoint 恢复：

```text
checkpoints/defomstereo_vitl_middlebury_pretrain.pth
```

训练数据混合为：

- `crestereo`
- `instereo2k`
- `carla_highres`
- `middlebury_2014`
- `middlebury_2021`
- `middlebury_H`
- `middlebury_F`
- `falling_things`

对应 fold：

- `1 50 50 200 200 200 200 5`

主要超参：

- `num_steps=100000`
- `image_size=512 768`
- `dinov2_encoder=vitl`

这就是当前仓库作者对 Middlebury 的现成 recipe。你第一轮最好不要偏离它太多。


## 13. 训练循环真正吃到的是什么

在 [train_stereo.py](D:/project/DEFOM-Stereo/train_stereo.py:141) 里，每个 batch 会取出：

- `data_blob["img1"]`
- `data_blob["img2"]`
- `data_blob["disp"]`
- `data_blob["valid"]`

然后前向调用：

```python
disp_predictions = model(image1, image2, iters=args.train_iters, scale_iters=args.scale_iters)
```

之后：

```python
loss, metrics = sequence_loss(disp_predictions, disp_gt, valid)
```

这里有个对你当前改动非常关键的结论：

- 数据集不会额外给出 `DepthAnythingV2` 特征
- 这些特征是在 `DEFOMStereo` 模型内部自己算的

所以你现在把 `BasicEncoder` 改成 deformable stem 之后，数据读取流程本身不需要跟着改。

这点非常重要，因为说明你当前这次结构修改没有破坏 dataset API。


## 14. 针对你当前 deformable stem 的第一轮训练建议

建议采用“只改网络，不改 recipe”的方式先做第一轮对比。

### 建议方案

1. 先确认当前改后的模型还能不能加载已有的 SceneFlow 预训练权重
2. 如果能加载，就直接从 `defomstereo_vitl_sceneflow.pth` 开始
3. 完整跑一遍原始的 Middlebury pretrain
4. 再完整跑一遍原始的 Middlebury fine-tune
5. 最后和未修改网络的 baseline 比：
   - `middlebury_F`
   - `middlebury_H`
   - `middlebury_Q`

### 原因

- 这样最容易隔离 deformable stem 的真实贡献
- 不会把结论和数据混合策略耦合在一起
- 如果结果变差，也更容易定位是结构问题，而不是训练配方问题


## 15. 这次改动下要特别注意的风险

### 风险 1：旧 checkpoint 不能严格加载

因为你改了 `BasicEncoder`，参数名和参数形状有变化的概率很高。

你要重点检查：

- 哪些 key missing
- 哪些 key unexpected
- 是否只有新加的 stem/guide head 参数未命中

如果发现旧的核心主干参数也大面积对不上，那就不能直接沿用原 checkpoint。


### 风险 2：Middlebury 过采样后可能很快过拟合

项目脚本里对：

- `middlebury_2014`
- `middlebury_2021`
- `middlebury_H`
- `middlebury_F`

给的 fold 都比较大。

这在原始模型上未必有问题，但你改了 feature stem 之后，收敛动态可能会变。

第一轮不要继续加大 fold。


### 风险 3：自定义 `DeformConv2dBlock` 速度可能明显慢于普通卷积

因为当前实现是：

- 手工构建 grid
- `grid_sample`
- `einsum`

这通常比 cuDNN 优化过的普通 `Conv2d` 更慢。

所以在开长程训练前，最好先做一个短 smoke test，看单 iter 时间和显存变化。


## 16. 推荐的实际执行顺序

### 第一步：检查目录

确认下面这些路径真实存在：

- `datasets/Middlebury/MiddEval3/trainingF/...`
- `datasets/Middlebury/MiddEval3/trainingH/...`
- `datasets/Middlebury/MiddEval3/trainingQ/...`
- `datasets/Middlebury/2014/...`
- `datasets/Middlebury/2021/data/...`

### 第二步：做 dataset sanity check

建议写一个非常小的检查脚本，至少打印：

- `Middlebury(split="H")` 的长度
- 第一个样本的左右图路径
- 第一个样本的 `disp_file`
- `img1/img2/disp/valid` 的 shape
- `valid` 的有效像素比例

这个步骤能帮你确认：

- 路径没写错
- reader 能正常读
- 增强后尺寸符合预期
- valid mask 没有全空

### 第三步：做短程训练 smoke test

先跑很短一段，比如：

- 100 到 500 step

主要看：

- 能不能正常前向反向
- 会不会 NaN
- checkpoint 能不能加载
- deformable stem 会不会引起显存爆炸或严重变慢

### 第四步：跑阶段 A

按原 `train_middlebury.sh` 的 pretrain 配方先跑完整阶段。

### 第五步：跑阶段 B

再按原脚本的 final fine-tune 配方跑第二阶段。

### 第六步：统一评测

最后至少评测：

- `middlebury_F`
- `middlebury_H`
- `middlebury_Q`

再和原始模型比较。


## 17. 一个当前阶段最务实的训练策略

如果你的目标是“先尽快得到一个可信的初版结果”，建议直接沿用原脚本的两阶段结构，只改实验名，例如：

- `defomstereo_vitl_middlebury_pretrain_dcnstem`
- `defomstereo_vitl_middlebury_dcnstem`

这样好处是：

- 与原始模型实验记录容易一一对应
- 训练日志和 checkpoint 不会混淆
- 后面做 ablation 更清楚


## 18. 目前这份方案里最关键的结论

可以先把最重要的几点压缩成下面几条：

- Middlebury 训练入口在 `train_stereo.py + fetch_dataset()`
- `middlebury_*` 通过 `core/stereo_datasets.py` 中的 `Middlebury` 类构建
- 当前仓库把 Middlebury 作为 `sparse=True` 数据集处理
- 视差读取使用 `readDispMiddlebury()`，有效像素规则主要是 `disp < 1e3`
- 训练样本最终返回的是 `img1/img2/disp/valid` 这个字典
- 你的 deformable stem 不需要改 dataset API，因为 DAV2 特征在模型内部生成
- 第一轮实验最好保持原始 Middlebury 两阶段训练 recipe 不变


## 19. 下一步建议

在真正开始大规模训练前，最值得立刻做的不是改更多网络代码，而是先补一个小的 dataset 检查脚本。

这个脚本只要做一件事：

- 真正实例化 `Middlebury(split="H")`
- 取一个样本
- 把 shape、文件名、valid ratio 打印出来

只要这一步完全确认了，你后面排查训练问题时就能少掉一大块不确定性。


## 20. 已发生的实际训练尝试记录

这里记录一次已经发生的真实尝试，方便后面继续调参时有基线可对照。

### 20.1 第一版配置：未跑通，CUDA OOM

最初尝试的是更接近原始大模型配置的一版，核心特征是：

- `dinov2_encoder=vitl`
- `resume_ckpt=checkpoints/defomstereo_vitl_sceneflow.pth`
- `batch_size=8`
- 4 张卡训练，因此每卡实际 batch 为 `2`
- `image_size=384 512`
- 未使用更保守的小模型设置

对应的完整训练命令为：

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=9993 train_stereo.py \
  --distributed \
  --launcher pytorch \
  --gpu_ids 0 1 2 3 \
  --name defomstereo_vitl_middeval3_x100 \
  --batch_size 8 \
  --num_workers 8 \
  --train_datasets middlebury_F middlebury_H middlebury_Q \
  --train_folds 100 100 100 \
  --num_steps 100000 \
  --n_downsample 2 \
  --train_iters 18 \
  --scale_iters 8 \
  --idepth_scale 0.5 \
  --corr_levels 2 \
  --corr_radius 4 \
  --scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
  --scale_corr_radius 2 \
  --dinov2_encoder vitl \
  --image_size 384 512 \
  --resume_ckpt checkpoints/defomstereo_vitl_sceneflow.pth
```

这版在训练启动后出现了显存不足：

- 单卡总显存约 `11GB`
- 报错时每卡只剩下约 `138MB` 可用显存
- PyTorch 已分配显存约 `10GB`

从现象上看，导致 OOM 的主要原因是几个高开销因素叠加：

1. `vitl` 本身比 `vits` 重很多
2. `384x512` 输入分辨率对 DAV2 特征和后续 stereo 特征都会明显增大显存占用
3. 总 `batch_size=8` 在 4 卡下意味着每卡 batch 为 `2`
4. 你当前加入的 deformable stem 额外引入了：
   - guide feature 分支
   - offset/mask/alpha head
   - `grid_sample` 与手工聚合

其中真正最可能把显存推爆的是前三项，deformable stem 则进一步加重了压力。


### 20.2 第二版配置：已跑通

后续改成了一版更保守的配置，能够正常训练。核心参数是：

- `dinov2_encoder=vits`
- `resume_ckpt=checkpoints/defomstereo_vits_sceneflow.pth`
- `batch_size=4`
- 4 张卡训练，因此每卡实际 batch 为 `1`
- `image_size=320 448`
- `--mixed_precision`

同时还附加了：

```bash
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

对应的完整训练命令为：

```bash
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=9993 train_stereo.py \
  --distributed \
  --launcher pytorch \
  --gpu_ids 0 1 2 3 \
  --name defomstereo_vits_middeval3_x100 \
  --batch_size 4 \
  --num_workers 8 \
  --train_datasets middlebury_F middlebury_H middlebury_Q \
  --train_folds 100 100 100 \
  --num_steps 100000 \
  --n_downsample 2 \
  --train_iters 18 \
  --scale_iters 8 \
  --idepth_scale 0.5 \
  --corr_levels 2 \
  --corr_radius 4 \
  --scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
  --scale_corr_radius 2 \
  --dinov2_encoder vits \
  --image_size 320 448 \
  --mixed_precision \
  --resume_ckpt checkpoints/defomstereo_vits_sceneflow.pth
```

这版能够跑通的主要原因是：

1. `vits` 的显存占用显著低于 `vitl`
2. 输入分辨率从 `384x512` 降到 `320x448`
3. 总 batch 从 `8` 降到 `4`，每卡 batch 从 `2` 降到 `1`
4. 开启了混合精度，进一步减少激活和中间张量的显存占用

从实际 `nvidia-smi` 观察结果看，这一版运行时每张卡大约使用：

- `~5.2GB / 11.3GB`

GPU 利用率大约在：

- `59% ~ 72%`

这说明第二版虽然稳定，但整体仍然偏保守：

- 显存没有吃满
- GPU 利用率也没有完全跑满


### 20.3 对这两版差异的直接结论

这次实验说明：

- 第一版 OOM 不是代码逻辑错误，而是配置过重
- 第二版能够正常跑通，说明当前修改后的网络整体是可训练的
- 目前的瓶颈主要是显存配置，不是数据读入或权重加载

也就是说，当前最合理的策略不是继续盲目降配置，而是在“已跑通的第二版”基础上逐步往上加。


### 20.4 下一步可能的改进方向

后续调参建议一次只改一个变量，这样才能知道到底是哪一项导致显存或吞吐变化。

推荐顺序如下。

#### 方案 A：先提高输入分辨率

保持当前可跑的 `vits` 配置不变，只把：

```bash
--image_size 320 448
```

提高到：

```bash
--image_size 384 512
```

优先做这一步的原因是：

- 分辨率对立体匹配效果通常比 batch size 更直接
- 当前显存只占用了约一半，最值得优先把分辨率加回去试


#### 方案 B：在方案 A 稳定后，再考虑增大 batch

如果 `vits + 384x512` 还能稳定运行，可以再尝试：

```bash
--batch_size 8
```

也就是把每卡 batch 从 `1` 提高到 `2`。

这一步能提升吞吐，但未必像提升分辨率那样直接改善效果，因此优先级低于方案 A。


#### 方案 C：最后再尝试切回 `vitl`

如果你的目标是尽量接近原始大模型配置，再尝试：

- `dinov2_encoder=vitl`
- `resume_ckpt=checkpoints/defomstereo_vitl_sceneflow.pth`

但这一步应当放在最后，因为它最容易再次触发 OOM。


### 20.5 当前最务实的建议

如果目标是先稳定做实验并尽快拿到可比较结果，当前建议是：

1. 先用已经跑通的第二版配置作为正式起点
2. 先尝试把分辨率提升回 `384x512`
3. 如果仍稳定，再尝试提高 batch size
4. 最后才考虑换回 `vitl`

这样做的好处是：

- 能逐步逼近原始配置
- 每一步都能明确知道新增的显存开销来自哪里
- 不会因为一次改太多参数而失去判断依据

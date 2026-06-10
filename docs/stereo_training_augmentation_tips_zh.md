# 立体匹配训练技巧：DEFOM-Stereo 的数据增强与 GT 同步规则

本文记录当前仓库里和立体匹配训练最相关的一部分技巧：**数据增强如何作用在左右图像与视差 GT 上**。  
重点不是“有哪些增强名字”，而是**每种增强是否需要同步修改 disparity / valid mask，以及为什么**。

对应实现文件：

- `core/utils/augmentor.py`
- `core/stereo_datasets.py`
- `train_stereo.py`

## 1. 为什么 stereo 的 augmentation 比分类更麻烦

分类任务里，大多数增强只要改输入图像就够了。  
但在立体匹配里，标签不是一个类别，而是**像素级视差场**，所以一旦做了空间变换：

- 图像坐标变了
- 视差数值定义也可能变
- 稀疏 GT 的有效点位置也可能变

如果只增强图像、不增强 GT，很容易造成：

- 输入与标签不对齐
- 训练噪声变成系统性错误
- loss 看起来还能下降，但模型学到的是错的几何关系

所以这类任务里，**增强策略本身就是训练稳定性的关键部分**。

## 2. 当前项目的增强入口

数据增强不是在训练主循环里做的，而是在 dataset 取样阶段做的。

在 [core/stereo_datasets.py](D:/project/DEFOM-Stereo/core/stereo_datasets.py:22) 中：

- 稠密视差数据走 `DispAugmentor`
- 稀疏视差数据走 `SparseDispAugmentor`

具体入口：

- 稠密数据增强调用见 [core/stereo_datasets.py](D:/project/DEFOM-Stereo/core/stereo_datasets.py:115)
- 稀疏数据增强调用见 [core/stereo_datasets.py](D:/project/DEFOM-Stereo/core/stereo_datasets.py:113)

训练参数入口在 [train_stereo.py](D:/project/DEFOM-Stereo/train_stereo.py:350) 附近。

## 3. 当前项目实际用了哪些增强

### 3.1 光照/颜色增强

实现位置：

- [core/utils/augmentor.py](D:/project/DEFOM-Stereo/core/utils/augmentor.py:88)
- [core/utils/augmentor.py](D:/project/DEFOM-Stereo/core/utils/augmentor.py:205)

当前包括：

- `ColorJitter`
  - brightness
  - contrast
  - saturation
  - hue
- `AdjustGamma`

作用对象：

- `img1`
- `img2`

不修改：

- `disp`
- `valid`

原因：

- 这类增强只改变外观，不改变几何
- 视差定义仍然成立

当前代码还区分了：

- 对左右图分别独立增强的 asymmetric augmentation
- 对左右图共同增强的 symmetric augmentation

这对 stereo 是有意义的，因为真实双目相机并不总是严格同曝光、同颜色响应。

### 3.2 Eraser / 遮挡增强

实现位置：

- [core/utils/augmentor.py](D:/project/DEFOM-Stereo/core/utils/augmentor.py:104)
- [core/utils/augmentor.py](D:/project/DEFOM-Stereo/core/utils/augmentor.py:211)

做法：

- 在 `img2` 上随机抹掉 1 到 2 个矩形区域
- 用图像均值颜色填充

不修改：

- `disp`
- `valid`

原因：

- 这是在模拟匹配中的遮挡或局部无纹理
- 几何没有被重新定义
- GT 仍然是左图对应像素的真实视差

这是 stereo 训练里很常见的一种鲁棒性增强。

### 3.3 空间缩放增强

实现位置：

- 稠密视差：[core/utils/augmentor.py](D:/project/DEFOM-Stereo/core/utils/augmentor.py:119)
- 稀疏视差：[core/utils/augmentor.py](D:/project/DEFOM-Stereo/core/utils/augmentor.py:258)

这部分最关键，因为它不只是 resize 图像，还必须同步处理 GT。

#### 稠密视差的处理

代码核心是：

```python
img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
disp = cv2.resize(disp, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
disp = disp * scale_x
```

为什么要乘 `scale_x`：

- disparity 是**水平像素位移**
- 当图像宽度缩放后，像素坐标系发生变化
- 视差数值也必须按水平方向缩放比例同步变化

这一点是 stereo augmentation 里最容易犯错的地方之一。

#### 稀疏视差的处理

稀疏 GT 不能简单对 `disp` 做双线性插值。  
当前实现用了 [resize_sparse_flow_map()](D:/project/DEFOM-Stereo/core/utils/augmentor.py:224)：

1. 先提取 `valid` 点
2. 对有效点坐标做缩放
3. 对视差值做 `fx` 缩放
4. 再把这些点重新投回新的稀疏图上

这比直接 resize 更合理，因为：

- 稀疏点不应该被虚构插值成稠密点
- `valid` mask 也需要跟着重建

### 3.4 随机裁剪

实现位置：

- 稠密视差：[core/utils/augmentor.py](D:/project/DEFOM-Stereo/core/utils/augmentor.py:151)
- 稀疏视差：[core/utils/augmentor.py](D:/project/DEFOM-Stereo/core/utils/augmentor.py:288)

处理方式：

- `img1`、`img2`、`disp` 同步 crop
- 稀疏数据还要同步 crop `valid`

这是最基础、也最必要的空间增强。  
如果 crop 不同步，训练会立即失真。

### 3.5 垂直翻转

实现位置：

- 稠密视差：[core/utils/augmentor.py](D:/project/DEFOM-Stereo/core/utils/augmentor.py:145)
- 稀疏视差：[core/utils/augmentor.py](D:/project/DEFOM-Stereo/core/utils/augmentor.py:281)

当前只支持：

- `v-flip`

代码逻辑是：

- 图像上下翻
- `disp` 上下翻
- 稀疏数据时 `valid` 也上下翻

为什么这里不用改视差数值本身：

- disparity 表示的是**水平位移**
- 上下翻只改变像素的纵向位置
- 不改变水平位移的数值定义

### 3.6 `yjitter`：模拟非完美极线校正

实现位置：

- [core/utils/augmentor.py](D:/project/DEFOM-Stereo/core/utils/augmentor.py:151)

这个增强不是普通的“同步裁剪”，而是：

- 左图从 `(y0, x0)` crop
- 右图从 `(y1, x0)` crop
- 其中 `y1` 相比 `y0` 允许有小范围偏移
- 视差 `disp` 仍然按左图区域 crop

这相当于人为给左右图加入少量垂直不对齐，用来模拟：

- 非完美双目标定
- 轻微极线校正误差

这不是 bug，而是一个有目的的鲁棒性增强。

## 4. 当前项目没有做什么增强

### 4.1 没有做水平翻转

训练参数里虽然有 `do_flip`，但当前逻辑只对 `do_flip == 'v'` 时生效，见 [core/utils/augmentor.py](D:/project/DEFOM-Stereo/core/utils/augmentor.py:146)。

也就是说，这个项目**没有做 horizontal flip**。

这是合理的，因为水平翻转对 stereo 来说风险很高：

- 左右相机关系会变
- disparity 符号和参考坐标系可能需要重定义
- 很容易把标签处理错

很多 stereo 项目都会避免这一类增强，除非专门设计过。

### 4.2 没有做旋转、仿射、透视形变

当前增强里没有看到：

- 随机旋转
- 任意仿射变换
- 透视扰动

这也很合理，因为这些操作会显著破坏极线几何。  
除非你同步重建新的 stereo 几何关系，否则通常不值得在双目匹配里乱加。

## 5. 稠密 GT 和稀疏 GT 的处理差异

这是当前实现里另一个值得注意的点。

### 5.1 稠密数据

稠密数据增强后，代码最后使用：

```python
valid = disp < 512
```

见 [core/stereo_datasets.py](D:/project/DEFOM-Stereo/core/stereo_datasets.py:123)。

也就是说：

- 稠密数据集默认不依赖单独的稀疏有效性标注
- 主要通过视差阈值定义有效区域

### 5.2 稀疏数据

稀疏数据会显式维护：

- `disp`
- `valid`

增强过程里，`valid` 会跟着一起变化，尤其是在缩放和裁剪时必须同步更新。

## 6. 训练参数里对应哪些 augmentation 开关

在 [train_stereo.py](D:/project/DEFOM-Stereo/train_stereo.py:350) 可以看到当前暴露给训练脚本的增强参数：

- `--img_gamma`
- `--saturation_range`
- `--do_flip`
- `--spatial_scale`
- `--noyjitter`

而在 [core/stereo_datasets.py](D:/project/DEFOM-Stereo/core/stereo_datasets.py:521) 里，这些参数会被整理成：

- `crop_size`
- `min_scale`
- `max_scale`
- `do_flip`
- `yjitter`
- `saturation_range`
- `gamma`

注意一点：

- `yjitter` 不是直接由 `--yjitter` 控制
- 而是由 `--noyjitter` 取反得到

也就是说：

- 默认会启用 `yjitter`
- 传 `--noyjitter` 才关闭

## 7. 对训练最有价值的几点经验

### 7.1 只要改了空间坐标，就必须重新审视 GT

对 stereo 来说，最重要的不是“增强多不多”，而是：

- 图像几何变了没有
- 如果变了，`disp` 数值要不要改
- `valid` 掩码要不要改

这个项目的实现里，缩放和裁剪都对 GT 做了同步处理，这是对的。

### 7.2 `disp` 不是普通 mask，resize 后必须考虑物理含义

语义分割标签 resize 时通常只关心类别插值方式。  
但视差不同，它是带物理意义的数值场。

尤其在宽度缩放后：

- 不能只 resize
- 还必须乘以 `scale_x`

### 7.3 稀疏 GT 不要直接双线性插值

这份代码对稀疏视差单独实现了 `resize_sparse_flow_map()`，这是正确方向。  
如果直接把稀疏 GT 当稠密图 resize，很容易制造出不存在的监督点。

### 7.4 不要轻易加水平翻转

很多视觉任务里 horizontal flip 是默认增强。  
但 stereo 不是。

当前项目避免水平翻转，是一个偏保守但正确的选择。

### 7.5 `yjitter` 很有价值，但要知道它在做什么

`yjitter` 不是“标注错了”，而是在故意制造轻微双目不完美校正。  
如果你的目标场景是真实双目设备，而不是严格合成极线对齐数据，它往往是有帮助的。

## 8. 一句话总结

当前 DEFOM-Stereo 的增强策略可以概括为：

- 外观增强改图像，不改 GT
- 空间增强必须同步改 `disp`
- 稀疏 GT 还要同步改 `valid`
- 避免高风险的水平翻转与复杂几何扰动
- 用 `yjitter` 模拟真实双目的轻微非理想校正

如果后面你要继续做训练策略整理，一个很自然的下一步就是把：

- augmentation
- 数据集混合比例
- `train_iters / scale_iters`
- 各 benchmark 微调配置

合并成一份完整的中文训练手册。

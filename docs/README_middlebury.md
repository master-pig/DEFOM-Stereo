# Middlebury 视差处理工具

基于 [Middlebury Stereo Evaluation](https://vision.middlebury.edu/stereo/) 场景数据，将官方 `disp0.pfm` 真值导出为 PNG，并仿真激光雷达稀疏深度、插值稠密深度，以及将视差反投影为点云。

## 环境

```bash
pip install -r requirements.txt
```

依赖：`numpy`、`Pillow`、`scipy`（`--lidar` 线性插值需要）。

## 数据目录约定

每个场景一个文件夹，例如 `dataset/Adirondack-perfect/`，至少包含：

| 文件 | 说明 |
|------|------|
| `disp0.pfm` | 左图 im0 浮点视差真值（脚本输入） |
| `calib.txt` | 标定：`cam0`、`baseline`、`doffs` 等 |
| `im0.png` / `im1.png` | 左右图（本工具不直接使用，仅供对照） |

标定与深度的关系（与 Middlebury 文档一致）：

```text
Z [mm] = baseline × f / (d + doffs)
```

其中 `f`、`baseline`、`doffs` 来自 `calib.txt` 的 `cam0` 与基线参数。

---

## 脚本一：`export_disparity.py`

### 思路

```text
disp0.pfm (稠密 GT)
    │
    ├─► disp0_gt.png / disp0_color.png     始终输出
    │
    ├─► [--noise] 深度域 N(0, σ) 高斯噪声 → 反投影视差
    │       └─► disp0_noise.png            仍为稠密图
    │
    └─► [--lidar] 仿真激光雷达
            1. 稠密深度 Z（由 GT 视差 + calib 得到）
            2. 可选：与 --noise 相同的标准差加在深度上
            3. 可选：深度量化 (--lidar-quant)
            4. 稀疏采样（默认 20 万点，scan/grid/random）
            5. 栅格化 → 空洞深度/视差
            6. 默认：对稀疏深度做 2D 线性插值 + 凸包外最近邻
            7. 反投影 → 全图稠密深度/视差
```

**设计要点**

- 噪声加在**深度 [mm]** 上再转回视差，而不是直接对视差加噪。
- 激光仿真在**采样后**才产生空洞；`disp0_lidar.png` 中 `0` 表示未测量。
- 稠密激光图 `*_lidar_dense` 在**图像平面 (u,v)** 上对深度线性插值，凸包外区域用最近邻补全；适合作为“带噪、全图有值”的初始化深度/视差，但插值区域并非真实测量。

### 视差 PNG 编码（所有 `disp0_*.png` 一致）

- **16 位灰度**
- `视差(像素) = 像素值 / 256`（支持亚像素）
- `0` = 无效 / 无深度

### 深度 PNG 编码（`depth0_*.png`）

- **16 位灰度**
- `深度(mm) ≈ 像素值`（四舍五入）
- `0` = 无效；大于 65535 mm 的值会被截断

### 输出文件一览

| 文件 | 何时生成 | 含义 |
|------|----------|------|
| `disp0_gt.png` | 默认 | 稠密视差真值 |
| `disp0_color.png` | 默认 | GT 伪彩色（仅可视化） |
| `disp0_noise.png` | `--noise` | 稠密、深度加噪后的视差 |
| `disp0_lidar.png` | `--lidar` | 稀疏视差（~3.5% 像素有值，默认 20 万点） |
| `depth0_lidar.png` | `--lidar` | 稀疏深度 [mm] |
| `disp0_lidar_color.png` | `--lidar` | 稀疏视差伪彩色 |
| `depth0_lidar_dense.png` | `--lidar`（默认插值） | 插值后的全图深度 |
| `disp0_lidar_dense.png` | `--lidar`（默认插值） | 插值后的全图视差 |
| `disp0_lidar_dense_color.png` | `--lidar`（默认插值） | 稠密视差伪彩色 |

### 用法示例

```bash
# 仅导出 GT + 伪彩色
python export_disparity.py dataset/Adirondack-perfect

# 批量：dataset 下所有含 disp0.pfm 的子目录
python export_disparity.py dataset

# 深度加噪（默认 σ = 30 mm = 3 cm）
python export_disparity.py dataset/Adirondack-perfect --noise --seed 0

# 稀疏激光 + 深度噪声 + 线性插值稠密图（推荐一条命令跑全）
python export_disparity.py dataset/Adirondack-perfect --lidar --noise --seed 0

# 指定输出目录
python export_disparity.py dataset/Adirondack-perfect -o output/

# 激光相关参数
python export_disparity.py dataset/Adirondack-perfect --lidar \
  --lidar-points 200000 \
  --lidar-mode scan \
  --lidar-quant 10 \
  --noise --noise-std 30 \
  --seed 0

# 不要稠密插值（仅稀疏 depth/disp）
python export_disparity.py dataset/Adirondack-perfect --lidar --no-lidar-interp
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `path` | `dataset` | 单场景目录或数据集根目录 |
| `-o, --output` | 场景目录 | 输出目录 |
| `--noise` | 关 | 输出 `disp0_noise.png` |
| `--noise-std` | `30` | 深度噪声标准差 [mm] |
| `--lidar` | 关 | 激光稀疏 + 可选稠密插值 |
| `--lidar-points` | `200000` | 最大采样点数 |
| `--lidar-mode` | `scan` | `scan` / `grid` / `random` |
| `--lidar-quant` | `0` | 深度量化步长 [mm]，0 表示关闭 |
| `--lidar-interp` | 开 | 线性插值生成 `*_dense` |
| `--no-lidar-interp` | — | 关闭稠密插值 |
| `--seed` | 随机 | 噪声与采样的随机种子 |

---

## 脚本二：`disparity_to_pcd.py`

### 思路

读取 **任意** 与 `export_disparity.py` 编码一致的视差 PNG 和对应的 `calib.txt`，在**左相机坐标系**下反投影为点云（单位 **mm**），写出 ASCII PCD（PCL / CloudCompare 可读）。

```text
视差 PNG  →  d = value/256（0 跳过）
calib.txt →  f, cx, cy, baseline, doffs
    │
    Z = baseline × f / (d + doffs)
    X = (u - cx) × Z / f
    Y = (v - cy) × Z / f
    │
    └─► *.pcd（仅包含有效视差像素）
```

稀疏图（如 `disp0_lidar.png`）只会生成对应数量的点；稠密图（`disp0_gt.png`、`disp0_lidar_dense.png`）点数为有效像素数（可达数百万）。

### 用法示例

**单文件模式（灵活指定路径，推荐）**

```bash
# 输出到指定文件
python disparity_to_pcd.py \
  --calib dataset/Adirondack-perfect/calib.txt \
  --disp dataset/Adirondack-perfect/disp0_lidar_dense.png \
  -o output/lidar_dense.pcd

# -o 为目录时：output/<视差文件名>.pcd
python disparity_to_pcd.py --calib .../calib.txt --disp .../disp0_gt.png -o output/

# 省略 -o：与视差图同目录，<stem>.pcd
python disparity_to_pcd.py --calib .../calib.txt --disp .../disp0_lidar.png

# 降采样（每 N 像素取 1 个，减小 PCD 体积）
python disparity_to_pcd.py --calib .../calib.txt --disp .../disp0_gt.png -o out.pcd --stride 2
```

**批量模式（按场景目录扫描）**

```bash
python disparity_to_pcd.py dataset/Adirondack-perfect --disp-name disp0_gt.png

python disparity_to_pcd.py dataset --disp-name disp0_lidar_dense.png -o output/
# 多场景时 -o 必须为目录，文件名为 <场景名>_<视差名>.pcd
```

| 参数 | 说明 |
|------|------|
| `--calib` | `calib.txt` 路径（单文件模式必填） |
| `--disp` | 视差 PNG 路径（单文件模式必填） |
| `-o, --output` | 输出 `.pcd` 文件或目录 |
| `--stride` | 像素步长，默认 1 |
| `path` | 批量：场景或数据集根目录 |
| `--disp-name` | 批量时各场景内的视差文件名 |

---

## 典型工作流

```bash
# 1. 从 Middlebury 场景生成 GT、稀疏/稠密激光视差与深度
python export_disparity.py dataset/Adirondack-perfect --lidar --noise --seed 0

# 2. 稀疏点云（约 20 万点）
python disparity_to_pcd.py \
  --calib dataset/Adirondack-perfect/calib.txt \
  --disp dataset/Adirondack-perfect/disp0_lidar.png \
  -o output/sparse.pcd

# 3. 插值后的稠密点云（有效像素与 GT 同量级）
python disparity_to_pcd.py \
  --calib dataset/Adirondack-perfect/calib.txt \
  --disp dataset/Adirondack-perfect/disp0_lidar_dense.png \
  -o output/dense.pcd
```

---

## 延伸阅读

- [`docs/lidar_disparity_for_defom.md`](docs/lidar_disparity_for_defom.md)：将激光仿真视差接入 DEFOM-Stereo（替代 Depth Anything 初始化）的思路与优化方向。

## 文件结构

```text
middlebury/
├── README.md
├── requirements.txt
├── export_disparity.py      # 视差导出与激光仿真
├── disparity_to_pcd.py      # 视差 → PCD
├── docs/
│   └── lidar_disparity_for_defom.md
└── dataset/
    └── <Scene>-perfect/
        ├── calib.txt
        ├── disp0.pfm
        ├── disp0_gt.png
        ├── disp0_lidar.png
        ├── disp0_lidar_dense.png
        └── ...
```

# AIC2026 面向城市场景的视觉多模态目标检测
## 当前工作总结与后续路线

**更新时间：2026-08-21**  
**当前阶段：单模态 baseline 已完成，Exp04 三模态 Early Fusion 数据管线已通过 smoke test，下一步进入 5 通道 YOLO11s 模型接入。**

---

# 1. 项目目标

比赛任务为城市场景多模态目标检测，输入包含空间对齐的：

- RGB
- Infrared / IR
- Depth

检测 12 类目标：

| ID | 类别 |
|---:|---|
| 0 | person |
| 1 | boat |
| 2 | animal |
| 3 | seat |
| 4 | sign |
| 5 | bicycle |
| 6 | car |
| 7 | ball |
| 8 | light |
| 9 | garbage_can |
| 10 | uav |
| 11 | tricycle |

核心比赛指标：

- **mAP@50-95**
- 每张图最多 **100 个预测框**

当前实验原则：

1. 先建立 RGB / IR / Depth 三条单模态 baseline。
2. 再做三模态融合。
3. 不使用三个独立检测器输出后简单投票/平均。
4. 三模态融合必须进入同一个端到端模型。
5. 所有实验尽量保持统一的 split、imgsz、模型规模和训练超参数。
6. Exp01 / Exp02 / Exp03 单模态 baseline 已封版，不再反复调参污染对照实验。

---

# 2. 当前工程规范

当前本机项目目录：

```text
/home/xhm/Desktop/aicomp
```

但 Python 脚本统一使用：

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
```

避免在源码内硬编码：

```text
/home/xhm/...
/root/...
```

当前主要目录：

```text
aicomp/
├── datasets/
├── environment/
├── pretrained/
├── runs/
├── scripts/
├── splits/
│   └── aic2026_group_stratified_v2/
├── yolo_views/
│   ├── rgb_v1/
│   ├── ir_v1/
│   └── depth8_v1/
└── ultralytics/
```

训练脚本统一放在：

```text
scripts/train_xxxx.py
```

通过：

```bash
python scripts/train_xxxx.py
```

启动，不使用 `yolo detect train ...` 作为正式训练方式。

---

# 3. 固定数据划分

全数据：

```text
2000 组三模态样本
```

固定划分：

```text
train = 1600
val   = 400
```

RGB、IR、Depth 三个 view 使用完全相同的样本 membership。

当前 view：

```text
yolo_views/rgb_v1/
yolo_views/ir_v1/
yolo_views/depth8_v1/
```

---

# 4. Exp01：RGB-only baseline

模型：

```text
YOLO11s
imgsz = 960
batch = 8
epochs = 200
patience = 50
seed = 2026
```

训练：

```text
runs/exp01_rgb_yolo11s_960/
```

正式验证规格：

```text
imgsz   = 960
conf    = 0.001
NMS IoU = 0.70
max_det = 100
val     = 固定 400 张
```

正式结果：

| 指标 | RGB |
|---|---:|
| Precision | 0.763071 |
| Recall | 0.637680 |
| mAP@50 | 0.642417 |
| mAP@75 | 0.372719 |
| **mAP@50-95** | **0.380843** |

各类别 AP50-95：

| 类别 | RGB |
|---|---:|
| person | 0.401 |
| boat | 0.293 |
| animal | 0.413 |
| seat | 0.721 |
| sign | 0.329 |
| bicycle | 0.336 |
| car | 0.244 |
| ball | 0.323 |
| light | 0.198 |
| garbage_can | 0.361 |
| uav | 0.630 |
| tricycle | 0.323 |

结论：

> RGB 是当前明显最强的单模态，因此后续融合应采用 **RGB 主导、IR/Depth 辅助** 的设计，而不是三路等权。

---

# 5. Exp02：IR-only baseline

数据：

```text
yolo_views/ir_v1/
```

训练：

```text
runs/exp02_ir_yolo11s_960/
```

正式结果：

| 指标 | IR |
|---|---:|
| Precision | 0.587656 |
| Recall | 0.367550 |
| mAP@50 | 0.368124 |
| mAP@75 | 0.205555 |
| **mAP@50-95** | **0.209259** |

各类别 AP50-95：

| 类别 | IR |
|---|---:|
| person | 0.207 |
| boat | 0.199 |
| animal | 0.119 |
| seat | 0.595 |
| sign | 0.216 |
| bicycle | 0.154 |
| car | 0.175 |
| ball | 0.230 |
| light | 0.076 |
| garbage_can | 0.214 |
| uav | 0.071 |
| tricycle | 0.255 |

结论：

- IR 单模态明显弱于 RGB。
- 但 `seat`、`car`、`tricycle` 等仍存在有效补充信息。
- 后续不应该放弃 IR，而应该使用自适应融合控制其贡献。

---

# 6. Depth 数据调查与统一表示

官方 Depth 实际存在两种数据形式。

## 6.1 uint16 metric Depth

数量：

```text
1851
```

典型分辨率：

```text
1080 × 1920
```

统计：

```text
global min/max       = 0 / 19999 mm
valid min/max        = 300 / 19999 mm
invalid(<300) ratio  ≈ 27.76%
valid ratio          ≈ 72.24%
```

有效深度中位数约：

```text
7377 mm
```

## 6.2 uint8 JPG Depth

数量：

```text
149
```

分辨率：

```text
360 × 640 × 3
```

分析确认：

- 三通道几乎完全相同。
- 本质上是灰度 Depth visualization。
- 不是伪彩色深度图。
- JPEG 压缩导致三个通道存在极小差异。
- 视觉方向为 **近亮、远暗**。

## 6.3 depth8_v1 统一方案

对于 uint16 metric Depth：

```text
d < 300 mm     -> 0
d > 20000 mm   -> 0
```

有效区域：

```text
300 mm   -> 255
20000 mm -> 1
```

采用反向线性映射：

```text
gray = 1 + (20000 - d) * 254 / (20000 - 300)
```

因此：

```text
0       = invalid
亮      = near
暗      = far
```

对于官方 uint8 JPG：

```text
BGR -> grayscale
保留原灰度分布
不再次归一化
```

最终统一保存为：

```text
uint8
3-channel PNG
```

用于 Depth-only baseline。

---

# 7. Exp03：Depth-only baseline

数据：

```text
yolo_views/depth8_v1/
```

训练：

```text
runs/exp03_depth_yolo11s_960/
```

正式验证结果：

| 指标 | Depth |
|---|---:|
| Precision | 0.541017 |
| Recall | 0.314882 |
| mAP@50 | 0.305194 |
| mAP@75 | 0.170475 |
| **mAP@50-95** | **0.177121** |

三条 baseline：

| 模态 | mAP@50-95 |
|---|---:|
| **RGB** | **0.380843** |
| IR | 0.209259 |
| Depth | 0.177121 |

各类别：

| 类别 | RGB | IR | Depth |
|---|---:|---:|---:|
| person | 0.401 | 0.207 | 0.207 |
| boat | 0.293 | 0.199 | 0.086 |
| animal | 0.413 | 0.119 | 0.201 |
| seat | 0.721 | 0.595 | 0.663 |
| sign | 0.329 | 0.216 | 0.117 |
| bicycle | 0.336 | 0.154 | 0.118 |
| car | 0.244 | 0.175 | 0.114 |
| ball | 0.323 | 0.230 | 0.104 |
| light | 0.198 | 0.076 | 0.047 |
| garbage_can | 0.361 | 0.214 | 0.157 |
| uav | 0.630 | 0.071 | 0.098 |
| tricycle | 0.323 | 0.255 | 0.213 |

重要观察：

- Depth 总体最弱。
- `seat` 的 Depth AP50-95 达到 **0.663**，非常强。
- `animal` 上 Depth 也明显强于 IR。
- Depth 更像几何辅助信号，而不是独立语义主模态。

因此三模态融合不应简单等权：

```text
RGB = 主干
IR = 辅助
Depth = 几何辅助
```

---

# 8. 三模态数据对齐审计

已完成：

```text
scripts/check_multimodal_alignment.py
```

核心结果：

```text
Aligned samples       = 2000

train:
RGB   = 1600
IR    = 1600
Depth = 1600

val:
RGB   = 400
IR    = 400
Depth = 400
```

空间对齐：

```text
Spatial matches = 2000 / 2000
```

标签：

```text
Label matches = 2000 / 2000
```

train / val：

```text
overlap = 0
```

分辨率组合：

```text
1851 组：
RGB / IR / Depth = 1080 × 1920

149 组：
RGB / IR / Depth = 360 × 640
```

结论：

> RGB、IR、Depth 在样本 ID、split、原始尺寸和检测标签上严格一一对应，可以进行端到端多模态训练。

---

# 9. 标签轻微越界问题

审计发现 3 个训练标签存在轻微归一化越界：

```text
000050:
x = 1.00339

003107:
x = 1.00651

003817:
h = 1.00185
```

进一步检查 Ultralytics cache：

```text
IR train.cache
Depth train.cache
```

确认三个样本全部实际进入 baseline Dataset，且 cache 中仍保留原始数值。

因此决定：

- 不删除这三张图。
- 不修改官方标签。
- 不人为 clip 成 1.0。
- 后续继续交给当前锁定 Ultralytics 的 Instances / augmentation / clipping 流程处理。

这 3 条应视为：

```text
WARNING
```

而不是三模态对齐错误。

---

# 10. Depth 低有效率样本

发现约 16 张 Depth 的有效像素比例低于 5%。

其中部分样本甚至：

```text
valid_ratio = 0
```

这些样本不删除。

原因：

1. 它们属于官方训练数据。
2. 测试集同样可能出现 Depth 失效。
3. 可以作为后续 modality-quality-aware fusion 的天然训练场景。

后面可以定义：

```python
depth_valid_ratio = (depth > 0).float().mean()
```

作为 Depth 质量指标。

---

# 11. 当前 Ultralytics 多通道兼容性调查

已检查当前项目实际锁定源码。

数据调用链：

```text
YOLODataset.__getitem__()
        ↓
get_image_and_label()
        ↓
load_image()
        ↓
transforms
```

确认：

## 原生可以继续使用

```text
LetterBox          PASS
Mosaic             PASS
RandomPerspective  PASS
RandomFlip         PASS
bbox / Instances   PASS
```

当前版本对 multispectral 已有一定原生支持。

## 需要特殊处理

### RandomHSV

原版仅对 3-channel BGR 生效：

```text
5 channel -> 原版 RandomHSV 会直接跳过
```

所以已经实现：

```text
RGBOnlyRandomHSV
```

仅增强：

```text
R,G,B
```

保持：

```text
IR    bit-identical
Depth bit-identical
```

### Format

原版 `Format` 只有在 `C == 3` 时才自动：

```text
BGR -> RGB
```

因此 5-channel 输入不能依赖它做通道转换。

当前方案提前把 RGB：

```text
BGR -> RGB
```

然后再拼：

```text
[R,G,B,I,D]
```

### Albumentations

Exp04 v1 中先禁用通用 Albumentations：

```text
MultimodalAlbumentationsNoOp
```

避免未知 3-channel transform 污染 IR/Depth 或破坏空间对齐。

---

# 12. 当前 Exp04 多模态 Dataset

已经实现：

```text
scripts/multimodal_dataset.py
```

核心表示：

```text
RGB:
BGR -> RGB

IR:
3ch -> grayscale

Depth:
3ch -> grayscale
```

拼接：

```text
[R,G,B,I,D]
```

raw：

```text
H × W × 5
uint8
```

Format 后：

```text
5 × H × W
uint8
```

随后 Trainer 的 preprocess 会再进行：

```text
float
/255
```

几何增强在拼接之后执行，因此：

```text
Mosaic
RandomPerspective
Flip
LetterBox
```

天然同步作用于全部五个通道。

---

# 13. Exp04 Dataset smoke test

已经实现：

```text
scripts/test_exp04_multimodal_dataset.py
```

本次实际运行结果：

```text
STATUS = PASS
```

数据：

```text
train = 1600
val   = 400
```

测试覆盖：

```text
raw samples checked           = 22
val formatted samples checked = 20
train augmented checked       = 20
```

全部通过：

```text
raw source mapping        PASS
5-channel raw             PASS
Format channel order      PASS
val tensor shape          PASS
RGB-only HSV              PASS
IR unchanged by HSV       PASS
Depth unchanged by HSV    PASS
training augmentation     PASS
bbox validation           PASS
collate_fn                PASS
```

Batch：

```text
val:
[4, 5, 960, 960]
torch.uint8

train:
[4, 5, 960, 960]
torch.uint8
```

像素范围：

```text
0 ~ 255
```

训练 transform：

```text
Mosaic
CopyPaste
RandomPerspective
MixUp
CutMix
MultimodalAlbumentationsNoOp
RGBOnlyRandomHSV
RandomFlip
RandomFlip
Format
```

其中当前正式配置：

```text
copy_paste = 0
mixup      = 0
cutmix     = 0
```

所以虽然 transform 对象存在，但对应概率为 0。

## 可视化确认

已生成：

```text
raw_triplet.jpg
val_formatted_triplet.jpg
train_augmented_triplet.jpg
```

本次 `train_augmented_triplet.jpg` 已检查。

Mosaic / augmentation 后：

- RGB
- IR
- Depth

中的场景布局、物体位置以及检测框仍保持空间对应。

因此：

> **Exp04 的三模态 Dataset 数据层已经可以封版。**

---

# 14. 当前阶段结论

到目前为止：

```text
Exp01 RGB baseline                     DONE
Exp02 IR baseline                      DONE
Exp03 Depth baseline                   DONE

Depth 数据格式调查                     DONE
Depth8 统一表示                        DONE
Depth 可视化方向检查                   DONE

三模态 stem 对齐                      DONE
三模态 resolution 对齐                DONE
三模态 label 对齐                     DONE
train/val 一致性                       DONE

Ultralytics multispectral 源码调查     DONE

MultimodalYOLODataset                  DONE
RGB-only HSV                           DONE
5-channel collate                      DONE
Exp04 Dataset smoke test               PASS
```

当前已经从：

```text
数据准备阶段
```

正式进入：

```text
模型融合阶段
```

---

# 15. 下一步：Exp04 5-channel Early Fusion

Exp04 目标非常明确：

> 判断 IR + Depth 加入 RGB 后，是否能够超过 RGB-only 的 **0.380843 mAP@50-95**。

模型输入：

```text
[R,G,B,I,D]
```

Tensor：

```text
[B, 5, H, W]
```

网络：

```text
RGB ─────── R,G,B ─┐
IR  ───────── I ───┼── 5-channel ── YOLO11s ── Detect
Depth ─────── D ───┘
```

---

# 16. 下一步模型改动

YOLO11s 第一层目前：

```text
Conv2d:
in_channels = 3
```

需要改成：

```text
Conv2d:
in_channels = 5
```

预训练权重初始化建议固定为：

```python
new_weight[:, 0:3] = pretrained_rgb_weight
new_weight[:, 3] = 0
new_weight[:, 4] = 0
```

解释：

```text
RGB   -> 完整继承 COCO pretrained
IR    -> zero init
Depth -> zero init
```

这样模型初始化时基本等价于原来的 pretrained RGB YOLO11s：

```text
F_initial ≈ F_RGB
```

不会因为新增两个随机通道在训练第一步就破坏 RGB representation。

IR / Depth 权重虽然从 0 开始，但依然可以正常接收梯度并学习。

---

# 17. 推荐接下来严格按这个顺序执行

## Step 1：实现 5-channel YOLO11s

建议建立：

```text
scripts/multimodal_model.py
```

负责：

1. 创建 5-channel YOLO11s。
2. 加载 `pretrained/yolo11s.pt`。
3. 第一层 `3 -> 5`。
4. RGB 三个通道复制 pretrained 权重。
5. IR / Depth 权重 zero-init。
6. 其余全部可匹配参数继续加载 pretrained。

---

## Step 2：做模型级 smoke test

建立：

```text
scripts/test_exp04_5ch_model.py
```

至少检查：

```text
first conv in_channels = 5             PASS

RGB weight copied exactly              PASS

IR channel weight == 0                 PASS
Depth channel weight == 0              PASS

input:
[B,5,960,960]

forward                                PASS
loss                                   PASS
backward                               PASS

IR/Depth first-conv gradients != 0     PASS
```

最后一项尤其重要：

> 验证 zero-init 的 IR / Depth 权重确实可以在第一次 backward 后得到非零梯度。

---

## Step 3：接入自定义 Trainer

Exp04 不能继续使用默认 `YOLODataset`。

需要让 DetectionTrainer 构造：

```text
MultimodalYOLODataset
```

而不是：

```text
YOLODataset
```

因此建议实现一个：

```text
MultimodalDetectionTrainer
```

主要只改：

```text
build_dataset()
```

以及必要的模型构造位置。

目标是尽量少修改 Ultralytics 源码：

```text
不要直接魔改 ultralytics/
```

而是在：

```text
scripts/
```

下通过 subclass 完成。

---

## Step 4：先跑 Exp04 smoke training

不要第一步直接启动 200 epoch。

建立独立脚本，例如：

```text
scripts/train_exp04_rgbid_early5_smoke_960.py
```

只验证：

```text
Dataset
↓
DataLoader
↓
5ch model
↓
forward
↓
loss
↓
backward
↓
optimizer.step
↓
validation
↓
best.pt
```

完整闭环能跑通。

建议 smoke：

```text
epochs = 2~3
```

它只是工程验证，不是正式实验。

---

## Step 5：正式 Exp04

smoke PASS 后建立：

```text
scripts/train_exp04_rgbid_early5_yolo11s_960.py
```

正式参数尽量继承 Exp01 / 02 / 03：

```text
model      = YOLO11s
imgsz      = 960
batch      = 8
epochs     = 200
patience   = 50
seed       = 2026
mosaic     = 1
close_mosaic = 10
fliplr     = 0.5
max_det(train val) = 300
```

三模态特殊处理只有：

```text
input = 5ch

RGB HSV      = ON
IR HSV       = OFF
Depth HSV    = OFF

generic Albumentations = OFF
```

---

# 18. Exp04 的成功判据

当前必须打败：

```text
RGB baseline = 0.380843
```

例如：

```text
Exp04 = 0.39
```

说明有轻微正增益。

```text
Exp04 = 0.41+
```

说明三模态 early fusion 已经证明存在较明确价值。

如果：

```text
Exp04 <= 0.380843
```

也不意味着多模态无效。

更可能说明：

> 5-channel early fusion 过于粗暴，RGB / IR / Depth 三种统计分布不同，第一层卷积难以同时有效建模。

此时继续进入 Feature-level Fusion，而不是回头删除 IR/Depth。

---

# 19. Exp05：Feature-level Gated Fusion

Exp04 完成后，推荐的真正主模型：

```text
RGB ── RGB Stem ───── F_rgb ─────────────┐
                                         │
IR ─── IR Stem ────── F_ir ── gate_ir ──┤
                                         ├── F_fused
Depth ─ Depth Stem ── F_d ── gate_d ────┘
                                         │
                                      YOLO Neck
                                         │
                                       Detect
```

RGB 始终为主干：

```python
F_fused = F_rgb + alpha_ir * F_ir + alpha_depth * F_depth
```

其中：

```text
alpha_ir    ∈ [0,1]
alpha_depth ∈ [0,1]
```

网络学习不同样本下辅助模态应贡献多少。

---

# 20. Exp06：Depth-quality-aware + Modality Dropout

Depth 本身已经有天然质量指标：

```python
depth_valid_ratio = (depth > 0).float().mean()
```

可以加入 Depth gate：

```text
Depth Feature
      +
Depth Valid Ratio
      ↓
Quality-aware Gate
```

训练时再加入 Modality Dropout：

```text
随机屏蔽 IR
随机屏蔽 Depth
少量情况同时屏蔽 IR + Depth
```

目的：

```text
RGB正常          -> 正常工作
IR失效           -> 不严重退化
Depth失效        -> 不严重退化
RGB质量下降      -> 利用 IR / Depth
```

这是比简单堆 Attention 更符合当前数据特点的方向。

---

# 21. 当前实验路线总览

```text
Exp01
RGB YOLO11s 960
mAP50-95 = 0.380843
        ✓

Exp02
IR YOLO11s 960
mAP50-95 = 0.209259
        ✓

Exp03
Depth YOLO11s 960
mAP50-95 = 0.177121
        ✓

        ↓

Exp04
RGB + IR + Depth
5-channel Early Fusion
        ← 当前阶段

        ↓

Exp05
Separate Stem
+
RGB-dominant Feature-level Gated Fusion

        ↓

Exp06
Exp05
+
Depth Quality-aware Gate
+
Modality Dropout
```

---

# 22. 当前最优先要做的事情

现在**不要继续处理数据，也不要继续分析 Depth**。

数据层已经足够稳定。

下一步唯一主线：

```text
实现 5-channel YOLO11s
        ↓
test_exp04_5ch_model.py
        ↓
Exp04 smoke training
        ↓
Exp04 正式训练
        ↓
official validation
        ↓
与 RGB 0.380843 对比
```

因此下一份应该写的代码是：

```text
scripts/multimodal_model.py
```

以及紧接着：

```text
scripts/test_exp04_5ch_model.py
```

在这两个模型级 smoke test 通过之前，不启动正式 Exp04 200 epoch。

---

# 23. 当前一句话状态

> **三条单模态 baseline 已完成，RGB=0.380843、IR=0.209259、Depth=0.177121；2000 组三模态已确认严格对齐；5-channel `[R,G,B,I,D]` Dataset、同步增强、RGB-only HSV 与 batch collate 均通过 smoke test。当前下一步是把 YOLO11s 第一层从 3 通道扩展为 5 通道并完成模型级 forward/backward smoke test，然后启动 Exp04 Early Fusion。**

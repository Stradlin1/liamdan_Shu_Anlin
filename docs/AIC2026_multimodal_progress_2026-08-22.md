# AIC2026 面向城市场景的视觉多模态目标检测
## 项目记忆与下一阶段路线（2026-08-22）

> 用途：作为后续对话、实验设计和工程恢复时的“记忆性文档”。  
> 当前核心结论：**5 通道 RGB+IR+Depth Early Fusion 已完成从数据层、模型层、正式训练、Official Val、推理 parity 到测试集提交的完整闭环；公开平台当前最好分数为 45.12，高于 RGB 单模态 43.611。**

---

# 1. 当前比赛与工程目标

任务：面向城市场景的视觉多模态目标检测。

输入模态：

- RGB
- Infrared / IR
- Depth

检测类别共 12 类：

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

核心评价指标：

- mAP@50-95
- 每张图片最多 100 个预测框

工程原则：

- 所有正式实验保持固定 train / val split。
- RGB / IR / Depth 必须严格使用同一组三模态样本 membership。
- 不使用多个独立检测器输出后简单投票/平均的“模型集成”路线。
- 多模态必须进入同一个端到端模型。
- 正式训练均使用独立 Python 脚本：
  `scripts/train_xxxx.py`
- 正式脚本统一通过：
  `PROJECT_ROOT = Path(__file__).resolve().parents[1]`
  获取项目根目录。
- 不在 Python 源码中硬编码 `/home/xhm/...` 或 `/root/...`。
- 项目内锁定版本 `ultralytics/`，不随意升级或重新安装其他版本。

---

# 2. 固定数据与目录状态

项目标准目录：

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

固定数据规模：

```text
总样本 = 2000 组三模态
train  = 1600
val    = 400
```

三模态已确认：

- sample stem 对齐
- resolution 对齐
- label 对齐
- train / val membership 对齐

数据层目前视为 **冻结 / 封版**，除非后续发现明确 bug，不再回头反复改数据。

---

# 3. 单模态 Baseline 已完成

## Exp01：RGB-only YOLO11s 960

正式本地验证：

| 指标 | RGB |
|---|---:|
| Precision | 0.763071 |
| Recall | 0.637680 |
| mAP50 | 0.642417 |
| mAP75 | 0.372719 |
| mAP50-95 | **0.380843** |

公开平台提交分数：

```text
RGB 单模态：43.611
```

这条结果是当前公开平台上的单模态参考线。

---

## Exp02：IR-only YOLO11s 960

正式本地验证：

```text
mAP50-95 = 0.209259
```

---

## Exp03：Depth-only YOLO11s 960

Depth 已统一到 `depth8_v1`：

- uint16 metric Depth：
  - `<300 mm` 或 `>20000 mm` -> 0
  - 300 mm -> 255
  - 20000 mm -> 1
  - near = bright
  - far = dark
- 官方 uint8 JPG：
  - BGR -> grayscale
  - 保留官方灰度方向
  - 不重新反转

正式本地验证：

```text
mAP50-95 = 0.177121
```

---

# 4. Exp04：5 通道 Early Fusion 已完成

## 4.1 输入定义

模型输入固定为：

```text
[R, G, B, IR, Depth]
```

Tensor：

```text
[B, 5, H, W]
```

数据类型流程：

```text
raw:
H x W x 5
uint8
0~255

Format / DataLoader:
5 x H x W
uint8

Trainer / inference:
float32 / 255
```

---

## 4.2 Multimodal Dataset

已实现：

```text
scripts/multimodal_dataset.py
```

处理规则：

```text
RGB:
BGR -> RGB

IR:
3ch -> grayscale

Depth:
3ch -> grayscale / depth8 representation

concat:
[R,G,B,I,D]
```

几何增强在 5 通道拼接后统一执行，因此：

- Mosaic
- RandomPerspective
- Flip
- LetterBox

对 RGB / IR / Depth 保持空间同步。

颜色增强：

```text
RGB HSV      = ON
IR HSV       = OFF
Depth HSV    = OFF
```

通用 Albumentations 在 Exp04 中关闭，避免未知 3 通道算子污染辅助模态。

---

## 4.3 Dataset smoke test

已实现：

```text
scripts/test_exp04_multimodal_dataset.py
```

状态：

```text
STATUS = PASS
```

确认过：

- raw source mapping
- 5-channel raw
- Format channel order
- val tensor shape
- RGB-only HSV
- IR unchanged by HSV
- Depth unchanged by HSV
- training augmentation
- bbox validation
- collate_fn

因此 Dataset 层已封版。

---

# 5. Exp04 模型层与 Trainer 已完成

已实现：

```text
scripts/multimodal_model.py
scripts/multimodal_trainer.py
scripts/test_exp04_5ch_model.py
```

第一层：

```text
Conv2d:
3 channels -> 5 channels
```

初始化策略：

```text
RGB    : 精确继承 pretrained YOLO11s 权重
IR     : zero-init
Depth  : zero-init
```

同时验证：

```text
first conv in_channels = 5
RGB pretrained exact copy
IR weight = 0 at initialization
Depth weight = 0 at initialization
forward PASS
loss PASS
backward PASS
IR gradient != 0
Depth gradient != 0
```

说明 IR / Depth 虽从 0 开始，但可以正常学习。

自定义 Trainer 使用：

```text
MultimodalYOLODataset
```

而不是默认 `YOLODataset`。

没有直接修改项目内 `ultralytics/` 源码。

---

# 6. Exp04 正式训练已完成

正式实验：

```text
runs/exp04_rgbid_early5_yolo11s_960/
```

模型：

```text
YOLO11s
input = 5ch
imgsz = 960
batch = 8
epochs = 200
patience = 50
seed = 2026
mosaic = 1
close_mosaic = 10
fliplr = 0.5
```

最终使用：

```text
weights/best.pt
```

而不是：

```text
last.pt
best0.pt
```

重要训练现象：

- 正式训练因 Early Stop 提前结束。
- 因配置为 `close_mosaic=10` 且训练未运行到原计划最后 10 epoch，**这一轮并没有真正经历完整的“最后 10 epoch 关闭 Mosaic”收敛阶段**。
- 这为后续 `no-Mosaic fine-tune` 留下了一个低成本实验机会。

---

# 7. Exp04 Official Val 已完成

正式固定 400 张 val。

协议：

```text
imgsz   = 960
conf    = 0.001
iou     = 0.70
max_det = 100
channels= 5
val     = 400
```

结果：

| 指标 | Exp04 5ch | RGB baseline | 差值 |
|---|---:|---:|---:|
| Precision | 0.712538 | 0.763071 | -0.050533 |
| Recall | 0.603216 | 0.637680 | -0.034464 |
| mAP50 | 0.628168 | 0.642417 | -0.014249 |
| mAP75 | **0.388485** | 0.372719 | **+0.015766** |
| mAP50-95 | **0.379090** | **0.380843** | **-0.001753** |

本地 val 结论：

```text
Exp04 Early Fusion 在 mAP50-95 上与 RGB 基本打平，略低 0.001753。
```

但：

```text
mAP75 明显提高
```

说明三模态对定位质量可能有正作用，而 Precision / Recall / mAP50 略有下降。

---

# 8. 推理 parity 已完成

为了验证测试提交脚本没有出现：

- 通道顺序错误
- LetterBox 错误
- NMS 错误
- 坐标恢复错误
- YOLO TXT 转换错误

已在固定 400 张 val 上执行手工提交推理 parity。

手工 pipeline：

```text
RGB + IR + Depth
      ↓
[R,G,B,IR,Depth]
      ↓
square LetterBox 960x960
      ↓
FP32 / 255
      ↓
best.pt
      ↓
per-image NMS
      ↓
restore original coordinates
      ↓
metric calculation
```

结果：

| 推理方式 | mAP50 | mAP75 | mAP50-95 | vs Official |
|---|---:|---:|---:|---:|
| Official Validator | 0.628168 | 0.388485 | **0.379090** | - |
| multi_label=False | 0.623487 | 0.399749 | **0.373872** | -0.005218 |
| multi_label=True | 0.625146 | 0.400082 | **0.374356** | **-0.004734** |

最终决定：

```text
multi_label = True
```

原因：

- 比 `False` 高约 0.000484 mAP50-95。
- 更接近 Official Validator。
- 400 张 val parity 判定通过。

当前提交脚本的安全策略：

```text
model forward = batch 4
NMS           = per-image
max_time_img  = 10 s
max_det       = 100
conf          = 0.001
iou           = 0.70
multi_label   = True
```

之前出现过：

```text
WARNING: NMS time limit 2.200s exceeded
```

已通过“batch forward + 单图 NMS”解决，避免一个复杂样本导致同 batch 后续图片被跳过。

---

# 9. 公开平台分数现状（2026-08-22）

当前截图中可见分数：

```text
45.120
44.978
44.541
44.541
44.134
42.744
43.611
```

用户已明确这些提交的含义：

## RGB 正式单模态

```text
43.611
```

## Exp04 正式三模态完整训练

```text
45.120
```

这是当前最好成绩。

## 其他三模态提交

```text
42.744
44.134
44.541
44.541
44.978
```

这些不是完整、规范的正式比较实验。

它们来自：

```text
三模态训练尚未结束
+
未先完成正式 val
+
直接拿阶段性 last.pt 推测试集
```

因此这些分数只能作为“训练过程中模型已经有竞争力”的旁证，**不能用于严肃判断某个 checkpoint 或训练策略优劣**。

---

# 10. 当前最重要的比赛结论

这是目前项目最关键的新信息：

```text
Local Official Val:
RGB   = 0.380843
RGBID = 0.379090
=> RGB 略高

Public Test / Leaderboard:
RGB   = 43.611
RGBID = 45.120
=> RGBID 明显更高
```

因此：

> **当前固定 400 张本地 val 与公开测试集存在一定分布/排序差异。**

不能再简单使用：

```text
本地 val 比 RGB 低一点
=> 三模态没有价值
```

来下结论。

公开测试已经证明：

> **三模态 5ch Early Fusion 在真实比赛测试集上存在有效增益。**

这意味着后续路线应继续围绕：

```text
RGB 主导
+
IR / Depth 辅助
```

做更好的融合，而不是退回纯 RGB。

同时：

- 本地 val 仍然用于**控制变量和排查 bug**；
- 平台分数用于**最终真实排序确认**；
- 不应该为了 leaderboard 每个 last.pt 都提交一次。

---

# 11. 当前模型基线重新定义

此前项目 baseline：

```text
Exp01 RGB
local mAP50-95 = 0.380843
leaderboard    = 43.611
```

从现在开始，新的比赛主 baseline 应改为：

```text
Exp04 RGB+IR+Depth 5ch Early Fusion
leaderboard = 45.120
```

后续任何新方法首先需要回答：

```text
能否稳定超过 45.12？
```

而不仅仅是：

```text
能否超过 RGB 43.611？
```

---

# 12. 不再做的事情

当前阶段建议停止：

1. 随机拿训练中途 `last.pt` 直接提交。
2. 没有 local val / parity 就推测试集。
3. 继续修改已经封版的 Depth8 表示。
4. 继续修改固定 train / val split。
5. 在当前 Exp04 上无目标地扫 conf / iou。
6. 为了追分直接堆很多未经控制变量验证的改动。
7. 回头删除 IR / Depth。

尤其是：

```text
44.978 / 44.541 / 44.134 / 42.744
```

这些历史阶段性提交到此为止，不再作为下一步优化主线。

---

# 13. 下一步优先级

后续建议按“成本最低 -> 收益验证最快 -> 结构升级”的顺序推进。

---

## P0：立即封存 Exp04 45.12

先把 45.12 这一套完整实验封存为不可变 baseline。

至少保存：

```text
runs/exp04_rgbid_early5_yolo11s_960/
├── weights/
│   ├── best.pt
│   └── last.pt
├── args.yaml
├── results.csv
├── val_official_960/
├── parity_val400/
└── submission_test_final/
```

同时保存：

```text
scripts/train_exp04_rgbid_early5_yolo11s_960.py
scripts/multimodal_dataset.py
scripts/multimodal_model.py
scripts/multimodal_trainer.py
scripts/val_exp04_rgbid_early5_960.py
scripts/test_exp04_inference_parity_val400.py
scripts/infer_exp04_rgbid_early5_960_submit_final.py
```

建议记录：

```text
Leaderboard score = 45.120
Date              = 2026-08-22
```

如果 Git 仓库已经建立，建议给这个状态打 tag，例如：

```text
exp04-lb45.12
```

以后 Exp05 / Exp06 都不得覆盖这些结果。

---

## P1：先做“模态消融”，不要马上改网络

这是下一步最有价值、成本最低的实验。

使用 **同一个 Exp04 best.pt**，不重新训练，只在固定 400 val 上测试：

```text
A. Full:
   RGB + IR + Depth

B. No IR:
   RGB + 0 + Depth

C. No Depth:
   RGB + IR + 0

D. RGB only inside 5ch model:
   RGB + 0 + 0
```

可选增加：

```text
E. shuffled IR
F. shuffled Depth
```

目标：

回答三个问题：

1. 当前 trained Exp04 到底有没有实际使用 IR？
2. 当前 trained Exp04 到底有没有实际使用 Depth？
3. 哪个辅助模态贡献更大，哪个可能在拖后腿？

输出至少比较：

```text
P
R
mAP50
mAP75
mAP50-95
per-class AP
```

这是进入 Exp05 gated fusion 前必须知道的信息。

### 决策规则

如果：

```text
Full > No IR
```

说明 IR 有正贡献。

如果：

```text
Full > No Depth
```

说明 Depth 有正贡献。

如果某个：

```text
No modality > Full
```

说明当前 Early Fusion 中该模态可能存在负迁移，需要 gate / quality-aware 机制。

---

## P2：做一次 Rect Inference A/B

当前 Official Val：

```text
rect=True
mAP50-95 = 0.379090
```

当前手工 square 960 parity：

```text
multi_label=True
mAP50-95 = 0.374356
```

差：

```text
0.004734
```

因此存在一个**不需要训练**的潜在收益点：

```text
把测试推理从 fixed square 960
改成更贴近 Official Validator 的 rect preprocessing。
```

建议：

1. 先在固定 400 val 上实现 `rect inference`。
2. 必须先达到或逼近 Official `0.379090`。
3. 确认坐标恢复、NMS、max_det=100 都一致。
4. 如果显著优于 square `0.374356`，并且比赛提交额度允许，再做一次公开测试提交。

这可能是当前成本最低的 leaderboard 提升机会之一。

---

## P3：Exp04b — No-Mosaic Fine-tune

当前 Exp04 正式训练提前 Early Stop。

原配置：

```text
epochs = 200
close_mosaic = 10
```

由于训练没有真正运行到计划末尾，因此最后的 no-Mosaic 收敛阶段没有完整发生。

建议建立独立训练实验：

```text
Exp04b:
start = Exp04 best.pt
mosaic = 0
保持同一 train/val split
保持 5ch model
低学习率
短周期 fine-tune
```

建议先跑：

```text
20~30 epochs
```

学习率原则：

```text
显著低于原正式训练的有效初始学习率
```

不要一开始大范围 sweep。

每个 checkpoint 必须：

```text
train
↓
Official Val
↓
parity
↓
只有明显优于当前 baseline 才提交
```

目标不是重训整个 Exp04，而是补一次干净的 box refinement / no-Mosaic 收敛。

---

# 14. 下一阶段主架构：Exp05

原项目路线仍然成立，但现在有了更强的理由继续做。

## Exp05：RGB-dominant Feature-level Gated Fusion

当前 5ch Early Fusion：

```text
[R,G,B,I,D]
      ↓
同一个第一层 Conv
```

缺点：

- RGB、IR、Depth 统计分布差异很大。
- 所有模态过早混合。
- 模型没有显式能力判断辅助模态质量。
- Depth 失效区域可能污染表示。

下一步应改为：

```text
RGB ── RGB Stem ───────── F_rgb ──────────────┐
                                              │
IR ─── IR Stem ────────── F_ir ── gate_ir ───┤
                                              ├── F_fused
Depth ─ Depth Stem ────── F_d ── gate_d ─────┘
                                              │
                                           YOLO Neck
                                              │
                                            Detect
```

核心：

```python
F_fused = (
    F_rgb
    + alpha_ir * F_ir
    + alpha_depth * F_depth
)
```

其中：

```text
alpha_ir    = learnable / sample-adaptive gate
alpha_depth = learnable / sample-adaptive gate
```

原则：

> RGB 是稳定主干，IR / Depth 只能“帮助”，不能无条件覆盖 RGB。

---

# 15. Exp06：Quality-aware Fusion + Modality Dropout

如果 Exp05 证明 gated fusion 有价值，再增加：

```text
Depth quality-aware gate
+
Modality Dropout
```

Depth 质量指标可以直接使用：

```python
depth_valid_ratio = (depth > 0).float().mean()
```

训练中随机：

```text
drop IR
drop Depth
少量 drop IR + Depth
```

目标：

```text
RGB 正常         -> 稳定检测
IR 无效          -> 模型自动弱化 IR
Depth 无效       -> 模型自动弱化 Depth
RGB 困难         -> IR / Depth 提供补充
```

---

# 16. 推荐接下来严格执行的顺序

```text
当前最好：
Exp04 5ch Early Fusion
Leaderboard = 45.12
        │
        ▼
[Step 1]
冻结 / 归档 Exp04
        │
        ▼
[Step 2]
Val400 模态消融
Full / No IR / No Depth / RGB-only-in-5ch
        │
        ▼
[Step 3]
Rect inference parity
与 square 960 比较
        │
        ├── 若 rect 明显更好
        │       ↓
        │   生成 rect test submission
        │
        ▼
[Step 4]
Exp04b
best.pt + no-Mosaic low-LR fine-tune
        │
        ▼
Official Val
        │
        ▼
只有确认提升才提交
        │
        ▼
[Step 5]
Exp05
Separate Stem
+ RGB-dominant Gated Fusion
        │
        ▼
[Step 6]
Exp06
Quality-aware Gate
+ Modality Dropout
```

---

# 17. 接下来最应该先写的脚本

按优先级：

```text
1. scripts/test_exp04_modality_ablation_val400.py
```

作用：

```text
Full
No IR
No Depth
RGB only
```

在同一个 `best.pt` 上做 val400 消融。

之后：

```text
2. scripts/test_exp04_rect_inference_parity_val400.py
```

作用：

```text
square inference
vs
rect inference
vs
Official Validator
```

再之后：

```text
3. scripts/train_exp04b_rgbid_early5_nomosaic_ft_960.py
```

最后才进入：

```text
4. Exp05 gated feature fusion
```

---

# 18. 当前判断

当前项目已经不处在“多模态能不能跑通”的阶段。

已经完成：

```text
数据准备
↓
单模态 baseline
↓
5ch Dataset
↓
5ch Model
↓
5ch Trainer
↓
smoke training
↓
正式训练
↓
Official Val
↓
manual inference parity
↓
正式测试集提交
↓
Leaderboard 验证
```

而且公开测试：

```text
RGB   = 43.611
RGBID = 45.120
```

已经给出了一个很重要的现实结论：

> **三模态方向值得继续。**

现在的目标应从：

```text
“证明多模态有用”
```

切换为：

```text
“搞清楚 IR / Depth 各自贡献什么，并把 45.12 提升成新的更强单模型。” 
```

---

# 19. 当前一句话记忆

> **截至 2026-08-22，AIC2026 项目已完成 RGB/IR/Depth 三条单模态 baseline、三模态 5ch Dataset/Model/Trainer、Exp04 YOLO11s 960 正式训练、Official Val、400 张 manual parity 和 1000 张测试集提交；RGB 平台分数 43.611，完整训练后的 Exp04 RGB+IR+Depth Early Fusion 达到 45.120，为当前最好成绩。其他 42.744~44.978 的三模态成绩来自未完成训练时直接使用阶段性 last.pt 的非正式提交，不作为后续比较基线。下一步先冻结 45.12 baseline，做 val400 模态消融与 rect inference A/B，再做 Exp04b no-Mosaic 低学习率微调，之后进入 RGB-dominant feature-level gated fusion（Exp05）。**

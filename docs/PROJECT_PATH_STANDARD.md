# AIC2026 工程路径配置规范

## 1. 目的

本规范用于统一 `aicomp` 项目中 Python 脚本、数据集配置、预训练权重、实验结果等路径的写法。

核心目标：

- 项目可以整体迁移到本地、AutoDL、学校服务器或其他 Linux 主机；
- 更换项目存放位置后，不需要修改 Python 源码中的路径；
- 禁止新脚本依赖某台机器特有的绝对路径；
- 保证训练、验证、推理和数据处理脚本具有可复现性。

---

## 2. 项目标准目录

项目根目录统一记为：

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
├── ultralytics/
└── PROJECT_PATH_STANDARD.md
```

各目录用途：

| 目录 | 用途 |
|---|---|
| `datasets/` | 官方数据集及必要的数据处理结果 |
| `environment/` | Conda、pip、版本、Git commit 等环境记录 |
| `pretrained/` | YOLO 等公开预训练权重 |
| `runs/` | 所有训练、验证、推理实验结果 |
| `scripts/` | 所有训练、数据处理、验证、推理 Python 脚本 |
| `splits/` | 固定的 train/val 数据划分 |
| `yolo_views/` | 为 Ultralytics 构造的 YOLO 数据视图 |
| `ultralytics/` | 本项目锁定版本的 Ultralytics 源码 |

---

## 3. 核心原则

### 3.1 禁止硬编码本机绝对路径

新脚本中禁止出现：

```python
"/home/xhm/Desktop/aicomp/..."
```

也禁止服务器专用路径：

```python
"/root/aicomp/..."
"/root/autodl-tmp/aicomp/..."
```

禁止通过修改源码适配不同服务器。

项目迁移后，脚本本身不应发生任何路径修改。

---

## 4. Python 脚本定位项目根目录

目前所有正式 Python 脚本统一放在：

```text
aicomp/scripts/
```

因此脚本必须使用：

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
```

例如脚本：

```text
aicomp/scripts/train_exp01_rgb_yolo11s_960.py
```

则：

```python
Path(__file__).resolve()
```

表示：

```text
<当前项目位置>/aicomp/scripts/train_exp01_rgb_yolo11s_960.py
```

而：

```python
Path(__file__).resolve().parents[1]
```

自动得到：

```text
<当前项目位置>/aicomp
```

因此无论项目位于：

```text
/home/xhm/Desktop/aicomp
```

还是：

```text
/root/aicomp
```

或：

```text
/root/autodl-tmp/aicomp
```

代码均无需修改。

---

## 5. 标准路径写法

### 5.1 数据集

正确：

```python
DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "AIC2026_Train_2000"
)
```

禁止：

```python
DATASET_ROOT = Path(
    "/home/xhm/Desktop/aicomp/datasets/AIC2026_Train_2000"
)
```

### 5.2 预训练模型

所有公开预训练模型统一放入：

```text
aicomp/pretrained/
```

例如：

```text
pretrained/yolo11n.pt
pretrained/yolo11s.pt
```

Python 中使用：

```python
MODEL_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "yolo11s.pt"
)
```

加载模型时：

```python
model = YOLO(
    str(MODEL_PATH)
)
```

不要使用：

```python
MODEL = "yolo11s.pt"
```

原因：

仅写文件名会依赖当前工作目录，并且文件不存在时可能触发额外的下载行为。

比赛工程优先使用项目中已经保存的固定权重。

---

## 6. 训练结果目录

所有实验统一保存至：

```text
aicomp/runs/
```

Python 写法：

```python
PROJECT_DIR = (
    PROJECT_ROOT
    / "runs"
)
```

实验名称独立定义：

```python
EXPERIMENT_NAME = (
    "exp01_rgb_yolo11s_960"
)
```

Ultralytics：

```python
results = model.train(
    project=str(PROJECT_DIR),
    name=EXPERIMENT_NAME,
)
```

最终结果目录：

```text
runs/exp01_rgb_yolo11s_960/
```

禁止在训练脚本中写：

```python
project="/home/xhm/Desktop/aicomp/runs"
```

---

## 7. YOLO 数据视图

YOLO 格式数据统一放在：

```text
aicomp/yolo_views/
```

例如：

```text
yolo_views/
└── rgb_v1/
    ├── data.yaml
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

Python 中：

```python
RGB_YOLO_VIEW = (
    PROJECT_ROOT
    / "yolo_views"
    / "rgb_v1"
)
```

训练配置：

```python
DATA_YAML = (
    PROJECT_ROOT
    / "yolo_views"
    / "rgb_v1"
    / "data.yaml"
)
```

---

## 8. `data.yaml` 路径规范

YOLO view 的 `data.yaml` 不写机器绝对路径。

推荐：

```yaml
train: images/train
val: images/val

names:
  0: person
  1: boat
  2: animal
  3: seat
  4: sign
  5: bicycle
  6: car
  7: ball
  8: light
  9: garbage_can
  10: uav
  11: tricycle
```

禁止：

```yaml
path: /home/xhm/Desktop/aicomp/yolo_views/rgb_v1
```

在本项目锁定的 Ultralytics 版本中，当 `path` 不指定时，数据集根目录可由 YAML 文件位置确定。

因此：

```yaml
train: images/train
val: images/val
```

能够随整个项目一起迁移。

---

## 9. 数据软链接规范

为了避免复制大量官方数据，`yolo_views` 可以使用软链接。

但必须创建：

```text
相对软链接
```

禁止创建：

```text
绝对软链接
```

正确示例：

```python
import os

relative_target = os.path.relpath(
    src,
    start=dst.parent
)

os.symlink(
    relative_target,
    dst
)
```

如果项目整体从：

```text
/home/xhm/Desktop/aicomp
```

移动到：

```text
/root/aicomp
```

相对软链接仍然有效。

如果使用绝对软链接，则迁移后会全部失效。

---

## 10. Train / Val 划分路径

固定比赛划分统一存放：

```text
aicomp/splits/aic2026_group_stratified_v2/
```

Python：

```python
SPLIT_ROOT = (
    PROJECT_ROOT
    / "splits"
    / "aic2026_group_stratified_v2"
)

TRAIN_SPLIT = (
    SPLIT_ROOT
    / "train.txt"
)

VAL_SPLIT = (
    SPLIT_ROOT
    / "val.txt"
)
```

正式实验禁止临时重新随机划分数据集。

所有可比较实验必须使用同一版本 split。

---

## 11. 文件存在性检查

正式脚本启动前应检查关键输入文件。

例如：

```python
def check_files():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"预训练权重不存在: {MODEL_PATH}"
        )

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML 不存在: {DATA_YAML}"
        )

    PROJECT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )
```

这样服务器缺失文件时能够立即停止，而不是训练开始后才出现难以定位的问题。

---

## 12. 不允许依赖当前终端目录

禁止使用：

```python
Path.cwd()
```

作为项目根目录。

禁止：

```python
DATA = Path("./datasets/...")
```

作为正式工程路径。

因为下面两个命令的工作目录不同：

```bash
cd aicomp
python scripts/train_xxx.py
```

和：

```bash
cd aicomp/scripts
python train_xxx.py
```

如果使用 `cwd`，可能得到不同路径。

应始终使用：

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
```

项目路径只取决于脚本自身的位置，不取决于用户从哪个目录启动。

---

## 13. 推荐的新训练脚本模板

以后新增：

```text
scripts/train_xxxx.py
```

建议从以下模板开始：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "yolo11s.pt"
)

DATA_YAML = (
    PROJECT_ROOT
    / "yolo_views"
    / "rgb_v1"
    / "data.yaml"
)

RUNS_DIR = (
    PROJECT_ROOT
    / "runs"
)

EXPERIMENT_NAME = "exp_xxxx"


def check_files():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATA_YAML}"
        )

    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def main():

    check_files()

    print("Project root :", PROJECT_ROOT)
    print("Model        :", MODEL_PATH)
    print("Dataset      :", DATA_YAML)
    print("Output       :", RUNS_DIR / EXPERIMENT_NAME)

    model = YOLO(
        str(MODEL_PATH)
    )

    results = model.train(
        data=str(DATA_YAML),
        project=str(RUNS_DIR),
        name=EXPERIMENT_NAME,
    )

    return results


if __name__ == "__main__":
    main()
```

所有训练仍统一使用：

```bash
python scripts/train_xxxx.py
```

不得将正式实验改成直接执行：

```bash
yolo detect train ...
```

以保证训练参数、路径和实验配置都有独立 Python 文件记录。

---

## 14. `project_paths.py` 使用规范

多个数据处理脚本需要共享同一组路径时，可以统一定义在：

```text
scripts/project_paths.py
```

例如：

```python
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "AIC2026_Train_2000"
)

VISIBLE_DIR = (
    DATASET_ROOT
    / "visible"
)

LABEL_DIR = (
    DATASET_ROOT
    / "labels"
)

SPLIT_ROOT = (
    PROJECT_ROOT
    / "splits"
    / "aic2026_group_stratified_v2"
)

TRAIN_SPLIT = (
    SPLIT_ROOT
    / "train.txt"
)

VAL_SPLIT = (
    SPLIT_ROOT
    / "val.txt"
)

RGB_YOLO_VIEW = (
    PROJECT_ROOT
    / "yolo_views"
    / "rgb_v1"
)
```

同目录脚本可以：

```python
from project_paths import (
    DATASET_ROOT,
    VISIBLE_DIR,
    LABEL_DIR,
    TRAIN_SPLIT,
    VAL_SPLIT,
    RGB_YOLO_VIEW,
)
```

原则：

- 通用路径集中管理；
- 实验特有参数留在对应实验脚本中；
- 不要在多个脚本中重复硬编码同一个绝对路径。

---

## 15. 服务器迁移原则

项目上传服务器后，允许改变的只有项目根目录位置。

例如本地：

```text
/home/xhm/Desktop/aicomp
```

服务器：

```text
/root/autodl-tmp/aicomp
```

迁移完成后：

```bash
cd /root/autodl-tmp/aicomp
```

安装项目内 Ultralytics：

```bash
cd ultralytics
python -m pip install -e .
```

然后回到项目根目录：

```bash
cd ..
```

训练：

```bash
python scripts/train_xxxx.py
```

正常情况下：

```text
不修改任何 Python 路径
不修改 data.yaml
不重新创建绝对软链接
```

---

## 16. Ultralytics 源码规范

本项目使用项目内：

```text
aicomp/ultralytics/
```

源码。

版本及 Git commit 应记录在：

```text
environment/version_info.txt
environment/ultralytics_commit.txt
```

服务器恢复环境时使用：

```bash
cd ultralytics
python -m pip install -e .
```

不要同时从 PyPI 再安装另一份不同版本 Ultralytics。

---

## 17. 路径审计

新增或者修改正式脚本后，应执行路径检查：

```bash
grep -RIn \
  --exclude-dir=.git \
  --exclude-dir=datasets \
  --exclude-dir=runs \
  --exclude-dir=ultralytics \
  '/home/xhm/Desktop/aicomp' \
  . 2>/dev/null
```

正式使用的代码和配置中原则上不应出现：

```text
/home/xhm/Desktop/aicomp
```

如果只出现在旧实验记录、历史日志或说明文档中，不影响运行，可以保留。

重点检查：

```text
scripts/
yolo_views/*/data.yaml
environment/
```

---

## 18. 历史脚本处理原则

已经完成、后续不会再次运行的旧实验脚本：

```text
允许保持原样
```

不要求为了迁移专门重构。

但：

```text
禁止复制旧脚本中的绝对路径写法到新实验。
```

如果旧实验需要重新运行，则在重新运行之前按照本规范完成路径重构。

---

## 19. 新脚本提交前检查清单

每个新的正式脚本至少确认：

- [ ] 脚本位于 `scripts/`
- [ ] 使用 `Path(__file__).resolve().parents[1]` 定位项目根目录
- [ ] 没有 `/home/xhm/...` 硬编码
- [ ] 没有 `/root/...` 服务器硬编码
- [ ] 预训练权重从 `pretrained/` 加载
- [ ] 数据集从 `datasets/` 或 `yolo_views/` 加载
- [ ] 数据划分从 `splits/` 加载
- [ ] 实验输出统一进入 `runs/`
- [ ] `data.yaml` 不包含机器绝对路径
- [ ] 软链接使用相对软链接
- [ ] 不依赖 `Path.cwd()` 判断项目位置
- [ ] 正式训练由独立 `train_xxxx.py` 启动
- [ ] 运行前检查关键文件是否存在
- [ ] 更换服务器路径后无需修改源码

---

## 20. 最终原则

本项目路径设计遵守一句话：

> **所有路径都相对于项目自身确定，而不是相对于某一台电脑确定。**

机器可以变化：

```text
/home/xhm/Desktop/aicomp
/root/aicomp
/root/autodl-tmp/aicomp
/data/aicomp
```

但工程内部结构保持不变：

```text
datasets/
pretrained/
runs/
scripts/
splits/
yolo_views/
ultralytics/
```

因此代码无需修改即可迁移。

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from collections import Counter, defaultdict
import csv
import math
import re
import sys

import numpy as np


# ============================================================
# 配置
# ============================================================

DATASET_ROOT = Path(
    "/home/xhm/Desktop/aicomp/datasets/AIC2026_Train_2000"
)

VISIBLE_DIR = DATASET_ROOT / "visible"
INFRARED_DIR = DATASET_ROOT / "infrared"
DEPTH_DIR = DATASET_ROOT / "depth"
LABEL_DIR = DATASET_ROOT / "labels"

# 上一步数据体检生成的报告
HEALTH_REPORT_DIR = Path(
    "/home/xhm/Desktop/aicomp/dataset_health_report"
)

ADJACENT_CSV = (
    HEALTH_REPORT_DIR / "adjacent_similarity.csv"
)

# 所有 split 文件都放在数据集外部
OUTPUT_DIR = Path(
    "/home/xhm/Desktop/aicomp/splits/aic2026_group_stratified_v1"
)

VAL_RATIO = 0.20
RANDOM_SEED = 2026

NUM_CLASSES = 12

CLASS_NAMES = {
    0: "person",
    1: "boat",
    2: "animal",
    3: "seat",
    4: "sign",
    5: "bicycle",
    6: "car",
    7: "ball",
    8: "light",
    9: "garbage_can",
    10: "uav",
    11: "tricycle",
}

IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
}

# 对类似 000001 / 000002 / ...
# 这种无法从文件名恢复真实 sequence 的数据，
# 连续每 25 张作为一个原子块。
#
# 不是移动数据，只是防止相邻图片被随机拆开。
NUMERIC_BLOCK_SIZE = 25

# 之前体检脚本的 dHash：
# <=5 认为极度相似，强制锁进同一 split。
SIMILARITY_LOCK_THRESHOLD = 5

# 6~10 不强制合并，但最终统计有多少跨 split。
SIMILARITY_WARNING_THRESHOLD = 10

# 搜索次数。2000 样本规模跑起来很快。
NUM_RESTARTS = 60
LOCAL_SEARCH_ITERS = 2500


# ============================================================
# 基础函数
# ============================================================

def natural_key(text):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", str(text))
    ]


def scan_images(folder):
    result = defaultdict(list)

    for p in folder.iterdir():
        if (
            p.is_file()
            and p.suffix.lower() in IMAGE_EXTS
        ):
            result[p.stem].append(p)

    return result


def scan_labels(folder):
    result = defaultdict(list)

    for p in folder.iterdir():
        if (
            p.is_file()
            and p.suffix.lower() == ".txt"
        ):
            result[p.stem].append(p)

    return result


def require_unique(mapping, stem, name):
    paths = mapping.get(stem, [])

    if len(paths) != 1:
        raise RuntimeError(
            f"{stem}: {name} 文件数量为 {len(paths)}，"
            f"要求必须恰好为 1"
        )

    return paths[0]


def read_classes(label_path):
    """
    这里只读取 class_id 用于 split 分层。
    不修改、过滤或重新写官方标签。
    """
    counts = Counter()

    text = label_path.read_text(
        encoding="utf-8",
        errors="replace"
    )

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if not parts:
            continue

        try:
            cls_raw = float(parts[0])
            cls_id = int(cls_raw)
        except Exception:
            continue

        if (
            abs(cls_raw - cls_id) < 1e-9
            and 0 <= cls_id < NUM_CLASSES
        ):
            counts[cls_id] += 1

    return counts


def source_family(stem, visible_path):
    """
    数据源类别仅用于让 train/val 尽量保持来源比例。
    """
    ext = visible_path.suffix.lower().lstrip(".")

    if stem.isdigit():
        return f"numeric_{len(stem)}digit_{ext}"

    s = stem.lower()

    if "suppl" in s:
        return f"suppl_{ext}"

    if s.startswith("shuming"):
        return f"shuming_{ext}"

    if s.startswith("hehe"):
        return f"hehe_{ext}"

    return f"structured_other_{ext}"


# ============================================================
# Union-Find
# ============================================================

class DSU:
    def __init__(self, items):
        self.parent = {
            x: x for x in items
        }

        self.rank = {
            x: 0 for x in items
        }

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = (
                self.parent[
                    self.parent[x]
                ]
            )
            x = self.parent[x]

        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra

        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# ============================================================
# 检查目录
# ============================================================

for folder in [
    VISIBLE_DIR,
    INFRARED_DIR,
    DEPTH_DIR,
    LABEL_DIR,
]:
    if not folder.exists():
        print(
            f"ERROR: 目录不存在：{folder}"
        )
        sys.exit(1)


# ============================================================
# 扫描四模态
# ============================================================

print("=" * 88)
print("AIC2026 Group-aware + Class-stratified Train/Val Split")
print("=" * 88)
print()
print("注意：本脚本不会修改、移动、复制或删除任何官方数据。")
print("只会在数据集外生成 split 索引。")
print()

visible_map = scan_images(VISIBLE_DIR)
infrared_map = scan_images(INFRARED_DIR)
depth_map = scan_images(DEPTH_DIR)
label_map = scan_labels(LABEL_DIR)

sets = [
    set(visible_map.keys()),
    set(infrared_map.keys()),
    set(depth_map.keys()),
    set(label_map.keys()),
]

if not (
    sets[0] == sets[1]
    == sets[2] == sets[3]
):
    print("ERROR: 四个子目录 basename 不完全一致。")
    print("请先查看之前 dataset_health_report。")
    sys.exit(1)

stems = sorted(
    sets[0],
    key=natural_key
)

N = len(stems)

print(f"样本总数：{N}")

if N != 2000:
    print(
        f"WARNING: 当前不是 2000 组，而是 {N} 组。"
    )

target_val_n = round(
    N * VAL_RATIO
)

print(
    f"目标划分：Train ≈ {N - target_val_n}, "
    f"Val ≈ {target_val_n}"
)
print()


# ============================================================
# 读取每个样本
# ============================================================

sample_info = {}

for stem in stems:

    vp = require_unique(
        visible_map,
        stem,
        "visible"
    )

    ip = require_unique(
        infrared_map,
        stem,
        "infrared"
    )

    dp = require_unique(
        depth_map,
        stem,
        "depth"
    )

    lp = require_unique(
        label_map,
        stem,
        "label"
    )

    cls_counts = read_classes(lp)

    sample_info[stem] = {
        "visible": vp,
        "infrared": ip,
        "depth": dp,
        "label": lp,
        "class_counts": cls_counts,
        "source": source_family(
            stem,
            vp
        ),
    }


# ============================================================
# 第一层 grouping
#
# A. 带结构化后缀：
#    000003_080_00000307
#    000003_080_00000409
#    -> 000003_080
#
# B. 纯数字：
#    连续每 NUMERIC_BLOCK_SIZE 张形成一个 block
# ============================================================

base_groups = defaultdict(list)

numeric_buckets = defaultdict(list)

for stem in stems:

    vp = sample_info[stem][
        "visible"
    ]

    if stem.isdigit():

        key = (
            len(stem),
            vp.suffix.lower(),
        )

        numeric_buckets[key].append(
            stem
        )

    else:

        # 去掉最后一个 "_数字帧号"
        match = re.match(
            r"^(.*)_([0-9]+)$",
            stem
        )

        if match:
            prefix = match.group(1)
        else:
            # 文件名不符合规律时，
            # 只能将自身作为单独组
            prefix = stem

        group_id = (
            f"sequence::{prefix}"
        )

        base_groups[
            group_id
        ].append(stem)


# ------------------------------------------------------------
# 纯数字文件按连续排序成 block
# ------------------------------------------------------------

for key, bucket in numeric_buckets.items():

    bucket = sorted(
        bucket,
        key=lambda x: int(x)
    )

    digits_len, ext = key

    for index, stem in enumerate(
        bucket
    ):

        block_id = (
            index // NUMERIC_BLOCK_SIZE
        )

        group_id = (
            f"numeric::{digits_len}"
            f"::{ext}"
            f"::block_{block_id:04d}"
        )

        base_groups[
            group_id
        ].append(stem)


# ============================================================
# DSU：先将 base group 内部锁定
# ============================================================

dsu = DSU(stems)

for members in base_groups.values():

    if len(members) <= 1:
        continue

    anchor = members[0]

    for stem in members[1:]:
        dsu.union(
            anchor,
            stem
        )


# ============================================================
# 使用上一次体检得到的相邻图 dHash
#
# <=5 强制同组
# ============================================================

similarity_rows = []

locked_pairs = 0

if ADJACENT_CSV.exists():

    print(
        f"读取相邻帧体检：{ADJACENT_CSV}"
    )

    with ADJACENT_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            try:
                a = row["stem_a"]
                b = row["stem_b"]
                dist = int(
                    row["dhash_distance"]
                )
            except Exception:
                continue

            similarity_rows.append(
                (a, b, dist)
            )

            if (
                a in sample_info
                and b in sample_info
                and dist
                <= SIMILARITY_LOCK_THRESHOLD
            ):
                dsu.union(a, b)
                locked_pairs += 1

    print(
        f"dHash <= {SIMILARITY_LOCK_THRESHOLD} "
        f"锁定相邻对：{locked_pairs}"
    )

else:

    print(
        "WARNING: 未找到 adjacent_similarity.csv"
    )

    print(
        "仍可划分，但不会加入视觉极相似帧约束。"
    )

print()


# ============================================================
# 得到最终 atomic groups
# ============================================================

groups_dict = defaultdict(list)

for stem in stems:
    groups_dict[
        dsu.find(stem)
    ].append(stem)

groups_members = list(
    groups_dict.values()
)

groups_members.sort(
    key=lambda x: natural_key(
        min(x, key=natural_key)
    )
)

group_sizes = [
    len(x)
    for x in groups_members
]

print("最终原子 Group")
print("-" * 88)
print(
    f"Group 数量：{len(groups_members)}"
)
print(
    f"最小 Group：{min(group_sizes)}"
)
print(
    f"最大 Group：{max(group_sizes)}"
)
print(
    f"平均 Group：{np.mean(group_sizes):.2f}"
)
print()


# ============================================================
# source 列表
# ============================================================

sources = sorted({
    sample_info[s]["source"]
    for s in stems
})

source_to_idx = {
    name: i
    for i, name in enumerate(sources)
}

print("识别出的数据来源：")

for src in sources:
    count = sum(
        1
        for s in stems
        if (
            sample_info[s]["source"]
            == src
        )
    )

    print(
        f"  {src:30s} {count:4d}"
    )

print()


# ============================================================
# 每个 group 编码成 vector
#
# [
#   sample_n,
#   class_instance[12],
#   class_image[12],
#   source_count[S]
# ]
# ============================================================

OFFSET_N = 0

OFFSET_INST = 1

OFFSET_IMG = (
    OFFSET_INST
    + NUM_CLASSES
)

OFFSET_SOURCE = (
    OFFSET_IMG
    + NUM_CLASSES
)

VECTOR_DIM = (
    OFFSET_SOURCE
    + len(sources)
)

G = np.zeros(
    (
        len(groups_members),
        VECTOR_DIM,
    ),
    dtype=np.float64,
)

for gi, members in enumerate(
    groups_members
):

    G[gi, OFFSET_N] = len(
        members
    )

    for stem in members:

        info = sample_info[stem]

        counts = info[
            "class_counts"
        ]

        for cls_id, count in (
            counts.items()
        ):

            G[
                gi,
                OFFSET_INST + cls_id
            ] += count

            G[
                gi,
                OFFSET_IMG + cls_id
            ] += 1

        src_idx = source_to_idx[
            info["source"]
        ]

        G[
            gi,
            OFFSET_SOURCE + src_idx
        ] += 1


TOTAL = np.sum(
    G,
    axis=0
)

TARGET = (
    TOTAL * VAL_RATIO
)

TARGET[
    OFFSET_N
] = target_val_n


# ============================================================
# Objective
# ============================================================

total_inst = TOTAL[
    OFFSET_INST:
    OFFSET_INST + NUM_CLASSES
]

total_img = TOTAL[
    OFFSET_IMG:
    OFFSET_IMG + NUM_CLASSES
]

total_source = TOTAL[
    OFFSET_SOURCE:
]


def objective(v):

    val_n = v[
        OFFSET_N
    ]

    inst = v[
        OFFSET_INST:
        OFFSET_INST + NUM_CLASSES
    ]

    img = v[
        OFFSET_IMG:
        OFFSET_IMG + NUM_CLASSES
    ]

    src = v[
        OFFSET_SOURCE:
    ]

    target_inst = (
        total_inst * VAL_RATIO
    )

    target_img = (
        total_img * VAL_RATIO
    )

    target_src = (
        total_source * VAL_RATIO
    )

    # --------------------------------------------------------
    # 1. 样本数
    # --------------------------------------------------------

    size_err = (
        (
            val_n - target_val_n
        )
        / max(
            target_val_n,
            1
        )
    ) ** 2

    # 样本数需要比较强的约束
    score = 50.0 * size_err

    # --------------------------------------------------------
    # 2. 12 类实例数
    #
    # 每个类别等价计算相对误差，
    # 不允许 person 把 tricycle 淹没。
    # --------------------------------------------------------

    inst_err = (
        (
            inst - target_inst
        )
        /
        (
            target_inst + 1.0
        )
    ) ** 2

    score += (
        5.0
        * float(
            np.mean(inst_err)
        )
    )

    # --------------------------------------------------------
    # 3. 各类别出现图片数
    #
    # 这个比单纯实例数更重要：
    # 希望 rare class 在 val 中分布到多张图。
    # --------------------------------------------------------

    img_err = (
        (
            img - target_img
        )
        /
        (
            target_img + 1.0
        )
    ) ** 2

    score += (
        8.0
        * float(
            np.mean(img_err)
        )
    )

    # --------------------------------------------------------
    # 4. 来源平衡
    # --------------------------------------------------------

    if len(src) > 0:

        source_err = (
            (
                src - target_src
            )
            /
            (
                target_src + 1.0
            )
        ) ** 2

        score += (
            3.0
            * float(
                np.mean(source_err)
            )
        )

    # --------------------------------------------------------
    # 5. 禁止有类别完全不出现在 val
    # --------------------------------------------------------

    for cls_id in range(
        NUM_CLASSES
    ):

        if (
            total_img[cls_id] > 0
            and img[cls_id] <= 0
        ):
            score += 100.0

    # --------------------------------------------------------
    # 6. 稀有类额外保护
    #
    # 比如 tricycle 只有约 21 张图片，
    # 至少尽量接近理论 20%。
    # --------------------------------------------------------

    for cls_id in range(
        NUM_CLASSES
    ):

        target = target_img[
            cls_id
        ]

        if (
            total_img[cls_id]
            <= 150
            and total_img[cls_id] > 0
        ):

            # 不希望低于理论 val 图片数量的 60%
            minimum_desired = max(
                1.0,
                target * 0.60
            )

            if (
                img[cls_id]
                < minimum_desired
            ):
                shortage = (
                    minimum_desired
                    - img[cls_id]
                ) / (
                    minimum_desired
                    + 1.0
                )

                score += (
                    12.0
                    * shortage ** 2
                )

    return float(score)


# ============================================================
# 优化 group -> val
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

num_groups = len(
    groups_members
)

best_mask = None
best_vec = None
best_score = float("inf")


def make_initial_mask():

    order = rng.permutation(
        num_groups
    )

    mask = np.zeros(
        num_groups,
        dtype=bool
    )

    current_n = 0.0

    for gi in order:

        if current_n >= target_val_n:
            break

        mask[gi] = True

        current_n += G[
            gi,
            OFFSET_N
        ]

    return mask


print(
    "开始搜索 group-aware / stratified split..."
)

for restart in range(
    NUM_RESTARTS
):

    mask = make_initial_mask()

    vec = np.sum(
        G[mask],
        axis=0
    )

    score = objective(
        vec
    )

    temperature = 0.05

    for iteration in range(
        LOCAL_SEARCH_ITERS
    ):

        # 约 55% swap，45% single flip
        if (
            rng.random() < 0.55
            and np.any(mask)
            and np.any(~mask)
        ):

            val_indices = np.flatnonzero(
                mask
            )

            train_indices = np.flatnonzero(
                ~mask
            )

            out_idx = int(
                rng.choice(
                    val_indices
                )
            )

            in_idx = int(
                rng.choice(
                    train_indices
                )
            )

            new_vec = (
                vec
                - G[out_idx]
                + G[in_idx]
            )

            new_score = objective(
                new_vec
            )

            delta = (
                new_score - score
            )

            accept = (
                delta < 0
                or rng.random()
                < math.exp(
                    -delta
                    / max(
                        temperature,
                        1e-9
                    )
                )
            )

            if accept:

                mask[out_idx] = False
                mask[in_idx] = True

                vec = new_vec
                score = new_score

        else:

            idx = int(
                rng.integers(
                    0,
                    num_groups
                )
            )

            if mask[idx]:
                new_vec = (
                    vec - G[idx]
                )
            else:
                new_vec = (
                    vec + G[idx]
                )

            # 避免 train 或 val 为空
            new_n = new_vec[
                OFFSET_N
            ]

            if (
                new_n <= 0
                or new_n >= N
            ):
                continue

            new_score = objective(
                new_vec
            )

            delta = (
                new_score - score
            )

            accept = (
                delta < 0
                or rng.random()
                < math.exp(
                    -delta
                    / max(
                        temperature,
                        1e-9
                    )
                )
            )

            if accept:
                mask[idx] = (
                    not mask[idx]
                )

                vec = new_vec
                score = new_score

        temperature *= 0.999

    if score < best_score:

        best_score = score
        best_mask = mask.copy()
        best_vec = vec.copy()

    if (
        (restart + 1) % 10 == 0
        or restart == 0
    ):
        print(
            f"[{restart + 1:02d}/"
            f"{NUM_RESTARTS}] "
            f"当前最好 objective="
            f"{best_score:.6f}, "
            f"val_n="
            f"{int(best_vec[OFFSET_N])}"
        )


# ============================================================
# 得到 train / val
# ============================================================

val_groups = {
    i
    for i in range(
        num_groups
    )
    if best_mask[i]
}

train_stems = []
val_stems = []

group_id_for_stem = {}

for gi, members in enumerate(
    groups_members
):

    for stem in members:

        group_id_for_stem[
            stem
        ] = gi

        if gi in val_groups:
            val_stems.append(stem)
        else:
            train_stems.append(stem)

train_stems = sorted(
    train_stems,
    key=natural_key
)

val_stems = sorted(
    val_stems,
    key=natural_key
)

assert (
    len(train_stems)
    + len(val_stems)
    == N
)

assert (
    set(train_stems).isdisjoint(
        set(val_stems)
    )
)


# ============================================================
# 统计函数
# ============================================================

def calc_stats(split_stems):

    cls_instances = Counter()
    cls_images = Counter()
    src_counts = Counter()

    total_boxes = 0

    for stem in split_stems:

        info = sample_info[stem]

        counts = info[
            "class_counts"
        ]

        for cls_id, count in (
            counts.items()
        ):
            cls_instances[
                cls_id
            ] += count

            cls_images[
                cls_id
            ] += 1

            total_boxes += count

        src_counts[
            info["source"]
        ] += 1

    return (
        cls_instances,
        cls_images,
        src_counts,
        total_boxes,
    )


(
    train_cls_inst,
    train_cls_img,
    train_sources,
    train_boxes,
) = calc_stats(
    train_stems
)

(
    val_cls_inst,
    val_cls_img,
    val_sources,
    val_boxes,
) = calc_stats(
    val_stems
)


# ============================================================
# 相似帧跨 split 检查
# ============================================================

split_lookup = {}

for s in train_stems:
    split_lookup[s] = "train"

for s in val_stems:
    split_lookup[s] = "val"

cross_very_similar = []
cross_similar = []

for a, b, dist in similarity_rows:

    if (
        a not in split_lookup
        or b not in split_lookup
    ):
        continue

    if (
        split_lookup[a]
        == split_lookup[b]
    ):
        continue

    if (
        dist
        <= SIMILARITY_LOCK_THRESHOLD
    ):
        cross_very_similar.append(
            (a, b, dist)
        )

    elif (
        dist
        <= SIMILARITY_WARNING_THRESHOLD
    ):
        cross_similar.append(
            (a, b, dist)
        )


# ============================================================
# 创建输出目录
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# train.txt / val.txt
#
# 每行 basename，不带后缀。
# 例如：
# 000002
# 000003_080_00000307
# ============================================================

(
    OUTPUT_DIR / "train.txt"
).write_text(
    "\n".join(train_stems)
    + "\n",
    encoding="utf-8"
)

(
    OUTPUT_DIR / "val.txt"
).write_text(
    "\n".join(val_stems)
    + "\n",
    encoding="utf-8"
)


# ============================================================
# manifest.csv
# ============================================================

with (
    OUTPUT_DIR / "split_manifest.csv"
).open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    fieldnames = [
        "stem",
        "split",
        "group_id",
        "source",
        "visible",
        "infrared",
        "depth",
        "label",
        "num_objects",
        "classes",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for stem in stems:

        info = sample_info[stem]

        counts = info[
            "class_counts"
        ]

        classes_text = ";".join(
            str(x)
            for x in sorted(
                counts.keys()
            )
        )

        writer.writerow({
            "stem":
                stem,

            "split":
                split_lookup[stem],

            "group_id":
                group_id_for_stem[
                    stem
                ],

            "source":
                info["source"],

            "visible":
                info[
                    "visible"
                ].name,

            "infrared":
                info[
                    "infrared"
                ].name,

            "depth":
                info[
                    "depth"
                ].name,

            "label":
                info[
                    "label"
                ].name,

            "num_objects":
                sum(
                    counts.values()
                ),

            "classes":
                classes_text,
        })


# ============================================================
# groups.csv
# ============================================================

with (
    OUTPUT_DIR / "groups.csv"
).open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "group_id",
        "split",
        "size",
        "first_stem",
        "last_stem",
    ])

    for gi, members in enumerate(
        groups_members
    ):

        ms = sorted(
            members,
            key=natural_key
        )

        writer.writerow([
            gi,
            (
                "val"
                if gi in val_groups
                else "train"
            ),
            len(ms),
            ms[0],
            ms[-1],
        ])


# ============================================================
# class stats CSV
# ============================================================

with (
    OUTPUT_DIR
    / "class_split_stats.csv"
).open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "class_id",
        "class_name",
        "total_instances",
        "train_instances",
        "val_instances",
        "val_instance_ratio",
        "total_images",
        "train_images",
        "val_images",
        "val_image_ratio",
    ])

    for cls_id in range(
        NUM_CLASSES
    ):

        total_i = (
            train_cls_inst[cls_id]
            + val_cls_inst[cls_id]
        )

        total_im = (
            train_cls_img[cls_id]
            + val_cls_img[cls_id]
        )

        writer.writerow([
            cls_id,
            CLASS_NAMES[cls_id],

            total_i,
            train_cls_inst[
                cls_id
            ],
            val_cls_inst[
                cls_id
            ],

            (
                val_cls_inst[
                    cls_id
                ] / total_i
                if total_i > 0
                else 0
            ),

            total_im,
            train_cls_img[
                cls_id
            ],
            val_cls_img[
                cls_id
            ],

            (
                val_cls_img[
                    cls_id
                ] / total_im
                if total_im > 0
                else 0
            ),
        ])


# ============================================================
# source stats CSV
# ============================================================

with (
    OUTPUT_DIR
    / "source_split_stats.csv"
).open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "source",
        "total",
        "train",
        "val",
        "val_ratio",
    ])

    for src in sources:

        tr = train_sources[src]
        va = val_sources[src]
        total = tr + va

        writer.writerow([
            src,
            total,
            tr,
            va,
            (
                va / total
                if total > 0
                else 0
            ),
        ])


# ============================================================
# 相似帧跨 split CSV
# ============================================================

with (
    OUTPUT_DIR
    / "cross_split_similar_pairs.csv"
).open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "stem_a",
        "split_a",
        "stem_b",
        "split_b",
        "dhash_distance",
        "level",
    ])

    for a, b, dist in (
        cross_very_similar
    ):

        writer.writerow([
            a,
            split_lookup[a],
            b,
            split_lookup[b],
            dist,
            "VERY_SIMILAR",
        ])

    for a, b, dist in (
        cross_similar
    ):

        writer.writerow([
            a,
            split_lookup[a],
            b,
            split_lookup[b],
            dist,
            "SIMILAR",
        ])


# ============================================================
# summary
# ============================================================

lines = []

lines.append(
    "=" * 88
)

lines.append(
    "AIC2026 Train / Validation Split Report"
)

lines.append(
    "=" * 88
)

lines.append("")

lines.append(
    "[重要说明]"
)

lines.append(
    "本次划分没有移动、复制、删除、重命名或修改任何官方训练数据。"
)

lines.append(
    "只生成 train/val 索引及统计文件。"
)

lines.append("")

lines.append(
    "[1] 总体"
)

lines.append(
    f"总样本数       : {N}"
)

lines.append(
    f"Train          : {len(train_stems)} "
    f"({len(train_stems)/N:.4%})"
)

lines.append(
    f"Validation     : {len(val_stems)} "
    f"({len(val_stems)/N:.4%})"
)

lines.append(
    f"目标 Val 比例  : {VAL_RATIO:.2%}"
)

lines.append(
    f"Objective      : {best_score:.8f}"
)

lines.append("")

lines.append(
    "[2] Group"
)

lines.append(
    f"Group 总数     : {len(groups_members)}"
)

lines.append(
    f"Train Groups   : "
    f"{len(groups_members) - len(val_groups)}"
)

lines.append(
    f"Val Groups     : {len(val_groups)}"
)

lines.append(
    f"最小 Group     : {min(group_sizes)}"
)

lines.append(
    f"最大 Group     : {max(group_sizes)}"
)

lines.append(
    f"平均 Group     : {np.mean(group_sizes):.2f}"
)

lines.append("")

lines.append(
    "[3] Bounding Box 实例"
)

lines.append(
    f"Train boxes    : {train_boxes}"
)

lines.append(
    f"Val boxes      : {val_boxes}"
)

lines.append(
    f"Total boxes    : {train_boxes + val_boxes}"
)

lines.append("")

lines.append(
    "[4] 类别分布"
)

lines.append(
    "id  class         "
    "total  train  val   "
    "val_inst%   "
    "train_img  val_img  val_img%"
)

for cls_id in range(
    NUM_CLASSES
):

    tr_i = train_cls_inst[
        cls_id
    ]

    va_i = val_cls_inst[
        cls_id
    ]

    total_i = tr_i + va_i

    tr_im = train_cls_img[
        cls_id
    ]

    va_im = val_cls_img[
        cls_id
    ]

    total_im = tr_im + va_im

    lines.append(
        f"{cls_id:2d}  "
        f"{CLASS_NAMES[cls_id]:12s} "
        f"{total_i:5d}  "
        f"{tr_i:5d}  "
        f"{va_i:4d}  "
        f"{(va_i/total_i if total_i else 0):9.2%}   "
        f"{tr_im:5d}     "
        f"{va_im:5d}   "
        f"{(va_im/total_im if total_im else 0):8.2%}"
    )

lines.append("")

lines.append(
    "[5] 数据来源分布"
)

for src in sources:

    tr = train_sources[src]
    va = val_sources[src]
    total = tr + va

    lines.append(
        f"{src:30s} "
        f"total={total:4d} "
        f"train={tr:4d} "
        f"val={va:4d} "
        f"val_ratio="
        f"{(va/total if total else 0):.2%}"
    )

lines.append("")

lines.append(
    "[6] 相邻图视觉泄漏检查"
)

lines.append(
    f"dHash <= "
    f"{SIMILARITY_LOCK_THRESHOLD} "
    f"跨 split："
    f"{len(cross_very_similar)}"
)

lines.append(
    f"dHash "
    f"{SIMILARITY_LOCK_THRESHOLD + 1}"
    f"~{SIMILARITY_WARNING_THRESHOLD} "
    f"跨 split："
    f"{len(cross_similar)}"
)

if len(cross_very_similar) == 0:

    lines.append(
        "极相似帧强约束：PASS"
    )

else:

    lines.append(
        "极相似帧强约束：WARNING"
    )

lines.append("")

lines.append(
    "[7] 稀有类别重点"
)

for cls_id in [
    1,   # boat
    7,   # ball
    10,  # uav
    11,  # tricycle
]:

    total_im = (
        train_cls_img[cls_id]
        + val_cls_img[cls_id]
    )

    lines.append(
        f"{CLASS_NAMES[cls_id]:12s}: "
        f"train_images="
        f"{train_cls_img[cls_id]}, "
        f"val_images="
        f"{val_cls_img[cls_id]}, "
        f"total_images={total_im}, "
        f"val_ratio="
        f"{(val_cls_img[cls_id]/total_im if total_im else 0):.2%}"
    )

lines.append("")

lines.append(
    "[8] 输出文件"
)

lines.append(
    "train.txt                      Train basename 列表"
)

lines.append(
    "val.txt                        Validation basename 列表"
)

lines.append(
    "split_manifest.csv             每个样本的最终 split"
)

lines.append(
    "groups.csv                     Group 分组情况"
)

lines.append(
    "class_split_stats.csv          12 类统计"
)

lines.append(
    "source_split_stats.csv         数据来源统计"
)

lines.append(
    "cross_split_similar_pairs.csv  相似帧泄漏检查"
)

lines.append(
    "split_stats.txt                本报告"
)

summary = "\n".join(
    lines
)

(
    OUTPUT_DIR
    / "split_stats.txt"
).write_text(
    summary,
    encoding="utf-8"
)


# ============================================================
# 最终打印
# ============================================================

print()
print(summary)

print()
print("=" * 88)
print("划分完成")
print("=" * 88)

print(
    f"输出目录：{OUTPUT_DIR}"
)

print()
print(
    "官方数据目录未发生任何修改。"
)


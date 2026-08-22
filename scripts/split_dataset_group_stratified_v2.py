#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from collections import Counter, defaultdict
import csv
import math
import re
import sys
import hashlib

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

HEALTH_REPORT_DIR = Path(
    "/home/xhm/Desktop/aicomp/dataset_health_report"
)

ADJACENT_CSV = (
    HEALTH_REPORT_DIR / "adjacent_similarity.csv"
)

OUTPUT_DIR = Path(
    "/home/xhm/Desktop/aicomp/splits/aic2026_group_stratified_v2"
)

VAL_RATIO = 0.20
RANDOM_SEED = 20260821

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

# V2:
# V1 是 25，这里缩小到 20，
# 使 numeric_6digit / numeric_8digit 来源更容易接近 20%。
NUMERIC_BLOCK_SIZE = 20

# V2:
# dHash <= 10 全部强制在同一个 split。
SIMILARITY_LOCK_THRESHOLD = 10

# 搜索力度。
# 2000 张数据规模下可以接受。
NUM_RESTARTS = 100
LOCAL_SEARCH_ITERS = 5000

# 理想的数据源验证集比例范围
SOURCE_RATIO_LOW = 0.18
SOURCE_RATIO_HIGH = 0.22


# ============================================================
# 工具函数
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
            f"{stem}: {name} 文件数量={len(paths)}，"
            "要求恰好为 1"
        )

    return paths[0]


def read_classes(label_path):
    """
    只读取官方标签用于类别分层。
    不修改、不重写标签。
    """

    counts = Counter()

    try:
        text = label_path.read_text(
            encoding="utf-8",
            errors="replace"
        )
    except Exception:
        return counts

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        # 官方目标检测标签理论上应为 5 列
        if len(parts) != 5:
            continue

        try:
            cls_raw = float(parts[0])
        except Exception:
            continue

        if not math.isfinite(cls_raw):
            continue

        cls_id = int(cls_raw)

        if abs(cls_raw - cls_id) > 1e-9:
            continue

        if 0 <= cls_id < NUM_CLASSES:
            counts[cls_id] += 1

    return counts


def source_family(stem, visible_path):
    """
    根据体检结果区分数据来源。
    """

    ext = (
        visible_path
        .suffix
        .lower()
        .lstrip(".")
    )

    if stem.isdigit():

        return (
            f"numeric_"
            f"{len(stem)}digit_"
            f"{ext}"
        )

    s = stem.lower()

    if "suppl" in s:
        return f"suppl_{ext}"

    if s.startswith("shuming"):
        return f"shuming_{ext}"

    if s.startswith("hehe"):
        return f"hehe_{ext}"

    return f"structured_other_{ext}"


def sha256_lines(lines):
    data = (
        "\n".join(lines)
        + "\n"
    ).encode("utf-8")

    return hashlib.sha256(
        data
    ).hexdigest()


# ============================================================
# Union-Find
# ============================================================

class DSU:

    def __init__(self, items):

        self.parent = {
            x: x
            for x in items
        }

        self.rank = {
            x: 0
            for x in items
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
# 安全检查
# ============================================================

DATASET_ROOT_RESOLVED = (
    DATASET_ROOT.resolve()
)

OUTPUT_DIR_RESOLVED = (
    OUTPUT_DIR.resolve()
)

try:
    OUTPUT_DIR_RESOLVED.relative_to(
        DATASET_ROOT_RESOLVED
    )

    print(
        "ERROR: OUTPUT_DIR 不能放在官方数据集目录内部。"
    )
    sys.exit(1)

except ValueError:
    pass


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
# 扫描数据
# ============================================================

print("=" * 90)
print("AIC2026 Train/Val Split V2")
print("=" * 90)

print()
print(
    "本脚本只读取官方数据，"
    "所有输出均写入数据集目录之外。"
)
print()

visible_map = scan_images(
    VISIBLE_DIR
)

infrared_map = scan_images(
    INFRARED_DIR
)

depth_map = scan_images(
    DEPTH_DIR
)

label_map = scan_labels(
    LABEL_DIR
)

sets = [
    set(visible_map.keys()),
    set(infrared_map.keys()),
    set(depth_map.keys()),
    set(label_map.keys()),
]

if not (
    sets[0]
    == sets[1]
    == sets[2]
    == sets[3]
):

    print(
        "ERROR: depth/infrared/labels/visible "
        "basename 不完全一致。"
    )

    sys.exit(1)


stems = sorted(
    sets[0],
    key=natural_key
)

N = len(stems)

target_val_n = round(
    N * VAL_RATIO
)

print(
    f"样本总数      : {N}"
)

print(
    f"目标 Train    : {N - target_val_n}"
)

print(
    f"目标 Val      : {target_val_n}"
)

print()


# ============================================================
# 样本信息
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

    sample_info[stem] = {
        "visible": vp,
        "infrared": ip,
        "depth": dp,
        "label": lp,
        "classes": read_classes(lp),
        "source": source_family(
            stem,
            vp
        ),
    }


# ============================================================
# 基础 Group
# ============================================================

base_groups = defaultdict(list)
numeric_buckets = defaultdict(list)


for stem in stems:

    visible_path = (
        sample_info[stem]["visible"]
    )

    if stem.isdigit():

        key = (
            len(stem),
            visible_path.suffix.lower(),
        )

        numeric_buckets[
            key
        ].append(stem)

    else:

        # 例如：
        #
        # 000003_080_00000307
        # 000003_080_00000409
        #
        # 都归入：
        #
        # 000003_080
        #
        match = re.match(
            r"^(.*)_([0-9]+)$",
            stem
        )

        if match:
            prefix = match.group(1)
        else:
            prefix = stem

        base_groups[
            f"sequence::{prefix}"
        ].append(stem)


# ============================================================
# 纯数字文件每 20 张作为一个基础块
# ============================================================

for key, bucket in (
    numeric_buckets.items()
):

    bucket = sorted(
        bucket,
        key=lambda x: int(x)
    )

    digits_len, ext = key

    for index, stem in enumerate(
        bucket
    ):

        block_id = (
            index
            // NUMERIC_BLOCK_SIZE
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
# 基础 Group 放入 DSU
# ============================================================

dsu = DSU(stems)

for members in (
    base_groups.values()
):

    if len(members) <= 1:
        continue

    anchor = members[0]

    for stem in members[1:]:

        dsu.union(
            anchor,
            stem
        )


# ============================================================
# 相似帧锁定
# ============================================================

similarity_rows = []
locked_similarity_pairs = 0

if ADJACENT_CSV.exists():

    print(
        "读取："
        f"{ADJACENT_CSV}"
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

                dsu.union(
                    a,
                    b
                )

                locked_similarity_pairs += 1

else:

    print()
    print(
        "ERROR: 找不到 adjacent_similarity.csv"
    )

    print(
        "请先运行之前的数据集体检脚本。"
    )

    sys.exit(1)


print(
    f"dHash <= {SIMILARITY_LOCK_THRESHOLD} "
    f"锁定相邻对：{locked_similarity_pairs}"
)

print()


# ============================================================
# 最终 Atomic Groups
# ============================================================

groups_dict = defaultdict(list)

for stem in stems:

    root = dsu.find(stem)

    groups_dict[
        root
    ].append(stem)


groups_members = list(
    groups_dict.values()
)

groups_members.sort(
    key=lambda members:
        natural_key(
            min(
                members,
                key=natural_key
            )
        )
)

group_sizes = np.array(
    [
        len(x)
        for x in groups_members
    ],
    dtype=np.int32
)


print("Atomic Groups")
print("-" * 90)

print(
    f"Group 数      : {len(groups_members)}"
)

print(
    f"最小 Group    : {group_sizes.min()}"
)

print(
    f"最大 Group    : {group_sizes.max()}"
)

print(
    f"平均 Group    : {group_sizes.mean():.2f}"
)

print()


# ============================================================
# 数据来源
# ============================================================

sources = sorted({
    sample_info[s]["source"]
    for s in stems
})

source_to_idx = {
    src: idx
    for idx, src in enumerate(sources)
}


print("数据来源")
print("-" * 90)

for src in sources:

    count = sum(
        1
        for stem in stems
        if sample_info[stem][
            "source"
        ] == src
    )

    print(
        f"{src:32s}"
        f"{count:5d}"
    )

print()


# ============================================================
# Group Feature Vector
#
# [
# sample_count,
# class_instances x12,
# class_images x12,
# source_count xS
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
    dtype=np.float64
)


for gi, members in enumerate(
    groups_members
):

    G[
        gi,
        OFFSET_N
    ] = len(members)

    for stem in members:

        info = sample_info[
            stem
        ]

        counts = info[
            "classes"
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

        source_idx = (
            source_to_idx[
                info["source"]
            ]
        )

        G[
            gi,
            OFFSET_SOURCE
            + source_idx
        ] += 1


TOTAL = np.sum(
    G,
    axis=0
)


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


# ============================================================
# Objective V2
# ============================================================

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

    score = 0.0

    # --------------------------------------------------------
    # A. 总样本数
    # 10 张误差作为一个大致尺度
    # --------------------------------------------------------

    size_error = (
        (
            val_n
            - target_val_n
        )
        / 10.0
    ) ** 2

    score += (
        10.0
        * size_error
    )

    # --------------------------------------------------------
    # B. 各类别实例比例
    # --------------------------------------------------------

    valid_inst = (
        total_inst > 0
    )

    inst_ratio = np.zeros(
        NUM_CLASSES,
        dtype=np.float64
    )

    inst_ratio[
        valid_inst
    ] = (
        inst[valid_inst]
        /
        total_inst[valid_inst]
    )

    inst_dev = (
        inst_ratio
        - VAL_RATIO
    )

    # 5 个百分点作为基础尺度
    inst_error = (
        inst_dev
        / 0.05
    ) ** 2

    score += (
        2.0
        * float(
            np.mean(
                inst_error[
                    valid_inst
                ]
            )
        )
    )

    # --------------------------------------------------------
    # C. 各类别出现图片比例
    #
    # 对最终 mAP 更重要。
    # 各类别基本平权。
    # --------------------------------------------------------

    valid_img = (
        total_img > 0
    )

    img_ratio = np.zeros(
        NUM_CLASSES,
        dtype=np.float64
    )

    img_ratio[
        valid_img
    ] = (
        img[valid_img]
        /
        total_img[valid_img]
    )

    img_dev = (
        img_ratio
        - VAL_RATIO
    )

    class_weights = np.ones(
        NUM_CLASSES,
        dtype=np.float64
    )

    # 稀有类额外加权
    for cls_id in range(
        NUM_CLASSES
    ):

        if total_img[cls_id] <= 30:
            class_weights[
                cls_id
            ] = 3.0

        elif total_img[cls_id] <= 80:
            class_weights[
                cls_id
            ] = 2.2

        elif total_img[cls_id] <= 150:
            class_weights[
                cls_id
            ] = 1.6

    class_img_error = (
        (
            img_dev
            / 0.04
        ) ** 2
        * class_weights
    )

    score += (
        7.0
        * float(
            np.mean(
                class_img_error[
                    valid_img
                ]
            )
        )
    )

    # --------------------------------------------------------
    # D. 数据来源比例
    #
    # V2 显著提高权重。
    # --------------------------------------------------------

    valid_source = (
        total_source > 0
    )

    source_ratio = np.zeros(
        len(sources),
        dtype=np.float64
    )

    source_ratio[
        valid_source
    ] = (
        src[valid_source]
        /
        total_source[
            valid_source
        ]
    )

    source_dev = (
        source_ratio
        - VAL_RATIO
    )

    source_error = (
        source_dev
        / 0.02
    ) ** 2

    score += (
        12.0
        * float(
            np.mean(
                source_error[
                    valid_source
                ]
            )
        )
    )

    # --------------------------------------------------------
    # E. 来源超出 18%-22% 时，
    # 再加额外罚分。
    # --------------------------------------------------------

    low_excess = np.maximum(
        SOURCE_RATIO_LOW
        - source_ratio,
        0.0
    )

    high_excess = np.maximum(
        source_ratio
        - SOURCE_RATIO_HIGH,
        0.0
    )

    source_outside = (
        low_excess
        + high_excess
    )

    score += (
        25.0
        * float(
            np.sum(
                (
                    source_outside
                    / 0.01
                ) ** 2
            )
        )
    )

    # --------------------------------------------------------
    # F. 禁止任何有数据的类别
    # 在 Val 中完全缺失
    # --------------------------------------------------------

    for cls_id in range(
        NUM_CLASSES
    ):

        if (
            total_img[cls_id] > 0
            and img[cls_id] <= 0
        ):
            score += 10000.0

    # --------------------------------------------------------
    # G. 稀有类最低保护
    # --------------------------------------------------------

    for cls_id in range(
        NUM_CLASSES
    ):

        total_cls_images = (
            total_img[cls_id]
        )

        if total_cls_images <= 0:
            continue

        # 至少希望达到 15% 左右
        min_images = max(
            1,
            math.floor(
                total_cls_images
                * 0.15
            )
        )

        if (
            total_cls_images <= 30
        ):

            # 对极稀有类再稍微严格：
            # 例如 tricycle 21 张，
            # 目标至少保住约 4 张。
            min_images = max(
                min_images,
                round(
                    total_cls_images
                    * VAL_RATIO
                )
            )

        if img[cls_id] < min_images:

            shortage = (
                min_images
                - img[cls_id]
            )

            score += (
                100.0
                * shortage ** 2
            )

    return float(score)


# ============================================================
# 初始方案
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

num_groups = len(
    groups_members
)


def make_initial_mask():

    order = rng.permutation(
        num_groups
    )

    mask = np.zeros(
        num_groups,
        dtype=bool
    )

    current = np.zeros(
        VECTOR_DIM,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # 每一步随机抽一批候选，
    # 选 objective 最好的 group。
    # --------------------------------------------------------

    remaining = set(
        int(x)
        for x in order
    )

    while (
        current[
            OFFSET_N
        ]
        < target_val_n
        and remaining
    ):

        candidates = list(
            remaining
        )

        if len(candidates) > 40:

            candidates = (
                rng.choice(
                    candidates,
                    size=40,
                    replace=False
                )
                .astype(int)
                .tolist()
            )

        best_candidate = None
        best_candidate_score = (
            float("inf")
        )

        for gi in candidates:

            candidate_vec = (
                current
                + G[gi]
            )

            candidate_score = (
                objective(
                    candidate_vec
                )
            )

            # 太严重超过目标数量时，
            # 初始阶段额外惩罚。
            excess = (
                candidate_vec[
                    OFFSET_N
                ]
                - target_val_n
            )

            if excess > 20:

                candidate_score += (
                    excess ** 2
                )

            if (
                candidate_score
                < best_candidate_score
            ):

                best_candidate_score = (
                    candidate_score
                )

                best_candidate = gi

        if best_candidate is None:
            break

        mask[
            best_candidate
        ] = True

        current += G[
            best_candidate
        ]

        remaining.remove(
            best_candidate
        )

    return mask


# ============================================================
# 多次重启 + Local Search
# ============================================================

best_mask = None
best_vec = None
best_score = float("inf")


print(
    "开始 V2 分层优化..."
)

print(
    f"Restarts={NUM_RESTARTS}, "
    f"Iterations={LOCAL_SEARCH_ITERS}"
)

print()


for restart in range(
    NUM_RESTARTS
):

    mask = make_initial_mask()

    if np.any(mask):

        vec = np.sum(
            G[mask],
            axis=0
        )

    else:

        vec = np.zeros(
            VECTOR_DIM,
            dtype=np.float64
        )

    score = objective(
        vec
    )

    temperature = 0.10

    for iteration in range(
        LOCAL_SEARCH_ITERS
    ):

        r = rng.random()

        # ----------------------------------------------------
        # 65%：val/train 各换一个 group
        # ----------------------------------------------------

        if (
            r < 0.65
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
                new_score
                - score
            )

            accept = (
                delta < 0
                or rng.random()
                < math.exp(
                    -delta
                    / max(
                        temperature,
                        1e-12
                    )
                )
            )

            if accept:

                mask[
                    out_idx
                ] = False

                mask[
                    in_idx
                ] = True

                vec = new_vec
                score = new_score

        # ----------------------------------------------------
        # 35%：单个 group 翻转
        # ----------------------------------------------------

        else:

            idx = int(
                rng.integers(
                    0,
                    num_groups
                )
            )

            if mask[idx]:

                new_vec = (
                    vec
                    - G[idx]
                )

            else:

                new_vec = (
                    vec
                    + G[idx]
                )

            new_n = (
                new_vec[
                    OFFSET_N
                ]
            )

            if (
                new_n <= 0
                or new_n >= N
            ):
                continue

            new_score = objective(
                new_vec
            )

            delta = (
                new_score
                - score
            )

            accept = (
                delta < 0
                or rng.random()
                < math.exp(
                    -delta
                    / max(
                        temperature,
                        1e-12
                    )
                )
            )

            if accept:

                mask[idx] = (
                    not mask[idx]
                )

                vec = new_vec
                score = new_score

        temperature *= 0.9992

    if score < best_score:

        best_score = score

        best_mask = (
            mask.copy()
        )

        best_vec = (
            vec.copy()
        )

    if (
        restart == 0
        or (restart + 1) % 10 == 0
    ):

        print(
            f"[{restart + 1:03d}/"
            f"{NUM_RESTARTS}] "
            f"best_score="
            f"{best_score:.6f}, "
            f"val_n="
            f"{int(best_vec[OFFSET_N])}"
        )


# ============================================================
# 最终 Split
# ============================================================

val_group_indices = {
    gi
    for gi in range(
        num_groups
    )
    if best_mask[gi]
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

        if gi in val_group_indices:

            val_stems.append(
                stem
            )

        else:

            train_stems.append(
                stem
            )


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

assert set(
    train_stems
).isdisjoint(
    set(val_stems)
)


# ============================================================
# Split 统计
# ============================================================

def calculate_split_stats(
    split_stems
):

    class_instances = Counter()
    class_images = Counter()
    source_counts = Counter()

    total_boxes = 0

    for stem in split_stems:

        info = sample_info[
            stem
        ]

        counts = info[
            "classes"
        ]

        for cls_id, count in (
            counts.items()
        ):

            class_instances[
                cls_id
            ] += count

            class_images[
                cls_id
            ] += 1

            total_boxes += count

        source_counts[
            info["source"]
        ] += 1

    return (
        class_instances,
        class_images,
        source_counts,
        total_boxes,
    )


(
    train_cls_inst,
    train_cls_img,
    train_sources,
    train_boxes,
) = calculate_split_stats(
    train_stems
)


(
    val_cls_inst,
    val_cls_img,
    val_sources,
    val_boxes,
) = calculate_split_stats(
    val_stems
)


# ============================================================
# 相似帧泄漏检查
# ============================================================

split_lookup = {}

for stem in train_stems:
    split_lookup[stem] = "train"

for stem in val_stems:
    split_lookup[stem] = "val"


cross_similar_pairs = []


for a, b, dist in (
    similarity_rows
):

    if (
        a not in split_lookup
        or b not in split_lookup
    ):
        continue

    if (
        split_lookup[a]
        != split_lookup[b]
        and dist
        <= SIMILARITY_LOCK_THRESHOLD
    ):

        cross_similar_pairs.append(
            (
                a,
                b,
                dist,
            )
        )


# ============================================================
# 输出
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# train.txt / val.txt
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# split hash
# ------------------------------------------------------------

train_sha = sha256_lines(
    train_stems
)

val_sha = sha256_lines(
    val_stems
)


# ------------------------------------------------------------
# manifest
# ------------------------------------------------------------

with (
    OUTPUT_DIR
    / "split_manifest.csv"
).open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    fields = [
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
        fieldnames=fields
    )

    writer.writeheader()

    for stem in stems:

        info = sample_info[
            stem
        ]

        counts = info[
            "classes"
        ]

        writer.writerow({
            "stem":
                stem,

            "split":
                split_lookup[
                    stem
                ],

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
                ";".join(
                    str(cls_id)
                    for cls_id
                    in sorted(
                        counts.keys()
                    )
                ),
        })


# ------------------------------------------------------------
# groups.csv
# ------------------------------------------------------------

with (
    OUTPUT_DIR
    / "groups.csv"
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

        members_sorted = sorted(
            members,
            key=natural_key
        )

        writer.writerow([
            gi,

            (
                "val"
                if gi in val_group_indices
                else "train"
            ),

            len(members),

            members_sorted[0],

            members_sorted[-1],
        ])


# ------------------------------------------------------------
# class stats
# ------------------------------------------------------------

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

        tr_i = train_cls_inst[
            cls_id
        ]

        va_i = val_cls_inst[
            cls_id
        ]

        total_i = (
            tr_i + va_i
        )

        tr_img = train_cls_img[
            cls_id
        ]

        va_img = val_cls_img[
            cls_id
        ]

        total_images = (
            tr_img + va_img
        )

        writer.writerow([
            cls_id,
            CLASS_NAMES[
                cls_id
            ],
            total_i,
            tr_i,
            va_i,

            (
                va_i / total_i
                if total_i
                else 0
            ),

            total_images,
            tr_img,
            va_img,

            (
                va_img
                / total_images
                if total_images
                else 0
            ),
        ])


# ------------------------------------------------------------
# source stats
# ------------------------------------------------------------

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
        "target_ratio",
        "inside_18_22_percent",
    ])

    for src in sources:

        tr = train_sources[src]
        va = val_sources[src]

        total = tr + va

        ratio = (
            va / total
            if total
            else 0
        )

        writer.writerow([
            src,
            total,
            tr,
            va,
            ratio,
            VAL_RATIO,

            (
                "PASS"
                if (
                    SOURCE_RATIO_LOW
                    <= ratio
                    <= SOURCE_RATIO_HIGH
                )
                else "CHECK"
            ),
        ])


# ------------------------------------------------------------
# cross split similar
# ------------------------------------------------------------

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
    ])

    for a, b, dist in (
        cross_similar_pairs
    ):

        writer.writerow([
            a,
            split_lookup[a],
            b,
            split_lookup[b],
            dist,
        ])


# ============================================================
# 报告
# ============================================================

lines = []

lines.append(
    "=" * 90
)

lines.append(
    "AIC2026 Group-aware Stratified Split V2"
)

lines.append(
    "=" * 90
)

lines.append("")

lines.append(
    "[0] 数据安全"
)

lines.append(
    "官方数据集：只读。"
)

lines.append(
    "脚本未移动、复制、删除、重命名或修改任何官方文件。"
)

lines.append(
    f"输出目录：{OUTPUT_DIR}"
)

lines.append("")


lines.append(
    "[1] 总体划分"
)

lines.append(
    f"Total       : {N}"
)

lines.append(
    f"Train       : "
    f"{len(train_stems)} "
    f"({len(train_stems)/N:.2%})"
)

lines.append(
    f"Validation  : "
    f"{len(val_stems)} "
    f"({len(val_stems)/N:.2%})"
)

lines.append(
    f"目标 Val    : "
    f"{VAL_RATIO:.2%}"
)

lines.append(
    f"Objective   : "
    f"{best_score:.6f}"
)

lines.append("")


lines.append(
    "[2] Split 固定指纹"
)

lines.append(
    f"train.txt SHA256: {train_sha}"
)

lines.append(
    f"val.txt   SHA256: {val_sha}"
)

lines.append(
    "后续正式锁定验证集后，可用这两个 SHA256 确认 split 没有变化。"
)

lines.append("")


lines.append(
    "[3] Atomic Groups"
)

lines.append(
    f"Groups      : "
    f"{len(groups_members)}"
)

lines.append(
    f"Train Groups: "
    f"{len(groups_members) - len(val_group_indices)}"
)

lines.append(
    f"Val Groups  : "
    f"{len(val_group_indices)}"
)

lines.append(
    f"Group min   : "
    f"{group_sizes.min()}"
)

lines.append(
    f"Group max   : "
    f"{group_sizes.max()}"
)

lines.append(
    f"Group mean  : "
    f"{group_sizes.mean():.2f}"
)

lines.append("")


lines.append(
    "[4] 类别分布"
)

lines.append(
    "id  class         "
    "instances(train/val/ratio)       "
    "images(train/val/ratio)"
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

    total_i = (
        tr_i + va_i
    )

    tr_img = train_cls_img[
        cls_id
    ]

    va_img = val_cls_img[
        cls_id
    ]

    total_images = (
        tr_img + va_img
    )

    inst_ratio = (
        va_i / total_i
        if total_i
        else 0
    )

    img_ratio = (
        va_img / total_images
        if total_images
        else 0
    )

    lines.append(
        f"{cls_id:2d}  "
        f"{CLASS_NAMES[cls_id]:12s} "
        f"{tr_i:5d}/{va_i:4d}/"
        f"{inst_ratio:6.2%}          "
        f"{tr_img:4d}/{va_img:3d}/"
        f"{img_ratio:6.2%}"
    )

lines.append("")


lines.append(
    "[5] 稀有类别重点"
)

for cls_id in [
    1,
    7,
    10,
    11,
]:

    tr_img = train_cls_img[
        cls_id
    ]

    va_img = val_cls_img[
        cls_id
    ]

    total_images = (
        tr_img + va_img
    )

    lines.append(
        f"{CLASS_NAMES[cls_id]:12s}: "
        f"train_images={tr_img}, "
        f"val_images={va_img}, "
        f"total={total_images}, "
        f"val_ratio="
        f"{(va_img/total_images if total_images else 0):.2%}"
    )

lines.append("")


lines.append(
    "[6] 数据来源"
)

source_all_pass = True

for src in sources:

    tr = train_sources[
        src
    ]

    va = val_sources[
        src
    ]

    total = tr + va

    ratio = (
        va / total
        if total
        else 0
    )

    passed = (
        SOURCE_RATIO_LOW
        <= ratio
        <= SOURCE_RATIO_HIGH
    )

    if not passed:
        source_all_pass = False

    lines.append(
        f"{src:32s} "
        f"total={total:4d} "
        f"train={tr:4d} "
        f"val={va:4d} "
        f"ratio={ratio:6.2%} "
        f"{'PASS' if passed else 'CHECK'}"
    )

lines.append("")

lines.append(
    "来源 18%-22% 总体："
    + (
        "PASS"
        if source_all_pass
        else "部分来源未达到目标范围"
    )
)

lines.append("")


lines.append(
    "[7] 相似帧泄漏"
)

lines.append(
    f"锁定阈值：dHash <= "
    f"{SIMILARITY_LOCK_THRESHOLD}"
)

lines.append(
    f"跨 Train/Val 相似对："
    f"{len(cross_similar_pairs)}"
)

if len(
    cross_similar_pairs
) == 0:

    lines.append(
        "相似帧隔离：PASS"
    )

else:

    lines.append(
        "相似帧隔离：WARNING"
    )

lines.append("")


lines.append(
    "[8] Bounding Box 数"
)

lines.append(
    f"Train boxes : "
    f"{train_boxes}"
)

lines.append(
    f"Val boxes   : "
    f"{val_boxes}"
)

lines.append(
    f"Total boxes : "
    f"{train_boxes + val_boxes}"
)

lines.append(
    "注：这里只按标签中的合法 class_id 统计，"
    "不修改官方标签内容。"
)

lines.append("")


lines.append(
    "[9] 输出文件"
)

lines.append(
    "train.txt"
)

lines.append(
    "val.txt"
)

lines.append(
    "split_manifest.csv"
)

lines.append(
    "groups.csv"
)

lines.append(
    "class_split_stats.csv"
)

lines.append(
    "source_split_stats.csv"
)

lines.append(
    "cross_split_similar_pairs.csv"
)

lines.append(
    "split_stats.txt"
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


print()
print(summary)

print()
print("=" * 90)
print("V2 划分完成")
print("=" * 90)

print()
print(
    "官方训练数据未被修改。"
)

print(
    f"结果：{OUTPUT_DIR}"
)


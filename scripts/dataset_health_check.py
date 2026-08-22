#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import math
import re
import sys

try:
    import cv2
except ImportError:
    print("ERROR: 缺少 opencv-python")
    print("安装：pip install opencv-python")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: 缺少 numpy")
    print("安装：pip install numpy")
    sys.exit(1)


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

REPORT_DIR = Path(
    "/home/xhm/Desktop/aicomp/dataset_health_report"
)

EXPECTED_SAMPLES = 2000

IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
}

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

# 竞赛细则给出的 Depth 大致有效距离
DEPTH_MIN_MM = 300
DEPTH_MAX_MM = 19999

# dHash 相邻帧阈值
# 越小越相似
VERY_SIMILAR_DHASH = 5
SIMILAR_DHASH = 10


# ============================================================
# 工具函数
# ============================================================

def natural_key(text):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", str(text))
    ]


def percentile(values, q):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def fmt_num(x, digits=4):
    if x is None:
        return "N/A"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "N/A"
    return f"{x:.{digits}f}"


def image_map(folder):
    """
    返回:
      stem -> [Path, Path, ...]
    如果同 stem 同时出现 000001.png / 000001.jpg，
    会保留多个文件，后续标记 duplicate stem。
    """
    result = defaultdict(list)

    if not folder.exists():
        return result

    for p in folder.iterdir():
        if not p.is_file():
            continue

        if p.suffix.lower() in IMAGE_EXTS:
            result[p.stem].append(p)

    return result


def label_map(folder):
    result = defaultdict(list)

    if not folder.exists():
        return result

    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() == ".txt":
            result[p.stem].append(p)

    return result


def choose_unique(mapping, stem):
    paths = mapping.get(stem, [])
    if len(paths) == 1:
        return paths[0]
    return None


def safe_imread(path):
    try:
        img = cv2.imread(
            str(path),
            cv2.IMREAD_UNCHANGED
        )
        return img
    except Exception:
        return None


def shape_text(img):
    if img is None:
        return "READ_ERROR"
    return "x".join(map(str, img.shape))


def dhash_from_bgr_or_gray(img):
    """
    64-bit dHash。
    用于粗略判断相邻 visible 图片是否高度相似。
    """
    if img is None:
        return None

    if img.ndim == 3:
        if img.shape[2] == 4:
            gray = cv2.cvtColor(
                img,
                cv2.COLOR_BGRA2GRAY
            )
        else:
            gray = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )
    else:
        gray = img

    # uint16 等先归一到 uint8
    if gray.dtype != np.uint8:
        minv = float(np.min(gray))
        maxv = float(np.max(gray))

        if maxv > minv:
            gray = (
                (gray.astype(np.float32) - minv)
                / (maxv - minv)
                * 255.0
            ).astype(np.uint8)
        else:
            gray = np.zeros_like(
                gray,
                dtype=np.uint8
            )

    small = cv2.resize(
        gray,
        (9, 8),
        interpolation=cv2.INTER_AREA
    )

    diff = small[:, 1:] > small[:, :-1]

    value = 0

    for bit in diff.flatten():
        value = (value << 1) | int(bit)

    return value


def hamming_distance(a, b):
    if a is None or b is None:
        return None
    return (a ^ b).bit_count()


def write_csv(path, fieldnames, rows):
    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 检查目录
# ============================================================

print("=" * 80)
print("AIC2026 多模态目标检测数据集体检")
print("=" * 80)

required_dirs = [
    VISIBLE_DIR,
    INFRARED_DIR,
    DEPTH_DIR,
    LABEL_DIR,
]

missing_dirs = [
    p for p in required_dirs
    if not p.exists()
]

if missing_dirs:
    print("\nERROR: 以下目录不存在：")
    for p in missing_dirs:
        print("  ", p)
    sys.exit(1)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print(f"数据集：{DATASET_ROOT}")
print(f"报告  ：{REPORT_DIR}")
print()


# ============================================================
# 建立文件索引
# ============================================================

visible_map = image_map(VISIBLE_DIR)
infrared_map = image_map(INFRARED_DIR)
depth_map = image_map(DEPTH_DIR)
labels_map = label_map(LABEL_DIR)

all_stems = set()
all_stems.update(visible_map)
all_stems.update(infrared_map)
all_stems.update(depth_map)
all_stems.update(labels_map)

all_stems = sorted(
    all_stems,
    key=natural_key
)

print("文件数量")
print("-" * 80)
print(f"visible  stems : {len(visible_map)}")
print(f"infrared stems : {len(infrared_map)}")
print(f"depth    stems : {len(depth_map)}")
print(f"labels   stems : {len(labels_map)}")
print(f"union    stems : {len(all_stems)}")
print()


# ============================================================
# 汇总变量
# ============================================================

issues = []
sample_rows = []

class_instance_count = Counter()
class_image_count = Counter()

targets_per_image = []

bbox_width_px = []
bbox_height_px = []
bbox_area_px = []
bbox_area_ratio = []
bbox_aspect_ratio = []

bbox_width_norm = []
bbox_height_norm = []

small_count = 0
medium_count = 0
large_count = 0

empty_labels = []

invalid_label_files = []

visible_shapes = Counter()
infrared_shapes = Counter()
depth_shapes = Counter()

visible_dtypes = Counter()
infrared_dtypes = Counter()
depth_dtypes = Counter()

depth_zero_ratios = []
depth_valid_ratios = []
depth_under_min_ratios = []
depth_over_max_ratios = []

depth_valid_values_sampled = []

infrared_black_ratios = []
infrared_channel_mads = []
infrared_exact_channel_equal = 0
infrared_3channel_count = 0

visible_hashes = {}

complete_samples = 0

format_counter = {
    "visible": Counter(),
    "infrared": Counter(),
    "depth": Counter(),
}


# ============================================================
# 主循环
# ============================================================

print("开始逐样本检查...")
print("-" * 80)

for idx, stem in enumerate(all_stems, start=1):

    v_paths = visible_map.get(stem, [])
    i_paths = infrared_map.get(stem, [])
    d_paths = depth_map.get(stem, [])
    l_paths = labels_map.get(stem, [])

    row_issues = []

    # --------------------------------------------------------
    # 文件对应关系
    # --------------------------------------------------------

    if len(v_paths) == 0:
        row_issues.append("missing_visible")
    elif len(v_paths) > 1:
        row_issues.append("duplicate_visible_stem")

    if len(i_paths) == 0:
        row_issues.append("missing_infrared")
    elif len(i_paths) > 1:
        row_issues.append("duplicate_infrared_stem")

    if len(d_paths) == 0:
        row_issues.append("missing_depth")
    elif len(d_paths) > 1:
        row_issues.append("duplicate_depth_stem")

    if len(l_paths) == 0:
        row_issues.append("missing_label")
    elif len(l_paths) > 1:
        row_issues.append("duplicate_label_stem")

    v_path = choose_unique(
        visible_map,
        stem
    )
    i_path = choose_unique(
        infrared_map,
        stem
    )
    d_path = choose_unique(
        depth_map,
        stem
    )
    l_path = choose_unique(
        labels_map,
        stem
    )

    if v_path:
        format_counter["visible"][
            v_path.suffix.lower()
        ] += 1

    if i_path:
        format_counter["infrared"][
            i_path.suffix.lower()
        ] += 1

    if d_path:
        format_counter["depth"][
            d_path.suffix.lower()
        ] += 1

    v_img = None
    i_img = None
    d_img = None

    # --------------------------------------------------------
    # visible
    # --------------------------------------------------------

    if v_path is not None:
        v_img = safe_imread(v_path)

        if v_img is None:
            row_issues.append(
                "visible_read_error"
            )
        else:
            visible_shapes[
                shape_text(v_img)
            ] += 1

            visible_dtypes[
                str(v_img.dtype)
            ] += 1

            visible_hashes[stem] = (
                dhash_from_bgr_or_gray(v_img)
            )

    # --------------------------------------------------------
    # infrared
    # --------------------------------------------------------

    ir_black_ratio = None
    ir_channel_mad = None

    if i_path is not None:
        i_img = safe_imread(i_path)

        if i_img is None:
            row_issues.append(
                "infrared_read_error"
            )
        else:
            infrared_shapes[
                shape_text(i_img)
            ] += 1

            infrared_dtypes[
                str(i_img.dtype)
            ] += 1

            # 黑色/配准无效区域比例
            if i_img.ndim == 3:
                gray_ir = cv2.cvtColor(
                    i_img,
                    cv2.COLOR_BGR2GRAY
                )
            else:
                gray_ir = i_img

            ir_black_ratio = float(
                np.mean(gray_ir <= 2)
            )

            infrared_black_ratios.append(
                ir_black_ratio
            )

            # 检查三通道是否接近相同
            if (
                i_img.ndim == 3
                and i_img.shape[2] >= 3
            ):
                infrared_3channel_count += 1

                b = i_img[:, :, 0].astype(
                    np.float32
                )
                g = i_img[:, :, 1].astype(
                    np.float32
                )
                r = i_img[:, :, 2].astype(
                    np.float32
                )

                mad_bg = float(
                    np.mean(np.abs(b - g))
                )
                mad_gr = float(
                    np.mean(np.abs(g - r))
                )
                mad_br = float(
                    np.mean(np.abs(b - r))
                )

                ir_channel_mad = (
                    mad_bg + mad_gr + mad_br
                ) / 3.0

                infrared_channel_mads.append(
                    ir_channel_mad
                )

                if (
                    np.array_equal(
                        i_img[:, :, 0],
                        i_img[:, :, 1]
                    )
                    and np.array_equal(
                        i_img[:, :, 1],
                        i_img[:, :, 2]
                    )
                ):
                    infrared_exact_channel_equal += 1

    # --------------------------------------------------------
    # depth
    # --------------------------------------------------------

    depth_zero_ratio = None
    depth_valid_ratio = None
    depth_under_ratio = None
    depth_over_ratio = None
    depth_min = None
    depth_max = None

    if d_path is not None:
        d_img = safe_imread(d_path)

        if d_img is None:
            row_issues.append(
                "depth_read_error"
            )
        else:
            depth_shapes[
                shape_text(d_img)
            ] += 1

            depth_dtypes[
                str(d_img.dtype)
            ] += 1

            if d_path.suffix.lower() in {
                ".jpg",
                ".jpeg",
            }:
                row_issues.append(
                    "depth_is_jpeg"
                )

            if d_img.dtype != np.uint16:
                row_issues.append(
                    f"depth_dtype_{d_img.dtype}"
                )

            if d_img.ndim != 2:
                row_issues.append(
                    f"depth_not_single_channel_shape_{d_img.shape}"
                )

                # 如果意外是三通道，
                # 为统计临时取第一个通道
                if d_img.ndim == 3:
                    depth_stat = d_img[:, :, 0]
                else:
                    depth_stat = d_img
            else:
                depth_stat = d_img

            if depth_stat.size > 0:
                depth_min = int(
                    np.min(depth_stat)
                )
                depth_max = int(
                    np.max(depth_stat)
                )

                zero_mask = (
                    depth_stat == 0
                )

                under_mask = (
                    (depth_stat > 0)
                    & (
                        depth_stat
                        < DEPTH_MIN_MM
                    )
                )

                over_mask = (
                    depth_stat
                    > DEPTH_MAX_MM
                )

                valid_mask = (
                    (depth_stat >= DEPTH_MIN_MM)
                    & (
                        depth_stat
                        <= DEPTH_MAX_MM
                    )
                )

                depth_zero_ratio = float(
                    np.mean(zero_mask)
                )

                depth_under_ratio = float(
                    np.mean(under_mask)
                )

                depth_over_ratio = float(
                    np.mean(over_mask)
                )

                depth_valid_ratio = float(
                    np.mean(valid_mask)
                )

                depth_zero_ratios.append(
                    depth_zero_ratio
                )

                depth_under_min_ratios.append(
                    depth_under_ratio
                )

                depth_over_max_ratios.append(
                    depth_over_ratio
                )

                depth_valid_ratios.append(
                    depth_valid_ratio
                )

                valid_values = (
                    depth_stat[valid_mask]
                )

                # 防止内存过大：
                # 每张图最多随机均匀采 5000 个有效深度
                if valid_values.size > 0:
                    if valid_values.size > 5000:
                        sample_idx = np.linspace(
                            0,
                            valid_values.size - 1,
                            5000
                        ).astype(np.int64)

                        valid_values = (
                            valid_values[
                                sample_idx
                            ]
                        )

                    depth_valid_values_sampled.extend(
                        valid_values.astype(
                            np.int32
                        ).tolist()
                    )

    # --------------------------------------------------------
    # 三模态尺寸对齐
    # --------------------------------------------------------

    if (
        v_img is not None
        and i_img is not None
        and d_img is not None
    ):
        v_hw = v_img.shape[:2]
        i_hw = i_img.shape[:2]
        d_hw = d_img.shape[:2]

        if not (
            v_hw == i_hw == d_hw
        ):
            row_issues.append(
                "modality_size_mismatch"
            )

    # --------------------------------------------------------
    # labels
    # --------------------------------------------------------

    target_count = None

    if l_path is not None:

        current_classes = set()
        valid_targets = 0

        try:
            text = l_path.read_text(
                encoding="utf-8",
                errors="replace"
            )

            lines = [
                x.strip()
                for x in text.splitlines()
                if x.strip()
            ]

            target_count = len(lines)

            if len(lines) == 0:
                empty_labels.append(stem)

            for line_num, line in enumerate(
                lines,
                start=1
            ):
                parts = line.split()

                if len(parts) != 5:
                    row_issues.append(
                        f"label_bad_columns_line_{line_num}"
                    )
                    continue

                try:
                    cls_raw = float(parts[0])
                    cx = float(parts[1])
                    cy = float(parts[2])
                    bw = float(parts[3])
                    bh = float(parts[4])
                except ValueError:
                    row_issues.append(
                        f"label_non_numeric_line_{line_num}"
                    )
                    continue

                vals = [
                    cls_raw,
                    cx,
                    cy,
                    bw,
                    bh,
                ]

                if not all(
                    math.isfinite(x)
                    for x in vals
                ):
                    row_issues.append(
                        f"label_nan_inf_line_{line_num}"
                    )
                    continue

                cls_id = int(cls_raw)

                if abs(
                    cls_raw - cls_id
                ) > 1e-9:
                    row_issues.append(
                        f"label_class_not_integer_line_{line_num}"
                    )
                    continue

                if not (
                    0 <= cls_id < NUM_CLASSES
                ):
                    row_issues.append(
                        f"label_invalid_class_{cls_id}_line_{line_num}"
                    )
                    continue

                if not (
                    0.0 <= cx <= 1.0
                    and 0.0 <= cy <= 1.0
                    and 0.0 < bw <= 1.0
                    and 0.0 < bh <= 1.0
                ):
                    row_issues.append(
                        f"label_invalid_xywh_line_{line_num}"
                    )
                    continue

                # 检查框实际边缘是否越界
                x1 = cx - bw / 2.0
                y1 = cy - bh / 2.0
                x2 = cx + bw / 2.0
                y2 = cy + bh / 2.0

                if (
                    x1 < -1e-6
                    or y1 < -1e-6
                    or x2 > 1.0 + 1e-6
                    or y2 > 1.0 + 1e-6
                ):
                    row_issues.append(
                        f"bbox_cross_image_boundary_line_{line_num}"
                    )

                class_instance_count[
                    cls_id
                ] += 1

                current_classes.add(
                    cls_id
                )

                valid_targets += 1

                bbox_width_norm.append(
                    bw
                )
                bbox_height_norm.append(
                    bh
                )
                bbox_area_ratio.append(
                    bw * bh
                )

                if bh > 0:
                    bbox_aspect_ratio.append(
                        bw / bh
                    )

                # 使用 visible 原图大小换算像素框
                if v_img is not None:
                    h, w = v_img.shape[:2]

                    bw_px = bw * w
                    bh_px = bh * h
                    area_px = (
                        bw_px * bh_px
                    )

                    bbox_width_px.append(
                        bw_px
                    )
                    bbox_height_px.append(
                        bh_px
                    )
                    bbox_area_px.append(
                        area_px
                    )

                    # COCO 的原图像素面积标准
                    if area_px < 32 ** 2:
                        small_count += 1
                    elif area_px < 96 ** 2:
                        medium_count += 1
                    else:
                        large_count += 1

            for cls_id in current_classes:
                class_image_count[
                    cls_id
                ] += 1

            targets_per_image.append(
                valid_targets
            )

        except Exception as e:
            row_issues.append(
                f"label_read_exception_{type(e).__name__}"
            )
            invalid_label_files.append(
                stem
            )

    # --------------------------------------------------------
    # 完整样本
    # --------------------------------------------------------

    if (
        v_path is not None
        and i_path is not None
        and d_path is not None
        and l_path is not None
    ):
        complete_samples += 1

    # --------------------------------------------------------
    # 记录 issue
    # --------------------------------------------------------

    for issue in row_issues:
        issues.append({
            "stem": stem,
            "issue": issue,
        })

    sample_rows.append({
        "stem": stem,

        "visible_file":
            v_path.name
            if v_path else "",

        "infrared_file":
            i_path.name
            if i_path else "",

        "depth_file":
            d_path.name
            if d_path else "",

        "label_file":
            l_path.name
            if l_path else "",

        "visible_shape":
            shape_text(v_img),

        "visible_dtype":
            str(v_img.dtype)
            if v_img is not None else "",

        "infrared_shape":
            shape_text(i_img),

        "infrared_dtype":
            str(i_img.dtype)
            if i_img is not None else "",

        "depth_shape":
            shape_text(d_img),

        "depth_dtype":
            str(d_img.dtype)
            if d_img is not None else "",

        "depth_min":
            depth_min
            if depth_min is not None else "",

        "depth_max":
            depth_max
            if depth_max is not None else "",

        "depth_zero_ratio":
            depth_zero_ratio
            if depth_zero_ratio is not None else "",

        "depth_valid_ratio":
            depth_valid_ratio
            if depth_valid_ratio is not None else "",

        "depth_under_300_ratio":
            depth_under_ratio
            if depth_under_ratio is not None else "",

        "depth_over_19999_ratio":
            depth_over_ratio
            if depth_over_ratio is not None else "",

        "infrared_black_ratio":
            ir_black_ratio
            if ir_black_ratio is not None else "",

        "infrared_channel_mad":
            ir_channel_mad
            if ir_channel_mad is not None else "",

        "target_count":
            target_count
            if target_count is not None else "",

        "issues":
            ";".join(row_issues),
    })

    if (
        idx % 100 == 0
        or idx == len(all_stems)
    ):
        print(
            f"[{idx:4d}/{len(all_stems):4d}] "
            f"checked"
        )


# ============================================================
# 相邻 visible 图片相似度
# ============================================================

print()
print("计算相邻 visible 图像相似度...")

adjacent_rows = []

visible_stems_sorted = sorted(
    visible_hashes.keys(),
    key=natural_key
)

very_similar_pairs = 0
similar_pairs = 0

for a, b in zip(
    visible_stems_sorted[:-1],
    visible_stems_sorted[1:]
):
    dist = hamming_distance(
        visible_hashes[a],
        visible_hashes[b]
    )

    if dist is None:
        continue

    if dist <= VERY_SIMILAR_DHASH:
        level = "very_similar"
        very_similar_pairs += 1
    elif dist <= SIMILAR_DHASH:
        level = "similar"
        similar_pairs += 1
    else:
        level = "normal"

    adjacent_rows.append({
        "stem_a": a,
        "stem_b": b,
        "dhash_distance": dist,
        "similarity_level": level,
    })


# ============================================================
# Class CSV
# ============================================================

class_rows = []

total_instances = sum(
    class_instance_count.values()
)

for cls_id in range(NUM_CLASSES):
    count = class_instance_count[
        cls_id
    ]

    class_rows.append({
        "class_id": cls_id,
        "class_name": CLASS_NAMES[
            cls_id
        ],
        "instances": count,
        "images_containing_class":
            class_image_count[
                cls_id
            ],
        "instance_ratio":
            (
                count / total_instances
                if total_instances > 0
                else 0
            ),
    })


# ============================================================
# 写 CSV
# ============================================================

write_csv(
    REPORT_DIR / "samples.csv",
    list(sample_rows[0].keys())
    if sample_rows else ["stem"],
    sample_rows
)

write_csv(
    REPORT_DIR / "issues.csv",
    ["stem", "issue"],
    issues
)

write_csv(
    REPORT_DIR / "class_stats.csv",
    [
        "class_id",
        "class_name",
        "instances",
        "images_containing_class",
        "instance_ratio",
    ],
    class_rows
)

write_csv(
    REPORT_DIR / "adjacent_similarity.csv",
    [
        "stem_a",
        "stem_b",
        "dhash_distance",
        "similarity_level",
    ],
    adjacent_rows
)


# ============================================================
# Summary
# ============================================================

issue_counter = Counter(
    x["issue"]
    for x in issues
)

summary = {
    "dataset_root":
        str(DATASET_ROOT),

    "expected_samples":
        EXPECTED_SAMPLES,

    "unique_stems_union":
        len(all_stems),

    "complete_samples":
        complete_samples,

    "visible_stems":
        len(visible_map),

    "infrared_stems":
        len(infrared_map),

    "depth_stems":
        len(depth_map),

    "label_stems":
        len(labels_map),

    "total_instances":
        total_instances,

    "empty_label_files":
        len(empty_labels),

    "issues_total":
        len(issues),

    "visible_shapes":
        dict(visible_shapes),

    "infrared_shapes":
        dict(infrared_shapes),

    "depth_shapes":
        dict(depth_shapes),

    "visible_dtypes":
        dict(visible_dtypes),

    "infrared_dtypes":
        dict(infrared_dtypes),

    "depth_dtypes":
        dict(depth_dtypes),

    "format_counts": {
        k: dict(v)
        for k, v
        in format_counter.items()
    },

    "depth": {
        "mean_zero_ratio":
            float(np.mean(
                depth_zero_ratios
            ))
            if depth_zero_ratios else None,

        "median_zero_ratio":
            percentile(
                depth_zero_ratios,
                50
            ),

        "mean_valid_ratio":
            float(np.mean(
                depth_valid_ratios
            ))
            if depth_valid_ratios else None,

        "median_valid_ratio":
            percentile(
                depth_valid_ratios,
                50
            ),

        "valid_depth_mm_p01":
            percentile(
                depth_valid_values_sampled,
                1
            ),

        "valid_depth_mm_p05":
            percentile(
                depth_valid_values_sampled,
                5
            ),

        "valid_depth_mm_p50":
            percentile(
                depth_valid_values_sampled,
                50
            ),

        "valid_depth_mm_p95":
            percentile(
                depth_valid_values_sampled,
                95
            ),

        "valid_depth_mm_p99":
            percentile(
                depth_valid_values_sampled,
                99
            ),
    },

    "infrared": {
        "mean_black_ratio":
            float(np.mean(
                infrared_black_ratios
            ))
            if infrared_black_ratios
            else None,

        "median_black_ratio":
            percentile(
                infrared_black_ratios,
                50
            ),

        "mean_channel_mad":
            float(np.mean(
                infrared_channel_mads
            ))
            if infrared_channel_mads
            else None,

        "three_channel_images":
            infrared_3channel_count,

        "exact_equal_channel_images":
            infrared_exact_channel_equal,
    },

    "bbox": {
        "width_px_p01":
            percentile(
                bbox_width_px,
                1
            ),

        "width_px_p05":
            percentile(
                bbox_width_px,
                5
            ),

        "width_px_p50":
            percentile(
                bbox_width_px,
                50
            ),

        "height_px_p01":
            percentile(
                bbox_height_px,
                1
            ),

        "height_px_p05":
            percentile(
                bbox_height_px,
                5
            ),

        "height_px_p50":
            percentile(
                bbox_height_px,
                50
            ),

        "area_ratio_p01":
            percentile(
                bbox_area_ratio,
                1
            ),

        "area_ratio_p05":
            percentile(
                bbox_area_ratio,
                5
            ),

        "area_ratio_p50":
            percentile(
                bbox_area_ratio,
                50
            ),

        "coco_small":
            small_count,

        "coco_medium":
            medium_count,

        "coco_large":
            large_count,
    },

    "adjacent_similarity": {
        "pairs_checked":
            len(adjacent_rows),

        "very_similar_dhash_le_5":
            very_similar_pairs,

        "similar_dhash_6_to_10":
            similar_pairs,
    },

    "issue_counts":
        dict(issue_counter),
}

with (
    REPORT_DIR / "summary.json"
).open(
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 人类可读 summary.txt
# ============================================================

summary_lines = []

summary_lines.append(
    "=" * 80
)
summary_lines.append(
    "AIC2026 数据集体检报告"
)
summary_lines.append(
    "=" * 80
)
summary_lines.append(
    ""
)

summary_lines.append(
    "[1] 数据完整性"
)
summary_lines.append(
    f"预期样本数        : {EXPECTED_SAMPLES}"
)
summary_lines.append(
    f"联合 basename 数  : {len(all_stems)}"
)
summary_lines.append(
    f"完整四件套样本数  : {complete_samples}"
)
summary_lines.append(
    f"visible           : {len(visible_map)}"
)
summary_lines.append(
    f"infrared          : {len(infrared_map)}"
)
summary_lines.append(
    f"depth             : {len(depth_map)}"
)
summary_lines.append(
    f"labels            : {len(labels_map)}"
)
summary_lines.append(
    ""
)

summary_lines.append(
    "[2] 文件格式"
)

for modality in [
    "visible",
    "infrared",
    "depth",
]:
    summary_lines.append(
        f"{modality:10s}: "
        f"{dict(format_counter[modality])}"
    )

summary_lines.append(
    ""
)

summary_lines.append(
    "[3] 图像尺寸"
)
summary_lines.append(
    f"visible  : {dict(visible_shapes)}"
)
summary_lines.append(
    f"infrared : {dict(infrared_shapes)}"
)
summary_lines.append(
    f"depth    : {dict(depth_shapes)}"
)
summary_lines.append(
    ""
)

summary_lines.append(
    "[4] 图像 dtype"
)
summary_lines.append(
    f"visible  : {dict(visible_dtypes)}"
)
summary_lines.append(
    f"infrared : {dict(infrared_dtypes)}"
)
summary_lines.append(
    f"depth    : {dict(depth_dtypes)}"
)
summary_lines.append(
    ""
)

summary_lines.append(
    "[5] Depth 数据"
)
summary_lines.append(
    "有效深度定义      : "
    f"{DEPTH_MIN_MM} ~ "
    f"{DEPTH_MAX_MM} mm"
)
summary_lines.append(
    "平均 zero ratio   : "
    + fmt_num(
        summary["depth"][
            "mean_zero_ratio"
        ]
    )
)
summary_lines.append(
    "中位 zero ratio   : "
    + fmt_num(
        summary["depth"][
            "median_zero_ratio"
        ]
    )
)
summary_lines.append(
    "平均 valid ratio  : "
    + fmt_num(
        summary["depth"][
            "mean_valid_ratio"
        ]
    )
)
summary_lines.append(
    "中位 valid ratio  : "
    + fmt_num(
        summary["depth"][
            "median_valid_ratio"
        ]
    )
)
summary_lines.append(
    "有效深度 P01/P05/P50/P95/P99 mm : "
    + " / ".join(
        fmt_num(
            summary["depth"][k],
            1
        )
        for k in [
            "valid_depth_mm_p01",
            "valid_depth_mm_p05",
            "valid_depth_mm_p50",
            "valid_depth_mm_p95",
            "valid_depth_mm_p99",
        ]
    )
)
summary_lines.append(
    ""
)

summary_lines.append(
    "[6] Infrared"
)
summary_lines.append(
    "平均近黑像素比例  : "
    + fmt_num(
        summary["infrared"][
            "mean_black_ratio"
        ]
    )
)
summary_lines.append(
    "中位近黑像素比例  : "
    + fmt_num(
        summary["infrared"][
            "median_black_ratio"
        ]
    )
)
summary_lines.append(
    "三通道图数量      : "
    f"{infrared_3channel_count}"
)
summary_lines.append(
    "三通道完全相等    : "
    f"{infrared_exact_channel_equal}"
)
summary_lines.append(
    "平均通道 MAD      : "
    + fmt_num(
        summary["infrared"][
            "mean_channel_mad"
        ]
    )
)
summary_lines.append(
    ""
)

summary_lines.append(
    "[7] 标签"
)
summary_lines.append(
    f"目标总实例数      : {total_instances}"
)
summary_lines.append(
    f"空标签文件        : {len(empty_labels)}"
)
summary_lines.append(
    "每图目标数 P05/P50/P95 : "
    + " / ".join(
        fmt_num(
            percentile(
                targets_per_image,
                q
            ),
            1
        )
        for q in [5, 50, 95]
    )
)
summary_lines.append(
    ""
)

summary_lines.append(
    "[8] 类别分布"
)

for cls_id in range(NUM_CLASSES):
    summary_lines.append(
        f"{cls_id:2d} "
        f"{CLASS_NAMES[cls_id]:12s} "
        f"instances={class_instance_count[cls_id]:6d} "
        f"images={class_image_count[cls_id]:5d}"
    )

summary_lines.append(
    ""
)

summary_lines.append(
    "[9] Bounding Box"
)
summary_lines.append(
    "width px P01/P05/P50 : "
    + " / ".join(
        fmt_num(
            percentile(
                bbox_width_px,
                q
            ),
            1
        )
        for q in [1, 5, 50]
    )
)
summary_lines.append(
    "height px P01/P05/P50: "
    + " / ".join(
        fmt_num(
            percentile(
                bbox_height_px,
                q
            ),
            1
        )
        for q in [1, 5, 50]
    )
)
summary_lines.append(
    "area ratio P01/P05/P50: "
    + " / ".join(
        fmt_num(
            percentile(
                bbox_area_ratio,
                q
            ),
            6
        )
        for q in [1, 5, 50]
    )
)
summary_lines.append(
    f"COCO small  : {small_count}"
)
summary_lines.append(
    f"COCO medium : {medium_count}"
)
summary_lines.append(
    f"COCO large  : {large_count}"
)
summary_lines.append(
    ""
)

summary_lines.append(
    "[10] 相邻 visible 图像相似度"
)
summary_lines.append(
    f"检查相邻对数       : {len(adjacent_rows)}"
)
summary_lines.append(
    "dHash <= 5 极相似  : "
    f"{very_similar_pairs}"
)
summary_lines.append(
    "dHash 6~10 较相似  : "
    f"{similar_pairs}"
)
summary_lines.append(
    ""
)
summary_lines.append(
    "说明：如果极相似相邻帧很多，"
    "后续不建议随机逐图划分 train/val。"
)
summary_lines.append(
    ""
)

summary_lines.append(
    "[11] Issues"
)
summary_lines.append(
    f"问题记录总数       : {len(issues)}"
)

if issue_counter:
    for issue, count in (
        issue_counter.most_common()
    ):
        summary_lines.append(
            f"{count:6d}  {issue}"
        )
else:
    summary_lines.append(
        "未发现结构性问题。"
    )

summary_lines.append(
    ""
)
summary_lines.append(
    "[12] 输出文件"
)
summary_lines.append(
    "summary.txt              总报告"
)
summary_lines.append(
    "summary.json             结构化总报告"
)
summary_lines.append(
    "samples.csv              每个样本详细信息"
)
summary_lines.append(
    "issues.csv               所有异常"
)
summary_lines.append(
    "class_stats.csv          类别统计"
)
summary_lines.append(
    "adjacent_similarity.csv 相邻帧相似度"
)

summary_text = "\n".join(
    summary_lines
)

(
    REPORT_DIR / "summary.txt"
).write_text(
    summary_text,
    encoding="utf-8"
)


# ============================================================
# 打印最终报告
# ============================================================

print()
print(summary_text)
print()
print("=" * 80)
print("体检完成")
print("=" * 80)
print(
    f"完整报告目录：{REPORT_DIR}"
)
print(
    f"重点先看：{REPORT_DIR / 'summary.txt'}"
)
print(
    f"异常明细：{REPORT_DIR / 'issues.csv'}"
)
print(
    f"类别统计：{REPORT_DIR / 'class_stats.csv'}"
)
print(
    "相邻帧："
    f"{REPORT_DIR / 'adjacent_similarity.csv'}"
)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 - Analyze JPG Depth Anomalies

背景：
    官方 2000 张 Depth 中：

        1851 张：
            uint16
            单通道
            1080x1920

        149 张：
            uint8
            3 通道
            360x640
            JPG

本脚本专门分析这 149 张 JPG Depth，判断它们究竟属于：

    A. 三通道完全相同的灰度深度图
    B. 三通道近似相同（JPEG 压缩造成轻微差异）
    C. 真正的伪彩色 / 彩色 Depth 表示
    D. 其他异常格式

统计内容：

1. JPG 文件数量
2. dtype / shape
3. B/G/R：
       min
       max
       mean
       std
4. 通道差异：
       |B-G|
       |B-R|
       |G-R|

       mean
       max
5. 每个像素满足：
       B == G == R
       max_channel_diff <= 1
       max_channel_diff <= 2
       max_channel_diff <= 5
       max_channel_diff <= 10
6. 灰度图统计：
       min
       max
       P1
       P5
       P50
       P95
       P99
7. 每张图独立统计
8. 自动生成 JPG Depth 样例拼图

输出：

runs/analyze_depth_jpg_anomalies/
├── jpg_depth_summary.json
├── jpg_depth_file_stats.csv
├── jpg_depth_samples.jpg
└── suspicious_files.txt

注意：
    本脚本不修改任何原始数据。
"""

from pathlib import Path
from collections import Counter
import csv
import json
import math

import cv2
import numpy as np


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "AIC2026_Train_2000"
)

RUNS_DIR = (
    PROJECT_ROOT
    / "runs"
)

OUTPUT_DIR = (
    RUNS_DIR
    / "analyze_depth_jpg_anomalies"
)


# ============================================================
# Configuration
# ============================================================

EXPECTED_JPG_COUNT = 149

# Number of sample images in montage
SAMPLE_COUNT = 24

# Montage tile size
TILE_WIDTH = 320
TILE_HEIGHT = 180

# Number of columns
MONTAGE_COLUMNS = 4


# ============================================================
# Locate Depth directory
# ============================================================

def find_depth_dir():

    candidates = [
        DATASET_ROOT / "depth",
        DATASET_ROOT / "Depth",
        DATASET_ROOT / "DEPTH",
    ]

    found = [
        path
        for path in candidates
        if path.is_dir()
    ]

    unique = []
    seen = set()

    for path in found:

        real = path.resolve()

        if real not in seen:
            seen.add(real)
            unique.append(path)

    if len(unique) == 1:
        return unique[0]

    if len(unique) > 1:
        raise RuntimeError(
            "检测到多个 Depth 目录：\n"
            + "\n".join(
                f"  {path}"
                for path in unique
            )
        )

    raise FileNotFoundError(
        "没有找到 Depth 数据目录：\n"
        + "\n".join(
            f"  {path}"
            for path in candidates
        )
    )


# ============================================================
# Utility
# ============================================================

def ratio(
    numerator,
    denominator,
):

    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator
    )


def percentile_values(array):

    if array.size == 0:

        return {
            "p1": None,
            "p5": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }

    values = np.percentile(
        array,
        [
            1,
            5,
            50,
            95,
            99,
        ],
    )

    return {
        "p1": float(values[0]),
        "p5": float(values[1]),
        "p50": float(values[2]),
        "p95": float(values[3]),
        "p99": float(values[4]),
    }


def channel_statistics(channel):

    return {
        "min": int(
            channel.min()
        ),

        "max": int(
            channel.max()
        ),

        "mean": float(
            channel.mean()
        ),

        "std": float(
            channel.std()
        ),
    }


# ============================================================
# Analyze one JPG
# ============================================================

def analyze_image(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    if image is None:

        return {
            "status": "read_failed",
        }

    shape = tuple(
        int(x)
        for x in image.shape
    )

    dtype = str(
        image.dtype
    )

    if (
        image.ndim != 3
        or image.shape[2] != 3
    ):

        return {
            "status": "not_3_channel",
            "shape": shape,
            "dtype": dtype,
        }

    if image.dtype != np.uint8:

        return {
            "status": "not_uint8",
            "shape": shape,
            "dtype": dtype,
        }

    # OpenCV = BGR
    b = image[:, :, 0].astype(
        np.int16
    )

    g = image[:, :, 1].astype(
        np.int16
    )

    r = image[:, :, 2].astype(
        np.int16
    )

    pixel_count = int(
        b.size
    )

    # ========================================================
    # Absolute channel difference
    # ========================================================

    diff_bg = np.abs(
        b - g
    )

    diff_br = np.abs(
        b - r
    )

    diff_gr = np.abs(
        g - r
    )

    max_channel_diff = np.maximum(
        np.maximum(
            diff_bg,
            diff_br,
        ),
        diff_gr,
    )

    equal_mask = (
        (b == g)
        & (b == r)
    )

    diff_le_1 = (
        max_channel_diff <= 1
    )

    diff_le_2 = (
        max_channel_diff <= 2
    )

    diff_le_5 = (
        max_channel_diff <= 5
    )

    diff_le_10 = (
        max_channel_diff <= 10
    )

    # ========================================================
    # Convert to grayscale only for statistical inspection.
    #
    # This DOES NOT imply we will necessarily use grayscale
    # conversion when building depth8_v1.
    # ========================================================

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray_percentiles = (
        percentile_values(
            gray.reshape(-1)
        )
    )

    # ========================================================
    # Per-image result
    # ========================================================

    result = {
        "status": "ok",

        "shape": shape,
        "dtype": dtype,

        "pixel_count":
            pixel_count,

        # ---------------------------------------------
        # Channel statistics
        # ---------------------------------------------

        "b_min":
            int(b.min()),

        "b_max":
            int(b.max()),

        "b_mean":
            float(b.mean()),

        "b_std":
            float(b.std()),

        "g_min":
            int(g.min()),

        "g_max":
            int(g.max()),

        "g_mean":
            float(g.mean()),

        "g_std":
            float(g.std()),

        "r_min":
            int(r.min()),

        "r_max":
            int(r.max()),

        "r_mean":
            float(r.mean()),

        "r_std":
            float(r.std()),

        # ---------------------------------------------
        # Difference statistics
        # ---------------------------------------------

        "mean_abs_bg":
            float(
                diff_bg.mean()
            ),

        "mean_abs_br":
            float(
                diff_br.mean()
            ),

        "mean_abs_gr":
            float(
                diff_gr.mean()
            ),

        "max_abs_bg":
            int(
                diff_bg.max()
            ),

        "max_abs_br":
            int(
                diff_br.max()
            ),

        "max_abs_gr":
            int(
                diff_gr.max()
            ),

        "mean_max_channel_diff":
            float(
                max_channel_diff.mean()
            ),

        "max_channel_diff":
            int(
                max_channel_diff.max()
            ),

        # ---------------------------------------------
        # Equality ratios
        # ---------------------------------------------

        "exact_equal_ratio":
            float(
                equal_mask.mean()
            ),

        "diff_le_1_ratio":
            float(
                diff_le_1.mean()
            ),

        "diff_le_2_ratio":
            float(
                diff_le_2.mean()
            ),

        "diff_le_5_ratio":
            float(
                diff_le_5.mean()
            ),

        "diff_le_10_ratio":
            float(
                diff_le_10.mean()
            ),

        # ---------------------------------------------
        # Gray distribution
        # ---------------------------------------------

        "gray_min":
            int(
                gray.min()
            ),

        "gray_max":
            int(
                gray.max()
            ),

        "gray_mean":
            float(
                gray.mean()
            ),

        "gray_std":
            float(
                gray.std()
            ),

        "gray_p1":
            gray_percentiles["p1"],

        "gray_p5":
            gray_percentiles["p5"],

        "gray_p50":
            gray_percentiles["p50"],

        "gray_p95":
            gray_percentiles["p95"],

        "gray_p99":
            gray_percentiles["p99"],
    }

    return result


# ============================================================
# Montage
# ============================================================

def choose_sample_files(
    files,
    count,
):
    """
    Select files evenly across filename order instead of taking
    only the first N.
    """

    if len(files) <= count:
        return files

    indices = np.linspace(
        0,
        len(files) - 1,
        count,
        dtype=int,
    )

    return [
        files[index]
        for index in indices
    ]


def create_montage(
    files,
    output_path,
):

    if not files:
        return

    tiles = []

    for path in files:

        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            continue

        tile = cv2.resize(
            image,
            (
                TILE_WIDTH,
                TILE_HEIGHT,
            ),
            interpolation=cv2.INTER_AREA,
        )

        # ----------------------------------------------------
        # Filename label
        # ----------------------------------------------------

        cv2.rectangle(
            tile,
            (0, 0),
            (TILE_WIDTH, 26),
            (0, 0, 0),
            thickness=-1,
        )

        cv2.putText(
            tile,
            path.name,
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        tiles.append(
            tile
        )

    if not tiles:
        return

    columns = MONTAGE_COLUMNS

    rows = math.ceil(
        len(tiles)
        / columns
    )

    blank = np.zeros(
        (
            TILE_HEIGHT,
            TILE_WIDTH,
            3,
        ),
        dtype=np.uint8,
    )

    while len(tiles) < rows * columns:

        tiles.append(
            blank.copy()
        )

    row_images = []

    for row_index in range(rows):

        start = (
            row_index
            * columns
        )

        row = np.hstack(
            tiles[
                start:
                start + columns
            ]
        )

        row_images.append(
            row
        )

    montage = np.vstack(
        row_images
    )

    cv2.imwrite(
        str(output_path),
        montage,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95,
        ],
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 90)
    print("AIC2026 - JPG Depth Anomaly Analysis")
    print("=" * 90)

    if not DATASET_ROOT.is_dir():

        raise FileNotFoundError(
            "Dataset root not found:\n"
            f"{DATASET_ROOT}"
        )

    depth_dir = (
        find_depth_dir()
    )

    jpg_files = sorted(
        [
            path
            for path in depth_dir.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in {
                    ".jpg",
                    ".jpeg",
                }
            )
        ],
        key=lambda path: path.name,
    )

    if not jpg_files:

        raise RuntimeError(
            "Depth directory contains no JPG images:\n"
            f"{depth_dir}"
        )

    if OUTPUT_DIR.exists():

        raise FileExistsError(
            "Output directory already exists:\n"
            f"  {OUTPUT_DIR}\n\n"
            "If you explicitly want to rerun:\n"
            "  rm -rf runs/analyze_depth_jpg_anomalies"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Depth dir    : {depth_dir}"
    )

    print(
        f"JPG count    : {len(jpg_files)}"
    )

    print(
        f"Expected     : {EXPECTED_JPG_COUNT}"
    )

    print("=" * 90)

    # ========================================================
    # Global accumulators
    # ========================================================

    rows = []

    anomalies = []

    shape_counter = Counter()
    dtype_counter = Counter()

    global_pixel_count = 0

    global_exact_equal = 0

    global_diff_le_1 = 0
    global_diff_le_2 = 0
    global_diff_le_5 = 0
    global_diff_le_10 = 0

    global_sum_bg = 0
    global_sum_br = 0
    global_sum_gr = 0

    global_max_bg = 0
    global_max_br = 0
    global_max_gr = 0

    # Global gray histogram
    gray_histogram = np.zeros(
        256,
        dtype=np.int64,
    )

    # ========================================================
    # Analyze
    # ========================================================

    for index, path in enumerate(
        jpg_files,
        start=1,
    ):

        image = cv2.imread(
            str(path),
            cv2.IMREAD_UNCHANGED,
        )

        if image is None:

            anomalies.append(
                {
                    "file": str(
                        path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "error": "read_failed",
                }
            )

            continue

        shape_counter[
            str(
                tuple(
                    int(x)
                    for x in image.shape
                )
            )
        ] += 1

        dtype_counter[
            str(
                image.dtype
            )
        ] += 1

        result = analyze_image(
            path
        )

        if result[
            "status"
        ] != "ok":

            anomalies.append(
                {
                    "file": str(
                        path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "error":
                        result["status"],
                    "shape":
                        result.get(
                            "shape"
                        ),
                    "dtype":
                        result.get(
                            "dtype"
                        ),
                }
            )

            continue

        # ----------------------------------------------------
        # Need original image again for exact global counts
        # ----------------------------------------------------

        b = image[
            :, :, 0
        ].astype(
            np.int16
        )

        g = image[
            :, :, 1
        ].astype(
            np.int16
        )

        r = image[
            :, :, 2
        ].astype(
            np.int16
        )

        diff_bg = np.abs(
            b - g
        )

        diff_br = np.abs(
            b - r
        )

        diff_gr = np.abs(
            g - r
        )

        max_diff = np.maximum(
            np.maximum(
                diff_bg,
                diff_br,
            ),
            diff_gr,
        )

        pixels = int(
            b.size
        )

        global_pixel_count += pixels

        global_exact_equal += int(
            (
                (b == g)
                & (b == r)
            ).sum()
        )

        global_diff_le_1 += int(
            (
                max_diff <= 1
            ).sum()
        )

        global_diff_le_2 += int(
            (
                max_diff <= 2
            ).sum()
        )

        global_diff_le_5 += int(
            (
                max_diff <= 5
            ).sum()
        )

        global_diff_le_10 += int(
            (
                max_diff <= 10
            ).sum()
        )

        global_sum_bg += int(
            diff_bg.sum()
        )

        global_sum_br += int(
            diff_br.sum()
        )

        global_sum_gr += int(
            diff_gr.sum()
        )

        global_max_bg = max(
            global_max_bg,
            int(
                diff_bg.max()
            ),
        )

        global_max_br = max(
            global_max_br,
            int(
                diff_br.max()
            ),
        )

        global_max_gr = max(
            global_max_gr,
            int(
                diff_gr.max()
            ),
        )

        # ----------------------------------------------------
        # Global grayscale histogram
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        gray_histogram += np.bincount(
            gray.reshape(-1),
            minlength=256,
        )

        # ----------------------------------------------------
        # Store CSV row
        # ----------------------------------------------------

        row = {
            "file": str(
                path.relative_to(
                    PROJECT_ROOT
                )
            ),

            **result,
        }

        rows.append(
            row
        )

        if (
            index == 1
            or index % 25 == 0
            or index == len(jpg_files)
        ):

            print(
                f"[{index:3d}/{len(jpg_files)}] "
                f"processed"
            )

    # ========================================================
    # Global gray percentiles
    # ========================================================

    gray_values = np.arange(
        256,
        dtype=np.int64,
    )

    gray_count = int(
        gray_histogram.sum()
    )

    cumulative = np.cumsum(
        gray_histogram
    )

    def gray_percentile(p):

        if gray_count == 0:
            return None

        rank = int(
            np.floor(
                p
                / 100.0
                * (gray_count - 1)
            )
        )

        index = int(
            np.searchsorted(
                cumulative,
                rank + 1,
                side="left",
            )
        )

        return int(
            gray_values[index]
        )

    gray_global = {
        "min":
            int(
                np.flatnonzero(
                    gray_histogram
                )[0]
            )
            if gray_count > 0
            else None,

        "max":
            int(
                np.flatnonzero(
                    gray_histogram
                )[-1]
            )
            if gray_count > 0
            else None,

        "p1":
            gray_percentile(1),

        "p5":
            gray_percentile(5),

        "p50":
            gray_percentile(50),

        "p95":
            gray_percentile(95),

        "p99":
            gray_percentile(99),
    }

    # ========================================================
    # Determine overall interpretation
    # ========================================================

    exact_ratio = ratio(
        global_exact_equal,
        global_pixel_count,
    )

    le1_ratio = ratio(
        global_diff_le_1,
        global_pixel_count,
    )

    le2_ratio = ratio(
        global_diff_le_2,
        global_pixel_count,
    )

    le5_ratio = ratio(
        global_diff_le_5,
        global_pixel_count,
    )

    le10_ratio = ratio(
        global_diff_le_10,
        global_pixel_count,
    )

    mean_bg = ratio(
        global_sum_bg,
        global_pixel_count,
    )

    mean_br = ratio(
        global_sum_br,
        global_pixel_count,
    )

    mean_gr = ratio(
        global_sum_gr,
        global_pixel_count,
    )

    # Conservative automatic classification.
    if exact_ratio >= 0.99:

        interpretation = (
            "channels_are_effectively_identical"
        )

    elif le2_ratio >= 0.99:

        interpretation = (
            "channels_are_nearly_identical_likely_jpeg_gray"
        )

    elif le5_ratio >= 0.99:

        interpretation = (
            "channels_are_highly_similar_likely_compressed_gray"
        )

    else:

        interpretation = (
            "channels_have_meaningful_color_difference"
        )

    # ========================================================
    # Suspicious files
    #
    # Flag images where more than 5% pixels have channel
    # difference > 5.
    # ========================================================

    suspicious_rows = []

    for row in rows:

        if (
            row["diff_le_5_ratio"]
            < 0.95
        ):

            suspicious_rows.append(
                row
            )

    # ========================================================
    # Summary JSON
    # ========================================================

    summary = {
        "jpg_files_found":
            len(jpg_files),

        "expected_jpg_files":
            EXPECTED_JPG_COUNT,

        "successfully_analyzed":
            len(rows),

        "format_anomalies":
            len(anomalies),

        "shape_distribution":
            dict(
                sorted(
                    shape_counter.items()
                )
            ),

        "dtype_distribution":
            dict(
                sorted(
                    dtype_counter.items()
                )
            ),

        "global_pixels":
            global_pixel_count,

        "channel_similarity": {

            "exact_bgr_equal_ratio":
                exact_ratio,

            "max_channel_diff_le_1_ratio":
                le1_ratio,

            "max_channel_diff_le_2_ratio":
                le2_ratio,

            "max_channel_diff_le_5_ratio":
                le5_ratio,

            "max_channel_diff_le_10_ratio":
                le10_ratio,

            "mean_abs_b_minus_g":
                mean_bg,

            "mean_abs_b_minus_r":
                mean_br,

            "mean_abs_g_minus_r":
                mean_gr,

            "max_abs_b_minus_g":
                global_max_bg,

            "max_abs_b_minus_r":
                global_max_br,

            "max_abs_g_minus_r":
                global_max_gr,
        },

        "gray_distribution":
            gray_global,

        "automatic_interpretation":
            interpretation,

        "suspicious_file_count":
            len(
                suspicious_rows
            ),

        "format_anomalies_detail":
            anomalies,
    }

    summary_path = (
        OUTPUT_DIR
        / "jpg_depth_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Per-image CSV
    # ========================================================

    csv_path = (
        OUTPUT_DIR
        / "jpg_depth_file_stats.csv"
    )

    fieldnames = [
        "file",
        "status",
        "shape",
        "dtype",
        "pixel_count",

        "b_min",
        "b_max",
        "b_mean",
        "b_std",

        "g_min",
        "g_max",
        "g_mean",
        "g_std",

        "r_min",
        "r_max",
        "r_mean",
        "r_std",

        "mean_abs_bg",
        "mean_abs_br",
        "mean_abs_gr",

        "max_abs_bg",
        "max_abs_br",
        "max_abs_gr",

        "mean_max_channel_diff",
        "max_channel_diff",

        "exact_equal_ratio",
        "diff_le_1_ratio",
        "diff_le_2_ratio",
        "diff_le_5_ratio",
        "diff_le_10_ratio",

        "gray_min",
        "gray_max",
        "gray_mean",
        "gray_std",

        "gray_p1",
        "gray_p5",
        "gray_p50",
        "gray_p95",
        "gray_p99",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                {
                    key:
                        row.get(
                            key,
                            "",
                        )
                    for key
                    in fieldnames
                }
            )

    # ========================================================
    # Suspicious report
    # ========================================================

    suspicious_path = (
        OUTPUT_DIR
        / "suspicious_files.txt"
    )

    with suspicious_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        if not suspicious_rows:

            f.write(
                "No suspicious JPG Depth files detected.\n"
            )

        else:

            for row in suspicious_rows:

                f.write(
                    f"{row['file']}\n"
                )

                f.write(
                    "  exact_equal_ratio="
                    f"{row['exact_equal_ratio']:.6f}\n"
                )

                f.write(
                    "  diff_le_5_ratio="
                    f"{row['diff_le_5_ratio']:.6f}\n"
                )

                f.write(
                    "  mean_max_channel_diff="
                    f"{row['mean_max_channel_diff']:.6f}\n"
                )

                f.write(
                    "  max_channel_diff="
                    f"{row['max_channel_diff']}\n"
                )

                f.write(
                    "\n"
                )

    # ========================================================
    # Montage
    # ========================================================

    sample_files = (
        choose_sample_files(
            jpg_files,
            SAMPLE_COUNT,
        )
    )

    montage_path = (
        OUTPUT_DIR
        / "jpg_depth_samples.jpg"
    )

    create_montage(
        sample_files,
        montage_path,
    )

    # ========================================================
    # Terminal output
    # ========================================================

    print()
    print("=" * 90)
    print("JPG DEPTH SUMMARY")
    print("=" * 90)

    print(
        f"JPG files found           : "
        f"{len(jpg_files)}"
    )

    print(
        f"Successfully analyzed     : "
        f"{len(rows)}"
    )

    print(
        f"Format anomalies          : "
        f"{len(anomalies)}"
    )

    print()

    print(
        "Shape distribution:"
    )

    for key, value in sorted(
        shape_counter.items()
    ):

        print(
            f"  {key:25s}: {value}"
        )

    print()

    print(
        "Dtype distribution:"
    )

    for key, value in sorted(
        dtype_counter.items()
    ):

        print(
            f"  {key:15s}: {value}"
        )

    print()

    print("-" * 90)
    print("CHANNEL SIMILARITY")
    print("-" * 90)

    print(
        f"B == G == R                 : "
        f"{exact_ratio:.6%}"
    )

    print(
        f"max channel diff <= 1      : "
        f"{le1_ratio:.6%}"
    )

    print(
        f"max channel diff <= 2      : "
        f"{le2_ratio:.6%}"
    )

    print(
        f"max channel diff <= 5      : "
        f"{le5_ratio:.6%}"
    )

    print(
        f"max channel diff <= 10     : "
        f"{le10_ratio:.6%}"
    )

    print()

    print(
        f"Mean |B-G|                 : "
        f"{mean_bg:.6f}"
    )

    print(
        f"Mean |B-R|                 : "
        f"{mean_br:.6f}"
    )

    print(
        f"Mean |G-R|                 : "
        f"{mean_gr:.6f}"
    )

    print()

    print(
        f"Max |B-G|                  : "
        f"{global_max_bg}"
    )

    print(
        f"Max |B-R|                  : "
        f"{global_max_br}"
    )

    print(
        f"Max |G-R|                  : "
        f"{global_max_gr}"
    )

    print()

    print("-" * 90)
    print("GRAYSCALE DISTRIBUTION")
    print("-" * 90)

    print(
        f"Min                         : "
        f"{gray_global['min']}"
    )

    print(
        f"Max                         : "
        f"{gray_global['max']}"
    )

    print(
        f"P1                          : "
        f"{gray_global['p1']}"
    )

    print(
        f"P5                          : "
        f"{gray_global['p5']}"
    )

    print(
        f"P50                         : "
        f"{gray_global['p50']}"
    )

    print(
        f"P95                         : "
        f"{gray_global['p95']}"
    )

    print(
        f"P99                         : "
        f"{gray_global['p99']}"
    )

    print()

    print("-" * 90)
    print("AUTOMATIC INTERPRETATION")
    print("-" * 90)

    print(
        interpretation
    )

    print()

    print(
        f"Suspicious files            : "
        f"{len(suspicious_rows)}"
    )

    print()

    print("-" * 90)
    print("OUTPUT")
    print("-" * 90)

    print(
        f"Summary JSON : {summary_path}"
    )

    print(
        f"Per-file CSV : {csv_path}"
    )

    print(
        f"Samples      : {montage_path}"
    )

    print(
        f"Suspicious   : {suspicious_path}"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()

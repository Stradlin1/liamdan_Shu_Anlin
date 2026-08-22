#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 - Depth Dataset Analysis

用途：
    在构建 depth8_v1 之前，对官方 Depth 数据进行完整统计。

统计内容：
1. 文件数量
2. 图像 shape / dtype / channel 分布
3. 全局 min / max
4. 0 值比例
5. <300 mm 比例
6. 0<depth<300 mm 比例
7. 300~20000 mm 有效深度比例
8. >20000 mm 比例
9. 全局 P1 / P5 / P50 / P95 / P99
10. 有效深度 P1 / P5 / P50 / P95 / P99
11. 每张图：
        min / max
        zero ratio
        valid ratio
        各分位数
12. 检查是否存在：
        非 uint16
        非单通道
        无法读取
        尺寸异常

输出：
    runs/analyze_depth_dataset/
        depth_dataset_summary.json
        depth_file_stats.csv
        depth_histogram.csv
        anomalies.txt

说明：
    - 不修改任何原始数据
    - 分位数通过 uint16 全局直方图精确计算
    - 不把所有图片一次性加载进内存
"""

from pathlib import Path
from collections import Counter
import csv
import json
import time

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
    / "analyze_depth_dataset"
)


# ============================================================
# Depth definition
# ============================================================

DEPTH_MIN_MM = 300
DEPTH_MAX_MM = 20000

EXPECTED_IMAGES = 2000

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


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
            "检测到多个可能的 Depth 目录：\n"
            + "\n".join(
                f"  {path}"
                for path in unique
            )
        )

    children = []

    if DATASET_ROOT.is_dir():
        children = sorted(
            path.name
            for path in DATASET_ROOT.iterdir()
            if path.is_dir()
        )

    raise FileNotFoundError(
        "没有找到 Depth 数据目录。\n\n"
        f"Dataset root:\n"
        f"  {DATASET_ROOT}\n\n"
        "尝试过：\n"
        + "\n".join(
            f"  {path}"
            for path in candidates
        )
        + "\n\n"
        "当前 dataset 下目录：\n"
        + (
            "\n".join(
                f"  {name}"
                for name in children
            )
            if children
            else "  <none>"
        )
    )


# ============================================================
# Histogram utilities
# ============================================================

def percentile_from_histogram(
    histogram,
    percentile,
    start_value=0,
    end_value=65535,
):
    """
    Calculate exact percentile from integer histogram.

    histogram[value] = number of pixels with that depth value.
    """

    section = histogram[
        start_value:end_value + 1
    ]

    total = int(
        section.sum()
    )

    if total == 0:
        return None

    target = (
        percentile
        / 100.0
        * (total - 1)
    )

    target_rank = int(
        np.floor(target)
    )

    cumulative = np.cumsum(
        section,
        dtype=np.int64,
    )

    index = int(
        np.searchsorted(
            cumulative,
            target_rank + 1,
            side="left",
        )
    )

    return int(
        start_value + index
    )


def percentiles_from_histogram(
    histogram,
    start_value=0,
    end_value=65535,
):

    return {
        "p1": percentile_from_histogram(
            histogram,
            1,
            start_value,
            end_value,
        ),

        "p5": percentile_from_histogram(
            histogram,
            5,
            start_value,
            end_value,
        ),

        "p50": percentile_from_histogram(
            histogram,
            50,
            start_value,
            end_value,
        ),

        "p95": percentile_from_histogram(
            histogram,
            95,
            start_value,
            end_value,
        ),

        "p99": percentile_from_histogram(
            histogram,
            99,
            start_value,
            end_value,
        ),
    }


def first_nonzero_bin(histogram):

    indices = np.flatnonzero(
        histogram
    )

    if len(indices) == 0:
        return None

    return int(
        indices[0]
    )


def last_nonzero_bin(histogram):

    indices = np.flatnonzero(
        histogram
    )

    if len(indices) == 0:
        return None

    return int(
        indices[-1]
    )


def ratio(
    numerator,
    denominator,
):

    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator
    )


# ============================================================
# Per-image analysis
# ============================================================

def analyze_single_image(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    if image is None:
        return {
            "error": "cv2.imread failed"
        }

    original_shape = tuple(
        int(x)
        for x in image.shape
    )

    dtype_name = str(
        image.dtype
    )

    # --------------------------------------------------------
    # Determine channel structure
    # --------------------------------------------------------

    if image.ndim == 2:

        channels = 1
        depth = image

    elif (
        image.ndim == 3
        and image.shape[2] == 1
    ):

        channels = 1
        depth = image[:, :, 0]

    else:

        if image.ndim == 3:
            channels = int(
                image.shape[2]
            )
        else:
            channels = -1

        return {
            "error": "not_single_channel",
            "shape": original_shape,
            "dtype": dtype_name,
            "channels": channels,
        }

    # --------------------------------------------------------
    # We want exact uint16 depth statistics.
    #
    # Do NOT silently reinterpret uint8 / float images as
    # official 16-bit depth.
    # --------------------------------------------------------

    if depth.dtype != np.uint16:

        return {
            "error": "not_uint16",
            "shape": original_shape,
            "dtype": dtype_name,
            "channels": channels,
        }

    flat = depth.reshape(-1)

    pixel_count = int(
        flat.size
    )

    # Exact 0~65535 histogram.
    hist = np.bincount(
        flat,
        minlength=65536,
    ).astype(
        np.int64,
        copy=False,
    )

    zero_count = int(
        hist[0]
    )

    lt_300_count = int(
        hist[:DEPTH_MIN_MM].sum()
    )

    nonzero_lt_300_count = int(
        hist[1:DEPTH_MIN_MM].sum()
    )

    valid_count = int(
        hist[
            DEPTH_MIN_MM:
            DEPTH_MAX_MM + 1
        ].sum()
    )

    gt_20000_count = int(
        hist[
            DEPTH_MAX_MM + 1:
        ].sum()
    )

    raw_min = first_nonzero_bin(
        hist
    )

    raw_max = last_nonzero_bin(
        hist
    )

    # raw_min above excludes zero by construction.
    # Actual minimum including zero:
    if zero_count > 0:
        actual_min = 0
    else:
        actual_min = raw_min

    actual_max = raw_max

    all_percentiles = (
        percentiles_from_histogram(
            hist,
            0,
            65535,
        )
    )

    valid_percentiles = (
        percentiles_from_histogram(
            hist,
            DEPTH_MIN_MM,
            DEPTH_MAX_MM,
        )
    )

    valid_hist = hist[
        DEPTH_MIN_MM:
        DEPTH_MAX_MM + 1
    ]

    valid_indices = np.flatnonzero(
        valid_hist
    )

    if len(valid_indices) > 0:

        valid_min = int(
            valid_indices[0]
            + DEPTH_MIN_MM
        )

        valid_max = int(
            valid_indices[-1]
            + DEPTH_MIN_MM
        )

    else:

        valid_min = None
        valid_max = None

    return {
        "error": None,

        "shape": original_shape,
        "dtype": dtype_name,
        "channels": channels,

        "pixel_count": pixel_count,

        "min_all": actual_min,
        "max_all": actual_max,

        "min_valid": valid_min,
        "max_valid": valid_max,

        "zero_count": zero_count,
        "lt_300_count": lt_300_count,
        "nonzero_lt_300_count":
            nonzero_lt_300_count,
        "valid_count": valid_count,
        "gt_20000_count":
            gt_20000_count,

        "zero_ratio": ratio(
            zero_count,
            pixel_count,
        ),

        "lt_300_ratio": ratio(
            lt_300_count,
            pixel_count,
        ),

        "nonzero_lt_300_ratio": ratio(
            nonzero_lt_300_count,
            pixel_count,
        ),

        "valid_ratio": ratio(
            valid_count,
            pixel_count,
        ),

        "gt_20000_ratio": ratio(
            gt_20000_count,
            pixel_count,
        ),

        "p1_all":
            all_percentiles["p1"],
        "p5_all":
            all_percentiles["p5"],
        "p50_all":
            all_percentiles["p50"],
        "p95_all":
            all_percentiles["p95"],
        "p99_all":
            all_percentiles["p99"],

        "p1_valid":
            valid_percentiles["p1"],
        "p5_valid":
            valid_percentiles["p5"],
        "p50_valid":
            valid_percentiles["p50"],
        "p95_valid":
            valid_percentiles["p95"],
        "p99_valid":
            valid_percentiles["p99"],

        "histogram": hist,
    }


# ============================================================
# Main
# ============================================================

def main():

    start_time = time.time()

    print("=" * 88)
    print("AIC2026 Depth Dataset Analysis")
    print("=" * 88)

    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            "Dataset root does not exist:\n"
            f"{DATASET_ROOT}"
        )

    depth_dir = find_depth_dir()

    image_files = sorted(
        path
        for path in depth_dir.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )

    if not image_files:
        raise RuntimeError(
            "Depth directory contains no images:\n"
            f"{depth_dir}"
        )

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            "Analysis output directory already exists:\n"
            f"  {OUTPUT_DIR}\n\n"
            "If you explicitly want to rerun the analysis:\n"
            "  rm -rf runs/analyze_depth_dataset"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Dataset root : {DATASET_ROOT}"
    )

    print(
        f"Depth dir    : {depth_dir}"
    )

    print(
        f"Image files  : {len(image_files)}"
    )

    print(
        f"Expected     : {EXPECTED_IMAGES}"
    )

    print(
        f"Valid range  : "
        f"[{DEPTH_MIN_MM}, {DEPTH_MAX_MM}] mm"
    )

    print("=" * 88)

    if len(image_files) != EXPECTED_IMAGES:

        print(
            "WARNING: Depth image count is not "
            f"{EXPECTED_IMAGES}."
        )

    # ========================================================
    # Global containers
    # ========================================================

    global_hist = np.zeros(
        65536,
        dtype=np.int64,
    )

    dtype_counter = Counter()
    shape_counter = Counter()
    channel_counter = Counter()
    extension_counter = Counter()

    anomalies = []

    rows = []

    valid_image_count = 0

    # ========================================================
    # Analyze all files
    # ========================================================

    for index, path in enumerate(
        image_files,
        start=1,
    ):

        result = analyze_single_image(
            path
        )

        extension_counter[
            path.suffix.lower()
        ] += 1

        if result.get("dtype") is not None:

            dtype_counter[
                result["dtype"]
            ] += 1

        if result.get("shape") is not None:

            shape_counter[
                str(
                    result["shape"]
                )
            ] += 1

        if result.get("channels") is not None:

            channel_counter[
                str(
                    result["channels"]
                )
            ] += 1

        error = result.get(
            "error"
        )

        if error is not None:

            anomalies.append(
                {
                    "file":
                        str(
                            path.relative_to(
                                PROJECT_ROOT
                            )
                        ),
                    "error":
                        error,
                    "shape":
                        result.get("shape"),
                    "dtype":
                        result.get("dtype"),
                    "channels":
                        result.get("channels"),
                }
            )

            rows.append(
                {
                    "file":
                        str(
                            path.relative_to(
                                PROJECT_ROOT
                            )
                        ),

                    "status": error,

                    "shape":
                        result.get("shape"),

                    "dtype":
                        result.get("dtype"),

                    "channels":
                        result.get("channels"),
                }
            )

        else:

            valid_image_count += 1

            global_hist += result[
                "histogram"
            ]

            rows.append(
                {
                    "file":
                        str(
                            path.relative_to(
                                PROJECT_ROOT
                            )
                        ),

                    "status": "ok",

                    "shape":
                        result["shape"],

                    "dtype":
                        result["dtype"],

                    "channels":
                        result["channels"],

                    "pixel_count":
                        result["pixel_count"],

                    "min_all":
                        result["min_all"],

                    "max_all":
                        result["max_all"],

                    "min_valid":
                        result["min_valid"],

                    "max_valid":
                        result["max_valid"],

                    "zero_ratio":
                        result["zero_ratio"],

                    "lt_300_ratio":
                        result["lt_300_ratio"],

                    "nonzero_lt_300_ratio":
                        result[
                            "nonzero_lt_300_ratio"
                        ],

                    "valid_ratio":
                        result["valid_ratio"],

                    "gt_20000_ratio":
                        result[
                            "gt_20000_ratio"
                        ],

                    "p1_all":
                        result["p1_all"],

                    "p5_all":
                        result["p5_all"],

                    "p50_all":
                        result["p50_all"],

                    "p95_all":
                        result["p95_all"],

                    "p99_all":
                        result["p99_all"],

                    "p1_valid":
                        result["p1_valid"],

                    "p5_valid":
                        result["p5_valid"],

                    "p50_valid":
                        result["p50_valid"],

                    "p95_valid":
                        result["p95_valid"],

                    "p99_valid":
                        result["p99_valid"],
                }
            )

        if (
            index == 1
            or index % 100 == 0
            or index == len(image_files)
        ):

            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"[{index:4d}/{len(image_files)}] "
                f"valid={valid_image_count:4d} "
                f"anomalies={len(anomalies):3d} "
                f"elapsed={elapsed:.1f}s"
            )

    # ========================================================
    # Global statistics
    # ========================================================

    total_pixels = int(
        global_hist.sum()
    )

    zero_count = int(
        global_hist[0]
    )

    lt_300_count = int(
        global_hist[
            :DEPTH_MIN_MM
        ].sum()
    )

    nonzero_lt_300_count = int(
        global_hist[
            1:DEPTH_MIN_MM
        ].sum()
    )

    valid_count = int(
        global_hist[
            DEPTH_MIN_MM:
            DEPTH_MAX_MM + 1
        ].sum()
    )

    gt_20000_count = int(
        global_hist[
            DEPTH_MAX_MM + 1:
        ].sum()
    )

    global_min = (
        first_nonzero_bin(
            global_hist
        )
    )

    if zero_count > 0:
        global_min_all = 0
    else:
        global_min_all = global_min

    global_max_all = (
        last_nonzero_bin(
            global_hist
        )
    )

    # Valid min / max
    valid_hist = global_hist[
        DEPTH_MIN_MM:
        DEPTH_MAX_MM + 1
    ]

    valid_indices = np.flatnonzero(
        valid_hist
    )

    if len(valid_indices) > 0:

        global_valid_min = int(
            valid_indices[0]
            + DEPTH_MIN_MM
        )

        global_valid_max = int(
            valid_indices[-1]
            + DEPTH_MIN_MM
        )

    else:

        global_valid_min = None
        global_valid_max = None

    all_percentiles = (
        percentiles_from_histogram(
            global_hist,
            0,
            65535,
        )
    )

    valid_percentiles = (
        percentiles_from_histogram(
            global_hist,
            DEPTH_MIN_MM,
            DEPTH_MAX_MM,
        )
    )

    elapsed_seconds = (
        time.time()
        - start_time
    )

    # ========================================================
    # Summary JSON
    # ========================================================

    summary = {
        "project_root":
            str(PROJECT_ROOT),

        "dataset_root":
            str(
                DATASET_ROOT.relative_to(
                    PROJECT_ROOT
                )
            ),

        "depth_directory":
            str(
                depth_dir.relative_to(
                    PROJECT_ROOT
                )
            ),

        "expected_images":
            EXPECTED_IMAGES,

        "found_images":
            len(image_files),

        "valid_uint16_single_channel_images":
            valid_image_count,

        "anomaly_images":
            len(anomalies),

        "depth_definition_mm": {
            "minimum_valid":
                DEPTH_MIN_MM,

            "maximum_valid":
                DEPTH_MAX_MM,
        },

        "file_extensions":
            dict(
                sorted(
                    extension_counter.items()
                )
            ),

        "dtype_distribution":
            dict(
                sorted(
                    dtype_counter.items()
                )
            ),

        "shape_distribution":
            dict(
                sorted(
                    shape_counter.items()
                )
            ),

        "channel_distribution":
            dict(
                sorted(
                    channel_counter.items()
                )
            ),

        "global": {
            "total_pixels":
                total_pixels,

            "min_all_mm":
                global_min_all,

            "max_all_mm":
                global_max_all,

            "min_valid_mm":
                global_valid_min,

            "max_valid_mm":
                global_valid_max,

            "zero_pixels":
                zero_count,

            "zero_ratio":
                ratio(
                    zero_count,
                    total_pixels,
                ),

            "lt_300_pixels":
                lt_300_count,

            "lt_300_ratio":
                ratio(
                    lt_300_count,
                    total_pixels,
                ),

            "nonzero_lt_300_pixels":
                nonzero_lt_300_count,

            "nonzero_lt_300_ratio":
                ratio(
                    nonzero_lt_300_count,
                    total_pixels,
                ),

            "valid_300_20000_pixels":
                valid_count,

            "valid_300_20000_ratio":
                ratio(
                    valid_count,
                    total_pixels,
                ),

            "gt_20000_pixels":
                gt_20000_count,

            "gt_20000_ratio":
                ratio(
                    gt_20000_count,
                    total_pixels,
                ),

            "percentiles_all_mm":
                all_percentiles,

            "percentiles_valid_300_20000_mm":
                valid_percentiles,
        },

        "elapsed_seconds":
            elapsed_seconds,

        "anomalies":
            anomalies,
    }

    summary_path = (
        OUTPUT_DIR
        / "depth_dataset_summary.json"
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
    # Per-file CSV
    # ========================================================

    csv_path = (
        OUTPUT_DIR
        / "depth_file_stats.csv"
    )

    fieldnames = [
        "file",
        "status",
        "shape",
        "dtype",
        "channels",
        "pixel_count",
        "min_all",
        "max_all",
        "min_valid",
        "max_valid",
        "zero_ratio",
        "lt_300_ratio",
        "nonzero_lt_300_ratio",
        "valid_ratio",
        "gt_20000_ratio",
        "p1_all",
        "p5_all",
        "p50_all",
        "p95_all",
        "p99_all",
        "p1_valid",
        "p5_valid",
        "p50_valid",
        "p95_valid",
        "p99_valid",
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

            output_row = {
                key: row.get(
                    key,
                    "",
                )
                for key in fieldnames
            }

            writer.writerow(
                output_row
            )

    # ========================================================
    # Histogram CSV
    # ========================================================

    histogram_path = (
        OUTPUT_DIR
        / "depth_histogram.csv"
    )

    with histogram_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "depth_mm",
                "pixel_count",
                "ratio",
            ]
        )

        for depth_value, count in enumerate(
            global_hist
        ):

            writer.writerow(
                [
                    depth_value,
                    int(count),
                    ratio(
                        int(count),
                        total_pixels,
                    ),
                ]
            )

    # ========================================================
    # Anomaly report
    # ========================================================

    anomaly_path = (
        OUTPUT_DIR
        / "anomalies.txt"
    )

    with anomaly_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        if not anomalies:

            f.write(
                "No anomalies detected.\n"
            )

        else:

            for item in anomalies:

                f.write(
                    json.dumps(
                        item,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    # ========================================================
    # Terminal summary
    # ========================================================

    print()
    print("=" * 88)
    print("DEPTH DATASET SUMMARY")
    print("=" * 88)

    print(
        f"Images found               : "
        f"{len(image_files)}"
    )

    print(
        f"Valid uint16 single-channel: "
        f"{valid_image_count}"
    )

    print(
        f"Anomalies                  : "
        f"{len(anomalies)}"
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

    print("-" * 88)

    print(
        f"Total pixels               : "
        f"{total_pixels}"
    )

    print(
        f"Global min/max             : "
        f"{global_min_all} / "
        f"{global_max_all} mm"
    )

    print(
        f"Valid min/max              : "
        f"{global_valid_min} / "
        f"{global_valid_max} mm"
    )

    print()

    print(
        f"Zero ratio                 : "
        f"{ratio(zero_count, total_pixels):.6%}"
    )

    print(
        f"<300 mm ratio              : "
        f"{ratio(lt_300_count, total_pixels):.6%}"
    )

    print(
        f"0<depth<300 mm ratio       : "
        f"{ratio(nonzero_lt_300_count, total_pixels):.6%}"
    )

    print(
        f"300~20000 mm valid ratio   : "
        f"{ratio(valid_count, total_pixels):.6%}"
    )

    print(
        f">20000 mm ratio            : "
        f"{ratio(gt_20000_count, total_pixels):.6%}"
    )

    print()

    print(
        "All-pixel percentiles (mm):"
    )

    print(
        f"  P1  : {all_percentiles['p1']}"
    )

    print(
        f"  P5  : {all_percentiles['p5']}"
    )

    print(
        f"  P50 : {all_percentiles['p50']}"
    )

    print(
        f"  P95 : {all_percentiles['p95']}"
    )

    print(
        f"  P99 : {all_percentiles['p99']}"
    )

    print()

    print(
        "Valid-depth percentiles "
        f"[{DEPTH_MIN_MM}, "
        f"{DEPTH_MAX_MM}] mm:"
    )

    print(
        f"  P1  : {valid_percentiles['p1']}"
    )

    print(
        f"  P5  : {valid_percentiles['p5']}"
    )

    print(
        f"  P50 : {valid_percentiles['p50']}"
    )

    print(
        f"  P95 : {valid_percentiles['p95']}"
    )

    print(
        f"  P99 : {valid_percentiles['p99']}"
    )

    print()

    print(
        f"Elapsed                    : "
        f"{elapsed_seconds:.2f} s"
    )

    print("-" * 88)

    print(
        f"Summary JSON : {summary_path}"
    )

    print(
        f"Per-file CSV : {csv_path}"
    )

    print(
        f"Histogram    : {histogram_path}"
    )

    print(
        f"Anomalies    : {anomaly_path}"
    )

    print("=" * 88)

    # --------------------------------------------------------
    # Final data-integrity warning
    # --------------------------------------------------------

    if len(image_files) != EXPECTED_IMAGES:

        print()
        print(
            "WARNING: Dataset does not contain exactly "
            f"{EXPECTED_IMAGES} Depth images."
        )

    if anomalies:

        print()
        print(
            "WARNING: Some Depth files are not canonical "
            "single-channel uint16 images."
        )

        print(
            "Do NOT build depth8_v1 until these anomalies "
            "have been inspected."
        )

    elif len(image_files) == EXPECTED_IMAGES:

        print()
        print(
            "Depth format preflight: PASS"
        )


if __name__ == "__main__":
    main()

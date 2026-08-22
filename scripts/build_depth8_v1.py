#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 - Build Depth8 YOLO View

输入：
    官方 Depth：
        1851 张 uint16 单通道 PNG，1080x1920
        149 张 uint8 三通道 JPG，360x640

输出：
    yolo_views/depth8_v1/

转换规则
------------------------------------------------------------

1. uint16 metric Depth

    d < 300 mm
        -> 0 (invalid)

    d > 20000 mm
        -> 0 (invalid)

    300 <= d <= 20000 mm
        -> 反向线性映射至 [1, 255]

    300 mm
        -> 255

    20000 mm
        -> 1

    因此：
        0       = invalid
        1       = far
        255     = near

2. uint8 JPG Depth

    官方 JPG Depth 已经是灰度深度可视化，只是以
    3-channel JPEG 形式保存。

    BGR
      -> grayscale
      -> 保留原始 0~255，不重新归一化

3. 两种来源最终都保存为：
       uint8
       3-channel BGR
       lossless PNG

4. Train / Val：
       完全复用 rgb_v1 的 split
       不重新随机划分

5. Labels：
       使用相对软链接
       不复制标签

输出报告：
    runs/build_depth8_v1/
        build_summary.json
        sample_mapping.csv
        depth8_samples.jpg
"""

from pathlib import Path
from collections import Counter
import csv
import json
import os
import shutil
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

RGB_VIEW_ROOT = (
    PROJECT_ROOT
    / "yolo_views"
    / "rgb_v1"
)

DEPTH_VIEW_ROOT = (
    PROJECT_ROOT
    / "yolo_views"
    / "depth8_v1"
)

TEMP_VIEW_ROOT = (
    PROJECT_ROOT
    / "yolo_views"
    / ".depth8_v1_building"
)

RUNS_DIR = (
    PROJECT_ROOT
    / "runs"
)

REPORT_DIR = (
    RUNS_DIR
    / "build_depth8_v1"
)


# ============================================================
# Dataset constants
# ============================================================

SPLITS = (
    "train",
    "val",
)

EXPECTED_TOTAL = 2000
EXPECTED_TRAIN = 1600
EXPECTED_VAL = 400

EXPECTED_UINT16_PNG = 1851
EXPECTED_UINT8_JPG = 149

DEPTH_MIN_MM = 300
DEPTH_MAX_MM = 20000

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

PREVIEW_COUNT_PER_TYPE = 12
PREVIEW_COLUMNS = 4
PREVIEW_WIDTH = 320
PREVIEW_HEIGHT = 180


# ============================================================
# Locate official Depth directory
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

    raise FileNotFoundError(
        "没有找到 Depth 数据目录。\n\n"
        + "\n".join(
            f"  {path}"
            for path in candidates
        )
    )


# ============================================================
# Helpers
# ============================================================

def list_images(directory):

    if not directory.is_dir():
        raise FileNotFoundError(
            f"目录不存在：{directory}"
        )

    return sorted(
        [
            path
            for path in directory.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ],
        key=lambda p: p.name,
    )


def create_relative_symlink(
    source,
    destination,
):

    source = source.resolve()

    try:
        source.relative_to(
            PROJECT_ROOT.resolve()
        )
    except ValueError as exc:
        raise RuntimeError(
            "软链接目标不在项目目录内部：\n"
            f"{source}"
        ) from exc

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    relative_target = os.path.relpath(
        source,
        start=destination.parent,
    )

    destination.symlink_to(
        relative_target
    )


def build_depth_index(depth_dir):

    depth_files = sorted(
        [
            path
            for path in depth_dir.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ],
        key=lambda p: p.name,
    )

    if not depth_files:
        raise RuntimeError(
            f"Depth 目录中没有图片：{depth_dir}"
        )

    by_stem = {}
    duplicates = []

    for path in depth_files:

        stem = path.stem

        if stem in by_stem:
            duplicates.append(
                stem
            )
        else:
            by_stem[stem] = path

    if duplicates:
        raise RuntimeError(
            "Depth 数据存在重复 stem，无法安全匹配：\n"
            + "\n".join(
                sorted(
                    set(duplicates)
                )[:30]
            )
        )

    return (
        depth_files,
        by_stem,
    )


# ============================================================
# Depth conversion
# ============================================================

def convert_metric_depth(
    depth,
):
    """
    uint16 metric depth -> uint8 inverse-linear depth.

    invalid:
        d < 300
        d > 20000
        => 0

    valid:
        300 mm   -> 255
        20000 mm -> 1
    """

    if depth.dtype != np.uint16:

        raise TypeError(
            f"Expected uint16, got {depth.dtype}"
        )

    if depth.ndim != 2:

        raise ValueError(
            f"Expected single-channel depth, got {depth.shape}"
        )

    output = np.zeros(
        depth.shape,
        dtype=np.uint8,
    )

    valid = (
        (depth >= DEPTH_MIN_MM)
        & (depth <= DEPTH_MAX_MM)
    )

    if np.any(valid):

        d = depth[
            valid
        ].astype(
            np.float32
        )

        mapped = (
            1.0
            + (
                (
                    DEPTH_MAX_MM
                    - d
                )
                * 254.0
                / (
                    DEPTH_MAX_MM
                    - DEPTH_MIN_MM
                )
            )
        )

        mapped = np.rint(
            mapped
        )

        mapped = np.clip(
            mapped,
            1,
            255,
        ).astype(
            np.uint8
        )

        output[
            valid
        ] = mapped

    return output


def convert_jpg_depth(
    image,
):
    """
    Official 3-channel uint8 JPG Depth -> grayscale.

    Do NOT normalize again.
    """

    if image.dtype != np.uint8:

        raise TypeError(
            f"Expected uint8 JPG, got {image.dtype}"
        )

    if (
        image.ndim != 3
        or image.shape[2] != 3
    ):

        raise ValueError(
            "Expected 3-channel JPG Depth, "
            f"got shape {image.shape}"
        )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )


def convert_depth_file(
    source,
):
    """
    Returns:
        output_bgr
        source_type
        metadata
    """

    image = cv2.imread(
        str(source),
        cv2.IMREAD_UNCHANGED,
    )

    if image is None:

        raise RuntimeError(
            f"无法读取 Depth：{source}"
        )

    suffix = source.suffix.lower()

    # --------------------------------------------------------
    # Canonical metric PNG
    # --------------------------------------------------------

    if (
        image.dtype == np.uint16
        and image.ndim == 2
    ):

        gray = convert_metric_depth(
            image
        )

        source_type = (
            "uint16_metric"
        )

        metadata = {
            "source_dtype":
                str(image.dtype),

            "source_shape":
                list(image.shape),

            "output_min":
                int(gray.min()),

            "output_max":
                int(gray.max()),

            "invalid_ratio":
                float(
                    (
                        (
                            image < DEPTH_MIN_MM
                        )
                        | (
                            image > DEPTH_MAX_MM
                        )
                    ).mean()
                ),
        }

    # --------------------------------------------------------
    # Official uint8 JPG visualization
    # --------------------------------------------------------

    elif (
        image.dtype == np.uint8
        and image.ndim == 3
        and image.shape[2] == 3
        and suffix in {
            ".jpg",
            ".jpeg",
        }
    ):

        gray = convert_jpg_depth(
            image
        )

        source_type = (
            "uint8_jpg"
        )

        metadata = {
            "source_dtype":
                str(image.dtype),

            "source_shape":
                list(image.shape),

            "output_min":
                int(gray.min()),

            "output_max":
                int(gray.max()),

            "invalid_ratio":
                float(
                    (
                        gray == 0
                    ).mean()
                ),
        }

    else:

        raise RuntimeError(
            "发现未定义的 Depth 格式：\n"
            f"  file  : {source}\n"
            f"  dtype : {image.dtype}\n"
            f"  shape : {image.shape}\n"
            f"  suffix: {suffix}"
        )

    # Explicit 3-channel uint8 output.
    output_bgr = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR,
    )

    if output_bgr.dtype != np.uint8:

        raise RuntimeError(
            "Converted Depth is not uint8."
        )

    if (
        output_bgr.ndim != 3
        or output_bgr.shape[2] != 3
    ):

        raise RuntimeError(
            "Converted Depth is not 3-channel."
        )

    return (
        output_bgr,
        source_type,
        metadata,
    )


# ============================================================
# data.yaml
# ============================================================

def write_data_yaml(
    root,
):

    text = """train: images/train
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
"""

    (
        root
        / "data.yaml"
    ).write_text(
        text,
        encoding="utf-8",
    )


# ============================================================
# Preflight
# ============================================================

def preflight():

    print("=" * 88)
    print("AIC2026 - Build depth8_v1")
    print("=" * 88)

    if not DATASET_ROOT.is_dir():

        raise FileNotFoundError(
            f"Dataset root 不存在：{DATASET_ROOT}"
        )

    if not RGB_VIEW_ROOT.is_dir():

        raise FileNotFoundError(
            f"rgb_v1 不存在：{RGB_VIEW_ROOT}"
        )

    if DEPTH_VIEW_ROOT.exists():

        raise FileExistsError(
            "depth8_v1 已经存在：\n"
            f"  {DEPTH_VIEW_ROOT}\n\n"
            "为防止覆盖正式数据，脚本停止。\n"
            "如明确需要重新构建，请先手动删除。"
        )

    if TEMP_VIEW_ROOT.exists():

        raise FileExistsError(
            "检测到上一次未完成的临时目录：\n"
            f"  {TEMP_VIEW_ROOT}\n\n"
            "确认不需要后可执行：\n"
            "  rm -rf yolo_views/.depth8_v1_building"
        )

    if REPORT_DIR.exists():

        raise FileExistsError(
            "构建报告目录已经存在：\n"
            f"  {REPORT_DIR}\n\n"
            "如需重新构建，请先手动删除该目录。"
        )

    depth_dir = (
        find_depth_dir()
    )

    (
        depth_files,
        depth_by_stem,
    ) = build_depth_index(
        depth_dir
    )

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Depth source : {depth_dir}"
    )

    print(
        f"RGB split    : {RGB_VIEW_ROOT}"
    )

    print(
        f"Output       : {DEPTH_VIEW_ROOT}"
    )

    print()

    print(
        f"Official Depth files: {len(depth_files)}"
    )

    if len(depth_files) != EXPECTED_TOTAL:

        raise RuntimeError(
            "官方 Depth 文件总数异常：\n"
            f"expected={EXPECTED_TOTAL}\n"
            f"actual={len(depth_files)}"
        )

    records = []

    missing_depth = []
    missing_labels = []

    split_stems = {}

    for split in SPLITS:

        rgb_image_dir = (
            RGB_VIEW_ROOT
            / "images"
            / split
        )

        rgb_label_dir = (
            RGB_VIEW_ROOT
            / "labels"
            / split
        )

        rgb_images = list_images(
            rgb_image_dir
        )

        stems = set()

        for rgb_image in rgb_images:

            stem = rgb_image.stem

            if stem in stems:

                raise RuntimeError(
                    f"{split} 中出现重复 stem：{stem}"
                )

            stems.add(
                stem
            )

            depth_source = (
                depth_by_stem.get(
                    stem
                )
            )

            if depth_source is None:

                missing_depth.append(
                    (
                        split,
                        stem,
                    )
                )

                continue

            label_source = (
                rgb_label_dir
                / f"{stem}.txt"
            )

            if not label_source.exists():

                missing_labels.append(
                    (
                        split,
                        label_source,
                    )
                )

                continue

            records.append(
                {
                    "split": split,
                    "stem": stem,
                    "rgb_reference":
                        rgb_image,
                    "depth_source":
                        depth_source,
                    "label_source":
                        label_source,
                }
            )

        split_stems[
            split
        ] = stems

        print(
            f"{split:5s} samples : "
            f"{len(rgb_images)}"
        )

    if len(
        split_stems["train"]
    ) != EXPECTED_TRAIN:

        raise RuntimeError(
            "train 数量异常："
            f"{len(split_stems['train'])}"
        )

    if len(
        split_stems["val"]
    ) != EXPECTED_VAL:

        raise RuntimeError(
            "val 数量异常："
            f"{len(split_stems['val'])}"
        )

    overlap = (
        split_stems["train"]
        & split_stems["val"]
    )

    if overlap:

        raise RuntimeError(
            "train / val 存在 stem 重叠：\n"
            + "\n".join(
                sorted(overlap)[:20]
            )
        )

    if missing_depth:

        print()
        print("Missing Depth:")

        for item in missing_depth[:20]:

            print(
                " ",
                item,
            )

        raise RuntimeError(
            f"缺失 Depth 数量：{len(missing_depth)}"
        )

    if missing_labels:

        print()
        print("Missing Labels:")

        for item in missing_labels[:20]:

            print(
                " ",
                item,
            )

        raise RuntimeError(
            f"缺失 Label 数量：{len(missing_labels)}"
        )

    if len(records) != EXPECTED_TOTAL:

        raise RuntimeError(
            "最终匹配数量不是 2000："
            f"{len(records)}"
        )

    print()
    print("=" * 88)
    print("Preflight PASS")
    print("=" * 88)

    print(
        f"train             : "
        f"{len(split_stems['train'])}"
    )

    print(
        f"val               : "
        f"{len(split_stems['val'])}"
    )

    print(
        f"total             : "
        f"{len(records)}"
    )

    print(
        f"missing depth     : "
        f"{len(missing_depth)}"
    )

    print(
        f"missing labels    : "
        f"{len(missing_labels)}"
    )

    print(
        f"train/val overlap : "
        f"{len(overlap)}"
    )

    print()

    return (
        records,
        depth_dir,
    )


# ============================================================
# Build
# ============================================================

def build(
    records,
):

    for split in SPLITS:

        (
            TEMP_VIEW_ROOT
            / "images"
            / split
        ).mkdir(
            parents=True,
            exist_ok=False,
        )

        (
            TEMP_VIEW_ROOT
            / "labels"
            / split
        ).mkdir(
            parents=True,
            exist_ok=False,
        )

    counters = Counter()

    mapping_rows = []

    png_preview = []
    jpg_preview = []

    try:

        for index, record in enumerate(
            records,
            start=1,
        ):

            split = record[
                "split"
            ]

            stem = record[
                "stem"
            ]

            depth_source = record[
                "depth_source"
            ]

            (
                converted,
                source_type,
                metadata,
            ) = convert_depth_file(
                depth_source
            )

            destination_image = (
                TEMP_VIEW_ROOT
                / "images"
                / split
                / f"{stem}.png"
            )

            destination_label = (
                TEMP_VIEW_ROOT
                / "labels"
                / split
                / f"{stem}.txt"
            )

            success = cv2.imwrite(
                str(
                    destination_image
                ),
                converted,
                [
                    cv2.IMWRITE_PNG_COMPRESSION,
                    3,
                ],
            )

            if not success:

                raise RuntimeError(
                    "cv2.imwrite failed:\n"
                    f"{destination_image}"
                )

            create_relative_symlink(
                record[
                    "label_source"
                ],
                destination_label,
            )

            counters[
                split
            ] += 1

            counters[
                source_type
            ] += 1

            counters[
                f"{split}_{source_type}"
            ] += 1

            if (
                source_type
                == "uint16_metric"
                and len(
                    png_preview
                )
                < PREVIEW_COUNT_PER_TYPE
            ):

                png_preview.append(
                    {
                        "path":
                            destination_image,
                        "label":
                            "PNG metric",
                    }
                )

            if (
                source_type
                == "uint8_jpg"
                and len(
                    jpg_preview
                )
                < PREVIEW_COUNT_PER_TYPE
            ):

                jpg_preview.append(
                    {
                        "path":
                            destination_image,
                        "label":
                            "JPG official",
                    }
                )

            mapping_rows.append(
                {
                    "split":
                        split,

                    "stem":
                        stem,

                    "source":
                        str(
                            depth_source.relative_to(
                                PROJECT_ROOT
                            )
                        ),

                    "source_extension":
                        depth_source.suffix.lower(),

                    "source_type":
                        source_type,

                    "source_dtype":
                        metadata[
                            "source_dtype"
                        ],

                    "source_shape":
                        str(
                            metadata[
                                "source_shape"
                            ]
                        ),

                    "output":
                        str(
                            (
                                Path(
                                    "yolo_views"
                                )
                                / "depth8_v1"
                                / "images"
                                / split
                                / f"{stem}.png"
                            )
                        ),

                    "output_min":
                        metadata[
                            "output_min"
                        ],

                    "output_max":
                        metadata[
                            "output_max"
                        ],

                    "invalid_ratio":
                        metadata[
                            "invalid_ratio"
                        ],
                }
            )

            if (
                index == 1
                or index % 100 == 0
                or index == len(records)
            ):

                print(
                    f"[{index:4d}/{len(records)}] "
                    f"train={counters['train']:4d} "
                    f"val={counters['val']:3d} "
                    f"metric={counters['uint16_metric']:4d} "
                    f"jpg={counters['uint8_jpg']:3d}"
                )

        write_data_yaml(
            TEMP_VIEW_ROOT
        )

    except Exception:

        print()
        print(
            "Build failed."
        )

        print(
            "临时目录保留用于排查："
        )

        print(
            f"  {TEMP_VIEW_ROOT}"
        )

        raise

    return (
        counters,
        mapping_rows,
        png_preview,
        jpg_preview,
    )


# ============================================================
# Verify generated view
# ============================================================

def verify(
    counters,
):

    if counters[
        "train"
    ] != EXPECTED_TRAIN:

        raise RuntimeError(
            "Generated train count mismatch."
        )

    if counters[
        "val"
    ] != EXPECTED_VAL:

        raise RuntimeError(
            "Generated val count mismatch."
        )

    if counters[
        "uint16_metric"
    ] != EXPECTED_UINT16_PNG:

        raise RuntimeError(
            "uint16 metric count mismatch:\n"
            f"expected={EXPECTED_UINT16_PNG}\n"
            f"actual={counters['uint16_metric']}"
        )

    if counters[
        "uint8_jpg"
    ] != EXPECTED_UINT8_JPG:

        raise RuntimeError(
            "uint8 JPG count mismatch:\n"
            f"expected={EXPECTED_UINT8_JPG}\n"
            f"actual={counters['uint8_jpg']}"
        )

    broken_links = []

    for path in TEMP_VIEW_ROOT.rglob("*"):

        if (
            path.is_symlink()
            and not path.exists()
        ):

            broken_links.append(
                path
            )

    if broken_links:

        raise RuntimeError(
            "发现 broken label symlink：\n"
            + "\n".join(
                str(path)
                for path in broken_links[:20]
            )
        )

    # --------------------------------------------------------
    # Check every generated image
    # --------------------------------------------------------

    image_count = 0

    bad_outputs = []

    for split in SPLITS:

        image_dir = (
            TEMP_VIEW_ROOT
            / "images"
            / split
        )

        for path in sorted(
            image_dir.glob(
                "*.png"
            )
        ):

            image_count += 1

            image = cv2.imread(
                str(path),
                cv2.IMREAD_UNCHANGED,
            )

            if image is None:

                bad_outputs.append(
                    (
                        path,
                        "read_failed",
                    )
                )

                continue

            if (
                image.dtype != np.uint8
                or image.ndim != 3
                or image.shape[2] != 3
            ):

                bad_outputs.append(
                    (
                        path,
                        (
                            f"dtype={image.dtype}, "
                            f"shape={image.shape}"
                        ),
                    )
                )

    if image_count != EXPECTED_TOTAL:

        raise RuntimeError(
            "Generated image count mismatch:\n"
            f"{image_count}"
        )

    if bad_outputs:

        raise RuntimeError(
            "发现异常输出图像：\n"
            + "\n".join(
                f"{path}: {reason}"
                for path, reason
                in bad_outputs[:20]
            )
        )

    return len(
        broken_links
    )


# ============================================================
# Preview montage
# ============================================================

def evenly_select(
    items,
    count,
):

    if len(items) <= count:
        return items

    indices = np.linspace(
        0,
        len(items) - 1,
        count,
        dtype=int,
    )

    return [
        items[index]
        for index in indices
    ]


def make_preview(
    records,
    output_path,
):

    metric_records = [
        record
        for record in records
        if record["source_type"]
        == "uint16_metric"
    ]

    jpg_records = [
        record
        for record in records
        if record["source_type"]
        == "uint8_jpg"
    ]

    selected = (
        evenly_select(
            metric_records,
            PREVIEW_COUNT_PER_TYPE,
        )
        +
        evenly_select(
            jpg_records,
            PREVIEW_COUNT_PER_TYPE,
        )
    )

    tiles = []

    for record in selected:

        output_relative = Path(
            record["output"]
        )

        image_path = (
            PROJECT_ROOT
            / output_relative
        )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            continue

        tile = cv2.resize(
            image,
            (
                PREVIEW_WIDTH,
                PREVIEW_HEIGHT,
            ),
            interpolation=cv2.INTER_AREA,
        )

        title = (
            f"{record['source_type']} "
            f"{record['stem']}"
        )

        cv2.rectangle(
            tile,
            (0, 0),
            (
                PREVIEW_WIDTH,
                25,
            ),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            tile,
            title,
            (6, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        tiles.append(
            tile
        )

    if not tiles:
        return

    rows = math.ceil(
        len(tiles)
        / PREVIEW_COLUMNS
    )

    blank = np.zeros(
        (
            PREVIEW_HEIGHT,
            PREVIEW_WIDTH,
            3,
        ),
        dtype=np.uint8,
    )

    while (
        len(tiles)
        < rows * PREVIEW_COLUMNS
    ):

        tiles.append(
            blank.copy()
        )

    row_images = []

    for row_index in range(rows):

        start = (
            row_index
            * PREVIEW_COLUMNS
        )

        row_images.append(
            np.hstack(
                tiles[
                    start:
                    start
                    + PREVIEW_COLUMNS
                ]
            )
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
# Report
# ============================================================

def write_report(
    counters,
    mapping_rows,
):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    csv_path = (
        REPORT_DIR
        / "sample_mapping.csv"
    )

    fieldnames = [
        "split",
        "stem",
        "source",
        "source_extension",
        "source_type",
        "source_dtype",
        "source_shape",
        "output",
        "output_min",
        "output_max",
        "invalid_ratio",
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

        writer.writerows(
            mapping_rows
        )

    invalid_ratios = [
        float(
            row[
                "invalid_ratio"
            ]
        )
        for row in mapping_rows
    ]

    summary = {
        "view":
            "depth8_v1",

        "total_samples":
            len(mapping_rows),

        "train_samples":
            int(
                counters["train"]
            ),

        "val_samples":
            int(
                counters["val"]
            ),

        "source_formats": {
            "uint16_metric":
                int(
                    counters[
                        "uint16_metric"
                    ]
                ),

            "uint8_jpg":
                int(
                    counters[
                        "uint8_jpg"
                    ]
                ),
        },

        "split_source_formats": {
            "train_uint16_metric":
                int(
                    counters[
                        "train_uint16_metric"
                    ]
                ),

            "train_uint8_jpg":
                int(
                    counters[
                        "train_uint8_jpg"
                    ]
                ),

            "val_uint16_metric":
                int(
                    counters[
                        "val_uint16_metric"
                    ]
                ),

            "val_uint8_jpg":
                int(
                    counters[
                        "val_uint8_jpg"
                    ]
                ),
        },

        "metric_depth_conversion": {
            "invalid_output":
                0,

            "valid_input_min_mm":
                DEPTH_MIN_MM,

            "valid_input_max_mm":
                DEPTH_MAX_MM,

            "near_output":
                255,

            "far_output":
                1,

            "mapping":
                "inverse_linear",
        },

        "jpg_conversion": {
            "operation":
                "BGR_to_grayscale",

            "renormalized":
                False,
        },

        "final_format": {
            "dtype":
                "uint8",

            "channels":
                3,

            "file_format":
                "PNG",
        },

        "invalid_ratio_per_image": {
            "mean":
                float(
                    np.mean(
                        invalid_ratios
                    )
                ),

            "median":
                float(
                    np.median(
                        invalid_ratios
                    )
                ),

            "min":
                float(
                    np.min(
                        invalid_ratios
                    )
                ),

            "max":
                float(
                    np.max(
                        invalid_ratios
                    )
                ),
        },
    }

    json_path = (
        REPORT_DIR
        / "build_summary.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return (
        csv_path,
        json_path,
    )


# ============================================================
# Main
# ============================================================

def main():

    (
        records,
        depth_dir,
    ) = preflight()

    print("=" * 88)
    print("Building depth8_v1")
    print("=" * 88)

    (
        counters,
        mapping_rows,
        _,
        _,
    ) = build(
        records
    )

    print()
    print("=" * 88)
    print("Verifying generated data")
    print("=" * 88)

    broken_links = verify(
        counters
    )

    # --------------------------------------------------------
    # Atomic-ish finalization:
    # only expose official depth8_v1 after complete verification
    # --------------------------------------------------------

    TEMP_VIEW_ROOT.rename(
        DEPTH_VIEW_ROOT
    )

    (
        csv_path,
        json_path,
    ) = write_report(
        counters,
        mapping_rows,
    )

    preview_path = (
        REPORT_DIR
        / "depth8_samples.jpg"
    )

    make_preview(
        mapping_rows,
        preview_path,
    )

    print()
    print("=" * 88)
    print("depth8_v1 BUILT SUCCESSFULLY")
    print("=" * 88)

    print(
        f"Output view          : "
        f"{DEPTH_VIEW_ROOT}"
    )

    print()

    print(
        f"Train samples        : "
        f"{counters['train']}"
    )

    print(
        f"Val samples          : "
        f"{counters['val']}"
    )

    print(
        f"Total samples        : "
        f"{counters['train'] + counters['val']}"
    )

    print()

    print(
        f"uint16 metric Depth  : "
        f"{counters['uint16_metric']}"
    )

    print(
        f"uint8 JPG Depth      : "
        f"{counters['uint8_jpg']}"
    )

    print()

    print(
        f"train metric         : "
        f"{counters['train_uint16_metric']}"
    )

    print(
        f"train JPG            : "
        f"{counters['train_uint8_jpg']}"
    )

    print(
        f"val metric           : "
        f"{counters['val_uint16_metric']}"
    )

    print(
        f"val JPG              : "
        f"{counters['val_uint8_jpg']}"
    )

    print()

    print(
        f"Broken label links   : "
        f"{broken_links}"
    )

    print()

    print(
        f"Data YAML            : "
        f"{DEPTH_VIEW_ROOT / 'data.yaml'}"
    )

    print(
        f"Build summary        : "
        f"{json_path}"
    )

    print(
        f"Sample mapping       : "
        f"{csv_path}"
    )

    print(
        f"Preview              : "
        f"{preview_path}"
    )

    print("=" * 88)

    print()
    print("Depth representation:")
    print(
        "  metric PNG : "
        "invalid=0, near=255, far=1"
    )
    print(
        "  JPG Depth  : "
        "official grayscale retained"
    )


if __name__ == "__main__":
    main()

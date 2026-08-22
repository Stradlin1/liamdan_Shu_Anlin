#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 - Multimodal Alignment Audit

目的：
    在构建 Exp04 RGB + Infrared + Depth 三模态模型之前，
    对当前三个 YOLO view 做严格的一致性审计。

检查对象：
    yolo_views/rgb_v1
    yolo_views/ir_v1
    yolo_views/depth8_v1

检查内容：
1. train / val 数量
2. RGB / IR / Depth 的 sample stem 是否完全一致
3. train / val 是否存在交集
4. 每组三模态图像是否都能成功读取
5. 每组三模态图像宽高是否完全一致
6. dtype / channel 是否满足：
       RGB    -> uint8, 3-channel
       IR     -> uint8, 3-channel
       Depth8 -> uint8, 3-channel
7. 三个 view 的同名标签文件是否存在
8. 三个 view 的标签内容是否完全一致
9. 标签中的 class_id / xywh 是否基本合法
10. 统计：
       图像尺寸分布
       文件扩展名分布
       dtype 分布
       channel 分布
11. 输出完整 sample mapping，供后续 multimodal_dataset.py 使用参考

本脚本：
    - 不修改任何数据
    - 不重新划分 train / val
    - 不创建新的训练 view

输出：
    runs/check_multimodal_alignment/
    ├── alignment_summary.json
    ├── sample_alignment.csv
    ├── errors.txt
    └── warnings.txt
"""

from pathlib import Path
from collections import Counter
import csv
import hashlib
import json

import cv2
import numpy as np


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

YOLO_VIEWS_ROOT = (
    PROJECT_ROOT
    / "yolo_views"
)

RGB_VIEW = (
    YOLO_VIEWS_ROOT
    / "rgb_v1"
)

IR_VIEW = (
    YOLO_VIEWS_ROOT
    / "ir_v1"
)

DEPTH_VIEW = (
    YOLO_VIEWS_ROOT
    / "depth8_v1"
)

RUNS_DIR = (
    PROJECT_ROOT
    / "runs"
)

OUTPUT_DIR = (
    RUNS_DIR
    / "check_multimodal_alignment"
)


# ============================================================
# Expected dataset structure
# ============================================================

SPLITS = (
    "train",
    "val",
)

EXPECTED_COUNTS = {
    "train": 1600,
    "val": 400,
}

EXPECTED_TOTAL = 2000

NUM_CLASSES = 12

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# Basic utilities
# ============================================================

def relative(path):
    """Return project-relative path as string."""

    return str(
        path.relative_to(
            PROJECT_ROOT
        )
    )


def sha256_file(path):
    """Calculate SHA256 of file content."""

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def list_images(directory):
    """
    Return:
        dict[stem] = image_path

    Fail immediately if duplicate stems exist.
    """

    if not directory.is_dir():

        raise FileNotFoundError(
            f"Image directory not found:\n"
            f"  {directory}"
        )

    mapping = {}

    for path in sorted(
        directory.iterdir(),
        key=lambda p: p.name,
    ):

        if not path.is_file():
            continue

        if (
            path.suffix.lower()
            not in IMAGE_EXTENSIONS
        ):
            continue

        stem = path.stem

        if stem in mapping:

            raise RuntimeError(
                "Duplicate image stem detected:\n"
                f"  directory : {directory}\n"
                f"  stem      : {stem}\n"
                f"  file A    : {mapping[stem]}\n"
                f"  file B    : {path}"
            )

        mapping[
            stem
        ] = path

    return mapping


def list_labels(directory):
    """
    Return:
        dict[stem] = label_path
    """

    if not directory.is_dir():

        raise FileNotFoundError(
            f"Label directory not found:\n"
            f"  {directory}"
        )

    mapping = {}

    for path in sorted(
        directory.iterdir(),
        key=lambda p: p.name,
    ):

        if (
            not path.is_file()
            or path.suffix.lower()
            != ".txt"
        ):
            continue

        stem = path.stem

        if stem in mapping:

            raise RuntimeError(
                "Duplicate label stem detected:\n"
                f"  directory : {directory}\n"
                f"  stem      : {stem}"
            )

        mapping[
            stem
        ] = path

    return mapping


# ============================================================
# Label validation
# ============================================================

def parse_label_file(path):
    """
    Parse YOLO detection label.

    Expected line:
        class_id x_center y_center width height
    """

    objects = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            # Empty label file is valid.
            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:

                raise ValueError(
                    "Invalid label column count:\n"
                    f"  file : {path}\n"
                    f"  line : {line_number}\n"
                    f"  text : {line}"
                )

            try:

                class_id = int(
                    parts[0]
                )

                x, y, w, h = [
                    float(value)
                    for value
                    in parts[1:]
                ]

            except ValueError as exc:

                raise ValueError(
                    "Invalid numeric value in label:\n"
                    f"  file : {path}\n"
                    f"  line : {line_number}\n"
                    f"  text : {line}"
                ) from exc

            if not (
                0 <= class_id
                < NUM_CLASSES
            ):

                raise ValueError(
                    "Class id out of range:\n"
                    f"  file     : {path}\n"
                    f"  line     : {line_number}\n"
                    f"  class_id : {class_id}"
                )

            for value_name, value in [
                ("x", x),
                ("y", y),
                ("w", w),
                ("h", h),
            ]:

                if not np.isfinite(
                    value
                ):

                    raise ValueError(
                        "Non-finite label coordinate:\n"
                        f"  file  : {path}\n"
                        f"  line  : {line_number}\n"
                        f"  field : {value_name}\n"
                        f"  value : {value}"
                    )

                if not (
                    0.0 <= value
                    <= 1.0
                ):

                    raise ValueError(
                        "Normalized coordinate outside [0,1]:\n"
                        f"  file  : {path}\n"
                        f"  line  : {line_number}\n"
                        f"  field : {value_name}\n"
                        f"  value : {value}"
                    )

            if w <= 0.0 or h <= 0.0:

                raise ValueError(
                    "Bounding box has zero width/height:\n"
                    f"  file : {path}\n"
                    f"  line : {line_number}\n"
                    f"  w={w}, h={h}"
                )

            objects.append(
                (
                    class_id,
                    x,
                    y,
                    w,
                    h,
                )
            )

    return objects


# ============================================================
# Image inspection
# ============================================================

def inspect_image(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    if image is None:

        return {
            "ok": False,
            "error": "cv2.imread failed",
        }

    dtype = str(
        image.dtype
    )

    shape = tuple(
        int(x)
        for x in image.shape
    )

    if image.ndim == 2:

        height, width = (
            image.shape
        )

        channels = 1

    elif image.ndim == 3:

        height, width = (
            image.shape[:2]
        )

        channels = int(
            image.shape[2]
        )

    else:

        return {
            "ok": False,
            "error":
                f"unsupported ndim={image.ndim}",
            "dtype":
                dtype,
            "shape":
                shape,
        }

    return {
        "ok": True,

        "dtype":
            dtype,

        "shape":
            shape,

        "height":
            int(height),

        "width":
            int(width),

        "channels":
            channels,

        "extension":
            path.suffix.lower(),
    }


# ============================================================
# Directory preflight
# ============================================================

def check_view_structure():

    views = {
        "rgb": RGB_VIEW,
        "ir": IR_VIEW,
        "depth": DEPTH_VIEW,
    }

    for modality, root in views.items():

        if not root.is_dir():

            raise FileNotFoundError(
                f"{modality} view not found:\n"
                f"  {root}"
            )

        data_yaml = (
            root
            / "data.yaml"
        )

        if not data_yaml.is_file():

            raise FileNotFoundError(
                f"{modality} data.yaml not found:\n"
                f"  {data_yaml}"
            )

        for split in SPLITS:

            image_dir = (
                root
                / "images"
                / split
            )

            label_dir = (
                root
                / "labels"
                / split
            )

            if not image_dir.is_dir():

                raise FileNotFoundError(
                    f"Missing image directory:\n"
                    f"  {image_dir}"
                )

            if not label_dir.is_dir():

                raise FileNotFoundError(
                    f"Missing label directory:\n"
                    f"  {label_dir}"
                )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 92)
    print("AIC2026 - Multimodal Alignment Audit")
    print("=" * 92)

    check_view_structure()

    if OUTPUT_DIR.exists():

        raise FileExistsError(
            "Output directory already exists:\n"
            f"  {OUTPUT_DIR}\n\n"
            "If you explicitly want to rerun:\n"
            "  rm -rf runs/check_multimodal_alignment"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"RGB view     : {RGB_VIEW}"
    )

    print(
        f"IR view      : {IR_VIEW}"
    )

    print(
        f"Depth view   : {DEPTH_VIEW}"
    )

    print()

    errors = []
    warnings = []

    rows = []

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    image_count_stats = {}

    label_count_stats = {}

    shape_stats = {
        "rgb": Counter(),
        "ir": Counter(),
        "depth": Counter(),
    }

    dtype_stats = {
        "rgb": Counter(),
        "ir": Counter(),
        "depth": Counter(),
    }

    channel_stats = {
        "rgb": Counter(),
        "ir": Counter(),
        "depth": Counter(),
    }

    extension_stats = {
        "rgb": Counter(),
        "ir": Counter(),
        "depth": Counter(),
    }

    resolution_triplets = Counter()

    split_stems = {}

    total_objects = 0

    # ========================================================
    # Process each split
    # ========================================================

    for split in SPLITS:

        print("=" * 92)
        print(
            f"Checking split: {split}"
        )
        print("=" * 92)

        rgb_images = list_images(
            RGB_VIEW
            / "images"
            / split
        )

        ir_images = list_images(
            IR_VIEW
            / "images"
            / split
        )

        depth_images = list_images(
            DEPTH_VIEW
            / "images"
            / split
        )

        rgb_labels = list_labels(
            RGB_VIEW
            / "labels"
            / split
        )

        ir_labels = list_labels(
            IR_VIEW
            / "labels"
            / split
        )

        depth_labels = list_labels(
            DEPTH_VIEW
            / "labels"
            / split
        )

        # ----------------------------------------------------
        # Counts
        # ----------------------------------------------------

        image_count_stats[
            f"{split}_rgb"
        ] = len(
            rgb_images
        )

        image_count_stats[
            f"{split}_ir"
        ] = len(
            ir_images
        )

        image_count_stats[
            f"{split}_depth"
        ] = len(
            depth_images
        )

        label_count_stats[
            f"{split}_rgb"
        ] = len(
            rgb_labels
        )

        label_count_stats[
            f"{split}_ir"
        ] = len(
            ir_labels
        )

        label_count_stats[
            f"{split}_depth"
        ] = len(
            depth_labels
        )

        print(
            "Images:"
        )

        print(
            f"  RGB   : {len(rgb_images)}"
        )

        print(
            f"  IR    : {len(ir_images)}"
        )

        print(
            f"  Depth : {len(depth_images)}"
        )

        print(
            "Labels:"
        )

        print(
            f"  RGB   : {len(rgb_labels)}"
        )

        print(
            f"  IR    : {len(ir_labels)}"
        )

        print(
            f"  Depth : {len(depth_labels)}"
        )

        expected = (
            EXPECTED_COUNTS[
                split
            ]
        )

        for modality, mapping in [
            ("rgb", rgb_images),
            ("ir", ir_images),
            ("depth", depth_images),
        ]:

            if len(
                mapping
            ) != expected:

                errors.append(
                    f"[{split}] "
                    f"{modality} image count "
                    f"expected={expected}, "
                    f"actual={len(mapping)}"
                )

        # ----------------------------------------------------
        # Stem equality
        # ----------------------------------------------------

        rgb_stems = set(
            rgb_images
        )

        ir_stems = set(
            ir_images
        )

        depth_stems = set(
            depth_images
        )

        common_stems = (
            rgb_stems
            & ir_stems
            & depth_stems
        )

        union_stems = (
            rgb_stems
            | ir_stems
            | depth_stems
        )

        split_stems[
            split
        ] = union_stems

        if not (
            rgb_stems
            == ir_stems
            == depth_stems
        ):

            missing_ir = sorted(
                rgb_stems
                - ir_stems
            )

            missing_depth = sorted(
                rgb_stems
                - depth_stems
            )

            extra_ir = sorted(
                ir_stems
                - rgb_stems
            )

            extra_depth = sorted(
                depth_stems
                - rgb_stems
            )

            if missing_ir:

                errors.append(
                    f"[{split}] "
                    f"IR missing {len(missing_ir)} stems: "
                    + ", ".join(
                        missing_ir[:20]
                    )
                )

            if missing_depth:

                errors.append(
                    f"[{split}] "
                    f"Depth missing "
                    f"{len(missing_depth)} stems: "
                    + ", ".join(
                        missing_depth[:20]
                    )
                )

            if extra_ir:

                errors.append(
                    f"[{split}] "
                    f"IR has extra "
                    f"{len(extra_ir)} stems: "
                    + ", ".join(
                        extra_ir[:20]
                    )
                )

            if extra_depth:

                errors.append(
                    f"[{split}] "
                    f"Depth has extra "
                    f"{len(extra_depth)} stems: "
                    + ", ".join(
                        extra_depth[:20]
                    )
                )

        print(
            f"Common stems: "
            f"{len(common_stems)}"
        )

        # ----------------------------------------------------
        # Label stem checks
        # ----------------------------------------------------

        for modality, label_map in [
            ("rgb", rgb_labels),
            ("ir", ir_labels),
            ("depth", depth_labels),
        ]:

            label_stems = set(
                label_map
            )

            missing_labels = (
                rgb_stems
                - label_stems
            )

            extra_labels = (
                label_stems
                - rgb_stems
            )

            if missing_labels:

                errors.append(
                    f"[{split}] "
                    f"{modality} missing "
                    f"{len(missing_labels)} labels: "
                    + ", ".join(
                        sorted(
                            missing_labels
                        )[:20]
                    )
                )

            if extra_labels:

                errors.append(
                    f"[{split}] "
                    f"{modality} has "
                    f"{len(extra_labels)} extra labels: "
                    + ", ".join(
                        sorted(
                            extra_labels
                        )[:20]
                    )
                )

        # ====================================================
        # Per-sample check
        # ====================================================

        stems_to_check = sorted(
            common_stems
        )

        for index, stem in enumerate(
            stems_to_check,
            start=1,
        ):

            rgb_path = (
                rgb_images[
                    stem
                ]
            )

            ir_path = (
                ir_images[
                    stem
                ]
            )

            depth_path = (
                depth_images[
                    stem
                ]
            )

            # ------------------------------------------------
            # Images
            # ------------------------------------------------

            rgb_info = inspect_image(
                rgb_path
            )

            ir_info = inspect_image(
                ir_path
            )

            depth_info = inspect_image(
                depth_path
            )

            sample_errors = []

            for modality, path, info in [
                (
                    "RGB",
                    rgb_path,
                    rgb_info,
                ),
                (
                    "IR",
                    ir_path,
                    ir_info,
                ),
                (
                    "Depth",
                    depth_path,
                    depth_info,
                ),
            ]:

                if not info[
                    "ok"
                ]:

                    sample_errors.append(
                        f"{modality} read failed: "
                        f"{path} "
                        f"({info.get('error')})"
                    )

            # Cannot safely continue dimension checks
            # if any image failed.
            if sample_errors:

                errors.extend(
                    f"[{split}/{stem}] {msg}"
                    for msg
                    in sample_errors
                )

                continue

            # ------------------------------------------------
            # Format statistics
            # ------------------------------------------------

            for modality, info in [
                ("rgb", rgb_info),
                ("ir", ir_info),
                ("depth", depth_info),
            ]:

                shape_stats[
                    modality
                ][
                    str(
                        info[
                            "shape"
                        ]
                    )
                ] += 1

                dtype_stats[
                    modality
                ][
                    info[
                        "dtype"
                    ]
                ] += 1

                channel_stats[
                    modality
                ][
                    str(
                        info[
                            "channels"
                        ]
                    )
                ] += 1

                extension_stats[
                    modality
                ][
                    info[
                        "extension"
                    ]
                ] += 1

            # ------------------------------------------------
            # Expected format:
            # all views are now uint8 3-channel.
            # ------------------------------------------------

            for modality, info in [
                ("RGB", rgb_info),
                ("IR", ir_info),
                ("Depth", depth_info),
            ]:

                if (
                    info[
                        "dtype"
                    ]
                    != "uint8"
                ):

                    errors.append(
                        f"[{split}/{stem}] "
                        f"{modality} dtype "
                        f"is {info['dtype']}, "
                        "expected uint8"
                    )

                if (
                    info[
                        "channels"
                    ]
                    != 3
                ):

                    errors.append(
                        f"[{split}/{stem}] "
                        f"{modality} channels "
                        f"is {info['channels']}, "
                        "expected 3"
                    )

            # ------------------------------------------------
            # Spatial size equality
            # ------------------------------------------------

            rgb_hw = (
                rgb_info[
                    "height"
                ],
                rgb_info[
                    "width"
                ],
            )

            ir_hw = (
                ir_info[
                    "height"
                ],
                ir_info[
                    "width"
                ],
            )

            depth_hw = (
                depth_info[
                    "height"
                ],
                depth_info[
                    "width"
                ],
            )

            spatial_match = (
                rgb_hw
                == ir_hw
                == depth_hw
            )

            if not spatial_match:

                errors.append(
                    f"[{split}/{stem}] "
                    "spatial size mismatch: "
                    f"RGB={rgb_hw}, "
                    f"IR={ir_hw}, "
                    f"Depth={depth_hw}"
                )

            resolution_triplets[
                (
                    str(rgb_hw),
                    str(ir_hw),
                    str(depth_hw),
                )
            ] += 1

            # ------------------------------------------------
            # Labels
            # ------------------------------------------------

            label_available = (
                stem in rgb_labels
                and stem in ir_labels
                and stem in depth_labels
            )

            label_match = False

            label_sha256 = ""

            object_count = None

            if not label_available:

                errors.append(
                    f"[{split}/{stem}] "
                    "one or more label files missing"
                )

            else:

                rgb_label_path = (
                    rgb_labels[
                        stem
                    ]
                )

                ir_label_path = (
                    ir_labels[
                        stem
                    ]
                )

                depth_label_path = (
                    depth_labels[
                        stem
                    ]
                )

                rgb_hash = sha256_file(
                    rgb_label_path
                )

                ir_hash = sha256_file(
                    ir_label_path
                )

                depth_hash = sha256_file(
                    depth_label_path
                )

                label_match = (
                    rgb_hash
                    == ir_hash
                    == depth_hash
                )

                label_sha256 = (
                    rgb_hash
                )

                if not label_match:

                    errors.append(
                        f"[{split}/{stem}] "
                        "label content mismatch:\n"
                        f"  RGB   : {rgb_label_path}\n"
                        f"  IR    : {ir_label_path}\n"
                        f"  Depth : {depth_label_path}"
                    )

                # Parse RGB label as canonical copy.
                try:

                    objects = (
                        parse_label_file(
                            rgb_label_path
                        )
                    )

                    object_count = len(
                        objects
                    )

                    total_objects += (
                        object_count
                    )

                except Exception as exc:

                    errors.append(
                        f"[{split}/{stem}] "
                        f"invalid label: {exc}"
                    )

            # ------------------------------------------------
            # Depth valid ratio
            #
            # This is NOT an alignment error.
            # It is stored for future quality-aware fusion.
            # ------------------------------------------------

            depth_image = cv2.imread(
                str(depth_path),
                cv2.IMREAD_GRAYSCALE,
            )

            if depth_image is None:

                depth_valid_ratio = None

            else:

                depth_valid_ratio = float(
                    (
                        depth_image > 0
                    ).mean()
                )

                if (
                    depth_valid_ratio
                    < 0.05
                ):

                    warnings.append(
                        f"[{split}/{stem}] "
                        "Depth valid ratio below 5%: "
                        f"{depth_valid_ratio:.6f}"
                    )

            # ------------------------------------------------
            # CSV row
            # ------------------------------------------------

            rows.append(
                {
                    "split":
                        split,

                    "stem":
                        stem,

                    "rgb_path":
                        relative(
                            rgb_path
                        ),

                    "ir_path":
                        relative(
                            ir_path
                        ),

                    "depth_path":
                        relative(
                            depth_path
                        ),

                    "rgb_extension":
                        rgb_path.suffix.lower(),

                    "ir_extension":
                        ir_path.suffix.lower(),

                    "depth_extension":
                        depth_path.suffix.lower(),

                    "rgb_height":
                        rgb_info[
                            "height"
                        ],

                    "rgb_width":
                        rgb_info[
                            "width"
                        ],

                    "ir_height":
                        ir_info[
                            "height"
                        ],

                    "ir_width":
                        ir_info[
                            "width"
                        ],

                    "depth_height":
                        depth_info[
                            "height"
                        ],

                    "depth_width":
                        depth_info[
                            "width"
                        ],

                    "spatial_match":
                        spatial_match,

                    "rgb_dtype":
                        rgb_info[
                            "dtype"
                        ],

                    "ir_dtype":
                        ir_info[
                            "dtype"
                        ],

                    "depth_dtype":
                        depth_info[
                            "dtype"
                        ],

                    "rgb_channels":
                        rgb_info[
                            "channels"
                        ],

                    "ir_channels":
                        ir_info[
                            "channels"
                        ],

                    "depth_channels":
                        depth_info[
                            "channels"
                        ],

                    "label_match":
                        label_match,

                    "label_sha256":
                        label_sha256,

                    "object_count":
                        (
                            object_count
                            if object_count
                            is not None
                            else ""
                        ),

                    "depth_valid_ratio":
                        (
                            depth_valid_ratio
                            if depth_valid_ratio
                            is not None
                            else ""
                        ),
                }
            )

            if (
                index == 1
                or index % 100 == 0
                or index
                == len(
                    stems_to_check
                )
            ):

                print(
                    f"[{split}] "
                    f"{index:4d}/"
                    f"{len(stems_to_check)} "
                    f"errors={len(errors):3d} "
                    f"warnings={len(warnings):3d}"
                )

        print()

    # ========================================================
    # Train / val overlap
    # ========================================================

    train_val_overlap = (
        split_stems[
            "train"
        ]
        & split_stems[
            "val"
        ]
    )

    if train_val_overlap:

        errors.append(
            "Train / val overlap detected: "
            + ", ".join(
                sorted(
                    train_val_overlap
                )[:30]
            )
        )

    # ========================================================
    # Total count
    # ========================================================

    total_aligned_samples = len(
        rows
    )

    if (
        total_aligned_samples
        != EXPECTED_TOTAL
    ):

        errors.append(
            "Aligned sample count mismatch: "
            f"expected={EXPECTED_TOTAL}, "
            f"actual={total_aligned_samples}"
        )

    # ========================================================
    # Write sample_alignment.csv
    # ========================================================

    csv_path = (
        OUTPUT_DIR
        / "sample_alignment.csv"
    )

    fieldnames = [
        "split",
        "stem",

        "rgb_path",
        "ir_path",
        "depth_path",

        "rgb_extension",
        "ir_extension",
        "depth_extension",

        "rgb_height",
        "rgb_width",

        "ir_height",
        "ir_width",

        "depth_height",
        "depth_width",

        "spatial_match",

        "rgb_dtype",
        "ir_dtype",
        "depth_dtype",

        "rgb_channels",
        "ir_channels",
        "depth_channels",

        "label_match",
        "label_sha256",

        "object_count",

        "depth_valid_ratio",
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
            rows
        )

    # ========================================================
    # Errors / warnings
    # ========================================================

    errors_path = (
        OUTPUT_DIR
        / "errors.txt"
    )

    with errors_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        if not errors:

            f.write(
                "No alignment errors detected.\n"
            )

        else:

            for item in errors:

                f.write(
                    item
                    + "\n"
                )

    warnings_path = (
        OUTPUT_DIR
        / "warnings.txt"
    )

    with warnings_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        if not warnings:

            f.write(
                "No warnings.\n"
            )

        else:

            for item in warnings:

                f.write(
                    item
                    + "\n"
                )

    # ========================================================
    # Summary
    # ========================================================

    spatial_match_count = sum(
        1
        for row in rows
        if row[
            "spatial_match"
        ]
    )

    label_match_count = sum(
        1
        for row in rows
        if row[
            "label_match"
        ]
    )

    depth_valid_values = [
        float(
            row[
                "depth_valid_ratio"
            ]
        )
        for row in rows
        if row[
            "depth_valid_ratio"
        ]
        != ""
    ]

    if depth_valid_values:

        depth_valid_summary = {
            "min":
                float(
                    np.min(
                        depth_valid_values
                    )
                ),

            "p5":
                float(
                    np.percentile(
                        depth_valid_values,
                        5,
                    )
                ),

            "median":
                float(
                    np.median(
                        depth_valid_values
                    )
                ),

            "mean":
                float(
                    np.mean(
                        depth_valid_values
                    )
                ),

            "p95":
                float(
                    np.percentile(
                        depth_valid_values,
                        95,
                    )
                ),

            "max":
                float(
                    np.max(
                        depth_valid_values
                    )
                ),
        }

    else:

        depth_valid_summary = {}

    summary = {
        "expected_total":
            EXPECTED_TOTAL,

        "aligned_rows":
            total_aligned_samples,

        "expected_split_counts":
            EXPECTED_COUNTS,

        "image_counts":
            image_count_stats,

        "label_counts":
            label_count_stats,

        "train_val_overlap":
            len(
                train_val_overlap
            ),

        "spatial_match_count":
            spatial_match_count,

        "spatial_mismatch_count":
            (
                total_aligned_samples
                - spatial_match_count
            ),

        "label_match_count":
            label_match_count,

        "label_mismatch_count":
            (
                total_aligned_samples
                - label_match_count
            ),

        "total_gt_objects":
            total_objects,

        "error_count":
            len(errors),

        "warning_count":
            len(warnings),

        "status":
            (
                "PASS"
                if not errors
                else "FAIL"
            ),

        "shape_distribution": {
            modality:
                dict(
                    counter
                )
            for modality, counter
            in shape_stats.items()
        },

        "dtype_distribution": {
            modality:
                dict(
                    counter
                )
            for modality, counter
            in dtype_stats.items()
        },

        "channel_distribution": {
            modality:
                dict(
                    counter
                )
            for modality, counter
            in channel_stats.items()
        },

        "extension_distribution": {
            modality:
                dict(
                    counter
                )
            for modality, counter
            in extension_stats.items()
        },

        "resolution_triplets": {
            str(key):
                int(value)
            for key, value
            in resolution_triplets.items()
        },

        "depth_valid_ratio":
            depth_valid_summary,
    }

    summary_path = (
        OUTPUT_DIR
        / "alignment_summary.json"
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
    # Terminal summary
    # ========================================================

    print()
    print("=" * 92)
    print("MULTIMODAL ALIGNMENT SUMMARY")
    print("=" * 92)

    print(
        f"Aligned samples          : "
        f"{total_aligned_samples}"
    )

    print()

    print(
        "Train images:"
    )

    print(
        f"  RGB                    : "
        f"{image_count_stats.get('train_rgb', 0)}"
    )

    print(
        f"  IR                     : "
        f"{image_count_stats.get('train_ir', 0)}"
    )

    print(
        f"  Depth                  : "
        f"{image_count_stats.get('train_depth', 0)}"
    )

    print()

    print(
        "Val images:"
    )

    print(
        f"  RGB                    : "
        f"{image_count_stats.get('val_rgb', 0)}"
    )

    print(
        f"  IR                     : "
        f"{image_count_stats.get('val_ir', 0)}"
    )

    print(
        f"  Depth                  : "
        f"{image_count_stats.get('val_depth', 0)}"
    )

    print()

    print(
        f"Spatial matches         : "
        f"{spatial_match_count}/"
        f"{total_aligned_samples}"
    )

    print(
        f"Label matches           : "
        f"{label_match_count}/"
        f"{total_aligned_samples}"
    )

    print(
        f"Train/val overlap       : "
        f"{len(train_val_overlap)}"
    )

    print(
        f"Total GT objects        : "
        f"{total_objects}"
    )

    print()

    print(
        f"Errors                  : "
        f"{len(errors)}"
    )

    print(
        f"Warnings                : "
        f"{len(warnings)}"
    )

    if depth_valid_summary:

        print()

        print(
            "Depth valid-ratio:"
        )

        print(
            f"  min                   : "
            f"{depth_valid_summary['min']:.6f}"
        )

        print(
            f"  P5                    : "
            f"{depth_valid_summary['p5']:.6f}"
        )

        print(
            f"  median                : "
            f"{depth_valid_summary['median']:.6f}"
        )

        print(
            f"  mean                  : "
            f"{depth_valid_summary['mean']:.6f}"
        )

        print(
            f"  P95                   : "
            f"{depth_valid_summary['p95']:.6f}"
        )

        print(
            f"  max                   : "
            f"{depth_valid_summary['max']:.6f}"
        )

    print()

    print("-" * 92)

    print(
        f"CSV     : {csv_path}"
    )

    print(
        f"Summary : {summary_path}"
    )

    print(
        f"Errors  : {errors_path}"
    )

    print(
        f"Warnings: {warnings_path}"
    )

    print("-" * 92)

    if not errors:

        print()
        print(
            "MULTIMODAL ALIGNMENT: PASS"
        )

        print(
            "RGB / IR / Depth are safe to use "
            "for the Exp04 multimodal data pipeline."
        )

    else:

        print()
        print(
            "MULTIMODAL ALIGNMENT: FAIL"
        )

        print(
            "Do NOT start Exp04 until errors.txt "
            "has been resolved."
        )

    print("=" * 92)


if __name__ == "__main__":
    main()

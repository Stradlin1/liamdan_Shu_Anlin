#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exp01 RGB Baseline - Official-spec Validation

目的：
1. 使用 Exp01 的 best.pt，而不是 last.pt
2. 固定验证集为 rgb_v1 / val
3. 固定 imgsz=960
4. 按比赛要求设置 max_det=100
5. 输出总体：
       Precision
       Recall
       mAP@50
       mAP@50-95
6. 输出每个类别：
       GT 数量
       Precision
       Recall
       AP@50
       AP@50-95
7. 保存 per_class_metrics.csv 和 validation_summary.json

注意：
- 本脚本只做验证，不重新训练
- 所有路径均相对于项目根目录确定
- 不依赖当前终端所在目录
"""

from pathlib import Path
import csv
import json

import numpy as np
from ultralytics import YOLO


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WEIGHTS_PATH = (
    PROJECT_ROOT
    / "runs"
    / "exp01_rgb_yolo11s_960"
    / "weights"
    / "best.pt"
)

DATA_YAML = (
    PROJECT_ROOT
    / "yolo_views"
    / "rgb_v1"
    / "data.yaml"
)

VAL_IMAGE_DIR = (
    PROJECT_ROOT
    / "yolo_views"
    / "rgb_v1"
    / "images"
    / "val"
)

VAL_LABEL_DIR = (
    PROJECT_ROOT
    / "yolo_views"
    / "rgb_v1"
    / "labels"
    / "val"
)

RUNS_DIR = (
    PROJECT_ROOT
    / "runs"
)

EXPERIMENT_NAME = "val_exp01_rgb_yolo11s_960"


# ============================================================
# Validation configuration
# ============================================================

IMAGE_SIZE = 960
BATCH_SIZE = 8
DEVICE = "0"
WORKERS = 8

# Ultralytics validation usually uses a very low confidence threshold
# so that PR / AP calculation sees enough candidate detections.
CONF_THRESHOLD = 0.001

# NMS IoU threshold.
NMS_IOU = 0.70

# Competition rule:
# each image may contain at most 100 submitted detections.
MAX_DET = 100

SEED = 2026


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
# Utility functions
# ============================================================

def check_files():
    """Check all required files/directories before validation."""

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Exp01 best.pt not found:\n{WEIGHTS_PATH}"
        )

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found:\n{DATA_YAML}"
        )

    if not VAL_IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Validation image directory not found:\n{VAL_IMAGE_DIR}"
        )

    if not VAL_LABEL_DIR.exists():
        raise FileNotFoundError(
            f"Validation label directory not found:\n{VAL_LABEL_DIR}"
        )

    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def normalize_names(raw_names):
    """
    Convert model.names into:
        {0: "person", 1: "boat", ...}
    """

    if isinstance(raw_names, dict):
        return {
            int(k): str(v)
            for k, v in raw_names.items()
        }

    if isinstance(raw_names, (list, tuple)):
        return {
            i: str(name)
            for i, name in enumerate(raw_names)
        }

    raise TypeError(
        f"Unsupported class-name format: {type(raw_names)}"
    )


def count_validation_images():
    """Count validation images."""

    count = 0

    for path in VAL_IMAGE_DIR.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            count += 1

    return count


def count_ground_truth(num_classes):
    """
    Count GT instances in labels/val.

    Returns:
        gt_counts: dict[class_id] -> number of GT boxes
        label_file_count
    """

    gt_counts = {
        class_id: 0
        for class_id in range(num_classes)
    }

    label_files = sorted(
        VAL_LABEL_DIR.rglob("*.txt")
    )

    for label_path in label_files:

        with label_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            for line_number, line in enumerate(
                f,
                start=1,
            ):
                line = line.strip()

                # Empty label files are valid.
                if not line:
                    continue

                parts = line.split()

                if len(parts) < 5:
                    raise ValueError(
                        "Invalid YOLO label line:\n"
                        f"file: {label_path}\n"
                        f"line: {line_number}\n"
                        f"text: {line}"
                    )

                try:
                    class_id = int(parts[0])
                except ValueError as exc:
                    raise ValueError(
                        "Invalid class id:\n"
                        f"file: {label_path}\n"
                        f"line: {line_number}\n"
                        f"text: {line}"
                    ) from exc

                if class_id not in gt_counts:
                    raise ValueError(
                        "Class id outside model range:\n"
                        f"file: {label_path}\n"
                        f"line: {line_number}\n"
                        f"class_id: {class_id}\n"
                        f"num_classes: {num_classes}"
                    )

                gt_counts[class_id] += 1

    return gt_counts, len(label_files)


def metric_value(
    array,
    row_index,
    class_id,
    evaluated_class_count,
):
    """
    Read a per-class metric robustly across slightly different
    Ultralytics Metric implementations.
    """

    if array is None:
        return None

    array = np.asarray(array)

    if array.size == 0:
        return None

    array = array.reshape(-1)

    # Standard Ultralytics case:
    # p/r/f1 correspond to ap_class_index order.
    if len(array) == evaluated_class_count:
        return float(array[row_index])

    # Fallback:
    # array directly indexed by class id.
    if class_id < len(array):
        return float(array[class_id])

    return None


def fmt(value):
    if value is None:
        return "-"
    return f"{value:.6f}"


# ============================================================
# Main
# ============================================================

def main():

    check_files()

    print("=" * 78)
    print("Exp01 RGB YOLO11s 960 - Official-spec Validation")
    print("=" * 78)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Weights      : {WEIGHTS_PATH}")
    print(f"Dataset      : {DATA_YAML}")
    print(f"Images       : {VAL_IMAGE_DIR}")
    print(f"Labels       : {VAL_LABEL_DIR}")
    print(f"Output root  : {RUNS_DIR}")
    print("-" * 78)
    print(f"imgsz        : {IMAGE_SIZE}")
    print(f"batch        : {BATCH_SIZE}")
    print(f"device       : {DEVICE}")
    print(f"conf         : {CONF_THRESHOLD}")
    print(f"NMS IoU      : {NMS_IOU}")
    print(f"max_det      : {MAX_DET}")
    print("=" * 78)

    # --------------------------------------------------------
    # Load trained Exp01 best checkpoint
    # --------------------------------------------------------

    model = YOLO(
        str(WEIGHTS_PATH)
    )

    names = normalize_names(
        model.names
    )

    num_classes = len(names)

    print()
    print(f"Number of classes: {num_classes}")

    for class_id in sorted(names):
        print(
            f"  {class_id:2d}: {names[class_id]}"
        )

    # --------------------------------------------------------
    # Dataset statistics
    # --------------------------------------------------------

    val_image_count = count_validation_images()

    gt_counts, label_file_count = count_ground_truth(
        num_classes
    )

    total_gt = sum(
        gt_counts.values()
    )

    print()
    print("=" * 78)
    print("Validation dataset statistics")
    print("=" * 78)
    print(f"Validation images : {val_image_count}")
    print(f"Label TXT files   : {label_file_count}")
    print(f"Total GT boxes    : {total_gt}")

    for class_id in sorted(names):
        print(
            f"  {class_id:2d} "
            f"{names[class_id]:15s} "
            f"GT={gt_counts[class_id]}"
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("Starting validation...")
    print("=" * 78)

    metrics = model.val(
        data=str(DATA_YAML),

        split="val",

        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,

        conf=CONF_THRESHOLD,
        iou=NMS_IOU,
        max_det=MAX_DET,

        augment=False,

        plots=True,
        save_json=False,
        save_txt=False,

        project=str(RUNS_DIR),
        name=EXPERIMENT_NAME,
        exist_ok=False,

        verbose=True,

        seed=SEED,
        deterministic=True,

        # Keep consistent with the project Ultralytics config
        # used by Exp01.
        cls_remap=True,
    )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    box = metrics.box

    overall = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map75": float(box.map75),
        "map50_95": float(box.map),
    }

    print()
    print("=" * 78)
    print("Overall validation metrics")
    print("=" * 78)
    print(
        f"Precision   : {overall['precision']:.6f}"
    )
    print(
        f"Recall      : {overall['recall']:.6f}"
    )
    print(
        f"mAP@50      : {overall['map50']:.6f}"
    )
    print(
        f"mAP@75      : {overall['map75']:.6f}"
    )
    print(
        f"mAP@50-95   : {overall['map50_95']:.6f}"
    )

    # --------------------------------------------------------
    # Per-class metrics
    # --------------------------------------------------------

    all_ap = np.asarray(
        box.all_ap,
        dtype=float,
    )

    if all_ap.ndim != 2:
        raise RuntimeError(
            "Unexpected box.all_ap shape: "
            f"{all_ap.shape}"
        )

    ap_class_index = np.asarray(
        box.ap_class_index,
        dtype=int,
    ).reshape(-1)

    evaluated_class_count = len(
        ap_class_index
    )

    precision_array = getattr(
        box,
        "p",
        None,
    )

    recall_array = getattr(
        box,
        "r",
        None,
    )

    f1_array = getattr(
        box,
        "f1",
        None,
    )

    class_to_row = {
        int(class_id): row_index
        for row_index, class_id
        in enumerate(ap_class_index)
    }

    per_class_rows = []

    for class_id in sorted(names):

        class_name = names[class_id]

        if class_id in class_to_row:

            row_index = class_to_row[class_id]

            ap_values = all_ap[
                row_index
            ]

            ap50 = float(
                ap_values[0]
            )

            ap50_95 = float(
                np.mean(ap_values)
            )

            precision = metric_value(
                precision_array,
                row_index,
                class_id,
                evaluated_class_count,
            )

            recall = metric_value(
                recall_array,
                row_index,
                class_id,
                evaluated_class_count,
            )

            f1 = metric_value(
                f1_array,
                row_index,
                class_id,
                evaluated_class_count,
            )

        else:

            # Normally this should only happen if a class has
            # no GT instances in the validation split.
            ap50 = None
            ap50_95 = None
            precision = None
            recall = None
            f1 = None

        per_class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "gt_count": gt_counts[class_id],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "ap50": ap50,
                "ap50_95": ap50_95,
            }
        )

    # --------------------------------------------------------
    # Print class table
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("Per-class metrics")
    print("=" * 110)

    print(
        f"{'ID':>3} "
        f"{'Class':<16} "
        f"{'GT':>7} "
        f"{'Precision':>12} "
        f"{'Recall':>12} "
        f"{'F1':>12} "
        f"{'AP50':>12} "
        f"{'AP50-95':>12}"
    )

    print("-" * 110)

    for row in per_class_rows:

        print(
            f"{row['class_id']:>3d} "
            f"{row['class_name']:<16} "
            f"{row['gt_count']:>7d} "
            f"{fmt(row['precision']):>12} "
            f"{fmt(row['recall']):>12} "
            f"{fmt(row['f1']):>12} "
            f"{fmt(row['ap50']):>12} "
            f"{fmt(row['ap50_95']):>12}"
        )

    # --------------------------------------------------------
    # Determine actual Ultralytics output directory
    # --------------------------------------------------------

    save_dir = Path(
        getattr(
            metrics,
            "save_dir",
            RUNS_DIR / EXPERIMENT_NAME,
        )
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save per-class CSV
    # --------------------------------------------------------

    csv_path = (
        save_dir
        / "per_class_metrics.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_id",
                "class_name",
                "gt_count",
                "precision",
                "recall",
                "f1",
                "ap50",
                "ap50_95",
            ],
        )

        writer.writeheader()

        for row in per_class_rows:
            writer.writerow(row)

    # --------------------------------------------------------
    # Save JSON summary
    # --------------------------------------------------------

    summary_path = (
        save_dir
        / "validation_summary.json"
    )

    summary = {
        "experiment": EXPERIMENT_NAME,
        "weights": str(
            WEIGHTS_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "data_yaml": str(
            DATA_YAML.relative_to(
                PROJECT_ROOT
            )
        ),
        "validation_images": val_image_count,
        "label_files": label_file_count,
        "total_gt_boxes": total_gt,
        "configuration": {
            "imgsz": IMAGE_SIZE,
            "batch": BATCH_SIZE,
            "device": DEVICE,
            "workers": WORKERS,
            "conf": CONF_THRESHOLD,
            "nms_iou": NMS_IOU,
            "max_det": MAX_DET,
            "seed": SEED,
        },
        "overall_metrics": overall,
        "per_class_metrics": per_class_rows,
    }

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

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("Validation completed.")
    print("=" * 78)
    print(f"Results directory : {save_dir}")
    print(f"Per-class CSV     : {csv_path}")
    print(f"Summary JSON      : {summary_path}")
    print()
    print(
        "Primary competition metric: "
        f"mAP@50-95 = {overall['map50_95']:.6f}"
    )
    print(
        "Competition max detections: "
        f"{MAX_DET} per image"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exp03 Depth-only YOLO11s 960 - Official-spec Validation

使用：
    runs/exp03_depth_yolo11s_960/weights/best.pt

验证：
    yolo_views/depth8_v1

正式比赛规格：
    imgsz   = 960
    max_det = 100

输出：
    overall metrics
    per-class Precision / Recall / F1 / AP50 / AP50-95
    per_class_metrics.csv
    validation_summary.json
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
    / "exp03_depth_yolo11s_960"
    / "weights"
    / "best.pt"
)

DATA_YAML = (
    PROJECT_ROOT
    / "yolo_views"
    / "depth8_v1"
    / "data.yaml"
)

VAL_IMAGE_DIR = (
    PROJECT_ROOT
    / "yolo_views"
    / "depth8_v1"
    / "images"
    / "val"
)

VAL_LABEL_DIR = (
    PROJECT_ROOT
    / "yolo_views"
    / "depth8_v1"
    / "labels"
    / "val"
)

RUNS_DIR = (
    PROJECT_ROOT
    / "runs"
)

EXPERIMENT_NAME = "val_exp03_depth_yolo11s_960"


# ============================================================
# Validation configuration
# ============================================================

IMAGE_SIZE = 960
BATCH_SIZE = 8
DEVICE = "0"
WORKERS = 8

CONF_THRESHOLD = 0.001
NMS_IOU = 0.70

# Competition rule:
# maximum 100 predictions per image
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
# Helpers
# ============================================================

def check_files():

    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            "Exp03 best.pt not found:\n"
            f"  {WEIGHTS_PATH}"
        )

    if not DATA_YAML.is_file():
        raise FileNotFoundError(
            "Depth data.yaml not found:\n"
            f"  {DATA_YAML}"
        )

    if not VAL_IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            "Validation image directory not found:\n"
            f"  {VAL_IMAGE_DIR}"
        )

    if not VAL_LABEL_DIR.is_dir():
        raise FileNotFoundError(
            "Validation label directory not found:\n"
            f"  {VAL_LABEL_DIR}"
        )

    output_dir = (
        RUNS_DIR
        / EXPERIMENT_NAME
    )

    if output_dir.exists():
        raise FileExistsError(
            "Validation output directory already exists:\n"
            f"  {output_dir}\n\n"
            "If you explicitly want to rerun it:\n"
            "  rm -rf runs/val_exp03_depth_yolo11s_960"
        )

    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def normalize_names(raw_names):

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
        f"Unsupported class names type: {type(raw_names)}"
    )


def count_validation_images():

    return sum(
        1
        for path in VAL_IMAGE_DIR.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )


def count_ground_truth(num_classes):

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

                if not line:
                    continue

                parts = line.split()

                if len(parts) < 5:

                    raise ValueError(
                        "Invalid YOLO label:\n"
                        f"file : {label_path}\n"
                        f"line : {line_number}\n"
                        f"text : {line}"
                    )

                class_id = int(
                    parts[0]
                )

                if class_id not in gt_counts:

                    raise ValueError(
                        "Invalid class id:\n"
                        f"file     : {label_path}\n"
                        f"class_id : {class_id}"
                    )

                gt_counts[
                    class_id
                ] += 1

    return (
        gt_counts,
        len(label_files),
    )


def metric_value(
    array,
    row_index,
    class_id,
    evaluated_class_count,
):

    if array is None:
        return None

    array = np.asarray(
        array
    ).reshape(-1)

    if array.size == 0:
        return None

    # Standard Ultralytics case
    if len(array) == evaluated_class_count:
        return float(
            array[row_index]
        )

    # Fallback
    if class_id < len(array):
        return float(
            array[class_id]
        )

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

    print("=" * 82)
    print("Exp03 Depth YOLO11s 960 - Official-spec Validation")
    print("=" * 82)

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Weights      : {WEIGHTS_PATH}"
    )

    print(
        f"Dataset      : {DATA_YAML}"
    )

    print(
        f"Images       : {VAL_IMAGE_DIR}"
    )

    print(
        f"Labels       : {VAL_LABEL_DIR}"
    )

    print("-" * 82)

    print(
        f"imgsz        : {IMAGE_SIZE}"
    )

    print(
        f"batch        : {BATCH_SIZE}"
    )

    print(
        f"device       : {DEVICE}"
    )

    print(
        f"conf         : {CONF_THRESHOLD}"
    )

    print(
        f"NMS IoU      : {NMS_IOU}"
    )

    print(
        f"max_det      : {MAX_DET}"
    )

    print("=" * 82)

    # --------------------------------------------------------
    # Load Exp03 best.pt
    # --------------------------------------------------------

    model = YOLO(
        str(WEIGHTS_PATH)
    )

    names = normalize_names(
        model.names
    )

    num_classes = len(
        names
    )

    # --------------------------------------------------------
    # Dataset statistics
    # --------------------------------------------------------

    val_image_count = (
        count_validation_images()
    )

    (
        gt_counts,
        label_file_count,
    ) = count_ground_truth(
        num_classes
    )

    total_gt = sum(
        gt_counts.values()
    )

    print()
    print("=" * 82)
    print("Validation dataset")
    print("=" * 82)

    print(
        f"Images       : {val_image_count}"
    )

    print(
        f"Label files  : {label_file_count}"
    )

    print(
        f"Total GT     : {total_gt}"
    )

    print()

    for class_id in sorted(names):

        print(
            f"{class_id:2d} "
            f"{names[class_id]:15s} "
            f"GT={gt_counts[class_id]}"
        )

    if val_image_count != 400:

        raise RuntimeError(
            "Validation image count is not 400:\n"
            f"  actual = {val_image_count}"
        )

    if label_file_count != 400:

        raise RuntimeError(
            "Validation label file count is not 400:\n"
            f"  actual = {label_file_count}"
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print()
    print("=" * 82)
    print("Starting official-spec validation")
    print("=" * 82)

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

        cls_remap=True,
    )

    box = metrics.box

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    overall = {
        "precision": float(
            box.mp
        ),

        "recall": float(
            box.mr
        ),

        "map50": float(
            box.map50
        ),

        "map75": float(
            box.map75
        ),

        "map50_95": float(
            box.map
        ),
    }

    print()
    print("=" * 82)
    print("Overall metrics")
    print("=" * 82)

    print(
        f"Precision   : "
        f"{overall['precision']:.6f}"
    )

    print(
        f"Recall      : "
        f"{overall['recall']:.6f}"
    )

    print(
        f"mAP@50      : "
        f"{overall['map50']:.6f}"
    )

    print(
        f"mAP@75      : "
        f"{overall['map75']:.6f}"
    )

    print(
        f"mAP@50-95   : "
        f"{overall['map50_95']:.6f}"
    )

    # --------------------------------------------------------
    # Per-class AP
    # --------------------------------------------------------

    all_ap = np.asarray(
        box.all_ap,
        dtype=float,
    )

    if all_ap.ndim != 2:

        raise RuntimeError(
            "Unexpected box.all_ap shape:\n"
            f"  {all_ap.shape}"
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
        in enumerate(
            ap_class_index
        )
    }

    rows = []

    for class_id in sorted(names):

        class_name = names[
            class_id
        ]

        if class_id in class_to_row:

            row_index = (
                class_to_row[
                    class_id
                ]
            )

            ap_values = all_ap[
                row_index
            ]

            ap50 = float(
                ap_values[0]
            )

            ap50_95 = float(
                np.mean(
                    ap_values
                )
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

            precision = None
            recall = None
            f1 = None
            ap50 = None
            ap50_95 = None

        rows.append(
            {
                "class_id":
                    class_id,

                "class_name":
                    class_name,

                "gt_count":
                    gt_counts[
                        class_id
                    ],

                "precision":
                    precision,

                "recall":
                    recall,

                "f1":
                    f1,

                "ap50":
                    ap50,

                "ap50_95":
                    ap50_95,
            }
        )

    # --------------------------------------------------------
    # Terminal table
    # --------------------------------------------------------

    print()
    print("=" * 112)
    print("Per-class metrics")
    print("=" * 112)

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

    print("-" * 112)

    for row in rows:

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
    # Output directory
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
    # CSV
    # --------------------------------------------------------

    csv_path = (
        save_dir
        / "per_class_metrics.csv"
    )

    fieldnames = [
        "class_id",
        "class_name",
        "gt_count",
        "precision",
        "recall",
        "f1",
        "ap50",
        "ap50_95",
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

    # --------------------------------------------------------
    # JSON summary
    # --------------------------------------------------------

    summary = {
        "experiment":
            EXPERIMENT_NAME,

        "modality":
            "depth8_v1",

        "weights":
            str(
                WEIGHTS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),

        "data_yaml":
            str(
                DATA_YAML.relative_to(
                    PROJECT_ROOT
                )
            ),

        "validation_images":
            val_image_count,

        "label_files":
            label_file_count,

        "total_gt_boxes":
            total_gt,

        "configuration": {
            "imgsz":
                IMAGE_SIZE,

            "batch":
                BATCH_SIZE,

            "device":
                DEVICE,

            "workers":
                WORKERS,

            "conf":
                CONF_THRESHOLD,

            "nms_iou":
                NMS_IOU,

            "max_det":
                MAX_DET,

            "seed":
                SEED,
        },

        "overall_metrics":
            overall,

        "per_class_metrics":
            rows,
    }

    json_path = (
        save_dir
        / "validation_summary.json"
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

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 82)
    print("Exp03 official validation completed")
    print("=" * 82)

    print(
        f"Results : {save_dir}"
    )

    print(
        f"CSV     : {csv_path}"
    )

    print(
        f"JSON    : {json_path}"
    )

    print()

    print(
        "Competition metric:"
        f" mAP@50-95 = "
        f"{overall['map50_95']:.6f}"
    )

    print(
        f"Competition max_det: {MAX_DET}"
    )

    print("=" * 82)


if __name__ == "__main__":
    main()

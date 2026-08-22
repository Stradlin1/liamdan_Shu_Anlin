#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exp03: Depth-only YOLO11s 960 Baseline

Controlled experiment:

    Exp01 -> RGB-only
    Exp02 -> Infrared-only
    Exp03 -> Depth-only

Depth input:
    yolo_views/depth8_v1/

depth8_v1 representation:
    metric uint16 PNG:
        invalid -> 0
        near    -> bright
        far     -> dark

    official uint8 JPG:
        BGR -> grayscale
        original grayscale distribution retained

All generated training images are 3-channel uint8 PNG.

Purpose:
1. Establish Depth-only baseline.
2. Compare Depth with RGB and IR on the same split.
3. Measure geometric information provided by Depth.
4. Prepare ablation evidence for multimodal fusion.
"""

from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "yolo11s.pt"
)

DATA_YAML = (
    PROJECT_ROOT
    / "yolo_views"
    / "depth8_v1"
    / "data.yaml"
)

DEPTH_VIEW_ROOT = (
    PROJECT_ROOT
    / "yolo_views"
    / "depth8_v1"
)

RUNS_DIR = (
    PROJECT_ROOT
    / "runs"
)

EXPERIMENT_NAME = (
    "exp03_depth_yolo11s_960"
)

OUTPUT_DIR = (
    RUNS_DIR
    / EXPERIMENT_NAME
)


# ============================================================
# Controlled training configuration
# ============================================================

EPOCHS = 200
PATIENCE = 50

BATCH_SIZE = 8
IMAGE_SIZE = 960

DEVICE = "0"
WORKERS = 8

SEED = 2026

EXPECTED_TRAIN_IMAGES = 1600
EXPECTED_VAL_IMAGES = 400

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
# Dataset checks
# ============================================================

def count_images(directory):

    return sum(
        1
        for path in directory.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )


def count_labels(directory):

    return sum(
        1
        for path in directory.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() == ".txt"
        )
    )


def check_files():

    if not MODEL_PATH.is_file():

        raise FileNotFoundError(
            "Pretrained model not found:\n"
            f"  {MODEL_PATH}"
        )

    if not DATA_YAML.is_file():

        raise FileNotFoundError(
            "Depth data.yaml not found:\n"
            f"  {DATA_YAML}\n\n"
            "Build depth8_v1 first."
        )

    train_images = (
        DEPTH_VIEW_ROOT
        / "images"
        / "train"
    )

    val_images = (
        DEPTH_VIEW_ROOT
        / "images"
        / "val"
    )

    train_labels = (
        DEPTH_VIEW_ROOT
        / "labels"
        / "train"
    )

    val_labels = (
        DEPTH_VIEW_ROOT
        / "labels"
        / "val"
    )

    required_dirs = [
        train_images,
        val_images,
        train_labels,
        val_labels,
    ]

    for path in required_dirs:

        if not path.is_dir():

            raise FileNotFoundError(
                "Required Depth dataset directory "
                "not found:\n"
                f"  {path}"
            )

    train_image_count = (
        count_images(
            train_images
        )
    )

    val_image_count = (
        count_images(
            val_images
        )
    )

    train_label_count = (
        count_labels(
            train_labels
        )
    )

    val_label_count = (
        count_labels(
            val_labels
        )
    )

    if (
        train_image_count
        != EXPECTED_TRAIN_IMAGES
    ):

        raise RuntimeError(
            "Unexpected train image count:\n"
            f"  expected = {EXPECTED_TRAIN_IMAGES}\n"
            f"  actual   = {train_image_count}"
        )

    if (
        val_image_count
        != EXPECTED_VAL_IMAGES
    ):

        raise RuntimeError(
            "Unexpected val image count:\n"
            f"  expected = {EXPECTED_VAL_IMAGES}\n"
            f"  actual   = {val_image_count}"
        )

    if (
        train_label_count
        != EXPECTED_TRAIN_IMAGES
    ):

        raise RuntimeError(
            "Unexpected train label count:\n"
            f"  expected = {EXPECTED_TRAIN_IMAGES}\n"
            f"  actual   = {train_label_count}"
        )

    if (
        val_label_count
        != EXPECTED_VAL_IMAGES
    ):

        raise RuntimeError(
            "Unexpected val label count:\n"
            f"  expected = {EXPECTED_VAL_IMAGES}\n"
            f"  actual   = {val_label_count}"
        )

    # Do not allow Ultralytics to silently create:
    # exp03_depth_yolo11s_9602, ...3, etc.
    if OUTPUT_DIR.exists():

        raise FileExistsError(
            "Formal Exp03 output already exists:\n"
            f"  {OUTPUT_DIR}\n\n"
            "Training stopped to protect the existing run."
        )

    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return {
        "train_images": train_image_count,
        "val_images": val_image_count,
        "train_labels": train_label_count,
        "val_labels": val_label_count,
    }


# ============================================================
# Environment report
# ============================================================

def print_environment(
    dataset_counts,
):

    print("=" * 82)
    print("Exp03 - Depth-only YOLO11s 960")
    print("=" * 82)

    print(
        f"Project root  : {PROJECT_ROOT}"
    )

    print(
        f"Model         : {MODEL_PATH}"
    )

    print(
        f"Dataset       : {DATA_YAML}"
    )

    print(
        f"Output        : {OUTPUT_DIR}"
    )

    print("-" * 82)

    print(
        f"Train images  : "
        f"{dataset_counts['train_images']}"
    )

    print(
        f"Train labels  : "
        f"{dataset_counts['train_labels']}"
    )

    print(
        f"Val images    : "
        f"{dataset_counts['val_images']}"
    )

    print(
        f"Val labels    : "
        f"{dataset_counts['val_labels']}"
    )

    print("-" * 82)

    print(
        f"PyTorch       : {torch.__version__}"
    )

    print(
        f"Ultralytics   : "
        f"{ultralytics.__version__}"
    )

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"CUDA version  : "
            f"{torch.version.cuda}"
        )

        print(
            f"GPU           : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print("-" * 82)

    print(
        f"Epochs        : {EPOCHS}"
    )

    print(
        f"Patience      : {PATIENCE}"
    )

    print(
        f"Batch         : {BATCH_SIZE}"
    )

    print(
        f"Image size    : {IMAGE_SIZE}"
    )

    print(
        f"Device        : {DEVICE}"
    )

    print(
        f"Workers       : {WORKERS}"
    )

    print(
        f"Seed          : {SEED}"
    )

    print("-" * 82)

    print("Controlled baseline:")
    print("  Exp01 = RGB")
    print("  Exp02 = Infrared")
    print("  Exp03 = Depth")

    print("=" * 82)


# ============================================================
# Main
# ============================================================

def main():

    dataset_counts = (
        check_files()
    )

    print_environment(
        dataset_counts
    )

    # --------------------------------------------------------
    # Same pretrained YOLO11s used by Exp01 / Exp02.
    # --------------------------------------------------------

    model = YOLO(
        str(MODEL_PATH)
    )

    # --------------------------------------------------------
    # Controlled training
    # --------------------------------------------------------

    results = model.train(

        # Dataset / output
        data=str(DATA_YAML),

        project=str(RUNS_DIR),

        name=EXPERIMENT_NAME,

        exist_ok=False,

        # Core
        epochs=EPOCHS,

        patience=PATIENCE,

        batch=BATCH_SIZE,

        imgsz=IMAGE_SIZE,

        device=DEVICE,

        workers=WORKERS,

        cache=False,

        # Reproducibility
        seed=SEED,

        deterministic=True,

        # Model / optimizer
        pretrained=True,

        optimizer="auto",

        amp=True,

        cls_remap=True,

        # Optimizer parameters
        lr0=0.01,

        lrf=0.01,

        momentum=0.937,

        weight_decay=0.0005,

        warmup_epochs=3.0,

        warmup_momentum=0.8,

        warmup_bias_lr=0.1,

        nbs=64,

        # Detection loss
        box=7.5,

        cls=0.5,

        dfl=1.5,

        # Geometric augmentation
        degrees=0.0,

        translate=0.1,

        scale=0.5,

        shear=0.0,

        perspective=0.0,

        flipud=0.0,

        fliplr=0.5,

        # ----------------------------------------------------
        # Keep Exp01 / Exp02 augmentation settings for the
        # first controlled Depth baseline.
        #
        # Since Depth images contain duplicated grayscale
        # channels, hue/saturation changes have little effect.
        # hsv_v modifies intensity and is intentionally kept
        # here for baseline consistency.
        # ----------------------------------------------------

        hsv_h=0.015,

        hsv_s=0.7,

        hsv_v=0.4,

        bgr=0.0,

        # Compound augmentation
        mosaic=1.0,

        close_mosaic=10,

        mixup=0.0,

        cutmix=0.0,

        copy_paste=0.0,

        # Other
        single_cls=False,

        rect=False,

        cos_lr=False,

        multi_scale=0.0,

        dropout=0.0,

        fraction=1.0,

        freeze=None,

        resume=False,

        compile=False,

        # ----------------------------------------------------
        # Validation during training
        #
        # Exp01 / Exp02 training both used max_det=300.
        # Keep 300 here for controlled comparison.
        #
        # Official-spec validation after training will use
        # max_det=100.
        # ----------------------------------------------------

        val=True,

        split="val",

        iou=0.70,

        max_det=300,

        augment=False,

        # Output
        save=True,

        save_period=-1,

        plots=True,

        save_json=False,

        save_txt=False,

        verbose=True,
    )

    print()
    print("=" * 82)
    print("Exp03 training finished")
    print("=" * 82)

    print(
        f"Results : {OUTPUT_DIR}"
    )

    print(
        "Best    : "
        f"{OUTPUT_DIR / 'weights' / 'best.pt'}"
    )

    print(
        "Last    : "
        f"{OUTPUT_DIR / 'weights' / 'last.pt'}"
    )

    print()
    print(
        "Next: official-spec validation "
        "with max_det=100."
    )

    print("=" * 82)

    return results


if __name__ == "__main__":
    main()

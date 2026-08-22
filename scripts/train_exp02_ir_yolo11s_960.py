#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exp02: Infrared-only YOLO11s 960 Baseline

控制变量实验：
    Exp01: RGB-only YOLO11s 960
    Exp02: IR-only  YOLO11s 960

除输入模态和实验名称外，其余训练参数与 Exp01 保持一致。

目的：
1. 建立 Infrared 单模态 baseline
2. 与 Exp01 RGB baseline 直接比较
3. 分析 IR 对低照度、小目标、弱纹理目标的独立检测能力
4. 为后续 RGB + IR + Depth 多模态融合提供消融基准
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
    / "ir_v1"
    / "data.yaml"
)

RUNS_DIR = (
    PROJECT_ROOT
    / "runs"
)

EXPERIMENT_NAME = "exp02_ir_yolo11s_960"

OUTPUT_DIR = (
    RUNS_DIR
    / EXPERIMENT_NAME
)


# ============================================================
# Exp02 training configuration
#
# IMPORTANT:
# These settings intentionally match Exp01.
# Do not casually change them, otherwise RGB/IR comparison
# will no longer be a clean controlled experiment.
# ============================================================

EPOCHS = 200
PATIENCE = 50

BATCH_SIZE = 8
IMAGE_SIZE = 960

DEVICE = "0"
WORKERS = 8

SEED = 2026


# ============================================================
# File checks
# ============================================================

def check_files():

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Pretrained model not found:\n"
            f"  {MODEL_PATH}"
        )

    if not DATA_YAML.is_file():
        raise FileNotFoundError(
            "IR data.yaml not found:\n"
            f"  {DATA_YAML}\n\n"
            "Please build yolo_views/ir_v1 first."
        )

    ir_view_root = (
        PROJECT_ROOT
        / "yolo_views"
        / "ir_v1"
    )

    required_dirs = [
        ir_view_root / "images" / "train",
        ir_view_root / "images" / "val",
        ir_view_root / "labels" / "train",
        ir_view_root / "labels" / "val",
    ]

    for path in required_dirs:

        if not path.is_dir():
            raise FileNotFoundError(
                "Required IR YOLO directory not found:\n"
                f"  {path}"
            )

    # Do not silently create exp02_ir_yolo11s_9602 etc.
    # A formal experiment name must uniquely correspond
    # to exactly one run.
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            "Experiment output directory already exists:\n"
            f"  {OUTPUT_DIR}\n\n"
            "To protect existing experiment results, "
            "training has been stopped.\n"
            "Do not delete it unless you explicitly intend "
            "to rerun Exp02 from scratch."
        )

    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Environment information
# ============================================================

def print_environment():

    print("=" * 80)
    print("Exp02 - Infrared-only YOLO11s 960")
    print("=" * 80)

    print(f"Project root      : {PROJECT_ROOT}")
    print(f"Model             : {MODEL_PATH}")
    print(f"Dataset           : {DATA_YAML}")
    print(f"Output            : {OUTPUT_DIR}")

    print("-" * 80)

    print(f"PyTorch           : {torch.__version__}")
    print(f"Ultralytics       : {ultralytics.__version__}")
    print(f"CUDA available    : {torch.cuda.is_available()}")

    if torch.cuda.is_available():

        print(
            f"CUDA version      : "
            f"{torch.version.cuda}"
        )

        print(
            f"GPU               : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print("-" * 80)

    print(f"Epochs            : {EPOCHS}")
    print(f"Patience          : {PATIENCE}")
    print(f"Batch             : {BATCH_SIZE}")
    print(f"Image size        : {IMAGE_SIZE}")
    print(f"Device            : {DEVICE}")
    print(f"Workers           : {WORKERS}")
    print(f"Seed              : {SEED}")

    print("-" * 80)

    print("Controlled comparison:")
    print("  Exp01 = RGB-only")
    print("  Exp02 = IR-only")
    print()
    print(
        "All major training hyperparameters "
        "are kept consistent with Exp01."
    )

    print("=" * 80)


# ============================================================
# Main
# ============================================================

def main():

    check_files()

    print_environment()

    # --------------------------------------------------------
    # Load the exact same YOLO11s pretrained model family
    # used by Exp01.
    # --------------------------------------------------------

    model = YOLO(
        str(MODEL_PATH)
    )

    # --------------------------------------------------------
    # Train
    #
    # Parameters below are aligned with Exp01 args.yaml.
    # --------------------------------------------------------

    results = model.train(

        # ----------------------------------------------------
        # Dataset / output
        # ----------------------------------------------------

        data=str(DATA_YAML),

        project=str(RUNS_DIR),

        name=EXPERIMENT_NAME,

        exist_ok=False,

        # ----------------------------------------------------
        # Core training parameters
        # ----------------------------------------------------

        epochs=EPOCHS,

        patience=PATIENCE,

        batch=BATCH_SIZE,

        imgsz=IMAGE_SIZE,

        device=DEVICE,

        workers=WORKERS,

        cache=False,

        # ----------------------------------------------------
        # Reproducibility
        # ----------------------------------------------------

        seed=SEED,

        deterministic=True,

        # ----------------------------------------------------
        # Model / optimizer
        # ----------------------------------------------------

        pretrained=True,

        optimizer="auto",

        amp=True,

        # Custom option in the project Ultralytics source.
        cls_remap=True,

        # ----------------------------------------------------
        # Optimization parameters
        #
        # Same values recorded by Exp01 args.yaml.
        # ----------------------------------------------------

        lr0=0.01,

        lrf=0.01,

        momentum=0.937,

        weight_decay=0.0005,

        warmup_epochs=3.0,

        warmup_momentum=0.8,

        warmup_bias_lr=0.1,

        nbs=64,

        # ----------------------------------------------------
        # Detection loss
        # ----------------------------------------------------

        box=7.5,

        cls=0.5,

        dfl=1.5,

        # ----------------------------------------------------
        # Geometric augmentation
        # ----------------------------------------------------

        degrees=0.0,

        translate=0.1,

        scale=0.5,

        shear=0.0,

        perspective=0.0,

        flipud=0.0,

        fliplr=0.5,

        # ----------------------------------------------------
        # Pixel / color augmentation
        #
        # IR supplied by the competition is stored as
        # 3-channel uint8 images.
        #
        # These parameters are intentionally retained from
        # Exp01 to preserve the controlled experiment.
        # ----------------------------------------------------

        hsv_h=0.015,

        hsv_s=0.7,

        hsv_v=0.4,

        bgr=0.0,

        # ----------------------------------------------------
        # Compound augmentation
        # ----------------------------------------------------

        mosaic=1.0,

        close_mosaic=10,

        mixup=0.0,

        cutmix=0.0,

        copy_paste=0.0,

        # ----------------------------------------------------
        # Other training options
        # ----------------------------------------------------

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
        # max_det remains 300 here because Exp01 training used
        # max_det=300. This preserves direct comparability of
        # training curves and best.pt selection.
        #
        # After training, a separate official-spec validation
        # will use max_det=100.
        # ----------------------------------------------------

        val=True,

        split="val",

        iou=0.70,

        max_det=300,

        augment=False,

        # ----------------------------------------------------
        # Logging / output
        # ----------------------------------------------------

        save=True,

        save_period=-1,

        plots=True,

        save_json=False,

        save_txt=False,

        verbose=True,
    )

    print()
    print("=" * 80)
    print("Exp02 training finished")
    print("=" * 80)

    print(
        "Results directory :",
        OUTPUT_DIR,
    )

    print(
        "Best weights      :",
        OUTPUT_DIR / "weights" / "best.pt",
    )

    print(
        "Last weights      :",
        OUTPUT_DIR / "weights" / "last.pt",
    )

    print()
    print(
        "Next step: run official-spec validation "
        "with max_det=100."
    )

    print("=" * 80)

    return results


if __name__ == "__main__":
    main()

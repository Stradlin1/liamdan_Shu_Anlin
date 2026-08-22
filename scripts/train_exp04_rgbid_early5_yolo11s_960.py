#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 Exp04
RGB + IR + Depth
5-channel Early Fusion
YOLO11s @ 960

Formal training experiment.

Input:
    [R, G, B, IR, Depth]

Model:
    YOLO11s
    5-channel input
    12-class Detect head

Initialization:
    RGB   <- COCO pretrained exactly
    IR    <- zero
    Depth <- zero

Formal configuration:
    imgsz       = 960
    batch       = 8
    epochs      = 200
    patience    = 50
    seed        = 2026
    mosaic      = 1.0
    close_mosaic= 10
    fliplr      = 0.5
    max_det     = 300

NOTE:
    max_det=300 here is for training/internal validation.

    Final official comparison must later use:

        imgsz   = 960
        conf    = 0.001
        iou     = 0.70
        max_det = 100
        val     = fixed 400 samples
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import ultralytics
from ultralytics import YOLO

from multimodal_dataset import (
    CHANNEL_NAMES,
    MULTIMODAL_CHANNELS,
)

from multimodal_trainer import (
    MultimodalDetectionTrainer,
)


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

PRETRAINED_MODEL_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "yolo11s.pt"
)

DATA_YAML = (
    PROJECT_ROOT
    / "yolo_views"
    / "rgb_v1"
    / "data_exp04_5ch.yaml"
)

RUNS_DIR = (
    PROJECT_ROOT
    / "runs"
)

EXPERIMENT_NAME = (
    "exp04_rgbid_early5_yolo11s_960"
)

RUN_DIR = (
    RUNS_DIR
    / EXPERIMENT_NAME
)


# ============================================================
# Formal experiment configuration
# ============================================================

IMAGE_SIZE = 960

BATCH_SIZE = 8

EPOCHS = 200

PATIENCE = 50

SEED = 2026

WORKERS = 4

MAX_DET_TRAIN_VAL = 300

EXPECTED_CHANNELS = 5

EXPECTED_CLASSES = 12


# RGB-only formal baseline
RGB_BASELINE_MAP5095 = 0.380843


# ============================================================
# Helpers
# ============================================================

def section(
    title: str,
) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def get_first_conv(
    model: nn.Module,
) -> nn.Conv2d:

    if not hasattr(
        model,
        "model",
    ):
        raise RuntimeError(
            "Model has no '.model' attribute."
        )

    first_block = model.model[0]

    if not hasattr(
        first_block,
        "conv",
    ):
        raise RuntimeError(
            "YOLO first block has no '.conv'."
        )

    conv = first_block.conv

    if not isinstance(
        conv,
        nn.Conv2d,
    ):
        raise TypeError(
            "YOLO first convolution is not nn.Conv2d."
        )

    return conv


# ============================================================
# Preflight
# ============================================================

def check_environment() -> None:

    section(
        "Exp04 Formal Training - Preflight"
    )

    if not PRETRAINED_MODEL_PATH.is_file():

        raise FileNotFoundError(
            "Missing pretrained model:\n"
            f"  {PRETRAINED_MODEL_PATH}"
        )

    if not DATA_YAML.is_file():

        raise FileNotFoundError(
            "Missing Exp04 dataset YAML:\n"
            f"  {DATA_YAML}"
        )

    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Never mix an old formal experiment with a new one.
    if RUN_DIR.exists():

        raise RuntimeError(
            "Formal experiment directory already exists:\n"
            f"  {RUN_DIR}\n\n"
            "The script stopped to prevent old/new runs "
            "from being mixed."
        )

    print(
        "Project root :",
        PROJECT_ROOT,
    )

    print(
        "Ultralytics  :",
        ultralytics.__file__,
    )

    print(
        "Version      :",
        getattr(
            ultralytics,
            "__version__",
            "unknown",
        ),
    )

    print(
        "Pretrained   :",
        PRETRAINED_MODEL_PATH,
    )

    print(
        "Dataset YAML :",
        DATA_YAML,
    )

    print(
        "Output       :",
        RUN_DIR,
    )

    print(
        "Input        :",
        CHANNEL_NAMES,
    )

    print(
        "Channels     :",
        MULTIMODAL_CHANNELS,
    )

    print(
        "Classes      :",
        EXPECTED_CLASSES,
    )

    print(
        "Image size   :",
        IMAGE_SIZE,
    )

    print(
        "Batch        :",
        BATCH_SIZE,
    )

    print(
        "Epochs       :",
        EPOCHS,
    )

    print(
        "Patience     :",
        PATIENCE,
    )

    print(
        "Seed         :",
        SEED,
    )

    print(
        "max_det      :",
        MAX_DET_TRAIN_VAL,
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is unavailable. "
            "Formal Exp04 training should not start on CPU."
        )

    print(
        "GPU          :",
        torch.cuda.get_device_name(
            0
        ),
    )

    # --------------------------------------------------------
    # Require project-local editable Ultralytics.
    # --------------------------------------------------------

    active_ultralytics = (
        Path(
            ultralytics.__file__
        ).resolve()
    )

    expected_root = (
        PROJECT_ROOT
        / "ultralytics"
    ).resolve()

    try:

        active_ultralytics.relative_to(
            expected_root
        )

    except ValueError as exc:

        raise RuntimeError(
            "Active Ultralytics is not the project-local "
            "editable source.\n\n"
            f"Active:\n  {active_ultralytics}\n"
            f"Expected under:\n  {expected_root}"
        ) from exc

    print(
        "[PASS] project-local editable Ultralytics"
    )


# ============================================================
# Trainer construction
# ============================================================

def build_trainer():

    overrides = {

        # ----------------------------------------------------
        # Model / Dataset
        # ----------------------------------------------------

        "model": str(
            PRETRAINED_MODEL_PATH
        ),

        "data": str(
            DATA_YAML
        ),

        "task": "detect",

        "pretrained": True,

        # Keep class-name based pretrained head remapping.
        "cls_remap": True,

        # ----------------------------------------------------
        # Formal training duration
        # ----------------------------------------------------

        "epochs": EPOCHS,

        "patience": PATIENCE,

        # ----------------------------------------------------
        # Geometry
        # ----------------------------------------------------

        "imgsz": IMAGE_SIZE,

        "batch": BATCH_SIZE,

        "rect": False,

        "multi_scale": 0.0,

        # ----------------------------------------------------
        # Device / DataLoader
        # ----------------------------------------------------

        "device": "0",

        "workers": WORKERS,

        "cache": False,

        # ----------------------------------------------------
        # Precision
        # ----------------------------------------------------

        "amp": True,

        # ----------------------------------------------------
        # Optimizer
        #
        # Keep Ultralytics auto optimizer behavior, consistent
        # with the validated Exp04 smoke pipeline.
        # ----------------------------------------------------

        "optimizer": "auto",

        # ----------------------------------------------------
        # Reproducibility
        # ----------------------------------------------------

        "seed": SEED,

        "deterministic": True,

        # ----------------------------------------------------
        # Augmentation
        #
        # MultimodalYOLODataset guarantees:
        #
        # RGB HSV   = ON
        # IR HSV    = OFF
        # Depth HSV = OFF
        #
        # Geometry transforms operate synchronously on all
        # five channels.
        # ----------------------------------------------------

        "mosaic": 1.0,

        "close_mosaic": 10,

        "fliplr": 0.5,

        "flipud": 0.0,

        "mixup": 0.0,

        "cutmix": 0.0,

        "copy_paste": 0.0,

        # ----------------------------------------------------
        # Validation during training
        # ----------------------------------------------------

        "val": True,

        "iou": 0.70,

        "max_det": MAX_DET_TRAIN_VAL,

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        "project": str(
            RUNS_DIR
        ),

        "name": EXPERIMENT_NAME,

        "exist_ok": False,

        "save": True,

        "save_period": -1,

        # Standard plot code is RGB-oriented.
        # Disable it for the 5-channel formal run.
        "plots": False,
    }

    return MultimodalDetectionTrainer(
        overrides=overrides
    )


# ============================================================
# Post-training verification
# ============================================================

def verify_best_checkpoint(
    trainer,
) -> None:

    section(
        "Exp04 Post-training Verification"
    )

    run_dir = (
        Path(
            trainer.save_dir
        ).resolve()
    )

    best_path = (
        run_dir
        / "weights"
        / "best.pt"
    )

    last_path = (
        run_dir
        / "weights"
        / "last.pt"
    )

    results_csv = (
        run_dir
        / "results.csv"
    )

    print(
        "Run directory:",
        run_dir,
    )

    for path in (
        best_path,
        last_path,
        results_csv,
    ):

        if not path.is_file():

            raise AssertionError(
                f"Required output missing: {path}"
            )

        print(
            "[PASS]",
            path.name,
        )

    # --------------------------------------------------------
    # Independent reload
    # --------------------------------------------------------

    best_wrapper = YOLO(
        str(
            best_path
        )
    )

    model = best_wrapper.model

    first_conv = (
        get_first_conv(
            model
        )
    )

    detect_head = (
        model.model[-1]
    )

    print()
    print(
        "best.pt first conv:",
        first_conv,
    )

    print(
        "best.pt weight shape:",
        tuple(
            first_conv.weight.shape
        ),
    )

    print(
        "best.pt Detect nc:",
        getattr(
            detect_head,
            "nc",
            None,
        ),
    )

    if (
        first_conv.in_channels
        != EXPECTED_CHANNELS
    ):

        raise AssertionError(
            "best.pt is not 5-channel."
        )

    if (
        getattr(
            detect_head,
            "nc",
            None,
        )
        != EXPECTED_CLASSES
    ):

        raise AssertionError(
            "best.pt is not 12-class."
        )

    print(
        "[PASS] best.pt 5-channel"
    )

    print(
        "[PASS] best.pt 12-class"
    )

    # --------------------------------------------------------
    # Auxiliary stem weights
    # --------------------------------------------------------

    weight = (
        first_conv
        .weight
        .detach()
        .cpu()
    )

    ir_weight = (
        weight[:, 3]
    )

    depth_weight = (
        weight[:, 4]
    )

    ir_nonzero = (
        torch.count_nonzero(
            ir_weight
        ).item()
    )

    depth_nonzero = (
        torch.count_nonzero(
            depth_weight
        ).item()
    )

    print()
    print(
        "IR stem:"
    )

    print(
        "  nonzero =",
        ir_nonzero,
    )

    print(
        "  abs sum =",
        ir_weight.abs().sum().item(),
    )

    print(
        "Depth stem:"
    )

    print(
        "  nonzero =",
        depth_nonzero,
    )

    print(
        "  abs sum =",
        depth_weight.abs().sum().item(),
    )

    if ir_nonzero == 0:

        raise AssertionError(
            "IR stem remained zero."
        )

    if depth_nonzero == 0:

        raise AssertionError(
            "Depth stem remained zero."
        )

    print(
        "[PASS] learned IR stem"
    )

    print(
        "[PASS] learned Depth stem"
    )


# ============================================================
# Metrics summary
# ============================================================

def print_training_metrics(
    trainer,
) -> None:

    section(
        "Internal Validation Summary"
    )

    metrics = (
        trainer.metrics
        if isinstance(
            trainer.metrics,
            dict,
        )
        else {}
    )

    precision = metrics.get(
        "metrics/precision(B)"
    )

    recall = metrics.get(
        "metrics/recall(B)"
    )

    map50 = metrics.get(
        "metrics/mAP50(B)"
    )

    map5095 = metrics.get(
        "metrics/mAP50-95(B)"
    )

    print(
        "Precision :",
        precision,
    )

    print(
        "Recall    :",
        recall,
    )

    print(
        "mAP50     :",
        map50,
    )

    print(
        "mAP50-95  :",
        map5095,
    )

    print()
    print(
        "RGB baseline official mAP50-95:",
        RGB_BASELINE_MAP5095,
    )

    if map5095 is not None:

        delta = (
            float(map5095)
            - RGB_BASELINE_MAP5095
        )

        print(
            "Internal-val delta vs RGB:",
            f"{delta:+.6f}",
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The value above uses training/internal validation "
        "with max_det=300."
    )

    print(
        "Do NOT declare Exp04 better/worse than RGB yet."
    )

    print(
        "Final comparison requires the separate official "
        "validation protocol:"
    )

    print(
        "imgsz=960, conf=0.001, iou=0.70, max_det=100."
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    check_environment()

    section(
        "Build Exp04 Formal Trainer"
    )

    trainer = (
        build_trainer()
    )

    print(
        "Trainer:",
        type(
            trainer
        ).__name__,
    )

    print(
        "Dataset channels:",
        trainer.data.get(
            "channels"
        ),
    )

    print(
        "Dataset classes:",
        trainer.data.get(
            "nc"
        ),
    )

    if (
        trainer.data.get(
            "channels"
        )
        != EXPECTED_CHANNELS
    ):

        raise AssertionError(
            "Dataset metadata channels != 5."
        )

    if (
        trainer.data.get(
            "nc"
        )
        != EXPECTED_CLASSES
    ):

        raise AssertionError(
            "Dataset metadata nc != 12."
        )

    section(
        "START EXP04 FORMAL TRAINING"
    )

    trainer.train()

    section(
        "EXP04 FORMAL TRAINING RETURNED"
    )

    verify_best_checkpoint(
        trainer
    )

    print_training_metrics(
        trainer
    )

    section(
        "FINAL RESULT"
    )

    print(
        "Formal training pipeline      PASS"
    )

    print(
        "5-channel model               PASS"
    )

    print(
        "12-class Detect head          PASS"
    )

    print(
        "200 epoch schedule            COMPLETE/STOPPED BY PATIENCE"
    )

    print(
        "best.pt                       PASS"
    )

    print(
        "last.pt                       PASS"
    )

    print(
        "IR stem learned               PASS"
    )

    print(
        "Depth stem learned            PASS"
    )

    print()
    print(
        "NEXT:"
    )

    print(
        "Official multimodal validation "
        "(conf=0.001, iou=0.70, max_det=100)"
    )


if __name__ == "__main__":
    main()

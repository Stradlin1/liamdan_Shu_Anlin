#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 Exp04b

No-Mosaic low-LR fine-tuning from the trained Exp04 best.pt.

Purpose:
    Exp04 formal training stopped before it completed a clean
    no-Mosaic convergence stage.

    Exp04b therefore starts from Exp04 best.pt and performs a
    short low-learning-rate fine-tune with Mosaic disabled.

Frozen:
    model structure
    5-channel [R,G,B,IR,Depth]
    train/val split
    imgsz = 960
    batch = 8
    all multimodal dataset logic

Changed:
    start weights = Exp04 best.pt
    mosaic        = 0
    close_mosaic  = 0
    optimizer     = MuSGD
    lr0           = 0.001
    epochs        = 25
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

EXP04_RUN_DIR = (
    PROJECT_ROOT
    / "runs"
    / "exp04_rgbid_early5_yolo11s_960"
)

START_WEIGHTS = (
    EXP04_RUN_DIR
    / "weights"
    / "best.pt"
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
    "exp04b_rgbid_early5_nomosaic_ft_960"
)

RUN_DIR = (
    RUNS_DIR
    / EXPERIMENT_NAME
)


# ============================================================
# Exp04b configuration
# ============================================================

IMAGE_SIZE = 960

BATCH_SIZE = 8

EPOCHS = 25

PATIENCE = 10

SEED = 2026

WORKERS = 4

EXPECTED_CHANNELS = 5

EXPECTED_CLASSES = 12

MAX_DET_TRAIN_VAL = 300


# ------------------------------------------------------------
# Fine-tuning optimizer
#
# Exp04 full training:
#     optimizer=auto
#     long run -> MuSGD(lr=0.01)
#
# Exp04b:
#     keep MuSGD explicitly
#     reduce initial LR by 10x
# ------------------------------------------------------------

OPTIMIZER = "MuSGD"

LR0 = 0.001

MOMENTUM = 0.9


# ============================================================
# Helpers
# ============================================================

def section(title: str) -> None:

    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


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
            "First YOLO block has no '.conv'."
        )

    conv = first_block.conv

    if not isinstance(
        conv,
        nn.Conv2d,
    ):
        raise TypeError(
            "First convolution is not nn.Conv2d."
        )

    return conv


# ============================================================
# Preflight
# ============================================================

def check_environment() -> None:

    section(
        "Exp04b No-Mosaic Fine-tune - Preflight"
    )

    if not START_WEIGHTS.is_file():

        raise FileNotFoundError(
            "Exp04 best.pt not found:\n"
            f"  {START_WEIGHTS}"
        )

    if not DATA_YAML.is_file():

        raise FileNotFoundError(
            "Dataset YAML not found:\n"
            f"  {DATA_YAML}"
        )

    if RUN_DIR.exists():

        raise RuntimeError(
            "Exp04b output directory already exists:\n"
            f"  {RUN_DIR}\n\n"
            "Stopped to prevent mixing old and new runs."
        )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is unavailable."
        )

    # --------------------------------------------------------
    # Require project-local editable Ultralytics.
    # --------------------------------------------------------

    active_ultralytics = (
        Path(
            ultralytics.__file__
        ).resolve()
    )

    expected_ultralytics = (
        PROJECT_ROOT
        / "ultralytics"
    ).resolve()

    try:

        active_ultralytics.relative_to(
            expected_ultralytics
        )

    except ValueError as exc:

        raise RuntimeError(
            "Active Ultralytics is not project-local.\n\n"
            f"Active:\n  {active_ultralytics}\n"
            f"Expected under:\n  {expected_ultralytics}"
        ) from exc

    # --------------------------------------------------------
    # Verify the START checkpoint itself.
    # --------------------------------------------------------

    wrapper = YOLO(
        str(
            START_WEIGHTS
        )
    )

    model = wrapper.model

    first_conv = get_first_conv(
        model
    )

    detect_head = (
        model.model[-1]
    )

    if (
        first_conv.in_channels
        != EXPECTED_CHANNELS
    ):

        raise RuntimeError(
            "Start checkpoint is not 5-channel.\n"
            f"in_channels = "
            f"{first_conv.in_channels}"
        )

    if (
        getattr(
            detect_head,
            "nc",
            None,
        )
        != EXPECTED_CLASSES
    ):

        raise RuntimeError(
            "Start checkpoint is not 12-class.\n"
            f"nc = "
            f"{getattr(detect_head, 'nc', None)}"
        )

    # Confirm learned auxiliary weights are really present.

    weight = (
        first_conv
        .weight
        .detach()
        .cpu()
    )

    ir_abs_sum = (
        weight[:, 3]
        .abs()
        .sum()
        .item()
    )

    depth_abs_sum = (
        weight[:, 4]
        .abs()
        .sum()
        .item()
    )

    if ir_abs_sum == 0:

        raise RuntimeError(
            "IR channel weights are still zero."
        )

    if depth_abs_sum == 0:

        raise RuntimeError(
            "Depth channel weights are still zero."
        )

    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Project root     :",
        PROJECT_ROOT,
    )

    print(
        "Ultralytics      :",
        active_ultralytics,
    )

    print(
        "Start weights    :",
        START_WEIGHTS,
    )

    print(
        "Dataset          :",
        DATA_YAML,
    )

    print(
        "Output           :",
        RUN_DIR,
    )

    print(
        "Input            :",
        CHANNEL_NAMES,
    )

    print(
        "Channels         :",
        MULTIMODAL_CHANNELS,
    )

    print(
        "Image size       :",
        IMAGE_SIZE,
    )

    print(
        "Batch            :",
        BATCH_SIZE,
    )

    print(
        "Epochs           :",
        EPOCHS,
    )

    print(
        "Patience         :",
        PATIENCE,
    )

    print(
        "Mosaic           : 0.0"
    )

    print(
        "close_mosaic     : 0"
    )

    print(
        "Optimizer        :",
        OPTIMIZER,
    )

    print(
        "lr0              :",
        LR0,
    )

    print(
        "Momentum         :",
        MOMENTUM,
    )

    print(
        "IR weight abs sum:",
        ir_abs_sum,
    )

    print(
        "D weight abs sum :",
        depth_abs_sum,
    )

    print(
        "GPU              :",
        torch.cuda.get_device_name(
            0
        ),
    )

    print()
    print(
        "[PASS] Exp04 best.pt is a learned "
        "5-channel checkpoint."
    )


# ============================================================
# Trainer
# ============================================================

def build_trainer():

    overrides = {

        # ----------------------------------------------------
        # Start from trained Exp04 5-channel best.pt
        # ----------------------------------------------------

        "model": str(
            START_WEIGHTS
        ),

        "data": str(
            DATA_YAML
        ),

        "task": "detect",

        "pretrained": True,

        # IMPORTANT:
        # New fine-tuning experiment.
        # Do not restore old optimizer / old epoch.
        "resume": False,

        "cls_remap": True,

        # ----------------------------------------------------
        # Fine-tune duration
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
        # Device / loader
        # ----------------------------------------------------

        "device": "0",

        "workers": WORKERS,

        "cache": False,

        "amp": True,

        # ----------------------------------------------------
        # Optimizer
        #
        # Do NOT use optimizer=auto here.
        #
        # Exp04's long formal run used MuSGD through auto.
        # A short Exp04b run would otherwise switch auto
        # optimizer to AdamW.
        # ----------------------------------------------------

        "optimizer": OPTIMIZER,

        "lr0": LR0,

        "momentum": MOMENTUM,

        # ----------------------------------------------------
        # Reproducibility
        # ----------------------------------------------------

        "seed": SEED,

        "deterministic": True,

        # ----------------------------------------------------
        # Augmentation
        #
        # Main experimental variable:
        #
        #     Mosaic OFF
        #
        # All five channels still go through the same spatial
        # transforms in MultimodalYOLODataset.
        # ----------------------------------------------------

        "mosaic": 0.0,

        "close_mosaic": 0,

        "fliplr": 0.5,

        "flipud": 0.0,

        "mixup": 0.0,

        "cutmix": 0.0,

        "copy_paste": 0.0,

        # ----------------------------------------------------
        # Internal validation
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

        "plots": False,
    }

    return MultimodalDetectionTrainer(
        overrides=overrides
    )


# ============================================================
# Post-training verification
# ============================================================

def verify_outputs(
    trainer,
) -> None:

    section(
        "Exp04b Post-training Verification"
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

    for path in (
        best_path,
        last_path,
        results_csv,
    ):

        if not path.is_file():

            raise RuntimeError(
                f"Required output missing: {path}"
            )

        print(
            "[PASS]",
            path,
        )

    wrapper = YOLO(
        str(
            best_path
        )
    )

    model = wrapper.model

    first_conv = get_first_conv(
        model
    )

    detect_head = (
        model.model[-1]
    )

    if (
        first_conv.in_channels
        != EXPECTED_CHANNELS
    ):

        raise RuntimeError(
            "Exp04b best.pt is not 5-channel."
        )

    if (
        getattr(
            detect_head,
            "nc",
            None,
        )
        != EXPECTED_CLASSES
    ):

        raise RuntimeError(
            "Exp04b best.pt is not 12-class."
        )

    weight = (
        first_conv
        .weight
        .detach()
        .cpu()
    )

    print()
    print(
        "Best checkpoint :",
        best_path,
    )

    print(
        "IR abs sum      :",
        weight[:, 3]
        .abs()
        .sum()
        .item(),
    )

    print(
        "Depth abs sum   :",
        weight[:, 4]
        .abs()
        .sum()
        .item(),
    )

    print()
    print(
        "[PASS] Exp04b best.pt "
        "remains 5-channel / 12-class."
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    check_environment()

    section(
        "Build Exp04b Trainer"
    )

    trainer = build_trainer()

    print(
        "Trainer:",
        type(
            trainer
        ).__name__,
    )

    section(
        "Start Exp04b Training"
    )

    trainer.train()

    verify_outputs(
        trainer
    )

    section(
        "FINAL"
    )

    print(
        "Exp04b training finished."
    )

    print()

    print(
        "DO NOT submit last.pt directly."
    )

    print(
        "Next:"
    )

    print(
        "1. fixed val400 official validation"
    )

    print(
        "2. rect inference parity"
    )

    print(
        "3. compare with Exp04 rect "
        "mAP50-95 = 0.380711"
    )

    print(
        "4. only then decide whether "
        "to submit to leaderboard"
    )

    print()

    print(
        "STATUS = PASS"
    )


if __name__ == "__main__":
    main()

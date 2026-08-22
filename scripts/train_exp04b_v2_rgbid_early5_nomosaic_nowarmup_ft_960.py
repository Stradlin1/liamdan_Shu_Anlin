#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 Exp04b-v2

5-channel RGB + IR + Depth Early Fusion
No-Mosaic + No-Warmup low-LR fine-tuning.

Start:
    Exp04 best.pt

Purpose:
    Exp04b-v1 used:
        warmup_epochs  = 3.0
        warmup_bias_lr = 0.1

    Therefore several bias/head parameter groups temporarily received
    learning rates far above the intended low fine-tuning LR.

    Exp04b-v2 removes warmup completely.

Frozen relative to Exp04b-v1:
    start checkpoint = Exp04 best.pt
    model structure  = same 5-channel Early Fusion
    train/val split  = same
    imgsz            = 960
    batch             = 8
    optimizer         = MuSGD
    lr0               = 0.001
    momentum          = 0.9
    mosaic            = 0
    epochs            = 25
    patience          = 10

Changed:
    warmup_epochs     = 0.0
    warmup_bias_lr    = 0.0
"""

from __future__ import annotations

import argparse
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
    "exp04b_v2_rgbid_early5_nomosaic_nowarmup_ft_960"
)

RUN_DIR = (
    RUNS_DIR
    / EXPERIMENT_NAME
)


# ============================================================
# Experiment configuration
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


# ============================================================
# Fine-tuning optimizer
# ============================================================

OPTIMIZER = "MuSGD"

LR0 = 0.001

MOMENTUM = 0.9

# Keep the standard linear schedule explicit.
LRF = 0.01

COS_LR = False


# ============================================================
# Critical v2 change
# ============================================================

WARMUP_EPOCHS = 0.0

WARMUP_BIAS_LR = 0.0


# ============================================================
# References
# ============================================================

EXP04_OFFICIAL_MAP5095 = 0.379090

EXP04_RECT_MANUAL_MAP5095 = 0.380711

EXP04_LEADERBOARD = 45.614


# ============================================================
# Helpers
# ============================================================

def section(
    title: str,
) -> None:

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

    if len(
        model.model
    ) == 0:
        raise RuntimeError(
            "Model contains no layers."
        )

    first_block = (
        model.model[0]
    )

    if not hasattr(
        first_block,
        "conv",
    ):
        raise RuntimeError(
            "First YOLO block has no '.conv'."
        )

    conv = (
        first_block.conv
    )

    if not isinstance(
        conv,
        nn.Conv2d,
    ):
        raise TypeError(
            "First convolution is not nn.Conv2d."
        )

    return conv


# ============================================================
# Trainer subclass
# ============================================================

class Exp04bV2Trainer(
    MultimodalDetectionTrainer
):
    """
    Same multimodal trainer as Exp04.

    Only adds optimizer sanity checks so this formal experiment
    cannot silently switch optimizer or start with an unexpectedly
    large LR.
    """

    def build_optimizer(
        self,
        model,
        name="auto",
        lr=0.001,
        momentum=0.9,
        decay=1e-5,
        iterations=1e5,
    ):

        optimizer = super().build_optimizer(
            model=model,
            name=name,
            lr=lr,
            momentum=momentum,
            decay=decay,
            iterations=iterations,
        )

        optimizer_name = (
            type(
                optimizer
            ).__name__
        )

        if optimizer_name != OPTIMIZER:

            raise RuntimeError(
                "Unexpected optimizer.\n"
                f"Expected = {OPTIMIZER}\n"
                f"Actual   = {optimizer_name}"
            )

        lrs = [
            float(
                group["lr"]
            )
            for group
            in optimizer.param_groups
        ]

        unique_lrs = sorted(
            {
                round(
                    value,
                    12,
                )
                for value
                in lrs
            }
        )

        # MuSGD in this Ultralytics source may apply 3x LR
        # to selected detection-head parameters.
        maximum_expected_lr = (
            LR0
            * 3.0
            * 1.001
        )

        if max(
            lrs
        ) > maximum_expected_lr:

            raise RuntimeError(
                "Unexpected initial optimizer LR.\n"
                f"LR groups = {unique_lrs}\n"
                f"Maximum expected ~= {LR0 * 3.0}"
            )

        section(
            "Exp04b-v2 Optimizer Audit"
        )

        print(
            "Optimizer       :",
            optimizer_name,
        )

        print(
            "Configured lr0  :",
            LR0,
        )

        print(
            "Initial LR groups:",
            unique_lrs,
        )

        print(
            "Warmup epochs   :",
            self.args.warmup_epochs,
        )

        print(
            "Warmup bias LR  :",
            self.args.warmup_bias_lr,
        )

        if float(
            self.args.warmup_epochs
        ) != 0.0:

            raise RuntimeError(
                "warmup_epochs is not zero."
            )

        if float(
            self.args.warmup_bias_lr
        ) != 0.0:

            raise RuntimeError(
                "warmup_bias_lr is not zero."
            )

        print()
        print(
            "[PASS] MuSGD low-LR / no-warmup configuration"
        )

        return optimizer


# ============================================================
# Preflight
# ============================================================

def check_environment() -> None:

    section(
        "Exp04b-v2 Preflight"
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
            "Exp04b-v2 run directory already exists:\n"
            f"  {RUN_DIR}\n\n"
            "Delete/rename it only if you intentionally "
            "want to rerun this experiment."
        )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is unavailable."
        )

    # --------------------------------------------------------
    # Require project-local editable Ultralytics
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
            f"Active:\n"
            f"  {active_ultralytics}\n\n"
            f"Expected under:\n"
            f"  {expected_ultralytics}"
        ) from exc

    # --------------------------------------------------------
    # Verify Exp04 checkpoint
    # --------------------------------------------------------

    wrapper = YOLO(
        str(
            START_WEIGHTS
        )
    )

    model = (
        wrapper.model
    )

    first_conv = (
        get_first_conv(
            model
        )
    )

    detect_head = (
        model.model[-1]
    )

    if (
        first_conv.in_channels
        != EXPECTED_CHANNELS
    ):

        raise RuntimeError(
            "Exp04 best.pt is not 5-channel.\n"
            f"in_channels = "
            f"{first_conv.in_channels}"
        )

    nc = getattr(
        detect_head,
        "nc",
        None,
    )

    if nc != EXPECTED_CLASSES:

        raise RuntimeError(
            "Exp04 best.pt class count is wrong.\n"
            f"Expected = {EXPECTED_CLASSES}\n"
            f"Actual   = {nc}"
        )

    weight = (
        first_conv
        .weight
        .detach()
        .float()
        .cpu()
    )

    ir_abs_sum = float(
        weight[
            :,
            3,
        ]
        .abs()
        .sum()
        .item()
    )

    depth_abs_sum = float(
        weight[
            :,
            4,
        ]
        .abs()
        .sum()
        .item()
    )

    if ir_abs_sum <= 0:

        raise RuntimeError(
            "IR first-conv weights are zero."
        )

    if depth_abs_sum <= 0:

        raise RuntimeError(
            "Depth first-conv weights are zero."
        )

    print(
        "Project root      :",
        PROJECT_ROOT,
    )

    print(
        "Ultralytics       :",
        active_ultralytics,
    )

    print(
        "Start weights     :",
        START_WEIGHTS,
    )

    print(
        "Dataset YAML      :",
        DATA_YAML,
    )

    print(
        "Output            :",
        RUN_DIR,
    )

    print()

    print(
        "Channels          :",
        CHANNEL_NAMES,
    )

    print(
        "Input channels    :",
        MULTIMODAL_CHANNELS,
    )

    print(
        "Classes           :",
        EXPECTED_CLASSES,
    )

    print()

    print(
        "Image size        :",
        IMAGE_SIZE,
    )

    print(
        "Batch             :",
        BATCH_SIZE,
    )

    print(
        "Epochs            :",
        EPOCHS,
    )

    print(
        "Patience          :",
        PATIENCE,
    )

    print()

    print(
        "Mosaic            : 0.0"
    )

    print(
        "close_mosaic      : 0"
    )

    print()

    print(
        "Optimizer         :",
        OPTIMIZER,
    )

    print(
        "lr0               :",
        LR0,
    )

    print(
        "momentum          :",
        MOMENTUM,
    )

    print(
        "lrf               :",
        LRF,
    )

    print(
        "cos_lr            :",
        COS_LR,
    )

    print()

    print(
        "warmup_epochs     :",
        WARMUP_EPOCHS,
    )

    print(
        "warmup_bias_lr    :",
        WARMUP_BIAS_LR,
    )

    print()

    print(
        "IR weight abs sum :",
        ir_abs_sum,
    )

    print(
        "D weight abs sum  :",
        depth_abs_sum,
    )

    print()

    print(
        "GPU               :",
        torch.cuda.get_device_name(
            0
        ),
    )

    print()

    print(
        "[PASS] Exp04 learned 5-channel checkpoint"
    )

    print(
        "[PASS] no-Mosaic configuration"
    )

    print(
        "[PASS] no-warmup configuration"
    )


# ============================================================
# Trainer construction
# ============================================================

def build_trainer():

    overrides = {

        # ----------------------------------------------------
        # Model / dataset
        # ----------------------------------------------------

        "model": str(
            START_WEIGHTS
        ),

        "data": str(
            DATA_YAML
        ),

        "task": "detect",

        # New experiment initialized from best.pt.
        # This is NOT optimizer-state resume.
        "pretrained": True,

        "resume": False,

        "cls_remap": True,

        # ----------------------------------------------------
        # Duration
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
        # Explicit optimizer
        # ----------------------------------------------------

        "optimizer": OPTIMIZER,

        "lr0": LR0,

        "momentum": MOMENTUM,

        "lrf": LRF,

        "cos_lr": COS_LR,

        # ----------------------------------------------------
        # Critical v2 change:
        # absolutely no warmup
        # ----------------------------------------------------

        "warmup_epochs": WARMUP_EPOCHS,

        "warmup_bias_lr": WARMUP_BIAS_LR,

        # Irrelevant when warmup_epochs=0, but explicitly keep
        # the target momentum consistent.
        "warmup_momentum": MOMENTUM,

        # ----------------------------------------------------
        # Reproducibility
        # ----------------------------------------------------

        "seed": SEED,

        "deterministic": True,

        # ----------------------------------------------------
        # Augmentation
        # ----------------------------------------------------

        "mosaic": 0.0,

        "close_mosaic": 0,

        "fliplr": 0.5,

        "flipud": 0.0,

        "mixup": 0.0,

        "cutmix": 0.0,

        "copy_paste": 0.0,

        # MultimodalYOLODataset continues to perform RGB-only
        # HSV while leaving IR and Depth unchanged.

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

    trainer = (
        Exp04bV2Trainer(
            overrides=overrides
        )
    )

    # Static sanity checks before training begins.
    assert (
        float(
            trainer.args.warmup_epochs
        )
        == 0.0
    )

    assert (
        float(
            trainer.args.warmup_bias_lr
        )
        == 0.0
    )

    assert (
        float(
            trainer.args.mosaic
        )
        == 0.0
    )

    assert (
        str(
            trainer.args.optimizer
        )
        == OPTIMIZER
    )

    return trainer


# ============================================================
# Post-training verification
# ============================================================

def verify_outputs(
    trainer,
) -> None:

    section(
        "Exp04b-v2 Post-training Verification"
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

    model = (
        wrapper.model
    )

    first_conv = (
        get_first_conv(
            model
        )
    )

    detect_head = (
        model.model[-1]
    )

    if (
        first_conv.in_channels
        != EXPECTED_CHANNELS
    ):

        raise RuntimeError(
            "Exp04b-v2 best.pt is not 5-channel."
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
            "Exp04b-v2 best.pt is not 12-class."
        )

    weight = (
        first_conv
        .weight
        .detach()
        .float()
        .cpu()
    )

    ir_abs_sum = float(
        weight[
            :,
            3,
        ]
        .abs()
        .sum()
        .item()
    )

    depth_abs_sum = float(
        weight[
            :,
            4,
        ]
        .abs()
        .sum()
        .item()
    )

    if (
        ir_abs_sum <= 0
        or depth_abs_sum <= 0
    ):

        raise RuntimeError(
            "Auxiliary modality weights are invalid."
        )

    print()

    print(
        "Best checkpoint :",
        best_path,
    )

    print(
        "IR abs sum      :",
        ir_abs_sum,
    )

    print(
        "Depth abs sum   :",
        depth_abs_sum,
    )

    print()

    print(
        "[PASS] Exp04b-v2 best.pt is "
        "5-channel / 12-class."
    )


# ============================================================
# Final internal metric summary
# ============================================================

def print_internal_metrics(
    trainer,
) -> None:

    section(
        "Exp04b-v2 Internal Validation"
    )

    metrics = (
        trainer.metrics
        if isinstance(
            trainer.metrics,
            dict,
        )
        else {}
    )

    for key in (
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ):

        print(
            f"{key:24s}:",
            metrics.get(
                key
            ),
        )

    print()
    print(
        "Reference only:"
    )

    print(
        "Exp04 Official val mAP50-95 :",
        EXP04_OFFICIAL_MAP5095,
    )

    print(
        "Exp04 manual Rect mAP50-95  :",
        EXP04_RECT_MANUAL_MAP5095,
    )

    print(
        "Exp04 leaderboard           :",
        EXP04_LEADERBOARD,
    )

    print()
    print(
        "Do NOT compare leaderboard performance "
        "from this internal metric alone."
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Exp04b-v2 no-Mosaic / "
            "no-warmup low-LR fine-tuning."
        )
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Run preflight checks only; "
            "do not create trainer or start training."
        ),
    )

    args = parser.parse_args()

    check_environment()

    if args.check_only:

        section(
            "CHECK ONLY"
        )

        print(
            "STATUS = PASS"
        )

        return

    section(
        "Build Exp04b-v2 Trainer"
    )

    trainer = (
        build_trainer()
    )

    print(
        "Trainer          :",
        type(
            trainer
        ).__name__,
    )

    print(
        "warmup_epochs    :",
        trainer.args.warmup_epochs,
    )

    print(
        "warmup_bias_lr   :",
        trainer.args.warmup_bias_lr,
    )

    print(
        "optimizer        :",
        trainer.args.optimizer,
    )

    print(
        "lr0              :",
        trainer.args.lr0,
    )

    print(
        "mosaic           :",
        trainer.args.mosaic,
    )

    section(
        "Start Exp04b-v2 Training"
    )

    trainer.train()

    verify_outputs(
        trainer
    )

    print_internal_metrics(
        trainer
    )

    section(
        "FINAL"
    )

    print(
        "Exp04b-v2 training completed."
    )

    print()

    print(
        "DO NOT submit best.pt directly."
    )

    print()

    print(
        "Next mandatory steps:"
    )

    print(
        "1. fixed val400 validation"
    )

    print(
        "2. manual Rect inference parity"
    )

    print(
        "3. compare against Exp04 Rect "
        "0.380711"
    )

    print(
        "4. only submit if the improvement "
        "is meaningful"
    )

    print()

    print(
        "STATUS = PASS"
    )


if __name__ == "__main__":
    main()

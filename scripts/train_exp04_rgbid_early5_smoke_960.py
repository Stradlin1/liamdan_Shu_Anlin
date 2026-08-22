#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 Exp04
RGB + IR + Depth 5-channel Early Fusion
Full training-chain smoke test.

Purpose:
    Verify the complete training pipeline:

        MultimodalYOLODataset
                ↓
        DataLoader
                ↓
        5-channel / 12-class YOLO11s
                ↓
        Detection Loss
                ↓
        Backward
                ↓
        Optimizer Step
                ↓
        Validation
                ↓
        last.pt / best.pt

This is NOT the formal Exp04 experiment.

Smoke configuration:
    epochs = 2
    imgsz = 960
    batch = 8

Input channel order:
    [R, G, B, IR, Depth]
"""

from __future__ import annotations

import csv
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
    "exp04_rgbid_early5_smoke_960"
)

RUN_DIR = (
    RUNS_DIR
    / EXPERIMENT_NAME
)


# ============================================================
# Smoke configuration
# ============================================================

IMAGE_SIZE = 960

BATCH_SIZE = 8

EPOCHS = 2

SEED = 2026

EXPECTED_CLASSES = 12
EXPECTED_CHANNELS = 5


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

    first_block = (
        model.model[0]
    )

    if not hasattr(
        first_block,
        "conv",
    ):
        raise RuntimeError(
            "YOLO first block has no '.conv'."
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
# Preflight
# ============================================================

def check_environment() -> None:

    section(
        "Preflight"
    )

    if not PRETRAINED_MODEL_PATH.is_file():

        raise FileNotFoundError(
            "Missing pretrained model:\n"
            f"  {PRETRAINED_MODEL_PATH}"
        )

    if not DATA_YAML.is_file():

        raise FileNotFoundError(
            "Missing dataset YAML:\n"
            f"  {DATA_YAML}"
        )

    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Do not silently mix an old smoke experiment with a new
    # one. Delete/rename it manually if a rerun is intended.
    # --------------------------------------------------------

    if RUN_DIR.exists():

        raise RuntimeError(
            "Smoke output directory already exists:\n"
            f"  {RUN_DIR}\n\n"
            "To prevent old/new experiment files from mixing, "
            "the script stopped.\n"
            "Rename or remove that directory before rerunning."
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
        "Pretrained   :",
        PRETRAINED_MODEL_PATH,
    )

    print(
        "Dataset      :",
        DATA_YAML,
    )

    print(
        "Output       :",
        RUN_DIR,
    )

    print(
        "Channels     :",
        MULTIMODAL_CHANNELS,
    )

    print(
        "Order        :",
        CHANNEL_NAMES,
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

    if torch.cuda.is_available():

        print(
            "GPU          :",
            torch.cuda.get_device_name(
                0
            ),
        )

    else:

        print(
            "GPU          : CUDA unavailable"
        )

    # --------------------------------------------------------
    # Verify editable project-local Ultralytics.
    # --------------------------------------------------------

    installed_ultralytics = (
        Path(
            ultralytics.__file__
        )
        .resolve()
    )

    expected_source_root = (
        PROJECT_ROOT
        / "ultralytics"
    ).resolve()

    try:

        installed_ultralytics.relative_to(
            expected_source_root
        )

    except ValueError as exc:

        raise RuntimeError(
            "The active Ultralytics package is NOT the "
            "project-local editable source.\n\n"
            f"Active:\n  {installed_ultralytics}\n"
            f"Expected under:\n  {expected_source_root}"
        ) from exc

    print(
        "[PASS] project-local editable Ultralytics"
    )


# ============================================================
# Trainer
# ============================================================

def build_trainer():
    """
    Build the real Exp04 smoke trainer.
    """

    device = (
        "0"
        if torch.cuda.is_available()
        else "cpu"
    )

    overrides = {

        # ----------------------------------------------------
        # Model / data
        # ----------------------------------------------------

        "model": str(
            PRETRAINED_MODEL_PATH
        ),

        "data": str(
            DATA_YAML
        ),

        "task": "detect",

        # ----------------------------------------------------
        # Smoke duration
        # ----------------------------------------------------

        "epochs": EPOCHS,

        "patience": 50,

        # ----------------------------------------------------
        # Baseline-compatible geometry
        # ----------------------------------------------------

        "imgsz": IMAGE_SIZE,

        "batch": BATCH_SIZE,

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        "device": device,

        "workers": 4,

        # ----------------------------------------------------
        # Reproducibility
        # ----------------------------------------------------

        "seed": SEED,

        "deterministic": True,

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

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        "val": True,

        # Internal training validation keeps normal YOLO
        # maximum detections. Official competition validation
        # will later be performed separately with max_det=100.
        "max_det": 300,

        # ----------------------------------------------------
        # Training precision
        #
        # Formal training is expected to use AMP, so smoke
        # should test AMP as part of the real pipeline.
        # ----------------------------------------------------

        "amp": True,

        # ----------------------------------------------------
        # Data
        # ----------------------------------------------------

        "cache": False,

        "rect": False,

        # ----------------------------------------------------
        # Exp01-compatible augmentation policy
        #
        # MultimodalYOLODataset itself ensures HSV only acts
        # on RGB and that geometry remains synchronized across
        # all 5 channels.
        # ----------------------------------------------------

        "mosaic": 1.0,

        "fliplr": 0.5,

        "mixup": 0.0,

        "cutmix": 0.0,

        "copy_paste": 0.0,

        # For a 2-epoch smoke test we intentionally do NOT
        # close Mosaic early. Formal Exp04 will restore:
        #
        #     close_mosaic = 10
        #
        "close_mosaic": 0,

        # ----------------------------------------------------
        # Keep smoke simple / deterministic
        # ----------------------------------------------------

        "multi_scale": 0.0,

        # 5-channel training plots are not needed for this
        # engineering test and may assume RGB visualization.
        "plots": False,
    }

    return MultimodalDetectionTrainer(
        overrides=overrides
    )


# ============================================================
# Post-training checkpoint verification
# ============================================================

def verify_training_outputs(
    trainer,
) -> None:

    section(
        "Post-training verification"
    )

    actual_run_dir = (
        Path(
            trainer.save_dir
        )
        .resolve()
    )

    print(
        "Actual run dir:",
        actual_run_dir,
    )

    weights_dir = (
        actual_run_dir
        / "weights"
    )

    best_path = (
        weights_dir
        / "best.pt"
    )

    last_path = (
        weights_dir
        / "last.pt"
    )

    results_csv = (
        actual_run_dir
        / "results.csv"
    )

    # --------------------------------------------------------
    # Required artifacts
    # --------------------------------------------------------

    if not last_path.is_file():

        raise AssertionError(
            f"last.pt missing: {last_path}"
        )

    print(
        "[PASS] last.pt exists"
    )

    if not best_path.is_file():

        raise AssertionError(
            f"best.pt missing: {best_path}"
        )

    print(
        "[PASS] best.pt exists"
    )

    if not results_csv.is_file():

        raise AssertionError(
            f"results.csv missing: {results_csv}"
        )

    print(
        "[PASS] results.csv exists"
    )

    # --------------------------------------------------------
    # Verify that two epochs actually completed.
    # --------------------------------------------------------

    with results_csv.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        rows = list(
            csv.DictReader(
                f
            )
        )

    print(
        "results.csv epoch rows:",
        len(
            rows
        ),
    )

    if len(rows) < EPOCHS:

        raise AssertionError(
            "Training did not record the expected "
            f"{EPOCHS} epochs."
        )

    print(
        f"[PASS] completed >= {EPOCHS} epochs"
    )

    # --------------------------------------------------------
    # Reload best.pt independently.
    # --------------------------------------------------------

    section(
        "Reload best.pt"
    )

    best_wrapper = YOLO(
        str(
            best_path
        )
    )

    best_model = (
        best_wrapper.model
    )

    first_conv = (
        get_first_conv(
            best_model
        )
    )

    detect_head = (
        best_model.model[-1]
    )

    print(
        "Reloaded first conv:",
        first_conv,
    )

    print(
        "Reloaded first weight shape:",
        tuple(
            first_conv.weight.shape
        ),
    )

    print(
        "Reloaded Detect nc:",
        getattr(
            detect_head,
            "nc",
            None,
        ),
    )

    # --------------------------------------------------------
    # Check model structure survived checkpoint save/reload.
    # --------------------------------------------------------

    if (
        first_conv.in_channels
        != EXPECTED_CHANNELS
    ):

        raise AssertionError(
            "Reloaded best.pt is not 5-channel."
        )

    print(
        "[PASS] best.pt input channels == 5"
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
            "Reloaded best.pt is not 12-class."
        )

    print(
        "[PASS] best.pt Detect nc == 12"
    )

    # --------------------------------------------------------
    # The auxiliary-channel weights started exactly at zero.
    #
    # After actual optimizer steps they should now be nonzero.
    # This verifies:
    #
    #     gradient
    #       ↓
    #     optimizer update
    #       ↓
    #     checkpoint persistence
    # --------------------------------------------------------

    weight = (
        first_conv
        .weight
        .detach()
        .cpu()
    )

    ir_nonzero = (
        torch.count_nonzero(
            weight[:, 3]
        ).item()
    )

    depth_nonzero = (
        torch.count_nonzero(
            weight[:, 4]
        ).item()
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

    print()
    print(
        "Reloaded IR stem:"
    )

    print(
        "  nonzero =",
        ir_nonzero,
    )

    print(
        "  abs sum =",
        ir_abs_sum,
    )

    print()
    print(
        "Reloaded Depth stem:"
    )

    print(
        "  nonzero =",
        depth_nonzero,
    )

    print(
        "  abs sum =",
        depth_abs_sum,
    )

    if ir_nonzero == 0:

        raise AssertionError(
            "IR weights are still entirely zero after "
            "actual training."
        )

    if depth_nonzero == 0:

        raise AssertionError(
            "Depth weights are still entirely zero after "
            "actual training."
        )

    print(
        "[PASS] optimizer learned nonzero IR weights"
    )

    print(
        "[PASS] optimizer learned nonzero Depth weights"
    )

    # --------------------------------------------------------
    # Print final validation metrics if available.
    # Smoke values are NOT performance results.
    # --------------------------------------------------------

    print()
    print(
        "Trainer final metrics:"
    )

    print(
        trainer.metrics
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    check_environment()

    section(
        "Build MultimodalDetectionTrainer"
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
        "Data channels:",
        trainer.data.get(
            "channels"
        ),
    )

    print(
        "Data classes:",
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
            "Trainer metadata is not 5-channel."
        )

    if (
        trainer.data.get(
            "nc"
        )
        != EXPECTED_CLASSES
    ):

        raise AssertionError(
            "Trainer metadata is not 12-class."
        )

    section(
        "START EXP04 SMOKE TRAINING"
    )

    trainer.train()

    section(
        "TRAINING RETURNED SUCCESSFULLY"
    )

    verify_training_outputs(
        trainer
    )

    section(
        "FINAL RESULT"
    )

    print(
        "Dataset -> DataLoader             PASS"
    )

    print(
        "5-channel model                   PASS"
    )

    print(
        "12-class Detect head              PASS"
    )

    print(
        "Loss -> Backward                  PASS"
    )

    print(
        "Optimizer step                    PASS"
    )

    print(
        "AMP training                      PASS"
    )

    print(
        "Validation                        PASS"
    )

    print(
        "results.csv                       PASS"
    )

    print(
        "last.pt                           PASS"
    )

    print(
        "best.pt                           PASS"
    )

    print(
        "best.pt reload                    PASS"
    )

    print(
        "IR weights learned                PASS"
    )

    print(
        "Depth weights learned             PASS"
    )

    print()
    print(
        "STATUS = PASS"
    )

    print()
    print(
        "Next step:"
    )

    print(
        "train_exp04_rgbid_early5_yolo11s_960.py"
    )


if __name__ == "__main__":
    main()

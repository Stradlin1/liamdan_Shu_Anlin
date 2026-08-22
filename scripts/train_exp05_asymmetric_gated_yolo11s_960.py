#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 Exp05
RGB-dominant Asymmetric Gated Feature Fusion
YOLO11s @ 960

Formal training experiment.

Input
=====
    [R, G, B, IR, Depth]

Architecture
============
RGB:
    Original pretrained YOLO11s main path.

Depth:
    1-channel encoder derived from RGB pretrained backbone
        ↓
    P3/8
        ↓
    channel-wise sigmoid gate
        ↓
    residual fusion after backbone layer 4

IR:
    1-channel encoder derived from RGB pretrained backbone
        ↓
    P3/8
        ↓
    AvgPool2d
        ↓
    identity-initialized 1x1 projection + BN
        ↓
    P4/16
        ↓
    channel-wise sigmoid gate
        ↓
    residual fusion after backbone layer 6

No separate P5 fusion.

Gate initialization
===================
    Depth = 0.10
    IR    = 0.02

Formal configuration
====================
    imgsz        = 960
    batch        = 8
    epochs       = 200
    patience     = 50
    seed         = 2026

    optimizer    = auto
    AMP          = True

    mosaic       = 1.0
    close_mosaic = 10
    fliplr       = 0.5

    train        = full fixed 1600
    val          = full fixed 400

Internal validation
===================
    iou     = 0.70
    max_det = 300

IMPORTANT
=========
Internal training validation is NOT the final official comparison.

After training, Exp05 must be evaluated separately with the same
production validation protocol used for Exp04:

    fixed val400
    imgsz   = 960
    rect     = True
    conf     = 0.001
    iou      = 0.70
    max_det  = 100

Primary baseline to beat:
    Exp04 Rect val400 mAP50-95 ~= 0.380711

This script intentionally does NOT modify Ultralytics source.
"""

from __future__ import annotations

import csv
import gc
import math
from pathlib import Path

import torch
import torch.nn as nn
import ultralytics

from ultralytics import YOLO

from multimodal_dataset import (
    CHANNEL_NAMES,
    MULTIMODAL_CHANNELS,
    MultimodalYOLODataset,
)

from multimodal_gated_model import (
    DEFAULT_DEPTH_GATE,
    DEFAULT_IR_GATE,
    DEPTH_FUSION_LAYER,
    IR_FUSION_LAYER,
    Exp05AsymmetricGatedDetectionModel,
)

from multimodal_gated_trainer import (
    Exp05GatedDetectionTrainer,
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
    "exp05_asymmetric_gated_yolo11s_960"
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

EXPECTED_TRAIN_SAMPLES = 1600

EXPECTED_VAL_SAMPLES = 400

EXPECTED_STRIDES = (
    8.0,
    16.0,
    32.0,
)


# ============================================================
# Reference baselines
# ============================================================

RGB_BASELINE_MAP5095 = (
    0.380843
)

EXP04_RECT_VAL400_MAP5095 = (
    0.380711
)

EXP04_OFFICIAL_VAL_MAP5095 = (
    0.379090
)


# ============================================================
# Console helpers
# ============================================================

def section(
    title: str,
) -> None:

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def pass_line(
    text: str,
) -> None:

    print(
        f"[PASS] {text}"
    )


# ============================================================
# Model helpers
# ============================================================

def first_conv(
    block: nn.Module,
) -> nn.Conv2d:

    if hasattr(
        block,
        "conv",
    ):

        conv = (
            block.conv
        )

        if isinstance(
            conv,
            nn.Conv2d,
        ):
            return conv

    for module in block.modules():

        if isinstance(
            module,
            nn.Conv2d,
        ):
            return module

    raise RuntimeError(
        "Could not find Conv2d in "
        f"{type(block).__name__}."
    )


def assert_finite_model(
    model: nn.Module,
    description: str,
) -> None:

    for (
        name,
        tensor,
    ) in model.state_dict().items():

        if not isinstance(
            tensor,
            torch.Tensor,
        ):
            continue

        if not tensor.is_floating_point():
            continue

        if not torch.isfinite(
            tensor
        ).all():

            raise AssertionError(
                f"{description} contains NaN/Inf:\n"
                f"  {name}"
            )


def verify_exp05_structure(
    model: nn.Module,
    description: str,
    require_trainable: bool = True,
) -> None:

    if not isinstance(
        model,
        Exp05AsymmetricGatedDetectionModel,
    ):

        raise AssertionError(
            f"{description}: expected "
            "Exp05AsymmetricGatedDetectionModel, "
            f"got {type(model).__name__}."
        )

    # --------------------------------------------------------
    # Main RGB path
    # --------------------------------------------------------

    rgb_first = (
        first_conv(
            model.model[0]
        )
    )

    if (
        rgb_first.in_channels
        != 3
    ):

        raise AssertionError(
            f"{description}: RGB main first conv "
            f"in_channels={rgb_first.in_channels}, "
            "expected 3."
        )

    # --------------------------------------------------------
    # Auxiliary inputs
    # --------------------------------------------------------

    depth_first = (
        first_conv(
            model.depth_encoder[0]
        )
    )

    ir_first = (
        first_conv(
            model.ir_encoder[0]
        )
    )

    if (
        depth_first.in_channels
        != 1
    ):

        raise AssertionError(
            f"{description}: Depth encoder "
            "is not 1-channel."
        )

    if (
        ir_first.in_channels
        != 1
    ):

        raise AssertionError(
            f"{description}: IR encoder "
            "is not 1-channel."
        )

    # --------------------------------------------------------
    # Detect head
    # --------------------------------------------------------

    detect_head = (
        model.model[-1]
    )

    head_nc = getattr(
        detect_head,
        "nc",
        None,
    )

    if (
        head_nc
        != EXPECTED_CLASSES
    ):

        raise AssertionError(
            f"{description}: Detect nc="
            f"{head_nc}, expected "
            f"{EXPECTED_CLASSES}."
        )

    # --------------------------------------------------------
    # Strides
    # --------------------------------------------------------

    strides = tuple(
        float(value)
        for value
        in model.stride.detach().cpu().tolist()
    )

    if (
        strides
        != EXPECTED_STRIDES
    ):

        raise AssertionError(
            f"{description}: unexpected strides "
            f"{strides}."
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    if (
        getattr(
            model,
            "multimodal_channels",
            None,
        )
        != EXPECTED_CHANNELS
    ):

        raise AssertionError(
            f"{description}: multimodal channel "
            "metadata mismatch."
        )

    # --------------------------------------------------------
    # Gates
    # --------------------------------------------------------

    if require_trainable:

        if not (
            model
            .depth_gate
            .logits
            .requires_grad
        ):

            raise AssertionError(
                f"{description}: Depth gate "
                "requires_grad=False."
            )

        if not (
            model
            .ir_gate
            .logits
            .requires_grad
        ):

            raise AssertionError(
                f"{description}: IR gate "
                "requires_grad=False."
            )

    assert_finite_model(
        model=model,
        description=description,
    )


# ============================================================
# Environment preflight
# ============================================================

def check_environment() -> None:

    section(
        "Exp05 Formal Training - Preflight"
    )

    # --------------------------------------------------------
    # Required files
    # --------------------------------------------------------

    if not PRETRAINED_MODEL_PATH.is_file():

        raise FileNotFoundError(
            "Missing pretrained YOLO11s:\n"
            f"  {PRETRAINED_MODEL_PATH}"
        )

    if not DATA_YAML.is_file():

        raise FileNotFoundError(
            "Missing multimodal data YAML:\n"
            f"  {DATA_YAML}"
        )

    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Never mix two formal runs.
    # --------------------------------------------------------

    if RUN_DIR.exists():

        raise RuntimeError(
            "Formal Exp05 run directory already exists:\n"
            f"  {RUN_DIR}\n\n"
            "The script stopped to prevent old/new "
            "experiments from being mixed.\n"
            "Rename or remove the directory only if a "
            "fresh rerun is intentionally required."
        )

    # --------------------------------------------------------
    # CUDA
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is unavailable. "
            "Formal Exp05 training must not start on CPU."
        )

    # --------------------------------------------------------
    # Project-local editable Ultralytics
    # --------------------------------------------------------

    active_ultralytics = (
        Path(
            ultralytics.__file__
        )
        .resolve()
    )

    expected_ultralytics_root = (
        PROJECT_ROOT
        / "ultralytics"
    ).resolve()

    try:

        active_ultralytics.relative_to(
            expected_ultralytics_root
        )

    except ValueError as exc:

        raise RuntimeError(
            "Active Ultralytics is not the "
            "project-local editable source.\n\n"
            f"Active:\n"
            f"  {active_ultralytics}\n\n"
            f"Expected under:\n"
            f"  {expected_ultralytics_root}"
        ) from exc

    # --------------------------------------------------------
    # Print locked configuration
    # --------------------------------------------------------

    print(
        "Project root    :",
        PROJECT_ROOT,
    )

    print(
        "Ultralytics     :",
        active_ultralytics,
    )

    print(
        "Version         :",
        getattr(
            ultralytics,
            "__version__",
            "unknown",
        ),
    )

    print(
        "Pretrained      :",
        PRETRAINED_MODEL_PATH,
    )

    print(
        "Dataset YAML    :",
        DATA_YAML,
    )

    print(
        "Output          :",
        RUN_DIR,
    )

    print(
        "Input           :",
        CHANNEL_NAMES,
    )

    print(
        "Channels        :",
        MULTIMODAL_CHANNELS,
    )

    print(
        "Classes         :",
        EXPECTED_CLASSES,
    )

    print(
        "Depth fusion    :",
        f"layer {DEPTH_FUSION_LAYER} / P3",
    )

    print(
        "IR fusion       :",
        f"layer {IR_FUSION_LAYER} / P4",
    )

    print(
        "Depth gate init :",
        DEFAULT_DEPTH_GATE,
    )

    print(
        "IR gate init    :",
        DEFAULT_IR_GATE,
    )

    print(
        "Image size      :",
        IMAGE_SIZE,
    )

    print(
        "Batch           :",
        BATCH_SIZE,
    )

    print(
        "Epochs          :",
        EPOCHS,
    )

    print(
        "Patience        :",
        PATIENCE,
    )

    print(
        "Workers         :",
        WORKERS,
    )

    print(
        "Seed            :",
        SEED,
    )

    print(
        "Internal max_det:",
        MAX_DET_TRAIN_VAL,
    )

    print(
        "GPU             :",
        torch.cuda.get_device_name(
            0
        ),
    )

    gpu_memory_gib = (
        torch.cuda
        .get_device_properties(
            0
        )
        .total_memory
        / 1024**3
    )

    print(
        "GPU memory      :",
        f"{gpu_memory_gib:.2f} GiB",
    )

    print()
    print(
        "Reference metrics:"
    )

    print(
        "  RGB baseline mAP50-95       :",
        RGB_BASELINE_MAP5095,
    )

    print(
        "  Exp04 Rect val400 mAP50-95  :",
        EXP04_RECT_VAL400_MAP5095,
    )

    print(
        "  Exp04 Official mAP50-95     :",
        EXP04_OFFICIAL_VAL_MAP5095,
    )

    pass_line(
        "project-local editable Ultralytics"
    )

    pass_line(
        "formal Exp05 preflight"
    )


# ============================================================
# Trainer construction
# ============================================================

def build_trainer():

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

        "pretrained": True,

        # Preserve pretrained class-name head remapping.
        "cls_remap": True,

        # ----------------------------------------------------
        # Formal duration
        # ----------------------------------------------------

        "epochs": EPOCHS,

        "patience": PATIENCE,

        # Explicitly restore the complete training split.
        "fraction": 1.0,

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

        # ----------------------------------------------------
        # Precision
        # ----------------------------------------------------

        "amp": True,

        # ----------------------------------------------------
        # Optimizer
        #
        # Keep Exp04 behavior.
        #
        # Ultralytics decides actual optimizer/lr when:
        #
        #     optimizer = auto
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
        # MultimodalYOLODataset already guarantees:
        #
        #     HSV:
        #         RGB   ON
        #         IR    OFF
        #         Depth OFF
        #
        #     geometric transforms:
        #         synchronized across all 5 channels
        # ----------------------------------------------------

        "mosaic": 1.0,

        "close_mosaic": 10,

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

        # Generic YOLO plots assume RGB in several paths.
        "plots": False,
    }

    return Exp05GatedDetectionTrainer(
        overrides=overrides
    )


# ============================================================
# Dataset / training output verification
# ============================================================

def verify_training_outputs(
    trainer,
) -> tuple[
    Path,
    Path,
]:

    section(
        "Exp05 Post-training Verification"
    )

    # --------------------------------------------------------
    # Dataset membership
    # --------------------------------------------------------

    train_dataset = (
        trainer
        .train_loader
        .dataset
    )

    val_dataset = (
        trainer
        .test_loader
        .dataset
    )

    print(
        "Train dataset:",
        type(
            train_dataset
        ).__name__,
        "samples=",
        len(
            train_dataset
        ),
    )

    print(
        "Val dataset  :",
        type(
            val_dataset
        ).__name__,
        "samples=",
        len(
            val_dataset
        ),
    )

    if not isinstance(
        train_dataset,
        MultimodalYOLODataset,
    ):

        raise AssertionError(
            "Formal train dataset is not "
            "MultimodalYOLODataset."
        )

    if not isinstance(
        val_dataset,
        MultimodalYOLODataset,
    ):

        raise AssertionError(
            "Formal val dataset is not "
            "MultimodalYOLODataset."
        )

    if (
        len(
            train_dataset
        )
        != EXPECTED_TRAIN_SAMPLES
    ):

        raise AssertionError(
            "Formal Exp05 train split mismatch: "
            f"expected={EXPECTED_TRAIN_SAMPLES}, "
            f"got={len(train_dataset)}."
        )

    if (
        len(
            val_dataset
        )
        != EXPECTED_VAL_SAMPLES
    ):

        raise AssertionError(
            "Formal Exp05 val split mismatch: "
            f"expected={EXPECTED_VAL_SAMPLES}, "
            f"got={len(val_dataset)}."
        )

    pass_line(
        "full fixed train1600"
    )

    pass_line(
        "full fixed val400"
    )

    # --------------------------------------------------------
    # Live model
    # --------------------------------------------------------

    verify_exp05_structure(
        model=trainer.model,
        description="trainer.model",
    )

    pass_line(
        "live trainer model structure"
    )

    # --------------------------------------------------------
    # Required artifacts
    # --------------------------------------------------------

    run_dir = (
        Path(
            trainer.save_dir
        )
        .resolve()
    )

    weights_dir = (
        run_dir
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
        run_dir
        / "results.csv"
    )

    print(
        "Run directory:",
        run_dir,
    )

    for (
        name,
        path,
    ) in (
        (
            "best.pt",
            best_path,
        ),
        (
            "last.pt",
            last_path,
        ),
        (
            "results.csv",
            results_csv,
        ),
    ):

        if not path.is_file():

            raise AssertionError(
                f"Required output missing:\n"
                f"  {path}"
            )

        pass_line(
            f"{name} exists"
        )

    # --------------------------------------------------------
    # Results row count
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
        "Recorded epoch rows:",
        len(
            rows
        ),
    )

    if not rows:

        raise AssertionError(
            "results.csv contains no epoch rows."
        )

    pass_line(
        "results.csv contains training history"
    )

    return (
        best_path,
        last_path,
    )


# ============================================================
# Independent checkpoint verification
# ============================================================

def verify_checkpoint(
    checkpoint_path: Path,
) -> None:

    section(
        f"Independent reload: {checkpoint_path.name}"
    )

    wrapper = YOLO(
        str(
            checkpoint_path
        )
    )

    model = (
        wrapper.model
    )

    verify_exp05_structure(
        model=model,
        description=checkpoint_path.name,
        require_trainable=False,
    )

    pass_line(
        f"{checkpoint_path.name} reloads as Exp05 model"
    )

    # --------------------------------------------------------
    # Gate state
    #
    # Checkpoint is EMA FP16. Therefore:
    #
    # DO NOT compare tiny checkpoint changes against the
    # original FP32 gate initialization as evidence of
    # optimizer learning.
    #
    # Live FP32 training participation was already verified
    # separately by:
    #
    #     test_exp05_trainer_live_update.py
    # --------------------------------------------------------

    gate_stats = (
        model
        .get_gate_statistics()
    )

    print(
        "Depth gate:",
        gate_stats[
            "depth"
        ],
    )

    print(
        "IR gate   :",
        gate_stats[
            "ir"
        ],
    )

    for modality in (
        "depth",
        "ir",
    ):

        for statistic in (
            "mean",
            "min",
            "max",
            "std",
        ):

            value = (
                gate_stats[
                    modality
                ][
                    statistic
                ]
            )

            if not math.isfinite(
                value
            ):

                raise AssertionError(
                    f"{checkpoint_path.name}: "
                    f"{modality}.{statistic} "
                    "is not finite."
                )

    pass_line(
        "checkpoint gate state finite"
    )

    # --------------------------------------------------------
    # 5-channel forward sanity
    # --------------------------------------------------------

    device = torch.device(
        "cuda:0"
    )

    model = (
        model
        .float()
        .to(
            device
        )
        .eval()
    )

    dummy = torch.zeros(
        (
            1,
            EXPECTED_CHANNELS,
            320,
            320,
        ),
        dtype=torch.float32,
        device=device,
    )

    with torch.inference_mode():

        output = model(
            dummy
        )

    prediction = (
        output[0]
        if isinstance(
            output,
            tuple,
        )
        else output
    )

    expected_shape = (
        1,
        4 + EXPECTED_CLASSES,
        2100,
    )

    print(
        "5-channel forward:",
        tuple(
            prediction.shape
        ),
    )

    if (
        tuple(
            prediction.shape
        )
        != expected_shape
    ):

        raise AssertionError(
            f"{checkpoint_path.name}: "
            "unexpected prediction shape: "
            f"{tuple(prediction.shape)}"
        )

    if not torch.isfinite(
        prediction
    ).all():

        raise AssertionError(
            f"{checkpoint_path.name}: "
            "forward contains NaN/Inf."
        )

    pass_line(
        "5-channel checkpoint forward"
    )

    del dummy
    del output
    del prediction
    del wrapper
    del model

    gc.collect()

    torch.cuda.empty_cache()


# ============================================================
# Metrics summary
# ============================================================

def print_metrics(
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
        "Reference:"
    )

    print(
        "RGB baseline             :",
        RGB_BASELINE_MAP5095,
    )

    print(
        "Exp04 Rect val400        :",
        EXP04_RECT_VAL400_MAP5095,
    )

    print(
        "Exp04 Official val       :",
        EXP04_OFFICIAL_VAL_MAP5095,
    )

    if map5095 is not None:

        internal_value = float(
            map5095
        )

        print()
        print(
            "Internal-val delta vs "
            "Exp04 Rect val400:",
            f"{internal_value - EXP04_RECT_VAL400_MAP5095:+.6f}",
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The Exp05 value above is the Trainer's "
        "internal validation metric with max_det=300."
    )

    print(
        "It is NOT the final apples-to-apples "
        "Exp04 comparison."
    )

    print(
        "Final comparison must use the separate "
        "Rect val400 protocol:"
    )

    print(
        "imgsz=960, rect=True, conf=0.001, "
        "iou=0.70, max_det=100."
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    check_environment()

    section(
        "Build Exp05 Formal Trainer"
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
            "Trainer metadata channels != 5."
        )

    if (
        trainer.data.get(
            "nc"
        )
        != EXPECTED_CLASSES
    ):

        raise AssertionError(
            "Trainer metadata nc != 12."
        )

    pass_line(
        "Trainer metadata"
    )

    # --------------------------------------------------------
    # Formal training
    # --------------------------------------------------------

    section(
        "START EXP05 FORMAL TRAINING"
    )

    trainer.train()

    section(
        "EXP05 FORMAL TRAINING RETURNED"
    )

    # --------------------------------------------------------
    # Post-training verification
    # --------------------------------------------------------

    (
        best_path,
        last_path,
    ) = verify_training_outputs(
        trainer
    )

    verify_checkpoint(
        best_path
    )

    verify_checkpoint(
        last_path
    )

    print_metrics(
        trainer
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    section(
        "FINAL RESULT"
    )

    print(
        "Full train1600                   : PASS"
    )

    print(
        "Full val400                      : PASS"
    )

    print(
        "Multimodal Dataset               : PASS"
    )

    print(
        "RGB pretrained 3-channel path    : PASS"
    )

    print(
        "Depth gated P3 fusion            : PASS"
    )

    print(
        "IR gated P4 fusion               : PASS"
    )

    print(
        "12-class Detect                  : PASS"
    )

    print(
        "AMP training                     : PASS"
    )

    print(
        "Internal validation              : PASS"
    )

    print(
        "EMA/checkpoint save              : PASS"
    )

    print(
        "best.pt reload                   : PASS"
    )

    print(
        "last.pt reload                   : PASS"
    )

    print(
        "5-channel checkpoint forward     : PASS"
    )

    print()
    print(
        "STATUS = PASS"
    )

    print()
    print(
        "Next:"
    )

    print(
        "1. Official/production Rect val400"
    )

    print(
        "2. Exp05 modality ablation"
    )

    print(
        "3. Test-set submission only if "
        "local validation clearly improves"
    )


if __name__ == "__main__":
    main()

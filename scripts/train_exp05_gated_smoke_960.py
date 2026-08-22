#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 Exp05
Asymmetric Gated Feature Fusion
Full Trainer integration smoke test.

Purpose
=======

Verify the complete real training chain:

    MultimodalYOLODataset
            ↓
    DataLoader
            ↓
    trainer preprocess: uint8 -> float / 255
            ↓
    Exp05 asymmetric gated model
            ↓
    Depth @ P3 / IR @ P4
            ↓
    YOLO detection loss
            ↓
    AMP backward
            ↓
    optimizer step
            ↓
    validation
            ↓
    EMA
            ↓
    last.pt / best.pt
            ↓
    independent checkpoint reload
            ↓
    5-channel forward after reload
            ↓
    trainer checkpoint reconstruction

This is NOT the formal Exp05 experiment.

Smoke configuration
===================

    imgsz         = 960
    batch         = 2
    epochs        = 2
    train fraction= 0.05

The full 400-image validation set is still used.

Performance from this run is meaningless.
Only engineering correctness matters.
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
from ultralytics.utils.torch_utils import unwrap_model

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
    "exp05_asymmetric_gated_smoke_960"
)

RUN_DIR = (
    RUNS_DIR
    / EXPERIMENT_NAME
)


# ============================================================
# Smoke configuration
# ============================================================

IMAGE_SIZE = 960
BATCH_SIZE = 2
EPOCHS = 2

TRAIN_FRACTION = 0.05

SEED = 2026

EXPECTED_CLASSES = 12
EXPECTED_CHANNELS = 5

EXPECTED_STRIDES = (
    8.0,
    16.0,
    32.0,
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

        conv = block.conv

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

        if (
            isinstance(
                tensor,
                torch.Tensor,
            )
            and tensor.is_floating_point()
            and not torch.isfinite(
                tensor
            ).all()
        ):

            raise AssertionError(
                f"{description} contains NaN/Inf "
                f"at state tensor: {name}"
            )


def assert_exp05_structure(
    model: nn.Module,
    description: str,
) -> None:

    if not isinstance(
        model,
        Exp05AsymmetricGatedDetectionModel,
    ):

        raise AssertionError(
            f"{description}: wrong model class: "
            f"{type(model).__name__}"
        )

    rgb_first = first_conv(
        model.model[0]
    )

    if (
        rgb_first.in_channels
        != 3
    ):

        raise AssertionError(
            f"{description}: RGB main path "
            f"in_channels={rgb_first.in_channels}, "
            "expected 3."
        )

    depth_first = first_conv(
        model.depth_encoder[0]
    )

    ir_first = first_conv(
        model.ir_encoder[0]
    )

    if (
        depth_first.in_channels
        != 1
    ):

        raise AssertionError(
            f"{description}: Depth input is not 1ch."
        )

    if (
        ir_first.in_channels
        != 1
    ):

        raise AssertionError(
            f"{description}: IR input is not 1ch."
        )

    head = model.model[-1]

    if (
        getattr(
            head,
            "nc",
            None,
        )
        != EXPECTED_CLASSES
    ):

        raise AssertionError(
            f"{description}: Detect nc="
            f"{getattr(head, 'nc', None)}, "
            f"expected {EXPECTED_CLASSES}."
        )

    strides = tuple(
        float(v)
        for v
        in model.stride.detach().cpu().tolist()
    )

    if (
        strides
        != EXPECTED_STRIDES
    ):

        raise AssertionError(
            f"{description}: stride mismatch: "
            f"{strides}"
        )

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

    assert_finite_model(
        model,
        description,
    )


# ============================================================
# Preflight
# ============================================================

def check_environment() -> None:

    section(
        "Preflight"
    )

    if not PRETRAINED_MODEL_PATH.is_file():

        raise FileNotFoundError(
            "Missing pretrained YOLO11s:\n"
            f"  {PRETRAINED_MODEL_PATH}"
        )

    if not DATA_YAML.is_file():

        raise FileNotFoundError(
            "Missing multimodal dataset YAML:\n"
            f"  {DATA_YAML}"
        )

    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if RUN_DIR.exists():

        raise RuntimeError(
            "Smoke output directory already exists:\n"
            f"  {RUN_DIR}\n\n"
            "To avoid mixing old and new runs, "
            "rename or remove it manually before rerunning."
        )

    print(
        "Project root   :",
        PROJECT_ROOT,
    )

    print(
        "Ultralytics    :",
        ultralytics.__file__,
    )

    print(
        "Pretrained     :",
        PRETRAINED_MODEL_PATH,
    )

    print(
        "Dataset        :",
        DATA_YAML,
    )

    print(
        "Output         :",
        RUN_DIR,
    )

    print(
        "Image size     :",
        IMAGE_SIZE,
    )

    print(
        "Batch          :",
        BATCH_SIZE,
    )

    print(
        "Epochs         :",
        EPOCHS,
    )

    print(
        "Train fraction :",
        TRAIN_FRACTION,
    )

    print(
        "Channels       :",
        EXPECTED_CHANNELS,
    )

    print(
        "Channel order  :",
        CHANNEL_NAMES,
    )

    print(
        "Depth fusion   :",
        f"layer {DEPTH_FUSION_LAYER} / P3",
    )

    print(
        "IR fusion      :",
        f"layer {IR_FUSION_LAYER} / P4",
    )

    print(
        "Depth gate init:",
        DEFAULT_DEPTH_GATE,
    )

    print(
        "IR gate init   :",
        DEFAULT_IR_GATE,
    )

    if torch.cuda.is_available():

        print(
            "GPU            :",
            torch.cuda.get_device_name(
                0
            ),
        )

        total_gb = (
            torch.cuda.get_device_properties(
                0
            ).total_memory
            / 1024**3
        )

        print(
            "GPU memory     :",
            f"{total_gb:.2f} GiB",
        )

    else:

        print(
            "GPU            : CUDA unavailable"
        )

    # --------------------------------------------------------
    # Must use project-local editable Ultralytics.
    # --------------------------------------------------------

    active_ultralytics = (
        Path(
            ultralytics.__file__
        )
        .resolve()
    )

    expected_source = (
        PROJECT_ROOT
        / "ultralytics"
    ).resolve()

    try:

        active_ultralytics.relative_to(
            expected_source
        )

    except ValueError as exc:

        raise RuntimeError(
            "Active Ultralytics is not the "
            "project-local editable source.\n\n"
            f"Active:\n  {active_ultralytics}\n"
            f"Expected under:\n  {expected_source}"
        ) from exc

    pass_line(
        "project-local editable Ultralytics"
    )


# ============================================================
# Trainer construction
# ============================================================

def build_trainer(
    model_path: Path = PRETRAINED_MODEL_PATH,
    experiment_name: str = EXPERIMENT_NAME,
):
    """
    Construct the real Exp05 Trainer.

    Training fraction is intentionally small because this is
    an engineering smoke test rather than an experiment.
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
            model_path
        ),

        "data": str(
            DATA_YAML
        ),

        "task": "detect",

        # ----------------------------------------------------
        # Duration
        # ----------------------------------------------------

        "epochs": EPOCHS,

        "patience": 50,

        # ----------------------------------------------------
        # Geometry
        # ----------------------------------------------------

        "imgsz": IMAGE_SIZE,

        "batch": BATCH_SIZE,

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        "device": device,

        "workers": 2,

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

        "name": experiment_name,

        "exist_ok": False,

        "save": True,

        "save_period": -1,

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        "val": True,

        # Internal training validation only.
        # Competition validation will remain max_det=100.
        "max_det": 300,

        # ----------------------------------------------------
        # Precision
        # ----------------------------------------------------

        "amp": True,

        # ----------------------------------------------------
        # Data
        # ----------------------------------------------------

        "cache": False,

        "rect": False,

        # Only shorten TRAIN dataset.
        # build_dataset() forces val fraction back to 1.0.
        "fraction": TRAIN_FRACTION,

        # ----------------------------------------------------
        # Augmentation
        # ----------------------------------------------------

        "mosaic": 1.0,

        "fliplr": 0.5,

        "mixup": 0.0,

        "cutmix": 0.0,

        "copy_paste": 0.0,

        "close_mosaic": 0,

        "multi_scale": 0.0,

        # Generic plotting may assume RGB.
        "plots": False,
    }

    return Exp05GatedDetectionTrainer(
        overrides=overrides
    )


# ============================================================
# Post-training checks
# ============================================================

def verify_dataset_and_model(
    trainer,
) -> None:

    section(
        "Trainer runtime structure"
    )

    model = unwrap_model(
        trainer.model
    )

    assert_exp05_structure(
        model=model,
        description="trainer.model",
    )

    pass_line(
        "trainer.model is Exp05 gated model"
    )

    print(
        "Trainer data channels:",
        trainer.data.get(
            "channels"
        ),
    )

    print(
        "Trainer data classes :",
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
            "Trainer dataset metadata is not 5-channel."
        )

    if (
        trainer.data.get(
            "nc"
        )
        != EXPECTED_CLASSES
    ):

        raise AssertionError(
            "Trainer dataset metadata is not 12-class."
        )

    pass_line(
        "trainer metadata channels == 5"
    )

    pass_line(
        "trainer metadata classes == 12"
    )

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
            "Train dataset is not MultimodalYOLODataset."
        )

    if not isinstance(
        val_dataset,
        MultimodalYOLODataset,
    ):

        raise AssertionError(
            "Val dataset is not MultimodalYOLODataset."
        )

    pass_line(
        "train Dataset is MultimodalYOLODataset"
    )

    pass_line(
        "val Dataset is MultimodalYOLODataset"
    )

    # fraction=0.05 should shorten train.
    # Val must remain the fixed complete 400-image split.
    if len(
        val_dataset
    ) != 400:

        raise AssertionError(
            "Smoke validation dataset is not the "
            f"full fixed val400: got {len(val_dataset)}."
        )

    pass_line(
        "validation keeps complete fixed val400"
    )


def verify_artifacts(
    trainer,
) -> tuple[
    Path,
    Path,
]:

    section(
        "Training artifacts"
    )

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
        description,
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
                f"{description} missing:\n"
                f"  {path}"
            )

        print(
            f"[PASS] {description} exists"
        )

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
        "results.csv rows:",
        len(
            rows
        ),
    )

    if len(
        rows
    ) < EPOCHS:

        raise AssertionError(
            "Training did not complete the "
            f"expected {EPOCHS} epochs."
        )

    pass_line(
        f"results.csv contains >= {EPOCHS} epochs"
    )

    print()
    print(
        "Final trainer metrics:"
    )

    print(
        trainer.metrics
    )

    return (
        best_path,
        last_path,
    )


# ============================================================
# Independent checkpoint reload
# ============================================================

def verify_checkpoint_reload(
    checkpoint_path: Path,
) -> Exp05AsymmetricGatedDetectionModel:

    section(
        f"Independent checkpoint reload: {checkpoint_path.name}"
    )

    wrapper = YOLO(
        str(
            checkpoint_path
        )
    )

    model = (
        wrapper.model
    )

    assert_exp05_structure(
        model=model,
        description=checkpoint_path.name,
    )

    pass_line(
        f"{checkpoint_path.name} reloads as Exp05 model"
    )

    gate_stats = (
        model.get_gate_statistics()
    )

    print(
        "Reloaded Depth gate:",
        gate_stats[
            "depth"
        ],
    )

    print(
        "Reloaded IR gate   :",
        gate_stats[
            "ir"
        ],
    )

    for modality in (
        "depth",
        "ir",
    ):

        for value in (
            gate_stats[
                modality
            ][
                "mean"
            ],
            gate_stats[
                modality
            ][
                "min"
            ],
            gate_stats[
                modality
            ][
                "max"
            ],
            gate_stats[
                modality
            ][
                "std"
            ],
        ):

            if not math.isfinite(
                value
            ):

                raise AssertionError(
                    f"{checkpoint_path.name}: "
                    f"{modality} gate contains non-finite values."
                )

    pass_line(
        "checkpoint gate values are finite"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Ultralytics serializes the EMA checkpoint in FP16.
    # Therefore comparing checkpoint gate values against the
    # original FP32 initialization is NOT a valid proof of
    # optimization.
    #
    # Real live-FP32 optimizer participation is verified by:
    #
    #   scripts/test_exp05_trainer_live_update.py
    #
    # Here we only verify that the checkpoint contains finite
    # gate state and can be reloaded / forwarded correctly.
    # --------------------------------------------------------

    print(
        "Checkpoint gate learning proof: "
        "delegated to test_exp05_trainer_live_update.py"
    )

    # --------------------------------------------------------
    # Verify actual checkpoint forward.
    # Keep this small; 960 forward already occurred throughout
    # training and validation.
    # --------------------------------------------------------

    device = (
        torch.device(
            "cuda:0"
        )
        if torch.cuda.is_available()
        else torch.device(
            "cpu"
        )
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
        "Reload forward prediction:",
        tuple(
            prediction.shape
        ),
    )

    if tuple(
        prediction.shape
    ) != expected_shape:

        raise AssertionError(
            "Reloaded Exp05 forward shape mismatch: "
            f"expected={expected_shape}, "
            f"got={tuple(prediction.shape)}."
        )

    if not torch.isfinite(
        prediction
    ).all():

        raise AssertionError(
            "Reloaded checkpoint forward contains NaN/Inf."
        )

    pass_line(
        "reloaded checkpoint 5-channel forward"
    )

    model = (
        model
        .cpu()
    )

    del dummy
    del output
    del prediction

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model


# ============================================================
# Trainer reconstruction from Exp05 checkpoint
# ============================================================

def verify_trainer_checkpoint_reconstruction(
    checkpoint_path: Path,
    expected_model: Exp05AsymmetricGatedDetectionModel,
) -> None:

    section(
        "Trainer reconstruction from Exp05 checkpoint"
    )

    reload_name = (
        "exp05_asymmetric_gated_smoke_reload_check"
    )

    # No training is launched here.
    # We only force BaseTrainer.setup_model() to go through:
    #
    #   load_checkpoint()
    #       ↓
    #   Exp05GatedDetectionTrainer.get_model()
    #       ↓
    #   source_has_exp05_parameters()
    #       ↓
    #   preserve learned auxiliary/gate state
    #
    reload_trainer = (
        build_trainer(
            model_path=checkpoint_path,
            experiment_name=reload_name,
        )
    )

    reload_trainer.setup_model()

    reconstructed = unwrap_model(
        reload_trainer.model
    )

    assert_exp05_structure(
        model=reconstructed,
        description=(
            "trainer reconstructed checkpoint"
        ),
    )

    source_state = (
        expected_model
        .float()
        .cpu()
        .state_dict()
    )

    target_state = (
        reconstructed
        .float()
        .cpu()
        .state_dict()
    )

    if (
        source_state.keys()
        != target_state.keys()
    ):

        raise AssertionError(
            "Trainer checkpoint reconstruction "
            "state_dict key mismatch."
        )

    checked = 0

    for key in source_state:

        source_tensor = (
            source_state[
                key
            ]
        )

        target_tensor = (
            target_state[
                key
            ]
        )

        if (
            source_tensor.shape
            != target_tensor.shape
        ):

            raise AssertionError(
                "Checkpoint reconstruction shape mismatch: "
                f"{key}"
            )

        if not torch.equal(
            source_tensor,
            target_tensor,
        ):

            if (
                source_tensor.is_floating_point()
                and target_tensor.is_floating_point()
            ):

                max_diff = (
                    source_tensor
                    - target_tensor
                ).abs().max().item()

            else:

                max_diff = "N/A"

            raise AssertionError(
                "Checkpoint reconstruction changed tensor: "
                f"{key}, max_abs_diff={max_diff}"
            )

        checked += 1

    print(
        "Exact reconstructed state tensors:",
        checked,
    )

    pass_line(
        "Trainer Exp05 checkpoint path preserves full learned state"
    )

    del reload_trainer
    del reconstructed

    gc.collect()


# ============================================================
# Main
# ============================================================

def main() -> None:

    check_environment()

    section(
        "Build Exp05GatedDetectionTrainer"
    )

    trainer = build_trainer()

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

    section(
        "START EXP05 TRAINER SMOKE"
    )

    trainer.train()

    section(
        "TRAINING RETURNED SUCCESSFULLY"
    )

    verify_dataset_and_model(
        trainer
    )

    (
        best_path,
        last_path,
    ) = verify_artifacts(
        trainer
    )

    # --------------------------------------------------------
    # Independently deserialize BOTH checkpoint paths.
    # --------------------------------------------------------

    best_model = (
        verify_checkpoint_reload(
            best_path
        )
    )

    last_model = (
        verify_checkpoint_reload(
            last_path
        )
    )

    # --------------------------------------------------------
    # Verify custom Trainer can reconstruct learned Exp05 state
    # from a .pt checkpoint without resetting gates/branches.
    # --------------------------------------------------------

    verify_trainer_checkpoint_reconstruction(
        checkpoint_path=best_path,
        expected_model=best_model,
    )

    del best_model
    del last_model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    section(
        "FINAL RESULT"
    )

    print(
        "Multimodal Dataset              : PASS"
    )

    print(
        "DataLoader                       : PASS"
    )

    print(
        "5-channel preprocess             : PASS"
    )

    print(
        "Exp05 custom model               : PASS"
    )

    print(
        "RGB 3-channel protected path     : PASS"
    )

    print(
        "Depth @ P3                       : PASS"
    )

    print(
        "IR @ P4                          : PASS"
    )

    print(
        "12-class Detect                  : PASS"
    )

    print(
        "AMP loss/backward                : PASS"
    )

    print(
        "Trainer optimization loop          : PASS"
    )

    print(
        "Custom live FP32 parameter updates : "
        "verified by separate live audit"
    )

    print(
        "EMA/checkpoint save              : PASS"
    )

    print(
        "Validation on val400             : PASS"
    )

    print(
        "best.pt reload                   : PASS"
    )

    print(
        "last.pt reload                   : PASS"
    )

    print(
        "Exp05 Trainer checkpoint rebuild : PASS"
    )

    print()
    print(
        "STATUS = PASS"
    )

    print()
    print(
        "If this passes, the next step is "
        "the formal Exp05 training script."
    )


if __name__ == "__main__":
    main()

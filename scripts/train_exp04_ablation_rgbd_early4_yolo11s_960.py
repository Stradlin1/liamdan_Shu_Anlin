#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 Exp04 modality ablation: RGB + Depth.

4-channel Early Fusion, YOLO11s @ 960.

This is a controlled Axis-A experiment. Relative to the established Exp04
RGB+IR+Depth run, the intended experimental variable is the selected input
modality set only.

Input:
    [R, G, B, Depth]

Model:
    YOLO11s
    4-channel input
    12-class Detect head

Initialization:
    RGB   <- COCO pretrained exactly
    Depth <- zero

Formal configuration kept aligned with Exp04:
    imgsz        = 960
    batch        = 8
    epochs       = 200
    patience     = 50
    seed         = 2026
    mosaic       = 1.0
    close_mosaic = 10
    fliplr       = 0.5
    max_det      = 300  # training/internal validation only

Final controlled comparison must later use the fixed val400 protocol:
    imgsz   = 960
    conf    = 0.001
    iou     = 0.70
    max_det = 100
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import ultralytics
from ultralytics import YOLO

from multimodal_config import (
    channel_names_for_modalities,
    channels_for_modalities,
)
from multimodal_trainer import MultimodalDetectionTrainer


MODALITIES = ("rgb", "depth")
CHANNEL_NAMES = channel_names_for_modalities(MODALITIES)
EXPECTED_CHANNELS = channels_for_modalities(MODALITIES)
EXPECTED_CLASSES = 12

EXPERIMENT_NAME = "exp04_ablation_rgbd_early4_yolo11s_960"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_MODEL_PATH = PROJECT_ROOT / "pretrained" / "yolo11s.pt"
DATA_YAML = PROJECT_ROOT / "yolo_views" / "rgb_v1" / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "runs"
RUN_DIR = RUNS_DIR / EXPERIMENT_NAME

IMAGE_SIZE = 960
BATCH_SIZE = 8
EPOCHS = 200
PATIENCE = 50
SEED = 2026
WORKERS = 4
MAX_DET_TRAIN_VAL = 300

RGB_BASELINE_MAP5095 = 0.380843


def section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def get_first_conv(model: nn.Module) -> nn.Conv2d:
    if not hasattr(model, "model"):
        raise RuntimeError("Model has no '.model' attribute.")

    first_block = model.model[0]
    if not hasattr(first_block, "conv"):
        raise RuntimeError("YOLO first block has no '.conv'.")

    conv = first_block.conv
    if not isinstance(conv, nn.Conv2d):
        raise TypeError("YOLO first convolution is not nn.Conv2d.")

    return conv


def check_environment() -> None:
    section("RGBD 4ch Formal Training - Preflight")

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

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    if RUN_DIR.exists():
        raise RuntimeError(
            "Formal experiment directory already exists:\n"
            f"  {RUN_DIR}\n\n"
            "The script stopped to prevent old/new runs from being mixed."
        )

    print("Project root :", PROJECT_ROOT)
    print("Ultralytics  :", ultralytics.__file__)
    print("Version      :", getattr(ultralytics, "__version__", "unknown"))
    print("Pretrained   :", PRETRAINED_MODEL_PATH)
    print("Dataset YAML :", DATA_YAML)
    print("Output       :", RUN_DIR)
    print("Modalities   :", MODALITIES)
    print("Input        :", CHANNEL_NAMES)
    print("Channels     :", EXPECTED_CHANNELS)
    print("Classes      :", EXPECTED_CLASSES)
    print("Image size   :", IMAGE_SIZE)
    print("Batch        :", BATCH_SIZE)
    print("Epochs       :", EPOCHS)
    print("Patience     :", PATIENCE)
    print("Seed         :", SEED)
    print("max_det      :", MAX_DET_TRAIN_VAL)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Formal RGBD training should not start on CPU."
        )

    print("GPU          :", torch.cuda.get_device_name(0))

    active_ultralytics = Path(ultralytics.__file__).resolve()
    expected_root = (PROJECT_ROOT / "ultralytics").resolve()

    try:
        active_ultralytics.relative_to(expected_root)
    except ValueError as exc:
        raise RuntimeError(
            "Active Ultralytics is not the project-local editable source.\n\n"
            f"Active:\n  {active_ultralytics}\n"
            f"Expected under:\n  {expected_root}"
        ) from exc

    print("[PASS] project-local editable Ultralytics")


def build_trainer() -> MultimodalDetectionTrainer:
    overrides = {
        "model": str(PRETRAINED_MODEL_PATH),
        "data": str(DATA_YAML),
        "task": "detect",
        "pretrained": True,
        "cls_remap": True,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "imgsz": IMAGE_SIZE,
        "batch": BATCH_SIZE,
        "rect": False,
        "multi_scale": 0.0,
        "device": "0",
        "workers": WORKERS,
        "cache": False,
        "amp": True,
        "optimizer": "auto",
        "seed": SEED,
        "deterministic": True,
        "mosaic": 1.0,
        "close_mosaic": 10,
        "fliplr": 0.5,
        "flipud": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "val": True,
        "iou": 0.70,
        "max_det": MAX_DET_TRAIN_VAL,
        "project": str(RUNS_DIR),
        "name": EXPERIMENT_NAME,
        "exist_ok": False,
        "save": True,
        "save_period": -1,
        "plots": False,
    }

    return MultimodalDetectionTrainer(
        overrides=overrides,
        modalities=MODALITIES,
    )


def verify_trainer_contract(trainer: MultimodalDetectionTrainer) -> None:
    if trainer.modalities != MODALITIES:
        raise AssertionError(
            f"Trainer modalities={trainer.modalities}, expected={MODALITIES}"
        )

    if trainer.channel_names != CHANNEL_NAMES:
        raise AssertionError(
            f"Trainer channel order={trainer.channel_names}, expected={CHANNEL_NAMES}"
        )

    if trainer.input_channels != EXPECTED_CHANNELS:
        raise AssertionError(
            f"Trainer channels={trainer.input_channels}, expected={EXPECTED_CHANNELS}"
        )

    if trainer.data.get("channels") != EXPECTED_CHANNELS:
        raise AssertionError(
            f"Dataset metadata channels={trainer.data.get('channels')}, "
            f"expected={EXPECTED_CHANNELS}"
        )

    if trainer.data.get("nc") != EXPECTED_CLASSES:
        raise AssertionError(
            f"Dataset metadata nc={trainer.data.get('nc')}, expected={EXPECTED_CLASSES}"
        )

    print("[PASS] trainer modalities/channel order/channel count")
    print("[PASS] dataset metadata channels=4, nc=12")


def write_experiment_manifest(trainer: MultimodalDetectionTrainer) -> None:
    run_dir = Path(trainer.save_dir).resolve()
    manifest_path = run_dir / "experiment_manifest.json"

    manifest = {
        "experiment": EXPERIMENT_NAME,
        "axis": "A_modality_ablation",
        "model": "yolo11s",
        "pretrained": str(PRETRAINED_MODEL_PATH.relative_to(PROJECT_ROOT)),
        "data_yaml": str(DATA_YAML.relative_to(PROJECT_ROOT)),
        "modalities": list(MODALITIES),
        "channel_names": list(CHANNEL_NAMES),
        "channels": EXPECTED_CHANNELS,
        "classes": EXPECTED_CLASSES,
        "imgsz": IMAGE_SIZE,
        "batch": BATCH_SIZE,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "seed": SEED,
        "max_det_train_val": MAX_DET_TRAIN_VAL,
        "final_val_protocol": {
            "imgsz": 960,
            "conf": 0.001,
            "iou": 0.70,
            "max_det": 100,
            "split": "fixed_val400",
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("[PASS] experiment manifest:", manifest_path)


def verify_best_checkpoint(trainer: MultimodalDetectionTrainer) -> None:
    section("RGBD 4ch Post-training Verification")

    run_dir = Path(trainer.save_dir).resolve()
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"
    manifest_path = run_dir / "experiment_manifest.json"

    print("Run directory:", run_dir)

    for path in (best_path, last_path, results_csv, manifest_path):
        if not path.is_file():
            raise AssertionError(f"Required output missing: {path}")
        print("[PASS]", path.name)

    best_wrapper = YOLO(str(best_path))
    model = best_wrapper.model
    first_conv = get_first_conv(model)
    detect_head = model.model[-1]

    print("best.pt weight shape:", tuple(first_conv.weight.shape))
    print("best.pt Detect nc:", getattr(detect_head, "nc", None))

    if first_conv.in_channels != EXPECTED_CHANNELS:
        raise AssertionError(
            f"best.pt channels={first_conv.in_channels}, expected={EXPECTED_CHANNELS}"
        )

    if getattr(detect_head, "nc", None) != EXPECTED_CLASSES:
        raise AssertionError("best.pt is not 12-class.")

    weight = first_conv.weight.detach().cpu()
    depth_weight = weight[:, 3]
    depth_nonzero = torch.count_nonzero(depth_weight).item()

    print("Depth stem nonzero =", depth_nonzero)
    print("Depth stem abs sum =", depth_weight.abs().sum().item())

    if depth_nonzero == 0:
        raise AssertionError("Depth stem remained zero after training.")

    print("[PASS] best.pt 4-channel RGBD")
    print("[PASS] best.pt 12-class")
    print("[PASS] learned Depth stem")


def print_training_metrics(trainer: MultimodalDetectionTrainer) -> None:
    section("Internal Validation Summary")

    metrics = trainer.metrics if isinstance(trainer.metrics, dict) else {}

    precision = metrics.get("metrics/precision(B)")
    recall = metrics.get("metrics/recall(B)")
    map50 = metrics.get("metrics/mAP50(B)")
    map5095 = metrics.get("metrics/mAP50-95(B)")

    print("Precision :", precision)
    print("Recall    :", recall)
    print("mAP50     :", map50)
    print("mAP50-95  :", map5095)
    print("RGB baseline official mAP50-95:", RGB_BASELINE_MAP5095)

    if map5095 is not None:
        delta = float(map5095) - RGB_BASELINE_MAP5095
        print("Internal-val delta vs RGB:", f"{delta:+.6f}")

    print()
    print("IMPORTANT:")
    print("This value uses training/internal validation with max_det=300.")
    print("Do not use it as the final controlled Axis-A comparison.")
    print("Final comparison: imgsz=960, conf=0.001, iou=0.70, max_det=100.")


def main() -> None:
    check_environment()

    section("Build RGBD 4ch Formal Trainer")
    trainer = build_trainer()

    print("Trainer          :", type(trainer).__name__)
    print("Modalities       :", trainer.modalities)
    print("Channel order    :", trainer.channel_names)
    print("Dataset channels :", trainer.data.get("channels"))
    print("Dataset classes  :", trainer.data.get("nc"))

    verify_trainer_contract(trainer)
    write_experiment_manifest(trainer)

    section("START RGBD 4ch FORMAL TRAINING")
    trainer.train()

    section("RGBD 4ch FORMAL TRAINING RETURNED")
    verify_best_checkpoint(trainer)
    print_training_metrics(trainer)

    section("FINAL RESULT")
    print("Formal training pipeline      PASS")
    print("RGBD 4-channel model          PASS")
    print("12-class Detect head          PASS")
    print("best.pt / last.pt             PASS")
    print("Depth stem learned            PASS")
    print()
    print("NEXT:")
    print("Run the fixed val400 protocol only after all Axis-A models are ready.")


if __name__ == "__main__":
    main()

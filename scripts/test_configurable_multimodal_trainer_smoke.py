#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trainer-level smoke test for AIC2026 configurable early fusion.

Run:

    python scripts/test_configurable_multimodal_trainer_smoke.py

Purpose
-------
The existing ``test_configurable_early_fusion_smoke.py`` verifies the
configuration, dataset, standalone model construction, forward, loss and
backward paths.

This test goes one layer higher and verifies the actual custom
``MultimodalDetectionTrainer`` integration used by formal training:

    Trainer
      -> data metadata
      -> MultimodalYOLODataset
      -> DetectionModel(ch=C, nc=12)
      -> pretrained first-conv transfer
      -> train DataLoader
      -> preprocess_batch
      -> real detection loss
      -> backward

Cases
-----
1. RGB + IR       : 4 channels
2. RGB + Depth    : 4 channels
3. RGB + IR + Depth using the trainer DEFAULT, as a regression check for
   the established Exp04 5-channel behavior.

This is intentionally NOT a formal experiment:

    imgsz   = 320
    batch   = 2
    fraction= 0.01 (training split only)
    amp     = False
    workers = 0

No epoch is trained. The test calls the locked trainer setup routine and then
runs exactly one real training batch through preprocessing, loss and backward.

A temporary smoke run directory is removed after each successful case. If a
case fails, its directory is preserved for debugging.
"""

from __future__ import annotations

import gc
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.utils.torch_utils import unwrap_model

from multimodal_config import (
    DEFAULT_MODALITIES,
    channel_names_for_modalities,
    channels_for_modalities,
)
from multimodal_dataset import MultimodalYOLODataset, RGB_VIEW_ROOT
from multimodal_trainer import MultimodalDetectionTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_MODEL_PATH = PROJECT_ROOT / "pretrained" / "yolo11s.pt"
DATA_YAML = RGB_VIEW_ROOT / "data.yaml"
SMOKE_PROJECT = PROJECT_ROOT / "runs" / "_smoke_configurable_trainer"

IMAGE_SIZE = 320
BATCH_SIZE = 2
TRAIN_FRACTION = 0.01
SEED = 2026
EXPECTED_CLASSES = 12

# name, modalities, instantiate trainer through default modalities?
CASES = (
    ("rgbi", ("rgb", "ir"), False),
    ("rgbd", ("rgb", "depth"), False),
    ("rgbid_default", DEFAULT_MODALITIES, True),
)


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def get_first_conv(model: nn.Module) -> nn.Conv2d:
    model = unwrap_model(model)
    check(hasattr(model, "model"), "Detection model has no '.model'")
    first_block = model.model[0]
    check(hasattr(first_block, "conv"), "First YOLO block has no '.conv'")
    conv = first_block.conv
    check(isinstance(conv, nn.Conv2d), "First convolution is not nn.Conv2d")
    return conv


def preflight() -> torch.Tensor:
    section("Preflight")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run this trainer-level smoke test in the "
            "same CUDA environment used for formal experiments."
        )

    missing = []
    for path in (PRETRAINED_MODEL_PATH, DATA_YAML):
        if not path.is_file():
            missing.append(path)

    for split in ("train", "val"):
        path = RGB_VIEW_ROOT / "images" / split
        if not path.is_dir():
            missing.append(path)

    if missing:
        raise FileNotFoundError(
            "Trainer smoke inputs are missing:\n  "
            + "\n  ".join(str(path) for path in missing)
        )

    print("GPU        :", torch.cuda.get_device_name(0))
    print("Pretrained :", PRETRAINED_MODEL_PATH)
    print("Data YAML  :", DATA_YAML)
    print("imgsz      :", IMAGE_SIZE)
    print("batch      :", BATCH_SIZE)
    print("fraction   :", TRAIN_FRACTION)

    reference = YOLO(str(PRETRAINED_MODEL_PATH))
    reference_conv = get_first_conv(reference.model)
    reference_rgb = reference_conv.weight.detach().cpu().clone()

    check(reference_conv.in_channels == 3, "Pretrained model is not 3-channel")

    del reference
    gc.collect()
    torch.cuda.empty_cache()

    print("[PASS] CUDA, paths, and 3-channel pretrained reference")
    return reference_rgb


def build_overrides(case_name: str) -> dict:
    return {
        "model": str(PRETRAINED_MODEL_PATH),
        "data": str(DATA_YAML),
        "task": "detect",
        "pretrained": True,
        "cls_remap": True,
        "epochs": 1,
        "patience": 1,
        "imgsz": IMAGE_SIZE,
        "batch": BATCH_SIZE,
        "fraction": TRAIN_FRACTION,
        "rect": False,
        "multi_scale": 0.0,
        "device": "0",
        "workers": 0,
        "cache": False,
        # Disable AMP here so this test isolates Trainer/Dataset/Model
        # integration. Formal Exp04 already established the AMP training path.
        "amp": False,
        "optimizer": "SGD",
        "seed": SEED,
        "deterministic": True,
        # Keep the smoke path simple; augmentation correctness is already
        # covered by test_configurable_early_fusion_smoke.py.
        "mosaic": 0.0,
        "close_mosaic": 0,
        "fliplr": 0.0,
        "flipud": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "val": True,
        "iou": 0.70,
        "max_det": 100,
        "project": str(SMOKE_PROJECT),
        "name": case_name,
        "exist_ok": True,
        "save": False,
        "plots": False,
        "verbose": False,
    }


def run_case(
    case_name: str,
    modalities: tuple[str, ...],
    use_default_constructor: bool,
    reference_rgb: torch.Tensor,
) -> None:
    expected_channels = channels_for_modalities(modalities)
    expected_names = channel_names_for_modalities(modalities)
    case_dir = SMOKE_PROJECT / case_name

    # Do not let artifacts from an older smoke run influence this one.
    shutil.rmtree(case_dir, ignore_errors=True)

    section(
        f"Trainer case: {case_name} | modalities={modalities} | "
        f"channels={expected_channels}"
    )

    success = False
    trainer = None
    batch = None
    loss = None

    try:
        overrides = build_overrides(case_name)

        if use_default_constructor:
            trainer = MultimodalDetectionTrainer(
                overrides=overrides,
            )
        else:
            trainer = MultimodalDetectionTrainer(
                overrides=overrides,
                modalities=modalities,
            )

        # ----------------------------------------------------
        # Constructor / data metadata contract
        # ----------------------------------------------------
        check(
            trainer.modalities == modalities,
            f"Trainer modalities={trainer.modalities}, expected={modalities}",
        )
        check(
            trainer.channel_names == expected_names,
            f"Trainer channel_names={trainer.channel_names}, expected={expected_names}",
        )
        check(
            trainer.input_channels == expected_channels,
            "Trainer input channel count mismatch",
        )
        check(
            int(trainer.data["channels"]) == expected_channels,
            f"data['channels']={trainer.data['channels']}, expected={expected_channels}",
        )
        check(
            int(trainer.data["nc"]) == EXPECTED_CLASSES,
            f"data['nc']={trainer.data['nc']}, expected={EXPECTED_CLASSES}",
        )

        print("[PASS] trainer configuration and dataset metadata")

        # ----------------------------------------------------
        # Actual locked Ultralytics setup path
        # ----------------------------------------------------
        trainer._setup_train()

        model = unwrap_model(trainer.model)
        conv = get_first_conv(model)
        detect_head = model.model[-1]

        check(
            conv.in_channels == expected_channels,
            f"Model in_channels={conv.in_channels}, expected={expected_channels}",
        )
        check(
            getattr(detect_head, "nc", None) == EXPECTED_CLASSES,
            "Detection head class count mismatch",
        )

        weight = conv.weight.detach().cpu()
        check(
            torch.equal(weight[:, :3], reference_rgb),
            "Trainer changed pretrained RGB first-conv weights",
        )

        for channel_index, channel_name in enumerate(
            expected_names[3:],
            start=3,
        ):
            check(
                torch.count_nonzero(weight[:, channel_index]).item() == 0,
                f"{channel_name} first-conv weights are not zero initialized",
            )

        print("[PASS] DetectionModel channels, nc=12, and pretrained initialization")

        # ----------------------------------------------------
        # Trainer -> Dataset -> DataLoader contract
        # ----------------------------------------------------
        train_dataset = trainer.train_loader.dataset
        val_dataset = trainer.test_loader.dataset

        for mode, dataset in (
            ("train", train_dataset),
            ("val", val_dataset),
        ):
            check(
                isinstance(dataset, MultimodalYOLODataset),
                f"{mode} dataset is not MultimodalYOLODataset",
            )
            check(
                dataset.modalities == modalities,
                f"{mode} dataset modalities={dataset.modalities}, expected={modalities}",
            )
            check(
                dataset.num_channels == expected_channels,
                f"{mode} dataset channels={dataset.num_channels}, expected={expected_channels}",
            )
            check(
                dataset.channel_names == expected_names,
                f"{mode} dataset channel order mismatch",
            )

        print(
            "[PASS] train/val datasets: "
            f"train={len(train_dataset)}, val={len(val_dataset)}"
        )

        # ----------------------------------------------------
        # One real training batch through the actual trainer path
        # ----------------------------------------------------
        batch = next(iter(trainer.train_loader))

        check(
            torch.is_tensor(batch["img"]),
            "DataLoader batch image is not a tensor",
        )
        check(
            batch["img"].ndim == 4,
            f"Unexpected batch image shape={tuple(batch['img'].shape)}",
        )
        check(
            batch["img"].shape[1] == expected_channels,
            f"Batch channels={batch['img'].shape[1]}, expected={expected_channels}",
        )

        batch = trainer.preprocess_batch(batch)

        check(
            batch["img"].dtype == torch.float32,
            f"Preprocessed dtype={batch['img'].dtype}",
        )
        check(
            torch.isfinite(batch["img"]).all().item(),
            "Preprocessed image contains non-finite values",
        )
        check(
            float(batch["img"].min()) >= 0.0
            and float(batch["img"].max()) <= 1.0,
            "Preprocessed image is outside [0, 1]",
        )

        trainer.model.train()
        trainer.optimizer.zero_grad(set_to_none=True)

        loss_result = trainer.model(batch)
        check(
            isinstance(loss_result, (tuple, list)) and len(loss_result) >= 1,
            "Trainer model did not return a detection loss tuple/list",
        )

        loss = loss_result[0]
        check(torch.isfinite(loss).all().item(), "Detection loss is not finite")
        loss.sum().backward()

        gradient = conv.weight.grad
        check(gradient is not None, "First-conv gradient is missing")
        check(torch.isfinite(gradient).all().item(), "First-conv gradient is non-finite")
        check(
            torch.count_nonzero(gradient[:, :3]).item() > 0,
            "RGB first-conv gradient is zero",
        )

        for channel_index, channel_name in enumerate(
            expected_names[3:],
            start=3,
        ):
            check(
                torch.count_nonzero(gradient[:, channel_index]).item() > 0,
                f"{channel_name} first-conv gradient is zero",
            )

        print(
            "[PASS] real Trainer batch: "
            f"shape={tuple(batch['img'].shape)}, "
            f"loss={float(loss.sum().detach().cpu()):.6f}, backward=OK"
        )

        success = True

    finally:
        del loss, batch, trainer
        gc.collect()
        torch.cuda.empty_cache()

        # Successful smoke artifacts are disposable. On failure, preserve
        # args.yaml and other files under case_dir for diagnosis.
        if success:
            shutil.rmtree(case_dir, ignore_errors=True)


def main() -> None:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    reference_rgb = preflight()

    for case in CASES:
        run_case(*case, reference_rgb=reference_rgb)

    # Remove empty smoke parent after all successful cases.
    if SMOKE_PROJECT.is_dir() and not any(SMOKE_PROJECT.iterdir()):
        SMOKE_PROJECT.rmdir()

    section("RESULT")
    print("[PASS] configurable multimodal Trainer smoke test")
    print("[PASS] RGBI 4ch Trainer path")
    print("[PASS] RGBD 4ch Trainer path")
    print("[PASS] RGBID 5ch default regression path")


if __name__ == "__main__":
    main()

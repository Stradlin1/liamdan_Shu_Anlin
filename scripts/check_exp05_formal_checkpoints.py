#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import torch
from ultralytics import YOLO

from multimodal_gated_model import (
    Exp05AsymmetricGatedDetectionModel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RUN_DIR = (
    PROJECT_ROOT
    / "runs"
    / "exp05_asymmetric_gated_yolo11s_960"
)

BEST_PATH = (
    RUN_DIR
    / "weights"
    / "best.pt"
)

LAST_PATH = (
    RUN_DIR
    / "weights"
    / "last.pt"
)

EXPECTED_CLASSES = 12
EXPECTED_CHANNELS = 5


def check_checkpoint(path: Path) -> None:

    print()
    print("=" * 100)
    print("Checking:", path)
    print("=" * 100)

    if not path.is_file():
        raise FileNotFoundError(path)

    wrapper = YOLO(str(path))
    model = wrapper.model

    print("Model class:", type(model).__name__)

    if not isinstance(
        model,
        Exp05AsymmetricGatedDetectionModel,
    ):
        raise AssertionError(
            "Checkpoint did not reload as "
            "Exp05AsymmetricGatedDetectionModel"
        )

    # RGB main path
    rgb_conv = model.model[0].conv

    print(
        "RGB first conv:",
        rgb_conv,
    )

    if rgb_conv.in_channels != 3:
        raise AssertionError(
            f"RGB main path expected 3ch, got {rgb_conv.in_channels}"
        )

    # Aux paths
    depth_conv = model.depth_encoder[0].conv
    ir_conv = model.ir_encoder[0].conv

    if depth_conv.in_channels != 1:
        raise AssertionError("Depth encoder is not 1ch")

    if ir_conv.in_channels != 1:
        raise AssertionError("IR encoder is not 1ch")

    # Detect head
    head = model.model[-1]

    print(
        "Detect nc:",
        head.nc,
    )

    if head.nc != EXPECTED_CLASSES:
        raise AssertionError(
            f"Expected {EXPECTED_CLASSES} classes, got {head.nc}"
        )

    # Gates
    stats = model.get_gate_statistics()

    print(
        "Depth gate:",
        stats["depth"],
    )

    print(
        "IR gate:",
        stats["ir"],
    )

    # requires_grad=False is NORMAL for stripped inference checkpoint
    print(
        "Depth gate requires_grad:",
        model.depth_gate.logits.requires_grad,
    )

    print(
        "IR gate requires_grad:",
        model.ir_gate.logits.requires_grad,
    )

    # Check finite state
    for name, tensor in model.state_dict().items():

        if (
            tensor.is_floating_point()
            and not torch.isfinite(tensor).all()
        ):
            raise AssertionError(
                f"Non-finite tensor: {name}"
            )

    print("[PASS] all state tensors finite")

    # Real 5-channel inference
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.float().to(device).eval()

    x = torch.zeros(
        1,
        EXPECTED_CHANNELS,
        320,
        320,
        device=device,
    )

    with torch.inference_mode():
        out = model(x)

    pred = (
        out[0]
        if isinstance(out, tuple)
        else out
    )

    print(
        "Prediction shape:",
        tuple(pred.shape),
    )

    expected = (
        1,
        4 + EXPECTED_CLASSES,
        2100,
    )

    if tuple(pred.shape) != expected:
        raise AssertionError(
            f"Expected {expected}, got {tuple(pred.shape)}"
        )

    if not torch.isfinite(pred).all():
        raise AssertionError(
            "Prediction contains NaN/Inf"
        )

    print("[PASS] Exp05 checkpoint structure")
    print("[PASS] 5-channel forward")
    print("[PASS]", path.name)


def main():

    check_checkpoint(BEST_PATH)
    check_checkpoint(LAST_PATH)

    print()
    print("=" * 100)
    print("FINAL RESULT")
    print("=" * 100)
    print("best.pt : PASS")
    print("last.pt : PASS")
    print("STATUS = PASS")


if __name__ == "__main__":
    main()

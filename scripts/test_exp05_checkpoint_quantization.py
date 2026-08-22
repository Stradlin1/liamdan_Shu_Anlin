#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exp05 checkpoint quantization audit.

Purpose:
    Distinguish REAL learned parameter changes from differences caused only
    by Ultralytics FP16 checkpoint serialization.

Compare:

    trained best.pt

against:

    fresh Exp05 initialization -> .half()

For representative auxiliary parameters.
"""

from pathlib import Path

import torch
from ultralytics import YOLO

from multimodal_gated_model import (
    PROJECT_ROOT,
    PRETRAINED_MODEL_PATH,
    build_exp05_yolo11s_gated,
)


BEST_PATH = (
    PROJECT_ROOT
    / "runs"
    / "exp05_asymmetric_gated_smoke_960"
    / "weights"
    / "best.pt"
)

LAST_PATH = (
    PROJECT_ROOT
    / "runs"
    / "exp05_asymmetric_gated_smoke_960"
    / "weights"
    / "last.pt"
)


COMPETITION_NAMES = {
    0: "person",
    1: "boat",
    2: "animal",
    3: "seat",
    4: "sign",
    5: "bicycle",
    6: "car",
    7: "ball",
    8: "light",
    9: "garbage_can",
    10: "uav",
    11: "tricycle",
}


TRACKED_KEYS = (
    "depth_gate.logits",
    "ir_gate.logits",
    "depth_encoder.0.conv.weight",
    "ir_encoder.0.conv.weight",
    "ir_to_p4.proj.weight",
)


def compare_tensor(
    name: str,
    trained: torch.Tensor,
    fresh_half: torch.Tensor,
) -> bool:

    trained = (
        trained.detach()
        .cpu()
        .float()
    )

    fresh_half = (
        fresh_half.detach()
        .cpu()
        .float()
    )

    if trained.shape != fresh_half.shape:
        raise AssertionError(
            f"{name}: shape mismatch "
            f"{tuple(trained.shape)} vs "
            f"{tuple(fresh_half.shape)}"
        )

    delta = (
        trained
        - fresh_half
    ).abs()

    equal = torch.equal(
        trained,
        fresh_half,
    )

    print()
    print(name)
    print(
        "  exact fresh-FP16 equality:",
        equal,
    )
    print(
        "  changed elements          :",
        torch.count_nonzero(
            delta
        ).item(),
        "/",
        delta.numel(),
    )
    print(
        "  delta abs max             :",
        delta.max().item(),
    )
    print(
        "  delta abs sum             :",
        delta.sum().item(),
    )

    return not equal


def load_state(
    path: Path,
):

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    model = YOLO(
        str(path)
    ).model

    if model is None:
        raise RuntimeError(
            f"No model in {path}"
        )

    return model.state_dict()


def main():

    print("=" * 100)
    print(
        "AIC2026 Exp05 - checkpoint FP16 quantization audit"
    )
    print("=" * 100)

    print(
        "Pretrained:",
        PRETRAINED_MODEL_PATH,
    )

    print(
        "best.pt   :",
        BEST_PATH,
    )

    print(
        "last.pt   :",
        LAST_PATH,
    )

    # --------------------------------------------------------
    # Fresh reference.
    # --------------------------------------------------------

    fresh = build_exp05_yolo11s_gated(
        pretrained_path=PRETRAINED_MODEL_PATH,
        nc=12,
        names=COMPETITION_NAMES,
        verbose=False,
    )

    # Ultralytics checkpoint save path serializes EMA as FP16.
    fresh = (
        fresh
        .eval()
        .half()
    )

    fresh_state = (
        fresh.state_dict()
    )

    best_state = load_state(
        BEST_PATH
    )

    last_state = load_state(
        LAST_PATH
    )

    print()
    print("=" * 100)
    print(
        "BEST.PT vs FRESH FP16 INITIALIZATION"
    )
    print("=" * 100)

    best_changed = {}

    for key in TRACKED_KEYS:

        if key not in fresh_state:
            raise KeyError(
                f"Fresh model missing: {key}"
            )

        if key not in best_state:
            raise KeyError(
                f"best.pt missing: {key}"
            )

        best_changed[key] = compare_tensor(
            name=key,
            trained=best_state[key],
            fresh_half=fresh_state[key],
        )

    print()
    print("=" * 100)
    print(
        "LAST.PT vs FRESH FP16 INITIALIZATION"
    )
    print("=" * 100)

    last_changed = {}

    for key in TRACKED_KEYS:

        last_changed[key] = compare_tensor(
            name=key,
            trained=last_state[key],
            fresh_half=fresh_state[key],
        )

    print()
    print("=" * 100)
    print(
        "SUMMARY"
    )
    print("=" * 100)

    for key in TRACKED_KEYS:

        print(
            f"{key:<35} "
            f"best_changed={best_changed[key]!s:<5} "
            f"last_changed={last_changed[key]!s:<5}"
        )

    print()
    print(
        "Interpretation:"
    )

    print(
        "changed=False means the checkpoint tensor is exactly equal "
        "to an UNTRAINED Exp05 tensor after the same FP16 serialization."
    )

    print(
        "Therefore any apparent difference from the original FP32 "
        "initialization was only checkpoint quantization."
    )

    print()
    print(
        "AUDIT COMPLETE"
    )


if __name__ == "__main__":
    main()

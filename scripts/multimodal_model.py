#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 Exp04
RGB + IR + Depth 5-channel Early Fusion YOLO11s

Input channel order:
    [R, G, B, IR, Depth]

Initialization strategy:
    RGB:
        Copy the original pretrained YOLO11s first-conv weights exactly.

    IR:
        Zero initialization.

    Depth:
        Zero initialization.

All layers except the first input convolution remain exactly as loaded
from pretrained/yolo11s.pt.

Important:
    This file only defines/builds the 5-channel model.

    Forward / loss / backward / gradient verification belongs to:
        scripts/test_exp04_5ch_model.py
"""

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from ultralytics import YOLO


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRETRAINED_MODEL_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "yolo11s.pt"
)


# ============================================================
# Multimodal channel definition
# ============================================================

RGB_CHANNELS = 3
MULTIMODAL_CHANNELS = 5

CHANNEL_NAMES: Tuple[str, ...] = (
    "R",
    "G",
    "B",
    "IR",
    "Depth",
)


# ============================================================
# First-convolution helpers
# ============================================================

def get_first_conv2d(yolo_model: YOLO) -> nn.Conv2d:
    """
    Return the actual torch.nn.Conv2d used by the first YOLO stem block.

    Expected Ultralytics structure:

        YOLO
          └── model
                └── model[0]
                      └── conv
                            └── nn.Conv2d

    We deliberately check the structure instead of silently assuming it,
    so an incompatible Ultralytics source version fails immediately.
    """

    detection_model = yolo_model.model

    if detection_model is None:
        raise RuntimeError(
            "YOLO wrapper contains no underlying model."
        )

    if not hasattr(detection_model, "model"):
        raise RuntimeError(
            "Unexpected Ultralytics model structure: "
            "detection_model has no '.model' attribute."
        )

    layers = detection_model.model

    if len(layers) == 0:
        raise RuntimeError(
            "Unexpected Ultralytics model structure: "
            "model layer list is empty."
        )

    first_block = layers[0]

    if not hasattr(first_block, "conv"):
        raise RuntimeError(
            "Unexpected YOLO first block: "
            f"{type(first_block).__name__} has no '.conv' attribute."
        )

    first_conv = first_block.conv

    if not isinstance(first_conv, nn.Conv2d):
        raise TypeError(
            "Expected first_block.conv to be torch.nn.Conv2d, "
            f"but got {type(first_conv).__name__}."
        )

    return first_conv


def _replace_first_conv_with_5ch(
    yolo_model: YOLO,
) -> None:
    """
    Replace YOLO11s input Conv2d:

        3 -> C_out

    with:

        5 -> C_out

    Initialization:

        channel 0 = pretrained R
        channel 1 = pretrained G
        channel 2 = pretrained B
        channel 3 = 0               # IR
        channel 4 = 0               # Depth

    BatchNorm, activation and every later layer are left untouched.
    """

    detection_model = yolo_model.model
    first_block = detection_model.model[0]

    old_conv = get_first_conv2d(yolo_model)

    if old_conv.in_channels != RGB_CHANNELS:
        raise RuntimeError(
            "Expected pretrained YOLO11s first convolution to have "
            f"{RGB_CHANNELS} input channels, "
            f"but got {old_conv.in_channels}."
        )

    # The normal YOLO stem is a standard convolution.
    # Explicitly reject an unexpected grouped/depthwise input convolution.
    if old_conv.groups != 1:
        raise RuntimeError(
            "Expected YOLO first convolution groups=1, "
            f"but got groups={old_conv.groups}."
        )

    old_weight = old_conv.weight.detach().clone()

    old_bias = None
    if old_conv.bias is not None:
        old_bias = old_conv.bias.detach().clone()

    # Preserve all structural properties of the pretrained convolution.
    new_conv = nn.Conv2d(
        in_channels=MULTIMODAL_CHANNELS,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=(old_conv.bias is not None),
        padding_mode=old_conv.padding_mode,
        device=old_conv.weight.device,
        dtype=old_conv.weight.dtype,
    )

    # --------------------------------------------------------
    # Explicit deterministic initialization
    # --------------------------------------------------------
    #
    # First set all 5 channels to zero.
    #
    # Then restore the pretrained RGB weights exactly.
    #
    # Therefore at initialization:
    #
    #   Conv(R,G,B,I,D)
    #
    # is equivalent to:
    #
    #   Conv_pretrained(R,G,B)
    #
    # because the IR and Depth contributions are exactly zero.
    # --------------------------------------------------------

    with torch.no_grad():

        new_conv.weight.zero_()

        new_conv.weight[
            :,
            0:RGB_CHANNELS,
            :,
            :
        ].copy_(old_weight)

        if old_bias is not None:
            new_conv.bias.copy_(old_bias)

    first_block.conv = new_conv

    # --------------------------------------------------------
    # Keep model metadata consistent where possible.
    #
    # The custom trainer will still explicitly construct/use a
    # 5-channel model later. This metadata update is not relied on
    # as the sole mechanism for model construction.
    # --------------------------------------------------------

    model_yaml = getattr(
        detection_model,
        "yaml",
        None,
    )

    if isinstance(model_yaml, dict):

        # Different Ultralytics revisions may use different names.
        # Keeping both at 5 is harmless and prevents stale 3-channel
        # metadata from being propagated.
        model_yaml["ch"] = MULTIMODAL_CHANNELS
        model_yaml["channels"] = MULTIMODAL_CHANNELS


# ============================================================
# Initialization verification
# ============================================================

def _verify_initialization(
    original_rgb_weight: torch.Tensor,
    yolo_model: YOLO,
) -> None:
    """
    Internal fail-fast verification.

    Full model-level testing is intentionally left for
    test_exp04_5ch_model.py.
    """

    conv = get_first_conv2d(yolo_model)

    if conv.in_channels != MULTIMODAL_CHANNELS:
        raise AssertionError(
            "5-channel conversion failed: "
            f"in_channels={conv.in_channels}"
        )

    new_weight = conv.weight.detach()

    # RGB must be bit-for-bit equal to pretrained weights.
    if not torch.equal(
        new_weight[:, 0:3],
        original_rgb_weight,
    ):
        max_diff = (
            new_weight[:, 0:3]
            - original_rgb_weight
        ).abs().max().item()

        raise AssertionError(
            "Pretrained RGB first-conv weights were not copied exactly. "
            f"max_abs_diff={max_diff}"
        )

    # IR must be exactly zero.
    if torch.count_nonzero(
        new_weight[:, 3]
    ).item() != 0:
        raise AssertionError(
            "IR first-conv weights are not zero initialized."
        )

    # Depth must be exactly zero.
    if torch.count_nonzero(
        new_weight[:, 4]
    ).item() != 0:
        raise AssertionError(
            "Depth first-conv weights are not zero initialized."
        )


# ============================================================
# Public model builder
# ============================================================

def build_yolo11s_5ch(
    pretrained_path: Path = PRETRAINED_MODEL_PATH,
) -> YOLO:
    """
    Build Exp04 5-channel YOLO11s.

    Workflow:

        1. Load complete pretrained YOLO11s checkpoint.
        2. Preserve the original pretrained RGB stem weights.
        3. Replace only the first Conv2d from 3 input channels to 5.
        4. Copy pretrained RGB channels exactly.
        5. Zero initialize IR and Depth channels.
        6. Leave every other pretrained parameter untouched.

    Returns:
        ultralytics.YOLO
    """

    pretrained_path = Path(
        pretrained_path
    ).resolve()

    if not pretrained_path.is_file():
        raise FileNotFoundError(
            "Pretrained YOLO11s checkpoint not found:\n"
            f"  {pretrained_path}"
        )

    # Load the FULL pretrained checkpoint first.
    #
    # This is deliberate:
    #
    # instead of rebuilding a fresh network and hoping every compatible
    # tensor is transferred, the complete pretrained model is loaded and
    # then ONLY the input convolution is replaced.
    #
    # Therefore all other pretrained weights remain untouched.
    yolo_model = YOLO(
        str(pretrained_path)
    )

    old_conv = get_first_conv2d(
        yolo_model
    )

    original_rgb_weight = (
        old_conv
        .weight
        .detach()
        .clone()
    )

    _replace_first_conv_with_5ch(
        yolo_model
    )

    _verify_initialization(
        original_rgb_weight=original_rgb_weight,
        yolo_model=yolo_model,
    )

    return yolo_model


# ============================================================
# Standalone construction check
# ============================================================

def main() -> None:
    """
    Construction-only sanity check.

    This does NOT replace test_exp04_5ch_model.py.
    """

    print("=" * 80)
    print("AIC2026 Exp04 - 5-channel YOLO11s")
    print("=" * 80)

    print(
        "Project root :",
        PROJECT_ROOT,
    )

    print(
        "Pretrained   :",
        PRETRAINED_MODEL_PATH,
    )

    print(
        "Channels     :",
        MULTIMODAL_CHANNELS,
    )

    print(
        "Channel order:",
        CHANNEL_NAMES,
    )

    print("-" * 80)

    model = build_yolo11s_5ch()

    conv = get_first_conv2d(
        model
    )

    weight = (
        conv
        .weight
        .detach()
    )

    print(
        "First conv   :",
        conv,
    )

    print(
        "Weight shape :",
        tuple(weight.shape),
    )

    print(
        "in_channels  :",
        conv.in_channels,
    )

    print(
        "out_channels :",
        conv.out_channels,
    )

    print(
        "IR abs max   :",
        weight[:, 3].abs().max().item(),
    )

    print(
        "Depth abs max:",
        weight[:, 4].abs().max().item(),
    )

    print("-" * 80)
    print("Construction : PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()

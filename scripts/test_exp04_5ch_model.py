#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 Exp04
5-channel YOLO11s model-level smoke test

Tests:

1. First Conv2d in_channels == 5
2. RGB pretrained first-conv weights copied exactly
3. IR first-conv weights == 0
4. Depth first-conv weights == 0
5. All other pretrained state_dict tensors remain unchanged
6. [B, 5, 960, 960] forward succeeds
7. Real YOLO detection loss succeeds
8. backward succeeds
9. RGB first-conv gradient is non-zero
10. IR first-conv gradient is non-zero
11. Depth first-conv gradient is non-zero

This is a MODEL smoke test.

It does not use the multimodal Dataset yet.
Dataset -> Trainer integration belongs to the next stage.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Iterable

import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg

from multimodal_model import (
    CHANNEL_NAMES,
    MULTIMODAL_CHANNELS,
    PRETRAINED_MODEL_PATH,
    PROJECT_ROOT,
    build_yolo11s_5ch,
    get_first_conv2d,
)


# ============================================================
# Test configuration
# ============================================================

IMAGE_SIZE = 960
BATCH_SIZE = 1

SEED = 2026


# ============================================================
# Utility functions
# ============================================================

def section(title: str) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def pass_line(name: str) -> None:

    print(
        f"[PASS] {name}"
    )


def find_parameter_name(
    module: torch.nn.Module,
    target_parameter: torch.nn.Parameter,
) -> str:
    """
    Find the state_dict/named_parameters name corresponding
    to a specific Parameter object.
    """

    for name, parameter in module.named_parameters():

        if parameter is target_parameter:
            return name

    raise RuntimeError(
        "Could not find parameter name."
    )


def iter_tensors(
    obj: Any,
) -> Iterable[torch.Tensor]:
    """
    Recursively yield tensors from nested YOLO outputs.

    Supports:
        Tensor
        dict
        list
        tuple
    """

    if torch.is_tensor(obj):

        yield obj
        return

    if isinstance(obj, dict):

        for value in obj.values():
            yield from iter_tensors(value)

        return

    if isinstance(
        obj,
        (list, tuple),
    ):

        for value in obj:
            yield from iter_tensors(value)

        return


def assert_tensor_finite(
    tensor: torch.Tensor,
    description: str,
) -> None:

    if not torch.isfinite(tensor).all():

        raise AssertionError(
            f"{description} contains NaN or Inf."
        )


def tensor_stats(
    tensor: torch.Tensor,
) -> str:

    detached = tensor.detach()

    return (
        f"shape={tuple(detached.shape)}, "
        f"dtype={detached.dtype}, "
        f"device={detached.device}, "
        f"min={detached.min().item():.6g}, "
        f"max={detached.max().item():.6g}"
    )


# ============================================================
# Determinism
# ============================================================

def set_seed() -> None:

    torch.manual_seed(
        SEED
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            SEED
        )


# ============================================================
# Pretrained reference
# ============================================================

def load_pretrained_reference():
    """
    Load untouched 3-channel YOLO11s and clone its state_dict.

    This provides an independent reference for checking that:

        RGB stem weights are unchanged

    and

        every non-input-conv pretrained tensor remains unchanged.
    """

    if not PRETRAINED_MODEL_PATH.is_file():

        raise FileNotFoundError(
            f"Pretrained model not found:\n"
            f"  {PRETRAINED_MODEL_PATH}"
        )

    reference_yolo = YOLO(
        str(PRETRAINED_MODEL_PATH)
    )

    reference_model = (
        reference_yolo.model
    )

    reference_conv = get_first_conv2d(
        reference_yolo
    )

    if reference_conv.in_channels != 3:

        raise AssertionError(
            "Reference YOLO11s first conv must have "
            f"3 channels, got {reference_conv.in_channels}."
        )

    first_conv_weight_name = (
        find_parameter_name(
            reference_model,
            reference_conv.weight,
        )
    )

    reference_state = {
        key: value.detach().cpu().clone()
        for key, value
        in reference_model.state_dict().items()
    }

    return (
        reference_yolo,
        reference_state,
        first_conv_weight_name,
    )


# ============================================================
# Initialization tests
# ============================================================

def test_initialization(
    multimodal_yolo: YOLO,
    reference_state: dict[str, torch.Tensor],
    first_conv_weight_name: str,
) -> None:

    section(
        "1. Initialization verification"
    )

    model = multimodal_yolo.model

    first_conv = get_first_conv2d(
        multimodal_yolo
    )

    weight = (
        first_conv
        .weight
        .detach()
        .cpu()
    )

    print(
        "First conv:",
        first_conv,
    )

    print(
        "Weight shape:",
        tuple(weight.shape),
    )

    # --------------------------------------------------------
    # 5-channel structure
    # --------------------------------------------------------

    if first_conv.in_channels != MULTIMODAL_CHANNELS:

        raise AssertionError(
            "Expected first conv in_channels="
            f"{MULTIMODAL_CHANNELS}, "
            f"got {first_conv.in_channels}."
        )

    pass_line(
        "first conv in_channels == 5"
    )

    # --------------------------------------------------------
    # RGB pretrained copy
    # --------------------------------------------------------

    pretrained_rgb = (
        reference_state[
            first_conv_weight_name
        ]
    )

    if pretrained_rgb.shape[1] != 3:

        raise AssertionError(
            "Reference first-conv tensor is not 3-channel."
        )

    if not torch.equal(
        weight[:, 0:3],
        pretrained_rgb,
    ):

        max_diff = (
            weight[:, 0:3]
            - pretrained_rgb
        ).abs().max().item()

        raise AssertionError(
            "RGB pretrained weights differ. "
            f"max_abs_diff={max_diff}"
        )

    pass_line(
        "RGB first-conv pretrained weights copied exactly"
    )

    # --------------------------------------------------------
    # IR zero initialization
    # --------------------------------------------------------

    ir_weight = (
        weight[:, 3]
    )

    ir_nonzero = (
        torch.count_nonzero(
            ir_weight
        ).item()
    )

    print(
        "IR weight nonzero count:",
        ir_nonzero,
    )

    print(
        "IR weight abs max:",
        ir_weight.abs().max().item(),
    )

    if ir_nonzero != 0:

        raise AssertionError(
            "IR first-conv weights are not exactly zero."
        )

    pass_line(
        "IR first-conv weights == 0"
    )

    # --------------------------------------------------------
    # Depth zero initialization
    # --------------------------------------------------------

    depth_weight = (
        weight[:, 4]
    )

    depth_nonzero = (
        torch.count_nonzero(
            depth_weight
        ).item()
    )

    print(
        "Depth weight nonzero count:",
        depth_nonzero,
    )

    print(
        "Depth weight abs max:",
        depth_weight.abs().max().item(),
    )

    if depth_nonzero != 0:

        raise AssertionError(
            "Depth first-conv weights are not exactly zero."
        )

    pass_line(
        "Depth first-conv weights == 0"
    )

    # --------------------------------------------------------
    # Every OTHER state_dict tensor must remain exactly
    # identical to the pretrained YOLO11s checkpoint.
    # --------------------------------------------------------

    multimodal_state = (
        model.state_dict()
    )

    missing_keys = []

    unexpected_keys = []

    changed_keys = []

    for key in reference_state:

        if key not in multimodal_state:

            missing_keys.append(
                key
            )

            continue

        # The first Conv weight is intentionally changed
        # from:
        #
        #   [Cout, 3, K, K]
        #
        # to:
        #
        #   [Cout, 5, K, K]
        #
        if key == first_conv_weight_name:
            continue

        reference_tensor = (
            reference_state[key]
        )

        multimodal_tensor = (
            multimodal_state[key]
            .detach()
            .cpu()
        )

        if (
            reference_tensor.shape
            != multimodal_tensor.shape
        ):

            changed_keys.append(
                (
                    key,
                    "shape",
                    tuple(reference_tensor.shape),
                    tuple(multimodal_tensor.shape),
                )
            )

            continue

        if not torch.equal(
            reference_tensor,
            multimodal_tensor,
        ):

            max_diff = (
                reference_tensor
                - multimodal_tensor
            ).abs().max().item()

            changed_keys.append(
                (
                    key,
                    "value",
                    max_diff,
                )
            )

    for key in multimodal_state:

        if key not in reference_state:

            unexpected_keys.append(
                key
            )

    if missing_keys:

        raise AssertionError(
            "Missing pretrained state_dict keys:\n"
            + "\n".join(
                missing_keys
            )
        )

    if unexpected_keys:

        raise AssertionError(
            "Unexpected multimodal state_dict keys:\n"
            + "\n".join(
                unexpected_keys
            )
        )

    if changed_keys:

        raise AssertionError(
            "Parameters/buffers other than the first input "
            "convolution changed unexpectedly:\n"
            + "\n".join(
                str(item)
                for item in changed_keys
            )
        )

    checked_count = (
        len(reference_state) - 1
    )

    print(
        "Unchanged state_dict tensors checked:",
        checked_count,
    )

    pass_line(
        "all other pretrained tensors remain exactly unchanged"
    )


# ============================================================
# Forward test
# ============================================================

def test_forward(
    multimodal_yolo: YOLO,
    device: torch.device,
) -> torch.Tensor:

    section(
        "2. 5-channel forward"
    )

    model = multimodal_yolo.model

    model.to(
        device
    )

    model.eval()

    # --------------------------------------------------------
    # Synthetic multimodal input
    #
    # Channel semantics:
    #
    #   0 R
    #   1 G
    #   2 B
    #   3 IR
    #   4 Depth
    #
    # Values are already represented in trainer-style float
    # [0, 1] range.
    # --------------------------------------------------------

    image = torch.rand(
        (
            BATCH_SIZE,
            MULTIMODAL_CHANNELS,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        dtype=torch.float32,
        device=device,
    )

    print(
        "Input:",
        tensor_stats(
            image
        ),
    )

    print(
        "Channel order:",
        CHANNEL_NAMES,
    )

    with torch.inference_mode():

        output = model(
            image
        )

    output_tensors = list(
        iter_tensors(
            output
        )
    )

    if not output_tensors:

        raise AssertionError(
            "Forward returned no tensors."
        )

    for index, tensor in enumerate(
        output_tensors
    ):

        assert_tensor_finite(
            tensor,
            f"forward output tensor #{index}",
        )

    print(
        "Forward tensor count:",
        len(output_tensors),
    )

    print(
        "First output tensors:"
    )

    for index, tensor in enumerate(
        output_tensors[:8]
    ):

        print(
            f"  [{index}] "
            f"{tensor_stats(tensor)}"
        )

    pass_line(
        "[1, 5, 960, 960] forward"
    )

    # Delete inference output before backward smoke test
    # to release memory.
    del output
    del output_tensors

    if device.type == "cuda":

        torch.cuda.empty_cache()

    return image


# ============================================================
# Real YOLO detection loss / backward test
# ============================================================

def test_loss_and_backward(
    multimodal_yolo: YOLO,
    image: torch.Tensor,
    device: torch.device,
) -> None:

    section(
        "3. YOLO detection loss + backward"
    )

    model = multimodal_yolo.model

    # --------------------------------------------------------
    # Direct model(batch) bypasses Ultralytics DetectionTrainer.
    #
    # A freshly loaded DetectionModel may keep model.args as a
    # plain dict, while v8DetectionLoss expects attribute access:
    #
    #     self.hyp.box
    #     self.hyp.cls
    #     self.hyp.dfl
    #
    # DetectionTrainer normally prepares this automatically.
    # For this standalone model smoke test we explicitly convert
    # the existing model args to Ultralytics config namespace.
    # --------------------------------------------------------

    if isinstance(model.args, dict):
        model.args = get_cfg(
            overrides=model.args
        )

    print(
        "model.args type:",
        type(model.args).__name__,
    )

    print(
        "Loss gains:"
    )

    print(
        "  box =",
        model.args.box,
    )

    print(
        "  cls =",
        model.args.cls,
    )

    print(
        "  dfl =",
        model.args.dfl,
    )

    # Defensive reset:
    # criterion stores self.hyp when it is first constructed.
    # Recreate it after preparing model.args.
    if hasattr(
        model,
        "criterion",
    ):
        delattr(
            model,
            "criterion",
        )

    model.train()

    model.zero_grad(
        set_to_none=True
    )

    # --------------------------------------------------------
    # Synthetic YOLO detection targets.
    #
    # Normalized xywh format.
    #
    # We deliberately use two objects so that box /
    # classification / DFL branches all participate in the
    # normal detection loss computation.
    # --------------------------------------------------------

    cls = torch.tensor(
        [
            [0.0],
            [3.0],
        ],
        dtype=torch.float32,
        device=device,
    )

    bboxes = torch.tensor(
        [
            [
                0.35,
                0.40,
                0.18,
                0.24,
            ],
            [
                0.68,
                0.62,
                0.22,
                0.20,
            ],
        ],
        dtype=torch.float32,
        device=device,
    )

    batch_idx = torch.tensor(
        [
            0.0,
            0.0,
        ],
        dtype=torch.float32,
        device=device,
    )

    batch = {
        "img": image,
        "cls": cls,
        "bboxes": bboxes,
        "batch_idx": batch_idx,
    }

    print(
        "Target classes:",
        cls.flatten().tolist(),
    )

    print(
        "Target boxes:"
    )

    for box in bboxes.tolist():

        print(
            " ",
            box,
        )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Because input is a dict, Ultralytics BaseModel.forward()
    # routes into:
    #
    #       model.loss(batch)
    #
    # This therefore exercises the actual YOLO detection loss,
    # not an artificial output.mean() surrogate loss.
    # --------------------------------------------------------

    loss_result = model(
        batch
    )

    if not isinstance(
        loss_result,
        (tuple, list),
    ):

        raise AssertionError(
            "Expected YOLO detection model to return "
            "(loss, loss_items), "
            f"got {type(loss_result).__name__}."
        )

    if len(loss_result) != 2:

        raise AssertionError(
            "Expected YOLO detection loss result length=2, "
            f"got {len(loss_result)}."
        )

    loss = loss_result[0]
    loss_items = loss_result[1]

    if not torch.is_tensor(
        loss
    ):

        raise TypeError(
            "YOLO loss is not a Tensor."
        )

    assert_tensor_finite(
        loss,
        "YOLO detection loss",
    )

    print(
        "Loss:",
        loss.detach().cpu(),
    )

    if torch.is_tensor(
        loss_items
    ):

        print(
            "Loss items:",
            loss_items.detach().cpu(),
        )

    else:

        print(
            "Loss items:",
            loss_items,
        )

    # Ensure scalar backward even if a future Ultralytics
    # revision changes the loss tensor shape.
    backward_loss = (
        loss.sum()
    )

    backward_loss.backward()

    pass_line(
        "YOLO detection loss"
    )

    pass_line(
        "loss.backward()"
    )

    # --------------------------------------------------------
    # First-layer gradient verification
    # --------------------------------------------------------

    first_conv = get_first_conv2d(
        multimodal_yolo
    )

    grad = (
        first_conv
        .weight
        .grad
    )

    if grad is None:

        raise AssertionError(
            "First-conv gradient is None after backward."
        )

    assert_tensor_finite(
        grad,
        "first-conv gradient",
    )

    grad = (
        grad
        .detach()
        .cpu()
    )

    rgb_grad = (
        grad[:, 0:3]
    )

    ir_grad = (
        grad[:, 3]
    )

    depth_grad = (
        grad[:, 4]
    )

    rgb_nonzero = (
        torch.count_nonzero(
            rgb_grad
        ).item()
    )

    ir_nonzero = (
        torch.count_nonzero(
            ir_grad
        ).item()
    )

    depth_nonzero = (
        torch.count_nonzero(
            depth_grad
        ).item()
    )

    print()
    print(
        "First-conv gradient shape:",
        tuple(
            grad.shape
        ),
    )

    print()
    print(
        "RGB gradient:"
    )

    print(
        "  nonzero =",
        rgb_nonzero,
    )

    print(
        "  abs max =",
        rgb_grad.abs().max().item(),
    )

    print(
        "  abs sum =",
        rgb_grad.abs().sum().item(),
    )

    print()
    print(
        "IR gradient:"
    )

    print(
        "  nonzero =",
        ir_nonzero,
    )

    print(
        "  abs max =",
        ir_grad.abs().max().item(),
    )

    print(
        "  abs sum =",
        ir_grad.abs().sum().item(),
    )

    print()
    print(
        "Depth gradient:"
    )

    print(
        "  nonzero =",
        depth_nonzero,
    )

    print(
        "  abs max =",
        depth_grad.abs().max().item(),
    )

    print(
        "  abs sum =",
        depth_grad.abs().sum().item(),
    )

    if rgb_nonzero == 0:

        raise AssertionError(
            "RGB first-conv gradient is entirely zero."
        )

    pass_line(
        "RGB first-conv gradient != 0"
    )

    if ir_nonzero == 0:

        raise AssertionError(
            "IR first-conv gradient is entirely zero. "
            "Zero-init auxiliary channel is not learning."
        )

    pass_line(
        "IR first-conv gradient != 0"
    )

    if depth_nonzero == 0:

        raise AssertionError(
            "Depth first-conv gradient is entirely zero. "
            "Zero-init auxiliary channel is not learning."
        )

    pass_line(
        "Depth first-conv gradient != 0"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    set_seed()

    section(
        "AIC2026 Exp04 - 5-channel model smoke test"
    )

    print(
        "Project root :",
        PROJECT_ROOT,
    )

    print(
        "Pretrained   :",
        PRETRAINED_MODEL_PATH,
    )

    print(
        "Image size   :",
        IMAGE_SIZE,
    )

    print(
        "Batch size   :",
        BATCH_SIZE,
    )

    print(
        "Channels     :",
        MULTIMODAL_CHANNELS,
    )

    print(
        "Channel order:",
        CHANNEL_NAMES,
    )

    if torch.cuda.is_available():

        device = torch.device(
            "cuda:0"
        )

        print(
            "Device       :",
            device,
        )

        print(
            "GPU          :",
            torch.cuda.get_device_name(
                0
            ),
        )

    else:

        device = torch.device(
            "cpu"
        )

        print(
            "Device       :",
            device,
        )

        print(
            "WARNING      : CUDA unavailable; "
            "960x960 backward may be slow."
        )

    # --------------------------------------------------------
    # Independent untouched pretrained reference
    # --------------------------------------------------------

    section(
        "0. Load pretrained reference"
    )

    (
        reference_yolo,
        reference_state,
        first_conv_weight_name,
    ) = load_pretrained_reference()

    print(
        "Reference first-conv parameter:",
        first_conv_weight_name,
    )

    print(
        "Reference state_dict tensors:",
        len(
            reference_state
        ),
    )

    pass_line(
        "pretrained reference loaded"
    )

    # --------------------------------------------------------
    # Build actual 5-channel model
    # --------------------------------------------------------

    del reference_yolo

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    multimodal_yolo = (
        build_yolo11s_5ch()
    )

    # --------------------------------------------------------
    # Test 1
    # --------------------------------------------------------

    test_initialization(
        multimodal_yolo=multimodal_yolo,
        reference_state=reference_state,
        first_conv_weight_name=first_conv_weight_name,
    )

    # Reference no longer needed.
    del reference_state

    gc.collect()

    # --------------------------------------------------------
    # Test 2
    # --------------------------------------------------------

    image = test_forward(
        multimodal_yolo=multimodal_yolo,
        device=device,
    )

    # --------------------------------------------------------
    # Test 3
    # --------------------------------------------------------

    test_loss_and_backward(
        multimodal_yolo=multimodal_yolo,
        image=image,
        device=device,
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    section(
        "FINAL RESULT"
    )

    print(
        "first conv in_channels = 5          PASS"
    )

    print(
        "RGB pretrained exact copy           PASS"
    )

    print(
        "IR zero initialization              PASS"
    )

    print(
        "Depth zero initialization           PASS"
    )

    print(
        "other pretrained tensors unchanged  PASS"
    )

    print(
        "[1,5,960,960] forward               PASS"
    )

    print(
        "YOLO detection loss                 PASS"
    )

    print(
        "backward                            PASS"
    )

    print(
        "RGB gradient != 0                   PASS"
    )

    print(
        "IR gradient != 0                    PASS"
    )

    print(
        "Depth gradient != 0                 PASS"
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
        "Implement MultimodalDetectionTrainer."
    )


if __name__ == "__main__":
    main()

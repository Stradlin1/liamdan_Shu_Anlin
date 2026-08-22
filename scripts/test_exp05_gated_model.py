#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 Exp05
Asymmetric Gated Feature Fusion model-level smoke test.

Tests
=====

1. Build actual 12-class Exp05 YOLO11s.
2. RGB YOLO layers 0..22 remain exactly pretrained.
3. RGB first Conv2d remains 3-channel.
4. Depth / IR auxiliary first Conv2d are 1-channel.
5. Auxiliary first-conv weights equal:
       W_R + W_G + W_B
6. Auxiliary layers 1..4 exactly copy RGB layers 1..4.
7. Depth gate initial probability ~= 0.10.
8. IR gate initial probability ~= 0.02.
9. IR P4 projector is identity initialized.
10. Real [1,5,960,960] forward succeeds.
11. Runtime fusion route shapes are correct:
       Depth P3  = [1,256,120,120]
       IR P3     = [1,256,120,120]
       IR P4     = [1,256,60,60]
       layer5 in = [1,256,120,120]  # fused P3
       layer7 in = [1,256,60,60]    # fused P4
12. Detect inputs remain P3/P4/P5:
       [128,120,120]
       [256,60,60]
       [512,30,30]
13. Real Ultralytics YOLO detection loss succeeds.
14. backward succeeds.
15. Non-zero finite gradients exist for:
       RGB main path
       Depth encoder
       IR encoder
       IR projector
       Depth gate
       IR gate
       late fused backbone
16. optimizer.step changes all major parameter groups.
17. Exp05 -> Exp05 reload preserves learned auxiliary/gate state.

This is a MODEL smoke test only.

Dataset / Trainer integration belongs to the next stage.
"""

from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.cfg import get_cfg

from multimodal_gated_model import (
    AUX_ENCODER_LAST_LAYER,
    CHANNEL_NAMES,
    DEFAULT_DEPTH_GATE,
    DEFAULT_IR_GATE,
    DEPTH_FUSION_LAYER,
    IR_FUSION_LAYER,
    MULTIMODAL_CHANNELS,
    P3_CHANNELS,
    P4_CHANNELS,
    PRETRAINED_MODEL_PATH,
    PROJECT_ROOT,
    Exp05AsymmetricGatedDetectionModel,
    build_exp05_yolo11s_gated,
    initialize_exp05_from_weights,
    source_has_exp05_parameters,
)


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 960
BATCH_SIZE = 1
SEED = 2026

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

NUM_CLASSES = len(
    COMPETITION_NAMES
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
    name: str,
) -> None:

    print(
        f"[PASS] {name}"
    )


def tensor_stats(
    tensor: torch.Tensor,
) -> str:

    tensor = (
        tensor
        .detach()
    )

    return (
        f"shape={tuple(tensor.shape)}, "
        f"dtype={tensor.dtype}, "
        f"device={tensor.device}, "
        f"min={tensor.min().item():.6g}, "
        f"max={tensor.max().item():.6g}, "
        f"abs_mean={tensor.abs().mean().item():.6g}"
    )


def iter_tensors(
    obj: Any,
) -> Iterable[torch.Tensor]:

    if torch.is_tensor(
        obj
    ):
        yield obj
        return

    if isinstance(
        obj,
        dict,
    ):

        for value in obj.values():
            yield from iter_tensors(
                value
            )

        return

    if isinstance(
        obj,
        (list, tuple),
    ):

        for value in obj:
            yield from iter_tensors(
                value
            )


def assert_finite(
    tensor: torch.Tensor,
    name: str,
) -> None:

    if not torch.isfinite(
        tensor
    ).all():

        raise AssertionError(
            f"{name} contains NaN/Inf."
        )


def assert_exact_tensor(
    actual: torch.Tensor,
    expected: torch.Tensor,
    name: str,
) -> None:

    actual = (
        actual
        .detach()
        .cpu()
    )

    expected = (
        expected
        .detach()
        .cpu()
    )

    if actual.shape != expected.shape:

        raise AssertionError(
            f"{name} shape mismatch: "
            f"{tuple(actual.shape)} vs "
            f"{tuple(expected.shape)}"
        )

    if not torch.equal(
        actual,
        expected,
    ):

        if (
            actual.is_floating_point()
            and expected.is_floating_point()
        ):
            max_diff = (
                actual.float()
                - expected.float()
            ).abs().max().item()
        else:
            max_diff = "N/A"

        raise AssertionError(
            f"{name} differs; "
            f"max_abs_diff={max_diff}"
        )


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


def set_seed() -> None:

    torch.manual_seed(
        SEED
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            SEED
        )


def get_device() -> torch.device:

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

        return device

    print(
        "Device       : cpu"
    )

    print(
        "WARNING      : CUDA unavailable; "
        "960 backward will be slow."
    )

    return torch.device(
        "cpu"
    )


def print_cuda_peak(
    label: str,
) -> None:

    if not torch.cuda.is_available():
        return

    peak = (
        torch.cuda
        .max_memory_allocated()
        / 1024**3
    )

    print(
        f"{label} peak CUDA memory: "
        f"{peak:.3f} GiB"
    )


# ============================================================
# Build independent reference
# ============================================================

def load_reference():

    if not PRETRAINED_MODEL_PATH.is_file():

        raise FileNotFoundError(
            "Pretrained YOLO11s not found:\n"
            f"  {PRETRAINED_MODEL_PATH}"
        )

    yolo = YOLO(
        str(
            PRETRAINED_MODEL_PATH
        )
    )

    model = yolo.model

    if model is None:

        raise RuntimeError(
            "Reference YOLO has no model."
        )

    return (
        yolo,
        model,
    )


# ============================================================
# Test 1: initialization
# ============================================================

def test_initialization(
    model: Exp05AsymmetricGatedDetectionModel,
    reference_model: nn.Module,
) -> None:

    section(
        "1. Structure + initialization verification"
    )

    # --------------------------------------------------------
    # Competition head
    # --------------------------------------------------------

    head = model.model[-1]

    if getattr(
        head,
        "nc",
        None,
    ) != NUM_CLASSES:

        raise AssertionError(
            "Detection head nc mismatch: "
            f"{getattr(head, 'nc', None)}"
        )

    print(
        "Detection classes:",
        head.nc,
    )

    pass_line(
        "competition Detect head nc == 12"
    )

    # --------------------------------------------------------
    # Strides
    # --------------------------------------------------------

    expected_stride = torch.tensor(
        [
            8.0,
            16.0,
            32.0,
        ],
        dtype=model.stride.dtype,
        device=model.stride.device,
    )

    if not torch.equal(
        model.stride,
        expected_stride,
    ):

        raise AssertionError(
            "Unexpected model stride: "
            f"{model.stride}"
        )

    print(
        "Model stride:",
        model.stride.tolist(),
    )

    pass_line(
        "model strides == [8,16,32]"
    )

    # --------------------------------------------------------
    # Main RGB path must remain three-channel.
    # --------------------------------------------------------

    rgb_conv = first_conv(
        model.model[0]
    )

    if rgb_conv.in_channels != 3:

        raise AssertionError(
            "RGB main path first conv is not 3-channel."
        )

    print(
        "RGB first conv:",
        rgb_conv,
    )

    pass_line(
        "RGB main-path first Conv2d == 3 channels"
    )

    # --------------------------------------------------------
    # All normal YOLO layers BEFORE Detect must be exact
    # pretrained copies.
    #
    # nc affects the Detect layer, so model[23] is excluded.
    # --------------------------------------------------------

    checked_tensors = 0

    for layer_index in range(
        23
    ):

        reference_state = (
            reference_model
            .model[layer_index]
            .state_dict()
        )

        actual_state = (
            model
            .model[layer_index]
            .state_dict()
        )

        if (
            reference_state.keys()
            != actual_state.keys()
        ):

            raise AssertionError(
                f"RGB layer {layer_index} "
                "state_dict key mismatch."
            )

        for key in reference_state:

            assert_exact_tensor(
                actual=actual_state[key],
                expected=reference_state[key],
                name=(
                    f"RGB model[{layer_index}].{key}"
                ),
            )

            checked_tensors += 1

    print(
        "Exact pretrained RGB tensors checked:",
        checked_tensors,
    )

    pass_line(
        "RGB YOLO layers 0..22 exactly preserve pretrained weights"
    )

    # --------------------------------------------------------
    # Auxiliary first convolutions
    # --------------------------------------------------------

    expected_gray_weight = (
        rgb_conv
        .weight
        .detach()
        .sum(
            dim=1,
            keepdim=True,
        )
    )

    for (
        branch_name,
        encoder,
    ) in (
        (
            "Depth",
            model.depth_encoder,
        ),
        (
            "IR",
            model.ir_encoder,
        ),
    ):

        aux_conv = first_conv(
            encoder[0]
        )

        print(
            f"{branch_name} first conv:",
            aux_conv,
        )

        if aux_conv.in_channels != 1:

            raise AssertionError(
                f"{branch_name} first conv "
                f"in_channels={aux_conv.in_channels}"
            )

        assert_exact_tensor(
            actual=aux_conv.weight,
            expected=expected_gray_weight,
            name=(
                f"{branch_name} gray first-conv init"
            ),
        )

        pass_line(
            f"{branch_name} first Conv2d == 1 channel"
        )

        pass_line(
            f"{branch_name} first Conv2d == RGB weight sum"
        )

        # Layers 1..4 must exactly match the loaded RGB backbone.
        for layer_index in range(
            1,
            AUX_ENCODER_LAST_LAYER + 1,
        ):

            rgb_state = (
                model
                .model[layer_index]
                .state_dict()
            )

            aux_state = (
                encoder[layer_index]
                .state_dict()
            )

            if (
                rgb_state.keys()
                != aux_state.keys()
            ):

                raise AssertionError(
                    f"{branch_name} layer {layer_index} "
                    "state_dict key mismatch."
                )

            for key in rgb_state:

                assert_exact_tensor(
                    actual=aux_state[key],
                    expected=rgb_state[key],
                    name=(
                        f"{branch_name} encoder "
                        f"layer {layer_index}.{key}"
                    ),
                )

        pass_line(
            f"{branch_name} encoder layers 1..4 "
            "exactly copy RGB backbone"
        )

    # --------------------------------------------------------
    # IR projector identity initialization
    # --------------------------------------------------------

    projector = (
        model.ir_to_p4
        .proj
        .weight
        .detach()
    )

    expected_projector = (
        torch.zeros_like(
            projector
        )
    )

    diagonal = torch.arange(
        P4_CHANNELS,
        device=projector.device,
    )

    expected_projector[
        diagonal,
        diagonal,
        0,
        0,
    ] = 1.0

    assert_exact_tensor(
        actual=projector,
        expected=expected_projector,
        name="IR P4 1x1 identity projector",
    )

    bn = model.ir_to_p4.bn

    assert_exact_tensor(
        actual=bn.weight,
        expected=torch.ones_like(
            bn.weight
        ),
        name="IR projector BN weight",
    )

    assert_exact_tensor(
        actual=bn.bias,
        expected=torch.zeros_like(
            bn.bias
        ),
        name="IR projector BN bias",
    )

    pass_line(
        "IR P4 projector is identity initialized"
    )

    # --------------------------------------------------------
    # Gate initialization
    # --------------------------------------------------------

    gate_stats = (
        model.get_gate_statistics()
    )

    print(
        "Depth gate:",
        gate_stats["depth"],
    )

    print(
        "IR gate   :",
        gate_stats["ir"],
    )

    if not math.isclose(
        gate_stats[
            "depth"
        ][
            "mean"
        ],
        DEFAULT_DEPTH_GATE,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):

        raise AssertionError(
            "Depth gate init mismatch."
        )

    if not math.isclose(
        gate_stats[
            "ir"
        ][
            "mean"
        ],
        DEFAULT_IR_GATE,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):

        raise AssertionError(
            "IR gate init mismatch."
        )

    if not model.depth_gate.logits.requires_grad:

        raise AssertionError(
            "Depth gate requires_grad=False."
        )

    if not model.ir_gate.logits.requires_grad:

        raise AssertionError(
            "IR gate requires_grad=False."
        )

    pass_line(
        "Depth gate init ~= 0.10 and trainable"
    )

    pass_line(
        "IR gate init ~= 0.02 and trainable"
    )


# ============================================================
# Test 2: runtime feature graph + forward
# ============================================================

def test_forward_graph(
    model: Exp05AsymmetricGatedDetectionModel,
    device: torch.device,
) -> torch.Tensor:

    section(
        "2. Runtime fusion graph + 960 forward"
    )

    model = (
        model
        .to(
            device
        )
        .float()
        .eval()
    )

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

    # --------------------------------------------------------
    # Direct branch shapes
    # --------------------------------------------------------

    with torch.inference_mode():

        depth_p3 = (
            model.depth_encoder(
                image[:, 4:5]
            )
        )

        ir_p3 = (
            model.ir_encoder(
                image[:, 3:4]
            )
        )

        ir_p4 = (
            model.ir_to_p4(
                ir_p3
            )
        )

    expected_depth_p3 = (
        BATCH_SIZE,
        P3_CHANNELS,
        IMAGE_SIZE // 8,
        IMAGE_SIZE // 8,
    )

    expected_ir_p3 = (
        BATCH_SIZE,
        P3_CHANNELS,
        IMAGE_SIZE // 8,
        IMAGE_SIZE // 8,
    )

    expected_ir_p4 = (
        BATCH_SIZE,
        P4_CHANNELS,
        IMAGE_SIZE // 16,
        IMAGE_SIZE // 16,
    )

    if tuple(
        depth_p3.shape
    ) != expected_depth_p3:

        raise AssertionError(
            "Depth P3 shape mismatch: "
            f"{tuple(depth_p3.shape)}"
        )

    if tuple(
        ir_p3.shape
    ) != expected_ir_p3:

        raise AssertionError(
            "IR P3 shape mismatch: "
            f"{tuple(ir_p3.shape)}"
        )

    if tuple(
        ir_p4.shape
    ) != expected_ir_p4:

        raise AssertionError(
            "IR P4 shape mismatch: "
            f"{tuple(ir_p4.shape)}"
        )

    print(
        "Depth P3:",
        tuple(
            depth_p3.shape
        ),
    )

    print(
        "IR P3   :",
        tuple(
            ir_p3.shape
        ),
    )

    print(
        "IR P4   :",
        tuple(
            ir_p4.shape
        ),
    )

    pass_line(
        "auxiliary runtime feature shapes"
    )

    del depth_p3
    del ir_p3
    del ir_p4

    # --------------------------------------------------------
    # Hook exact inputs to layer 5 / layer 7 / Detect.
    #
    # layer5 pre-hook sees the already Depth-fused P3 tensor.
    # layer7 pre-hook sees the already IR-fused P4 tensor.
    # --------------------------------------------------------

    captured = {}

    def capture_layer5(
        module,
        args,
    ):

        x = args[0]

        captured[
            "layer5_input"
        ] = tuple(
            int(v)
            for v in x.shape
        )

    def capture_layer7(
        module,
        args,
    ):

        x = args[0]

        captured[
            "layer7_input"
        ] = tuple(
            int(v)
            for v in x.shape
        )

    def capture_detect(
        module,
        args,
    ):

        features = args[0]

        captured[
            "detect_inputs"
        ] = [
            tuple(
                int(v)
                for v in feature.shape
            )
            for feature
            in features
        ]

    handles = [
        model
        .model[
            DEPTH_FUSION_LAYER + 1
        ]
        .register_forward_pre_hook(
            capture_layer5
        ),

        model
        .model[
            IR_FUSION_LAYER + 1
        ]
        .register_forward_pre_hook(
            capture_layer7
        ),

        model
        .model[-1]
        .register_forward_pre_hook(
            capture_detect
        ),
    ]

    try:

        if (
            device.type
            == "cuda"
        ):
            torch.cuda.reset_peak_memory_stats()

        with torch.inference_mode():

            output = model(
                image
            )

        if (
            device.type
            == "cuda"
        ):
            torch.cuda.synchronize(
                device
            )

    finally:

        for handle in handles:
            handle.remove()

    print(
        "layer5 input:",
        captured.get(
            "layer5_input"
        ),
    )

    print(
        "layer7 input:",
        captured.get(
            "layer7_input"
        ),
    )

    print(
        "Detect inputs:",
        captured.get(
            "detect_inputs"
        ),
    )

    expected_layer5 = (
        BATCH_SIZE,
        256,
        120,
        120,
    )

    expected_layer7 = (
        BATCH_SIZE,
        256,
        60,
        60,
    )

    expected_detect = [
        (
            BATCH_SIZE,
            128,
            120,
            120,
        ),
        (
            BATCH_SIZE,
            256,
            60,
            60,
        ),
        (
            BATCH_SIZE,
            512,
            30,
            30,
        ),
    ]

    if (
        captured.get(
            "layer5_input"
        )
        != expected_layer5
    ):

        raise AssertionError(
            "Fused P3 routing mismatch."
        )

    if (
        captured.get(
            "layer7_input"
        )
        != expected_layer7
    ):

        raise AssertionError(
            "Fused P4 routing mismatch."
        )

    if (
        captured.get(
            "detect_inputs"
        )
        != expected_detect
    ):

        raise AssertionError(
            "Detect pyramid routing mismatch."
        )

    pass_line(
        "Depth-fused P3 enters layer 5"
    )

    pass_line(
        "IR-fused P4 enters layer 7"
    )

    pass_line(
        "Detect receives unchanged P3/P4/P5 pyramid shapes"
    )

    # --------------------------------------------------------
    # Eval prediction shape
    # --------------------------------------------------------

    output_tensors = list(
        iter_tensors(
            output
        )
    )

    if not output_tensors:

        raise AssertionError(
            "Forward produced no tensors."
        )

    for index, tensor in enumerate(
        output_tensors
    ):

        assert_finite(
            tensor,
            f"forward tensor {index}",
        )

    if not isinstance(
        output,
        tuple,
    ):

        raise AssertionError(
            "Expected eval Detect output tuple."
        )

    prediction = output[0]

    expected_prediction_shape = (
        BATCH_SIZE,
        4 + NUM_CLASSES,
        18900,
    )

    if tuple(
        prediction.shape
    ) != expected_prediction_shape:

        raise AssertionError(
            "Prediction shape mismatch: "
            f"expected={expected_prediction_shape}, "
            f"got={tuple(prediction.shape)}"
        )

    print(
        "Prediction:",
        tensor_stats(
            prediction
        ),
    )

    pass_line(
        "[1,5,960,960] forward"
    )

    pass_line(
        "12-class prediction shape == [1,16,18900]"
    )

    print_cuda_peak(
        "Forward"
    )

    del output
    del output_tensors
    del prediction

    if (
        device.type
        == "cuda"
    ):
        torch.cuda.empty_cache()

    return image


# ============================================================
# Gradient utilities
# ============================================================

def check_gradient(
    parameter: nn.Parameter,
    name: str,
) -> dict[str, float]:

    grad = parameter.grad

    if grad is None:

        raise AssertionError(
            f"{name} gradient is None."
        )

    assert_finite(
        grad,
        f"{name} gradient",
    )

    detached = (
        grad
        .detach()
        .float()
    )

    nonzero = int(
        torch.count_nonzero(
            detached
        ).item()
    )

    abs_sum = float(
        detached
        .abs()
        .sum()
        .item()
    )

    abs_max = float(
        detached
        .abs()
        .max()
        .item()
    )

    print(
        f"{name}:"
    )

    print(
        "  shape   =",
        tuple(
            detached.shape
        ),
    )

    print(
        "  nonzero =",
        nonzero,
    )

    print(
        "  abs sum =",
        abs_sum,
    )

    print(
        "  abs max =",
        abs_max,
    )

    if nonzero == 0:

        raise AssertionError(
            f"{name} gradient is entirely zero."
        )

    if abs_sum <= 0.0:

        raise AssertionError(
            f"{name} gradient abs_sum <= 0."
        )

    pass_line(
        f"{name} gradient != 0"
    )

    return {
        "nonzero": nonzero,
        "abs_sum": abs_sum,
        "abs_max": abs_max,
    }


def first_parameter(
    module: nn.Module,
) -> nn.Parameter:

    for parameter in module.parameters():
        return parameter

    raise RuntimeError(
        f"No parameter in {type(module).__name__}."
    )


# ============================================================
# Test 3: real YOLO loss / backward / optimizer
# ============================================================

def test_loss_backward_optimizer(
    model: Exp05AsymmetricGatedDetectionModel,
    image: torch.Tensor,
    device: torch.device,
) -> None:

    section(
        "3. Real YOLO detection loss + backward + optimizer.step"
    )

    # Standalone model smoke tests bypass DetectionTrainer.
    # Prepare Ultralytics hyperparameter namespace manually.
    if isinstance(
        model.args,
        dict,
    ):

        model.args = get_cfg(
            overrides=model.args
        )

    print(
        "model.args type:",
        type(
            model.args
        ).__name__,
    )

    print(
        "Loss gains:",
        {
            "box": model.args.box,
            "cls": model.args.cls,
            "dfl": model.args.dfl,
        },
    )

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
    # Synthetic 12-class YOLO targets.
    # --------------------------------------------------------

    cls = torch.tensor(
        [
            [0.0],   # person
            [6.0],   # car
            [10.0],  # uav
        ],
        dtype=torch.float32,
        device=device,
    )

    bboxes = torch.tensor(
        [
            [
                0.30,
                0.42,
                0.15,
                0.28,
            ],
            [
                0.65,
                0.60,
                0.25,
                0.18,
            ],
            [
                0.78,
                0.20,
                0.08,
                0.07,
            ],
        ],
        dtype=torch.float32,
        device=device,
    )

    batch_idx = torch.tensor(
        [
            0.0,
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
    # Parameter groups that MUST receive gradient.
    # --------------------------------------------------------

    rgb_first_weight = (
        first_conv(
            model.model[0]
        )
        .weight
    )

    depth_first_weight = (
        first_conv(
            model.depth_encoder[0]
        )
        .weight
    )

    ir_first_weight = (
        first_conv(
            model.ir_encoder[0]
        )
        .weight
    )

    ir_projector_weight = (
        model.ir_to_p4
        .proj
        .weight
    )

    depth_gate_logits = (
        model.depth_gate
        .logits
    )

    ir_gate_logits = (
        model.ir_gate
        .logits
    )

    late_backbone_weight = (
        first_parameter(
            model.model[10]
        )
    )

    tracked = {
        "RGB first conv": (
            rgb_first_weight
        ),
        "Depth first conv": (
            depth_first_weight
        ),
        "IR first conv": (
            ir_first_weight
        ),
        "IR P4 projector": (
            ir_projector_weight
        ),
        "Depth gate logits": (
            depth_gate_logits
        ),
        "IR gate logits": (
            ir_gate_logits
        ),
        "Late fused backbone": (
            late_backbone_weight
        ),
    }

    before_step = {
        name: (
            parameter
            .detach()
            .clone()
        )
        for name, parameter
        in tracked.items()
    }

    # --------------------------------------------------------
    # Real Ultralytics detection loss.
    # --------------------------------------------------------

    if (
        device.type
        == "cuda"
    ):
        torch.cuda.reset_peak_memory_stats()

    loss_result = model(
        batch
    )

    if not isinstance(
        loss_result,
        (tuple, list),
    ):

        raise AssertionError(
            "Expected (loss, loss_items), "
            f"got {type(loss_result).__name__}."
        )

    if len(
        loss_result
    ) != 2:

        raise AssertionError(
            "Unexpected loss-result length: "
            f"{len(loss_result)}"
        )

    loss = loss_result[0]
    loss_items = loss_result[1]

    if not torch.is_tensor(
        loss
    ):

        raise TypeError(
            "YOLO loss is not Tensor."
        )

    assert_finite(
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

    loss.sum().backward()

    if (
        device.type
        == "cuda"
    ):
        torch.cuda.synchronize(
            device
        )

    pass_line(
        "real YOLO detection loss"
    )

    pass_line(
        "loss.backward()"
    )

    # --------------------------------------------------------
    # Gradient verification
    # --------------------------------------------------------

    print()

    for (
        name,
        parameter,
    ) in tracked.items():

        check_gradient(
            parameter=parameter,
            name=name,
        )

    print_cuda_peak(
        "Backward"
    )

    # --------------------------------------------------------
    # One disposable optimizer step.
    #
    # This model instance is only a smoke-test model.
    # --------------------------------------------------------

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.0,
        weight_decay=0.0,
    )

    optimizer.step()

    changed_groups = []

    for (
        name,
        parameter,
    ) in tracked.items():

        after = (
            parameter
            .detach()
        )

        before = (
            before_step[
                name
            ]
        )

        if not torch.equal(
            after,
            before,
        ):

            changed_groups.append(
                name
            )

            print(
                f"[UPDATED] {name}"
            )

        else:

            raise AssertionError(
                f"optimizer.step did not change {name}."
            )

    if len(
        changed_groups
    ) != len(
        tracked
    ):

        raise AssertionError(
            "Not all tracked parameter groups updated."
        )

    pass_line(
        "optimizer.step updates all major Exp05 paths"
    )

    print(
        "Gate statistics after one disposable step:",
        model.get_gate_statistics(),
    )


# ============================================================
# Test 4: Exp05 source/reload preservation
# ============================================================

def compare_full_state_dict(
    source: nn.Module,
    target: nn.Module,
) -> None:

    source_state = (
        source
        .state_dict()
    )

    target_state = (
        target
        .state_dict()
    )

    if (
        source_state.keys()
        != target_state.keys()
    ):

        source_only = sorted(
            set(
                source_state
            )
            - set(
                target_state
            )
        )

        target_only = sorted(
            set(
                target_state
            )
            - set(
                source_state
            )
        )

        raise AssertionError(
            "Reload state_dict key mismatch.\n"
            f"source only={source_only}\n"
            f"target only={target_only}"
        )

    checked = 0

    for key in source_state:

        assert_exact_tensor(
            actual=target_state[key],
            expected=source_state[key],
            name=f"reload:{key}",
        )

        checked += 1

    print(
        "Reload state_dict tensors checked:",
        checked,
    )


def test_exp05_reload(
    trained_model: Exp05AsymmetricGatedDetectionModel,
) -> None:

    section(
        "4. Exp05 checkpoint/state reload preservation"
    )

    trained_model = (
        trained_model
        .cpu()
        .eval()
    )

    if not source_has_exp05_parameters(
        trained_model
    ):

        raise AssertionError(
            "source_has_exp05_parameters() failed "
            "to identify an Exp05 model."
        )

    pass_line(
        "Exp05 source checkpoint signature detected"
    )

    source_gate_stats = (
        trained_model
        .get_gate_statistics()
    )

    # Build clean target from RGB pretrained.
    reload_target = (
        build_exp05_yolo11s_gated(
            pretrained_path=(
                PRETRAINED_MODEL_PATH
            ),
            nc=NUM_CLASSES,
            names=COMPETITION_NAMES,
            verbose=False,
        )
    )

    # Load the learned Exp05 source.
    detected_exp05 = (
        initialize_exp05_from_weights(
            model=reload_target,
            weights=trained_model,
            verbose=False,
        )
    )

    if not detected_exp05:

        raise AssertionError(
            "Exp05 source was incorrectly treated "
            "as RGB-only source."
        )

    pass_line(
        "Exp05 reload path does not reinitialize auxiliary branches"
    )

    compare_full_state_dict(
        source=trained_model,
        target=reload_target,
    )

    target_gate_stats = (
        reload_target
        .get_gate_statistics()
    )

    print(
        "Source gate stats:",
        source_gate_stats,
    )

    print(
        "Reload gate stats:",
        target_gate_stats,
    )

    for modality in (
        "depth",
        "ir",
    ):

        for stat in (
            "mean",
            "min",
            "max",
            "std",
        ):

            if not math.isclose(
                source_gate_stats[
                    modality
                ][
                    stat
                ],
                target_gate_stats[
                    modality
                ][
                    stat
                ],
                rel_tol=0.0,
                abs_tol=1e-8,
            ):

                raise AssertionError(
                    "Gate state changed during reload: "
                    f"{modality}.{stat}"
                )

    pass_line(
        "learned gate state preserved exactly"
    )

    pass_line(
        "full Exp05 state_dict preserved exactly"
    )

    del reload_target

    gc.collect()


# ============================================================
# Main
# ============================================================

def main() -> None:

    set_seed()

    section(
        "AIC2026 Exp05 - gated model smoke test"
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

    print(
        "Classes      :",
        NUM_CLASSES,
    )

    print(
        "Depth fusion :",
        f"layer {DEPTH_FUSION_LAYER} / P3",
    )

    print(
        "IR fusion    :",
        f"layer {IR_FUSION_LAYER} / P4",
    )

    device = get_device()

    # --------------------------------------------------------
    # Independent untouched RGB reference
    # --------------------------------------------------------

    section(
        "0. Load untouched pretrained YOLO11s reference"
    )

    (
        reference_yolo,
        reference_model,
    ) = load_reference()

    print(
        "Reference nc:",
        reference_model.model[-1].nc,
    )

    print(
        "Reference first conv:",
        first_conv(
            reference_model.model[0]
        ),
    )

    pass_line(
        "pretrained RGB reference loaded"
    )

    # --------------------------------------------------------
    # Actual 12-class Exp05 model
    # --------------------------------------------------------

    model = build_exp05_yolo11s_gated(
        pretrained_path=(
            PRETRAINED_MODEL_PATH
        ),
        nc=NUM_CLASSES,
        names=COMPETITION_NAMES,
        verbose=False,
    )

    test_initialization(
        model=model,
        reference_model=reference_model,
    )

    # Reference is no longer needed.
    del reference_yolo
    del reference_model

    gc.collect()

    if (
        torch.cuda.is_available()
    ):
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Actual 960 forward
    # --------------------------------------------------------

    image = test_forward_graph(
        model=model,
        device=device,
    )

    # --------------------------------------------------------
    # Real loss/backward/update
    # --------------------------------------------------------

    test_loss_backward_optimizer(
        model=model,
        image=image,
        device=device,
    )

    del image

    if (
        torch.cuda.is_available()
    ):
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Checkpoint/state reload semantics
    # --------------------------------------------------------

    test_exp05_reload(
        trained_model=model,
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    section(
        "FINAL RESULT"
    )

    print(
        "RGB pretrained path         : PASS"
    )

    print(
        "Depth encoder initialization: PASS"
    )

    print(
        "IR encoder initialization   : PASS"
    )

    print(
        "Depth@P3 routing             : PASS"
    )

    print(
        "IR@P4 routing                : PASS"
    )

    print(
        "12-class 960 forward         : PASS"
    )

    print(
        "YOLO detection loss          : PASS"
    )

    print(
        "Backward gradients           : PASS"
    )

    print(
        "Optimizer step               : PASS"
    )

    print(
        "Exp05 reload preservation    : PASS"
    )

    print()
    print(
        "STATUS = PASS"
    )


if __name__ == "__main__":
    main()

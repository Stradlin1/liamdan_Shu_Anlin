#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 Exp05
RGB-dominant Asymmetric Gated Feature Fusion for YOLO11s.

Architecture
============

Input:
    [R, G, B, IR, Depth]
    shape = [B, 5, H, W]

RGB main path:
    Original pretrained YOLO11s backbone / neck / detect head.

Depth branch:
    1-channel input
        ↓
    RGB-pretrained-derived encoder (YOLO layers 0..4)
        ↓
    P3/8 feature: [B, 256, H/8, W/8]
        ↓
    channel-wise learnable gate
        ↓
    residual add AFTER RGB backbone layer 4

IR branch:
    1-channel input
        ↓
    RGB-pretrained-derived encoder (YOLO layers 0..4)
        ↓
    P3/8 feature
        ↓
    AvgPool2d(stride=2)
        ↓
    identity-initialized 1x1 projection + BN
        ↓
    P4/16 feature: [B, 256, H/16, W/16]
        ↓
    channel-wise learnable gate
        ↓
    residual add AFTER RGB backbone layer 6

After P4:
    The fused RGB + Depth + IR representation continues through
    the original YOLO11s layer 7..10, neck, and Detect head.

Important design choices
========================

1. RGB is the protected primary modality.
2. Depth is the stronger auxiliary modality and enters at P3.
3. IR is weaker / more conditional and enters later at P4.
4. P5 is not given an additional explicit fusion module.
5. Gate v1 is deliberately simple:
       one learnable gate value per feature channel.
6. No SE / CBAM / cross-attention / quality network in Exp05-v1.
7. Auxiliary one-channel first-convolution weights are initialized as:

       W_gray = W_R + W_G + W_B

   This is mathematically equivalent to applying the original RGB
   convolution to a grayscale image replicated into three channels.
8. The original Ultralytics source is NOT modified.

Locked architecture assumption from the Exp05 audit
====================================================

YOLO11s @ 960:

    layer 4:
        P3/8
        [B, 256, 120, 120]

    layer 6:
        P4/16
        [B, 256, 60, 60]

    layer 10:
        P5/32
        [B, 512, 30, 30]

    Detect:
        from [16, 19, 22]

This module intentionally fails fast if the locked YOLO graph changes.
"""

from __future__ import annotations

import argparse
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER


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
# Exp05 channel definition
# ============================================================

RGB_CHANNELS = 3
MULTIMODAL_CHANNELS = 5

RGB_SLICE = slice(0, 3)
IR_CHANNEL_INDEX = 3
DEPTH_CHANNEL_INDEX = 4

CHANNEL_NAMES = (
    "R",
    "G",
    "B",
    "IR",
    "Depth",
)


# ============================================================
# Locked YOLO11s fusion topology
# ============================================================

DEPTH_FUSION_LAYER = 4
IR_FUSION_LAYER = 6

P3_CHANNELS = 256
P4_CHANNELS = 256

AUX_ENCODER_LAST_LAYER = 4

EXPECTED_TOP_LEVEL_LAYERS = 24

EXPECTED_DETECT_FROM = [
    16,
    19,
    22,
]

EXPECTED_SAVE_LAYERS = {
    4,
    6,
    10,
    13,
    16,
    19,
    22,
}


# ============================================================
# Gate initialization
# ============================================================

DEFAULT_DEPTH_GATE = 0.10
DEFAULT_IR_GATE = 0.02


# ============================================================
# Generic helpers
# ============================================================

def _probability_to_logit(
    probability: float,
) -> float:
    """
    Convert p in (0, 1) to logit(p).
    """

    probability = float(
        probability
    )

    if not (
        0.0
        < probability
        < 1.0
    ):
        raise ValueError(
            "Gate initialization must satisfy "
            f"0 < p < 1, got {probability}."
        )

    return math.log(
        probability
        / (
            1.0
            - probability
        )
    )


def _shape_repr(
    obj: Any,
    depth: int = 0,
) -> str:
    """
    Compact recursive representation of outputs.
    """

    if depth > 4:
        return "..."

    if isinstance(
        obj,
        torch.Tensor,
    ):
        return str(
            tuple(
                int(v)
                for v in obj.shape
            )
        )

    if isinstance(
        obj,
        list,
    ):
        return (
            "["
            + ", ".join(
                _shape_repr(
                    item,
                    depth + 1,
                )
                for item in obj[:8]
            )
            + (
                ", ..."
                if len(obj) > 8
                else ""
            )
            + "]"
        )

    if isinstance(
        obj,
        tuple,
    ):
        return (
            "("
            + ", ".join(
                _shape_repr(
                    item,
                    depth + 1,
                )
                for item in obj[:8]
            )
            + (
                ", ..."
                if len(obj) > 8
                else ""
            )
            + ")"
        )

    if isinstance(
        obj,
        dict,
    ):
        parts = []

        for index, (
            key,
            value,
        ) in enumerate(
            obj.items()
        ):
            if index >= 8:
                parts.append(
                    "..."
                )
                break

            parts.append(
                f"{key}:"
                f"{_shape_repr(value, depth + 1)}"
            )

        return (
            "{"
            + ", ".join(parts)
            + "}"
        )

    if obj is None:
        return "None"

    return type(
        obj
    ).__name__


def _extract_source_module(
    weights: Any,
) -> nn.Module | None:
    """
    Extract an nn.Module from common Ultralytics checkpoint forms.

    Supported:
        nn.Module
        {"model": nn.Module, ...}
        {"ema": nn.Module, ...}
    """

    if weights is None:
        return None

    if isinstance(
        weights,
        nn.Module,
    ):
        return weights

    if isinstance(
        weights,
        Mapping,
    ):
        model = weights.get(
            "model"
        )

        if isinstance(
            model,
            nn.Module,
        ):
            return model

        ema = weights.get(
            "ema"
        )

        if isinstance(
            ema,
            nn.Module,
        ):
            return ema

    return None


def source_has_exp05_parameters(
    weights: Any,
) -> bool:
    """
    Determine whether source weights are already an Exp05 checkpoint.

    This is important for resume:
        Exp05 checkpoint:
            load learned auxiliary encoders + gates as-is.

        Original RGB pretrained checkpoint:
            load RGB path first,
            then initialize auxiliary encoders from RGB weights.
    """

    source = _extract_source_module(
        weights
    )

    if source is None:
        return False

    keys = set(
        source
        .state_dict()
        .keys()
    )

    required_prefixes = (
        "depth_encoder.",
        "ir_encoder.",
        "ir_to_p4.",
        "depth_gate.",
        "ir_gate.",
    )

    return all(
        any(
            key.startswith(
                prefix
            )
            for key in keys
        )
        for prefix in required_prefixes
    )


def _get_first_conv2d(
    block: nn.Module,
) -> nn.Conv2d:
    """
    Locate the first Conv2d inside a YOLO block.
    """

    if hasattr(
        block,
        "conv",
    ):
        conv = getattr(
            block,
            "conv",
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
        "Could not locate Conv2d "
        f"inside {type(block).__name__}."
    )


def _replace_block_input_conv_with_1ch(
    first_block: nn.Module,
) -> None:
    """
    Convert copied YOLO layer-0 convolution:

        Conv2d(3, C, ...)

    into:

        Conv2d(1, C, ...)

    with:

        W_1ch = W_R + W_G + W_B

    BN and activation from the copied pretrained block are preserved.
    """

    if not hasattr(
        first_block,
        "conv",
    ):
        raise RuntimeError(
            "Expected copied YOLO layer 0 "
            "to expose '.conv'."
        )

    old_conv = first_block.conv

    if not isinstance(
        old_conv,
        nn.Conv2d,
    ):
        raise TypeError(
            "Expected layer0.conv to be nn.Conv2d, "
            f"got {type(old_conv).__name__}."
        )

    if (
        old_conv.in_channels
        != RGB_CHANNELS
    ):
        raise RuntimeError(
            "Expected RGB layer-0 Conv2d "
            f"in_channels=3, got {old_conv.in_channels}."
        )

    if old_conv.groups != 1:
        raise RuntimeError(
            "Unexpected grouped first convolution: "
            f"groups={old_conv.groups}."
        )

    old_weight = (
        old_conv
        .weight
        .detach()
        .clone()
    )

    old_bias = None

    if old_conv.bias is not None:
        old_bias = (
            old_conv
            .bias
            .detach()
            .clone()
        )

    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=1,
        bias=(
            old_conv.bias
            is not None
        ),
        padding_mode=old_conv.padding_mode,
        device=old_conv.weight.device,
        dtype=old_conv.weight.dtype,
    )

    with torch.no_grad():

        gray_weight = (
            old_weight
            .sum(
                dim=1,
                keepdim=True,
            )
        )

        new_conv.weight.copy_(
            gray_weight
        )

        if old_bias is not None:
            new_conv.bias.copy_(
                old_bias
            )

    first_block.conv = new_conv


# ============================================================
# Channel-wise residual gate
# ============================================================

class ChannelWiseResidualGate(
    nn.Module
):
    """
    One learnable sigmoid gate per feature channel.

    Parameter:
        logits shape = [1, C, 1, 1]

    Forward:
        gated_residual =
            sigmoid(logits) * residual

    This is deliberately simple for Exp05-v1.
    """

    def __init__(
        self,
        channels: int,
        initial_probability: float,
    ) -> None:

        super().__init__()

        channels = int(
            channels
        )

        if channels <= 0:
            raise ValueError(
                f"Invalid channel count: {channels}"
            )

        self.channels = channels

        self.initial_probability = float(
            initial_probability
        )

        initial_logit = (
            _probability_to_logit(
                self.initial_probability
            )
        )

        self.logits = nn.Parameter(
            torch.full(
                (
                    1,
                    channels,
                    1,
                    1,
                ),
                fill_value=initial_logit,
                dtype=torch.float32,
            )
        )

    def reset(
        self,
        probability: float | None = None,
    ) -> None:
        """
        Reset gate logits to a fixed probability.
        """

        if probability is None:
            probability = (
                self.initial_probability
            )

        logit = (
            _probability_to_logit(
                float(
                    probability
                )
            )
        )

        with torch.no_grad():
            self.logits.fill_(
                logit
            )

    def probabilities(
        self,
    ) -> torch.Tensor:
        """
        Return sigmoid gate probabilities.
        """

        return torch.sigmoid(
            self.logits
        )

    def forward(
        self,
        residual: torch.Tensor,
    ) -> torch.Tensor:

        if residual.ndim != 4:
            raise RuntimeError(
                "Gate expects BCHW tensor, "
                f"got shape={tuple(residual.shape)}."
            )

        if (
            residual.shape[1]
            != self.channels
        ):
            raise RuntimeError(
                "Gate channel mismatch: "
                f"expected={self.channels}, "
                f"got={residual.shape[1]}."
            )

        return (
            residual
            * self.probabilities()
        )


# ============================================================
# IR P3 -> P4 lightweight projector
# ============================================================

class IRP4Projector(
    nn.Module
):
    """
    Lightweight IR transition:

        [B,256,H/8,W/8]
            ↓ AvgPool2d(2)
        [B,256,H/16,W/16]
            ↓ identity-initialized 1x1 Conv
            ↓ BatchNorm
        [B,256,H/16,W/16]

    No extra activation is placed after the projection in Exp05-v1.

    Reason:
        the incoming P3 feature is already nonlinear,
        and a linear residual projection is easier to interpret.
    """

    def __init__(
        self,
        channels: int = P4_CHANNELS,
    ) -> None:

        super().__init__()

        channels = int(
            channels
        )

        self.channels = channels

        self.pool = nn.AvgPool2d(
            kernel_size=2,
            stride=2,
        )

        self.proj = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(
            channels
        )

        self.reset_identity()

    def reset_identity(
        self,
    ) -> None:
        """
        Initialize the 1x1 projection approximately as identity.
        """

        with torch.no_grad():

            self.proj.weight.zero_()

            diagonal = torch.arange(
                self.channels,
                device=(
                    self.proj
                    .weight
                    .device
                ),
            )

            self.proj.weight[
                diagonal,
                diagonal,
                0,
                0,
            ] = 1.0

            self.bn.weight.fill_(
                1.0
            )

            self.bn.bias.zero_()

            self.bn.running_mean.zero_()

            self.bn.running_var.fill_(
                1.0
            )

            self.bn.num_batches_tracked.zero_()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 4:
            raise RuntimeError(
                "IR projector expects BCHW input."
            )

        if x.shape[1] != self.channels:
            raise RuntimeError(
                "IR projector channel mismatch: "
                f"expected={self.channels}, "
                f"got={x.shape[1]}."
            )

        x = self.pool(
            x
        )

        x = self.proj(
            x
        )

        x = self.bn(
            x
        )

        return x


# ============================================================
# Exp05 DetectionModel
# ============================================================

class Exp05AsymmetricGatedDetectionModel(
    DetectionModel
):
    """
    YOLO11 DetectionModel with asymmetric multimodal feature fusion.

    The original self.model remains the complete RGB YOLO11 graph.

    Additional trainable modules:

        depth_encoder
        ir_encoder
        ir_to_p4
        depth_gate
        ir_gate

    The standard DetectionModel loss / Detect implementation remains intact.
    """

    def __init__(
        self,
        cfg: str | dict,
        nc: int | None = None,
        verbose: bool = True,
        depth_gate_init: float = DEFAULT_DEPTH_GATE,
        ir_gate_init: float = DEFAULT_IR_GATE,
    ) -> None:

        # ----------------------------------------------------
        # DetectionModel.__init__ internally performs a dummy
        # forward to build detection strides.
        #
        # At that moment auxiliary branches do not exist yet,
        # so _predict_once() must temporarily behave exactly
        # like the original RGB DetectionModel.
        # ----------------------------------------------------

        object.__setattr__(
            self,
            "_exp05_ready",
            False,
        )

        # ----------------------------------------------------
        # The internal RGB backbone is intentionally built as
        # a normal 3-channel YOLO11 model.
        #
        # The OVERALL Exp05 model accepts five channels, but
        # only the RGB slice is sent through self.model[0].
        # ----------------------------------------------------

        super().__init__(
            cfg=cfg,
            ch=RGB_CHANNELS,
            nc=nc,
            verbose=verbose,
        )

        self.depth_gate_init = float(
            depth_gate_init
        )

        self.ir_gate_init = float(
            ir_gate_init
        )

        self._assert_locked_yolo11s_topology()

        # ----------------------------------------------------
        # Build auxiliary encoders from the current RGB path.
        #
        # When the model is first created these are copies of
        # the current RGB initialization.
        #
        # If loading original pretrained weights, call:
        #
        #   initialize_auxiliary_from_rgb()
        #
        # AFTER pretrained RGB weights have been transferred.
        #
        # If loading an Exp05 checkpoint, normal state_dict
        # loading restores these modules directly.
        # ----------------------------------------------------

        self.depth_encoder = (
            self._build_single_channel_p3_encoder()
        )

        self.ir_encoder = (
            self._build_single_channel_p3_encoder()
        )

        self.ir_to_p4 = (
            IRP4Projector(
                channels=P4_CHANNELS
            )
        )

        self.depth_gate = (
            ChannelWiseResidualGate(
                channels=P3_CHANNELS,
                initial_probability=(
                    self.depth_gate_init
                ),
            )
        )

        self.ir_gate = (
            ChannelWiseResidualGate(
                channels=P4_CHANNELS,
                initial_probability=(
                    self.ir_gate_init
                ),
            )
        )

        # Overall model metadata.
        if isinstance(
            self.yaml,
            dict,
        ):
            self.yaml[
                "channels"
            ] = MULTIMODAL_CHANNELS

            self.yaml[
                "ch"
            ] = MULTIMODAL_CHANNELS

        self.multimodal_channels = (
            MULTIMODAL_CHANNELS
        )

        self.channel_names = (
            CHANNEL_NAMES
        )

        self._exp05_ready = True

    # --------------------------------------------------------
    # Architecture validation
    # --------------------------------------------------------

    def _assert_locked_yolo11s_topology(
        self,
    ) -> None:
        """
        Fail fast if the locked YOLO11s graph differs from the audited graph.
        """

        if len(
            self.model
        ) != EXPECTED_TOP_LEVEL_LAYERS:

            raise RuntimeError(
                "Unexpected YOLO top-level layer count: "
                f"expected={EXPECTED_TOP_LEVEL_LAYERS}, "
                f"got={len(self.model)}."
            )

        # Auxiliary encoders copy layers 0..4 as a sequential path.
        # These must all consume only the previous layer.
        for index in range(
            AUX_ENCODER_LAST_LAYER + 1
        ):

            layer_from = getattr(
                self.model[index],
                "f",
                None,
            )

            if layer_from != -1:
                raise RuntimeError(
                    "Auxiliary encoder assumption violated: "
                    f"model[{index}].f={layer_from}, "
                    "expected -1."
                )

        detect_from = list(
            getattr(
                self.model[-1],
                "f",
                [],
            )
        )

        if (
            detect_from
            != EXPECTED_DETECT_FROM
        ):
            raise RuntimeError(
                "Unexpected Detect routing: "
                f"expected={EXPECTED_DETECT_FROM}, "
                f"got={detect_from}."
            )

        actual_save = set(
            int(v)
            for v in self.save
        )

        if not (
            EXPECTED_SAVE_LAYERS
            <= actual_save
        ):
            raise RuntimeError(
                "Required YOLO save layers are missing: "
                f"required={sorted(EXPECTED_SAVE_LAYERS)}, "
                f"actual={sorted(actual_save)}."
            )

        # Explicitly verify audited backbone skip routes.
        expected_routes = {
            12: [-1, 6],
            15: [-1, 4],
            21: [-1, 10],
        }

        for (
            layer_index,
            expected_from,
        ) in expected_routes.items():

            actual_from = list(
                getattr(
                    self.model[
                        layer_index
                    ],
                    "f",
                    [],
                )
            )

            if (
                actual_from
                != expected_from
            ):
                raise RuntimeError(
                    "Unexpected backbone/head route: "
                    f"model[{layer_index}].f="
                    f"{actual_from}, "
                    f"expected={expected_from}."
                )

    # --------------------------------------------------------
    # Auxiliary encoder construction
    # --------------------------------------------------------

    def _build_single_channel_p3_encoder(
        self,
    ) -> nn.Sequential:
        """
        Clone RGB YOLO layers 0..4 and adapt layer-0 to one channel.
        """

        blocks = [
            deepcopy(
                self.model[index]
            )
            for index in range(
                AUX_ENCODER_LAST_LAYER + 1
            )
        ]

        encoder = nn.Sequential(
            *blocks
        )

        _replace_block_input_conv_with_1ch(
            encoder[0]
        )

        return encoder

    def initialize_auxiliary_from_rgb(
        self,
        reset_gates: bool = True,
        reset_ir_projection: bool = True,
    ) -> None:
        """
        Initialize Depth and IR encoders from CURRENT RGB backbone weights.

        This must be called AFTER original RGB pretrained weights are loaded
        when starting Exp05 from yolo11s.pt.

        Do NOT call this after loading a learned Exp05 checkpoint.
        """

        reference_depth = (
            self._build_single_channel_p3_encoder()
        )

        reference_ir = (
            self._build_single_channel_p3_encoder()
        )

        self.depth_encoder.load_state_dict(
            reference_depth.state_dict(),
            strict=True,
        )

        self.ir_encoder.load_state_dict(
            reference_ir.state_dict(),
            strict=True,
        )

        if reset_ir_projection:
            self.ir_to_p4.reset_identity()

        if reset_gates:

            self.depth_gate.reset(
                self.depth_gate_init
            )

            self.ir_gate.reset(
                self.ir_gate_init
            )

    # --------------------------------------------------------
    # Gate helpers
    # --------------------------------------------------------

    def set_gate_probabilities(
        self,
        depth_probability: float,
        ir_probability: float,
    ) -> None:

        self.depth_gate.reset(
            depth_probability
        )

        self.ir_gate.reset(
            ir_probability
        )

    @staticmethod
    def _gate_stats(
        gate: ChannelWiseResidualGate,
    ) -> dict[str, float]:

        values = (
            gate
            .probabilities()
            .detach()
            .float()
            .cpu()
        )

        return {
            "mean": float(
                values.mean()
            ),
            "min": float(
                values.min()
            ),
            "max": float(
                values.max()
            ),
            "std": float(
                values.std(
                    unbiased=False
                )
            ),
        }

    def get_gate_statistics(
        self,
    ) -> dict[str, dict[str, float]]:

        return {
            "depth": self._gate_stats(
                self.depth_gate
            ),
            "ir": self._gate_stats(
                self.ir_gate
            ),
        }

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    @staticmethod
    def _validate_multimodal_input(
        x: torch.Tensor,
    ) -> None:

        if not isinstance(
            x,
            torch.Tensor,
        ):
            raise TypeError(
                "Exp05 expects torch.Tensor input, "
                f"got {type(x).__name__}."
            )

        if x.ndim != 4:
            raise RuntimeError(
                "Exp05 expects BCHW input, "
                f"got shape={tuple(x.shape)}."
            )

        if (
            x.shape[1]
            != MULTIMODAL_CHANNELS
        ):
            raise RuntimeError(
                "Exp05 expects 5-channel input "
                "[R,G,B,IR,Depth], "
                f"got C={x.shape[1]}."
            )

    @staticmethod
    def _assert_fusion_shapes(
        primary: torch.Tensor,
        residual: torch.Tensor,
        label: str,
    ) -> None:

        if (
            primary.shape
            != residual.shape
        ):
            raise RuntimeError(
                f"{label} fusion shape mismatch:\n"
                f"  primary : {tuple(primary.shape)}\n"
                f"  residual: {tuple(residual.shape)}"
            )

    # --------------------------------------------------------
    # Exp05 forward graph
    # --------------------------------------------------------

    def _predict_once(
        self,
        x: torch.Tensor,
        profile: bool = False,
        embed=None,
    ):
        """
        Exp05 forward pass.

        During DetectionModel.__init__:
            _exp05_ready == False
            -> execute original RGB YOLO forward.

        During normal Exp05 operation:
            input C must be 5
            -> split modalities
            -> auxiliary feature extraction
            -> gated fusion at RGB layer 4 and layer 6
            -> original YOLO remainder.
        """

        if not getattr(
            self,
            "_exp05_ready",
            False,
        ):
            return super()._predict_once(
                x,
                profile=profile,
                embed=embed,
            )

        self._validate_multimodal_input(
            x
        )

        rgb = x[
            :,
            RGB_SLICE,
            :,
            :,
        ]

        ir = x[
            :,
            IR_CHANNEL_INDEX:
            IR_CHANNEL_INDEX + 1,
            :,
            :,
        ]

        depth = x[
            :,
            DEPTH_CHANNEL_INDEX:
            DEPTH_CHANNEL_INDEX + 1,
            :,
            :,
        ]

        # ----------------------------------------------------
        # Auxiliary branches
        # ----------------------------------------------------

        depth_p3 = (
            self.depth_encoder(
                depth
            )
        )

        ir_p3 = (
            self.ir_encoder(
                ir
            )
        )

        ir_p4 = (
            self.ir_to_p4(
                ir_p3
            )
        )

        # ----------------------------------------------------
        # Original Ultralytics graph execution
        # ----------------------------------------------------

        y = []
        dt = []
        embeddings = []

        embed = (
            frozenset(
                embed
            )
            if embed
            else {
                -1
            }
        )

        max_idx = max(
            embed
        )

        current = rgb

        for module in self.model:

            module_from = getattr(
                module,
                "f",
                -1,
            )

            if module_from != -1:

                if isinstance(
                    module_from,
                    int,
                ):

                    current = y[
                        module_from
                    ]

                else:

                    current = [
                        (
                            current
                            if source == -1
                            else y[source]
                        )
                        for source
                        in module_from
                    ]

            if profile:

                self._profile_one_layer(
                    module,
                    current,
                    dt,
                )

            current = module(
                current
            )

            module_index = int(
                getattr(
                    module,
                    "i",
                    -1,
                )
            )

            # ------------------------------------------------
            # Depth residual fusion after RGB backbone layer 4
            # ------------------------------------------------

            if (
                module_index
                == DEPTH_FUSION_LAYER
            ):

                if not isinstance(
                    current,
                    torch.Tensor,
                ):
                    raise RuntimeError(
                        "RGB P3 output is not a Tensor."
                    )

                self._assert_fusion_shapes(
                    primary=current,
                    residual=depth_p3,
                    label="Depth@P3",
                )

                current = (
                    current
                    + self.depth_gate(
                        depth_p3
                    )
                )

            # ------------------------------------------------
            # IR residual fusion after RGBD backbone layer 6
            # ------------------------------------------------

            if (
                module_index
                == IR_FUSION_LAYER
            ):

                if not isinstance(
                    current,
                    torch.Tensor,
                ):
                    raise RuntimeError(
                        "RGBD P4 output is not a Tensor."
                    )

                self._assert_fusion_shapes(
                    primary=current,
                    residual=ir_p4,
                    label="IR@P4",
                )

                current = (
                    current
                    + self.ir_gate(
                        ir_p4
                    )
                )

            y.append(
                current
                if module_index
                in self.save
                else None
            )

            if (
                module_index
                in embed
            ):

                if not isinstance(
                    current,
                    torch.Tensor,
                ):
                    raise RuntimeError(
                        "Embedding requested from a "
                        "non-Tensor layer output."
                    )

                embeddings.append(
                    F.adaptive_avg_pool2d(
                        current,
                        (
                            1,
                            1,
                        ),
                    )
                    .squeeze(
                        -1
                    )
                    .squeeze(
                        -1
                    )
                )

                if (
                    module_index
                    == max_idx
                ):
                    return torch.unbind(
                        torch.cat(
                            embeddings,
                            dim=1,
                        ),
                        dim=0,
                    )

        return current


# ============================================================
# Weight-loading logic
# ============================================================

def initialize_exp05_from_weights(
    model: Exp05AsymmetricGatedDetectionModel,
    weights: Any,
    verbose: bool = True,
) -> bool:
    """
    Load weights into Exp05.

    Returns:
        True:
            source was already an Exp05 checkpoint.

        False:
            source was a normal RGB model and auxiliary branches were
            initialized from the newly loaded RGB backbone.

    Resume safety:
        If source already contains Exp05 parameters, NEVER reset the
        learned auxiliary encoders or gates.
    """

    if not isinstance(
        model,
        Exp05AsymmetricGatedDetectionModel,
    ):
        raise TypeError(
            "Expected Exp05AsymmetricGatedDetectionModel."
        )

    is_exp05_source = (
        source_has_exp05_parameters(
            weights
        )
    )

    model.load(
        weights,
        verbose=verbose,
    )

    if is_exp05_source:

        LOGGER.info(
            "Exp05 source checkpoint detected: "
            "preserving learned auxiliary encoders and gates."
        )

    else:

        model.initialize_auxiliary_from_rgb(
            reset_gates=True,
            reset_ir_projection=True,
        )

        LOGGER.info(
            "RGB source checkpoint detected: "
            "initialized Depth/IR encoders from RGB backbone; "
            f"depth_gate={model.depth_gate_init:.4f}, "
            f"ir_gate={model.ir_gate_init:.4f}."
        )

    return is_exp05_source


# ============================================================
# Initialization verification
# ============================================================

def verify_auxiliary_initialization(
    model: Exp05AsymmetricGatedDetectionModel,
) -> None:
    """
    Fail-fast check for Exp05 initialization.

    Checks:
        1. auxiliary first Conv2d is 1-channel
        2. first Conv weights equal RGB channel-sum initialization
        3. auxiliary YOLO layers 1..4 equal RGB layers 1..4 exactly
        4. gate probabilities match configured initialization
    """

    rgb_first = _get_first_conv2d(
        model.model[0]
    )

    if (
        rgb_first.in_channels
        != RGB_CHANNELS
    ):
        raise AssertionError(
            "RGB backbone first Conv2d "
            f"in_channels={rgb_first.in_channels}, "
            "expected 3."
        )

    expected_gray = (
        rgb_first
        .weight
        .detach()
        .sum(
            dim=1,
            keepdim=True,
        )
    )

    for (
        name,
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

        aux_first = (
            _get_first_conv2d(
                encoder[0]
            )
        )

        if aux_first.in_channels != 1:
            raise AssertionError(
                f"{name} first Conv2d expected "
                f"in_channels=1, got {aux_first.in_channels}."
            )

        if not torch.equal(
            aux_first
            .weight
            .detach(),
            expected_gray,
        ):

            max_diff = (
                aux_first
                .weight
                .detach()
                - expected_gray
            ).abs().max().item()

            raise AssertionError(
                f"{name} first-conv initialization mismatch: "
                f"max_abs_diff={max_diff}"
            )

        # Layers 1..4 should be exact copies of RGB backbone.
        for index in range(
            1,
            AUX_ENCODER_LAST_LAYER + 1,
        ):

            rgb_state = (
                model
                .model[index]
                .state_dict()
            )

            aux_state = (
                encoder[index]
                .state_dict()
            )

            if (
                rgb_state.keys()
                != aux_state.keys()
            ):
                raise AssertionError(
                    f"{name} encoder layer {index} "
                    "state_dict key mismatch."
                )

            for key in rgb_state:

                if not torch.equal(
                    rgb_state[key],
                    aux_state[key],
                ):

                    max_diff = (
                        rgb_state[key]
                        .detach()
                        .float()
                        - aux_state[key]
                        .detach()
                        .float()
                    ).abs().max().item()

                    raise AssertionError(
                        f"{name} encoder layer {index} "
                        f"parameter mismatch at '{key}', "
                        f"max_abs_diff={max_diff}"
                    )

    stats = (
        model.get_gate_statistics()
    )

    depth_mean = (
        stats[
            "depth"
        ][
            "mean"
        ]
    )

    ir_mean = (
        stats[
            "ir"
        ][
            "mean"
        ]
    )

    if not math.isclose(
        depth_mean,
        model.depth_gate_init,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise AssertionError(
            "Depth gate initialization mismatch: "
            f"expected={model.depth_gate_init}, "
            f"got={depth_mean}"
        )

    if not math.isclose(
        ir_mean,
        model.ir_gate_init,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise AssertionError(
            "IR gate initialization mismatch: "
            f"expected={model.ir_gate_init}, "
            f"got={ir_mean}"
        )


# ============================================================
# Public builder
# ============================================================

def build_exp05_yolo11s_gated(
    pretrained_path: Path = PRETRAINED_MODEL_PATH,
    nc: int | None = None,
    names: dict[int, str] | None = None,
    depth_gate_init: float = DEFAULT_DEPTH_GATE,
    ir_gate_init: float = DEFAULT_IR_GATE,
    verbose: bool = True,
) -> Exp05AsymmetricGatedDetectionModel:
    """
    Build Exp05 from the project's pretrained YOLO11s checkpoint.

    Standalone use:
        model = build_exp05_yolo11s_gated()

    Competition trainer use:
        nc should be set to dataset nc (=12).

    Workflow:
        1. Load original pretrained/yolo11s.pt.
        2. Rebuild identical RGB YOLO11s graph.
        3. Transfer pretrained RGB weights.
        4. Build one-channel Depth / IR encoders.
        5. Initialize auxiliary encoders from loaded RGB backbone.
        6. Initialize asymmetric channel gates.
    """

    pretrained_path = Path(
        pretrained_path
    ).resolve()

    if not pretrained_path.is_file():
        raise FileNotFoundError(
            "Pretrained YOLO11s checkpoint not found:\n"
            f"  {pretrained_path}"
        )

    source_yolo = YOLO(
        str(
            pretrained_path
        )
    )

    source_model = (
        source_yolo.model
    )

    if source_model is None:
        raise RuntimeError(
            "Pretrained YOLO wrapper contains no model."
        )

    source_cfg = deepcopy(
        source_model.yaml
    )

    source_head = (
        source_model.model[-1]
    )

    source_nc = getattr(
        source_head,
        "nc",
        None,
    )

    if source_nc is None:
        raise RuntimeError(
            "Could not determine source model nc."
        )

    target_nc = (
        int(source_nc)
        if nc is None
        else int(nc)
    )

    model = (
        Exp05AsymmetricGatedDetectionModel(
            cfg=source_cfg,
            nc=target_nc,
            verbose=verbose,
            depth_gate_init=(
                depth_gate_init
            ),
            ir_gate_init=(
                ir_gate_init
            ),
        )
    )

    # Set names before load when possible so Ultralytics class-remapping
    # logic has access to meaningful destination class names.
    if names is not None:

        if len(names) != target_nc:
            raise ValueError(
                "names length does not match target nc: "
                f"len(names)={len(names)}, nc={target_nc}"
            )

        model.names = deepcopy(
            names
        )

    elif (
        target_nc
        == int(source_nc)
        and isinstance(
            getattr(
                source_model,
                "names",
                None,
            ),
            dict,
        )
    ):

        model.names = deepcopy(
            source_model.names
        )

    initialize_exp05_from_weights(
        model=model,
        weights=source_model,
        verbose=verbose,
    )

    # Preserve useful source runtime args for standalone use.
    if hasattr(
        source_model,
        "args",
    ):
        model.args = deepcopy(
            source_model.args
        )

    # Verify main RGB layer-0 was transferred exactly.
    source_first = (
        _get_first_conv2d(
            source_model.model[0]
        )
    )

    target_first = (
        _get_first_conv2d(
            model.model[0]
        )
    )

    if (
        source_first.weight.shape
        == target_first.weight.shape
    ):

        if not torch.equal(
            source_first
            .weight
            .detach()
            .cpu(),
            target_first
            .weight
            .detach()
            .cpu(),
        ):

            max_diff = (
                source_first
                .weight
                .detach()
                .cpu()
                - target_first
                .weight
                .detach()
                .cpu()
            ).abs().max().item()

            raise AssertionError(
                "RGB pretrained first-conv transfer failed: "
                f"max_abs_diff={max_diff}"
            )

    verify_auxiliary_initialization(
        model
    )

    return model


# ============================================================
# Parameter statistics
# ============================================================

def parameter_count(
    module: nn.Module,
) -> int:

    return int(
        sum(
            p.numel()
            for p in module.parameters()
        )
    )


def trainable_parameter_count(
    module: nn.Module,
) -> int:

    return int(
        sum(
            p.numel()
            for p in module.parameters()
            if p.requires_grad
        )
    )


# ============================================================
# Standalone sanity check
# ============================================================

def _resolve_device(
    requested: str,
) -> torch.device:

    requested = (
        requested
        .strip()
        .lower()
    )

    if requested == "auto":

        if torch.cuda.is_available():
            return torch.device(
                "cuda:0"
            )

        return torch.device(
            "cpu"
        )

    device = torch.device(
        requested
    )

    if (
        device.type
        == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    return device


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Construct and sanity-check "
            "AIC2026 Exp05 asymmetric gated YOLO11s."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=PRETRAINED_MODEL_PATH,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help=(
            "Standalone forward-test size. "
            "Formal experiment remains 960."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    if args.imgsz <= 0:
        raise ValueError(
            f"Invalid imgsz={args.imgsz}"
        )

    if args.imgsz % 32 != 0:
        raise ValueError(
            "imgsz must be divisible by 32."
        )

    print("=" * 100)
    print(
        "AIC2026 Exp05 - Asymmetric Gated Feature Fusion"
    )
    print("=" * 100)

    print(
        "Project root    :",
        PROJECT_ROOT,
    )

    print(
        "Pretrained      :",
        Path(
            args.model
        ).resolve(),
    )

    print(
        "Channel order   :",
        CHANNEL_NAMES,
    )

    print(
        "Depth fusion    :",
        f"layer {DEPTH_FUSION_LAYER} / P3/8",
    )

    print(
        "IR fusion       :",
        f"layer {IR_FUSION_LAYER} / P4/16",
    )

    print(
        "Depth gate init :",
        DEFAULT_DEPTH_GATE,
    )

    print(
        "IR gate init    :",
        DEFAULT_IR_GATE,
    )

    print("-" * 100)

    model = build_exp05_yolo11s_gated(
        pretrained_path=args.model,
        verbose=False,
    )

    print(
        "Model class     :",
        type(model).__name__,
    )

    print(
        "Main YOLO params:",
        f"{parameter_count(model.model):,}",
    )

    print(
        "Depth encoder   :",
        f"{parameter_count(model.depth_encoder):,}",
    )

    print(
        "IR encoder      :",
        f"{parameter_count(model.ir_encoder):,}",
    )

    print(
        "IR P4 projector :",
        f"{parameter_count(model.ir_to_p4):,}",
    )

    print(
        "Depth gate      :",
        f"{parameter_count(model.depth_gate):,}",
    )

    print(
        "IR gate         :",
        f"{parameter_count(model.ir_gate):,}",
    )

    print(
        "Total params    :",
        f"{parameter_count(model):,}",
    )

    print(
        "Trainable params:",
        f"{trainable_parameter_count(model):,}",
    )

    gate_stats = (
        model.get_gate_statistics()
    )

    print(
        "Depth gate stats:",
        gate_stats[
            "depth"
        ],
    )

    print(
        "IR gate stats   :",
        gate_stats[
            "ir"
        ],
    )

    print("-" * 100)

    device = _resolve_device(
        args.device
    )

    print(
        "Forward device  :",
        device,
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
            MULTIMODAL_CHANNELS,
            args.imgsz,
            args.imgsz,
        ),
        dtype=torch.float32,
        device=device,
    )

    with torch.inference_mode():
        output = model(
            dummy
        )

    if device.type == "cuda":
        torch.cuda.synchronize(
            device
        )

    print(
        "Input shape     :",
        tuple(
            dummy.shape
        ),
    )

    print(
        "Output          :",
        _shape_repr(
            output
        ),
    )

    print("-" * 100)

    print(
        "Aux initialization : PASS"
    )

    print(
        "5-channel forward  : PASS"
    )

    print(
        "Topology check      : PASS"
    )

    print("=" * 100)
    print(
        "EXP05 GATED MODEL CONSTRUCTION: PASS"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()

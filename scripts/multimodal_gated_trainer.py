#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 Exp05
Custom Ultralytics DetectionTrainer for asymmetric gated multimodal fusion.

Input:
    [R, G, B, IR, Depth] = 5 channels

Model:
    RGB:
        original 3-channel YOLO11s main path

    Depth:
        independent 1-channel encoder
        fusion after backbone layer 4 / P3/8

    IR:
        independent 1-channel encoder
        fusion after backbone layer 6 / P4/16

Dataset:
    MultimodalYOLODataset

This trainer intentionally leaves the locked Ultralytics source untouched.

Ultralytics continues to manage:
    - optimizer
    - scheduler
    - AMP
    - EMA
    - detection loss
    - validation
    - early stopping
    - checkpoint save
    - resume
    - DDP

Exp05 customizations:
    1. use 5-channel multimodal Dataset
    2. build Exp05AsymmetricGatedDetectionModel
    3. preserve 3-channel RGB main path internally
    4. initialize auxiliary branches from RGB pretrained weights
    5. preserve learned Exp05 auxiliary/gate state when resuming
"""

from __future__ import annotations

from copy import copy
from typing import Any

import torch
import torch.nn as nn

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER, RANK, colorstr
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
    initialize_exp05_from_weights,
    source_has_exp05_parameters,
)


# ============================================================
# Constants
# ============================================================

RGB_MAIN_CHANNELS = 3

EXPECTED_STRIDES = (
    8.0,
    16.0,
    32.0,
)


# ============================================================
# Helpers
# ============================================================

def _get_rgb_first_conv(
    model: nn.Module,
) -> nn.Conv2d:
    """
    Return Exp05 RGB main-path first Conv2d.

    Expected:
        model.model[0].conv
    """

    if not hasattr(
        model,
        "model",
    ):
        raise RuntimeError(
            "Detection model has no '.model' attribute."
        )

    if len(
        model.model
    ) == 0:
        raise RuntimeError(
            "Detection model layer list is empty."
        )

    first_block = (
        model.model[0]
    )

    if not hasattr(
        first_block,
        "conv",
    ):
        raise RuntimeError(
            "YOLO layer 0 has no '.conv' attribute."
        )

    conv = (
        first_block.conv
    )

    if not isinstance(
        conv,
        nn.Conv2d,
    ):
        raise TypeError(
            "Expected model.model[0].conv to be nn.Conv2d, "
            f"got {type(conv).__name__}."
        )

    return conv


def _verify_exp05_model_structure(
    model: Exp05AsymmetricGatedDetectionModel,
    expected_nc: int,
) -> None:
    """
    Trainer-level fail-fast structural validation.
    """

    if not isinstance(
        model,
        Exp05AsymmetricGatedDetectionModel,
    ):
        raise TypeError(
            "Trainer created wrong model class: "
            f"{type(model).__name__}"
        )

    # --------------------------------------------------------
    # Overall model receives 5-channel dataset tensors,
    # but the RGB main branch must remain 3-channel.
    # --------------------------------------------------------

    if (
        getattr(
            model,
            "multimodal_channels",
            None,
        )
        != MULTIMODAL_CHANNELS
    ):
        raise AssertionError(
            "Exp05 multimodal channel metadata mismatch."
        )

    rgb_first_conv = (
        _get_rgb_first_conv(
            model
        )
    )

    if (
        rgb_first_conv.in_channels
        != RGB_MAIN_CHANNELS
    ):
        raise AssertionError(
            "Exp05 RGB main path must remain 3-channel, "
            f"got {rgb_first_conv.in_channels}."
        )

    # --------------------------------------------------------
    # Detection classes
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
        != expected_nc
    ):
        raise AssertionError(
            "Detection head class count mismatch: "
            f"head.nc={head_nc}, "
            f"dataset.nc={expected_nc}"
        )

    # --------------------------------------------------------
    # Strides
    # --------------------------------------------------------

    actual_strides = tuple(
        float(value)
        for value
        in model.stride.detach().cpu().tolist()
    )

    if (
        actual_strides
        != EXPECTED_STRIDES
    ):
        raise AssertionError(
            "Unexpected Exp05 strides: "
            f"{actual_strides}"
        )

    # --------------------------------------------------------
    # Auxiliary branches
    # --------------------------------------------------------

    depth_first = (
        model
        .depth_encoder[0]
        .conv
    )

    ir_first = (
        model
        .ir_encoder[0]
        .conv
    )

    if not isinstance(
        depth_first,
        nn.Conv2d,
    ):
        raise TypeError(
            "Depth encoder first conv is not nn.Conv2d."
        )

    if not isinstance(
        ir_first,
        nn.Conv2d,
    ):
        raise TypeError(
            "IR encoder first conv is not nn.Conv2d."
        )

    if (
        depth_first.in_channels
        != 1
    ):
        raise AssertionError(
            "Depth encoder must accept one channel."
        )

    if (
        ir_first.in_channels
        != 1
    ):
        raise AssertionError(
            "IR encoder must accept one channel."
        )

    # --------------------------------------------------------
    # Gate trainability
    # --------------------------------------------------------

    if not (
        model
        .depth_gate
        .logits
        .requires_grad
    ):
        raise AssertionError(
            "Depth gate requires_grad=False."
        )

    if not (
        model
        .ir_gate
        .logits
        .requires_grad
    ):
        raise AssertionError(
            "IR gate requires_grad=False."
        )


# ============================================================
# Trainer
# ============================================================

class Exp05GatedDetectionTrainer(
    DetectionTrainer
):
    """
    DetectionTrainer for Exp05 asymmetric gated fusion.
    """

    # --------------------------------------------------------
    # Dataset metadata
    # --------------------------------------------------------

    def get_dataset(
        self,
    ):
        """
        Load standard detection metadata and mark the in-memory
        dataset representation as five-channel.

        Important:
            This does NOT mean YOLO's RGB first convolution becomes 5ch.

        Exp05:
            Dataset tensor C = 5

            Model internally:
                RGB branch C = 3
                IR branch  C = 1
                Depth      C = 1
        """

        data = super().get_dataset()

        original_channels = (
            data.get(
                "channels",
                3,
            )
        )

        data[
            "channels"
        ] = MULTIMODAL_CHANNELS

        LOGGER.info(
            "Exp05 dataset channels: "
            f"{original_channels} -> "
            f"{MULTIMODAL_CHANNELS} "
            f"{CHANNEL_NAMES}"
        )

        return data

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    def build_dataset(
        self,
        img_path: str,
        mode: str = "train",
        batch: int | None = None,
    ):
        """
        Build the already validated MultimodalYOLODataset.
        """

        if mode not in {
            "train",
            "val",
        }:
            raise ValueError(
                f"Unsupported dataset mode: {mode}"
            )

        gs = max(
            int(
                unwrap_model(
                    self.model
                )
                .stride
                .max()
            ),
            32,
        )

        # Match the behavior already validated in Exp04.
        rect = bool(
            self.args.rect
            or mode == "val"
        )

        pad = (
            0.0
            if mode == "train"
            else 0.5
        )

        fraction = (
            self.args.fraction
            if mode == "train"
            else 1.0
        )

        dataset = (
            MultimodalYOLODataset(
                img_path=img_path,
                imgsz=self.args.imgsz,
                batch_size=batch,
                augment=(
                    mode
                    == "train"
                ),
                hyp=copy(
                    self.args
                ),
                rect=rect,
                cache=(
                    self.args.cache
                    or None
                ),
                single_cls=(
                    self.args.single_cls
                    or False
                ),
                stride=gs,
                pad=pad,
                prefix=colorstr(
                    f"{mode}: "
                ),
                task=self.args.task,
                classes=self.args.classes,
                data=self.data,
                fraction=fraction,
            )
        )

        if not isinstance(
            dataset,
            MultimodalYOLODataset,
        ):
            raise AssertionError(
                "Exp05 trainer constructed "
                "the wrong Dataset class."
            )

        LOGGER.info(
            f"Exp05 {mode} dataset: "
            f"{type(dataset).__name__}, "
            f"samples={len(dataset)}, "
            f"channels={MULTIMODAL_CHANNELS}"
        )

        return dataset

    # --------------------------------------------------------
    # Batch preprocessing
    # --------------------------------------------------------

    def preprocess_batch(
        self,
        batch: dict,
    ) -> dict:
        """
        Reuse standard Ultralytics preprocessing:

            device transfer
            float()
            /255
            optional multi-scale

        then verify the multimodal tensor contract.
        """

        batch = super().preprocess_batch(
            batch
        )

        image = batch.get(
            "img"
        )

        if not isinstance(
            image,
            torch.Tensor,
        ):
            raise TypeError(
                "Preprocessed batch has no Tensor 'img'."
            )

        if image.ndim != 4:
            raise RuntimeError(
                "Expected BCHW batch image, "
                f"got shape={tuple(image.shape)}."
            )

        if (
            image.shape[1]
            != MULTIMODAL_CHANNELS
        ):
            raise RuntimeError(
                "Exp05 batch channel mismatch: "
                f"expected={MULTIMODAL_CHANNELS}, "
                f"got={image.shape[1]}."
            )

        if not (
            image.is_floating_point()
        ):
            raise RuntimeError(
                "Exp05 batch image is not floating point "
                "after preprocessing."
            )

        if not torch.isfinite(
            image
        ).all():
            raise RuntimeError(
                "Exp05 batch contains NaN/Inf."
            )

        return batch

    # --------------------------------------------------------
    # Model construction
    # --------------------------------------------------------

    def get_model(
        self,
        cfg: str | dict | None = None,
        weights: Any = None,
        verbose: bool = True,
    ):
        """
        Build Exp05 custom DetectionModel.

        RGB pretrained start:
            - construct 12-class Exp05 model
            - transfer normal YOLO11s RGB weights
            - initialize Depth / IR from current RGB backbone
            - reset gates to Exp05 initial values

        Exp05 resume:
            - construct the same Exp05 topology
            - restore complete learned Exp05 state
            - DO NOT reset Depth / IR / gate parameters
        """

        if cfg is None:

            # Normally BaseTrainer.setup_model() obtains cfg from
            # the loaded .pt model before calling this function.
            #
            # Keep a fallback for direct Trainer construction.
            source_model = None

            if isinstance(
                weights,
                nn.Module,
            ):
                source_model = weights

            elif isinstance(
                weights,
                dict,
            ):
                source_model = (
                    weights.get(
                        "model"
                    )
                    or weights.get(
                        "ema"
                    )
                )

            if (
                source_model
                is not None
                and hasattr(
                    source_model,
                    "yaml",
                )
            ):
                cfg = (
                    source_model.yaml
                )

            else:
                raise RuntimeError(
                    "Exp05 get_model() received cfg=None "
                    "and could not recover model YAML "
                    "from source weights."
                )

        # ----------------------------------------------------
        # Construct target Exp05 topology.
        #
        # Internally this custom model deliberately builds the
        # RGB YOLO path with ch=3.
        #
        # Do NOT call:
        #
        #   DetectionModel(... ch=self.data['channels'])
        #
        # because self.data['channels'] == 5 refers to the
        # external multimodal tensor, not RGB main-path width.
        # ----------------------------------------------------

        model = (
            Exp05AsymmetricGatedDetectionModel(
                cfg=cfg,
                nc=self.data[
                    "nc"
                ],
                verbose=(
                    verbose
                    and RANK == -1
                ),
                depth_gate_init=(
                    DEFAULT_DEPTH_GATE
                ),
                ir_gate_init=(
                    DEFAULT_IR_GATE
                ),
            )
        )

        # Give BaseModel.load() meaningful destination names
        # before pretrained class-head remapping.
        model = (
            self
            .set_model_names_for_load(
                model
            )
        )

        # ----------------------------------------------------
        # Weight initialization / resume
        # ----------------------------------------------------

        if weights is not None:

            source_is_exp05 = (
                source_has_exp05_parameters(
                    weights
                )
            )

            initialize_exp05_from_weights(
                model=model,
                weights=weights,
                verbose=verbose,
            )

            if source_is_exp05:

                LOGGER.info(
                    "Exp05 Trainer: learned gated checkpoint "
                    "loaded without auxiliary reinitialization."
                )

            else:

                LOGGER.info(
                    "Exp05 Trainer: RGB pretrained source loaded; "
                    "Depth/IR encoders initialized from RGB."
                )

        else:

            # Scratch mode is not the intended formal experiment,
            # but keep the model internally valid.
            LOGGER.warning(
                "Exp05 Trainer is starting without pretrained weights. "
                "Auxiliary branches are derived from the model's "
                "random RGB initialization."
            )

        # ----------------------------------------------------
        # Trainer-level fail-fast verification
        # ----------------------------------------------------

        _verify_exp05_model_structure(
            model=model,
            expected_nc=self.data[
                "nc"
            ],
        )

        gate_stats = (
            model
            .get_gate_statistics()
        )

        LOGGER.info(
            "Exp05 model ready: "
            f"external_channels={MULTIMODAL_CHANNELS}, "
            f"rgb_main_channels={RGB_MAIN_CHANNELS}, "
            f"classes={self.data['nc']}, "
            f"Depth@layer{DEPTH_FUSION_LAYER}, "
            f"IR@layer{IR_FUSION_LAYER}, "
            f"depth_gate_mean="
            f"{gate_stats['depth']['mean']:.6f}, "
            f"ir_gate_mean="
            f"{gate_stats['ir']['mean']:.6f}"
        )

        return model


# ============================================================
# Import sanity check
# ============================================================

def main() -> None:

    print("=" * 100)
    print(
        "AIC2026 Exp05 - Gated DetectionTrainer"
    )
    print("=" * 100)

    print(
        "Trainer :",
        Exp05GatedDetectionTrainer,
    )

    print(
        "Dataset :",
        MultimodalYOLODataset,
    )

    print(
        "Channels:",
        MULTIMODAL_CHANNELS,
    )

    print(
        "Order   :",
        CHANNEL_NAMES,
    )

    print(
        "Depth   :",
        f"layer {DEPTH_FUSION_LAYER} / P3"
    )

    print(
        "IR      :",
        f"layer {IR_FUSION_LAYER} / P4"
    )

    print(
        "Depth gate init:",
        DEFAULT_DEPTH_GATE,
    )

    print(
        "IR gate init   :",
        DEFAULT_IR_GATE,
    )

    print("=" * 100)
    print(
        "IMPORT CHECK: PASS"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()

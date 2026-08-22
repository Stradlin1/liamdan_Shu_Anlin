#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 Exp04
Custom Ultralytics DetectionTrainer for RGB + IR + Depth Early Fusion.

Responsibilities:

1. Force model input channels to:
       [R, G, B, IR, Depth] = 5 channels

2. Build:
       MultimodalYOLODataset
   instead of:
       YOLODataset

3. Build the detection model using the actual competition class count
   from data.yaml (12 classes).

4. Load pretrained YOLO11s weights.

5. For a 3-channel pretrained checkpoint:
       RGB   <- pretrained exactly
       IR    <- zero
       Depth <- zero

6. Preserve learned 5-channel weights when resuming from a 5-channel
   checkpoint. Never zero IR/Depth during resume.

Ultralytics source itself is not modified.
"""

from __future__ import annotations

from copy import copy
from typing import Any

import torch
import torch.nn as nn

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.torch_utils import unwrap_model

from multimodal_dataset import (
    CHANNEL_NAMES,
    MULTIMODAL_CHANNELS,
    MultimodalYOLODataset,
)


FIRST_CONV_KEY = "model.0.conv.weight"


# ============================================================
# Helper functions
# ============================================================

def _get_first_conv(
    model: nn.Module,
) -> nn.Conv2d:
    """
    Return YOLO's first Conv2d.

    Expected structure:

        DetectionModel
            -> model[0]
                -> conv
    """

    if not hasattr(model, "model"):
        raise RuntimeError(
            "Detection model has no '.model' attribute."
        )

    if len(model.model) == 0:
        raise RuntimeError(
            "Detection model contains no layers."
        )

    first_block = model.model[0]

    if not hasattr(first_block, "conv"):
        raise RuntimeError(
            "YOLO first block has no '.conv' attribute."
        )

    conv = first_block.conv

    if not isinstance(
        conv,
        nn.Conv2d,
    ):
        raise TypeError(
            "Expected first_block.conv to be nn.Conv2d, "
            f"got {type(conv).__name__}."
        )

    return conv


def _extract_source_first_conv_weight(
    weights: Any,
) -> torch.Tensor | None:
    """
    Extract the first-conv weight tensor from a pretrained/resume model.

    Returns a CPU FP32 clone so initialization can be verified after
    Ultralytics performs its normal weight transfer.
    """

    if weights is None:
        return None

    source_model = (
        weights.get("model")
        if isinstance(weights, dict)
        else weights
    )

    if source_model is None:
        return None

    if not hasattr(
        source_model,
        "state_dict",
    ):
        return None

    source_state = (
        source_model
        .float()
        .state_dict()
    )

    weight = source_state.get(
        FIRST_CONV_KEY
    )

    if weight is None:
        return None

    return (
        weight
        .detach()
        .cpu()
        .clone()
    )


# ============================================================
# Multimodal Detection Trainer
# ============================================================

class MultimodalDetectionTrainer(
    DetectionTrainer
):
    """
    Ultralytics DetectionTrainer adapted for Exp04.

    We intentionally keep the customization small.

    Original Ultralytics continues to handle:

        optimizer
        scheduler
        AMP
        EMA
        loss
        validation
        checkpoint saving
        early stopping
        DDP
        resume

    We only replace the multimodal-specific parts.
    """

    # --------------------------------------------------------
    # Dataset metadata
    # --------------------------------------------------------

    def get_dataset(self):
        """
        Load normal detection dataset metadata, then override the
        in-memory channel count to 5.

        We do NOT modify rgb_v1/data.yaml on disk because that YAML is
        also part of the RGB baseline view.

        This keeps Exp01 reproducible while allowing Exp04 to use the
        same RGB image/label membership as its anchor view.
        """

        data = super().get_dataset()

        original_channels = data.get(
            "channels",
            3,
        )

        data["channels"] = (
            MULTIMODAL_CHANNELS
        )

        LOGGER.info(
            "Exp04 multimodal channels: "
            f"{original_channels} -> "
            f"{MULTIMODAL_CHANNELS} "
            f"{CHANNEL_NAMES}"
        )

        return data

    # --------------------------------------------------------
    # Dataset construction
    # --------------------------------------------------------

    def build_dataset(
        self,
        img_path: str,
        mode: str = "train",
        batch: int | None = None,
    ):
        """
        Build MultimodalYOLODataset using the same arguments that
        Ultralytics normally supplies to YOLODataset.
        """

        if mode not in {
            "train",
            "val",
        }:
            raise ValueError(
                f"Unsupported mode: {mode}"
            )

        # Same stride logic as DetectionTrainer.
        gs = max(
            int(
                unwrap_model(
                    self.model
                ).stride.max()
            ),
            32,
        )

        # Mirror Ultralytics build_yolo_dataset behavior.
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

        dataset = MultimodalYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=copy(self.args),
            rect=rect,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
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

        if not isinstance(
            dataset,
            MultimodalYOLODataset,
        ):
            raise AssertionError(
                "Trainer constructed the wrong dataset class."
            )

        LOGGER.info(
            f"Exp04 {mode} dataset: "
            f"{type(dataset).__name__}, "
            f"samples={len(dataset)}"
        )

        return dataset

    # --------------------------------------------------------
    # Model construction
    # --------------------------------------------------------

    def get_model(
        self,
        cfg: str | dict | None = None,
        weights=None,
        verbose: bool = True,
    ):
        """
        Build the normal Ultralytics detection model, but ensure:

            input channels = 5
            output classes = dataset nc

        For 3-channel pretrained weights:

            pretrained[:, 0:3] -> RGB
            new[:, 3]          -> zero
            new[:, 4]          -> zero

        For 5-channel resume checkpoints:

            load all five learned channels normally
            DO NOT zero channels 3/4.
        """

        source_first_weight = (
            _extract_source_first_conv_weight(
                weights
            )
        )

        # ----------------------------------------------------
        # Let the locked Ultralytics DetectionTrainer perform
        # its normal model creation:
        #
        #   DetectionModel(
        #       cfg,
        #       nc=self.data["nc"],
        #       ch=self.data["channels"],
        #   )
        #
        # This is important because the competition model must
        # have a 12-class head rather than the 80-class COCO head
        # used by the standalone model smoke test.
        # ----------------------------------------------------

        model = super().get_model(
            cfg=cfg,
            weights=weights,
            verbose=verbose,
        )

        first_conv = _get_first_conv(
            model
        )

        if (
            first_conv.in_channels
            != MULTIMODAL_CHANNELS
        ):
            raise AssertionError(
                "Trainer model has incorrect input channels: "
                f"{first_conv.in_channels}"
            )

        # Check detection head class count.
        detect_head = model.model[-1]

        head_nc = getattr(
            detect_head,
            "nc",
            None,
        )

        expected_nc = self.data["nc"]

        if head_nc != expected_nc:
            raise AssertionError(
                "Detection head class count mismatch: "
                f"head.nc={head_nc}, "
                f"dataset.nc={expected_nc}"
            )

        # ----------------------------------------------------
        # Initialization policy
        # ----------------------------------------------------

        if source_first_weight is None:

            # Scratch-model fallback.
            #
            # Exp04 normally does NOT use this path because we train
            # from pretrained/yolo11s.pt.
            with torch.no_grad():
                first_conv.weight[
                    :,
                    3:MULTIMODAL_CHANNELS,
                ].zero_()

            LOGGER.warning(
                "No source first-conv weight found. "
                "RGB uses model initialization; "
                "IR/Depth were zero-initialized."
            )

        else:

            source_channels = (
                source_first_weight.shape[1]
            )

            # ------------------------------------------------
            # Standard Exp04 start:
            #
            # COCO checkpoint = 3 channels.
            # ------------------------------------------------

            if source_channels == 3:

                with torch.no_grad():

                    first_conv.weight[
                        :,
                        3:MULTIMODAL_CHANNELS,
                    ].zero_()

                target_rgb = (
                    first_conv
                    .weight[
                        :,
                        0:3,
                    ]
                    .detach()
                    .cpu()
                )

                source_rgb = (
                    source_first_weight[
                        :,
                        0:3,
                    ]
                )

                if (
                    target_rgb.shape
                    != source_rgb.shape
                ):
                    raise AssertionError(
                        "RGB first-conv shape mismatch after "
                        "pretrained loading:\n"
                        f"target={tuple(target_rgb.shape)}\n"
                        f"source={tuple(source_rgb.shape)}"
                    )

                if not torch.equal(
                    target_rgb,
                    source_rgb,
                ):

                    max_diff = (
                        target_rgb
                        - source_rgb
                    ).abs().max().item()

                    raise AssertionError(
                        "Trainer failed to copy pretrained RGB "
                        "first-conv weights exactly. "
                        f"max_abs_diff={max_diff}"
                    )

                ir_nonzero = torch.count_nonzero(
                    first_conv.weight[:, 3]
                ).item()

                depth_nonzero = torch.count_nonzero(
                    first_conv.weight[:, 4]
                ).item()

                if ir_nonzero != 0:
                    raise AssertionError(
                        "IR first-conv initialization is not zero."
                    )

                if depth_nonzero != 0:
                    raise AssertionError(
                        "Depth first-conv initialization is not zero."
                    )

                LOGGER.info(
                    "Exp04 pretrained stem initialization: "
                    "RGB=exact pretrained, IR=zero, Depth=zero"
                )

            # ------------------------------------------------
            # Resume / fine-tune from an already-trained
            # multimodal checkpoint.
            #
            # Absolutely do not zero auxiliary channels here.
            # ------------------------------------------------

            elif (
                source_channels
                == MULTIMODAL_CHANNELS
            ):

                target_weight = (
                    first_conv
                    .weight
                    .detach()
                    .cpu()
                )

                if (
                    target_weight.shape
                    == source_first_weight.shape
                    and not torch.equal(
                        target_weight,
                        source_first_weight,
                    )
                ):
                    max_diff = (
                        target_weight
                        - source_first_weight
                    ).abs().max().item()

                    raise AssertionError(
                        "5-channel resume first-conv weights "
                        "were not restored exactly. "
                        f"max_abs_diff={max_diff}"
                    )

                LOGGER.info(
                    "Exp04 5-channel checkpoint detected: "
                    "preserving learned RGB/IR/Depth stem weights."
                )

            else:

                raise RuntimeError(
                    "Unsupported pretrained input channel count: "
                    f"{source_channels}. "
                    "Expected 3 or 5."
                )

        LOGGER.info(
            "Exp04 model ready: "
            f"channels={first_conv.in_channels}, "
            f"classes={head_nc}, "
            f"order={CHANNEL_NAMES}"
        )

        return model


# ============================================================
# Import-only sanity check
# ============================================================

def main() -> None:

    print(
        "Import OK"
    )

    print(
        "Trainer :",
        MultimodalDetectionTrainer,
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


if __name__ == "__main__":
    main()

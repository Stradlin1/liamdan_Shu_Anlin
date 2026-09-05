#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 configurable early-fusion DetectionTrainer.

Supported input configurations:

    ("rgb",)                  -> [R, G, B] = 3 channels
    ("rgb", "ir")            -> [R, G, B, IR] = 4 channels
    ("rgb", "depth")         -> [R, G, B, Depth] = 4 channels
    ("rgb", "ir", "depth") -> [R, G, B, IR, Depth] = 5 channels

Responsibilities:

1. Keep RGB as the canonical YOLO dataset/label/split anchor.
2. Build MultimodalYOLODataset with the selected modalities.
3. Build the detection model using the selected input-channel count and
   the actual competition class count from data.yaml.
4. Load pretrained YOLO11s weights without modifying Ultralytics source.
5. For a 3-channel pretrained checkpoint:
       RGB <- pretrained exactly
       selected auxiliary channels <- zero
6. When resuming from an already-trained multimodal checkpoint, preserve
   all learned input-channel weights exactly.

Backward compatibility:

    modalities is optional. Omitting it keeps the original Exp04 default:
        RGB + IR + Depth = 5 channels.

Important:

    RGB+IR and RGB+Depth are both 4-channel tensors. Tensor shape alone
    cannot distinguish their semantic meaning, so callers must only resume
    a 4-channel checkpoint with the same modality configuration.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import copy
from typing import Any

import torch
import torch.nn as nn

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.torch_utils import unwrap_model

from multimodal_config import (
    CHANNEL_NAMES,
    DEFAULT_MODALITIES,
    MULTIMODAL_CHANNELS,
    channel_names_for_modalities,
    channels_for_modalities,
    normalize_modalities,
)
from multimodal_dataset import MultimodalYOLODataset


FIRST_CONV_KEY = "model.0.conv.weight"
RGB_CHANNELS = 3


# ============================================================
# Helper functions
# ============================================================

def _get_first_conv(
    model: nn.Module,
) -> nn.Conv2d:
    """Return YOLO's first Conv2d."""

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
    Ultralytics DetectionTrainer for configurable early-fusion inputs.

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

    This trainer only owns multimodal-specific configuration, dataset
    construction, channel metadata, and first-convolution verification.
    """

    def __init__(
        self,
        *args,
        modalities: Iterable[str] | str | None = DEFAULT_MODALITIES,
        **kwargs,
    ):
        # These attributes must exist before DetectionTrainer.__init__ runs,
        # because the parent constructor may call get_dataset()/get_model().
        self.modalities = normalize_modalities(
            modalities
        )
        self.channel_names = channel_names_for_modalities(
            self.modalities
        )
        self.input_channels = channels_for_modalities(
            self.modalities
        )

        super().__init__(
            *args,
            **kwargs,
        )

    # --------------------------------------------------------
    # Dataset metadata
    # --------------------------------------------------------

    def get_dataset(self):
        """
        Load normal detection dataset metadata, then override only the
        in-memory input-channel count for the selected modality set.

        The RGB view YAML is not modified on disk, so existing RGB and
        Exp04 experiments remain reproducible.
        """

        data = super().get_dataset()

        original_channels = data.get(
            "channels",
            RGB_CHANNELS,
        )

        data["channels"] = (
            self.input_channels
        )

        LOGGER.info(
            "Configurable multimodal channels: "
            f"{original_channels} -> "
            f"{self.input_channels}; "
            f"modalities={self.modalities}; "
            f"order={self.channel_names}"
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
        Build MultimodalYOLODataset with the selected modalities using
        the same core arguments Ultralytics supplies to YOLODataset.
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
            modalities=self.modalities,
        )

        if not isinstance(
            dataset,
            MultimodalYOLODataset,
        ):
            raise AssertionError(
                "Trainer constructed the wrong dataset class."
            )

        if dataset.modalities != self.modalities:
            raise AssertionError(
                "Trainer/Dataset modality mismatch: "
                f"trainer={self.modalities}, "
                f"dataset={dataset.modalities}"
            )

        if dataset.num_channels != self.input_channels:
            raise AssertionError(
                "Trainer/Dataset channel-count mismatch: "
                f"trainer={self.input_channels}, "
                f"dataset={dataset.num_channels}"
            )

        LOGGER.info(
            f"Configurable {mode} dataset: "
            f"{type(dataset).__name__}, "
            f"samples={len(dataset)}, "
            f"modalities={self.modalities}, "
            f"channels={self.input_channels}"
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
        Build the normal Ultralytics detection model while enforcing:

            input channels = selected modality channel count
            output classes = dataset nc

        Initialization policy:

            3-channel source checkpoint:
                RGB <- source exactly
                selected auxiliary channels <- zero

            same-channel multimodal checkpoint:
                preserve the complete learned first convolution exactly
        """

        source_first_weight = (
            _extract_source_first_conv_weight(
                weights
            )
        )

        # Let the locked Ultralytics DetectionTrainer perform its normal
        # DetectionModel creation using self.data["channels"] and nc.
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
            != self.input_channels
        ):
            raise AssertionError(
                "Trainer model has incorrect input channels: "
                f"actual={first_conv.in_channels}, "
                f"expected={self.input_channels}"
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
        # Initialization / resume policy
        # ----------------------------------------------------

        if source_first_weight is None:

            # Scratch-model fallback. RGB keeps the model's normal random
            # initialization while every selected auxiliary input is zeroed.
            with torch.no_grad():
                first_conv.weight[
                    :,
                    RGB_CHANNELS:self.input_channels,
                ].zero_()

            LOGGER.warning(
                "No source first-conv weight found. "
                "RGB uses model initialization; selected auxiliary "
                f"channels were zero-initialized: "
                f"{self.channel_names[RGB_CHANNELS:]}"
            )

        else:

            source_channels = int(
                source_first_weight.shape[1]
            )

            # ------------------------------------------------
            # Standard start from a 3-channel COCO checkpoint.
            # ------------------------------------------------

            if source_channels == RGB_CHANNELS:

                with torch.no_grad():
                    first_conv.weight[
                        :,
                        RGB_CHANNELS:self.input_channels,
                    ].zero_()

                target_rgb = (
                    first_conv
                    .weight[
                        :,
                        0:RGB_CHANNELS,
                    ]
                    .detach()
                    .cpu()
                )

                source_rgb = (
                    source_first_weight[
                        :,
                        0:RGB_CHANNELS,
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

                for channel_index, channel_name in enumerate(
                    self.channel_names[RGB_CHANNELS:],
                    start=RGB_CHANNELS,
                ):
                    nonzero = torch.count_nonzero(
                        first_conv.weight[:, channel_index]
                    ).item()

                    if nonzero != 0:
                        raise AssertionError(
                            f"{channel_name} first-conv initialization "
                            "is not zero."
                        )

                LOGGER.info(
                    "Pretrained stem initialization: "
                    "RGB=exact pretrained; "
                    f"auxiliary=zero {self.channel_names[RGB_CHANNELS:]}"
                )

            # ------------------------------------------------
            # Resume/fine-tune from a same-channel checkpoint.
            # ------------------------------------------------

            elif source_channels == self.input_channels:

                target_weight = (
                    first_conv
                    .weight
                    .detach()
                    .cpu()
                )

                if (
                    target_weight.shape
                    != source_first_weight.shape
                ):
                    raise AssertionError(
                        "Resume first-conv shape mismatch:\n"
                        f"target={tuple(target_weight.shape)}\n"
                        f"source={tuple(source_first_weight.shape)}"
                    )

                if not torch.equal(
                    target_weight,
                    source_first_weight,
                ):
                    max_diff = (
                        target_weight
                        - source_first_weight
                    ).abs().max().item()

                    raise AssertionError(
                        "Resume first-conv weights were not restored "
                        "exactly. "
                        f"max_abs_diff={max_diff}"
                    )

                if self.input_channels == 4:
                    LOGGER.warning(
                        "4-channel checkpoint detected. Tensor shape "
                        "cannot distinguish RGB+IR from RGB+Depth; "
                        "caller must ensure checkpoint modalities match "
                        f"current modalities={self.modalities}."
                    )

                LOGGER.info(
                    "Same-channel checkpoint detected: preserving learned "
                    f"stem weights for {self.channel_names}."
                )

            else:

                raise RuntimeError(
                    "Unsupported pretrained input channel count: "
                    f"{source_channels}. Expected either "
                    f"{RGB_CHANNELS} (RGB pretrained) or "
                    f"{self.input_channels} (same-channel resume)."
                )

        LOGGER.info(
            "Configurable model ready: "
            f"modalities={self.modalities}, "
            f"channels={first_conv.in_channels}, "
            f"classes={head_nc}, "
            f"order={self.channel_names}"
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
        "Default modalities:",
        DEFAULT_MODALITIES,
    )

    print(
        "Default channels:",
        MULTIMODAL_CHANNELS,
    )

    print(
        "Default order   :",
        CHANNEL_NAMES,
    )


if __name__ == "__main__":
    main()

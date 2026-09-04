#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared modality configuration for AIC2026 early-fusion experiments."""

from __future__ import annotations

from collections.abc import Iterable


RGB_MODALITY = "rgb"
IR_MODALITY = "ir"
DEPTH_MODALITY = "depth"

DEFAULT_MODALITIES = (
    RGB_MODALITY,
    IR_MODALITY,
    DEPTH_MODALITY,
)

SUPPORTED_MODALITY_CONFIGS = (
    (RGB_MODALITY,),
    (RGB_MODALITY, IR_MODALITY),
    (RGB_MODALITY, DEPTH_MODALITY),
    DEFAULT_MODALITIES,
)

MODALITY_CHANNEL_NAMES = {
    RGB_MODALITY: ("R", "G", "B"),
    IR_MODALITY: ("IR",),
    DEPTH_MODALITY: ("Depth",),
}


def normalize_modalities(
    modalities: Iterable[str] | str | None,
) -> tuple[str, ...]:
    """Validate and normalize one supported early-fusion configuration."""

    if modalities is None:
        normalized = DEFAULT_MODALITIES
    elif isinstance(modalities, str):
        normalized = tuple(
            item.strip().lower()
            for item in modalities.split(",")
            if item.strip()
        )
    else:
        normalized = tuple(
            str(item).strip().lower()
            for item in modalities
        )

    if normalized not in SUPPORTED_MODALITY_CONFIGS:
        supported = ", ".join(
            "+".join(config)
            for config in SUPPORTED_MODALITY_CONFIGS
        )
        raise ValueError(
            f"Unsupported modalities={normalized!r}. "
            f"Expected one of: {supported}."
        )

    return normalized


def channel_names_for_modalities(
    modalities: Iterable[str] | str | None,
) -> tuple[str, ...]:
    """Return the exact tensor channel order for a configuration."""

    normalized = normalize_modalities(modalities)
    return tuple(
        channel_name
        for modality in normalized
        for channel_name in MODALITY_CHANNEL_NAMES[modality]
    )


def channels_for_modalities(
    modalities: Iterable[str] | str | None,
) -> int:
    """Return the number of input channels for a configuration."""

    return len(channel_names_for_modalities(modalities))


# Backward-compatible Exp04 defaults. Existing Exp04 imports continue to
# describe [R, G, B, IR, Depth] and therefore do not silently change behavior.
MULTIMODAL_CHANNELS = channels_for_modalities(DEFAULT_MODALITIES)
CHANNEL_NAMES = channel_names_for_modalities(DEFAULT_MODALITIES)


__all__ = [
    "RGB_MODALITY",
    "IR_MODALITY",
    "DEPTH_MODALITY",
    "DEFAULT_MODALITIES",
    "SUPPORTED_MODALITY_CONFIGS",
    "MODALITY_CHANNEL_NAMES",
    "MULTIMODAL_CHANNELS",
    "CHANNEL_NAMES",
    "normalize_modalities",
    "channel_names_for_modalities",
    "channels_for_modalities",
]

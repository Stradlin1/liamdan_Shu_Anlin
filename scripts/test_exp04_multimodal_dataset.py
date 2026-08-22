#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exp04 Multimodal Dataset Smoke Test

Test target:
    scripts/multimodal_dataset.py

Expected multimodal representation:
    H x W x 5 uint8

Channel order:
    0 = R
    1 = G
    2 = B
    3 = Infrared
    4 = Depth8

Tests:
1. Dataset construction
2. Raw RGB / IR / Depth source consistency
3. Raw HWC 5-channel representation
4. Validation Format -> CHW [5, 960, 960]
5. Format channel-order preservation
6. RGB-only HSV:
       RGB may change
       IR must remain bit-identical
       Depth must remain bit-identical
7. Training augmentation pipeline
8. Mosaic / RandomPerspective / Flip 5-channel compatibility
9. Bounding-box validity
10. collate_fn -> [B, 5, 960, 960]
11. Visual RGB / IR / Depth triplets

No dataset files are modified.

Outputs:
    runs/test_exp04_multimodal_dataset/
    ├── test_summary.json
    ├── raw_triplet.jpg
    ├── val_formatted_triplet.jpg
    └── train_augmented_triplet.jpg
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import random

import cv2
import numpy as np
import torch
import yaml

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data.augment import (
    Albumentations,
    LetterBox,
)

from multimodal_dataset import (
    MultimodalYOLODataset,
    MultimodalAlbumentationsNoOp,
    RGBOnlyRandomHSV,
    PROJECT_ROOT,
    RGB_VIEW_ROOT,
    IR_VIEW_ROOT,
    DEPTH_VIEW_ROOT,
    MULTIMODAL_CHANNELS,
    CHANNEL_NAMES,
    _read_grayscale_uint8,
    _resize_gray_to_hw,
)


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 960
BATCH_SIZE = 8
SEED = 2026

NUM_RAW_SAMPLES_TO_CHECK = 20
NUM_VAL_SAMPLES_TO_CHECK = 20
NUM_TRAIN_AUG_SAMPLES_TO_CHECK = 20

SMOKE_BATCH_SIZE = 4


DATA_YAML = (
    RGB_VIEW_ROOT
    / "data.yaml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "test_exp04_multimodal_dataset"
)


# ============================================================
# Utility
# ============================================================

def fail(
    message: str,
) -> None:

    raise RuntimeError(
        "\n"
        + "=" * 90
        + "\nTEST FAILED\n"
        + "=" * 90
        + "\n"
        + message
    )


def check(
    condition: bool,
    message: str,
) -> None:

    if not condition:
        fail(
            message
        )


def load_data_yaml() -> dict:

    if not DATA_YAML.is_file():

        raise FileNotFoundError(
            f"data.yaml not found:\n"
            f"  {DATA_YAML}"
        )

    with DATA_YAML.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = yaml.safe_load(
            f
        )

    if not isinstance(
        data,
        dict,
    ):

        raise TypeError(
            "data.yaml did not produce a dictionary."
        )

    if "names" not in data:

        raise KeyError(
            "data.yaml missing 'names'."
        )

    return data


def make_train_hyp():

    hyp = deepcopy(
        DEFAULT_CFG
    )

    # --------------------------------------------------------
    # Match Exp01 / Exp02 / Exp03 baseline where applicable.
    # --------------------------------------------------------

    hyp.mosaic = 1.0

    hyp.mixup = 0.0
    hyp.cutmix = 0.0
    hyp.copy_paste = 0.0

    hyp.degrees = 0.0
    hyp.translate = 0.1
    hyp.scale = 0.5
    hyp.shear = 0.0
    hyp.perspective = 0.0

    hyp.flipud = 0.0
    hyp.fliplr = 0.5

    hyp.hsv_h = 0.015
    hyp.hsv_s = 0.7
    hyp.hsv_v = 0.4

    # Critical:
    # 5-channel tensor is explicitly:
    # [R, G, B, IR, Depth]
    hyp.bgr = 0.0

    # Generic Albumentations is intentionally disabled
    # by MultimodalYOLODataset.build_transforms().
    try:
        hyp.augmentations = None
    except Exception:
        pass

    return hyp


def make_val_hyp():

    hyp = deepcopy(
        DEFAULT_CFG
    )

    hyp.bgr = 0.0

    return hyp


# ============================================================
# Dataset construction
# ============================================================

def build_dataset(
    split: str,
    augment: bool,
):

    data = load_data_yaml()

    if split not in {
        "train",
        "val",
    }:

        raise ValueError(
            f"Unsupported split: {split}"
        )

    hyp = (
        make_train_hyp()
        if augment
        else make_val_hyp()
    )

    image_dir = (
        RGB_VIEW_ROOT
        / "images"
        / split
    )

    dataset = MultimodalYOLODataset(

        img_path=str(
            image_dir
        ),

        imgsz=IMAGE_SIZE,

        cache=False,

        augment=augment,

        hyp=hyp,

        prefix=(
            f"Exp04 {split}: "
        ),

        rect=False,

        batch_size=BATCH_SIZE,

        stride=32,

        pad=0.5,

        single_cls=False,

        classes=None,

        fraction=1.0,

        data=data,

        task="detect",

        ir_view_root=IR_VIEW_ROOT,

        depth_view_root=DEPTH_VIEW_ROOT,
    )

    return dataset


# ============================================================
# Transform introspection
# ============================================================

def flatten_transforms(
    obj,
):

    result = []

    children = getattr(
        obj,
        "transforms",
        None,
    )

    if children is None:

        result.append(
            obj
        )

        return result

    for child in children:

        result.extend(
            flatten_transforms(
                child
            )
        )

    return result


def inspect_training_transforms(
    dataset,
):

    transforms = flatten_transforms(
        dataset.transforms
    )

    names = [
        type(t).__name__
        for t in transforms
    ]

    rgb_hsv_count = sum(
        isinstance(
            t,
            RGBOnlyRandomHSV,
        )
        for t in transforms
    )

    normal_hsv_count = sum(
        (
            type(t).__name__
            == "RandomHSV"
        )
        for t in transforms
    )

    noop_albu_count = sum(
        isinstance(
            t,
            MultimodalAlbumentationsNoOp,
        )
        for t in transforms
    )

    normal_albu_count = sum(
        isinstance(
            t,
            Albumentations,
        )
        for t in transforms
    )

    check(
        rgb_hsv_count == 1,
        "Training pipeline does not contain exactly one "
        "RGBOnlyRandomHSV.\n"
        f"Transforms:\n{names}"
    )

    check(
        normal_hsv_count == 0,
        "Original 3-channel RandomHSV still exists in "
        "multimodal training pipeline."
    )

    check(
        normal_albu_count == 0,
        "Original Albumentations still exists in "
        "multimodal training pipeline."
    )

    return {
        "transform_names":
            names,

        "rgb_only_hsv_count":
            rgb_hsv_count,

        "multimodal_albumentations_noop_count":
            noop_albu_count,
    }


# ============================================================
# Sample selection
# ============================================================

def deterministic_indices(
    length: int,
    count: int,
    seed: int,
):

    count = min(
        count,
        length,
    )

    rng = random.Random(
        seed
    )

    return sorted(
        rng.sample(
            range(
                length
            ),
            count,
        )
    )


def find_resolution_representative_indices(
    dataset,
):

    result = {}

    for i, label in enumerate(
        dataset.labels
    ):

        shape = label.get(
            "shape"
        )

        if shape is None:
            continue

        key = tuple(
            int(x)
            for x in shape
        )

        if key not in result:

            result[
                key
            ] = i

    return result


# ============================================================
# Raw source verification
# ============================================================

def check_raw_sample(
    dataset,
    index: int,
):

    raw = dataset.get_image_and_label(
        index
    )

    img = raw[
        "img"
    ]

    check(
        isinstance(
            img,
            np.ndarray,
        ),
        f"Raw image is not numpy array: {type(img)}"
    )

    check(
        img.ndim == 3,
        f"Raw image ndim != 3: {img.shape}"
    )

    check(
        img.shape[2]
        == MULTIMODAL_CHANNELS,
        f"Raw channel count is not 5: {img.shape}"
    )

    check(
        img.dtype
        == np.uint8,
        f"Raw dtype is not uint8: {img.dtype}"
    )

    paths = dataset.get_multimodal_paths(
        index
    )

    ori_h, ori_w = (
        int(
            raw[
                "ori_shape"
            ][0]
        ),
        int(
            raw[
                "ori_shape"
            ][1]
        ),
    )

    target_h, target_w = (
        int(
            raw[
                "resized_shape"
            ][0]
        ),
        int(
            raw[
                "resized_shape"
            ][1]
        ),
    )

    # --------------------------------------------------------
    # RGB expected
    # --------------------------------------------------------

    rgb_bgr = cv2.imread(
        str(
            paths[
                "rgb"
            ]
        ),
        cv2.IMREAD_COLOR,
    )

    check(
        rgb_bgr is not None,
        f"Cannot read RGB: {paths['rgb']}"
    )

    check(
        rgb_bgr.shape[:2]
        == (
            ori_h,
            ori_w,
        ),
        "RGB original shape mismatch."
    )

    if (
        rgb_bgr.shape[0]
        != target_h
        or rgb_bgr.shape[1]
        != target_w
    ):

        rgb_bgr = cv2.resize(
            rgb_bgr,
            (
                target_w,
                target_h,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

    expected_rgb = cv2.cvtColor(
        rgb_bgr,
        cv2.COLOR_BGR2RGB,
    )

    check(
        np.array_equal(
            img[
                ...,
                :3,
            ],
            expected_rgb,
        ),
        "Raw RGB channels do not exactly match expected "
        "RGB source after resize and BGR->RGB."
    )

    # --------------------------------------------------------
    # IR expected
    # --------------------------------------------------------

    ir = _read_grayscale_uint8(
        paths[
            "ir"
        ],
        "IR",
    )

    ir = _resize_gray_to_hw(
        ir,
        (
            target_h,
            target_w,
        ),
    )

    check(
        np.array_equal(
            img[
                ...,
                3,
            ],
            ir,
        ),
        "Raw IR channel does not exactly match source."
    )

    # --------------------------------------------------------
    # Depth expected
    # --------------------------------------------------------

    depth = _read_grayscale_uint8(
        paths[
            "depth"
        ],
        "Depth",
    )

    depth = _resize_gray_to_hw(
        depth,
        (
            target_h,
            target_w,
        ),
    )

    check(
        np.array_equal(
            img[
                ...,
                4,
            ],
            depth,
        ),
        "Raw Depth channel does not exactly match source."
    )

    return raw


# ============================================================
# Validation Format verification
# ============================================================

def check_val_formatted_sample(
    dataset,
    index: int,
):

    # Raw pre-transform sample.
    raw = dataset.get_image_and_label(
        index
    )

    raw_img = raw[
        "img"
    ].copy()

    # Full validation pipeline.
    formatted = dataset[
        index
    ]

    tensor = formatted[
        "img"
    ]

    check(
        torch.is_tensor(
            tensor
        ),
        f"Formatted image is not torch Tensor: {type(tensor)}"
    )

    check(
        tuple(
            tensor.shape
        )
        == (
            MULTIMODAL_CHANNELS,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        "Unexpected validation tensor shape:\n"
        f"  expected=(5,{IMAGE_SIZE},{IMAGE_SIZE})\n"
        f"  actual={tuple(tensor.shape)}"
    )

    check(
        tensor.dtype
        == torch.uint8,
        f"Dataset Format dtype expected uint8, got {tensor.dtype}"
    )

    # --------------------------------------------------------
    # Reproduce validation image transform manually:
    #
    # raw HWC 5ch
    # -> LetterBox
    # -> HWC -> CHW
    #
    # There must be NO 5-channel reversal.
    # --------------------------------------------------------

    letterbox = LetterBox(
        new_shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        scaleup=False,
    )

    expected_hwc = letterbox(
        image=raw_img
    )

    expected_chw = np.ascontiguousarray(
        expected_hwc.transpose(
            2,
            0,
            1,
        )
    )

    actual = (
        tensor
        .cpu()
        .numpy()
    )

    check(
        np.array_equal(
            actual,
            expected_chw,
        ),
        "Validation Format output does not exactly match "
        "expected [R,G,B,IR,Depth] CHW tensor.\n"
        "Possible channel-order or LetterBox problem."
    )

    check_bboxes(
        formatted,
        context=(
            f"val index={index}"
        ),
    )

    return formatted


# ============================================================
# RGB-only HSV isolation test
# ============================================================

def check_rgb_only_hsv(
    raw_sample,
):

    original = raw_sample[
        "img"
    ].copy()

    labels = {
        "img":
            original.copy()
    }

    augmenter = RGBOnlyRandomHSV(
        hgain=0.015,
        sgain=0.7,
        vgain=0.4,
    )

    np.random.seed(
        SEED
    )

    result = augmenter.apply_image(
        labels
    )

    after = result[
        "img"
    ]

    # Absolutely critical:
    # auxiliary modalities must be untouched.

    check(
        np.array_equal(
            original[
                ...,
                3,
            ],
            after[
                ...,
                3,
            ],
        ),
        "RGBOnlyRandomHSV changed the IR channel."
    )

    check(
        np.array_equal(
            original[
                ...,
                4,
            ],
            after[
                ...,
                4,
            ],
        ),
        "RGBOnlyRandomHSV changed the Depth channel."
    )

    rgb_changed_pixels = int(
        np.count_nonzero(
            original[
                ...,
                :3,
            ]
            != after[
                ...,
                :3,
            ]
        )
    )

    check(
        rgb_changed_pixels > 0,
        "RGBOnlyRandomHSV did not modify any RGB pixels "
        "with the deterministic smoke-test seed."
    )

    return {
        "rgb_changed_values":
            rgb_changed_pixels,

        "ir_identical":
            True,

        "depth_identical":
            True,
    }


# ============================================================
# Bounding-box validation
# ============================================================

def check_bboxes(
    sample,
    context: str,
):

    bboxes = sample.get(
        "bboxes"
    )

    cls = sample.get(
        "cls"
    )

    check(
        bboxes is not None,
        f"{context}: bboxes missing."
    )

    check(
        cls is not None,
        f"{context}: cls missing."
    )

    check(
        torch.is_tensor(
            bboxes
        ),
        f"{context}: bboxes is not Tensor."
    )

    check(
        torch.is_tensor(
            cls
        ),
        f"{context}: cls is not Tensor."
    )

    check(
        bboxes.ndim == 2
        and bboxes.shape[1] == 4,
        f"{context}: invalid bbox shape {tuple(bboxes.shape)}"
    )

    check(
        len(
            bboxes
        )
        == len(
            cls
        ),
        f"{context}: bbox/cls count mismatch."
    )

    if len(
        bboxes
    ) == 0:

        return

    check(
        torch.isfinite(
            bboxes
        ).all().item(),
        f"{context}: non-finite bbox detected."
    )

    tolerance = 1e-5

    check(
        (
            bboxes
            >= -tolerance
        ).all().item(),
        f"{context}: bbox value below 0 after augmentation."
    )

    check(
        (
            bboxes
            <= 1.0
            + tolerance
        ).all().item(),
        f"{context}: bbox value above 1 after augmentation."
    )

    # xywh
    widths = bboxes[
        :,
        2
    ]

    heights = bboxes[
        :,
        3
    ]

    check(
        (
            widths > 0
        ).all().item(),
        f"{context}: zero/negative bbox width."
    )

    check(
        (
            heights > 0
        ).all().item(),
        f"{context}: zero/negative bbox height."
    )


# ============================================================
# Tensor / visualization helpers
# ============================================================

def tensor_to_modalities(
    tensor: torch.Tensor,
):

    array = (
        tensor
        .detach()
        .cpu()
        .numpy()
    )

    check(
        array.ndim == 3
        and array.shape[0] == 5,
        f"Expected CHW 5-channel tensor, got {array.shape}"
    )

    rgb = array[
        0:3
    ].transpose(
        1,
        2,
        0,
    )

    ir = array[
        3
    ]

    depth = array[
        4
    ]

    rgb = np.clip(
        rgb,
        0,
        255,
    ).astype(
        np.uint8
    )

    ir = np.clip(
        ir,
        0,
        255,
    ).astype(
        np.uint8
    )

    depth = np.clip(
        depth,
        0,
        255,
    ).astype(
        np.uint8
    )

    return (
        rgb,
        ir,
        depth,
    )


def draw_bboxes(
    image_bgr: np.ndarray,
    sample,
):

    output = image_bgr.copy()

    bboxes = sample.get(
        "bboxes"
    )

    cls = sample.get(
        "cls"
    )

    if (
        bboxes is None
        or cls is None
        or len(
            bboxes
        ) == 0
    ):

        return output

    boxes = (
        bboxes
        .detach()
        .cpu()
        .numpy()
    )

    classes = (
        cls
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    h, w = output.shape[:2]

    for box, class_id in zip(
        boxes,
        classes,
    ):

        xc, yc, bw, bh = [
            float(x)
            for x in box
        ]

        x1 = int(
            round(
                (
                    xc
                    - bw / 2
                )
                * w
            )
        )

        y1 = int(
            round(
                (
                    yc
                    - bh / 2
                )
                * h
            )
        )

        x2 = int(
            round(
                (
                    xc
                    + bw / 2
                )
                * w
            )
        )

        y2 = int(
            round(
                (
                    yc
                    + bh / 2
                )
                * h
            )
        )

        x1 = max(
            0,
            min(
                w - 1,
                x1,
            ),
        )

        y1 = max(
            0,
            min(
                h - 1,
                y1,
            ),
        )

        x2 = max(
            0,
            min(
                w - 1,
                x2,
            ),
        )

        y2 = max(
            0,
            min(
                h - 1,
                y2,
            ),
        )

        cv2.rectangle(
            output,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            (
                255,
                255,
                255,
            ),
            2,
        )

        cv2.putText(
            output,
            str(
                int(
                    class_id
                )
            ),
            (
                x1,
                max(
                    18,
                    y1 - 4,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                255,
                255,
                255,
            ),
            1,
            cv2.LINE_AA,
        )

    return output


def add_title(
    image: np.ndarray,
    title: str,
):

    image = image.copy()

    h, w = image.shape[:2]

    bar_h = 42

    canvas = np.zeros(
        (
            h + bar_h,
            w,
            3,
        ),
        dtype=np.uint8,
    )

    canvas[
        bar_h:
    ] = image

    cv2.putText(
        canvas,
        title,
        (
            12,
            28,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (
            255,
            255,
            255,
        ),
        2,
        cv2.LINE_AA,
    )

    return canvas


def resize_panel(
    image,
    width=480,
):

    h, w = image.shape[:2]

    ratio = (
        width
        / w
    )

    new_h = int(
        round(
            h
            * ratio
        )
    )

    return cv2.resize(
        image,
        (
            width,
            new_h,
        ),
        interpolation=cv2.INTER_AREA,
    )


def save_tensor_triplet(
    tensor,
    sample,
    output_path: Path,
    title_prefix: str,
    draw_boxes: bool = True,
):

    rgb, ir, depth = (
        tensor_to_modalities(
            tensor
        )
    )

    rgb_bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )

    ir_bgr = cv2.cvtColor(
        ir,
        cv2.COLOR_GRAY2BGR,
    )

    depth_bgr = cv2.cvtColor(
        depth,
        cv2.COLOR_GRAY2BGR,
    )

    if draw_boxes:

        rgb_bgr = draw_bboxes(
            rgb_bgr,
            sample,
        )

        ir_bgr = draw_bboxes(
            ir_bgr,
            sample,
        )

        depth_bgr = draw_bboxes(
            depth_bgr,
            sample,
        )

    rgb_bgr = resize_panel(
        rgb_bgr
    )

    ir_bgr = resize_panel(
        ir_bgr
    )

    depth_bgr = resize_panel(
        depth_bgr
    )

    rgb_bgr = add_title(
        rgb_bgr,
        f"{title_prefix} - RGB",
    )

    ir_bgr = add_title(
        ir_bgr,
        f"{title_prefix} - IR",
    )

    depth_bgr = add_title(
        depth_bgr,
        f"{title_prefix} - Depth",
    )

    # Ensure equal panel heights.
    min_h = min(
        rgb_bgr.shape[0],
        ir_bgr.shape[0],
        depth_bgr.shape[0],
    )

    rgb_bgr = rgb_bgr[
        :min_h
    ]

    ir_bgr = ir_bgr[
        :min_h
    ]

    depth_bgr = depth_bgr[
        :min_h
    ]

    triplet = np.hstack(
        (
            rgb_bgr,
            ir_bgr,
            depth_bgr,
        )
    )

    ok = cv2.imwrite(
        str(
            output_path
        ),
        triplet,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95,
        ],
    )

    check(
        ok,
        f"Failed to write visualization: {output_path}"
    )


def save_raw_triplet(
    raw,
    output_path: Path,
):

    img = raw[
        "img"
    ]

    rgb = img[
        ...,
        :3
    ]

    ir = img[
        ...,
        3
    ]

    depth = img[
        ...,
        4
    ]

    rgb_bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )

    ir_bgr = cv2.cvtColor(
        ir,
        cv2.COLOR_GRAY2BGR,
    )

    depth_bgr = cv2.cvtColor(
        depth,
        cv2.COLOR_GRAY2BGR,
    )

    rgb_bgr = add_title(
        resize_panel(
            rgb_bgr
        ),
        "RAW - RGB",
    )

    ir_bgr = add_title(
        resize_panel(
            ir_bgr
        ),
        "RAW - IR",
    )

    depth_bgr = add_title(
        resize_panel(
            depth_bgr
        ),
        "RAW - Depth",
    )

    min_h = min(
        rgb_bgr.shape[0],
        ir_bgr.shape[0],
        depth_bgr.shape[0],
    )

    triplet = np.hstack(
        (
            rgb_bgr[
                :min_h
            ],
            ir_bgr[
                :min_h
            ],
            depth_bgr[
                :min_h
            ],
        )
    )

    ok = cv2.imwrite(
        str(
            output_path
        ),
        triplet,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95,
        ],
    )

    check(
        ok,
        f"Failed to write {output_path}"
    )


# ============================================================
# Batch collate check
# ============================================================

def check_batch_collate(
    dataset,
    indices,
    context,
):

    samples = [
        dataset[
            i
        ]
        for i in indices
    ]

    batch = dataset.collate_fn(
        samples
    )

    check(
        "img" in batch,
        f"{context}: collated batch missing img."
    )

    images = batch[
        "img"
    ]

    check(
        torch.is_tensor(
            images
        ),
        f"{context}: batch img is not Tensor."
    )

    expected_shape = (
        len(
            indices
        ),
        MULTIMODAL_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    check(
        tuple(
            images.shape
        )
        == expected_shape,
        f"{context}: bad batch shape.\n"
        f"  expected={expected_shape}\n"
        f"  actual={tuple(images.shape)}"
    )

    check(
        images.dtype
        == torch.uint8,
        f"{context}: expected uint8 dataset batch, "
        f"got {images.dtype}"
    )

    return {
        "shape":
            list(
                images.shape
            ),

        "dtype":
            str(
                images.dtype
            ),

        "min":
            int(
                images.min().item()
            ),

        "max":
            int(
                images.max().item()
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    random.seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    print("=" * 94)
    print("Exp04 Multimodal Dataset Smoke Test")
    print("=" * 94)

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"RGB view     : {RGB_VIEW_ROOT}"
    )

    print(
        f"IR view      : {IR_VIEW_ROOT}"
    )

    print(
        f"Depth view   : {DEPTH_VIEW_ROOT}"
    )

    print(
        f"Image size   : {IMAGE_SIZE}"
    )

    print(
        f"Channels     : {MULTIMODAL_CHANNELS}"
    )

    print(
        f"Order        : {CHANNEL_NAMES}"
    )

    print("=" * 94)

    if OUTPUT_DIR.exists():

        raise FileExistsError(
            "Smoke-test output directory already exists:\n"
            f"  {OUTPUT_DIR}\n\n"
            "To rerun explicitly:\n"
            "  rm -rf runs/test_exp04_multimodal_dataset"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    # ========================================================
    # Build validation dataset
    # ========================================================

    print()
    print("[1/8] Building validation dataset...")

    val_dataset = build_dataset(
        split="val",
        augment=False,
    )

    check(
        len(
            val_dataset
        )
        == 400,
        f"Expected 400 val samples, got {len(val_dataset)}"
    )

    print(
        f"      val samples = {len(val_dataset)}"
    )

    # ========================================================
    # Build training dataset
    # ========================================================

    print()
    print("[2/8] Building training dataset...")

    train_dataset = build_dataset(
        split="train",
        augment=True,
    )

    check(
        len(
            train_dataset
        )
        == 1600,
        f"Expected 1600 train samples, got {len(train_dataset)}"
    )

    print(
        f"      train samples = {len(train_dataset)}"
    )

    transform_summary = inspect_training_transforms(
        train_dataset
    )

    print(
        "      RGBOnlyRandomHSV = PASS"
    )

    print(
        "      original RandomHSV removed = PASS"
    )

    print(
        "      original Albumentations removed = PASS"
    )

    # ========================================================
    # Raw source checks
    # ========================================================

    print()
    print("[3/8] Checking raw 5-channel source mapping...")

    raw_indices = deterministic_indices(
        len(
            val_dataset
        ),
        NUM_RAW_SAMPLES_TO_CHECK,
        SEED,
    )

    # Explicitly include one sample for every native resolution
    # represented in the validation set.

    resolution_indices = (
        find_resolution_representative_indices(
            val_dataset
        )
    )

    raw_indices = sorted(
        set(
            raw_indices
        )
        | set(
            resolution_indices.values()
        )
    )

    first_raw = None

    for position, index in enumerate(
        raw_indices,
        start=1,
    ):

        raw = check_raw_sample(
            val_dataset,
            index,
        )

        if first_raw is None:

            first_raw = raw

        print(
            f"      raw "
            f"{position:02d}/{len(raw_indices):02d} "
            f"index={index:3d} "
            f"shape={raw['img'].shape} PASS"
        )

    check(
        first_raw is not None,
        "No raw sample was checked."
    )

    save_raw_triplet(
        first_raw,
        OUTPUT_DIR
        / "raw_triplet.jpg",
    )

    # ========================================================
    # RGB-only HSV check
    # ========================================================

    print()
    print("[4/8] Checking RGB-only HSV isolation...")

    hsv_summary = check_rgb_only_hsv(
        first_raw
    )

    print(
        "      RGB changed       : PASS"
    )

    print(
        "      IR bit-identical  : PASS"
    )

    print(
        "      Depth bit-identical: PASS"
    )

    # ========================================================
    # Validation Format checks
    # ========================================================

    print()
    print("[5/8] Checking validation LetterBox + Format...")

    val_indices = deterministic_indices(
        len(
            val_dataset
        ),
        NUM_VAL_SAMPLES_TO_CHECK,
        SEED + 1,
    )

    first_val_formatted = None

    for position, index in enumerate(
        val_indices,
        start=1,
    ):

        formatted = check_val_formatted_sample(
            val_dataset,
            index,
        )

        if (
            first_val_formatted
            is None
        ):

            first_val_formatted = (
                formatted
            )

        print(
            f"      val "
            f"{position:02d}/{len(val_indices):02d} "
            f"index={index:3d} "
            f"tensor={tuple(formatted['img'].shape)} "
            "PASS"
        )

    save_tensor_triplet(
        first_val_formatted[
            "img"
        ],
        first_val_formatted,
        OUTPUT_DIR
        / "val_formatted_triplet.jpg",
        "VAL 5CH",
        draw_boxes=True,
    )

    # ========================================================
    # Warm mosaic buffer
    # ========================================================

    print()
    print("[6/8] Warming training mosaic buffer...")

    warm_count = min(
        64,
        len(
            train_dataset
        ),
    )

    for i in range(
        warm_count
    ):

        train_dataset.get_image_and_label(
            i
        )

    print(
        f"      buffer size = "
        f"{len(train_dataset.buffer)}"
    )

    check(
        len(
            train_dataset.buffer
        ) > 0,
        "Training Mosaic buffer is empty."
    )

    # ========================================================
    # Full train augmentation
    # ========================================================

    print()
    print("[7/8] Checking full training augmentation pipeline...")

    train_indices = deterministic_indices(
        len(
            train_dataset
        ),
        NUM_TRAIN_AUG_SAMPLES_TO_CHECK,
        SEED + 2,
    )

    first_train_augmented = None

    for position, index in enumerate(
        train_indices,
        start=1,
    ):

        sample = train_dataset[
            index
        ]

        tensor = sample[
            "img"
        ]

        check(
            torch.is_tensor(
                tensor
            ),
            "Train augmented image is not Tensor."
        )

        check(
            tuple(
                tensor.shape
            )
            == (
                5,
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            "Train augmented tensor shape is wrong:\n"
            f"  index={index}\n"
            f"  shape={tuple(tensor.shape)}"
        )

        check(
            tensor.dtype
            == torch.uint8,
            "Train augmented tensor dtype is not uint8."
        )

        check_bboxes(
            sample,
            context=(
                f"train index={index}"
            ),
        )

        if (
            first_train_augmented
            is None
        ):

            first_train_augmented = (
                sample
            )

        print(
            f"      train "
            f"{position:02d}/{len(train_indices):02d} "
            f"index={index:4d} "
            f"tensor={tuple(tensor.shape)} "
            f"boxes={len(sample['bboxes']):3d} "
            "PASS"
        )

    save_tensor_triplet(
        first_train_augmented[
            "img"
        ],
        first_train_augmented,
        OUTPUT_DIR
        / "train_augmented_triplet.jpg",
        "TRAIN AUG",
        draw_boxes=True,
    )

    # ========================================================
    # Collate test
    # ========================================================

    print()
    print("[8/8] Checking collate_fn...")

    val_batch_indices = (
        val_indices[
            :SMOKE_BATCH_SIZE
        ]
    )

    train_batch_indices = (
        train_indices[
            :SMOKE_BATCH_SIZE
        ]
    )

    val_batch_summary = (
        check_batch_collate(
            val_dataset,
            val_batch_indices,
            "validation",
        )
    )

    train_batch_summary = (
        check_batch_collate(
            train_dataset,
            train_batch_indices,
            "training",
        )
    )

    print(
        "      val batch   : "
        f"{val_batch_summary['shape']} PASS"
    )

    print(
        "      train batch : "
        f"{train_batch_summary['shape']} PASS"
    )

    # ========================================================
    # Summary
    # ========================================================

    summary = {
        "status":
            "PASS",

        "project_root":
            str(
                PROJECT_ROOT
            ),

        "representation": {
            "channels":
                MULTIMODAL_CHANNELS,

            "channel_order":
                list(
                    CHANNEL_NAMES
                ),

            "image_size":
                IMAGE_SIZE,

            "raw_layout":
                "HWC",

            "formatted_layout":
                "CHW",

            "dtype":
                "uint8",
        },

        "dataset": {
            "train_samples":
                len(
                    train_dataset
                ),

            "val_samples":
                len(
                    val_dataset
                ),
        },

        "native_resolution_representatives": {
            str(
                shape
            ):
                int(
                    index
                )
            for shape, index
            in resolution_indices.items()
        },

        "tests": {
            "raw_samples_checked":
                len(
                    raw_indices
                ),

            "val_formatted_samples_checked":
                len(
                    val_indices
                ),

            "train_augmented_samples_checked":
                len(
                    train_indices
                ),

            "raw_source_mapping":
                "PASS",

            "five_channel_raw":
                "PASS",

            "format_channel_order":
                "PASS",

            "val_tensor_shape":
                "PASS",

            "rgb_only_hsv":
                "PASS",

            "ir_hsv_unchanged":
                "PASS",

            "depth_hsv_unchanged":
                "PASS",

            "train_augmentation":
                "PASS",

            "bbox_validation":
                "PASS",

            "collate":
                "PASS",
        },

        "hsv_test":
            hsv_summary,

        "transforms":
            transform_summary,

        "val_batch":
            val_batch_summary,

        "train_batch":
            train_batch_summary,

        "outputs": {
            "raw_triplet":
                "raw_triplet.jpg",

            "val_formatted_triplet":
                "val_formatted_triplet.jpg",

            "train_augmented_triplet":
                "train_augmented_triplet.jpg",
        },
    }

    summary_path = (
        OUTPUT_DIR
        / "test_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Final report
    # ========================================================

    print()
    print("=" * 94)
    print("EXP04 MULTIMODAL DATASET SMOKE TEST: PASS")
    print("=" * 94)

    print(
        f"Train samples            : "
        f"{len(train_dataset)}"
    )

    print(
        f"Val samples              : "
        f"{len(val_dataset)}"
    )

    print(
        "Raw representation       : "
        "H x W x 5 uint8"
    )

    print(
        "Channel order            : "
        "[R, G, B, IR, Depth]"
    )

    print(
        "Formatted representation : "
        f"[5, {IMAGE_SIZE}, {IMAGE_SIZE}] uint8"
    )

    print(
        "RGB-only HSV             : PASS"
    )

    print(
        "IR untouched by HSV      : PASS"
    )

    print(
        "Depth untouched by HSV   : PASS"
    )

    print(
        "Mosaic/affine/flip       : PASS"
    )

    print(
        "BBox validation          : PASS"
    )

    print(
        "Batch collate            : PASS"
    )

    print()

    print(
        f"Results : {OUTPUT_DIR}"
    )

    print(
        f"Summary : {summary_path}"
    )

    print(
        "Visual  : "
        f"{OUTPUT_DIR / 'raw_triplet.jpg'}"
    )

    print(
        "Visual  : "
        f"{OUTPUT_DIR / 'val_formatted_triplet.jpg'}"
    )

    print(
        "Visual  : "
        f"{OUTPUT_DIR / 'train_augmented_triplet.jpg'}"
    )

    print()
    print(
        "Next step:"
    )

    print(
        "  Build YOLO11s 5-channel input model "
        "(3 -> 5 input channels)."
    )

    print("=" * 94)


if __name__ == "__main__":
    main()

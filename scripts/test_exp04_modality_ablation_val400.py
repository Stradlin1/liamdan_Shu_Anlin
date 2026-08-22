#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 Exp04 modality ablation on the fixed 400-image validation split.

Same trained checkpoint, no retraining:

    full:
        RGB + IR + Depth

    no_ir:
        RGB + 0 + Depth

    no_depth:
        RGB + IR + 0

    rgb_only:
        RGB + 0 + 0

The script intentionally uses the already parity-tested square 960 inference
protocol before the separate rect-inference A/B experiment:

    input order = [R, G, B, IR, Depth]
    imgsz       = 960 x 960 square LetterBox
    scaleup     = False
    FP32 / 255
    conf        = 0.001
    NMS IoU     = 0.70
    max_det     = 100
    multi_label = True
    NMS         = per-image

Metrics are computed from the competition's stated 101-point AP procedure
for IoU thresholds 0.50:0.05:0.95.

Important:
This is a TEST-TIME modality ablation of one trained 5-channel model.
It measures how much that trained model relies on each auxiliary modality.
It is not equivalent to retraining RGB+Depth, RGB+IR, or RGB-only models.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    from ultralytics.utils.ops import non_max_suppression


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RGB_VIEW = (
    PROJECT_ROOT
    / "yolo_views"
    / "rgb_v1"
)

IR_VIEW = (
    PROJECT_ROOT
    / "yolo_views"
    / "ir_v1"
)

DEPTH_VIEW = (
    PROJECT_ROOT
    / "yolo_views"
    / "depth8_v1"
)

RGB_IMAGE_DIR = (
    RGB_VIEW
    / "images"
    / "val"
)

IR_IMAGE_DIR = (
    IR_VIEW
    / "images"
    / "val"
)

DEPTH_IMAGE_DIR = (
    DEPTH_VIEW
    / "images"
    / "val"
)

LABEL_DIR = (
    RGB_VIEW
    / "labels"
    / "val"
)

VAL_SPLIT = (
    PROJECT_ROOT
    / "splits"
    / "aic2026_group_stratified_v2"
    / "val.txt"
)

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "exp04_rgbid_early5_yolo11s_960"
    / "weights"
    / "best.pt"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "exp04_rgbid_early5_yolo11s_960"
    / "modality_ablation_val400"
)


# ============================================================
# Evaluation configuration
# ============================================================

IMAGE_SIZE = 960

CONF_THRESHOLD = 0.001
IOU_THRESHOLD = 0.70
MAX_DET = 100

EXPECTED_CLASSES = 12
EXPECTED_CHANNELS = 5
EXPECTED_VAL_SAMPLES = 400

IOU_THRESHOLDS = np.arange(
    0.50,
    0.96,
    0.05,
    dtype=np.float64,
)

CLASS_NAMES = (
    "person",
    "boat",
    "animal",
    "seat",
    "sign",
    "bicycle",
    "car",
    "ball",
    "light",
    "garbage_can",
    "uav",
    "tricycle",
)

ALL_MODES = (
    "full",
    "no_ir",
    "no_depth",
    "rgb_only",
)

MODE_LABELS = {
    "full": "RGB + IR + Depth",
    "no_ir": "RGB + 0 + Depth",
    "no_depth": "RGB + IR + 0",
    "rgb_only": "RGB + 0 + 0",
}

# Previous manual square-parity result.
#
# This is only a sanity reference.
# The present script independently calculates competition-style AP.
PREVIOUS_SQUARE_FULL_MAP50_95 = 0.374356

PARITY_WARN_TOLERANCE = 0.003

IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# Utility
# ============================================================

def section(title: str) -> None:

    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def resolve_project_path(
    text: str | None,
    default: Path,
) -> Path:

    if text is None:
        return default

    path = Path(text)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


# ============================================================
# Dataset checks
# ============================================================

def image_map(
    root: Path,
) -> dict[str, Path]:

    result: dict[str, Path] = {}

    for path in root.rglob("*"):

        if not (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_SUFFIXES
        ):
            continue

        stem = path.stem

        if stem in result:

            raise RuntimeError(
                "Duplicate image stem:\n"
                f"  root   = {root}\n"
                f"  stem   = {stem}\n"
                f"  first  = {result[stem]}\n"
                f"  second = {path}"
            )

        result[stem] = path

    return result


def read_split_stems(
    path: Path,
) -> set[str]:

    stems: set[str] = set()

    for raw_line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        line = raw_line.strip()

        if not line:
            continue

        stems.add(
            Path(line).stem
        )

    return stems


def check_data():

    required_dirs = (
        RGB_IMAGE_DIR,
        IR_IMAGE_DIR,
        DEPTH_IMAGE_DIR,
        LABEL_DIR,
    )

    for path in required_dirs:

        if not path.is_dir():

            raise FileNotFoundError(
                f"Required directory not found: {path}"
            )

    rgb_map = image_map(
        RGB_IMAGE_DIR
    )

    ir_map = image_map(
        IR_IMAGE_DIR
    )

    depth_map = image_map(
        DEPTH_IMAGE_DIR
    )

    rgb_stems = set(
        rgb_map
    )

    ir_stems = set(
        ir_map
    )

    depth_stems = set(
        depth_map
    )

    if not (
        rgb_stems
        == ir_stems
        == depth_stems
    ):

        raise RuntimeError(
            "Validation modalities are not exactly aligned.\n"
            f"RGB   = {len(rgb_stems)}\n"
            f"IR    = {len(ir_stems)}\n"
            f"Depth = {len(depth_stems)}\n"
            f"Common= "
            f"{len(rgb_stems & ir_stems & depth_stems)}"
        )

    if len(
        rgb_stems
    ) != EXPECTED_VAL_SAMPLES:

        raise RuntimeError(
            "Unexpected val sample count.\n"
            f"Expected = {EXPECTED_VAL_SAMPLES}\n"
            f"Actual   = {len(rgb_stems)}"
        )

    if VAL_SPLIT.is_file():

        split_stems = read_split_stems(
            VAL_SPLIT
        )

        if split_stems != rgb_stems:

            missing_in_view = sorted(
                split_stems
                - rgb_stems
            )

            extra_in_view = sorted(
                rgb_stems
                - split_stems
            )

            raise RuntimeError(
                "yolo_views/*/val does not match "
                "the fixed val.txt split.\n"
                f"val.txt count   = {len(split_stems)}\n"
                f"view count      = {len(rgb_stems)}\n"
                f"missing in view = {missing_in_view[:20]}\n"
                f"extra in view   = {extra_in_view[:20]}"
            )

    stems = sorted(
        rgb_stems
    )

    return (
        stems,
        rgb_map,
        ir_map,
        depth_map,
    )


# ============================================================
# Multimodal loading
# ============================================================

def load_rgb(
    path: Path,
) -> np.ndarray:

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:

        raise RuntimeError(
            f"Failed to read RGB image: {path}"
        )

    # Training representation:
    # BGR -> RGB
    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )


def load_ir_gray(
    path: Path,
) -> np.ndarray:

    image = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    if image is None:

        raise RuntimeError(
            f"Failed to read IR image: {path}"
        )

    if image.ndim == 2:

        gray = image

    elif (
        image.ndim == 3
        and image.shape[2] == 3
    ):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    elif (
        image.ndim == 3
        and image.shape[2] == 4
    ):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2GRAY,
        )

    else:

        raise RuntimeError(
            "Unsupported IR shape:\n"
            f"  path  = {path}\n"
            f"  shape = {image.shape}"
        )

    if gray.dtype != np.uint8:

        raise RuntimeError(
            "IR image is expected to be uint8.\n"
            f"  path  = {path}\n"
            f"  dtype = {gray.dtype}"
        )

    return gray


def metric_depth_to_uint8(
    depth: np.ndarray,
) -> np.ndarray:

    depth_f = depth.astype(
        np.float32
    )

    gray = np.zeros(
        depth.shape,
        dtype=np.uint8,
    )

    valid = (
        (depth_f >= 300.0)
        & (depth_f <= 20000.0)
    )

    values = (
        1.0
        + (
            20000.0
            - depth_f[valid]
        )
        * 254.0
        / (
            20000.0
            - 300.0
        )
    )

    gray[valid] = np.clip(
        np.rint(
            values
        ),
        1,
        255,
    ).astype(
        np.uint8
    )

    return gray


def load_depth_gray(
    path: Path,
) -> np.ndarray:

    depth = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    if depth is None:

        raise RuntimeError(
            f"Failed to read Depth image: {path}"
        )

    # Existing official uint8 visualization:
    # preserve near-bright / far-dark.
    if depth.dtype == np.uint8:

        if depth.ndim == 2:

            return depth

        if (
            depth.ndim == 3
            and depth.shape[2] == 3
        ):

            return cv2.cvtColor(
                depth,
                cv2.COLOR_BGR2GRAY,
            )

        if (
            depth.ndim == 3
            and depth.shape[2] == 4
        ):

            return cv2.cvtColor(
                depth,
                cv2.COLOR_BGRA2GRAY,
            )

        raise RuntimeError(
            "Unsupported uint8 Depth shape:\n"
            f"  path  = {path}\n"
            f"  shape = {depth.shape}"
        )

    # Metric Depth.
    if depth.ndim == 3:

        depth = depth[
            ...,
            0,
        ]

    if depth.ndim != 2:

        raise RuntimeError(
            "Unsupported metric Depth shape:\n"
            f"  path  = {path}\n"
            f"  shape = {depth.shape}"
        )

    return metric_depth_to_uint8(
        depth
    )


def load_5ch(
    rgb_path: Path,
    ir_path: Path,
    depth_path: Path,
):

    rgb = load_rgb(
        rgb_path
    )

    ir = load_ir_gray(
        ir_path
    )

    depth = load_depth_gray(
        depth_path
    )

    h, w = rgb.shape[:2]

    if ir.shape != (
        h,
        w,
    ):

        raise RuntimeError(
            "RGB/IR spatial mismatch:\n"
            f"  RGB = {rgb_path}: {rgb.shape}\n"
            f"  IR  = {ir_path}: {ir.shape}"
        )

    if depth.shape != (
        h,
        w,
    ):

        raise RuntimeError(
            "RGB/Depth spatial mismatch:\n"
            f"  RGB   = {rgb_path}: {rgb.shape}\n"
            f"  Depth = {depth_path}: {depth.shape}"
        )

    image = np.concatenate(
        [
            rgb,
            ir[
                ...,
                None,
            ],
            depth[
                ...,
                None,
            ],
        ],
        axis=2,
    )

    if (
        image.dtype != np.uint8
        or image.shape[2]
        != EXPECTED_CHANNELS
    ):

        raise AssertionError(
            "Invalid 5-channel image:\n"
            f"  shape = {image.shape}\n"
            f"  dtype = {image.dtype}"
        )

    return (
        image,
        (
            h,
            w,
        ),
    )


# ============================================================
# 5-channel square LetterBox
# ============================================================

def letterbox_5ch(
    image: np.ndarray,
    new_shape=(
        IMAGE_SIZE,
        IMAGE_SIZE,
    ),
    scaleup=False,
    padding_value=114,
):

    h0, w0 = image.shape[:2]

    new_h, new_w = new_shape

    r = min(
        new_h / h0,
        new_w / w0,
    )

    if not scaleup:

        r = min(
            r,
            1.0,
        )

    resized_w = int(
        round(
            w0 * r
        )
    )

    resized_h = int(
        round(
            h0 * r
        )
    )

    if (
        resized_w != w0
        or resized_h != h0
    ):

        resized = cv2.resize(
            image,
            (
                resized_w,
                resized_h,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

    else:

        resized = image

    dw = (
        new_w
        - resized_w
    )

    dh = (
        new_h
        - resized_h
    )

    left = int(
        round(
            dw / 2.0
            - 0.1
        )
    )

    top = int(
        round(
            dh / 2.0
            - 0.1
        )
    )

    output = np.full(
        (
            new_h,
            new_w,
            EXPECTED_CHANNELS,
        ),
        padding_value,
        dtype=np.uint8,
    )

    output[
        top:
        top + resized_h,
        left:
        left + resized_w,
    ] = resized

    scale_x = (
        resized_w
        / float(w0)
    )

    scale_y = (
        resized_h
        / float(h0)
    )

    return (
        output,
        scale_x,
        scale_y,
        left,
        top,
    )


# ============================================================
# Ground-truth / box handling
# ============================================================

def restore_xyxy_to_original(
    boxes: torch.Tensor,
    original_shape,
    scale_x,
    scale_y,
    left,
    top,
):

    h, w = original_shape

    boxes = boxes.clone()

    boxes[
        :,
        [
            0,
            2,
        ],
    ] -= float(
        left
    )

    boxes[
        :,
        [
            1,
            3,
        ],
    ] -= float(
        top
    )

    boxes[
        :,
        [
            0,
            2,
        ],
    ] /= float(
        scale_x
    )

    boxes[
        :,
        [
            1,
            3,
        ],
    ] /= float(
        scale_y
    )

    boxes[
        :,
        0,
    ].clamp_(
        0,
        w,
    )

    boxes[
        :,
        2,
    ].clamp_(
        0,
        w,
    )

    boxes[
        :,
        1,
    ].clamp_(
        0,
        h,
    )

    boxes[
        :,
        3,
    ].clamp_(
        0,
        h,
    )

    return boxes


def read_ground_truth(
    stem,
    original_shape,
):

    label_path = (
        LABEL_DIR
        / f"{stem}.txt"
    )

    if not label_path.exists():

        return np.zeros(
            (
                0,
                5,
            ),
            dtype=np.float64,
        )

    h, w = original_shape

    rows = []

    for (
        line_number,
        raw_line,
    ) in enumerate(
        label_path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):

        line = raw_line.strip()

        if not line:
            continue

        fields = line.split()

        if len(fields) < 5:

            raise RuntimeError(
                "Invalid YOLO label line:\n"
                f"  file = {label_path}\n"
                f"  line = {line_number}\n"
                f"  text = {line}"
            )

        class_id = int(
            float(
                fields[0]
            )
        )

        (
            xc,
            yc,
            bw,
            bh,
        ) = map(
            float,
            fields[
                1:
                5
            ],
        )

        if not (
            0
            <= class_id
            < EXPECTED_CLASSES
        ):

            raise RuntimeError(
                f"Illegal class id in "
                f"{label_path}: {class_id}"
            )

        x1 = (
            xc
            - bw / 2.0
        ) * w

        y1 = (
            yc
            - bh / 2.0
        ) * h

        x2 = (
            xc
            + bw / 2.0
        ) * w

        y2 = (
            yc
            + bh / 2.0
        ) * h

        # Defensive clipping for the few known slightly
        # out-of-range official labels.
        x1 = min(
            max(
                x1,
                0.0,
            ),
            float(w),
        )

        x2 = min(
            max(
                x2,
                0.0,
            ),
            float(w),
        )

        y1 = min(
            max(
                y1,
                0.0,
            ),
            float(h),
        )

        y2 = min(
            max(
                y2,
                0.0,
            ),
            float(h),
        )

        if (
            x2 <= x1
            or y2 <= y1
        ):

            continue

        rows.append(
            (
                float(class_id),
                x1,
                y1,
                x2,
                y2,
            )
        )

    if not rows:

        return np.zeros(
            (
                0,
                5,
            ),
            dtype=np.float64,
        )

    return np.asarray(
        rows,
        dtype=np.float64,
    )


# ============================================================
# Model
# ============================================================

def load_model(
    model_path,
    device,
):

    if not model_path.is_file():

        raise FileNotFoundError(
            "Exp04 checkpoint not found:\n"
            f"  {model_path}"
        )

    wrapper = YOLO(
        str(
            model_path
        )
    )

    model = wrapper.model

    first_conv = (
        model
        .model[0]
        .conv
    )

    detect_head = (
        model
        .model[-1]
    )

    if (
        first_conv.in_channels
        != EXPECTED_CHANNELS
    ):

        raise RuntimeError(
            "Checkpoint is not a 5-channel "
            "Exp04 model.\n"
            f"first_conv.in_channels = "
            f"{first_conv.in_channels}"
        )

    if (
        getattr(
            detect_head,
            "nc",
            None,
        )
        != EXPECTED_CLASSES
    ):

        raise RuntimeError(
            "Checkpoint is not a 12-class model.\n"
            f"detect.nc = "
            f"{getattr(detect_head, 'nc', None)}"
        )

    model = (
        model
        .to(device)
        .float()
        .eval()
    )

    return (
        wrapper,
        model,
    )


# ============================================================
# Modality ablation
# ============================================================

def apply_ablation(
    batch,
    mode,
):

    if mode == "full":

        return batch

    output = batch.clone()

    if mode == "no_ir":

        # Channel 3 = IR
        output[
            :,
            3,
            :,
            :,
        ] = 0.0

    elif mode == "no_depth":

        # Channel 4 = Depth
        output[
            :,
            4,
            :,
            :,
        ] = 0.0

    elif mode == "rgb_only":

        # Remove both auxiliary modalities.
        output[
            :,
            3:
            5,
            :,
            :,
        ] = 0.0

    else:

        raise ValueError(
            f"Unknown ablation mode: {mode}"
        )

    return output


# ============================================================
# Inference
# ============================================================

@torch.inference_mode()
def predict_batch(
    model,
    batch,
):

    raw = model(
        batch
    )

    if isinstance(
        raw,
        (
            tuple,
            list,
        ),
    ):

        prediction = raw[0]

    elif torch.is_tensor(
        raw
    ):

        prediction = raw

    else:

        raise RuntimeError(
            "Unsupported model inference output type: "
            f"{type(raw).__name__}"
        )

    if prediction.ndim != 3:

        raise RuntimeError(
            "Unexpected YOLO prediction shape: "
            f"{tuple(prediction.shape)}"
        )

    detections = []

    # Deliberately NMS one image at a time.
    for image_index in range(
        prediction.shape[0]
    ):

        pred_one = prediction[
            image_index:
            image_index + 1
        ]

        det_one = non_max_suppression(
            pred_one,
            conf_thres=CONF_THRESHOLD,
            iou_thres=IOU_THRESHOLD,
            classes=None,
            agnostic=False,
            multi_label=True,
            max_det=MAX_DET,
            nc=EXPECTED_CLASSES,
            max_time_img=10.0,
            max_nms=30000,
        )[0]

        detections.append(
            det_one
        )

    return detections


def prepare_base_batch(
    stems,
    rgb_map,
    ir_map,
    depth_map,
    device,
):

    tensors = []
    metas = []

    for stem in stems:

        (
            image,
            original_shape,
        ) = load_5ch(
            rgb_map[stem],
            ir_map[stem],
            depth_map[stem],
        )

        (
            image,
            scale_x,
            scale_y,
            left,
            top,
        ) = letterbox_5ch(
            image
        )

        tensor = torch.from_numpy(
            np.ascontiguousarray(
                image.transpose(
                    2,
                    0,
                    1,
                )
            )
        )

        tensors.append(
            tensor
        )

        metas.append(
            {
                "stem": stem,
                "original_shape": original_shape,
                "scale_x": scale_x,
                "scale_y": scale_y,
                "left": left,
                "top": top,
            }
        )

    batch = torch.stack(
        tensors,
        dim=0,
    )

    batch = (
        batch
        .to(
            device,
            non_blocking=True,
        )
        .float()
        / 255.0
    )

    return (
        batch,
        metas,
    )


# ============================================================
# Competition metric
# ============================================================

def box_iou_one_to_many(
    box,
    boxes,
):

    if boxes.size == 0:

        return np.zeros(
            (
                0,
            ),
            dtype=np.float64,
        )

    x1 = np.maximum(
        box[0],
        boxes[
            :,
            0,
        ],
    )

    y1 = np.maximum(
        box[1],
        boxes[
            :,
            1,
        ],
    )

    x2 = np.minimum(
        box[2],
        boxes[
            :,
            2,
        ],
    )

    y2 = np.minimum(
        box[3],
        boxes[
            :,
            3,
        ],
    )

    inter_w = np.maximum(
        0.0,
        x2 - x1,
    )

    inter_h = np.maximum(
        0.0,
        y2 - y1,
    )

    intersection = (
        inter_w
        * inter_h
    )

    area_box = (
        max(
            0.0,
            box[2] - box[0],
        )
        * max(
            0.0,
            box[3] - box[1],
        )
    )

    area_boxes = (
        np.maximum(
            0.0,
            boxes[
                :,
                2,
            ]
            - boxes[
                :,
                0,
            ],
        )
        * np.maximum(
            0.0,
            boxes[
                :,
                3,
            ]
            - boxes[
                :,
                1,
            ],
        )
    )

    union = (
        area_box
        + area_boxes
        - intersection
    )

    return np.divide(
        intersection,
        union,
        out=np.zeros_like(
            intersection,
            dtype=np.float64,
        ),
        where=union > 0.0,
    )


def collect_class_data(
    predictions,
    ground_truths,
    stems,
    class_id,
):

    gt_by_image = {}
    pred_records = []

    n_gt = 0

    for stem in stems:

        gt = ground_truths[
            stem
        ]

        if gt.size:

            gt_boxes = gt[
                gt[
                    :,
                    0,
                ]
                == class_id,
                1:
                5,
            ]

        else:

            gt_boxes = np.zeros(
                (
                    0,
                    4,
                ),
                dtype=np.float64,
            )

        gt_by_image[
            stem
        ] = gt_boxes

        n_gt += len(
            gt_boxes
        )

        pred = predictions[
            stem
        ]

        if not pred.size:
            continue

        selected = pred[
            pred[
                :,
                5,
            ].astype(
                np.int64
            )
            == class_id
        ]

        for row in selected:

            pred_records.append(
                (
                    float(
                        row[4]
                    ),
                    stem,
                    row[
                        0:
                        4
                    ].astype(
                        np.float64,
                        copy=False,
                    ),
                )
            )

    pred_records.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    return (
        gt_by_image,
        pred_records,
        n_gt,
    )


def match_predictions(
    gt_by_image,
    pred_records,
    iou_threshold,
):

    matched = {
        stem:
        np.zeros(
            len(boxes),
            dtype=bool,
        )
        for (
            stem,
            boxes,
        ) in gt_by_image.items()
    }

    scores = np.zeros(
        len(pred_records),
        dtype=np.float64,
    )

    tp = np.zeros(
        len(pred_records),
        dtype=np.float64,
    )

    fp = np.zeros(
        len(pred_records),
        dtype=np.float64,
    )

    for (
        index,
        (
            confidence,
            stem,
            pred_box,
        ),
    ) in enumerate(
        pred_records
    ):

        scores[
            index
        ] = confidence

        gt_boxes = gt_by_image[
            stem
        ]

        unmatched_indices = np.flatnonzero(
            ~matched[
                stem
            ]
        )

        if (
            unmatched_indices.size
            == 0
        ):

            fp[
                index
            ] = 1.0

            continue

        unmatched_boxes = gt_boxes[
            unmatched_indices
        ]

        ious = box_iou_one_to_many(
            pred_box,
            unmatched_boxes,
        )

        best_local = int(
            np.argmax(
                ious
            )
        )

        best_iou = float(
            ious[
                best_local
            ]
        )

        if (
            best_iou
            >= iou_threshold
        ):

            gt_index = int(
                unmatched_indices[
                    best_local
                ]
            )

            matched[
                stem
            ][
                gt_index
            ] = True

            tp[
                index
            ] = 1.0

        else:

            fp[
                index
            ] = 1.0

    return (
        scores,
        tp,
        fp,
    )


def ap_101_point(
    tp,
    fp,
    n_gt,
):

    if n_gt <= 0:

        return math.nan

    if tp.size == 0:

        return 0.0

    tp_cum = np.cumsum(
        tp
    )

    fp_cum = np.cumsum(
        fp
    )

    recall = (
        tp_cum
        / float(n_gt)
    )

    precision = np.divide(
        tp_cum,
        tp_cum + fp_cum,
        out=np.zeros_like(
            tp_cum,
            dtype=np.float64,
        ),
        where=(
            tp_cum
            + fp_cum
            > 0.0
        ),
    )

    samples = np.linspace(
        0.0,
        1.0,
        101,
    )

    interpolated = np.zeros(
        101,
        dtype=np.float64,
    )

    for (
        index,
        recall_sample,
    ) in enumerate(
        samples
    ):

        eligible = precision[
            recall
            >= recall_sample
        ]

        if eligible.size:

            interpolated[
                index
            ] = float(
                eligible.max()
            )

    return float(
        interpolated.mean()
    )


def evaluate_mode(
    predictions,
    ground_truths,
    stems,
):

    ap_matrix = np.full(
        (
            EXPECTED_CLASSES,
            len(
                IOU_THRESHOLDS
            ),
        ),
        np.nan,
        dtype=np.float64,
    )

    gt_counts = np.zeros(
        EXPECTED_CLASSES,
        dtype=np.int64,
    )

    global_iou50_status = []

    for class_id in range(
        EXPECTED_CLASSES
    ):

        (
            gt_by_image,
            pred_records,
            n_gt,
        ) = collect_class_data(
            predictions,
            ground_truths,
            stems,
            class_id,
        )

        gt_counts[
            class_id
        ] = n_gt

        for (
            threshold_index,
            threshold,
        ) in enumerate(
            IOU_THRESHOLDS
        ):

            (
                scores,
                tp,
                fp,
            ) = match_predictions(
                gt_by_image,
                pred_records,
                float(
                    threshold
                ),
            )

            ap_matrix[
                class_id,
                threshold_index,
            ] = ap_101_point(
                tp,
                fp,
                n_gt,
            )

            # Used only for a useful P/R summary.
            if threshold_index == 0:

                for (
                    score,
                    is_tp,
                ) in zip(
                    scores,
                    tp,
                ):

                    global_iou50_status.append(
                        (
                            float(
                                score
                            ),
                            float(
                                is_tp
                            ),
                        )
                    )

    valid_classes = (
        gt_counts > 0
    )

    if not valid_classes.any():

        raise RuntimeError(
            "No ground-truth boxes were found in val400."
        )

    idx50 = int(
        np.argmin(
            np.abs(
                IOU_THRESHOLDS
                - 0.50
            )
        )
    )

    idx75 = int(
        np.argmin(
            np.abs(
                IOU_THRESHOLDS
                - 0.75
            )
        )
    )

    map50 = float(
        np.nanmean(
            ap_matrix[
                valid_classes,
                idx50,
            ]
        )
    )

    map75 = float(
        np.nanmean(
            ap_matrix[
                valid_classes,
                idx75,
            ]
        )
    )

    map50_95 = float(
        np.nanmean(
            ap_matrix[
                valid_classes,
                :,
            ]
        )
    )

    # --------------------------------------------------------
    # P/R summary
    #
    # The competition score is mAP.
    # For P/R we report micro P/R at the confidence point that
    # maximizes F1 at IoU=0.50.
    #
    # This is deliberately labelled separately from Ultralytics'
    # console P/R fields.
    # --------------------------------------------------------

    global_iou50_status.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    if global_iou50_status:

        scores = np.asarray(
            [
                x[0]
                for x
                in global_iou50_status
            ],
            dtype=np.float64,
        )

        tp = np.asarray(
            [
                x[1]
                for x
                in global_iou50_status
            ],
            dtype=np.float64,
        )

        fp = (
            1.0
            - tp
        )

        tp_cum = np.cumsum(
            tp
        )

        fp_cum = np.cumsum(
            fp
        )

        precision = np.divide(
            tp_cum,
            tp_cum + fp_cum,
            out=np.zeros_like(
                tp_cum
            ),
            where=(
                tp_cum
                + fp_cum
                > 0.0
            ),
        )

        total_gt = int(
            gt_counts.sum()
        )

        recall = (
            tp_cum
            / float(
                max(
                    total_gt,
                    1,
                )
            )
        )

        f1 = np.divide(
            2.0
            * precision
            * recall,
            precision
            + recall,
            out=np.zeros_like(
                precision
            ),
            where=(
                precision
                + recall
                > 0.0
            ),
        )

        best_index = int(
            np.argmax(
                f1
            )
        )

        p_best = float(
            precision[
                best_index
            ]
        )

        r_best = float(
            recall[
                best_index
            ]
        )

        best_conf = float(
            scores[
                best_index
            ]
        )

    else:

        p_best = 0.0
        r_best = 0.0
        best_conf = 1.0

    prediction_boxes = sum(
        len(
            predictions[
                stem
            ]
        )
        for stem
        in stems
    )

    empty_images = sum(
        len(
            predictions[
                stem
            ]
        )
        == 0
        for stem
        in stems
    )

    per_class = []

    for (
        class_id,
        class_name,
    ) in enumerate(
        CLASS_NAMES
    ):

        per_class.append(
            {
                "class_id":
                class_id,

                "class_name":
                class_name,

                "gt_count":
                int(
                    gt_counts[
                        class_id
                    ]
                ),

                "ap50":
                float(
                    ap_matrix[
                        class_id,
                        idx50,
                    ]
                ),

                "ap75":
                float(
                    ap_matrix[
                        class_id,
                        idx75,
                    ]
                ),

                "ap50_95":
                float(
                    np.nanmean(
                        ap_matrix[
                            class_id,
                            :,
                        ]
                    )
                ),
            }
        )

    return {
        "p50_best_f1":
        p_best,

        "r50_best_f1":
        r_best,

        "best_conf50":
        best_conf,

        "map50":
        map50,

        "map75":
        map75,

        "map50_95":
        map50_95,

        "prediction_boxes":
        int(
            prediction_boxes
        ),

        "empty_images":
        int(
            empty_images
        ),

        "per_class":
        per_class,
    }


# ============================================================
# Results
# ============================================================

def safe_float(
    value,
):

    value = float(
        value
    )

    if math.isnan(
        value
    ):

        return None

    return value


def write_outputs(
    output_dir,
    model_path,
    modes,
    results,
    elapsed_seconds,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_result = results.get(
        "full"
    )

    # --------------------------------------------------------
    # summary.csv
    # --------------------------------------------------------

    summary_csv = (
        output_dir
        / "summary.csv"
    )

    with summary_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "input",
                "p50_best_f1",
                "r50_best_f1",
                "best_conf50",
                "map50",
                "map75",
                "map50_95",
                "drop_map50_95_from_full",
                "prediction_boxes",
                "empty_images",
            ],
        )

        writer.writeheader()

        for mode in modes:

            item = results[
                mode
            ]

            drop = ""

            if full_result is not None:

                drop = (
                    full_result[
                        "map50_95"
                    ]
                    - item[
                        "map50_95"
                    ]
                )

            writer.writerow(
                {
                    "mode":
                    mode,

                    "input":
                    MODE_LABELS[
                        mode
                    ],

                    "p50_best_f1":
                    f"{item['p50_best_f1']:.9f}",

                    "r50_best_f1":
                    f"{item['r50_best_f1']:.9f}",

                    "best_conf50":
                    f"{item['best_conf50']:.9f}",

                    "map50":
                    f"{item['map50']:.9f}",

                    "map75":
                    f"{item['map75']:.9f}",

                    "map50_95":
                    f"{item['map50_95']:.9f}",

                    "drop_map50_95_from_full":
                    (
                        ""
                        if drop == ""
                        else f"{drop:.9f}"
                    ),

                    "prediction_boxes":
                    item[
                        "prediction_boxes"
                    ],

                    "empty_images":
                    item[
                        "empty_images"
                    ],
                }
            )

    # --------------------------------------------------------
    # per_class.csv
    # --------------------------------------------------------

    per_class_csv = (
        output_dir
        / "per_class.csv"
    )

    full_class_map = {}

    if full_result is not None:

        full_class_map = {
            item[
                "class_id"
            ]:
            item

            for item
            in full_result[
                "per_class"
            ]
        }

    with per_class_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "class_id",
                "class_name",
                "gt_count",
                "ap50",
                "ap75",
                "ap50_95",
                "drop_ap50_95_from_full",
            ],
        )

        writer.writeheader()

        for mode in modes:

            for item in results[
                mode
            ][
                "per_class"
            ]:

                drop = ""

                if full_class_map:

                    drop = (
                        full_class_map[
                            item[
                                "class_id"
                            ]
                        ][
                            "ap50_95"
                        ]
                        - item[
                            "ap50_95"
                        ]
                    )

                writer.writerow(
                    {
                        "mode":
                        mode,

                        "class_id":
                        item[
                            "class_id"
                        ],

                        "class_name":
                        item[
                            "class_name"
                        ],

                        "gt_count":
                        item[
                            "gt_count"
                        ],

                        "ap50":
                        f"{item['ap50']:.9f}",

                        "ap75":
                        f"{item['ap75']:.9f}",

                        "ap50_95":
                        f"{item['ap50_95']:.9f}",

                        "drop_ap50_95_from_full":
                        (
                            ""
                            if drop == ""
                            else f"{drop:.9f}"
                        ),
                    }
                )

    # --------------------------------------------------------
    # results.json
    # --------------------------------------------------------

    serializable_results = {}

    for mode in modes:

        item = results[
            mode
        ]

        serializable_results[
            mode
        ] = {
            key:
            value

            for (
                key,
                value,
            ) in item.items()

            if key
            != "per_class"
        }

        serializable_results[
            mode
        ][
            "per_class"
        ] = [
            {
                key:
                (
                    safe_float(
                        value
                    )
                    if key
                    in {
                        "ap50",
                        "ap75",
                        "ap50_95",
                    }
                    else value
                )

                for (
                    key,
                    value,
                ) in row.items()
            }

            for row
            in item[
                "per_class"
            ]
        ]

    json_path = (
        output_dir
        / "results.json"
    )

    payload = {
        "experiment":
        "Exp04 val400 test-time modality ablation",

        "model":
        str(
            model_path
        ),

        "protocol":
        {
            "imgsz":
            IMAGE_SIZE,

            "square_letterbox":
            True,

            "scaleup":
            False,

            "conf":
            CONF_THRESHOLD,

            "nms_iou":
            IOU_THRESHOLD,

            "max_det":
            MAX_DET,

            "multi_label":
            True,

            "channels":
            [
                "R",
                "G",
                "B",
                "IR",
                "Depth",
            ],

            "iou_thresholds":
            [
                float(x)
                for x
                in IOU_THRESHOLDS
            ],

            "ap_interpolation_points":
            101,
        },

        "elapsed_seconds":
        float(
            elapsed_seconds
        ),

        "previous_square_full_map50_95":
        PREVIOUS_SQUARE_FULL_MAP50_95,

        "results":
        serializable_results,
    }

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # report.md
    # --------------------------------------------------------

    report_path = (
        output_dir
        / "report.md"
    )

    lines = [
        "# Exp04 Modality Ablation Val400",
        "",
        f"- Model: `{model_path}`",
        f"- Samples: {EXPECTED_VAL_SAMPLES}",
        "- Input: `[R,G,B,IR,Depth]`",
        (
            f"- Inference: square "
            f"`{IMAGE_SIZE}x{IMAGE_SIZE}`, "
            f"FP32, conf={CONF_THRESHOLD}, "
            f"NMS IoU={IOU_THRESHOLD}, "
            f"max_det={MAX_DET}, "
            f"multi_label=True"
        ),
        (
            "- Metric: 101-point AP, "
            "IoU 0.50:0.05:0.95"
        ),
        (
            "- P/R: micro P/R at the "
            "IoU=0.50 confidence point "
            "with maximum F1; not the "
            "Ultralytics displayed P/R field"
        ),
        "",
        "## Summary",
        "",
        (
            "| Mode | Input | P@bestF1 | R@bestF1 | "
            "mAP50 | mAP75 | mAP50-95 | "
            "Drop from Full |"
        ),
        (
            "|---|---|---:|---:|---:|---:|---:|---:|"
        ),
    ]

    for mode in modes:

        item = results[
            mode
        ]

        drop_text = "-"

        if full_result is not None:

            drop_text = (
                f"{full_result['map50_95'] - item['map50_95']:+.6f}"
            )

        lines.append(
            f"| {mode} "
            f"| {MODE_LABELS[mode]} "
            f"| {item['p50_best_f1']:.6f} "
            f"| {item['r50_best_f1']:.6f} "
            f"| {item['map50']:.6f} "
            f"| {item['map75']:.6f} "
            f"| {item['map50_95']:.6f} "
            f"| {drop_text} |"
        )

    if full_result is not None:

        lines.extend(
            [
                "",
                "## Ablation interpretation",
                "",
            ]
        )

        if "no_ir" in results:

            ir_drop = (
                full_result[
                    "map50_95"
                ]
                - results[
                    "no_ir"
                ][
                    "map50_95"
                ]
            )

            lines.append(
                "- IR ablation drop "
                "(Full - No IR): "
                f"`{ir_drop:+.6f}`"
            )

        if "no_depth" in results:

            depth_drop = (
                full_result[
                    "map50_95"
                ]
                - results[
                    "no_depth"
                ][
                    "map50_95"
                ]
            )

            lines.append(
                "- Depth ablation drop "
                "(Full - No Depth): "
                f"`{depth_drop:+.6f}`"
            )

        if "rgb_only" in results:

            both_drop = (
                full_result[
                    "map50_95"
                ]
                - results[
                    "rgb_only"
                ][
                    "map50_95"
                ]
            )

            lines.append(
                "- Auxiliary-modality drop "
                "(Full - RGB only): "
                f"`{both_drop:+.6f}`"
            )

        lines.extend(
            [
                "",
                (
                    "Positive drop means removing that modality "
                    "reduced val400 mAP50-95 for this trained "
                    "Exp04 checkpoint."
                ),
                (
                    "Negative drop means the ablated input scored "
                    "higher than Full under this test-time protocol, "
                    "which is evidence of possible negative transfer "
                    "or poor modality use, not proof that retraining "
                    "without that modality will be better."
                ),
            ]
        )

    report_path.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Saved:")
    print(
        " ",
        summary_csv,
    )
    print(
        " ",
        per_class_csv,
    )
    print(
        " ",
        json_path,
    )
    print(
        " ",
        report_path,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Exp04 val400 test-time "
            "modality ablation."
        )
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help=(
            "Checkpoint path. Relative paths are resolved "
            "from PROJECT_ROOT. Default: "
            "runs/exp04_rgbid_early5_yolo11s_960/"
            "weights/best.pt"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory. Relative paths are "
            "resolved from PROJECT_ROOT."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help=(
            "CUDA index such as 0, cuda:0, "
            "or cpu."
        ),
    )

    parser.add_argument(
        "--modes",
        nargs="+",
        choices=ALL_MODES,
        default=list(
            ALL_MODES
        ),
        help=(
            "Ablation modes to run. "
            "Default runs all four."
        ),
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Check paths, split alignment, "
            "and model shape without inference."
        ),
    )

    args = parser.parse_args()

    if args.batch_size <= 0:

        raise ValueError(
            "--batch-size must be > 0"
        )

    # Remove duplicate modes while keeping order.
    modes = []

    for mode in args.modes:

        if mode not in modes:

            modes.append(
                mode
            )

    model_path = resolve_project_path(
        args.weights,
        DEFAULT_MODEL_PATH,
    )

    output_dir = resolve_project_path(
        args.output_dir,
        DEFAULT_OUTPUT_DIR,
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if args.device.lower() == "cpu":

        device = torch.device(
            "cpu"
        )

    else:

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA is unavailable. "
                "Use --device cpu only if you "
                "intentionally want CPU inference."
            )

        device_text = args.device

        if device_text.startswith(
            "cuda:"
        ):

            device = torch.device(
                device_text
            )

        else:

            device = torch.device(
                f"cuda:{device_text}"
            )

    section(
        "Exp04 Modality Ablation Val400"
    )

    print(
        "Project root :",
        PROJECT_ROOT,
    )

    print(
        "Model        :",
        model_path,
    )

    print(
        "Output       :",
        output_dir,
    )

    print(
        "Device       :",
        device,
    )

    print(
        "Batch size   :",
        args.batch_size,
    )

    print(
        "Modes        :",
        ", ".join(
            modes
        ),
    )

    print(
        "Protocol     : "
        "square 960 / "
        "conf 0.001 / "
        "IoU 0.70 / "
        "max_det 100 / "
        "multi_label=True"
    )

    # --------------------------------------------------------
    # Fixed val400
    # --------------------------------------------------------

    section(
        "Check fixed val400"
    )

    (
        stems,
        rgb_map,
        ir_map,
        depth_map,
    ) = check_data()

    print(
        "RGB val      :",
        RGB_IMAGE_DIR,
    )

    print(
        "IR val       :",
        IR_IMAGE_DIR,
    )

    print(
        "Depth val    :",
        DEPTH_IMAGE_DIR,
    )

    print(
        "Labels       :",
        LABEL_DIR,
    )

    print(
        "Val samples  :",
        len(
            stems
        ),
    )

    print(
        "First stem   :",
        stems[0],
    )

    print(
        "Last stem    :",
        stems[-1],
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    section(
        "Load Exp04 best.pt"
    )

    (
        _,
        model,
    ) = load_model(
        model_path,
        device,
    )

    first_conv = (
        model
        .model[0]
        .conv
    )

    print(
        "First conv   :",
        first_conv,
    )

    print(
        "Input ch     :",
        first_conv.in_channels,
    )

    print(
        "Classes      :",
        getattr(
            model.model[-1],
            "nc",
            None,
        ),
    )

    if args.check_only:

        section(
            "CHECK ONLY"
        )

        print(
            "STATUS = PASS"
        )

        return

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    predictions = {
        mode: {}
        for mode
        in modes
    }

    ground_truths = {}

    total_batches = math.ceil(
        len(stems)
        / args.batch_size
    )

    start_time = time.time()

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    section(
        "Run four-way test-time ablation"
    )

    for batch_index in range(
        total_batches
    ):

        start = (
            batch_index
            * args.batch_size
        )

        end = min(
            start
            + args.batch_size,
            len(stems),
        )

        batch_stems = stems[
            start:
            end
        ]

        (
            base_batch,
            metas,
        ) = prepare_base_batch(
            batch_stems,
            rgb_map,
            ir_map,
            depth_map,
            device,
        )

        # GT only needs to be loaded once.
        for meta in metas:

            stem = meta[
                "stem"
            ]

            if stem not in ground_truths:

                ground_truths[
                    stem
                ] = read_ground_truth(
                    stem,
                    meta[
                        "original_shape"
                    ],
                )

        # Reuse the same loaded/preprocessed batch for all modes.
        for mode in modes:

            mode_batch = apply_ablation(
                base_batch,
                mode,
            )

            detections = predict_batch(
                model,
                mode_batch,
            )

            for (
                det,
                meta,
            ) in zip(
                detections,
                metas,
            ):

                stem = meta[
                    "stem"
                ]

                if det.numel() == 0:

                    predictions[
                        mode
                    ][
                        stem
                    ] = np.zeros(
                        (
                            0,
                            6,
                        ),
                        dtype=np.float64,
                    )

                    continue

                boxes = restore_xyxy_to_original(
                    det[
                        :,
                        0:
                        4,
                    ],
                    meta[
                        "original_shape"
                    ],
                    meta[
                        "scale_x"
                    ],
                    meta[
                        "scale_y"
                    ],
                    meta[
                        "left"
                    ],
                    meta[
                        "top"
                    ],
                )

                output = torch.cat(
                    [
                        boxes,
                        det[
                            :,
                            4:
                            6,
                        ],
                    ],
                    dim=1,
                )

                predictions[
                    mode
                ][
                    stem
                ] = (
                    output
                    .detach()
                    .cpu()
                    .double()
                    .numpy()
                )

            if mode_batch is not base_batch:

                del mode_batch

        del base_batch

        if (
            (
                batch_index
                + 1
            )
            % 10
            == 0
            or (
                batch_index
                + 1
            )
            == total_batches
        ):

            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"[{batch_index + 1:3d}/"
                f"{total_batches}] "
                f"{end:3d}/"
                f"{len(stems)} images "
                f"| elapsed "
                f"{elapsed / 60.0:.1f} min"
            )

    if device.type == "cuda":

        torch.cuda.synchronize(
            device
        )

    elapsed_seconds = (
        time.time()
        - start_time
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    section(
        "Evaluate competition mAP"
    )

    results = {}

    for mode in modes:

        result = evaluate_mode(
            predictions[
                mode
            ],
            ground_truths,
            stems,
        )

        results[
            mode
        ] = result

        print()

        print(
            f"{mode:9s} "
            f"| {MODE_LABELS[mode]:17s} "
            f"| P={result['p50_best_f1']:.6f} "
            f"R={result['r50_best_f1']:.6f} "
            f"mAP50={result['map50']:.6f} "
            f"mAP75={result['map75']:.6f} "
            f"mAP50-95={result['map50_95']:.6f}"
        )

    # --------------------------------------------------------
    # Full parity sanity check
    # --------------------------------------------------------

    if "full" in results:

        full_map = results[
            "full"
        ][
            "map50_95"
        ]

        diff = (
            full_map
            - PREVIOUS_SQUARE_FULL_MAP50_95
        )

        print()

        print(
            "Previous square Full reference :",
            f"{PREVIOUS_SQUARE_FULL_MAP50_95:.6f}",
        )

        print(
            "Current  square Full result    :",
            f"{full_map:.6f}",
        )

        print(
            "Difference                     :",
            f"{diff:+.6f}",
        )

        if (
            abs(
                diff
            )
            > PARITY_WARN_TOLERANCE
        ):

            print()

            print(
                "WARNING: Full result differs from the "
                "previous square parity reference by more "
                f"than {PARITY_WARN_TOLERANCE:.3f}."
            )

            print(
                "Do not interpret modality deltas until "
                "preprocessing, checkpoint, NMS, labels, "
                "and metric implementation are checked."
            )

    # --------------------------------------------------------
    # Main ablation numbers
    # --------------------------------------------------------

    section(
        "Ablation drops"
    )

    if "full" in results:

        full_map = results[
            "full"
        ][
            "map50_95"
        ]

        if "no_ir" in results:

            print(
                "IR drop    = Full - No IR    =",
                f"{full_map - results['no_ir']['map50_95']:+.6f}",
            )

        if "no_depth" in results:

            print(
                "Depth drop = Full - No Depth =",
                f"{full_map - results['no_depth']['map50_95']:+.6f}",
            )

        if "rgb_only" in results:

            print(
                "Aux drop   = Full - RGB only =",
                f"{full_map - results['rgb_only']['map50_95']:+.6f}",
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    write_outputs(
        output_dir,
        model_path,
        modes,
        results,
        elapsed_seconds,
    )

    section(
        "FINAL"
    )

    print(
        "This experiment is test-time ablation "
        "of the SAME trained Exp04 best.pt."
    )

    print(
        "Positive Full-minus-ablated mAP means "
        "that removing that modality hurts this "
        "trained model on val400."
    )

    print(
        "Use the per-class CSV to decide whether "
        "Exp05 should gate IR, Depth, or both "
        "more aggressively."
    )

    print()

    print(
        "STATUS = PASS"
    )


if __name__ == "__main__":
    main()

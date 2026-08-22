#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 Exp04
RGB + IR + Depth 5-channel Early Fusion
Test-set inference + submission TXT + ZIP.

Model:
    runs/exp04_rgbid_early5_yolo11s_960/weights/best.pt

Input:
    [R, G, B, IR, Depth]

Submission:
    one TXT per RGB test image

    class_id
    norm_center_x
    norm_center_y
    norm_w
    norm_h
    confidence

    max 100 predictions / image

The script processes test images in small batches and writes results
immediately, so it does NOT keep all 1000 images/results in GPU memory.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    # Compatibility fallback for older locked Ultralytics revisions.
    from ultralytics.utils.ops import non_max_suppression


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

DATASETS_ROOT = (
    PROJECT_ROOT
    / "datasets"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "exp04_rgbid_early5_yolo11s_960"
    / "weights"
    / "best0.pt"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "exp04_rgbid_early5_yolo11s_960"
    / "submission_test"
)

TXT_DIR = (
    OUTPUT_ROOT
    / "txt"
)

ZIP_PATH = (
    OUTPUT_ROOT
    / "exp04_rgbid_early5_yolo11s_960_submission.zip"
)


# ============================================================
# Competition / inference configuration
# ============================================================

IMAGE_SIZE = 960

CONF_THRESHOLD = 0.001

IOU_THRESHOLD = 0.70

MAX_DET = 100

EXPECTED_CLASSES = 12

EXPECTED_CHANNELS = 5

EXPECTED_TEST_SAMPLES = 1000

# Conservative for 8 GB GPU.
# This is manual streaming inference, so batch=4 should be safe.
BATCH_SIZE = 4

DEVICE_INDEX = 0

IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# Logging
# ============================================================

def section(title: str) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# Test-set discovery
# ============================================================

def normalize_name(name: str) -> str:

    return (
        name.lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )


def classify_modality_dir(name: str) -> str | None:

    n = normalize_name(name)

    # Depth first: avoid accidental substring ambiguity.
    if "depth" in n:
        return "depth"

    if (
        "infrared" in n
        or "infra" in n
        or "thermal" in n
        or n == "ir"
    ):
        return "ir"

    if (
        "rgb" in n
        or n in {
            "color",
            "colour",
            "visible",
            "visiblelight",
        }
    ):
        return "rgb"

    return None


def image_map(root: Path) -> dict[str, Path]:
    """
    Recursively index images by filename stem.

    Different modalities may use different extensions, therefore
    matching is performed by stem rather than full filename.
    """

    result: dict[str, Path] = {}

    for path in root.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
        ):

            stem = path.stem

            if stem in result:

                raise RuntimeError(
                    "Duplicate image stem inside modality directory:\n"
                    f"  stem={stem}\n"
                    f"  first={result[stem]}\n"
                    f"  second={path}"
                )

            result[stem] = path

    return result


def scan_triplet_candidates(
    datasets_root: Path,
) -> list[dict]:
    """
    Search for a directory whose immediate children include:

        RGB
        Infrared / IR
        Depth

    Then count exactly aligned image stems.
    """

    candidates = []

    root_depth = len(
        datasets_root.parts
    )

    for current, dirnames, _ in os.walk(
        datasets_root
    ):

        current_path = Path(
            current
        )

        depth = (
            len(current_path.parts)
            - root_depth
        )

        # Do not recursively crawl arbitrarily deep directory trees.
        if depth >= 6:
            dirnames[:] = []
            continue

        child_modalities: dict[str, Path] = {}

        for dirname in dirnames:

            modality = classify_modality_dir(
                dirname
            )

            if modality is None:
                continue

            child_path = (
                current_path
                / dirname
            )

            # If duplicate modality directories exist under one root,
            # this root is ambiguous and will not be selected.
            if modality in child_modalities:
                child_modalities[
                    modality
                ] = None
            else:
                child_modalities[
                    modality
                ] = child_path

        if set(
            child_modalities
        ) != {
            "rgb",
            "ir",
            "depth",
        }:
            continue

        if any(
            path is None
            for path in child_modalities.values()
        ):
            continue

        rgb_map = image_map(
            child_modalities["rgb"]
        )

        ir_map = image_map(
            child_modalities["ir"]
        )

        depth_map = image_map(
            child_modalities["depth"]
        )

        aligned = (
            set(rgb_map)
            & set(ir_map)
            & set(depth_map)
        )

        root_text = (
            str(current_path).lower()
        )

        score = len(
            aligned
        )

        if len(aligned) == EXPECTED_TEST_SAMPLES:
            score += 100000

        if (
            len(rgb_map) == EXPECTED_TEST_SAMPLES
            and len(ir_map) == EXPECTED_TEST_SAMPLES
            and len(depth_map) == EXPECTED_TEST_SAMPLES
        ):
            score += 10000

        if (
            "test" in root_text
            or "测试" in root_text
            or "初赛" in root_text
        ):
            score += 1000

        candidates.append(
            {
                "root": current_path,
                "rgb_dir": child_modalities["rgb"],
                "ir_dir": child_modalities["ir"],
                "depth_dir": child_modalities["depth"],
                "rgb_map": rgb_map,
                "ir_map": ir_map,
                "depth_map": depth_map,
                "aligned": aligned,
                "score": score,
            }
        )

    return candidates


def discover_test_set(
    manual_root: Path | None = None,
):
    """
    Discover the previous 1000-sample test set.

    If --test-root is supplied, only that directory is inspected.
    """

    search_root = (
        manual_root
        if manual_root is not None
        else DATASETS_ROOT
    )

    if not search_root.is_dir():

        raise FileNotFoundError(
            f"Dataset search root does not exist:\n"
            f"  {search_root}"
        )

    candidates = scan_triplet_candidates(
        search_root
    )

    if not candidates:

        raise RuntimeError(
            "No RGB / Infrared / Depth sibling-directory "
            "test set was found under:\n"
            f"  {search_root}\n\n"
            "You can inspect datasets/ and rerun with:\n"
            "  --test-root <relative-or-absolute-test-root>"
        )

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    best = candidates[0]

    # Print candidates for auditability.
    print(
        "Detected multimodal candidates:"
    )

    for index, candidate in enumerate(
        candidates[:10],
        start=1,
    ):

        print(
            f"  [{index}] "
            f"{candidate['root']} | "
            f"RGB={len(candidate['rgb_map'])}, "
            f"IR={len(candidate['ir_map'])}, "
            f"Depth={len(candidate['depth_map'])}, "
            f"aligned={len(candidate['aligned'])}"
        )

    print()

    if (
        len(best["aligned"])
        != EXPECTED_TEST_SAMPLES
    ):

        raise RuntimeError(
            "Best detected test candidate does not contain "
            f"{EXPECTED_TEST_SAMPLES} aligned triplets.\n"
            f"Best root: {best['root']}\n"
            f"Aligned  : {len(best['aligned'])}"
        )

    # Strictly require all three modality sets to correspond to
    # exactly the same 1000 image stems.
    rgb_stems = set(
        best["rgb_map"]
    )

    ir_stems = set(
        best["ir_map"]
    )

    depth_stems = set(
        best["depth_map"]
    )

    if not (
        rgb_stems
        == ir_stems
        == depth_stems
    ):

        raise RuntimeError(
            "Test modalities are not exactly one-to-one aligned.\n"
            f"RGB   : {len(rgb_stems)}\n"
            f"IR    : {len(ir_stems)}\n"
            f"Depth : {len(depth_stems)}\n"
            f"Common: {len(best['aligned'])}"
        )

    stems = sorted(
        best["aligned"]
    )

    return (
        best,
        stems,
    )


# ============================================================
# Modality loading
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
    # BGR -> RGB before concatenating the 5 channels.
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    return image


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
            f"  path={path}\n"
            f"  shape={image.shape}"
        )

    if gray.dtype != np.uint8:

        raise RuntimeError(
            "IR image is expected to be uint8:\n"
            f"  path={path}\n"
            f"  dtype={gray.dtype}"
        )

    return gray


def metric_depth_to_uint8(
    depth: np.ndarray,
) -> np.ndarray:
    """
    Same Depth8 representation used by the project:

        invalid:
            d < 300 mm
            d > 20000 mm
            -> 0

        valid:
            300 mm   -> 255
            20000 mm -> 1

    near = bright
    far  = dark
    """

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

    # --------------------------------------------------------
    # Official uint8 JPG / PNG visualization.
    #
    # The three channels contain effectively the same grayscale
    # information. Preserve its existing near-bright/far-dark
    # distribution.
    # --------------------------------------------------------

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
            f"  path={path}\n"
            f"  shape={depth.shape}"
        )

    # --------------------------------------------------------
    # Metric uint16 Depth.
    # --------------------------------------------------------

    if depth.ndim == 3:

        # Metric Depth should normally be one channel.
        # If stored redundantly, use the first channel.
        depth = depth[..., 0]

    if depth.ndim != 2:

        raise RuntimeError(
            "Unsupported metric Depth shape:\n"
            f"  path={path}\n"
            f"  shape={depth.shape}"
        )

    return metric_depth_to_uint8(
        depth
    )


def load_5ch(
    rgb_path: Path,
    ir_path: Path,
    depth_path: Path,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Build raw:
        H x W x 5 uint8

    exact channel order:
        [R, G, B, IR, Depth]
    """

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
            f"  RGB={rgb_path}: {rgb.shape}\n"
            f"  IR ={ir_path}: {ir.shape}"
        )

    if depth.shape != (
        h,
        w,
    ):

        raise RuntimeError(
            "RGB/Depth spatial mismatch:\n"
            f"  RGB  ={rgb_path}: {rgb.shape}\n"
            f"  Depth={depth_path}: {depth.shape}"
        )

    image = np.concatenate(
        [
            rgb,
            ir[..., None],
            depth[..., None],
        ],
        axis=2,
    )

    if (
        image.dtype != np.uint8
        or image.shape[2] != EXPECTED_CHANNELS
    ):

        raise AssertionError(
            "Invalid 5-channel image:\n"
            f"  shape={image.shape}\n"
            f"  dtype={image.dtype}"
        )

    return (
        image,
        (
            h,
            w,
        ),
    )


# ============================================================
# 5-channel LetterBox
# ============================================================

def letterbox_5ch(
    image: np.ndarray,
    new_shape: tuple[int, int] = (
        IMAGE_SIZE,
        IMAGE_SIZE,
    ),
    scaleup: bool = False,
    padding_value: int = 114,
):
    """
    Ultralytics-style centered LetterBox, extended explicitly
    to all 5 channels.

    scaleup=False matches validation-style preprocessing:
    small images are not artificially enlarged.

    Returns:
        padded image
        scale_x
        scale_y
        left padding
        top padding
    """

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
        top:top + resized_h,
        left:left + resized_w,
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
# Box conversion
# ============================================================

def restore_xyxy_to_original(
    boxes: torch.Tensor,
    original_shape: tuple[int, int],
    scale_x: float,
    scale_y: float,
    left: int,
    top: int,
) -> torch.Tensor:

    h, w = original_shape

    boxes = boxes.clone()

    boxes[:, [0, 2]] -= float(
        left
    )

    boxes[:, [1, 3]] -= float(
        top
    )

    boxes[:, [0, 2]] /= float(
        scale_x
    )

    boxes[:, [1, 3]] /= float(
        scale_y
    )

    boxes[:, 0].clamp_(
        0,
        w,
    )

    boxes[:, 2].clamp_(
        0,
        w,
    )

    boxes[:, 1].clamp_(
        0,
        h,
    )

    boxes[:, 3].clamp_(
        0,
        h,
    )

    return boxes


def xyxy_to_submission(
    xyxy: torch.Tensor,
    conf: torch.Tensor,
    cls: torch.Tensor,
    original_shape: tuple[int, int],
) -> list[str]:

    h, w = original_shape

    lines = []

    for (
        box,
        confidence,
        class_id,
    ) in zip(
        xyxy,
        conf,
        cls,
    ):

        x1, y1, x2, y2 = [
            float(x)
            for x in box.tolist()
        ]

        bw = max(
            0.0,
            x2 - x1,
        )

        bh = max(
            0.0,
            y2 - y1,
        )

        if (
            bw <= 0.0
            or bh <= 0.0
        ):
            continue

        xc = (
            x1 + x2
        ) / 2.0

        yc = (
            y1 + y2
        ) / 2.0

        xc /= float(
            w
        )

        yc /= float(
            h
        )

        bw /= float(
            w
        )

        bh /= float(
            h
        )

        # Defensive clipping against numerical roundoff.
        xc = min(
            max(
                xc,
                0.0,
            ),
            1.0,
        )

        yc = min(
            max(
                yc,
                0.0,
            ),
            1.0,
        )

        bw = min(
            max(
                bw,
                0.0,
            ),
            1.0,
        )

        bh = min(
            max(
                bh,
                0.0,
            ),
            1.0,
        )

        confidence = float(
            confidence
        )

        class_id = int(
            class_id
        )

        if not (
            0
            <= class_id
            < EXPECTED_CLASSES
        ):

            raise RuntimeError(
                f"Illegal class id: {class_id}"
            )

        if not (
            math.isfinite(
                confidence
            )
            and 0.0
            <= confidence
            <= 1.0
        ):

            raise RuntimeError(
                f"Illegal confidence: {confidence}"
            )

        lines.append(
            f"{class_id} "
            f"{xc:.8f} "
            f"{yc:.8f} "
            f"{bw:.8f} "
            f"{bh:.8f} "
            f"{confidence:.8f}"
        )

    return lines


# ============================================================
# Model
# ============================================================

def load_model():

    if not MODEL_PATH.is_file():

        raise FileNotFoundError(
            "Exp04 best.pt not found:\n"
            f"  {MODEL_PATH}"
        )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA unavailable."
        )

    wrapper = YOLO(
        str(
            MODEL_PATH
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

    print(
        "First conv:",
        first_conv,
    )

    print(
        "Detect nc :",
        getattr(
            detect_head,
            "nc",
            None,
        ),
    )

    if (
        first_conv.in_channels
        != EXPECTED_CHANNELS
    ):

        raise RuntimeError(
            "Loaded checkpoint is not a 5-channel Exp04 model."
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
            "Loaded checkpoint is not a 12-class model."
        )

    device = torch.device(
        f"cuda:{DEVICE_INDEX}"
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
        device,
    )


# ============================================================
# Batch preparation
# ============================================================

def prepare_batch(
    stems: list[str],
    candidate: dict,
    device: torch.device,
):

    tensors = []

    metas = []

    for stem in stems:

        rgb_path = (
            candidate["rgb_map"][
                stem
            ]
        )

        ir_path = (
            candidate["ir_map"][
                stem
            ]
        )

        depth_path = (
            candidate["depth_map"][
                stem
            ]
        )

        (
            image,
            original_shape,
        ) = load_5ch(
            rgb_path,
            ir_path,
            depth_path,
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
                "rgb_path": rgb_path,
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

    # Same Trainer preprocessing:
    #
    # uint8 -> float32 -> /255
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
# Prediction
# ============================================================

@torch.inference_mode()
def predict_batch(
    model,
    batch: torch.Tensor,
):
    """
    5-channel batched forward + per-image NMS.

    Important:

    The model forward remains batched for GPU efficiency.

    NMS is deliberately performed one image at a time so that
    Ultralytics' NMS time-limit protection can never cause later
    images in the same forward batch to be silently skipped.
    """

    raw = model(
        batch
    )

    # Normal YOLO11 DetectionModel eval output:
    #
    #     (decoded_predictions, feature_maps)
    #
    if isinstance(
        raw,
        (tuple, list),
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

    if (
        prediction.shape[0]
        != batch.shape[0]
    ):

        raise RuntimeError(
            "Prediction batch size mismatch: "
            f"prediction={prediction.shape[0]}, "
            f"input={batch.shape[0]}"
        )

    detections = []

    # --------------------------------------------------------
    # CRITICAL:
    #
    # Run NMS separately for each image.
    #
    # With the old implementation:
    #
    #     NMS(batch=4)
    #
    # Ultralytics uses one total time budget for all 4 images.
    # A timeout can break the loop and leave later images empty.
    #
    # With:
    #
    #     NMS(batch=1) x 4
    #
    # each image is independently completed.
    # --------------------------------------------------------

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
            multi_label=False,
            max_det=MAX_DET,
            nc=EXPECTED_CLASSES,

            # Default is only 0.05 s/image.
            #
            # conf=0.001 at imgsz=960 can create many candidate
            # boxes, so use a generous safety budget.
            max_time_img=10.0,

            # YOLO11s@960 has fewer raw anchors than this in our
            # configuration, so this does not intentionally
            # truncate normal candidates.
            max_nms=30000,
        )[0]

        detections.append(
            det_one
        )

    return detections


# ============================================================
# Submission validation
# ============================================================

def validate_txt_dir(
    expected_stems: list[str],
) -> dict:

    files = sorted(
        TXT_DIR.glob(
            "*.txt"
        )
    )

    if (
        len(files)
        != EXPECTED_TEST_SAMPLES
    ):

        raise RuntimeError(
            "Submission TXT count mismatch:\n"
            f"  expected={EXPECTED_TEST_SAMPLES}\n"
            f"  actual={len(files)}"
        )

    actual_stems = {
        path.stem
        for path in files
    }

    expected_set = set(
        expected_stems
    )

    if actual_stems != expected_set:

        missing = sorted(
            expected_set
            - actual_stems
        )

        extra = sorted(
            actual_stems
            - expected_set
        )

        raise RuntimeError(
            "Submission filenames mismatch.\n"
            f"Missing: {missing[:20]}\n"
            f"Extra  : {extra[:20]}"
        )

    total_boxes = 0

    empty_files = 0

    max_boxes_seen = 0

    for path in files:

        lines = [
            line.strip()
            for line in path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        if not lines:
            empty_files += 1

        if len(lines) > MAX_DET:

            raise RuntimeError(
                f"{path.name} contains "
                f"{len(lines)} boxes > {MAX_DET}."
            )

        max_boxes_seen = max(
            max_boxes_seen,
            len(lines),
        )

        total_boxes += len(
            lines
        )

        for line_number, line in enumerate(
            lines,
            start=1,
        ):

            fields = line.split()

            if len(fields) != 6:

                raise RuntimeError(
                    f"Invalid submission line:\n"
                    f"  file={path}\n"
                    f"  line={line_number}\n"
                    f"  text={line}"
                )

            try:

                class_id = int(
                    fields[0]
                )

                values = [
                    float(x)
                    for x in fields[1:]
                ]

            except ValueError as exc:

                raise RuntimeError(
                    f"Non-numeric submission line: "
                    f"{path}:{line_number}"
                ) from exc

            if not (
                0
                <= class_id
                < EXPECTED_CLASSES
            ):

                raise RuntimeError(
                    f"Illegal class id in {path.name}: "
                    f"{class_id}"
                )

            xc, yc, bw, bh, confidence = values

            for value in (
                xc,
                yc,
                bw,
                bh,
                confidence,
            ):

                if not (
                    math.isfinite(
                        value
                    )
                    and 0.0
                    <= value
                    <= 1.0
                ):

                    raise RuntimeError(
                        f"Illegal normalized value in "
                        f"{path.name}: {value}"
                    )

    return {
        "txt_files": len(files),
        "total_boxes": total_boxes,
        "empty_files": empty_files,
        "max_boxes": max_boxes_seen,
    }


def create_zip() -> None:

    if ZIP_PATH.exists():

        ZIP_PATH.unlink()

    with zipfile.ZipFile(
        ZIP_PATH,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as zf:

        # IMPORTANT:
        # TXT files go directly into the ZIP root.
        # No extra "txt/" folder inside the submission archive.
        for txt_path in sorted(
            TXT_DIR.glob(
                "*.txt"
            )
        ):

            zf.write(
                txt_path,
                arcname=txt_path.name,
            )


def validate_zip(
    expected_stems: list[str],
) -> None:

    with zipfile.ZipFile(
        ZIP_PATH,
        mode="r",
    ) as zf:

        names = zf.namelist()

    if len(names) != EXPECTED_TEST_SAMPLES:

        raise RuntimeError(
            "ZIP file count mismatch:\n"
            f"  expected={EXPECTED_TEST_SAMPLES}\n"
            f"  actual={len(names)}"
        )

    if any(
        "/" in name
        for name in names
    ):

        raise RuntimeError(
            "ZIP contains directories. "
            "Submission TXT files must be at ZIP root."
        )

    expected_names = {
        f"{stem}.txt"
        for stem in expected_stems
    }

    if set(
        names
    ) != expected_names:

        raise RuntimeError(
            "ZIP filenames do not exactly match test stems."
        )


# ============================================================
# Main inference
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test-root",
        type=str,
        default=None,
        help=(
            "Optional test root. "
            "Normally omit this and let the script "
            "auto-detect the 1000 aligned test triplets."
        ),
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Only detect and verify test-set paths; "
            "do not run model inference."
        ),
    )

    args = parser.parse_args()

    section(
        "AIC2026 Exp04 Submission Inference"
    )

    print(
        "Project root :",
        PROJECT_ROOT,
    )

    print(
        "Model        :",
        MODEL_PATH,
    )

    print(
        "Datasets root:",
        DATASETS_ROOT,
    )

    print(
        "Image size   :",
        IMAGE_SIZE,
    )

    print(
        "Confidence   :",
        CONF_THRESHOLD,
    )

    print(
        "NMS IoU      :",
        IOU_THRESHOLD,
    )

    print(
        "Max det      :",
        MAX_DET,
    )

    print(
        "Batch size   :",
        BATCH_SIZE,
    )

    manual_root = None

    if args.test_root is not None:

        supplied = Path(
            args.test_root
        )

        manual_root = (
            supplied
            if supplied.is_absolute()
            else (
                PROJECT_ROOT
                / supplied
            )
        ).resolve()

    section(
        "Discover test set"
    )

    (
        candidate,
        stems,
    ) = discover_test_set(
        manual_root
    )

    print(
        "Selected test root:",
        candidate["root"],
    )

    print(
        "RGB directory :",
        candidate["rgb_dir"],
    )

    print(
        "IR directory  :",
        candidate["ir_dir"],
    )

    print(
        "Depth directory:",
        candidate["depth_dir"],
    )

    print(
        "Aligned samples:",
        len(
            stems
        ),
    )

    print(
        "First stem     :",
        stems[0],
    )

    print(
        "Last stem      :",
        stems[-1],
    )

    # --------------------------------------------------------
    # Inspect first real triplet before model inference.
    # --------------------------------------------------------

    first_stem = stems[0]

    (
        first_image,
        first_shape,
    ) = load_5ch(
        candidate["rgb_map"][
            first_stem
        ],
        candidate["ir_map"][
            first_stem
        ],
        candidate["depth_map"][
            first_stem
        ],
    )

    print()
    print(
        "First raw 5ch shape:",
        first_image.shape,
    )

    print(
        "First raw dtype    :",
        first_image.dtype,
    )

    print(
        "First original HW  :",
        first_shape,
    )

    print(
        "Channel min/max:"
    )

    for channel_index, channel_name in enumerate(
        (
            "R",
            "G",
            "B",
            "IR",
            "Depth",
        )
    ):

        channel = first_image[
            ...,
            channel_index
        ]

        print(
            f"  {channel_name:5s}: "
            f"{int(channel.min())} .. "
            f"{int(channel.max())}"
        )

    if args.check_only:

        section(
            "CHECK ONLY"
        )

        print(
            "STATUS = PASS"
        )

        return

    section(
        "Load model"
    )

    (
        wrapper,
        model,
        device,
    ) = load_model()

    print(
        "Device:",
        device,
    )

    print(
        "GPU   :",
        torch.cuda.get_device_name(
            DEVICE_INDEX
        ),
    )

    # --------------------------------------------------------
    # Prepare clean output directory.
    # --------------------------------------------------------

    if TXT_DIR.exists():

        shutil.rmtree(
            TXT_DIR
        )

    TXT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if ZIP_PATH.exists():

        ZIP_PATH.unlink()

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    section(
        "Inference"
    )

    total_batches = math.ceil(
        len(stems)
        / BATCH_SIZE
    )

    total_predictions = 0

    empty_predictions = 0

    for batch_index, start in enumerate(
        range(
            0,
            len(stems),
            BATCH_SIZE,
        ),
        start=1,
    ):

        if (
            batch_index == 1
            or batch_index % 10 == 0
            or batch_index == total_batches
        ):
            print(
                f"[Inference] "
                f"batch {batch_index}/{total_batches} "
                f"({min(start + BATCH_SIZE, len(stems))}/{len(stems)} images)"
            )

        batch_stems = stems[
            start:
            start + BATCH_SIZE
        ]

        (
            batch,
            metas,
        ) = prepare_batch(
            batch_stems,
            candidate,
            device,
        )

        detections = predict_batch(
            model,
            batch,
        )

        if len(
            detections
        ) != len(
            metas
        ):

            raise RuntimeError(
                "NMS output batch length mismatch."
            )

        for det, meta in zip(
            detections,
            metas,
        ):

            stem = meta[
                "stem"
            ]

            txt_path = (
                TXT_DIR
                / f"{stem}.txt"
            )

            # Required by competition:
            # even no-detection images must have an empty TXT.
            if (
                det is None
                or len(det) == 0
            ):

                txt_path.write_text(
                    "",
                    encoding="utf-8",
                )

                empty_predictions += 1

                continue

            # Sort by confidence descending and strictly cap at 100.
            order = torch.argsort(
                det[:, 4],
                descending=True,
            )

            det = det[
                order
            ][
                :MAX_DET
            ]

            boxes = restore_xyxy_to_original(
                det[:, :4],
                original_shape=meta[
                    "original_shape"
                ],
                scale_x=meta[
                    "scale_x"
                ],
                scale_y=meta[
                    "scale_y"
                ],
                left=meta[
                    "left"
                ],
                top=meta[
                    "top"
                ],
            )

            lines = xyxy_to_submission(
                xyxy=boxes,
                conf=det[:, 4],
                cls=det[:, 5],
                original_shape=meta[
                    "original_shape"
                ],
            )

            # Defensive final max-det cap after invalid zero-area
            # boxes have been removed.
            lines = lines[
                :MAX_DET
            ]

            total_predictions += len(
                lines
            )

            txt_path.write_text(
                (
                    "\n".join(
                        lines
                    )
                    + (
                        "\n"
                        if lines
                        else ""
                    )
                ),
                encoding="utf-8",
            )

        # Release batch GPU memory before the next batch.
        del batch
        del detections

    # --------------------------------------------------------
    # Validate TXT submission.
    # --------------------------------------------------------

    section(
        "Validate TXT files"
    )

    stats = validate_txt_dir(
        stems
    )

    print(
        "TXT files        :",
        stats[
            "txt_files"
        ],
    )

    print(
        "Total boxes      :",
        stats[
            "total_boxes"
        ],
    )

    print(
        "Empty TXT files  :",
        stats[
            "empty_files"
        ],
    )

    print(
        "Max boxes / image:",
        stats[
            "max_boxes"
        ],
    )

    # --------------------------------------------------------
    # ZIP
    # --------------------------------------------------------

    section(
        "Create submission ZIP"
    )

    create_zip()

    validate_zip(
        stems
    )

    zip_size_mb = (
        ZIP_PATH.stat().st_size
        / 1024**2
    )

    print(
        "TXT directory:",
        TXT_DIR,
    )

    print(
        "ZIP          :",
        ZIP_PATH,
    )

    print(
        "ZIP size     :",
        f"{zip_size_mb:.2f} MB",
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    section(
        "FINAL RESULT"
    )

    print(
        "5-channel test loading             PASS"
    )

    print(
        "RGB / IR / Depth alignment         PASS"
    )

    print(
        "Exp04 best.pt                      PASS"
    )

    print(
        "Inference                          PASS"
    )

    print(
        "1000 same-name TXT files           PASS"
    )

    print(
        "Empty predictions preserved        PASS"
    )

    print(
        "Submission line format             PASS"
    )

    print(
        "max_det <= 100                     PASS"
    )

    print(
        "ZIP root structure                 PASS"
    )

    print()
    print(
        "STATUS = PASS"
    )

    print()
    print(
        "SUBMIT THIS ZIP:"
    )

    print(
        ZIP_PATH
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 Exp05
Asymmetric Gated Multimodal YOLO11s
Rect test inference + submission ZIP.

Frozen production protocol
==========================

Checkpoint:
    runs/exp05_asymmetric_gated_yolo11s_960/weights/best.pt

Input:
    [R, G, B, IR, Depth]

Model:
    RGB main path : pretrained 3-channel YOLO11s
    Depth         : gated P3 fusion
    IR            : gated P4 fusion

Inference:
    imgsz          = 960
    rect group     = 8
    forward batch  = 4
    rect pad       = 0.5

    FP32 / 255
    conf           = 0.001
    NMS IoU        = 0.70
    max_det        = 100
    multi_label    = True
    NMS            = per-image

Submission:
    exactly 1000 TXT files
    empty detections -> empty TXT
    <= 100 detections per image
    ZIP contains TXT files directly at root

Validated dependencies
======================

infer_exp04_rgbid_early5_960_submit.py
    -> test-set discovery
    -> final competition TXT formatter

test_exp04_modality_ablation_val400.py
    -> RGB/IR/Depth loading
    -> Depth conversion
    -> 5-channel composition
    -> per-image NMS
    -> box restoration

test_exp04_rect_inference_parity_val400.py
    -> validated Ultralytics-like Rect preprocessing

Important
=========

Do NOT use base.load_model() here.

Exp04:
    external input = 5
    first main conv = 5

Exp05:
    external input = 5
    RGB main conv   = 3
    Depth encoder   = 1
    IR encoder      = 1

The Exp05 custom model performs the modality split internally.
"""

from __future__ import annotations

import argparse
import math
import shutil
import time
import zipfile
from collections import Counter
from pathlib import Path

import torch
from ultralytics import YOLO

# IMPORTANT:
# Import the custom model definition before checkpoint deserialization.
from multimodal_gated_model import (
    Exp05AsymmetricGatedDetectionModel,
)

import infer_exp04_rgbid_early5_960_submit as submit_base
import test_exp04_modality_ablation_val400 as base
import test_exp04_rect_inference_parity_val400 as rect


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
    / "exp05_asymmetric_gated_yolo11s_960"
    / "weights"
    / "best.pt"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "exp05_asymmetric_gated_yolo11s_960"
    / "submission_test_rect"
)

TXT_DIR = (
    OUTPUT_ROOT
    / "txt"
)

ZIP_PATH = (
    OUTPUT_ROOT
    / "exp05_asymmetric_gated_960_submit_rect.zip"
)


# ============================================================
# Frozen submission protocol
# ============================================================

EXPECTED_TEST_SAMPLES = 1000

EXPECTED_CLASSES = 12

EXPECTED_EXTERNAL_CHANNELS = 5

EXPECTED_RGB_CHANNELS = 3

EXPECTED_AUX_CHANNELS = 1

MAX_DET = 100

FORWARD_BATCH = 4

DEVICE_INDEX = 0


# ============================================================
# Console
# ============================================================

def section(
    title: str,
) -> None:

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def pass_line(
    text: str,
) -> None:

    print(
        f"[PASS] {text}"
    )


# ============================================================
# Path helper
# ============================================================

def resolve_test_root(
    text: str | None,
) -> Path | None:

    if text is None:

        return None

    path = Path(
        text
    )

    if not path.is_absolute():

        path = (
            PROJECT_ROOT
            / path
        )

    return path.resolve()


# ============================================================
# Protocol guard
# ============================================================

def verify_frozen_protocol() -> None:

    section(
        "Frozen production protocol"
    )

    print(
        "imgsz            :",
        rect.IMAGE_SIZE,
    )

    print(
        "conf             :",
        base.CONF_THRESHOLD,
    )

    print(
        "NMS IoU          :",
        base.IOU_THRESHOLD,
    )

    print(
        "max_det          :",
        base.MAX_DET,
    )

    print(
        "multi_label      : True"
    )

    print(
        "forward batch    :",
        FORWARD_BATCH,
    )

    print(
        "rect group batch :",
        rect.RECT_GROUP_BATCH,
    )

    print(
        "rect pad         :",
        rect.RECT_PAD,
    )

    if (
        rect.IMAGE_SIZE
        != 960
    ):

        raise RuntimeError(
            "Rect IMAGE_SIZE is no longer 960."
        )

    if abs(
        float(
            base.CONF_THRESHOLD
        )
        - 0.001
    ) > 1e-12:

        raise RuntimeError(
            "CONF_THRESHOLD changed."
        )

    if abs(
        float(
            base.IOU_THRESHOLD
        )
        - 0.70
    ) > 1e-12:

        raise RuntimeError(
            "IOU_THRESHOLD changed."
        )

    if (
        base.MAX_DET
        != 100
    ):

        raise RuntimeError(
            "base.MAX_DET changed."
        )

    if (
        MAX_DET
        != base.MAX_DET
    ):

        raise RuntimeError(
            "Local MAX_DET does not match "
            "validated evaluator."
        )

    if (
        rect.RECT_GROUP_BATCH
        != 8
    ):

        raise RuntimeError(
            "RECT_GROUP_BATCH changed."
        )

    if abs(
        float(
            rect.RECT_PAD
        )
        - 0.5
    ) > 1e-12:

        raise RuntimeError(
            "RECT_PAD changed."
        )

    if (
        FORWARD_BATCH
        != 4
    ):

        raise RuntimeError(
            "FORWARD_BATCH changed."
        )

    pass_line(
        "production inference protocol unchanged"
    )


# ============================================================
# Exp05 model loading
# ============================================================

def load_exp05_model(
    device: torch.device,
):

    section(
        "Load frozen Exp05 best.pt"
    )

    if not MODEL_PATH.is_file():

        raise FileNotFoundError(
            "Exp05 checkpoint not found:\n"
            f"  {MODEL_PATH}"
        )

    wrapper = YOLO(
        str(
            MODEL_PATH
        )
    )

    model = (
        wrapper.model
    )

    print(
        "Checkpoint        :",
        MODEL_PATH,
    )

    print(
        "Model class       :",
        type(
            model
        ).__name__,
    )

    if not isinstance(
        model,
        Exp05AsymmetricGatedDetectionModel,
    ):

        raise RuntimeError(
            "Checkpoint did not deserialize as "
            "Exp05AsymmetricGatedDetectionModel.\n"
            f"Actual class: {type(model).__name__}"
        )

    # --------------------------------------------------------
    # External multimodal input
    # --------------------------------------------------------

    external_channels = getattr(
        model,
        "multimodal_channels",
        None,
    )

    print(
        "External channels :",
        external_channels,
    )

    if (
        external_channels
        != EXPECTED_EXTERNAL_CHANNELS
    ):

        raise RuntimeError(
            "Unexpected Exp05 external channel count: "
            f"{external_channels}"
        )

    # --------------------------------------------------------
    # RGB main path
    # --------------------------------------------------------

    rgb_first = (
        model
        .model[0]
        .conv
    )

    print(
        "RGB stem channels :",
        rgb_first.in_channels,
    )

    if (
        rgb_first.in_channels
        != EXPECTED_RGB_CHANNELS
    ):

        raise RuntimeError(
            "Exp05 RGB main stem is not 3-channel."
        )

    # --------------------------------------------------------
    # Depth / IR auxiliary paths
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

    print(
        "Depth stem ch     :",
        depth_first.in_channels,
    )

    print(
        "IR stem ch        :",
        ir_first.in_channels,
    )

    if (
        depth_first.in_channels
        != EXPECTED_AUX_CHANNELS
    ):

        raise RuntimeError(
            "Depth encoder is not 1-channel."
        )

    if (
        ir_first.in_channels
        != EXPECTED_AUX_CHANNELS
    ):

        raise RuntimeError(
            "IR encoder is not 1-channel."
        )

    # --------------------------------------------------------
    # Detection head
    # --------------------------------------------------------

    detect_head = (
        model
        .model[-1]
    )

    nc = getattr(
        detect_head,
        "nc",
        None,
    )

    print(
        "Classes           :",
        nc,
    )

    if (
        nc
        != EXPECTED_CLASSES
    ):

        raise RuntimeError(
            "Unexpected class count: "
            f"{nc}"
        )

    # --------------------------------------------------------
    # Stride
    # --------------------------------------------------------

    strides = tuple(
        float(
            value
        )
        for value
        in model.stride.detach().cpu().tolist()
    )

    print(
        "Strides           :",
        strides,
    )

    if (
        strides
        != (
            8.0,
            16.0,
            32.0,
        )
    ):

        raise RuntimeError(
            "Unexpected Exp05 strides: "
            f"{strides}"
        )

    # --------------------------------------------------------
    # Gate audit
    # --------------------------------------------------------

    gate_stats = (
        model
        .get_gate_statistics()
    )

    print()
    print(
        "Depth gate:",
        gate_stats[
            "depth"
        ],
    )

    print(
        "IR gate   :",
        gate_stats[
            "ir"
        ],
    )

    # --------------------------------------------------------
    # Finite checkpoint state
    # --------------------------------------------------------

    for (
        name,
        tensor,
    ) in model.state_dict().items():

        if not isinstance(
            tensor,
            torch.Tensor,
        ):

            continue

        if not (
            tensor.is_floating_point()
        ):

            continue

        if not torch.isfinite(
            tensor
        ).all():

            raise RuntimeError(
                "Non-finite checkpoint tensor:\n"
                f"  {name}"
            )

    model = (
        model
        .to(
            device
        )
        .float()
        .eval()
    )

    pass_line(
        "Exp05 best.pt structure"
    )

    pass_line(
        "checkpoint tensors finite"
    )

    return (
        wrapper,
        model,
        gate_stats,
    )


# ============================================================
# TXT validation
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
        len(
            files
        )
        != EXPECTED_TEST_SAMPLES
    ):

        raise RuntimeError(
            "TXT count mismatch: "
            f"expected={EXPECTED_TEST_SAMPLES}, "
            f"actual={len(files)}"
        )

    expected = set(
        expected_stems
    )

    actual = {
        path.stem
        for path
        in files
    }

    if (
        actual
        != expected
    ):

        raise RuntimeError(
            "TXT stem mismatch.\n"
            f"missing="
            f"{sorted(expected - actual)[:20]}\n"
            f"extra="
            f"{sorted(actual - expected)[:20]}"
        )

    total_boxes = 0

    empty_files = 0

    max_boxes_seen = 0

    class_counts = Counter()

    for path in files:

        lines = [
            line.strip()
            for line
            in path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        if not lines:

            empty_files += 1

        if (
            len(
                lines
            )
            > MAX_DET
        ):

            raise RuntimeError(
                f"{path.name}: "
                f"{len(lines)} boxes > {MAX_DET}"
            )

        max_boxes_seen = max(
            max_boxes_seen,
            len(
                lines
            ),
        )

        total_boxes += len(
            lines
        )

        for (
            line_no,
            line,
        ) in enumerate(
            lines,
            start=1,
        ):

            fields = (
                line.split()
            )

            if (
                len(
                    fields
                )
                != 6
            ):

                raise RuntimeError(
                    "Bad submission line:\n"
                    f"  file={path}\n"
                    f"  line={line_no}\n"
                    f"  value={line}"
                )

            try:

                class_id = int(
                    fields[
                        0
                    ]
                )

                values = [
                    float(
                        value
                    )
                    for value
                    in fields[
                        1:
                    ]
                ]

            except ValueError as exc:

                raise RuntimeError(
                    "Non-numeric submission line:\n"
                    f"  {path}:{line_no}"
                ) from exc

            if not (
                0
                <= class_id
                < EXPECTED_CLASSES
            ):

                raise RuntimeError(
                    "Illegal class id:\n"
                    f"  file={path.name}\n"
                    f"  class={class_id}"
                )

            class_counts[
                class_id
            ] += 1

            for value in values:

                if not (
                    math.isfinite(
                        value
                    )
                    and 0.0
                    <= value
                    <= 1.0
                ):

                    raise RuntimeError(
                        "Illegal normalized value:\n"
                        f"  file={path.name}\n"
                        f"  value={value}"
                    )

    return {
        "txt_files":
            len(
                files
            ),

        "total_boxes":
            total_boxes,

        "empty_files":
            empty_files,

        "max_boxes":
            max_boxes_seen,

        "class_counts":
            class_counts,
    }


# ============================================================
# ZIP creation + validation
# ============================================================

def create_and_validate_zip(
    expected_stems: list[str],
) -> None:

    if ZIP_PATH.exists():

        ZIP_PATH.unlink()

    with zipfile.ZipFile(
        ZIP_PATH,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as zf:

        for path in sorted(
            TXT_DIR.glob(
                "*.txt"
            )
        ):

            zf.write(
                path,
                arcname=path.name,
            )

    # --------------------------------------------------------
    # Read back ZIP and validate exact structure.
    # --------------------------------------------------------

    with zipfile.ZipFile(
        ZIP_PATH,
        "r",
    ) as zf:

        names = (
            zf.namelist()
        )

        bad_file = (
            zf.testzip()
        )

    if (
        bad_file
        is not None
    ):

        raise RuntimeError(
            "ZIP CRC validation failed:\n"
            f"  {bad_file}"
        )

    if (
        len(
            names
        )
        != EXPECTED_TEST_SAMPLES
    ):

        raise RuntimeError(
            "ZIP count mismatch: "
            f"expected={EXPECTED_TEST_SAMPLES}, "
            f"actual={len(names)}"
        )

    if any(
        "/" in name
        for name
        in names
    ):

        raise RuntimeError(
            "ZIP contains subdirectories. "
            "TXT files must be directly at ZIP root."
        )

    expected_names = {
        f"{stem}.txt"
        for stem
        in expected_stems
    }

    if (
        set(
            names
        )
        != expected_names
    ):

        missing = (
            expected_names
            - set(
                names
            )
        )

        extra = (
            set(
                names
            )
            - expected_names
        )

        raise RuntimeError(
            "ZIP filename mismatch.\n"
            f"missing={sorted(missing)[:20]}\n"
            f"extra={sorted(extra)[:20]}"
        )

    pass_line(
        "submission ZIP structure"
    )

    pass_line(
        "submission ZIP CRC"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "AIC2026 Exp05 asymmetric gated "
            "Rect test inference and submission ZIP."
        )
    )

    parser.add_argument(
        "--test-root",
        type=str,
        default=None,
        help=(
            "Optional test root. "
            "If omitted, auto-detect under project datasets/."
        ),
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Audit dataset, checkpoint and Rect geometry, "
            "then execute one real 5-channel inference batch "
            "without creating submission files."
        ),
    )

    args = (
        parser.parse_args()
    )

    section(
        "AIC2026 Exp05 Rect Submission"
    )

    print(
        "Project root:",
        PROJECT_ROOT,
    )

    print(
        "Model       :",
        MODEL_PATH,
    )

    print(
        "Output      :",
        OUTPUT_ROOT,
    )

    # ========================================================
    # Freeze inference protocol
    # ========================================================

    verify_frozen_protocol()

    # ========================================================
    # Discover test set
    # ========================================================

    section(
        "Discover test set"
    )

    manual_root = (
        resolve_test_root(
            args.test_root
        )
    )

    (
        candidate,
        stems,
    ) = (
        submit_base
        .discover_test_set(
            manual_root
        )
    )

    if (
        len(
            stems
        )
        != EXPECTED_TEST_SAMPLES
    ):

        raise RuntimeError(
            "Unexpected test-set size:\n"
            f"  expected={EXPECTED_TEST_SAMPLES}\n"
            f"  actual={len(stems)}"
        )

    print(
        "Selected root   :",
        candidate[
            "root"
        ],
    )

    print(
        "RGB             :",
        candidate[
            "rgb_dir"
        ],
    )

    print(
        "IR              :",
        candidate[
            "ir_dir"
        ],
    )

    print(
        "Depth           :",
        candidate[
            "depth_dir"
        ],
    )

    print(
        "Aligned samples :",
        len(
            stems
        ),
    )

    print(
        "First / last    :",
        stems[
            0
        ],
        "/",
        stems[
            -1
        ],
    )

    # --------------------------------------------------------
    # Verify first real 5-channel sample.
    # --------------------------------------------------------

    first_stem = (
        stems[
            0
        ]
    )

    (
        first_image,
        first_hw,
    ) = base.load_5ch(
        candidate[
            "rgb_map"
        ][
            first_stem
        ],
        candidate[
            "ir_map"
        ][
            first_stem
        ],
        candidate[
            "depth_map"
        ][
            first_stem
        ],
    )

    print(
        "First raw 5ch   :",
        first_image.shape,
        first_image.dtype,
    )

    print(
        "First original HW:",
        first_hw,
    )

    if (
        first_image.ndim
        != 3
        or first_image.shape[
            2
        ]
        != EXPECTED_EXTERNAL_CHANNELS
    ):

        raise RuntimeError(
            "Test input is not 5-channel."
        )

    pass_line(
        "aligned RGB/IR/Depth test set"
    )

    # ========================================================
    # Device
    # ========================================================

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA unavailable."
        )

    device = torch.device(
        f"cuda:{DEVICE_INDEX}"
    )

    print()
    print(
        "Device          :",
        device,
    )

    print(
        "GPU             :",
        torch.cuda.get_device_name(
            DEVICE_INDEX
        ),
    )

    # ========================================================
    # Exp05 checkpoint
    # ========================================================

    (
        wrapper,
        model,
        gate_stats,
    ) = load_exp05_model(
        device
    )

    stride = (
        rect.get_stride(
            model
        )
    )

    print(
        "Max stride       :",
        stride,
    )

    # ========================================================
    # Rect geometry
    # ========================================================

    section(
        "Rect geometry audit"
    )

    (
        shape_by_stem,
        source_counts,
    ) = rect.inspect_shapes(
        stems,
        candidate[
            "rgb_map"
        ],
    )

    print(
        "Source shape distribution:"
    )

    for (
        (
            h,
            w,
        ),
        count,
    ) in sorted(
        source_counts.items()
    ):

        print(
            f"  {h}x{w}: "
            f"{count}"
        )

    groups = (
        rect.build_rect_groups(
            stems,
            shape_by_stem,
            stride,
        )
    )

    grouped_stems = [
        stem
        for group
        in groups
        for stem
        in group[
            "stems"
        ]
    ]

    if (
        len(
            grouped_stems
        )
        != EXPECTED_TEST_SAMPLES
    ):

        raise RuntimeError(
            "Rect grouping sample count mismatch."
        )

    if (
        len(
            set(
                grouped_stems
            )
        )
        != EXPECTED_TEST_SAMPLES
    ):

        raise RuntimeError(
            "Rect grouping contains duplicate stems."
        )

    if (
        set(
            grouped_stems
        )
        != set(
            stems
        )
    ):

        raise RuntimeError(
            "Rect grouping changed test membership."
        )

    rect_counts = Counter(
        group[
            "shape"
        ]
        for group
        in groups
    )

    print()
    print(
        "Rect group shape distribution:"
    )

    for (
        (
            h,
            w,
        ),
        count,
    ) in sorted(
        rect_counts.items()
    ):

        print(
            f"  {h}x{w}: "
            f"{count} logical batches"
        )

    ratios = sorted(
        {
            round(
                h
                / float(
                    w
                ),
                8,
            )
            for (
                h,
                w,
            ) in shape_by_stem.values()
        }
    )

    print()
    print(
        "H/W ratios:",
        ratios,
    )

    (
        resized,
        sx,
        sy,
    ) = (
        rect
        .resize_long_side_like_dataset(
            first_image
        )
    )

    print(
        "First long-side resize:",
        resized.shape[
            :2
        ],
    )

    print(
        "First pre-scale       :",
        f"{sx:.6f}",
        f"{sy:.6f}",
    )

    pass_line(
        "Rect grouping covers test1000 exactly once"
    )

    # ========================================================
    # One real batch audit
    # ========================================================

    section(
        "Real Exp05 5-channel inference audit"
    )

    first_group = (
        groups[
            0
        ]
    )

    check_stems = (
        first_group[
            "stems"
        ][
            :
            min(
                FORWARD_BATCH,
                len(
                    first_group[
                        "stems"
                    ]
                ),
            )
        ]
    )

    (
        check_batch,
        check_metas,
    ) = rect.prepare_rect_batch(
        check_stems,
        first_group[
            "shape"
        ],
        candidate[
            "rgb_map"
        ],
        candidate[
            "ir_map"
        ],
        candidate[
            "depth_map"
        ],
        device,
    )

    print(
        "Audit batch shape:",
        tuple(
            check_batch.shape
        ),
    )

    if (
        check_batch.ndim
        != 4
        or check_batch.shape[
            1
        ]
        != EXPECTED_EXTERNAL_CHANNELS
    ):

        raise RuntimeError(
            "Rect batch is not BCHW 5-channel."
        )

    with torch.inference_mode():

        check_detections = (
            base.predict_batch(
                model,
                check_batch,
            )
        )

    if (
        len(
            check_detections
        )
        != len(
            check_metas
        )
    ):

        raise RuntimeError(
            "Audit NMS batch length mismatch."
        )

    audit_boxes = sum(
        0
        if (
            det is None
            or det.numel() == 0
        )
        else int(
            det.shape[
                0
            ]
        )
        for det
        in check_detections
    )

    print(
        "Audit detections :",
        audit_boxes,
    )

    pass_line(
        "real Exp05 5-channel forward + NMS"
    )

    del check_batch
    del check_metas
    del check_detections

    torch.cuda.empty_cache()

    # ========================================================
    # Check-only exit
    # ========================================================

    if args.check_only:

        section(
            "CHECK ONLY"
        )

        print(
            "Exp05 checkpoint             PASS"
        )

        print(
            "5-channel test alignment     PASS"
        )

        print(
            "Rect preprocessing           PASS"
        )

        print(
            "Real model forward           PASS"
        )

        print(
            "Per-image NMS                PASS"
        )

        print()

        print(
            "STATUS = PASS"
        )

        return

    # ========================================================
    # Clean only THIS Exp05 output
    # ========================================================

    section(
        "Prepare clean Exp05 submission output"
    )

    if TXT_DIR.exists():

        shutil.rmtree(
            TXT_DIR
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    TXT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if ZIP_PATH.exists():

        ZIP_PATH.unlink()

    print(
        "TXT directory:",
        TXT_DIR,
    )

    print(
        "ZIP target   :",
        ZIP_PATH,
    )

    # ========================================================
    # Full Rect inference
    # ========================================================

    section(
        "Exp05 Rect test inference"
    )

    total_forward = sum(
        math.ceil(
            len(
                group[
                    "stems"
                ]
            )
            / FORWARD_BATCH
        )
        for group
        in groups
    )

    forward_idx = 0

    processed = 0

    tracked_boxes = 0

    tracked_empty = 0

    torch.cuda.reset_peak_memory_stats(
        device
    )

    torch.cuda.synchronize(
        device
    )

    t0 = (
        time.time()
    )

    with torch.inference_mode():

        for group in groups:

            group_stems = (
                group[
                    "stems"
                ]
            )

            for start in range(
                0,
                len(
                    group_stems
                ),
                FORWARD_BATCH,
            ):

                forward_idx += 1

                batch_stems = (
                    group_stems[
                        start:
                        start
                        + FORWARD_BATCH
                    ]
                )

                (
                    batch,
                    metas,
                ) = (
                    rect
                    .prepare_rect_batch(
                        batch_stems,
                        group[
                            "shape"
                        ],
                        candidate[
                            "rgb_map"
                        ],
                        candidate[
                            "ir_map"
                        ],
                        candidate[
                            "depth_map"
                        ],
                        device,
                    )
                )

                # --------------------------------------------
                # Exact validated prediction path:
                #
                # conf        = 0.001
                # NMS IoU     = 0.70
                # max_det     = 100
                # multi_label = True
                # NMS         = per-image
                # --------------------------------------------

                detections = (
                    base.predict_batch(
                        model,
                        batch,
                    )
                )

                if (
                    len(
                        detections
                    )
                    != len(
                        metas
                    )
                ):

                    raise RuntimeError(
                        "NMS output batch "
                        "length mismatch."
                    )

                # --------------------------------------------
                # One TXT per test image
                # --------------------------------------------

                for (
                    det,
                    meta,
                ) in zip(
                    detections,
                    metas,
                ):

                    stem = (
                        meta[
                            "stem"
                        ]
                    )

                    txt_path = (
                        TXT_DIR
                        / f"{stem}.txt"
                    )

                    # ----------------------------------------
                    # Empty detection:
                    #
                    # competition requires an EMPTY TXT,
                    # not a missing TXT file.
                    # ----------------------------------------

                    if (
                        det is None
                        or det.numel() == 0
                    ):

                        txt_path.write_text(
                            "",
                            encoding="utf-8",
                        )

                        tracked_empty += 1

                        continue

                    # ----------------------------------------
                    # Defensive confidence sort.
                    # base.predict_batch already enforces
                    # max_det=100, but retain this final guard.
                    # ----------------------------------------

                    order = (
                        torch.argsort(
                            det[
                                :,
                                4
                            ],
                            descending=True,
                        )
                    )

                    det = (
                        det[
                            order
                        ][
                            :MAX_DET
                        ]
                    )

                    # ----------------------------------------
                    # Rect coordinates -> original image
                    # ----------------------------------------

                    boxes = (
                        base
                        .restore_xyxy_to_original(
                            det[
                                :,
                                :4
                            ],
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
                    )

                    # ----------------------------------------
                    # Reuse the exact validated competition
                    # TXT coordinate formatter.
                    #
                    # Do not duplicate the formatting logic.
                    # ----------------------------------------

                    lines = (
                        submit_base
                        .xyxy_to_submission(
                            xyxy=boxes,
                            conf=det[
                                :,
                                4
                            ],
                            cls=det[
                                :,
                                5
                            ],
                            original_shape=meta[
                                "original_shape"
                            ],
                        )[
                            :MAX_DET
                        ]
                    )

                    if (
                        len(
                            lines
                        )
                        > MAX_DET
                    ):

                        raise RuntimeError(
                            f"{stem}: "
                            "formatter returned >100 boxes."
                        )

                    tracked_boxes += len(
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

                processed += len(
                    batch_stems
                )

                # --------------------------------------------
                # Progress
                # --------------------------------------------

                if (
                    forward_idx == 1
                    or forward_idx % 10 == 0
                    or forward_idx
                    == total_forward
                ):

                    (
                        h,
                        w,
                    ) = (
                        group[
                            "shape"
                        ]
                    )

                    print(
                        f"[Rect] "
                        f"batch "
                        f"{forward_idx}/"
                        f"{total_forward}  "
                        f"{processed}/"
                        f"{len(stems)} images  "
                        f"shape={h}x{w}  "
                        f"boxes={tracked_boxes}  "
                        f"elapsed="
                        f"{time.time() - t0:.1f}s"
                    )

                del batch
                del detections

    torch.cuda.synchronize(
        device
    )

    elapsed = (
        time.time()
        - t0
    )

    peak_memory_gib = (
        torch.cuda
        .max_memory_allocated(
            device
        )
        / 1024**3
    )

    if (
        processed
        != EXPECTED_TEST_SAMPLES
    ):

        raise RuntimeError(
            "Processed-image count mismatch:\n"
            f"  expected={EXPECTED_TEST_SAMPLES}\n"
            f"  actual={processed}"
        )

    # ========================================================
    # Validate TXT files
    # ========================================================

    section(
        "Validate submission TXT files"
    )

    stats = (
        validate_txt_dir(
            stems
        )
    )

    print(
        "TXT files         :",
        stats[
            "txt_files"
        ],
    )

    print(
        "Total boxes       :",
        stats[
            "total_boxes"
        ],
    )

    print(
        "Empty TXT files   :",
        stats[
            "empty_files"
        ],
    )

    print(
        "Max boxes / image :",
        stats[
            "max_boxes"
        ],
    )

    print(
        "Inference seconds :",
        f"{elapsed:.2f}",
    )

    print(
        "Peak CUDA memory  :",
        f"{peak_memory_gib:.3f} GiB",
    )

    print()
    print(
        "Class detection counts:"
    )

    for class_id in range(
        EXPECTED_CLASSES
    ):

        print(
            f"  class {class_id:2d}: "
            f"{stats['class_counts'][class_id]}"
        )

    if (
        stats[
            "total_boxes"
        ]
        != tracked_boxes
    ):

        raise RuntimeError(
            "Box-count mismatch:\n"
            f"  audited={stats['total_boxes']}\n"
            f"  tracked={tracked_boxes}"
        )

    if (
        stats[
            "empty_files"
        ]
        != tracked_empty
    ):

        raise RuntimeError(
            "Empty-count mismatch:\n"
            f"  audited={stats['empty_files']}\n"
            f"  tracked={tracked_empty}"
        )

    pass_line(
        "1000 same-name TXT files"
    )

    pass_line(
        "empty detections preserved as empty TXT"
    )

    pass_line(
        "max_det <= 100"
    )

    # ========================================================
    # Create + validate ZIP
    # ========================================================

    section(
        "Create submission ZIP"
    )

    create_and_validate_zip(
        stems
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
        f"{ZIP_PATH.stat().st_size / 1024**2:.2f} MB",
    )

    # ========================================================
    # Final
    # ========================================================

    section(
        "FINAL RESULT"
    )

    print(
        "Frozen Exp05 best.pt                PASS"
    )

    print(
        "Exp05 custom gated model            PASS"
    )

    print(
        "External 5-channel input            PASS"
    )

    print(
        "RGB 3-channel protected main path   PASS"
    )

    print(
        "Depth gated P3 path                 PASS"
    )

    print(
        "IR gated P4 path                    PASS"
    )

    print(
        "RGB/IR/Depth test alignment         PASS"
    )

    print(
        "Validated Rect preprocessing        PASS"
    )

    print(
        "FP32 / 255                          PASS"
    )

    print(
        "conf=0.001                          PASS"
    )

    print(
        "NMS IoU=0.70                       PASS"
    )

    print(
        "multi_label=True                    PASS"
    )

    print(
        "Per-image NMS                       PASS"
    )

    print(
        "max_det <= 100                      PASS"
    )

    print(
        "1000 same-name TXT files            PASS"
    )

    print(
        "Empty TXT files preserved           PASS"
    )

    print(
        "ZIP files directly at root          PASS"
    )

    print(
        "ZIP CRC validation                  PASS"
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

    print()

    print(
        "Frozen best.pt gate state:"
    )

    print(
        "  Depth:",
        gate_stats[
            "depth"
        ],
    )

    print(
        "  IR   :",
        gate_stats[
            "ir"
        ],
    )

    del model
    del wrapper

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 Exp05
Strict Rect val400 evaluation.

Model:
    Exp05 asymmetric gated YOLO11s

Checkpoint:
    runs/exp05_asymmetric_gated_yolo11s_960/weights/best.pt

Protocol:
    fixed val400
    RGB + IR + Depth
    imgsz   = 960
    rect     = Ultralytics-like rectangular batching
    conf     = 0.001
    NMS IoU  = 0.70
    max_det  = 100
    multi_label = True

Metric:
    competition-style 101-point AP
    IoU = 0.50:0.05:0.95

Important:
    This script deliberately reuses the already parity-tested
    Exp04 loading / preprocessing / NMS / evaluator implementation.

Primary reference:
    Exp04 Rect val400 mAP50-95 = 0.380711
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path

import torch
from ultralytics import YOLO

# Required BEFORE YOLO checkpoint deserialization.
from multimodal_gated_model import (
    Exp05AsymmetricGatedDetectionModel,
)

import test_exp04_modality_ablation_val400 as base
import test_exp04_rect_inference_parity_val400 as rect


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "exp05_asymmetric_gated_yolo11s_960"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "exp05_asymmetric_gated_yolo11s_960"
    / "rect_val400"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "summary.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "summary.json"
)

SUMMARY_TXT = (
    OUTPUT_DIR
    / "summary.txt"
)


# ============================================================
# Frozen evaluation protocol
# ============================================================

IMAGE_SIZE = 960

EXPECTED_VAL = 400
EXPECTED_CLASSES = 12
EXPECTED_EXTERNAL_CHANNELS = 5

DEVICE_INDEX = 0

EXP04_RECT_VAL400_MAP50_95 = 0.380711

RGB_BASELINE_MAP50_95 = 0.380843

# Merely for interpretation; never fail evaluation on performance.
CLEAR_GAIN = 0.005
POSITIVE_GAIN = 0.001


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
# Protocol verification
# ============================================================

def verify_protocol() -> None:

    section(
        "Frozen evaluation protocol"
    )

    checks = {
        "base.IMAGE_SIZE":
            base.IMAGE_SIZE,

        "base.CONF_THRESHOLD":
            base.CONF_THRESHOLD,

        "base.IOU_THRESHOLD":
            base.IOU_THRESHOLD,

        "base.MAX_DET":
            base.MAX_DET,

        "rect.IMAGE_SIZE":
            rect.IMAGE_SIZE,

        "rect.RECT_GROUP_BATCH":
            rect.RECT_GROUP_BATCH,

        "rect.RECT_PAD":
            rect.RECT_PAD,

        "rect.FORWARD_BATCH":
            rect.FORWARD_BATCH,
    }

    for key, value in checks.items():

        print(
            f"{key:<30}:",
            value,
        )

    if base.IMAGE_SIZE != 960:
        raise AssertionError(
            "base IMAGE_SIZE changed."
        )

    if abs(
        base.CONF_THRESHOLD
        - 0.001
    ) > 1e-12:
        raise AssertionError(
            "conf threshold changed."
        )

    if abs(
        base.IOU_THRESHOLD
        - 0.70
    ) > 1e-12:
        raise AssertionError(
            "NMS IoU threshold changed."
        )

    if base.MAX_DET != 100:
        raise AssertionError(
            "max_det changed."
        )

    if rect.IMAGE_SIZE != 960:
        raise AssertionError(
            "rect IMAGE_SIZE changed."
        )

    if rect.RECT_GROUP_BATCH != 8:
        raise AssertionError(
            "rect logical batch changed."
        )

    if abs(
        rect.RECT_PAD
        - 0.5
    ) > 1e-12:
        raise AssertionError(
            "rect padding changed."
        )

    pass_line(
        "evaluation protocol unchanged"
    )


# ============================================================
# Model
# ============================================================

def load_exp05_model(
    device: torch.device,
):

    section(
        "Load Exp05 best.pt"
    )

    if not MODEL_PATH.is_file():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
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
        "Checkpoint :",
        MODEL_PATH,
    )

    print(
        "Class      :",
        type(
            model
        ).__name__,
    )

    if not isinstance(
        model,
        Exp05AsymmetricGatedDetectionModel,
    ):

        raise AssertionError(
            "Checkpoint is not "
            "Exp05AsymmetricGatedDetectionModel."
        )

    # --------------------------------------------------------
    # Exp05 has:
    #
    # external input = 5 channels
    # RGB main stem  = 3 channels
    # --------------------------------------------------------

    rgb_first = (
        model
        .model[0]
        .conv
    )

    print(
        "RGB stem in_channels:",
        rgb_first.in_channels,
    )

    if (
        rgb_first.in_channels
        != 3
    ):

        raise AssertionError(
            "Exp05 RGB main stem is not 3-channel."
        )

    external_channels = getattr(
        model,
        "multimodal_channels",
        None,
    )

    print(
        "External channels    :",
        external_channels,
    )

    if (
        external_channels
        != EXPECTED_EXTERNAL_CHANNELS
    ):

        raise AssertionError(
            "Exp05 external input is not 5-channel."
        )

    detect_head = (
        model.model[-1]
    )

    print(
        "Detect nc            :",
        detect_head.nc,
    )

    if (
        detect_head.nc
        != EXPECTED_CLASSES
    ):

        raise AssertionError(
            "Exp05 Detect head is not 12-class."
        )

    strides = tuple(
        float(x)
        for x
        in model.stride.detach().cpu().tolist()
    )

    print(
        "Strides              :",
        strides,
    )

    if strides != (
        8.0,
        16.0,
        32.0,
    ):

        raise AssertionError(
            f"Unexpected model strides: {strides}"
        )

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

    # Inference checkpoint may have requires_grad=False.
    # That is normal and irrelevant here.

    model = (
        model
        .float()
        .to(
            device
        )
        .eval()
    )

    pass_line(
        "Exp05 checkpoint structure"
    )

    return (
        wrapper,
        model,
        gate_stats,
    )


# ============================================================
# Dataset
# ============================================================

def prepare_dataset():

    section(
        "Fixed val400 dataset"
    )

    (
        stems,
        rgb_map,
        ir_map,
        depth_map,
    ) = base.check_data()

    print(
        "Samples:",
        len(
            stems
        ),
    )

    if len(
        stems
    ) != EXPECTED_VAL:

        raise AssertionError(
            f"Expected {EXPECTED_VAL} validation samples, "
            f"got {len(stems)}."
        )

    (
        shape_by_stem,
        shape_counts,
    ) = rect.inspect_shapes(
        stems,
        rgb_map,
    )

    print()
    print(
        "Original image shape distribution:"
    )

    for (
        shape,
        count,
    ) in sorted(
        shape_counts.items()
    ):

        print(
            f"  {shape[0]}x{shape[1]} : {count}"
        )

    pass_line(
        "fixed aligned val400"
    )

    return (
        stems,
        rgb_map,
        ir_map,
        depth_map,
        shape_by_stem,
    )


# ============================================================
# Rect geometry
# ============================================================

def prepare_rect_groups(
    model,
    stems,
    shape_by_stem,
):

    section(
        "Rect batch geometry"
    )

    stride = rect.get_stride(
        model
    )

    print(
        "Max stride:",
        stride,
    )

    groups = rect.build_rect_groups(
        stems,
        shape_by_stem,
        stride,
    )

    print(
        "Logical rect groups:",
        len(
            groups
        ),
    )

    shape_counts = Counter(
        group[
            "shape"
        ]
        for group in groups
    )

    print()
    print(
        "Rect group shape distribution:"
    )

    for (
        shape,
        count,
    ) in sorted(
        shape_counts.items()
    ):

        print(
            f"  {shape[0]}x{shape[1]} : "
            f"{count} groups"
        )

    total_samples = sum(
        len(
            group[
                "stems"
            ]
        )
        for group
        in groups
    )

    if total_samples != EXPECTED_VAL:

        raise AssertionError(
            "Rect grouping lost/duplicated samples: "
            f"{total_samples}"
        )

    pass_line(
        "rect grouping covers val400 exactly once"
    )

    return groups


# ============================================================
# Evaluation
# ============================================================

def run_evaluation(
    model,
    groups,
    stems,
    rgb_map,
    ir_map,
    depth_map,
    device,
):

    section(
        "Exp05 Rect val400 inference"
    )

    ground_truths = {}

    torch.cuda.reset_peak_memory_stats(
        device
    )

    start = time.time()

    with torch.inference_mode():

        (
            predictions,
            inference_seconds,
        ) = rect.run_rect(
            model,
            groups,
            rgb_map,
            ir_map,
            depth_map,
            device,
            ground_truths,
        )

    total_seconds = (
        time.time()
        - start
    )

    if set(
        predictions
    ) != set(
        stems
    ):

        missing = (
            set(
                stems
            )
            - set(
                predictions
            )
        )

        extra = (
            set(
                predictions
            )
            - set(
                stems
            )
        )

        raise AssertionError(
            "Prediction stem mismatch.\n"
            f"missing={sorted(missing)[:20]}\n"
            f"extra={sorted(extra)[:20]}"
        )

    if set(
        ground_truths
    ) != set(
        stems
    ):

        raise AssertionError(
            "Ground-truth collection does not "
            "cover all val400 samples."
        )

    section(
        "Competition-style evaluation"
    )

    result = base.evaluate_mode(
        predictions,
        ground_truths,
        stems,
    )

    peak_memory_gib = (
        torch.cuda
        .max_memory_allocated(
            device
        )
        / 1024**3
    )

    return (
        result,
        inference_seconds,
        total_seconds,
        peak_memory_gib,
    )


# ============================================================
# Save summary
# ============================================================

def save_summary(
    result,
    gate_stats,
    inference_seconds,
    total_seconds,
    peak_memory_gib,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    map50 = float(
        result[
            "map50"
        ]
    )

    map75 = float(
        result[
            "map75"
        ]
    )

    map50_95 = float(
        result[
            "map50_95"
        ]
    )

    delta_exp04 = (
        map50_95
        - EXP04_RECT_VAL400_MAP50_95
    )

    delta_rgb = (
        map50_95
        - RGB_BASELINE_MAP50_95
    )

    if delta_exp04 >= CLEAR_GAIN:

        verdict = (
            "CLEAR_GAIN"
        )

    elif delta_exp04 >= POSITIVE_GAIN:

        verdict = (
            "POSITIVE_GAIN"
        )

    elif delta_exp04 > 0.0:

        verdict = (
            "SMALL_POSITIVE"
        )

    else:

        verdict = (
            "NO_GAIN"
        )

    summary = {
        "experiment":
            "exp05_asymmetric_gated_yolo11s_960",

        "checkpoint":
            str(
                MODEL_PATH
            ),

        "samples":
            EXPECTED_VAL,

        "imgsz":
            IMAGE_SIZE,

        "rect":
            True,

        "conf":
            float(
                base.CONF_THRESHOLD
            ),

        "iou":
            float(
                base.IOU_THRESHOLD
            ),

        "max_det":
            int(
                base.MAX_DET
            ),

        "map50":
            map50,

        "map75":
            map75,

        "map50_95":
            map50_95,

        "p50_best_f1":
            float(
                result.get(
                    "p50_best_f1",
                    float("nan"),
                )
            ),

        "r50_best_f1":
            float(
                result.get(
                    "r50_best_f1",
                    float("nan"),
                )
            ),

        "best_conf50":
            float(
                result.get(
                    "best_conf50",
                    float("nan"),
                )
            ),

        "prediction_boxes":
            int(
                result.get(
                    "prediction_boxes",
                    0,
                )
            ),

        "empty_images":
            int(
                result.get(
                    "empty_images",
                    0,
                )
            ),

        "exp04_rect_map50_95":
            EXP04_RECT_VAL400_MAP50_95,

        "delta_vs_exp04_rect":
            delta_exp04,

        "rgb_baseline_map50_95":
            RGB_BASELINE_MAP50_95,

        "delta_vs_rgb":
            delta_rgb,

        "verdict":
            verdict,

        "inference_seconds":
            float(
                inference_seconds
            ),

        "total_seconds":
            float(
                total_seconds
            ),

        "peak_cuda_memory_gib":
            float(
                peak_memory_gib
            ),

        "depth_gate":
            gate_stats[
                "depth"
            ],

        "ir_gate":
            gate_stats[
                "ir"
            ],
    }

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    with SUMMARY_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        fieldnames = (
            "map50",
            "map75",
            "map50_95",
            "delta_vs_exp04_rect",
            "delta_vs_rgb",
            "p50_best_f1",
            "r50_best_f1",
            "best_conf50",
            "prediction_boxes",
            "empty_images",
            "inference_seconds",
            "peak_cuda_memory_gib",
            "verdict",
        )

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerow(
            {
                key:
                    summary[
                        key
                    ]
                for key
                in fieldnames
            }
        )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    lines = [
        "AIC2026 Exp05 Rect val400",
        "",
        f"checkpoint = {MODEL_PATH}",
        "",
        "Protocol:",
        "  val      = fixed 400",
        "  imgsz    = 960",
        "  rect     = True",
        "  conf     = 0.001",
        "  iou      = 0.70",
        "  max_det  = 100",
        "",
        "Metrics:",
        f"  mAP50    = {map50:.9f}",
        f"  mAP75    = {map75:.9f}",
        f"  mAP50-95 = {map50_95:.9f}",
        "",
        "Reference:",
        (
            "  Exp04 Rect val400 = "
            f"{EXP04_RECT_VAL400_MAP50_95:.9f}"
        ),
        (
            "  delta vs Exp04    = "
            f"{delta_exp04:+.9f}"
        ),
        (
            "  RGB baseline      = "
            f"{RGB_BASELINE_MAP50_95:.9f}"
        ),
        (
            "  delta vs RGB      = "
            f"{delta_rgb:+.9f}"
        ),
        "",
        f"Verdict = {verdict}",
        "",
        (
            "Depth gate mean = "
            f"{gate_stats['depth']['mean']:.9f}"
        ),
        (
            "IR gate mean    = "
            f"{gate_stats['ir']['mean']:.9f}"
        ),
    ]

    SUMMARY_TXT.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )

    return summary


# ============================================================
# Main
# ============================================================

def main() -> None:

    section(
        "AIC2026 Exp05 - Strict Rect val400"
    )

    print(
        "Project root:",
        PROJECT_ROOT,
    )

    print(
        "Checkpoint  :",
        MODEL_PATH,
    )

    print(
        "Output      :",
        OUTPUT_DIR,
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA unavailable."
        )

    device = torch.device(
        f"cuda:{DEVICE_INDEX}"
    )

    print(
        "GPU         :",
        torch.cuda.get_device_name(
            DEVICE_INDEX
        ),
    )

    verify_protocol()

    (
        wrapper,
        model,
        gate_stats,
    ) = load_exp05_model(
        device
    )

    (
        stems,
        rgb_map,
        ir_map,
        depth_map,
        shape_by_stem,
    ) = prepare_dataset()

    groups = prepare_rect_groups(
        model,
        stems,
        shape_by_stem,
    )

    (
        result,
        inference_seconds,
        total_seconds,
        peak_memory_gib,
    ) = run_evaluation(
        model,
        groups,
        stems,
        rgb_map,
        ir_map,
        depth_map,
        device,
    )

    summary = save_summary(
        result,
        gate_stats,
        inference_seconds,
        total_seconds,
        peak_memory_gib,
    )

    section(
        "FINAL RESULT"
    )

    print(
        "mAP50     :",
        f"{summary['map50']:.9f}",
    )

    print(
        "mAP75     :",
        f"{summary['map75']:.9f}",
    )

    print(
        "mAP50-95  :",
        f"{summary['map50_95']:.9f}",
    )

    print()

    print(
        "Exp04 Rect:",
        f"{EXP04_RECT_VAL400_MAP50_95:.9f}",
    )

    print(
        "Delta     :",
        f"{summary['delta_vs_exp04_rect']:+.9f}",
    )

    print()

    print(
        "RGB base  :",
        f"{RGB_BASELINE_MAP50_95:.9f}",
    )

    print(
        "Delta RGB :",
        f"{summary['delta_vs_rgb']:+.9f}",
    )

    print()

    print(
        "Prediction boxes:",
        summary[
            "prediction_boxes"
        ],
    )

    print(
        "Empty images    :",
        summary[
            "empty_images"
        ],
    )

    print(
        "Inference time  :",
        f"{inference_seconds:.3f}s",
    )

    print(
        "Peak CUDA memory:",
        f"{peak_memory_gib:.3f} GiB",
    )

    print()

    print(
        "Verdict:",
        summary[
            "verdict"
        ],
    )

    print()

    print(
        "Saved:"
    )

    print(
        " ",
        SUMMARY_CSV,
    )

    print(
        " ",
        SUMMARY_JSON,
    )

    print(
        " ",
        SUMMARY_TXT,
    )

    print()

    print(
        "STATUS = PASS"
    )

    del wrapper
    del model

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

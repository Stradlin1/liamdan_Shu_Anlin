#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 Exp04b-v2 Rect Val400 A/B

Strict apples-to-apples comparison:

A:
    Exp04 original best.pt

B:
    Exp04b-v2 no-Mosaic / no-warmup best.pt

Both checkpoints use the EXACT SAME validated Rect inference pipeline:
    input       = [R,G,B,IR,Depth]
    imgsz       = 960
    rect group  = 8
    forward     = 4
    rect pad    = 0.5
    FP32 / 255
    conf        = 0.001
    NMS IoU     = 0.70
    max_det     = 100
    multi_label = True
    NMS         = per-image

Evaluation:
    fixed val400
    101-point AP
    IoU = 0.50:0.05:0.95

Decision:
    Exp04b-v2 must improve Rect mAP50-95 by >= +0.001
    to become a leaderboard submission candidate.

This script intentionally imports the already validated Rect implementation
rather than reimplementing preprocessing/NMS/coordinate restoration.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import test_exp04_modality_ablation_val400 as base
import test_exp04_rect_inference_parity_val400 as rect


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

EXP04_WEIGHTS = (
    PROJECT_ROOT
    / "runs"
    / "exp04_rgbid_early5_yolo11s_960"
    / "weights"
    / "best.pt"
)

EXP04B_V2_WEIGHTS = (
    PROJECT_ROOT
    / "runs"
    / "exp04b_v2_rgbid_early5_nomosaic_nowarmup_ft_960"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "exp04b_v2_rgbid_early5_nomosaic_nowarmup_ft_960"
    / "rect_val400"
)


# ============================================================
# Frozen references
# ============================================================

EXPECTED_VAL = 400
EXPECTED_CHANNELS = 5
EXPECTED_CLASSES = 12

# Previous validated manual Rect score.
# Only used as a sanity reference.
EXP04_PREVIOUS_RECT_MAP5095 = 0.380711

# Current leaderboard baseline.
EXP04_LEADERBOARD = 45.614

# Require at least this much local Rect gain before considering submission.
MIN_SUBMISSION_GAIN = 0.001

# If the newly rerun Exp04 differs from historical Rect result by more than
# this, stop interpretation and investigate parity first.
EXP04_REFERENCE_TOL = 0.003


# ============================================================
# Helpers
# ============================================================

def section(
    title: str,
) -> None:

    print()
    print("=" * 92)
    print(title)
    print("=" * 92)


def check_checkpoint(
    path: Path,
    device: torch.device,
):

    if not path.is_file():

        raise FileNotFoundError(
            f"Checkpoint not found:\n  {path}"
        )

    _, model = base.load_model(
        path,
        device,
    )

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
            f"{path.name} is not a "
            f"{EXPECTED_CHANNELS}-channel model.\n"
            f"in_channels = {first_conv.in_channels}"
        )

    nc = getattr(
        detect_head,
        "nc",
        None,
    )

    if nc != EXPECTED_CLASSES:

        raise RuntimeError(
            f"{path.name} class count mismatch.\n"
            f"Expected = {EXPECTED_CLASSES}\n"
            f"Actual   = {nc}"
        )

    return model


def print_result(
    name: str,
    result: dict,
    seconds: float,
) -> None:

    print(
        f"{name:12s}"
        f" P={result['p50_best_f1']:.6f}"
        f" R={result['r50_best_f1']:.6f}"
        f" mAP50={result['map50']:.6f}"
        f" mAP75={result['map75']:.6f}"
        f" mAP50-95={result['map50_95']:.6f}"
        f" time={seconds:.2f}s"
    )


# ============================================================
# Single-checkpoint Rect evaluation
# ============================================================

def evaluate_checkpoint_rect(
    label: str,
    weights: Path,
    device: torch.device,
    groups,
    stems,
    rgb_map,
    ir_map,
    depth_map,
):

    section(
        f"Rect inference: {label}"
    )

    print(
        "Weights:",
        weights,
    )

    model = check_checkpoint(
        weights,
        device,
    )

    ground_truths = {}

    predictions, elapsed = rect.run_rect(
        model=model,
        groups=groups,
        rgb_map=rgb_map,
        ir_map=ir_map,
        depth_map=depth_map,
        device=device,
        ground_truths=ground_truths,
    )

    if set(
        predictions
    ) != set(
        stems
    ):

        missing = sorted(
            set(stems)
            - set(predictions)
        )

        raise RuntimeError(
            "Missing predictions:\n"
            f"{missing[:20]}"
        )

    if set(
        ground_truths
    ) != set(
        stems
    ):

        missing = sorted(
            set(stems)
            - set(ground_truths)
        )

        raise RuntimeError(
            "Missing GT entries:\n"
            f"{missing[:20]}"
        )

    result = base.evaluate_mode(
        predictions,
        ground_truths,
        stems,
    )

    print()

    print_result(
        label,
        result,
        elapsed,
    )

    del predictions
    del ground_truths
    del model

    if device.type == "cuda":

        torch.cuda.empty_cache()

    return (
        result,
        elapsed,
    )


# ============================================================
# Output
# ============================================================

def save_results(
    exp04_result,
    exp04_time,
    v2_result,
    v2_time,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline = (
        exp04_result[
            "map50_95"
        ]
    )

    candidate = (
        v2_result[
            "map50_95"
        ]
    )

    gain = (
        candidate
        - baseline
    )

    # --------------------------------------------------------
    # summary.csv
    # --------------------------------------------------------

    summary_path = (
        OUTPUT_DIR
        / "summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "p50_best_f1",
                "r50_best_f1",
                "best_conf50",
                "map50",
                "map75",
                "map50_95",
                "delta_vs_exp04",
                "seconds",
            ],
        )

        writer.writeheader()

        writer.writerow(
            {
                "model":
                "exp04",

                "p50_best_f1":
                f"{exp04_result['p50_best_f1']:.9f}",

                "r50_best_f1":
                f"{exp04_result['r50_best_f1']:.9f}",

                "best_conf50":
                f"{exp04_result['best_conf50']:.9f}",

                "map50":
                f"{exp04_result['map50']:.9f}",

                "map75":
                f"{exp04_result['map75']:.9f}",

                "map50_95":
                f"{exp04_result['map50_95']:.9f}",

                "delta_vs_exp04":
                "+0.000000000",

                "seconds":
                f"{exp04_time:.3f}",
            }
        )

        writer.writerow(
            {
                "model":
                "exp04b_v2",

                "p50_best_f1":
                f"{v2_result['p50_best_f1']:.9f}",

                "r50_best_f1":
                f"{v2_result['r50_best_f1']:.9f}",

                "best_conf50":
                f"{v2_result['best_conf50']:.9f}",

                "map50":
                f"{v2_result['map50']:.9f}",

                "map75":
                f"{v2_result['map75']:.9f}",

                "map50_95":
                f"{v2_result['map50_95']:.9f}",

                "delta_vs_exp04":
                f"{gain:+.9f}",

                "seconds":
                f"{v2_time:.3f}",
            }
        )

    # --------------------------------------------------------
    # per_class.csv
    # --------------------------------------------------------

    per_class_path = (
        OUTPUT_DIR
        / "per_class.csv"
    )

    exp04_pc = {
        row[
            "class_id"
        ]:
        row
        for row
        in exp04_result[
            "per_class"
        ]
    }

    v2_pc = {
        row[
            "class_id"
        ]:
        row
        for row
        in v2_result[
            "per_class"
        ]
    }

    with per_class_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_id",
                "class_name",
                "gt_count",
                "exp04_ap50",
                "exp04_ap75",
                "exp04_ap50_95",
                "exp04b_v2_ap50",
                "exp04b_v2_ap75",
                "exp04b_v2_ap50_95",
                "delta_ap50_95",
            ],
        )

        writer.writeheader()

        for class_id in range(
            EXPECTED_CLASSES
        ):

            a = exp04_pc[
                class_id
            ]

            b = v2_pc[
                class_id
            ]

            writer.writerow(
                {
                    "class_id":
                    class_id,

                    "class_name":
                    a[
                        "class_name"
                    ],

                    "gt_count":
                    a[
                        "gt_count"
                    ],

                    "exp04_ap50":
                    f"{a['ap50']:.9f}",

                    "exp04_ap75":
                    f"{a['ap75']:.9f}",

                    "exp04_ap50_95":
                    f"{a['ap50_95']:.9f}",

                    "exp04b_v2_ap50":
                    f"{b['ap50']:.9f}",

                    "exp04b_v2_ap75":
                    f"{b['ap75']:.9f}",

                    "exp04b_v2_ap50_95":
                    f"{b['ap50_95']:.9f}",

                    "delta_ap50_95":
                    f"{b['ap50_95'] - a['ap50_95']:+.9f}",
                }
            )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    json_path = (
        OUTPUT_DIR
        / "results.json"
    )

    payload = {
        "protocol": {
            "val_samples":
            EXPECTED_VAL,

            "imgsz":
            rect.IMAGE_SIZE,

            "rect_group_batch":
            rect.RECT_GROUP_BATCH,

            "forward_batch":
            rect.FORWARD_BATCH,

            "rect_pad":
            rect.RECT_PAD,

            "conf":
            base.CONF_THRESHOLD,

            "nms_iou":
            base.IOU_THRESHOLD,

            "max_det":
            base.MAX_DET,

            "multi_label":
            True,

            "ap":
            "101-point",

            "iou_thresholds":
            [
                float(x)
                for x
                in base.IOU_THRESHOLDS
            ],
        },

        "weights": {
            "exp04":
            str(
                EXP04_WEIGHTS
            ),

            "exp04b_v2":
            str(
                EXP04B_V2_WEIGHTS
            ),
        },

        "historical": {
            "exp04_previous_rect_map50_95":
            EXP04_PREVIOUS_RECT_MAP5095,

            "exp04_leaderboard":
            EXP04_LEADERBOARD,
        },

        "results": {
            "exp04": {
                key:
                value
                for key, value
                in exp04_result.items()
                if key
                != "per_class"
            },

            "exp04b_v2": {
                key:
                value
                for key, value
                in v2_result.items()
                if key
                != "per_class"
            },

            "delta_map50_95":
            gain,

            "submission_threshold":
            MIN_SUBMISSION_GAIN,

            "submission_candidate":
            bool(
                gain
                >= MIN_SUBMISSION_GAIN
            ),
        },
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

    print()

    print(
        "Saved:",
        summary_path,
    )

    print(
        "Saved:",
        per_class_path,
    )

    print(
        "Saved:",
        json_path,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Strict Rect val400 comparison: "
            "Exp04 vs Exp04b-v2."
        )
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Check checkpoints/data/Rect geometry "
            "without inference."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow replacing an existing "
            "rect_val400 output directory."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Output protection
    # --------------------------------------------------------

    if (
        OUTPUT_DIR.exists()
        and any(
            OUTPUT_DIR.iterdir()
        )
    ):

        if not args.overwrite:

            raise RuntimeError(
                "Output directory already contains files:\n"
                f"  {OUTPUT_DIR}\n\n"
                "Use --overwrite only if you intentionally "
                "want to replace the previous evaluation."
            )

        shutil.rmtree(
            OUTPUT_DIR
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA unavailable."
        )

    device = torch.device(
        "cuda:0"
    )

    section(
        "Exp04 vs Exp04b-v2 Rect Val400"
    )

    print(
        "Project root       :",
        PROJECT_ROOT,
    )

    print(
        "Exp04 weights      :",
        EXP04_WEIGHTS,
    )

    print(
        "Exp04b-v2 weights  :",
        EXP04B_V2_WEIGHTS,
    )

    print(
        "Output             :",
        OUTPUT_DIR,
    )

    print(
        "Device             :",
        device,
    )

    print(
        "GPU                :",
        torch.cuda.get_device_name(
            0
        ),
    )

    print()

    print(
        "Rect imgsz         :",
        rect.IMAGE_SIZE,
    )

    print(
        "Rect group batch   :",
        rect.RECT_GROUP_BATCH,
    )

    print(
        "Forward batch      :",
        rect.FORWARD_BATCH,
    )

    print(
        "Rect pad           :",
        rect.RECT_PAD,
    )

    print()

    print(
        "conf               :",
        base.CONF_THRESHOLD,
    )

    print(
        "NMS IoU            :",
        base.IOU_THRESHOLD,
    )

    print(
        "max_det            :",
        base.MAX_DET,
    )

    print(
        "multi_label        : True"
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
    ) = base.check_data()

    if len(
        stems
    ) != EXPECTED_VAL:

        raise RuntimeError(
            f"Expected {EXPECTED_VAL} validation samples, "
            f"got {len(stems)}."
        )

    print(
        "Val samples        :",
        len(
            stems
        ),
    )

    print(
        "First / last       :",
        stems[0],
        "/",
        stems[-1],
    )

    # --------------------------------------------------------
    # Check both checkpoints
    # --------------------------------------------------------

    section(
        "Checkpoint audit"
    )

    model_a = check_checkpoint(
        EXP04_WEIGHTS,
        device,
    )

    model_b = check_checkpoint(
        EXP04B_V2_WEIGHTS,
        device,
    )

    print(
        "[PASS] Exp04 best.pt = "
        "5-channel / 12-class"
    )

    print(
        "[PASS] Exp04b-v2 best.pt = "
        "5-channel / 12-class"
    )

    stride_a = (
        rect.get_stride(
            model_a
        )
    )

    stride_b = (
        rect.get_stride(
            model_b
        )
    )

    if stride_a != stride_b:

        raise RuntimeError(
            "Model strides differ:\n"
            f"Exp04    = {stride_a}\n"
            f"Exp04b-v2= {stride_b}"
        )

    stride = stride_a

    print(
        "Model stride       :",
        stride,
    )

    del model_a
    del model_b

    torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Rect geometry
    # --------------------------------------------------------

    section(
        "Rect geometry audit"
    )

    (
        shape_by_stem,
        source_counts,
    ) = rect.inspect_shapes(
        stems,
        rgb_map,
    )

    for (
        shape,
        count,
    ) in sorted(
        source_counts.items()
    ):

        print(
            f"source "
            f"{shape[0]}x{shape[1]}: "
            f"{count}"
        )

    groups = rect.build_rect_groups(
        stems,
        shape_by_stem,
        stride,
    )

    rect_counts = Counter(
        group[
            "shape"
        ]
        for group
        in groups
    )

    for (
        shape,
        count,
    ) in sorted(
        rect_counts.items()
    ):

        print(
            f"rect   "
            f"{shape[0]}x{shape[1]}: "
            f"{count} logical batches"
        )

    print(
        "Logical rect groups:",
        len(
            groups
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

    # ========================================================
    # Strict A/B
    # ========================================================

    exp04_result, exp04_time = (
        evaluate_checkpoint_rect(
            label="Exp04",
            weights=EXP04_WEIGHTS,
            device=device,
            groups=groups,
            stems=stems,
            rgb_map=rgb_map,
            ir_map=ir_map,
            depth_map=depth_map,
        )
    )

    v2_result, v2_time = (
        evaluate_checkpoint_rect(
            label="Exp04b-v2",
            weights=EXP04B_V2_WEIGHTS,
            device=device,
            groups=groups,
            stems=stems,
            rgb_map=rgb_map,
            ir_map=ir_map,
            depth_map=depth_map,
        )
    )

    # ========================================================
    # Decision
    # ========================================================

    section(
        "Strict Rect A/B Result"
    )

    print_result(
        "Exp04",
        exp04_result,
        exp04_time,
    )

    print_result(
        "Exp04b-v2",
        v2_result,
        v2_time,
    )

    baseline = (
        exp04_result[
            "map50_95"
        ]
    )

    candidate = (
        v2_result[
            "map50_95"
        ]
    )

    delta50 = (
        v2_result[
            "map50"
        ]
        - exp04_result[
            "map50"
        ]
    )

    delta75 = (
        v2_result[
            "map75"
        ]
        - exp04_result[
            "map75"
        ]
    )

    delta5095 = (
        candidate
        - baseline
    )

    historical_diff = (
        baseline
        - EXP04_PREVIOUS_RECT_MAP5095
    )

    print()

    print(
        "Historical Exp04 Rect :",
        f"{EXP04_PREVIOUS_RECT_MAP5095:.6f}",
    )

    print(
        "Current Exp04 Rect    :",
        f"{baseline:.6f}",
    )

    print(
        "Parity difference     :",
        f"{historical_diff:+.6f}",
    )

    parity_ok = (
        abs(
            historical_diff
        )
        <= EXP04_REFERENCE_TOL
    )

    print(
        "Exp04 parity          :",
        (
            "PASS"
            if parity_ok
            else "REVIEW"
        ),
    )

    print()

    print(
        "Delta mAP50           :",
        f"{delta50:+.6f}",
    )

    print(
        "Delta mAP75           :",
        f"{delta75:+.6f}",
    )

    print(
        "Delta mAP50-95        :",
        f"{delta5095:+.6f}",
    )

    print()

    print(
        "Required gain         :",
        f"+{MIN_SUBMISSION_GAIN:.6f}",
    )

    submission_candidate = (
        parity_ok
        and delta5095
        >= MIN_SUBMISSION_GAIN
    )

    print()

    print(
        "EXP04B-V2 TEST SUBMISSION CANDIDATE =",
        (
            "YES"
            if submission_candidate
            else "NO"
        ),
    )

    # ========================================================
    # Per-class deltas
    # ========================================================

    section(
        "Per-class AP50-95 delta"
    )

    exp04_pc = {
        row[
            "class_id"
        ]:
        row
        for row
        in exp04_result[
            "per_class"
        ]
    }

    v2_pc = {
        row[
            "class_id"
        ]:
        row
        for row
        in v2_result[
            "per_class"
        ]
    }

    rows = []

    for class_id in range(
        EXPECTED_CLASSES
    ):

        a = exp04_pc[
            class_id
        ]

        b = v2_pc[
            class_id
        ]

        delta = (
            b[
                "ap50_95"
            ]
            - a[
                "ap50_95"
            ]
        )

        rows.append(
            (
                delta,
                class_id,
                a[
                    "class_name"
                ],
                a[
                    "gt_count"
                ],
                a[
                    "ap50_95"
                ],
                b[
                    "ap50_95"
                ],
            )
        )

    rows.sort(
        reverse=True
    )

    for (
        delta,
        class_id,
        name,
        gt_count,
        old_ap,
        new_ap,
    ) in rows:

        print(
            f"{class_id:2d} "
            f"{name:12s} "
            f"GT={gt_count:4d} "
            f"Exp04={old_ap:.6f} "
            f"V2={new_ap:.6f} "
            f"delta={delta:+.6f}"
        )

    # ========================================================
    # Save
    # ========================================================

    save_results(
        exp04_result=exp04_result,
        exp04_time=exp04_time,
        v2_result=v2_result,
        v2_time=v2_time,
    )

    # ========================================================
    # Final
    # ========================================================

    section(
        "FINAL"
    )

    if not parity_ok:

        print(
            "RESULT = REVIEW"
        )

        print(
            "Reason: rerun Exp04 Rect result "
            "does not reproduce the previous "
            "0.380711 closely enough."
        )

        print(
            "Do not judge Exp04b-v2 yet."
        )

    elif submission_candidate:

        print(
            "RESULT = EXP04B-V2 POSITIVE"
        )

        print(
            "Exp04b-v2 exceeded Exp04 by at least "
            f"{MIN_SUBMISSION_GAIN:.3f} "
            "mAP50-95 under the exact same Rect protocol."
        )

        print(
            "Next: generate an Exp04b-v2 Rect "
            "test submission."
        )

    else:

        print(
            "RESULT = EXP04B CLOSED"
        )

        print(
            "Exp04b-v2 did not produce a meaningful "
            "Rect val400 improvement."
        )

        print(
            "Do NOT submit Exp04b-v2."
        )

        print(
            "Keep Exp04 + Rect leaderboard 45.614 "
            "as the main baseline."
        )

        print(
            "Next: Exp05 asymmetric gated fusion."
        )

    print()

    print(
        "STATUS = PASS"
    )


if __name__ == "__main__":
    main()

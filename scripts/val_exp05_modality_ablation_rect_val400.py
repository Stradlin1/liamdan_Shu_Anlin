#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 Exp05
Rect val400 modality ablation.

Same trained checkpoint, NO retraining.

Modes:
    full:
        RGB + IR + Depth

    no_ir:
        RGB + 0 + Depth

    no_depth:
        RGB + IR + 0

    rgb_only:
        RGB + 0 + 0

Frozen protocol:
    fixed val400
    imgsz       = 960
    rect         = True
    conf         = 0.001
    NMS IoU      = 0.70
    max_det      = 100
    multi_label  = True
    competition 101-point AP
    IoU          = 0.50:0.05:0.95

This script reuses the already validated Exp04 evaluator and
Exp05 Rect preprocessing path.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import torch

import test_exp04_modality_ablation_val400 as base
import test_exp04_rect_inference_parity_val400 as rect
import val_exp05_asymmetric_gated_rect_val400 as exp05


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "exp05_asymmetric_gated_yolo11s_960"
    / "modality_ablation_rect_val400"
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
# Frozen references
# ============================================================

EXP05_FULL_RECT_REFERENCE = (
    0.40306017849386494
)

FULL_PARITY_TOLERANCE = (
    0.001
)

EXPECTED_VAL = 400

MODES = (
    "full",
    "no_ir",
    "no_depth",
    "rgb_only",
)

MODE_LABELS = {
    "full":
        "RGB + IR + Depth",

    "no_ir":
        "RGB + 0 + Depth",

    "no_depth":
        "RGB + IR + 0",

    "rgb_only":
        "RGB + 0 + 0",
}


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


# ============================================================
# Rect inference for one modality mode
# ============================================================

def run_mode(
    model,
    mode: str,
    groups,
    rgb_map,
    ir_map,
    depth_map,
    device,
    ground_truths,
):

    predictions = {}

    total_forward = sum(
        math.ceil(
            len(
                group[
                    "stems"
                ]
            )
            / rect.FORWARD_BATCH
        )
        for group
        in groups
    )

    forward_index = 0
    done = 0

    start_time = (
        time.time()
    )

    with torch.inference_mode():

        for group in groups:

            stems = (
                group[
                    "stems"
                ]
            )

            for start in range(
                0,
                len(
                    stems
                ),
                rect.FORWARD_BATCH,
            ):

                forward_index += 1

                batch_stems = (
                    stems[
                        start:
                        start
                        + rect.FORWARD_BATCH
                    ]
                )

                (
                    batch,
                    metas,
                ) = rect.prepare_rect_batch(
                    batch_stems,
                    group[
                        "shape"
                    ],
                    rgb_map,
                    ir_map,
                    depth_map,
                    device,
                )

                # --------------------------------------------
                # Same trained checkpoint.
                # Only zero selected external modalities.
                # --------------------------------------------

                ablated_batch = (
                    base.apply_ablation(
                        batch,
                        mode,
                    )
                )

                detections = (
                    base.predict_batch(
                        model,
                        ablated_batch,
                    )
                )

                rect.collect_outputs(
                    detections,
                    metas,
                    predictions,
                    ground_truths,
                )

                done += len(
                    batch_stems
                )

                del detections
                del ablated_batch
                del batch

                if (
                    forward_index == 1
                    or forward_index % 10 == 0
                    or forward_index
                    == total_forward
                ):

                    h, w = (
                        group[
                            "shape"
                        ]
                    )

                    print(
                        f"[{mode:<8}] "
                        f"{forward_index:3d}/"
                        f"{total_forward}  "
                        f"{done:3d}/{EXPECTED_VAL}  "
                        f"shape={h}x{w}"
                    )

    elapsed = (
        time.time()
        - start_time
    )

    return (
        predictions,
        elapsed,
    )


# ============================================================
# Serialization
# ============================================================

def result_row(
    mode: str,
    result,
    elapsed: float,
    full_map: float,
):

    value = float(
        result[
            "map50_95"
        ]
    )

    return {
        "mode":
            mode,

        "label":
            MODE_LABELS[
                mode
            ],

        "map50":
            float(
                result[
                    "map50"
                ]
            ),

        "map75":
            float(
                result[
                    "map75"
                ]
            ),

        "map50_95":
            value,

        "drop_from_full":
            (
                full_map
                - value
            ),

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

        "elapsed_seconds":
            float(
                elapsed
            ),
    }


def save_results(
    results,
    timings,
    gate_stats,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_map = float(
        results[
            "full"
        ][
            "map50_95"
        ]
    )

    rows = [
        result_row(
            mode,
            results[
                mode
            ],
            timings[
                mode
            ],
            full_map,
        )
        for mode
        in MODES
    ]

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    with SUMMARY_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    payload = {
        "checkpoint":
            str(
                exp05.MODEL_PATH
            ),

        "protocol": {
            "samples":
                EXPECTED_VAL,

            "imgsz":
                base.IMAGE_SIZE,

            "rect":
                True,

            "conf":
                base.CONF_THRESHOLD,

            "iou":
                base.IOU_THRESHOLD,

            "max_det":
                base.MAX_DET,
        },

        "exp05_full_rect_reference":
            EXP05_FULL_RECT_REFERENCE,

        "gate_stats":
            gate_stats,

        "results": {
            row[
                "mode"
            ]:
                row
            for row
            in rows
        },
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # TXT
    # --------------------------------------------------------

    lines = [
        "AIC2026 Exp05 Rect val400 modality ablation",
        "",
        (
            "Checkpoint: "
            f"{exp05.MODEL_PATH}"
        ),
        "",
        "Protocol:",
        "  val     = fixed 400",
        "  imgsz   = 960",
        "  rect    = True",
        "  conf    = 0.001",
        "  iou     = 0.70",
        "  max_det = 100",
        "",
        "Results:",
        "",
    ]

    for row in rows:

        lines.extend(
            [
                (
                    f"{row['mode']}: "
                    f"{row['label']}"
                ),
                (
                    "  mAP50    = "
                    f"{row['map50']:.9f}"
                ),
                (
                    "  mAP75    = "
                    f"{row['map75']:.9f}"
                ),
                (
                    "  mAP50-95 = "
                    f"{row['map50_95']:.9f}"
                ),
                (
                    "  drop     = "
                    f"{row['drop_from_full']:+.9f}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "Interpretation:",
            (
                "  IR drop        = "
                f"{full_map - results['no_ir']['map50_95']:+.9f}"
            ),
            (
                "  Depth drop     = "
                f"{full_map - results['no_depth']['map50_95']:+.9f}"
            ),
            (
                "  Aux total drop = "
                f"{full_map - results['rgb_only']['map50_95']:+.9f}"
            ),
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
    )

    SUMMARY_TXT.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )

    return rows


# ============================================================
# Main
# ============================================================

def main():

    section(
        "AIC2026 Exp05 - Rect Modality Ablation"
    )

    print(
        "Checkpoint:",
        exp05.MODEL_PATH,
    )

    print(
        "Output    :",
        OUTPUT_DIR,
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA unavailable."
        )

    device = torch.device(
        "cuda:0"
    )

    print(
        "GPU       :",
        torch.cuda.get_device_name(
            0
        ),
    )

    # --------------------------------------------------------
    # Frozen protocol checks
    # --------------------------------------------------------

    exp05.verify_protocol()

    # --------------------------------------------------------
    # Exp05 checkpoint
    # --------------------------------------------------------

    (
        wrapper,
        model,
        gate_stats,
    ) = exp05.load_exp05_model(
        device
    )

    # --------------------------------------------------------
    # Fixed val400 + Rect grouping
    # --------------------------------------------------------

    (
        stems,
        rgb_map,
        ir_map,
        depth_map,
        shape_by_stem,
    ) = exp05.prepare_dataset()

    groups = (
        exp05.prepare_rect_groups(
            model,
            stems,
            shape_by_stem,
        )
    )

    # --------------------------------------------------------
    # Run all modes
    # --------------------------------------------------------

    results = {}
    timings = {}

    # Reuse identical GT objects across all four modes.
    ground_truths = {}

    for mode in MODES:

        section(
            f"MODE: {mode} - "
            f"{MODE_LABELS[mode]}"
        )

        torch.cuda.empty_cache()

        (
            predictions,
            elapsed,
        ) = run_mode(
            model,
            mode,
            groups,
            rgb_map,
            ir_map,
            depth_map,
            device,
            ground_truths,
        )

        if set(
            predictions
        ) != set(
            stems
        ):

            raise AssertionError(
                f"{mode}: prediction set "
                "does not match val400."
            )

        result = (
            base.evaluate_mode(
                predictions,
                ground_truths,
                stems,
            )
        )

        results[
            mode
        ] = result

        timings[
            mode
        ] = elapsed

        print()
        print(
            "mAP50    :",
            f"{result['map50']:.9f}",
        )

        print(
            "mAP75    :",
            f"{result['map75']:.9f}",
        )

        print(
            "mAP50-95 :",
            f"{result['map50_95']:.9f}",
        )

        print(
            "Boxes     :",
            result.get(
                "prediction_boxes",
                0,
            ),
        )

        del predictions

    # --------------------------------------------------------
    # Full-mode parity guard
    # --------------------------------------------------------

    section(
        "Full-mode parity check"
    )

    full_map = float(
        results[
            "full"
        ][
            "map50_95"
        ]
    )

    full_diff = (
        full_map
        - EXP05_FULL_RECT_REFERENCE
    )

    print(
        "Previous Exp05 Full Rect:",
        f"{EXP05_FULL_RECT_REFERENCE:.9f}",
    )

    print(
        "Current Full Rect       :",
        f"{full_map:.9f}",
    )

    print(
        "Difference              :",
        f"{full_diff:+.9f}",
    )

    if abs(
        full_diff
    ) > FULL_PARITY_TOLERANCE:

        raise AssertionError(
            "Full-mode Rect result no longer "
            "matches the validated Exp05 reference."
        )

    print(
        "[PASS] Full-mode evaluator parity"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    rows = save_results(
        results,
        timings,
        gate_stats,
    )

    # --------------------------------------------------------
    # Final interpretation
    # --------------------------------------------------------

    ir_drop = (
        full_map
        - float(
            results[
                "no_ir"
            ][
                "map50_95"
            ]
        )
    )

    depth_drop = (
        full_map
        - float(
            results[
                "no_depth"
            ][
                "map50_95"
            ]
        )
    )

    aux_drop = (
        full_map
        - float(
            results[
                "rgb_only"
            ][
                "map50_95"
            ]
        )
    )

    section(
        "FINAL RESULT"
    )

    for row in rows:

        print(
            f"{row['mode']:<10} "
            f"mAP50-95="
            f"{row['map50_95']:.9f}  "
            f"drop="
            f"{row['drop_from_full']:+.9f}"
        )

    print()
    print(
        "IR contribution:"
    )

    print(
        "  Full - No IR    =",
        f"{ir_drop:+.9f}",
    )

    print()
    print(
        "Depth contribution:"
    )

    print(
        "  Full - No Depth =",
        f"{depth_drop:+.9f}",
    )

    print()
    print(
        "Total auxiliary contribution:"
    )

    print(
        "  Full - RGB only =",
        f"{aux_drop:+.9f}",
    )

    print()

    if ir_drop > 0:

        print(
            "[INFO] IR is positively contributing."
        )

    elif ir_drop < 0:

        print(
            "[REVIEW] Removing IR improves AP."
        )

    else:

        print(
            "[INFO] IR has zero measured contribution."
        )

    if depth_drop > 0:

        print(
            "[INFO] Depth is positively contributing."
        )

    elif depth_drop < 0:

        print(
            "[REVIEW] Removing Depth improves AP."
        )

    else:

        print(
            "[INFO] Depth has zero measured contribution."
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

    del model
    del wrapper

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

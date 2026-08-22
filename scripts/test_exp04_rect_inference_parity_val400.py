#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exp04 fixed-val400: square inference vs Ultralytics-like rect inference.

Reuses test_exp04_modality_ablation_val400.py for:
- 5-channel [R,G,B,IR,Depth] loading
- Depth conversion
- fixed val400 checks
- per-image NMS, multi_label=True
- box restoration
- 101-point competition mAP evaluation
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

import test_exp04_modality_ablation_val400 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "exp04_rgbid_early5_yolo11s_960"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "exp04_rgbid_early5_yolo11s_960"
    / "rect_inference_parity_val400"
)

IMAGE_SIZE = 960

FORWARD_BATCH = 4

# Exp04 正式 val 的 batch=8。
# 这个值只用于计算 rect batch shape。
# 真正 GPU forward 仍然 batch=4。
RECT_GROUP_BATCH = 8

# Ultralytics validation dataset 常用 pad=0.5。
RECT_PAD = 0.5

DEVICE_INDEX = 0

EXPECTED_VAL = 400
EXPECTED_CHANNELS = 5
EXPECTED_CLASSES = 12


# ============================================================
# Frozen references for current Exp04 best.pt
# ============================================================

# 最新 modality-ablation 脚本使用同一 evaluator 得到的 Full square。
LATEST_SQUARE_REFERENCE = 0.375390

# Official validator:
OFFICIAL_MAP50 = 0.628168
OFFICIAL_MAP75 = 0.388485
OFFICIAL_MAP50_95 = 0.379090


# Parity tolerances
SQUARE_TOL = 0.003
RECT_TOL = 0.003

# Rect 至少提升这么多，才认为值得继续做 test submission。
MIN_USEFUL_GAIN = 0.001


# ============================================================
# Logging
# ============================================================

def section(title: str) -> None:

    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


# ============================================================
# Model geometry
# ============================================================

def get_stride(model) -> int:

    stride = getattr(
        model,
        "stride",
        None,
    )

    if stride is None:

        return 32

    if torch.is_tensor(
        stride
    ):

        return int(
            stride.max().item()
        )

    return int(
        max(
            stride
        )
    )


# ============================================================
# Dataset geometry inspection
# ============================================================

def read_hw(
    path: Path,
) -> tuple[int, int]:

    image = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    if image is None:

        raise RuntimeError(
            f"Cannot read image: {path}"
        )

    return image.shape[:2]


def inspect_shapes(
    stems,
    rgb_map,
):

    shape_by_stem = {}

    counts = Counter()

    for stem in stems:

        hw = read_hw(
            rgb_map[
                stem
            ]
        )

        shape_by_stem[
            stem
        ] = hw

        counts[
            hw
        ] += 1

    return (
        shape_by_stem,
        counts,
    )


# ============================================================
# Rectangular batch geometry
# ============================================================

def build_rect_groups(
    stems,
    shape_by_stem,
    stride: int,
):
    """
    Reproduce Ultralytics-style rectangular batch geometry.

    Aspect ratio:
        h / w

    Images are sorted by aspect ratio, then one common stride-aligned
    HxW is assigned to every logical rect batch.
    """

    records = [
        (
            stem,
            shape_by_stem[
                stem
            ][0]
            / float(
                shape_by_stem[
                    stem
                ][1]
            ),
        )
        for stem in stems
    ]

    records.sort(
        key=lambda x: x[1]
    )

    groups = []

    for start in range(
        0,
        len(records),
        RECT_GROUP_BATCH,
    ):

        chunk = records[
            start:
            start + RECT_GROUP_BATCH
        ]

        ars = np.asarray(
            [
                x[1]
                for x in chunk
            ],
            dtype=np.float64,
        )

        min_ar = float(
            ars.min()
        )

        max_ar = float(
            ars.max()
        )

        if max_ar < 1.0:

            ratio = np.array(
                [
                    max_ar,
                    1.0,
                ],
                dtype=np.float64,
            )

        elif min_ar > 1.0:

            ratio = np.array(
                [
                    1.0,
                    1.0
                    / min_ar,
                ],
                dtype=np.float64,
            )

        else:

            ratio = np.array(
                [
                    1.0,
                    1.0,
                ],
                dtype=np.float64,
            )

        shape = (
            np.ceil(
                ratio
                * IMAGE_SIZE
                / stride
                + RECT_PAD
            )
            .astype(
                np.int64
            )
            * stride
        )

        groups.append(
            {
                "stems": [
                    x[0]
                    for x in chunk
                ],

                "shape": (
                    int(
                        shape[0]
                    ),
                    int(
                        shape[1]
                    ),
                ),
            }
        )

    return groups


# ============================================================
# BaseDataset-like first resize
# ============================================================

def resize_long_side_like_dataset(
    image: np.ndarray,
):
    """
    Mimic BaseDataset.load_image() rect-mode resize.

    Critical difference from the previous manual square path:

        360x640 source
            ↓
        long side -> 960
            ↓
        ~540x960

    The previous direct square LetterBox(scaleup=False)
    kept the 360x640 source small.

    Returns:
        resized
        scale_x
        scale_y
    """

    h0, w0 = image.shape[:2]

    r = (
        IMAGE_SIZE
        / float(
            max(
                h0,
                w0,
            )
        )
    )

    if abs(
        r
        - 1.0
    ) < 1e-12:

        return (
            image,
            1.0,
            1.0,
        )

    new_w = min(
        math.ceil(
            w0
            * r
        ),
        IMAGE_SIZE,
    )

    new_h = min(
        math.ceil(
            h0
            * r
        ),
        IMAGE_SIZE,
    )

    interpolation = (
        cv2.INTER_LINEAR
        if r > 1.0
        else cv2.INTER_AREA
    )

    resized = cv2.resize(
        image,
        (
            new_w,
            new_h,
        ),
        interpolation=interpolation,
    )

    return (
        resized,
        new_w
        / float(
            w0
        ),
        new_h
        / float(
            h0
        ),
    )


# ============================================================
# Rect batch preparation
# ============================================================

def prepare_rect_batch(
    stems,
    rect_shape,
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
        ) = base.load_5ch(
            rgb_map[
                stem
            ],
            ir_map[
                stem
            ],
            depth_map[
                stem
            ],
        )

        # ----------------------------------------------------
        # Stage 1:
        # BaseDataset-like long-side resize.
        # ----------------------------------------------------

        (
            image,
            pre_sx,
            pre_sy,
        ) = resize_long_side_like_dataset(
            image
        )

        # ----------------------------------------------------
        # Stage 2:
        # Rectangular LetterBox.
        # ----------------------------------------------------

        (
            image,
            lb_sx,
            lb_sy,
            left,
            top,
        ) = base.letterbox_5ch(
            image,
            new_shape=rect_shape,
            scaleup=False,
            padding_value=114,
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
                "stem":
                stem,

                "original_shape":
                original_shape,

                # Original -> pre-resize -> LetterBox
                "scale_x":
                pre_sx
                * lb_sx,

                "scale_y":
                pre_sy
                * lb_sy,

                "left":
                left,

                "top":
                top,
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
# Prediction result collection
# ============================================================

def collect_outputs(
    detections,
    metas,
    predictions,
    ground_truths,
):

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

        if stem not in ground_truths:

            ground_truths[
                stem
            ] = base.read_ground_truth(
                stem,
                meta[
                    "original_shape"
                ],
            )

        if (
            det is None
            or det.numel() == 0
        ):

            predictions[
                stem
            ] = np.zeros(
                (
                    0,
                    6,
                ),
                dtype=np.float64,
            )

            continue

        boxes = base.restore_xyxy_to_original(
            det[
                :,
                :4,
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

        predictions[
            stem
        ] = (
            torch.cat(
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
            .cpu()
            .double()
            .numpy()
        )


# ============================================================
# Existing square inference
# ============================================================

def run_square(
    model,
    stems,
    rgb_map,
    ir_map,
    depth_map,
    device,
    ground_truths,
):

    predictions = {}

    total = math.ceil(
        len(stems)
        / FORWARD_BATCH
    )

    t0 = time.time()

    for (
        bi,
        start,
    ) in enumerate(
        range(
            0,
            len(stems),
            FORWARD_BATCH,
        ),
        start=1,
    ):

        batch_stems = stems[
            start:
            start + FORWARD_BATCH
        ]

        (
            batch,
            metas,
        ) = base.prepare_base_batch(
            batch_stems,
            rgb_map,
            ir_map,
            depth_map,
            device,
        )

        detections = base.predict_batch(
            model,
            batch,
        )

        collect_outputs(
            detections,
            metas,
            predictions,
            ground_truths,
        )

        del batch
        del detections

        if (
            bi == 1
            or bi % 10 == 0
            or bi == total
        ):

            print(
                f"[square] "
                f"{bi:3d}/{total}  "
                f"{min(start + FORWARD_BATCH, len(stems)):3d}/"
                f"{len(stems)}"
            )

    return (
        predictions,
        time.time()
        - t0,
    )


# ============================================================
# Rect inference
# ============================================================

def run_rect(
    model,
    groups,
    rgb_map,
    ir_map,
    depth_map,
    device,
    ground_truths,
):

    predictions = {}

    total_fw = sum(
        math.ceil(
            len(
                group[
                    "stems"
                ]
            )
            / FORWARD_BATCH
        )
        for group in groups
    )

    done = 0

    fw = 0

    t0 = time.time()

    for group in groups:

        for start in range(
            0,
            len(
                group[
                    "stems"
                ]
            ),
            FORWARD_BATCH,
        ):

            fw += 1

            batch_stems = group[
                "stems"
            ][
                start:
                start + FORWARD_BATCH
            ]

            (
                batch,
                metas,
            ) = prepare_rect_batch(
                batch_stems,
                group[
                    "shape"
                ],
                rgb_map,
                ir_map,
                depth_map,
                device,
            )

            detections = base.predict_batch(
                model,
                batch,
            )

            collect_outputs(
                detections,
                metas,
                predictions,
                ground_truths,
            )

            done += len(
                batch_stems
            )

            del batch
            del detections

            if (
                fw == 1
                or fw % 10 == 0
                or fw == total_fw
            ):

                (
                    h,
                    w,
                ) = group[
                    "shape"
                ]

                print(
                    f"[rect]   "
                    f"{fw:3d}/{total_fw}  "
                    f"{done:3d}/{EXPECTED_VAL}  "
                    f"shape={h}x{w}"
                )

    return (
        predictions,
        time.time()
        - t0,
    )


# ============================================================
# CSV
# ============================================================

def save_csv(
    results,
    timings,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    square_map = results[
        "square"
    ][
        "map50_95"
    ]

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    with (
        OUTPUT_DIR
        / "summary.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "mode",
                "map50",
                "map75",
                "map50_95",
                "delta_vs_square",
                "delta_vs_official",
                "seconds",
            ]
        )

        for mode in (
            "square",
            "rect",
        ):

            result = results[
                mode
            ]

            writer.writerow(
                [
                    mode,

                    f"{result['map50']:.9f}",

                    f"{result['map75']:.9f}",

                    f"{result['map50_95']:.9f}",

                    f"{result['map50_95'] - square_map:+.9f}",

                    f"{result['map50_95'] - OFFICIAL_MAP50_95:+.9f}",

                    f"{timings[mode]:.3f}",
                ]
            )

    # --------------------------------------------------------
    # Per-class
    # --------------------------------------------------------

    square_pc = {
        row[
            "class_id"
        ]:
        row

        for row
        in results[
            "square"
        ][
            "per_class"
        ]
    }

    with (
        OUTPUT_DIR
        / "per_class.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "mode",
                "class_id",
                "class_name",
                "gt_count",
                "ap50",
                "ap75",
                "ap50_95",
                "delta_vs_square",
            ]
        )

        for mode in (
            "square",
            "rect",
        ):

            for row in results[
                mode
            ][
                "per_class"
            ]:

                delta = (
                    row[
                        "ap50_95"
                    ]
                    - square_pc[
                        row[
                            "class_id"
                        ]
                    ][
                        "ap50_95"
                    ]
                )

                writer.writerow(
                    [
                        mode,

                        row[
                            "class_id"
                        ],

                        row[
                            "class_name"
                        ],

                        row[
                            "gt_count"
                        ],

                        f"{row['ap50']:.9f}",

                        f"{row['ap75']:.9f}",

                        f"{row['ap50_95']:.9f}",

                        f"{delta:+.9f}",
                    ]
                )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--check-only",
        action="store_true",
    )

    args = parser.parse_args()

    section(
        "Exp04 Rect Inference Parity Val400"
    )

    print(
        "Project root     :",
        PROJECT_ROOT,
    )

    print(
        "Model            :",
        MODEL_PATH,
    )

    print(
        "Output           :",
        OUTPUT_DIR,
    )

    print(
        "Forward batch    :",
        FORWARD_BATCH,
    )

    print(
        "Rect group batch :",
        RECT_GROUP_BATCH,
    )

    print(
        "NMS              : "
        "conf=0.001, "
        "IoU=0.70, "
        "max_det=100, "
        "multi_label=True"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    section(
        "Check frozen val400"
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
            f"Expected {EXPECTED_VAL} "
            f"val samples, "
            f"got {len(stems)}"
        )

    print(
        "Samples          :",
        len(
            stems
        ),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    if not MODEL_PATH.is_file():

        raise FileNotFoundError(
            MODEL_PATH
        )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA unavailable"
        )

    device = torch.device(
        f"cuda:{DEVICE_INDEX}"
    )

    (
        _,
        model,
    ) = base.load_model(
        MODEL_PATH,
        device,
    )

    first_conv = (
        model
        .model[0]
        .conv
    )

    nc = getattr(
        model.model[-1],
        "nc",
        None,
    )

    if (
        first_conv.in_channels
        != EXPECTED_CHANNELS
        or nc
        != EXPECTED_CLASSES
    ):

        raise RuntimeError(
            "Loaded checkpoint is not "
            "Exp04 5ch / 12-class model"
        )

    stride = get_stride(
        model
    )

    print(
        "Stride           :",
        stride,
    )

    # --------------------------------------------------------
    # Rect geometry
    # --------------------------------------------------------

    section(
        "Rect geometry audit"
    )

    (
        shape_by_stem,
        shape_counts,
    ) = inspect_shapes(
        stems,
        rgb_map,
    )

    for (
        (
            h,
            w,
        ),
        count,
    ) in sorted(
        shape_counts.items()
    ):

        print(
            f"source "
            f"{h}x{w}: "
            f"{count}"
        )

    groups = build_rect_groups(
        stems,
        shape_by_stem,
        stride,
    )

    rect_counts = Counter(
        group[
            "shape"
        ]
        for group in groups
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
            f"rect   "
            f"{h}x{w}: "
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
            )
            in shape_by_stem.values()
        }
    )

    print(
        "H/W ratios       :",
        ratios,
    )

    # --------------------------------------------------------
    # First sample audit
    # --------------------------------------------------------

    first = stems[0]

    (
        raw,
        original_shape,
    ) = base.load_5ch(
        rgb_map[
            first
        ],
        ir_map[
            first
        ],
        depth_map[
            first
        ],
    )

    (
        resized,
        sx,
        sy,
    ) = resize_long_side_like_dataset(
        raw
    )

    print(
        "First sample      :",
        first,
    )

    print(
        "Raw HxW           :",
        original_shape,
    )

    print(
        "Long-side HxW     :",
        resized.shape[:2],
    )

    print(
        "Pre-scale x/y     :",
        f"{sx:.6f}",
        f"{sy:.6f}",
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
    # A/B inference
    # --------------------------------------------------------

    ground_truths = {}

    results = {}

    timings = {}

    section(
        "A - Existing square 960"
    )

    (
        predictions,
        timings[
            "square"
        ],
    ) = run_square(
        model,
        stems,
        rgb_map,
        ir_map,
        depth_map,
        device,
        ground_truths,
    )

    results[
        "square"
    ] = base.evaluate_mode(
        predictions,
        ground_truths,
        stems,
    )

    del predictions

    torch.cuda.empty_cache()

    section(
        "B - Ultralytics-like rect"
    )

    (
        predictions,
        timings[
            "rect"
        ],
    ) = run_rect(
        model,
        groups,
        rgb_map,
        ir_map,
        depth_map,
        device,
        ground_truths,
    )

    results[
        "rect"
    ] = base.evaluate_mode(
        predictions,
        ground_truths,
        stems,
    )

    del predictions

    torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    section(
        "Square vs Rect vs Official"
    )

    print(
        f"official     "
        f"mAP50={OFFICIAL_MAP50:.6f}  "
        f"mAP75={OFFICIAL_MAP75:.6f}  "
        f"mAP50-95={OFFICIAL_MAP50_95:.6f}"
    )

    for mode in (
        "square",
        "rect",
    ):

        result = results[
            mode
        ]

        print(
            f"{mode:12s} "
            f"mAP50={result['map50']:.6f}  "
            f"mAP75={result['map75']:.6f}  "
            f"mAP50-95={result['map50_95']:.6f}  "
            f"vsOfficial="
            f"{result['map50_95'] - OFFICIAL_MAP50_95:+.6f}"
        )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    square = results[
        "square"
    ][
        "map50_95"
    ]

    rect = results[
        "rect"
    ][
        "map50_95"
    ]

    square_diff = (
        square
        - LATEST_SQUARE_REFERENCE
    )

    rect_diff = (
        rect
        - OFFICIAL_MAP50_95
    )

    gain = (
        rect
        - square
    )

    old_gap = (
        OFFICIAL_MAP50_95
        - square
    )

    new_gap = (
        OFFICIAL_MAP50_95
        - rect
    )

    section(
        "Parity decision"
    )

    print(
        "Square reference :",
        f"{LATEST_SQUARE_REFERENCE:.6f}",
    )

    print(
        "Current square   :",
        f"{square:.6f}",
    )

    print(
        "Square diff      :",
        f"{square_diff:+.6f}",
    )

    print(
        "Square parity    :",
        (
            "PASS"
            if abs(
                square_diff
            )
            <= SQUARE_TOL
            else "REVIEW"
        ),
    )

    print()

    print(
        "Official rect    :",
        f"{OFFICIAL_MAP50_95:.6f}",
    )

    print(
        "Current rect     :",
        f"{rect:.6f}",
    )

    print(
        "Rect diff        :",
        f"{rect_diff:+.6f}",
    )

    print(
        "Rect parity      :",
        (
            "PASS"
            if abs(
                rect_diff
            )
            <= RECT_TOL
            else "REVIEW"
        ),
    )

    print()

    print(
        "Rect gain        :",
        f"{gain:+.6f}",
    )

    print(
        "Official gap old :",
        f"{old_gap:+.6f}",
    )

    print(
        "Official gap new :",
        f"{new_gap:+.6f}",
    )

    if old_gap > 0:

        print(
            "Gap recovered    :",
            f"{gain / old_gap:.1%}",
        )

    candidate = (
        abs(
            square_diff
        )
        <= SQUARE_TOL
        and gain
        >= MIN_USEFUL_GAIN
        and abs(
            new_gap
        )
        < abs(
            old_gap
        )
    )

    print()

    print(
        "RECT TEST SUBMISSION CANDIDATE =",
        (
            "YES"
            if candidate
            else "NO"
        ),
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_csv(
        results,
        timings,
    )

    print(
        "Saved            :",
        OUTPUT_DIR
        / "summary.csv",
    )

    print(
        "Saved            :",
        OUTPUT_DIR
        / "per_class.csv",
    )

    section(
        "FINAL"
    )

    print(
        "STATUS = PASS"
    )


if __name__ == "__main__":

    main()

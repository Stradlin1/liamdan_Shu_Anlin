#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 Exp04 FINAL Submission Inference

This is the final submission wrapper around the already-tested
infer_exp04_rgbid_early5_960_submit.py pipeline.

Locked final configuration:

    model       = best.pt
    input       = [R, G, B, IR, Depth]
    imgsz       = 960
    batch       = 4
    precision   = FP32
    conf        = 0.001
    iou         = 0.70
    max_det     = 100
    multi_label = True
    NMS         = per-image

The preprocessing / LetterBox / coordinate restoration / TXT / ZIP
implementation is reused from the parity-tested submission script.

Final outputs are written to a NEW directory so previous inference
results are never overwritten.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

# Import torchvision when available.
# Ultralytics NMS may use torchvision.ops.nms on CUDA when torchvision
# is already loaded.
try:
    import torchvision  # noqa: F401
except Exception:
    torchvision = None

import infer_exp04_rgbid_early5_960_submit as base


# ============================================================
# Final locked configuration
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

RUN_DIR = (
    PROJECT_ROOT
    / "runs"
    / "exp04_rgbid_early5_yolo11s_960"
)

FINAL_MODEL_PATH = (
    RUN_DIR
    / "weights"
    / "best.pt"
)

FINAL_OUTPUT_ROOT = (
    RUN_DIR
    / "submission_test_final"
)

FINAL_TXT_DIR = (
    FINAL_OUTPUT_ROOT
    / "txt"
)

FINAL_ZIP_PATH = (
    FINAL_OUTPUT_ROOT
    / "exp04_rgbid_early5_yolo11s_960_submission_final.zip"
)


# ============================================================
# Official / parity-tested inference parameters
# ============================================================

IMAGE_SIZE = 960

BATCH_SIZE = 4

CONF_THRESHOLD = 0.001

IOU_THRESHOLD = 0.70

MAX_DET = 100

EXPECTED_CLASSES = 12

EXPECTED_CHANNELS = 5

EXPECTED_TEST_SAMPLES = 1000

MULTI_LABEL = True


# ============================================================
# Apply final configuration to tested base implementation
# ============================================================

base.MODEL_PATH = FINAL_MODEL_PATH

base.OUTPUT_ROOT = FINAL_OUTPUT_ROOT

base.TXT_DIR = FINAL_TXT_DIR

base.ZIP_PATH = FINAL_ZIP_PATH

base.IMAGE_SIZE = IMAGE_SIZE

base.BATCH_SIZE = BATCH_SIZE

base.CONF_THRESHOLD = CONF_THRESHOLD

base.IOU_THRESHOLD = IOU_THRESHOLD

base.MAX_DET = MAX_DET

base.EXPECTED_CLASSES = EXPECTED_CLASSES

base.EXPECTED_CHANNELS = EXPECTED_CHANNELS

base.EXPECTED_TEST_SAMPLES = EXPECTED_TEST_SAMPLES


# ============================================================
# Final per-image NMS
# ============================================================

@torch.inference_mode()
def predict_batch_final(
    model,
    batch: torch.Tensor,
):
    """
    Batched 5-channel model forward + per-image NMS.

    Important:
        - GPU forward remains batch=4.
        - NMS is performed independently for every image.
        - multi_label=True was selected after val400 parity testing.
        - generous NMS time budget prevents the previous batch-level
          timeout problem from dropping later images.
    """

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

        prediction = raw[
            0
        ]

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
            "Prediction batch size mismatch:\n"
            f"  prediction={prediction.shape[0]}\n"
            f"  input={batch.shape[0]}"
        )

    detections = []

    for image_index in range(
        prediction.shape[0]
    ):

        # Clone so NMS is free to modify coordinates internally
        # without touching the raw batch prediction tensor.
        pred_one = (
            prediction[
                image_index:
                image_index + 1
            ]
            .clone()
        )

        det_one = base.non_max_suppression(
            pred_one,
            conf_thres=CONF_THRESHOLD,
            iou_thres=IOU_THRESHOLD,
            classes=None,
            agnostic=False,

            # Locked after val400 parity test.
            multi_label=MULTI_LABEL,

            max_det=MAX_DET,
            nc=EXPECTED_CLASSES,

            # Per-image NMS.
            # Current Ultralytics computes:
            #
            #   time_limit = 2.0 + max_time_img * batch_size
            #
            # Here batch_size for NMS is always 1.
            max_time_img=10.0,

            max_nms=30000,
        )[0]

        # Explicit confidence ordering.
        if len(
            det_one
        ):

            order = torch.argsort(
                det_one[
                    :,
                    4,
                ],
                descending=True,
            )

            det_one = det_one[
                order
            ][
                :MAX_DET
            ]

        detections.append(
            det_one
        )

    return detections


# Replace only prediction function in base implementation.
base.predict_batch = predict_batch_final


# ============================================================
# Enhanced final submission statistics
# ============================================================

_original_validate_txt_dir = (
    base.validate_txt_dir
)


def validate_txt_dir_final(
    expected_stems: list[str],
) -> dict:
    """
    Run the original strict submission validator, then add useful
    statistics for final submission auditing.
    """

    stats = _original_validate_txt_dir(
        expected_stems
    )

    counts = []

    class_counts = np.zeros(
        EXPECTED_CLASSES,
        dtype=np.int64,
    )

    confidence_values = []

    saturated_files = []

    for path in sorted(
        FINAL_TXT_DIR.glob(
            "*.txt"
        )
    ):

        lines = [
            line.strip()
            for line in path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        counts.append(
            len(
                lines
            )
        )

        if (
            len(lines)
            == MAX_DET
        ):

            saturated_files.append(
                path.name
            )

        for line in lines:

            fields = line.split()

            class_id = int(
                fields[
                    0
                ]
            )

            confidence = float(
                fields[
                    5
                ]
            )

            class_counts[
                class_id
            ] += 1

            confidence_values.append(
                confidence
            )

    counts_array = np.asarray(
        counts,
        dtype=np.int64,
    )

    if confidence_values:

        conf_array = np.asarray(
            confidence_values,
            dtype=np.float64,
        )

        mean_conf = float(
            conf_array.mean()
        )

        median_conf = float(
            np.median(
                conf_array
            )
        )

    else:

        mean_conf = 0.0

        median_conf = 0.0

    stats.update(
        {
            "mean_boxes": float(
                counts_array.mean()
            ),
            "median_boxes": float(
                np.median(
                    counts_array
                )
            ),
            "p95_boxes": float(
                np.percentile(
                    counts_array,
                    95,
                )
            ),
            "saturated_100": int(
                len(
                    saturated_files
                )
            ),
            "mean_conf": mean_conf,
            "median_conf": median_conf,
            "class_counts": (
                class_counts.tolist()
            ),
        }
    )

    print()
    print(
        "-" * 80
    )

    print(
        "FINAL SUBMISSION STATISTICS"
    )

    print(
        "-" * 80
    )

    print(
        "Mean boxes / image  :",
        f"{stats['mean_boxes']:.2f}",
    )

    print(
        "Median boxes / image:",
        f"{stats['median_boxes']:.2f}",
    )

    print(
        "P95 boxes / image   :",
        f"{stats['p95_boxes']:.2f}",
    )

    print(
        "100-box saturated   :",
        f"{stats['saturated_100']}"
        f"/{EXPECTED_TEST_SAMPLES}",
    )

    print(
        "Mean confidence     :",
        f"{stats['mean_conf']:.6f}",
    )

    print(
        "Median confidence   :",
        f"{stats['median_conf']:.6f}",
    )

    print()
    print(
        "Predictions by class:"
    )

    for class_id, count in enumerate(
        class_counts.tolist()
    ):

        print(
            f"  class {class_id:2d}: "
            f"{count}"
        )

    if saturated_files:

        print()
        print(
            "First saturated images:"
        )

        for name in saturated_files[
            :20
        ]:

            print(
                " ",
                name,
            )

    return stats


base.validate_txt_dir = (
    validate_txt_dir_final
)


# ============================================================
# SHA256
# ============================================================

def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                1024
                * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


# ============================================================
# Final preflight
# ============================================================

def final_preflight() -> None:

    print()
    print("=" * 80)
    print(
        "AIC2026 Exp04 FINAL SUBMISSION CONFIG"
    )
    print("=" * 80)

    print(
        "Model       :",
        FINAL_MODEL_PATH,
    )

    print(
        "Output root :",
        FINAL_OUTPUT_ROOT,
    )

    print(
        "Input       :",
        "[R, G, B, IR, Depth]",
    )

    print(
        "imgsz       :",
        IMAGE_SIZE,
    )

    print(
        "batch       :",
        BATCH_SIZE,
    )

    print(
        "precision   :",
        "FP32",
    )

    print(
        "conf        :",
        CONF_THRESHOLD,
    )

    print(
        "iou         :",
        IOU_THRESHOLD,
    )

    print(
        "max_det     :",
        MAX_DET,
    )

    print(
        "multi_label :",
        MULTI_LABEL,
    )

    print(
        "NMS         :",
        "per-image",
    )

    if (
        FINAL_MODEL_PATH.name
        != "best.pt"
    ):

        raise RuntimeError(
            "FINAL MODEL IS NOT best.pt."
        )

    if not FINAL_MODEL_PATH.is_file():

        raise FileNotFoundError(
            "Final best.pt does not exist:\n"
            f"  {FINAL_MODEL_PATH}"
        )

    print()
    print(
        "[PASS] final model = best.pt"
    )

    print(
        "[PASS] multi_label = True"
    )

    print(
        "[PASS] per-image NMS enabled"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    final_preflight()

    # Run the already-tested base submission pipeline with the
    # final overrides above.
    base.main()

    # --check-only returns before ZIP creation.
    if not FINAL_ZIP_PATH.is_file():

        return

    print()
    print("=" * 80)
    print(
        "FINAL ZIP CHECKSUM"
    )
    print("=" * 80)

    zip_sha256 = sha256_file(
        FINAL_ZIP_PATH
    )

    model_sha256 = sha256_file(
        FINAL_MODEL_PATH
    )

    print(
        "Model SHA256:"
    )

    print(
        model_sha256
    )

    print()

    print(
        "ZIP SHA256:"
    )

    print(
        zip_sha256
    )

    print()
    print(
        "FINAL SUBMISSION ZIP:"
    )

    print(
        FINAL_ZIP_PATH
    )


if __name__ == "__main__":
    main()

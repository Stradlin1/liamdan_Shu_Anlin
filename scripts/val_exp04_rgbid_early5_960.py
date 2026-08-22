#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 Exp04 Official Validation

Model:
    runs/exp04_rgbid_early5_yolo11s_960/weights/best.pt

Input:
    [R, G, B, IR, Depth]

Official validation protocol:
    imgsz   = 960
    conf    = 0.001
    iou     = 0.70
    max_det = 100

Validation set:
    fixed 400 multimodal samples
"""

from __future__ import annotations

import json
from copy import copy
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator

from multimodal_dataset import (
    CHANNEL_NAMES,
    MULTIMODAL_CHANNELS,
)

from multimodal_trainer import (
    MultimodalDetectionTrainer,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

RUN_DIR = (
    PROJECT_ROOT
    / "runs"
    / "exp04_rgbid_early5_yolo11s_960"
)

MODEL_PATH = (
    RUN_DIR
    / "weights"
    / "best.pt"
)

DATA_YAML = (
    PROJECT_ROOT
    / "yolo_views"
    / "rgb_v1"
    / "data_exp04_5ch.yaml"
)

VAL_DIR = (
    RUN_DIR
    / "val_official_960"
)

METRICS_JSON = (
    VAL_DIR
    / "official_metrics.json"
)


# ============================================================
# Official protocol
# ============================================================

IMAGE_SIZE = 960

BATCH_SIZE = 8

CONF_THRESHOLD = 0.001

IOU_THRESHOLD = 0.70

MAX_DET = 100

EXPECTED_CHANNELS = 5

EXPECTED_CLASSES = 12

EXPECTED_VAL_IMAGES = 400


# ============================================================
# RGB baseline
# ============================================================

RGB_BASELINE = {
    "precision": 0.763071,
    "recall": 0.637680,
    "map50": 0.642417,
    "map75": 0.372719,
    "map5095": 0.380843,
}


# ============================================================
# Helpers
# ============================================================

def section(title: str) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# Preflight
# ============================================================

def preflight() -> None:

    section(
        "Exp04 Official Validation - Preflight"
    )

    if not MODEL_PATH.is_file():

        raise FileNotFoundError(
            f"best.pt not found:\n  {MODEL_PATH}"
        )

    if not DATA_YAML.is_file():

        raise FileNotFoundError(
            f"5-channel data YAML not found:\n"
            f"  {DATA_YAML}"
        )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is unavailable."
        )

    VAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
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
        "Dataset      :",
        DATA_YAML,
    )

    print(
        "Output       :",
        VAL_DIR,
    )

    print(
        "Channels     :",
        CHANNEL_NAMES,
    )

    print(
        "imgsz        :",
        IMAGE_SIZE,
    )

    print(
        "batch        :",
        BATCH_SIZE,
    )

    print(
        "conf         :",
        CONF_THRESHOLD,
    )

    print(
        "iou          :",
        IOU_THRESHOLD,
    )

    print(
        "max_det      :",
        MAX_DET,
    )

    print(
        "GPU          :",
        torch.cuda.get_device_name(0),
    )


# ============================================================
# Load checkpoint
# ============================================================

def load_and_check_model():

    section(
        "Load best.pt"
    )

    wrapper = YOLO(
        str(MODEL_PATH)
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
        "Weight shape:",
        tuple(
            first_conv.weight.shape
        ),
    )

    print(
        "Detect nc:",
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
            "best.pt input channels != 5"
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
            "best.pt Detect nc != 12"
        )

    print(
        "[PASS] best.pt is 5-channel"
    )

    print(
        "[PASS] best.pt is 12-class"
    )

    return wrapper


# ============================================================
# Build custom multimodal validation DataLoader
# ============================================================

def build_val_loader(
    wrapper: YOLO,
):

    section(
        "Build Multimodal Validation DataLoader"
    )

    overrides = {

        "model": str(
            MODEL_PATH
        ),

        "data": str(
            DATA_YAML
        ),

        "task": "detect",

        "imgsz": IMAGE_SIZE,

        "batch": BATCH_SIZE,

        "device": "0",

        "workers": 4,

        "cache": False,

        "rect": True,

        "conf": CONF_THRESHOLD,

        "iou": IOU_THRESHOLD,

        "max_det": MAX_DET,

        "plots": False,

        "save_json": False,

        "save_txt": False,

        "project": str(
            RUN_DIR
        ),

        "name": "val_official_960",

        "exist_ok": True,
    }

    trainer = MultimodalDetectionTrainer(
        overrides=overrides
    )

    # We do not start training.
    #
    # Give the trainer the already-loaded best.pt model so that
    # build_dataset/get_dataloader can obtain the real YOLO stride.
    trainer.model = (
        wrapper
        .model
        .to(
            trainer.device
        )
    )

    trainer.set_model_attributes()

    if (
        trainer.data.get("channels")
        != EXPECTED_CHANNELS
    ):

        raise RuntimeError(
            "Trainer data channels != 5"
        )

    if (
        trainer.data.get("nc")
        != EXPECTED_CLASSES
    ):

        raise RuntimeError(
            "Trainer data nc != 12"
        )

    val_loader = trainer.get_dataloader(
        trainer.data["val"],
        batch_size=BATCH_SIZE,
        rank=-1,
        mode="val",
    )

    dataset = val_loader.dataset

    print(
        "Dataset class:",
        type(dataset).__name__,
    )

    print(
        "Samples      :",
        len(dataset),
    )

    if (
        type(dataset).__name__
        != "MultimodalYOLODataset"
    ):

        raise RuntimeError(
            "Validation is not using "
            "MultimodalYOLODataset."
        )

    if (
        len(dataset)
        != EXPECTED_VAL_IMAGES
    ):

        raise RuntimeError(
            "Unexpected validation sample count:\n"
            f"  expected={EXPECTED_VAL_IMAGES}\n"
            f"  actual={len(dataset)}"
        )

    # --------------------------------------------------------
    # Inspect one actual batch.
    # --------------------------------------------------------

    first_batch = next(
        iter(
            val_loader
        )
    )

    images = first_batch[
        "img"
    ]

    print()
    print(
        "First val batch shape:",
        tuple(
            images.shape
        ),
    )

    print(
        "First val batch dtype:",
        images.dtype,
    )

    print(
        "First val batch min/max:",
        int(images.min()),
        int(images.max()),
    )

    if (
        images.ndim != 4
        or images.shape[1]
        != EXPECTED_CHANNELS
    ):

        raise RuntimeError(
            "Validation batch is not BCHW 5-channel."
        )

    print()
    print(
        "[PASS] fixed 400-sample val split"
    )

    print(
        "[PASS] MultimodalYOLODataset"
    )

    print(
        "[PASS] 5-channel validation batch"
    )

    return (
        trainer,
        val_loader,
    )


# ============================================================
# Official validation
# ============================================================

def run_official_validation(
    trainer,
    val_loader,
):

    section(
        "RUN OFFICIAL VALIDATION"
    )

    validator_args = copy(
        trainer.args
    )

    # Explicitly lock official protocol.
    validator_args.model = str(
        MODEL_PATH
    )

    validator_args.data = str(
        DATA_YAML
    )

    validator_args.split = "val"

    validator_args.imgsz = IMAGE_SIZE

    validator_args.batch = BATCH_SIZE

    validator_args.conf = (
        CONF_THRESHOLD
    )

    validator_args.iou = (
        IOU_THRESHOLD
    )

    validator_args.max_det = (
        MAX_DET
    )

    validator_args.plots = False

    validator_args.save_json = False

    validator_args.save_txt = False

    validator_args.device = "0"

    validator = DetectionValidator(
        dataloader=val_loader,
        save_dir=VAL_DIR,
        args=validator_args,
    )

    stats = validator(
        model=str(
            MODEL_PATH
        )
    )

    return (
        validator,
        stats,
    )


# ============================================================
# Metrics
# ============================================================

def report_metrics(
    validator,
    stats,
) -> None:

    section(
        "OFFICIAL METRICS"
    )

    box = (
        validator
        .metrics
        .box
    )

    precision = float(
        box.mp
    )

    recall = float(
        box.mr
    )

    map50 = float(
        box.map50
    )

    map75 = float(
        box.map75
    )

    map5095 = float(
        box.map
    )

    metrics = {

        "protocol": {
            "imgsz": IMAGE_SIZE,
            "conf": CONF_THRESHOLD,
            "iou": IOU_THRESHOLD,
            "max_det": MAX_DET,
            "val_images": EXPECTED_VAL_IMAGES,
            "channels": EXPECTED_CHANNELS,
        },

        "exp04": {
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map75": map75,
            "map5095": map5095,
        },

        "rgb_baseline": (
            RGB_BASELINE
        ),

        "delta_vs_rgb": {
            "precision": (
                precision
                - RGB_BASELINE[
                    "precision"
                ]
            ),

            "recall": (
                recall
                - RGB_BASELINE[
                    "recall"
                ]
            ),

            "map50": (
                map50
                - RGB_BASELINE[
                    "map50"
                ]
            ),

            "map75": (
                map75
                - RGB_BASELINE[
                    "map75"
                ]
            ),

            "map5095": (
                map5095
                - RGB_BASELINE[
                    "map5095"
                ]
            ),
        },
    }

    print(
        "Exp04"
    )

    print(
        f"Precision   : {precision:.6f}"
    )

    print(
        f"Recall      : {recall:.6f}"
    )

    print(
        f"mAP50       : {map50:.6f}"
    )

    print(
        f"mAP75       : {map75:.6f}"
    )

    print(
        f"mAP50-95    : {map5095:.6f}"
    )

    print()
    print(
        "RGB baseline"
    )

    print(
        f"Precision   : "
        f"{RGB_BASELINE['precision']:.6f}"
    )

    print(
        f"Recall      : "
        f"{RGB_BASELINE['recall']:.6f}"
    )

    print(
        f"mAP50       : "
        f"{RGB_BASELINE['map50']:.6f}"
    )

    print(
        f"mAP75       : "
        f"{RGB_BASELINE['map75']:.6f}"
    )

    print(
        f"mAP50-95    : "
        f"{RGB_BASELINE['map5095']:.6f}"
    )

    delta = (
        map5095
        - RGB_BASELINE[
            "map5095"
        ]
    )

    print()
    print(
        "mAP50-95 delta:",
        f"{delta:+.6f}",
    )

    if delta > 0:

        print(
            "RESULT: Exp04 > RGB baseline"
        )

    elif delta < 0:

        print(
            "RESULT: Exp04 < RGB baseline"
        )

    else:

        print(
            "RESULT: exact tie"
        )

    METRICS_JSON.write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Saved:",
        METRICS_JSON,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    preflight()

    wrapper = (
        load_and_check_model()
    )

    (
        trainer,
        val_loader,
    ) = build_val_loader(
        wrapper
    )

    (
        validator,
        stats,
    ) = run_official_validation(
        trainer,
        val_loader,
    )

    report_metrics(
        validator,
        stats,
    )

    section(
        "FINAL RESULT"
    )

    print(
        "best.pt                        PASS"
    )

    print(
        "5-channel model                PASS"
    )

    print(
        "12-class Detect head           PASS"
    )

    print(
        "400-sample fixed val           PASS"
    )

    print(
        "MultimodalYOLODataset          PASS"
    )

    print(
        "imgsz=960                      PASS"
    )

    print(
        "conf=0.001                     PASS"
    )

    print(
        "iou=0.70                       PASS"
    )

    print(
        "max_det=100                    PASS"
    )

    print()
    print(
        "STATUS = PASS"
    )


if __name__ == "__main__":
    main()

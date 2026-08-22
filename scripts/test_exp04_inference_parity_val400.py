#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 Exp04
Manual submission-inference parity test on the fixed 400-image val set.

Purpose
-------
Compare the exact manual submission preprocessing / inference pipeline against
the already measured Official Validation result.

Manual pipeline tested here:

    RGB + IR + Depth
        ↓
    [R, G, B, IR, Depth]
        ↓
    square LetterBox to 960 x 960
        ↓
    float32 / 255
        ↓
    Exp04 best.pt
        ↓
    per-image NMS
        ↓
    restore boxes to original image coordinates
        ↓
    compare against original YOLO labels
        ↓
    compute P / R / mAP50 / mAP75 / mAP50-95

Two NMS modes are evaluated from the SAME model forward:

    A. SUBMIT_FALSE
       multi_label=False
       -> matches current submission script

    B. VAL_TRUE
       multi_label=True
       -> matches Ultralytics DetectionValidator behavior more closely

Official reference:
    imgsz   = 960
    conf    = 0.001
    iou     = 0.70
    max_det = 100
    val     = 400

This script DOES NOT modify the model, dataset, or submission script.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

# Import torchvision if available so Ultralytics NMS can use its CUDA NMS path.
try:
    import torchvision  # noqa: F401
except Exception:
    torchvision = None

from ultralytics import YOLO
from ultralytics.utils.metrics import (
    ap_per_class,
    box_iou,
)

try:
    from ultralytics.utils.nms import (
        non_max_suppression,
    )
except ImportError:
    from ultralytics.utils.ops import (
        non_max_suppression,
    )

# ------------------------------------------------------------
# IMPORTANT:
#
# Reuse the exact image-loading / LetterBox / normalization /
# coordinate-restoration code from the submission script.
# ------------------------------------------------------------

from infer_exp04_rgbid_early5_960_submit import (
    load_5ch,
    letterbox_5ch,
    restore_xyxy_to_original,
)


# ============================================================
# Project paths
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

RGB_VAL_DIR = (
    PROJECT_ROOT
    / "yolo_views"
    / "rgb_v1"
    / "images"
    / "val"
)

IR_VAL_DIR = (
    PROJECT_ROOT
    / "yolo_views"
    / "ir_v1"
    / "images"
    / "val"
)

DEPTH_VAL_DIR = (
    PROJECT_ROOT
    / "yolo_views"
    / "depth8_v1"
    / "images"
    / "val"
)

LABEL_VAL_DIR = (
    PROJECT_ROOT
    / "yolo_views"
    / "rgb_v1"
    / "labels"
    / "val"
)

OFFICIAL_METRICS_JSON = (
    RUN_DIR
    / "val_official_960"
    / "official_metrics.json"
)

OUTPUT_DIR = (
    RUN_DIR
    / "parity_val400"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "parity_metrics.json"
)


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 960

BATCH_SIZE = 4

CONF_THRESHOLD = 0.001

IOU_THRESHOLD = 0.70

MAX_DET = 100

EXPECTED_CHANNELS = 5

EXPECTED_CLASSES = 12

EXPECTED_VAL_IMAGES = 400

DEVICE_INDEX = 0

IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

IOU_THRESHOLDS = torch.linspace(
    0.50,
    0.95,
    10,
)


# ============================================================
# Official reference
# ============================================================

OFFICIAL_REFERENCE_FALLBACK = {
    "precision": 0.712538,
    "recall": 0.603216,
    "map50": 0.628168,
    "map75": 0.388485,
    "map5095": 0.379090,
}


# ============================================================
# Parity thresholds
# ============================================================

# The manual submission path uses square 960 LetterBox, while
# Official Val used rect=True. Therefore exact bitwise equality
# is not expected.
#
# mAP50-95 difference:
#   <= 0.005 : PASS
#   <= 0.010 : WARN
#   >  0.010 : FAIL

PARITY_PASS_TOL = 0.005

PARITY_WARN_TOL = 0.010


# ============================================================
# Helpers
# ============================================================

def section(
    title: str,
) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def image_map(
    root: Path,
) -> dict[str, Path]:

    result: dict[str, Path] = {}

    for path in root.iterdir():

        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_SUFFIXES
        ):

            stem = path.stem

            if stem in result:

                raise RuntimeError(
                    "Duplicate image stem:\n"
                    f"  root={root}\n"
                    f"  stem={stem}"
                )

            result[
                stem
            ] = path

    return result


def load_official_reference() -> dict:

    if OFFICIAL_METRICS_JSON.is_file():

        data = json.loads(
            OFFICIAL_METRICS_JSON.read_text(
                encoding="utf-8"
            )
        )

        exp04 = data.get(
            "exp04"
        )

        if isinstance(
            exp04,
            dict,
        ):

            required = {
                "precision",
                "recall",
                "map50",
                "map75",
                "map5095",
            }

            if required.issubset(
                exp04
            ):

                print(
                    "Official reference loaded from:"
                )

                print(
                    f"  {OFFICIAL_METRICS_JSON}"
                )

                return {
                    key: float(
                        exp04[key]
                    )
                    for key in required
                }

    print(
        "WARNING: official_metrics.json not found."
    )

    print(
        "Using locked Official Val fallback values."
    )

    return dict(
        OFFICIAL_REFERENCE_FALLBACK
    )


# ============================================================
# Validation set
# ============================================================

def build_val_maps():

    section(
        "Build fixed val400 triplets"
    )

    for path in (
        RGB_VAL_DIR,
        IR_VAL_DIR,
        DEPTH_VAL_DIR,
        LABEL_VAL_DIR,
    ):

        if not path.is_dir():

            raise FileNotFoundError(
                f"Missing directory:\n  {path}"
            )

    rgb_map = image_map(
        RGB_VAL_DIR
    )

    ir_map = image_map(
        IR_VAL_DIR
    )

    depth_map = image_map(
        DEPTH_VAL_DIR
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

    print(
        "RGB images  :",
        len(
            rgb_stems
        ),
    )

    print(
        "IR images   :",
        len(
            ir_stems
        ),
    )

    print(
        "Depth images:",
        len(
            depth_stems
        ),
    )

    if not (
        rgb_stems
        == ir_stems
        == depth_stems
    ):

        raise RuntimeError(
            "RGB / IR / Depth val stems are not identical."
        )

    stems = sorted(
        rgb_stems
    )

    if (
        len(stems)
        != EXPECTED_VAL_IMAGES
    ):

        raise RuntimeError(
            "Unexpected val sample count:\n"
            f"  expected={EXPECTED_VAL_IMAGES}\n"
            f"  actual={len(stems)}"
        )

    missing_labels = []

    for stem in stems:

        label_path = (
            LABEL_VAL_DIR
            / f"{stem}.txt"
        )

        if not label_path.is_file():

            missing_labels.append(
                label_path
            )

    if missing_labels:

        raise RuntimeError(
            "Missing validation labels:\n"
            + "\n".join(
                str(x)
                for x
                in missing_labels[:20]
            )
        )

    candidate = {
        "rgb_map": rgb_map,
        "ir_map": ir_map,
        "depth_map": depth_map,
    }

    print(
        "[PASS] 400 RGB / IR / Depth triplets"
    )

    print(
        "[PASS] 400 label files"
    )

    return (
        candidate,
        stems,
    )


# ============================================================
# Ground truth
# ============================================================

def read_yolo_label(
    stem: str,
):

    path = (
        LABEL_VAL_DIR
        / f"{stem}.txt"
    )

    text = path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:

        return np.zeros(
            (
                0,
                5,
            ),
            dtype=np.float32,
        )

    rows = []

    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):

        fields = line.split()

        if len(fields) != 5:

            raise RuntimeError(
                "Invalid YOLO label:\n"
                f"  file={path}\n"
                f"  line={line_number}\n"
                f"  text={line}"
            )

        values = [
            float(x)
            for x in fields
        ]

        rows.append(
            values
        )

    labels = np.asarray(
        rows,
        dtype=np.float32,
    )

    # Ultralytics dataset verification removes exact duplicate
    # labels. Mirror that behavior here.
    if len(labels):

        labels = np.unique(
            labels,
            axis=0,
        )

    class_ids = labels[
        :,
        0,
    ]

    if np.any(
        class_ids < 0
    ) or np.any(
        class_ids >= EXPECTED_CLASSES
    ):

        raise RuntimeError(
            f"Illegal class id in {path}"
        )

    return labels


def gt_to_original_xyxy(
    labels: np.ndarray,
    original_shape: tuple[int, int],
    device: torch.device,
):

    h, w = original_shape

    if len(labels) == 0:

        return (
            torch.zeros(
                (
                    0,
                    4,
                ),
                dtype=torch.float32,
                device=device,
            ),
            torch.zeros(
                (
                    0,
                ),
                dtype=torch.long,
                device=device,
            ),
        )

    cls = torch.from_numpy(
        labels[
            :,
            0,
        ]
    ).to(
        device=device,
        dtype=torch.long,
    )

    xywh = torch.from_numpy(
        labels[
            :,
            1:5,
        ]
    ).to(
        device=device,
        dtype=torch.float32,
    )

    xc = (
        xywh[
            :,
            0,
        ]
        * float(w)
    )

    yc = (
        xywh[
            :,
            1,
        ]
        * float(h)
    )

    bw = (
        xywh[
            :,
            2,
        ]
        * float(w)
    )

    bh = (
        xywh[
            :,
            3,
        ]
        * float(h)
    )

    x1 = (
        xc
        - bw / 2.0
    )

    y1 = (
        yc
        - bh / 2.0
    )

    x2 = (
        xc
        + bw / 2.0
    )

    y2 = (
        yc
        + bh / 2.0
    )

    boxes = torch.stack(
        (
            x1,
            y1,
            x2,
            y2,
        ),
        dim=1,
    )

    return (
        boxes,
        cls,
    )


# ============================================================
# Manual submission preprocessing
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
            candidate[
                "rgb_map"
            ][
                stem
            ]
        )

        ir_path = (
            candidate[
                "ir_map"
            ][
                stem
            ]
        )

        depth_path = (
            candidate[
                "depth_map"
            ][
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
            image,
            new_shape=(
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            scaleup=False,
            padding_value=114,
        )

        if (
            image.ndim != 3
            or image.shape[2]
            != EXPECTED_CHANNELS
        ):

            raise RuntimeError(
                "Manual preprocessing did not "
                "produce HWC 5-channel data."
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
                "original_shape": (
                    int(
                        original_shape[
                            0
                        ]
                    ),
                    int(
                        original_shape[
                            1
                        ]
                    ),
                ),
                "scale_x": float(
                    scale_x
                ),
                "scale_y": float(
                    scale_y
                ),
                "left": int(
                    left
                ),
                "top": int(
                    top
                ),
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
# Model
# ============================================================

def load_model():

    section(
        "Load Exp04 best.pt"
    )

    if not MODEL_PATH.is_file():

        raise FileNotFoundError(
            f"Missing best.pt:\n  {MODEL_PATH}"
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
        "Model:",
        MODEL_PATH,
    )

    print(
        "First conv:",
        first_conv,
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
            "best.pt is not 5-channel."
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
            "best.pt is not 12-class."
        )

    device = torch.device(
        f"cuda:{DEVICE_INDEX}"
    )

    model = (
        model
        .to(
            device
        )
        .float()
        .eval()
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(
            DEVICE_INDEX
        ),
    )

    print(
        "[PASS] 5-channel model"
    )

    print(
        "[PASS] 12-class model"
    )

    return (
        wrapper,
        model,
        device,
    )


# ============================================================
# Model forward
# ============================================================

@torch.inference_mode()
def model_forward(
    model,
    batch: torch.Tensor,
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

        prediction = raw[
            0
        ]

    elif torch.is_tensor(
        raw
    ):

        prediction = raw

    else:

        raise RuntimeError(
            "Unexpected model output type: "
            f"{type(raw).__name__}"
        )

    if (
        prediction.ndim != 3
    ):

        raise RuntimeError(
            "Unexpected prediction shape: "
            f"{tuple(prediction.shape)}"
        )

    if (
        prediction.shape[
            0
        ]
        != batch.shape[
            0
        ]
    ):

        raise RuntimeError(
            "Prediction batch size mismatch."
        )

    return prediction


# ============================================================
# Per-image NMS
# ============================================================

@torch.inference_mode()
def per_image_nms(
    prediction: torch.Tensor,
    multi_label: bool,
):

    detections = []

    for image_index in range(
        prediction.shape[
            0
        ]
    ):

        # CRITICAL:
        #
        # non_max_suppression() modifies prediction coordinates
        # internally in some Ultralytics versions.
        #
        # Clone each image so the second NMS mode receives the
        # exact same raw model output.
        pred_one = (
            prediction[
                image_index:
                image_index + 1
            ]
            .clone()
        )

        det = non_max_suppression(
            pred_one,
            conf_thres=CONF_THRESHOLD,
            iou_thres=IOU_THRESHOLD,
            classes=None,
            agnostic=False,
            multi_label=multi_label,
            max_det=MAX_DET,
            nc=EXPECTED_CLASSES,
            max_time_img=10.0,
            max_nms=30000,
        )[
            0
        ]

        # Current submission script explicitly re-sorts by
        # confidence and caps at 100 once more.
        if len(
            det
        ):

            order = torch.argsort(
                det[
                    :,
                    4,
                ],
                descending=True,
            )

            det = det[
                order
            ][
                :MAX_DET
            ]

        detections.append(
            det
        )

    return detections


# ============================================================
# Matching
# ============================================================

def match_predictions(
    pred_classes: torch.Tensor,
    true_classes: torch.Tensor,
    iou: torch.Tensor,
) -> np.ndarray:
    """
    Mirror Ultralytics BaseValidator.match_predictions().

    Returns:
        [num_predictions, 10] boolean array
        for IoU thresholds 0.50 ... 0.95.
    """

    correct = np.zeros(
        (
            pred_classes.shape[
                0
            ],
            len(
                IOU_THRESHOLDS
            ),
        ),
        dtype=bool,
    )

    if (
        pred_classes.numel()
        == 0
        or true_classes.numel()
        == 0
    ):

        return correct

    correct_class = (
        true_classes[
            :,
            None,
        ]
        == pred_classes[
            None,
            :,
        ]
    )

    iou = (
        iou
        * correct_class
    )

    iou_np = (
        iou
        .detach()
        .cpu()
        .numpy()
    )

    for threshold_index, threshold in enumerate(
        IOU_THRESHOLDS.tolist()
    ):

        matches = np.nonzero(
            iou_np
            >= threshold
        )

        matches = np.array(
            matches
        ).T

        if (
            matches.shape[
                0
            ]
            == 0
        ):

            continue

        if (
            matches.shape[
                0
            ]
            > 1
        ):

            # Highest IoU first.
            matches = matches[
                iou_np[
                    matches[
                        :,
                        0,
                    ],
                    matches[
                        :,
                        1,
                    ],
                ]
                .argsort()[
                    ::-1
                ]
            ]

            # One detection can match only one target.
            matches = matches[
                np.unique(
                    matches[
                        :,
                        1,
                    ],
                    return_index=True,
                )[
                    1
                ]
            ]

            # One target can match only one detection.
            matches = matches[
                np.unique(
                    matches[
                        :,
                        0,
                    ],
                    return_index=True,
                )[
                    1
                ]
            ]

        correct[
            matches[
                :,
                1,
            ].astype(
                int
            ),
            threshold_index,
        ] = True

    return correct


# ============================================================
# Metric accumulator
# ============================================================

class MetricAccumulator:

    def __init__(
        self,
        name: str,
    ):

        self.name = name

        self.tp = []

        self.conf = []

        self.pred_cls = []

        self.target_cls = []

        self.total_predictions = 0

        self.empty_prediction_images = 0

        self.saturated_100_images = 0

        self.total_targets = 0


    def update(
        self,
        det: torch.Tensor,
        meta: dict,
    ) -> None:

        stem = meta[
            "stem"
        ]

        labels = read_yolo_label(
            stem
        )

        (
            gt_boxes,
            gt_cls,
        ) = gt_to_original_xyxy(
            labels,
            meta[
                "original_shape"
            ],
            det.device,
        )

        self.total_targets += (
            gt_cls.numel()
        )

        # ----------------------------------------------------
        # Restore prediction boxes EXACTLY like submission.
        # ----------------------------------------------------

        if (
            det is None
            or len(
                det
            )
            == 0
        ):

            pred_boxes = torch.zeros(
                (
                    0,
                    4,
                ),
                dtype=torch.float32,
                device=gt_boxes.device,
            )

            pred_conf = torch.zeros(
                (
                    0,
                ),
                dtype=torch.float32,
                device=gt_boxes.device,
            )

            pred_cls = torch.zeros(
                (
                    0,
                ),
                dtype=torch.long,
                device=gt_boxes.device,
            )

        else:

            pred_boxes = (
                restore_xyxy_to_original(
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
            )

            pred_conf = det[
                :,
                4,
            ]

            pred_cls = det[
                :,
                5,
            ].long()

            # Submission xyxy_to_submission() discards
            # zero-area boxes after coordinate restoration.
            valid = (
                (
                    pred_boxes[
                        :,
                        2,
                    ]
                    > pred_boxes[
                        :,
                        0,
                    ]
                )
                & (
                    pred_boxes[
                        :,
                        3,
                    ]
                    > pred_boxes[
                        :,
                        1,
                    ]
                )
            )

            pred_boxes = (
                pred_boxes[
                    valid
                ]
            )

            pred_conf = (
                pred_conf[
                    valid
                ]
            )

            pred_cls = (
                pred_cls[
                    valid
                ]
            )

        num_predictions = (
            pred_cls.numel()
        )

        self.total_predictions += (
            num_predictions
        )

        if (
            num_predictions
            == 0
        ):

            self.empty_prediction_images += 1

        if (
            num_predictions
            >= MAX_DET
        ):

            self.saturated_100_images += 1

        # ----------------------------------------------------
        # Match predictions against GT.
        # ----------------------------------------------------

        if (
            gt_boxes.shape[
                0
            ]
            and pred_boxes.shape[
                0
            ]
        ):

            iou = box_iou(
                gt_boxes,
                pred_boxes,
            )

            correct = match_predictions(
                pred_classes=pred_cls,
                true_classes=gt_cls,
                iou=iou,
            )

        else:

            correct = np.zeros(
                (
                    num_predictions,
                    len(
                        IOU_THRESHOLDS
                    ),
                ),
                dtype=bool,
            )

        self.tp.append(
            correct
        )

        self.conf.append(
            pred_conf
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        self.pred_cls.append(
            pred_cls
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        self.target_cls.append(
            gt_cls
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )


    def compute(
        self,
        names,
    ) -> dict:

        tp = np.concatenate(
            self.tp,
            axis=0,
        )

        conf = np.concatenate(
            self.conf,
            axis=0,
        )

        pred_cls = np.concatenate(
            self.pred_cls,
            axis=0,
        )

        target_cls = np.concatenate(
            self.target_cls,
            axis=0,
        )

        if (
            target_cls.size
            == 0
        ):

            raise RuntimeError(
                "No validation ground truth found."
            )

        if (
            conf.size
            == 0
        ):

            raise RuntimeError(
                f"No predictions for mode {self.name}."
            )

        result = ap_per_class(
            tp=tp,
            conf=conf,
            pred_cls=pred_cls,
            target_cls=target_cls,
            plot=False,
            names=names,
        )

        if (
            len(
                result
            )
            < 7
        ):

            raise RuntimeError(
                "Unexpected ap_per_class() return signature."
            )

        p = np.asarray(
            result[
                2
            ]
        )

        r = np.asarray(
            result[
                3
            ]
        )

        ap = np.asarray(
            result[
                5
            ]
        )

        ap_classes = np.asarray(
            result[
                6
            ]
        ).astype(
            int
        )

        if (
            ap.ndim != 2
            or ap.shape[
                1
            ]
            != 10
        ):

            raise RuntimeError(
                "Unexpected AP matrix shape: "
                f"{ap.shape}"
            )

        precision = float(
            p.mean()
        )

        recall = float(
            r.mean()
        )

        map50 = float(
            ap[
                :,
                0,
            ].mean()
        )

        map75 = float(
            ap[
                :,
                5,
            ].mean()
        )

        map5095 = float(
            ap.mean()
        )

        per_class = []

        for result_index, class_id in enumerate(
            ap_classes
        ):

            if isinstance(
                names,
                dict,
            ):

                class_name = names.get(
                    int(
                        class_id
                    ),
                    str(
                        class_id
                    ),
                )

            else:

                class_name = (
                    names[
                        int(
                            class_id
                        )
                    ]
                    if int(
                        class_id
                    )
                    < len(
                        names
                    )
                    else str(
                        class_id
                    )
                )

            per_class.append(
                {
                    "class_id": int(
                        class_id
                    ),
                    "class_name": str(
                        class_name
                    ),
                    "precision": float(
                        p[
                            result_index
                        ]
                    ),
                    "recall": float(
                        r[
                            result_index
                        ]
                    ),
                    "map50": float(
                        ap[
                            result_index,
                            0,
                        ]
                    ),
                    "map75": float(
                        ap[
                            result_index,
                            5,
                        ]
                    ),
                    "map5095": float(
                        ap[
                            result_index
                        ].mean()
                    ),
                }
            )

        return {
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map75": map75,
            "map5095": map5095,
            "total_predictions": int(
                self.total_predictions
            ),
            "total_targets": int(
                self.total_targets
            ),
            "empty_prediction_images": int(
                self.empty_prediction_images
            ),
            "saturated_100_images": int(
                self.saturated_100_images
            ),
            "mean_predictions_per_image": float(
                self.total_predictions
                / EXPECTED_VAL_IMAGES
            ),
            "per_class": per_class,
        }


# ============================================================
# Parity status
# ============================================================

def parity_status(
    manual_map: float,
    official_map: float,
):

    delta = (
        manual_map
        - official_map
    )

    absolute_delta = abs(
        delta
    )

    if (
        absolute_delta
        <= PARITY_PASS_TOL
    ):

        status = "PASS"

    elif (
        absolute_delta
        <= PARITY_WARN_TOL
    ):

        status = "WARN"

    else:

        status = "FAIL"

    return {
        "status": status,
        "delta": float(
            delta
        ),
        "absolute_delta": float(
            absolute_delta
        ),
    }


# ============================================================
# Report
# ============================================================

def print_metric_block(
    title: str,
    result: dict,
    official: dict,
):

    section(
        title
    )

    print(
        f"Precision : "
        f"{result['precision']:.6f}"
    )

    print(
        f"Recall    : "
        f"{result['recall']:.6f}"
    )

    print(
        f"mAP50     : "
        f"{result['map50']:.6f}"
    )

    print(
        f"mAP75     : "
        f"{result['map75']:.6f}"
    )

    print(
        f"mAP50-95  : "
        f"{result['map5095']:.6f}"
    )

    print()

    print(
        "Total predictions :",
        result[
            "total_predictions"
        ],
    )

    print(
        "Mean preds/image  :",
        f"{result['mean_predictions_per_image']:.2f}",
    )

    print(
        "Empty images      :",
        result[
            "empty_prediction_images"
        ],
    )

    print(
        "100-box saturated :",
        result[
            "saturated_100_images"
        ],
    )

    parity = parity_status(
        result[
            "map5095"
        ],
        official[
            "map5095"
        ],
    )

    print()

    print(
        "Official mAP50-95 :",
        f"{official['map5095']:.6f}",
    )

    print(
        "Delta             :",
        f"{parity['delta']:+.6f}",
    )

    print(
        "Absolute delta    :",
        f"{parity['absolute_delta']:.6f}",
    )

    print(
        "Parity status     :",
        parity[
            "status"
        ],
    )

    return parity


# ============================================================
# Main
# ============================================================

def main() -> None:

    section(
        "Exp04 Manual Inference Parity Test"
    )

    print(
        "Model:",
        MODEL_PATH,
    )

    print(
        "Manual preprocessing:"
    )

    print(
        "  5ch RGB+IR+Depth"
    )

    print(
        "  square LetterBox 960 x 960"
    )

    print(
        "  scaleup=False"
    )

    print(
        "  padding=114"
    )

    print(
        "  FP32 / 255"
    )

    print(
        "  batch forward=4"
    )

    print(
        "  per-image NMS"
    )

    print(
        "  conf=0.001"
    )

    print(
        "  iou=0.70"
    )

    print(
        "  max_det=100"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    official = (
        load_official_reference()
    )

    section(
        "Official reference"
    )

    print(
        f"Precision : "
        f"{official['precision']:.6f}"
    )

    print(
        f"Recall    : "
        f"{official['recall']:.6f}"
    )

    print(
        f"mAP50     : "
        f"{official['map50']:.6f}"
    )

    print(
        f"mAP75     : "
        f"{official['map75']:.6f}"
    )

    print(
        f"mAP50-95  : "
        f"{official['map5095']:.6f}"
    )

    (
        candidate,
        stems,
    ) = build_val_maps()

    (
        wrapper,
        model,
        device,
    ) = load_model()

    names = (
        wrapper.names
    )

    submit_false = MetricAccumulator(
        "SUBMIT_FALSE"
    )

    val_true = MetricAccumulator(
        "VAL_TRUE"
    )

    section(
        "Run val400 manual inference"
    )

    total_batches = math.ceil(
        len(
            stems
        )
        / BATCH_SIZE
    )

    for batch_index, start in enumerate(
        range(
            0,
            len(
                stems
            ),
            BATCH_SIZE,
        ),
        start=1,
    ):

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

        if (
            batch.shape[
                1
            ]
            != EXPECTED_CHANNELS
        ):

            raise RuntimeError(
                "Manual batch is not 5-channel."
            )

        if (
            batch.shape[
                2:
            ]
            != (
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
        ):

            raise RuntimeError(
                "Manual batch is not 960x960:\n"
                f"  shape={tuple(batch.shape)}"
            )

        prediction = model_forward(
            model,
            batch,
        )

        # ----------------------------------------------------
        # A: exact current submission mode.
        # ----------------------------------------------------

        detections_false = per_image_nms(
            prediction,
            multi_label=False,
        )

        # ----------------------------------------------------
        # B: validator-like multi-label NMS.
        # ----------------------------------------------------

        detections_true = per_image_nms(
            prediction,
            multi_label=True,
        )

        if not (
            len(
                detections_false
            )
            == len(
                metas
            )
            == len(
                detections_true
            )
        ):

            raise RuntimeError(
                "Prediction batch length mismatch."
            )

        for (
            det_false,
            det_true,
            meta,
        ) in zip(
            detections_false,
            detections_true,
            metas,
        ):

            submit_false.update(
                det_false,
                meta,
            )

            val_true.update(
                det_true,
                meta,
            )

        if (
            batch_index == 1
            or batch_index % 10 == 0
            or batch_index
            == total_batches
        ):

            done = min(
                start
                + BATCH_SIZE,
                len(
                    stems
                ),
            )

            print(
                f"[Parity] "
                f"batch "
                f"{batch_index}/"
                f"{total_batches} "
                f"({done}/"
                f"{len(stems)} images)"
            )

        del batch

        del prediction

        del detections_false

        del detections_true

    # ========================================================
    # Metrics
    # ========================================================

    result_false = (
        submit_false.compute(
            names
        )
    )

    result_true = (
        val_true.compute(
            names
        )
    )

    parity_false = print_metric_block(
        "A. SUBMIT_FALSE - current submission behavior",
        result_false,
        official,
    )

    parity_true = print_metric_block(
        "B. VAL_TRUE - validator-like multi_label=True",
        result_true,
        official,
    )

    # ========================================================
    # Comparison
    # ========================================================

    section(
        "A/B comparison"
    )

    print(
        "Official:"
    )

    print(
        f"  mAP50-95 = "
        f"{official['map5095']:.6f}"
    )

    print()

    print(
        "SUBMIT_FALSE:"
    )

    print(
        f"  mAP50-95 = "
        f"{result_false['map5095']:.6f}"
    )

    print(
        f"  delta     = "
        f"{parity_false['delta']:+.6f}"
    )

    print()

    print(
        "VAL_TRUE:"
    )

    print(
        f"  mAP50-95 = "
        f"{result_true['map5095']:.6f}"
    )

    print(
        f"  delta     = "
        f"{parity_true['delta']:+.6f}"
    )

    nms_delta = (
        result_true[
            "map5095"
        ]
        - result_false[
            "map5095"
        ]
    )

    print()

    print(
        "multi_label=True "
        "minus False:"
    )

    print(
        f"  {nms_delta:+.6f}"
    )

    if (
        result_true[
            "map5095"
        ]
        > result_false[
            "map5095"
        ]
    ):

        validation_winner = (
            "multi_label=True"
        )

    elif (
        result_true[
            "map5095"
        ]
        < result_false[
            "map5095"
        ]
    ):

        validation_winner = (
            "multi_label=False"
        )

    else:

        validation_winner = "tie"

    print()

    print(
        "Manual val winner:",
        validation_winner,
    )

    # ========================================================
    # Save report
    # ========================================================

    report = {
        "protocol": {
            "model": str(
                MODEL_PATH
            ),
            "val_images": EXPECTED_VAL_IMAGES,
            "channels": EXPECTED_CHANNELS,
            "imgsz": IMAGE_SIZE,
            "manual_letterbox": "square",
            "batch": BATCH_SIZE,
            "precision": "fp32",
            "conf": CONF_THRESHOLD,
            "iou": IOU_THRESHOLD,
            "max_det": MAX_DET,
        },
        "official_reference": official,
        "submit_false": result_false,
        "submit_false_parity": parity_false,
        "val_true": result_true,
        "val_true_parity": parity_true,
        "multi_label_true_minus_false_map5095": float(
            nms_delta
        ),
        "manual_val_winner": validation_winner,
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    section(
        "FINAL RESULT"
    )

    print(
        "400 aligned multimodal val samples   PASS"
    )

    print(
        "5-channel manual loading             PASS"
    )

    print(
        "square 960 submission LetterBox      PASS"
    )

    print(
        "best.pt forward                      PASS"
    )

    print(
        "per-image NMS                        PASS"
    )

    print(
        "original-coordinate restoration      PASS"
    )

    print(
        "YOLO GT matching                     PASS"
    )

    print(
        "COCO-style AP calculation            PASS"
    )

    print()

    print(
        "Current submission parity:"
    )

    print(
        f"  {parity_false['status']}"
    )

    print()

    print(
        "Validator-like NMS parity:"
    )

    print(
        f"  {parity_true['status']}"
    )

    print()

    print(
        "Report saved:"
    )

    print(
        OUTPUT_JSON
    )

    print()
    print(
        "NEXT:"
    )

    print(
        "Compare these results before "
        "running the 1000-image test set."
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 Exp04 rect test submission.

Frozen variables from the passed val400 rect A/B:
  model       = runs/exp04_rgbid_early5_yolo11s_960/weights/best.pt
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

This script intentionally reuses the already validated project helpers:
  - infer_exp04_rgbid_early5_960_submit.py
      only for test-set discovery and TXT coordinate formatting
  - test_exp04_modality_ablation_val400.py
      for 5ch model loading, per-image NMS and box restoration
  - test_exp04_rect_inference_parity_val400.py
      for the exact rect preprocessing that passed val400 parity

It does NOT call the legacy submit script's load_model()/predict_batch().
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

import infer_exp04_rgbid_early5_960_submit as submit_base
import test_exp04_modality_ablation_val400 as base
import test_exp04_rect_inference_parity_val400 as rect


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = PROJECT_ROOT / "datasets"

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "exp04_rgbid_early5_yolo11s_960"
    / "weights"
    / "best.pt"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "exp04_rgbid_early5_yolo11s_960"
    / "submission_test_rect"
)

TXT_DIR = OUTPUT_ROOT / "txt"

ZIP_PATH = (
    OUTPUT_ROOT
    / "exp04_rgbid_early5_yolo11s_960_submit_rect.zip"
)

EXPECTED_TEST_SAMPLES = 1000
EXPECTED_CLASSES = 12
EXPECTED_CHANNELS = 5
MAX_DET = 100

FORWARD_BATCH = 4
DEVICE_INDEX = 0


def section(title: str) -> None:

    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def resolve_test_root(
    text: str | None,
) -> Path | None:

    if text is None:
        return None

    path = Path(text)

    return (
        path
        if path.is_absolute()
        else PROJECT_ROOT / path
    ).resolve()


def validate_txt_dir(
    expected_stems: list[str],
) -> dict:

    files = sorted(
        TXT_DIR.glob("*.txt")
    )

    if len(files) != EXPECTED_TEST_SAMPLES:

        raise RuntimeError(
            "TXT count mismatch: "
            f"expected={EXPECTED_TEST_SAMPLES}, "
            f"actual={len(files)}"
        )

    expected = set(
        expected_stems
    )

    actual = {
        p.stem
        for p in files
    }

    if actual != expected:

        raise RuntimeError(
            "TXT stem mismatch. "
            f"missing={sorted(expected - actual)[:20]}, "
            f"extra={sorted(actual - expected)[:20]}"
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
                f"{path.name}: "
                f"{len(lines)} boxes > {MAX_DET}"
            )

        max_boxes_seen = max(
            max_boxes_seen,
            len(lines),
        )

        total_boxes += len(
            lines
        )

        for line_no, line in enumerate(
            lines,
            start=1,
        ):

            fields = line.split()

            if len(fields) != 6:

                raise RuntimeError(
                    f"Bad line "
                    f"{path}:{line_no}: "
                    f"{line}"
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
                    f"Non-numeric line "
                    f"{path}:{line_no}"
                ) from exc

            if not (
                0
                <= class_id
                < EXPECTED_CLASSES
            ):

                raise RuntimeError(
                    f"Illegal class id "
                    f"in {path.name}: "
                    f"{class_id}"
                )

            for value in values:

                if not (
                    math.isfinite(value)
                    and 0.0 <= value <= 1.0
                ):

                    raise RuntimeError(
                        f"Illegal value in "
                        f"{path.name}: "
                        f"{value}"
                    )

    return {
        "txt_files":
        len(files),

        "total_boxes":
        total_boxes,

        "empty_files":
        empty_files,

        "max_boxes":
        max_boxes_seen,
    }


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
            TXT_DIR.glob("*.txt")
        ):

            zf.write(
                path,
                arcname=path.name,
            )

    with zipfile.ZipFile(
        ZIP_PATH,
        "r",
    ) as zf:

        names = zf.namelist()

    if len(names) != EXPECTED_TEST_SAMPLES:

        raise RuntimeError(
            "ZIP count mismatch: "
            f"expected={EXPECTED_TEST_SAMPLES}, "
            f"actual={len(names)}"
        )

    if any(
        "/" in name
        for name in names
    ):

        raise RuntimeError(
            "ZIP contains a subdirectory; "
            "TXT files must be at ZIP root."
        )

    expected_names = {
        f"{stem}.txt"
        for stem in expected_stems
    }

    if set(names) != expected_names:

        raise RuntimeError(
            "ZIP filenames do not "
            "exactly match test stems."
        )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Exp04 rect test inference "
            "and submission ZIP generation."
        )
    )

    parser.add_argument(
        "--test-root",
        type=str,
        default=None,
        help=(
            "Optional test root; otherwise "
            "auto-detect under datasets/."
        ),
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Audit data/model/rect geometry only; "
            "do not infer."
        ),
    )

    args = parser.parse_args()

    section(
        "AIC2026 Exp04 Rect Submission"
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
        OUTPUT_ROOT,
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

    # ========================================================
    # Discover test set
    # ========================================================

    section(
        "Discover test set"
    )

    manual_root = resolve_test_root(
        args.test_root
    )

    candidate, stems = (
        submit_base.discover_test_set(
            manual_root
        )
    )

    if len(stems) != EXPECTED_TEST_SAMPLES:

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_TEST_SAMPLES} stems, "
            f"got {len(stems)}"
        )

    print(
        "Selected root    :",
        candidate["root"],
    )

    print(
        "RGB              :",
        candidate["rgb_dir"],
    )

    print(
        "IR               :",
        candidate["ir_dir"],
    )

    print(
        "Depth            :",
        candidate["depth_dir"],
    )

    print(
        "Aligned samples  :",
        len(stems),
    )

    print(
        "First / last     :",
        stems[0],
        "/",
        stems[-1],
    )

    first = stems[0]

    first_image, first_hw = (
        base.load_5ch(
            candidate["rgb_map"][first],
            candidate["ir_map"][first],
            candidate["depth_map"][first],
        )
    )

    print(
        "First raw 5ch    :",
        first_image.shape,
        first_image.dtype,
    )

    print(
        "First original HW:",
        first_hw,
    )

    # ========================================================
    # Frozen Exp04 best.pt
    # ========================================================

    section(
        "Load frozen Exp04 best.pt"
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA unavailable"
        )

    device = torch.device(
        f"cuda:{DEVICE_INDEX}"
    )

    _, model = base.load_model(
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
            "Unexpected model: "
            f"in_ch={first_conv.in_channels}, "
            f"nc={nc}"
        )

    stride = rect.get_stride(
        model
    )

    print(
        "Device           :",
        device,
    )

    print(
        "GPU              :",
        torch.cuda.get_device_name(
            DEVICE_INDEX
        ),
    )

    print(
        "Stride           :",
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
        candidate["rgb_map"],
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
            f"source {h}x{w}: "
            f"{count}"
        )

    groups = rect.build_rect_groups(
        stems,
        shape_by_stem,
        stride,
    )

    rect_counts = Counter(
        group["shape"]
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
            f"rect   {h}x{w}: "
            f"{count} logical batches"
        )

    ratios = sorted(
        {
            round(
                h / float(w),
                8,
            )
            for h, w
            in shape_by_stem.values()
        }
    )

    print(
        "H/W ratios       :",
        ratios,
    )

    resized, sx, sy = (
        rect.resize_long_side_like_dataset(
            first_image
        )
    )

    print(
        "First long-side  :",
        resized.shape[:2],
    )

    print(
        "First pre-scale  :",
        f"{sx:.6f}",
        f"{sy:.6f}",
    )

    # Build one real rect input tensor.
    first_group = groups[0]

    check_stems = (
        first_group["stems"][
            :
            min(
                FORWARD_BATCH,
                len(
                    first_group["stems"]
                ),
            )
        ]
    )

    (
        check_batch,
        check_metas,
    ) = rect.prepare_rect_batch(
        check_stems,
        first_group["shape"],
        candidate["rgb_map"],
        candidate["ir_map"],
        candidate["depth_map"],
        device,
    )

    print(
        "First rect tensor:",
        tuple(
            check_batch.shape
        ),
    )

    del check_batch
    del check_metas

    torch.cuda.empty_cache()

    if args.check_only:

        section(
            "CHECK ONLY"
        )

        print(
            "STATUS = PASS"
        )

        return

    # ========================================================
    # Clean NEW rect output only
    # ========================================================

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

    # ========================================================
    # Rect inference
    # ========================================================

    section(
        "Rect inference"
    )

    total_forward = sum(
        math.ceil(
            len(
                group["stems"]
            )
            / FORWARD_BATCH
        )
        for group in groups
    )

    forward_idx = 0
    processed = 0
    tracked_boxes = 0
    tracked_empty = 0

    t0 = time.time()

    for group in groups:

        for start in range(
            0,
            len(
                group["stems"]
            ),
            FORWARD_BATCH,
        ):

            forward_idx += 1

            batch_stems = (
                group["stems"][
                    start:
                    start
                    + FORWARD_BATCH
                ]
            )

            (
                batch,
                metas,
            ) = rect.prepare_rect_batch(
                batch_stems,
                group["shape"],
                candidate["rgb_map"],
                candidate["ir_map"],
                candidate["depth_map"],
                device,
            )

            # IMPORTANT:
            # This is the validated implementation:
            #
            # multi_label=True
            # per-image NMS
            # conf=0.001
            # IoU=0.70
            # max_det=100
            detections = (
                base.predict_batch(
                    model,
                    batch,
                )
            )

            if len(
                detections
            ) != len(
                metas
            ):

                raise RuntimeError(
                    "NMS output batch "
                    "length mismatch"
                )

            for det, meta in zip(
                detections,
                metas,
            ):

                txt_path = (
                    TXT_DIR
                    / f"{meta['stem']}.txt"
                )

                # Competition requires empty TXT,
                # not missing files.
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

                order = torch.argsort(
                    det[:, 4],
                    descending=True,
                )

                det = (
                    det[
                        order
                    ][
                        :MAX_DET
                    ]
                )

                boxes = (
                    base.restore_xyxy_to_original(
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
                )

                # IMPORTANT:
                # Only reuse the old submit script's
                # pure TXT formatter here.
                #
                # We DO NOT use its predict_batch().
                lines = (
                    submit_base.xyxy_to_submission(
                        xyxy=boxes,
                        conf=det[:, 4],
                        cls=det[:, 5],
                        original_shape=meta[
                            "original_shape"
                        ],
                    )[
                        :MAX_DET
                    ]
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

            if (
                forward_idx == 1
                or forward_idx % 10 == 0
                or forward_idx
                == total_forward
            ):

                h, w = group["shape"]

                print(
                    f"[Rect] "
                    f"batch "
                    f"{forward_idx}/"
                    f"{total_forward} "
                    f"({processed}/"
                    f"{len(stems)} images) "
                    f"shape={h}x{w} "
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

    # ========================================================
    # TXT audit
    # ========================================================

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

    print(
        "Inference seconds:",
        f"{elapsed:.2f}",
    )

    if (
        stats["total_boxes"]
        != tracked_boxes
    ):

        raise RuntimeError(
            "Box-count mismatch: "
            f"files="
            f"{stats['total_boxes']}, "
            f"tracked="
            f"{tracked_boxes}"
        )

    if (
        stats["empty_files"]
        != tracked_empty
    ):

        raise RuntimeError(
            "Empty-count mismatch: "
            f"files="
            f"{stats['empty_files']}, "
            f"tracked="
            f"{tracked_empty}"
        )

    # ========================================================
    # ZIP
    # ========================================================

    section(
        "Create submission ZIP"
    )

    create_and_validate_zip(
        stems
    )

    print(
        "TXT directory    :",
        TXT_DIR,
    )

    print(
        "ZIP              :",
        ZIP_PATH,
    )

    print(
        "ZIP size         :",
        f"{ZIP_PATH.stat().st_size / 1024**2:.2f} MB",
    )

    # ========================================================
    # Final
    # ========================================================

    section(
        "FINAL RESULT"
    )

    print(
        "Frozen Exp04 best.pt               PASS"
    )

    print(
        "5-channel RGB/IR/Depth alignment   PASS"
    )

    print(
        "Validated rect preprocessing       PASS"
    )

    print(
        "multi_label=True                   PASS"
    )

    print(
        "Per-image NMS                      PASS"
    )

    print(
        "1000 same-name TXT files           PASS"
    )

    print(
        "Empty TXT files preserved          PASS"
    )

    print(
        "max_det <= 100                     PASS"
    )

    print(
        "ZIP files directly at root         PASS"
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

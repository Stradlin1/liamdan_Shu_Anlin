#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 Exp05
YOLO11s architecture / feature-shape audit.

Purpose:
    Before implementing Exp05 asymmetric gated feature fusion, inspect the
    ACTUAL YOLO11s model loaded from:

        pretrained/yolo11s.pt

    under the CURRENT editable Ultralytics source.

The script reports:

    1. Project / PyTorch / Ultralytics environment.
    2. Actual Ultralytics source path and Git commit.
    3. Top-level YOLO layer graph.
    4. Layer index / from / module / parameter count.
    5. Runtime input/output tensor shapes at imgsz=960.
    6. Effective feature stride.
    7. Backbone feature maps reused by the neck.
    8. Detect input layers and their feature shapes.
    9. A JSON report saved under runs/architecture_audits/.

Important:
    This script is READ-ONLY with respect to the model.
    It does not modify weights, YAML files, or Ultralytics source.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import ultralytics
from ultralytics import YOLO


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "yolo11s.pt"
)

LOCAL_ULTRALYTICS_DIR = (
    PROJECT_ROOT
    / "ultralytics"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "architecture_audits"
)


# ============================================================
# Helpers
# ============================================================

def get_git_commit(repo_dir: Path) -> str:
    """
    Return Git HEAD for a local repository.

    If unavailable, return a readable marker instead of aborting the audit.
    """

    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_dir),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        return result.stdout.strip()

    except Exception as exc:
        return f"UNAVAILABLE ({type(exc).__name__})"


def resolve_device(requested: str) -> torch.device:
    """
    Resolve:
        auto
        cpu
        cuda
        cuda:0
        ...
    """

    requested = requested.strip().lower()

    if requested == "auto":

        if torch.cuda.is_available():
            return torch.device("cuda:0")

        return torch.device("cpu")

    device = torch.device(requested)

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA device requested, but torch.cuda.is_available() is False."
        )

    return device


def shape_repr(obj: Any, depth: int = 0) -> str:
    """
    Produce a compact recursive representation of tensor shapes.

    Examples:

        Tensor:
            (1, 256, 60, 60)

        Concat / Detect inputs:
            [(1,256,60,60), (1,256,60,60)]

        Detect output:
            ((1,...), [...])
    """

    if depth > 4:
        return "..."

    if isinstance(obj, torch.Tensor):
        return str(tuple(int(v) for v in obj.shape))

    if isinstance(obj, list):

        parts = [
            shape_repr(
                item,
                depth=depth + 1,
            )
            for item in obj[:8]
        ]

        if len(obj) > 8:
            parts.append("...")

        return "[" + ", ".join(parts) + "]"

    if isinstance(obj, tuple):

        parts = [
            shape_repr(
                item,
                depth=depth + 1,
            )
            for item in obj[:8]
        ]

        if len(obj) > 8:
            parts.append("...")

        return "(" + ", ".join(parts) + ")"

    if isinstance(obj, dict):

        parts = []

        for index, (key, value) in enumerate(
            obj.items()
        ):
            if index >= 8:
                parts.append("...")
                break

            parts.append(
                f"{key}:"
                f"{shape_repr(value, depth=depth + 1)}"
            )

        return "{" + ", ".join(parts) + "}"

    if obj is None:
        return "None"

    return type(obj).__name__


def direct_4d_meta(obj: Any) -> dict[str, int] | None:
    """
    Return shape metadata only when output is directly BCHW tensor.
    """

    if not isinstance(obj, torch.Tensor):
        return None

    if obj.ndim != 4:
        return None

    return {
        "batch": int(obj.shape[0]),
        "channels": int(obj.shape[1]),
        "height": int(obj.shape[2]),
        "width": int(obj.shape[3]),
    }


def module_name(module: nn.Module) -> str:
    """
    Return readable Ultralytics module name.
    """

    name = getattr(
        module,
        "type",
        module.__class__.__name__,
    )

    name = str(name)

    prefixes = (
        "ultralytics.nn.modules.",
        "torch.nn.modules.",
    )

    for prefix in prefixes:
        name = name.replace(
            prefix,
            "",
        )

    return name


def get_first_conv2d(
    detection_model: nn.Module,
) -> nn.Conv2d:
    """
    Find first actual Conv2d under YOLO top-level layer 0.
    """

    if not hasattr(
        detection_model,
        "model",
    ):
        raise RuntimeError(
            "Detection model has no '.model' layer container."
        )

    layers = detection_model.model

    if len(layers) == 0:
        raise RuntimeError(
            "Detection model layer container is empty."
        )

    for module in layers[0].modules():

        if isinstance(
            module,
            nn.Conv2d,
        ):
            return module

    raise RuntimeError(
        "Could not locate Conv2d inside model layer 0."
    )


def calculate_stride(
    imgsz: int,
    meta: dict[str, int] | None,
) -> str:
    """
    Infer effective H/W stride from a direct BCHW feature map.
    """

    if meta is None:
        return "-"

    h = meta["height"]
    w = meta["width"]

    if h <= 0 or w <= 0:
        return "-"

    stride_h = imgsz / h
    stride_w = imgsz / w

    if abs(
        stride_h
        - stride_w
    ) < 1e-9:

        if abs(
            stride_h
            - round(stride_h)
        ) < 1e-9:

            return str(
                int(
                    round(
                        stride_h
                    )
                )
            )

        return f"{stride_h:.4f}"

    return (
        f"{stride_h:.3f}"
        f"x"
        f"{stride_w:.3f}"
    )


def normalize_from(value: Any) -> list[int]:
    """
    Normalize Ultralytics layer .f into list[int].
    """

    if isinstance(value, int):
        return [value]

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            int(v)
            for v in value
        ]

    return []


def shorten(
    text: str,
    width: int,
) -> str:

    if len(text) <= width:
        return text

    if width <= 3:
        return text[:width]

    return (
        text[: width - 3]
        + "..."
    )


# ============================================================
# Runtime feature audit
# ============================================================

def capture_runtime_shapes(
    detection_model: nn.Module,
    imgsz: int,
    device: torch.device,
) -> tuple[
    dict[int, dict[str, Any]],
    Any,
]:
    """
    Attach hooks to every top-level YOLO layer and perform one real forward.

    Only shape metadata is retained.
    Actual feature tensors are NOT retained.
    """

    layers = detection_model.model

    records: dict[
        int,
        dict[str, Any],
    ] = {}

    handles = []

    for index, layer in enumerate(
        layers
    ):

        records[index] = {
            "index": index,
            "from": getattr(
                layer,
                "f",
                None,
            ),
            "module": module_name(
                layer
            ),
            "params": int(
                sum(
                    p.numel()
                    for p in layer.parameters()
                )
            ),
            "input": "NOT_CAPTURED",
            "output": "NOT_CAPTURED",
            "output_4d": None,
        }

        def make_pre_hook(
            layer_index: int,
        ):
            def pre_hook(
                module: nn.Module,
                args: tuple[Any, ...],
            ) -> None:

                if len(args) == 1:
                    obj = args[0]
                else:
                    obj = args

                records[
                    layer_index
                ][
                    "input"
                ] = shape_repr(
                    obj
                )

            return pre_hook

        def make_post_hook(
            layer_index: int,
        ):
            def post_hook(
                module: nn.Module,
                args: tuple[Any, ...],
                output: Any,
            ) -> None:

                records[
                    layer_index
                ][
                    "output"
                ] = shape_repr(
                    output
                )

                records[
                    layer_index
                ][
                    "output_4d"
                ] = direct_4d_meta(
                    output
                )

            return post_hook

        handles.append(
            layer.register_forward_pre_hook(
                make_pre_hook(
                    index
                )
            )
        )

        handles.append(
            layer.register_forward_hook(
                make_post_hook(
                    index
                )
            )
        )

    dummy = torch.zeros(
        (
            1,
            3,
            imgsz,
            imgsz,
        ),
        dtype=torch.float32,
        device=device,
    )

    final_output = None

    try:

        with torch.inference_mode():

            final_output = detection_model(
                dummy
            )

        if device.type == "cuda":
            torch.cuda.synchronize(
                device
            )

    finally:

        for handle in handles:
            handle.remove()

    return (
        records,
        final_output,
    )


# ============================================================
# Reporting
# ============================================================

def print_environment(
    model_path: Path,
    device: torch.device,
) -> dict[str, Any]:

    ultralytics_file = Path(
        ultralytics.__file__
    ).resolve()

    ultralytics_git_commit = (
        get_git_commit(
            LOCAL_ULTRALYTICS_DIR
        )
    )

    print("=" * 100)
    print(
        "AIC2026 Exp05 - YOLO11s Architecture Audit"
    )
    print("=" * 100)

    print(
        "Project root       :",
        PROJECT_ROOT,
    )

    print(
        "Checkpoint         :",
        model_path,
    )

    print(
        "PyTorch            :",
        torch.__version__,
    )

    print(
        "Ultralytics version:",
        getattr(
            ultralytics,
            "__version__",
            "UNKNOWN",
        ),
    )

    print(
        "Ultralytics import :",
        ultralytics_file,
    )

    print(
        "Ultralytics git    :",
        ultralytics_git_commit,
    )

    print(
        "Device             :",
        device,
    )

    if device.type == "cuda":

        index = (
            device.index
            if device.index is not None
            else 0
        )

        print(
            "CUDA device        :",
            torch.cuda.get_device_name(
                index
            ),
        )

    expected_root = (
        LOCAL_ULTRALYTICS_DIR
        .resolve()
    )

    try:
        using_project_source = (
            ultralytics_file
            .is_relative_to(
                expected_root
            )
        )
    except AttributeError:
        using_project_source = (
            str(
                ultralytics_file
            ).startswith(
                str(
                    expected_root
                )
            )
        )

    print(
        "Project source     :",
        (
            "PASS"
            if using_project_source
            else "WARNING - imported Ultralytics is outside project/ultralytics"
        ),
    )

    print("=" * 100)

    return {
        "project_root": str(
            PROJECT_ROOT
        ),
        "checkpoint": str(
            model_path
        ),
        "torch_version": str(
            torch.__version__
        ),
        "ultralytics_version": str(
            getattr(
                ultralytics,
                "__version__",
                "UNKNOWN",
            )
        ),
        "ultralytics_import": str(
            ultralytics_file
        ),
        "ultralytics_git_commit": (
            ultralytics_git_commit
        ),
        "device": str(
            device
        ),
        "project_ultralytics_source": bool(
            using_project_source
        ),
    }


def print_model_metadata(
    detection_model: nn.Module,
    imgsz: int,
) -> tuple[int, dict[str, Any]]:

    yaml_cfg = getattr(
        detection_model,
        "yaml",
        {},
    )

    if not isinstance(
        yaml_cfg,
        dict,
    ):
        yaml_cfg = {}

    backbone_cfg = yaml_cfg.get(
        "backbone",
        [],
    )

    head_cfg = yaml_cfg.get(
        "head",
        [],
    )

    backbone_count = len(
        backbone_cfg
    )

    layer_count = len(
        detection_model.model
    )

    first_conv = get_first_conv2d(
        detection_model
    )

    print()
    print(
        "MODEL METADATA"
    )
    print("-" * 100)

    print(
        "yaml_file           :",
        getattr(
            detection_model,
            "yaml_file",
            "UNKNOWN",
        ),
    )

    print(
        "scale               :",
        yaml_cfg.get(
            "scale",
            "UNKNOWN",
        ),
    )

    print(
        "nc                  :",
        yaml_cfg.get(
            "nc",
            getattr(
                detection_model,
                "nc",
                "UNKNOWN",
            ),
        ),
    )

    print(
        "top-level layers    :",
        layer_count,
    )

    print(
        "backbone YAML layers:",
        backbone_count,
    )

    print(
        "head YAML layers    :",
        len(
            head_cfg
        ),
    )

    print(
        "model.save          :",
        getattr(
            detection_model,
            "save",
            "UNKNOWN",
        ),
    )

    print(
        "audit input         :",
        f"[1, 3, {imgsz}, {imgsz}]",
    )

    print(
        "first Conv2d        :",
        first_conv,
    )

    print(
        "first conv weight   :",
        tuple(
            int(v)
            for v
            in first_conv.weight.shape
        ),
    )

    if first_conv.in_channels != 3:

        raise RuntimeError(
            "Architecture audit expects original RGB YOLO11s "
            "with first Conv2d in_channels=3, "
            f"but found {first_conv.in_channels}."
        )

    print("-" * 100)

    metadata = {
        "yaml_file": str(
            getattr(
                detection_model,
                "yaml_file",
                "UNKNOWN",
            )
        ),
        "scale": yaml_cfg.get(
            "scale",
            None,
        ),
        "nc": yaml_cfg.get(
            "nc",
            getattr(
                detection_model,
                "nc",
                None,
            ),
        ),
        "top_level_layers": (
            layer_count
        ),
        "backbone_layers": (
            backbone_count
        ),
        "head_layers": len(
            head_cfg
        ),
        "save": list(
            getattr(
                detection_model,
                "save",
                [],
            )
        ),
        "first_conv": {
            "in_channels": int(
                first_conv.in_channels
            ),
            "out_channels": int(
                first_conv.out_channels
            ),
            "kernel_size": list(
                first_conv.kernel_size
            ),
            "stride": list(
                first_conv.stride
            ),
            "padding": list(
                first_conv.padding
            ),
            "weight_shape": list(
                first_conv.weight.shape
            ),
        },
    }

    return (
        backbone_count,
        metadata,
    )


def print_layer_table(
    records: dict[int, dict[str, Any]],
    backbone_count: int,
    imgsz: int,
) -> None:

    print()
    print(
        "ACTUAL TOP-LEVEL LAYER GRAPH @ "
        f"{imgsz}x{imgsz}"
    )
    print("=" * 180)

    header = (
        f"{'idx':>3}  "
        f"{'part':<8}  "
        f"{'from':<14}  "
        f"{'module':<30}  "
        f"{'params':>10}  "
        f"{'stride':>7}  "
        f"{'input shape':<43}  "
        f"{'output shape'}"
    )

    print(
        header
    )

    print("-" * 180)

    for index in sorted(
        records
    ):

        rec = records[
            index
        ]

        part = (
            "backbone"
            if index < backbone_count
            else "head"
        )

        stride = calculate_stride(
            imgsz=imgsz,
            meta=rec[
                "output_4d"
            ],
        )

        from_text = str(
            rec["from"]
        )

        module_text = str(
            rec["module"]
        )

        input_text = shorten(
            rec["input"],
            43,
        )

        output_text = shorten(
            rec["output"],
            70,
        )

        print(
            f"{index:>3}  "
            f"{part:<8}  "
            f"{from_text:<14}  "
            f"{module_text:<30}  "
            f"{rec['params']:>10,}  "
            f"{stride:>7}  "
            f"{input_text:<43}  "
            f"{output_text}"
        )

    print("=" * 180)


def collect_backbone_sources_used_by_head(
    detection_model: nn.Module,
    backbone_count: int,
) -> list[int]:

    sources = set()

    for index, layer in enumerate(
        detection_model.model
    ):

        if index < backbone_count:
            continue

        refs = normalize_from(
            getattr(
                layer,
                "f",
                -1,
            )
        )

        for ref in refs:

            if (
                ref >= 0
                and ref < backbone_count
            ):
                sources.add(
                    ref
                )

    return sorted(
        sources
    )


def print_backbone_feature_routes(
    detection_model: nn.Module,
    records: dict[int, dict[str, Any]],
    backbone_count: int,
    imgsz: int,
) -> list[dict[str, Any]]:

    sources = (
        collect_backbone_sources_used_by_head(
            detection_model=detection_model,
            backbone_count=backbone_count,
        )
    )

    entries = []

    print()
    print(
        "BACKBONE FEATURES REUSED BY THE NECK"
    )
    print("-" * 100)

    if not sources:

        print(
            "No explicit backbone -> head references found."
        )

        return entries

    sortable = []

    for source in sources:

        meta = records[
            source
        ][
            "output_4d"
        ]

        stride_text = calculate_stride(
            imgsz=imgsz,
            meta=meta,
        )

        try:
            stride_num = float(
                stride_text
            )
        except ValueError:
            stride_num = 999999.0

        sortable.append(
            (
                stride_num,
                source,
            )
        )

    sortable.sort()

    for (
        stride_num,
        source,
    ) in sortable:

        rec = records[
            source
        ]

        meta = rec[
            "output_4d"
        ]

        stride_text = calculate_stride(
            imgsz=imgsz,
            meta=meta,
        )

        level = "-"

        if stride_text == "8":
            level = "P3/8"

        elif stride_text == "16":
            level = "P4/16"

        elif stride_text == "32":
            level = "P5/32"

        print(
            f"{level:<6}  "
            f"layer={source:<3}  "
            f"module={rec['module']:<28}  "
            f"shape={rec['output']:<24}  "
            f"stride={stride_text}"
        )

        entries.append(
            {
                "level": level,
                "layer": source,
                "module": rec[
                    "module"
                ],
                "shape": rec[
                    "output"
                ],
                "stride": stride_text,
                "output_4d": meta,
            }
        )

    print("-" * 100)

    return entries


def print_head_external_routes(
    detection_model: nn.Module,
    records: dict[int, dict[str, Any]],
    backbone_count: int,
    imgsz: int,
) -> list[dict[str, Any]]:

    print()
    print(
        "HEAD LAYERS THAT DIRECTLY READ BACKBONE FEATURES"
    )
    print("-" * 120)

    entries = []

    for index, layer in enumerate(
        detection_model.model
    ):

        if index < backbone_count:
            continue

        refs = normalize_from(
            getattr(
                layer,
                "f",
                -1,
            )
        )

        backbone_refs = [
            ref
            for ref in refs
            if (
                ref >= 0
                and ref < backbone_count
            )
        ]

        if not backbone_refs:
            continue

        target_module = module_name(
            layer
        )

        for source in backbone_refs:

            source_record = records[
                source
            ]

            stride = calculate_stride(
                imgsz=imgsz,
                meta=source_record[
                    "output_4d"
                ],
            )

            print(
                f"head layer {index:<3} "
                f"{target_module:<28} "
                f"<- backbone layer {source:<3} "
                f"{source_record['module']:<28} "
                f"shape={source_record['output']} "
                f"stride={stride}"
            )

            entries.append(
                {
                    "head_layer": index,
                    "head_module": (
                        target_module
                    ),
                    "backbone_layer": (
                        source
                    ),
                    "backbone_module": (
                        source_record[
                            "module"
                        ]
                    ),
                    "shape": (
                        source_record[
                            "output"
                        ]
                    ),
                    "stride": stride,
                }
            )

    print("-" * 120)

    return entries


def print_detect_inputs(
    detection_model: nn.Module,
    records: dict[int, dict[str, Any]],
    imgsz: int,
) -> list[dict[str, Any]]:

    print()
    print(
        "DETECT INPUT FEATURES"
    )
    print("-" * 100)

    entries = []

    detect_found = False

    for index, layer in enumerate(
        detection_model.model
    ):

        name = module_name(
            layer
        )

        class_name = (
            layer
            .__class__
            .__name__
        )

        if (
            "Detect" not in name
            and "Detect" not in class_name
        ):
            continue

        detect_found = True

        refs = normalize_from(
            getattr(
                layer,
                "f",
                -1,
            )
        )

        print(
            f"Detect layer = {index}"
        )

        print(
            f"Detect from  = {refs}"
        )

        for source in refs:

            if source < 0:
                continue

            rec = records[
                source
            ]

            stride = calculate_stride(
                imgsz=imgsz,
                meta=rec[
                    "output_4d"
                ],
            )

            level = "-"

            if stride == "8":
                level = "P3/8"

            elif stride == "16":
                level = "P4/16"

            elif stride == "32":
                level = "P5/32"

            print(
                f"  {level:<6} "
                f"layer={source:<3} "
                f"module={rec['module']:<28} "
                f"shape={rec['output']:<24} "
                f"stride={stride}"
            )

            entries.append(
                {
                    "detect_layer": index,
                    "level": level,
                    "source_layer": source,
                    "source_module": (
                        rec[
                            "module"
                        ]
                    ),
                    "shape": (
                        rec[
                            "output"
                        ]
                    ),
                    "stride": stride,
                    "output_4d": (
                        rec[
                            "output_4d"
                        ]
                    ),
                }
            )

    if not detect_found:

        print(
            "WARNING: no Detect layer identified."
        )

    print("-" * 100)

    return entries


def print_fusion_candidate_summary(
    backbone_features: list[dict[str, Any]],
) -> None:

    print()
    print(
        "EXP05 FUSION-POINT AUDIT SUMMARY"
    )
    print("=" * 100)

    print(
        "The script does NOT choose the Exp05 fusion point automatically."
    )

    print(
        "Use the runtime results below to decide where "
        "Depth and IR should enter the RGB path."
    )

    print()

    for item in backbone_features:

        print(
            f"{item['level']:<6} "
            f"backbone layer {item['layer']:<3} "
            f"{item['module']:<28} "
            f"{item['shape']}"
        )

    print()
    print(
        "Next decision:"
    )

    print(
        "  1. exact RGB fusion layer"
    )

    print(
        "  2. Depth fusion layer"
    )

    print(
        "  3. IR fusion layer"
    )

    print(
        "  4. whether Depth and IR fuse at the same or different stage"
    )

    print(
        "  5. gate initialization and auxiliary-stem channel widths"
    )

    print("=" * 100)


def save_json_report(
    imgsz: int,
    environment: dict[str, Any],
    model_metadata: dict[str, Any],
    records: dict[int, dict[str, Any]],
    backbone_features: list[dict[str, Any]],
    head_routes: list[dict[str, Any]],
    detect_inputs: list[dict[str, Any]],
) -> Path:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        REPORT_DIR
        / (
            "exp05_yolo11s_"
            f"architecture_{imgsz}.json"
        )
    )

    serializable_layers = []

    for index in sorted(
        records
    ):

        rec = records[
            index
        ]

        serializable_layers.append(
            {
                "index": int(
                    rec[
                        "index"
                    ]
                ),
                "from": (
                    rec[
                        "from"
                    ]
                ),
                "module": str(
                    rec[
                        "module"
                    ]
                ),
                "params": int(
                    rec[
                        "params"
                    ]
                ),
                "input": str(
                    rec[
                        "input"
                    ]
                ),
                "output": str(
                    rec[
                        "output"
                    ]
                ),
                "output_4d": (
                    rec[
                        "output_4d"
                    ]
                ),
                "effective_stride": (
                    calculate_stride(
                        imgsz=imgsz,
                        meta=rec[
                            "output_4d"
                        ],
                    )
                ),
            }
        )

    report = {
        "experiment": (
            "Exp05 YOLO11s architecture audit"
        ),
        "imgsz": imgsz,
        "environment": environment,
        "model": model_metadata,
        "layers": serializable_layers,
        "backbone_features_used_by_head": (
            backbone_features
        ),
        "head_backbone_routes": (
            head_routes
        ),
        "detect_inputs": (
            detect_inputs
        ),
    }

    path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return path


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Audit actual YOLO11s architecture and "
            "runtime feature shapes for Exp05."
        )
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help=(
            "Square audit input size. "
            "Default: 960"
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "auto, cpu, cuda, cuda:0, ... "
            "Default: auto"
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=(
            "YOLO checkpoint path. "
            "Default: project/pretrained/yolo11s.pt"
        ),
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    if args.imgsz <= 0:
        raise ValueError(
            f"Invalid imgsz: {args.imgsz}"
        )

    if args.imgsz % 32 != 0:
        raise ValueError(
            "For this audit, imgsz must be divisible by 32. "
            f"Got {args.imgsz}."
        )

    model_path = Path(
        args.model
    ).resolve()

    if not model_path.is_file():
        raise FileNotFoundError(
            "YOLO11s checkpoint not found:\n"
            f"  {model_path}"
        )

    device = resolve_device(
        args.device
    )

    environment = print_environment(
        model_path=model_path,
        device=device,
    )

    print()
    print(
        "Loading actual pretrained YOLO11s checkpoint..."
    )

    yolo = YOLO(
        str(
            model_path
        )
    )

    detection_model = yolo.model

    if detection_model is None:
        raise RuntimeError(
            "YOLO wrapper contains no underlying model."
        )

    # Shape audit only. Force FP32 for predictable execution.
    detection_model = (
        detection_model
        .float()
        .to(
            device
        )
        .eval()
    )

    (
        backbone_count,
        model_metadata,
    ) = print_model_metadata(
        detection_model=detection_model,
        imgsz=args.imgsz,
    )

    print()
    print(
        "Running one real forward pass for runtime hooks..."
    )

    (
        records,
        final_output,
    ) = capture_runtime_shapes(
        detection_model=detection_model,
        imgsz=args.imgsz,
        device=device,
    )

    print(
        "Forward output      :",
        shorten(
            shape_repr(
                final_output
            ),
            180,
        ),
    )

    print_layer_table(
        records=records,
        backbone_count=backbone_count,
        imgsz=args.imgsz,
    )

    backbone_features = (
        print_backbone_feature_routes(
            detection_model=detection_model,
            records=records,
            backbone_count=backbone_count,
            imgsz=args.imgsz,
        )
    )

    head_routes = (
        print_head_external_routes(
            detection_model=detection_model,
            records=records,
            backbone_count=backbone_count,
            imgsz=args.imgsz,
        )
    )

    detect_inputs = (
        print_detect_inputs(
            detection_model=detection_model,
            records=records,
            imgsz=args.imgsz,
        )
    )

    print_fusion_candidate_summary(
        backbone_features=backbone_features,
    )

    report_path = save_json_report(
        imgsz=args.imgsz,
        environment=environment,
        model_metadata=model_metadata,
        records=records,
        backbone_features=backbone_features,
        head_routes=head_routes,
        detect_inputs=detect_inputs,
    )

    print()
    print(
        "JSON report saved:"
    )

    print(
        f"  {report_path}"
    )

    print()
    print("=" * 100)
    print(
        "ARCHITECTURE AUDIT: PASS"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()

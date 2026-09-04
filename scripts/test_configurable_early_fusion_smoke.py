#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Smoke-test configurable RGB+IR and RGB+Depth early fusion.

Run from the project environment:

    python scripts/test_configurable_early_fusion_smoke.py

The test uses one real validation sample plus synthetic detection targets. It
does not train an epoch and does not write a run directory.
"""

from __future__ import annotations

import gc
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG, get_cfg

from multimodal_config import (
    SUPPORTED_MODALITY_CONFIGS,
    channel_names_for_modalities,
    channels_for_modalities,
    normalize_modalities,
)
from multimodal_dataset import (
    DEPTH_VIEW_ROOT,
    IR_VIEW_ROOT,
    RGB_VIEW_ROOT,
    MultimodalYOLODataset,
    RGBOnlyRandomHSV,
    _read_grayscale_uint8,
    _resize_gray_to_hw,
)
from multimodal_model import (
    PRETRAINED_MODEL_PATH,
    build_yolo11s_multimodal,
    get_first_conv2d,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = RGB_VIEW_ROOT / "data.yaml"
IMAGE_SIZE = 320
SEED = 2026

CASES = (
    ("RGBI", ("rgb", "ir"), "ir", IR_VIEW_ROOT),
    ("RGBD", ("rgb", "depth"), "depth", DEPTH_VIEW_ROOT),
)


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def preflight() -> dict:
    missing = [
        path
        for path in (PRETRAINED_MODEL_PATH, DATA_YAML)
        if not path.is_file()
    ]
    for view_root in (RGB_VIEW_ROOT, IR_VIEW_ROOT, DEPTH_VIEW_ROOT):
        if not (view_root / "images" / "val").is_dir():
            missing.append(view_root / "images" / "val")
    if missing:
        raise FileNotFoundError(
            "Smoke-test inputs are missing:\n  "
            + "\n  ".join(str(path) for path in missing)
        )

    with DATA_YAML.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    check(isinstance(data, dict) and "names" in data, "Invalid data.yaml")
    return data


def make_dataset(
    modalities: tuple[str, ...],
    data: dict,
) -> MultimodalYOLODataset:
    hyp = deepcopy(DEFAULT_CFG)
    hyp.bgr = 0.0

    unused_root = PROJECT_ROOT / "_unused_modality_must_not_be_read"
    return MultimodalYOLODataset(
        img_path=str(RGB_VIEW_ROOT / "images" / "val"),
        imgsz=IMAGE_SIZE,
        cache=False,
        augment=False,
        hyp=hyp,
        prefix=f"smoke {'+'.join(modalities)}: ",
        rect=False,
        batch_size=1,
        stride=32,
        pad=0.5,
        single_cls=False,
        classes=None,
        fraction=1.0,
        data=data,
        task="detect",
        modalities=modalities,
        ir_view_root=(IR_VIEW_ROOT if "ir" in modalities else unused_root),
        depth_view_root=(
            DEPTH_VIEW_ROOT if "depth" in modalities else unused_root
        ),
    )


def test_configuration_contract() -> None:
    section("Configuration contract")
    expected = {
        ("rgb",): ("R", "G", "B"),
        ("rgb", "ir"): ("R", "G", "B", "IR"),
        ("rgb", "depth"): ("R", "G", "B", "Depth"),
        ("rgb", "ir", "depth"): ("R", "G", "B", "IR", "Depth"),
    }
    check(tuple(expected) == SUPPORTED_MODALITY_CONFIGS, "Supported configs changed")
    for modalities, channel_names in expected.items():
        check(normalize_modalities(modalities) == modalities, "Normalization failed")
        check(
            channel_names_for_modalities(modalities) == channel_names,
            f"Wrong channel order for {modalities}",
        )
        check(
            channels_for_modalities(modalities) == len(channel_names),
            f"Wrong channel count for {modalities}",
        )
    for invalid in (("ir",), ("rgb", "depth", "ir"), ("rgb", "rgb")):
        try:
            normalize_modalities(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid configuration accepted: {invalid}")
    print("[PASS] supported combinations, counts, order, and rejection paths")


def test_four_channel_hsv_isolation() -> None:
    section("Four-channel HSV isolation")
    image = np.random.default_rng(SEED).integers(
        0,
        256,
        size=(32, 48, 4),
        dtype=np.uint8,
    )
    auxiliary_before = image[..., 3].copy()
    transform = RGBOnlyRandomHSV(
        hgain=0.015,
        sgain=0.7,
        vgain=0.4,
        expected_channels=4,
    )
    result = transform.apply_image({"img": image})["img"]
    check(
        np.array_equal(result[..., 3], auxiliary_before),
        "HSV changed the selected auxiliary channel",
    )
    print("[PASS] HSV is restricted to channels R/G/B")


def test_dataset_case(
    name: str,
    modalities: tuple[str, ...],
    auxiliary_name: str,
    auxiliary_root: Path,
    data: dict,
) -> None:
    section(f"{name} dataset")
    dataset = make_dataset(modalities, data)
    check(dataset.modalities == modalities, "Dataset modalities mismatch")
    check(dataset.num_channels == 4, "Dataset must have four channels")
    check(
        dataset.channel_names == channel_names_for_modalities(modalities),
        "Dataset channel order mismatch",
    )

    paths = dataset.get_multimodal_paths(0)
    check(set(paths) == {"rgb", auxiliary_name}, "Dataset read an unused modality")
    check(paths[auxiliary_name].is_file(), "Selected auxiliary path is missing")
    check(auxiliary_root in paths[auxiliary_name].parents, "Wrong auxiliary view")

    raw = dataset.get_image_and_label(0)
    image = raw["img"]
    check(image.ndim == 3 and image.shape[2] == 4, f"Raw shape={image.shape}")
    check(image.dtype == np.uint8, f"Raw dtype={image.dtype}")

    rgb_bgr = cv2.imread(str(paths["rgb"]), cv2.IMREAD_COLOR)
    target_hw = tuple(int(value) for value in raw["resized_shape"])
    if rgb_bgr.shape[:2] != target_hw:
        rgb_bgr = cv2.resize(
            rgb_bgr,
            (target_hw[1], target_hw[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    expected_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    check(np.array_equal(image[..., :3], expected_rgb), "RGB channel mismatch")

    expected_auxiliary = _read_grayscale_uint8(
        paths[auxiliary_name],
        auxiliary_name,
    )
    expected_auxiliary = _resize_gray_to_hw(expected_auxiliary, target_hw)
    check(
        np.array_equal(image[..., 3], expected_auxiliary),
        f"{auxiliary_name} channel mismatch",
    )

    formatted = dataset[0]["img"]
    check(torch.is_tensor(formatted), "Formatted image is not a tensor")
    check(
        tuple(formatted.shape) == (4, IMAGE_SIZE, IMAGE_SIZE),
        f"Formatted shape={tuple(formatted.shape)}",
    )
    print(f"[PASS] {name}: paths, HWC order, and CHW Format")


def iter_tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from iter_tensors(item)


def test_model_case(
    name: str,
    modalities: tuple[str, ...],
    reference_weight: torch.Tensor,
    device: torch.device,
) -> None:
    section(f"{name} model")
    yolo = build_yolo11s_multimodal(
        pretrained_path=PRETRAINED_MODEL_PATH,
        modalities=modalities,
    )
    model = yolo.model.to(device)
    conv = get_first_conv2d(yolo)
    weight = conv.weight.detach().cpu()
    check(conv.in_channels == 4, "First convolution must have four channels")
    check(torch.equal(weight[:, :3], reference_weight), "RGB weights changed")
    check(torch.count_nonzero(weight[:, 3]).item() == 0, "Aux weight is nonzero")

    image = torch.rand((1, 4, IMAGE_SIZE, IMAGE_SIZE), device=device)
    model.eval()
    with torch.inference_mode():
        output = model(image)
    outputs = list(iter_tensors(output))
    check(outputs, "Forward returned no tensor")
    check(all(torch.isfinite(item).all() for item in outputs), "Non-finite forward")

    if isinstance(model.args, dict):
        model.args = get_cfg(overrides=model.args)
    if hasattr(model, "criterion"):
        delattr(model, "criterion")
    model.train()
    model.zero_grad(set_to_none=True)
    batch = {
        "img": image,
        "cls": torch.tensor([[0.0], [3.0]], device=device),
        "bboxes": torch.tensor(
            [[0.35, 0.40, 0.18, 0.24], [0.68, 0.62, 0.22, 0.20]],
            device=device,
        ),
        "batch_idx": torch.tensor([0.0, 0.0], device=device),
    }
    loss_result = model(batch)
    check(isinstance(loss_result, (tuple, list)), "Loss result type changed")
    loss = loss_result[0]
    check(torch.isfinite(loss).all(), "Loss is not finite")
    loss.sum().backward()
    gradient = conv.weight.grad
    check(gradient is not None, "First-conv gradient is missing")
    check(torch.isfinite(gradient).all(), "Gradient is not finite")
    check(torch.count_nonzero(gradient[:, :3]).item() > 0, "RGB gradient is zero")
    check(torch.count_nonzero(gradient[:, 3]).item() > 0, "Aux gradient is zero")
    print(f"[PASS] {name}: init, forward, detection loss, and backward")

    del output, outputs, loss_result, loss, image, batch, model, yolo
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = preflight()
    test_configuration_contract()
    test_four_channel_hsv_isolation()
    for case in CASES:
        test_dataset_case(*case, data=data)

    reference = YOLO(str(PRETRAINED_MODEL_PATH))
    reference_weight = get_first_conv2d(reference).weight.detach().cpu().clone()
    del reference
    gc.collect()

    for name, modalities, _, _ in CASES:
        test_model_case(name, modalities, reference_weight, device)

    section("RESULT")
    print("[PASS] configurable RGBI/RGBD early-fusion smoke test")


if __name__ == "__main__":
    main()

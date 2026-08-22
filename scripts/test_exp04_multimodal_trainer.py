#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 Exp04
MultimodalDetectionTrainer integration smoke test.

This test verifies the real integration:

    pretrained/yolo11s.pt
            ↓
    MultimodalDetectionTrainer
            ↓
    5-channel / 12-class DetectionModel
            ↓
    MultimodalYOLODataset
            ↓
    DataLoader
            ↓
    real [R,G,B,IR,Depth] batch
            ↓
    preprocess /255
            ↓
    YOLO detection loss
            ↓
    backward
            ↓
    IR / Depth gradients

This is NOT an epoch training test yet.
"""

from __future__ import annotations

import gc
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO

from multimodal_dataset import (
    CHANNEL_NAMES,
    MULTIMODAL_CHANNELS,
    MultimodalYOLODataset,
)

from multimodal_trainer import (
    MultimodalDetectionTrainer,
)


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

PRETRAINED_MODEL_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "yolo11s.pt"
)

DATA_YAML = (
    PROJECT_ROOT
    / "yolo_views"
    / "rgb_v1"
    / "data.yaml"
)

RUNS_DIR = (
    PROJECT_ROOT
    / "runs"
)

TEST_NAME = (
    "_test_exp04_multimodal_trainer"
)


# ============================================================
# Expected Exp04 configuration
# ============================================================

IMAGE_SIZE = 960
BATCH_SIZE = 1

EXPECTED_CHANNELS = 5
EXPECTED_CLASSES = 12

EXPECTED_TRAIN_SAMPLES = 1600
EXPECTED_VAL_SAMPLES = 400

SEED = 2026


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


def pass_line(
    text: str,
) -> None:

    print(
        f"[PASS] {text}"
    )


def get_first_conv(
    model: nn.Module,
) -> nn.Conv2d:

    if not hasattr(
        model,
        "model",
    ):
        raise RuntimeError(
            "Model has no '.model' attribute."
        )

    first_block = model.model[0]

    if not hasattr(
        first_block,
        "conv",
    ):
        raise RuntimeError(
            "First YOLO block has no '.conv'."
        )

    conv = first_block.conv

    if not isinstance(
        conv,
        nn.Conv2d,
    ):
        raise TypeError(
            "First convolution is not nn.Conv2d."
        )

    return conv


def check_files() -> None:

    if not PRETRAINED_MODEL_PATH.is_file():

        raise FileNotFoundError(
            "Pretrained model not found:\n"
            f"  {PRETRAINED_MODEL_PATH}"
        )

    if not DATA_YAML.is_file():

        raise FileNotFoundError(
            "Dataset YAML not found:\n"
            f"  {DATA_YAML}"
        )

    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def assert_finite(
    tensor: torch.Tensor,
    name: str,
) -> None:

    if not torch.isfinite(
        tensor
    ).all():

        raise AssertionError(
            f"{name} contains NaN or Inf."
        )


# ============================================================
# Reference pretrained RGB stem
# ============================================================

def load_reference_rgb_weight() -> torch.Tensor:

    reference = YOLO(
        str(
            PRETRAINED_MODEL_PATH
        )
    )

    conv = get_first_conv(
        reference.model
    )

    if conv.in_channels != 3:

        raise AssertionError(
            "Pretrained reference model "
            f"must be 3-channel, got {conv.in_channels}."
        )

    weight = (
        conv.weight
        .detach()
        .cpu()
        .clone()
    )

    del reference

    gc.collect()

    return weight


# ============================================================
# Trainer construction
# ============================================================

def build_trainer():
    """
    Instantiate Trainer only.

    trainer.train() is deliberately NOT called here.
    """

    device = (
        "0"
        if torch.cuda.is_available()
        else "cpu"
    )

    overrides = {
        "model": str(
            PRETRAINED_MODEL_PATH
        ),

        "data": str(
            DATA_YAML
        ),

        "task": "detect",

        "imgsz": IMAGE_SIZE,

        "batch": BATCH_SIZE,

        "epochs": 1,

        "device": device,

        "workers": 0,

        "project": str(
            RUNS_DIR
        ),

        "name": TEST_NAME,

        "exist_ok": True,

        "save": False,

        "plots": False,

        "val": True,

        "cache": False,

        # Explicit FP32 for this integration test.
        "amp": False,

        "seed": SEED,

        "deterministic": True,

        # Keep Exp04 augmentation policy.
        "mosaic": 1.0,

        "mixup": 0.0,

        "cutmix": 0.0,

        "copy_paste": 0.0,

        "fliplr": 0.5,

        # Avoid requiring trainer.stride inside
        # preprocess_batch before full training setup.
        "multi_scale": 0.0,
    }

    return MultimodalDetectionTrainer(
        overrides=overrides
    )


# ============================================================
# Dataset metadata test
# ============================================================

def test_dataset_metadata(
    trainer,
) -> None:

    section(
        "1. Dataset metadata"
    )

    print(
        "data path:",
        trainer.data["path"],
    )

    print(
        "train:",
        trainer.data["train"],
    )

    print(
        "val:",
        trainer.data["val"],
    )

    print(
        "nc:",
        trainer.data["nc"],
    )

    print(
        "channels:",
        trainer.data["channels"],
    )

    print(
        "names:",
        trainer.data["names"],
    )

    if (
        trainer.data["channels"]
        != EXPECTED_CHANNELS
    ):

        raise AssertionError(
            "Trainer dataset metadata "
            f"channels={trainer.data['channels']}, "
            f"expected {EXPECTED_CHANNELS}."
        )

    pass_line(
        "data['channels'] == 5"
    )

    if (
        trainer.data["nc"]
        != EXPECTED_CLASSES
    ):

        raise AssertionError(
            "Dataset class count mismatch: "
            f"{trainer.data['nc']}"
        )

    pass_line(
        "data['nc'] == 12"
    )


# ============================================================
# Model construction test
# ============================================================

def test_model(
    trainer,
    pretrained_rgb_weight: torch.Tensor,
) -> None:

    section(
        "2. Trainer model construction"
    )

    # BaseTrainer.__init__ only stores the model path.
    # setup_model() performs the actual model construction
    # through MultimodalDetectionTrainer.get_model().
    trainer.setup_model()

    trainer.model = (
        trainer.model.to(
            trainer.device
        )
    )

    # Normal DetectionTrainer training setup attaches:
    #
    #   nc
    #   names
    #   args
    #
    # Do it explicitly for this standalone integration test.
    trainer.set_model_attributes()

    model = trainer.model

    first_conv = get_first_conv(
        model
    )

    detect_head = (
        model.model[-1]
    )

    print(
        "Device:",
        trainer.device,
    )

    print(
        "First conv:",
        first_conv,
    )

    print(
        "First conv weight shape:",
        tuple(
            first_conv.weight.shape
        ),
    )

    print(
        "Detect head:",
        type(
            detect_head
        ).__name__,
    )

    print(
        "Detect head nc:",
        getattr(
            detect_head,
            "nc",
            None,
        ),
    )

    # --------------------------------------------------------
    # 5-channel model
    # --------------------------------------------------------

    if (
        first_conv.in_channels
        != MULTIMODAL_CHANNELS
    ):

        raise AssertionError(
            "Trainer model is not 5-channel."
        )

    pass_line(
        "trainer model input channels == 5"
    )

    # --------------------------------------------------------
    # 12-class detection head
    # --------------------------------------------------------

    if (
        getattr(
            detect_head,
            "nc",
            None,
        )
        != EXPECTED_CLASSES
    ):

        raise AssertionError(
            "Detection head is not 12-class."
        )

    pass_line(
        "Detect head nc == 12"
    )

    # --------------------------------------------------------
    # RGB pretrained exact copy
    # --------------------------------------------------------

    target_weight = (
        first_conv
        .weight
        .detach()
        .cpu()
    )

    if not torch.equal(
        target_weight[:, 0:3],
        pretrained_rgb_weight,
    ):

        max_diff = (
            target_weight[:, 0:3]
            - pretrained_rgb_weight
        ).abs().max().item()

        raise AssertionError(
            "Trainer RGB pretrained stem mismatch. "
            f"max_abs_diff={max_diff}"
        )

    pass_line(
        "RGB pretrained stem copied exactly"
    )

    # --------------------------------------------------------
    # IR / Depth zero initialization
    # --------------------------------------------------------

    ir_nonzero = torch.count_nonzero(
        target_weight[:, 3]
    ).item()

    depth_nonzero = torch.count_nonzero(
        target_weight[:, 4]
    ).item()

    print(
        "IR initial nonzero:",
        ir_nonzero,
    )

    print(
        "Depth initial nonzero:",
        depth_nonzero,
    )

    if ir_nonzero != 0:

        raise AssertionError(
            "IR stem is not zero initialized."
        )

    if depth_nonzero != 0:

        raise AssertionError(
            "Depth stem is not zero initialized."
        )

    pass_line(
        "IR stem == 0"
    )

    pass_line(
        "Depth stem == 0"
    )


# ============================================================
# DataLoader integration
# ============================================================

def find_good_real_train_batch(
    train_loader,
):
    """
    Find one real batch that contains:

        at least one GT object
        non-zero IR data
        non-zero Depth data

    This makes the gradient test meaningful.
    """

    iterator = iter(
        train_loader
    )

    max_attempts = 20

    for attempt in range(
        1,
        max_attempts + 1,
    ):

        batch = next(
            iterator
        )

        img = batch["img"]

        has_gt = (
            batch["cls"].numel()
            > 0
        )

        ir_nonzero = (
            torch.count_nonzero(
                img[:, 3]
            ).item()
        )

        depth_nonzero = (
            torch.count_nonzero(
                img[:, 4]
            ).item()
        )

        if (
            has_gt
            and ir_nonzero > 0
            and depth_nonzero > 0
        ):

            print(
                "Selected train batch attempt:",
                attempt,
            )

            print(
                "Instances:",
                batch["cls"].shape[0],
            )

            print(
                "IR nonzero pixels:",
                ir_nonzero,
            )

            print(
                "Depth nonzero pixels:",
                depth_nonzero,
            )

            return batch

    raise RuntimeError(
        "Could not find a train batch with "
        "GT + nonzero IR + nonzero Depth "
        f"within {max_attempts} attempts."
    )


def test_dataloaders(
    trainer,
):
    section(
        "3. Dataset + DataLoader integration"
    )

    train_loader = (
        trainer.get_dataloader(
            dataset_path=trainer.data[
                "train"
            ],
            batch_size=BATCH_SIZE,
            rank=-1,
            mode="train",
        )
    )

    val_loader = (
        trainer.get_dataloader(
            dataset_path=trainer.data[
                "val"
            ],
            batch_size=BATCH_SIZE,
            rank=-1,
            mode="val",
        )
    )

    train_dataset = (
        train_loader.dataset
    )

    val_dataset = (
        val_loader.dataset
    )

    print(
        "Train dataset class:",
        type(
            train_dataset
        ).__name__,
    )

    print(
        "Val dataset class:",
        type(
            val_dataset
        ).__name__,
    )

    print(
        "Train samples:",
        len(
            train_dataset
        ),
    )

    print(
        "Val samples:",
        len(
            val_dataset
        ),
    )

    # --------------------------------------------------------
    # Correct custom Dataset class
    # --------------------------------------------------------

    if not isinstance(
        train_dataset,
        MultimodalYOLODataset,
    ):

        raise AssertionError(
            "Train DataLoader is not using "
            "MultimodalYOLODataset."
        )

    if not isinstance(
        val_dataset,
        MultimodalYOLODataset,
    ):

        raise AssertionError(
            "Val DataLoader is not using "
            "MultimodalYOLODataset."
        )

    pass_line(
        "train loader uses MultimodalYOLODataset"
    )

    pass_line(
        "val loader uses MultimodalYOLODataset"
    )

    # --------------------------------------------------------
    # Expected split sizes
    # --------------------------------------------------------

    if (
        len(train_dataset)
        != EXPECTED_TRAIN_SAMPLES
    ):

        raise AssertionError(
            "Unexpected train sample count: "
            f"{len(train_dataset)}"
        )

    if (
        len(val_dataset)
        != EXPECTED_VAL_SAMPLES
    ):

        raise AssertionError(
            "Unexpected val sample count: "
            f"{len(val_dataset)}"
        )

    pass_line(
        "train samples == 1600"
    )

    pass_line(
        "val samples == 400"
    )

    # --------------------------------------------------------
    # Real train batch
    # --------------------------------------------------------

    train_batch = (
        find_good_real_train_batch(
            train_loader
        )
    )

    train_img = (
        train_batch["img"]
    )

    print()
    print(
        "Raw train batch shape:",
        tuple(
            train_img.shape
        ),
    )

    print(
        "Raw train batch dtype:",
        train_img.dtype,
    )

    print(
        "Raw train batch min/max:",
        train_img.min().item(),
        train_img.max().item(),
    )

    if train_img.ndim != 4:

        raise AssertionError(
            "Train img tensor must be BCHW."
        )

    if (
        train_img.shape[1]
        != MULTIMODAL_CHANNELS
    ):

        raise AssertionError(
            "Train batch is not 5-channel."
        )

    if (
        tuple(
            train_img.shape[-2:]
        )
        != (
            IMAGE_SIZE,
            IMAGE_SIZE,
        )
    ):

        raise AssertionError(
            "Train batch is not 960x960: "
            f"{tuple(train_img.shape)}"
        )

    if (
        train_img.dtype
        != torch.uint8
    ):

        raise AssertionError(
            "Raw train batch must be uint8."
        )

    pass_line(
        "real train batch == [B,5,960,960] uint8"
    )

    # --------------------------------------------------------
    # Real val batch
    #
    # Rectangular validation may use a non-square spatial size,
    # so only enforce BCHW + 5 channels here.
    # --------------------------------------------------------

    val_batch = next(
        iter(
            val_loader
        )
    )

    val_img = (
        val_batch["img"]
    )

    print()
    print(
        "Raw val batch shape:",
        tuple(
            val_img.shape
        ),
    )

    print(
        "Raw val batch dtype:",
        val_img.dtype,
    )

    if (
        val_img.ndim != 4
        or val_img.shape[1]
        != MULTIMODAL_CHANNELS
    ):

        raise AssertionError(
            "Validation batch is not 5-channel BCHW."
        )

    if (
        val_img.dtype
        != torch.uint8
    ):

        raise AssertionError(
            "Raw val batch must be uint8."
        )

    pass_line(
        "real val batch is 5-channel uint8"
    )

    return train_batch


# ============================================================
# Preprocess + real loss + backward
# ============================================================

def test_real_batch_backward(
    trainer,
    batch,
) -> None:

    section(
        "4. Real multimodal batch -> loss -> backward"
    )

    model = trainer.model

    # Trainer preprocessing:
    #
    # uint8 [0,255]
    #       ↓
    # float32 [0,1]
    batch = trainer.preprocess_batch(
        batch
    )

    img = batch["img"]

    print(
        "Processed batch shape:",
        tuple(
            img.shape
        ),
    )

    print(
        "Processed batch dtype:",
        img.dtype,
    )

    print(
        "Processed batch device:",
        img.device,
    )

    print(
        "Processed min/max:",
        img.min().item(),
        img.max().item(),
    )

    if (
        img.dtype
        != torch.float32
    ):

        raise AssertionError(
            "preprocess_batch did not convert "
            "image to float32."
        )

    if (
        img.shape[1]
        != MULTIMODAL_CHANNELS
    ):

        raise AssertionError(
            "preprocess_batch changed channel count."
        )

    if (
        img.min().item() < 0.0
        or img.max().item() > 1.0
    ):

        raise AssertionError(
            "Processed image outside [0,1]."
        )

    pass_line(
        "Trainer preprocess_batch /255"
    )

    # --------------------------------------------------------
    # Real YOLO detection loss
    # --------------------------------------------------------

    model.train()

    model.zero_grad(
        set_to_none=True
    )

    result = model(
        batch
    )

    if not isinstance(
        result,
        (tuple, list),
    ):

        raise AssertionError(
            "Expected model(batch) -> "
            "(loss, loss_items)."
        )

    if len(result) != 2:

        raise AssertionError(
            "Unexpected detection loss output."
        )

    loss = result[0]
    loss_items = result[1]

    assert_finite(
        loss,
        "Detection loss",
    )

    print()
    print(
        "Loss:",
        loss.detach().cpu(),
    )

    print(
        "Loss items:",
        {
            key: (
                value.detach().cpu()
                if torch.is_tensor(value)
                else value
            )
            for key, value
            in loss_items.items()
        }
        if isinstance(
            loss_items,
            dict,
        )
        else loss_items,
    )

    backward_loss = (
        loss.sum()
    )

    backward_loss.backward()

    pass_line(
        "real YOLO detection loss"
    )

    pass_line(
        "real loss backward"
    )

    # --------------------------------------------------------
    # Auxiliary-channel gradients
    # --------------------------------------------------------

    conv = get_first_conv(
        model
    )

    grad = conv.weight.grad

    if grad is None:

        raise AssertionError(
            "First-conv gradient is None."
        )

    assert_finite(
        grad,
        "First-conv gradient",
    )

    grad = (
        grad.detach().cpu()
    )

    rgb_grad = (
        grad[:, 0:3]
    )

    ir_grad = (
        grad[:, 3]
    )

    depth_grad = (
        grad[:, 4]
    )

    rgb_nonzero = (
        torch.count_nonzero(
            rgb_grad
        ).item()
    )

    ir_nonzero = (
        torch.count_nonzero(
            ir_grad
        ).item()
    )

    depth_nonzero = (
        torch.count_nonzero(
            depth_grad
        ).item()
    )

    print()
    print(
        "RGB grad nonzero:",
        rgb_nonzero,
    )

    print(
        "RGB grad abs sum:",
        rgb_grad.abs().sum().item(),
    )

    print(
        "IR grad nonzero:",
        ir_nonzero,
    )

    print(
        "IR grad abs sum:",
        ir_grad.abs().sum().item(),
    )

    print(
        "Depth grad nonzero:",
        depth_nonzero,
    )

    print(
        "Depth grad abs sum:",
        depth_grad.abs().sum().item(),
    )

    if rgb_nonzero == 0:

        raise AssertionError(
            "RGB first-conv gradient is zero."
        )

    if ir_nonzero == 0:

        raise AssertionError(
            "IR first-conv gradient is zero."
        )

    if depth_nonzero == 0:

        raise AssertionError(
            "Depth first-conv gradient is zero."
        )

    pass_line(
        "RGB real-batch gradient != 0"
    )

    pass_line(
        "IR real-batch gradient != 0"
    )

    pass_line(
        "Depth real-batch gradient != 0"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    check_files()

    section(
        "AIC2026 Exp04 - Multimodal Trainer integration test"
    )

    print(
        "Project root :",
        PROJECT_ROOT,
    )

    print(
        "Pretrained   :",
        PRETRAINED_MODEL_PATH,
    )

    print(
        "Dataset YAML :",
        DATA_YAML,
    )

    print(
        "Channels     :",
        MULTIMODAL_CHANNELS,
    )

    print(
        "Channel order:",
        CHANNEL_NAMES,
    )

    print(
        "Image size   :",
        IMAGE_SIZE,
    )

    print(
        "Batch size   :",
        BATCH_SIZE,
    )

    # Independent pretrained reference.
    pretrained_rgb_weight = (
        load_reference_rgb_weight()
    )

    trainer = (
        build_trainer()
    )

    test_dataset_metadata(
        trainer
    )

    test_model(
        trainer,
        pretrained_rgb_weight,
    )

    del pretrained_rgb_weight

    gc.collect()

    real_train_batch = (
        test_dataloaders(
            trainer
        )
    )

    test_real_batch_backward(
        trainer,
        real_train_batch,
    )

    section(
        "FINAL RESULT"
    )

    print(
        "Trainer import                           PASS"
    )

    print(
        "data channels = 5                       PASS"
    )

    print(
        "dataset classes = 12                    PASS"
    )

    print(
        "model input channels = 5                PASS"
    )

    print(
        "Detect head nc = 12                     PASS"
    )

    print(
        "RGB pretrained exact                    PASS"
    )

    print(
        "IR / Depth zero-init                    PASS"
    )

    print(
        "train MultimodalYOLODataset             PASS"
    )

    print(
        "val MultimodalYOLODataset               PASS"
    )

    print(
        "train samples = 1600                    PASS"
    )

    print(
        "val samples = 400                       PASS"
    )

    print(
        "real [B,5,960,960] DataLoader batch     PASS"
    )

    print(
        "Trainer preprocess /255                 PASS"
    )

    print(
        "real detection loss                     PASS"
    )

    print(
        "real backward                           PASS"
    )

    print(
        "real IR / Depth gradients != 0          PASS"
    )

    print()
    print(
        "STATUS = PASS"
    )

    print()
    print(
        "Next step:"
    )

    print(
        "train_exp04_rgbid_early5_smoke_960.py"
    )


if __name__ == "__main__":
    main()

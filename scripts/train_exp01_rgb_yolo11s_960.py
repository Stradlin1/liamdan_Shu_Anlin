#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import torch
from ultralytics import YOLO


# ============================================================
# AIC2026
# Exp01: RGB-only YOLO11s 960 Baseline
#
# 第一个正式 baseline。
#
# Exp01 RGB
#   ↓
# Exp02 Early Fusion
#   ↓
# Exp03 Depth Mask
#   ↓
# Exp04 Feature Fusion
#   ↓
# Exp05 Reliability Fusion
#
# 固定使用 V2 Train/Val split。
# ============================================================


# ============================================================
# 项目路径
#
# 根据当前脚本位置自动确定 aicomp 根目录。
# 禁止硬编码 /home/xhm/... 等机器相关路径。
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_YAML = (
    PROJECT_ROOT
    / "yolo_views"
    / "rgb_v1"
    / "data.yaml"
)

PROJECT_DIR = (
    PROJECT_ROOT
    / "runs"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "yolo11s.pt"
)

EXPERIMENT_NAME = (
    "exp01_rgb_yolo11s_960"
)


# ============================================================
# 训练参数
# ============================================================

IMAGE_SIZE = 960

EPOCHS = 200

PATIENCE = 50

BATCH_SIZE = 8

WORKERS = 8

DEVICE = 0

SEED = 2026


def check_environment():

    print("=" * 80)
    print("Environment")
    print("=" * 80)

    print(
        "PyTorch       :",
        torch.__version__
    )

    print(
        "CUDA runtime  :",
        torch.version.cuda
    )

    print(
        "CUDA available:",
        torch.cuda.is_available()
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA 不可用，停止训练。"
        )

    print(
        "GPU           :",
        torch.cuda.get_device_name(
            DEVICE
        )
    )

    gpu = torch.cuda.get_device_properties(
        DEVICE
    )

    print(
        "VRAM          :",
        f"{gpu.total_memory / 1024**3:.2f} GB"
    )

    print()


def check_files():

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML 不存在: {DATA_YAML}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"预训练权重不存在: {MODEL_PATH}"
        )

    PROJECT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def main():

    check_environment()

    check_files()

    print("=" * 80)
    print(
        "AIC2026 Exp01 - RGB YOLO11s 960"
    )
    print("=" * 80)

    print(
        f"Root        : {PROJECT_ROOT}"
    )

    print(
        f"Model       : {MODEL_PATH}"
    )

    print(
        f"Data        : {DATA_YAML}"
    )

    print(
        f"Image size  : {IMAGE_SIZE}"
    )

    print(
        f"Epochs      : {EPOCHS}"
    )

    print(
        f"Patience    : {PATIENCE}"
    )

    print(
        f"Batch       : {BATCH_SIZE}"
    )

    print(
        f"Device      : {DEVICE}"
    )

    print(
        f"Seed        : {SEED}"
    )

    print(
        f"Output      : "
        f"{PROJECT_DIR / EXPERIMENT_NAME}"
    )

    print("=" * 80)
    print()

    # --------------------------------------------------------
    # 项目 pretrained/ 中的 COCO pretrained YOLO11s
    # --------------------------------------------------------

    model = YOLO(
        str(MODEL_PATH)
    )

    # --------------------------------------------------------
    # 正式 RGB baseline
    # --------------------------------------------------------

    results = model.train(

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        data=str(
            DATA_YAML
        ),

        # ----------------------------------------------------
        # Model input
        # ----------------------------------------------------

        imgsz=IMAGE_SIZE,

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        epochs=EPOCHS,

        patience=PATIENCE,

        batch=BATCH_SIZE,

        device=DEVICE,

        workers=WORKERS,

        # ----------------------------------------------------
        # Reproducibility
        # ----------------------------------------------------

        seed=SEED,

        deterministic=True,

        # ----------------------------------------------------
        # Pretrained weights
        # ----------------------------------------------------

        pretrained=True,

        # ----------------------------------------------------
        # Precision
        # ----------------------------------------------------

        amp=True,

        # ----------------------------------------------------
        # Dataset cache
        # ----------------------------------------------------

        cache=False,

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        optimizer="auto",

        # ----------------------------------------------------
        # Augmentation
        # ----------------------------------------------------

        mosaic=1.0,

        close_mosaic=10,

        mixup=0.0,

        degrees=0.0,

        translate=0.1,

        scale=0.5,

        fliplr=0.5,

        flipud=0.0,

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val=True,

        plots=True,

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        project=str(
            PROJECT_DIR
        ),

        name=EXPERIMENT_NAME,

        # 正式训练不覆盖旧实验。
        # 如果重复运行，Ultralytics 创建新目录。
        exist_ok=False,

        save=True,

        verbose=True,
    )

    print()
    print("=" * 80)
    print("Exp01 finished")
    print("=" * 80)

    print(
        "Result root:",
        PROJECT_DIR
    )

    return results


if __name__ == "__main__":
    main()

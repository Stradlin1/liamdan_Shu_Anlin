#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from ultralytics import YOLO


# ============================================================
# AIC2026
# Exp00: RGB-only Smoke Test
#
# 目的：
#   1. 验证 Ultralytics 环境
#   2. 验证 GPU / CUDA
#   3. 验证 train / val 数据读取
#   4. 验证 12 类标签
#   5. 验证完整训练和验证流程
#
# 注意：
#   这不是正式 baseline，只跑 3 epochs。
# ============================================================


# ============================================================
# 项目路径
#
# 无论项目位于：
#   <PROJECT_ROOT>
#   /root/aicomp
#   /root/autodl-tmp/aicomp
#
# 都会自动定位项目根目录。
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "yolo11n.pt"
)

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

EXPERIMENT_NAME = (
    "exp00_rgb_smoke_640"
)


def check_files():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"预训练权重不存在: {MODEL_PATH}"
        )

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML 不存在: {DATA_YAML}"
        )

    PROJECT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def main():

    check_files()

    print("=" * 80)
    print("AIC2026 - Exp00 RGB Smoke Test")
    print("=" * 80)

    print(f"Root    : {PROJECT_ROOT}")
    print(f"Model   : {MODEL_PATH}")
    print(f"Data    : {DATA_YAML}")
    print(f"Project : {PROJECT_DIR}")
    print(f"Name    : {EXPERIMENT_NAME}")

    print("=" * 80)

    # --------------------------------------------------------
    # 加载项目 pretrained/ 中的 COCO 预训练 YOLO11n
    # --------------------------------------------------------

    model = YOLO(
        str(MODEL_PATH)
    )

    # --------------------------------------------------------
    # Smoke Test
    # --------------------------------------------------------

    results = model.train(

        # 数据
        data=str(DATA_YAML),

        # 输入分辨率
        imgsz=640,

        # Smoke test 只跑 3 个 epoch
        epochs=3,

        batch=16,

        # GPU 0
        device=0,

        # DataLoader
        workers=8,

        # 固定随机种子
        seed=2026,

        # 尽量保证实验可复现
        deterministic=True,

        # 暂时不缓存数据
        cache=False,

        # 输出
        project=str(PROJECT_DIR),
        name=EXPERIMENT_NAME,

        # Smoke test 允许覆盖同名目录
        exist_ok=True,

        save=True,

        val=True,

        plots=True,

        verbose=True,
    )

    print()
    print("=" * 80)
    print("Exp00 training finished.")
    print("=" * 80)

    print(
        "Results directory:",
        PROJECT_DIR / EXPERIMENT_NAME
    )

    return results


if __name__ == "__main__":
    main()

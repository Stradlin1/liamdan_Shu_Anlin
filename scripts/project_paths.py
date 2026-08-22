#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path


# ============================================================
# 项目根目录
#
# 当前文件位于：
# aicomp/scripts/project_paths.py
#
# parents[1] 即：
# aicomp/
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# 官方数据集
# ============================================================

DATASETS_ROOT = PROJECT_ROOT / "datasets"

DATASET_ROOT = (
    DATASETS_ROOT
    / "AIC2026_Train_2000"
)

VISIBLE_DIR = DATASET_ROOT / "visible"
INFRARED_DIR = DATASET_ROOT / "infrared"
DEPTH_DIR = DATASET_ROOT / "depth"
LABEL_DIR = DATASET_ROOT / "labels"


# ============================================================
# 固定 V2 Train / Val Split
# ============================================================

SPLITS_ROOT = PROJECT_ROOT / "splits"

SPLIT_ROOT = (
    SPLITS_ROOT
    / "aic2026_group_stratified_v2"
)

TRAIN_SPLIT = SPLIT_ROOT / "train.txt"
VAL_SPLIT = SPLIT_ROOT / "val.txt"


# ============================================================
# YOLO 数据视图
# ============================================================

YOLO_VIEWS_ROOT = (
    PROJECT_ROOT
    / "yolo_views"
)

RGB_YOLO_VIEW = (
    YOLO_VIEWS_ROOT
    / "rgb_v1"
)

RGB_DATA_YAML = (
    RGB_YOLO_VIEW
    / "data.yaml"
)


# ============================================================
# 公开预训练权重
# ============================================================

PRETRAINED_ROOT = (
    PROJECT_ROOT
    / "pretrained"
)

YOLO11N_WEIGHT = (
    PRETRAINED_ROOT
    / "yolo11n.pt"
)

YOLO11S_WEIGHT = (
    PRETRAINED_ROOT
    / "yolo11s.pt"
)


# ============================================================
# 实验结果
# ============================================================

RUNS_ROOT = PROJECT_ROOT / "runs"


# ============================================================
# 环境记录
# ============================================================

ENVIRONMENT_ROOT = (
    PROJECT_ROOT
    / "environment"
)


# ============================================================
# Ultralytics editable 源码
# ============================================================

ULTRALYTICS_ROOT = (
    PROJECT_ROOT
    / "ultralytics"
)


def print_paths():

    print("=" * 80)
    print("AIC2026 Project Paths")
    print("=" * 80)

    print("PROJECT_ROOT   :", PROJECT_ROOT)
    print("DATASET_ROOT   :", DATASET_ROOT)
    print("VISIBLE_DIR    :", VISIBLE_DIR)
    print("INFRARED_DIR   :", INFRARED_DIR)
    print("DEPTH_DIR      :", DEPTH_DIR)
    print("LABEL_DIR      :", LABEL_DIR)

    print("SPLIT_ROOT     :", SPLIT_ROOT)
    print("RGB_YOLO_VIEW  :", RGB_YOLO_VIEW)

    print("PRETRAINED_ROOT:", PRETRAINED_ROOT)
    print("RUNS_ROOT      :", RUNS_ROOT)

    print("ULTRALYTICS    :", ULTRALYTICS_ROOT)


if __name__ == "__main__":
    print_paths()

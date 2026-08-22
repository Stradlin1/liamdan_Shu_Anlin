#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import shutil
import zipfile

import torch
from ultralytics import YOLO


# ============================================================
# Exp01: RGB-only YOLO11s 960
# Phase 1 测试集提交结果生成脚本
#
# 输入:
#   datasets/AIC2026_PHASE_1_1000/visible
#
# 模型:
#   runs/exp01_rgb_yolo11s_960/weights/best.pt
#
# 输出:
#   runs/exp01_rgb_yolo11s_960/submission_phase1/
#
# 每张测试图对应一个同名 TXT:
#   class_id x_center y_center width height confidence
#
# 坐标全部为 [0, 1] 归一化坐标。
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "AIC2026_PHASE_1_1000"
)

VISIBLE_DIR = (
    TEST_ROOT
    / "visible"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "exp01_rgb_yolo11s_960"
    / "weights"
    / "best.pt"
)

SUBMISSION_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "exp01_rgb_yolo11s_960"
    / "submission_phase1"
)

PRED_DIR = (
    SUBMISSION_ROOT
    / "labels"
)

ZIP_PATH = (
    SUBMISSION_ROOT
    / "exp01_rgb_yolo11s_960_phase1_submit.zip"
)


# ============================================================
# 推理参数
# ============================================================

IMAGE_SIZE = 960

# mAP 计算需要尽量保留低置信度预测框，
# 因此提交推理不使用常见的 0.25 阈值。
CONF_THRESHOLD = 0.001

IOU_THRESHOLD = 0.70

# 比赛要求单张图片最多 100 个预测框
MAX_DET = 100

# RTX 5060 Laptop 8GB 下先使用较保守的 batch
BATCH_SIZE = 1

EXPECTED_IMAGE_COUNT = 1000

NUM_CLASSES = 12

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def check_paths():

    if not VISIBLE_DIR.exists():
        raise FileNotFoundError(
            f"测试集 visible 目录不存在:\n{VISIBLE_DIR}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"模型不存在:\n{MODEL_PATH}"
        )

    SUBMISSION_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )


def find_images():

    images = sorted(
        p
        for p in VISIBLE_DIR.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise RuntimeError(
            f"没有在以下目录找到测试图片:\n{VISIBLE_DIR}"
        )

    # --------------------------------------------------------
    # 检查 basename 是否重复。
    #
    # 因为提交格式要求 image.xxx -> image.txt，
    # 如果存在两个不同图片同名，就会产生 TXT 覆盖。
    # --------------------------------------------------------
    stems = [p.stem for p in images]

    if len(stems) != len(set(stems)):

        seen = set()
        duplicated = set()

        for stem in stems:
            if stem in seen:
                duplicated.add(stem)
            seen.add(stem)

        raise RuntimeError(
            "检测到重复图片 basename，无法安全生成提交文件:\n"
            + "\n".join(sorted(duplicated))
        )

    return images


def prepare_output():

    # 每次重新推理时清理旧结果，
    # 避免旧 TXT 混入新提交。
    if PRED_DIR.exists():
        shutil.rmtree(PRED_DIR)

    PRED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()


def save_prediction(result):

    image_path = Path(result.path)

    txt_path = (
        PRED_DIR
        / f"{image_path.stem}.txt"
    )

    boxes = result.boxes

    # --------------------------------------------------------
    # 没检测到任何目标：
    # 比赛仍要求存在同名空 TXT
    # --------------------------------------------------------
    if boxes is None or len(boxes) == 0:
        txt_path.touch()
        return 0

    xywhn = (
        boxes.xywhn
        .detach()
        .cpu()
    )

    classes = (
        boxes.cls
        .detach()
        .cpu()
    )

    confidences = (
        boxes.conf
        .detach()
        .cpu()
    )

    # --------------------------------------------------------
    # 再次显式按照 confidence 从高到低排序。
    # 即便 Ultralytics 已经排序，这里仍保证提交结果确定。
    # --------------------------------------------------------
    order = torch.argsort(
        confidences,
        descending=True
    )

    order = order[:MAX_DET]

    xywhn = xywhn[order]
    classes = classes[order]
    confidences = confidences[order]

    lines = []

    for cls_tensor, box_tensor, conf_tensor in zip(
        classes,
        xywhn,
        confidences,
    ):

        cls_id = int(
            cls_tensor.item()
        )

        if not 0 <= cls_id < NUM_CLASSES:
            raise RuntimeError(
                f"发现非法类别 ID: {cls_id}\n"
                f"图片: {image_path}"
            )

        x, y, w, h = [
            float(v)
            for v in box_tensor.tolist()
        ]

        confidence = float(
            conf_tensor.item()
        )

        # ----------------------------------------------------
        # 数值安全处理。
        # 正常 YOLO 输出本身就在 [0,1]，
        # 这里防止极端浮点误差造成非法坐标。
        # ----------------------------------------------------
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0)
        h = min(max(h, 0.0), 1.0)

        confidence = min(
            max(confidence, 0.0),
            1.0
        )

        lines.append(
            f"{cls_id} "
            f"{x:.8f} "
            f"{y:.8f} "
            f"{w:.8f} "
            f"{h:.8f} "
            f"{confidence:.8f}"
        )

    txt_path.write_text(
        "\n".join(lines)
        + ("\n" if lines else ""),
        encoding="utf-8",
    )

    return len(lines)


def validate_submission(image_paths):

    txt_files = sorted(
        PRED_DIR.glob("*.txt")
    )

    image_stems = {
        p.stem
        for p in image_paths
    }

    txt_stems = {
        p.stem
        for p in txt_files
    }

    missing = sorted(
        image_stems - txt_stems
    )

    extra = sorted(
        txt_stems - image_stems
    )

    if missing:
        raise RuntimeError(
            "以下测试图片缺少预测 TXT:\n"
            + "\n".join(missing[:20])
        )

    if extra:
        raise RuntimeError(
            "发现没有对应测试图片的多余 TXT:\n"
            + "\n".join(extra[:20])
        )

    if len(txt_files) != len(image_paths):
        raise RuntimeError(
            "预测 TXT 数量与测试图片数量不一致:\n"
            f"images = {len(image_paths)}\n"
            f"txt    = {len(txt_files)}"
        )

    # --------------------------------------------------------
    # 对每个 TXT 做基本格式检查
    # --------------------------------------------------------
    total_boxes = 0
    empty_files = 0

    for txt_path in txt_files:

        text = txt_path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            empty_files += 1
            continue

        lines = text.splitlines()

        if len(lines) > MAX_DET:
            raise RuntimeError(
                f"{txt_path.name} "
                f"预测框数量超过 {MAX_DET}: "
                f"{len(lines)}"
            )

        for line_index, line in enumerate(
            lines,
            start=1,
        ):

            parts = line.split()

            if len(parts) != 6:
                raise RuntimeError(
                    f"提交格式错误:\n"
                    f"{txt_path}:{line_index}\n"
                    f"期望 6 列，实际 {len(parts)} 列\n"
                    f"{line}"
                )

            cls_id = int(parts[0])

            if not 0 <= cls_id < NUM_CLASSES:
                raise RuntimeError(
                    f"非法 class_id:\n"
                    f"{txt_path}:{line_index}\n"
                    f"{cls_id}"
                )

            values = [
                float(v)
                for v in parts[1:]
            ]

            if not all(
                0.0 <= v <= 1.0
                for v in values
            ):
                raise RuntimeError(
                    f"发现超出 [0,1] 的预测值:\n"
                    f"{txt_path}:{line_index}\n"
                    f"{line}"
                )

        total_boxes += len(lines)

    return (
        len(txt_files),
        total_boxes,
        empty_files,
    )


def create_zip():

    # --------------------------------------------------------
    # ZIP 内不增加 labels/ 这一层目录。
    #
    # ZIP 打开后直接是：
    #   xxx.txt
    #   xxx.txt
    #   ...
    # --------------------------------------------------------
    with zipfile.ZipFile(
        ZIP_PATH,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zip_file:

        for txt_path in sorted(
            PRED_DIR.glob("*.txt")
        ):

            zip_file.write(
                txt_path,
                arcname=txt_path.name,
            )


def main():

    check_paths()

    image_paths = find_images()

    print("=" * 80)
    print("Exp01 Phase-1 RGB Submission Inference")
    print("=" * 80)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Model        : {MODEL_PATH}")
    print(f"Visible test : {VISIBLE_DIR}")
    print(f"Images       : {len(image_paths)}")
    print(f"Image size   : {IMAGE_SIZE}")
    print(f"Conf         : {CONF_THRESHOLD}")
    print(f"IoU          : {IOU_THRESHOLD}")
    print(f"Max det      : {MAX_DET}")
    print(f"Output TXT   : {PRED_DIR}")
    print(f"Output ZIP   : {ZIP_PATH}")
    print("=" * 80)

    if len(image_paths) != EXPECTED_IMAGE_COUNT:
        raise RuntimeError(
            "Phase-1 测试图片数量不是预期的 "
            f"{EXPECTED_IMAGE_COUNT} 张。\n"
            f"实际检测到: {len(image_paths)}\n"
            "为避免生成错误提交，脚本已停止。"
        )

    prepare_output()

    device = (
        0
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device       : {device}")

    if torch.cuda.is_available():
        print(
            "GPU          : "
            + torch.cuda.get_device_name(0)
        )

    print("=" * 80)

    model = YOLO(
        str(MODEL_PATH)
    )

    processed = 0
    total_predictions = 0

    for image_path in image_paths:

        results = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            max_det=MAX_DET,
            device=device,
            save=False,
            save_txt=False,
            save_conf=False,
            verbose=False,
            augment=False,
            agnostic_nms=False,
        )

        result = results[0]

        num_predictions = save_prediction(
            result
        )

        total_predictions += num_predictions
        processed += 1

        if (
            processed % 50 == 0
            or processed == len(image_paths)
        ):
            print(
                f"[{processed:4d}/{len(image_paths)}] "
                f"累计预测框: {total_predictions}"
            )

    print()
    print("开始检查提交结果...")

    (
        txt_count,
        total_boxes,
        empty_count,
    ) = validate_submission(
        image_paths
    )

    print("结果检查通过。")
    print(f"TXT 数量     : {txt_count}")
    print(f"预测框总数   : {total_boxes}")
    print(f"空 TXT 数量  : {empty_count}")

    print()
    print("开始创建 ZIP...")

    create_zip()

    zip_size_mb = (
        ZIP_PATH.stat().st_size
        / 1024
        / 1024
    )

    print()
    print("=" * 80)
    print("提交文件生成完成")
    print("=" * 80)
    print(f"模型          : {MODEL_PATH}")
    print(f"测试图片      : {len(image_paths)}")
    print(f"预测 TXT      : {txt_count}")
    print(f"预测框总数    : {total_boxes}")
    print(f"空 TXT        : {empty_count}")
    print(f"ZIP           : {ZIP_PATH}")
    print(f"ZIP 大小      : {zip_size_mb:.2f} MB")
    print("=" * 80)
    print()
    print("这个 ZIP 即为 Phase-1 排行榜提交文件。")


if __name__ == "__main__":
    main()

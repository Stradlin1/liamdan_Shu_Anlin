#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 - Build IR-only YOLO View

目标：
    构建：
        yolo_views/ir_v1/

原则：
1. 不重新划分 train / val。
2. 完全复用 rgb_v1 中已经固定的 train / val 成员。
3. images 指向官方 infrared 图像。
4. labels 指向与 rgb_v1 相同的官方标签。
5. 所有软链接均为相对软链接。
6. 不复制原始图像。
7. 不对 Infrared 做灰度化、归一化或其他预处理。
8. 构建前先完整检查，避免生成半成品目录。

预期结构：

yolo_views/ir_v1/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
"""

from pathlib import Path
import os


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "AIC2026_Train_2000"
)

RGB_VIEW_ROOT = (
    PROJECT_ROOT
    / "yolo_views"
    / "rgb_v1"
)

IR_VIEW_ROOT = (
    PROJECT_ROOT
    / "yolo_views"
    / "ir_v1"
)


# ============================================================
# Constants
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

SPLITS = (
    "train",
    "val",
)

EXPECTED_TOTAL_SAMPLES = 2000


# ============================================================
# Dataset path detection
# ============================================================

def find_infrared_dir():
    """
    Locate the official Infrared directory.

    Preferred project convention:
        datasets/AIC2026_Train_2000/infrared

    A few capitalization variants are accepted so that the
    builder does not depend on filesystem capitalization.
    """

    candidates = [
        DATASET_ROOT / "infrared",
        DATASET_ROOT / "Infrared",
        DATASET_ROOT / "INFRARED",
        DATASET_ROOT / "ir",
        DATASET_ROOT / "IR",
    ]

    found = [
        path
        for path in candidates
        if path.is_dir()
    ]

    # Remove duplicates after resolve().
    unique = []

    seen = set()

    for path in found:

        real = path.resolve()

        if real not in seen:
            seen.add(real)
            unique.append(path)

    if len(unique) == 1:
        return unique[0]

    if len(unique) > 1:
        raise RuntimeError(
            "检测到多个可能的 Infrared 目录，"
            "为避免使用错误数据，停止构建：\n"
            + "\n".join(
                f"  {path}"
                for path in unique
            )
        )

    children = []

    if DATASET_ROOT.is_dir():

        children = sorted(
            path.name
            for path in DATASET_ROOT.iterdir()
            if path.is_dir()
        )

    raise FileNotFoundError(
        "没有找到 Infrared 数据目录。\n\n"
        f"Dataset root:\n  {DATASET_ROOT}\n\n"
        "尝试过：\n"
        + "\n".join(
            f"  {path}"
            for path in candidates
        )
        + "\n\n"
        "当前 Dataset root 下的目录：\n"
        + (
            "\n".join(
                f"  {name}"
                for name in children
            )
            if children
            else "  <none>"
        )
    )


# ============================================================
# Helpers
# ============================================================

def list_images(directory):
    """Return all image files directly inside a YOLO split."""

    if not directory.is_dir():
        raise FileNotFoundError(
            f"目录不存在：{directory}"
        )

    images = [
        path
        for path in directory.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]

    return sorted(
        images,
        key=lambda p: p.name,
    )


def build_ir_index(infrared_dir):
    """
    Build lookup tables for Infrared files.

    Normally RGB / Infrared use exactly the same filename.
    We additionally index by stem as a conservative fallback.
    """

    image_files = sorted(
        path
        for path in infrared_dir.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )

    if not image_files:
        raise RuntimeError(
            "Infrared 目录中没有找到图片：\n"
            f"{infrared_dir}"
        )

    by_name = {}
    by_stem = {}

    duplicate_names = set()
    duplicate_stems = set()

    for path in image_files:

        if path.name in by_name:
            duplicate_names.add(
                path.name
            )
        else:
            by_name[path.name] = path

        if path.stem in by_stem:
            duplicate_stems.add(
                path.stem
            )
        else:
            by_stem[path.stem] = path

    # Ambiguous entries are removed so they can never be
    # selected silently.
    for name in duplicate_names:
        by_name.pop(
            name,
            None,
        )

    for stem in duplicate_stems:
        by_stem.pop(
            stem,
            None,
        )

    return (
        image_files,
        by_name,
        by_stem,
        duplicate_names,
        duplicate_stems,
    )


def find_ir_image(
    rgb_view_image,
    by_name,
    by_stem,
):
    """
    Locate corresponding Infrared image.

    Priority:
        1. exact filename
        2. exact stem
    """

    source = by_name.get(
        rgb_view_image.name
    )

    if source is not None:
        return source

    source = by_stem.get(
        rgb_view_image.stem
    )

    if source is not None:
        return source

    return None


def ensure_inside_project(path):
    """
    Ensure symlink targets stay inside the project.

    This preserves whole-project portability.
    """

    real_path = path.resolve()
    real_root = PROJECT_ROOT.resolve()

    try:
        real_path.relative_to(
            real_root
        )
    except ValueError as exc:
        raise RuntimeError(
            "检测到项目目录之外的软链接目标，"
            "这会破坏工程迁移能力：\n"
            f"  target: {real_path}\n"
            f"  root  : {real_root}"
        ) from exc

    return real_path


def create_relative_symlink(
    source,
    destination,
):
    """Create a project-portable relative symlink."""

    source = ensure_inside_project(
        source
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    relative_target = os.path.relpath(
        source,
        start=destination.parent,
    )

    destination.symlink_to(
        relative_target
    )


def write_data_yaml():
    """Write portable YOLO dataset configuration."""

    data_yaml = """train: images/train
val: images/val

names:
  0: person
  1: boat
  2: animal
  3: seat
  4: sign
  5: bicycle
  6: car
  7: ball
  8: light
  9: garbage_can
  10: uav
  11: tricycle
"""

    path = (
        IR_VIEW_ROOT
        / "data.yaml"
    )

    path.write_text(
        data_yaml,
        encoding="utf-8",
    )


# ============================================================
# Preflight
# ============================================================

def preflight():

    print("=" * 80)
    print("AIC2026 - Build IR YOLO View")
    print("=" * 80)

    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            "Dataset root 不存在：\n"
            f"{DATASET_ROOT}"
        )

    if not RGB_VIEW_ROOT.is_dir():
        raise FileNotFoundError(
            "rgb_v1 不存在：\n"
            f"{RGB_VIEW_ROOT}"
        )

    if IR_VIEW_ROOT.exists():
        raise FileExistsError(
            "目标目录已经存在，为避免新旧数据混合，"
            "脚本停止：\n"
            f"{IR_VIEW_ROOT}\n\n"
            "确认需要重新构建时先执行：\n"
            "rm -rf yolo_views/ir_v1"
        )

    infrared_dir = (
        find_infrared_dir()
    )

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Dataset root : {DATASET_ROOT}"
    )

    print(
        f"Infrared     : {infrared_dir}"
    )

    print(
        f"RGB view     : {RGB_VIEW_ROOT}"
    )

    print(
        f"IR view      : {IR_VIEW_ROOT}"
    )

    print()

    (
        ir_files,
        by_name,
        by_stem,
        duplicate_names,
        duplicate_stems,
    ) = build_ir_index(
        infrared_dir
    )

    print(
        f"Infrared source images: "
        f"{len(ir_files)}"
    )

    if duplicate_names:
        print(
            "Warning: duplicate IR filenames: "
            f"{len(duplicate_names)}"
        )

    if duplicate_stems:
        print(
            "Warning: duplicate IR stems: "
            f"{len(duplicate_stems)}"
        )

    records = []

    missing_ir = []
    missing_labels = []

    split_names = {}

    for split in SPLITS:

        rgb_image_dir = (
            RGB_VIEW_ROOT
            / "images"
            / split
        )

        rgb_label_dir = (
            RGB_VIEW_ROOT
            / "labels"
            / split
        )

        rgb_images = list_images(
            rgb_image_dir
        )

        if not rgb_images:
            raise RuntimeError(
                "RGB split 中没有图片：\n"
                f"{rgb_image_dir}"
            )

        names = set()

        for rgb_image in rgb_images:

            if rgb_image.name in names:
                raise RuntimeError(
                    "RGB view 内存在重复图片名：\n"
                    f"{rgb_image.name}"
                )

            names.add(
                rgb_image.name
            )

            label = (
                rgb_label_dir
                / f"{rgb_image.stem}.txt"
            )

            if not label.exists():
                missing_labels.append(
                    (
                        split,
                        rgb_image,
                        label,
                    )
                )

                continue

            ir_image = find_ir_image(
                rgb_image,
                by_name,
                by_stem,
            )

            if ir_image is None:

                missing_ir.append(
                    (
                        split,
                        rgb_image,
                    )
                )

                continue

            records.append(
                {
                    "split": split,
                    "filename": rgb_image.name,
                    "stem": rgb_image.stem,
                    "ir_source": ir_image,
                    "label_source": label,
                }
            )

        split_names[split] = names

        print(
            f"{split:5s} RGB samples : "
            f"{len(rgb_images)}"
        )

    # --------------------------------------------------------
    # Train / val overlap check
    # --------------------------------------------------------

    overlap = (
        split_names["train"]
        & split_names["val"]
    )

    if overlap:
        raise RuntimeError(
            "train / val 存在重叠样本：\n"
            + "\n".join(
                sorted(overlap)[:20]
            )
        )

    # --------------------------------------------------------
    # Missing source checks
    # --------------------------------------------------------

    if missing_labels:

        print()
        print("=" * 80)
        print("ERROR: Missing labels")
        print("=" * 80)

        for (
            split,
            image,
            label,
        ) in missing_labels[:20]:

            print(
                f"[{split}] "
                f"{image.name}"
            )

            print(
                f"  expected: {label}"
            )

        raise RuntimeError(
            f"缺失 label 数量："
            f"{len(missing_labels)}"
        )

    if missing_ir:

        print()
        print("=" * 80)
        print("ERROR: Missing Infrared images")
        print("=" * 80)

        for (
            split,
            image,
        ) in missing_ir[:30]:

            print(
                f"[{split}] "
                f"{image.name}"
            )

        raise RuntimeError(
            f"找不到对应 Infrared 图像："
            f"{len(missing_ir)} 个。\n"
            "脚本没有创建 ir_v1。"
        )

    total_rgb = (
        len(split_names["train"])
        + len(split_names["val"])
    )

    if total_rgb != EXPECTED_TOTAL_SAMPLES:
        raise RuntimeError(
            "RGB view 的总样本数不是官方训练集的 "
            f"{EXPECTED_TOTAL_SAMPLES} 组：\n"
            f"实际：{total_rgb}\n"
            "停止构建，先检查 rgb_v1。"
        )

    if len(records) != total_rgb:
        raise RuntimeError(
            "Preflight 内部计数不一致：\n"
            f"records={len(records)}, "
            f"rgb={total_rgb}"
        )

    print()
    print("=" * 80)
    print("Preflight PASS")
    print("=" * 80)

    print(
        "train samples :",
        len(
            split_names["train"]
        ),
    )

    print(
        "val samples   :",
        len(
            split_names["val"]
        ),
    )

    print(
        "total samples :",
        total_rgb,
    )

    print(
        "missing IR    :",
        len(missing_ir),
    )

    print(
        "missing label :",
        len(missing_labels),
    )

    print(
        "train/val overlap:",
        len(overlap),
    )

    print()

    return records


# ============================================================
# Build
# ============================================================

def build(records):

    for split in SPLITS:

        (
            IR_VIEW_ROOT
            / "images"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            IR_VIEW_ROOT
            / "labels"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    counters = {
        "train": 0,
        "val": 0,
    }

    for record in records:

        split = record["split"]

        image_destination = (
            IR_VIEW_ROOT
            / "images"
            / split
            / record["ir_source"].name
        )

        label_destination = (
            IR_VIEW_ROOT
            / "labels"
            / split
            / f"{record['stem']}.txt"
        )

        create_relative_symlink(
            record["ir_source"],
            image_destination,
        )

        create_relative_symlink(
            record["label_source"],
            label_destination,
        )

        counters[split] += 1

    write_data_yaml()

    return counters


# ============================================================
# Final verification
# ============================================================

def verify(counters):

    broken_links = []

    for path in IR_VIEW_ROOT.rglob("*"):

        if path.is_symlink():

            if not path.exists():
                broken_links.append(
                    path
                )

    if broken_links:

        raise RuntimeError(
            "构建完成后发现 broken symlink：\n"
            + "\n".join(
                str(path)
                for path in broken_links[:20]
            )
        )

    print("=" * 80)
    print("IR YOLO View built successfully")
    print("=" * 80)

    print(
        "Output:",
        IR_VIEW_ROOT,
    )

    print()

    print(
        "Train images:",
        counters["train"],
    )

    print(
        "Train labels:",
        counters["train"],
    )

    print(
        "Val images  :",
        counters["val"],
    )

    print(
        "Val labels  :",
        counters["val"],
    )

    print(
        "Broken links:",
        len(broken_links),
    )

    print()

    print(
        "Data YAML:",
        IR_VIEW_ROOT / "data.yaml",
    )

    print("=" * 80)


def main():

    records = preflight()

    counters = build(
        records
    )

    verify(
        counters
    )


if __name__ == "__main__":
    main()

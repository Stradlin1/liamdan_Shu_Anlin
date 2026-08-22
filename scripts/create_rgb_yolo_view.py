#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import os
import shutil

from project_paths import (
    VISIBLE_DIR,
    LABEL_DIR,
    TRAIN_SPLIT,
    VAL_SPLIT,
    RGB_YOLO_VIEW,
)


IMAGE_EXTS = [
    ".png",
    ".jpg",
    ".jpeg",
]


CLASS_NAMES = [
    "person",
    "boat",
    "animal",
    "seat",
    "sign",
    "bicycle",
    "car",
    "ball",
    "light",
    "garbage_can",
    "uav",
    "tricycle",
]


def find_image(stem):

    found = []

    for ext in IMAGE_EXTS:

        path = (
            VISIBLE_DIR
            / f"{stem}{ext}"
        )

        if path.exists():
            found.append(path)

    if len(found) != 1:

        raise RuntimeError(
            f"{stem}: visible 图片数量={len(found)}, "
            f"found={found}"
        )

    return found[0]


def create_relative_symlink(src, dst):
    """
    创建相对软链接。

    不使用绝对路径，因此整个 aicomp 项目
    移动到其他机器后仍然有效。
    """

    dst.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    relative_target = os.path.relpath(
        src,
        start=dst.parent
    )

    os.symlink(
        relative_target,
        dst
    )


def read_split(path):

    return [
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def build_split(
    split_name,
    split_file,
):

    stems = read_split(
        split_file
    )

    image_dir = (
        RGB_YOLO_VIEW
        / "images"
        / split_name
    )

    label_dir = (
        RGB_YOLO_VIEW
        / "labels"
        / split_name
    )

    image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    label_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for stem in stems:

        image_src = find_image(
            stem
        )

        label_src = (
            LABEL_DIR
            / f"{stem}.txt"
        )

        if not label_src.exists():

            raise FileNotFoundError(
                label_src
            )

        image_dst = (
            image_dir
            / image_src.name
        )

        label_dst = (
            label_dir
            / label_src.name
        )

        create_relative_symlink(
            image_src,
            image_dst
        )

        create_relative_symlink(
            label_src,
            label_dst
        )

    return stems


def create_yaml():
    """
    创建可迁移的 YOLO data.yaml。

    不写绝对 path。
    train / val 均相对于 data.yaml 所在目录解析。

    因此无论项目位于：
      <PROJECT_ROOT>
      /root/aicomp
      /root/autodl-tmp/aicomp
    均不需要修改 YAML。
    """

    yaml_text = (
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "names:\n"
    )

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        yaml_text += (
            f"  {class_id}: "
            f"{class_name}\n"
        )

    yaml_path = (
        RGB_YOLO_VIEW
        / "data.yaml"
    )

    yaml_path.write_text(
        yaml_text,
        encoding="utf-8"
    )

    return yaml_path


def main():

    print("=" * 80)
    print("Create RGB YOLO View")
    print("=" * 80)

    print(
        "Output:",
        RGB_YOLO_VIEW
    )

    # --------------------------------------------------------
    # 只删除 YOLO view。
    # 官方 datasets 完全不会被修改。
    # --------------------------------------------------------

    if RGB_YOLO_VIEW.exists():

        print(
            "Removing old RGB YOLO view..."
        )

        shutil.rmtree(
            RGB_YOLO_VIEW
        )

    train_stems = build_split(
        "train",
        TRAIN_SPLIT
    )

    val_stems = build_split(
        "val",
        VAL_SPLIT
    )

    yaml_path = create_yaml()

    print()
    print("=" * 80)
    print("RGB YOLO view created")
    print("=" * 80)

    print(
        "Train:",
        len(train_stems)
    )

    print(
        "Val  :",
        len(val_stems)
    )

    print(
        "YAML :",
        yaml_path
    )

    print()
    print(
        "官方数据未修改。"
    )

    print(
        "YOLO images / labels 均为相对软链接。"
    )


if __name__ == "__main__":
    main()

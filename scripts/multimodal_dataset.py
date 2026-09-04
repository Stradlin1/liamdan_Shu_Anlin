#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 configurable early-fusion YOLO Dataset.

Supported representations:

    ("rgb",)                -> [R, G, B]
    ("rgb", "ir")          -> [R, G, B, IR]
    ("rgb", "depth")       -> [R, G, B, Depth]
    ("rgb", "ir", "depth") -> [R, G, B, IR, Depth]

The Exp04 five-channel representation remains the default for backward
compatibility. Images are HWC uint8 before Format and CHW uint8 afterwards.

The normal Ultralytics preprocessing later converts it to float and /255.

Design principles
-----------------
1. RGB view remains the canonical YOLO dataset:
       - image list
       - labels
       - train / val split

2. Selected auxiliary modalities are matched by:
       split + stem

3. RGB is loaded/resized by the original BaseDataset.load_image().
   This preserves the project's locked Ultralytics behavior.

4. Selected auxiliary modalities are resized to RGB's resized_shape.

5. Selected modalities are concatenated BEFORE:
       Mosaic
       RandomPerspective
       RandomFlip
       LetterBox

   Therefore all geometric augmentations are naturally synchronized.

6. Standard RandomHSV is replaced with RGBOnlyRandomHSV:
       RGB   -> HSV augmentation
       auxiliary channels -> unchanged

7. Standard Ultralytics Albumentations is disabled in Exp04 v1.
   Reason:
       generic Albumentations is designed primarily around 3-channel
       images and may introduce modality-inconsistent photometric or
       spatial transforms.

8. RGB is converted:
       OpenCV BGR -> RGB
   before building a 4/5-channel image.

   This is required because Ultralytics Format only auto-reverses
   channels for 3-channel images. For 4/5-channel images it keeps the
   channel order unchanged.

9. No machine-specific absolute paths are used.

10. This module does NOT modify any source files under:
        ultralytics/

    Existing trainer integration remains separate.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ultralytics.data.augment import (
    Albumentations,
    RandomHSV,
)
from ultralytics.data.dataset import YOLODataset

from multimodal_config import (
    CHANNEL_NAMES,
    DEFAULT_MODALITIES,
    DEPTH_MODALITY,
    IR_MODALITY,
    MULTIMODAL_CHANNELS,
    channel_names_for_modalities,
    channels_for_modalities,
    normalize_modalities,
)


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

YOLO_VIEWS_ROOT = (
    PROJECT_ROOT
    / "yolo_views"
)

RGB_VIEW_ROOT = (
    YOLO_VIEWS_ROOT
    / "rgb_v1"
)

IR_VIEW_ROOT = (
    YOLO_VIEWS_ROOT
    / "ir_v1"
)

DEPTH_VIEW_ROOT = (
    YOLO_VIEWS_ROOT
    / "depth8_v1"
)


# ============================================================
# Multimodal definition
# ============================================================

SUPPORTED_SPLITS = (
    "train",
    "val",
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# Simple no-op augmentation
# ============================================================

class MultimodalAlbumentationsNoOp:
    """
    Intentionally disable generic Albumentations for Exp04 v1.

    The standard Ultralytics Albumentations wrapper is primarily designed
    for conventional 3-channel images. Applying an arbitrary transform to
    a 5-channel RGB/IR/Depth tensor could:

        - modify IR / Depth photometrically;
        - apply an image-only spatial transform;
        - break cross-modal spatial alignment.

    Exp04 keeps the normal YOLO geometric augmentations and RGB HSV
    augmentation, but skips this generic Albumentations stage.
    """

    def __call__(
        self,
        labels: dict[str, Any],
    ) -> dict[str, Any]:

        return labels

    def __repr__(self) -> str:

        return (
            self.__class__.__name__
            + "()"
        )


# ============================================================
# RGB-only HSV augmentation
# ============================================================

class RGBOnlyRandomHSV(RandomHSV):
    """
    RandomHSV variant for a 4/5-channel early-fusion image.

    Input channel order begins with:

        [R, G, B, ...]

    Only channels 0:3 are modified.

    Every selected auxiliary channel remains bit-identical.
    """

    def __init__(
        self,
        *args,
        expected_channels: int = MULTIMODAL_CHANNELS,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.expected_channels = int(expected_channels)

    def apply_image(
        self,
        labels: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:

        img = labels["img"]

        if (
            img.ndim != 3
            or img.shape[2] != self.expected_channels
        ):

            raise RuntimeError(
                "RGBOnlyRandomHSV expected a "
                f"{self.expected_channels}-channel HWC image, "
                f"but received shape={img.shape}"
            )

        if not (
            self.hgain
            or self.sgain
            or self.vgain
        ):

            return labels

        if img.dtype != np.uint8:

            raise TypeError(
                "RGBOnlyRandomHSV expects uint8 input, "
                f"but received dtype={img.dtype}"
            )

        # ----------------------------------------------------
        # Same gain-generation logic as Ultralytics RandomHSV
        # ----------------------------------------------------

        r = (
            np.random.uniform(
                -1,
                1,
                3,
            )
            * [
                self.hgain,
                self.sgain,
                self.vgain,
            ]
        )

        x = np.arange(
            0,
            256,
            dtype=r.dtype,
        )

        lut_hue = (
            (
                x
                + r[0] * 180
            )
            % 180
        ).astype(
            np.uint8
        )

        lut_sat = np.clip(
            x * (
                r[1]
                + 1
            ),
            0,
            255,
        ).astype(
            np.uint8
        )

        lut_val = np.clip(
            x * (
                r[2]
                + 1
            ),
            0,
            255,
        ).astype(
            np.uint8
        )

        lut_sat[0] = 0

        # ----------------------------------------------------
        # Our first three channels are already RGB.
        #
        # Standard Ultralytics RandomHSV expects BGR.
        # Therefore use RGB <-> HSV explicitly here.
        # ----------------------------------------------------

        rgb = np.ascontiguousarray(
            img[
                ...,
                0:3,
            ]
        )

        hsv = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2HSV,
        )

        hue, sat, val = cv2.split(
            hsv
        )

        hsv_aug = cv2.merge(
            (
                cv2.LUT(
                    hue,
                    lut_hue,
                ),
                cv2.LUT(
                    sat,
                    lut_sat,
                ),
                cv2.LUT(
                    val,
                    lut_val,
                ),
            )
        )

        rgb_aug = cv2.cvtColor(
            hsv_aug,
            cv2.COLOR_HSV2RGB,
        )

        # Only overwrite RGB.
        #
        # img[..., 3] -> IR stays unchanged.
        # img[..., 4] -> Depth stays unchanged.

        img[
            ...,
            0:3,
        ] = rgb_aug

        labels["img"] = img

        return labels


# ============================================================
# Utility functions
# ============================================================

def _build_image_index(
    view_root: Path,
) -> dict[tuple[str, str], Path]:
    """
    Build:

        (split, stem) -> image_path

    for one YOLO view.
    """

    index: dict[
        tuple[str, str],
        Path,
    ] = {}

    images_root = (
        view_root
        / "images"
    )

    if not images_root.is_dir():

        raise FileNotFoundError(
            "YOLO view images directory not found:\n"
            f"  {images_root}"
        )

    for split in SUPPORTED_SPLITS:

        split_dir = (
            images_root
            / split
        )

        if not split_dir.is_dir():

            raise FileNotFoundError(
                "YOLO view split directory not found:\n"
                f"  {split_dir}"
            )

        for path in sorted(
            split_dir.iterdir(),
            key=lambda p: p.name,
        ):

            if not path.is_file():
                continue

            if (
                path.suffix.lower()
                not in IMAGE_EXTENSIONS
            ):
                continue

            key = (
                split,
                path.stem,
            )

            if key in index:

                raise RuntimeError(
                    "Duplicate multimodal image stem:\n"
                    f"  view  : {view_root}\n"
                    f"  split : {split}\n"
                    f"  stem  : {path.stem}\n"
                    f"  file1 : {index[key]}\n"
                    f"  file2 : {path}"
                )

            index[
                key
            ] = path

    return index


def _read_grayscale_uint8(
    path: Path,
    modality_name: str,
) -> np.ndarray:
    """
    Read IR / Depth8 and return:

        H x W
        uint8

    Supported input:
        H x W
        H x W x 1
        H x W x 3
        H x W x 4
    """

    image = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    if image is None:

        raise FileNotFoundError(
            f"{modality_name} image cannot be read:\n"
            f"  {path}"
        )

    if image.dtype != np.uint8:

        raise TypeError(
            f"{modality_name} must be uint8 in the "
            "current YOLO view:\n"
            f"  file  : {path}\n"
            f"  dtype : {image.dtype}"
        )

    if image.ndim == 2:

        gray = image

    elif (
        image.ndim == 3
        and image.shape[2] == 1
    ):

        gray = image[
            ...,
            0
        ]

    elif (
        image.ndim == 3
        and image.shape[2] == 3
    ):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    elif (
        image.ndim == 3
        and image.shape[2] == 4
    ):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2GRAY,
        )

    else:

        raise RuntimeError(
            f"Unsupported {modality_name} image shape:\n"
            f"  file  : {path}\n"
            f"  shape : {image.shape}"
        )

    return np.ascontiguousarray(
        gray
    )


def _resize_gray_to_hw(
    gray: np.ndarray,
    target_hw: tuple[int, int],
) -> np.ndarray:
    """
    Resize a grayscale modality to exact (height, width).

    INTER_LINEAR is intentionally used because Exp02/Exp03 ordinary
    YOLO image loading also used normal image interpolation.
    """

    target_h = int(
        target_hw[0]
    )

    target_w = int(
        target_hw[1]
    )

    if (
        gray.shape[0] == target_h
        and gray.shape[1] == target_w
    ):

        return gray

    resized = cv2.resize(
        gray,
        (
            target_w,
            target_h,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    return np.ascontiguousarray(
        resized
    )


# ============================================================
# Multimodal Dataset
# ============================================================

class MultimodalYOLODataset(YOLODataset):
    """
    Configurable early-fusion YOLO detection Dataset.

    Supported inputs are RGB, RGB+IR, RGB+Depth, and RGB+IR+Depth.

    Canonical image list and labels come from rgb_v1.

    Auxiliary modalities are located by:

        (split, image_stem)
    """

    def __init__(
        self,
        *args,
        modalities: Iterable[str] | str | None = DEFAULT_MODALITIES,
        ir_view_root: str | Path | None = None,
        depth_view_root: str | Path | None = None,
        **kwargs,
    ):

        # ----------------------------------------------------
        # Exp04 v1 intentionally keeps cache=False.
        #
        # The original RGB cache system knows only about the
        # canonical RGB files and not auxiliary modalities.
        #
        # Supporting a full 5-channel RAM/disk cache can be
        # added later, but should not be mixed into the first
        # controlled fusion experiment.
        # ----------------------------------------------------

        cache_mode = kwargs.get(
            "cache",
            False,
        )

        if cache_mode not in {
            False,
            None,
        }:

            raise ValueError(
                "MultimodalYOLODataset Exp04 v1 requires "
                "cache=False.\n"
                f"Received cache={cache_mode!r}"
            )

        self.modalities = normalize_modalities(modalities)
        self.channel_names = channel_names_for_modalities(self.modalities)
        self.num_channels = channels_for_modalities(self.modalities)

        self.ir_view_root = Path(
            ir_view_root
            if ir_view_root is not None
            else IR_VIEW_ROOT
        )

        self.depth_view_root = Path(
            depth_view_root
            if depth_view_root is not None
            else DEPTH_VIEW_ROOT
        )

        # Build standard RGB YOLO dataset first.
        #
        # This preserves:
        #   labels
        #   cache parsing
        #   split
        #   bbox Instances
        #   Ultralytics transforms

        super().__init__(
            *args,
            **kwargs,
        )

        # Build indexes only for selected auxiliary modalities. This matters
        # for controlled RGB+IR and RGB+Depth experiments: the unused view is
        # neither required nor accidentally read.
        self._ir_index = (
            _build_image_index(self.ir_view_root)
            if IR_MODALITY in self.modalities
            else None
        )

        self._depth_index = (
            _build_image_index(self.depth_view_root)
            if DEPTH_MODALITY in self.modalities
            else None
        )

        # Fail immediately if a canonical RGB sample has no corresponding
        # pair for any selected auxiliary modality.

        self._validate_multimodal_pairs()

    # ========================================================
    # Multimodal pairing
    # ========================================================

    def _rgb_key(
        self,
        index: int,
    ) -> tuple[str, str]:

        rgb_path = Path(
            self.im_files[
                index
            ]
        )

        split = (
            rgb_path
            .parent
            .name
        )

        stem = (
            rgb_path
            .stem
        )

        if (
            split
            not in SUPPORTED_SPLITS
        ):

            raise RuntimeError(
                "Cannot infer train/val split from RGB path:\n"
                f"  {rgb_path}\n\n"
                "Expected parent directory to be one of:\n"
                f"  {SUPPORTED_SPLITS}"
            )

        return (
            split,
            stem,
        )

    def get_multimodal_paths(
        self,
        index: int,
    ) -> dict[str, Path]:
        """
        Return source paths for one multimodal sample.

        Useful for smoke tests and debugging.
        """

        key = self._rgb_key(
            index
        )

        rgb_path = Path(
            self.im_files[
                index
            ]
        )

        paths = {"rgb": rgb_path}

        for modality, modality_index, label in (
            (IR_MODALITY, self._ir_index, "IR"),
            (DEPTH_MODALITY, self._depth_index, "Depth"),
        ):
            if modality not in self.modalities:
                continue
            if modality_index is None:
                raise AssertionError(f"{label} index was not initialized.")
            try:
                paths[modality] = modality_index[key]
            except KeyError as exc:
                raise FileNotFoundError(
                    f"{label} pair missing:\n"
                    f"  RGB   : {rgb_path}\n"
                    f"  split : {key[0]}\n"
                    f"  stem  : {key[1]}"
                ) from exc

        return paths

    def _validate_multimodal_pairs(
        self,
    ) -> None:

        missing_ir = []
        missing_depth = []

        for index in range(
            len(
                self.im_files
            )
        ):

            key = self._rgb_key(
                index
            )

            if (
                self._ir_index is not None
                and key not in self._ir_index
            ):

                missing_ir.append(
                    key
                )

            if (
                self._depth_index is not None
                and key not in self._depth_index
            ):

                missing_depth.append(
                    key
                )

        if (
            missing_ir
            or missing_depth
        ):

            messages = [
                "Multimodal dataset pairing failed."
            ]

            if missing_ir:

                messages.append(
                    "Missing IR samples: "
                    f"{len(missing_ir)}"
                )

                messages.extend(
                    "  IR missing: "
                    f"{split}/{stem}"
                    for split, stem
                    in missing_ir[:20]
                )

            if missing_depth:

                messages.append(
                    "Missing Depth samples: "
                    f"{len(missing_depth)}"
                )

                messages.extend(
                    "  Depth missing: "
                    f"{split}/{stem}"
                    for split, stem
                    in missing_depth[:20]
                )

            raise RuntimeError(
                "\n".join(
                    messages
                )
            )

    # ========================================================
    # Image loading
    # ========================================================

    def get_image_and_label(
        self,
        index: int,
    ) -> dict[str, Any]:
        """
        Load the selected modalities and return one sample before augmentation.

        Important:
            RGB resizing is performed by the ORIGINAL
            BaseDataset.load_image().

            Selected auxiliary modalities are resized to exactly the same
            resized_shape.

        This avoids reimplementing Ultralytics resize logic.
        """

        # Preserve the exact standard Ultralytics path for the RGB-only
        # control experiment. In particular, Format owns BGR -> RGB for a
        # normal 3-channel image.
        if self.modalities == ("rgb",):
            return super().get_image_and_label(index)

        # ----------------------------------------------------
        # Start from canonical RGB label.
        #
        # Same logic as BaseDataset.get_image_and_label().
        # ----------------------------------------------------

        label = deepcopy(
            self.labels[
                index
            ]
        )

        label.pop(
            "shape",
            None,
        )

        # ----------------------------------------------------
        # Canonical RGB loading.
        #
        # Calling super().load_image() intentionally bypasses
        # any future load_image override in this class.
        # ----------------------------------------------------

        (
            rgb_bgr,
            ori_shape,
            resized_shape,
        ) = super().load_image(
            index
        )

        if rgb_bgr is None:

            raise RuntimeError(
                "RGB image loading returned None:\n"
                f"  {self.im_files[index]}"
            )

        if (
            rgb_bgr.ndim != 3
            or rgb_bgr.shape[2] != 3
        ):

            raise RuntimeError(
                "Canonical RGB image must be 3-channel:\n"
                f"  file  : {self.im_files[index]}\n"
                f"  shape : {rgb_bgr.shape}"
            )

        if rgb_bgr.dtype != np.uint8:

            raise TypeError(
                "Canonical RGB image must be uint8:\n"
                f"  file  : {self.im_files[index]}\n"
                f"  dtype : {rgb_bgr.dtype}"
            )

        # ----------------------------------------------------
        # Resolve auxiliary modality paths.
        # ----------------------------------------------------

        paths = self.get_multimodal_paths(
            index
        )

        modality_labels = {
            IR_MODALITY: "IR",
            DEPTH_MODALITY: "Depth",
        }
        auxiliary_images = {
            modality: _read_grayscale_uint8(
                paths[modality],
                modality_labels[modality],
            )
            for modality in self.modalities[1:]
        }

        # ----------------------------------------------------
        # Raw spatial-alignment assertion.
        #
        # check_multimodal_alignment.py already confirmed this
        # globally, but this runtime assertion protects against
        # future accidental data changes.
        # ----------------------------------------------------

        expected_ori_shape = (
            int(
                ori_shape[0]
            ),
            int(
                ori_shape[1]
            ),
        )

        for modality, gray in auxiliary_images.items():
            auxiliary_shape = (
                int(gray.shape[0]),
                int(gray.shape[1]),
            )
            if auxiliary_shape != expected_ori_shape:
                label_name = modality_labels[modality]
                raise RuntimeError(
                    f"RGB / {label_name} raw spatial mismatch:\n"
                    f"  RGB    : {paths['rgb']}\n"
                    f"  {label_name:<7}: {paths[modality]}\n"
                    f"  RGB HW : {expected_ori_shape}\n"
                    f"  {label_name} HW: {auxiliary_shape}"
                )

        # ----------------------------------------------------
        # Resize auxiliary modalities to exactly the output
        # geometry produced by RGB BaseDataset.load_image().
        # ----------------------------------------------------

        target_hw = (
            int(
                resized_shape[0]
            ),
            int(
                resized_shape[1]
            ),
        )

        auxiliary_images = {
            modality: _resize_gray_to_hw(gray, target_hw)
            for modality, gray in auxiliary_images.items()
        }

        if (
            rgb_bgr.shape[:2]
            != target_hw
        ):

            raise RuntimeError(
                "Unexpected RGB resized shape:\n"
                f"  image shape   : {rgb_bgr.shape[:2]}\n"
                f"  resized_shape : {target_hw}"
            )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Original OpenCV RGB image is BGR.
        #
        # Normal 3-channel YOLO Format converts BGR -> RGB.
        # But Format does NOT reverse channels for 5-channel
        # images.
        #
        # Therefore perform BGR -> RGB HERE.
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            rgb_bgr,
            cv2.COLOR_BGR2RGB,
        )

        # ----------------------------------------------------
        # Build H x W x C in the exact order described by self.modalities.
        # ----------------------------------------------------

        channel_arrays = [rgb]
        channel_arrays.extend(
            auxiliary_images[modality][..., None]
            for modality in self.modalities[1:]
        )
        multimodal = np.concatenate(channel_arrays, axis=2)

        multimodal = np.ascontiguousarray(
            multimodal
        )

        if (
            multimodal.ndim != 3
            or multimodal.shape[2]
            != self.num_channels
        ):

            raise RuntimeError(
                "Multimodal concatenation failed:\n"
                f"  modalities={self.modalities}\n"
                f"  expected channels={self.num_channels}\n"
                f"  shape={multimodal.shape}"
            )

        if (
            multimodal.dtype
            != np.uint8
        ):

            raise TypeError(
                "Multimodal image must remain uint8:\n"
                f"  dtype={multimodal.dtype}"
            )

        # ----------------------------------------------------
        # Same metadata behavior as BaseDataset.
        # ----------------------------------------------------

        label[
            "img"
        ] = multimodal

        label[
            "ori_shape"
        ] = ori_shape

        label[
            "resized_shape"
        ] = resized_shape

        label[
            "ratio_pad"
        ] = (
            resized_shape[0]
            / ori_shape[0],

            resized_shape[1]
            / ori_shape[1],
        )

        if self.rect:

            label[
                "rect_shape"
            ] = self.batch_shapes[
                self.batch[
                    index
                ]
            ]

        # YOLODataset converts raw bbox/segment data into
        # Ultralytics Instances here.

        return self.update_labels_info(
            label
        )

    # ========================================================
    # Transform customization
    # ========================================================

    def build_transforms(
        self,
        hyp=None,
    ):
        """
        Keep the current locked Ultralytics transform pipeline,
        but make two multimodal-specific changes:

        1. RandomHSV
             ->
           RGBOnlyRandomHSV

        2. Albumentations
             ->
           no-op

        Mosaic / Perspective / Flip / LetterBox / Format remain
        the project's original implementations.
        """

        if self.modalities == ("rgb",):
            return super().build_transforms(hyp)

        if (
            self.augment
            and float(
                getattr(
                    hyp,
                    "bgr",
                    0.0,
                )
            )
            != 0.0
        ):

            raise ValueError(
                "Exp04 multimodal pipeline requires bgr=0.0.\n"
                "The first three channels are explicitly stored "
                "as RGB before Format."
            )

        transforms = (
            super()
            .build_transforms(
                hyp
            )
        )

        if not self.augment:

            # Validation:
            #
            # LetterBox + Format
            #
            # The 4/5-channel Format keeps channel order unchanged.

            return transforms

        transform_list = getattr(
            transforms,
            "transforms",
            None,
        )

        if transform_list is None:

            raise RuntimeError(
                "Current Ultralytics Compose object does not "
                "expose .transforms; cannot safely patch "
                "multimodal augmentations."
            )

        hsv_replaced = 0
        albumentations_replaced = 0

        for i, transform in enumerate(
            transform_list
        ):

            if isinstance(
                transform,
                RandomHSV,
            ):

                transform_list[
                    i
                ] = RGBOnlyRandomHSV(
                    hgain=transform.hgain,
                    sgain=transform.sgain,
                    vgain=transform.vgain,
                    expected_channels=self.num_channels,
                )

                hsv_replaced += 1

            elif isinstance(
                transform,
                Albumentations,
            ):

                transform_list[
                    i
                ] = (
                    MultimodalAlbumentationsNoOp()
                )

                albumentations_replaced += 1

        if hsv_replaced != 1:

            raise RuntimeError(
                "Expected exactly one RandomHSV transform in "
                "the current Ultralytics training pipeline, "
                f"but replaced {hsv_replaced}."
            )

        # Albumentations may be represented differently in
        # future Ultralytics revisions. In this locked version
        # one instance is expected.

        if (
            albumentations_replaced
            > 1
        ):

            raise RuntimeError(
                "Unexpected number of Albumentations transforms: "
                f"{albumentations_replaced}"
            )

        return transforms


# ============================================================
# Public aliases
# ============================================================

__all__ = [
    "MultimodalYOLODataset",
    "RGBOnlyRandomHSV",
    "MultimodalAlbumentationsNoOp",
    "PROJECT_ROOT",
    "RGB_VIEW_ROOT",
    "IR_VIEW_ROOT",
    "DEPTH_VIEW_ROOT",
    "MULTIMODAL_CHANNELS",
    "CHANNEL_NAMES",
    "DEFAULT_MODALITIES",
    "normalize_modalities",
    "channel_names_for_modalities",
    "channels_for_modalities",
]

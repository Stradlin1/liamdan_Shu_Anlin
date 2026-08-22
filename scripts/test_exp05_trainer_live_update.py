#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIC2026 Exp05
Live Trainer optimizer-update audit.

Purpose
=======

The normal Ultralytics checkpoint path serializes EMA as FP16.

Therefore:

    trained checkpoint
        vs
    fresh FP32 initialization

is NOT a valid test of whether small parameters were actually updated.

This audit instead observes parameters directly around the REAL:

    Exp05GatedDetectionTrainer.optimizer_step()

while the model is still live FP32.

It verifies:

1. Exp05 custom parameters are actually present in Trainer optimizer groups.
2. They receive finite non-zero gradients from a real multimodal batch.
3. The real Ultralytics optimizer step changes their FP32 values.

No checkpoint comparison is used as evidence of learning.
"""

from __future__ import annotations

import gc
from pathlib import Path

import torch
import torch.nn as nn
import ultralytics

from ultralytics.utils.torch_utils import unwrap_model

from multimodal_dataset import (
    CHANNEL_NAMES,
    MULTIMODAL_CHANNELS,
)

from multimodal_gated_model import (
    Exp05AsymmetricGatedDetectionModel,
)

from multimodal_gated_trainer import (
    Exp05GatedDetectionTrainer,
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
    / "data_exp04_5ch.yaml"
)

RUNS_DIR = (
    PROJECT_ROOT
    / "runs"
)

EXPERIMENT_NAME = (
    "exp05_trainer_live_update_audit_960"
)

RUN_DIR = (
    RUNS_DIR
    / EXPERIMENT_NAME
)


# ============================================================
# Audit configuration
# ============================================================

IMAGE_SIZE = 960
BATCH_SIZE = 2
EPOCHS = 1
TRAIN_FRACTION = 0.01

SEED = 2026


# ============================================================
# Parameters that MUST participate in real Trainer training
# ============================================================

TRACKED_PARAMETER_NAMES = (
    "model.0.conv.weight",
    "depth_encoder.0.conv.weight",
    "ir_encoder.0.conv.weight",
    "ir_to_p4.proj.weight",
    "depth_gate.logits",
    "ir_gate.logits",
)


# ============================================================
# Helpers
# ============================================================

def section(
    title: str,
) -> None:

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def pass_line(
    text: str,
) -> None:

    print(
        f"[PASS] {text}"
    )


def parameter_stats(
    tensor: torch.Tensor,
) -> dict[str, float]:

    tensor = (
        tensor
        .detach()
        .float()
    )

    return {
        "abs_sum": float(
            tensor
            .abs()
            .sum()
            .item()
        ),
        "abs_max": float(
            tensor
            .abs()
            .max()
            .item()
        ),
    }


# ============================================================
# Trainer subclass with live optimizer audit
# ============================================================

class Exp05LiveUpdateAuditTrainer(
    Exp05GatedDetectionTrainer
):
    """
    Observe the REAL Ultralytics optimizer step.

    We deliberately audit live FP32 parameters instead of checkpoint
    tensors so FP16 serialization cannot hide or imitate updates.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        self.live_step_count = 0

        self.audit_history = {
            name: []
            for name
            in TRACKED_PARAMETER_NAMES
        }

        self.optimizer_membership_checked = False

        super().__init__(
            *args,
            **kwargs,
        )

    def _get_tracked_parameters(
        self,
    ) -> dict[str, nn.Parameter]:

        model = unwrap_model(
            self.model
        )

        if not isinstance(
            model,
            Exp05AsymmetricGatedDetectionModel,
        ):

            raise AssertionError(
                "Live audit Trainer is not using "
                "Exp05AsymmetricGatedDetectionModel."
            )

        parameters = dict(
            model.named_parameters()
        )

        missing = [
            name
            for name
            in TRACKED_PARAMETER_NAMES
            if name not in parameters
        ]

        if missing:

            raise AssertionError(
                "Tracked parameters missing:\n"
                + "\n".join(
                    missing
                )
            )

        return {
            name: parameters[name]
            for name
            in TRACKED_PARAMETER_NAMES
        }

    def _check_optimizer_membership(
        self,
        tracked: dict[str, nn.Parameter],
    ) -> None:

        optimizer_ids = {
            id(parameter)
            for group
            in self.optimizer.param_groups
            for parameter
            in group["params"]
        }

        print()
        print(
            "Optimizer membership:"
        )

        for (
            name,
            parameter,
        ) in tracked.items():

            present = (
                id(parameter)
                in optimizer_ids
            )

            print(
                f"  {name:<35} "
                f"present={present} "
                f"requires_grad={parameter.requires_grad}"
            )

            if not present:

                raise AssertionError(
                    f"{name} is NOT present "
                    "in Trainer optimizer groups."
                )

            if not parameter.requires_grad:

                raise AssertionError(
                    f"{name} has requires_grad=False."
                )

        self.optimizer_membership_checked = True

        pass_line(
            "all tracked Exp05 parameters are in real Trainer optimizer"
        )

    def optimizer_step(
        self,
    ):
        """
        Wrap the exact optimizer step used by BaseTrainer.
        """

        tracked = (
            self._get_tracked_parameters()
        )

        if not self.optimizer_membership_checked:

            self._check_optimizer_membership(
                tracked
            )

        # ----------------------------------------------------
        # Capture live FP32 values and gradients BEFORE the
        # actual Ultralytics optimizer step.
        # ----------------------------------------------------

        before = {}

        gradient_info = {}

        for (
            name,
            parameter,
        ) in tracked.items():

            before[name] = (
                parameter
                .detach()
                .float()
                .clone()
            )

            grad = (
                parameter.grad
            )

            if grad is None:

                gradient_info[name] = {
                    "exists": False,
                    "finite": False,
                    "nonzero": 0,
                    "abs_sum": 0.0,
                    "abs_max": 0.0,
                }

                continue

            grad_float = (
                grad
                .detach()
                .float()
            )

            finite = bool(
                torch.isfinite(
                    grad_float
                ).all()
                .item()
            )

            nonzero = int(
                torch.count_nonzero(
                    grad_float
                ).item()
            )

            stats = parameter_stats(
                grad_float
            )

            gradient_info[name] = {
                "exists": True,
                "finite": finite,
                "nonzero": nonzero,
                **stats,
            }

        # ----------------------------------------------------
        # REAL Ultralytics optimizer step.
        # AMP/scaler/EMA behavior remains controlled by the
        # locked trainer implementation.
        # ----------------------------------------------------

        super().optimizer_step()

        self.live_step_count += 1

        print()
        print(
            "=" * 100
        )

        print(
            f"LIVE OPTIMIZER STEP #{self.live_step_count}"
        )

        print(
            "=" * 100
        )

        # ----------------------------------------------------
        # Compare live FP32 values AFTER the step.
        # ----------------------------------------------------

        for (
            name,
            parameter,
        ) in tracked.items():

            after = (
                parameter
                .detach()
                .float()
            )

            delta = (
                after
                - before[name]
            ).abs()

            changed_elements = int(
                torch.count_nonzero(
                    delta
                ).item()
            )

            delta_abs_sum = float(
                delta
                .sum()
                .item()
            )

            delta_abs_max = float(
                delta
                .max()
                .item()
            )

            grad_info = (
                gradient_info[
                    name
                ]
            )

            print()
            print(
                name
            )

            print(
                "  grad exists     :",
                grad_info[
                    "exists"
                ],
            )

            print(
                "  grad finite     :",
                grad_info[
                    "finite"
                ],
            )

            print(
                "  grad nonzero    :",
                grad_info[
                    "nonzero"
                ],
            )

            print(
                "  grad abs sum    :",
                grad_info[
                    "abs_sum"
                ],
            )

            print(
                "  grad abs max    :",
                grad_info[
                    "abs_max"
                ],
            )

            print(
                "  changed elements:",
                changed_elements,
                "/",
                delta.numel(),
            )

            print(
                "  delta abs sum   :",
                delta_abs_sum,
            )

            print(
                "  delta abs max   :",
                delta_abs_max,
            )

            self.audit_history[
                name
            ].append(
                {
                    "step": (
                        self.live_step_count
                    ),
                    "gradient": (
                        grad_info
                    ),
                    "changed_elements": (
                        changed_elements
                    ),
                    "delta_abs_sum": (
                        delta_abs_sum
                    ),
                    "delta_abs_max": (
                        delta_abs_max
                    ),
                }
            )

    def assert_live_update_results(
        self,
    ) -> None:

        section(
            "Live optimizer audit result"
        )

        if (
            self.live_step_count
            <= 0
        ):

            raise AssertionError(
                "No real optimizer step was observed."
            )

        if not (
            self.optimizer_membership_checked
        ):

            raise AssertionError(
                "Optimizer membership was never checked."
            )

        print(
            "Observed optimizer steps:",
            self.live_step_count,
        )

        for name in TRACKED_PARAMETER_NAMES:

            history = (
                self.audit_history[
                    name
                ]
            )

            if not history:

                raise AssertionError(
                    f"No audit history for {name}."
                )

            any_finite_nonzero_grad = any(
                record[
                    "gradient"
                ][
                    "exists"
                ]
                and record[
                    "gradient"
                ][
                    "finite"
                ]
                and record[
                    "gradient"
                ][
                    "nonzero"
                ] > 0
                for record
                in history
            )

            any_fp32_update = any(
                record[
                    "changed_elements"
                ] > 0
                and record[
                    "delta_abs_sum"
                ] > 0.0
                for record
                in history
            )

            print()
            print(
                name
            )

            print(
                "  finite nonzero gradient:",
                any_finite_nonzero_grad,
            )

            print(
                "  live FP32 update       :",
                any_fp32_update,
            )

            if not any_finite_nonzero_grad:

                raise AssertionError(
                    f"{name} never received "
                    "a finite non-zero gradient."
                )

            if not any_fp32_update:

                raise AssertionError(
                    f"{name} never changed across "
                    "a real Trainer optimizer step."
                )

            pass_line(
                f"{name} participates in real Trainer optimization"
            )


# ============================================================
# Build audit Trainer
# ============================================================

def build_trainer():

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

        "pretrained": True,

        "cls_remap": True,

        # ----------------------------------------------------
        # Controlled optimizer audit.
        #
        # This is NOT a performance experiment.
        #
        # No warmup is used here because the purpose is to
        # directly prove that every custom parameter receives
        # a real non-zero optimizer update.
        # ----------------------------------------------------

        "epochs": EPOCHS,

        "imgsz": IMAGE_SIZE,

        "batch": BATCH_SIZE,

        "fraction": TRAIN_FRACTION,

        "optimizer": "AdamW",

        "lr0": 0.001,

        "lrf": 1.0,

        "momentum": 0.9,

        "weight_decay": 0.0005,

        "warmup_epochs": 0.0,

        # Keep effective accumulation simple for this audit.
        "nbs": BATCH_SIZE,

        # ----------------------------------------------------

        "device": device,

        "workers": 0,

        "amp": True,

        "seed": SEED,

        "deterministic": True,

        "cache": False,

        "rect": False,

        "multi_scale": 0.0,

        # Disable augmentation complexity.
        "mosaic": 0.0,

        "close_mosaic": 0,

        "fliplr": 0.0,

        "flipud": 0.0,

        "mixup": 0.0,

        "cutmix": 0.0,

        "copy_paste": 0.0,

        # Validation may still occur on the final epoch under
        # Ultralytics internals; that is harmless.
        "val": False,

        "plots": False,

        "project": str(
            RUNS_DIR
        ),

        "name": EXPERIMENT_NAME,

        "exist_ok": False,

        "save": False,
    }

    return Exp05LiveUpdateAuditTrainer(
        overrides=overrides
    )


# ============================================================
# Main
# ============================================================

def main():

    section(
        "AIC2026 Exp05 - live Trainer optimizer audit"
    )

    print(
        "Project root  :",
        PROJECT_ROOT,
    )

    print(
        "Ultralytics   :",
        ultralytics.__file__,
    )

    print(
        "Pretrained    :",
        PRETRAINED_MODEL_PATH,
    )

    print(
        "Dataset       :",
        DATA_YAML,
    )

    print(
        "Channels      :",
        MULTIMODAL_CHANNELS,
    )

    print(
        "Channel order :",
        CHANNEL_NAMES,
    )

    print(
        "Image size    :",
        IMAGE_SIZE,
    )

    print(
        "Batch         :",
        BATCH_SIZE,
    )

    print(
        "Train fraction:",
        TRAIN_FRACTION,
    )

    if RUN_DIR.exists():

        raise RuntimeError(
            "Audit output directory already exists:\n"
            f"  {RUN_DIR}\n"
            "Remove/rename it before rerunning."
        )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA unavailable."
        )

    print(
        "GPU           :",
        torch.cuda.get_device_name(
            0
        ),
    )

    trainer = build_trainer()

    section(
        "START LIVE TRAINER AUDIT"
    )

    trainer.train()

    trainer.assert_live_update_results()

    if torch.cuda.is_available():

        print()
        print(
            "Peak CUDA memory:",
            f"{torch.cuda.max_memory_allocated() / 1024**3:.3f} GiB",
        )

    section(
        "FINAL RESULT"
    )

    print(
        "Optimizer membership : PASS"
    )

    print(
        "Real batch gradients  : PASS"
    )

    print(
        "RGB live FP32 update  : PASS"
    )

    print(
        "Depth live FP32 update: PASS"
    )

    print(
        "IR live FP32 update   : PASS"
    )

    print(
        "IR projector update   : PASS"
    )

    print(
        "Depth gate update      : PASS"
    )

    print(
        "IR gate update         : PASS"
    )

    print()
    print(
        "STATUS = PASS"
    )

    del trainer

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

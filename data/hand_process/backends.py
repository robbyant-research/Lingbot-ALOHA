from __future__ import annotations

import importlib
from fractions import Fraction
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DetectorBackend(Protocol):
    def predict_many(
        self, paths: list[Path], batch_size: int | None = None
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]: ...

    def to(self, device: str) -> None: ...


@runtime_checkable
class SegmenterBackend(Protocol):
    def track(
        self,
        frame_dir: str | Path,
        prompts: dict[int, tuple[np.ndarray, np.ndarray]],
        output_dir: str | Path,
    ) -> None: ...

    def to(self, device: str) -> None: ...


# The tracker interface also serves external plugins.
TrackerBackend = SegmenterBackend


@runtime_checkable
class InpainterBackend(Protocol):
    def run_chunk(
        self,
        frame_dir: str | Path,
        mask_dir: str | Path,
        output_root: str | Path,
        fps: Fraction,
        log_name: str,
    ) -> Path: ...


@runtime_checkable
class MovableInpainterBackend(InpainterBackend, Protocol):
    def to(self, device: str) -> None: ...


def _resolve_backend(
    backend: str, builtins: dict[str, type]
) -> type:
    if backend in builtins:
        return builtins[backend]
    if ":" not in backend:
        raise ValueError(
            f"unknown backend {backend!r}; expected one of "
            f"{sorted(builtins)} or package.module:Class"
        )
    module_name, class_name = backend.split(":", 1)
    if not module_name or not class_name:
        raise ValueError(
            f"invalid backend target {backend!r}; "
            "expected package.module:Class"
        )
    module = importlib.import_module(module_name)
    try:
        backend_class = getattr(module, class_name)
    except AttributeError as error:
        raise ValueError(
            f"backend class does not exist: {backend}"
        ) from error
    if not isinstance(backend_class, type):
        raise ValueError(f"backend target is not a class: {backend}")
    return backend_class


def build_detector(config: dict, *, device: str) -> DetectorBackend:
    from .care_ego import CareEgoSegmenter

    from .grounding_dino_sam2 import GroundingDinoSam2Segmenter

    backend_class = _resolve_backend(
        str(config.get("backend", "care_ego")),
        {
            "care_ego": CareEgoSegmenter,
            "grounding_dino_sam2": GroundingDinoSam2Segmenter,
        },
    )
    backend = backend_class(config, device=device)
    if not isinstance(backend, DetectorBackend):
        raise TypeError(
            f"detector backend does not implement DetectorBackend: "
            f"{backend_class}"
        )
    return backend


def build_segmenter(config: dict, *, device: str) -> SegmenterBackend:
    from .sam2_tracker import Sam2Tracker

    backend_class = _resolve_backend(
        str(config.get("backend", "sam2")),
        {"sam2": Sam2Tracker},
    )
    backend = backend_class(config, device=device)
    if not isinstance(backend, SegmenterBackend):
        raise TypeError(
            f"segmenter backend does not implement SegmenterBackend: "
            f"{backend_class}"
        )
    return backend


build_tracker = build_segmenter


def detector_config(config: dict) -> dict:
    models = config["models"]
    selected = models.get("detector", models.get("care_ego"))
    if not isinstance(selected, dict):
        raise ValueError(
            "models.detector must be configured "
            "(legacy models.care_ego is also accepted)"
        )
    return selected


def segmenter_config(config: dict) -> dict:
    models = config["models"]
    selected = models.get("segmenter", models.get("sam2"))
    if not isinstance(selected, dict):
        raise ValueError(
            "models.segmenter must be configured "
            "(legacy models.sam2 is also accepted)"
        )
    return selected


def build_inpainter(
    config: dict,
    *,
    log_root: str | Path,
    device: str,
    persistent: bool,
) -> InpainterBackend:
    from .propainter import PersistentProPainterRunner, ProPainterRunner

    backend_name = str(config.get("backend", "propainter"))
    if backend_name == "propainter":
        backend_class = (
            PersistentProPainterRunner if persistent else ProPainterRunner
        )
    else:
        builtin_targets = {
            "diffueraser": (
                "hand_process.diffueraser_runner:"
                "PersistentDiffuEraserRunner"
            ),
        }
        backend_class = _resolve_backend(
            builtin_targets.get(backend_name, backend_name), {}
        )
    if persistent:
        backend = backend_class(config, log_root, device=device)
    else:
        backend = backend_class(config, log_root)
    if not isinstance(backend, InpainterBackend):
        raise TypeError(
            f"inpainter backend does not implement InpainterBackend: "
            f"{backend_class}"
        )
    if persistent and not isinstance(backend, MovableInpainterBackend):
        raise TypeError(
            "persistent inpainter backend must implement to(device): "
            f"{backend_class}"
        )
    return backend

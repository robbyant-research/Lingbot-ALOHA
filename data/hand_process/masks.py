from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class MaskFrameStats:
    raw_area_ratio: float
    final_area_ratio: float
    num_components: int
    dilation_px: int
    feather_px: int

    def to_dict(self) -> dict:
        return asdict(self)


def _remove_small_components(
    binary: np.ndarray, min_area: int
) -> tuple[np.ndarray, int]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    output = np.zeros_like(binary, dtype=np.uint8)
    kept = 0
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            output[labels == label] = 1
            kept += 1
    return output, kept


def postprocess_probability(
    probability: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, MaskFrameStats]:
    if probability.ndim != 2:
        raise ValueError(f"probability must be HxW, got {probability.shape}")
    height, width = probability.shape
    num_pixels = height * width
    binary = probability >= 0.5
    raw_area_ratio = float(binary.mean())
    min_area = max(1, round(num_pixels * config["min_component_area_ratio"]))
    binary, num_components = _remove_small_components(binary, min_area)
    short_side = min(height, width)
    binary = ndimage.binary_fill_holes(binary).astype(np.uint8)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)
    )

    dilation = int(
        np.clip(
            round(short_side * config["dilation_ratio"]),
            config["dilation_min_px"],
            config["dilation_max_px"],
        )
    )
    feather = int(
        np.clip(
            round(short_side * config["feather_ratio"]),
            config["feather_min_px"],
            config["feather_max_px"],
        )
    )
    if binary.any():
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilation + 1, 2 * dilation + 1)
        )
        dilated = cv2.dilate(binary, kernel)
        sigma = max(0.5, feather / 2)
        alpha = cv2.GaussianBlur(
            dilated.astype(np.float32),
            (2 * feather + 1, 2 * feather + 1),
            sigmaX=sigma,
            sigmaY=sigma,
        )
        alpha = np.clip(alpha, 0.0, 1.0)
        hard = alpha >= float(config["hard_alpha_threshold"])
    else:
        hard = binary.astype(bool)
        alpha = binary.astype(np.float32)
    stats = MaskFrameStats(
        raw_area_ratio=raw_area_ratio,
        final_area_ratio=float(hard.mean()),
        num_components=num_components,
        dilation_px=dilation,
        feather_px=feather,
    )
    return hard.astype(np.uint8) * 255, alpha.astype(np.float32), stats


def temporal_iou(previous: np.ndarray, current: np.ndarray) -> float | None:
    previous_binary = previous > 0
    current_binary = current > 0
    union = np.logical_or(previous_binary, current_binary).sum()
    if union == 0:
        return None
    intersection = np.logical_and(previous_binary, current_binary).sum()
    return float(intersection / union)

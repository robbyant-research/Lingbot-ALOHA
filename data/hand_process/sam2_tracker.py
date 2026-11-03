from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np


_OUTPUT_COMPONENT_FILTER_KEYS = frozenset(
    {
        "enabled",
        "min_area_ratio",
        "max_components",
        "prefer_edge_connected",
        "edge_margin_ratio",
        "edge_score_bonus",
        "center_distance_score_weight",
    }
)


def _validate_output_component_filter(config: object) -> dict[str, object]:
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("sam2.output_component_filter must be a mapping")
    unknown = set(config) - _OUTPUT_COMPONENT_FILTER_KEYS
    if unknown:
        raise ValueError(
            "unknown sam2.output_component_filter settings: "
            f"{sorted(unknown)}"
        )

    enabled = config.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(
            "sam2.output_component_filter.enabled must be a boolean"
        )

    prefer_edge_connected = config.get("prefer_edge_connected", False)
    if not isinstance(prefer_edge_connected, bool):
        raise ValueError(
            "sam2.output_component_filter.prefer_edge_connected must be "
            "a boolean"
        )

    values: dict[str, object] = {
        "enabled": enabled,
        "prefer_edge_connected": prefer_edge_connected,
    }
    for name, default in (
        ("min_area_ratio", 0.0),
        ("edge_margin_ratio", 0.0),
        ("edge_score_bonus", 0.0),
        ("center_distance_score_weight", 0.0),
    ):
        try:
            value = float(config.get(name, default))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"sam2.output_component_filter.{name} must be a finite number"
            ) from error
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"sam2.output_component_filter.{name} must be finite and "
                "non-negative"
            )
        values[name] = value

    if values["min_area_ratio"] >= 1.0:
        raise ValueError(
            "sam2.output_component_filter.min_area_ratio must be less than 1"
        )
    if values["edge_margin_ratio"] > 0.5:
        raise ValueError(
            "sam2.output_component_filter.edge_margin_ratio must be at most 0.5"
        )

    max_components = config.get("max_components", 0)
    if isinstance(max_components, bool):
        raise ValueError(
            "sam2.output_component_filter.max_components must be an integer"
        )
    try:
        max_components = int(max_components)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "sam2.output_component_filter.max_components must be an integer"
        ) from error
    if max_components < 0:
        raise ValueError(
            "sam2.output_component_filter.max_components must be non-negative"
        )
    values["max_components"] = max_components
    return values


def filter_mask_components(
    probability: np.ndarray,
    *,
    min_area_ratio: float = 0.0,
    max_components: int = 0,
    prefer_edge_connected: bool = False,
    edge_margin_ratio: float = 0.0,
    edge_score_bonus: float = 0.0,
    center_distance_score_weight: float = 0.0,
) -> np.ndarray:
    """Remove small mask islands and optionally rank the remaining components.

    ``prefer_edge_connected`` identifies the wearer by retaining components
    that enter through any image boundary. If none are close to a boundary, it
    safely retains all candidates instead of guessing which hand is the wearer.
    Spatial score terms only influence ranking when ``max_components`` is set.
    """

    if not isinstance(probability, np.ndarray):
        raise TypeError("probability must be a numpy.ndarray")
    if probability.ndim != 2:
        raise ValueError("probability must have shape (height, width)")
    if not np.issubdtype(probability.dtype, np.floating):
        raise TypeError("probability must use a floating-point dtype")
    if not np.all(np.isfinite(probability)):
        raise ValueError("probability must contain only finite values")
    settings = _validate_output_component_filter(
        {
            "enabled": True,
            "min_area_ratio": min_area_ratio,
            "max_components": max_components,
            "prefer_edge_connected": prefer_edge_connected,
            "edge_margin_ratio": edge_margin_ratio,
            "edge_score_bonus": edge_score_bonus,
            "center_distance_score_weight": center_distance_score_weight,
        }
    )
    min_area_ratio = float(settings["min_area_ratio"])
    max_components = int(settings["max_components"])
    prefer_edge_connected = bool(settings["prefer_edge_connected"])
    edge_margin_ratio = float(settings["edge_margin_ratio"])
    edge_score_bonus = float(settings["edge_score_bonus"])
    center_distance_score_weight = float(
        settings["center_distance_score_weight"]
    )

    # This fast path is also the baseline compatibility guarantee: with the
    # feature disabled (or all filtering controls at their defaults), no copy
    # or numerical operation is performed.
    if (
        min_area_ratio == 0.0
        and max_components == 0
        and not prefer_edge_connected
    ):
        return probability

    foreground = probability >= 0.5
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return probability

    height, width = probability.shape
    image_area = height * width
    min_area = min_area_ratio * image_area
    candidates = [
        label
        for label in range(1, num_labels)
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area
    ]

    def distance_to_edge(label: int) -> int:
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        right = width - (left + component_width)
        bottom = height - (top + component_height)
        return min(left, top, right, bottom)

    edge_margin = edge_margin_ratio * min(height, width)
    if prefer_edge_connected:
        edge_candidates = [
            label
            for label in candidates
            if distance_to_edge(label) <= edge_margin
        ]
        if edge_candidates:
            candidates = edge_candidates

    if max_components and len(candidates) > max_components:
        largest_area = max(
            int(stats[label, cv2.CC_STAT_AREA]) for label in candidates
        )
        half_diagonal = math.hypot(width - 1, height - 1) / 2.0

        def component_score(label: int) -> tuple[float, int, int]:
            edge_distance = distance_to_edge(label)
            edge_proximity = (
                max(0.0, 1.0 - edge_distance / edge_margin)
                if edge_margin > 0.0
                else 0.0
            )
            center_x, center_y = centroids[label]
            center_distance = (
                math.hypot(
                    center_x - (width - 1) / 2.0,
                    center_y - (height - 1) / 2.0,
                )
                / half_diagonal
                if half_diagonal > 0.0
                else 0.0
            )
            area = int(stats[label, cv2.CC_STAT_AREA])
            score = (
                area / largest_area
                + edge_score_bonus * edge_proximity
                + center_distance_score_weight * center_distance
            )
            # Deterministic tie-breaking prefers area, then scan order.
            return score, area, -label

        candidates = sorted(
            candidates, key=component_score, reverse=True
        )[:max_components]

    selected = np.isin(labels, candidates)
    result = probability.copy()
    result[foreground & ~selected] = 0.0
    return result


class Sam2Tracker:
    def __init__(self, config: dict, device: str = "cuda:0") -> None:
        repo = str(Path(config["repo"]))
        if repo not in sys.path:
            sys.path.insert(0, repo)
        import torch
        from sam2.build_sam import build_sam2_video_predictor

        self.torch = torch
        self.device = device
        self.mask_threshold = float(config.get("mask_threshold", 0.0))
        self.min_prompt_area_ratio = float(
            config.get("min_prompt_area_ratio", 0.0)
        )
        self.max_prompt_area_ratio = float(
            config.get("max_prompt_area_ratio", 1.0)
        )
        self.max_prompt_components = int(
            config.get("max_prompt_components", 0)
        )
        self.merge_hands_for_tracking = bool(
            config.get("merge_hands_for_tracking", False)
        )
        self.output_component_filter = _validate_output_component_filter(
            config.get("output_component_filter")
        )
        if not 0.0 <= self.min_prompt_area_ratio < 1.0:
            raise ValueError(
                "sam2.min_prompt_area_ratio must be in [0, 1)"
            )
        if not self.min_prompt_area_ratio < self.max_prompt_area_ratio <= 1.0:
            raise ValueError(
                "sam2.max_prompt_area_ratio must be greater than "
                "min_prompt_area_ratio and at most 1"
            )
        if self.max_prompt_components < 0:
            raise ValueError(
                "sam2.max_prompt_components must be non-negative"
            )
        self.offload_video_to_cpu = bool(config.get("offload_video_to_cpu", True))
        self.offload_state_to_cpu = bool(config.get("offload_state_to_cpu", False))
        self.predictor = build_sam2_video_predictor(
            config["config"],
            config["checkpoint"],
            device=device,
            apply_postprocessing=True,
            vos_optimized=False,
        )

    def to(self, device: str) -> None:
        self.device = device
        self.predictor.to(device)

    @staticmethod
    def _write_probability(path: Path, probability: np.ndarray) -> None:
        encoded = np.rint(np.clip(probability, 0.0, 1.0) * 65535).astype(
            np.uint16
        )
        if not cv2.imwrite(str(path), encoded):
            raise RuntimeError(f"failed to write probability mask: {path}")

    @staticmethod
    def _conditioning_bounds(
        nonempty: dict[int, list[int]],
        active_ids: list[int],
    ) -> tuple[int, int]:
        frame_indexes = [
            frame_index
            for obj_id in active_ids
            for frame_index in nonempty[obj_id]
        ]
        return min(frame_indexes), max(frame_indexes)

    def _valid_prompt(self, probability: np.ndarray) -> bool:
        binary = probability >= 0.5
        area_ratio = float(binary.mean())
        if not (
            self.min_prompt_area_ratio
            <= area_ratio
            <= self.max_prompt_area_ratio
        ):
            return False
        if self.max_prompt_components:
            num_components = cv2.connectedComponents(
                binary.astype(np.uint8), connectivity=8
            )[0] - 1
            if num_components > self.max_prompt_components:
                return False
        return True

    def _prompt_objects(
        self, left: np.ndarray, right: np.ndarray
    ) -> dict[int, np.ndarray]:
        if self.merge_hands_for_tracking:
            return {1: np.clip(left + right, 0.0, 1.0)}
        return {1: left, 2: right}

    def track(
        self,
        frame_dir: str | Path,
        prompts: dict[int, tuple[np.ndarray, np.ndarray]],
        output_dir: str | Path,
    ) -> None:
        torch = self.torch
        frame_root = Path(frame_dir)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        frame_count = len(list(frame_root.glob("*.jpg")))
        if frame_count == 0:
            raise ValueError(f"no JPEG frames in {frame_root}")

        nonempty: dict[int, list[int]] = {1: [], 2: []}
        for frame_index, (left, right) in prompts.items():
            for obj_id, probability in self._prompt_objects(
                left, right
            ).items():
                if self._valid_prompt(probability):
                    nonempty[obj_id].append(frame_index)
        active_ids = [obj_id for obj_id, frames in nonempty.items() if frames]
        if not active_ids:
            sample = cv2.imread(str(next(frame_root.glob("*.jpg"))))
            zeros = np.zeros(sample.shape[:2], dtype=np.float32)
            for frame_index in range(frame_count):
                self._write_probability(output / f"{frame_index:06d}.png", zeros)
            return

        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        with torch.inference_mode(), autocast:
            state = self.predictor.init_state(
                str(frame_root),
                offload_video_to_cpu=self.offload_video_to_cpu,
                offload_state_to_cpu=self.offload_state_to_cpu,
                async_loading_frames=False,
            )
            for frame_index in sorted(prompts):
                left, right = prompts[frame_index]
                for obj_id, probability in self._prompt_objects(
                    left, right
                ).items():
                    if (
                        obj_id in active_ids
                        and self._valid_prompt(probability)
                    ):
                        self.predictor.add_new_mask(
                            state,
                            frame_index,
                            obj_id=obj_id,
                            mask=probability >= 0.5,
                        )

            first_conditioning, last_conditioning = self._conditioning_bounds(
                nonempty, active_ids
            )

            written: set[int] = set()

            def consume(iterator) -> None:
                for frame_index, _object_ids, mask_logits in iterator:
                    probability = (
                        (mask_logits.float() - self.mask_threshold)
                        .sigmoid()
                        .amax(dim=0)
                        .squeeze(0)
                        .cpu()
                        .numpy()
                    )
                    if self.output_component_filter["enabled"]:
                        probability = filter_mask_components(
                            probability,
                            min_area_ratio=float(
                                self.output_component_filter["min_area_ratio"]
                            ),
                            max_components=int(
                                self.output_component_filter["max_components"]
                            ),
                            prefer_edge_connected=bool(
                                self.output_component_filter[
                                    "prefer_edge_connected"
                                ]
                            ),
                            edge_margin_ratio=float(
                                self.output_component_filter[
                                    "edge_margin_ratio"
                                ]
                            ),
                            edge_score_bonus=float(
                                self.output_component_filter[
                                    "edge_score_bonus"
                                ]
                            ),
                            center_distance_score_weight=float(
                                self.output_component_filter[
                                    "center_distance_score_weight"
                                ]
                            ),
                        )
                    self._write_probability(
                        output / f"{frame_index:06d}.png", probability
                    )
                    written.add(int(frame_index))

            if last_conditioning > 0:
                consume(
                    self.predictor.propagate_in_video(
                        state,
                        start_frame_idx=last_conditioning,
                        max_frame_num_to_track=last_conditioning + 1,
                        reverse=True,
                    )
                )
            consume(
                self.predictor.propagate_in_video(
                    state,
                    start_frame_idx=first_conditioning,
                    max_frame_num_to_track=frame_count,
                    reverse=False,
                )
            )
            self.predictor.reset_state(state)

        if len(written) != frame_count:
            missing = sorted(set(range(frame_count)) - written)
            raise RuntimeError(
                f"SAM 2 did not produce all frames: missing={missing[:20]}"
            )

from __future__ import annotations

import inspect
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


DEFAULT_HAND_PROMPTS = (
    "human hand",
    "gloved hand",
    "forearm",
)


def normalize_grounding_prompt(prompts: str | Iterable[str]) -> str:
    """Build the period-separated phrase format expected by Grounding DINO."""

    if isinstance(prompts, str):
        phrases = [prompts]
    else:
        phrases = list(prompts)
    normalized = [
        str(phrase).strip().strip(".").strip()
        for phrase in phrases
        if str(phrase).strip().strip(".").strip()
    ]
    if not normalized:
        raise ValueError("grounding_dino_sam2.prompts must not be empty")
    return ". ".join(normalized) + "."


def expand_and_clip_boxes(
    boxes: np.ndarray,
    *,
    width: int,
    height: int,
    expansion_ratio: float,
    min_expansion_px: float = 0.0,
) -> np.ndarray:
    """Expand XYXY boxes around their centers and clip them to the image."""

    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (num_boxes, 4)")
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    if not math.isfinite(expansion_ratio) or expansion_ratio < 0:
        raise ValueError("expansion_ratio must be finite and non-negative")
    if not math.isfinite(min_expansion_px) or min_expansion_px < 0:
        raise ValueError("min_expansion_px must be finite and non-negative")
    if boxes.size == 0:
        return boxes.copy()

    expanded = boxes.copy()
    box_widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    box_heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    pad_x = np.maximum(box_widths * expansion_ratio, min_expansion_px)
    pad_y = np.maximum(box_heights * expansion_ratio, min_expansion_px)
    expanded[:, 0] -= pad_x
    expanded[:, 1] -= pad_y
    expanded[:, 2] += pad_x
    expanded[:, 3] += pad_y
    expanded[:, [0, 2]] = np.clip(expanded[:, [0, 2]], 0.0, width - 1.0)
    expanded[:, [1, 3]] = np.clip(
        expanded[:, [1, 3]], 0.0, height - 1.0
    )
    return expanded


def class_agnostic_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    *,
    max_detections: int | None = None,
) -> np.ndarray:
    """Return score-ordered indexes after class-agnostic XYXY NMS."""

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (num_boxes, 4)")
    if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
        raise ValueError("scores must have shape (num_boxes,)")
    if not math.isfinite(iou_threshold) or not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    if max_detections is not None and max_detections < 1:
        raise ValueError("max_detections must be positive when provided")

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(-scores, kind="stable")
    kept: list[int] = []
    while order.size:
        current = int(order[0])
        kept.append(current)
        if max_detections is not None and len(kept) >= max_detections:
            break
        remaining = order[1:]
        if not remaining.size:
            break
        intersection_width = np.maximum(
            0.0,
            np.minimum(x2[current], x2[remaining])
            - np.maximum(x1[current], x1[remaining]),
        )
        intersection_height = np.maximum(
            0.0,
            np.minimum(y2[current], y2[remaining])
            - np.maximum(y1[current], y1[remaining]),
        )
        intersection = intersection_width * intersection_height
        union = areas[current] + areas[remaining] - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        order = remaining[iou <= iou_threshold]
    return np.asarray(kept, dtype=np.int64)


def union_sam_masks(
    masks: Iterable[np.ndarray],
    *,
    height: int,
    width: int,
    max_area_ratio: float = 1.0,
) -> np.ndarray:
    """Merge plausible SAM masks into the detector probability map."""

    if not math.isfinite(max_area_ratio) or not 0.0 <= max_area_ratio <= 1.0:
        raise ValueError("max_area_ratio must be in [0, 1]")
    union = np.zeros((height, width), dtype=bool)
    for mask in masks:
        candidate = np.asarray(mask)
        candidate = np.squeeze(candidate)
        if candidate.shape != union.shape:
            raise ValueError(
                f"SAM mask shape {candidate.shape} does not match "
                f"image shape {union.shape}"
            )
        candidate = candidate > 0
        if float(candidate.mean()) <= max_area_ratio:
            union |= candidate
    return union.astype(np.float32)


def _validate_unit_interval(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


class GroundingDinoSam2Segmenter:
    """No-MMCV hand detector refined by the project-pinned SAM 2 image model."""

    def __init__(self, config: dict, device: str = "cuda:0") -> None:
        if config.get("acknowledge_non_ego_hand_removal") is not True:
            raise ValueError(
                "grounding_dino_sam2 detects every matching hand; set "
                "acknowledge_non_ego_hand_removal: true only for scenes "
                "where removing non-ego hands is acceptable"
            )
        os.environ.setdefault("USE_TF", "0")
        import torch
        from transformers import (
            GroundingDinoForObjectDetection,
            GroundingDinoProcessor,
        )

        model_dir = Path(config["model_dir"])
        if not model_dir.is_dir():
            raise ValueError(
                "grounding_dino_sam2.model_dir is not a directory: "
                f"{model_dir}"
            )
        self.torch = torch
        self.device = torch.device(device)
        self.prompt = normalize_grounding_prompt(
            config.get("prompts", DEFAULT_HAND_PROMPTS)
        )
        self.box_threshold = _validate_unit_interval(
            "grounding_dino_sam2.box_threshold",
            config.get("box_threshold", 0.25),
        )
        self.text_threshold = _validate_unit_interval(
            "grounding_dino_sam2.text_threshold",
            config.get("text_threshold", 0.20),
        )
        self.nms_iou_threshold = _validate_unit_interval(
            "grounding_dino_sam2.nms_iou_threshold",
            config.get("nms_iou_threshold", 0.5),
        )
        self.sam_score_threshold = _validate_unit_interval(
            "grounding_dino_sam2.sam_score_threshold",
            config.get("sam_score_threshold", 0.0),
        )
        self.max_sam_mask_area_ratio = _validate_unit_interval(
            "grounding_dino_sam2.max_sam_mask_area_ratio",
            config.get("max_sam_mask_area_ratio", 1.0),
        )
        self.box_expansion_ratio = float(
            config.get("box_expansion_ratio", 0.12)
        )
        self.min_box_expansion_px = float(
            config.get("min_box_expansion_px", 4)
        )
        self.batch_size = int(config.get("batch_size", 4))
        self.max_detections = int(config.get("max_detections", 8))
        if self.batch_size < 1:
            raise ValueError(
                "grounding_dino_sam2.batch_size must be at least 1"
            )
        if self.max_detections < 1:
            raise ValueError(
                "grounding_dino_sam2.max_detections must be at least 1"
            )
        # Validate expansion settings before model loading so bad profiles fail
        # quickly and with the same rules as the pure helper.
        expand_and_clip_boxes(
            np.empty((0, 4), dtype=np.float32),
            width=1,
            height=1,
            expansion_ratio=self.box_expansion_ratio,
            min_expansion_px=self.min_box_expansion_px,
        )

        self.processor = GroundingDinoProcessor.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        self.detector = (
            GroundingDinoForObjectDetection.from_pretrained(
                model_dir,
                local_files_only=True,
                use_safetensors=True,
            )
            .to(self.device)
            .eval()
        )

        sam2_repo = str(Path(config["sam2_repo"]))
        if sam2_repo not in sys.path:
            sys.path.insert(0, sam2_repo)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam_model = build_sam2(
            config["sam2_config"],
            config["sam2_checkpoint"],
            device=device,
            apply_postprocessing=True,
        )
        self.sam_predictor = SAM2ImagePredictor(sam_model)

    def to(self, device: str) -> None:
        self.device = self.torch.device(device)
        self.detector.to(self.device)
        self.sam_predictor.model.to(self.device)
        self.sam_predictor.reset_predictor()

    def _post_process(
        self,
        outputs,
        input_ids,
        target_sizes: list[tuple[int, int]],
    ) -> list[dict[str, Any]]:
        method = self.processor.post_process_grounded_object_detection
        parameters = inspect.signature(method).parameters
        threshold_name = (
            "threshold" if "threshold" in parameters else "box_threshold"
        )
        return method(
            outputs,
            input_ids=input_ids,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes,
            **{threshold_name: self.box_threshold},
        )

    def _refine(
        self, image: np.ndarray, boxes: np.ndarray
    ) -> np.ndarray:
        height, width = image.shape[:2]
        if not len(boxes):
            return np.zeros((height, width), dtype=np.float32)
        self.sam_predictor.set_image(image)
        accepted = []
        for box in boxes:
            masks, scores, _low_resolution = self.sam_predictor.predict(
                box=box,
                multimask_output=False,
                return_logits=False,
            )
            score = float(np.asarray(scores).reshape(-1)[0])
            if score >= self.sam_score_threshold:
                accepted.append(np.asarray(masks)[0])
        self.sam_predictor.reset_predictor()
        return union_sam_masks(
            accepted,
            height=height,
            width=width,
            max_area_ratio=self.max_sam_mask_area_ratio,
        )

    def predict_many(
        self, paths: list[Path], batch_size: int | None = None
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        selected_batch_size = (
            self.batch_size if batch_size is None else int(batch_size)
        )
        if selected_batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        torch = self.torch
        for start in range(0, len(paths), selected_batch_size):
            batch_paths = paths[start : start + selected_batch_size]
            images = [
                np.asarray(Image.open(path).convert("RGB"))
                for path in batch_paths
            ]
            encoded = self.processor(
                images=images,
                text=[self.prompt] * len(images),
                return_tensors="pt",
                padding=True,
            )
            encoded = {
                name: value.to(self.device)
                if hasattr(value, "to")
                else value
                for name, value in encoded.items()
            }
            autocast_enabled = self.device.type == "cuda"
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                model_outputs = self.detector(**encoded)
            detections = self._post_process(
                model_outputs,
                encoded["input_ids"],
                [image.shape[:2] for image in images],
            )

            for path, image, detection in zip(
                batch_paths, images, detections
            ):
                boxes = detection["boxes"].detach().float().cpu().numpy()
                scores = detection["scores"].detach().float().cpu().numpy()
                keep = class_agnostic_nms(
                    boxes,
                    scores,
                    self.nms_iou_threshold,
                    max_detections=self.max_detections,
                )
                boxes = expand_and_clip_boxes(
                    boxes[keep],
                    width=image.shape[1],
                    height=image.shape[0],
                    expansion_ratio=self.box_expansion_ratio,
                    min_expansion_px=self.min_box_expansion_px,
                )
                union = self._refine(image, boxes)
                results[int(path.stem)] = (
                    union,
                    np.zeros_like(union),
                )
        return results

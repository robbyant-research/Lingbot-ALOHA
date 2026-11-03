from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from .masks import temporal_iou
from .video import VideoInfo, frame_paths, probe_video


def _encoded_outside_metrics(
    original_paths: list[Path],
    mask_paths: list[Path],
    video_path: Path,
    max_samples: int = 50,
) -> tuple[float, float]:
    import av

    sample_indexes = set(
        np.linspace(
            0,
            max(0, len(original_paths) - 1),
            num=min(max_samples, len(original_paths)),
            dtype=int,
        ).tolist()
    )
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index not in sample_indexes:
                continue
            encoded = frame.to_ndarray(format="rgb24")
            original = np.asarray(
                Image.open(original_paths[frame_index]).convert("RGB")
            )
            mask = (
                np.asarray(Image.open(mask_paths[frame_index]).convert("L")) > 0
            )
            outside = ~mask
            if outside.any():
                difference = (
                    encoded[outside].astype(np.float32)
                    - original[outside].astype(np.float32)
                )
                mse = float(np.mean(difference * difference))
                psnr_values.append(
                    99.0 if mse == 0 else 10 * math.log10(255.0**2 / mse)
                )
            comparison = encoded.copy()
            comparison[mask] = original[mask]
            ssim_values.append(
                float(
                    structural_similarity(
                        original,
                        comparison,
                        channel_axis=2,
                        data_range=255,
                    )
                )
            )
    return (
        min(psnr_values) if psnr_values else 99.0,
        min(ssim_values) if ssim_values else 1.0,
    )


def evaluate_episode(
    *,
    source_info: VideoInfo,
    expected_output_info: VideoInfo,
    original_frame_dir: Path,
    mask_dir: Path,
    final_frame_dir: Path,
    inpainted_video: Path,
    mask_frame_stats: list[dict],
    config: dict,
) -> dict:
    original_paths = frame_paths(original_frame_dir)
    mask_paths = frame_paths(mask_dir)
    final_paths = frame_paths(final_frame_dir)
    output_info = probe_video(inpainted_video)
    structural_errors: list[str] = []
    expected_count = len(original_paths)
    if len(mask_paths) != expected_count or len(final_paths) != expected_count:
        structural_errors.append("intermediate frame count mismatch")
    if output_info.num_frames != expected_count:
        structural_errors.append(
            f"encoded frame count {output_info.num_frames} != {expected_count}"
        )
    if (output_info.width, output_info.height) != (
        expected_output_info.width,
        expected_output_info.height,
    ):
        structural_errors.append("encoded resolution mismatch")
    if output_info.fps != expected_output_info.fps:
        structural_errors.append(
            f"encoded fps {output_info.fps} != expected "
            f"{expected_output_info.fps}"
        )

    areas = [
        float(np.mean(np.asarray(Image.open(path).convert("L")) > 0))
        for path in mask_paths
    ]
    temporal_ious: list[float] = []
    previous = None
    for path in mask_paths:
        current = np.asarray(Image.open(path).convert("L"))
        if previous is not None:
            value = temporal_iou(previous, current)
            if value is not None:
                temporal_ious.append(value)
        previous = current
    area_jumps = [
        abs(current - previous) for previous, current in zip(areas, areas[1:])
    ]
    temporal_iou_percentile = float(
        config["qc"].get("temporal_iou_percentile", 0.0)
    )
    temporal_iou_guard = (
        float(np.percentile(temporal_ious, temporal_iou_percentile))
        if temporal_ious
        else None
    )
    encoded_psnr, encoded_ssim = _encoded_outside_metrics(
        original_paths, mask_paths, inpainted_video
    )

    review_reasons: list[str] = []
    if areas and max(areas) > config["mask"]["max_area_ratio"]:
        review_reasons.append("mask_area_above_limit")
    if area_jumps and max(area_jumps) > config["qc"]["max_area_jump_ratio"]:
        review_reasons.append("mask_area_jump")
    if (
        temporal_iou_guard is not None
        and temporal_iou_guard < config["qc"]["min_temporal_iou"]
    ):
        review_reasons.append("low_temporal_mask_iou")
    if encoded_psnr < config["qc"]["min_encoded_outside_psnr"]:
        review_reasons.append("low_outside_psnr")
    if encoded_ssim < config["qc"]["min_encoded_outside_ssim"]:
        review_reasons.append("low_outside_ssim")

    status = (
        "failed"
        if structural_errors
        else "needs_review"
        if review_reasons
        else "pass"
    )
    return {
        "status": status,
        "structural_errors": structural_errors,
        "review_reasons": review_reasons,
        "source": source_info.to_dict(),
        "output": output_info.to_dict(),
        "mask": {
            "area_min": min(areas) if areas else 0.0,
            "area_median": float(np.median(areas)) if areas else 0.0,
            "area_max": max(areas) if areas else 0.0,
            "empty_frames": sum(area == 0 for area in areas),
            "max_area_jump": max(area_jumps) if area_jumps else 0.0,
            "temporal_iou_min": min(temporal_ious)
            if temporal_ious
            else None,
            "temporal_iou_median": float(np.median(temporal_ious))
            if temporal_ious
            else None,
            "temporal_iou_p01": float(np.percentile(temporal_ious, 1))
            if temporal_ious
            else None,
            "temporal_iou_p05": float(np.percentile(temporal_ious, 5))
            if temporal_ious
            else None,
            "temporal_iou_p10": float(np.percentile(temporal_ious, 10))
            if temporal_ious
            else None,
            "temporal_iou_percentile": temporal_iou_percentile,
            "temporal_iou_guard": temporal_iou_guard,
            "frames": mask_frame_stats,
        },
        "inpainted": {
            "encoded_outside_psnr_min": encoded_psnr,
            "encoded_outside_ssim_min": encoded_ssim,
        },
    }


def write_json(path: str | Path, value: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(output)


def build_manifest(run_root: str | Path) -> list[dict]:
    root = Path(run_root)
    records: list[dict] = []
    for qc_path in sorted((root / "qc").glob("**/*.json")):
        with qc_path.open("r", encoding="utf-8") as handle:
            records.append(json.load(handle))
    manifest_path = root / "manifest.jsonl"
    temporary = manifest_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    summary = {
        "total": len(records),
        "pass": sum(record["status"] == "pass" for record in records),
        "needs_review": sum(
            record["status"] == "needs_review" for record in records
        ),
        "failed": sum(record["status"] == "failed" for record in records),
    }
    write_json(root / "summary.json", summary)
    return records

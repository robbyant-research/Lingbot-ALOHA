from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from queue import Empty
from typing import Callable, Iterable

import cv2
import numpy as np
from PIL import Image

from .backends import InpainterBackend, detector_config, segmenter_config
from .datasets import Episode
from .frame_store import FrameValue, frame_value_rgb, load_frame_store
from .fs import (
    atomic_copy_file,
    blend_weight,
    chunk_ranges,
    hardlink_or_copy,
    recreate_directory,
    relative_symlink,
    remove_directory,
)
from .masks import postprocess_probability
from .qc import build_manifest, evaluate_episode, write_json
from .video import (
    decode_video_to_pngs,
    encode_mask_pngs,
    encode_rgb_frames,
    frame_paths,
    iter_png_rgb,
    make_preview_frames,
    probe_video,
    scaled_dimensions,
    video_is_all_keyframes,
)


logger = logging.getLogger(__name__)
PIPELINE_OUTPUT_VERSION = 2


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _global_progress_line(
    completed: int,
    total: int,
    failed: int,
    elapsed_seconds: float,
    *,
    latest: str | None = None,
    rate_completed: int | None = None,
    width: int = 28,
) -> str:
    if total < 0:
        raise ValueError("progress total must be non-negative")
    if completed < 0 or completed > total:
        raise ValueError("progress completed must be in [0, total]")
    if failed < 0 or failed > completed:
        raise ValueError("progress failed must be in [0, completed]")
    if rate_completed is not None and (
        rate_completed < 0 or rate_completed > completed
    ):
        raise ValueError("progress rate_completed must be in [0, completed]")
    if width < 1:
        raise ValueError("progress width must be at least 1")
    ratio = completed / total if total else 1.0
    filled = min(width, int(ratio * width))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = max(0.0, elapsed_seconds)
    rate_count = completed if rate_completed is None else rate_completed
    if rate_count and completed < total and elapsed > 0:
        eta = elapsed * (total - completed) / rate_count
        eta_text = _format_duration(eta)
    elif completed >= total:
        eta_text = "00:00"
    else:
        eta_text = "--:--"
    line = (
        f"[overall] [{bar}] {completed}/{total} "
        f"({ratio * 100:5.1f}%) failed={failed} "
        f"elapsed={_format_duration(elapsed)} eta={eta_text}"
    )
    if latest:
        line += f" | {latest}"
    return line


def _episode_paths(config: dict, run_id: str, episode: Episode) -> dict[str, Path]:
    run_root = Path(config["run_root"]) / run_id
    relative = episode.relative_dir
    work_relative = (
        episode.source_relative_path.with_suffix("")
        if episode.source_relative_path is not None
        else Path(episode.stem)
    )
    dataset_config = config.get("datasets", {}).get(episode.dataset, {})
    final_video_root = Path(
        dataset_config.get(
            "final_video_root",
            config.get("final_video_root", run_root / "inpainted"),
        )
    )
    output_relative = relative
    if "final_video_root" in dataset_config:
        source_parent = (
            episode.source_relative_path.parent
            if episode.source_relative_path is not None
            else Path(".")
        )
        output_relative = (
            Path(dataset_config.get("output_relative_path", "."))
            / source_parent
        )
    return {
        "run": run_root,
        "original": run_root / "original" / relative / episode.source.name,
        "mask_dir": run_root / "hand_mask" / relative / episode.stem,
        "mask_video": run_root / "hand_mask" / relative / episode.source.name,
        "inpainted": final_video_root / output_relative / episode.source.name,
        "qc": run_root / "qc" / relative / f"{episode.stem}.json",
        "preview": run_root / "previews" / relative / episode.source.name,
        "work": (
            Path(config["work_root"])
            / run_id
            / episode.dataset
            / work_relative
        ),
    }


def _processing_options(config: dict) -> tuple[int, bool]:
    processing_config = config.get("processing", {})
    target_height_value = processing_config.get("target_height", 480)
    if isinstance(target_height_value, bool) or not isinstance(
        target_height_value, int
    ):
        raise ValueError("processing.target_height must be an integer")
    if target_height_value < 2:
        raise ValueError("processing.target_height must be at least 2")
    paste_back_value = processing_config.get("paste_back_to_source", False)
    if not isinstance(paste_back_value, bool):
        raise ValueError("processing.paste_back_to_source must be true or false")
    return target_height_value, paste_back_value


def _output_contract(config: dict) -> dict:
    from .model_service import inpainter_backend_name, inpainter_config

    target_height, paste_back = _processing_options(config)
    selected = inpainter_config(config)
    output = config["output"]
    return {
        "pipeline_version": PIPELINE_OUTPUT_VERSION,
        "processing": {
            "target_height": target_height,
            "paste_back_to_source": paste_back,
        },
        "inpainting": {
            "backend": inpainter_backend_name(config),
            "output_storage": str(selected.get("output_storage", "png")),
            "implementation_version": "generated-frame-store-v1",
        },
        "encoding": {
            "video_codec": str(output["video_codec"]),
            "crf": int(output["crf"]),
            "preset": str(output["preset"]),
            "pixel_format": "yuv420p",
            "all_keyframes": True,
        },
    }


def _validate_work_root(config: dict) -> Path:
    work_root = Path(config["work_root"]).expanduser().resolve()
    require_tmpfs = config.get("runtime", {}).get("require_tmpfs", False)
    if not isinstance(require_tmpfs, bool):
        raise ValueError("runtime.require_tmpfs must be true or false")
    if require_tmpfs:
        from .fs import ensure_within

        return ensure_within(work_root, "/dev/shm")
    return work_root


def _keyframe_indexes(num_frames: int, ranges: list[tuple[int, int]], interval: int):
    indexes = {0, num_frames - 1}
    indexes.update(range(0, num_frames, interval))
    for start, end in ranges:
        indexes.add(start)
        indexes.add(end - 1)
    return sorted(index for index in indexes if 0 <= index < num_frames)


def _make_sam_jpegs(
    originals: list[Path], start: int, end: int, destination: Path
) -> None:
    recreate_directory(destination, destination.parent)
    for local_index, global_index in enumerate(range(start, end)):
        with Image.open(originals[global_index]) as image:
            image.convert("RGB").save(
                destination / f"{local_index:06d}.jpg",
                format="JPEG",
                quality=95,
                subsampling=0,
            )


def _run_sam_chunks(
    *,
    originals: list[Path],
    detector,
    segmenter,
    config: dict,
    work: Path,
    activate_model: Callable[[str], None] | None = None,
) -> Path:
    sam_config = segmenter_config(config)
    ranges = chunk_ranges(
        len(originals),
        int(sam_config["chunk_frames"]),
        int(sam_config["chunk_overlap"]),
    )
    keyframes = _keyframe_indexes(
        len(originals), ranges, int(sam_config["keyframe_interval"])
    )
    if activate_model is not None:
        activate_model("detector")
    prompts = detector.predict_many([originals[index] for index in keyframes])
    if activate_model is not None:
        activate_model("segmenter")
    probability_dir = work / "sam_probability"
    probability_dir.mkdir(parents=True, exist_ok=True)
    accumulated_weights = np.zeros(len(originals), dtype=np.float32)

    for chunk_index, (start, end) in enumerate(ranges):
        chunk_root = work / "sam_chunks" / f"{chunk_index:04d}"
        jpeg_dir = chunk_root / "frames"
        raw_dir = chunk_root / "raw"
        _make_sam_jpegs(originals, start, end, jpeg_dir)
        local_prompts = {
            global_index - start: prompts[global_index]
            for global_index in keyframes
            if start <= global_index < end
        }
        segmenter.track(jpeg_dir, local_prompts, raw_dir)
        raw_paths = frame_paths(raw_dir)
        if len(raw_paths) != end - start:
            raise RuntimeError(
                f"SAM 2 chunk {chunk_index} returned {len(raw_paths)} "
                f"frames, expected {end - start}"
            )
        for local_index, raw_path in enumerate(raw_paths):
            global_index = start + local_index
            current = cv2.imread(str(raw_path), cv2.IMREAD_UNCHANGED)
            if current is None:
                raise RuntimeError(f"cannot read SAM 2 probability: {raw_path}")
            current = current.astype(np.float32) / 65535.0
            weight = blend_weight(
                local_index, end - start, int(sam_config["chunk_overlap"])
            )
            destination = probability_dir / f"{global_index:06d}.png"
            previous_weight = float(accumulated_weights[global_index])
            if previous_weight > 0:
                previous = cv2.imread(str(destination), cv2.IMREAD_UNCHANGED)
                if previous is None:
                    raise RuntimeError(f"cannot read merged probability: {destination}")
                previous = previous.astype(np.float32) / 65535.0
                current = (previous * previous_weight + current * weight) / (
                    previous_weight + weight
                )
            encoded = np.rint(np.clip(current, 0.0, 1.0) * 65535).astype(np.uint16)
            if not cv2.imwrite(str(destination), encoded):
                raise RuntimeError(f"cannot write merged probability: {destination}")
            accumulated_weights[global_index] += weight
        remove_directory(chunk_root, work)

    if np.any(accumulated_weights == 0):
        missing = np.flatnonzero(accumulated_weights == 0).tolist()
        raise RuntimeError(f"SAM 2 merge missed frames: {missing[:20]}")
    return probability_dir


def _finalize_masks(
    probability_dir: Path,
    mask_dir: Path,
    alpha_dir: Path,
    config: dict,
) -> list[dict]:
    mask_dir.mkdir(parents=True, exist_ok=True)
    alpha_dir.mkdir(parents=True, exist_ok=True)
    probability_paths = frame_paths(probability_dir)
    num_workers = int(
        config.get("runtime", {}).get(
            "mask_workers",
            config.get("runtime", {}).get("composite_workers", 4),
        )
    )
    if num_workers < 1:
        raise ValueError("runtime.mask_workers must be at least 1")

    def finalize(probability_path: Path) -> dict:
        encoded = cv2.imread(str(probability_path), cv2.IMREAD_UNCHANGED)
        if encoded is None:
            raise RuntimeError(f"cannot read probability mask: {probability_path}")
        probability = encoded.astype(np.float32) / 65535.0
        hard, alpha, frame_stats = postprocess_probability(probability, config["mask"])
        if not cv2.imwrite(str(mask_dir / probability_path.name), hard):
            raise RuntimeError(f"cannot write hand mask: {probability_path.name}")
        alpha_encoded = np.rint(alpha * 65535).astype(np.uint16)
        if not cv2.imwrite(str(alpha_dir / probability_path.name), alpha_encoded):
            raise RuntimeError(f"cannot write alpha mask: {probability_path.name}")
        frame_record = frame_stats.to_dict()
        frame_record["frame_index"] = int(probability_path.stem)
        return frame_record

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(finalize, probability_paths))


def _write_composited_frame(
    original_path: Path,
    generated_frame: FrameValue,
    alpha_path: Path,
    destination: Path,
) -> None:
    with Image.open(original_path) as image:
        original = np.asarray(image.convert("RGB"), dtype=np.float32)
    generated = frame_value_rgb(generated_frame).astype(np.float32)
    alpha = cv2.imread(str(alpha_path), cv2.IMREAD_UNCHANGED)
    if alpha is None:
        raise RuntimeError(f"missing alpha: {alpha_path}")
    alpha = (alpha.astype(np.float32) / 65535.0)[..., None]
    composite = np.rint(generated * alpha + original * (1.0 - alpha)).astype(np.uint8)
    Image.fromarray(composite).save(destination, format="PNG", compress_level=0)


def _episode_rotation(episode: Episode) -> int | None:
    parent_name = episode.source.parent.name.casefold()
    has_left = "left" in parent_name
    has_right = "right" in parent_name
    if has_left and has_right:
        raise ValueError(
            f"video parent directory contains both left and right: {parent_name}"
        )
    if has_left:
        return cv2.ROTATE_90_CLOCKWISE
    if has_right:
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    return None


def _inverse_rotation(rotation: int) -> int:
    if rotation == cv2.ROTATE_90_CLOCKWISE:
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    if rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return cv2.ROTATE_90_CLOCKWISE
    raise ValueError(f"unsupported processing rotation: {rotation}")


def _rotation_name(rotation: int | None) -> str | None:
    if rotation == cv2.ROTATE_90_CLOCKWISE:
        return "clockwise_90"
    if rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return "counterclockwise_90"
    if rotation is None:
        return None
    raise ValueError(f"unsupported processing rotation: {rotation}")


def _rotate_png_directory(
    source_dir: Path,
    output_dir: Path,
    *,
    rotation: int,
    num_workers: int,
) -> None:
    if num_workers < 1:
        raise ValueError("rotation workers must be at least 1")
    sources = frame_paths(source_dir)
    if not sources:
        raise ValueError(f"rotation source contains no PNG frames: {source_dir}")
    recreate_directory(output_dir, output_dir.parent)

    def rotate_one(source: Path) -> None:
        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"cannot read rotation source: {source}")
        destination = output_dir / source.name
        if not cv2.imwrite(str(destination), cv2.rotate(image, rotation)):
            raise RuntimeError(f"cannot write rotated frame: {destination}")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(rotate_one, source) for source in sources]
        for future in futures:
            future.result()


def _write_pasted_correction_frame(
    processed_original_path: Path,
    processed_final_path: Path,
    processed_alpha_path: Path,
    source_original_path: Path,
    output_path: Path,
    output_mask_path: Path,
) -> None:
    with Image.open(processed_original_path) as image:
        processed_original = np.asarray(image.convert("RGB"), dtype=np.float32)
    with Image.open(processed_final_path) as image:
        processed_final = np.asarray(image.convert("RGB"), dtype=np.float32)
    with Image.open(source_original_path) as image:
        source_original = np.asarray(image.convert("RGB"), dtype=np.float32)
    correction = processed_final - processed_original
    source_height, source_width = source_original.shape[:2]
    correction = cv2.resize(
        correction,
        (source_width, source_height),
        interpolation=cv2.INTER_LINEAR,
    )
    output = np.rint(np.clip(source_original + correction, 0.0, 255.0)).astype(np.uint8)
    Image.fromarray(output).save(output_path, format="PNG", compress_level=0)

    alpha = cv2.imread(str(processed_alpha_path), cv2.IMREAD_UNCHANGED)
    if alpha is None:
        raise RuntimeError(f"missing alpha: {processed_alpha_path}")
    output_alpha = cv2.resize(
        alpha,
        (source_width, source_height),
        interpolation=cv2.INTER_LINEAR,
    )
    output_mask = np.where(output_alpha > 0, 255, 0).astype(np.uint8)
    if not cv2.imwrite(str(output_mask_path), output_mask):
        raise RuntimeError(f"cannot write pasted mask: {output_mask_path}")


def _paste_processed_corrections(
    *,
    processed_original_dir: Path,
    processed_final_dir: Path,
    processed_alpha_dir: Path,
    source_original_dir: Path,
    output_dir: Path,
    output_mask_dir: Path,
    num_workers: int,
) -> None:
    processed_originals = frame_paths(processed_original_dir)
    processed_finals = frame_paths(processed_final_dir)
    processed_alphas = frame_paths(processed_alpha_dir)
    source_originals = frame_paths(source_original_dir)
    num_frames = len(processed_originals)
    if not (
        num_frames
        == len(processed_finals)
        == len(processed_alphas)
        == len(source_originals)
    ):
        raise RuntimeError("paste-back frame counts do not match")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _write_pasted_correction_frame,
                processed_originals[index],
                processed_finals[index],
                processed_alphas[index],
                source_originals[index],
                output_dir / f"{index:06d}.png",
                output_mask_dir / f"{index:06d}.png",
            )
            for index in range(num_frames)
        ]
        for future in futures:
            future.result()


def _run_inpainter_chunks(
    *,
    originals: list[Path],
    mask_dir: Path,
    alpha_dir: Path,
    final_dir: Path,
    runner: InpainterBackend,
    fps,
    config: dict,
    work: Path,
    log_prefix: str,
) -> None:
    from .model_service import inpainter_config

    selected_config = inpainter_config(config)
    overlap = int(selected_config["chunk_overlap"])
    dynamic_mask_ranges = selected_config.get("dynamic_mask_ranges", False)
    if not isinstance(dynamic_mask_ranges, bool):
        raise ValueError("inpainter.dynamic_mask_ranges must be true or false")
    mask_context_frames = int(selected_config.get("mask_context_frames", 0))
    if mask_context_frames < 0:
        raise ValueError("inpainter.mask_context_frames must be >= 0")
    if dynamic_mask_ranges:
        mask_activity = []
        for global_index in range(len(originals)):
            mask_path = mask_dir / f"{global_index:06d}.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"cannot read inpainting mask: {mask_path}")
            mask_activity.append(bool(np.any(mask)))
        active_ranges = _expanded_mask_ranges(
            mask_activity,
            context_frames=mask_context_frames,
            min_frames=int(selected_config.get("min_chunk_frames", 1)),
        )
        ranges = [
            (start + local_start, start + local_end)
            for start, end in active_ranges
            for local_start, local_end in chunk_ranges(
                end - start,
                int(selected_config["chunk_frames"]),
                overlap,
                min_size=int(selected_config.get("min_chunk_frames", 1)),
            )
        ]
    else:
        ranges = chunk_ranges(
            len(originals),
            int(selected_config["chunk_frames"]),
            overlap,
            min_size=int(selected_config.get("min_chunk_frames", 1)),
        )
    final_dir.mkdir(parents=True, exist_ok=True)
    best_weights = np.full(len(originals), -1.0, dtype=np.float32)
    composite_workers = int(config.get("runtime", {}).get("composite_workers", 4))
    if composite_workers < 1:
        raise ValueError("runtime.composite_workers must be at least 1")

    covered = np.zeros(len(originals), dtype=bool)
    for start, end in ranges:
        covered[start:end] = True
    for global_index in np.flatnonzero(~covered).tolist():
        relative_symlink(
            originals[global_index], final_dir / f"{global_index:06d}.png"
        )
        best_weights[global_index] = 0.0

    for chunk_index, (start, end) in enumerate(ranges):
        chunk_root = work / "inpainter_chunks" / f"{chunk_index:04d}"
        frame_dir = recreate_directory(chunk_root / "frames", work)
        chunk_mask_dir = recreate_directory(chunk_root / "masks", work)
        masks_are_empty = True
        for local_index, global_index in enumerate(range(start, end)):
            relative_symlink(
                originals[global_index], frame_dir / f"{local_index:06d}.png"
            )
            source_mask = mask_dir / f"{global_index:06d}.png"
            relative_symlink(source_mask, chunk_mask_dir / f"{local_index:06d}.png")
            if masks_are_empty:
                mask = cv2.imread(str(source_mask), cv2.IMREAD_GRAYSCALE)
                masks_are_empty = mask is None or not np.any(mask)

        if masks_are_empty:
            generated_frames = None
        else:
            output_root = chunk_root / "output"
            generated_store = runner.run_chunk(
                frame_dir,
                chunk_mask_dir,
                output_root,
                fps,
                f"{log_prefix}.propainter.{chunk_index:04d}",
            )
            generated_frames = load_frame_store(generated_store)
            if len(generated_frames) != end - start:
                raise RuntimeError(
                    f"inpainter chunk {chunk_index} returned "
                    f"{len(generated_frames)} frames, expected {end - start}"
                )

        updates: list[tuple[int, float]] = []
        futures = []
        with ThreadPoolExecutor(max_workers=composite_workers) as executor:
            for local_index, global_index in enumerate(range(start, end)):
                weight = blend_weight(local_index, end - start, overlap)
                if weight <= best_weights[global_index]:
                    continue
                destination = final_dir / f"{global_index:06d}.png"
                if generated_frames is None:
                    relative_symlink(originals[global_index], destination)
                else:
                    futures.append(
                        executor.submit(
                            _write_composited_frame,
                            originals[global_index],
                            generated_frames[local_index],
                            alpha_dir / f"{global_index:06d}.png",
                            destination,
                        )
                    )
                updates.append((global_index, weight))
            for future in futures:
                future.result()
        for global_index, weight in updates:
            best_weights[global_index] = weight
        remove_directory(chunk_root, work)

    if np.any(best_weights < 0) or len(frame_paths(final_dir)) != len(originals):
        missing = np.flatnonzero(best_weights < 0).tolist()
        raise RuntimeError(f"ProPainter merge missed frames: {missing[:20]}")


def _expanded_mask_ranges(
    mask_activity: list[bool],
    *,
    context_frames: int,
    min_frames: int = 1,
) -> list[tuple[int, int]]:
    if context_frames < 0:
        raise ValueError("context_frames must be >= 0")
    if min_frames < 1:
        raise ValueError("min_frames must be >= 1")
    active_indexes = np.flatnonzero(mask_activity).tolist()
    if not active_indexes:
        return []

    raw_ranges: list[tuple[int, int]] = []
    start = previous = active_indexes[0]
    for index in active_indexes[1:]:
        if index != previous + 1:
            raw_ranges.append((start, previous + 1))
            start = index
        previous = index
    raw_ranges.append((start, previous + 1))

    total_frames = len(mask_activity)
    target_frames = min(min_frames, total_frames)
    expanded = []
    for start, end in raw_ranges:
        start = max(0, start - context_frames)
        end = min(total_frames, end + context_frames)
        missing = target_frames - (end - start)
        if missing > 0:
            extend_left = min(start, missing // 2)
            start -= extend_left
            missing -= extend_left
            extend_right = min(total_frames - end, missing)
            end += extend_right
            missing -= extend_right
            if missing > 0:
                start -= min(start, missing)
        expanded.append((start, end))
    merged = [expanded[0]]
    for start, end in expanded[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _fallback_to_source_video(
    *,
    config: dict,
    episode: Episode,
    paths: dict[str, Path],
    work: Path,
    error: Exception,
    error_traceback: str,
    target_height: int,
    source_info,
    rotation: int | None,
) -> dict:
    warning_message = (
        f"Inpainting failed for {episode.identity}; copying the source video "
        f"to {paths['inpainted']} as a fallback: "
        f"{type(error).__name__}: {error}"
    )
    logger.warning(
        warning_message,
        exc_info=(type(error), error, error.__traceback__),
    )
    atomic_copy_file(episode.source, paths["inpainted"])
    output_contract = _output_contract(config)
    record = {
        "status": "needs_review",
        "structural_errors": [],
        "review_reasons": ["inpainting_failed_source_copy_fallback"],
        "identity": episode.identity,
        "dataset": episode.dataset,
        "source_path": str(episode.source),
        "inpainted_path": str(paths["inpainted"]),
        "output_all_keyframes": False,
        "pipeline_version": output_contract["pipeline_version"],
        "inpainting": output_contract["inpainting"],
        "encoding": output_contract["encoding"],
        "output_contract": output_contract,
        "processing": {
            "target_height": target_height,
            "width": source_info.width,
            "height": source_info.height,
            "paste_back_to_source": False,
            "rotation": _rotation_name(rotation),
        },
        "fallback": {
            "stage": "inpainting",
            "action": "source_video_copy",
            "warning": warning_message,
            "error": f"{type(error).__name__}: {error}",
            "traceback": error_traceback,
        },
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(paths["qc"], record)
    remove_directory(work, config["work_root"])
    return record


def _process_episode(
    *,
    config: dict,
    run_id: str,
    episode: Episode,
    stage: str,
    preprocess_semaphore,
    backend,
) -> dict:
    paths = _episode_paths(config, run_id, episode)
    work = recreate_directory(paths["work"], config["work_root"])
    original_frames = work / "original_frames"
    alpha_dir = work / "alpha"
    final_frames = work / "final_frames"
    source_frames = work / "source_frames"
    model_original_frames = work / "model_original_frames"
    model_source_frames = work / "model_source_frames"
    pasted_frames = work / "pasted_frames"
    pasted_mask_dir = work / "pasted_mask"
    restored_frames = work / "restored_frames"
    restored_mask_dir = work / "restored_mask"
    retain_debug_artifacts = stage != "full"
    # Per-frame masks are scratch data. Keep them with the decoded frames so
    # calibration runs avoid retaining thousands of individual frame files.
    mask_dir = work / "hand_mask"
    try:
        with preprocess_semaphore:
            if retain_debug_artifacts:
                hardlink_or_copy(episode.source, paths["original"])
            target_height, paste_back = _processing_options(config)
            rotation = _episode_rotation(episode)
            rotation_workers = int(
                config.get("runtime", {}).get("composite_workers", 4)
            )
            source_info = probe_video(episode.source)
            processing_info = decode_video_to_pngs(
                episode.source,
                original_frames,
                target_height=target_height,
                full_resolution_dir=source_frames if paste_back else None,
            )
            if rotation is not None:
                _rotate_png_directory(
                    original_frames,
                    model_original_frames,
                    rotation=rotation,
                    num_workers=rotation_workers,
                )
                if paste_back:
                    _rotate_png_directory(
                        source_frames,
                        model_source_frames,
                        rotation=rotation,
                        num_workers=rotation_workers,
                    )
            else:
                model_original_frames = original_frames
                model_source_frames = source_frames
            originals = frame_paths(model_original_frames)
            probability_dir = backend.segment(
                originals=originals, config=config, work=work
            )
            mask_stats = _finalize_masks(probability_dir, mask_dir, alpha_dir, config)
        try:
            _run_inpainter_chunks(
                originals=originals,
                mask_dir=mask_dir,
                alpha_dir=alpha_dir,
                final_dir=final_frames,
                runner=backend,
                fps=source_info.fps,
                config=config,
                work=work,
                log_prefix=episode.identity.replace("/", "."),
            )
        except Exception as error:
            return _fallback_to_source_video(
                config=config,
                episode=episode,
                paths=paths,
                work=work,
                error=error,
                error_traceback=traceback.format_exc(),
                target_height=target_height,
                source_info=source_info,
                rotation=rotation,
            )
        if paste_back:
            composite_workers = int(
                config.get("runtime", {}).get("composite_workers", 4)
            )
            _paste_processed_corrections(
                processed_original_dir=original_frames,
                processed_final_dir=final_frames,
                processed_alpha_dir=alpha_dir,
                source_original_dir=model_source_frames,
                output_dir=pasted_frames,
                output_mask_dir=pasted_mask_dir,
                num_workers=composite_workers,
            )
            output_frame_dir = pasted_frames
            output_mask_dir = pasted_mask_dir
            qc_original_frame_dir = source_frames
            expected_output_info = source_info
        else:
            output_frame_dir = final_frames
            output_mask_dir = mask_dir
            qc_original_frame_dir = original_frames
            expected_output_info = processing_info
        if rotation is not None:
            inverse_rotation = _inverse_rotation(rotation)
            _rotate_png_directory(
                output_frame_dir,
                restored_frames,
                rotation=inverse_rotation,
                num_workers=rotation_workers,
            )
            _rotate_png_directory(
                output_mask_dir,
                restored_mask_dir,
                rotation=inverse_rotation,
                num_workers=rotation_workers,
            )
            output_frame_dir = restored_frames
            output_mask_dir = restored_mask_dir
        output_config = config["output"]
        encoded_frames = encode_rgb_frames(
            iter_png_rgb(output_frame_dir),
            paths["inpainted"],
            fps=processing_info.fps,
            width=expected_output_info.width,
            height=expected_output_info.height,
            codec=output_config["video_codec"],
            crf=int(output_config["crf"]),
            preset=output_config["preset"],
            all_keyframes=True,
        )
        if encoded_frames != processing_info.num_frames:
            raise RuntimeError(
                f"encoded {encoded_frames} frames, expected "
                f"{processing_info.num_frames}"
            )
        if not video_is_all_keyframes(paths["inpainted"]):
            raise RuntimeError(
                f"encoded output is not all-keyframe: {paths['inpainted']}"
            )
        if retain_debug_artifacts:
            preview_height = int(output_config["preview_height"])
            preview_width = (
                max(
                    2,
                    int(
                        round(
                            expected_output_info.width
                            * preview_height
                            / expected_output_info.height
                        )
                    )
                    // 2
                    * 2,
                )
                * 3
            )
            encode_rgb_frames(
                make_preview_frames(
                    qc_original_frame_dir,
                    output_mask_dir,
                    output_frame_dir,
                    preview_height=preview_height,
                ),
                paths["preview"],
                fps=processing_info.fps,
                width=preview_width,
                height=preview_height,
                codec=output_config["video_codec"],
                crf=23,
                preset="fast",
            )
        record = evaluate_episode(
            source_info=source_info,
            expected_output_info=expected_output_info,
            original_frame_dir=qc_original_frame_dir,
            mask_dir=output_mask_dir,
            final_frame_dir=output_frame_dir,
            inpainted_video=paths["inpainted"],
            mask_frame_stats=(mask_stats if retain_debug_artifacts else []),
            config=config,
        )
        if retain_debug_artifacts:
            encode_mask_pngs(
                output_mask_dir,
                paths["mask_video"],
                fps=processing_info.fps,
                width=expected_output_info.width,
                height=expected_output_info.height,
                crf=int(output_config["mask_crf"]),
            )
            remove_directory(output_mask_dir, config["work_root"])
        record.update(
            {
                "identity": episode.identity,
                "dataset": episode.dataset,
                "source_path": str(episode.source),
                "original_path": (
                    str(paths["original"]) if retain_debug_artifacts else None
                ),
                "hand_mask_path": (
                    str(paths["mask_video"]) if retain_debug_artifacts else None
                ),
                "inpainted_path": str(paths["inpainted"]),
                "output_all_keyframes": True,
                "pipeline_version": PIPELINE_OUTPUT_VERSION,
                "inpainting": _output_contract(config)["inpainting"],
                "encoding": _output_contract(config)["encoding"],
                "output_contract": _output_contract(config),
                "processing": {
                    "target_height": target_height,
                    "width": processing_info.width,
                    "height": processing_info.height,
                    "paste_back_to_source": paste_back,
                    "rotation": _rotation_name(rotation),
                },
                "preview_path": (
                    str(paths["preview"]) if retain_debug_artifacts else None
                ),
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
        )
        write_json(paths["qc"], record)
        remove_directory(work, config["work_root"])
        return record
    except Exception as error:
        failure = {
            "status": "failed",
            "identity": episode.identity,
            "dataset": episode.dataset,
            "source_path": str(episode.source),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "work_path": str(work),
            "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        write_json(paths["qc"], failure)
        return failure


class _LocalModelBackend:
    """In-process compatibility backend used when no service is configured."""

    def __init__(self, config: dict) -> None:
        from .backends import (
            build_detector,
            build_inpainter,
            build_segmenter,
        )

        self.detector = build_detector(detector_config(config), device="cuda:0")
        self.segmenter = build_segmenter(
            segmenter_config(config), device="cuda:0"
        )
        self.propainter = build_inpainter(
            config["models"]["propainter"],
            log_root=config["log_root"],
            device="cuda:0",
            persistent=False,
        )

    def segment(self, *, originals: list[Path], config: dict, work: Path) -> Path:
        return _run_sam_chunks(
            originals=originals,
            detector=self.detector,
            segmenter=self.segmenter,
            config=config,
            work=work,
        )

    def run_chunk(
        self,
        frame_dir: str | Path,
        mask_dir: str | Path,
        output_root: str | Path,
        fps,
        log_name: str,
    ) -> Path:
        return self.propainter.run_chunk(
            frame_dir, mask_dir, output_root, fps, log_name
        )


def _worker(
    config: dict,
    run_id: str,
    episodes: list[Episode],
    gpu: str,
    lane: int,
    stage: str,
    startup_delay_seconds: float,
    preprocess_semaphore,
    result_queue,
    service_dir: str | None,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    configured_cpu_threads = config.get("runtime", {}).get("cpu_threads", 4)
    cpu_threads = int(
        os.environ.get("HAND_PROCESS_CPU_THREADS", str(configured_cpu_threads))
    )
    cv2.setNumThreads(min(2, cpu_threads))
    if startup_delay_seconds > 0:
        time.sleep(startup_delay_seconds)
    completed = 0
    failed = 0
    try:
        if service_dir is not None:
            from .model_service import ModelServiceClient, socket_path

            backend = ModelServiceClient(
                socket_path(service_dir, gpu, lane), config
            )
        else:
            import torch

            torch.set_num_threads(cpu_threads)
            torch.set_num_interop_threads(1)
            backend = _LocalModelBackend(config)
    except Exception as error:
        detail = f"{type(error).__name__}: {error}"
        for episode in episodes:
            paths = _episode_paths(config, run_id, episode)
            write_json(
                paths["qc"],
                {
                    "status": "failed",
                    "identity": episode.identity,
                    "dataset": episode.dataset,
                    "source_path": str(episode.source),
                    "error": f"worker initialization failed: {detail}",
                    "traceback": traceback.format_exc(),
                },
            )
            failed += 1
            result_queue.put(
                {
                    "kind": "episode",
                    "gpu": gpu,
                    "lane": lane,
                    "dataset": episode.dataset,
                    "identity": episode.identity,
                    "status": "failed",
                }
            )
        result_queue.put(
            {
                "kind": "worker",
                "gpu": gpu,
                "lane": lane,
                "completed": 0,
                "failed": failed,
            }
        )
        return

    current_dataset: str | None = None
    for episode in episodes:
        if episode.dataset != current_dataset:
            current_dataset = episode.dataset
            result_queue.put(
                {
                    "kind": "dataset_start",
                    "gpu": gpu,
                    "lane": lane,
                    "dataset": episode.dataset,
                }
            )
        print(
            f"[gpu {gpu} lane {lane}] starting {episode.identity}",
            flush=True,
        )
        result = _process_episode(
            config=config,
            run_id=run_id,
            episode=episode,
            stage=stage,
            preprocess_semaphore=preprocess_semaphore,
            backend=backend,
        )
        if result["status"] == "failed":
            failed += 1
            detail = result.get("error") or "; ".join(
                result.get("structural_errors", ["unknown QC failure"])
            )
            print(
                f"[gpu {gpu} lane {lane}] failed {episode.identity}: {detail}",
                flush=True,
            )
        else:
            completed += 1
            print(
                f"[gpu {gpu} lane {lane}] finished {episode.identity}: "
                f"{result['status']}",
                flush=True,
            )
        result_queue.put(
            {
                "kind": "episode",
                "gpu": gpu,
                "lane": lane,
                "dataset": episode.dataset,
                "identity": episode.identity,
                "status": result["status"],
            }
        )
    result_queue.put(
        {
            "kind": "worker",
            "gpu": gpu,
            "lane": lane,
            "completed": completed,
            "failed": failed,
        }
    )


def _balance_assignments(
    episodes: list[Episode],
    num_workers: int,
    *,
    target_height: int | None = None,
) -> tuple[list[list[Episode]], list[int]]:
    """Greedily balance work using decoded pixel-frames as the cost proxy."""

    assignments: list[list[Episode]] = [[] for _ in range(num_workers)]
    loads = [0 for _ in range(num_workers)]
    scored = []
    for episode in episodes:
        info = probe_video(episode.source)
        width, height = (
            scaled_dimensions(info.width, info.height, target_height=target_height)
            if target_height is not None
            else (info.width, info.height)
        )
        cost = info.num_frames * width * height
        scored.append((cost, episode.identity, episode))
    for cost, _identity, episode in sorted(scored, reverse=True):
        worker_index = min(range(num_workers), key=lambda index: (loads[index], index))
        assignments[worker_index].append(episode)
        loads[worker_index] += cost
    return assignments, loads


def _ordered_assignments(
    episodes: list[Episode],
    num_workers: int,
    *,
    target_height: int | None = None,
    ordered_datasets: bool = False,
) -> tuple[list[list[Episode]], list[int]]:
    if not ordered_datasets:
        return _balance_assignments(
            episodes,
            num_workers,
            target_height=target_height,
        )

    grouped: dict[str, list[Episode]] = {}
    for episode in episodes:
        grouped.setdefault(episode.dataset, []).append(episode)
    ordered_assignments: list[list[Episode]] = [
        [] for _ in range(num_workers)
    ]
    total_loads = [0] * num_workers
    for dataset_episodes in grouped.values():
        assignments, loads = _balance_assignments(
            dataset_episodes,
            num_workers,
            target_height=target_height,
        )
        for worker_index, assignment in enumerate(assignments):
            ordered_assignments[worker_index].extend(assignment)
        total_loads = [
            current + added for current, added in zip(total_loads, loads)
        ]
    return ordered_assignments, total_loads


def _has_resumable_output(
    qc_path: Path,
    *,
    expected_rotation: str | None = None,
    expected_contract: dict | None = None,
) -> bool:
    """Return whether a QC record represents a completed retained output."""

    if not qc_path.is_file():
        return False
    try:
        with qc_path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    if record.get("status") not in {"pass", "needs_review"}:
        return False
    output_value = record.get("inpainted_path")
    if not output_value or record.get("output_all_keyframes") is not True:
        return False
    if record.get("processing", {}).get("rotation") != expected_rotation:
        return False
    if expected_contract is not None and record.get("output_contract") != expected_contract:
        return False
    output = Path(output_value)
    return output.is_file() and output.stat().st_size > 0


def run_pipeline(
    *,
    config: dict,
    run_id: str,
    stage: str,
    episodes: Iterable[Episode],
    gpus: list[str],
    resume: bool,
    service_dir: str | None = None,
    lanes_per_gpu: int = 1,
) -> dict:
    if not run_id or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in run_id
    ):
        raise ValueError("run_id may contain only letters, digits, '-' and '_'")
    if not gpus:
        raise ValueError("at least one GPU is required")
    if lanes_per_gpu < 1:
        raise ValueError("lanes_per_gpu must be at least 1")
    if service_dir is None and lanes_per_gpu != 1:
        raise ValueError("multiple lanes require a model service directory")
    _validate_work_root(config)
    if service_dir is not None:
        from .model_service import (
            model_service_endpoints,
            validate_service_ready,
        )

        endpoints = model_service_endpoints(
            service_dir,
            gpus,
            lanes_per_gpu=lanes_per_gpu,
        )
        for endpoint in endpoints:
            validate_service_ready(endpoint.address, config)
        worker_specs = [
            (endpoint.gpu, endpoint.lane) for endpoint in endpoints
        ]
    else:
        worker_specs = [(gpu, 0) for gpu in gpus]

    selected = list(episodes)
    run_root = Path(config["run_root"]) / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    output_contract = _output_contract(config)
    pending: list[Episode] = []
    for episode in selected:
        qc_path = _episode_paths(config, run_id, episode)["qc"]
        if resume and _has_resumable_output(
            qc_path,
            expected_rotation=_rotation_name(_episode_rotation(episode)),
            expected_contract=output_contract,
        ):
            continue
        pending.append(episode)
    runtime_config = config.get("runtime", {})
    ordered_datasets = runtime_config.get("ordered_datasets", False)
    if not isinstance(ordered_datasets, bool):
        raise ValueError("runtime.ordered_datasets must be true or false")
    target_height, _paste_back = _processing_options(config)
    assignments, estimated_loads = _ordered_assignments(
        pending,
        len(worker_specs),
        target_height=target_height,
        ordered_datasets=ordered_datasets,
    )
    write_json(
        run_root / "run.json",
        {
            "run_id": run_id,
            "stage": stage,
            "retained_artifacts": (
                ["original", "hand_mask", "inpainted", "preview", "qc"]
                if stage != "full"
                else ["inpainted", "qc"]
            ),
            "selected": len(selected),
            "pending": len(pending),
            "gpus": gpus,
            "lanes_per_gpu": lanes_per_gpu,
            "ordered_datasets": ordered_datasets,
            "processing": config.get("processing", {}),
            "estimated_pixel_frames_per_worker": {
                f"gpu-{gpu}-lane-{lane}": estimated_loads[index]
                for index, (gpu, lane) in enumerate(worker_specs)
            },
            "estimated_pixel_frames_per_gpu": {
                gpu: sum(
                    estimated_loads[index]
                    for index, (worker_gpu, _lane) in enumerate(worker_specs)
                    if worker_gpu == gpu
                )
                for gpu in gpus
            },
            "config_path": config.get("_config_path"),
            "model_service_dir": service_dir,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    progress_started = time.monotonic()
    progress_completed = len(selected) - len(pending)
    progress_processed = 0
    progress_failed = 0
    dataset_order = list(dict.fromkeys(episode.dataset for episode in selected))
    dataset_indexes = {
        dataset: index + 1 for index, dataset in enumerate(dataset_order)
    }
    dataset_totals = Counter(episode.dataset for episode in selected)
    dataset_completed = Counter(episode.dataset for episode in selected)
    dataset_completed.subtract(episode.dataset for episode in pending)
    dataset_labels = {
        dataset: str(
            config.get("datasets", {})
            .get(dataset, {})
            .get("display_name", dataset)
        )
        for dataset in dataset_order
    }
    initial_detail = (
        f"resumed {progress_completed}"
        if progress_completed
        else None
    )
    print(
        _global_progress_line(
            progress_completed,
            len(selected),
            progress_failed,
            0.0,
            latest=initial_detail,
        ),
        file=sys.stderr,
        flush=True,
    )
    if not pending:
        records = build_manifest(run_root)
        return {"selected": len(selected), "processed": 0, "records": len(records)}

    context = mp.get_context("spawn")
    configured_preprocess_workers = int(
        runtime_config.get("preprocess_workers", len(worker_specs))
    )
    if configured_preprocess_workers < 1:
        raise ValueError("runtime.preprocess_workers must be at least 1")
    preprocess_workers = min(configured_preprocess_workers, len(worker_specs))
    startup_stagger = float(runtime_config.get("worker_start_stagger_seconds", 10.0))
    if startup_stagger < 0:
        raise ValueError("runtime.worker_start_stagger_seconds must be >= 0")
    preprocess_semaphore = context.BoundedSemaphore(preprocess_workers)
    worker_totals: dict[tuple[str, int], dict] = {}
    bad_processes: list[dict] = []

    def consume_result(message: dict) -> None:
        nonlocal progress_completed, progress_failed, progress_processed
        kind = message.get("kind")
        if kind == "worker":
            gpu = str(message.get("gpu", "?"))
            lane = int(message.get("lane", 0))
            aggregate = worker_totals.setdefault(
                (gpu, lane),
                {"gpu": gpu, "lane": lane, "completed": 0, "failed": 0},
            )
            aggregate["completed"] += int(message.get("completed", 0))
            aggregate["failed"] += int(message.get("failed", 0))
            return
        dataset = str(message.get("dataset", "unknown"))
        dataset_index = dataset_indexes.get(dataset, 0)
        dataset_label = dataset_labels.get(dataset, dataset)
        gpu = str(message.get("gpu", "?"))
        lane = str(message.get("lane", "?"))
        if kind == "dataset_start":
            print(
                _global_progress_line(
                    progress_completed,
                    len(selected),
                    progress_failed,
                    time.monotonic() - progress_started,
                    latest=(
                        f"dataset={dataset_index}/{len(dataset_order)} "
                        f"{dataset_label} "
                        f"{dataset_completed[dataset]}/"
                        f"{dataset_totals[dataset]} starting "
                        f"(gpu {gpu} lane {lane})"
                    ),
                    rate_completed=progress_processed,
                ),
                file=sys.stderr,
                flush=True,
            )
            return
        if kind != "episode":
            raise ValueError(f"unknown worker result kind: {kind!r}")
        progress_completed += 1
        progress_processed += 1
        status = str(message.get("status", "unknown"))
        if status == "failed":
            progress_failed += 1
        identity = str(message.get("identity", "unknown"))
        dataset_completed[dataset] += 1
        dataset_detail = (
            f"dataset={dataset_index}/{len(dataset_order)} {dataset_label} "
            f"{dataset_completed[dataset]}/{dataset_totals[dataset]}"
        )
        print(
            _global_progress_line(
                progress_completed,
                len(selected),
                progress_failed,
                time.monotonic() - progress_started,
                latest=(
                    f"{dataset_detail} | {status} {identity} "
                    f"(gpu {gpu} lane {lane})"
                ),
                rate_completed=progress_processed,
            ),
            file=sys.stderr,
            flush=True,
        )

    result_queue = context.Queue()
    processes = []
    for worker_index, ((gpu, lane), assignment) in enumerate(
        zip(worker_specs, assignments)
    ):
        if not assignment:
            continue
        process = context.Process(
            target=_worker,
            args=(
                config,
                run_id,
                assignment,
                gpu,
                lane,
                stage,
                worker_index * startup_stagger,
                preprocess_semaphore,
                result_queue,
                service_dir,
            ),
            name=f"hand-process-gpu-{gpu}-lane-{lane}",
        )
        process.start()
        processes.append(process)

    while any(process.is_alive() for process in processes):
        try:
            consume_result(result_queue.get(timeout=1))
        except Empty:
            pass
    for process in processes:
        process.join()
    while True:
        try:
            consume_result(result_queue.get_nowait())
        except Empty:
            break
    bad_processes.extend(
        {"name": process.name, "exit_code": process.exitcode}
        for process in processes
        if process.exitcode != 0
    )
    result_queue.close()
    result_queue.join_thread()

    worker_results = [
        worker_totals[(gpu, lane)]
        for gpu, lane in worker_specs
        if (gpu, lane) in worker_totals
    ]
    records = build_manifest(run_root)
    summary_path = run_root / "summary.json"
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    summary.update(
        {
            "selected": len(selected),
            "processed_this_invocation": len(pending),
            "workers": worker_results,
            "bad_processes": bad_processes,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    write_json(summary_path, summary)
    return summary


def compact_run_masks(config: dict, run_id: str) -> dict:
    """Convert retained debug PNG masks to one validated MP4 per episode."""

    run_root = Path(config["run_root"]) / run_id
    summary_path = run_root / "summary.json"
    previous_summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {}
    )
    converted = skipped = 0
    for qc_path in sorted((run_root / "qc").glob("**/*.json")):
        with qc_path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        mask_value = record.get("hand_mask_path")
        if not mask_value:
            skipped += 1
            continue
        mask_dir = Path(mask_value)
        if mask_dir.is_file():
            skipped += 1
            continue
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"mask path does not exist: {mask_dir}")
        source = record["source"]
        mask_video = mask_dir.with_suffix(".mp4")
        encode_mask_pngs(
            mask_dir,
            mask_video,
            fps=Fraction(
                int(source["fps_numerator"]),
                int(source["fps_denominator"]),
            ),
            width=int(source["width"]),
            height=int(source["height"]),
            crf=int(config["output"]["mask_crf"]),
        )
        remove_directory(mask_dir, run_root)
        record["hand_mask_path"] = str(mask_video)
        write_json(qc_path, record)
        converted += 1
    build_manifest(run_root)
    current_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    current_summary.update(
        {
            key: value
            for key, value in previous_summary.items()
            if key not in {"total", "pass", "needs_review", "failed"}
        }
    )
    write_json(summary_path, current_summary)
    return {"converted": converted, "skipped": skipped}

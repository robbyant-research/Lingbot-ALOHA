from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps_numerator: int
    fps_denominator: int
    num_frames: int
    duration_seconds: float
    codec: str
    has_audio: bool

    @property
    def fps(self) -> Fraction:
        return Fraction(self.fps_numerator, self.fps_denominator)

    def to_dict(self) -> dict:
        return asdict(self)


def probe_video(path: str | Path) -> VideoInfo:
    import av

    video_path = Path(path)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.base_rate or Fraction(30, 1)
        duration = (
            float(stream.duration * stream.time_base)
            if stream.duration is not None
            else float(container.duration / av.time_base)
            if container.duration is not None
            else 0.0
        )
        num_frames = int(stream.frames or round(duration * float(rate)))
        return VideoInfo(
            width=int(stream.width),
            height=int(stream.height),
            fps_numerator=int(rate.numerator),
            fps_denominator=int(rate.denominator),
            num_frames=num_frames,
            duration_seconds=duration,
            codec=str(stream.codec_context.name),
            has_audio=bool(container.streams.audio),
        )


def scaled_dimensions(
    width: int, height: int, *, target_height: int
) -> tuple[int, int]:
    if width < 1 or height < 1:
        raise ValueError("source dimensions must be positive")
    if target_height < 2:
        raise ValueError("target_height must be at least 2")
    output_height = target_height // 2 * 2
    output_width = max(
        2, int(round(width * output_height / height)) // 2 * 2
    )
    return output_width, output_height


def decode_video_to_pngs(
    path: str | Path,
    output_dir: str | Path,
    *,
    target_height: int | None = None,
    full_resolution_dir: str | Path | None = None,
) -> VideoInfo:
    import av

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    full_output = (
        Path(full_resolution_dir) if full_resolution_dir is not None else None
    )
    if full_output is not None:
        full_output.mkdir(parents=True, exist_ok=True)
    source_info = probe_video(path)
    output_width, output_height = (
        scaled_dimensions(
            source_info.width,
            source_info.height,
            target_height=target_height,
        )
        if target_height is not None
        else (source_info.width, source_info.height)
    )
    count = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for count, frame in enumerate(container.decode(stream), start=1):
            image = frame.to_image()
            filename = f"{count - 1:06d}.png"
            if full_output is not None:
                image.save(
                    full_output / filename,
                    format="PNG",
                    compress_level=0,
                )
            if image.size != (output_width, output_height):
                image = image.resize(
                    (output_width, output_height),
                    Image.Resampling.LANCZOS,
                )
            image.save(output / filename, format="PNG", compress_level=0)
    if count == 0:
        raise RuntimeError(f"video contains no frames: {path}")
    return VideoInfo(
        width=output_width,
        height=output_height,
        fps_numerator=source_info.fps_numerator,
        fps_denominator=source_info.fps_denominator,
        num_frames=count,
        duration_seconds=count / float(source_info.fps),
        codec=source_info.codec,
        has_audio=source_info.has_audio,
    )


def frame_paths(directory: str | Path, suffix: str = ".png") -> list[Path]:
    return sorted(Path(directory).glob(f"*{suffix}"), key=lambda path: int(path.stem))


def encode_rgb_frames(
    frames: Iterable[np.ndarray],
    output_path: str | Path,
    *,
    fps: Fraction,
    width: int,
    height: int,
    codec: str = "libx264",
    crf: int = 18,
    preset: str = "medium",
    all_keyframes: bool = False,
) -> int:
    import av

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with av.open(str(output), mode="w") as container:
        stream = container.add_stream(codec, rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": preset}
        if all_keyframes:
            stream.codec_context.gop_size = 1
            stream.codec_context.max_b_frames = 0
            stream.options.update(
                {"g": "1", "keyint_min": "1", "sc_threshold": "0"}
            )
        for count, array in enumerate(frames, start=1):
            if array.shape[:2] != (height, width):
                raise ValueError(
                    f"frame shape {array.shape[:2]} != expected {(height, width)}"
                )
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return count


def video_is_all_keyframes(path: str | Path) -> bool:
    import av

    found_packet = False
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for packet in container.demux(stream):
            if packet.size <= 0:
                continue
            found_packet = True
            if not packet.is_keyframe:
                return False
    return found_packet


def iter_png_rgb(directory: str | Path) -> Iterator[np.ndarray]:
    for path in frame_paths(directory):
        with Image.open(path) as image:
            yield np.asarray(image.convert("RGB"))


def encode_mask_pngs(
    mask_dir: str | Path,
    output_path: str | Path,
    *,
    fps: Fraction,
    width: int,
    height: int,
    crf: int,
) -> int:
    source = Path(mask_dir)
    output = Path(output_path)
    expected_count = len(frame_paths(source))
    if expected_count == 0:
        raise ValueError(f"mask directory contains no PNG frames: {source}")
    if output.is_file():
        info = probe_video(output)
        if (
            info.num_frames == expected_count
            and info.width == width
            and info.height == height
            and info.fps == fps
        ):
            return expected_count
        raise FileExistsError(f"existing mask video failed validation: {output}")

    temporary = output.with_name(
        f"{output.stem}.tmp-{os.getpid()}{output.suffix}"
    )
    try:
        count = encode_rgb_frames(
            iter_png_rgb(source),
            temporary,
            fps=fps,
            width=width,
            height=height,
            codec="libx264",
            crf=crf,
            preset="medium",
        )
        info = probe_video(temporary)
        if (
            count != expected_count
            or info.num_frames != expected_count
            or info.width != width
            or info.height != height
            or info.fps != fps
        ):
            raise RuntimeError(
                f"encoded mask video failed validation: {temporary}"
            )
        temporary.rename(output)
        return count
    finally:
        if temporary.exists():
            temporary.unlink()


def make_preview_frames(
    original_dir: str | Path,
    mask_dir: str | Path,
    inpainted_dir: str | Path,
    *,
    preview_height: int,
) -> Iterator[np.ndarray]:
    import cv2

    originals = frame_paths(original_dir)
    masks = frame_paths(mask_dir)
    inpainted = frame_paths(inpainted_dir)
    if not (len(originals) == len(masks) == len(inpainted)):
        raise ValueError("preview inputs have different frame counts")
    for original_path, mask_path, inpainted_path in zip(
        originals, masks, inpainted
    ):
        original = np.asarray(Image.open(original_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path).convert("L"))
        result = np.asarray(Image.open(inpainted_path).convert("RGB"))
        overlay = original.copy()
        green = np.zeros_like(overlay)
        green[..., 1] = 255
        alpha = (mask.astype(np.float32) / 255.0 * 0.55)[..., None]
        overlay = np.rint(overlay * (1 - alpha) + green * alpha).astype(np.uint8)
        scale = preview_height / original.shape[0]
        preview_width = max(2, int(round(original.shape[1] * scale)) // 2 * 2)
        pieces = [
            cv2.resize(item, (preview_width, preview_height), cv2.INTER_AREA)
            for item in (original, overlay, result)
        ]
        yield np.concatenate(pieces, axis=1)

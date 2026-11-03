from fractions import Fraction

import av
import numpy as np
from PIL import Image

from hand_process.video import (
    decode_video_to_pngs,
    encode_mask_pngs,
    encode_rgb_frames,
    frame_paths,
    probe_video,
    scaled_dimensions,
    video_is_all_keyframes,
)


def test_scaled_dimensions_preserves_aspect_ratio_and_even_width():
    assert scaled_dimensions(848, 480, target_height=480) == (848, 480)
    assert scaled_dimensions(1344, 1344, target_height=480) == (480, 480)
    assert scaled_dimensions(1920, 1080, target_height=480) == (852, 480)


def test_decode_video_to_pngs_resizes_and_optionally_keeps_source(tmp_path):
    video_path = tmp_path / "source.mp4"
    frames = [
        np.full((48, 64, 3), index * 20, dtype=np.uint8)
        for index in range(4)
    ]
    encode_rgb_frames(
        frames,
        video_path,
        fps=Fraction(25, 1),
        width=64,
        height=48,
        crf=0,
    )
    processing_dir = tmp_path / "processing"
    source_dir = tmp_path / "source"
    info = decode_video_to_pngs(
        video_path,
        processing_dir,
        target_height=24,
        full_resolution_dir=source_dir,
    )
    assert (info.width, info.height, info.num_frames) == (32, 24, 4)
    assert Image.open(frame_paths(processing_dir)[0]).size == (32, 24)
    assert Image.open(frame_paths(source_dir)[0]).size == (64, 48)


def test_encode_mask_pngs_preserves_timing_and_thresholdable_shape(tmp_path):
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    expected = []
    for index in range(4):
        mask = np.zeros((48, 64), dtype=np.uint8)
        mask[10:30, 8 + index : 28 + index] = 255
        expected.append(mask > 0)
        Image.fromarray(mask).save(mask_dir / f"{index:06d}.png")

    output = tmp_path / "mask.mp4"
    count = encode_mask_pngs(
        mask_dir,
        output,
        fps=Fraction(25, 1),
        width=64,
        height=48,
        crf=12,
    )
    assert count == 4
    info = probe_video(output)
    assert (info.num_frames, info.width, info.height, info.fps) == (
        4,
        64,
        48,
        Fraction(25, 1),
    )

    decoded = []
    with av.open(str(output)) as container:
        for frame in container.decode(container.streams.video[0]):
            decoded.append(frame.to_ndarray(format="gray") > 127)
    for reference, reconstructed in zip(expected, decoded):
        union = np.logical_or(reference, reconstructed).sum()
        intersection = np.logical_and(reference, reconstructed).sum()
        assert intersection / union > 0.98


def test_encode_rgb_frames_can_force_every_frame_to_be_a_keyframe(tmp_path):
    output = tmp_path / "all-keyframes.mp4"
    frames = [
        np.full((48, 64, 3), index * 20, dtype=np.uint8)
        for index in range(6)
    ]

    assert encode_rgb_frames(
        frames,
        output,
        fps=Fraction(25, 1),
        width=64,
        height=48,
        crf=18,
        all_keyframes=True,
    ) == len(frames)
    assert video_is_all_keyframes(output)

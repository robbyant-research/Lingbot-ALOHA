import json
import logging
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from hand_process.datasets import Episode
from hand_process.frame_store import Hdf5VideoWriter
from hand_process.pipeline import (
    _balance_assignments,
    _episode_paths,
    _episode_rotation,
    _expanded_mask_ranges,
    _fallback_to_source_video,
    _format_duration,
    _global_progress_line,
    _has_resumable_output,
    _inverse_rotation,
    _ordered_assignments,
    _rotate_png_directory,
    _run_inpainter_chunks,
    _validate_work_root,
    _write_pasted_correction_frame,
    _write_composited_frame,
)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "00:00"), (65.9, "01:05"), (3661, "1:01:01")],
)
def test_format_duration(seconds, expected):
    assert _format_duration(seconds) == expected


def test_global_progress_line_reports_overall_status_and_eta():
    line = _global_progress_line(
        3,
        12,
        1,
        30.0,
        latest="pass data/episode_000003.mp4 (gpu 2 lane 1)",
        width=12,
    )

    assert "[###---------]" in line
    assert "3/12 ( 25.0%)" in line
    assert "failed=1" in line
    assert "elapsed=00:30 eta=01:30" in line
    assert "pass data/episode_000003.mp4 (gpu 2 lane 1)" in line


def test_global_progress_line_handles_empty_completed_run():
    line = _global_progress_line(0, 0, 0, 0.0, width=4)

    assert "[####]" in line
    assert "0/0 (100.0%)" in line
    assert "eta=00:00" in line


def test_global_progress_eta_excludes_resumed_videos_from_rate():
    line = _global_progress_line(
        91,
        100,
        0,
        10.0,
        rate_completed=1,
        width=10,
    )

    assert "91/100 ( 91.0%)" in line
    assert "eta=01:30" in line


def test_episode_paths_can_write_final_video_directly_to_output(tmp_path):
    episode = Episode(
        "input_videos",
        tmp_path / "input" / "episode_000001.mp4",
        Path("."),
    )
    config = {
        "run_root": str(tmp_path / "runtime"),
        "work_root": str(tmp_path / "work"),
        "final_video_root": str(tmp_path / "output"),
    }
    paths = _episode_paths(config, "run", episode)
    assert paths["inpainted"] == (
        tmp_path / "output" / "episode_000001.mp4"
    )
    assert paths["qc"] == (
        tmp_path / "runtime" / "run" / "qc" / "episode_000001.json"
    )


def test_episode_paths_use_dataset_specific_output_root(tmp_path):
    episode = Episode(
        "input_videos_002",
        tmp_path / "b_in" / "nested" / "episode.mp4",
        Path("nested"),
        Path("nested/episode.mp4"),
    )
    config = {
        "run_root": str(tmp_path / "runtime"),
        "work_root": str(tmp_path / "work"),
        "datasets": {
            "input_videos_001": {
                "final_video_root": str(tmp_path / "a_out")
            },
            "input_videos_002": {
                "final_video_root": str(tmp_path / "b_out")
            },
        },
    }

    paths = _episode_paths(config, "run", episode)

    assert paths["inpainted"] == (
        tmp_path / "b_out" / "nested" / "episode.mp4"
    )


def test_episode_paths_isolate_same_stem_from_different_directories(tmp_path):
    config = {
        "run_root": str(tmp_path / "runtime"),
        "work_root": str(tmp_path / "work"),
        "final_video_root": str(tmp_path / "output"),
    }
    left = Episode(
        "input_videos",
        tmp_path / "videos" / "left" / "episode_000001.mp4",
        Path("left"),
        Path("left/episode_000001.mp4"),
    )
    right = Episode(
        "input_videos",
        tmp_path / "videos" / "right" / "episode_000001.mp4",
        Path("right"),
        Path("right/episode_000001.mp4"),
    )

    left_paths = _episode_paths(config, "run", left)
    right_paths = _episode_paths(config, "run", right)

    assert left_paths["work"] == (
        tmp_path / "work" / "run" / "input_videos" / "left" / "episode_000001"
    )
    assert right_paths["work"] == (
        tmp_path / "work" / "run" / "input_videos" / "right" / "episode_000001"
    )
    assert left_paths["work"] != right_paths["work"]


@pytest.mark.parametrize(
    ("parent", "expected"),
    [
        ("observation.image.left", cv2.ROTATE_90_CLOCKWISE),
        ("observation.image.right", cv2.ROTATE_90_COUNTERCLOCKWISE),
        ("observation.image.head", None),
    ],
)
def test_episode_rotation_uses_immediate_parent_name(tmp_path, parent, expected):
    episode = Episode(
        "input_videos",
        tmp_path / parent / "episode.mp4",
        Path(parent),
    )
    assert _episode_rotation(episode) == expected


def test_rotate_png_directory_round_trips(tmp_path):
    source = tmp_path / "source"
    rotated = tmp_path / "rotated"
    restored = tmp_path / "restored"
    source.mkdir()
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    assert cv2.imwrite(str(source / "000000.png"), image)

    _rotate_png_directory(
        source,
        rotated,
        rotation=cv2.ROTATE_90_CLOCKWISE,
        num_workers=1,
    )
    _rotate_png_directory(
        rotated,
        restored,
        rotation=_inverse_rotation(cv2.ROTATE_90_CLOCKWISE),
        num_workers=1,
    )

    np.testing.assert_array_equal(
        cv2.imread(str(restored / "000000.png"), cv2.IMREAD_UNCHANGED),
        image,
    )


def test_expanded_mask_ranges_keep_context_and_merge_overlaps():
    activity = [False] * 50
    activity[20] = True
    activity[38:40] = [True, True]

    assert _expanded_mask_ranges(activity, context_frames=11) == [(9, 50)]


def test_expanded_mask_ranges_keep_separate_distant_activity():
    activity = [False] * 80
    activity[15] = True
    activity[60] = True

    assert _expanded_mask_ranges(activity, context_frames=11) == [
        (4, 27),
        (49, 72),
    ]


def test_expanded_mask_ranges_skip_fully_empty_video():
    assert _expanded_mask_ranges([False] * 20, context_frames=11) == []


@pytest.mark.parametrize(
    ("active_index", "expected"),
    [(0, (0, 22)), (49, (28, 50))],
)
def test_expanded_mask_ranges_extend_boundary_activity_to_minimum(
    active_index, expected
):
    activity = [False] * 50
    activity[active_index] = True

    assert _expanded_mask_ranges(
        activity,
        context_frames=11,
        min_frames=22,
    ) == [expected]


def test_expanded_mask_ranges_use_whole_video_when_shorter_than_minimum():
    activity = [False] * 20
    activity[0] = True

    assert _expanded_mask_ranges(
        activity,
        context_frames=11,
        min_frames=22,
    ) == [(0, 20)]


def test_inpainting_failure_copies_source_and_logs_warning(
    tmp_path, caplog
):
    source = tmp_path / "input" / "episode.mp4"
    source.parent.mkdir()
    source.write_bytes(b"source-video")
    episode = Episode("input_videos", source, Path("."))
    config = {
        "run_root": str(tmp_path / "runtime"),
        "work_root": str(tmp_path / "work"),
        "final_video_root": str(tmp_path / "output"),
        "processing": {
            "target_height": 360,
            "paste_back_to_source": False,
        },
        "models": {"inpainter": {"backend": "propainter"}},
        "output": {
            "video_codec": "libx264",
            "crf": 18,
            "preset": "medium",
        },
    }
    paths = _episode_paths(config, "run", episode)
    paths["work"].mkdir(parents=True)
    error = RuntimeError("inpainter unavailable")

    with caplog.at_level(logging.WARNING, logger="hand_process.pipeline"):
        record = _fallback_to_source_video(
            config=config,
            episode=episode,
            paths=paths,
            work=paths["work"],
            error=error,
            error_traceback="inpainting traceback",
            target_height=360,
            source_info=SimpleNamespace(width=640, height=480),
            rotation=None,
        )

    assert paths["inpainted"].read_bytes() == b"source-video"
    assert record["status"] == "needs_review"
    assert record["review_reasons"] == [
        "inpainting_failed_source_copy_fallback"
    ]
    assert record["output_all_keyframes"] is False
    assert record["fallback"]["action"] == "source_video_copy"
    assert "Inpainting failed" in caplog.text
    assert "inpainter unavailable" in caplog.text
    assert not paths["work"].exists()
    assert json.loads(paths["qc"].read_text(encoding="utf-8")) == record


def test_balance_assignments_uses_pixel_frame_cost(monkeypatch, tmp_path):
    episodes = [
        Episode("data", tmp_path / f"episode_{index}.mp4", tmp_path)
        for index in range(4)
    ]
    costs = {
        episodes[0].source: (100, 10, 10),
        episodes[1].source: (80, 10, 10),
        episodes[2].source: (20, 10, 10),
        episodes[3].source: (20, 10, 10),
    }

    def fake_probe(path):
        frames, width, height = costs[path]
        return SimpleNamespace(
            num_frames=frames,
            width=width,
            height=height,
        )

    monkeypatch.setattr("hand_process.pipeline.probe_video", fake_probe)
    assignments, loads = _balance_assignments(episodes, 2)
    assert sorted(len(values) for values in assignments) == [2, 2]
    assert loads == [12000, 10000]


def test_each_worker_queue_keeps_datasets_in_input_order(
    monkeypatch, tmp_path
):
    episodes = [
        Episode("dataset_a", tmp_path / "a_1.mp4", Path(".")),
        Episode("dataset_a", tmp_path / "a_2.mp4", Path(".")),
        Episode("dataset_b", tmp_path / "b_1.mp4", Path(".")),
        Episode("dataset_b", tmp_path / "b_2.mp4", Path(".")),
    ]

    def fake_probe(_path):
        return SimpleNamespace(num_frames=10, width=10, height=10)

    monkeypatch.setattr("hand_process.pipeline.probe_video", fake_probe)
    assignments, loads = _ordered_assignments(
        episodes,
        2,
        ordered_datasets=True,
    )

    assert [episode.dataset for episode in assignments[0]] == [
        "dataset_a",
        "dataset_b",
    ]
    assert [episode.dataset for episode in assignments[1]] == [
        "dataset_a",
        "dataset_b",
    ]
    assert loads == [2000, 2000]


def test_required_tmpfs_rejects_persistent_frame_scratch(tmp_path):
    config = {
        "work_root": str(tmp_path),
        "runtime": {"require_tmpfs": True},
    }
    with pytest.raises(ValueError, match="path escapes allowed root"):
        _validate_work_root(config)


def test_required_tmpfs_accepts_dev_shm():
    config = {
        "work_root": "/dev/shm/hand-process-test",
        "runtime": {"require_tmpfs": True},
    }
    assert _validate_work_root(config) == Path("/dev/shm/hand-process-test")


def test_resumable_output_requires_successful_qc_and_video(tmp_path):
    qc_path = tmp_path / "qc.json"
    output_path = tmp_path / "inpainted.mp4"
    output_path.write_bytes(b"video")

    qc_path.write_text(
        f'{{"status": "pass", "inpainted_path": "{output_path}", '
        '"output_all_keyframes": true}',
        encoding="utf-8",
    )
    assert _has_resumable_output(qc_path)

    qc_path.write_text(
        f'{{"status": "pass", "inpainted_path": "{output_path}"}}',
        encoding="utf-8",
    )
    assert not _has_resumable_output(qc_path)

    qc_path.write_text(
        f'{{"status": "failed", "inpainted_path": "{output_path}"}}',
        encoding="utf-8",
    )
    assert not _has_resumable_output(qc_path)
    qc_path.write_text("{", encoding="utf-8")
    assert not _has_resumable_output(qc_path)


def test_resumable_output_requires_matching_processing_rotation(tmp_path):
    output_path = tmp_path / "inpainted.mp4"
    output_path.write_bytes(b"video")
    qc_path = tmp_path / "qc.json"
    qc_path.write_text(
        f'{{"status": "pass", "inpainted_path": "{output_path}", '
        '"output_all_keyframes": true, '
        '"processing": {"rotation": "clockwise_90"}}',
        encoding="utf-8",
    )

    assert _has_resumable_output(
        qc_path, expected_rotation="clockwise_90"
    )
    assert not _has_resumable_output(
        qc_path, expected_rotation="counterclockwise_90"
    )


def test_resumable_output_requires_exact_output_contract(tmp_path):
    output_path = tmp_path / "inpainted.mp4"
    output_path.write_bytes(b"video")
    qc_path = tmp_path / "qc.json"
    contract = {
        "pipeline_version": 2,
        "processing": {
            "target_height": 480,
            "paste_back_to_source": False,
        },
        "inpainting": {
            "backend": "diffueraser",
            "output_storage": "hdf5",
            "implementation_version": "generated-frame-store-v1",
        },
        "encoding": {
            "video_codec": "libx264",
            "crf": 1,
            "preset": "medium",
            "pixel_format": "yuv420p",
            "all_keyframes": True,
        },
    }
    qc_path.write_text(
        json.dumps(
            {
                "status": "pass",
                "inpainted_path": str(output_path),
                "output_all_keyframes": True,
                "processing": {"rotation": None},
                "output_contract": contract,
            }
        ),
        encoding="utf-8",
    )

    assert _has_resumable_output(qc_path, expected_contract=contract)
    changed = {**contract, "pipeline_version": 3}
    assert not _has_resumable_output(qc_path, expected_contract=changed)


def test_write_composited_frame_preserves_alpha_blend(tmp_path):
    original = np.full((2, 2, 3), 20, dtype=np.uint8)
    generated = np.full((2, 2, 3), 220, dtype=np.uint8)
    alpha = np.array([[0, 65535], [32768, 16384]], dtype=np.uint16)
    original_path = tmp_path / "original.png"
    generated_path = tmp_path / "generated.png"
    alpha_path = tmp_path / "alpha.png"
    output_path = tmp_path / "output.png"
    Image.fromarray(original).save(original_path)
    Image.fromarray(generated).save(generated_path)
    assert cv2.imwrite(str(alpha_path), alpha)

    _write_composited_frame(
        original_path, generated_path, alpha_path, output_path
    )

    with Image.open(output_path) as image:
        actual = np.asarray(image)
    expected = np.rint(
        generated.astype(np.float32) * (alpha.astype(np.float32) / 65535.0)[..., None]
        + original.astype(np.float32)
        * (1.0 - alpha.astype(np.float32) / 65535.0)[..., None]
    ).astype(np.uint8)
    np.testing.assert_array_equal(actual, expected)


def test_write_composited_frame_accepts_hdf5_array_frame(tmp_path):
    original = np.full((2, 2, 3), 20, dtype=np.uint8)
    generated = np.full((2, 2, 3), 220, dtype=np.uint8)
    alpha = np.full((2, 2), 65535, dtype=np.uint16)
    original_path = tmp_path / "original.png"
    alpha_path = tmp_path / "alpha.png"
    output_path = tmp_path / "output.png"
    Image.fromarray(original).save(original_path)
    assert cv2.imwrite(str(alpha_path), alpha)

    _write_composited_frame(
        original_path, generated, alpha_path, output_path
    )

    np.testing.assert_array_equal(np.asarray(Image.open(output_path)), generated)


def test_run_inpainter_chunks_consumes_and_removes_hdf5_store(tmp_path):
    pytest.importorskip("h5py")
    originals_dir = tmp_path / "originals"
    mask_dir = tmp_path / "masks"
    alpha_dir = tmp_path / "alpha"
    final_dir = tmp_path / "final"
    work = tmp_path / "work"
    for directory in (originals_dir, mask_dir, alpha_dir, work):
        directory.mkdir()
    originals = []
    for index in range(2):
        original_path = originals_dir / f"{index:06d}.png"
        Image.fromarray(np.full((2, 2, 3), 20, dtype=np.uint8)).save(
            original_path
        )
        Image.fromarray(np.full((2, 2), 255, dtype=np.uint8)).save(
            mask_dir / f"{index:06d}.png"
        )
        assert cv2.imwrite(
            str(alpha_dir / f"{index:06d}.png"),
            np.full((2, 2), 65535, dtype=np.uint16),
        )
        originals.append(original_path)

    class Hdf5Runner:
        def run_chunk(
            self, frame_dir, chunk_mask_dir, output_root, fps, log_name
        ):
            del frame_dir, chunk_mask_dir, log_name
            output_path = Path(output_root) / "frames.h5"
            writer = Hdf5VideoWriter(
                output_path,
                expected_num_frames=2,
                fps=fps,
                frame_size=(2, 2),
            )
            for _ in range(2):
                writer.write(np.full((2, 2, 3), 220, dtype=np.uint8))
            writer.release()
            return output_path

    config = {
        "models": {
            "inpainter": {
                "backend": "diffueraser",
                "chunk_frames": 2,
                "chunk_overlap": 0,
                "min_chunk_frames": 1,
            }
        },
        "runtime": {"composite_workers": 2},
    }
    _run_inpainter_chunks(
        originals=originals,
        mask_dir=mask_dir,
        alpha_dir=alpha_dir,
        final_dir=final_dir,
        runner=Hdf5Runner(),
        fps=Fraction(24, 1),
        config=config,
        work=work,
        log_prefix="test",
    )

    assert not list(work.rglob("*.h5"))
    for output_path in sorted(final_dir.glob("*.png")):
        np.testing.assert_array_equal(
            np.asarray(Image.open(output_path)),
            np.full((2, 2, 3), 220, dtype=np.uint8),
        )


def test_write_pasted_correction_frame_applies_low_resolution_delta(tmp_path):
    processed_original = np.full((2, 2, 3), 20, dtype=np.uint8)
    processed_final = np.full((2, 2, 3), 50, dtype=np.uint8)
    source_original = np.full((4, 4, 3), 100, dtype=np.uint8)
    alpha = np.full((2, 2), 65535, dtype=np.uint16)
    paths = {
        name: tmp_path / f"{name}.png"
        for name in (
            "processed_original",
            "processed_final",
            "source_original",
            "alpha",
            "output",
            "mask",
        )
    }
    Image.fromarray(processed_original).save(paths["processed_original"])
    Image.fromarray(processed_final).save(paths["processed_final"])
    Image.fromarray(source_original).save(paths["source_original"])
    assert cv2.imwrite(str(paths["alpha"]), alpha)

    _write_pasted_correction_frame(
        paths["processed_original"],
        paths["processed_final"],
        paths["alpha"],
        paths["source_original"],
        paths["output"],
        paths["mask"],
    )

    with Image.open(paths["output"]) as image:
        np.testing.assert_array_equal(
            np.asarray(image), np.full((4, 4, 3), 130, dtype=np.uint8)
        )
    with Image.open(paths["mask"]) as image:
        np.testing.assert_array_equal(
            np.asarray(image), np.full((4, 4), 255, dtype=np.uint8)
        )

import pytest

from hand_process.fs import (
    atomic_copy_file,
    blend_weight,
    chunk_ranges,
    ensure_within,
)


def test_chunk_ranges_cover_without_gaps():
    ranges = chunk_ranges(725, size=300, overlap=30)
    assert ranges == [(0, 300), (270, 570), (540, 725)]
    covered = {index for start, end in ranges for index in range(start, end)}
    assert covered == set(range(725))


def test_chunk_ranges_validate_parameters():
    with pytest.raises(ValueError):
        chunk_ranges(10, size=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_ranges(10, size=10, overlap=0, min_size=11)


def test_chunk_ranges_extend_short_tail_to_minimum_size():
    assert chunk_ranges(80, size=66, overlap=0, min_size=22) == [
        (0, 66),
        (58, 80),
    ]
    assert chunk_ranges(70, size=66, overlap=11, min_size=22) == [
        (0, 66),
        (48, 70),
    ]


def test_blend_weight_prefers_chunk_center():
    assert blend_weight(10, 100, 20) > blend_weight(0, 100, 20)
    assert blend_weight(89, 100, 20) > blend_weight(99, 100, 20)


def test_ensure_within_rejects_escape(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    assert ensure_within(allowed / "child", allowed) == allowed / "child"
    with pytest.raises(ValueError):
        ensure_within(tmp_path / "outside", allowed)


def test_atomic_copy_file_publishes_complete_artifact(tmp_path):
    source = tmp_path / "scratch" / "artifact.mp4"
    destination = tmp_path / "results" / "artifact.mp4"
    source.parent.mkdir()
    source.write_bytes(b"complete-video")

    atomic_copy_file(source, destination)

    assert destination.read_bytes() == b"complete-video"
    assert not list(destination.parent.glob(".*.tmp-*"))

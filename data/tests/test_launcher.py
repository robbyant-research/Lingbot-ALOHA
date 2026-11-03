import pytest

from hand_process.launcher import (
    _base_config_path,
    _datasets_for_pairs,
    _prepare_path_pairs,
    _run_id,
    _run_id_for_pairs,
    _video_inventory,
)


def test_server_config_can_be_overridden_by_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HAND_PROCESS_SERVER_CONFIG", "configs/custom.yaml")
    assert _base_config_path(tmp_path, "server") == (
        tmp_path / "configs" / "custom.yaml"
    )


def test_server_config_uses_deployment_default(tmp_path, monkeypatch):
    monkeypatch.delenv("HAND_PROCESS_SERVER_CONFIG", raising=False)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    deployment_default = config_dir / "server_default.yaml"
    deployment_default.write_text("extends: custom.yaml\n", encoding="utf-8")
    assert _base_config_path(tmp_path, "server") == deployment_default


def test_server_config_falls_back_to_server(tmp_path, monkeypatch):
    monkeypatch.delenv("HAND_PROCESS_SERVER_CONFIG", raising=False)
    assert _base_config_path(tmp_path, "server") == (
        tmp_path / "configs" / "server.yaml"
    )


def test_edge_config_ignores_server_override(tmp_path, monkeypatch):
    monkeypatch.setenv("HAND_PROCESS_SERVER_CONFIG", "/tmp/server.yaml")
    assert _base_config_path(tmp_path, "edge") == (
        tmp_path / "configs" / "low_memory.yaml"
    )


def test_video_inventory_recurses_and_accepts_mp4_case_insensitively(tmp_path):
    (tmp_path / "episode_000000.mp4").write_bytes(b"video")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "episode_000001.MP4").write_bytes(b"nested")
    assert _video_inventory(tmp_path) == [
        tmp_path / "episode_000000.mp4",
        nested / "episode_000001.MP4",
    ]


def test_video_inventory_requires_at_least_one_video(tmp_path):
    with pytest.raises(ValueError, match="no \\*\\.mp4"):
        _video_inventory(tmp_path)


def test_path_pairs_require_an_even_number_of_paths():
    with pytest.raises(ValueError, match="ordered pairs"):
        _prepare_path_pairs(["a_in", "a_out", "b_in"])


def test_path_pairs_preserve_input_order_and_do_not_create_outputs(tmp_path):
    first_input = tmp_path / "a_in"
    second_input = tmp_path / "b_in"
    first_output = tmp_path / "a_out"
    second_output = tmp_path / "b_out"
    first_input.mkdir()
    second_input.mkdir()
    second_output.mkdir()
    (first_input / "a.mp4").write_bytes(b"a")
    (second_input / "b.mp4").write_bytes(b"b")

    pairs = _prepare_path_pairs(
        [
            str(first_input),
            str(first_output),
            str(second_input),
            str(second_output),
        ]
    )

    assert [pair[0] for pair in pairs] == [first_input, second_input]
    assert [pair[1] for pair in pairs] == [first_output, second_output]
    assert not first_output.exists()
    assert list(second_output.iterdir()) == []


def test_path_pairs_reject_nonempty_output_before_creating_any_output(tmp_path):
    first_input = tmp_path / "a_in"
    second_input = tmp_path / "b_in"
    first_output = tmp_path / "a_out"
    second_output = tmp_path / "b_out"
    first_input.mkdir()
    second_input.mkdir()
    second_output.mkdir()
    (first_input / "a.mp4").write_bytes(b"a")
    (second_input / "b.mp4").write_bytes(b"b")
    (second_output / "existing.txt").write_text("occupied", encoding="utf-8")

    with pytest.raises(ValueError, match="must not exist or must be empty"):
        _prepare_path_pairs(
            [
                str(first_input),
                str(first_output),
                str(second_input),
                str(second_output),
            ]
        )

    assert not first_output.exists()


def test_path_pairs_reject_output_inside_any_input_tree(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "video.mp4").write_bytes(b"video")

    with pytest.raises(ValueError, match="outside every input tree"):
        _prepare_path_pairs([str(input_dir), str(input_dir / "output")])


def test_datasets_for_pairs_keep_order_and_independent_outputs(tmp_path):
    pairs = [
        (tmp_path / "a_in", tmp_path / "a_out", []),
        (tmp_path / "b_in", tmp_path / "b_out", []),
    ]

    datasets = _datasets_for_pairs(pairs)

    assert list(datasets) == ["input_videos_001", "input_videos_002"]
    assert datasets["input_videos_001"]["input_order"] == 0
    assert datasets["input_videos_001"]["relative_path"] == "input_videos_001"
    assert datasets["input_videos_001"]["output_relative_path"] == "."
    assert datasets["input_videos_001"]["final_video_root"] == str(
        tmp_path / "a_out"
    )
    assert datasets["input_videos_002"]["input_order"] == 1
    assert datasets["input_videos_002"]["final_video_root"] == str(
        tmp_path / "b_out"
    )


def test_run_id_changes_when_video_inventory_changes(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    video = input_dir / "episode_000000.mp4"
    video.write_bytes(b"a")
    first = _run_id(
        "edge", input_dir, output_dir, _video_inventory(input_dir)
    )
    video.write_bytes(b"changed")
    second = _run_id(
        "edge", input_dir, output_dir, _video_inventory(input_dir)
    )
    assert first != second


def test_multi_pair_run_id_depends_on_pair_order(tmp_path):
    first_input = tmp_path / "a_in"
    second_input = tmp_path / "b_in"
    first_input.mkdir()
    second_input.mkdir()
    first_video = first_input / "a.mp4"
    second_video = second_input / "b.mp4"
    first_video.write_bytes(b"a")
    second_video.write_bytes(b"b")
    first = (
        first_input,
        tmp_path / "a_out",
        [first_video],
    )
    second = (
        second_input,
        tmp_path / "b_out",
        [second_video],
    )

    assert _run_id_for_pairs("server", [first, second]) != _run_id_for_pairs(
        "server", [second, first]
    )

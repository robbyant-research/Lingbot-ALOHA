from pathlib import Path

import pytest

from hand_process.datasets import (
    Episode,
    discover_episodes,
    select_identities,
    select_stage,
)


def _episode(dataset: str, index: int) -> Episode:
    return Episode(
        dataset=dataset,
        source=Path(f"/input/episode_{index:06d}.mp4"),
        relative_dir=Path(dataset),
    )


def test_smoke_selects_episode_zero_per_dataset():
    episodes = [_episode("a", 3), _episode("a", 0), _episode("b", 2)]
    assert [item.identity for item in select_stage(episodes, "smoke")] == [
        "a/episode_000000.mp4",
        "b/episode_000002.mp4",
    ]


def test_full_is_deterministically_sorted():
    episodes = [_episode("b", 3), _episode("a", 4), _episode("a", 1)]
    assert [item.identity for item in select_stage(episodes, "full")] == [
        "a/episode_000001.mp4",
        "a/episode_000004.mp4",
        "b/episode_000003.mp4",
    ]


def test_select_identities_is_exact_and_validated():
    episodes = [_episode("b", 3), _episode("a", 4), _episode("a", 1)]
    assert [
        item.identity
        for item in select_identities(
            episodes,
            ["b/episode_000003.mp4", "a/episode_000001.mp4"],
        )
    ] == ["a/episode_000001.mp4", "b/episode_000003.mp4"]
    with pytest.raises(ValueError, match="not found"):
        select_identities(episodes, ["a/episode_999999.mp4"])


def test_discover_episodes_skips_disabled_dataset(tmp_path):
    enabled = tmp_path / "enabled"
    disabled = tmp_path / "disabled"
    enabled.mkdir()
    disabled.mkdir()
    (enabled / "episode_000000.mp4").touch()
    (disabled / "episode_000000.mp4").touch()
    config = {
        "datasets": {
            "enabled": {
                "enabled": True,
                "input_dir": str(enabled),
                "relative_path": "enabled",
            },
            "disabled": {
                "enabled": False,
                "input_dir": str(disabled),
                "relative_path": "disabled",
            },
        }
    }
    episodes = discover_episodes(config)
    assert [episode.dataset for episode in episodes] == ["enabled"]


def test_discover_episodes_preserves_recursive_source_tree(tmp_path):
    input_dir = tmp_path / "videos"
    nested = input_dir / "task" / "camera"
    nested.mkdir(parents=True)
    source = nested / "clip.mp4"
    source.touch()
    config = {
        "datasets": {
            "input_videos": {
                "input_dir": str(input_dir),
                "relative_path": ".",
                "glob": "**/*.mp4",
                "preserve_relative_tree": True,
            }
        }
    }

    assert discover_episodes(config) == [
        Episode(
            dataset="input_videos",
            source=source,
            relative_dir=Path("task/camera"),
            source_relative_path=Path("task/camera/clip.mp4"),
        )
    ]
    assert discover_episodes(config)[0].identity == (
        "input_videos/task/camera/clip.mp4"
    )

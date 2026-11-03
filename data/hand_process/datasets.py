from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .video import probe_video


@dataclass(frozen=True)
class Episode:
    dataset: str
    source: Path
    relative_dir: Path
    source_relative_path: Path | None = None

    @property
    def stem(self) -> str:
        return self.source.stem

    @property
    def identity(self) -> str:
        relative = self.source_relative_path or Path(self.source.name)
        return f"{self.dataset}/{relative.as_posix()}"


def discover_episodes(config: dict) -> list[Episode]:
    episodes: list[Episode] = []
    for dataset_name, dataset_config in config["datasets"].items():
        if not dataset_config.get("enabled", True):
            continue
        input_dir = Path(dataset_config["input_dir"])
        pattern = dataset_config.get("glob", "episode_*.mp4")
        relative_root = Path(dataset_config["relative_path"])
        preserve_relative_tree = dataset_config.get(
            "preserve_relative_tree", False
        )
        if not isinstance(preserve_relative_tree, bool):
            raise ValueError(
                f"datasets.{dataset_name}.preserve_relative_tree must be "
                "true or false"
            )
        for source in sorted(input_dir.glob(pattern)):
            if source.is_file():
                source_relative_path = (
                    source.relative_to(input_dir)
                    if preserve_relative_tree
                    else None
                )
                relative_dir = (
                    relative_root / source_relative_path.parent
                    if source_relative_path is not None
                    else relative_root
                )
                episodes.append(
                    Episode(
                        dataset=dataset_name,
                        source=source,
                        relative_dir=relative_dir,
                        source_relative_path=source_relative_path,
                    )
                )
    return episodes


def _duration_seconds(episode: Episode) -> float:
    return probe_video(episode.source).duration_seconds


def select_stage(episodes: Iterable[Episode], stage: str) -> list[Episode]:
    grouped: dict[str, list[Episode]] = {}
    for episode in episodes:
        grouped.setdefault(episode.dataset, []).append(episode)

    if stage == "full":
        return sorted(
            (episode for values in grouped.values() for episode in values),
            key=lambda episode: episode.identity,
        )

    selected: list[Episode] = []
    for dataset in sorted(grouped):
        values = sorted(grouped[dataset], key=lambda episode: episode.source.name)
        if not values:
            continue
        if stage == "smoke":
            exact = next(
                (
                    episode
                    for episode in values
                    if episode.source.name == "episode_000000.mp4"
                ),
                values[0],
            )
            selected.append(exact)
            continue
        if stage == "calibration":
            by_duration = sorted(
                ((episode, _duration_seconds(episode)) for episode in values),
                key=lambda item: (item[1], item[0].source.name),
            )
            indexes = {
                round(percentile * (len(by_duration) - 1))
                for percentile in (0.0, 0.25, 0.5, 0.75, 1.0)
            }
            selected.extend(by_duration[index][0] for index in sorted(indexes))
            continue
        raise ValueError(f"unknown stage: {stage}")
    return selected


def select_identities(
    episodes: Iterable[Episode], identities: Iterable[str]
) -> list[Episode]:
    requested = set(identities)
    selected = sorted(
        (episode for episode in episodes if episode.identity in requested),
        key=lambda episode: episode.identity,
    )
    found = {episode.identity for episode in selected}
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"episode identities not found: {missing}")
    return selected

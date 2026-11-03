from pathlib import Path

import pytest

from hand_process.backends import (
    DetectorBackend,
    InpainterBackend,
    MovableInpainterBackend,
    SegmenterBackend,
    TrackerBackend,
    _resolve_backend,
    detector_config,
    segmenter_config,
)


class DummyDetector:
    def predict_many(self, paths, batch_size=None):
        return {}

    def to(self, device):
        pass


class DummyTracker:
    def track(self, frame_dir, prompts, output_dir):
        pass

    def to(self, device):
        pass


class DummyInpainter:
    def run_chunk(
        self, frame_dir, mask_dir, output_root, fps, log_name
    ):
        return Path(output_root)

    def to(self, device):
        pass


def test_backend_protocols_are_independently_replaceable():
    assert isinstance(DummyDetector(), DetectorBackend)
    assert isinstance(DummyTracker(), SegmenterBackend)
    assert isinstance(DummyTracker(), TrackerBackend)
    assert isinstance(DummyInpainter(), InpainterBackend)
    assert isinstance(DummyInpainter(), MovableInpainterBackend)


def test_backend_resolver_accepts_builtin_and_import_target():
    assert _resolve_backend("dummy", {"dummy": DummyDetector}) is DummyDetector
    assert _resolve_backend("pathlib:Path", {}) is Path


def test_backend_resolver_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown backend"):
        _resolve_backend("missing", {})


def test_stage_configs_prefer_stage_names_and_accept_aliases():
    current = {
        "models": {
            "detector": {"backend": "care_ego"},
            "segmenter": {"backend": "sam2"},
        }
    }
    assert detector_config(current)["backend"] == "care_ego"
    assert segmenter_config(current)["backend"] == "sam2"

    aliases = {
        "models": {
            "care_ego": {"backend": "care_ego"},
            "sam2": {"backend": "sam2"},
        }
    }
    assert detector_config(aliases)["backend"] == "care_ego"
    assert segmenter_config(aliases)["backend"] == "sam2"

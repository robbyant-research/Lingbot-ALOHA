import pytest

from hand_process.cli import _lanes_per_gpu


def test_lanes_per_gpu_defaults_to_one():
    assert _lanes_per_gpu({}, None) == 1


def test_lanes_per_gpu_honors_config_and_explicit_override():
    config = {"runtime": {"model_service": {"lanes_per_gpu": 1}}}
    assert _lanes_per_gpu(config, None) == 1
    assert _lanes_per_gpu(config, 2) == 2


def test_lanes_per_gpu_rejects_invalid_values():
    with pytest.raises(ValueError, match="at least 1"):
        _lanes_per_gpu({}, 0)

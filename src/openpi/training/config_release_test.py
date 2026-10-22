"""Regression checks for the Mobile ALOHA configuration release (2026-10-21)."""

import dataclasses
from unittest import mock

import numpy as np
import pytest

from openpi.training import config


@pytest.mark.parametrize("rtc", [False, True])
@pytest.mark.parametrize("history_length", [0, 1, 7])
def test_factory_and_data_pipeline(tmp_path, rtc, history_length):
    factory = config.make_pi05_mobile_aloha_rtc_config if rtc else config.make_pi05_mobile_aloha_config
    kwargs = dict(
        name="pipeline_test",
        repo_id="local/mobile_aloha",
        norm_name="test_stats",
        batch_size=4,
        num_steps=12,
        use_episodes=3,
        action_horizon=10,
        history_length=history_length,
    )
    if rtc:
        kwargs.update(rtc_max_delay=4, rtc_prefix_len=2, rtc_exec_steps=2, per_step_time_rtc=True)
    cfg = factory(**kwargs)
    assert cfg.batch_size == 4
    assert cfg.num_train_steps == 12
    assert cfg.model.action_horizon == 10
    assert cfg.model.pi05
    assert cfg.data.episodes == 3
    assert cfg.data.assets.asset_id == "test_stats"
    if rtc:
        assert cfg.model.use_rtc
        assert (cfg.model.rtc_max_delay, cfg.model.rtc_prefix_len, cfg.model.rtc_exec_steps) == (4, 2, 2)
        assert cfg.model.per_step_time_rtc

    data_factory = dataclasses.replace(cfg.data, assets=config.AssetsConfig(assets_dir=str(tmp_path), asset_id="missing"))
    # Tokenizer initialization can download an external asset. This check exercises the data transforms without it.
    with mock.patch.object(config._tokenizer, "PaligemmaTokenizer", return_value=object()):
        data_cfg = data_factory.create(tmp_path, cfg.model)
    assert data_cfg.use_quantile_norm
    assert data_cfg.prompt_from_task
    assert data_cfg.history_length == history_length
    assert data_cfg.action_sequence_keys == ("action.right", "action.left", "action.base")

    frames = cfg.model.action_horizon + history_length
    left = np.arange(frames * 7, dtype=np.float32).reshape(frames, 7)
    right = left + 100
    base = np.arange(frames * 3, dtype=np.float32).reshape(frames, 3)
    raw = {
        **{f"observation.image.{name}": np.ones((3, 8, 8), dtype=np.float32) for name in ("head", "left", "right")},
        **{f"observation.states.{name}": np.arange(10, dtype=np.float32) for name in ("left", "right")},
        "action.left": left,
        "action.right": right,
        "action.base": base,
        "prompt": "move the object",
    }
    value = raw
    for transform in (*data_cfg.repack_transforms.inputs, *data_cfg.data_transforms.inputs):
        value = transform(value)
    expected = np.concatenate([left[history_length:], right[history_length:], base[history_length:, [0, -1]]], axis=-1)
    np.testing.assert_array_equal(value["actions"], expected)
    assert value["actions"].shape == (10, 16)
    assert value["image"]["base_0_rgb"].shape == (8, 8, 3)
    assert value["prompt"] == raw["prompt"]
    expected_state_size = 20 + (2 * len(range(0, history_length, 6)) if history_length > 1 else 2)
    assert value["state"].shape == (expected_state_size,)
    for transform in data_cfg.data_transforms.outputs:
        value = transform({"actions": np.zeros((10, 32), dtype=np.float32)})
    assert value["actions"].shape == (10, 16)


def test_registry_scope():
    assert set(config._CONFIGS_DICT) == {"pi05_mobile_aloha_rtc"}
    assert len(config._CONFIGS) == 1
    assert config.get_config("pi05_mobile_aloha_rtc").model.use_rtc
    for cfg in config._CONFIGS:
        assert config.get_config(cfg.name) is cfg
        assert type(cfg.data) is config.LeRobotMobileAlohaDataConfig
    for name in ("pi0_aloha_sim", "pi05_droid", "debug", "whole_body_39_rtc", "lingbot_arx_x5_rtc_200_gripper"):
        with pytest.raises(ValueError, match="not found"):
            config.get_config(name)

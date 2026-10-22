import types

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi05_per_step_time_rtc_reaches_model():
    config = _pi0_config.Pi0Config(pi05=True, per_step_time_rtc=True)
    model = nnx.eval_shape(config.create, jax.random.key(0))
    assert model.per_step_time_rtc


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


class _Projection:
    out_features = 4

    def __call__(self, value):
        return jnp.zeros((*value.shape[:-1], self.out_features))


def test_pi05_per_step_time_rtc_controls_adarms_condition_shape():
    observation = _model.Observation(images={}, image_masks={}, state=jnp.zeros((2, 2)))
    actions = jnp.zeros((2, 3, 2))
    timestep = jnp.array([[0.0, 0.5, 0.75], [0.0, 0.25, 0.75]])

    common = {
        "pi05": True,
        "action_horizon": 3,
        "action_in_proj": _Projection(),
        "time_mlp_in": lambda value: value,
        "time_mlp_out": lambda value: value,
    }
    scalar_time_model = types.SimpleNamespace(**common, per_step_time_rtc=False)
    per_step_time_model = types.SimpleNamespace(**common, per_step_time_rtc=True)

    *_, scalar_condition = _pi0.Pi0.embed_suffix(scalar_time_model, observation, actions, timestep)
    *_, per_step_condition = _pi0.Pi0.embed_suffix(per_step_time_model, observation, actions, timestep)

    assert scalar_condition.shape == (2, 4)
    assert per_step_condition.shape == (2, 3, 4)
    assert not jnp.allclose(per_step_condition[:, 0], per_step_condition[:, 1])

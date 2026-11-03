import sys
import types

import pytest

from hand_process.diffusers_compat import enable_peft_adapter_loading


def test_enable_peft_adapter_loading_patches_legacy_model(monkeypatch):
    class FakePeftAdapterMixin:
        def load_lora_adapter(self):
            return "loaded"

    loaders = types.ModuleType("diffusers.loaders")
    loaders.PeftAdapterMixin = FakePeftAdapterMixin
    diffusers = types.ModuleType("diffusers")
    diffusers.loaders = loaders
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.loaders", loaders)

    class LegacyModel:
        def _optionally_disable_offloading(self):
            return False, False

    assert enable_peft_adapter_loading(LegacyModel) is True
    assert LegacyModel().load_lora_adapter() == "loaded"
    assert LegacyModel._hf_peft_config_loaded is False
    assert LegacyModel._prepare_lora_hotswap_kwargs is None
    assert enable_peft_adapter_loading(LegacyModel) is False


def test_enable_peft_adapter_loading_rejects_incompatible_model():
    class IncompatibleModel:
        pass

    with pytest.raises(TypeError, match="offloading hook"):
        enable_peft_adapter_loading(IncompatibleModel)

from __future__ import annotations

from collections.abc import Mapping
import inspect
from typing import Any


def disable_optional_xformers() -> None:
    """Use PyTorch SDPA without importing optional xformers extensions."""
    import diffusers.utils.import_utils as import_utils

    # Diffusers detects extensions through package metadata. Explicitly disable
    # this backend so an incompatible optional wheel cannot change inference.
    import_utils._xformers_available = False


def configure_attention_backend(
    models: Mapping[str, Any], backend: str
) -> dict[str, tuple[str, ...]]:
    """Configure and verify diffusion-model attention processors."""
    normalized = str(backend).strip().lower()
    if normalized != "sdpa":
        raise ValueError(f"unsupported attention backend {backend!r}; expected 'sdpa'")

    disable_optional_xformers()

    import torch.nn.functional as functional
    from diffusers.models.attention_processor import AttnProcessor2_0

    if not hasattr(functional, "scaled_dot_product_attention"):
        raise RuntimeError(
            "attention_backend='sdpa' requires "
            "torch.nn.functional.scaled_dot_product_attention"
        )

    configured: dict[str, tuple[str, ...]] = {}
    for name, model in models.items():
        configured_processors: list[Any] = []
        modules = getattr(model, "modules", None)
        if callable(modules):
            for module in modules():
                setter = getattr(module, "set_processor", None)
                if not callable(setter):
                    continue
                parameters = inspect.signature(setter).parameters
                if "_remove_lora" in parameters:
                    setter(AttnProcessor2_0(), _remove_lora=False)
                else:
                    setter(AttnProcessor2_0())
                configured_processors.append(getattr(module, "processor", None))
        if not configured_processors:
            raise TypeError(f"{name} does not expose attention modules")
        processor_types = tuple(
            sorted({type(processor).__name__ for processor in configured_processors})
        )
        if processor_types != ("AttnProcessor2_0",):
            raise RuntimeError(
                f"{name} did not switch entirely to PyTorch SDPA: {processor_types}"
            )
        configured[name] = processor_types
    return configured


def enable_peft_adapter_loading(model_class: type) -> bool:
    """Expose the current diffusers PEFT loader on an older custom UNet."""
    if hasattr(model_class, "load_lora_adapter"):
        return False
    if not hasattr(model_class, "_optionally_disable_offloading"):
        raise TypeError(
            f"{model_class.__name__} does not provide the diffusers "
            "offloading hook required by load_lora_adapter"
        )

    from diffusers.loaders import PeftAdapterMixin

    model_class.load_lora_adapter = PeftAdapterMixin.load_lora_adapter
    model_class._hf_peft_config_loaded = False
    model_class._prepare_lora_hotswap_kwargs = None
    return True

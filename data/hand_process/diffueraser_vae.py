from __future__ import annotations

import torch


def decode_vae_latents_in_batches(
    vae,
    latents: torch.Tensor,
    *,
    weight_dtype: torch.dtype,
    batch_size: int,
) -> torch.Tensor:
    """Decode image latents in bounded batches while preserving frame order."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise ValueError("VAE decode batch_size must be an integer")
    if batch_size < 1:
        raise ValueError("VAE decode batch_size must be at least 1")
    if latents.shape[0] < 1:
        raise ValueError("VAE decode requires at least one latent")

    scaled_latents = 1 / vae.config.scaling_factor * latents
    decoded = []
    for start in range(0, scaled_latents.shape[0], batch_size):
        batch = scaled_latents[start : start + batch_size].to(weight_dtype)
        decoded.append(vae.decode(batch).sample)
    return torch.cat(decoded, dim=0).float()

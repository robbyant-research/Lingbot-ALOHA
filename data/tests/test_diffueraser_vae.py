from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from hand_process.diffueraser_vae import decode_vae_latents_in_batches


def test_decode_vae_latents_in_batches_preserves_order_and_tail():
    torch = pytest.importorskip("torch")

    class FakeVAE:
        config = SimpleNamespace(scaling_factor=2.0)

        def __init__(self):
            self.batch_sizes = []

        def decode(self, latents):
            self.batch_sizes.append(latents.shape[0])
            return SimpleNamespace(sample=latents + 1)

    vae = FakeVAE()
    latents = torch.arange(10, dtype=torch.float16).reshape(10, 1, 1, 1)
    decoded = decode_vae_latents_in_batches(
        vae,
        latents,
        weight_dtype=torch.float16,
        batch_size=4,
    )

    assert vae.batch_sizes == [4, 4, 2]
    assert decoded.dtype == torch.float32
    torch.testing.assert_close(decoded, latents.float() / 2 + 1)


@pytest.mark.parametrize("batch_size", [0, True, 1.5])
def test_decode_vae_latents_in_batches_validates_batch_size(batch_size):
    torch = pytest.importorskip("torch")
    vae = SimpleNamespace(config=SimpleNamespace(scaling_factor=1.0))
    with pytest.raises(ValueError, match="batch_size"):
        decode_vae_latents_in_batches(
            vae,
            torch.zeros(1, 1, 1, 1),
            weight_dtype=torch.float32,
            batch_size=batch_size,
        )

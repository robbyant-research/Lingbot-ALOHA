from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")

from hand_process.diffueraser_batch import (
    gather_context_batch,
    group_contexts,
    split_context_batch,
)

from hand_process.diffueraser_runner import (
    PersistentDiffuEraserRunner,
    _prepare_pcm_runtime,
)

from hand_process.diffusers_compat import configure_attention_backend
from hand_process.frame_store import Hdf5VideoWriter, load_frame_store


def test_prepare_pcm_runtime_links_assets_without_mutating_source(tmp_path):
    pcm = tmp_path / "models" / "PCM_Weights"
    pcm.mkdir(parents=True)
    runtime = tmp_path / "runtime"

    assert (
        _prepare_pcm_runtime(
            pcm_weights_path=pcm,
            runtime_root=runtime,
        )
        == runtime
    )

    link = runtime / "weights" / "PCM_Weights"
    assert link.is_symlink()
    assert link.resolve() == pcm.resolve()


def test_prepare_pcm_runtime_rejects_conflicting_link(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    runtime = tmp_path / "runtime"
    (runtime / "weights").mkdir(parents=True)
    (runtime / "weights" / "PCM_Weights").symlink_to(first, target_is_directory=True)

    with pytest.raises(RuntimeError, match="targets"):
        _prepare_pcm_runtime(
            pcm_weights_path=second,
            runtime_root=runtime,
        )


def test_prepare_pcm_runtime_requires_existing_weights(tmp_path):
    with pytest.raises(FileNotFoundError, match="PCM weights"):
        _prepare_pcm_runtime(
            pcm_weights_path=Path(tmp_path / "missing"),
            runtime_root=tmp_path / "runtime",
        )


def test_restore_exact_background_updates_hdf5_in_place(tmp_path):
    pytest.importorskip("h5py")
    frame_path = tmp_path / "original.png"
    mask_path = tmp_path / "mask.png"
    store_path = tmp_path / "frames.h5"
    original = np.full((2, 2, 3), 10, dtype=np.uint8)
    generated_rgb = np.full((2, 2, 3), 200, dtype=np.uint8)
    mask = np.array([[255, 0], [0, 0]], dtype=np.uint8)
    Image.fromarray(original).save(frame_path)
    Image.fromarray(mask).save(mask_path)
    writer = Hdf5VideoWriter(
        store_path,
        expected_num_frames=1,
        fps=Fraction(24, 1),
        frame_size=(2, 2),
    )
    writer.write(generated_rgb[..., ::-1])
    writer.release()

    PersistentDiffuEraserRunner._restore_exact_background_hdf5(
        store_path, [frame_path], [mask_path]
    )

    expected = original.copy()
    expected[0, 0] = generated_rgb[0, 0]
    np.testing.assert_array_equal(load_frame_store(store_path)[0], expected)


def test_forward_to_hdf5_restores_upstream_cv2_after_failure(tmp_path):
    original_cv2 = SimpleNamespace(identity="original")
    upstream_module = SimpleNamespace(cv2=original_cv2)

    class FailingEngine:
        def forward(self, *_args, **_kwargs):
            assert upstream_module.cv2 is not original_cv2
            raise RuntimeError("inference failed")

    runner = object.__new__(PersistentDiffuEraserRunner)
    runner._diffueraser_module = upstream_module
    runner.engine = FailingEngine()
    runner.config = {}
    runner._request_seed = 42
    destination = tmp_path / "frames.h5"

    with pytest.raises(RuntimeError, match="inference failed"):
        runner._forward_to_hdf5(
            input_video=tmp_path / "input.mp4",
            mask_video=tmp_path / "mask.mp4",
            prior_video=tmp_path / "prior.mp4",
            result=destination,
            num_frames=1,
            width=2,
            height=2,
            fps=Fraction(24, 1),
            duration=1.0,
        )

    assert upstream_module.cv2 is original_cv2
    assert not destination.exists()


@pytest.mark.parametrize(
    ("batch_size", "expected"),
    [
        (1, [[0], [1], [2], [3], [4]]),
        (2, [[0, 1], [2, 3], [4]]),
        (4, [[0, 1, 2, 3], [4]]),
    ],
)
def test_group_contexts_preserves_order_and_partial_tail(batch_size, expected):
    assert group_contexts(list(range(5)), batch_size) == expected


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_group_contexts_rejects_invalid_batch_size(value):
    with pytest.raises(ValueError, match="window_batch_size"):
        group_contexts([0, 1], value)


def test_gather_and_split_context_batch_preserve_window_order():
    torch = pytest.importorskip("torch")
    values = torch.arange(8).reshape(8, 1)
    contexts = [[0, 1, 2], [5, 6, 7]]

    gathered = gather_context_batch(values, contexts)
    assert gathered[:, 0].tolist() == [0, 1, 2, 5, 6, 7]
    split = split_context_batch(gathered, group_size=2, num_frames=3)
    assert [item[:, 0].tolist() for item in split] == [
        [0, 1, 2],
        [5, 6, 7],
    ]


def test_context_batch_helpers_validate_shape():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="equal length"):
        gather_context_batch(torch.arange(4), [[0, 1], [2]])
    with pytest.raises(ValueError, match="expected 6"):
        split_context_batch(torch.arange(5), group_size=2, num_frames=3)


def test_runner_window_batch_size_defaults_to_one_and_validates():
    runner = object.__new__(PersistentDiffuEraserRunner)
    runner.config = {}
    assert runner._window_batch_size() == 1

    runner.config = {"window_batch_size": 4}
    assert runner._window_batch_size() == 4

    runner.config = {"window_batch_size": 0}
    with pytest.raises(ValueError, match="window_batch_size"):
        runner._window_batch_size()


def test_runner_vae_decode_batch_size_defaults_to_one_and_validates():
    runner = object.__new__(PersistentDiffuEraserRunner)
    runner.config = {}
    assert runner._vae_decode_batch_size() == 1

    runner.config = {"vae_decode_batch_size": 8}
    assert runner._vae_decode_batch_size() == 8

    runner.config = {"vae_decode_batch_size": 0}
    with pytest.raises(ValueError, match="vae_decode_batch_size"):
        runner._vae_decode_batch_size()


def test_runner_skip_redundant_pre_inference_decode_validates_boolean():
    runner = object.__new__(PersistentDiffuEraserRunner)
    runner.config = {}
    assert runner._skip_redundant_pre_inference_decode() is True

    runner.config = {"skip_redundant_pre_inference_decode": False}
    assert runner._skip_redundant_pre_inference_decode() is False

    runner.config = {"skip_redundant_pre_inference_decode": 1}
    with pytest.raises(
        ValueError, match="skip_redundant_pre_inference_decode"
    ):
        runner._skip_redundant_pre_inference_decode()


def test_configure_attention_backend_uses_sdpa():
    pytest.importorskip("diffusers")
    class Attention:
        def __init__(self):
            self.processor = object()

        def set_processor(self, processor):
            self.processor = processor

    class Model:
        def __init__(self):
            self.attention = Attention()

        @property
        def attn_processors(self):
            return {"attention.processor": self.attention.processor}

        def modules(self):
            return [self, self.attention]

    configured = configure_attention_backend({"unet": Model()}, "sdpa")

    assert configured == {"unet": ("AttnProcessor2_0",)}


def test_configure_attention_backend_rejects_unknown_backend():
    with pytest.raises(ValueError, match="expected 'sdpa'"):
        configure_attention_backend({}, "xformers")

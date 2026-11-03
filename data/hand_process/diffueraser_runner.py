from __future__ import annotations

import contextlib
import os
import shutil
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from .frame_store import OpenCVHdf5Proxy, is_hdf5_frame_store
from .fs import recreate_directory, remove_directory
from .h5_utils import require_h5py
from .propainter import PersistentProPainterRunner
from .video import (
    decode_video_to_pngs,
    encode_mask_pngs,
    encode_rgb_frames,
    frame_paths,
    iter_png_rgb,
)


@contextlib.contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _prepare_pcm_runtime(*, pcm_weights_path: Path, runtime_root: Path) -> Path:
    if not pcm_weights_path.is_dir():
        raise FileNotFoundError(
            f"DiffuEraser PCM weights are missing: {pcm_weights_path}"
        )
    weights_root = runtime_root / "weights"
    weights_root.mkdir(parents=True, exist_ok=True)
    link = weights_root / "PCM_Weights"
    if link.is_symlink():
        if link.resolve() != pcm_weights_path.resolve():
            raise RuntimeError(
                f"DiffuEraser PCM runtime link targets {link.resolve()}, "
                f"expected {pcm_weights_path.resolve()}"
            )
    elif link.exists():
        raise RuntimeError(f"DiffuEraser PCM runtime path is not a symlink: {link}")
    else:
        try:
            link.symlink_to(pcm_weights_path.resolve(), target_is_directory=True)
        except FileExistsError:
            # Multiple persistent GPU workers can initialize the shared
            # runtime concurrently. Accept only the identical link created by
            # a sibling; every other collision remains an error.
            if not link.is_symlink() or link.resolve() != pcm_weights_path.resolve():
                raise
    return runtime_root


class PersistentDiffuEraserRunner:
    """Persistent DiffuEraser refinement over a ProPainter prior."""

    def __init__(
        self,
        config: dict,
        log_root: str | Path,
        device: str = "cuda:0",
    ) -> None:
        self.config = config
        self.repo = Path(config["repo"])
        self.log_root = Path(log_root)
        self.device = device
        self.last_profile: dict[str, float] = {}
        source = self.repo / "diffueraser" / "diffueraser.py"
        if not source.is_file():
            raise FileNotFoundError(
                f"DiffuEraser source checkout is incomplete: {self.repo}"
            )
        repo_string = str(self.repo)
        if repo_string not in sys.path:
            sys.path.insert(0, repo_string)
        if bool(config.get("disable_xformers", True)):
            from .diffusers_compat import disable_optional_xformers

            disable_optional_xformers()
        pcm_root = Path(config["pcm_weights_path"])
        runtime_root = _prepare_pcm_runtime(
            pcm_weights_path=pcm_root,
            runtime_root=self.log_root / "diffueraser-runtime",
        )
        from libs.unet_motion_model import UNetMotionModel

        from .diffusers_compat import enable_peft_adapter_loading

        # Diffusers 0.34 delegates LoRA loading to PeftAdapterMixin. Expose
        # that loader on the custom motion UNet before loading PCM weights.
        enable_peft_adapter_loading(UNetMotionModel)
        from diffueraser.diffueraser import DiffuEraser
        from diffueraser import diffueraser as diffueraser_module

        # Upstream resolves weights/PCM_Weights relative to cwd. Keep its
        # pinned source tree immutable and expose the configured asset through
        # an isolated runtime link instead.
        with _working_directory(runtime_root):
            self.engine = DiffuEraser(
                device,
                str(Path(config["base_model_path"])),
                str(Path(config["vae_path"])),
                str(Path(config["diffueraser_path"])),
                revision=config.get("revision"),
                ckpt=str(config.get("checkpoint_variant", "2-Step")),
                mode=str(config.get("pcm_mode", "sd15")),
            )
        self._diffueraser_module = diffueraser_module
        from .diffusers_compat import configure_attention_backend

        self.attention_processors = configure_attention_backend(
            {
                "unet_main": self.engine.unet_main,
                "brushnet": self.engine.brushnet,
            },
            str(config.get("attention_backend", "sdpa")),
        )
        prior_config = dict(config["prior"])
        self.prior = PersistentProPainterRunner(
            prior_config,
            self.log_root / "diffueraser-prior",
            device=device,
        )
        self._request_seed = 42

    def _window_batch_size(self) -> int:
        value = self.config.get("window_batch_size", 1)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("DiffuEraser window_batch_size must be an integer")
        if value < 1:
            raise ValueError("DiffuEraser window_batch_size must be at least 1")
        return value

    def _vae_decode_batch_size(self) -> int:
        value = self.config.get("vae_decode_batch_size", 1)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                "DiffuEraser vae_decode_batch_size must be an integer"
            )
        if value < 1:
            raise ValueError(
                "DiffuEraser vae_decode_batch_size must be at least 1"
            )
        return value

    def _skip_redundant_pre_inference_decode(self) -> bool:
        value = self.config.get(
            "skip_redundant_pre_inference_decode", True
        )
        if not isinstance(value, bool):
            raise ValueError(
                "DiffuEraser skip_redundant_pre_inference_decode must be "
                "a boolean"
            )
        return value

    def _detailed_profile(self) -> bool:
        return bool(self.config.get("detailed_profile", False))

    def set_request_context(self, *, global_start: int, seed: int) -> None:
        self._request_seed = int(seed) + int(global_start)
        self.prior.set_request_context(global_start=global_start, seed=seed)

    def to(self, device: str) -> None:
        self.prior.to(device)
        self.engine.pipeline.to(device)
        self.engine.device = device
        self.device = device

    @staticmethod
    def _input_geometry(
        frame_root: Path, mask_root: Path
    ) -> tuple[list[Path], list[Path], int, int]:
        frames = frame_paths(frame_root)
        masks = frame_paths(mask_root)
        if not frames:
            raise ValueError(f"no PNG frames in {frame_root}")
        if len(frames) != len(masks):
            raise ValueError(
                f"DiffuEraser frame/mask count mismatch: {len(frames)} != {len(masks)}"
            )
        with Image.open(frames[0]) as image:
            width, height = image.size
        return frames, masks, width, height

    @staticmethod
    def _restore_exact_background(
        generated_dir: Path,
        frames: list[Path],
        masks: list[Path],
    ) -> None:
        generated = frame_paths(generated_dir)
        if len(generated) != len(frames):
            raise RuntimeError(
                "DiffuEraser output frame count mismatch: "
                f"{len(generated)} != {len(frames)}"
            )
        for generated_path, frame_path, mask_path in zip(generated, frames, masks):
            with Image.open(generated_path) as image:
                result = np.asarray(image.convert("RGB"))
            with Image.open(frame_path) as image:
                original = np.asarray(image.convert("RGB"))
            with Image.open(mask_path) as image:
                mask = np.asarray(image.convert("L")) >= 128
            if result.shape != original.shape:
                result = np.asarray(
                    Image.fromarray(result).resize(
                        (original.shape[1], original.shape[0]),
                        Image.Resampling.BICUBIC,
                    )
                )
            final = np.where(mask[..., None], result, original)
            Image.fromarray(final.astype(np.uint8)).save(
                generated_path, format="PNG", compress_level=0
            )

    @staticmethod
    def _restore_exact_background_hdf5(
        generated_store: Path,
        frames: list[Path],
        masks: list[Path],
    ) -> None:
        h5py = require_h5py()
        with h5py.File(generated_store, "r+") as handle:
            generated = handle["frames"]
            if generated.shape[0] != len(frames):
                raise RuntimeError(
                    "DiffuEraser output frame count mismatch: "
                    f"{generated.shape[0]} != {len(frames)}"
                )
            for index, (frame_path, mask_path) in enumerate(zip(frames, masks)):
                result = np.asarray(generated[index], dtype=np.uint8)
                with Image.open(frame_path) as image:
                    original = np.asarray(image.convert("RGB"))
                with Image.open(mask_path) as image:
                    mask = np.asarray(image.convert("L")) >= 128
                if result.shape != original.shape:
                    result = np.asarray(
                        Image.fromarray(result).resize(
                            (original.shape[1], original.shape[0]),
                            Image.Resampling.BICUBIC,
                        )
                    )
                generated[index] = np.where(mask[..., None], result, original)
            handle.flush()

    def _forward_to_hdf5(
        self,
        *,
        input_video: Path,
        mask_video: Path,
        prior_video: Path,
        result: Path,
        num_frames: int,
        width: int,
        height: int,
        fps: Fraction,
        duration: float,
    ) -> None:
        original_cv2 = self._diffueraser_module.cv2
        proxy = OpenCVHdf5Proxy(
            original_cv2,
            result,
            expected_num_frames=num_frames,
            fps=fps,
            frame_size=(width, height),
        )
        self._diffueraser_module.cv2 = proxy
        try:
            self.engine.forward(
                str(input_video),
                str(mask_video),
                str(prior_video),
                str(result.with_suffix(".mp4")),
                max_img_size=int(self.config.get("max_img_size", 960)),
                video_length=duration,
                mask_dilation_iter=int(self.config.get("mask_dilation_iter", 1)),
                nframes=int(self.config.get("nframes", 22)),
                seed=self._request_seed,
                guidance_scale=self.config.get("guidance_scale"),
                blended=bool(self.config.get("blended", True)),
                window_batch_size=self._window_batch_size(),
                vae_decode_batch_size=self._vae_decode_batch_size(),
                skip_redundant_pre_inference_decode=(
                    self._skip_redundant_pre_inference_decode()
                ),
                detailed_profile=self._detailed_profile(),
            )
            proxy.require_completed()
        except Exception:
            proxy.abort()
            raise
        finally:
            self._diffueraser_module.cv2 = original_cv2

    def run_chunk(
        self,
        frame_dir: str | Path,
        mask_dir: str | Path,
        output_root: str | Path,
        fps: Fraction,
        log_name: str,
    ) -> Path:
        frame_root = Path(frame_dir)
        mask_root = Path(mask_dir)
        output = Path(output_root)
        frames, masks, width, height = self._input_geometry(frame_root, mask_root)
        if len(frames) < 22:
            raise ValueError("DiffuEraser requires at least 22 frames per request")
        work_parent = output / ".diffueraser-work"
        work_parent.mkdir(parents=True, exist_ok=True)
        safe_log_name = "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in log_name
        )
        work = recreate_directory(work_parent / safe_log_name, work_parent)
        output_storage = str(self.config.get("output_storage", "png"))
        if output_storage not in {"png", "hdf5"}:
            raise ValueError("DiffuEraser output_storage must be 'png' or 'hdf5'")
        result = (
            output / f"{frame_root.name}.h5"
            if output_storage == "hdf5"
            else output / frame_root.name / "frames"
        )
        try:
            encode_started = time.perf_counter()
            input_video = work / "input.mp4"
            mask_video = work / "mask.mp4"
            encode_rgb_frames(
                iter_png_rgb(frame_root),
                input_video,
                fps=fps,
                width=width,
                height=height,
                crf=1,
                preset="fast",
            )
            encode_mask_pngs(
                mask_root,
                mask_video,
                fps=fps,
                width=width,
                height=height,
                crf=1,
            )
            encode_seconds = time.perf_counter() - encode_started

            prior_started = time.perf_counter()
            prior_frames = self.prior.run_chunk(
                frame_root,
                mask_root,
                work / "prior-frames",
                fps,
                f"{safe_log_name}-prior",
            )
            prior_video = work / "prior.mp4"
            encode_rgb_frames(
                iter_png_rgb(prior_frames),
                prior_video,
                fps=fps,
                width=width,
                height=height,
                crf=1,
                preset="fast",
            )
            prior_seconds = time.perf_counter() - prior_started

            if bool(self.config.get("retain_intermediates", False)):
                retained = output / "intermediates" / safe_log_name
                retained.mkdir(parents=True, exist_ok=True)
                shutil.copy2(input_video, retained / "input.mp4")
                shutil.copy2(mask_video, retained / "mask.mp4")
                shutil.copy2(prior_video, retained / "propainter-prior.mp4")

            inference_started = time.perf_counter()
            output_video = work / "diffueraser.mp4"
            duration = (len(frames) + 0.5) / float(fps)
            if output_storage == "hdf5":
                result.unlink(missing_ok=True)
                self._forward_to_hdf5(
                    input_video=input_video,
                    mask_video=mask_video,
                    prior_video=prior_video,
                    result=result,
                    num_frames=len(frames),
                    width=width,
                    height=height,
                    fps=fps,
                    duration=duration,
                )
            else:
                self.engine.forward(
                    str(input_video),
                    str(mask_video),
                    str(prior_video),
                    str(output_video),
                    max_img_size=int(self.config.get("max_img_size", 960)),
                    video_length=duration,
                    mask_dilation_iter=int(self.config.get("mask_dilation_iter", 1)),
                    nframes=int(self.config.get("nframes", 22)),
                    seed=self._request_seed,
                    guidance_scale=self.config.get("guidance_scale"),
                    blended=bool(self.config.get("blended", True)),
                    window_batch_size=self._window_batch_size(),
                    vae_decode_batch_size=self._vae_decode_batch_size(),
                    skip_redundant_pre_inference_decode=(
                        self._skip_redundant_pre_inference_decode()
                    ),
                    detailed_profile=self._detailed_profile(),
                )
            inference_seconds = time.perf_counter() - inference_started

            decode_started = time.perf_counter()
            if is_hdf5_frame_store(result):
                self._restore_exact_background_hdf5(result, frames, masks)
            else:
                if result.exists():
                    remove_directory(result, output)
                decode_video_to_pngs(output_video, result)
                self._restore_exact_background(result, frames, masks)
            decode_seconds = time.perf_counter() - decode_started
        finally:
            remove_directory(work, work_parent)

        self.log_root.mkdir(parents=True, exist_ok=True)
        (self.log_root / f"{safe_log_name}.log").write_text(
            f"persistent DiffuEraser request: {len(frames)} frames\n"
            f"checkpoint_variant: "
            f"{self.config.get('checkpoint_variant', '2-Step')}\n"
            f"attention_backend: "
            f"{self.config.get('attention_backend', 'sdpa')}\n"
            f"attention_processors: {self.attention_processors}\n"
            f"seed: {self._request_seed}\n"
            f"output_storage: {output_storage}\n"
            f"window_batch_size: {self._window_batch_size()}\n"
            f"vae_decode_batch_size: {self._vae_decode_batch_size()}\n"
            "skip_redundant_pre_inference_decode: "
            f"{self._skip_redundant_pre_inference_decode()}\n",
            encoding="utf-8",
        )
        self.last_profile = {
            "encode_inputs": encode_seconds,
            "propainter_prior": prior_seconds,
            "diffueraser_inference": inference_seconds,
            "decode_outputs": decode_seconds,
        }
        if self._detailed_profile():
            detailed = getattr(self.engine, "last_detailed_profile", {})
            self.last_profile.update(
                {
                    f"diffueraser_detail_{name}": float(seconds)
                    for name, seconds in detailed.items()
                }
            )
        window_stats = getattr(self.engine.pipeline, "last_window_batch_stats", {})
        if window_stats:
            with (self.log_root / f"{safe_log_name}.log").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(
                    "window_batch_stats: "
                    + ", ".join(
                        f"{name}={value}"
                        for name, value in sorted(window_stats.items())
                    )
                    + "\n"
                )
        return result

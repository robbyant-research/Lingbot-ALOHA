from __future__ import annotations

import os
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _mask_dilation(config: dict) -> int:
    dilation = int(config.get("mask_dilation", 0))
    if dilation < 0:
        raise ValueError("ProPainter mask_dilation must be >= 0")
    return dilation


class ProPainterRunner:
    """Compatibility runner that starts the upstream CLI for every chunk."""

    def __init__(self, config: dict, log_root: str | Path) -> None:
        self.config = config
        self.repo = Path(config["repo"])
        self.weights = Path(config["weights"])
        self.log_root = Path(log_root)
        self.last_profile: dict[str, float] = {}
        for name in (
            "ProPainter.pth",
            "raft-things.pth",
            "recurrent_flow_completion.pth",
        ):
            target = self.repo / "weights" / name
            if not target.exists():
                raise FileNotFoundError(
                    f"ProPainter weight link is missing or broken: {target}"
                )

    def run_chunk(
        self,
        frame_dir: str | Path,
        mask_dir: str | Path,
        output_root: str | Path,
        fps: Fraction,
        log_name: str,
    ) -> Path:
        started = time.perf_counter()
        frame_root = Path(frame_dir)
        output = Path(output_root)
        output.mkdir(parents=True, exist_ok=True)
        self.log_root.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(self.repo / "inference_propainter.py"),
            "--video",
            str(frame_root),
            "--mask",
            str(mask_dir),
            "--output",
            str(output),
            "--mask_dilation",
            str(_mask_dilation(self.config)),
            "--ref_stride",
            str(self.config["ref_stride"]),
            "--neighbor_length",
            str(self.config["neighbor_length"]),
            "--subvideo_length",
            str(self.config["subvideo_length"]),
            "--raft_iter",
            str(self.config["raft_iter"]),
            "--save_fps",
            str(round(float(fps))),
            "--save_frames",
        ]
        if self.config.get("fp16", True):
            command.append("--fp16")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = (
            str(self.repo)
            + os.pathsep
            + environment.get("PYTHONPATH", "")
        )
        log_path = self.log_root / f"{log_name}.log"
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(
                command,
                cwd=self.repo,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                text=True,
            )
        result = output / frame_root.name / "frames"
        if not result.is_dir() or not any(result.glob("*.png")):
            raise RuntimeError(
                f"ProPainter produced no output frames; see {log_path}"
            )
        self.last_profile = {
            "subprocess": time.perf_counter() - started,
        }
        return result


class PersistentProPainterRunner:
    """Keep all three ProPainter networks loaded across requests."""

    def __init__(
        self,
        config: dict,
        log_root: str | Path,
        device: str = "cuda:0",
    ) -> None:
        self.config = config
        self.repo = Path(config["repo"])
        self.weights = Path(config["weights"])
        self.log_root = Path(log_root)
        self.device = device
        self.last_profile: dict[str, float] = {}
        self._validate_weights()
        repo_string = str(self.repo)
        demo_string = str(self.repo / "web-demos" / "hugging_face")
        for source_path in (repo_string, demo_string):
            if source_path not in sys.path:
                sys.path.insert(0, source_path)
        from inpainter.base_inpainter import ProInpainter

        self.engine = ProInpainter(
            str(self.weights / "ProPainter.pth"),
            str(self.weights / "raft-things.pth"),
            str(self.weights / "recurrent_flow_completion.pth"),
            device=device,
            use_half=bool(config.get("fp16", True)),
        )
        if config.get("torch_compile", False):
            import torch

            compile_mode = str(
                config.get("torch_compile_mode", "reduce-overhead")
            )
            self.engine.fix_raft = torch.compile(
                self.engine.fix_raft, mode=compile_mode, dynamic=True
            )
            self.engine.fix_flow_complete = torch.compile(
                self.engine.fix_flow_complete,
                mode=compile_mode,
                dynamic=True,
            )
            self.engine.model = torch.compile(
                self.engine.model, mode=compile_mode, dynamic=True
            )

    def set_request_context(self, *, global_start: int, seed: int) -> None:
        import torch

        request_seed = int(seed) + int(global_start)
        torch.manual_seed(request_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(request_seed)

    def _validate_weights(self) -> None:
        for name in (
            "ProPainter.pth",
            "raft-things.pth",
            "recurrent_flow_completion.pth",
        ):
            target = self.weights / name
            if not target.exists():
                raise FileNotFoundError(
                    f"ProPainter weight is missing: {target}"
                )

    def to(self, device: str) -> None:
        import torch

        target = torch.device(device)
        self.engine.fix_raft.to(target)
        self.engine.fix_flow_complete.to(target)
        self.engine.model.to(target)
        self.engine.device = target
        self.device = str(target)

    @staticmethod
    def _read_frames(frame_root: Path) -> list[np.ndarray]:
        frames = []
        for path in sorted(frame_root.glob("*.png")):
            with Image.open(path) as image:
                frames.append(np.asarray(image.convert("RGB")))
        if not frames:
            raise ValueError(f"no PNG frames in {frame_root}")
        return frames

    @staticmethod
    def _read_masks(mask_root: Path) -> list[np.ndarray]:
        masks = []
        for path in sorted(mask_root.glob("*.png")):
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"cannot read mask: {path}")
            masks.append(mask)
        if not masks:
            raise ValueError(f"no PNG masks in {mask_root}")
        return masks

    def run_chunk(
        self,
        frame_dir: str | Path,
        mask_dir: str | Path,
        output_root: str | Path,
        fps: Fraction,
        log_name: str,
    ) -> Path:
        del fps
        frame_root = Path(frame_dir)
        mask_root = Path(mask_dir)
        read_started = time.perf_counter()
        frames = self._read_frames(frame_root)
        masks = self._read_masks(mask_root)
        if len(frames) != len(masks):
            raise RuntimeError(
                f"ProPainter frame/mask count mismatch: "
                f"{len(frames)} != {len(masks)}"
            )
        read_seconds = time.perf_counter() - read_started
        self.log_root.mkdir(parents=True, exist_ok=True)
        log_path = self.log_root / f"{log_name}.log"
        log_path.write_text(
            f"persistent ProPainter request: {len(frames)} frames\n",
            encoding="utf-8",
        )
        inference_started = time.perf_counter()
        generated = self.engine.inpaint(
            np.asarray(frames),
            masks,
            ratio=1.0,
            dilate_radius=_mask_dilation(self.config),
            raft_iter=int(self.config["raft_iter"]),
            subvideo_length=int(self.config["subvideo_length"]),
            neighbor_length=int(self.config["neighbor_length"]),
            ref_stride=int(self.config["ref_stride"]),
        )
        inference_seconds = time.perf_counter() - inference_started
        if self.config.get("release_cache_after_request", False):
            import gc
            import torch

            gc.collect()
            torch.cuda.empty_cache()
        result = Path(output_root) / frame_root.name / "frames"
        result.mkdir(parents=True, exist_ok=True)
        write_started = time.perf_counter()
        for index, frame in enumerate(generated):
            Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(
                result / f"{index:04d}.png",
                format="PNG",
                compress_level=0,
            )
        self.last_profile = {
            "read_inputs": read_seconds,
            "inference": inference_seconds,
            "write_outputs": time.perf_counter() - write_started,
        }
        return result

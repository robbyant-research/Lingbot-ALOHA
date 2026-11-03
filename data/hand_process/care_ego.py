from __future__ import annotations

import copy
import importlib.machinery
import math
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PROMPT_MODES = frozenset(
    {
        "per_class_argmax",
        "union_hand_probability",
    }
)


def _validate_prompt_settings(
    confidence_threshold: float, prompt_mode: str
) -> tuple[float, str]:
    try:
        threshold = float(confidence_threshold)
    except (TypeError, ValueError) as error:
        raise ValueError("confidence_threshold must be a finite number") from error
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if not isinstance(prompt_mode, str) or prompt_mode not in PROMPT_MODES:
        raise ValueError(
            f"care_ego.prompt_mode must be one of {sorted(PROMPT_MODES)}, "
            f"got {prompt_mode!r}"
        )
    return threshold, prompt_mode


def build_hand_prompts(
    hand_probability: np.ndarray,
    confidence_threshold: float,
    prompt_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert three-class CaRe-Ego probabilities into SAM prompt maps."""

    if not isinstance(hand_probability, np.ndarray):
        raise TypeError("hand_probability must be a numpy.ndarray")
    if hand_probability.ndim != 3 or hand_probability.shape[0] != 3:
        raise ValueError("hand_probability must have shape (3, height, width)")
    if not np.issubdtype(hand_probability.dtype, np.floating):
        raise TypeError("hand_probability must use a floating-point dtype")
    if not np.all(np.isfinite(hand_probability)):
        raise ValueError("hand_probability must contain only finite values")
    if np.any(hand_probability < 0.0) or np.any(hand_probability > 1.0):
        raise ValueError("hand_probability values must be in [0, 1]")
    threshold, prompt_mode = _validate_prompt_settings(
        confidence_threshold, prompt_mode
    )

    if prompt_mode == "per_class_argmax":
        labels = hand_probability.argmax(axis=0)
        left = np.where(
            (labels == 1) & (hand_probability[1] >= threshold),
            hand_probability[1],
            0.0,
        )
        right = np.where(
            (labels == 2) & (hand_probability[2] >= threshold),
            hand_probability[2],
            0.0,
        )
        return left, right

    union = hand_probability[1] + hand_probability[2]
    union = np.where(union >= threshold, union, 0.0)
    return union, np.zeros_like(union)


def _install_unused_mmcv_ops_stub() -> None:
    """Allow pure-PyTorch MMSeg models when the optional MMCV ABI is stale.

    MMCV custom extensions may be unavailable in a pure-Python installation.
    CaRe-Ego does not call MMCV custom operators, but MMSeg imports their
    wrappers during registry initialization. The stub makes those wrappers
    importable and raises immediately if an operator is unexpectedly called.
    """

    try:
        __import__("mmcv._ext")
        return
    except ImportError:
        sys.modules.pop("mmcv._ext", None)

    extension = types.ModuleType("mmcv._ext")
    extension.__file__ = "<hand-process-unused-mmcv-ops>"
    extension.__spec__ = importlib.machinery.ModuleSpec("mmcv._ext", loader=None)

    def missing_operator(name: str):
        if name.startswith("__"):
            raise AttributeError(name)

        def unavailable(*_args, **_kwargs):
            raise RuntimeError(
                f"MMCV operator {name!r} is unavailable; CaRe-Ego should not "
                "execute MMCV custom operators"
            )

        return unavailable

    extension.__getattr__ = missing_operator
    sys.modules["mmcv._ext"] = extension


def _install_unused_mmseg_import_stubs() -> None:
    """Keep MMSeg model registration from importing unrelated frameworks."""

    if "mmengine.runner" not in sys.modules:
        runner = types.ModuleType("mmengine.runner")
        runner.__file__ = "<hand-process-unused-mmengine-runner>"
        runner.__path__ = []
        runner.__spec__ = importlib.machinery.ModuleSpec(
            "mmengine.runner", loader=None, is_package=True
        )
        checkpoint = types.ModuleType("mmengine.runner.checkpoint")
        checkpoint.__file__ = "<hand-process-unused-mmengine-checkpoint>"
        checkpoint.__spec__ = importlib.machinery.ModuleSpec(
            "mmengine.runner.checkpoint", loader=None
        )

        def unavailable(*_args, **_kwargs):
            raise RuntimeError(
                "MMEngine checkpoint helpers are unavailable in CaRe-Ego "
                "inference; the pinned state dict is loaded directly"
            )

        class UnusedCheckpointLoader:
            load_checkpoint = staticmethod(unavailable)

        for module in (runner, checkpoint):
            module.CheckpointLoader = UnusedCheckpointLoader
            module.load_checkpoint = unavailable
        checkpoint.load_state_dict = unavailable
        checkpoint._load_checkpoint = unavailable
        runner.checkpoint = checkpoint
        sys.modules["mmengine.runner"] = runner
        sys.modules["mmengine.runner.checkpoint"] = checkpoint

    # MMSeg imports the optional latent-diffusion VPD backbone as part of its
    # global model package. CaRe-Ego only needs SwinTransformer; importing VPD
    # pulls TensorFlow and other GPU runtimes into this process.
    if "mmseg.models.backbones.vpd" not in sys.modules:
        vpd = types.ModuleType("mmseg.models.backbones.vpd")
        vpd.__file__ = "<hand-process-unused-vpd>"
        vpd.__spec__ = importlib.machinery.ModuleSpec(
            "mmseg.models.backbones.vpd", loader=None
        )
        vpd.VPD = type("UnusedVPD", (), {})
        sys.modules["mmseg.models.backbones.vpd"] = vpd


class CareEgoSegmenter:
    """Project-local CaRe-Ego inference without modifying system MMSeg."""

    def __init__(self, config: dict, device: str = "cuda:0") -> None:
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
        _install_unused_mmcv_ops_stub()
        _install_unused_mmseg_import_stubs()
        import torch
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
        from mmseg.registry import MODELS

        self.torch = torch
        self.device = torch.device(device)
        self.repo = Path(config["repo"])
        self.input_height, self.input_width = map(int, config["input_size"])
        self.confidence_threshold, self.prompt_mode = _validate_prompt_settings(
            config["confidence_threshold"],
            config.get("prompt_mode", "per_class_argmax"),
        )
        self.batch_size = int(config.get("batch_size", 4))
        if self.batch_size < 1:
            raise ValueError("care_ego.batch_size must be at least 1")

        repo_string = str(self.repo)
        if repo_string not in sys.path:
            sys.path.insert(0, repo_string)
        # Register only model components. MMSeg's broad
        # register_all_modules() imports datasets, evaluators and unrelated
        # frameworks that this inference path does not use.
        __import__("mmseg.models")
        init_default_scope("mmseg")
        __import__("models")

        model_config = Config.fromfile(config["config"])
        model_spec = copy.deepcopy(model_config.model)
        backbone_spec = model_spec["backbone"]
        backbone_spec.pop("pretrained", None)
        self.data_preprocessor = MODELS.build(model_spec["data_preprocessor"])
        self.backbone = MODELS.build(backbone_spec)
        self.decode_head = MODELS.build(model_spec["decode_head1"])
        self.test_config = model_spec["test_cfg"]
        # This is the pinned official CaRe-Ego checkpoint. It contains
        # MMEngine training metadata in addition to tensors, so PyTorch 2.6+
        # cannot open it with the new weights_only default.
        checkpoint = torch.load(
            config["checkpoint"],
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        state_dict = checkpoint["state_dict"]
        self.backbone.load_state_dict(
            {
                key.removeprefix("backbone."): value
                for key, value in state_dict.items()
                if key.startswith("backbone.")
            },
            strict=True,
        )
        self.decode_head.load_state_dict(
            {
                key.removeprefix("decode_head1."): value
                for key, value in state_dict.items()
                if key.startswith("decode_head1.")
            },
            strict=True,
        )
        del checkpoint
        self.data_preprocessor.to(self.device)
        self.backbone.to(self.device).eval()
        self.decode_head.to(self.device).eval()

    def to(self, device: str) -> None:
        self.device = self.torch.device(device)
        self.data_preprocessor.to(self.device)
        self.backbone.to(self.device)
        self.decode_head.to(self.device)

    def _load_input(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        rgb = np.asarray(Image.open(path).convert("RGB"))
        resized_rgb = cv2.resize(
            rgb,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        bgr = np.ascontiguousarray(resized_rgb[..., ::-1])
        tensor = self.torch.from_numpy(bgr).permute(2, 0, 1).float()
        return rgb, tensor

    def predict_many(
        self, paths: list[Path], batch_size: int | None = None
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        torch = self.torch
        batch_size = self.batch_size if batch_size is None else batch_size
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        outputs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            originals: list[np.ndarray] = []
            tensors = []
            for path in batch_paths:
                original, tensor = self._load_input(path)
                originals.append(original)
                tensors.append(tensor)
            processed = self.data_preprocessor(
                {"inputs": tensors, "data_samples": None}, training=False
            )
            inputs = processed["inputs"].to(self.device)
            metadata = [
                {
                    "ori_shape": (self.input_height, self.input_width),
                    "img_shape": (self.input_height, self.input_width),
                    "pad_shape": (self.input_height, self.input_width),
                    "padding_size": [0, 0, 0, 0],
                }
                for _ in batch_paths
            ]
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                features = self.backbone(inputs)
                hand_logits, _features = self.decode_head.predict(
                    features, metadata, self.test_config
                )
                hand_probability = hand_logits.float().softmax(dim=1)
            for batch_index, (path, original) in enumerate(
                zip(batch_paths, originals)
            ):
                original_height, original_width = original.shape[:2]
                probability = hand_probability[batch_index].cpu().numpy()
                left, right = build_hand_prompts(
                    probability,
                    self.confidence_threshold,
                    self.prompt_mode,
                )
                left = cv2.resize(
                    left,
                    (original_width, original_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                right = cv2.resize(
                    right,
                    (original_width, original_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                outputs[int(path.stem)] = (
                    left.astype(np.float32),
                    right.astype(np.float32),
                )
        return outputs

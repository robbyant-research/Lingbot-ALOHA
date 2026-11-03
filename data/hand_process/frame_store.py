from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import TypeAlias

import numpy as np
from PIL import Image

from .h5_utils import require_h5py
from .video import frame_paths


HDF5_FRAME_STORE_SCHEMA = "hand-process-generated-frames"
HDF5_FRAME_STORE_VERSION = 1
FrameValue: TypeAlias = Path | np.ndarray


def is_hdf5_frame_store(path: str | Path) -> bool:
    candidate = Path(path)
    return candidate.is_file() and candidate.suffix.casefold() in {".h5", ".hdf5"}


def _validated_dataset(store_path: Path, handle):
    if handle.attrs.get("schema") != HDF5_FRAME_STORE_SCHEMA:
        raise ValueError(f"unsupported HDF5 frame store schema: {store_path}")
    if int(handle.attrs.get("schema_version", -1)) != HDF5_FRAME_STORE_VERSION:
        raise ValueError(f"unsupported HDF5 frame store version: {store_path}")
    if "frames" not in handle:
        raise ValueError(f"HDF5 frame store has no frames dataset: {store_path}")
    frames = handle["frames"]
    if frames.dtype != np.dtype(np.uint8) or len(frames.shape) != 4:
        raise ValueError(
            f"HDF5 frame store frames must be uint8 [T,H,W,3]: {store_path}"
        )
    if frames.shape[-1] != 3:
        raise ValueError(
            f"HDF5 frame store frames must have three channels: {store_path}"
        )
    expected_shape = (
        int(handle.attrs.get("num_frames", -1)),
        int(handle.attrs.get("height", -1)),
        int(handle.attrs.get("width", -1)),
        3,
    )
    if frames.shape != expected_shape:
        raise ValueError(
            f"HDF5 frame store shape mismatch: {frames.shape} != {expected_shape}"
        )
    return frames


def frame_store_count(path: str | Path) -> int:
    store_path = Path(path)
    if store_path.is_dir():
        return len(frame_paths(store_path))
    if not is_hdf5_frame_store(store_path):
        raise ValueError(f"unsupported generated frame store: {store_path}")
    h5py = require_h5py()
    with h5py.File(store_path, "r") as handle:
        return int(_validated_dataset(store_path, handle).shape[0])


def load_frame_store(path: str | Path) -> list[FrameValue]:
    store_path = Path(path)
    if store_path.is_dir():
        return frame_paths(store_path)
    if not is_hdf5_frame_store(store_path):
        raise ValueError(f"unsupported generated frame store: {store_path}")
    h5py = require_h5py()
    with h5py.File(store_path, "r") as handle:
        frames = np.asarray(_validated_dataset(store_path, handle), dtype=np.uint8)
    return [frame for frame in frames]


def frame_value_rgb(value: FrameValue) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError(
                "generated array frame must be uint8 [H,W,3], got "
                f"{value.dtype} {value.shape}"
            )
        return value
    with Image.open(value) as image:
        return np.asarray(image.convert("RGB"))


class Hdf5VideoWriter:
    """OpenCV VideoWriter-compatible sink for lossless generated RGB frames."""

    def __init__(
        self,
        destination: str | Path,
        *,
        expected_num_frames: int,
        fps: Fraction,
        frame_size: tuple[int, int],
    ) -> None:
        if expected_num_frames < 1:
            raise ValueError("expected_num_frames must be at least 1")
        width, height = (int(value) for value in frame_size)
        if width < 1 or height < 1:
            raise ValueError("frame_size must contain positive dimensions")
        self.destination = Path(destination)
        self.temporary = self.destination.with_suffix(
            self.destination.suffix + ".tmp"
        )
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.temporary.unlink(missing_ok=True)
        self.destination.unlink(missing_ok=True)
        h5py = require_h5py()
        self._handle = h5py.File(self.temporary, "w")
        self._frames = self._handle.create_dataset(
            "frames",
            shape=(expected_num_frames, height, width, 3),
            dtype=np.uint8,
            chunks=None,
            compression=None,
        )
        self._handle.attrs.update(
            {
                "schema": HDF5_FRAME_STORE_SCHEMA,
                "schema_version": HDF5_FRAME_STORE_VERSION,
                "num_frames": expected_num_frames,
                "width": width,
                "height": height,
                "fps_numerator": int(fps.numerator),
                "fps_denominator": int(fps.denominator),
                "color_space": "RGB",
            }
        )
        self._expected_num_frames = expected_num_frames
        self._frame_shape = (height, width, 3)
        self._index = 0
        self._released = False

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV compatibility
        return not self._released and self._handle is not None

    def write(self, frame: np.ndarray) -> None:
        if not self.isOpened():
            raise RuntimeError("cannot write to a released HDF5 video writer")
        if self._index >= self._expected_num_frames:
            raise RuntimeError(
                "DiffuEraser generated more frames than expected: "
                f"> {self._expected_num_frames}"
            )
        bgr = np.asarray(frame)
        if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[-1] != 3:
            raise ValueError(
                "DiffuEraser generated frame must be uint8 [H,W,3], got "
                f"{bgr.dtype} {bgr.shape}"
            )
        rgb = bgr[..., ::-1]
        if rgb.shape != self._frame_shape:
            rgb = np.asarray(
                Image.fromarray(rgb).resize(
                    (self._frame_shape[1], self._frame_shape[0]),
                    Image.Resampling.BICUBIC,
                )
            )
        self._frames[self._index] = rgb
        self._index += 1

    def release(self) -> None:
        if self._released:
            return
        try:
            if self._index != self._expected_num_frames:
                raise RuntimeError(
                    "DiffuEraser generated frame count mismatch: "
                    f"{self._index} != {self._expected_num_frames}"
                )
            self._handle.flush()
            self._handle.close()
            self._handle = None
            self.temporary.replace(self.destination)
            self._released = True
        except Exception:
            self.abort()
            raise

    def abort(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is not None:
            handle.close()
        self.temporary.unlink(missing_ok=True)
        if not self._released:
            self.destination.unlink(missing_ok=True)

    @property
    def completed(self) -> bool:
        return self._released and self.destination.is_file()


class OpenCVHdf5Proxy:
    """Delegate OpenCV except for one expected VideoWriter construction."""

    def __init__(
        self,
        cv2_module,
        destination: str | Path,
        *,
        expected_num_frames: int,
        fps: Fraction,
        frame_size: tuple[int, int],
    ) -> None:
        self._cv2 = cv2_module
        self._destination = Path(destination)
        self._expected_num_frames = expected_num_frames
        self._fps = fps
        self._frame_size = frame_size
        self._writer: Hdf5VideoWriter | None = None

    def __getattr__(self, name: str):
        return getattr(self._cv2, name)

    def VideoWriter(self, _path, _fourcc, fps, frame_size):  # noqa: N802
        if self._writer is not None:
            raise RuntimeError("DiffuEraser created more than one output writer")
        if abs(float(fps) - float(self._fps)) > 1e-6:
            raise RuntimeError(
                f"DiffuEraser output fps changed unexpectedly: {fps} != {self._fps}"
            )
        self._writer = Hdf5VideoWriter(
            self._destination,
            expected_num_frames=self._expected_num_frames,
            fps=self._fps,
            frame_size=self._frame_size,
        )
        return self._writer

    def require_completed(self) -> None:
        if self._writer is None or not self._writer.completed:
            raise RuntimeError("DiffuEraser did not complete its output writer")

    def abort(self) -> None:
        if self._writer is not None and not self._writer.completed:
            self._writer.abort()

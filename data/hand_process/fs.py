from __future__ import annotations

import errno
import os
import shutil
import uuid
from pathlib import Path


def ensure_within(path: str | Path, root: str | Path) -> Path:
    candidate = Path(path).resolve()
    allowed = Path(root).resolve()
    if candidate != allowed and not candidate.is_relative_to(allowed):
        raise ValueError(f"path escapes allowed root: {candidate} not under {allowed}")
    return candidate


def recreate_directory(path: str | Path, root: str | Path) -> Path:
    candidate = ensure_within(path, root)
    if candidate.exists():
        shutil.rmtree(candidate)
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def remove_directory(path: str | Path, root: str | Path) -> None:
    candidate = ensure_within(path, root)
    if candidate.exists():
        shutil.rmtree(candidate)


def hardlink_or_copy(source: str | Path, destination: str | Path) -> None:
    src = Path(source)
    dst = Path(destination)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if src.stat().st_size == dst.stat().st_size:
            return
        raise FileExistsError(f"destination exists with different size: {dst}")
    try:
        os.link(src, dst)
    except OSError as error:
        if error.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES):
            raise
        shutil.copy2(src, dst)


def atomic_copy_file(source: str | Path, destination: str | Path) -> None:
    """Copy one artifact across filesystems without exposing partial output."""

    src = Path(source)
    dst = Path(destination)
    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary = dst.with_name(f".{dst.name}.tmp-{uuid.uuid4().hex}")
    try:
        shutil.copy2(src, temporary)
        os.replace(temporary, dst)
    finally:
        temporary.unlink(missing_ok=True)


def relative_symlink(source: str | Path, destination: str | Path) -> None:
    src = Path(source).resolve()
    dst = Path(destination)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(os.path.relpath(src, dst.parent))


def chunk_ranges(
    total: int,
    size: int,
    overlap: int,
    *,
    min_size: int = 1,
) -> list[tuple[int, int]]:
    if total <= 0:
        return []
    if size <= 0 or overlap < 0 or overlap >= size or min_size <= 0 or min_size > size:
        raise ValueError(
            f"invalid chunk parameters size={size}, overlap={overlap}, "
            f"min_size={min_size}"
        )
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(total, start + size)
        ranges.append((start, end))
        if end == total:
            break
        start = end - overlap
    if total >= min_size and ranges[-1][1] - ranges[-1][0] < min_size:
        ranges[-1] = (total - min_size, total)
    return ranges


def blend_weight(local_index: int, length: int, overlap: int) -> float:
    if overlap <= 0:
        return 1.0
    left = min(1.0, (local_index + 1) / overlap)
    right = min(1.0, (length - local_index) / overlap)
    return max(1e-3, min(left, right))

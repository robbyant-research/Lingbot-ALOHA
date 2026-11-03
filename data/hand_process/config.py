from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def _merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config_file(
    config_path: Path, loading: tuple[Path, ...]
) -> dict[str, Any]:
    if config_path in loading:
        chain = " -> ".join(str(path) for path in (*loading, config_path))
        raise ValueError(f"configuration extends cycle: {chain}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    extends = config.pop("extends", None)
    if extends is None:
        return config
    if not isinstance(extends, str) or not extends:
        raise ValueError(f"configuration extends must be a path: {config_path}")
    base_path = Path(extends).expanduser()
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path
    base_path = base_path.resolve()
    return _merge_config(
        _load_config_file(base_path, (*loading, config_path)), config
    )


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    config = _load_config_file(config_path, ())
    if config.get("schema_version") != 1:
        raise ValueError(
            f"unsupported schema_version={config.get('schema_version')!r}"
        )
    config = _expand_paths(config)
    config["_config_path"] = str(config_path)
    return config


def _expand_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Expand portable path tokens without changing SAM's Hydra config names."""
    root = Path(__file__).resolve().parents[1]
    identity = hashlib.sha256(str(root).encode()).hexdigest()[:10]
    scratch = Path(os.environ.get(
        "HAND_PROCESS_WORK_ROOT", f"/dev/shm/hand-process-{os.getuid()}-{identity}"
    )).expanduser().resolve()
    assets = Path(os.environ.get(
        "HAND_PROCESS_ASSET_ROOT", str(root / "assets" / "models")
    )).expanduser().resolve()
    sockets = Path(tempfile.gettempdir()) / f"hp-{os.getuid()}-{identity}"
    replacements = {
        "${project_root}": str(root),
        "${asset_root}": str(assets),
        "${work_root}": str(scratch),
        "${socket_root}": str(sockets),
    }

    def expand(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, str):
            for token, replacement in replacements.items():
                value = value.replace(token, replacement)
        return value

    return expand(config)


def require_path(config: dict[str, Any], *keys: str) -> Path:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing configuration key: {'.'.join(keys)}")
        value = value[key]
    if not isinstance(value, str):
        raise ValueError(f"configuration path is not a string: {'.'.join(keys)}")
    return Path(value)

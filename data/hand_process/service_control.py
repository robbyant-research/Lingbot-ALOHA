from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

import yaml

from .config import load_config


_PROFILE_CONFIGS = tuple(
    f"configs/{name}.yaml"
    for name in ("base", "server", "low_memory", "grounding_dino")
)


def _resolve_socket_dir(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def discover_socket_dirs(project_root: str | Path) -> list[Path]:
    root = Path(project_root).resolve()
    directories: set[Path] = set()
    for relative in _PROFILE_CONFIGS:
        config_path = root / relative
        if not config_path.is_file():
            continue
        config = load_config(config_path)
        value = config.get("runtime", {}).get("model_service", {}).get("socket_dir")
        if value:
            directories.add(_resolve_socket_dir(root, value))

    generated_root = root / "runtime" / "generated-configs"
    for config_path in generated_root.glob("*.yaml"):
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            value = config["runtime"]["model_service"]["socket_dir"]
        except (KeyError, OSError, TypeError, yaml.YAMLError):
            continue
        directories.add(_resolve_socket_dir(root, value))

    runtime_root = root / "runtime"
    if runtime_root.is_dir():
        for socket in runtime_root.rglob("gpu-*.sock"):
            directories.add(socket.parent.resolve())
        for ready in runtime_root.rglob("gpu-*.ready.json"):
            directories.add(ready.parent.resolve())
    return sorted(directories)


def _process_cwd(pid: int) -> Path | None:
    try:
        return Path(f"/proc/{pid}/cwd").resolve(strict=True)
    except OSError:
        return None


def _process_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (OSError, UnicodeDecodeError):
        return ""


def _parent_pid(pid: int) -> int | None:
    try:
        lines = Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines()
        value = next(line for line in lines if line.startswith("PPid:"))
        return int(value.split(":", 1)[1].strip())
    except (OSError, StopIteration, ValueError):
        return None


def _pid_is_project_service(pid: int, project_root: Path) -> bool:
    if pid <= 1:
        return False
    cwd = _process_cwd(pid)
    if cwd != project_root:
        return False
    try:
        os.kill(pid, 0)
    except (PermissionError, ProcessLookupError):
        return False
    if "multiprocessing.spawn" not in _process_cmdline(pid):
        return False
    parent_pid = _parent_pid(pid)
    if parent_pid is None or _process_cwd(parent_pid) != project_root:
        return False
    parent_cmdline = _process_cmdline(parent_pid)
    return "hand_process.cli serve" in parent_cmdline


def _read_ready(ready_path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(ready_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def stop_project_services(
    project_root: str | Path,
    *,
    dry_run: bool = False,
    timeout_seconds: float = 30,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    socket_dirs = discover_socket_dirs(root)
    targeted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for socket_dir in socket_dirs:
        ready_paths = sorted(socket_dir.glob("gpu-*.ready.json"))
        for ready_path in ready_paths:
            status = _read_ready(ready_path)
            try:
                pid = int(status["pid"]) if status is not None else -1
            except (KeyError, TypeError, ValueError):
                pid = -1
            record = {
                "pid": pid,
                "ready_file": str(ready_path),
                "socket": str(
                    ready_path.with_name(
                        ready_path.name.removesuffix(".ready.json") + ".sock"
                    )
                ),
            }
            if not _pid_is_project_service(pid, root):
                record["reason"] = "PID is not a live service in this project"
                skipped.append(record)
                continue
            targeted.append(record)
            if not dry_run:
                os.kill(pid, signal.SIGTERM)

    if not dry_run and targeted:
        deadline = time.monotonic() + timeout_seconds
        pending = {record["pid"] for record in targeted}
        while pending and time.monotonic() < deadline:
            pending = {pid for pid in pending if _pid_is_project_service(pid, root)}
            if pending:
                time.sleep(0.2)
        for pid in pending:
            os.kill(pid, signal.SIGKILL)

        for record in targeted:
            Path(record["socket"]).unlink(missing_ok=True)
            Path(record["ready_file"]).unlink(missing_ok=True)

    return {
        "project_root": str(root),
        "dry_run": dry_run,
        "stopped": 0 if dry_run else len(targeted),
        "would_stop": len(targeted) if dry_run else 0,
        "services": targeted,
        "skipped": skipped,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stop model services launched by this project"
    )
    parser.add_argument("--project-root", default=Path(__file__).parents[1])
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = stop_project_services(
        args.project_root,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

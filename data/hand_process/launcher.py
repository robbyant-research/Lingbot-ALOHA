from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import yaml

from .assets import validate_config_assets
from .config import load_config
from .datasets import discover_episodes
from .model_service import (
    DEFAULT_LANES_PER_GPU,
    model_config_digest,
    model_service_endpoints,
    validate_service_ready,
)
from .pipeline import run_pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ordered hand-removal launcher: IN1 OUT1 [IN2 OUT2 ...]"
        )
    )
    parser.add_argument("target", choices=("server", "edge"))
    parser.add_argument("paths", nargs="+", metavar="PATH")
    return parser


def _video_inventory(input_dir: Path) -> list[Path]:
    videos = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".mp4"
    )
    if not videos:
        raise ValueError(f"no *.mp4 videos found in {input_dir}")
    return videos


PathPair = tuple[Path, Path, list[Path]]


def _prepare_path_pairs(path_values: Sequence[str]) -> list[PathPair]:
    if len(path_values) < 2 or len(path_values) % 2:
        raise ValueError(
            "input/output paths must be provided in ordered pairs: "
            "IN1 OUT1 [IN2 OUT2 ...]"
        )
    pairs: list[PathPair] = []
    for index in range(0, len(path_values), 2):
        input_dir = Path(path_values[index]).expanduser().resolve()
        output_dir = Path(path_values[index + 1]).expanduser().resolve()
        if not input_dir.is_dir():
            raise NotADirectoryError(input_dir)
        videos = _video_inventory(input_dir)
        if output_dir.is_symlink() and not output_dir.exists():
            raise ValueError(f"output directory is a dangling symlink: {output_dir}")
        if output_dir.exists():
            if not output_dir.is_dir():
                raise NotADirectoryError(output_dir)
            if any(output_dir.iterdir()):
                raise ValueError(
                    f"output directory must not exist or must be empty: {output_dir}"
                )
        pairs.append((input_dir, output_dir, videos))

    outputs = [output_dir for _input_dir, output_dir, _videos in pairs]
    if len(set(outputs)) != len(outputs):
        raise ValueError("output directories must be unique")
    for index, output_dir in enumerate(outputs):
        for other in outputs[index + 1 :]:
            if output_dir.is_relative_to(other) or other.is_relative_to(output_dir):
                raise ValueError(
                    "output directories must not contain one another: "
                    f"{output_dir}, {other}"
                )
        for input_dir, _pair_output, _videos in pairs:
            if output_dir == input_dir or output_dir.is_relative_to(input_dir):
                raise ValueError(
                    "output video directories must be outside every input tree: "
                    f"{output_dir}"
                )
    return pairs


def _base_config_path(project_root: Path, target: str) -> Path:
    if target == "server":
        configured = os.environ.get("HAND_PROCESS_SERVER_CONFIG")
        if configured:
            path = Path(configured).expanduser()
            return path if path.is_absolute() else project_root / path
        deployment_default = project_root / "configs" / "server_default.yaml"
        if deployment_default.is_file():
            return deployment_default
        return project_root / "configs" / "server.yaml"
    return project_root / "configs" / "low_memory.yaml"


def _run_id(
    target: str,
    input_dir: Path,
    output_dir: Path,
    videos: list[Path],
) -> str:
    return _run_id_for_pairs(target, [(input_dir, output_dir, videos)])


def _run_id_for_pairs(target: str, path_pairs: Sequence[PathPair]) -> str:
    digest = hashlib.sha256()
    digest.update(target.encode())
    digest.update(b"\0")
    for input_dir, output_dir, videos in path_pairs:
        for value in (str(input_dir), str(output_dir)):
            digest.update(value.encode())
            digest.update(b"\0")
        for video in videos:
            stat = video.stat()
            digest.update(video.relative_to(input_dir).as_posix().encode())
            digest.update(f":{stat.st_size}:{stat.st_mtime_ns}\0".encode())
    return f"launcher-{target}-{digest.hexdigest()[:16]}"


def _datasets_for_pairs(path_pairs: Sequence[PathPair]) -> dict[str, dict]:
    datasets: dict[str, dict] = {}
    multiple = len(path_pairs) > 1
    for index, (input_dir, output_dir, _videos) in enumerate(path_pairs):
        name = f"input_videos_{index + 1:03d}" if multiple else "input_videos"
        datasets[name] = {
            "input_dir": str(input_dir),
            "final_video_root": str(output_dir),
            "display_name": input_dir.name or str(input_dir),
            "input_order": index,
            "relative_path": name if multiple else ".",
            "output_relative_path": ".",
            "glob": "**/*.mp4",
            "preserve_relative_tree": True,
        }
    return datasets


def _server_gpus() -> list[str]:
    configured = os.environ.get("HAND_PROCESS_GPUS")
    if configured is not None:
        gpus = [value.strip() for value in configured.split(",") if value.strip()]
        if not gpus or len(gpus) != len(set(gpus)) or any(not value.isdigit() for value in gpus):
            raise ValueError("HAND_PROCESS_GPUS must contain unique physical GPU indexes")
        return gpus
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    gpus = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not gpus:
        raise RuntimeError("nvidia-smi reported no GPUs")
    return gpus


def _write_runtime_config(
    *,
    project_root: Path,
    target: str,
    path_pairs: Sequence[PathPair],
    run_id: str,
) -> tuple[dict, Path]:
    config = load_config(_base_config_path(project_root, target))
    config.pop("_config_path", None)
    config["datasets"] = _datasets_for_pairs(path_pairs)
    runtime_root = project_root / "runtime"
    config["run_root"] = str(runtime_root / "launcher-runs")
    config["log_root"] = str(project_root / "logs")
    config.pop("final_video_root", None)
    config.setdefault("runtime", {})["ordered_datasets"] = True
    generated_root = runtime_root / "generated-configs"
    generated_root.mkdir(parents=True, exist_ok=True)
    config_path = generated_root / f"{run_id}.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    config["_config_path"] = str(config_path)
    return config, config_path


def _configured_lanes_per_gpu(config: dict) -> int:
    value = (
        config.get("runtime", {})
        .get("model_service", {})
        .get("lanes_per_gpu", DEFAULT_LANES_PER_GPU)
    )
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("runtime.model_service.lanes_per_gpu must be >= 1")
    return value


def _all_services_ready(
    config: dict,
    gpus: list[str],
    socket_dir: Path,
    lanes_per_gpu: int,
) -> bool:
    try:
        for endpoint in model_service_endpoints(
            socket_dir,
            gpus,
            lanes_per_gpu=lanes_per_gpu,
        ):
            validate_service_ready(endpoint.address, config)
    except RuntimeError:
        return False
    return True


def _ensure_model_services(
    *,
    config: dict,
    config_path: Path,
    project_root: Path,
    gpus: list[str],
    lanes_per_gpu: int,
) -> Path:
    service_config = config["runtime"]["model_service"]
    configured_dir = Path(service_config["socket_dir"])
    if _all_services_ready(config, gpus, configured_dir, lanes_per_gpu):
        return configured_dir

    endpoints = model_service_endpoints(
        configured_dir,
        gpus,
        lanes_per_gpu=lanes_per_gpu,
    )
    occupied = any(endpoint.address.exists() for endpoint in endpoints)
    if occupied:
        digest = model_config_digest(config)[:16]
        configured_dir = (
            configured_dir.parent / f"{digest}-lanes-{lanes_per_gpu}"
        )
        service_config["socket_dir"] = str(configured_dir)
        config_path.write_text(
            yaml.safe_dump(
                {key: value for key, value in config.items() if key != "_config_path"},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
    configured_dir.mkdir(parents=True, exist_ok=True)
    log_path = project_root / "logs" / (
        f"launcher-service-{config.get('profile', 'default')}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hand_process.cli",
            "serve",
            "--config",
            str(config_path),
            "--gpus",
            ",".join(gpus),
            "--socket-dir",
            str(configured_dir),
            "--lanes-per-gpu",
            str(lanes_per_gpu),
        ],
        cwd=project_root,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log.close()
    deadline = time.monotonic() + float(
        service_config.get("startup_timeout_seconds", 900)
    )
    while time.monotonic() < deadline:
        if _all_services_ready(config, gpus, configured_dir, lanes_per_gpu):
            return configured_dir
        if process.poll() is not None:
            raise RuntimeError(
                f"model service exited with {process.returncode}; see {log_path}"
            )
        time.sleep(1)
    raise TimeoutError(f"model service did not become ready; see {log_path}")


def launch(
    target: str,
    *path_values: str,
) -> dict:
    project_root = Path(__file__).resolve().parents[1]
    path_pairs = _prepare_path_pairs(path_values)
    run_id = _run_id_for_pairs(target, path_pairs)
    config, config_path = _write_runtime_config(
        project_root=project_root,
        target=target,
        path_pairs=path_pairs,
        run_id=run_id,
    )
    validate_config_assets(config)
    for _input_dir, output_dir, _videos in path_pairs:
        output_dir.mkdir(parents=True, exist_ok=True)
    episodes = discover_episodes(config)
    gpus = _server_gpus() if target == "server" else ["0"]
    lanes_per_gpu = _configured_lanes_per_gpu(config)
    service_dir = _ensure_model_services(
        config=config,
        config_path=config_path,
        project_root=project_root,
        gpus=gpus,
        lanes_per_gpu=lanes_per_gpu,
    )
    summary = run_pipeline(
        config=config,
        run_id=run_id,
        stage="full",
        episodes=episodes,
        gpus=gpus,
        resume=True,
        service_dir=str(service_dir),
        lanes_per_gpu=lanes_per_gpu,
    )
    result = {
        "target": target,
        "path_pairs": [
            {
                "input_video_dir": str(input_dir),
                "output_video_dir": str(output_dir),
                "videos": len(videos),
            }
            for input_dir, output_dir, videos in path_pairs
        ],
        "run_id": run_id,
        "videos": sum(len(videos) for _input, _output, videos in path_pairs),
        "gpus": gpus,
        "lanes_per_gpu": lanes_per_gpu,
        "summary": summary,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = launch(args.target, *args.paths)
    except (NotADirectoryError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    return 1 if result["summary"].get("failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())

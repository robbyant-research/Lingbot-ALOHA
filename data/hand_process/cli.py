from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .assets import (
    build_asset_manifest,
    download_assets,
    validate_config_assets,
)
from .config import load_config
from .datasets import discover_episodes, select_identities, select_stage
from .video import probe_video


def _lanes_per_gpu(config: dict, explicit: int | None) -> int:
    from .model_service import DEFAULT_LANES_PER_GPU

    value = (
        explicit
        if explicit is not None
        else config.get("runtime", {})
        .get("model_service", {})
        .get("lanes_per_gpu", DEFAULT_LANES_PER_GPU)
    )
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("lanes_per_gpu must be at least 1")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ego-video hand removal pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config(subparser):
        subparser.add_argument(
            "--config", default="configs/server.yaml", help="YAML configuration"
        )

    inspect_parser = subparsers.add_parser("inspect")
    add_config(inspect_parser)

    manifest_parser = subparsers.add_parser("assets-manifest")
    add_config(manifest_parser)
    manifest_parser.add_argument("--output")
    manifest_parser.add_argument("--asset-root")
    manifest_parser.add_argument("--validate", action="store_true")

    download_assets_parser = subparsers.add_parser("assets-download")
    add_config(download_assets_parser)
    download_assets_parser.add_argument("--manifest", required=True)

    run_parser = subparsers.add_parser("run")
    add_config(run_parser)
    run_parser.add_argument(
        "--stage", choices=("smoke", "calibration", "full"), required=True
    )
    run_parser.add_argument("--run-id")
    run_parser.add_argument("--gpus", required=True)
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument(
        "--lanes-per-gpu",
        type=int,
        help="parallel model-service lanes per GPU (default: config or 1)",
    )
    run_parser.add_argument(
        "--service-dir",
        help="directory containing persistent per-GPU model sockets",
    )
    run_parser.add_argument(
        "--identity",
        action="append",
        help="run only this dataset/filename identity; may be repeated",
    )
    run_parser.add_argument(
        "--identity-file",
        action="append",
        help="UTF-8 file with one dataset/filename identity per line",
    )

    qc_parser = subparsers.add_parser("qc")
    add_config(qc_parser)
    qc_parser.add_argument("--run-id", required=True)

    compact_parser = subparsers.add_parser("compact-masks")
    add_config(compact_parser)
    compact_parser.add_argument("--run-id", required=True)

    serve_parser = subparsers.add_parser("serve")
    add_config(serve_parser)
    serve_parser.add_argument("--gpus", required=True)
    serve_parser.add_argument("--socket-dir")
    serve_parser.add_argument(
        "--lanes-per-gpu",
        type=int,
        help="independent model-service processes per GPU (default: config or 1)",
    )

    service_status_parser = subparsers.add_parser("service-status")
    add_config(service_status_parser)
    service_status_parser.add_argument("--gpus", required=True)
    service_status_parser.add_argument("--socket-dir")
    service_status_parser.add_argument(
        "--lanes-per-gpu",
        type=int,
        help="independent model-service processes per GPU (default: config or 1)",
    )

    return parser


def _inspect(config: dict) -> dict:
    episodes = discover_episodes(config)
    datasets: dict[str, dict] = {}
    total_frames = 0
    total_duration = 0.0
    for episode in episodes:
        info = probe_video(episode.source)
        current = datasets.setdefault(
            episode.dataset,
            {
                "videos": 0,
                "frames": 0,
                "duration_seconds": 0.0,
                "resolutions": {},
                "fps": {},
            },
        )
        current["videos"] += 1
        current["frames"] += info.num_frames
        current["duration_seconds"] += info.duration_seconds
        resolution = f"{info.width}x{info.height}"
        fps = f"{info.fps_numerator}/{info.fps_denominator}"
        current["resolutions"][resolution] = (
            current["resolutions"].get(resolution, 0) + 1
        )
        current["fps"][fps] = current["fps"].get(fps, 0) + 1
        total_frames += info.num_frames
        total_duration += info.duration_seconds
    return {
        "videos": len(episodes),
        "frames": total_frames,
        "duration_seconds": total_duration,
        "datasets": datasets,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "inspect":
        print(json.dumps(_inspect(config), indent=2, sort_keys=True))
        return 0
    if args.command == "assets-manifest":
        if args.asset_root:
            config["asset_root"] = str(Path(args.asset_root).resolve())
        manifest = (
            validate_config_assets(config)
            if args.validate
            else build_asset_manifest(config["asset_root"])
        )
        rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        if args.output:
            Path(args.output).write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
        return 0
    if args.command == "assets-download":
        print(
            json.dumps(
                download_assets(config, args.manifest), indent=2, sort_keys=True
            )
        )
        return 0
    if args.command == "run":
        from .pipeline import run_pipeline

        validate_config_assets(config)
        discovered = discover_episodes(config)
        requested_identities = list(args.identity or [])
        for identity_file in args.identity_file or []:
            requested_identities.extend(
                line.strip()
                for line in Path(identity_file).read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
        selected = (
            select_identities(discovered, requested_identities)
            if requested_identities
            else select_stage(discovered, args.stage)
        )
        if args.limit is not None:
            selected = selected[: args.limit]
        run_id = args.run_id or (
            f"{args.stage}-" + time.strftime("%Y%m%d-%H%M%S")
        )
        service_config = config.get("runtime", {}).get("model_service", {})
        service_dir = args.service_dir
        if service_dir is None and service_config.get("required", False):
            service_dir = service_config.get("socket_dir")
            if not service_dir:
                raise ValueError(
                    "required model service has no socket_dir configured"
                )
        if service_dir is not None:
            lanes_per_gpu = _lanes_per_gpu(config, args.lanes_per_gpu)
        elif args.lanes_per_gpu is None:
            lanes_per_gpu = 1
        else:
            lanes_per_gpu = _lanes_per_gpu({}, args.lanes_per_gpu)
        summary = run_pipeline(
            config=config,
            run_id=run_id,
            stage=args.stage,
            episodes=selected,
            gpus=[item.strip() for item in args.gpus.split(",") if item.strip()],
            resume=args.resume,
            service_dir=service_dir,
            lanes_per_gpu=lanes_per_gpu,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1 if summary.get("failed", 0) else 0
    if args.command == "qc":
        from .qc import build_manifest

        records = build_manifest(Path(config["run_root"]) / args.run_id)
        print(json.dumps({"records": len(records)}, indent=2))
        return 0
    if args.command == "compact-masks":
        from .pipeline import compact_run_masks

        print(
            json.dumps(
                compact_run_masks(config, args.run_id),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command in {"serve", "service-status"}:
        from .model_service import fleet_status, serve_model_fleet

        gpus = [
            item.strip() for item in args.gpus.split(",") if item.strip()
        ]
        service_config = (
            config.get("runtime", {}).get("model_service", {})
        )
        lanes_per_gpu = _lanes_per_gpu(config, args.lanes_per_gpu)
        socket_dir = args.socket_dir or service_config.get("socket_dir")
        if not socket_dir:
            raise ValueError(
                "model service socket directory must be configured or passed"
            )
        if args.command == "serve":
            validate_config_assets(config)
            serve_model_fleet(
                config=config,
                gpus=gpus,
                socket_dir=socket_dir,
                lanes_per_gpu=lanes_per_gpu,
            )
            return 0
        print(
            json.dumps(
                fleet_status(
                    config=config,
                    gpus=gpus,
                    socket_dir=socket_dir,
                    lanes_per_gpu=lanes_per_gpu,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())

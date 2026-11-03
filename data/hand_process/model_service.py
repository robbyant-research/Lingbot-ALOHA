from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import signal
import time
import traceback
from dataclasses import asdict, dataclass
from fractions import Fraction
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import Any


_AUTHKEY = b"hand-process-model-service-v1"
_DEFAULT_OPERATIONS = ("segment", "inpaint")
DEFAULT_LANES_PER_GPU = 1
MODEL_SERVICE_IMPLEMENTATION_VERSION = 2


@dataclass(frozen=True)
class InpaintRequest:
    frame_dir: Path
    mask_dir: Path
    output_root: Path
    fps: Fraction
    log_name: str
    global_start: int = 0
    seed: int = 42

    def to_rpc(self) -> dict[str, Any]:
        return {
            "operation": "inpaint",
            "frame_dir": str(self.frame_dir),
            "mask_dir": str(self.mask_dir),
            "output_root": str(self.output_root),
            "fps": [self.fps.numerator, self.fps.denominator],
            "log_name": self.log_name,
            "global_start": self.global_start,
            "seed": self.seed,
            "response_format": "profiled-v1",
        }

    @classmethod
    def from_rpc(cls, value: dict[str, Any]) -> InpaintRequest:
        numerator, denominator = value["fps"]
        global_start = int(value.get("global_start", 0))
        if global_start < 0:
            raise ValueError("inpaint global_start must be >= 0")
        return cls(
            frame_dir=Path(value["frame_dir"]),
            mask_dir=Path(value["mask_dir"]),
            output_root=Path(value["output_root"]),
            fps=Fraction(int(numerator), int(denominator)),
            log_name=str(value["log_name"]),
            global_start=global_start,
            seed=int(value.get("seed", 42)),
        )


@dataclass(frozen=True)
class InpaintResult:
    output_dir: Path
    backend: str
    num_frames: int
    wall_seconds: float
    stage_seconds: dict[str, float]
    peak_cuda_allocated_bytes: int | None
    peak_cuda_reserved_bytes: int | None

    def to_rpc(self) -> dict[str, Any]:
        value = asdict(self)
        value["output_dir"] = str(self.output_dir)
        return value

    @classmethod
    def from_rpc(cls, value: str | dict[str, Any]) -> InpaintResult:
        if isinstance(value, str):
            return cls(
                output_dir=Path(value),
                backend="legacy",
                num_frames=0,
                wall_seconds=0.0,
                stage_seconds={},
                peak_cuda_allocated_bytes=None,
                peak_cuda_reserved_bytes=None,
            )
        return cls(
            output_dir=Path(value["output_dir"]),
            backend=str(value["backend"]),
            num_frames=int(value["num_frames"]),
            wall_seconds=float(value["wall_seconds"]),
            stage_seconds={
                str(name): float(seconds)
                for name, seconds in value.get("stage_seconds", {}).items()
            },
            peak_cuda_allocated_bytes=_optional_int(
                value.get("peak_cuda_allocated_bytes")
            ),
            peak_cuda_reserved_bytes=_optional_int(
                value.get("peak_cuda_reserved_bytes")
            ),
        )


@dataclass(frozen=True)
class ModelServiceEndpoint:
    gpu: str
    lane: int
    address: Path


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def service_operations(config: dict) -> tuple[str, ...]:
    configured = (
        config.get("runtime", {})
        .get("model_service", {})
        .get("operations", _DEFAULT_OPERATIONS)
    )
    if not isinstance(configured, (list, tuple)) or not configured:
        raise ValueError(
            "runtime.model_service.operations must be a non-empty list"
        )
    operations = tuple(str(operation) for operation in configured)
    unknown = sorted(set(operations) - set(_DEFAULT_OPERATIONS))
    if unknown:
        raise ValueError(
            f"unsupported model service operations: {unknown}; "
            f"expected a subset of {_DEFAULT_OPERATIONS}"
        )
    if len(set(operations)) != len(operations):
        raise ValueError("runtime.model_service.operations contains duplicates")
    return operations


def inpainter_config(config: dict) -> dict:
    models = config["models"]
    legacy = models.get("propainter")
    selected = models.get("inpainter")
    if selected is None:
        selected = legacy
    if not isinstance(selected, dict):
        raise ValueError(
            "models.inpainter or legacy models.propainter must be configured"
        )
    if isinstance(legacy, dict):
        legacy_backend = str(legacy.get("backend", "propainter"))
        selected_backend = str(
            selected.get("backend", legacy_backend)
        )
        if selected_backend == legacy_backend:
            return {**legacy, **selected}
    return selected


def inpainter_backend_name(config: dict) -> str:
    return str(inpainter_config(config).get("backend", "propainter"))


def model_config_digest(config: dict) -> str:
    payload = {
        "implementation_version": MODEL_SERVICE_IMPLEMENTATION_VERSION,
        "models": config["models"],
        "profile": config.get("profile"),
        "service_operations": service_operations(config),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def socket_path(socket_dir: str | Path, gpu: str, lane: int = 0) -> Path:
    if lane < 0:
        raise ValueError("model service lane must be >= 0")
    lane_suffix = "" if lane == 0 else f"-lane-{lane}"
    return Path(socket_dir) / f"gpu-{gpu}{lane_suffix}.sock"


def model_service_endpoints(
    socket_dir: str | Path,
    gpus: list[str],
    *,
    lanes_per_gpu: int = DEFAULT_LANES_PER_GPU,
) -> list[ModelServiceEndpoint]:
    if lanes_per_gpu < 1:
        raise ValueError("lanes_per_gpu must be at least 1")
    if len(set(gpus)) != len(gpus):
        raise ValueError("model service GPU list contains duplicates")
    return [
        ModelServiceEndpoint(
            gpu=gpu,
            lane=lane,
            address=socket_path(socket_dir, gpu, lane),
        )
        for gpu in gpus
        for lane in range(lanes_per_gpu)
    ]


def validate_service_ready(
    address: str | Path, config: dict
) -> dict[str, Any]:
    socket = Path(address)
    ready_path = socket.with_suffix(".ready.json")
    if not socket.exists() or not ready_path.is_file():
        raise RuntimeError(f"model service is not ready at {socket}")
    try:
        status = json.loads(ready_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"invalid model service readiness file: {ready_path}"
        ) from error
    expected = model_config_digest(config)
    if status.get("config_digest") != expected:
        raise RuntimeError(
            "model service configuration does not match the run: "
            f"{status.get('config_digest')} != {expected}"
        )
    try:
        os.kill(int(status["pid"]), 0)
    except (KeyError, TypeError, ValueError, ProcessLookupError) as error:
        raise RuntimeError(
            f"model service process is not alive at {socket}"
        ) from error
    return status


class ModelServiceClient:
    def __init__(self, address: str | Path, config: dict) -> None:
        self.address = Path(address)
        self.expected_digest = model_config_digest(config)
        status = self.request({"operation": "status"})
        actual_digest = status["config_digest"]
        if actual_digest != self.expected_digest:
            raise RuntimeError(
                "model service configuration does not match the run: "
                f"{actual_digest} != {self.expected_digest}"
            )

    def request(self, request: dict[str, Any]) -> Any:
        connection = Client(
            str(self.address), family="AF_UNIX", authkey=_AUTHKEY
        )
        try:
            connection.send(request)
            response = connection.recv()
        finally:
            connection.close()
        if not response.get("ok"):
            raise RuntimeError(
                f"model service request failed: {response.get('error')}\n"
                f"{response.get('traceback', '')}"
            )
        return response.get("result")

    def segment(
        self, *, originals: list[Path], config: dict, work: Path
    ) -> Path:
        del config
        result = self.request(
            {
                "operation": "segment",
                "originals": [str(path) for path in originals],
                "work": str(work),
            }
        )
        return Path(result)

    def run_chunk(
        self,
        frame_dir: str | Path,
        mask_dir: str | Path,
        output_root: str | Path,
        fps: Fraction,
        log_name: str,
    ) -> Path:
        return self.run_chunk_profiled(
            frame_dir,
            mask_dir,
            output_root,
            fps,
            log_name,
        ).output_dir

    def run_chunk_profiled(
        self,
        frame_dir: str | Path,
        mask_dir: str | Path,
        output_root: str | Path,
        fps: Fraction,
        log_name: str,
        *,
        global_start: int = 0,
        seed: int = 42,
    ) -> InpaintResult:
        request = InpaintRequest(
            frame_dir=Path(frame_dir),
            mask_dir=Path(mask_dir),
            output_root=Path(output_root),
            fps=fps,
            log_name=log_name,
            global_start=global_start,
            seed=seed,
        )
        return InpaintResult.from_rpc(self.request(request.to_rpc()))


class ModelRuntime:
    def __init__(self, config: dict, device: str = "cuda:0") -> None:
        import cv2
        import torch

        from .backends import (
            build_detector,
            build_inpainter,
            build_segmenter,
            detector_config,
            segmenter_config,
        )

        runtime_config = config.get("runtime", {})
        service_config = runtime_config.get("model_service", {})
        residency = service_config.get("residency", "resident")
        if residency not in {"resident", "staged"}:
            raise ValueError(
                "runtime.model_service.residency must be resident or staged"
        )
        self.config = config
        self.device = device
        require_tmpfs = runtime_config.get("require_tmpfs", False)
        if not isinstance(require_tmpfs, bool):
            raise ValueError("runtime.require_tmpfs must be true or false")
        require_tmpfs_override = os.environ.get("HAND_PROCESS_REQUIRE_TMPFS")
        if require_tmpfs_override is not None:
            if require_tmpfs_override not in {"0", "1"}:
                raise ValueError("HAND_PROCESS_REQUIRE_TMPFS must be 0 or 1")
            require_tmpfs = require_tmpfs_override == "1"
        self.require_tmpfs = require_tmpfs
        self.work_root: Path | None = None
        if self.require_tmpfs:
            from .fs import ensure_within

            self.work_root = ensure_within(config["work_root"], "/dev/shm")
        self.max_cuda_memory_gb: float | None = None
        limit_value = os.environ.get("HAND_PROCESS_MAX_CUDA_MEMORY_GB")
        if limit_value is not None:
            try:
                limit_gb = float(limit_value)
            except ValueError as error:
                raise ValueError(
                    "HAND_PROCESS_MAX_CUDA_MEMORY_GB must be a number"
                ) from error
            if not math.isfinite(limit_gb) or limit_gb <= 0:
                raise ValueError(
                    "HAND_PROCESS_MAX_CUDA_MEMORY_GB must be positive"
                )
            if str(device).startswith("cuda") and torch.cuda.is_available():
                total_bytes = torch.cuda.get_device_properties(
                    device
                ).total_memory
                limit_bytes = limit_gb * 1024**3
                torch.cuda.set_per_process_memory_fraction(
                    min(1.0, limit_bytes / total_bytes),
                    device=device,
                )
                self.max_cuda_memory_gb = limit_gb
        self.residency = residency
        self.operations = service_operations(config)
        self.active_model: str | None = None
        self.detector = None
        self.segmenter = None
        self.inpainter = None
        configured_cpu_threads = runtime_config.get("cpu_threads", 4)
        cpu_threads = int(
            os.environ.get(
                "HAND_PROCESS_CPU_THREADS", str(configured_cpu_threads)
            )
        )
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(min(2, cpu_threads))

        if "segment" in self.operations:
            self.detector = build_detector(
                detector_config(config), device=device
            )
            if residency == "staged":
                self.detector.to("cpu")
                torch.cuda.empty_cache()
            self.segmenter = build_segmenter(
                segmenter_config(config), device=device
            )
            if residency == "staged":
                self.segmenter.to("cpu")
                torch.cuda.empty_cache()
        if "inpaint" in self.operations:
            backend_log_root = (
                self.work_root
                / "backend-logs"
                / hashlib.sha256(
                    str(config.get("profile", "default")).encode("utf-8")
                ).hexdigest()[:12]
                if self.require_tmpfs and self.work_root is not None
                else Path(config["log_root"])
            )
            self.inpainter = build_inpainter(
                inpainter_config(config),
                log_root=backend_log_root,
                device=device,
                persistent=True,
            )
            if residency == "staged":
                self.inpainter.to("cpu")
                torch.cuda.empty_cache()
        if residency == "resident":
            self.active_model = (
                "all" if len(self.operations) > 1 else self.operations[0]
            )

    def _activate(self, name: str) -> None:
        if self.residency == "resident":
            return
        if self.active_model == name:
            return
        import torch

        models = {
            "detector": self.detector,
            "segmenter": self.segmenter,
            "inpainter": self.inpainter,
        }
        if self.active_model is not None:
            active = models[self.active_model]
            assert active is not None
            active.to("cpu")
            torch.cuda.empty_cache()
        selected = models[name]
        assert selected is not None
        selected.to(self.device)
        self.active_model = name

    def segment(self, originals: list[Path], work: Path) -> Path:
        if "segment" not in self.operations:
            raise ValueError("segment operation is disabled for this service")
        from .pipeline import _run_sam_chunks

        assert self.detector is not None
        assert self.segmenter is not None
        if self.require_tmpfs:
            from .fs import ensure_within

            assert self.work_root is not None
            work = ensure_within(work, self.work_root)
        return _run_sam_chunks(
            originals=originals,
            detector=self.detector,
            segmenter=self.segmenter,
            config=self.config,
            work=work,
            activate_model=self._activate,
        )

    def inpaint(
        self,
        request: InpaintRequest,
    ) -> InpaintResult:
        if "inpaint" not in self.operations:
            raise ValueError("inpaint operation is disabled for this service")
        import torch

        self._activate("inpainter")
        assert self.inpainter is not None
        if self.require_tmpfs:
            from .fs import ensure_within

            assert self.work_root is not None
            ensure_within(request.frame_dir, self.work_root)
            ensure_within(request.mask_dir, self.work_root)
            ensure_within(request.output_root, self.work_root)
        set_request_context = getattr(
            self.inpainter, "set_request_context", None
        )
        if set_request_context is not None:
            set_request_context(
                global_start=request.global_start,
                seed=request.seed,
            )
        is_cuda = str(self.device).startswith("cuda") and torch.cuda.is_available()
        if is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        output_dir = self.inpainter.run_chunk(
            request.frame_dir,
            request.mask_dir,
            request.output_root,
            request.fps,
            request.log_name,
        )
        if is_cuda:
            torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - started
        stage_seconds = {
            str(name): float(seconds)
            for name, seconds in getattr(
                self.inpainter, "last_profile", {}
            ).items()
        }
        from .frame_store import frame_store_count

        num_frames = frame_store_count(output_dir)
        return InpaintResult(
            output_dir=Path(output_dir),
            backend=inpainter_backend_name(self.config),
            num_frames=num_frames,
            wall_seconds=wall_seconds,
            stage_seconds=stage_seconds,
            peak_cuda_allocated_bytes=(
                int(torch.cuda.max_memory_allocated()) if is_cuda else None
            ),
            peak_cuda_reserved_bytes=(
                int(torch.cuda.max_memory_reserved()) if is_cuda else None
            ),
        )

    def status(self) -> dict[str, Any]:
        return {
            "pid": os.getpid(),
            "device": self.device,
            "residency": self.residency,
            "operations": list(self.operations),
            "inpainter_backend": (
                inpainter_backend_name(self.config)
                if "inpaint" in self.operations
                else None
            ),
            "active_model": self.active_model,
            "config_digest": model_config_digest(self.config),
            "profile": self.config.get("profile"),
            "max_cuda_memory_gb": self.max_cuda_memory_gb,
            "require_tmpfs": self.require_tmpfs,
        }


def _serve_one(
    config: dict, gpu: str, address: str, startup_delay_seconds: float
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if startup_delay_seconds > 0:
        time.sleep(startup_delay_seconds)
    path = Path(address)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"model service socket already exists: {path}")
    runtime = ModelRuntime(config)
    listener = Listener(str(path), family="AF_UNIX", authkey=_AUTHKEY)
    ready_path = path.with_suffix(".ready.json")
    ready_path.write_text(
        json.dumps(runtime.status(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    running = True

    def send(payload: dict[str, Any]) -> None:
        try:
            connection.send(payload)
        except (BrokenPipeError, EOFError, OSError):
            pass

    try:
        while running:
            connection = listener.accept()
            try:
                request = connection.recv()
                operation = request.get("operation")
                if operation == "status":
                    result = runtime.status()
                elif operation == "segment":
                    result = str(
                        runtime.segment(
                            [Path(value) for value in request["originals"]],
                            Path(request["work"]),
                        )
                    )
                elif operation == "inpaint":
                    result_value = runtime.inpaint(
                        InpaintRequest.from_rpc(request)
                    )
                    result = (
                        result_value.to_rpc()
                        if request.get("response_format") == "profiled-v1"
                        else str(result_value.output_dir)
                    )
                elif operation == "shutdown":
                    result = {"shutdown": True}
                    running = False
                else:
                    raise ValueError(f"unknown operation: {operation!r}")
                send({"ok": True, "result": result})
            except Exception as error:
                send(
                    {
                        "ok": False,
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                )
            finally:
                connection.close()
    finally:
        listener.close()
        ready_path.unlink(missing_ok=True)
        path.unlink(missing_ok=True)


def _wait_until_ready(
    config: dict, addresses: list[Path], processes: list[mp.Process]
) -> None:
    deadline = time.monotonic() + float(
        config.get("runtime", {})
        .get("model_service", {})
        .get("startup_timeout_seconds", 600)
    )
    pending = set(addresses)
    while pending:
        for process in processes:
            if process.exitcode is not None:
                raise RuntimeError(
                    f"model service {process.name} exited with "
                    f"code {process.exitcode}"
                )
        for address in tuple(pending):
            if address.with_suffix(".ready.json").is_file():
                pending.remove(address)
        if not pending:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"model services did not become ready: "
                f"{sorted(str(path) for path in pending)}"
            )
        time.sleep(0.25)


def serve_model_fleet(
    *,
    config: dict,
    gpus: list[str],
    socket_dir: str | Path,
    lanes_per_gpu: int = DEFAULT_LANES_PER_GPU,
) -> None:
    if not gpus:
        raise ValueError("at least one GPU is required")
    context = mp.get_context("spawn")
    endpoints = model_service_endpoints(
        socket_dir,
        gpus,
        lanes_per_gpu=lanes_per_gpu,
    )
    for endpoint in endpoints:
        address = endpoint.address
        if not address.exists():
            continue
        try:
            connection = Client(
                str(address), family="AF_UNIX", authkey=_AUTHKEY
            )
            connection.send({"operation": "status"})
            connection.recv()
            connection.close()
        except (ConnectionError, EOFError, OSError):
            address.unlink()
            address.with_suffix(".ready.json").unlink(missing_ok=True)
        else:
            raise RuntimeError(
                f"model service is already running at {address}"
            )
    processes = []
    startup_stagger = float(
        config.get("runtime", {})
        .get("model_service", {})
        .get("worker_start_stagger_seconds", 0)
    )
    if startup_stagger < 0:
        raise ValueError(
            "model service worker_start_stagger_seconds must be >= 0"
        )
    for index, endpoint in enumerate(endpoints):
        process_name = f"hand-model-service-gpu-{endpoint.gpu}"
        if endpoint.lane:
            process_name += f"-lane-{endpoint.lane}"
        process = context.Process(
            target=_serve_one,
            args=(
                config,
                endpoint.gpu,
                str(endpoint.address),
                index * startup_stagger,
            ),
            name=process_name,
        )
        process.start()
        processes.append(process)
    addresses = [endpoint.address for endpoint in endpoints]
    _wait_until_ready(config, addresses, processes)
    print(
        json.dumps(
            {
                "status": "ready",
                "gpus": gpus,
                "lanes_per_gpu": lanes_per_gpu,
                "socket_dir": str(socket_dir),
                "profile": config.get("profile"),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        while not stopping and any(
            process.is_alive() for process in processes
        ):
            time.sleep(0.5)
    finally:
        for endpoint in endpoints:
            address = endpoint.address
            if not address.exists():
                continue
            try:
                connection = Client(
                    str(address), family="AF_UNIX", authkey=_AUTHKEY
                )
                connection.send({"operation": "shutdown"})
                connection.recv()
                connection.close()
            except (ConnectionError, EOFError, OSError):
                pass
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join()


def fleet_status(
    *,
    config: dict,
    gpus: list[str],
    socket_dir: str | Path,
    lanes_per_gpu: int = DEFAULT_LANES_PER_GPU,
) -> list[dict[str, Any]]:
    results = []
    endpoints = model_service_endpoints(
        socket_dir,
        gpus,
        lanes_per_gpu=lanes_per_gpu,
    )
    for endpoint in endpoints:
        client = ModelServiceClient(endpoint.address, config)
        status = client.request({"operation": "status"})
        status["gpu"] = endpoint.gpu
        status["lane"] = endpoint.lane
        status["socket"] = str(endpoint.address)
        results.append(status)
    return results

import json
import os
from fractions import Fraction
from pathlib import Path

import pytest

from hand_process.config import load_config
from hand_process.model_service import (
    InpaintRequest,
    InpaintResult,
    inpainter_backend_name,
    inpainter_config,
    model_service_endpoints,
    model_config_digest,
    service_operations,
    socket_path,
    validate_service_ready,
)


def test_model_config_digest_ignores_runtime_output_changes():
    first = {
        "profile": "edge",
        "models": {"sam2": {"checkpoint": "model.pt"}},
        "run_root": "first",
    }
    second = {
        "profile": "edge",
        "models": {"sam2": {"checkpoint": "model.pt"}},
        "run_root": "second",
    }
    assert model_config_digest(first) == model_config_digest(second)


def test_model_config_digest_detects_profile_or_model_changes():
    base = {
        "profile": "edge",
        "models": {"sam2": {"checkpoint": "model.pt"}},
    }
    changed_profile = {**base, "profile": "server"}
    changed_model = {
        **base,
        "models": {"sam2": {"checkpoint": "other.pt"}},
    }
    assert model_config_digest(base) != model_config_digest(changed_profile)
    assert model_config_digest(base) != model_config_digest(changed_model)


def test_model_config_digest_detects_service_capability_changes():
    base = {
        "profile": "server",
        "models": {"propainter": {}},
        "runtime": {"model_service": {"operations": ["segment", "inpaint"]}},
    }
    changed = {
        **base,
        "runtime": {"model_service": {"operations": ["inpaint"]}},
    }
    assert model_config_digest(base) != model_config_digest(changed)


def test_service_operations_default_and_validation():
    assert service_operations({}) == ("segment", "inpaint")
    assert service_operations(
        {"runtime": {"model_service": {"operations": ["inpaint"]}}}
    ) == ("inpaint",)
    with pytest.raises(ValueError, match="non-empty"):
        service_operations({"runtime": {"model_service": {"operations": []}}})
    with pytest.raises(ValueError, match="unsupported"):
        service_operations({"runtime": {"model_service": {"operations": ["train"]}}})
    with pytest.raises(ValueError, match="duplicates"):
        service_operations(
            {"runtime": {"model_service": {"operations": ["inpaint", "inpaint"]}}}
        )


def test_inpainter_config_prefers_new_key_and_supports_legacy():
    legacy = {"models": {"propainter": {"backend": "propainter"}}}
    current = {
        "models": {
            "propainter": {"backend": "legacy"},
            "inpainter": {"backend": "new"},
        }
    }
    assert inpainter_config(legacy)["backend"] == "propainter"
    assert inpainter_config(current)["backend"] == "new"
    assert inpainter_backend_name(current) == "new"


def test_inpainter_config_merges_same_backend_overrides():
    config = {
        "models": {
            "propainter": {
                "backend": "propainter",
                "repo": "/repo",
                "weights": "/weights",
                "raft_iter": 20,
            },
            "inpainter": {
                "backend": "propainter",
                "raft_iter": 8,
            },
        }
    }
    assert inpainter_config(config) == {
        "backend": "propainter",
        "repo": "/repo",
        "weights": "/weights",
        "raft_iter": 8,
    }


def test_server_starts_one_lane_with_bounded_cpu_workers():
    config = load_config(Path(__file__).parents[1] / "configs/server.yaml")

    assert config["runtime"]["model_service"]["lanes_per_gpu"] == 1
    assert config["runtime"]["model_service"]["worker_start_stagger_seconds"] == 0
    assert config["runtime"]["worker_start_stagger_seconds"] == 0
    assert config["runtime"]["preprocess_workers"] == 4
    assert config["models"]["inpainter"]["dynamic_mask_ranges"] is True
    assert config["models"]["inpainter"]["mask_context_frames"] == 11
    assert config["models"]["inpainter"]["window_batch_size"] == 1


def test_profiled_inpaint_rpc_round_trip():
    request = InpaintRequest(
        frame_dir=Path("/frames"),
        mask_dir=Path("/masks"),
        output_root=Path("/output"),
        fps=Fraction(25, 1),
        log_name="episode",
        global_start=120,
        seed=7,
    )
    assert InpaintRequest.from_rpc(request.to_rpc()) == request

    result = InpaintResult(
        output_dir=Path("/output/frames"),
        backend="propainter",
        num_frames=12,
        wall_seconds=3.5,
        stage_seconds={"inference": 3.0},
        peak_cuda_allocated_bytes=10,
        peak_cuda_reserved_bytes=20,
    )
    assert InpaintResult.from_rpc(result.to_rpc()) == result


def test_profiled_inpaint_result_accepts_legacy_server_response():
    result = InpaintResult.from_rpc("/legacy/output")
    assert result.output_dir == Path("/legacy/output")
    assert result.backend == "legacy"


def test_socket_path_is_per_physical_gpu(tmp_path):
    assert socket_path(tmp_path, "7") == tmp_path / "gpu-7.sock"


def test_model_service_endpoints_add_lanes_without_changing_lane_zero(tmp_path):
    endpoints = model_service_endpoints(
        tmp_path,
        ["7"],
        lanes_per_gpu=2,
    )

    assert [(endpoint.gpu, endpoint.lane) for endpoint in endpoints] == [
        ("7", 0),
        ("7", 1),
    ]
    assert endpoints[0].address == tmp_path / "gpu-7.sock"
    assert endpoints[1].address == tmp_path / "gpu-7-lane-1.sock"


def test_model_service_endpoints_default_to_one_lane(tmp_path):
    endpoints = model_service_endpoints(tmp_path, ["7"])
    assert [endpoint.lane for endpoint in endpoints] == [0]


def test_model_service_endpoints_validate_lane_count_and_gpu_list(tmp_path):
    with pytest.raises(ValueError, match="at least 1"):
        model_service_endpoints(tmp_path, ["0"], lanes_per_gpu=0)
    with pytest.raises(ValueError, match="duplicates"):
        model_service_endpoints(tmp_path, ["0", "0"])


def test_validate_service_ready_uses_nonblocking_readiness_file(tmp_path):
    config = {"profile": "edge", "models": {"sam2": {}}}
    socket = tmp_path / "gpu-0.sock"
    socket.touch()
    socket.with_suffix(".ready.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "config_digest": model_config_digest(config),
            }
        ),
        encoding="utf-8",
    )
    assert validate_service_ready(socket, config)["pid"] == os.getpid()


def test_validate_service_ready_rejects_wrong_config(tmp_path):
    config = {"profile": "edge", "models": {"sam2": {}}}
    socket = tmp_path / "gpu-0.sock"
    socket.touch()
    socket.with_suffix(".ready.json").write_text(
        json.dumps({"pid": os.getpid(), "config_digest": "wrong"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="does not match"):
        validate_service_ready(socket, config)

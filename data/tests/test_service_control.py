import json

from hand_process.service_control import (
    discover_socket_dirs,
    stop_project_services,
)


def test_discover_socket_dirs_finds_generated_runtime_socket(tmp_path):
    generated = tmp_path / "runtime" / "generated-configs"
    generated.mkdir(parents=True)
    (generated / "run.yaml").write_text(
        "runtime:\n  model_service:\n    socket_dir: runtime/model-services/test\n",
        encoding="utf-8",
    )

    assert (
        tmp_path / "runtime" / "model-services" / "test"
    ).resolve() in discover_socket_dirs(tmp_path)


def test_stop_project_services_skips_unowned_pid(tmp_path):
    socket_dir = tmp_path / "runtime" / "model-services" / "test"
    socket_dir.mkdir(parents=True)
    (socket_dir / "gpu-0.sock").touch()
    (socket_dir / "gpu-0.ready.json").write_text(
        json.dumps({"pid": 999999999}),
        encoding="utf-8",
    )

    result = stop_project_services(tmp_path)

    assert result["stopped"] == 0
    assert len(result["skipped"]) == 1

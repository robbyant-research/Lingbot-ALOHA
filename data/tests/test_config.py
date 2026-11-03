from pathlib import Path

import pytest

from hand_process.config import load_config


def test_config_extends_deep_merges_profiles(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        "schema_version: 1\nruntime:\n  workers: 8\n  threads: 4\n",
        encoding="utf-8",
    )
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        "extends: base.yaml\nruntime:\n  workers: 1\nprofile: edge\n",
        encoding="utf-8",
    )

    config = load_config(profile)

    assert config["runtime"] == {"workers": 1, "threads": 4}
    assert config["profile"] == "edge"
    assert config["_config_path"] == str(profile.resolve())


def test_config_extends_rejects_cycle(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("extends: second.yaml\nschema_version: 1\n")
    second.write_text("extends: first.yaml\nschema_version: 1\n")

    with pytest.raises(ValueError, match="extends cycle"):
        load_config(first)


def test_default_and_grounding_dino_profiles_are_explicit():
    default = load_config("configs/server.yaml")
    grounding = load_config(
        "configs/grounding_dino.yaml"
    )

    assert default["project_root"] == str(Path(__file__).resolve().parents[1])
    assert default["models"]["detector"].get("backend", "care_ego") == "care_ego"
    assert default["models"]["segmenter"].get("backend", "sam2") == "sam2"
    assert default["models"]["inpainter"]["backend"] == "diffueraser"
    assert default["models"]["inpainter"]["max_img_size"] == 768
    assert default["models"]["inpainter"]["attention_backend"] == "sdpa"
    assert default["models"]["inpainter"]["vae_decode_batch_size"] == 8
    assert (
        default["models"]["inpainter"][
            "skip_redundant_pre_inference_decode"
        ]
        is True
    )

    detector = grounding["models"]["detector"]
    assert detector["backend"] == "grounding_dino_sam2"
    assert detector["acknowledge_non_ego_hand_removal"] is True
    assert grounding["models"]["segmenter"]["keyframe_interval"] == 5
    assert (
        grounding["runtime"]["model_service"]["socket_dir"]
        != default["runtime"]["model_service"]["socket_dir"]
    )


def test_paths_resolve_independently_of_current_directory(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HAND_PROCESS_ASSET_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("HAND_PROCESS_WORK_ROOT", str(tmp_path / "scratch"))
    config = load_config(root / "configs/server.yaml")
    assert config["work_root"] == str(tmp_path / "scratch")
    assert config["asset_root"] == str(tmp_path / "models")
    assert config["models"]["inpainter"]["repo"] == str(root / "vendor/diffueraser")
    assert config["models"]["inpainter"]["base_model_path"].startswith(str(tmp_path / "models"))
    assert config["models"]["segmenter"]["config"] == "configs/sam2.1/sam2.1_hiera_l.yaml"
    assert Path(config["models"]["detector"]["config"]).is_file()

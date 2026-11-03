import hashlib
import io
import json

import pytest

from hand_process import assets


def _record(payload=b"weights"):
    return {
        "relative_path": "sam2/model.pt",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "source": "https://example.org/model.pt",
    }


def test_verified_download_rejects_corruption_without_replacing_file(tmp_path, monkeypatch):
    destination = tmp_path / "model.pt"
    destination.write_bytes(b"previous")
    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"corrupt"))
    with pytest.raises(RuntimeError, match="integrity"):
        assets._download_verified(_record(), destination)
    assert destination.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [destination]


def test_download_validates_and_reuses_existing_asset(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "assets": [_record()]}))
    calls = []

    def request(*args, **kwargs):
        calls.append(args)
        return io.BytesIO(b"weights")

    monkeypatch.setattr(assets.urllib.request, "urlopen", request)
    config = {"asset_root": str(tmp_path / "models")}
    for _ in range(2):
        result = assets.download_assets(config, manifest)
        assert all(record["valid"] for record in result["assets"])
    assert len(calls) == 1


def test_asset_destination_rejects_path_and_symlink_escape(tmp_path):
    with pytest.raises(ValueError, match="inside"):
        assets._asset_destination(tmp_path, "../other.pt")
    (tmp_path / "escape").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        assets._asset_destination(tmp_path, "escape/model.pt")


def test_profile_selects_diffusion_weights_only_when_required():
    server = {"models": {"inpainter": {"backend": "diffueraser"}}}
    low_memory = {"models": {"propainter": {}}}
    assert any(r["relative_path"].startswith("diffueraser/") for r in assets._selected_records(server))
    assert not any(r["relative_path"].startswith("diffueraser/") for r in assets._selected_records(low_memory))


def test_manifest_keeps_expected_hash_when_file_is_corrupt(tmp_path):
    spec = _record()
    (tmp_path / "sam2").mkdir()
    (tmp_path / spec["relative_path"]).write_bytes(b"invalid")
    record = assets.build_asset_manifest(tmp_path, records=[spec])["assets"][0]
    assert record["sha256"] == spec["sha256"]
    assert record["actual_sha256"] != spec["sha256"]
    assert record["valid"] is False

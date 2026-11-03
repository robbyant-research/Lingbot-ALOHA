from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from .fs import relative_symlink


MANIFEST_PATH = Path(__file__).resolve().parents[1] / "model_assets.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_file(path: Path, *, expected_size: int, expected_sha256: str) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == expected_size
        and sha256_file(path) == expected_sha256
    )


def _asset_records(manifest_path: str | Path = MANIFEST_PATH) -> list[dict]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported asset manifest schema")
    return manifest["assets"]


def _selected_records(config: dict, manifest_path: str | Path = MANIFEST_PATH) -> list[dict]:
    models = config.get("models", {})
    detector = models.get("detector", {}).get("backend", "care_ego")
    inpainter = models.get("inpainter", {}).get("backend", "propainter")
    return [
        record for record in _asset_records(manifest_path)
        if (not record["relative_path"].startswith("diffueraser/") or inpainter == "diffueraser")
        and (not record["relative_path"].startswith("care_ego/") or detector == "care_ego")
    ]


def _asset_destination(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"asset path must stay inside the asset directory: {relative}")
    destination = root / path
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"asset path resolves outside the asset directory: {relative}")
    return destination


def build_asset_manifest(asset_root: str | Path, *, records: list[dict] | None = None) -> dict:
    root = Path(asset_root)
    checked = []
    for spec in _asset_records() if records is None else records:
        path = _asset_destination(root, spec["relative_path"])
        record = dict(spec, path=str(path), valid=False)
        if path.is_file():
            record["actual_size"] = path.stat().st_size
            record["actual_sha256"] = sha256_file(path)
            record["valid"] = (
                record["actual_size"] == spec["size"]
                and record["actual_sha256"] == spec["sha256"]
            )
        checked.append(record)
    return {"schema_version": 1, "assets": checked}


def validate_assets(asset_root: str | Path, *, records: list[dict] | None = None) -> dict:
    manifest = build_asset_manifest(asset_root, records=records)
    invalid = [record["relative_path"] for record in manifest["assets"] if not record["valid"]]
    if invalid:
        raise RuntimeError(
            "missing or invalid model assets: " + ", ".join(invalid)
            + "; see weights/README.md"
        )
    return manifest


def validate_config_assets(config: dict) -> dict:
    """Check the weights selected by the profile before starting GPU services."""
    manifest = validate_assets(config["asset_root"], records=_selected_records(config))
    detector = config.get("models", {}).get("detector", {})
    if detector.get("backend") == "grounding_dino_sam2":
        root = Path(detector["model_dir"])
        required = ["config.json", "model.safetensors", "preprocessor_config.json", "tokenizer_config.json"]
        if any(not (root / name).is_file() for name in required):
            raise RuntimeError(f"Grounding DINO assets are incomplete: {root}; see weights/README.md")
    link_propainter_weights(config)
    return manifest


def _download_verified(record: dict, destination: Path) -> None:
    """Publish an asset only after its size and SHA-256 match the manifest."""
    url = record["source"]
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("model asset sources must use HTTPS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=destination.name + ".download-", dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        if parsed.hostname == "drive.google.com":
            try:
                import gdown
            except ImportError as error:
                raise RuntimeError("Install gdown or download CaRe-Ego manually; see weights/README.md") from error
            if gdown.download(url=url, output=str(temporary), fuzzy=True, quiet=False) is None:
                raise RuntimeError("CaRe-Ego download failed; see weights/README.md for manual retrieval")
        else:
            request = urllib.request.Request(url, headers={"User-Agent": "lingbot-hand-removal/1"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
        if not _valid_file(temporary, expected_size=int(record["size"]), expected_sha256=record["sha256"]):
            raise RuntimeError(f"asset integrity validation failed: {record['relative_path']}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def download_assets(config: dict, manifest_path: str | Path = MANIFEST_PATH) -> dict:
    records = _selected_records(config, manifest_path)
    for record in records:
        destination = _asset_destination(Path(config["asset_root"]), record["relative_path"])
        if _valid_file(destination, expected_size=int(record["size"]), expected_sha256=record["sha256"]):
            print(f"[assets] valid, skipping: {record['relative_path']}", file=sys.stderr)
            continue
        print(f"[assets] downloading: {record['relative_path']}", file=sys.stderr)
        _download_verified(record, destination)
    result = validate_assets(config["asset_root"], records=records)
    link_propainter_weights(config)
    return result


def link_propainter_weights(config: dict) -> None:
    model = config.get("models", {}).get("propainter")
    if not model:
        return
    repo = Path(model["repo"])
    root = Path(model["weights"])
    for name in ("ProPainter.pth", "raft-things.pth", "recurrent_flow_completion.pth"):
        relative_symlink(root / name, repo / "weights" / name)

"""Checksum logic of scripts/fetch_models.py, without network access."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_models.py"
_spec = importlib.util.spec_from_file_location("fetch_models", _SCRIPT)
fm = importlib.util.module_from_spec(_spec)
sys.modules["fetch_models"] = fm  # dataclasses look up their module here
_spec.loader.exec_module(fm)

PAYLOAD = b"model-bytes"
GOOD = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture(autouse=True)
def fake_download(monkeypatch):
    def _download(url: str, dest: Path, retries: int = 3) -> None:
        dest.write_bytes(PAYLOAD)

    monkeypatch.setattr(fm, "download", _download)


def _src(digest: str) -> "fm.Source":
    return fm.Source(key="yunet", url="https://example.invalid/m.onnx", algo="sha256", digest=digest)


def _target(models_dir: Path) -> Path:
    return models_dir / fm.MODEL_FILES["yunet"]


def test_wrong_digest_fails_and_leaves_nothing(tmp_path: Path):
    with pytest.raises(SystemExit):
        fm.fetch(_src("0" * 64), tmp_path)
    target = _target(tmp_path)
    assert not target.exists()
    assert list(target.parent.iterdir()) == []  # no leftover temp file


def test_right_digest_writes_target_0644(tmp_path: Path):
    fm.fetch(_src(GOOD), tmp_path)
    target = _target(tmp_path)
    assert target.read_bytes() == PAYLOAD
    assert target.stat().st_mode & 0o777 == 0o644
    assert list(target.parent.iterdir()) == [target]


def test_corrupted_existing_target_is_replaced(tmp_path: Path):
    target = _target(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"corrupted")
    fm.fetch(_src(GOOD), tmp_path)
    assert target.read_bytes() == PAYLOAD


def test_main_exits_when_a_model_has_no_source(monkeypatch):
    monkeypatch.setattr(fm, "SOURCES", [s for s in fm.SOURCES if s.key != "ser"])
    monkeypatch.setattr(sys, "argv", ["fetch_models.py", "--models-dir", "/nonexistent"])
    with pytest.raises(SystemExit) as exc:
        fm.main()
    assert "ser" in str(exc.value)

"""/api/health matrix rows that can be simulated without Docker or a GPU."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import health as health_mod
from app.config import MODEL_FILES


@pytest.fixture()
def models_dir(tmp_path: Path) -> Path:
    root = tmp_path / "models"
    for rel in MODEL_FILES.values():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    return root


@pytest.fixture()
def make_client(monkeypatch, tmp_path: Path, models_dir: Path):
    def _make(variant: str = "cpu", gpu_name: str | None = None) -> TestClient:
        monkeypatch.setenv("MODELS_DIR", str(models_dir))
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("APP_VARIANT", variant)
        monkeypatch.setattr(health_mod, "probe_cuda", lambda: gpu_name)
        health_mod._cached_device.cache_clear()
        from app.main import app

        return TestClient(app)

    yield _make
    health_mod._cached_device.cache_clear()


ALL_TRUE = {"yunet": True, "fer": True, "vad": True, "ser": True}


def test_cpu_start(make_client):
    with make_client("cpu") as client:
        r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "device": "cpu", "models": ALL_TRUE}


def test_cpu_variant_never_probes_cuda(make_client, monkeypatch):
    def boom():
        raise AssertionError("CPU image must not probe CUDA")

    with make_client("cpu") as client:
        monkeypatch.setattr(health_mod, "probe_cuda", boom)
        health_mod._cached_device.cache_clear()
        assert client.get("/api/health").json()["device"] == "cpu"


def test_gpu_start(make_client):
    with make_client("gpu", gpu_name="NVIDIA GeForce RTX 3050 Laptop GPU") as client:
        body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["device"] == "cuda"
    assert body["gpu_name"] == "NVIDIA GeForce RTX 3050 Laptop GPU"
    assert "warning" not in body
    assert body["models"] == ALL_TRUE


def test_gpu_image_without_gpu_falls_back(make_client):
    with make_client("gpu", gpu_name=None) as client:
        r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["device"] == "cpu"
    assert body["warning"] == "cuda_unavailable"
    assert "gpu_name" not in body


def test_probe_cuda_returns_none_when_subprocess_fails(monkeypatch):
    import subprocess

    def failed(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 1, stdout="", stderr="ImportError: no torch")

    monkeypatch.setattr(health_mod.subprocess, "run", failed)
    assert health_mod.probe_cuda() is None


def test_probe_cuda_returns_none_on_timeout(monkeypatch):
    import subprocess

    def hang(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs.get("timeout", 0))

    monkeypatch.setattr(health_mod.subprocess, "run", hang)
    assert health_mod.probe_cuda() is None


def test_probe_cuda_returns_gpu_name(monkeypatch):
    import subprocess

    def ok(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="NVIDIA GeForce RTX 3050\n", stderr="")

    monkeypatch.setattr(health_mod.subprocess, "run", ok)
    assert health_mod.probe_cuda() == "NVIDIA GeForce RTX 3050"


@pytest.mark.parametrize("missing", sorted(MODEL_FILES))
def test_missing_weight_degrades(make_client, models_dir: Path, missing: str):
    (models_dir / MODEL_FILES[missing]).unlink()
    with make_client("cpu") as client:
        r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    assert body["models"][missing] is False
    assert all(v for k, v in body["models"].items() if k != missing)


def test_directory_in_place_of_file_counts_as_missing(make_client, models_dir: Path):
    target = models_dir / MODEL_FILES["ser"]
    target.unlink()
    target.mkdir()
    with make_client("cpu") as client:
        body = client.get("/api/health").json()
    assert body["models"]["ser"] is False
    assert body["status"] == "degraded"


def test_ws_live_stub_accepts_and_closes(make_client):
    from starlette.websockets import WebSocketDisconnect

    with make_client("cpu") as client:
        with client.websocket_connect("/ws/live") as ws:
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_text()
    assert exc.value.code == 1000

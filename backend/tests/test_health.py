"""/api/health matrix rows that can be simulated without Docker or a GPU (fake inference worker)."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import MODEL_FILES
from inference.worker import WorkerStartError


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
    def _make(variant: str = "cpu") -> TestClient:
        monkeypatch.setenv("MODELS_DIR", str(models_dir))
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("APP_VARIANT", variant)
        from app.main import app

        return TestClient(app)

    return _make


def health(client: TestClient, wait: bool = True) -> dict:
    if wait:
        client.app.state.hub.wait_ready(5.0)
    r = client.get("/api/health")
    assert r.status_code == 200
    return r.json()


ALL_TRUE = {"yunet": True, "fer": True, "vad": True, "ser": True}
ALL_FALSE = {k: False for k in ALL_TRUE}


def test_cpu_start(make_client):
    with make_client("cpu") as client:
        assert health(client) == {"status": "ok", "device": "cpu", "models": ALL_TRUE}


def test_gpu_start(make_client, fake_worker):
    fake_worker.info = {"device": "cuda", "gpu_name": "NVIDIA GeForce RTX 3050 Laptop GPU"}
    with make_client("gpu") as client:
        body = health(client)
    assert body["status"] == "ok"
    assert body["device"] == "cuda"
    assert body["gpu_name"] == "NVIDIA GeForce RTX 3050 Laptop GPU"
    assert "warning" not in body
    assert body["models"] == ALL_TRUE


def test_gpu_image_without_gpu_falls_back(make_client, fake_worker):
    fake_worker.info = {"device": "cpu", "gpu_name": None}
    with make_client("gpu") as client:
        body = health(client)
    assert body["status"] == "ok"
    assert body["device"] == "cpu"
    assert body["warning"] == "cuda_unavailable"
    assert "gpu_name" not in body


def test_models_loading(make_client, fake_worker):
    fake_worker.start_gate = threading.Event()
    with make_client("cpu") as client:
        body = health(client, wait=False)  # startup is not blocked by model loading
        fake_worker.start_gate.set()
        assert body == {"status": "degraded", "device": "cpu", "models": ALL_FALSE, "warning": "models_loading"}
        assert health(client)["status"] == "ok"


def test_worker_failed_to_start(make_client, fake_worker):
    fake_worker.start_error = WorkerStartError("CUDAExecutionProvider requested but not active")
    with make_client("gpu") as client:
        body = health(client)
    assert body["status"] == "degraded"
    assert body["models"] == ALL_FALSE
    assert body["warning"] == "inference_failed"


def test_worker_died(make_client, fake_worker):
    with make_client("cpu") as client:
        client.app.state.hub.wait_ready(5.0)
        fake_worker.die()
        for _ in range(50):
            if client.app.state.hub.state == "failed":
                break
            threading.Event().wait(0.05)
        body = health(client, wait=False)
    assert body["status"] == "degraded"
    assert body["warning"] == "inference_failed"


@pytest.mark.parametrize("missing", sorted(MODEL_FILES))
def test_missing_weight_degrades(make_client, models_dir: Path, missing: str):
    (models_dir / MODEL_FILES[missing]).unlink()
    with make_client("cpu") as client:
        body = health(client)
    assert body["status"] == "degraded"
    assert body["models"][missing] is False
    assert all(v for k, v in body["models"].items() if k != missing)


def test_directory_in_place_of_file_counts_as_missing(make_client, models_dir: Path):
    target = models_dir / MODEL_FILES["ser"]
    target.unlink()
    target.mkdir()
    with make_client("cpu") as client:
        body = health(client)
    assert body["models"]["ser"] is False
    assert body["status"] == "degraded"

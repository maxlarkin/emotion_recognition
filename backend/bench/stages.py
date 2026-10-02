"""Per-stage benchmark of the inference core (story 1.2; CAP-6 groundwork).

    python -m bench.stages [--device auto|cpu|cuda] [--seconds N] [--json out.json]

Phase 1 (separate spawn process): p50/p95 ms of every stage on synthetic input -
detection on a 640x360 frame, FER at batch 1/4/8, detection+FER per frame, VAD per
512-sample chunk, SER per 3 s window.
Phase 2: the real InferenceWorker, end-to-end frame latency (submit -> result received)
with 1 and 3 synthetic sources at 8 fps each, plus 500 ms chunks of real speech
(bench/data/speech.wav, looped), so VAD and SER run in the worker too.
Throughout: peak VRAM of the whole device (NVML) and peak RSS of this process tree (psutil).

Exit code 0 on success, 1 on error, 2 on a stop condition (see `violations`): no face in the
bench frame, a starved source, and on CUDA models' VRAM (peak - baseline) > 4096 MB, peak VRAM
> 95% of the device, or detection+FER p95 > 100 ms.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

VRAM_LIMIT_MB = 4096.0
DETFER_P95_LIMIT_MS = 100.0
FPS = 8.0
AUDIO_CHUNK_S = 0.5
FACE_IMAGE = Path(__file__).resolve().parent / "data" / "face.jpg"
SPEECH_WAV = Path(__file__).resolve().parent / "data" / "speech.wav"


# ------------------------------------------------------------------------------ inputs


def synthetic_frame(width: int = 640, height: int = 360) -> np.ndarray:
    """640x360 BGR frame with one real face (bench/data/face.jpg) on a gradient background."""
    import cv2

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = np.linspace(40, 200, width, dtype=np.uint8)[np.newaxis, :, np.newaxis]
    face = cv2.imread(str(FACE_IMAGE))
    if face is None:
        raise FileNotFoundError(f"bench face image missing or unreadable: {FACE_IMAGE}")
    side = min(height, face.shape[0])
    face = cv2.resize(face, (side, side))
    x0 = (width - side) // 2
    frame[height - side :, x0 : x0 + side] = face
    return frame


def speech_audio(seconds: float) -> np.ndarray:
    """Real speech (bench/data/speech.wav, PCM16 16 kHz mono) looped to `seconds`."""
    with wave.open(str(SPEECH_WAV), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
            raise ValueError(f"{SPEECH_WAV} must be PCM16 mono 16 kHz")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    n = int(seconds * 16000)
    return np.resize(pcm, n).astype(np.int16)


# ------------------------------------------------------------------------------ stats / memory


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "p50": float("nan"), "p95": float("nan"), "mean": float("nan")}
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": int(a.size),
        "p50": round(float(np.percentile(a, 50)), 3),
        "p95": round(float(np.percentile(a, 95)), 3),
        "mean": round(float(a.mean()), 3),
    }


def time_stage(fn: Callable[[], Any], seconds: float, warmup: int = 3, min_iters: int = 20, max_iters: int = 5000) -> dict:
    for _ in range(warmup):
        fn()
    times: list[float] = []
    deadline = time.perf_counter() + seconds
    while len(times) < max_iters and (len(times) < min_iters or time.perf_counter() < deadline):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return percentiles(times)


class MemorySampler:
    """Background sampler: whole-device VRAM via NVML (None without a GPU) and RSS of this process tree."""

    def __init__(self, interval: float = 0.05):
        import psutil

        self._psutil = psutil
        self._proc = psutil.Process(os.getpid())
        self.interval = interval
        self.peak_rss_mb = 0.0
        self.peak_vram_mb: Optional[float] = None
        self.baseline_vram_mb: Optional[float] = None
        self.total_vram_mb: Optional[float] = None
        self._nvml_handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.baseline_vram_mb = self._vram()
            self.total_vram_mb = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle).total / 2**20
            self.peak_vram_mb = self.baseline_vram_mb
        except Exception:  # noqa: BLE001 - CPU image / no driver: VRAM reported as null
            self._nvml_handle = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _vram(self) -> Optional[float]:
        if self._nvml_handle is None:
            return None
        return self._nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle).used / 2**20

    def _rss(self) -> float:
        total = 0
        for p in [self._proc, *self._proc.children(recursive=True)]:
            try:
                total += p.memory_info().rss
            except self._psutil.Error:
                pass
        return total / 2**20

    def sample(self) -> None:
        self.peak_rss_mb = max(self.peak_rss_mb, self._rss())
        v = self._vram()
        if v is not None:
            self.peak_vram_mb = max(self.peak_vram_mb or 0.0, v)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.sample()
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "MemorySampler":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(2)


# ------------------------------------------------------------------------------ phase 1: stages


def _stages_main(out: "mp.Queue", device_req: str, stage_s: float, settings) -> None:
    try:
        out.put(("ok", run_stages(device_req, stage_s, settings)))
    except Exception as exc:  # noqa: BLE001
        out.put(("error", f"{type(exc).__name__}: {exc}"))


def run_stages(device_req: str, stage_s: float, settings) -> dict:
    from inference.detector import FaceDetector
    from inference.fer import FacialEmotion, crop_rgb
    from inference.ser import SpeechEmotion
    from inference.vad import CHUNK, SileroVAD, VadStream
    from inference.worker import WorkerConfig, check_model_files, select_device

    check_model_files(settings)
    device = select_device(device_req, settings.variant)
    cfg = WorkerConfig()
    det = FaceDetector(settings.model_path("yunet"))
    fer = FacialEmotion(settings.model_path("fer"), device=device, cuda_batch=cfg.fer_cuda_batch)
    vad = SileroVAD(settings.model_path("vad"))
    ser = SpeechEmotion(settings.model_path("ser"), device=device, fp16=cfg.ser_fp16)

    frame = synthetic_frame()
    boxes = det.detect(frame)
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 255, (160, 140, 3), dtype=np.uint8) for _ in range(8)]
    audio = speech_audio(3.0).astype(np.float32) / 32768.0
    stream = VadStream()
    vad_chunk = audio[:CHUNK].copy()

    def det_fer() -> None:
        b = det.detect(frame)
        fer.predict([crop_rgb(frame, x) for x in b])

    stages = {
        "detect_640x360": time_stage(lambda: det.detect(frame), stage_s),
        "fer_b1": time_stage(lambda: fer.predict(crops[:1]), stage_s),
        "fer_b4": time_stage(lambda: fer.predict(crops[:4]), stage_s),
        "fer_b8": time_stage(lambda: fer.predict(crops[:8]), stage_s),
        "detect_plus_fer": time_stage(det_fer, stage_s),
        "vad_512": time_stage(lambda: vad.chunk_prob(stream, vad_chunk), stage_s),
        "ser_3s": time_stage(lambda: ser.predict(audio[np.newaxis, :]), stage_s, warmup=2, min_iters=5),
    }
    return {"device": device, "ser_fp16": ser.fp16, "faces_in_frame": len(boxes), "stages": stages}


# ------------------------------------------------------------------------------ phase 2: worker e2e


def run_e2e(worker, n_sources: int, seconds: float, tag: str) -> dict:
    from inference.types import FrameResult

    frame = synthetic_frame()
    audio = speech_audio(60.0)
    chunk_len = int(AUDIO_CHUNK_S * 16000)
    sids = [f"{tag}-s{i + 1}" for i in range(n_sources)]
    for sid in sids:
        worker.register_source(sid)
    worker.results()  # drain leftovers of an earlier run

    lat: dict[str, list[float]] = {s: [] for s in sids}
    submitted = {s: 0 for s in sids}
    rejected = {s: 0 for s in sids}
    audio_results = {s: 0 for s in sids}
    errors: list[str] = []
    timings: dict[str, list[float]] = {"queue": [], "detect": [], "fer": [], "total": [], "batch_frames": []}

    def collect(timeout: float) -> None:
        now_results = worker.results(timeout=timeout)
        recv = time.time()
        for r in now_results:
            if isinstance(r, FrameResult):
                if r.source_id in lat:
                    lat[r.source_id].append((recv - r.ts) * 1000.0)
                    for k, v in timings.items():
                        v.append(r.timings.get(k, 0.0))
            elif hasattr(r, "speech"):
                if r.source_id in audio_results:
                    audio_results[r.source_id] += 1
            elif getattr(r, "source_id", None) in lat:  # ErrorResult of this run only
                errors.append(str(r))

    t_start = time.time()
    next_frame = t_start
    next_audio = t_start
    audio_pos = 0
    while time.time() - t_start < seconds:
        now = time.time()
        if now >= next_frame:
            for sid in sids:
                ok = worker.submit_frame(sid, frame, ts=time.time())
                submitted[sid] += 1
                rejected[sid] += 0 if ok else 1
            next_frame += 1.0 / FPS
            if next_frame < time.time():  # fell behind: skip missed ticks instead of bursting
                next_frame = time.time() + 1.0 / FPS
        if now >= next_audio:
            seg = audio[audio_pos : audio_pos + chunk_len]
            audio_pos = (audio_pos + chunk_len) % (audio.shape[0] - chunk_len)
            for sid in sids:
                worker.submit_audio(sid, seg, ts=now)
            next_audio += AUDIO_CHUNK_S
            if next_audio < time.time():
                next_audio = time.time() + AUDIO_CHUNK_S
        collect(timeout=max(0.0, min(next_frame, next_audio) - time.time()))
    elapsed = time.time() - t_start
    drain_until = time.time() + 3.0
    while time.time() < drain_until:
        collect(timeout=0.2)
    for sid in sids:
        worker.unregister_source(sid)

    per_source = {}
    for sid in sids:
        st = percentiles(lat[sid])
        st.update(
            submitted=submitted[sid],
            rejected=rejected[sid],
            results=len(lat[sid]),
            fps=round(len(lat[sid]) / elapsed, 2),
            audio_results=audio_results[sid],
        )
        per_source[sid] = st
    all_lat = [x for s in sids for x in lat[s]]
    counts = [len(lat[s]) for s in sids]
    return {
        "sources": n_sources,
        "seconds": round(elapsed, 2),
        "latency_ms": percentiles(all_lat),
        "min_fps": min(per_source[s]["fps"] for s in sids),
        "starved": bool(min(counts) == 0 or min(counts) < 0.5 * max(counts)),
        "worker_timings_ms": {k: percentiles(v) for k, v in timings.items()},
        "per_source": per_source,
        "errors": errors[:10],
    }


# ------------------------------------------------------------------------------ main


def violations(report: dict) -> list[str]:
    """The story's stop conditions; any hit -> exit code 2."""
    found = []
    if report.get("faces_in_frame", 0) == 0:
        found.append("no face detected in the bench frame (detection/FER not exercised)")
    for name, e in (report.get("e2e") or {}).items():
        if e.get("starved"):
            found.append(f"e2e {name}: a source was starved")
    if report.get("device") == "cuda":
        peak, base, total = report.get("peak_vram_mb"), report.get("vram_baseline_mb"), report.get("vram_total_mb")
        if peak is not None and base is not None and peak - base > VRAM_LIMIT_MB:
            found.append(f"VRAM used by the models {peak - base:.0f} MB > {VRAM_LIMIT_MB:.0f} MB")
        if peak is not None and total and peak > 0.95 * total:
            found.append(f"peak VRAM {peak:.0f} MB > 95% of the device ({total:.0f} MB)")
        p95 = report.get("stages", {}).get("detect_plus_fer", {}).get("p95", 0.0)
        if p95 > DETFER_P95_LIMIT_MS:
            found.append(f"detection+FER p95 {p95:.1f} ms > {DETFER_P95_LIMIT_MS:.0f} ms")
    return found


def fmt(st: dict) -> str:
    return f"p50 {st['p50']:8.2f}  p95 {st['p95']:8.2f}  mean {st['mean']:8.2f}  n={st['n']}"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.stages", description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seconds", type=float, default=20.0, help="approximate total measuring time")
    parser.add_argument("--json", type=Path, help="write the report as JSON to this file")
    args = parser.parse_args(argv)

    from app.config import get_settings
    from inference.worker import InferenceWorker, WorkerConfig, WorkerStartError

    settings = get_settings()
    stage_s = max(0.5, 0.08 * args.seconds)  # 7 stages ~ 0.56 N
    e2e_s = max(2.0, 0.22 * args.seconds)  # 2 runs ~ 0.44 N
    report: dict[str, Any] = {"requested_device": args.device, "variant": settings.variant}

    with MemorySampler() as mem:
        report["vram_baseline_mb"] = mem.baseline_vram_mb
        report["vram_total_mb"] = mem.total_vram_mb
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_stages_main, args=(q, args.device, stage_s, settings))
        p.start()
        status, payload = "error", "stage benchmark timed out"
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            try:
                status, payload = q.get(timeout=0.5)
                break
            except queue.Empty:
                if not p.is_alive():
                    status, payload = "error", f"stage benchmark process died (exit code {p.exitcode})"
                    break
        p.join(30)
        if p.is_alive():
            p.terminate()
            p.join(5)
        if status != "ok":
            print(f"ERROR: {payload}", file=sys.stderr)
            return 1
        report.update(payload)
        mem.sample()
        report["stages_peak_vram_mb"] = mem.peak_vram_mb
        report["stages_peak_rss_mb"] = round(mem.peak_rss_mb, 1)

        worker = InferenceWorker(settings, WorkerConfig(device=args.device))
        try:
            worker.start()
        except WorkerStartError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        try:
            report["worker_device"] = worker.device
            report["e2e"] = {
                "1_source": run_e2e(worker, 1, e2e_s, "one"),
                "3_sources": run_e2e(worker, 3, e2e_s, "three"),
            }
        finally:
            worker.stop()
        mem.sample()
        report["peak_vram_mb"] = None if mem.peak_vram_mb is None else round(mem.peak_vram_mb, 1)
        report["peak_rss_mb"] = round(mem.peak_rss_mb, 1)

    device = report["device"]
    report["violations"] = found = violations(report)

    print(f"device: {device} (worker: {report.get('worker_device')}, SER fp16: {report['ser_fp16']}, "
          f"faces in bench frame: {report['faces_in_frame']})")
    print("stage                      ms")
    for name, st in report["stages"].items():
        print(f"  {name:<22} {fmt(st)}")
    for name, e in report["e2e"].items():
        print(f"  e2e {name:<18} {fmt(e['latency_ms'])}  min_fps {e['min_fps']:.2f}  starved {e['starved']}")
        for sid, st in e["per_source"].items():
            print(f"      {sid:<16} results {st['results']:4d}/{st['submitted']:<4d} fps {st['fps']:5.2f}  "
                  f"p50 {st['p50']:8.2f}  p95 {st['p95']:8.2f}  audio {st['audio_results']}")
        wt = e["worker_timings_ms"]
        print("      worker p50 ms: " + "  ".join(f"{k} {wt[k]['p50']:.1f}" for k in ("queue", "detect", "fer", "total"))
              + f"  frames/batch {wt['batch_frames']['mean']:.2f}")
        for err in e["errors"]:
            print(f"      error: {err}")
    vram = "n/a" if report["peak_vram_mb"] is None else f"{report['peak_vram_mb']:.0f} MB"
    base = "n/a" if report["vram_baseline_mb"] is None else f"{report['vram_baseline_mb']:.0f} MB"
    total = "n/a" if report["vram_total_mb"] is None else f"{report['vram_total_mb']:.0f} MB"
    print(f"peak VRAM (whole device): {vram} (baseline {base}, total {total})")
    print(f"peak RSS (process tree):  {report['peak_rss_mb']:.0f} MB")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if found:
        print("STOP: " + "; ".join(found), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

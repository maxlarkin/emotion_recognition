"""Scripted /ws/live latency client (story 1.3, CAP-1): a browser-less latency number.

    python -m bench.live_ws --url ws://localhost/ws/live --seconds 60 [--fps 8] [--image face.jpg] [--json out.json]

Sends JPEG frames (longest side <= 640, quality 80) at --fps with the client wall clock as
capture_ts, and measures result receipt - capture_ts per frame_result. Prints p50/p95 of that
latency, the server's proc_ms, the result rate and the share of results with a face.
Unlike the UI number it excludes browser capture/encode and canvas paint.

Exit code 0 when p95 < 1000 ms, 2 when it is not (or no result arrived), 1 on error.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from bench.stages import percentiles, synthetic_frame
from sources.webcam import encode_frame

P95_LIMIT_MS = 1000.0
MAX_SIDE = 640
JPEG_QUALITY = 80
LOADING_PREFIX = "Модели ещё загружаются"


def load_jpeg(image: Optional[str]) -> tuple[bytes, tuple[int, int]]:
    import cv2

    if image:
        img = cv2.imread(image)
        if img is None:
            raise SystemExit(f"cannot read image {image}")
    else:
        img = synthetic_frame()  # 640x360 with bench/data/face.jpg
    h, w = img.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise SystemExit("JPEG encoding failed")
    return buf.tobytes(), (img.shape[1], img.shape[0])


def start_session(ws, wait_ready_s: float) -> str:
    deadline = time.monotonic() + wait_ready_s
    while True:
        ws.send(json.dumps({"type": "start"}))
        msg = json.loads(ws.recv(timeout=30))
        if msg.get("type") == "session":
            return msg["session_id"]
        text = msg.get("message", "")
        if msg.get("type") == "error" and text.startswith(LOADING_PREFIX) and time.monotonic() < deadline:
            time.sleep(2.0)
            continue
        raise SystemExit(f"start failed: {msg}")


def run(url: str, seconds: float, fps: float, jpeg: bytes, warmup_s: float, wait_ready_s: float) -> dict:
    from websockets.sync.client import connect

    results: list[dict] = []
    errors: list[str] = []
    stop = threading.Event()

    with connect(url, max_size=None, open_timeout=10) as ws:
        session_id = start_session(ws, wait_ready_s)

        def receiver() -> None:
            while not stop.is_set():
                try:
                    raw = ws.recv(timeout=0.5)
                except TimeoutError:
                    continue
                except Exception:  # noqa: BLE001 - connection closed
                    return
                now = time.time() * 1000.0
                msg = json.loads(raw)
                if msg.get("type") == "frame_result":
                    results.append(
                        {
                            "frame_id": msg["frame_id"],
                            "latency_ms": now - msg["capture_ts"],
                            "proc_ms": msg["proc_ms"],
                            "faces": len(msg["faces"]),
                            "recv": now,
                        }
                    )
                elif msg.get("type") == "error":
                    errors.append(msg.get("message", ""))

        rx = threading.Thread(target=receiver, daemon=True)
        rx.start()
        t0 = time.time() * 1000.0
        period = 1.0 / fps
        n_frames = int(seconds * fps)
        next_tick = time.monotonic()
        for fid in range(1, n_frames + 1):
            ws.send(encode_frame(time.time() * 1000.0, fid, jpeg))
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        time.sleep(1.5)  # trailing results
        ws.send(json.dumps({"type": "stop"}))
        stop.set()
        rx.join(timeout=2.0)

    measured = [r for r in results if r["recv"] - t0 >= warmup_s * 1000.0]
    span_s = max(1e-9, seconds - warmup_s)
    return {
        "url": url,
        "session_id": session_id,
        "seconds": seconds,
        "fps_sent": fps,
        "frames_sent": n_frames,
        "results": len(results),
        "warmup_s": warmup_s,
        "result_rate": round(len(measured) / span_s, 2),
        "face_share": round(sum(1 for r in measured if r["faces"]) / len(measured), 3) if measured else 0.0,
        "latency_ms": percentiles([r["latency_ms"] for r in measured]),
        "proc_ms": percentiles([r["proc_ms"] for r in measured]),
        "errors": errors[:20],
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="ws://localhost/ws/live")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--image", default=None, help="image to send (default: bench/data/face.jpg on a 640x360 frame)")
    ap.add_argument("--warmup", type=float, default=2.0, help="seconds of results excluded from the statistics")
    ap.add_argument("--wait-ready", type=float, default=300.0, help="retry start while models load, seconds")
    ap.add_argument("--json", default=None, help="write the report to this file")
    args = ap.parse_args(argv)

    try:
        jpeg, size = load_jpeg(args.image)
        report = run(args.url, args.seconds, args.fps, jpeg, args.warmup, args.wait_ready)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    report["jpeg"] = {"width": size[0], "height": size[1], "bytes": len(jpeg)}

    lat, proc = report["latency_ms"], report["proc_ms"]
    print(f"frames sent {report['frames_sent']} at {args.fps:g} fps, results {report['results']}, "
          f"result rate {report['result_rate']}/s, with face {report['face_share']:.0%}")
    print(f"latency (result - capture): p50 {lat['p50']} ms, p95 {lat['p95']} ms (n={lat['n']})")
    print(f"server proc_ms:             p50 {proc['p50']} ms, p95 {proc['p95']} ms")
    if report["errors"]:
        print(f"errors: {report['errors']}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = lat["n"] > 0 and not np.isnan(lat["p95"]) and lat["p95"] < P95_LIMIT_MS
    print("PASS: p95 < 1000 ms" if ok else "FAIL: p95 >= 1000 ms or no results")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())

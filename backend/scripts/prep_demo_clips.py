"""Prepare the 20-stream demo clip set (story 1.8). Run once, with network, inside the backend image:

    docker run --rm --gpus all --network host --user "$(id -u):$(id -g)" \\
        -v "$PWD/demo-videos:/demo:z" emotion-recognition/backend:gpu \\
        python scripts/prep_demo_clips.py --demo /demo

Steps:
1. download the pinned sources (`/demo/sources.json`) into `/demo/cache/`, verify SHA-256 (or the
   original's SHA-1 from Commons as a fallback);
2. analyse each source at 5 fps with the project models (YuNet, FER, Silero VAD + GigaAM SER,
   fusion) and find negative/positive windows by the CAP-11 rule (group >= 0.6 for >= 2 s);
3. choose 20 non-overlapping 75 s segments (6 webcam, 8 sector, 6 security), favouring negative and
   positive windows and visible faces;
4. encode each for its camera profile (scale so the median face lands in the profile's band,
   crop or pad to the canvas, 10 fps H.264 GOP 20, AAC 16 kHz mono where the profile has audio)
   into `/demo/clips/camNN.mp4`, then measure the achieved face size;
5. write `/demo/segments.md`.

The analysis is cached per source (`cache/<id>.analysis.json`), so a re-run without network only
re-encodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import statistics
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

log = logging.getLogger("prep")

USER_AGENT = "emotion-recognition-demo-prep/1.0 (educational project; offline demo clip preparation)"
ANALYSIS_FPS = 5
ANALYSIS_MAX_S = 900  # long interviews: the first 15 min are enough to find segments
SEGMENT_S = 75
SEGMENT_STEP_S = 15
OUT_FPS = 10
GOP = 20
THRESHOLD = 0.6
MIN_HELD_S = 2.0
NEGATIVE = ("anger", "disgust", "fear", "sadness")

# camera-profiles.md, 20-stream demo mix
PROFILES = {
    "webcam": {"count": 6, "canvas": (640, 480), "face": (90, 200), "target": 130, "audio": True},
    "sector": {"count": 8, "canvas": (1920, 1080), "face": (50, 100), "target": 75, "audio": True},
    "security": {"count": 6, "canvas": (2560, 1440), "face": (25, 45), "target": 35, "audio": False},
}
BACKGROUND = (48, 40, 32)  # BGR, dark neutral wall around the scaled scene


# ---------------------------------------------------------------------------------------- pure parts


@dataclass
class Window:
    kind: str  # negative | positive
    track: int
    start: float
    end: float
    peak: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def find_windows(track: int, series: list[tuple[float, float]], kind: str, threshold: float = THRESHOLD,
                 min_held: float = MIN_HELD_S, max_gap: float = 0.5) -> list[Window]:
    """Runs where the value stays >= threshold for >= min_held seconds (gaps up to max_gap tolerated)."""
    out: list[Window] = []
    start: Optional[float] = None
    last: Optional[float] = None
    peak = 0.0
    for t, v in series:
        if v >= threshold and (last is None or start is None or t - last <= max_gap):
            if start is None:
                start, peak = t, v
            peak = max(peak, v)
            last = t
            continue
        if start is not None and last is not None and last - start >= min_held:
            out.append(Window(kind, track, start, last, round(peak, 3)))
        start, last, peak = (t, t, v) if v >= threshold else (None, None, 0.0)
    if start is not None and last is not None and last - start >= min_held:
        out.append(Window(kind, track, start, last, round(peak, 3)))
    return out


@dataclass
class Candidate:
    source: str
    start: float
    end: float
    windows: list[Window]
    face_ratio: float  # share of analysed frames with a face
    median_face: float  # px at analysis resolution
    face_center: tuple[float, float]  # median centre of the largest face, px
    has_audio: bool
    score: float = 0.0

    def overlaps(self, other: "Candidate") -> bool:
        return self.source == other.source and self.start < other.end and other.start < self.end


def score(c: Candidate) -> float:
    """Both kinds count the same: the model already sees negative windows almost everywhere in
    acted footage, so a clip with positive moments as well is worth more than one more negative."""
    neg = sum(1 for w in c.windows if w.kind == "negative")
    pos = sum(1 for w in c.windows if w.kind == "positive")
    return 2.0 * min(neg, 2) + 2.0 * min(pos, 2) + 2.0 * c.face_ratio


WEBCAM_MIN_SOURCE_FACE = 60  # px at analysis size: smaller faces are wide scenes, not a desk webcam


def plan_mix(cands: list[Candidate], profiles: dict = PROFILES, per_source: int = 3) -> list[tuple[str, Candidate]]:
    """Greedy: best-scoring non-overlapping segments, at most `per_source` per source; webcam first
    takes the closest faces, then sector, then security (no audio needed)."""
    for c in cands:
        c.score = score(c)
    chosen: list[tuple[str, Candidate]] = []
    used: dict[str, int] = {}

    def free(c: Candidate) -> bool:
        return used.get(c.source, 0) < per_source and not any(c.overlaps(o) for _, o in chosen)

    order = {
        "webcam": lambda c: (c.has_audio, c.score + c.median_face / 50.0),
        "sector": lambda c: (c.has_audio, c.score),
        "security": lambda c: (True, c.score),
    }
    for profile in ("webcam", "sector", "security"):
        need = profiles[profile]["count"]
        pool = [
            c for c in cands
            if c.face_ratio >= 0.5
            and (c.has_audio or not profiles[profile]["audio"])
            # the webcam crop never downscales a 640x480-ish source, so the source face must already fit the band
            and (profile != "webcam" or WEBCAM_MIN_SOURCE_FACE <= c.median_face <= profiles["webcam"]["face"][1])
        ]
        pool.sort(key=order[profile], reverse=True)
        for c in pool:
            if need == 0:
                break
            if free(c):
                chosen.append((profile, c))
                used[c.source] = used.get(c.source, 0) + 1
                need -= 1
        if need:
            raise RuntimeError(f"not enough segments for {profile}: {need} missing")
    return chosen


@dataclass
class Geometry:
    scale: float
    # crop window in the scaled frame (webcam) or paste offset on the canvas (sector/security)
    crop: Optional[tuple[int, int]] = None
    offset: Optional[tuple[int, int]] = None
    canvas: tuple[int, int] = (0, 0)
    scaled: tuple[int, int] = (0, 0)


def fit_geometry(src: tuple[int, int], face_px: float, face_center: tuple[float, float], profile: str,
                 profiles: dict = PROFILES) -> Geometry:
    """Scale so the median face becomes the profile's target size, then crop (when the scaled frame
    is larger than the canvas) around the face, or paste onto the canvas (when smaller)."""
    p = profiles[profile]
    cw, ch = p["canvas"]
    scale = p["target"] / max(face_px, 1.0)
    if profile == "webcam":
        scale = max(scale, cw / src[0], ch / src[1])  # the crop must fill 640x480
    sw, sh = max(2, int(round(src[0] * scale / 2)) * 2), max(2, int(round(src[1] * scale / 2)) * 2)
    fx, fy = face_center[0] * scale, face_center[1] * scale
    if sw >= cw and sh >= ch:
        x = int(min(max(fx - cw / 2, 0), sw - cw))
        y = int(min(max(fy - ch / 2.5, 0), sh - ch))
        return Geometry(scale, crop=(x, y), canvas=(cw, ch), scaled=(sw, sh))
    if sw > cw or sh > ch:  # wider (or taller) than the canvas on one side only: crop that side around the face
        x = int(min(max(fx - cw / 2, 0), sw - cw)) if sw > cw else 0
        y = int(min(max(fy - ch / 2.5, 0), sh - ch)) if sh > ch else 0
        ox = 0 if sw > cw else int(max(0, min(cw - sw, cw / 2 - fx)))
        oy = 0 if sh > ch else int(max(0, min(ch - sh, ch / 3 - fy)))
        return Geometry(scale, crop=(x, y), offset=(ox, oy), canvas=(cw, ch), scaled=(sw, sh))
    # pasted scene: faces around the upper third, horizontally centred
    x = int(max(0, min(cw - sw, cw / 2 - fx)))
    y = int(max(0, min(ch - sh, ch / 3 - fy)))
    return Geometry(scale, offset=(x, y), canvas=(cw, ch), scaled=(sw, sh))


def apply_geometry(img: np.ndarray, g: Geometry) -> np.ndarray:
    import cv2

    cw, ch = g.canvas
    scaled = cv2.resize(img, (int(round(img.shape[1] * g.scale)), int(round(img.shape[0] * g.scale))), interpolation=cv2.INTER_AREA if g.scale < 1 else cv2.INTER_LINEAR)
    if g.crop is not None and g.offset is None:
        x, y = g.crop
        out = scaled[y : y + ch, x : x + cw]
        if out.shape[:2] != (ch, cw):  # rounding at the edge
            out = cv2.resize(out, (cw, ch))
        return np.ascontiguousarray(out)
    if g.crop is not None:  # cropped on one side, pasted on the other
        x, y = g.crop
        scaled = scaled[y : y + ch, x : x + cw]
    canvas = np.empty((ch, cw, 3), np.uint8)
    canvas[:] = BACKGROUND
    x, y = g.offset or (0, 0)
    h, w = min(scaled.shape[0], ch - y), min(scaled.shape[1], cw - x)
    canvas[y : y + h, x : x + w] = scaled[:h, :w]
    return canvas


# ---------------------------------------------------------------------------------------- download


def sha(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, path: Path) -> None:
    """Download to `path` (via a .part file); a short or failed download raises SystemExit naming the file."""
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=600) as r, open(tmp, "wb") as f:
            expected = int(r.headers.get("Content-Length") or 0)
            while block := r.read(1 << 20):
                f.write(block)
        if expected and tmp.stat().st_size != expected:
            raise OSError(f"got {tmp.stat().st_size} of {expected} bytes")
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"download of {path.name} failed: {exc}") from exc
    tmp.rename(path)


def ensure_source(src: dict, cache: Path) -> Path:
    """The pinned 480p transcode (SHA-256), or the original upload (Commons SHA-1) if the transcode changed.
    A verified file already in the cache is used without network (re-runs offline)."""
    path = cache / f"{src['id']}.480p.webm"
    orig = cache / f"{src['id']}.original.webm"
    if path.exists() and sha(path, "sha256") == src["sha256"]:
        return path
    if orig.exists() and sha(orig, "sha1") == src["original_sha1"]:
        return orig
    path.unlink(missing_ok=True)
    log.info("downloading %s", src["id"])
    fetch(src["url"], path)
    if sha(path, "sha256") == src["sha256"]:
        return path
    log.warning("%s: transcode checksum changed, using the original upload", src["id"])
    path.unlink()
    fetch(src["original_url"], orig)
    if sha(orig, "sha1") != src["original_sha1"]:
        orig.unlink()
        raise SystemExit(f"checksum mismatch for {src['id']} ({orig.name}); refusing to use it")
    return orig


# ---------------------------------------------------------------------------------------- analysis


def analyse(path: Path, source_id: str, engine, max_s: float = ANALYSIS_MAX_S) -> dict:
    """Faces + fused probabilities at ANALYSIS_FPS, speech windows; returns a JSON-able dict."""
    import av

    from inference.types import AudioChunk, ErrorResult, VideoFrame

    frames: list[dict] = []
    container = av.open(str(path))
    vs = container.streams.video[0]
    aus = container.streams.audio[0] if container.streams.audio else None
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000) if aus is not None else None
    streams = [vs] + ([aus] if aus is not None else [])
    next_t = 0.0
    audio_buf: list[AudioChunk] = []
    size = (vs.codec_context.width, vs.codec_context.height)
    fid = 0
    for packet in container.demux(*streams):
        if packet.dts is None:
            continue
        for frame in packet.decode():
            if frame.time is None:
                continue  # no timestamp: cannot place it on the clock
            t = float(frame.time)
            if t > max_s:
                break
            if packet.stream is vs:
                if t < next_t:
                    continue
                next_t = t + 1.0 / ANALYSIS_FPS
                if audio_buf:  # audio first, so this frame can be fused with the window that ended before it
                    for r in engine.process_audio({source_id: audio_buf}):
                        if isinstance(r, ErrorResult):
                            log.warning("%s audio at %.1f s: %s", source_id, t, r.message)
                    audio_buf = []
                fid += 1
                res = engine.process_frames([VideoFrame(source_id, fid, t, frame.to_ndarray(format="bgr24"))])[0]
                if isinstance(res, ErrorResult):
                    log.warning("%s frame at %.1f s skipped: %s", source_id, t, res.message)
                    continue
                faces = [{"track": f.track_id, "box": list(f.box), "fused": f.fused} for f in res.faces]
                frames.append({"t": round(t, 3), "faces": faces})
            elif resampler is not None:
                for out in resampler.resample(frame):
                    audio_buf.append(AudioChunk(source_id, t, out.to_ndarray().reshape(-1).astype(np.int16)))
                    t += out.samples / 16000
        else:
            continue
        break
    container.close()
    return {"file": path.name, "size": list(size), "has_audio": aus is not None, "frames": frames}


def merge_windows(ws: list[Window]) -> list[Window]:
    """One moment seen on several track ids (tracker re-IDs, overlapping faces) counts once:
    windows of the same kind that overlap in time are merged."""
    out: list[Window] = []
    for w in sorted(ws, key=lambda w: (w.kind, w.start)):
        last = out[-1] if out and out[-1].kind == w.kind else None
        if last is not None and w.start <= last.end:
            out[-1] = Window(w.kind, last.track, last.start, max(last.end, w.end), max(last.peak, w.peak))
        else:
            out.append(w)
    return sorted(out, key=lambda w: w.start)


def windows_of(analysis: dict) -> list[Window]:
    per_track: dict[int, dict[str, list[tuple[float, float]]]] = {}
    for fr in analysis["frames"]:
        for f in fr["faces"]:
            s = per_track.setdefault(f["track"], {"negative": [], "positive": []})
            s["negative"].append((fr["t"], sum(f["fused"][k] for k in NEGATIVE)))
            s["positive"].append((fr["t"], f["fused"]["happiness"]))
    out: list[Window] = []
    for track, s in per_track.items():
        out += find_windows(track, s["negative"], "negative") + find_windows(track, s["positive"], "positive")
    return merge_windows(out)


def candidates_of(source: str, analysis: dict, windows: list[Window], duration: float) -> list[Candidate]:
    frames = analysis["frames"]
    out = []
    t = 0.0
    end_all = min(duration, frames[-1]["t"] if frames else 0.0)
    while t + SEGMENT_S <= end_all:
        seg = [fr for fr in frames if t <= fr["t"] < t + SEGMENT_S]
        with_face = [fr for fr in seg if fr["faces"]]
        if seg and with_face:
            biggest = [max(fr["faces"], key=lambda f: f["box"][2]) for fr in with_face]
            out.append(Candidate(
                source=source, start=t, end=t + SEGMENT_S,
                windows=[w for w in windows if t <= w.start and w.end <= t + SEGMENT_S],
                face_ratio=len(with_face) / len(seg),
                median_face=float(statistics.median(f["box"][2] for f in biggest)),
                face_center=(float(statistics.median(f["box"][0] + f["box"][2] / 2 for f in biggest)),
                             float(statistics.median(f["box"][1] + f["box"][3] / 2 for f in biggest))),
                has_audio=analysis["has_audio"],
            ))
        t += SEGMENT_STEP_S
    return out


# ---------------------------------------------------------------------------------------- encoding


def encode(path: Path, cand: Candidate, profile: str, src_size: tuple[int, int], out_path: Path) -> None:
    """Encode [cand.start, cand.end) for the profile. Video and audio both cover exactly that span,
    so the clip loops with -c copy without drift. Too short a result raises."""
    import av

    g = fit_geometry(src_size, cand.median_face, cand.face_center, profile)
    with_audio = PROFILES[profile]["audio"]
    seconds = cand.end - cand.start
    want_video, want_audio = int(round(seconds * OUT_FPS)), int(round(seconds * 16000))
    tmp = out_path.with_name(f".{out_path.name}.part")  # never matches the emulator's cam[0-9][0-9].mp4
    inp = av.open(str(path))
    vs = inp.streams.video[0]
    aus = inp.streams.audio[0] if (with_audio and inp.streams.audio) else None
    inp.seek(int(max(0.0, cand.start - 2.0) * av.time_base))
    out = av.open(str(tmp), "w", format="mp4", options={"movflags": "+faststart"})
    ov = out.add_stream("libx264", rate=OUT_FPS, options={"preset": "veryfast", "crf": "23", "g": str(GOP), "keyint_min": str(GOP), "sc_threshold": "0", "bf": "0"})
    ov.width, ov.height, ov.pix_fmt = g.canvas[0], g.canvas[1], "yuv420p"
    oa = None
    if aus is not None:
        oa = out.add_stream("aac", rate=16000)
        oa.layout = "mono"
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000, frame_size=1024)  # AAC frame
    n_video = 0
    n_audio = 0
    done = False
    for packet in inp.demux(*([vs] + ([aus] if aus is not None else []))):
        if packet.dts is None:
            continue
        for frame in packet.decode():
            if frame.time is None:
                continue
            t = float(frame.time)
            if t < cand.start:
                continue
            if packet.stream is vs and t >= cand.end:
                done = True
                break
            if packet.stream is vs:
                while n_video < want_video and cand.start + n_video / OUT_FPS <= t:  # resample to 10 fps by time
                    img = apply_geometry(frame.to_ndarray(format="bgr24"), g)
                    vf = av.VideoFrame.from_ndarray(img, format="bgr24")
                    vf.pts, vf.time_base = n_video, Fraction(1, OUT_FPS)
                    for p in ov.encode(vf):
                        out.mux(p)
                    n_video += 1
            elif oa is not None and n_audio < want_audio and t < cand.end:
                for af in resampler.resample(frame):
                    if n_audio >= want_audio:
                        break
                    af.pts, af.time_base = n_audio, Fraction(1, 16000)
                    n_audio += af.samples
                    for p in oa.encode(af):
                        out.mux(p)
        if done or (n_video >= want_video and (oa is None or n_audio >= want_audio)):
            break
    if oa is not None and n_audio < want_audio:
        for af in resampler.resample(None):  # flush the last partial AAC frame
            af.pts, af.time_base = n_audio, Fraction(1, 16000)
            n_audio += af.samples
            for p in oa.encode(af):
                out.mux(p)
    for p in ov.encode(None):
        out.mux(p)
    if oa is not None:
        for p in oa.encode(None):
            out.mux(p)
    out.close()
    inp.close()
    if n_video < want_video * 0.98:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{out_path.name}: only {n_video} of {want_video} frames (source ended early?)")
    tmp.rename(out_path)


def measure_faces(path: Path, detector, samples: int = 12) -> Optional[float]:
    """Median width of the largest face over a few frames of an encoded clip."""
    import av

    widths = []
    with av.open(str(path)) as c:
        vs = c.streams.video[0]
        n = 0
        for frame in c.decode(vs):
            n += 1
            if n % max(1, SEGMENT_S * OUT_FPS // samples) != 1:
                continue
            boxes = detector.detect(frame.to_ndarray(format="bgr24"))
            if boxes:
                widths.append(max(b[2] for b in boxes))
    return float(statistics.median(widths)) if widths else None


# ---------------------------------------------------------------------------------------- main


def mmss(s: float) -> str:
    return f"{int(s) // 60:02d}:{int(s) % 60:02d}"


def in_band(face: Optional[float], profile: str) -> bool:
    lo, hi = PROFILES[profile]["face"]
    return face is not None and lo * 0.9 <= face <= hi * 1.1


def write_report(demo: Path, sources: dict, plan: list[tuple[str, Candidate]], measured: dict[str, Optional[float]]) -> None:
    lines = [
        "# Demo segments",
        "",
        "Generated by `backend/scripts/prep_demo_clips.py`. Windows follow the CAP-11 rule on the project's own",
        f"models (fused group >= {THRESHOLD} held >= {MIN_HELD_S:.0f} s on one face track, analysed at {ANALYSIS_FPS} fps).",
        "",
        "| Clip | Profile | Source | Segment | Median face (target band) | Negative windows | Positive windows |",
        "|---|---|---|---|---|---|---|",
    ]
    no_negative = []
    for i, (profile, c) in enumerate(plan, start=1):
        name = f"cam{i:02d}"
        band = PROFILES[profile]["face"]
        neg = [w for w in c.windows if w.kind == "negative"]
        pos = [w for w in c.windows if w.kind == "positive"]
        def fmt(ws: list[Window]) -> str:
            shown = ", ".join(f"{mmss(w.start - c.start)} ({w.duration:.1f} s, {w.peak:.2f})" for w in ws[:4])
            return (shown + (f" … {len(ws)} in all" if len(ws) > 4 else "")) or "—"

        face = measured.get(name)
        lines.append(f"| {name} | {profile} | {sources[c.source]['title']} | {mmss(c.start)}–{mmss(c.end)} | "
                     f"{'—' if face is None else f'{face:.0f} px'} ({band[0]}–{band[1]}) | {fmt(neg)} | {fmt(pos)} |")
        if not neg:
            no_negative.append(name)
    no_positive = [f"cam{i:02d}" for i, (_, c) in enumerate(plan, start=1) if not any(w.kind == "positive" for w in c.windows)]
    outside = [
        f"cam{i:02d}" for i, (p, _) in enumerate(plan, start=1)
        if not in_band(measured.get(f"cam{i:02d}"), p)
    ]
    lines += [
        "",
        f"Clips without a negative window: {', '.join(no_negative) or 'none'}.",
        f"Clips without a positive window: {', '.join(no_positive) or 'none'}.",
        f"Clips outside their face-size band (±10 %) or without a detected face: {', '.join(outside) or 'none'}.",
        "",
        "Windows are mined on the sources (about 100 px faces, with speech fused). Security clips have no audio and",
        "25–45 px faces, and sector/security clips are a scaled scene on a plain canvas, so the demo can trigger",
        "fewer windows there than listed.",
        "",
        "The FER model reports the negative group (mostly sadness and anger) far more often than people would",
        "label it, especially on acted footage: the windows are what the demo will trigger, not ground truth.",
        "",
    ]
    (demo / "segments.md").write_text("\n".join(lines), encoding="utf-8")


def check_stream_list(plan: list[tuple[str, Candidate]], path: Path) -> None:
    """demo-cams.json labels camN with a camera type; it must match the profile of clip camNN."""
    streams = json.loads(path.read_text(encoding="utf-8"))
    types = [s["camera_type"] for s in streams]
    profiles = [p for p, _ in plan]
    if types != profiles:
        raise SystemExit(f"{path.name} camera types {types} do not match the clip plan {profiles}")


def write_attribution(demo: Path, sources: dict, plan: list[tuple[str, Candidate]]) -> None:
    used: dict[str, list[str]] = {}
    for i, (_, c) in enumerate(plan, start=1):
        used.setdefault(c.source, []).append(f"cam{i:02d} {mmss(c.start)}–{mmss(c.end)}")
    licence_url = {"CC BY 3.0": "https://creativecommons.org/licenses/by/3.0/", "CC BY 4.0": "https://creativecommons.org/licenses/by/4.0/",
                   "CC BY-SA 4.0": "https://creativecommons.org/licenses/by-sa/4.0/", "CC0": "https://creativecommons.org/publicdomain/zero/1.0/"}
    lines = [
        "# Demo media",
        "",
        "Openly licensed sources of the 20-camera demo set. The clips are produced locally by",
        "`backend/scripts/prep_demo_clips.py` and are not committed. **Changes made:** segments were cut, scaled,",
        "cropped or padded onto a plain background, resampled to 10 fps and re-encoded (H.264, AAC 16 kHz mono);",
        "security clips have no audio. Details per clip: [segments.md](segments.md).",
        "",
        "| Source | Author | Licence | Commons page | Used in |",
        "|---|---|---|---|---|",
    ]
    for sid, s in sources.items():
        lic = s["licence"]
        lic_md = f"[{lic}]({licence_url[lic]})" if lic in licence_url else lic
        lines.append(f"| {s['title'].rsplit('.', 1)[0]} | {s['author']} | {lic_md} | {s['page']} | {', '.join(used.get(sid, [])) or 'not used in the current set'} |")
    lines += [
        "",
        "The 480p transcodes served by Wikimedia Commons are used (SHA-256 pinned in `sources.json`); if Commons",
        "re-transcodes a file, the prep falls back to the original upload, verified by its Commons SHA-1.",
        "",
        "Fallback synthetic cameras (no clips prepared) are built from files already in the repository:",
        "",
        "| File | Source | Licence |",
        "|---|---|---|",
        '| `backend/bench/data/face.jpg` | "astronaut" test image from scikit-image (NASA photograph of Eileen Collins) | Public domain |',
        "| `backend/bench/data/speech.wav` | excerpt of `examples/c++/aepyx.wav`, snakers4/silero-vad | MIT |",
        "",
    ]
    (demo / "ATTRIBUTION.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--demo", type=Path, default=Path("/demo"), help="demo-videos directory")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from app.config import get_settings
    from inference.worker import Engine, WorkerConfig, select_device

    demo = args.demo
    cache, clips = demo / "cache", demo / "clips"
    cache.mkdir(exist_ok=True)
    sources = {s["id"]: s for s in json.loads((demo / "sources.json").read_text(encoding="utf-8"))}
    settings = get_settings()
    engine = None

    all_cands: list[Candidate] = []
    files: dict[str, tuple[Path, tuple[int, int]]] = {}
    for sid, src in sources.items():
        path = ensure_source(src, cache)
        cached = cache / f"{sid}.analysis.json"
        analysis = json.loads(cached.read_text()) if cached.exists() else None
        if analysis is not None and analysis.get("file", f"{sid}.480p.webm") != path.name:
            analysis = None  # analysed another file (480p vs original): boxes would not match
        if analysis is None:
            if engine is None:
                device = select_device(args.device, settings.variant)
                log.info("loading models on %s", device)
                engine = Engine(settings, device, WorkerConfig())
            t0 = time.time()
            analysis = analyse(path, sid, engine)
            engine.remove(sid)
            cached.write_text(json.dumps(analysis))
            log.info("%s analysed in %.0f s (%d frames)", sid, time.time() - t0, len(analysis["frames"]))
        wins = windows_of(analysis)
        log.info("%s: %d negative, %d positive windows", sid, sum(w.kind == "negative" for w in wins), sum(w.kind == "positive" for w in wins))
        all_cands += candidates_of(sid, analysis, wins, src["duration_s"])
        files[sid] = (path, tuple(analysis["size"]))

    plan = plan_mix(all_cands)
    check_stream_list(plan, demo / "demo-cams.json")
    from inference.detector import FaceDetector

    detector = FaceDetector(settings.model_path("yunet"))
    measured: dict[str, Optional[float]] = {}
    # Encode into a staging directory and swap it in only when all clips succeeded: a failed run
    # never leaves a partial set for the emulator.
    staging = demo / ".clips-staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    for i, (profile, c) in enumerate(plan, start=1):
        name = f"cam{i:02d}"
        path, size = files[c.source]
        t0 = time.time()
        encode(path, c, profile, size, staging / f"{name}.mp4")
        measured[name] = measure_faces(staging / f"{name}.mp4", detector)
        log.info("%s %s from %s %s–%s: face %s px, %.0f s", name, profile, c.source, mmss(c.start), mmss(c.end), measured[name], time.time() - t0)
    (staging / "plan.json").write_text(json.dumps([{"clip": f"cam{i:02d}", "profile": p, **{k: v for k, v in asdict(c).items() if k != "windows"}, "windows": [asdict(w) for w in c.windows]} for i, (p, c) in enumerate(plan, start=1)], ensure_ascii=False, indent=1))
    shutil.rmtree(clips, ignore_errors=True)
    staging.rename(clips)
    write_report(demo, sources, plan, measured)
    write_attribution(demo, sources, plan)
    log.info("done: %d clips in %s", len(plan), clips)


if __name__ == "__main__":
    main()

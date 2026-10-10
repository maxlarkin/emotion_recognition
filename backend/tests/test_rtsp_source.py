"""sources.rtsp on a generated local file (same PyAV path as RTSP, no network)."""

from __future__ import annotations

import threading
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from sources import rtsp


def make_clip(path: Path, seconds: float = 2.0, size=(320, 240), fps: int = 25, audio: bool = True) -> Path:
    import av

    with av.open(str(path), "w") as out:
        vs = out.add_stream("mpeg4", rate=fps)
        vs.width, vs.height, vs.pix_fmt = size[0], size[1], "yuv420p"
        aus = out.add_stream("aac", rate=44100) if audio else None
        if aus is not None:
            aus.layout = "mono"
        for i in range(int(seconds * fps)):
            img = np.full((size[1], size[0], 3), (i * 5) % 255, np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="bgr24")
            frame.pts = i
            frame.time_base = Fraction(1, fps)
            for p in vs.encode(frame):
                out.mux(p)
        if aus is not None:
            n = 1024
            for k in range(int(seconds * 44100 / n)):
                t = (np.arange(n) + k * n) / 44100
                samples = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32).reshape(1, -1)
                af = av.AudioFrame.from_ndarray(samples, format="flt", layout="mono")
                af.sample_rate = 44100
                af.pts = k * n
                for p in aus.encode(af):
                    out.mux(p)
            for p in aus.encode(None):
                out.mux(p)
        for p in vs.encode(None):
            out.mux(p)
    return path


class Sink:
    def __init__(self) -> None:
        self.info = None
        self.frames: list[rtsp.DecodedFrame] = []
        self.audio_chunks: list[tuple[int, float]] = []

    def opened(self, info):
        self.info = info

    def frame(self, frame):
        self.frames.append(frame)

    def audio(self, pcm, ts):
        self.audio_chunks.append((len(pcm), ts))


@pytest.fixture()
def clip(tmp_path):
    return make_clip(tmp_path / "clip.mkv")


def test_fit_keeps_aspect_and_never_upscales():
    assert rtsp.fit((3840, 2160), 1920) == (1920, 1080)
    assert rtsp.fit((640, 480), 1920) == (640, 480)
    assert rtsp.fit((2560, 1440), 640) == (640, 360)


def test_run_samples_scales_and_cuts_audio(clip):
    sink = Sink()
    with pytest.raises(rtsp.SourceError) as err:
        rtsp.run(str(clip), detection_size=160, fps=5, stop=threading.Event(), sink=sink)
    assert err.value.message == "Поток закончился"
    assert sink.info == rtsp.OpenInfo((320, 240), True, "cpu")
    # 2 s at 25 fps sampled to 5 fps
    assert 9 <= len(sink.frames) <= 11
    f = sink.frames[0]
    assert f.image.shape == (120, 160, 3) and f.tile.shape == (120, 160, 3)
    ts = [x.ts for x in sink.frames]
    assert all(b > a for a, b in zip(ts, ts[1:]))
    assert abs((ts[-1] - ts[0]) - 1.8) < 0.15  # media clock, not decode speed
    # 500 ms PCM16 chunks on the same clock
    assert len(sink.audio_chunks) >= 3
    assert all(n == rtsp.AUDIO_CHUNK_SAMPLES * 2 for n, _ in sink.audio_chunks)
    starts = [t for _, t in sink.audio_chunks]
    assert abs((starts[1] - starts[0]) - 0.5) < 1e-6
    assert abs(starts[0] - ts[0]) < 0.2


def test_tile_is_capped_at_640(tmp_path):
    big = make_clip(tmp_path / "big.mkv", seconds=0.4, size=(1280, 720), audio=False)
    sink = Sink()
    with pytest.raises(rtsp.SourceError):
        rtsp.run(str(big), detection_size=1920, fps=5, stop=threading.Event(), sink=sink)
    assert sink.info.has_audio is False
    assert sink.frames[0].image.shape == (720, 1280, 3)
    assert sink.frames[0].tile.shape == (360, 640, 3)


def test_stop_returns_without_error(clip):
    stop = threading.Event()

    class Stopper(Sink):
        def frame(self, frame):
            super().frame(frame)
            stop.set()

    sink = Stopper()
    rtsp.run(str(clip), detection_size=160, fps=5, stop=stop, sink=sink)
    assert len(sink.frames) == 1


def test_probe(clip, tmp_path):
    assert rtsp.probe(str(clip)) == rtsp.OpenInfo((320, 240), True, "cpu")
    with pytest.raises(rtsp.SourceError) as err:
        rtsp.probe(str(tmp_path / "missing.mkv"))
    assert err.value.message in ("Не удалось открыть поток", "Поток не найден по этому адресу")


def test_error_messages_are_russian():
    assert rtsp.mask_error(TimeoutError()) == "Нет ответа от камеры"
    assert rtsp.mask_error(OSError("401 Unauthorized")) == "Камера отклонила логин или пароль"
    assert rtsp.mask_error(OSError("Connection refused")) == "Камера отклонила подключение"


def test_audio_cutter_restarts_after_a_gap():
    sink = Sink()
    cut = rtsp._AudioCutter()
    half = b"\x00\x00" * (rtsp.AUDIO_CHUNK_SAMPLES // 2)
    cut.push(half, 10.0, sink)  # 250 ms buffered
    cut.push(half, 12.0, sink)  # 2 s later: the buffered quarter is dropped, chunk restarts at 12.0
    cut.push(half, 12.25, sink)
    assert sink.audio_chunks == [(rtsp.AUDIO_CHUNK_SAMPLES * 2, 12.0)]

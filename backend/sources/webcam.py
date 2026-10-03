"""Webcam source: binary messages of WebSocket /ws/live (interfaces.md, Live session).

    video: 0x01 | f64 capture_ts (ms, client clock) | u32 frame_id | JPEG bytes
    audio: 0x02 | f64 capture_ts (ms, client clock) | PCM16 LE mono 16 kHz
All fields little-endian.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Union

import numpy as np

VIDEO_TYPE = 0x01
AUDIO_TYPE = 0x02
VIDEO_HEADER = struct.Struct("<BdI")  # 13 bytes
AUDIO_HEADER = struct.Struct("<Bd")  # 9 bytes


class BadMessage(ValueError):
    """A binary message that cannot be used. `message` is Russian text for the client."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


@dataclass(frozen=True)
class WebcamFrame:
    capture_ts: float  # ms, client clock
    frame_id: int
    jpeg: bytes


@dataclass(frozen=True)
class AudioPacket:
    capture_ts: float  # ms, client clock
    pcm: bytes


def encode_frame(capture_ts: float, frame_id: int, jpeg: bytes) -> bytes:
    """Client-side encoding (tests, bench.live_ws)."""
    return VIDEO_HEADER.pack(VIDEO_TYPE, float(capture_ts), int(frame_id)) + bytes(jpeg)


def parse_message(data: bytes) -> Union[WebcamFrame, AudioPacket]:
    if not data:
        raise BadMessage("empty", "Пустое сообщение")
    kind = data[0]
    if kind == VIDEO_TYPE:
        if len(data) < VIDEO_HEADER.size:
            raise BadMessage("short", f"Некорректный кадр: сообщение короче {VIDEO_HEADER.size} байт")
        _, ts, frame_id = VIDEO_HEADER.unpack_from(data)
        if not math.isfinite(ts):
            raise BadMessage("bad_ts", "Некорректный кадр: неверная метка времени")
        return WebcamFrame(ts, frame_id, bytes(data[VIDEO_HEADER.size :]))
    if kind == AUDIO_TYPE:
        if len(data) < AUDIO_HEADER.size:
            raise BadMessage("short", f"Некорректный аудиофрагмент: сообщение короче {AUDIO_HEADER.size} байт")
        _, ts = AUDIO_HEADER.unpack_from(data)
        return AudioPacket(ts, bytes(data[AUDIO_HEADER.size :]))
    raise BadMessage("unknown_type", f"Неизвестный тип сообщения: 0x{kind:02x}")


def decode_jpeg(jpeg: bytes) -> np.ndarray:
    """JPEG -> BGR uint8 HxWx3. Blocking (run it off the event loop)."""
    import cv2

    if not jpeg:
        raise BadMessage("bad_jpeg", "Не удалось декодировать кадр: нет данных изображения")
    try:
        img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        img = None
    if img is None or img.ndim != 3 or img.shape[0] == 0 or img.shape[1] == 0:
        raise BadMessage("bad_jpeg", "Не удалось декодировать кадр (ожидается JPEG)")
    return img

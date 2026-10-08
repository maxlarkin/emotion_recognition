"""Shared data contract of the inference engine (used by every source in later stories).

Everything here is plain data (numpy + dataclasses) so it pickles cheaply across the
"source -> queue -> worker -> results" seam.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np

# Output emotion keys, always in this order.
EMOTIONS: tuple[str, ...] = ("anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral")

# Class order of the EmotiEffLib 8-class models (enet_b0_8_va_mtl logits 0..7).
FER_CLASSES: tuple[str, ...] = (
    "anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise",
)
# For each EMOTIONS key, its index in FER_CLASSES (contempt is dropped).
_FER_TO_EMOTIONS = np.array([FER_CLASSES.index(e) for e in EMOTIONS])

# GigaAM-Emo (Dusha) labels -> output keys. Other output keys get 0.
SER_LABEL_MAP: dict[str, str] = {
    "angry": "anger",
    "sad": "sadness",
    "neutral": "neutral",
    "positive": "happiness",
}
# Output keys that have an audio counterpart (fused); the rest come from the face only.
AUDIO_EMOTIONS: tuple[str, ...] = tuple(SER_LABEL_MAP.values())

SAMPLE_RATE = 16000


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def renormalise(p: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    s = p.sum(axis=axis, keepdims=True)
    return np.divide(p, s, out=np.full_like(p, 1.0 / p.shape[axis]), where=s > 0)


def fer_probs_to_emotions(probs8: np.ndarray) -> np.ndarray:
    """[..., 8] probabilities in FER_CLASSES order -> [..., 7] in EMOTIONS order (contempt dropped, renormalised)."""
    probs8 = np.asarray(probs8, dtype=np.float64)
    if probs8.shape[-1] != len(FER_CLASSES):
        raise ValueError(f"expected {len(FER_CLASSES)} FER classes, got {probs8.shape[-1]}")
    return renormalise(probs8[..., _FER_TO_EMOTIONS])


def ser_probs_to_emotions(probs: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    """[..., K] SER probabilities with model label names -> [..., 7] in EMOTIONS order."""
    probs = np.asarray(probs, dtype=np.float64)
    out = np.zeros(probs.shape[:-1] + (len(EMOTIONS),), dtype=np.float64)
    for i, name in enumerate(labels):
        key = SER_LABEL_MAP.get(str(name))
        if key is None:
            raise ValueError(f"unknown SER label {name!r}")
        out[..., EMOTIONS.index(key)] += probs[..., i]
    return out


def to_dict(p: np.ndarray) -> dict[str, float]:
    return {k: float(v) for k, v in zip(EMOTIONS, p, strict=True)}


def from_dict(d: Mapping[str, float]) -> np.ndarray:
    return np.array([float(d.get(k, 0.0)) for k in EMOTIONS], dtype=np.float64)


def top_label(p: np.ndarray) -> str:
    return EMOTIONS[int(np.argmax(p))]


# ---------------------------------------------------------------------------------- inputs


@dataclass
class VideoFrame:
    source_id: str
    frame_id: int
    ts: float  # capture/decode timestamp (seconds, caller's clock); also drives track expiry
    image: np.ndarray  # BGR uint8 [H, W, 3]
    submit_ts: float = 0.0  # time.time() when handed to the worker (set by InferenceWorker)


@dataclass
class AudioChunk:
    source_id: str
    ts: float  # timestamp of the first sample (seconds, caller's clock)
    samples: np.ndarray  # PCM16 mono 16 kHz, int16 [N]


# ---------------------------------------------------------------------------------- outputs


@dataclass
class FaceResult:
    track_id: int
    box: tuple[int, int, int, int]  # x, y, w, h in frame pixels
    face: dict[str, float]  # EMA-smoothed facial probabilities (7 keys)
    face_label: str
    fused: dict[str, float]  # late fusion with the latest speech window (== face in silence)
    fused_label: str
    confidence: float  # fused probability of fused_label


@dataclass
class AudioResult:
    source_id: str
    speech: bool
    probs: Optional[dict[str, float]]  # 7 keys when speech, else None
    label: Optional[str]
    window_end_ts: float
    speech_ratio: float
    ser_ms: float = 0.0
    # True for an older speech window of a backlog: SER ran only on the newest one (probs None).
    skipped: bool = False


@dataclass
class FrameResult:
    source_id: str
    frame_id: int
    ts: float
    faces: list[FaceResult]
    audio: Optional[AudioResult]  # latest audio block of this source (None before the first window)
    submit_ts: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)  # ms: queue, detect, fer, total


@dataclass
class ErrorResult:
    source_id: Optional[str]
    kind: str  # e.g. "unknown_source", "frame_failed"
    message: str

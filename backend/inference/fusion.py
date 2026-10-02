"""Per-track EMA over facial probabilities and weighted late fusion with the vocal emotion (stack.md, Fusion)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from inference.types import AUDIO_EMOTIONS, EMOTIONS, renormalise

EMA_ALPHA = 0.4
W_FACE = 0.7
W_AUDIO = 0.3

_AUDIO_MASK = np.array([e in AUDIO_EMOTIONS for e in EMOTIONS])


class TrackEMA:
    """ema = alpha * p + (1 - alpha) * ema_prev, per track id; the first observation is taken as is."""

    def __init__(self, alpha: float = EMA_ALPHA):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("EMA alpha must be in (0, 1]")
        self.alpha = alpha
        self._state: dict[int, np.ndarray] = {}

    def update(self, track_id: int, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float64)
        prev = self._state.get(track_id)
        cur = probs if prev is None else self.alpha * probs + (1.0 - self.alpha) * prev
        self._state[track_id] = cur
        return cur

    def drop(self, track_id: int) -> None:
        self._state.pop(track_id, None)

    def __contains__(self, track_id: int) -> bool:
        return track_id in self._state


@dataclass
class FusionWeights:
    w_face: float = W_FACE
    w_audio: float = W_AUDIO

    def __post_init__(self) -> None:
        if self.w_face < 0 or self.w_audio < 0 or self.w_face + self.w_audio <= 0:
            raise ValueError("fusion weights must be non-negative and not both zero")


def fuse(face: np.ndarray, audio: Optional[np.ndarray], weights: FusionWeights) -> np.ndarray:
    """Late fusion in EMOTIONS order: renormalise(w_face * face + w_audio * audio).

    `audio` is the 7-key vocal distribution with zeros on classes that have no audio
    counterpart (fear, disgust, surprise), so those classes get the face term only.
    No speech (audio is None) -> w_audio = 0 and the result equals the face distribution.
    """
    face = np.asarray(face, dtype=np.float64)
    if audio is None or weights.w_audio == 0:
        return face.copy()  # exactly the face distribution (already normalised)
    audio = np.where(_AUDIO_MASK, np.asarray(audio, dtype=np.float64), 0.0)
    return renormalise(weights.w_face * face + weights.w_audio * audio)

"""Speech emotion recognition: GigaAM-Emo on 3 s windows (1 s hop), mapped to the 7 output keys.

The forward pass reproduces GigaAMEmo.get_probs (preprocessor -> encoder -> mean over time
-> head -> softmax) on in-memory float32 16 kHz windows instead of a file path.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from inference.types import SAMPLE_RATE, ser_probs_to_emotions
from inference.vad import CHUNK, SPEECH_THRESHOLD

WINDOW_S = 3.0
HOP_S = 1.0
MIN_SPEECH_RATIO = 0.5


@dataclass
class Window:
    samples: np.ndarray  # float32 [WINDOW_S * 16000]
    speech_ratio: float
    end_ts: float


class SpeechWindow:
    """Per-source sliding window over VAD-processed 512-sample chunks."""

    def __init__(self, window_s: float = WINDOW_S, hop_s: float = HOP_S):
        self.window_len = int(round(window_s * SAMPLE_RATE))
        self.hop_len = int(round(hop_s * SAMPLE_RATE))
        self._chunks: deque[np.ndarray] = deque(maxlen=-(-self.window_len // CHUNK))
        self._flags: deque[bool] = deque(maxlen=self._chunks.maxlen)
        self._since_emit = 0
        self._total = 0

    def push(self, chunk: np.ndarray, speech_prob: float, end_ts: float) -> Optional[Window]:
        self._chunks.append(chunk)
        self._flags.append(speech_prob >= SPEECH_THRESHOLD)
        self._total += chunk.shape[0]
        self._since_emit += chunk.shape[0]
        if self._total < self.window_len or self._since_emit < self.hop_len:
            return None
        first = self._since_emit == self._total  # first full window: the hop count starts here
        # Later windows carry the overshoot over, so the average hop stays exact with 512-sample chunks.
        self._since_emit = 0 if first else self._since_emit - self.hop_len
        samples = np.concatenate(list(self._chunks))[-self.window_len :]
        return Window(samples=samples, speech_ratio=float(np.mean(self._flags)), end_ts=end_ts)


class SpeechEmotion:
    def __init__(self, model_path: Path, device: str = "cpu", fp16: bool = False):
        import torch  # imported lazily: only the worker process needs torch
        import gigaam

        model_path = Path(model_path)
        if model_path.name != "emo.ckpt":
            raise ValueError(f"GigaAM-Emo checkpoint must be named emo.ckpt, got {model_path}")
        self._torch = torch
        self.device = torch.device(device)
        self.fp16 = bool(fp16 and self.device.type == "cuda")
        # The file exists, so gigaam only verifies its md5 and never downloads.
        self.model = gigaam.load_model(
            "emo", fp16_encoder=self.fp16, device=self.device, download_root=str(model_path.parent)
        )
        self.labels = [str(x) for x in self.model.id2name]

    def predict(self, windows: np.ndarray) -> np.ndarray:
        """[B, T] float32 16 kHz -> [B, 7] probabilities in EMOTIONS order."""
        torch = self._torch
        windows = np.ascontiguousarray(np.atleast_2d(windows), dtype=np.float32)
        b, t = windows.shape
        with torch.inference_mode():
            wav = torch.from_numpy(windows).to(self.device)
            length = torch.full([b], t, device=self.device)
            features, feat_len = self.model.preprocessor(wav, length)
            if self.fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    encoded, _ = self.model.encoder(features, feat_len)
            else:
                encoded, _ = self.model.encoder(features, feat_len)
            pooled = torch.nn.functional.avg_pool1d(encoded, kernel_size=encoded.shape[-1]).squeeze(-1)
            logits = self.model.head(pooled.float())
            probs = torch.nn.functional.softmax(logits.float(), dim=-1).cpu().numpy()
        return ser_probs_to_emotions(probs, self.labels)

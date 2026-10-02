"""Silero VAD (ONNX) with per-source recurrent state; 512-sample chunks at 16 kHz.

Reproduces silero_vad.utils_vad.OnnxWrapper.__call__: each chunk is prefixed with the last
64 samples of the previous one, and the [2, 1, 128] state is carried per source.
Runs on the CPU EP even in the GPU image: a tiny model called per 32 ms chunk is faster there.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from inference.types import SAMPLE_RATE

CHUNK = 512
CONTEXT = 64
SPEECH_THRESHOLD = 0.5


class VadStream:
    """Per-source VAD state: recurrent state, 64-sample context and leftover samples (< 512)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((CONTEXT,), dtype=np.float32)
        self.pending = np.zeros((0,), dtype=np.float32)


class SileroVAD:
    def __init__(self, model_path: Path):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)

    def chunk_prob(self, stream: VadStream, chunk: np.ndarray) -> float:
        """One 512-sample float32 chunk -> speech probability; advances the stream state."""
        if chunk.shape[0] != CHUNK:
            raise ValueError(f"VAD chunk must be {CHUNK} samples, got {chunk.shape[0]}")
        x = np.concatenate([stream.context, chunk.astype(np.float32, copy=False)])[np.newaxis, :]
        out, state = self.session.run(None, {"input": x, "state": stream.state, "sr": self._sr})
        stream.state = state
        stream.context = x[0, -CONTEXT:].copy()
        return float(out.reshape(-1)[0])

    def feed(self, stream: VadStream, samples: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Append float32 samples; return (chunk, speech probability) for every completed 512-chunk.

        Samples that do not fill a chunk stay in `stream.pending` until the next call.
        """
        buf = np.concatenate([stream.pending, samples.astype(np.float32, copy=False)])
        n = buf.shape[0] // CHUNK
        out = []
        for i in range(n):
            chunk = buf[i * CHUNK : (i + 1) * CHUNK]
            out.append((chunk, self.chunk_prob(stream, chunk)))
        stream.pending = buf[n * CHUNK :].copy()
        return out

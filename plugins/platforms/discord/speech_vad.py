"""Silero VAD (ONNX) as a level-independent speech classifier for barge-in.

The level gate in ``VoiceReceiver.speech_in_progress`` tells loud from quiet,
not speech from noise: a phone's own voice detection passes road noise at
a level that depends on the device, so a dBFS threshold that fits the car
cuts real speech over headphones and vice versa.  A classifier judges the
*structure* of the audio and needs no per-device calibration — the
production stacks (Pipecat, LiveKit) run Silero behind a coarse energy
gate for exactly this reason (researched 2026-09-05).

Runs on ``onnxruntime`` + ``numpy`` (both in the Hermes venv); torch is not
needed.  The model file is not part of the repository:

    curl -L -o ~/.hermes/models/silero_vad.onnx \\
      https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx

(2 327 524 bytes, sha256 1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3,
fetched 2026-09-05).  Without it :meth:`SileroVAD.load` returns ``None`` and
the caller keeps the level gate alone.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "~/.hermes/models/silero_vad.onnx"
MODEL_RATE = 16000          # the ONNX model accepts 8 or 16 kHz
CHUNK = 512                 # samples per inference at 16 kHz = 32 ms
SOURCE_RATE = 48000         # Discord PCM: 48 kHz, stereo, s16le
SOURCE_CHANNELS = 2


class SileroVAD:
    """Per-chunk speech probabilities for Discord PCM."""

    def __init__(self, model_path: str, threshold: float = 0.6):
        import numpy as np
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.log_severity_level = 3
        self._np = np
        self._session = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"],
        )
        self.model_path = model_path
        self.threshold = float(threshold)
        # onnxruntime sessions are thread-safe for run(); the lock keeps the
        # two callers (5 Hz poll, completed-utterance path) from interleaving
        # log lines rather than protecting the session.
        self._lock = threading.Lock()

    @classmethod
    def load(cls, model_path: Optional[str] = None,
             threshold: float = 0.6) -> Optional["SileroVAD"]:
        """Return a ready classifier, or ``None`` (logged once) if it cannot run."""
        path = os.path.expanduser(model_path or DEFAULT_MODEL_PATH)
        if not os.path.isfile(path):
            logger.warning(
                "Silero VAD model not found at %s — barge-in falls back to the "
                "level gate alone (see plugins/platforms/discord/speech_vad.py)", path,
            )
            return None
        try:
            vad = cls(path, threshold)
        except Exception as e:  # missing onnxruntime/numpy, corrupt file
            logger.warning("Silero VAD unavailable (%s) — level gate alone", e)
            return None
        logger.info("Silero VAD loaded from %s (threshold %.2f)", path, threshold)
        return vad

    # -- audio preparation --------------------------------------------------

    def _to_model_rate(self, pcm: bytes):
        """48 kHz stereo s16le -> 16 kHz mono float32 in [-1, 1]."""
        np = self._np
        usable = len(pcm) - (len(pcm) % (2 * SOURCE_CHANNELS))
        if usable <= 0:
            return np.zeros(0, dtype=np.float32)
        samples = np.frombuffer(pcm[:usable], dtype="<i2").astype(np.float32)
        mono = samples.reshape(-1, SOURCE_CHANNELS).mean(axis=1)
        factor = SOURCE_RATE // MODEL_RATE
        trimmed = mono[: (len(mono) // factor) * factor]
        if len(trimmed) == 0:
            return np.zeros(0, dtype=np.float32)
        # Box-filter decimation: adequate anti-aliasing for a VAD, no scipy.
        return (trimmed.reshape(-1, factor).mean(axis=1) / 32768.0).astype(np.float32)

    # -- inference ----------------------------------------------------------

    def probabilities(self, pcm: bytes) -> List[float]:
        """Speech probability per 32 ms chunk, state reset per call."""
        np = self._np
        audio = self._to_model_rate(pcm)
        chunks = len(audio) // CHUNK
        if chunks == 0:
            return []
        state = np.zeros((2, 1, 128), dtype=np.float32)
        sr = np.array(MODEL_RATE, dtype=np.int64)
        probs: List[float] = []
        with self._lock:
            for i in range(chunks):
                x = audio[i * CHUNK:(i + 1) * CHUNK][None, :]
                out, state = self._session.run(None, {"input": x, "state": state, "sr": sr})
                probs.append(float(out[0][0]))
        return probs

    def speech_ratio(self, pcm: bytes) -> float:
        """Share of chunks above ``threshold``; 0.0 for audio shorter than one chunk."""
        probs = self.probabilities(pcm)
        if not probs:
            return 0.0
        return sum(1 for p in probs if p >= self.threshold) / len(probs)

    def max_window_ratio(self, pcm: bytes, window_seconds: float) -> float:
        """Best ``speech_ratio`` over any ``window_seconds`` stretch (hop = half window).

        A finished utterance has no "most recent" window; judging it as a
        whole lets a quiet lead-in dilute the words, judging its loudest
        window keeps the onset semantics (same reasoning as
        ``pcm_peak_window_dbfs``).
        """
        probs = self.probabilities(pcm)
        if not probs:
            return 0.0
        n = max(1, int(round(window_seconds * MODEL_RATE / CHUNK)))
        if len(probs) <= n:
            return sum(1 for p in probs if p >= self.threshold) / len(probs)
        hop = max(1, n // 2)
        best = 0.0
        starts = list(range(0, len(probs) - n + 1, hop))
        if starts[-1] != len(probs) - n:
            starts.append(len(probs) - n)
        for start in starts:
            window = probs[start:start + n]
            best = max(best, sum(1 for p in window if p >= self.threshold) / n)
        return best

"""Queue-backed ``discord.AudioSource`` for streaming TTS playback.

The whole-file voice path (``FFmpegPCMAudio`` over a finished mp3) cannot
start before the last word has been synthesised.  This source is the other
half of ``gateway/streaming_tts_consumer.py``: the gateway writes PCM for
clause *n* while the model is still generating clause *n+1*, and Discord
pulls 20 ms frames out the moment the first clause arrives.

Format handling lives here on purpose.  Streaming providers declare their
own PCM format (``AudioFormat``: rate, channels, sample width) and Discord
only speaks 48 kHz stereo signed-16 LE; the adapter is the layer that knows
the platform's native format, so it converts.  Conversion happens in
:meth:`feed`, on the caller's thread — never in :meth:`read`, which
discord.py's sender thread calls every 20 ms and which must not do work
that can jitter.

Threading: ``feed``/``end``/``abort`` run on the gateway event loop, ``read``
on discord.py's player thread.  All shared state is under one plain lock.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import shape mirrors voice_mixer.py
    from .voice_mixer import CHANNELS, FRAME_SIZE, SAMPLE_RATE, SAMPLE_WIDTH, SILENCE_FRAME
except ImportError:  # pragma: no cover
    from voice_mixer import CHANNELS, FRAME_SIZE, SAMPLE_RATE, SAMPLE_WIDTH, SILENCE_FRAME

try:
    import discord
except Exception:  # pragma: no cover - discord is optional at import time
    discord = None  # type: ignore[assignment]


_AudioSourceBase = discord.AudioSource if discord is not None else object


def _require_numpy():
    """Import numpy or raise a message that names the extra.

    numpy is the "voice" extra, same as for :mod:`voice_mixer`. Resampling
    without it would mean hand-rolling interpolation in pure Python at
    48000 samples a second, which is exactly the work ``read`` must not do.
    """
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "numpy is required for streaming voice playback "
            "(pip install 'hermes-agent[voice]')"
        ) from exc
    return np


def numpy_available() -> bool:
    """True when the format converter can run."""
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:  # pragma: no cover - environment-dependent
        return False


def to_discord_pcm(pcm: bytes, sample_rate: int, channels: int,
                   sample_width: int = 2) -> bytes:
    """Convert one PCM chunk to Discord's native 48 kHz stereo s16le.

    Only signed 16-bit input is accepted — every registered streaming
    provider declares ``sample_width=2``, and silently mangling a format we
    do not understand is worse than refusing it.

    Resampling is linear interpolation. For the common exact-integer ratios
    (24 kHz → 48 kHz, 16 kHz → 48 kHz) that is what an ideal 2×/3× upsampler
    degrades to at this quality level anyway, and speech at 48 kHz has no
    content near Nyquist for imaging to matter.
    """
    if sample_width != SAMPLE_WIDTH:
        raise ValueError(
            f"unsupported sample width {sample_width} (expected {SAMPLE_WIDTH})"
        )
    if channels not in (1, 2):
        raise ValueError(f"unsupported channel count {channels}")
    if not pcm:
        return b""

    np = _require_numpy()
    samples = np.frombuffer(pcm, dtype="<i2")
    if channels == 2:
        # De-interleave so each channel is resampled on its own.
        usable = (len(samples) // 2) * 2
        samples = samples[:usable].reshape(-1, 2)
    else:
        samples = samples.reshape(-1, 1)

    if samples.shape[0] == 0:
        return b""

    if sample_rate != SAMPLE_RATE:
        src_n = samples.shape[0]
        dst_n = max(1, int(round(src_n * SAMPLE_RATE / float(sample_rate))))
        src_x = np.arange(src_n, dtype=np.float64)
        dst_x = np.linspace(0, src_n - 1, dst_n, dtype=np.float64)
        resampled = np.empty((dst_n, samples.shape[1]), dtype=np.float64)
        for c in range(samples.shape[1]):
            resampled[:, c] = np.interp(dst_x, src_x, samples[:, c])
        samples = np.clip(np.rint(resampled), -32768, 32767).astype("<i2")

    if samples.shape[1] == 1 and CHANNELS == 2:
        samples = np.repeat(samples, 2, axis=1)

    return samples.astype("<i2").tobytes()


class StreamingPCMSource(_AudioSourceBase):
    """A ``discord.AudioSource`` fed incrementally with PCM.

    Lifecycle: ``feed`` zero or more times, then ``end`` (producer is done)
    or ``abort`` (barge-in / error). ``read`` returns 20 ms frames until the
    buffer is drained *and* the producer has ended, then returns ``b""`` so
    discord.py stops the player and fires its ``after`` callback.

    While the producer is still running but the buffer is empty, ``read``
    returns silence rather than ``b""`` — ending the source there would cut
    the reply at the first gap between two clauses. ``max_starve_seconds``
    bounds that: a producer that dies without calling ``end`` cannot hold
    the voice connection open forever.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS,
                 sample_width: int = SAMPLE_WIDTH, *,
                 max_starve_seconds: float = 10.0,
                 name: str = "streaming-tts"):
        self.name = name
        self._src_rate = int(sample_rate)
        self._src_channels = int(channels)
        self._src_width = int(sample_width)
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._producer_done = False
        self._aborted = False
        self._finished = False
        self._starve_frames = 0
        self._max_starve_frames = max(1, int(max_starve_seconds * 1000 / 20))
        # Set once the buffer has run dry after end() — the adapter waits on
        # this instead of guessing a duration from the byte count.
        self.drained = threading.Event()
        # Total 20 ms frames actually handed to Discord; the adapter reports
        # it so "we streamed but nothing was audible" is distinguishable.
        self.frames_played = 0

    # -- producer side (gateway event loop) ---------------------------------

    def feed(self, pcm: bytes) -> None:
        """Append one PCM chunk in the producer's declared format."""
        if not pcm:
            return
        with self._lock:
            if self._aborted or self._producer_done:
                return
        converted = to_discord_pcm(
            pcm, self._src_rate, self._src_channels, self._src_width
        )
        if not converted:
            return
        with self._lock:
            if self._aborted or self._producer_done:
                return
            self._buf.extend(converted)
            self._starve_frames = 0

    def feed_native(self, pcm: bytes) -> None:
        """Append PCM that is already 48 kHz stereo s16le (no conversion).

        Used for the warm-up lead of silence the adapter prepends: it is
        generated in Discord's own format, and running it through the
        converter would be a no-op that can only introduce rounding.
        """
        if not pcm:
            return
        with self._lock:
            if self._aborted or self._producer_done:
                return
            self._buf.extend(pcm)
            self._starve_frames = 0

    def end(self) -> None:
        """Signal that no more audio will be fed; play out what is buffered."""
        with self._lock:
            self._producer_done = True
            if not self._buf:
                self._finished = True
        if self._finished:
            self.drained.set()

    def abort(self) -> None:
        """Drop buffered audio and end immediately (barge-in, error)."""
        with self._lock:
            self._aborted = True
            self._producer_done = True
            self._finished = True
            self._buf.clear()
        self.drained.set()

    @property
    def aborted(self) -> bool:
        with self._lock:
            return self._aborted

    @property
    def pending_bytes(self) -> int:
        with self._lock:
            return len(self._buf)

    # -- consumer side (discord.py player thread) ---------------------------

    def is_opus(self) -> bool:
        return False

    def read(self) -> bytes:
        with self._lock:
            if self._finished:
                return b""

            if len(self._buf) >= FRAME_SIZE:
                frame = bytes(self._buf[:FRAME_SIZE])
                del self._buf[:FRAME_SIZE]
                self._starve_frames = 0
                self.frames_played += 1
                return frame

            if self._producer_done:
                # Last partial frame: pad rather than drop, then stop.
                if self._buf:
                    frame = bytes(self._buf) + b"\x00" * (FRAME_SIZE - len(self._buf))
                    self._buf.clear()
                    self._finished = True
                    self.frames_played += 1
                    self.drained.set()
                    return frame
                self._finished = True
                self.drained.set()
                return b""

            # Underrun while the producer is still working.
            self._starve_frames += 1
            if self._starve_frames > self._max_starve_frames:
                logger.warning(
                    "Streaming TTS source %s starved for %.1fs — ending",
                    self.name, self._starve_frames * 0.02,
                )
                self._finished = True
                self.drained.set()
                return b""
            self.frames_played += 1
            return SILENCE_FRAME

    def cleanup(self) -> None:
        """discord.py calls this when the player stops."""
        with self._lock:
            self._finished = True
            self._buf.clear()
        self.drained.set()

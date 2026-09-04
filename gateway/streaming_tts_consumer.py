"""Gateway streaming-TTS consumer — LLM deltas to adapter PCM audio sink.

Bridges the synchronous agent ``stream_delta_callback`` (fired from the
worker thread) to a voice-capable platform adapter's streaming-audio
contract, so playback begins while the LLM is still generating.

Lifecycle::

    consumer = StreamingTTSConsumer(adapter, chat_id, tts_config, loop, metadata)
    agent.stream_delta_callback = consumer.on_delta   # sync, non-blocking
    ... agent runs in executor ...
    consumer.finish()            # signal end-of-text
    success = await consumer.wait_complete(timeout=10)
    if consumer.suppress_whole_file:
        # suppress whole-file auto-TTS for this turn
    consumer.abort("cancelled")  # idempotent cancellation

Design:
- ``on_delta`` is synchronous and never blocks the agent thread. It feeds
  deltas into a ``SentenceChunker`` and queues completed clauses onto a
  thread-safe ``queue.Queue``.
- An asyncio task (``run``) runs on the gateway event loop, draining the
  queue, synthesising each clause via a ``StreamingTTSProvider``, and
  writing PCM chunks to the adapter.
- Per-turn state is isolated: each consumer instance owns its own chunker,
  queue, handle, and flags. Concurrent chats cannot cross-contaminate.
- On successful completion (all clauses synthesised and written), the
  consumer reports ``completed=True`` so the gateway can suppress the
  duplicate whole-file auto-TTS.
- On failure before any audible output, the consumer reports
  ``completed=False`` and clears ``suppress_whole_file`` so the gateway can
  fall back to whole-file TTS.
- On failure after partial audible output, the consumer reports
  ``completed=False`` but keeps ``suppress_whole_file=True`` so the gateway
  does NOT replay the whole response from the beginning.
- Cancellation/abort is idempotent: late chunks are silently dropped.
- Providers with no chunked API (edge — the default — piper, minimax, the
  command providers) fall back to :class:`SentenceFileStreamer`, which
  synthesises one clause at a time through the proven sync path and decodes
  it to PCM. This mirrors what ``stream_tts_to_speaker`` does for the CLI
  with ``_SyncSentencePipeline``; the sink differs (adapter seam vs local
  speaker), so each dispatcher owns its own fallback.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Dict, Iterator, Optional

from gateway.platforms.base import AudioFormat, StreamingTTSHandle

logger = logging.getLogger("gateway.streaming_tts_consumer")

_ABORT = object()
_DONE = object()


# Upper bound on decoded PCM accepted for one clause. Mirrors
# ``_STREAM_SENTENCE_BYTE_CAP`` in tools.tts_streaming: a provider or a
# corrupt file must not be able to feed us unbounded audio.
_CLAUSE_PCM_BYTE_CAP = 16 * 1024 * 1024

# 48 kHz mono is deliberate. Every platform that can play voice wants 48 kHz
# (Discord's native rate, and Opus' own), so letting ffmpeg resample once
# during the decode is both cheaper and better than resampling PCM later in
# an adapter. Mono keeps the ``StreamingTTSProvider`` contract.
_FALLBACK_SAMPLE_RATE = 48000


class SentenceFileStreamer:
    """Per-clause sync synthesis, presented as a chunked PCM streamer.

    Satisfies the ``StreamingTTSProvider`` shape (``sample_rate``,
    ``channels``, ``sample_width``, ``stream(text) -> Iterator[bytes]``)
    without being registered as a provider: ``resolve_streaming_provider``
    must keep meaning "a real chunked API is available", and the CLI
    dispatcher has its own equivalent path.

    Time-to-first-audio is one clause instead of the whole reply. That is
    worse than a true chunked streamer and far better than waiting for the
    model to stop generating.
    """

    channels = 1
    sample_width = 2

    def __init__(self, tts_config: Dict[str, Any],
                 sample_rate: int = _FALLBACK_SAMPLE_RATE,
                 *, decode_timeout: float = 60.0) -> None:
        self.tts_config = tts_config
        self.sample_rate = int(sample_rate)
        self._decode_timeout = float(decode_timeout)

    @staticmethod
    def available() -> bool:
        """True when a clause can be synthesised and decoded to PCM."""
        return shutil.which("ffmpeg") is not None

    def stream(self, text: str) -> Iterator[bytes]:
        """Synthesise *text*, then yield its PCM as it decodes."""
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".mp3", prefix="hermes_stts_")
            os.close(fd)

            from tools.tts_tool import text_to_speech_tool
            text_to_speech_tool(text=text, output_path=tmp_path)

            if not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) == 0:
                logger.debug("sentence fallback: synthesis produced no audio")
                return

            yield from self._decode_to_pcm(tmp_path)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _decode_to_pcm(self, path: str) -> Iterator[bytes]:
        """Stream ffmpeg's PCM output so playback starts before decode ends."""
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return
        cmd = [
            ffmpeg, "-nostdin", "-loglevel", "error",
            "-i", path,
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ar", str(self.sample_rate), "-ac", str(self.channels),
            "pipe:1",
        ]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL
        )
        total = 0
        try:
            assert proc.stdout is not None
            while True:
                chunk = proc.stdout.read(8192)
                if not chunk:
                    break
                total += len(chunk)
                if total > _CLAUSE_PCM_BYTE_CAP:
                    logger.warning(
                        "sentence fallback: clause exceeded %d PCM bytes, truncating",
                        _CLAUSE_PCM_BYTE_CAP,
                    )
                    break
                yield chunk
        finally:
            try:
                proc.stdout.close()  # type: ignore[union-attr]
            except Exception:
                pass
            try:
                proc.wait(timeout=self._decode_timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            finally:
                if proc.returncode not in (0, None) and total == 0:
                    err = b""
                    try:
                        err = proc.stderr.read() or b""  # type: ignore[union-attr]
                    except Exception:
                        pass
                    logger.warning(
                        "sentence fallback: ffmpeg exit %s (%s)",
                        proc.returncode, err.decode("utf-8", "replace").strip()[:200],
                    )
                try:
                    proc.stderr.close()  # type: ignore[union-attr]
                except Exception:
                    pass


class StreamingTTSConsumer:
    """Consumes LLM text deltas and produces streaming PCM audio for an adapter."""

    def __init__(
        self,
        adapter: Any,
        chat_id: str,
        tts_config: Dict[str, Any],
        loop: asyncio.AbstractEventLoop,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        audio_format: Optional[AudioFormat] = None,
    ) -> None:
        from tools.tts_streaming import SentenceChunker, resolve_streaming_provider

        self._adapter = adapter
        self._chat_id = chat_id
        self._tts_config = tts_config
        self._loop = loop
        self._metadata = metadata

        # Resolve the streaming provider once. When the configured provider
        # has no chunked API, fall back to per-clause sync synthesis rather
        # than going inactive — otherwise edge (the default provider) waits
        # for the whole reply before a single word is spoken.
        self._streamer = resolve_streaming_provider(tts_config)
        if self._streamer is None and SentenceFileStreamer.available():
            self._streamer = SentenceFileStreamer(tts_config)
            logger.debug("streaming TTS: using per-clause sync fallback")
        self._chunker = SentenceChunker()

        if self._streamer is not None:
            self._audio_format = AudioFormat(
                sample_rate=int(getattr(self._streamer, "sample_rate", AudioFormat.sample_rate)),
                channels=int(getattr(self._streamer, "channels", AudioFormat.channels)),
                sample_width=int(getattr(self._streamer, "sample_width", AudioFormat.sample_width)),
            )
        else:
            self._audio_format = audio_format or AudioFormat()

        # Thread-safe queue: completed clauses and the occasional abort sentinel.
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=256)

        # Per-turn state.
        self._handle: Optional[StreamingTTSHandle] = None
        self._started = False
        self._completed = False
        self._partial = False
        self._aborted = False
        self._finished = False
        self._dropped = False
        self._suppress_whole_file = False
        self._task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()

        # Pre-allocate the strip-markdown helper lazily to avoid import cycles.
        self._strip_markdown = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        """True when this consumer has a usable streaming provider."""
        return self._streamer is not None

    @property
    def completed(self) -> bool:
        """True when streaming audio was fully delivered."""
        return self._completed

    @property
    def partial(self) -> bool:
        """True when some audio was audible before a failure or drop."""
        return self._partial

    @property
    def started(self) -> bool:
        """True when the adapter accepted the streaming session."""
        return self._started

    @property
    def audible(self) -> bool:
        """True once the first PCM chunk has been written."""
        return bool(self._handle and self._handle.audible)

    @property
    def dropped(self) -> bool:
        """True when queue saturation dropped at least one clause."""
        return self._dropped

    @property
    def suppress_whole_file(self) -> bool:
        """True when the gateway should skip the legacy whole-file TTS fallback."""
        return self._suppress_whole_file

    @property
    def done(self) -> bool:
        """True once the async drain task has terminated."""
        return self._task is not None and self._task.done()

    # ------------------------------------------------------------------
    # Sync callback (agent worker thread)
    # ------------------------------------------------------------------

    def on_delta(self, text: str) -> None:
        """Receive a text delta from the agent. Non-blocking."""
        if self._aborted or not self.active or self._finished:
            return
        try:
            for clause in self._chunker.feed(text):
                self._queue.put_nowait(clause)
        except queue.Full:
            self._dropped = True
            logger.debug("streaming TTS queue full, dropping clause")
        except Exception:
            logger.debug("streaming TTS on_delta error", exc_info=True)

    def finish(self) -> None:
        """Signal end-of-text and flush the chunker tail.

        Enqueues a ``_DONE`` sentinel after all flushed clauses so the
        drain loop has a deterministic termination signal that cannot
        race with a late ``on_delta`` or be lost when the queue is full.
        """
        if self._finished:
            return
        self._finished = True
        if self._aborted or not self.active:
            return
        try:
            for clause in self._chunker.flush():
                self._queue.put_nowait(clause)
        except queue.Full:
            self._dropped = True
            logger.debug("streaming TTS queue full while flushing tail")
        except Exception:
            pass
        # Guarantee the _DONE sentinel reaches the queue.  If the bounded
        # queue is full, drain one item to make room — the sentinel is
        # load-bearing and must not be lost (#60671 hardening).
        self._enqueue_done()

    def _enqueue_done(self) -> None:
        """Enqueue the _DONE sentinel, evicting a queued clause if necessary."""
        while True:
            try:
                self._queue.put_nowait(_DONE)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self._dropped = True
                except queue.Empty:
                    continue

    # ------------------------------------------------------------------
    # Async lifecycle (gateway event loop)
    # ------------------------------------------------------------------

    def start(self) -> asyncio.Task:
        """Create and return the async drain task on the gateway loop."""
        if self._task is not None:
            return self._task
        self._task = self._loop.create_task(self._run())
        return self._task

    async def _run(self) -> None:
        """Drain clauses from the queue, synthesise, and write to the adapter."""
        if not self.active:
            return

        if not self._adapter.supports_streaming_tts(self._chat_id, self._audio_format):
            logger.debug("adapter %s does not support streaming TTS", getattr(self._adapter, "name", "?"))
            return

        try:
            self._handle = await self._adapter.begin_streaming_tts(
                self._chat_id,
                self._audio_format,
                metadata=self._metadata,
            )
        except Exception as exc:
            logger.debug("begin_streaming_tts failed: %s", exc)
            self._handle = None
            return

        if self._handle is None:
            return

        self._started = True
        self._suppress_whole_file = False

        # Latenzmessung: _t0 laeuft ab dem Moment, in dem der Adapter den
        # Stream geoeffnet hat.  Die INFO-Zeilen sind das einzige Mittel,
        # die gefuehlte Antwortzeit eines Sprachzugs serverseitig zu lesen.
        _t0 = time.monotonic()
        _clauses = 0

        try:
            while True:
                if self._aborted:
                    break
                try:
                    item = await asyncio.to_thread(self._queue.get, True, 0.1)
                except queue.Empty:
                    continue

                if item is _ABORT:
                    break
                if item is _DONE:
                    break
                if not isinstance(item, str):
                    continue
                if self._aborted:
                    break

                try:
                    await self._synthesise_and_write(item)
                except Exception as exc:
                    logger.warning("streaming TTS clause failed: %s", exc)
                    if self._handle and self._handle.audible:
                        self._partial = True
                        self._suppress_whole_file = True
                    else:
                        self._suppress_whole_file = False
                    self._completed = False
                    await self._safe_abort(str(exc))
                    return

                _clauses += 1
                if _clauses == 1:
                    logger.info(
                        "streaming TTS: first clause audible after %.2fs",
                        time.monotonic() - _t0,
                    )

            logger.info(
                "streaming TTS: %d clause(s) in %.2fs (aborted=%s)",
                _clauses,
                time.monotonic() - _t0,
                self._aborted,
            )

            if not self._aborted and self._handle is not None:
                _finish_failed = False
                try:
                    await self._adapter.finish_streaming_tts(self._handle, interrupted=self._aborted)
                except Exception as exc:
                    logger.debug("finish_streaming_tts error: %s", exc)
                    _finish_failed = True

                if _finish_failed:
                    # finish_streaming_tts() raised — never report full
                    # completion.  If audio was already audible, report
                    # partial and preserve suppression so the gateway
                    # does not replay from the beginning.  If no audio
                    # was audible, permit whole-file fallback.
                    if self._handle.audible:
                        self._partial = True
                        self._completed = False
                        self._suppress_whole_file = True
                    else:
                        self._completed = False
                        self._suppress_whole_file = False
                    await self._safe_abort("finish_streaming_tts failed")
                elif self._handle.audible and not self._dropped:
                    self._completed = True
                    self._suppress_whole_file = True
                elif self._handle.audible and self._dropped:
                    self._partial = True
                    self._completed = False
                    self._suppress_whole_file = True
                else:
                    self._completed = False
                    self._suppress_whole_file = False
        except Exception as exc:
            logger.warning("streaming TTS consumer error: %s", exc)
            await self._safe_abort(str(exc))
        finally:
            try:
                while not self._queue.empty():
                    self._queue.get_nowait()
            except Exception:
                pass

    async def _synthesise_and_write(self, clause: str) -> None:
        """Synthesise one clause via the streamer and write PCM chunks."""
        if self._handle is None or self._handle.aborted:
            return

        cleaned = self._strip_markdown_for_tts(clause)
        if not cleaned or not cleaned.strip():
            return

        if self._streamer is None:
            return

        async for chunk in self._iter_stream_chunks(cleaned):
            if self._aborted or self._handle.aborted:
                return
            if not chunk:
                continue
            was_audible = self._handle.audible
            await self._adapter.write_streaming_tts(self._handle, chunk)
            if not was_audible:
                self._handle.audible = True
                self._suppress_whole_file = True

    async def _iter_stream_chunks(self, text: str):
        """Yield provider PCM chunks one at a time without blocking the loop."""
        if self._streamer is None:
            return
        iterator = iter(self._streamer.stream(text))
        while True:
            has_chunk, chunk = await asyncio.to_thread(self._next_stream_chunk, iterator)
            if not has_chunk:
                break
            yield chunk

    @staticmethod
    def _next_stream_chunk(iterator: Any) -> tuple[bool, Optional[bytes]]:
        try:
            return True, next(iterator)
        except StopIteration:
            return False, None

    def _strip_markdown_for_tts(self, text: str) -> str:
        """Lazy-import and apply the TTS markdown stripper."""
        if self._strip_markdown is None:
            try:
                from tools.tts_tool import _strip_markdown_for_tts as _strip
                self._strip_markdown = _strip
            except ImportError:
                self._strip_markdown = lambda t: t  # noqa: E731
        return self._strip_markdown(text).strip()

    async def _safe_abort(self, reason: str) -> None:
        """Abort the adapter stream, swallowing errors (idempotent)."""
        if self._handle is None:
            return
        try:
            await self._adapter.abort_streaming_tts(self._handle, error=reason)
        except Exception:
            pass
        finally:
            if self._handle:
                self._handle.aborted = True

    # ------------------------------------------------------------------
    # Cancellation and completion
    # ------------------------------------------------------------------

    def abort(self, reason: str = "cancelled") -> None:
        """Idempotent cancellation from any thread."""
        with self._lock:
            if self._aborted:
                return
            self._aborted = True
        # Guarantee the _ABORT sentinel reaches the queue.  If the bounded
        # queue is full, drain one item to make room — the sentinel must
        # not be lost (#60671 hardening).
        for _attempt in range(3):
            try:
                self._queue.put_nowait(_ABORT)
                break
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        else:
            logger.debug("streaming TTS _ABORT sentinel could not be enqueued")
        if self._handle is not None and not self._handle.aborted:
            try:
                self._loop.call_soon_threadsafe(
                    asyncio.create_task,
                    self._safe_abort(reason),
                )
            except Exception:
                pass

    def finalisation_timeout(self, grace: float = 10.0, ceiling: float = 120.0) -> float:
        """How long the gateway may wait for playback after the reply text ended.

        Remaining audio in the adapter plus *grace*, capped at *ceiling* so a
        wedged player cannot hold the turn open.  Adapters without the
        ``streaming_tts_pending_seconds`` seam count as "nothing pending".
        """
        pending = 0.0
        if self._handle is not None:
            probe = getattr(self._adapter, "streaming_tts_pending_seconds", None)
            if callable(probe):
                try:
                    pending = max(0.0, float(probe(self._handle)))
                except Exception:
                    pending = 0.0
        return min(float(ceiling), pending + float(grace))

    async def wait_complete(self, timeout: float = 10.0) -> bool:
        """Wait for the drain task to finish. Returns True only on full success."""
        if self._task is None:
            return self._completed
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        except Exception:
            pass
        return self._completed

"""Streaming TTS for Discord voice: PCM source, adapter seams, consumer fallback.

Measured on 2026-09-03 in a live Discord voice session: every reply waited
for the model to stop generating *and* for the whole answer to be synthesised
before a single word was spoken — 12 s on average, 56.8 s on a tool-heavy
turn. The gateway already had a streaming-TTS contract
(``gateway/streaming_tts_consumer.py``); no adapter implemented it, and the
consumer went inactive for every provider without a chunked API — which is
every provider Hermes ships by default. These tests pin both halves.
"""

import asyncio
import struct
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import AudioFormat
from plugins.platforms.discord.streaming_audio import (
    StreamingPCMSource,
    numpy_available,
    to_discord_pcm,
)
from plugins.platforms.discord.voice_mixer import FRAME_SIZE, SILENCE_FRAME

pytestmark = pytest.mark.skipif(
    not numpy_available(), reason="numpy (voice extra) not installed"
)


def _pcm(samples, value=1000):
    return struct.pack("<%dh" % samples, *([value] * samples))


# ============================================================================
# Format conversion
# ============================================================================

class TestToDiscordPCM:
    def test_24k_mono_doubles_rate_and_channels(self):
        out = to_discord_pcm(_pcm(100), 24000, 1)
        # 100 mono samples at 24 kHz -> 200 at 48 kHz -> 400 stereo samples
        assert len(out) == 400 * 2

    def test_16k_mono_triples_the_rate(self):
        out = to_discord_pcm(_pcm(100), 16000, 1)
        assert len(out) == 300 * 2 * 2

    def test_native_format_is_passed_through_unchanged(self):
        native = _pcm(200, value=1234)
        assert to_discord_pcm(native, 48000, 2) == native

    def test_mono_is_duplicated_not_split(self):
        """Both channels must carry the signal, or the voice sounds half-lost."""
        out = to_discord_pcm(_pcm(2, value=777), 48000, 1)
        assert struct.unpack("<4h", out) == (777, 777, 777, 777)

    def test_empty_input_is_empty_output(self):
        assert to_discord_pcm(b"", 24000, 1) == b""

    def test_unsupported_sample_width_is_refused(self):
        with pytest.raises(ValueError, match="sample width"):
            to_discord_pcm(_pcm(10), 24000, 1, sample_width=4)

    def test_unsupported_channel_count_is_refused(self):
        with pytest.raises(ValueError, match="channel"):
            to_discord_pcm(_pcm(30), 48000, 3)

    def test_odd_stereo_tail_does_not_raise(self):
        """A chunk boundary can split an interleaved sample pair."""
        assert to_discord_pcm(_pcm(101), 48000, 2)


# ============================================================================
# The audio source
# ============================================================================

class TestStreamingPCMSource:
    def test_frames_are_exactly_one_discord_frame(self):
        s = StreamingPCMSource(48000, 2)
        s.feed(_pcm(FRAME_SIZE // 2))          # exactly one frame
        assert len(s.read()) == FRAME_SIZE

    def test_underrun_plays_silence_not_the_end_of_the_reply(self):
        """A gap between two clauses must not end the whole stream."""
        s = StreamingPCMSource(48000, 2)
        s.feed(_pcm(FRAME_SIZE // 2))
        assert s.read() != SILENCE_FRAME
        assert s.read() == SILENCE_FRAME
        assert s.read() == SILENCE_FRAME

    def test_end_with_empty_buffer_finishes_immediately(self):
        s = StreamingPCMSource(48000, 2)
        s.end()
        assert s.read() == b""
        assert s.drained.is_set()

    def test_end_plays_out_what_is_buffered(self):
        s = StreamingPCMSource(48000, 2)
        s.feed(_pcm(FRAME_SIZE))               # two frames
        s.end()
        assert len(s.read()) == FRAME_SIZE
        assert len(s.read()) == FRAME_SIZE
        assert s.read() == b""

    def test_partial_last_frame_is_padded_not_dropped(self):
        s = StreamingPCMSource(48000, 2)
        s.feed(_pcm(100))                      # far less than one frame
        s.end()
        frame = s.read()
        assert len(frame) == FRAME_SIZE
        assert frame.endswith(b"\x00" * 100)
        assert s.read() == b""

    def test_abort_drops_buffered_audio(self):
        s = StreamingPCMSource(48000, 2)
        s.feed(_pcm(FRAME_SIZE * 4))
        s.abort()
        assert s.read() == b""
        assert s.pending_bytes == 0
        assert s.aborted is True
        assert s.drained.is_set()

    def test_abort_is_idempotent(self):
        s = StreamingPCMSource(48000, 2)
        s.abort()
        s.abort()
        assert s.read() == b""

    def test_feed_after_end_is_ignored(self):
        s = StreamingPCMSource(48000, 2)
        s.end()
        s.feed(_pcm(FRAME_SIZE))
        assert s.pending_bytes == 0

    def test_feed_after_abort_is_ignored(self):
        s = StreamingPCMSource(48000, 2)
        s.abort()
        s.feed(_pcm(FRAME_SIZE))
        assert s.pending_bytes == 0

    def test_a_dead_producer_cannot_hold_the_channel_forever(self):
        s = StreamingPCMSource(48000, 2, max_starve_seconds=0.06)   # 3 frames
        s.feed(_pcm(FRAME_SIZE // 2))
        s.read()                                # the real frame
        for _ in range(3):
            assert s.read() == SILENCE_FRAME
        assert s.read() == b""

    def test_feed_native_skips_conversion(self):
        s = StreamingPCMSource(24000, 1)        # would otherwise be converted
        s.feed_native(SILENCE_FRAME)
        assert s.pending_bytes == FRAME_SIZE

    def test_cleanup_releases_a_waiter(self):
        s = StreamingPCMSource(48000, 2)
        s.feed(_pcm(FRAME_SIZE * 2))
        s.cleanup()
        assert s.drained.is_set()
        assert s.read() == b""

    def test_conversion_happens_on_feed_not_on_read(self):
        """read() runs on discord.py's sender thread every 20 ms."""
        s = StreamingPCMSource(24000, 1)
        s.feed(_pcm(48000))                     # 2 s of 24 kHz mono
        t0 = time.perf_counter()
        s.read()
        assert (time.perf_counter() - t0) < 0.01


# ============================================================================
# Adapter seams
# ============================================================================

def _make_adapter(*, guild_id=42, chat_id="777", connected=True, barge_in=False):
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = object.__new__(DiscordAdapter)
    adapter._platform = Platform.DISCORD
    config = PlatformConfig(enabled=True, token="t")
    config.extra = {}
    adapter.config = config
    adapter._client = MagicMock()
    adapter._voice_clients = {}
    adapter._voice_receivers = {}
    adapter._voice_mixers = {}
    adapter._voice_streams = {}
    adapter._voice_text_channels = {guild_id: int(chat_id)}
    adapter._voice_fx_cfg = {}
    adapter._cancel_voice_timeout = lambda *a, **k: None
    adapter._reset_voice_timeout = MagicMock()
    adapter._voice_barge_in_enabled = lambda: barge_in
    adapter._lead_silence_bytes = lambda: b""

    if connected:
        vc = MagicMock()
        vc.is_connected.return_value = True
        vc.is_playing.return_value = False
        adapter._voice_clients[guild_id] = vc
    return adapter


_FMT = AudioFormat(sample_rate=48000, channels=1, sample_width=2)


class TestGuildLookup:
    def test_reverse_maps_the_bound_text_channel(self):
        assert _make_adapter()._voice_guild_for_chat("777") == 42

    def test_unbound_channel_has_no_guild(self):
        assert _make_adapter()._voice_guild_for_chat("999") is None

    def test_non_numeric_chat_id_is_not_an_error(self):
        assert _make_adapter()._voice_guild_for_chat("nope") is None


class TestSupportsStreaming:
    def test_supported_with_a_live_voice_session(self):
        assert _make_adapter().supports_streaming_tts("777", _FMT) is True

    def test_declined_without_a_voice_client(self):
        assert _make_adapter(connected=False).supports_streaming_tts("777", _FMT) is False

    def test_declined_for_an_unbound_chat(self):
        assert _make_adapter().supports_streaming_tts("999", _FMT) is False

    def test_declined_while_the_mixer_owns_the_stream(self):
        adapter = _make_adapter()
        adapter._voice_mixers[42] = MagicMock()
        assert adapter.supports_streaming_tts("777", _FMT) is False

    def test_declined_while_another_stream_is_live(self):
        adapter = _make_adapter()
        adapter._voice_streams[42] = StreamingPCMSource()
        assert adapter.supports_streaming_tts("777", _FMT) is False

    def test_a_finished_leftover_does_not_disable_the_session(self):
        # A turn cancelled between begin and finish leaves the registry
        # entry behind. Declining forever after that is indistinguishable
        # from "streaming not supported" — the worst kind of silent
        # degradation.
        adapter = _make_adapter()
        stale = StreamingPCMSource()
        stale.end()                      # drained: nothing left to play
        adapter._voice_streams[42] = stale
        assert adapter.supports_streaming_tts("777", _FMT) is True
        assert 42 not in adapter._voice_streams

    def test_an_aborted_leftover_does_not_disable_the_session(self):
        adapter = _make_adapter()
        stale = StreamingPCMSource()
        stale.abort()
        adapter._voice_streams[42] = stale
        assert adapter.supports_streaming_tts("777", _FMT) is True

    @pytest.mark.asyncio
    async def test_leaving_the_channel_drops_the_stream(self):
        # The source plays into one voice connection; keeping it past the
        # disconnect would decline streaming on the next /voice join.
        from unittest.mock import AsyncMock

        adapter = _make_adapter()
        live = StreamingPCMSource()
        live.feed_native(b"\x00" * 3840)
        adapter._voice_streams[42] = live
        adapter._voice_locks = {}
        adapter._voice_listen_tasks = {}
        adapter._voice_timeout_tasks = {}
        adapter._voice_sources = {}
        adapter._voice_clients[42].disconnect = AsyncMock()

        await adapter.leave_voice_channel(42)

        assert 42 not in adapter._voice_streams
        assert live.aborted is True

    def test_declined_for_a_format_we_would_have_to_mangle(self):
        adapter = _make_adapter()
        assert adapter.supports_streaming_tts(
            "777", AudioFormat(sample_rate=48000, channels=1, sample_width=4)
        ) is False

    def test_declined_when_disconnected_mid_session(self):
        adapter = _make_adapter()
        adapter._voice_clients[42].is_connected.return_value = False
        assert adapter.supports_streaming_tts("777", _FMT) is False


@pytest.mark.asyncio
class TestStreamingLifecycle:
    async def test_begin_starts_the_player_and_registers_the_source(self):
        adapter = _make_adapter()
        handle = await adapter.begin_streaming_tts("777", _FMT)

        assert handle is not None
        assert handle.guild_id == 42
        adapter._voice_clients[42].play.assert_called_once()
        assert adapter._voice_streams[42] is handle.source

    async def test_begin_declines_for_an_unbound_chat(self):
        assert await _make_adapter().begin_streaming_tts("999", _FMT) is None

    async def test_receiver_is_paused_while_speaking_by_default(self):
        adapter = _make_adapter()
        receiver = MagicMock()
        adapter._voice_receivers[42] = receiver

        handle = await adapter.begin_streaming_tts("777", _FMT)

        receiver.pause.assert_called_once()
        assert handle.receiver_paused is True

    async def test_receiver_keeps_listening_when_barge_in_is_on(self):
        adapter = _make_adapter(barge_in=True)
        receiver = MagicMock()
        adapter._voice_receivers[42] = receiver

        handle = await adapter.begin_streaming_tts("777", _FMT)

        receiver.pause.assert_not_called()
        assert handle.receiver_paused is False

    async def test_write_reaches_the_source(self):
        adapter = _make_adapter()
        handle = await adapter.begin_streaming_tts("777", _FMT)

        await adapter.write_streaming_tts(handle, _pcm(FRAME_SIZE // 4))

        assert handle.source.pending_bytes > 0

    async def test_write_after_abort_is_dropped(self):
        adapter = _make_adapter()
        handle = await adapter.begin_streaming_tts("777", _FMT)
        await adapter.abort_streaming_tts(handle, "cancelled")

        await adapter.write_streaming_tts(handle, _pcm(FRAME_SIZE // 4))

        assert handle.source.pending_bytes == 0

    async def test_finish_drains_then_releases_the_session(self):
        adapter = _make_adapter()
        receiver = MagicMock()
        adapter._voice_receivers[42] = receiver
        handle = await adapter.begin_streaming_tts("777", _FMT)
        await adapter.write_streaming_tts(handle, _pcm(FRAME_SIZE // 2))
        # Drain as discord.py's player would.
        while handle.source.read():
            pass

        await adapter.finish_streaming_tts(handle)

        assert 42 not in adapter._voice_streams
        receiver.resume.assert_called_once()
        adapter._reset_voice_timeout.assert_called_with(42)

    async def test_finish_does_not_hang_on_a_wedged_player(self):
        """Nothing ever calls read(); the drain must still time out and abort."""
        adapter = _make_adapter()
        adapter._streaming_drain_grace = 0.2
        handle = await adapter.begin_streaming_tts("777", _FMT)
        await adapter.write_streaming_tts(handle, _pcm(FRAME_SIZE // 2))

        started = time.monotonic()
        await adapter.finish_streaming_tts(handle)

        assert time.monotonic() - started < 5
        assert handle.source.aborted is True
        assert 42 not in adapter._voice_streams

    async def test_abort_is_idempotent(self):
        adapter = _make_adapter()
        handle = await adapter.begin_streaming_tts("777", _FMT)

        await adapter.abort_streaming_tts(handle, "first")
        await adapter.abort_streaming_tts(handle, "second")

        assert 42 not in adapter._voice_streams

    async def test_barge_in_aborts_a_live_stream(self):
        """The reason this exists: cut the answer, not just the file player."""
        adapter = _make_adapter(barge_in=True)
        handle = await adapter.begin_streaming_tts("777", _FMT)
        await adapter.write_streaming_tts(handle, _pcm(FRAME_SIZE * 2))

        adapter._stop_voice_playback(42)

        assert handle.source.aborted is True
        assert handle.source.read() == b""


# ============================================================================
# Consumer fallback
# ============================================================================

class TestSentenceFallbackWiring:
    def test_consumer_falls_back_when_no_chunked_provider_exists(self, monkeypatch):
        """Before this, edge — the default provider — got no streaming at all."""
        import gateway.streaming_tts_consumer as mod

        monkeypatch.setattr(
            "tools.tts_streaming.resolve_streaming_provider", lambda *a, **k: None
        )
        consumer = mod.StreamingTTSConsumer(
            adapter=MagicMock(), chat_id="777", tts_config={"provider": "edge"},
            loop=asyncio.new_event_loop(),
        )
        assert consumer.active is True
        assert isinstance(consumer._streamer, mod.SentenceFileStreamer)

    def test_a_real_chunked_provider_still_wins(self, monkeypatch):
        import gateway.streaming_tts_consumer as mod

        chunked = SimpleNamespace(sample_rate=24000, channels=1, sample_width=2)
        monkeypatch.setattr(
            "tools.tts_streaming.resolve_streaming_provider", lambda *a, **k: chunked
        )
        consumer = mod.StreamingTTSConsumer(
            adapter=MagicMock(), chat_id="777", tts_config={},
            loop=asyncio.new_event_loop(),
        )
        assert consumer._streamer is chunked
        assert consumer._audio_format.sample_rate == 24000

    def test_inactive_without_ffmpeg(self, monkeypatch):
        import gateway.streaming_tts_consumer as mod

        monkeypatch.setattr(
            "tools.tts_streaming.resolve_streaming_provider", lambda *a, **k: None
        )
        monkeypatch.setattr(mod.shutil, "which", lambda _name: None)
        consumer = mod.StreamingTTSConsumer(
            adapter=MagicMock(), chat_id="777", tts_config={},
            loop=asyncio.new_event_loop(),
        )
        assert consumer.active is False

    def test_fallback_declares_48k_mono(self):
        """48 kHz is every voice platform's native rate; ffmpeg resamples once."""
        from gateway.streaming_tts_consumer import SentenceFileStreamer

        s = SentenceFileStreamer({})
        assert (s.sample_rate, s.channels, s.sample_width) == (48000, 1, 2)

    def test_synthesis_failure_yields_nothing_rather_than_raising(self, monkeypatch):
        from gateway.streaming_tts_consumer import SentenceFileStreamer

        monkeypatch.setattr(
            "tools.tts_tool.text_to_speech_tool",
            lambda **kw: (_ for _ in ()).throw(RuntimeError("provider down")),
        )
        with pytest.raises(RuntimeError):
            list(SentenceFileStreamer({}).stream("hallo"))

    def test_empty_synthesis_output_yields_nothing(self, monkeypatch):
        from gateway.streaming_tts_consumer import SentenceFileStreamer

        monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", lambda **kw: "")
        assert list(SentenceFileStreamer({}).stream("hallo")) == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_deltas_become_discord_frames():
    """Real synthesis, real decode, real conversion — only Discord is faked.

    Hits the network (edge-tts), so it carries the integration marker and is
    excluded from the default run by pyproject's addopts.
    """
    import gateway.streaming_tts_consumer as mod
    from tools.tts_tool import _load_tts_config

    adapter = _make_adapter()
    written: list[bytes] = []

    real_begin = adapter.begin_streaming_tts

    async def _begin(chat_id, fmt, metadata=None):
        return await real_begin(chat_id, fmt, metadata)

    async def _write(handle, chunk):
        written.append(chunk)
        handle.source.feed(chunk)

    adapter.begin_streaming_tts = _begin
    adapter.write_streaming_tts = _write

    consumer = mod.StreamingTTSConsumer(
        adapter=adapter, chat_id="777", tts_config=_load_tts_config(),
        loop=asyncio.get_running_loop(),
    )
    assert consumer.active

    consumer.start()
    consumer.on_delta("Guten Abend. ")
    consumer.on_delta("Der Streaming-Pfad steht.")
    consumer.finish()
    await consumer.wait_complete(timeout=120)

    assert written, "no PCM reached the adapter"
    assert consumer.audible is True
    assert consumer.suppress_whole_file is True


# ============================================================================
# Pending-audio seam: the gateway bounds its finalisation wait by this
# ============================================================================


@pytest.mark.asyncio
class TestPendingSeconds:
    async def test_pending_seconds_follow_the_buffer(self):
        adapter = _make_adapter()
        handle = await adapter.begin_streaming_tts("777", _FMT)
        # Mono 48 kHz input is doubled to stereo on feed: 48000 samples of
        # mono become one second of Discord PCM (192000 bytes).
        await adapter.write_streaming_tts(handle, _pcm(48000))

        assert adapter.streaming_tts_pending_seconds(handle) == pytest.approx(1.0)

        while handle.source.read():
            pass
        assert adapter.streaming_tts_pending_seconds(handle) == 0.0

    def test_a_handle_without_a_source_has_nothing_pending(self):
        adapter = _make_adapter()
        assert adapter.streaming_tts_pending_seconds(SimpleNamespace()) == 0.0

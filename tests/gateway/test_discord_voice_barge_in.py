"""Discord voice: configurable end-of-utterance window and barge-in.

Both defaults were measured on 2026-09-03 in a live Discord voice session:
the fixed 1.5s silence window cut the speaker off mid-thought seven times,
and the receiver's pause during TTS made it impossible to interrupt a
69-second answer. These tests pin the knobs that fix that.
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform, PlatformConfig


# The voice knobs come from ``read_raw_config()["discord"]``, not from
# ``PlatformConfig.extra``. Reading the wrong source is indistinguishable
# from a key that was never set, so every test goes through the real path.
_DISCORD_BLOCK: dict = {}


@pytest.fixture(autouse=True)
def _patch_raw_config(monkeypatch):
    _DISCORD_BLOCK.clear()
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda *a, **k: {"discord": dict(_DISCORD_BLOCK)},
    )
    yield
    _DISCORD_BLOCK.clear()


def _make_adapter(extra=None):
    from plugins.platforms.discord.adapter import DiscordAdapter

    _DISCORD_BLOCK.clear()
    _DISCORD_BLOCK.update(extra or {})

    adapter = object.__new__(DiscordAdapter)
    adapter._platform = Platform.DISCORD
    config = PlatformConfig(enabled=True, token="t")
    config.extra = {}
    adapter.config = config
    adapter._allowed_user_ids = set()
    adapter._allowed_role_ids = set()
    adapter._voice_clients = {}
    adapter._voice_receivers = {}
    adapter._voice_listen_tasks = {}
    adapter._voice_timeout_tasks = {}
    adapter._voice_mixers = {}
    adapter._client = MagicMock()
    return adapter


def _make_receiver(silence_threshold=None):
    from plugins.platforms.discord.adapter import VoiceReceiver

    return VoiceReceiver(
        MagicMock(), allowed_user_ids=set(), silence_threshold=silence_threshold
    )


# ============================================================================
# Silence threshold
# ============================================================================

class TestSilenceThreshold:
    def test_default_is_the_class_constant(self):
        from plugins.platforms.discord.adapter import VoiceReceiver

        adapter = _make_adapter()
        assert adapter._voice_silence_threshold() == VoiceReceiver.SILENCE_THRESHOLD

    def test_config_override(self):
        adapter = _make_adapter({"voice_silence_threshold_seconds": 3.0})
        assert adapter._voice_silence_threshold() == 3.0

    def test_string_from_env_style_config(self):
        adapter = _make_adapter({"voice_silence_threshold_seconds": "2.5"})
        assert adapter._voice_silence_threshold() == 2.5

    @pytest.mark.parametrize("bad", ["abc", -1, 0, float("inf"), float("nan"), None])
    def test_invalid_values_fall_back_to_the_default(self, bad):
        from plugins.platforms.discord.adapter import VoiceReceiver

        adapter = _make_adapter({"voice_silence_threshold_seconds": bad})
        assert adapter._voice_silence_threshold() == VoiceReceiver.SILENCE_THRESHOLD

    def test_platform_config_extra_is_not_the_source(self):
        """Regression, 2026-09-03: the first cut read ``PlatformConfig.extra``.

        That dict never carries the top-level ``discord:`` block, so the knob
        looked configured and silently did nothing — a green run that proves
        only that the wrong dict was empty.
        """
        from plugins.platforms.discord.adapter import VoiceReceiver

        adapter = _make_adapter()
        adapter.config.extra = {
            "voice_silence_threshold_seconds": 9.0,
            "voice_barge_in": True,
        }

        assert adapter._voice_silence_threshold() == VoiceReceiver.SILENCE_THRESHOLD
        assert adapter._voice_barge_in_enabled() is False

    def test_absurd_value_is_capped(self):
        """A typo like 25 must not stall every utterance for 25 seconds."""
        from plugins.platforms.discord.adapter import DiscordAdapter

        adapter = _make_adapter({"voice_silence_threshold_seconds": 900})
        assert adapter._voice_silence_threshold() == DiscordAdapter._MAX_SILENCE_THRESHOLD

    def test_receiver_honours_the_instance_override(self):
        from plugins.platforms.discord.adapter import VoiceReceiver

        receiver = _make_receiver(silence_threshold=3.0)
        assert receiver.SILENCE_THRESHOLD == 3.0
        # The class default is untouched for every other receiver.
        assert VoiceReceiver.SILENCE_THRESHOLD == 1.5
        assert _make_receiver().SILENCE_THRESHOLD == 1.5

    def test_utterance_completes_only_after_the_configured_window(self):
        """A 2s pause ends the utterance at 1.5s but not at 3.0s."""
        bytes_per_second = 48000 * 2 * 2

        def _receiver_with(threshold):
            receiver = _make_receiver(silence_threshold=threshold)
            receiver._buffers[7] = bytearray(bytes_per_second)  # 1s of speech
            receiver._last_packet_time[7] = time.monotonic() - 2.0
            receiver._ssrc_to_user[7] = 99
            return receiver

        assert _receiver_with(1.5).check_silence() == [(99, bytes(bytes_per_second))]
        assert _receiver_with(3.0).check_silence() == []


# ============================================================================
# Barge-in
# ============================================================================

class TestBargeInFlag:
    def test_off_by_default(self):
        assert _make_adapter()._voice_barge_in_enabled() is False

    @pytest.mark.parametrize("value", [True, "true", "TRUE", "yes", "on", "1"])
    def test_truthy_values(self, value):
        assert _make_adapter({"voice_barge_in": value})._voice_barge_in_enabled() is True

    @pytest.mark.parametrize("value", [False, "false", "no", "off", "0"])
    def test_falsy_values(self, value):
        assert _make_adapter({"voice_barge_in": value})._voice_barge_in_enabled() is False


class TestStopVoicePlayback:
    def test_stops_a_playing_voice_client(self):
        adapter = _make_adapter()
        vc = MagicMock()
        vc.is_connected.return_value = True
        vc.is_playing.return_value = True
        adapter._voice_clients[42] = vc

        adapter._stop_voice_playback(42)

        vc.stop.assert_called_once()

    def test_silent_client_is_left_alone(self):
        adapter = _make_adapter()
        vc = MagicMock()
        vc.is_connected.return_value = True
        vc.is_playing.return_value = False
        adapter._voice_clients[42] = vc

        adapter._stop_voice_playback(42)

        vc.stop.assert_not_called()

    def test_mixer_path_takes_precedence(self):
        adapter = _make_adapter()
        mixer = MagicMock()
        mixer.speech_active = True
        adapter._voice_mixers[42] = mixer
        vc = MagicMock()
        vc.is_connected.return_value = True
        vc.is_playing.return_value = True
        adapter._voice_clients[42] = vc

        adapter._stop_voice_playback(42)

        mixer.stop_speech.assert_called_once()
        vc.stop.assert_not_called()

    def test_a_broken_client_does_not_raise(self):
        """Barge-in runs inside the listen loop — it must never kill it."""
        adapter = _make_adapter()
        vc = MagicMock()
        vc.is_connected.side_effect = RuntimeError("socket gone")
        adapter._voice_clients[42] = vc

        adapter._stop_voice_playback(42)  # must not raise

    def test_unknown_guild_is_a_no_op(self):
        _make_adapter()._stop_voice_playback(999)


# ============================================================================
# Listen loop
# ============================================================================

async def _run_one_listen_pass(adapter, guild_id=42):
    """Drive _voice_listen_loop for exactly one completed utterance."""
    receiver = adapter._voice_receivers[guild_id]
    task = asyncio.ensure_future(adapter._voice_listen_loop(guild_id))
    for _ in range(60):
        await asyncio.sleep(0.05)
        if adapter._processed:
            break
    receiver._running = False
    task.cancel()
    with __import__("contextlib").suppress(asyncio.CancelledError):
        await task


def _prepare_loop(adapter, guild_id=42):
    receiver = MagicMock()
    receiver._running = True
    receiver.check_silence.side_effect = [[(99, b"pcm")]] + [[]] * 200
    # A bare MagicMock returns a truthy object, which would fake an onset
    # on every poll and make "stopped once" impossible to assert.
    receiver.speech_in_progress.return_value = False
    adapter._voice_receivers[guild_id] = receiver

    vc = MagicMock()
    vc.is_connected.return_value = True
    vc.is_playing.return_value = True
    adapter._voice_clients[guild_id] = vc

    adapter._processed = []
    adapter._is_allowed_user = lambda *a, **k: True
    adapter._reset_voice_timeout = lambda *a, **k: None

    async def _process(gid, uid, pcm):
        adapter._processed.append((gid, uid, pcm))

    adapter._process_voice_input = _process
    return vc


@pytest.mark.asyncio
async def test_barge_in_cuts_playback_before_the_new_turn():
    adapter = _make_adapter({"voice_barge_in": True})
    vc = _prepare_loop(adapter)

    await _run_one_listen_pass(adapter)

    assert adapter._processed == [(42, 99, b"pcm")]
    vc.stop.assert_called_once()


@pytest.mark.asyncio
async def test_without_barge_in_playback_is_left_running():
    adapter = _make_adapter()
    vc = _prepare_loop(adapter)

    await _run_one_listen_pass(adapter)

    assert adapter._processed == [(42, 99, b"pcm")]
    vc.stop.assert_not_called()


# ============================================================================
# Playback pause
# ============================================================================

async def _play(adapter, monkeypatch, guild_id=42):
    """Call play_in_voice_channel with the ffmpeg/timeout machinery stubbed.

    The ``discord`` name on the adapter module is replaced wholesale rather
    than reached into: another module in tests/gateway sets it to None and
    never restores it, so anything that reads ``mod.discord.X`` here fails
    depending on collection order.
    """
    import plugins.platforms.discord.adapter as mod

    async def _timeout(_path):
        return 5.0

    adapter._playback_timeout_for_audio = _timeout
    adapter._cancel_voice_timeout = lambda *a, **k: None
    adapter._reset_voice_timeout = lambda *a, **k: None
    adapter._voice_fx_cfg = {}
    adapter._lead_silence_bytes = lambda: b""

    vc = adapter._voice_clients[guild_id]

    def _play_now(source, after=None):
        vc.is_playing.return_value = False
        if after:
            after(None)

    vc.play.side_effect = _play_now

    monkeypatch.setattr(
        mod,
        "discord",
        SimpleNamespace(
            FFmpegPCMAudio=MagicMock(),
            PCMVolumeTransformer=MagicMock(),
        ),
    )
    monkeypatch.setattr(mod, "resolve_ffmpeg_executable", lambda: "/usr/bin/ffmpeg")
    return await adapter.play_in_voice_channel(guild_id, "/tmp/reply.mp3")


def _prepare_playback(adapter, guild_id=42):
    vc = MagicMock()
    vc.is_connected.return_value = True
    vc.is_playing.return_value = False
    adapter._voice_clients[guild_id] = vc
    receiver = MagicMock()
    adapter._voice_receivers[guild_id] = receiver
    return receiver


@pytest.mark.asyncio
async def test_receiver_is_paused_while_speaking_by_default(monkeypatch):
    """Echo prevention: without headphones the bot would hear itself."""
    adapter = _make_adapter()
    receiver = _prepare_playback(adapter)

    assert await _play(adapter, monkeypatch) is True

    receiver.pause.assert_called_once()
    receiver.resume.assert_called_once()


@pytest.mark.asyncio
async def test_receiver_keeps_listening_when_barge_in_is_on(monkeypatch):
    adapter = _make_adapter({"voice_barge_in": True})
    receiver = _prepare_playback(adapter)

    assert await _play(adapter, monkeypatch) is True

    receiver.pause.assert_not_called()


# ============================================================================
# Speech onset — the reason barge-in was still useless after the first fix
# ============================================================================

def _fill(receiver, ssrc=7, seconds=1.0, age=0.0):
    """Put ``seconds`` of PCM into a receiver buffer, last packet ``age`` ago."""
    per_second = receiver.SAMPLE_RATE * receiver.CHANNELS * 2
    receiver._buffers[ssrc] = bytearray(int(per_second * seconds))
    receiver._last_packet_time[ssrc] = time.monotonic() - age


class TestSpeechInProgress:
    def test_empty_receiver_is_silent(self):
        assert _make_receiver().speech_in_progress() is False

    def test_live_audio_counts_as_speech(self):
        r = _make_receiver()
        _fill(r, seconds=1.0, age=0.0)
        assert r.speech_in_progress() is True

    def test_a_click_is_not_speech(self):
        """Below MIN_SPEECH_DURATION the audio would be discarded anyway."""
        r = _make_receiver()
        _fill(r, seconds=0.2, age=0.0)
        assert r.speech_in_progress() is False

    def test_a_finished_utterance_is_not_live(self):
        """The buffer is still full, but no packet has arrived for a while."""
        r = _make_receiver()
        _fill(r, seconds=3.0, age=1.0)
        assert r.speech_in_progress() is False

    def test_gap_boundary_is_configurable(self):
        r = _make_receiver()
        _fill(r, seconds=3.0, age=1.0)
        assert r.speech_in_progress(max_gap=2.0) is True

    def test_min_duration_is_configurable(self):
        r = _make_receiver()
        _fill(r, seconds=0.2, age=0.0)
        assert r.speech_in_progress(min_duration=0.1) is True

    def test_buffer_without_a_packet_time_is_ignored(self):
        """Stale entry from a cleared utterance must not read as speech."""
        r = _make_receiver()
        per_second = r.SAMPLE_RATE * r.CHANNELS * 2
        r._buffers[7] = bytearray(per_second)
        assert r.speech_in_progress() is False

    def test_onset_fires_far_earlier_than_check_silence(self):
        """The whole point: cut at 0.5s of speech, not 0.5s plus the window."""
        r = _make_receiver(silence_threshold=2.5)
        _fill(r, seconds=0.6, age=0.0)
        assert r.speech_in_progress() is True
        assert r.check_silence() == []


class TestBargeInMinSpeech:
    def test_default_matches_the_receiver_minimum(self):
        from plugins.platforms.discord.adapter import VoiceReceiver

        adapter = _make_adapter()
        assert adapter._voice_barge_in_min_speech() == VoiceReceiver.MIN_SPEECH_DURATION

    def test_config_override(self):
        adapter = _make_adapter({"voice_barge_in_min_speech_seconds": 0.3})
        assert adapter._voice_barge_in_min_speech() == 0.3

    @pytest.mark.parametrize("value", [0, -1, "nonsense", None, float("nan")])
    def test_nonsense_falls_back(self, value):
        from plugins.platforms.discord.adapter import VoiceReceiver

        adapter = _make_adapter({"voice_barge_in_min_speech_seconds": value})
        assert adapter._voice_barge_in_min_speech() == VoiceReceiver.MIN_SPEECH_DURATION

    def test_absurd_value_is_capped(self):
        adapter = _make_adapter({"voice_barge_in_min_speech_seconds": 99})
        assert adapter._voice_barge_in_min_speech() == 5.0


@pytest.mark.asyncio
async def test_loop_cuts_playback_on_onset_without_a_finished_utterance():
    """The failure of 2026-09-03: playback ran to the end of the utterance."""
    adapter = _make_adapter({"voice_barge_in": True})
    receiver = MagicMock()
    receiver._running = True
    receiver.check_silence.return_value = []          # nothing finished, ever
    receiver.speech_in_progress.return_value = True   # but somebody is talking
    adapter._voice_receivers[42] = receiver

    vc = MagicMock()
    vc.is_connected.return_value = True
    vc.is_playing.return_value = True
    adapter._voice_clients[42] = vc
    adapter._processed = []

    task = asyncio.ensure_future(adapter._voice_listen_loop(42))
    for _ in range(40):
        await asyncio.sleep(0.05)
        if vc.stop.called:
            break
    receiver._running = False
    task.cancel()
    with __import__("contextlib").suppress(asyncio.CancelledError):
        await task

    assert vc.stop.called
    assert adapter._processed == []


@pytest.mark.asyncio
async def test_onset_is_not_polled_when_barge_in_is_off():
    adapter = _make_adapter()
    receiver = MagicMock()
    receiver._running = True
    receiver.check_silence.return_value = []
    receiver.speech_in_progress.return_value = True
    adapter._voice_receivers[42] = receiver

    vc = MagicMock()
    vc.is_connected.return_value = True
    vc.is_playing.return_value = True
    adapter._voice_clients[42] = vc

    task = asyncio.ensure_future(adapter._voice_listen_loop(42))
    await asyncio.sleep(0.7)
    receiver._running = False
    task.cancel()
    with __import__("contextlib").suppress(asyncio.CancelledError):
        await task

    vc.stop.assert_not_called()


@pytest.mark.asyncio
async def test_config_is_read_once_not_five_times_a_second():
    """The poll runs at 5 Hz; reading config.yaml at that rate is a bug."""
    adapter = _make_adapter({"voice_barge_in": True})
    receiver = MagicMock()
    receiver._running = True
    receiver.check_silence.return_value = []
    receiver.speech_in_progress.return_value = False
    adapter._voice_receivers[42] = receiver

    calls = []
    real = adapter._voice_barge_in_enabled
    adapter._voice_barge_in_enabled = lambda: (calls.append(1), real())[1]

    task = asyncio.ensure_future(adapter._voice_listen_loop(42))
    await asyncio.sleep(0.9)
    receiver._running = False
    task.cancel()
    with __import__("contextlib").suppress(asyncio.CancelledError):
        await task

    assert len(calls) == 1


# ============================================================================
# Level gate — the reason barge-in fired on road noise (2026-09-05)
# ============================================================================

def _tone(receiver, seconds=1.0, amplitude=8000, ssrc=7, age=0.0):
    """Fill a buffer with a square wave of the given peak amplitude."""
    import struct

    frames = int(receiver.SAMPLE_RATE * seconds)
    sample = struct.pack("<h", amplitude) + struct.pack("<h", -amplitude)
    receiver._buffers[ssrc] = bytearray(sample * frames)
    receiver._last_packet_time[ssrc] = time.monotonic() - age


class TestPcmDbfs:
    def test_empty_and_silence_are_minus_infinity(self):
        from plugins.platforms.discord.adapter import pcm_dbfs

        assert pcm_dbfs(b"") == float("-inf")
        assert pcm_dbfs(bytes(4000)) == float("-inf")

    def test_full_scale_square_wave_is_zero_dbfs(self):
        from plugins.platforms.discord.adapter import pcm_dbfs

        r = _make_receiver()
        _tone(r, seconds=0.1, amplitude=32767)
        assert pcm_dbfs(bytes(r._buffers[7])) == pytest.approx(0.0, abs=0.01)

    def test_quiet_signal_reads_lower(self):
        from plugins.platforms.discord.adapter import pcm_dbfs

        r = _make_receiver()
        _tone(r, seconds=0.1, amplitude=328)  # ~ -40 dBFS
        assert pcm_dbfs(bytes(r._buffers[7])) == pytest.approx(-40.0, abs=0.1)

    def test_odd_trailing_byte_is_ignored(self):
        from plugins.platforms.discord.adapter import pcm_dbfs

        assert pcm_dbfs(b"\x00") == float("-inf")


class TestSpeechLevelGate:
    def test_no_gate_keeps_the_packets_only_behaviour(self):
        r = _make_receiver()
        _fill(r, seconds=1.0)               # digital silence
        assert r.speech_in_progress() is True

    def test_noise_below_the_gate_is_not_speech(self):
        r = _make_receiver()
        _tone(r, seconds=1.0, amplitude=100)  # ~ -50 dBFS
        assert r.speech_in_progress(min_dbfs=-40.0) is False

    def test_speech_above_the_gate_cuts(self):
        r = _make_receiver()
        _tone(r, seconds=1.0, amplitude=3000)  # ~ -21 dBFS
        assert r.speech_in_progress(min_dbfs=-40.0) is True

    def test_gate_reads_the_recent_window_not_the_whole_buffer(self):
        """A long quiet lead-in must not drown the words that follow."""
        r = _make_receiver()
        per_second = r.SAMPLE_RATE * r.CHANNELS * 2
        _tone(r, seconds=0.6, amplitude=3000)
        loud = bytes(r._buffers[7])
        r._buffers[7] = bytearray(per_second * 5) + bytearray(loud)
        assert r.speech_in_progress(min_duration=0.5, min_dbfs=-40.0) is True

    def test_verdict_is_logged_once_per_buffer(self, caplog):
        import logging

        r = _make_receiver()
        _tone(r, seconds=1.0, amplitude=100)
        with caplog.at_level(logging.INFO, logger="plugins.platforms.discord.adapter"):
            for _ in range(5):
                r.speech_in_progress(min_dbfs=-40.0)
        lines = [m for m in caplog.messages if m.startswith("Speech onset")]
        assert len(lines) == 1
        assert "ignored" in lines[0]

    def test_verdict_flip_is_logged_again(self, caplog):
        import logging

        r = _make_receiver()
        _tone(r, seconds=1.0, amplitude=100)
        with caplog.at_level(logging.INFO, logger="plugins.platforms.discord.adapter"):
            r.speech_in_progress(min_dbfs=-40.0)
            _tone(r, seconds=1.0, amplitude=3000)
            assert r.speech_in_progress(min_dbfs=-40.0) is True
        lines = [m for m in caplog.messages if m.startswith("Speech onset")]
        assert [("speech" in m) for m in lines] == [False, True]

    def test_verdict_is_forgotten_with_the_buffer(self):
        r = _make_receiver(silence_threshold=0.1)
        _tone(r, seconds=1.0, amplitude=3000, age=0.2)
        r.speech_in_progress(min_dbfs=-40.0)
        r.check_silence()
        assert r._onset_verdict == {}


class TestBargeInMinDbfs:
    def test_default(self):
        adapter = _make_adapter()
        assert adapter._voice_barge_in_min_dbfs() == -40.0

    def test_config_override(self):
        adapter = _make_adapter({"voice_barge_in_min_dbfs": -33})
        assert adapter._voice_barge_in_min_dbfs() == -33.0

    @pytest.mark.parametrize("value", [-90, -120, "-100"])
    def test_at_or_below_minus_ninety_switches_the_gate_off(self, value):
        adapter = _make_adapter({"voice_barge_in_min_dbfs": value})
        assert adapter._voice_barge_in_min_dbfs() is None

    @pytest.mark.parametrize("value", [5, "nonsense", None, float("nan"), float("inf")])
    def test_nonsense_falls_back(self, value):
        adapter = _make_adapter({"voice_barge_in_min_dbfs": value})
        assert adapter._voice_barge_in_min_dbfs() == -40.0


def _square(seconds=0.6, amplitude=8000):
    import struct

    return (struct.pack("<h", amplitude) + struct.pack("<h", -amplitude)) * int(48000 * seconds)


@pytest.mark.asyncio
async def test_completed_quiet_utterance_does_not_cut_playback():
    """Codex-Review 2026-09-05: the fallback stop bypassed the level gate."""
    adapter = _make_adapter({"voice_barge_in": True, "voice_barge_in_min_dbfs": -40})
    vc = _prepare_loop(adapter)
    receiver = adapter._voice_receivers[42]
    receiver.check_silence.side_effect = [[(99, _square(amplitude=100))]] + [[]] * 200  # ~ -50 dBFS

    await _run_one_listen_pass(adapter)

    vc.stop.assert_not_called()
    assert len(adapter._processed) == 1      # still transcribed, just not a barge-in


@pytest.mark.asyncio
async def test_completed_loud_utterance_still_cuts_playback():
    adapter = _make_adapter({"voice_barge_in": True, "voice_barge_in_min_dbfs": -40})
    vc = _prepare_loop(adapter)
    receiver = adapter._voice_receivers[42]
    receiver.check_silence.side_effect = [[(99, _square(amplitude=3000))]] + [[]] * 200  # ~ -21 dBFS

    await _run_one_listen_pass(adapter)

    vc.stop.assert_called_once()


@pytest.mark.asyncio
async def test_loop_passes_the_gate_to_the_receiver():
    adapter = _make_adapter({"voice_barge_in": True, "voice_barge_in_min_dbfs": -35})
    receiver = MagicMock()
    receiver._running = True
    receiver.check_silence.return_value = []
    receiver.speech_in_progress.return_value = False
    adapter._voice_receivers[42] = receiver
    adapter._voice_clients[42] = MagicMock()

    task = asyncio.ensure_future(adapter._voice_listen_loop(42))
    await asyncio.sleep(0.5)
    receiver._running = False
    task.cancel()
    with __import__("contextlib").suppress(asyncio.CancelledError):
        await task

    assert receiver.speech_in_progress.called
    _, kwargs = receiver.speech_in_progress.call_args
    assert kwargs == {"min_dbfs": -35.0}

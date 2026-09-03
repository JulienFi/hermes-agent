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

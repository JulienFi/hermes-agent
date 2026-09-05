"""Discord voice: the Silero classifier behind the level gate, and short-clip padding.

The level gate (2026-09-05, morning) stopped noise from cutting playback in a
car, but a dBFS threshold is device-dependent. The classifier judges speech by
structure. Model-dependent tests skip when the ONNX file is absent.
"""

import os
from unittest.mock import MagicMock

import pytest

from tests.gateway.test_discord_voice_barge_in import (  # noqa: F401 — _patch_raw_config registers the autouse fixture here
    _make_adapter, _make_receiver, _patch_raw_config, _prepare_loop, _run_one_listen_pass, _tone, _square,
)

MODEL = os.path.expanduser("~/.hermes/models/silero_vad.onnx")
needs_model = pytest.mark.skipif(not os.path.isfile(MODEL), reason="silero_vad.onnx not installed")


class FakeVAD:
    def __init__(self, ratio):
        self.ratio = ratio
        self.calls = []

    def speech_ratio(self, pcm):
        self.calls.append(len(pcm))
        return self.ratio

    def max_window_ratio(self, pcm, window_seconds):
        self.calls.append(("max", len(pcm), window_seconds))
        return self.ratio


# ============================================================================
# The real model, when present
# ============================================================================

@needs_model
class TestSileroModel:
    def _vad(self):
        from plugins.platforms.discord.speech_vad import SileroVAD

        vad = SileroVAD.load(MODEL, 0.6)
        assert vad is not None
        return vad

    def test_silence_is_not_speech(self):
        vad = self._vad()
        assert vad.probabilities(bytes(192000)) and max(vad.probabilities(bytes(192000))) < 0.1
        assert vad.speech_ratio(bytes(192000)) == 0.0

    def test_a_loud_tone_is_not_speech_either(self):
        """The whole point: loud enough for the level gate, still not speech."""
        from plugins.platforms.discord.adapter import pcm_dbfs

        r = _make_receiver()
        _tone(r, seconds=1.0, amplitude=8000)
        pcm = bytes(r._buffers[7])
        assert pcm_dbfs(pcm) > -40.0
        assert self._vad().speech_ratio(pcm) < 0.5

    def test_chunk_count_follows_the_audio_length(self):
        vad = self._vad()
        one_second = bytes(192000)       # 16000 samples at model rate -> 31 full chunks
        assert len(vad.probabilities(one_second)) == 16000 // 512
        assert vad.probabilities(b"") == []
        assert vad.probabilities(bytes(100)) == []

    def test_max_window_ratio_on_silence_is_zero(self):
        assert self._vad().max_window_ratio(bytes(192000 * 3), 1.0) == 0.0

    def test_missing_model_returns_none(self, caplog):
        from plugins.platforms.discord.speech_vad import SileroVAD

        assert SileroVAD.load("/nonexistent/silero.onnx", 0.6) is None
        assert any("falls back to the level gate" in m for m in caplog.messages)


# ============================================================================
# Receiver: classifier behind the level gate
# ============================================================================

class TestReceiverWithClassifier:
    def test_loud_but_not_speech_is_ignored(self, caplog):
        import logging

        vad = FakeVAD(0.1)
        r = _make_receiver()
        r._vad, r._vad_min_ratio = vad, 0.6
        _tone(r, seconds=1.0, amplitude=3000)
        with caplog.at_level(logging.INFO, logger="plugins.platforms.discord.adapter"):
            assert r.speech_in_progress(min_dbfs=-40.0) is False
        assert vad.calls == [int(0.5 * 192000)]      # the onset window, not the whole buffer
        assert any("vad=0.10 -> ignored" in m for m in caplog.messages)

    def test_loud_and_speech_cuts(self):
        r = _make_receiver()
        r._vad, r._vad_min_ratio = FakeVAD(0.9), 0.6
        _tone(r, seconds=1.0, amplitude=3000)
        assert r.speech_in_progress(min_dbfs=-40.0) is True

    def test_quiet_audio_never_reaches_the_classifier(self):
        vad = FakeVAD(1.0)
        r = _make_receiver()
        r._vad, r._vad_min_ratio = vad, 0.6
        _tone(r, seconds=1.0, amplitude=100)
        assert r.speech_in_progress(min_dbfs=-40.0) is False
        assert vad.calls == []

    def test_without_a_level_gate_the_classifier_is_not_consulted(self):
        """min_dbfs=None is the documented packets-only mode."""
        vad = FakeVAD(0.0)
        r = _make_receiver()
        r._vad = vad
        _tone(r, seconds=1.0, amplitude=3000)
        assert r.speech_in_progress() is True
        assert vad.calls == []

    def test_a_failing_classifier_leaves_the_level_verdict(self):
        r = _make_receiver()
        r._vad = MagicMock()
        r._vad.speech_ratio.side_effect = RuntimeError("onnx boom")
        _tone(r, seconds=1.0, amplitude=3000)
        assert r.speech_in_progress(min_dbfs=-40.0) is True

    def test_constructor_wires_the_classifier(self):
        from plugins.platforms.discord.adapter import VoiceReceiver

        vad = FakeVAD(0.5)
        r = VoiceReceiver(MagicMock(), allowed_user_ids=set(), vad=vad, vad_min_ratio=0.7)
        assert r._vad is vad and r._vad_min_ratio == 0.7


# ============================================================================
# Adapter: config and the completed-utterance fallback
# ============================================================================

class TestClassifierConfig:
    def test_switch_off_returns_none_without_loading(self):
        adapter = _make_adapter({"voice_barge_in_vad": False})
        assert adapter._voice_barge_in_vad() is None
        assert not hasattr(adapter, "_voice_vad")

    def test_default_is_on(self):
        """Without the key the adapter tries to load — visible as the failed
        attempt when the model path points nowhere."""
        from tests.gateway.test_discord_voice_barge_in import _DISCORD_BLOCK

        adapter = _make_adapter({"voice_barge_in_vad_model": "/nonexistent/silero.onnx"})
        _DISCORD_BLOCK.pop("voice_barge_in_vad")
        assert adapter._voice_barge_in_vad() is None
        assert adapter._voice_vad_failed is True

    def test_missing_model_is_remembered_and_not_retried(self, caplog):
        adapter = _make_adapter({"voice_barge_in_vad": True, "voice_barge_in_vad_model": "/nonexistent/silero.onnx"})
        assert adapter._voice_barge_in_vad() is None
        assert adapter._voice_vad_failed is True
        n = len(caplog.messages)
        assert adapter._voice_barge_in_vad() is None
        assert len(caplog.messages) == n

    @needs_model
    def test_model_is_loaded_once_and_cached(self):
        adapter = _make_adapter({"voice_barge_in_vad": True, "voice_barge_in_vad_model": MODEL,
                                 "voice_barge_in_vad_threshold": 0.7})
        first = adapter._voice_barge_in_vad()
        assert first is not None and first.threshold == 0.7
        assert adapter._voice_barge_in_vad() is first

    def test_min_ratio_default_and_override(self):
        assert _make_adapter()._voice_barge_in_vad_min_ratio() == 0.6
        assert _make_adapter({"voice_barge_in_vad_min_ratio": 0.4})._voice_barge_in_vad_min_ratio() == 0.4

    @pytest.mark.parametrize("value", [0, 1.5, -1, "x", float("nan")])
    def test_min_ratio_nonsense_falls_back(self, value):
        assert _make_adapter({"voice_barge_in_vad_min_ratio": value})._voice_barge_in_vad_min_ratio() == 0.6


@pytest.mark.asyncio
async def test_completed_loud_noise_does_not_cut_when_classifier_says_no():
    adapter = _make_adapter({"voice_barge_in": True, "voice_barge_in_min_dbfs": -40, "voice_barge_in_vad": True})
    adapter._voice_vad = FakeVAD(0.1)
    vc = _prepare_loop(adapter)
    receiver = adapter._voice_receivers[42]
    receiver.check_silence.side_effect = [[(99, _square(seconds=1.2, amplitude=3000))]] + [[]] * 200

    await _run_one_listen_pass(adapter)

    vc.stop.assert_not_called()
    assert len(adapter._processed) == 1


@pytest.mark.asyncio
async def test_completed_speech_still_cuts_with_classifier():
    adapter = _make_adapter({"voice_barge_in": True, "voice_barge_in_min_dbfs": -40, "voice_barge_in_vad": True})
    adapter._voice_vad = FakeVAD(0.9)
    vc = _prepare_loop(adapter)
    receiver = adapter._voice_receivers[42]
    receiver.check_silence.side_effect = [[(99, _square(seconds=1.2, amplitude=3000))]] + [[]] * 200

    await _run_one_listen_pass(adapter)

    vc.stop.assert_called_once()
    assert ("max", len(_square(seconds=1.2, amplitude=3000)), adapter._voice_barge_in_min_speech()) in adapter._voice_vad.calls


# ============================================================================
# Short clips are padded before STT
# ============================================================================

class TestShortClipPadding:
    def test_short_clip_gets_silence_on_both_ends(self):
        from plugins.platforms.discord.adapter import DiscordAdapter

        clip = _square(seconds=1.0, amplitude=3000)
        padded = DiscordAdapter._pad_short_clip(clip)
        pad = int(0.5 * 192000) & ~3
        assert len(padded) == len(clip) + 2 * pad
        assert padded[:pad] == bytes(pad) and padded[-pad:] == bytes(pad)
        assert padded[pad:pad + len(clip)] == clip

    def test_long_clip_is_untouched(self):
        from plugins.platforms.discord.adapter import DiscordAdapter

        clip = _square(seconds=2.0, amplitude=3000)
        assert DiscordAdapter._pad_short_clip(clip) is clip

    @pytest.mark.asyncio
    async def test_process_voice_input_transcribes_the_padded_clip(self, monkeypatch):
        from plugins.platforms.discord import adapter as adapter_module
        import tools.transcription_tools as tt

        adapter = _make_adapter()
        adapter._voice_input_callback = None
        seen = {}
        monkeypatch.setattr(adapter_module.VoiceReceiver, "pcm_to_wav",
                            staticmethod(lambda pcm, path, **kw: seen.setdefault("len", len(pcm))))
        monkeypatch.setattr(tt, "transcribe_audio", lambda path: {"success": True, "transcript": "Hallo"})
        clip = _square(seconds=1.0, amplitude=3000)

        await adapter._process_voice_input(42, 99, clip)

        assert seen["len"] == len(clip) + 2 * (int(0.5 * 192000) & ~3)

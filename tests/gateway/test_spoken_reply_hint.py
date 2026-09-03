"""Voice turns get a shorter answer than text turns.

Measured on 2026-09-03: with ~40k tokens of session context, the SOUL.md
brevity rule (verified present in the assembled prompt, 947 chars) did not
survive — spoken replies ran 69 seconds. The fix is not "make the rule
arrive" but "state it on the turn where it applies", which is what the
ephemeral prompt layer is for
(``website/docs/developer-guide/prompt-assembly.md``: API-call-time layers
are deliberately kept out of the cached prefix, so text turns in the same
session still send a byte-identical cached system prompt).

These tests pin that seam: the flag on the turn, the append rule, the
config read, and the agreement between the documented defaults and the
values the Discord adapter actually falls back to.
"""

from types import SimpleNamespace

from gateway.turn_context import TurnContext


def _make_turn_runner(ctx, hint=""):
    """A TurnRunner over a stub GatewayRunner (same shape as test_turn_context)."""
    from gateway.run import TurnRunner

    class _StubGatewayRunner:
        def _adapter_for_source(self, source):
            return None

        def _spoken_reply_hint(self):
            return hint

    return TurnRunner(_StubGatewayRunner(), ctx)


class TestSpokenReplyFlag:
    def test_defaults_to_false(self):
        # A turn is text unless something says otherwise: no adapter, no
        # voice input and no auto-TTS may not add prompt text.
        assert TurnContext().spoken_reply is False

    def test_is_a_plain_read_only_field(self):
        ctx = TurnContext()
        ctx.spoken_reply = True
        assert ctx.spoken_reply is True


class TestAppendSpokenReplyHint:
    def test_text_turn_is_untouched(self):
        runner = _make_turn_runner(TurnContext(spoken_reply=False), hint="SPEAK SHORT")
        assert runner._append_spoken_reply_hint("base") == "base"

    def test_voice_turn_appends_after_the_channel_prompts(self):
        runner = _make_turn_runner(TurnContext(spoken_reply=True), hint="SPEAK SHORT")
        assert runner._append_spoken_reply_hint("base") == "base\n\nSPEAK SHORT"

    def test_empty_hint_is_the_off_switch(self):
        runner = _make_turn_runner(TurnContext(spoken_reply=True), hint="")
        assert runner._append_spoken_reply_hint("base") == "base"

    def test_no_leading_blank_lines_when_nothing_precedes_it(self):
        # Otherwise every voice turn would open its ephemeral block with two
        # stray newlines.
        runner = _make_turn_runner(TurnContext(spoken_reply=True), hint="SPEAK SHORT")
        assert runner._append_spoken_reply_hint("") == "SPEAK SHORT"
        assert runner._append_spoken_reply_hint(None) == "SPEAK SHORT"


class TestSpokenReplyHintConfig:
    """``GatewayRunner._spoken_reply_hint`` reads voice.spoken_reply_hint.

    The method touches no instance state, so it is exercised unbound over a
    placeholder self rather than by building a whole GatewayRunner.
    """

    def _call(self, monkeypatch, voice_block):
        from gateway.run import GatewayRunner

        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda *a, **k: {"voice": voice_block},
        )
        return GatewayRunner._spoken_reply_hint(SimpleNamespace())

    def test_reads_and_strips_the_configured_text(self, monkeypatch):
        got = self._call(monkeypatch, {"spoken_reply_hint": "  Keep it short.\n"})
        assert got == "Keep it short."

    def test_empty_string_disables(self, monkeypatch):
        assert self._call(monkeypatch, {"spoken_reply_hint": "   "}) == ""

    def test_missing_key_disables(self, monkeypatch):
        assert self._call(monkeypatch, {}) == ""

    def test_non_string_is_ignored_rather_than_stringified(self, monkeypatch):
        # `spoken_reply_hint: 12` in YAML must not put "12" in the prompt.
        assert self._call(monkeypatch, {"spoken_reply_hint": 12}) == ""

    def test_unreadable_config_disables(self, monkeypatch):
        from gateway.run import GatewayRunner

        def _boom(*a, **k):
            raise OSError("config.yaml is gone")

        monkeypatch.setattr("hermes_cli.config.load_config_readonly", _boom)
        assert GatewayRunner._spoken_reply_hint(SimpleNamespace()) == ""


class TestDocumentedDefaults:
    """DEFAULT_CONFIG must agree with the fallbacks the code actually uses.

    A default documented in config_defaults.py but contradicted in the
    adapter is worse than none: `hermes setup` writes the documented value
    into config.yaml and the behaviour changes without anyone editing it.
    """

    def test_spoken_reply_hint_ships_a_usable_default(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        hint = DEFAULT_CONFIG["voice"]["spoken_reply_hint"]
        assert isinstance(hint, str) and hint.strip()
        assert "SPOKEN ALOUD" in hint

    def test_discord_voice_knobs_are_documented(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        discord = DEFAULT_CONFIG["discord"]
        assert discord["voice_barge_in"] is False
        assert discord["voice_silence_threshold_seconds"] == 1.5
        assert discord["voice_barge_in_min_speech_seconds"] == 0.5

    def test_documented_values_match_the_adapter_fallbacks(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG
        from plugins.platforms.discord.adapter import VoiceReceiver

        discord = DEFAULT_CONFIG["discord"]
        assert discord["voice_silence_threshold_seconds"] == VoiceReceiver.SILENCE_THRESHOLD
        assert discord["voice_barge_in_min_speech_seconds"] == VoiceReceiver.MIN_SPEECH_DURATION

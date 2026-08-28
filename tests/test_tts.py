from __future__ import annotations

from pathlib import Path

import pytest

from assistant.tts import (
    DEFAULT_ENGLISH_VOICE_ID,
    DEFAULT_PORTUGUESE_VOICE_ID,
    TTSError,
    TTSManager,
    TTSResult,
    VoiceSpec,
    _culture_from_windows_value,
    _voice_from_registry_record,
    resolve_voice_command_target,
)


class FakeStore:
    def __init__(self) -> None:
        self.data = {}

    def load(self) -> dict:
        return dict(self.data)

    def update(self, **changes):
        self.data.update(changes)
        return dict(self.data)


class FakeProvider:
    name = "fake_tts"

    def __init__(self, voices: list[VoiceSpec] | None = None) -> None:
        self.voices = voices or []
        self.synthesized: list[tuple[str, VoiceSpec]] = []
        self.stopped = False

    def synthesize(self, text: str, voice: VoiceSpec) -> TTSResult:
        self.synthesized.append((text, voice))
        return TTSResult(
            audio_path=Path("fake.wav"),
            provider=self.name,
            voice_id=voice.voice_id,
            language=voice.language,
            synthesis_ms=12.0,
            text_chars=len(text),
        )

    def list_voices(self) -> list[VoiceSpec]:
        return list(self.voices)

    def is_available(self) -> bool:
        return True

    def warmup(self) -> None:
        return None

    def stop(self) -> None:
        self.stopped = True


def sapi_voice(name: str, locale: str, *, gender: str = "female") -> VoiceSpec:
    return VoiceSpec(
        voice_id=f"sapi:{name}",
        display_name=name,
        provider="fake_tts",
        language=locale.split("-")[0],
        locale=locale,
        reference=name,
        available=True,
        gender=gender,
    )


def test_voice_registry_exposes_british_and_ptpt_slots() -> None:
    provider = FakeProvider([sapi_voice("Microsoft Hazel", "en-GB"), sapi_voice("Microsoft Helia", "pt-PT")])
    manager = TTSManager(provider=provider, store=FakeStore())

    voices = {voice.voice_id: voice for voice in manager.list_voices()}

    assert voices[DEFAULT_ENGLISH_VOICE_ID].available is True
    assert voices[DEFAULT_ENGLISH_VOICE_ID].language == "en"
    assert voices[DEFAULT_PORTUGUESE_VOICE_ID].available is True
    assert voices[DEFAULT_PORTUGUESE_VOICE_ID].language == "pt"


def test_windows_lcid_locale_parsing_for_sapi_and_onecore_values() -> None:
    assert _culture_from_windows_value("0816") == "pt-PT"
    assert _culture_from_windows_value("409") == "en-US"
    assert _culture_from_windows_value("0809") == "en-GB"
    assert _culture_from_windows_value("pt_PT") == "pt-PT"


def test_registry_record_parses_helia_onecore_as_ptpt_but_not_synthesizable() -> None:
    voice = _voice_from_registry_record(
        {
            "Backend": "onecore",
            "RawId": "MSTTS_V110_ptPT_Helia",
            "Name": "Microsoft Helia",
            "Culture": "0816",
            "Gender": "Female",
            "Synthesizable": False,
            "RejectionReason": "onecore_not_supported_for_playback",
        }
    )

    assert voice.display_name == "Microsoft Helia"
    assert voice.locale == "pt-PT"
    assert voice.language == "pt"
    assert voice.synthesizable is False
    assert voice.rejection_reason == "onecore_not_supported_for_playback"


def test_helia_like_engb_voice_resolves_to_british_slot() -> None:
    provider = FakeProvider([sapi_voice("Microsoft Hazel", "en-GB")])
    manager = TTSManager(provider=provider, store=FakeStore())

    voices = {voice.voice_id: voice for voice in manager.list_voices()}

    assert voices[DEFAULT_ENGLISH_VOICE_ID].available is True
    assert voices[DEFAULT_ENGLISH_VOICE_ID].reference == "Microsoft Hazel"


def test_english_slot_can_use_explicit_language_fallback_when_no_british_voice_exists() -> None:
    provider = FakeProvider([sapi_voice("Microsoft Zira Desktop", "en-US")])
    manager = TTSManager(provider=provider, store=FakeStore())

    voices = {voice.voice_id: voice for voice in manager.list_voices()}

    assert voices[DEFAULT_ENGLISH_VOICE_ID].available is True
    assert voices[DEFAULT_ENGLISH_VOICE_ID].rejection_reason == "language_fallback"


def test_ptpt_response_never_falls_back_to_english_voice() -> None:
    provider = FakeProvider([sapi_voice("Microsoft Zira Desktop", "en-US")])
    manager = TTSManager(provider=provider, store=FakeStore())

    with pytest.raises(TTSError, match="voz portuguesa"):
        manager.speak_to_file("Olá.", response_language="pt-PT")

    assert provider.synthesized == []


def test_ptpt_response_never_falls_back_to_ptbr_voice() -> None:
    provider = FakeProvider([sapi_voice("Microsoft Maria Brazil", "pt-BR")])
    manager = TTSManager(provider=provider, store=FakeStore())

    with pytest.raises(TTSError, match="voz portuguesa"):
        manager.speak_to_file("Olá.", response_language="pt-PT")

    assert provider.synthesized == []


def test_ptpt_response_uses_only_playable_ptpt_voice() -> None:
    provider = FakeProvider([sapi_voice("Microsoft Helia", "pt-PT"), sapi_voice("Microsoft Zira Desktop", "en-US")])
    manager = TTSManager(provider=provider, store=FakeStore())

    result = manager.speak_to_file("Olá.", response_language="pt-PT")

    assert result.voice_id == DEFAULT_PORTUGUESE_VOICE_ID
    assert result.fallback_used is False
    assert provider.synthesized[0][1].reference == "Microsoft Helia"


def test_engb_response_can_fall_back_to_english_voice_with_explicit_telemetry() -> None:
    provider = FakeProvider([sapi_voice("Microsoft Zira Desktop", "en-US")])
    manager = TTSManager(provider=provider, store=FakeStore())

    result = manager.speak_to_file("Hello.", response_language="en-GB")

    assert result.voice_id == DEFAULT_ENGLISH_VOICE_ID
    assert result.fallback_used is True
    assert result.fallback_reason == "exact_locale_unavailable"
    assert provider.synthesized[0][1].language == "en"


def test_portuguese_slot_does_not_substitute_ptbr() -> None:
    provider = FakeProvider([sapi_voice("Microsoft Maria Brazil", "pt-BR")])
    manager = TTSManager(provider=provider, store=FakeStore())

    voices = {voice.voice_id: voice for voice in manager.list_voices()}

    assert voices[DEFAULT_PORTUGUESE_VOICE_ID].available is False
    assert voices[DEFAULT_PORTUGUESE_VOICE_ID].reference == ""


def test_onecore_ptpt_voice_does_not_make_tts_available_without_playback_support() -> None:
    onecore_helia = VoiceSpec(
        voice_id="onecore:MSTTS_V110_ptPT_Helia",
        display_name="Microsoft Helia",
        provider="fake_tts",
        language="pt",
        locale="pt-PT",
        reference="Microsoft Helia",
        available=True,
        synthesizable=False,
        rejection_reason="onecore_not_supported_for_playback",
    )
    manager = TTSManager(provider=FakeProvider([onecore_helia]), store=FakeStore())

    assert manager.is_available() is False
    diagnostics = manager.diagnostics()
    assert diagnostics["available"] is False


def test_active_voice_is_persisted_in_user_settings_store() -> None:
    store = FakeStore()
    manager = TTSManager(provider=FakeProvider([sapi_voice("Microsoft Helia", "pt-PT")]), store=store)

    manager.set_active_voice(DEFAULT_PORTUGUESE_VOICE_ID)

    assert store.data["active_voice_id"] == DEFAULT_PORTUGUESE_VOICE_ID
    assert manager.active_voice_id() == DEFAULT_PORTUGUESE_VOICE_ID


def test_voice_switch_commands_resolve_naturally() -> None:
    assert resolve_voice_command_target("muda para voz portuguesa") == DEFAULT_PORTUGUESE_VOICE_ID
    assert resolve_voice_command_target("usa a voz britânica") == DEFAULT_ENGLISH_VOICE_ID
    assert resolve_voice_command_target("troca de voz", DEFAULT_ENGLISH_VOICE_ID) == DEFAULT_PORTUGUESE_VOICE_ID
    assert resolve_voice_command_target("troca de voz", DEFAULT_PORTUGUESE_VOICE_ID) == DEFAULT_ENGLISH_VOICE_ID
    assert resolve_voice_command_target("olá, como estás?") is None


def test_handle_voice_command_returns_natural_confirmation() -> None:
    store = FakeStore()
    manager = TTSManager(provider=FakeProvider([sapi_voice("Microsoft Helia", "pt-PT")]), store=store)

    result = manager.handle_voice_command("muda para voz portuguesa")

    assert result.handled is True
    assert result.active_voice_id == DEFAULT_PORTUGUESE_VOICE_ID
    assert result.message == "Fica ativa a voz portuguesa."


def test_speak_to_file_uses_language_compatible_fallback_when_exact_voice_is_unavailable() -> None:
    provider = FakeProvider([sapi_voice("Microsoft Zira Desktop", "en-US")])
    store = FakeStore()
    store.update(active_voice_id=DEFAULT_PORTUGUESE_VOICE_ID)
    manager = TTSManager(provider=provider, store=store)

    result = manager.speak_to_file("Hello.", response_language="en-GB")

    assert result.fallback_used is True
    assert result.fallback_reason == "exact_locale_unavailable"
    assert result.voice_id == DEFAULT_ENGLISH_VOICE_ID
    assert provider.synthesized[0][0] == "Hello."


def test_missing_voice_reference_fails_cleanly() -> None:
    store = FakeStore()
    store.update(active_voice_id=DEFAULT_PORTUGUESE_VOICE_ID)
    manager = TTSManager(provider=FakeProvider([]), store=store)

    with pytest.raises(TTSError, match="nenhuma voz TTS local"):
        manager.speak_to_file("Olá.")


def test_stop_delegates_to_provider() -> None:
    provider = FakeProvider()
    manager = TTSManager(provider=provider, store=FakeStore())

    manager.stop()

    assert provider.stopped is True

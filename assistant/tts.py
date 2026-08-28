from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from assistant.response_language import ENGLISH_GB, PORTUGUESE_PT, normalize_response_language

try:
    import winsound
except ImportError:  # pragma: no cover - Windows-only provider guard.
    winsound = None

try:
    import winreg
except ImportError:  # pragma: no cover - Windows-only registry guard.
    winreg = None


DEFAULT_ENGLISH_VOICE_ID = "english_british_primary"
DEFAULT_PORTUGUESE_VOICE_ID = "portuguese_ptpt_primary"


class TTSError(RuntimeError):
    """Raised when text-to-speech cannot synthesize or play audio."""


@dataclass(frozen=True)
class VoiceSpec:
    voice_id: str
    display_name: str
    provider: str
    language: str
    locale: str
    reference: str = ""
    available: bool = False
    enabled: bool = True
    gender: str = ""
    style: str = ""
    description: str = ""
    synthesizable: bool = True
    rejection_reason: str = ""


@dataclass(frozen=True)
class TTSResult:
    audio_path: Path
    provider: str
    voice_id: str
    language: str
    synthesis_ms: float
    text_chars: int
    fallback_used: bool = False
    fallback_reason: str = ""


@dataclass(frozen=True)
class VoiceCommandResult:
    handled: bool
    message: str = ""
    active_voice_id: str = ""
    needs_clarification: bool = False


class TTSProvider(Protocol):
    name: str

    def synthesize(self, text: str, voice: VoiceSpec) -> TTSResult:
        ...

    def list_voices(self) -> list[VoiceSpec]:
        ...

    def is_available(self) -> bool:
        ...

    def warmup(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def diagnostics(self) -> dict[str, Any]:
        ...


class WindowsSapiProvider:
    """Local Windows SAPI TTS provider hidden behind the provider abstraction."""

    name = "windows_sapi"

    def __init__(self, powershell: str = "powershell") -> None:
        self.powershell = powershell

    def is_available(self) -> bool:
        return sys.platform.startswith("win")

    def warmup(self) -> None:
        if not self.is_available():
            raise TTSError("Windows SAPI não está disponível neste sistema.")

    def stop(self) -> None:
        if winsound is None:
            return
        winsound.PlaySound(None, winsound.SND_PURGE)

    def list_voices(self) -> list[VoiceSpec]:
        return [voice for voice in self.discover_voices() if voice.synthesizable]

    def discover_voices(self) -> list[VoiceSpec]:
        voices: list[VoiceSpec] = []
        voices.extend(self._discover_sapi_desktop_voices())
        voices.extend(self._discover_registry_voices("sapi_desktop", r"HKLM:\SOFTWARE\Microsoft\Speech\Voices\Tokens", True))
        voices.extend(self._discover_registry_voices("sapi_desktop", r"HKCU:\SOFTWARE\Microsoft\Speech\Voices\Tokens", True))
        voices.extend(self._discover_onecore_voices())
        return _dedupe_voices(voices)

    def diagnostics(self) -> dict[str, Any]:
        voices = self.discover_voices()
        for voice in voices:
            _log_voice_discovery(voice)
        return {
            "provider": self.name,
            "provider_available": self.is_available(),
            "discovered_voice_count": len(voices),
            "synthesizable_voice_count": len([voice for voice in voices if voice.synthesizable]),
            "voices": [voice_to_diagnostic(voice) for voice in voices],
        }

    def _discover_sapi_desktop_voices(self) -> list[VoiceSpec]:
        if not self.is_available():
            return []
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$s.GetInstalledVoices() | ForEach-Object { "
            "$i=$_.VoiceInfo; "
            "[PSCustomObject]@{Backend='sapi_desktop'; RawId=$i.Id; Name=$i.Name; Culture=$i.Culture.Name; "
            "Gender=$i.Gender.ToString(); Synthesizable=$true; Usable=$true; RejectionReason=''} "
            "} | ConvertTo-Json -Compress"
        )
        try:
            completed = subprocess.run(
                [self.powershell, "-NoProfile", "-Command", script],
                check=True,
                capture_output=True,
                text=True,
                timeout=8,
            )
        except Exception:
            return []
        return [_voice_from_sapi_record(record) for record in _json_records(completed.stdout)]

    def _discover_onecore_voices(self) -> list[VoiceSpec]:
        voices = []
        voices.extend(self._discover_registry_voices("onecore", r"HKLM:\SOFTWARE\Microsoft\Speech_OneCore\Voices\Tokens", False))
        voices.extend(self._discover_registry_voices("onecore", r"HKCU:\SOFTWARE\Microsoft\Speech_OneCore\Voices\Tokens", False))
        return voices

    def _discover_registry_voices(self, backend: str, root: str, synthesizable: bool) -> list[VoiceSpec]:
        if not self.is_available() or winreg is None:
            return []
        hive_name, subkey = _split_registry_root(root)
        if hive_name is None:
            return []
        hive = winreg.HKEY_LOCAL_MACHINE if hive_name == "HKLM" else winreg.HKEY_CURRENT_USER
        voices: list[VoiceSpec] = []
        try:
            with winreg.OpenKey(hive, subkey) as tokens:
                count = winreg.QueryInfoKey(tokens)[0]
                for index in range(count):
                    token_name = winreg.EnumKey(tokens, index)
                    record = _registry_voice_record(tokens, token_name, backend, synthesizable)
                    if record:
                        voices.append(_voice_from_registry_record(record))
        except OSError:
            return []
        return voices

    def synthesize(self, text: str, voice: VoiceSpec) -> TTSResult:
        clean_text = str(text or "").strip()
        if not clean_text:
            raise TTSError("Não há texto para falar.")
        if not self.is_available():
            raise TTSError("Windows SAPI não está disponível.")
        output = Path(tempfile.NamedTemporaryFile(prefix="echo_tts_", suffix=".wav", delete=False).name)
        output.unlink(missing_ok=True)
        started = time.perf_counter()
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$name = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($env:ECHO_TTS_VOICE_B64)); "
            "$text = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($env:ECHO_TTS_TEXT_B64)); "
            "$out = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($env:ECHO_TTS_OUT_B64)); "
            "if ($name) { $s.SelectVoice($name) }; "
            "$s.SetOutputToWaveFile($out); "
            "$s.Speak($text); "
            "$s.Dispose()"
        )
        env = {
            **os.environ,
            "ECHO_TTS_VOICE_B64": _b64(voice.reference or voice.display_name),
            "ECHO_TTS_TEXT_B64": _b64(clean_text),
            "ECHO_TTS_OUT_B64": _b64(str(output)),
        }
        try:
            subprocess.run(
                [self.powershell, "-NoProfile", "-Command", script],
                check=True,
                capture_output=True,
                text=True,
                timeout=max(8, min(60, len(clean_text) // 12 + 8)),
                env=env,
            )
        except Exception as exc:
            output.unlink(missing_ok=True)
            raise TTSError(f"Não consegui sintetizar áudio com Windows SAPI: {exc}") from exc
        return TTSResult(
            audio_path=output,
            provider=self.name,
            voice_id=voice.voice_id,
            language=voice.language,
            synthesis_ms=(time.perf_counter() - started) * 1000.0,
            text_chars=len(clean_text),
        )


class VoiceRegistry:
    def __init__(self, provider: TTSProvider) -> None:
        self.provider = provider

    def configured_voices(self) -> list[VoiceSpec]:
        installed = self.provider.list_voices()
        return [
            self._slot(
                DEFAULT_ENGLISH_VOICE_ID,
                "British English",
                "en",
                "en-GB",
                installed,
                ("en-gb", "united kingdom", "british", "hazel", "susan", "sophie", "serena"),
                gender="female",
                style="calm, elegant, warm",
                description="Voz inglesa feminina, calma e elegante.",
                allow_language_fallback=True,
            ),
            self._slot(
                DEFAULT_PORTUGUESE_VOICE_ID,
                "Português de Portugal",
                "pt",
                "pt-PT",
                installed,
                ("pt-pt", "portugal", "portuguese", "português", "helia", "maria"),
                gender="",
                style="natural pt-PT",
                description="Voz real de português europeu.",
                strict_locale=True,
            ),
        ]

    def get(self, voice_id: str) -> VoiceSpec | None:
        for voice in self.configured_voices():
            if voice.voice_id == voice_id:
                return voice
        return None

    def resolve(self, requested_voice_id: str, *, preferred_language: str = "") -> tuple[VoiceSpec | None, bool, str]:
        voices = [voice for voice in self.configured_voices() if voice.enabled]
        requested = next((voice for voice in voices if voice.voice_id == requested_voice_id), None)
        if requested and requested.available:
            return requested, False, ""
        language = (requested.language if requested else preferred_language).strip().lower()
        if language:
            fallback = next((voice for voice in voices if voice.available and voice.language == language), None)
            if fallback:
                return fallback, True, "selected_voice_unavailable"
        fallback = next((voice for voice in voices if voice.available), None)
        if fallback:
            return fallback, True, "selected_voice_unavailable"
        return requested, False, "no_available_voice"

    def _slot(
        self,
        voice_id: str,
        display_name: str,
        language: str,
        locale: str,
        installed: list[VoiceSpec],
        match_terms: tuple[str, ...],
        *,
        gender: str,
        style: str,
        description: str,
        strict_locale: bool = False,
        allow_language_fallback: bool = False,
    ) -> VoiceSpec:
        chosen = _best_voice_match(installed, locale, match_terms)
        fallback_reason = ""
        if strict_locale and (chosen is None or _normalize(chosen.locale) != _normalize(locale)):
            chosen = None
        if chosen is None and allow_language_fallback:
            chosen = next((voice for voice in installed if voice.language == language and voice.synthesizable), None)
            if chosen is not None:
                fallback_reason = "language_fallback"
        return VoiceSpec(
            voice_id=voice_id,
            display_name=display_name if chosen is None else f"{display_name} ({chosen.display_name})",
            provider=self.provider.name,
            language=language,
            locale=locale,
            reference=chosen.reference if chosen else "",
            available=chosen is not None,
            enabled=True,
            gender=gender,
            style=style,
            description=description,
            synthesizable=bool(chosen and chosen.synthesizable),
            rejection_reason=fallback_reason if chosen and chosen.synthesizable else "no_matching_synthesizable_voice",
        )


class TTSManager:
    def __init__(
        self,
        *,
        provider: TTSProvider | None = None,
        store: Any | None = None,
        default_voice_id: str = DEFAULT_ENGLISH_VOICE_ID,
    ) -> None:
        self.provider = provider or WindowsSapiProvider()
        self.registry = VoiceRegistry(self.provider)
        self.store = store
        self.default_voice_id = default_voice_id

    def active_voice_id(self) -> str:
        data = self.store.load() if self.store is not None else {}
        return str(data.get("active_voice_id") or self.default_voice_id)

    def set_active_voice(self, voice_id: str) -> VoiceSpec:
        voice = self.registry.get(voice_id)
        if voice is None:
            raise TTSError("Essa voz não está configurada.")
        if self.store is not None:
            self.store.update(active_voice_id=voice.voice_id)
        return voice

    def list_voices(self) -> list[VoiceSpec]:
        return self.registry.configured_voices()

    def is_available(self, response_language: str = "") -> bool:
        voice, _fallback_used, _fallback_reason = self._resolved_voice(response_language=response_language)
        return bool(voice and voice.available and voice.synthesizable)

    def diagnostics(self) -> dict[str, Any]:
        provider_diagnostics = (
            self.provider.diagnostics()
            if callable(getattr(self.provider, "diagnostics", None))
            else {"provider": getattr(self.provider, "name", ""), "voices": []}
        )
        slots = self.registry.configured_voices()
        active = self.active_voice_id()
        response_language = self._response_language()
        voice, fallback_used, fallback_reason = self._resolved_voice(response_language=response_language)
        return {
            "provider": provider_diagnostics.get("provider") or getattr(self.provider, "name", ""),
            "provider_available": bool(getattr(self.provider, "is_available", lambda: False)()),
            "active_voice_id": active,
            "response_language": response_language,
            "available": bool(voice and voice.available and voice.synthesizable),
            "resolved_voice_id": voice.voice_id if voice else "",
            "resolved_voice_name": voice.reference if voice else "",
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "logical_slots": [voice_to_diagnostic(slot) for slot in slots],
            "raw": provider_diagnostics,
        }

    def speak_to_file(self, text: str, *, response_language: str = "") -> TTSResult:
        language = normalize_response_language(response_language) or self._response_language()
        voice, fallback_used, fallback_reason = self._resolved_voice(response_language=language)
        if voice is None or not voice.available:
            if language == PORTUGUESE_PT:
                raise TTSError("A voz portuguesa ainda não está disponível.")
            raise TTSError("Não há nenhuma voz TTS local disponível.")
        if not voice.synthesizable:
            raise TTSError("A voz TTS selecionada não está disponível para reprodução local.")
        result = self.provider.synthesize(text, voice)
        return TTSResult(
            audio_path=result.audio_path,
            provider=result.provider,
            voice_id=result.voice_id,
            language=result.language,
            synthesis_ms=result.synthesis_ms,
            text_chars=result.text_chars,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )

    def stop(self) -> None:
        self.provider.stop()

    def _resolved_voice(self, *, response_language: str = "") -> tuple[VoiceSpec | None, bool, str]:
        language = normalize_response_language(response_language) or self._response_language()
        if language == PORTUGUESE_PT:
            voice = self.registry.get(DEFAULT_PORTUGUESE_VOICE_ID)
            if voice and voice.available and voice.synthesizable and _normalize(voice.locale) == _normalize(PORTUGUESE_PT):
                return voice, False, ""
            return voice, False, "no_compatible_ptpt_voice"
        if language == ENGLISH_GB:
            voice = self.registry.get(DEFAULT_ENGLISH_VOICE_ID)
            if voice and voice.available and voice.synthesizable and voice.language == "en":
                if voice.rejection_reason == "language_fallback":
                    return voice, True, "exact_locale_unavailable"
                return voice, False, ""
            return voice, False, "no_compatible_english_voice"
        active_id = self.active_voice_id()
        active_voice = self.registry.get(active_id)
        preferred_language = active_voice.language if active_voice else ""
        return self.registry.resolve(active_id, preferred_language=preferred_language)

    def _response_language(self) -> str:
        if self.store is not None:
            try:
                language = normalize_response_language(self.store.load().get("response_language"))
            except Exception:
                language = ""
            if language:
                return language
        return ENGLISH_GB

    def handle_voice_command(self, message: str) -> VoiceCommandResult:
        target = resolve_voice_command_target(message, self.active_voice_id())
        if target is None:
            return VoiceCommandResult(handled=False)
        if target == "clarify":
            return VoiceCommandResult(
                handled=True,
                needs_clarification=True,
                message="Queres usar a voz portuguesa ou voltar à voz inglesa?",
                active_voice_id=self.active_voice_id(),
            )
        try:
            voice = self.set_active_voice(target)
        except TTSError as exc:
            return VoiceCommandResult(handled=True, message=str(exc), active_voice_id=self.active_voice_id())
        if voice.voice_id == DEFAULT_PORTUGUESE_VOICE_ID:
            message_text = "Fica ativa a voz portuguesa."
        else:
            message_text = "Fica ativa a voz inglesa."
        if not voice.available:
            message_text += " Ainda não está disponível neste sistema, por isso vou usar fallback quando for preciso falar."
        return VoiceCommandResult(handled=True, message=message_text, active_voice_id=voice.voice_id)


def resolve_voice_command_target(message: str, active_voice_id: str = DEFAULT_ENGLISH_VOICE_ID) -> str | None:
    text = _normalize(message)
    if not _looks_like_voice_switch(text):
        return None
    if any(marker in text for marker in ("portugues", "portuguese", "pt-pt", "pt pt")):
        return DEFAULT_PORTUGUESE_VOICE_ID
    if any(marker in text for marker in ("ingles", "inglesa", "britanica", "british", "english")):
        return DEFAULT_ENGLISH_VOICE_ID
    if "muda de voz" in text or "troca de voz" in text:
        return DEFAULT_PORTUGUESE_VOICE_ID if active_voice_id == DEFAULT_ENGLISH_VOICE_ID else DEFAULT_ENGLISH_VOICE_ID
    return None


def play_audio_file(audio_path: Path, stop_checker: Any | None = None) -> float:
    started = time.perf_counter()
    if stop_checker is not None and callable(stop_checker) and stop_checker():
        return 0.0
    if winsound is None:
        raise TTSError("A reprodução local de áudio só está disponível no Windows nesta versão.")
    winsound.PlaySound(str(audio_path), winsound.SND_FILENAME)
    return (time.perf_counter() - started) * 1000.0


def _voice_from_sapi_record(record: dict[str, Any]) -> VoiceSpec:
    return _voice_from_registry_record({**record, "Backend": "sapi_desktop", "Synthesizable": True, "Usable": True})


def _voice_from_registry_record(record: dict[str, Any]) -> VoiceSpec:
    name = str(record.get("Name") or "").strip()
    culture = _culture_from_windows_value(record.get("Culture") or record.get("Locale") or record.get("Language"))
    gender = str(record.get("Gender") or "").strip().lower()
    language = _language_from_locale(culture)
    backend = str(record.get("Backend") or "windows").strip()
    raw_id = str(record.get("RawId") or record.get("Id") or name).strip()
    synthesizable = _truthy(record.get("Synthesizable", True))
    rejection = str(record.get("RejectionReason") or "").strip()
    return VoiceSpec(
        voice_id=f"{backend}:{raw_id or name}",
        display_name=name,
        provider=WindowsSapiProvider.name,
        language=language,
        locale=culture,
        reference=name,
        available=bool(name),
        gender=gender,
        synthesizable=synthesizable,
        rejection_reason=rejection,
    )


def _best_voice_match(installed: list[VoiceSpec], locale: str, match_terms: tuple[str, ...]) -> VoiceSpec | None:
    normalized_terms = tuple(_normalize(term) for term in match_terms)
    exact = [voice for voice in installed if _normalize(voice.locale) == _normalize(locale)]
    if exact:
        return exact[0]
    for voice in installed:
        haystack = _normalize(f"{voice.display_name} {voice.locale} {voice.gender}")
        if any(term and term in haystack for term in normalized_terms):
            return voice
    return None


def _json_records(text: str) -> list[dict[str, Any]]:
    clean = str(text or "").strip()
    if not clean:
        return []
    try:
        payload = json.loads(clean)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def voice_to_diagnostic(voice: VoiceSpec) -> dict[str, Any]:
    return {
        "raw_id": voice.voice_id,
        "name": voice.display_name,
        "locale": voice.locale,
        "language": voice.language,
        "gender": voice.gender,
        "provider": voice.provider,
        "usable": bool(voice.available and voice.synthesizable),
        "rejection_reason": "" if voice.available and voice.synthesizable else voice.rejection_reason or "unavailable",
    }


def _log_voice_discovery(voice: VoiceSpec) -> None:
    diagnostic = voice_to_diagnostic(voice)
    print(
        "[TTS_DISCOVERY]",
        f"raw_id={diagnostic['raw_id']}",
        f"name={diagnostic['name']}",
        f"locale={diagnostic['locale']}",
        f"language={diagnostic['language']}",
        f"gender={diagnostic['gender']}",
        f"provider={diagnostic['provider']}",
        f"usable={str(diagnostic['usable']).lower()}",
        f"rejection_reason={diagnostic['rejection_reason']}",
    )


def _dedupe_voices(voices: list[VoiceSpec]) -> list[VoiceSpec]:
    seen: set[tuple[str, str, str]] = set()
    result: list[VoiceSpec] = []
    for voice in voices:
        key = (_normalize(voice.display_name), _normalize(voice.locale), str(voice.synthesizable))
        if key in seen:
            continue
        seen.add(key)
        result.append(voice)
    return result


def _split_registry_root(root: str) -> tuple[str | None, str]:
    text = str(root or "").strip()
    if text.upper().startswith("HKLM:\\"):
        return "HKLM", text[6:]
    if text.upper().startswith("HKCU:\\"):
        return "HKCU", text[6:]
    return None, ""


def _registry_voice_record(tokens_key: Any, token_name: str, backend: str, synthesizable: bool) -> dict[str, Any] | None:
    if winreg is None:
        return None
    rejection = "" if synthesizable else "onecore_not_supported_for_playback"
    try:
        with winreg.OpenKey(tokens_key, token_name) as token:
            token_values = _registry_values(token)
            try:
                with winreg.OpenKey(token, "Attributes") as attrs:
                    attr_values = _registry_values(attrs)
            except OSError:
                attr_values = {}
    except OSError:
        return None
    name = attr_values.get("Name") or token_values.get("Name") or token_name
    culture = (
        attr_values.get("Culture")
        or token_values.get("Culture")
        or attr_values.get("Locale")
        or token_values.get("Locale")
        or attr_values.get("Language")
        or token_values.get("Language")
    )
    gender = attr_values.get("Gender") or token_values.get("Gender") or ""
    return {
        "Backend": backend,
        "RawId": token_name,
        "Name": name,
        "Culture": culture,
        "Gender": gender,
        "Synthesizable": synthesizable,
        "Usable": synthesizable,
        "RejectionReason": rejection,
    }


def _registry_values(key: Any) -> dict[str, Any]:
    if winreg is None:
        return {}
    values: dict[str, Any] = {}
    try:
        count = winreg.QueryInfoKey(key)[1]
    except OSError:
        return values
    for index in range(count):
        try:
            name, value, _kind = winreg.EnumValue(key, index)
        except OSError:
            continue
        values[str(name)] = value
    return values


def _culture_from_windows_value(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    normalized = raw.replace("_", "-")
    if "-" in normalized and any(ch.isalpha() for ch in normalized):
        parts = normalized.split("-")
        if len(parts) >= 2:
            return f"{parts[0].lower()}-{parts[1].upper()}"
        return normalized
    clean = normalized.lower().removeprefix("0x")
    try:
        number = int(clean, 16)
    except ValueError:
        return normalized
    lcid_map = {
        0x0409: "en-US",
        0x0809: "en-GB",
        0x0816: "pt-PT",
        0x0416: "pt-BR",
    }
    return lcid_map.get(number, normalized)


def _language_from_locale(locale: str) -> str:
    return str(locale or "").split("-")[0].lower()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _looks_like_voice_switch(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "muda para",
            "usa a voz",
            "volta a voz",
            "volta à voz",
            "muda de voz",
            "troca de voz",
            "change to",
            "use the",
            "voice",
        )
    ) and any(marker in text for marker in ("voz", "voice", "portugues", "portuguese", "ingles", "british", "english"))


def _normalize(text: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFKD", str(text or "").lower())
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _b64(text: str) -> str:
    import base64

    return base64.b64encode(str(text or "").encode("utf-8")).decode("ascii")

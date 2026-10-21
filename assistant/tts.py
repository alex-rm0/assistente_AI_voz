from __future__ import annotations

import json
import os
import queue
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass, replace
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
    reference_audio_path: str = ""
    synthesis_options: dict[str, Any] | None = None


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
    model: str = ""
    device: str = ""
    model_load_ms: float = 0.0
    worker_launch_ms: float = 0.0
    reference_prepare_ms: float = 0.0
    first_audio_ms: float = 0.0
    audio_duration_ms: float = 0.0
    reference_used: bool = False
    reference_cache_hit: bool = False
    # Phase 7.3.3: the specific predefined/catalog voice name bound to this
    # synthesis (e.g. "anna"), when the provider used one -- "" for a
    # reference-audio (cloning) synthesis or a provider with no such concept.
    provider_voice: str = ""


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
        self._voice_cache: list[VoiceSpec] | None = None
        self._last_logged_voice_signature: tuple | None = None

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
        if self._voice_cache is not None:
            return list(self._voice_cache)
        voices: list[VoiceSpec] = []
        voices.extend(self._discover_sapi_desktop_voices())
        voices.extend(self._discover_registry_voices("sapi_desktop", r"HKLM:\SOFTWARE\Microsoft\Speech\Voices\Tokens", True))
        voices.extend(self._discover_registry_voices("sapi_desktop", r"HKCU:\SOFTWARE\Microsoft\Speech\Voices\Tokens", True))
        voices.extend(self._discover_onecore_voices())
        self._voice_cache = _dedupe_voices(voices)
        return list(self._voice_cache)

    def diagnostics(self) -> dict[str, Any]:
        voices = self.discover_voices()
        # Phase 7.3.4 (section 7): discover_voices() is already cached
        # (self._voice_cache), but diagnostics() used to re-emit a
        # [TTS_DISCOVERY] line per voice on EVERY call regardless -- and
        # diagnostics() is called repeatedly per turn/SPEAK (once from the
        # UI's debounced availability poll, again from speakText()'s own
        # eligibility check, etc.), producing duplicate identical log bursts
        # for what is conceptually one user action. Only log again when the
        # discovered voice set actually changed.
        signature = tuple((voice.voice_id, voice.available, voice.synthesizable) for voice in voices)
        if signature != self._last_logged_voice_signature:
            for voice in voices:
                _log_voice_discovery(voice)
            self._last_logged_voice_signature = signature
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


class ChatterboxLocalProvider:
    """Local Chatterbox TTS provider with lazy model loading and reference voices."""

    name = "chatterbox"
    model_name = "ResembleAI/chatterbox"

    def __init__(
        self,
        *,
        env: dict[str, str] | None = None,
        output_dir: Path | None = None,
        runtime_python: Path | str | None = None,
        runner_path: Path | str | None = None,
        command_runner: Any | None = None,
        process_factory: Any | None = None,
        clock: Any = time.perf_counter,
        load_timeout_seconds: float | None = None,
        synthesis_timeout_seconds: float | None = None,
    ) -> None:
        self.env = env if env is not None else os.environ
        self.output_dir = output_dir or (Path("data") / "tts")
        self.runtime_python = Path(runtime_python) if runtime_python is not None else _default_chatterbox_runtime_python()
        self.runner_path = Path(runner_path) if runner_path is not None else Path(__file__).with_name("chatterbox_runner.py")
        self.command_runner = command_runner
        self.process_factory = process_factory or subprocess.Popen
        self._use_persistent_server = process_factory is not None or command_runner is None
        self.clock = clock
        self.load_timeout_seconds = (
            float(load_timeout_seconds)
            if load_timeout_seconds is not None
            else _float_from_env(self.env, "ECHO_TTS_LOAD_TIMEOUT_SECONDS", 180.0)
        )
        self.synthesis_timeout_seconds = (
            float(synthesis_timeout_seconds)
            if synthesis_timeout_seconds is not None
            else _float_from_env(self.env, "ECHO_TTS_SYNTHESIS_TIMEOUT_SECONDS", 180.0)
        )
        self._model_load_ms = 0.0
        self._last_error = ""
        self._device = self._select_device()
        self._probe_cache: bool | None = None
        self._process: _ChatterboxServerProcess | None = None
        self._server_failed = False
        self._request_counter = 0
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        return self._dependency_available()

    def warmup(self) -> None:
        self._server_process()

    def stop(self) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
        if process is not None:
            process.close()

    def list_voices(self) -> list[VoiceSpec]:
        return [self._profile(DEFAULT_ENGLISH_VOICE_ID, ENGLISH_GB), self._profile(DEFAULT_PORTUGUESE_VOICE_ID, PORTUGUESE_PT)]

    def diagnostics(self) -> dict[str, Any]:
        voices = self.list_voices()
        return {
            "provider": self.name,
            "provider_available": self.is_available(),
            "model": self.model_name,
            "model_loaded": False,
            "model_load_ms": int(self._model_load_ms),
            "runtime_python": str(self.runtime_python),
            "runner_path": str(self.runner_path),
            "device": self._device,
            "last_error": self._last_error,
            "discovered_voice_count": len(voices),
            "synthesizable_voice_count": len([voice for voice in voices if voice.available and voice.synthesizable]),
            "voices": [voice_to_diagnostic(voice) for voice in voices],
        }

    def synthesize(self, text: str, voice: VoiceSpec) -> TTSResult:
        clean_text = str(text or "").strip()
        if not clean_text:
            raise TTSError("Não há texto para falar.")
        reference_path = _resolve_reference_audio_path(voice.reference_audio_path)
        if reference_path is None or not reference_path.exists():
            raise TTSError("A referência de voz Chatterbox não está disponível.")
        if not self.is_available():
            raise TTSError("Chatterbox TTS não está instalado neste ambiente.")

        started = self.clock()
        output = Path(tempfile.NamedTemporaryFile(prefix="echo_chatterbox_", suffix=".wav", delete=False).name)
        output.unlink(missing_ok=True)
        options = dict(voice.synthesis_options or {})
        try:
            payload = self._run_synthesis(clean_text, reference_path, output, options)
        except Exception as exc:
            output.unlink(missing_ok=True)
            self._last_error = sanitize_tts_error(exc)
            self._server_failed = True
            raise TTSError(f"Não consegui sintetizar áudio com Chatterbox: {self._last_error}") from exc
        return TTSResult(
            audio_path=output,
            provider=self.name,
            voice_id=voice.voice_id,
            language=voice.language,
            synthesis_ms=float(payload.get("synthesis_ms") or ((self.clock() - started) * 1000.0)),
            text_chars=len(clean_text),
            model=str(payload.get("model") or self.model_name),
            device=str(payload.get("device") or self._device),
            model_load_ms=float(payload.get("model_load_ms") or 0.0),
            reference_used=True,
        )

    def _profile(self, voice_id: str, locale: str) -> VoiceSpec:
        language = _language_from_locale(locale)
        reference_path = self._reference_path_for(voice_id)
        resolved = _resolve_reference_audio_path(reference_path)
        dependency_available = self.is_available()
        reference_available = bool(resolved and resolved.exists())
        available = dependency_available and reference_available and not self._server_failed
        if not dependency_available:
            rejection = "chatterbox_dependency_unavailable"
        elif self._server_failed:
            rejection = self._last_error or "chatterbox_server_unavailable"
        elif not reference_available:
            rejection = "reference_audio_missing"
        else:
            rejection = ""
        display_name = "Chatterbox British Reference" if locale == ENGLISH_GB else "Chatterbox Português de Portugal"
        return VoiceSpec(
            voice_id=f"{self.name}:{voice_id}",
            display_name=display_name,
            provider=self.name,
            language=language,
            locale=locale,
            reference=str(resolved or reference_path or ""),
            available=available,
            enabled=True,
            gender="female" if locale == ENGLISH_GB else "",
            style="calm, elegant, warm, conversational" if locale == ENGLISH_GB else "natural pt-PT",
            description="Voz Chatterbox local com referência configurável.",
            synthesizable=available,
            rejection_reason=rejection,
            reference_audio_path=str(resolved or reference_path or ""),
            synthesis_options=self._options_for(voice_id),
        )

    def _reference_path_for(self, voice_id: str) -> str:
        if voice_id == DEFAULT_PORTUGUESE_VOICE_ID:
            return str(self.env.get("ECHO_TTS_PORTUGUESE_REFERENCE_AUDIO") or "")
        return str(self.env.get("ECHO_TTS_ENGLISH_REFERENCE_AUDIO") or "")

    def _options_for(self, voice_id: str) -> dict[str, Any]:
        prefix = "ECHO_TTS_PORTUGUESE" if voice_id == DEFAULT_PORTUGUESE_VOICE_ID else "ECHO_TTS_ENGLISH"
        defaults = {
            "exaggeration": 0.6 if voice_id == DEFAULT_ENGLISH_VOICE_ID else 0.5,
            "cfg_weight": 0.45,
            "temperature": 0.8,
            "pace": 1.0,
        }
        options: dict[str, Any] = {}
        for key, default in defaults.items():
            env_key = f"{prefix}_{key.upper()}"
            options[key] = _float_from_env(self.env, env_key, default)
        seed = self.env.get(f"{prefix}_SEED")
        if seed not in {None, ""}:
            try:
                options["seed"] = max(0, int(str(seed).strip()))
            except ValueError:
                pass
        return options

    def _dependency_available(self) -> bool:
        if self._probe_cache is not None:
            return self._probe_cache
        if not self.runtime_python.exists() or not self.runner_path.exists():
            self._probe_cache = False
            return False
        try:
            runner = self.command_runner or subprocess.run
            completed = runner(
                [str(self.runtime_python), str(self.runner_path), "--probe", "--device", self._device],
                check=False,
                capture_output=True,
                text=True,
                timeout=25,
                env=self._runner_env(),
            )
        except Exception as exc:
            self._last_error = sanitize_tts_error(exc)
            self._probe_cache = False
            return False
        payload = _json_object(completed.stdout)
        if completed.returncode == 0 and payload.get("ok") is True:
            self._probe_cache = True
            self._last_error = ""
            return True
        self._last_error = sanitize_tts_error(RuntimeError(str(payload.get("error") or completed.stderr or "probe_failed")))
        self._probe_cache = False
        return False

    def _run_synthesis(self, text: str, reference_path: Path, output: Path, options: dict[str, Any]) -> dict[str, Any]:
        if not self._use_persistent_server:
            return self._run_synthesis_once(text, reference_path, output, options)
        server = self._server_process()
        with self._lock:
            self._request_counter += 1
            request_id = f"tts-{self._request_counter}"
        return server.synthesize(
            request_id=request_id,
            text=text,
            reference_path=reference_path,
            output=output,
            options=options,
            timeout_seconds=self.synthesis_timeout_seconds,
        )

    def _run_synthesis_once(self, text: str, reference_path: Path, output: Path, options: dict[str, Any]) -> dict[str, Any]:
        command = [
            str(self.runtime_python),
            str(self.runner_path),
            "--synthesize",
            "--text",
            text,
            "--reference",
            str(reference_path),
            "--output",
            str(output),
            "--device",
            self._device,
            "--options-json",
            json.dumps(options, ensure_ascii=False),
        ]
        completed = self.command_runner(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(120, len(text) // 4 + 120),
            env=self._runner_env(),
        )
        payload = _json_object(completed.stdout)
        if completed.returncode != 0 or payload.get("ok") is not True:
            raise TTSError(str(payload.get("error") or completed.stderr or "Chatterbox synthesis failed"))
        if not output.exists():
            raise TTSError("O runner Chatterbox não devolveu áudio.")
        return payload

    def _server_process(self) -> "_ChatterboxServerProcess":
        with self._lock:
            if self._process is not None and self._process.is_alive():
                return self._process
            self._process = _ChatterboxServerProcess(
                runtime_python=self.runtime_python,
                runner_path=self.runner_path,
                device=self._device,
                env=self._runner_env(),
                process_factory=self.process_factory,
                load_timeout_seconds=self.load_timeout_seconds,
            )
            ready = self._process.start()
            self._model_load_ms = float(ready.get("model_load_ms") or 0.0)
            return self._process

    def _runner_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update({key: value for key, value in self.env.items() if isinstance(value, str)})
        hf_home = env.get("ECHO_TTS_HF_HOME") or env.get("HF_HOME") or str(Path.cwd() / ".hf_cache")
        env["HF_HOME"] = hf_home
        env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        return env

    def _select_device(self) -> str:
        requested = str(self.env.get("ECHO_TTS_CHATTERBOX_DEVICE") or "auto").strip().lower()
        if requested in {"cpu", "cuda", "mps"}:
            return requested
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


class PocketTTSLocalProvider:
    """CPU-oriented local Pocket TTS provider hidden behind an isolated runner."""

    name = "pocket_tts"
    model_name = "kyutai/pocket-tts"

    def __init__(
        self,
        *,
        env: dict[str, str] | None = None,
        output_dir: Path | None = None,
        runtime_python: Path | str | None = None,
        runner_path: Path | str | None = None,
        command_runner: Any | None = None,
        process_factory: Any | None = None,
        clock: Any = time.perf_counter,
        load_timeout_seconds: float | None = None,
        synthesis_timeout_seconds: float | None = None,
    ) -> None:
        self.env = env if env is not None else os.environ
        self.output_dir = output_dir or (Path("data") / "tts")
        self.runtime_python = Path(runtime_python) if runtime_python is not None else _default_pocket_runtime_python()
        self.runner_path = Path(runner_path) if runner_path is not None else Path(__file__).with_name("pocket_tts_runner.py")
        self.command_runner = command_runner
        self.process_factory = process_factory or subprocess.Popen
        self._use_persistent_server = process_factory is not None or command_runner is None
        self.clock = clock
        self.load_timeout_seconds = (
            float(load_timeout_seconds)
            if load_timeout_seconds is not None
            else _float_from_env(self.env, "ECHO_TTS_POCKET_LOAD_TIMEOUT_SECONDS", 60.0)
        )
        self.synthesis_timeout_seconds = (
            float(synthesis_timeout_seconds)
            if synthesis_timeout_seconds is not None
            else _float_from_env(self.env, "ECHO_TTS_POCKET_SYNTHESIS_TIMEOUT_SECONDS", 30.0)
        )
        self._model_load_ms = 0.0
        self._worker_launch_ms = 0.0
        self._last_error = ""
        self._probe_cache: bool | None = None
        self._process: _PocketTTSServerProcess | None = None
        self._server_failed = False
        self._has_voice_cloning = True
        self._request_counter = 0
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        return self._dependency_available()

    def warmup(self) -> None:
        self._server_process()

    def stop(self) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
        if process is not None:
            process.close()

    def list_voices(self) -> list[VoiceSpec]:
        return [self._profile(DEFAULT_ENGLISH_VOICE_ID, ENGLISH_GB), self._profile(DEFAULT_PORTUGUESE_VOICE_ID, PORTUGUESE_PT)]

    def diagnostics(self) -> dict[str, Any]:
        voices = self.list_voices()
        return {
            "provider": self.name,
            "provider_available": self.is_available(),
            "model": self.model_name,
            "model_loaded": bool(self._process and self._process.is_alive()),
            "model_load_ms": int(self._model_load_ms),
            "worker_launch_ms": int(self._worker_launch_ms),
            "runtime_python": str(self.runtime_python),
            "runner_path": str(self.runner_path),
            "device": "cpu",
            "last_error": self._last_error,
            "discovered_voice_count": len(voices),
            "synthesizable_voice_count": len([voice for voice in voices if voice.available and voice.synthesizable]),
            "voices": [voice_to_diagnostic(voice) for voice in voices],
        }

    def synthesize(self, text: str, voice: VoiceSpec) -> TTSResult:
        clean_text = str(text or "").strip()
        if not clean_text:
            raise TTSError("Não há texto para falar.")
        options = dict(voice.synthesis_options or {})
        predefined_voice = str(options.get("predefined_voice") or "").strip()
        reference_path: Path | None = None
        if not predefined_voice:
            reference_path = _resolve_reference_audio_path(voice.reference_audio_path)
            if reference_path is None or not reference_path.exists():
                raise TTSError("A referência de voz Pocket TTS não está disponível.")
        if not self.is_available():
            raise TTSError("Pocket TTS não está instalado neste ambiente.")
        if self._server_failed:
            raise TTSError(self._last_error or "pocket_tts_server_unavailable")

        started = self.clock()
        output = Path(tempfile.NamedTemporaryFile(prefix="echo_pocket_tts_", suffix=".wav", delete=False).name)
        output.unlink(missing_ok=True)
        try:
            payload = self._run_synthesis(clean_text, reference_path, output, options, predefined_voice=predefined_voice)
        except Exception as exc:
            output.unlink(missing_ok=True)
            self._last_error = sanitize_tts_error(exc)
            self._server_failed = True
            raise TTSError(f"Não consegui sintetizar áudio com Pocket TTS: {self._last_error}") from exc
        _debug_tts_trace(
            "pocket_tts_result",
            f"model_load_ms={int(float(payload.get('model_load_ms') or 0.0))}",
            f"reference_prepare_ms={int(float(payload.get('reference_prepare_ms') or 0.0))}",
            f"synthesis_ms={int(float(payload.get('synthesis_ms') or 0.0))}",
            f"audio_duration_ms={int(float(payload.get('audio_duration_ms') or 0.0))}",
        )
        return TTSResult(
            audio_path=output,
            provider=self.name,
            voice_id=voice.voice_id,
            language=voice.language,
            synthesis_ms=float(payload.get("synthesis_ms") or ((self.clock() - started) * 1000.0)),
            text_chars=len(clean_text),
            model=str(payload.get("model") or self.model_name),
            device=str(payload.get("device") or "cpu"),
            model_load_ms=float(payload.get("model_load_ms") or 0.0),
            worker_launch_ms=float(payload.get("worker_launch_ms") or self._worker_launch_ms),
            reference_prepare_ms=float(payload.get("reference_prepare_ms") or 0.0),
            first_audio_ms=float(payload.get("first_audio_ms") or payload.get("synthesis_ms") or 0.0),
            audio_duration_ms=float(payload.get("audio_duration_ms") or 0.0),
            reference_used=bool(reference_path),
            reference_cache_hit=bool(payload.get("reference_cache_hit")),
            provider_voice=str(payload.get("predefined_voice") or predefined_voice or ""),
        )

    def _profile(self, voice_id: str, locale: str) -> VoiceSpec:
        language = _language_from_locale(locale)
        if locale != ENGLISH_GB:
            return VoiceSpec(
                voice_id=f"{self.name}:{voice_id}",
                display_name="Pocket TTS Português de Portugal",
                provider=self.name,
                language=language,
                locale=locale,
                available=False,
                enabled=True,
                synthesizable=False,
                rejection_reason="ptpt_not_supported_in_this_phase",
            )
        dependency_available = self.is_available()
        predefined_voice = self._predefined_voice_for(voice_id)
        if predefined_voice:
            # Phase 7.3.3: the production binding for english_british_primary
            # -- a predefined/catalog Pocket voice (default "anna", section
            # 3) needs no reference audio file at all, and works even when
            # this environment's voice-cloning weights are unavailable
            # (section 3/7: "must work independently"). The Chloe reference
            # (ECHO_TTS_ENGLISH_REFERENCE_AUDIO) stays available for future
            # experimentation but is not required, and is not consulted here
            # while a predefined voice is configured.
            available = dependency_available and not self._server_failed
            if not dependency_available:
                rejection = "pocket_tts_dependency_unavailable"
            elif self._server_failed:
                rejection = self._last_error or "pocket_tts_server_unavailable"
            else:
                rejection = ""
            return VoiceSpec(
                voice_id=f"{self.name}:{voice_id}",
                display_name=f"Pocket TTS ({predefined_voice})",
                provider=self.name,
                language="en",
                locale=ENGLISH_GB,
                reference=predefined_voice,
                available=available,
                enabled=True,
                gender="female",
                # Deliberately not claiming a certified accent (e.g.
                # "British") here -- this voice was selected by listening
                # quality during the Phase 7.3 audition, not by certified
                # accent metadata (see Phase 7.3.3 context).
                style="calm, warm, conversational English (accent not certified)",
                description=f"Voz Pocket TTS predefinida do catálogo ('{predefined_voice}') -- selecionada por qualidade auditiva.",
                synthesizable=available,
                rejection_reason=rejection,
                reference_audio_path="",
                synthesis_options=self._options_for(voice_id, predefined_voice=predefined_voice),
            )

        reference_path = self._reference_path_for(voice_id)
        resolved = _resolve_reference_audio_path(reference_path)
        reference_available = bool(resolved and resolved.exists())
        available = dependency_available and reference_available and not self._server_failed
        if not dependency_available:
            rejection = "pocket_tts_dependency_unavailable"
        elif self._server_failed:
            rejection = self._last_error or "pocket_tts_server_unavailable"
        elif not reference_available:
            rejection = "reference_audio_missing"
        else:
            rejection = ""
        return VoiceSpec(
            voice_id=f"{self.name}:{voice_id}",
            display_name="Pocket TTS British Reference",
            provider=self.name,
            language="en",
            locale=ENGLISH_GB,
            reference=str(resolved or reference_path or ""),
            available=available,
            enabled=True,
            gender="female",
            style="calm, warm, conversational British English",
            description="Voz Pocket TTS local com referência configurável.",
            synthesizable=available,
            rejection_reason=rejection,
            reference_audio_path=str(resolved or reference_path or ""),
            synthesis_options=self._options_for(voice_id),
        )

    def _predefined_voice_for(self, voice_id: str) -> str:
        if voice_id != DEFAULT_ENGLISH_VOICE_ID:
            return ""
        # Defaults to the Phase 7.3 audition winner ("anna") -- set this env
        # var to "" explicitly to fall back to reference-audio cloning
        # instead (e.g. for future Chloe experimentation).
        return str(self.env.get("ECHO_TTS_ENGLISH_POCKET_VOICE", "anna")).strip()

    def _reference_path_for(self, voice_id: str) -> str:
        if voice_id != DEFAULT_ENGLISH_VOICE_ID:
            return ""
        return str(self.env.get("ECHO_TTS_ENGLISH_REFERENCE_AUDIO") or "")

    def _options_for(self, voice_id: str, *, predefined_voice: str = "") -> dict[str, Any]:
        if voice_id != DEFAULT_ENGLISH_VOICE_ID:
            return {}
        options: dict[str, Any] = {"max_tokens": int(_float_from_env(self.env, "ECHO_TTS_POCKET_MAX_TOKENS", 120.0))}
        if predefined_voice:
            options["predefined_voice"] = predefined_voice
        return options

    def _dependency_available(self) -> bool:
        if self._probe_cache is not None:
            return self._probe_cache
        if not self.runtime_python.exists() or not self.runner_path.exists():
            self._probe_cache = False
            return False
        try:
            runner = self.command_runner or subprocess.run
            completed = runner(
                [str(self.runtime_python), str(self.runner_path), "--probe"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
                env=self._runner_env(),
            )
        except Exception as exc:
            self._last_error = sanitize_tts_error(exc)
            self._probe_cache = False
            return False
        payload = _json_object(completed.stdout)
        if completed.returncode == 0 and payload.get("ok") is True:
            self._probe_cache = True
            self._last_error = ""
            return True
        self._last_error = sanitize_tts_error(RuntimeError(str(payload.get("error") or completed.stderr or "probe_failed")))
        self._probe_cache = False
        return False

    def _run_synthesis(
        self, text: str, reference_path: Path | None, output: Path, options: dict[str, Any], *, predefined_voice: str = ""
    ) -> dict[str, Any]:
        if not self._use_persistent_server:
            return self._run_synthesis_once(text, reference_path, output, options, predefined_voice=predefined_voice)
        server = self._server_process()
        with self._lock:
            self._request_counter += 1
            request_id = f"pocket-tts-{self._request_counter}"
        return server.synthesize(
            request_id=request_id,
            text=text,
            reference_path=reference_path,
            predefined_voice=predefined_voice,
            output=output,
            options=options,
            timeout_seconds=self.synthesis_timeout_seconds,
        )

    def _run_synthesis_once(
        self, text: str, reference_path: Path | None, output: Path, options: dict[str, Any], *, predefined_voice: str = ""
    ) -> dict[str, Any]:
        command = [
            str(self.runtime_python),
            str(self.runner_path),
            "--synthesize",
            "--text",
            text,
            "--output",
            str(output),
            "--language",
            "english",
            "--options-json",
            json.dumps(options, ensure_ascii=False),
        ]
        if predefined_voice:
            command += ["--predefined-voice", predefined_voice]
        else:
            command += ["--reference", str(reference_path)]
        completed = self.command_runner(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(30, int(self.synthesis_timeout_seconds)),
            env=self._runner_env(),
        )
        payload = _json_object(completed.stdout)
        if completed.returncode != 0 or payload.get("ok") is not True:
            raise TTSError(str(payload.get("error") or completed.stderr or "Pocket TTS synthesis failed"))
        if not output.exists():
            raise TTSError("O runner Pocket TTS não devolveu áudio.")
        return payload

    def _server_process(self) -> "_PocketTTSServerProcess":
        with self._lock:
            if self._process is not None and self._process.is_alive():
                return self._process
            self._process = _PocketTTSServerProcess(
                runtime_python=self.runtime_python,
                runner_path=self.runner_path,
                env=self._runner_env(),
                process_factory=self.process_factory,
                load_timeout_seconds=self.load_timeout_seconds,
            )
            ready = self._process.start()
            self._model_load_ms = float(ready.get("model_load_ms") or 0.0)
            self._worker_launch_ms = float(ready.get("worker_launch_ms") or 0.0)
            # Phase 7.3.3 (section 3/7): has_voice_cloning=False only means
            # RAW REFERENCE-AUDIO cloning is unavailable -- a predefined/
            # catalog voice (e.g. "anna") works fine regardless, so this is
            # no longer a fatal server-startup condition. Recorded for
            # diagnostics only; an actual reference-cloning request still
            # fails on its own (via the runner's per-request check) when
            # this is False.
            self._has_voice_cloning = ready.get("has_voice_cloning") is not False
            return self._process

    def _runner_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update({key: value for key, value in self.env.items() if isinstance(value, str)})
        hf_home = env.get("ECHO_TTS_HF_HOME") or env.get("HF_HOME") or str(Path.cwd() / ".hf_cache")
        env["HF_HOME"] = hf_home
        env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        thread_limit = env.get("ECHO_TTS_POCKET_THREADS") or env.get("ECHO_TTS_CPU_THREADS") or "2"
        env.setdefault("OMP_NUM_THREADS", thread_limit)
        env.setdefault("MKL_NUM_THREADS", thread_limit)
        env.setdefault("NUMEXPR_NUM_THREADS", thread_limit)
        return env


class LocalTTSProvider:
    """Provider chain: Chatterbox reference voices first, SAPI as temporary fallback."""

    name = "local_tts"

    def __init__(self, providers: list[TTSProvider] | None = None) -> None:
        self.providers = providers or [PocketTTSLocalProvider(), ChatterboxLocalProvider(), WindowsSapiProvider()]
        self._last_error = ""
        self._voice_cache: list[VoiceSpec] | None = None

    def is_available(self) -> bool:
        return any(bool(getattr(provider, "is_available", lambda: False)()) for provider in self.providers)

    def warmup(self) -> None:
        first = self.providers[0] if self.providers else None
        if first is None:
            raise TTSError("Não há provider TTS local configurado.")
        first.warmup()

    def stop(self) -> None:
        for provider in self.providers:
            try:
                provider.stop()
            except Exception:
                continue

    def list_voices(self) -> list[VoiceSpec]:
        if self._voice_cache is not None:
            return list(self._voice_cache)
        voices: list[VoiceSpec] = []
        for provider in self.providers:
            try:
                voices.extend(provider.list_voices())
            except Exception as exc:
                self._last_error = sanitize_tts_error(exc)
        self._voice_cache = voices
        return list(self._voice_cache)

    def diagnostics(self) -> dict[str, Any]:
        providers = []
        voices: list[VoiceSpec] = []
        for provider in self.providers:
            try:
                diagnostic = provider.diagnostics()
                providers.append(diagnostic)
                voices.extend(provider.list_voices())
            except Exception as exc:
                self._last_error = sanitize_tts_error(exc)
                providers.append({"provider": getattr(provider, "name", ""), "provider_available": False, "last_error": self._last_error})
        return {
            "provider": self.name,
            "provider_available": self.is_available(),
            "providers": providers,
            "last_error": self._last_error,
            "discovered_voice_count": len(voices),
            "synthesizable_voice_count": len([voice for voice in voices if voice.available and voice.synthesizable]),
            "voices": [voice_to_diagnostic(voice) for voice in voices],
        }

    def synthesize(self, text: str, voice: VoiceSpec) -> TTSResult:
        provider_name = str(voice.provider or "").strip()
        first_error = ""
        for provider in self.providers:
            if getattr(provider, "name", "") != provider_name:
                continue
            try:
                return provider.synthesize(text, voice)
            except Exception as exc:
                first_error = sanitize_tts_error(exc)
                self._last_error = first_error
                self._voice_cache = None
                break
        for provider in self.providers:
            if getattr(provider, "name", "") == provider_name:
                continue
            fallback = next(
                (
                    candidate
                    for candidate in provider.list_voices()
                    if candidate.available and candidate.synthesizable and candidate.language == voice.language
                ),
                None,
            )
            if fallback is None:
                continue
            _debug_tts_trace("tts_provider_fallback", f"from={provider_name}", f"to={getattr(provider, 'name', '')}")
            return replace(
                provider.synthesize(text, fallback),
                fallback_used=True,
                fallback_reason=f"provider_failed:{provider_name}",
            )
        if first_error:
            raise TTSError(first_error)
        raise TTSError(f"O provider TTS '{provider_name}' não está disponível.")


class _ChatterboxServerProcess:
    def __init__(
        self,
        *,
        runtime_python: Path,
        runner_path: Path,
        device: str,
        env: dict[str, str],
        process_factory: Any,
        load_timeout_seconds: float,
    ) -> None:
        self.runtime_python = runtime_python
        self.runner_path = runner_path
        self.device = device
        self.env = env
        self.process_factory = process_factory
        self.load_timeout_seconds = load_timeout_seconds
        self.process: Any | None = None
        self.responses: queue.Queue[dict[str, Any]] = queue.Queue()
        self.reader: threading.Thread | None = None
        self.stderr_reader: threading.Thread | None = None

    def start(self) -> dict[str, Any]:
        _debug_tts_trace("chatterbox_server_start")
        self.process = self.process_factory(
            [str(self.runtime_python), str(self.runner_path), "--server", "--device", self.device],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=self.env,
        )
        self.reader = threading.Thread(target=self._read_stdout, name="EchoChatterboxReader", daemon=True)
        self.reader.start()
        self.stderr_reader = threading.Thread(target=self._drain_stderr, name="EchoChatterboxStderr", daemon=True)
        self.stderr_reader.start()
        ready = self._next_payload(timeout_seconds=self.load_timeout_seconds)
        if ready.get("ok") is not True:
            self.close()
            raise TTSError(str(ready.get("error") or "chatterbox_server_start_failed"))
        _debug_tts_trace(
            "chatterbox_server_ready",
            f"model_load_ms={int(float(ready.get('model_load_ms') or 0.0))}",
            f"device={ready.get('device', '')}",
        )
        return ready

    def synthesize(
        self,
        *,
        request_id: str,
        text: str,
        reference_path: Path,
        output: Path,
        options: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        if not self.is_alive():
            raise TTSError("chatterbox_server_not_running")
        request = {
            "type": "synthesize",
            "id": request_id,
            "text": text,
            "reference": str(reference_path),
            "output": str(output),
            "options": options,
        }
        _debug_tts_trace("chatterbox_synthesis_request", f"request_id={request_id}", f"text_chars={len(text)}")
        stdin = getattr(self.process, "stdin", None)
        if stdin is None:
            raise TTSError("chatterbox_server_stdin_unavailable")
        stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        stdin.flush()
        while True:
            payload = self._next_payload(timeout_seconds=timeout_seconds)
            if str(payload.get("id") or "") and payload.get("id") != request_id:
                continue
            if payload.get("ok") is not True:
                raise TTSError(str(payload.get("error") or "chatterbox_synthesis_failed"))
            _debug_tts_trace(
                "chatterbox_synthesis_ready",
                f"request_id={request_id}",
                f"synthesis_ms={int(float(payload.get('synthesis_ms') or 0.0))}",
            )
            if not output.exists():
                raise TTSError("O runner Chatterbox não devolveu áudio.")
            return payload

    def is_alive(self) -> bool:
        process = self.process
        return bool(process is not None and getattr(process, "poll", lambda: None)() is None)

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        try:
            stdin = getattr(process, "stdin", None)
            if stdin is not None and getattr(process, "poll", lambda: None)() is None:
                stdin.write(json.dumps({"type": "shutdown", "id": "shutdown"}, ensure_ascii=False) + "\n")
                stdin.flush()
        except Exception:
            pass
        try:
            if getattr(process, "poll", lambda: None)() is None:
                process.terminate()
        except Exception:
            pass
        try:
            wait = getattr(process, "wait", None)
            if callable(wait):
                wait(timeout=2)
        except Exception:
            try:
                kill = getattr(process, "kill", None)
                if callable(kill):
                    kill()
            except Exception:
                pass

    def _read_stdout(self) -> None:
        stdout = getattr(self.process, "stdout", None)
        if stdout is None:
            self.responses.put({"ok": False, "error": "chatterbox_stdout_unavailable"})
            return
        for line in stdout:
            payload = _json_object(line)
            if payload:
                self.responses.put(payload)

    def _drain_stderr(self) -> None:
        stderr = getattr(self.process, "stderr", None)
        if stderr is None:
            return
        for line in stderr:
            text = str(line or "").strip()
            if text:
                _debug_tts_trace("chatterbox_stderr", text[:200])

    def _next_payload(self, *, timeout_seconds: float) -> dict[str, Any]:
        try:
            return self.responses.get(timeout=max(0.1, float(timeout_seconds)))
        except queue.Empty as exc:
            self.close()
            raise TTSError("chatterbox_timeout") from exc


class _PocketTTSServerProcess:
    def __init__(
        self,
        *,
        runtime_python: Path,
        runner_path: Path,
        env: dict[str, str],
        process_factory: Any,
        load_timeout_seconds: float,
    ) -> None:
        self.runtime_python = runtime_python
        self.runner_path = runner_path
        self.env = env
        self.process_factory = process_factory
        self.load_timeout_seconds = load_timeout_seconds
        self.process: Any | None = None
        self.responses: queue.Queue[dict[str, Any]] = queue.Queue()
        self.reader: threading.Thread | None = None
        self.stderr_reader: threading.Thread | None = None

    def start(self) -> dict[str, Any]:
        _debug_tts_trace("pocket_tts_server_start")
        launched = time.perf_counter()
        self.process = self.process_factory(
            [str(self.runtime_python), str(self.runner_path), "--server", "--language", "english"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=self.env,
        )
        worker_launch_ms = (time.perf_counter() - launched) * 1000.0
        self.reader = threading.Thread(target=self._read_stdout, name="EchoPocketTTSReader", daemon=True)
        self.reader.start()
        self.stderr_reader = threading.Thread(target=self._drain_stderr, name="EchoPocketTTSStderr", daemon=True)
        self.stderr_reader.start()
        ready = self._next_payload(timeout_seconds=self.load_timeout_seconds)
        ready["worker_launch_ms"] = worker_launch_ms
        if ready.get("ok") is not True:
            self.close()
            raise TTSError(str(ready.get("error") or "pocket_tts_server_start_failed"))
        _debug_tts_trace(
            "pocket_tts_server_ready",
            f"worker_launch_ms={int(worker_launch_ms)}",
            f"model_load_ms={int(float(ready.get('model_load_ms') or 0.0))}",
        )
        return ready

    def synthesize(
        self,
        *,
        request_id: str,
        text: str,
        reference_path: Path | None,
        output: Path,
        options: dict[str, Any],
        timeout_seconds: float,
        predefined_voice: str = "",
    ) -> dict[str, Any]:
        if not self.is_alive():
            raise TTSError("pocket_tts_server_not_running")
        request = {
            "type": "synthesize",
            "id": request_id,
            "text": text,
            "output": str(output),
            "language": "english",
            "options": options,
        }
        if predefined_voice:
            request["predefined_voice"] = predefined_voice
        else:
            request["reference"] = str(reference_path)
        _debug_tts_trace("pocket_tts_synthesis_request", f"request_id={request_id}", f"text_chars={len(text)}")
        stdin = getattr(self.process, "stdin", None)
        if stdin is None:
            raise TTSError("pocket_tts_server_stdin_unavailable")
        stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        stdin.flush()
        while True:
            payload = self._next_payload(timeout_seconds=timeout_seconds)
            if str(payload.get("id") or "") and payload.get("id") != request_id:
                continue
            if payload.get("ok") is not True:
                raise TTSError(str(payload.get("error") or "pocket_tts_synthesis_failed"))
            _debug_tts_trace(
                "pocket_tts_synthesis_ready",
                f"request_id={request_id}",
                f"reference_prepare_ms={int(float(payload.get('reference_prepare_ms') or 0.0))}",
                f"synthesis_ms={int(float(payload.get('synthesis_ms') or 0.0))}",
            )
            if not output.exists():
                raise TTSError("O runner Pocket TTS não devolveu áudio.")
            return payload

    def is_alive(self) -> bool:
        process = self.process
        return bool(process is not None and getattr(process, "poll", lambda: None)() is None)

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        try:
            stdin = getattr(process, "stdin", None)
            if stdin is not None and getattr(process, "poll", lambda: None)() is None:
                stdin.write(json.dumps({"type": "shutdown", "id": "shutdown"}, ensure_ascii=False) + "\n")
                stdin.flush()
        except Exception:
            pass
        try:
            if getattr(process, "poll", lambda: None)() is None:
                process.terminate()
        except Exception:
            pass
        try:
            wait = getattr(process, "wait", None)
            if callable(wait):
                wait(timeout=2)
        except Exception:
            try:
                kill = getattr(process, "kill", None)
                if callable(kill):
                    kill()
            except Exception:
                pass

    def _read_stdout(self) -> None:
        stdout = getattr(self.process, "stdout", None)
        if stdout is None:
            self.responses.put({"ok": False, "error": "pocket_tts_stdout_unavailable"})
            return
        for line in stdout:
            payload = _json_object(line)
            if payload:
                self.responses.put(payload)

    def _drain_stderr(self) -> None:
        stderr = getattr(self.process, "stderr", None)
        if stderr is None:
            return
        for line in stderr:
            text = str(line or "").strip()
            if text:
                _debug_tts_trace("pocket_tts_stderr", text[:200])

    def _next_payload(self, *, timeout_seconds: float) -> dict[str, Any]:
        try:
            return self.responses.get(timeout=max(0.1, float(timeout_seconds)))
        except queue.Empty as exc:
            self.close()
            raise TTSError("pocket_tts_timeout") from exc


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
            provider=chosen.provider if chosen else self.provider.name,
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
            reference_audio_path=chosen.reference_audio_path if chosen else "",
            synthesis_options=dict(chosen.synthesis_options or {}) if chosen else None,
        )


class TTSManager:
    def __init__(
        self,
        *,
        provider: TTSProvider | None = None,
        store: Any | None = None,
        default_voice_id: str = DEFAULT_ENGLISH_VOICE_ID,
    ) -> None:
        self.provider = provider or LocalTTSProvider()
        self.registry = VoiceRegistry(self.provider)
        self.store = store
        self.default_voice_id = default_voice_id
        self._last_logged_availability_signature: tuple | None = None

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
        diagnostic = self.availability_diagnostics(response_language=response_language)
        # Phase 7.3.4 (section 7): the frontend re-checks availability from
        # several places in quick succession for one conceptual action (a
        # debounced poll on every responseReady, plus speakText()'s own
        # eligibility gate) -- state is still recomputed fresh every time
        # (no staleness risk), only the duplicate identical [TTS_AVAILABILITY]
        # log line is suppressed.
        signature = (
            diagnostic.get("response_language"),
            diagnostic.get("active_voice_id"),
            diagnostic.get("resolved_voice_id"),
            diagnostic.get("speak_available"),
            diagnostic.get("reason"),
            diagnostic.get("fallback_used"),
        )
        if signature != self._last_logged_availability_signature:
            _log_tts_availability(diagnostic)
            self._last_logged_availability_signature = signature
        return bool(diagnostic["speak_available"])

    def availability_diagnostics(self, response_language: str = "") -> dict[str, Any]:
        language = normalize_response_language(response_language) or self._response_language()
        voice, fallback_used, fallback_reason = self._resolved_voice(response_language=language)
        provider_diagnostics = (
            self.provider.diagnostics()
            if callable(getattr(self.provider, "diagnostics", None))
            else {"provider": getattr(self.provider, "name", ""), "voices": []}
        )
        providers = provider_diagnostics.get("providers")
        provider_items = providers if isinstance(providers, list) else [provider_diagnostics]
        pocket = _provider_diagnostic(provider_items, "pocket_tts")
        chatterbox = _provider_diagnostic(provider_items, "chatterbox")
        sapi = _provider_diagnostic(provider_items, "windows_sapi")
        pocket_reference_available = any(
            bool(item.get("reference_available"))
            for item in pocket.get("voices", [])
            if isinstance(item, dict)
        )
        pocket_available = bool(pocket.get("provider_available")) and any(
            _diagnostic_voice_usable(item)
            for item in pocket.get("voices", [])
            if isinstance(item, dict)
        )
        chatterbox_reference_available = any(
            bool(item.get("reference_available"))
            for item in chatterbox.get("voices", [])
            if isinstance(item, dict)
        )
        chatterbox_available = bool(chatterbox.get("provider_available")) and any(
            _diagnostic_voice_usable(item)
            for item in chatterbox.get("voices", [])
            if isinstance(item, dict)
        )
        sapi_available = bool(sapi.get("provider_available")) and any(
            _diagnostic_voice_usable(item)
            for item in sapi.get("voices", [])
            if isinstance(item, dict)
        )
        speak_available = bool(voice and voice.available and voice.synthesizable)
        compatible_fallback_available = (
            language == ENGLISH_GB
            and speak_available
            and bool(voice)
            and voice.language == "en"
            and bool(fallback_used)
        )
        if speak_available and compatible_fallback_available:
            reason = "compatible_english_fallback"
        elif speak_available:
            reason = "available"
        elif language == PORTUGUESE_PT:
            reason = "no_playable_ptpt_voice"
        elif chatterbox and not chatterbox_reference_available:
            reason = "chatterbox_reference_missing"
        else:
            reason = fallback_reason or "no_playable_voice"
        return {
            "response_language": language,
            "active_voice_id": self.active_voice_id(),
            "pocket_tts_available": pocket_available,
            "pocket_tts_reference_available": pocket_reference_available,
            "chatterbox_available": chatterbox_available,
            "chatterbox_reference_available": chatterbox_reference_available,
            "sapi_available": sapi_available,
            "compatible_fallback_available": compatible_fallback_available,
            "speak_available": speak_available,
            "reason": reason,
            "resolved_voice_id": voice.voice_id if voice else "",
            "resolved_voice_language": voice.language if voice else "",
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
        }

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
        combined_fallback_used = bool(fallback_used or result.fallback_used)
        combined_fallback_reason = fallback_reason or result.fallback_reason
        return TTSResult(
            audio_path=result.audio_path,
            provider=result.provider,
            voice_id=result.voice_id,
            language=result.language,
            synthesis_ms=result.synthesis_ms,
            text_chars=result.text_chars,
            fallback_used=combined_fallback_used,
            fallback_reason=combined_fallback_reason,
            model=result.model,
            device=result.device,
            model_load_ms=result.model_load_ms,
            worker_launch_ms=result.worker_launch_ms,
            reference_prepare_ms=result.reference_prepare_ms,
            first_audio_ms=result.first_audio_ms,
            audio_duration_ms=result.audio_duration_ms,
            reference_used=result.reference_used,
            reference_cache_hit=result.reference_cache_hit,
            provider_voice=result.provider_voice,
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


def play_audio_file(
    audio_path: Path,
    stop_checker: Any | None = None,
    *,
    on_playback_event: Any | None = None,
    allow_winsound_fallback: bool = True,
) -> float:
    """Phase 7.6.2's fix (winsound SND_ASYNC + polled SND_PURGE) turned out
    NOT to be reliable in real use -- real logs showed playback still
    running to its full natural duration (e.g. playback_ms=38751) no
    matter when STOP was pressed. winsound's async playback + purge is a
    thin wrapper over the legacy Windows mmsystem sound API, whose
    interruption behavior is documented as unreliable for anything beyond
    trivial system sounds; it is not a dependable backend for stoppable
    speech playback (section 3).

    Phase 7.6.3 replaced it with sounddevice (already a project dependency
    for STT capture) via a DEDICATED sd.OutputStream this function owns
    end-to-end. Phase 7.6.4 found that this backend was silently falling
    back to winsound on every real Pocket TTS utterance: Pocket writes its
    WAV via `scipy.io.wavfile.write` on a raw float32 numpy array, which
    encodes as WAV format code 3 (IEEE float) -- a format the stdlib
    `wave` module cannot parse at all (`wave.Error: unknown format: 3`),
    silently swallowed by the old decoder's bare except. `_decode_wav_for_playback`
    below replaces `wave` with a small dependency-free RIFF/WAVE chunk
    parser that handles both PCM (format 1) and IEEE float (format 3, the
    real format Pocket produces), and now RAISES descriptive exceptions
    instead of swallowing them, so every fallback is diagnosed via a
    `backend_failed` event rather than happening silently (section 3).

    Voice Conversation Mode must never silently degrade to the
    non-interruptible winsound fallback and pretend STOP/barge-in still
    work (section 7): when `allow_winsound_fallback` is False, a failure
    to open the interruptible sounddevice path raises TTSError instead of
    falling back -- callers already treat a raised TTSError from playback
    as a bounded TTS failure (text stays visible, voice mode recovers).
    `allow_winsound_fallback=True` (the default) preserves the old
    degrade-gracefully behavior for manual/non-interactive TTS."""
    started = time.perf_counter()
    if stop_checker is not None and callable(stop_checker) and stop_checker():
        return 0.0

    def _fallback_or_raise(exc: Exception, extra_fields: dict[str, Any]) -> float:
        _emit_playback_event(
            on_playback_event,
            "backend_failed",
            {
                "backend": "sounddevice",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc)[:200],
                "fallback": "winsound" if allow_winsound_fallback else "none",
                **extra_fields,
            },
        )
        if not allow_winsound_fallback:
            raise TTSError(
                f"Reprodução interrompível indisponível ({type(exc).__name__}: {exc})"
            ) from exc
        return _play_audio_file_winsound_fallback(audio_path, started, on_playback_event)

    try:
        audio, sample_rate, channels, dtype_name, bits_per_sample, frame_count = _decode_wav_for_playback(
            audio_path
        )
    except Exception as exc:
        return _fallback_or_raise(exc, {})

    try:
        import sounddevice as sd
    except Exception as exc:
        return _fallback_or_raise(
            exc,
            {
                "samplerate": sample_rate,
                "channels": channels,
                "dtype": dtype_name,
            },
        )

    device_index, device_name = _select_output_device(sd)
    _emit_playback_event(
        on_playback_event,
        "started",
        {
            "backend": "sounddevice",
            "sample_rate": sample_rate,
            "channels": channels,
            "sample_width": bits_per_sample // 8,
            "numpy_dtype": dtype_name,
            "frame_count": frame_count,
            "device": device_name,
        },
    )
    total_frames = audio.shape[0]
    position = 0
    finished = threading.Event()
    callback_exception: list[BaseException] = []

    def _callback(outdata, frame_count, time_info, status) -> None:
        nonlocal position
        try:
            remaining = total_frames - position
            if remaining <= 0:
                outdata[:] = 0
                raise sd.CallbackStop()
            chunk = min(frame_count, remaining)
            outdata[:chunk] = audio[position : position + chunk]
            if chunk < frame_count:
                outdata[chunk:] = 0
            position += chunk
            if position >= total_frames:
                raise sd.CallbackStop()
        except sd.CallbackStop:
            raise
        except Exception as exc:  # PortAudio otherwise hides callback errors silently
            callback_exception.append(exc)
            raise sd.CallbackStop()

    try:
        stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype=dtype_name,
            device=device_index,
            callback=_callback,
            finished_callback=finished.set,
        )
    except Exception as exc:
        # A real device/driver failure opening a dedicated output stream.
        return _fallback_or_raise(
            exc,
            {
                "samplerate": sample_rate,
                "channels": channels,
                "dtype": dtype_name,
                "device": device_name,
            },
        )

    stopped_early = False
    poll_interval_seconds = 0.02
    with stream:
        while not finished.is_set():
            if stop_checker is not None and callable(stop_checker) and stop_checker():
                stopped_early = True
                stream.abort()
                break
            time.sleep(poll_interval_seconds)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if callback_exception:
        _emit_playback_event(
            on_playback_event,
            "callback_error",
            {
                "backend": "sounddevice",
                "exception_type": type(callback_exception[0]).__name__,
                "exception_message": str(callback_exception[0])[:200],
            },
        )
    _emit_playback_event(
        on_playback_event,
        "stopped" if stopped_early else "completed",
        {"backend": "sounddevice", "elapsed_ms": int(elapsed_ms)},
    )
    return elapsed_ms


def _play_audio_file_winsound_fallback(audio_path: Path, started: float, on_playback_event: Any | None) -> float:
    if winsound is None:
        raise TTSError("A reprodução local de áudio só está disponível no Windows nesta versão.")
    _emit_playback_event(on_playback_event, "started", {"backend": "winsound_fallback"})
    winsound.PlaySound(str(audio_path), winsound.SND_FILENAME)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    _emit_playback_event(on_playback_event, "completed", {"backend": "winsound_fallback", "elapsed_ms": int(elapsed_ms)})
    return elapsed_ms


def _emit_playback_event(on_playback_event: Any | None, event: str, fields: dict[str, Any]) -> None:
    if on_playback_event is None or not callable(on_playback_event):
        return
    try:
        on_playback_event(event, fields)
    except Exception:
        pass  # diagnostics must never break playback


def _select_output_device(sd: Any) -> tuple[int | None, str]:
    """Only overrides sounddevice's own default output device when that
    default is NOT actually output-capable -- picks the first
    output-capable device deterministically (lowest index) in that case.
    Never hardcodes a machine-specific index and never touches the
    process-wide sd.default.device unless asked to via the return value
    consumed by a single OutputStream call (section 5)."""
    try:
        devices = sd.query_devices()
        default_output = sd.default.device
        # sd.default.device is normally a 2-item [input, output] sequence,
        # but sounddevice returns its own _InputOutputPair type (not a
        # list/tuple instance) -- duck-type via subscripting instead of
        # isinstance, since a plain int (single-device override) doesn't
        # support [1] and should be used as-is.
        try:
            default_output = default_output[1]
        except (TypeError, IndexError):
            pass
        if default_output is not None and 0 <= default_output < len(devices):
            info = devices[default_output]
            if info.get("max_output_channels", 0) > 0:
                return None, f"default:{info.get('name', default_output)}"
        for index, info in enumerate(devices):
            if info.get("max_output_channels", 0) > 0:
                return index, f"{index}:{info.get('name', index)}"
    except Exception:
        pass
    return None, "default"


def _decode_wav_for_playback(path: Path) -> tuple[Any, int, int, str, int, int]:
    """Decode a WAV file into a (frames, channels)-shaped numpy array ready
    for sounddevice, using only the stdlib (`struct`) + numpy -- NOT the
    stdlib `wave` module, which cannot parse IEEE-float WAV (format code 3)
    at all and raises `wave.Error: unknown format: 3`. This matters
    because Pocket TTS's own output IS exactly that format:
    assistant/pocket_tts_runner.py writes the raw float32 numpy array
    straight from the model via `scipy.io.wavfile.write`, which encodes it
    as IEEE-float, not PCM -- confirmed by reproducing that exact write
    call in Pocket's own dedicated venv and observing the stdlib `wave`
    module fail on the result.

    Walks RIFF chunks directly to support standard PCM (format 1) and IEEE
    float (format 3), plus WAVE_FORMAT_EXTENSIBLE (0xFFFE) by reading the
    real sub-format code from the extended fmt chunk. Raises a descriptive
    exception (ValueError, OSError) for anything else, rather than
    swallowing it -- callers are responsible for logging and deciding
    whether to fall back (section 3)."""
    import numpy as np

    raw = path.read_bytes()
    if raw[0:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"not a RIFF/WAVE file (header={raw[0:12]!r})")

    pos = 12
    fmt_code: int | None = None
    channels: int | None = None
    sample_rate: int | None = None
    bits_per_sample: int | None = None
    data_bytes: bytes | None = None
    while pos + 8 <= len(raw):
        chunk_id = raw[pos : pos + 4]
        chunk_size = struct.unpack_from("<I", raw, pos + 4)[0]
        chunk_start = pos + 8
        chunk_end = chunk_start + chunk_size
        if chunk_end > len(raw):
            break
        if chunk_id == b"fmt ":
            fmt_code, channels, sample_rate, _, _, bits_per_sample = struct.unpack_from(
                "<HHIIHH", raw, chunk_start
            )
            if fmt_code == 0xFFFE and chunk_size >= 40:
                fmt_code = struct.unpack_from("<H", raw, chunk_start + 24)[0]
        elif chunk_id == b"data":
            data_bytes = raw[chunk_start:chunk_end]
        pos = chunk_end + (chunk_end % 2)  # chunks are word-aligned

    if fmt_code is None:
        raise ValueError("no fmt chunk found in WAV file")
    if data_bytes is None:
        raise ValueError("no data chunk found in WAV file")
    if not channels:
        raise ValueError(f"invalid channel count in fmt chunk: {channels}")

    if fmt_code == 1:  # PCM
        if bits_per_sample == 16:
            dtype_name = "int16"
        elif bits_per_sample == 8:
            dtype_name = "uint8"
        elif bits_per_sample == 32:
            dtype_name = "int32"
        else:
            raise ValueError(f"unsupported PCM bits_per_sample={bits_per_sample}")
    elif fmt_code == 3:  # IEEE float -- the format Pocket TTS actually writes
        if bits_per_sample == 32:
            dtype_name = "float32"
        elif bits_per_sample == 64:
            dtype_name = "float64"
        else:
            raise ValueError(f"unsupported IEEE-float bits_per_sample={bits_per_sample}")
    else:
        raise ValueError(f"unsupported WAV format code={fmt_code}")

    audio = np.frombuffer(data_bytes, dtype=dtype_name)
    usable_frames = audio.shape[0] // channels
    audio = audio[: usable_frames * channels].reshape(usable_frames, channels)
    return audio, sample_rate, channels, dtype_name, bits_per_sample, usable_frames


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
    usable = [voice for voice in installed if voice.available and voice.synthesizable]
    exact = [voice for voice in usable if _normalize(voice.locale) == _normalize(locale)]
    if exact:
        return exact[0]
    for voice in usable:
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
        "reference_available": bool(voice.reference_audio_path and Path(voice.reference_audio_path).exists()),
        "reference_used": bool(voice.reference_audio_path),
        "synthesis_options": dict(voice.synthesis_options or {}),
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


def _provider_diagnostic(provider_items: list[Any], provider_name: str) -> dict[str, Any]:
    for item in provider_items:
        if isinstance(item, dict) and item.get("provider") == provider_name:
            return item
    return {}


def _diagnostic_voice_usable(item: dict[str, Any]) -> bool:
    if "usable" in item:
        return bool(item.get("usable"))
    return bool(item.get("available")) and bool(item.get("synthesizable", True))


def _log_tts_availability(diagnostic: dict[str, Any]) -> None:
    print(
        "[TTS_AVAILABILITY] "
        f"response_language={diagnostic.get('response_language', '')} "
        f"active_voice_id={diagnostic.get('active_voice_id', '')} "
        f"pocket_tts_available={diagnostic.get('pocket_tts_available', False)} "
        f"pocket_tts_reference_available={diagnostic.get('pocket_tts_reference_available', False)} "
        f"chatterbox_available={diagnostic.get('chatterbox_available', False)} "
        f"chatterbox_reference_available={diagnostic.get('chatterbox_reference_available', False)} "
        f"sapi_available={diagnostic.get('sapi_available', False)} "
        f"compatible_fallback_available={diagnostic.get('compatible_fallback_available', False)} "
        f"speak_available={diagnostic.get('speak_available', False)} "
        f"reason={diagnostic.get('reason', '')}"
    )


def _debug_tts_trace(stage: str, *parts: str) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    suffix = " ".join(str(part) for part in parts if str(part))
    print(f"[TTS_TRACE] ts={timestamp} stage={stage}" + (f" {suffix}" if suffix else ""))


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


def _default_chatterbox_runtime_python() -> Path:
    if sys.platform.startswith("win"):
        return Path.cwd() / ".venv_tts" / "Scripts" / "python.exe"
    return Path.cwd() / ".venv_tts" / "bin" / "python"


def _default_pocket_runtime_python() -> Path:
    if sys.platform.startswith("win"):
        return Path.cwd() / ".venv_pocket_tts" / "Scripts" / "python.exe"
    return Path.cwd() / ".venv_pocket_tts" / "bin" / "python"


def _resolve_reference_audio_path(value: str) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(raw)))
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _float_from_env(env: dict[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw in {None, ""}:
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        return default


def sanitize_tts_error(exc: Exception) -> str:
    text = str(exc or "").strip() or exc.__class__.__name__
    for key in ("ECHO_TTS_ENGLISH_REFERENCE_AUDIO", "ECHO_TTS_PORTUGUESE_REFERENCE_AUDIO"):
        value = os.environ.get(key)
        if value:
            text = text.replace(value, "<reference_audio>")
    return text[:240]


def _json_object(text: str) -> dict[str, Any]:
    clean = str(text or "").strip()
    if not clean:
        return {}
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}

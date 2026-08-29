from __future__ import annotations

import json
import queue
from pathlib import Path

import pytest

from assistant.tts import (
    ChatterboxLocalProvider,
    DEFAULT_ENGLISH_VOICE_ID,
    DEFAULT_PORTUGUESE_VOICE_ID,
    LocalTTSProvider,
    PocketTTSLocalProvider,
    TTSError,
    TTSManager,
    TTSResult,
    VoiceSpec,
    WindowsSapiProvider,
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
        self.list_voice_calls = 0

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
        self.list_voice_calls += 1
        return list(self.voices)

    def is_available(self) -> bool:
        return True

    def diagnostics(self) -> dict:
        return {
            "provider": self.name,
            "provider_available": self.is_available(),
            "voices": [voice_to_test_diagnostic(voice) for voice in self.voices],
        }

    def warmup(self) -> None:
        return None

    def stop(self) -> None:
        self.stopped = True


def voice_to_test_diagnostic(voice: VoiceSpec) -> dict:
    return {
        "voice_id": voice.voice_id,
        "display_name": voice.display_name,
        "provider": voice.provider,
        "language": voice.language,
        "locale": voice.locale,
        "reference": voice.reference,
        "available": voice.available,
        "synthesizable": voice.synthesizable,
        "rejection_reason": voice.rejection_reason,
        "reference_available": bool(voice.reference_audio_path),
    }


class FakeCompletedProcess:
    def __init__(self, *, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeChatterboxRunner:
    def __init__(self, *, probe_ok: bool = True, synth_ok: bool = True) -> None:
        self.probe_ok = probe_ok
        self.synth_ok = synth_ok
        self.commands: list[list[str]] = []

    def __call__(self, command, **_kwargs):
        command = [str(part) for part in command]
        self.commands.append(command)
        if "--probe" in command:
            payload = {"ok": self.probe_ok, "model": "ResembleAI/chatterbox", "device": "cpu"}
            return FakeCompletedProcess(returncode=0 if self.probe_ok else 1, stdout=json.dumps(payload))
        if "--synthesize" in command and self.synth_ok:
            output = Path(command[command.index("--output") + 1])
            output.write_bytes(b"RIFF....WAVE")
            payload = {
                "ok": True,
                "model": "ResembleAI/chatterbox",
                "device": "cpu",
                "model_load_ms": 1234.0,
                "synthesis_ms": 456.0,
            }
            return FakeCompletedProcess(stdout=json.dumps(payload))
        return FakeCompletedProcess(returncode=1, stdout=json.dumps({"ok": False, "error": "synthesis_failed"}))


class FakeServerStdout:
    def __init__(self) -> None:
        self.lines: queue.Queue[str | None] = queue.Queue()

    def push(self, payload: dict) -> None:
        self.lines.put(json.dumps(payload, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self.lines.put(None)

    def __iter__(self):
        return self

    def __next__(self) -> str:
        line = self.lines.get()
        if line is None:
            raise StopIteration
        return line


class FakeServerStdin:
    def __init__(self, process: "FakeChatterboxProcess") -> None:
        self.process = process

    def write(self, value: str) -> int:
        self.process.received.append(value)
        request = json.loads(value)
        if request.get("type") == "shutdown":
            self.process.stdout.push({"type": "shutdown", "ok": True, "id": request.get("id")})
            return len(value)
        self.process.synthesis_count += 1
        output = Path(request["output"])
        output.write_bytes(b"RIFF....WAVE")
        provider_name = getattr(self.process, "provider_name", "chatterbox")
        model_name = getattr(self.process, "model_name", "ResembleAI/chatterbox")
        self.process.stdout.push(
            {
                "type": "result",
                "ok": True,
                "id": request.get("id"),
                "provider": provider_name,
                "model": model_name,
                "device": "cpu",
                "model_load_ms": 1111.0,
                "worker_launch_ms": 2.0,
                "reference_prepare_ms": 33.0 if self.process.synthesis_count == 1 else 1.0,
                "first_audio_ms": 222.0,
                "synthesis_ms": 222.0,
                "audio_duration_ms": 900.0,
                "reference_cache_hit": self.process.synthesis_count > 1,
                "output": str(output),
            }
        )
        return len(value)

    def flush(self) -> None:
        return None


class FakeChatterboxProcess:
    def __init__(self, *_args, **_kwargs) -> None:
        self.stdout = FakeServerStdout()
        self.provider_name = "chatterbox"
        self.model_name = "ResembleAI/chatterbox"
        self.stdin = FakeServerStdin(self)
        self.stderr = []
        self.received: list[str] = []
        self.synthesis_count = 0
        self.terminated = False
        self.stdout.push(
            {
                "type": "ready",
                "ok": True,
                "model": "ResembleAI/chatterbox",
                "device": "cpu",
                "model_load_ms": 1111.0,
            }
        )

    def poll(self):
        return None if not self.terminated else 0

    def terminate(self) -> None:
        self.terminated = True
        self.stdout.close()


class FakePocketTTSProcess(FakeChatterboxProcess):
    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__(*_args, **_kwargs)
        self.provider_name = "pocket_tts"
        self.model_name = "kyutai/pocket-tts"
        self.stdout = FakeServerStdout()
        self.stdin = FakeServerStdin(self)
        self.stderr = []
        self.received = []
        self.synthesis_count = 0
        self.terminated = False
        self.stdout.push(
            {
                "type": "ready",
                "ok": True,
                "model": "kyutai/pocket-tts",
                "device": "cpu",
                "worker_launch_ms": 3.0,
                "model_load_ms": 88.0,
            }
        )


def sapi_voice(name: str, locale: str, *, gender: str = "female", provider: str = "fake_tts") -> VoiceSpec:
    return VoiceSpec(
        voice_id=f"sapi:{name}",
        display_name=name,
        provider=provider,
        language=locale.split("-")[0],
        locale=locale,
        reference=name,
        available=True,
        gender=gender,
    )


def fake_chatterbox_provider(
    tmp_path: Path,
    *,
    reference_name: str = "voice.wav",
    runner: FakeChatterboxRunner | None = None,
) -> ChatterboxLocalProvider:
    tmp_path.mkdir(parents=True, exist_ok=True)
    reference = tmp_path / reference_name
    reference.write_bytes(b"reference")
    runtime_python = tmp_path / "python.exe"
    runner_path = tmp_path / "chatterbox_runner.py"
    runtime_python.write_text("", encoding="utf-8")
    runner_path.write_text("", encoding="utf-8")

    return ChatterboxLocalProvider(
        env={
            "ECHO_TTS_ENGLISH_REFERENCE_AUDIO": str(reference),
            "ECHO_TTS_CHATTERBOX_DEVICE": "cpu",
            "ECHO_TTS_ENGLISH_EXAGGERATION": "0.62",
            "ECHO_TTS_ENGLISH_CFG_WEIGHT": "0.44",
            "ECHO_TTS_ENGLISH_TEMPERATURE": "0.75",
            "ECHO_TTS_ENGLISH_PACE": "1.0",
        },
        runtime_python=runtime_python,
        runner_path=runner_path,
        command_runner=runner or FakeChatterboxRunner(),
    )


def fake_pocket_provider(
    tmp_path: Path,
    *,
    reference_name: str = "voice.wav",
    process_factory=None,
    runner: FakeChatterboxRunner | None = None,
    predefined_voice: str = "",
) -> PocketTTSLocalProvider:
    """`predefined_voice=""` (the default here) disables the Phase 7.3.3
    "anna" production default so pre-existing reference-audio-cloning tests
    keep exercising exactly that path -- pass predefined_voice="anna" (or
    any other name) to exercise the new predefined-voice binding instead."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    reference = tmp_path / reference_name
    reference.write_bytes(b"reference")
    runtime_python = tmp_path / "python.exe"
    runner_path = tmp_path / "pocket_tts_runner.py"
    runtime_python.write_text("", encoding="utf-8")
    runner_path.write_text("", encoding="utf-8")

    return PocketTTSLocalProvider(
        env={
            "ECHO_TTS_ENGLISH_REFERENCE_AUDIO": str(reference),
            "ECHO_TTS_POCKET_THREADS": "2",
            "ECHO_TTS_ENGLISH_POCKET_VOICE": predefined_voice,
        },
        runtime_python=runtime_python,
        runner_path=runner_path,
        command_runner=runner,
        process_factory=process_factory,
    )


def test_voice_registry_exposes_british_and_ptpt_slots() -> None:
    provider = FakeProvider([sapi_voice("Microsoft Hazel", "en-GB"), sapi_voice("Microsoft Helia", "pt-PT")])
    manager = TTSManager(provider=provider, store=FakeStore())

    voices = {voice.voice_id: voice for voice in manager.list_voices()}

    assert voices[DEFAULT_ENGLISH_VOICE_ID].available is True
    assert voices[DEFAULT_ENGLISH_VOICE_ID].language == "en"
    assert voices[DEFAULT_PORTUGUESE_VOICE_ID].available is True
    assert voices[DEFAULT_PORTUGUESE_VOICE_ID].language == "pt"


def test_pocket_tts_provider_conforms_to_tts_provider(tmp_path: Path) -> None:
    provider = fake_pocket_provider(tmp_path, process_factory=FakePocketTTSProcess, runner=FakeChatterboxRunner())

    assert provider.name == "pocket_tts"
    assert provider.is_available() is True
    assert provider.list_voices()[0].provider == "pocket_tts"


def test_pocket_tts_provider_unavailable_without_isolated_runtime() -> None:
    provider = PocketTTSLocalProvider(
        env={},
        runtime_python=Path("missing") / "python.exe",
        runner_path=Path("missing") / "pocket_tts_runner.py",
    )

    assert provider.is_available() is False
    voices = {voice.locale: voice for voice in provider.list_voices()}
    assert voices["en-GB"].available is False
    assert voices["en-GB"].rejection_reason == "pocket_tts_dependency_unavailable"


def test_pocket_tts_missing_reference_marks_voice_unavailable(tmp_path: Path) -> None:
    runtime_python = tmp_path / "python.exe"
    runner_path = tmp_path / "pocket_tts_runner.py"
    runtime_python.write_text("", encoding="utf-8")
    runner_path.write_text("", encoding="utf-8")
    provider = PocketTTSLocalProvider(
        env={"ECHO_TTS_ENGLISH_REFERENCE_AUDIO": str(tmp_path / "missing.wav"), "ECHO_TTS_ENGLISH_POCKET_VOICE": ""},
        runtime_python=runtime_python,
        runner_path=runner_path,
        command_runner=FakeChatterboxRunner(),
    )

    voices = {voice.locale: voice for voice in provider.list_voices()}

    assert voices["en-GB"].available is False
    assert voices["en-GB"].rejection_reason == "reference_audio_missing"


def test_pocket_tts_persistent_runner_reuses_model_and_reference_state(tmp_path: Path) -> None:
    processes: list[FakePocketTTSProcess] = []

    def process_factory(*args, **kwargs):
        process = FakePocketTTSProcess(*args, **kwargs)
        processes.append(process)
        return process

    provider = fake_pocket_provider(tmp_path, process_factory=process_factory, runner=FakeChatterboxRunner())
    manager = TTSManager(provider=provider, store=FakeStore())

    first = manager.speak_to_file("Hi. How are you?", response_language="en-GB")
    second = manager.speak_to_file("Good to hear.", response_language="en-GB")

    assert first.provider == "pocket_tts"
    assert second.provider == "pocket_tts"
    assert len(processes) == 1
    assert processes[0].synthesis_count == 2
    assert first.reference_cache_hit is False
    assert second.reference_cache_hit is True
    assert second.reference_prepare_ms <= first.reference_prepare_ms


def test_pocket_tts_timeout_falls_back_to_sapi_without_repeating_timeout(tmp_path: Path) -> None:
    class SilentStdout(FakeServerStdout):
        def __init__(self) -> None:
            super().__init__()
            self.push({"type": "ready", "ok": True, "model_load_ms": 10.0, "device": "cpu"})

    class SilentPocketProcess(FakePocketTTSProcess):
        def __init__(self, *_args, **_kwargs) -> None:
            self.stdout = SilentStdout()

            class SilentStdin:
                def write(inner_self, value: str) -> int:
                    self.received.append(value)
                    return len(value)

                def flush(inner_self) -> None:
                    return None

            self.stdin = SilentStdin()
            self.stderr = []
            self.received = []
            self.synthesis_count = 0
            self.terminated = False

    processes: list[SilentPocketProcess] = []

    def process_factory(*args, **kwargs):
        process = SilentPocketProcess(*args, **kwargs)
        processes.append(process)
        return process

    pocket = fake_pocket_provider(tmp_path, process_factory=process_factory, runner=FakeChatterboxRunner())
    pocket.synthesis_timeout_seconds = 0.01
    sapi = FakeProvider([sapi_voice("Microsoft Hazel", "en-GB", provider=WindowsSapiProvider.name)])
    sapi.name = WindowsSapiProvider.name
    manager = TTSManager(provider=LocalTTSProvider([pocket, sapi]), store=FakeStore())

    first = manager.speak_to_file("Hello.", response_language="en-GB")
    second = manager.speak_to_file("Again.", response_language="en-GB")

    assert first.provider == WindowsSapiProvider.name
    assert second.provider == WindowsSapiProvider.name
    assert first.fallback_used is True
    assert first.fallback_reason == "provider_failed:pocket_tts"
    assert len(processes) == 1


def test_pocket_tts_without_voice_cloning_falls_back_before_synthesis(tmp_path: Path) -> None:
    """Phase 7.3.3 (section 3/7): has_voice_cloning=False is no longer a
    fatal SERVER-STARTUP condition -- a predefined/catalog voice (e.g.
    "anna") works fine regardless (see the dedicated
    test_pocket_tts_predefined_voice_works_without_voice_cloning below).
    Only a genuine REFERENCE-AUDIO (cloning) request still fails, and now
    does so per-request (the runner actually attempts and rejects it) rather
    than the whole server refusing to start."""

    class NoCloneStdout(FakeServerStdout):
        def __init__(self) -> None:
            super().__init__()
            self.push(
                {
                    "type": "ready",
                    "ok": True,
                    "model_load_ms": 10.0,
                    "device": "cpu",
                    "has_voice_cloning": False,
                }
            )

    class NoCloneStdin(FakeServerStdin):
        def write(self, value: str) -> int:
            self.process.received.append(value)
            request = json.loads(value)
            if request.get("type") == "shutdown":
                self.process.stdout.push({"type": "shutdown", "ok": True, "id": request.get("id")})
                return len(value)
            self.process.synthesis_count += 1
            if not request.get("predefined_voice"):
                self.process.stdout.push(
                    {"type": "result", "ok": False, "id": request.get("id"), "error": "voice_cloning_weights_unavailable"}
                )
                return len(value)
            return super().write(value)

    class NoClonePocketProcess(FakePocketTTSProcess):
        def __init__(self, *_args, **_kwargs) -> None:
            super().__init__(*_args, **_kwargs)
            self.stdout = NoCloneStdout()
            self.stdin = NoCloneStdin(self)

    processes: list[NoClonePocketProcess] = []

    def process_factory(*args, **kwargs):
        process = NoClonePocketProcess(*args, **kwargs)
        processes.append(process)
        return process

    pocket = fake_pocket_provider(tmp_path, process_factory=process_factory, runner=FakeChatterboxRunner())
    sapi = FakeProvider([sapi_voice("Microsoft Hazel", "en-GB", provider=WindowsSapiProvider.name)])
    sapi.name = WindowsSapiProvider.name
    manager = TTSManager(provider=LocalTTSProvider([pocket, sapi]), store=FakeStore())

    result = manager.speak_to_file("Hello.", response_language="en-GB")
    second = manager.speak_to_file("Again.", response_language="en-GB")

    assert result.provider == WindowsSapiProvider.name
    assert second.provider == WindowsSapiProvider.name
    assert result.fallback_used is True
    assert result.fallback_reason == "provider_failed:pocket_tts"
    assert processes[0].synthesis_count == 1
    assert len(processes) == 1


def test_local_tts_prefers_pocket_then_chatterbox_then_sapi_for_english(tmp_path: Path) -> None:
    pocket_processes: list[FakePocketTTSProcess] = []

    def pocket_factory(*args, **kwargs):
        process = FakePocketTTSProcess(*args, **kwargs)
        pocket_processes.append(process)
        return process

    pocket = fake_pocket_provider(tmp_path / "pocket", process_factory=pocket_factory, runner=FakeChatterboxRunner())
    chatterbox = fake_chatterbox_provider(tmp_path / "chatterbox")
    sapi = FakeProvider([sapi_voice("Microsoft Hazel", "en-GB", provider=WindowsSapiProvider.name)])
    sapi.name = WindowsSapiProvider.name
    manager = TTSManager(provider=LocalTTSProvider([pocket, chatterbox, sapi]), store=FakeStore())

    result = manager.speak_to_file("Hello.", response_language="en-GB")

    assert result.provider == "pocket_tts"
    assert pocket_processes[0].synthesis_count == 1
    assert sapi.synthesized == []


def test_local_tts_falls_back_from_missing_pocket_reference_to_chatterbox(tmp_path: Path) -> None:
    pocket = PocketTTSLocalProvider(
        env={"ECHO_TTS_ENGLISH_REFERENCE_AUDIO": str(tmp_path / "missing.wav"), "ECHO_TTS_ENGLISH_POCKET_VOICE": ""},
        runtime_python=tmp_path / "python.exe",
        runner_path=tmp_path / "pocket_tts_runner.py",
        command_runner=FakeChatterboxRunner(),
    )
    (tmp_path / "python.exe").write_text("", encoding="utf-8")
    (tmp_path / "pocket_tts_runner.py").write_text("", encoding="utf-8")
    chatterbox = fake_chatterbox_provider(tmp_path / "chatterbox")
    manager = TTSManager(provider=LocalTTSProvider([pocket, chatterbox]), store=FakeStore())

    result = manager.speak_to_file("Hello.", response_language="en-GB")

    assert result.provider == "chatterbox"


def test_english_british_primary_resolves_to_pocket_anna(tmp_path: Path) -> None:
    """A. Phase 7.3.3: the production binding -- english_british_primary
    resolves to the Pocket TTS provider, using the predefined 'anna' voice,
    with no fallback needed."""
    pocket = fake_pocket_provider(tmp_path, process_factory=FakePocketTTSProcess, runner=FakeChatterboxRunner(), predefined_voice="anna")
    manager = TTSManager(provider=LocalTTSProvider([pocket]), store=FakeStore())

    result = manager.speak_to_file("Good morning.", response_language="en-GB")

    assert result.provider == "pocket_tts"
    assert result.voice_id == DEFAULT_ENGLISH_VOICE_ID
    assert result.provider_voice == "anna"
    assert result.fallback_used is False


def test_pocket_anna_available_without_any_reference_audio(tmp_path: Path) -> None:
    """B. Anna does not require cloning/reference audio -- no
    ECHO_TTS_ENGLISH_REFERENCE_AUDIO is set at all."""
    runtime_python = tmp_path / "python.exe"
    runner_path = tmp_path / "pocket_tts_runner.py"
    runtime_python.write_text("", encoding="utf-8")
    runner_path.write_text("", encoding="utf-8")
    provider = PocketTTSLocalProvider(
        env={"ECHO_TTS_ENGLISH_POCKET_VOICE": "anna"},
        runtime_python=runtime_python,
        runner_path=runner_path,
        command_runner=FakeChatterboxRunner(),
    )

    voices = {voice.locale: voice for voice in provider.list_voices()}

    assert voices["en-GB"].available is True
    assert voices["en-GB"].synthesizable is True
    assert voices["en-GB"].rejection_reason == ""
    assert voices["en-GB"].reference_audio_path == ""
    assert voices["en-GB"].synthesis_options["predefined_voice"] == "anna"


def test_pocket_anna_available_even_with_missing_chloe_reference(tmp_path: Path) -> None:
    """C. A missing Chloe reference file does not disable Anna -- the
    predefined-voice binding never consults ECHO_TTS_ENGLISH_REFERENCE_AUDIO
    at all once configured."""
    runtime_python = tmp_path / "python.exe"
    runner_path = tmp_path / "pocket_tts_runner.py"
    runtime_python.write_text("", encoding="utf-8")
    runner_path.write_text("", encoding="utf-8")
    provider = PocketTTSLocalProvider(
        env={
            "ECHO_TTS_ENGLISH_REFERENCE_AUDIO": str(tmp_path / "chloe_missing.wav"),
            "ECHO_TTS_ENGLISH_POCKET_VOICE": "anna",
        },
        runtime_python=runtime_python,
        runner_path=runner_path,
        command_runner=FakeChatterboxRunner(),
    )

    voices = {voice.locale: voice for voice in provider.list_voices()}

    assert voices["en-GB"].available is True
    assert voices["en-GB"].rejection_reason == ""


def test_pocket_anna_unavailable_falls_back_to_chatterbox(tmp_path: Path) -> None:
    """D. Pocket unavailable (isolated runtime missing) -> falls through to
    the next English-compatible provider in the chain."""
    pocket = PocketTTSLocalProvider(
        env={"ECHO_TTS_ENGLISH_POCKET_VOICE": "anna"},
        runtime_python=Path("missing") / "python.exe",
        runner_path=Path("missing") / "pocket_tts_runner.py",
    )
    chatterbox = fake_chatterbox_provider(tmp_path / "chatterbox")
    manager = TTSManager(provider=LocalTTSProvider([pocket, chatterbox]), store=FakeStore())

    result = manager.speak_to_file("Hello.", response_language="en-GB")

    assert result.provider == "chatterbox"


def test_pocket_anna_reuses_persistent_worker_across_requests(tmp_path: Path) -> None:
    """E. The Pocket worker/model is not reloaded per SPEAK -- two
    syntheses through the same provider instance reuse one process."""
    processes: list[FakePocketTTSProcess] = []

    def process_factory(*args, **kwargs):
        process = FakePocketTTSProcess(*args, **kwargs)
        processes.append(process)
        return process

    pocket = fake_pocket_provider(tmp_path, process_factory=process_factory, runner=FakeChatterboxRunner(), predefined_voice="anna")
    manager = TTSManager(provider=LocalTTSProvider([pocket]), store=FakeStore())

    first = manager.speak_to_file("Good morning.", response_language="en-GB")
    second = manager.speak_to_file("Anything else?", response_language="en-GB")

    assert first.provider == "pocket_tts"
    assert second.provider == "pocket_tts"
    assert len(processes) == 1
    assert processes[0].synthesis_count == 2


def test_pocket_anna_voice_state_prepared_once_and_reused(tmp_path: Path) -> None:
    """F. Anna's conditioning state is prepared once and reused -- the
    second synthesis reports a cache hit, mirroring the reference-audio
    cache behavior this same mechanism already had."""
    pocket = fake_pocket_provider(tmp_path, process_factory=FakePocketTTSProcess, runner=FakeChatterboxRunner(), predefined_voice="anna")
    manager = TTSManager(provider=LocalTTSProvider([pocket]), store=FakeStore())

    first = manager.speak_to_file("Good morning.", response_language="en-GB")
    second = manager.speak_to_file("Anything else?", response_language="en-GB")

    assert first.reference_cache_hit is False
    assert second.reference_cache_hit is True
    assert second.reference_prepare_ms <= first.reference_prepare_ms


def test_pocket_process_recreated_after_genuine_worker_crash(tmp_path: Path) -> None:
    """J. A genuinely dead worker process (crashed -- not just a UI Stop or
    a new user message arriving mid-playback) IS replaced with a fresh one
    on the next request. This is the ONE legitimate restart trigger,
    distinct from the fixed bug (Phase 7.3.4 section 5) where merely
    stopping playback used to also tear down a perfectly healthy worker."""
    processes: list[FakePocketTTSProcess] = []

    def process_factory(*args, **kwargs):
        process = FakePocketTTSProcess(*args, **kwargs)
        processes.append(process)
        return process

    pocket = fake_pocket_provider(tmp_path, process_factory=process_factory, runner=FakeChatterboxRunner(), predefined_voice="anna")
    manager = TTSManager(provider=LocalTTSProvider([pocket]), store=FakeStore())

    first = manager.speak_to_file("Good morning.", response_language="en-GB")
    assert len(processes) == 1

    # Simulate the underlying worker process having genuinely crashed --
    # NOT an app-triggered stop/shutdown (those never touch .terminated).
    processes[0].terminated = True

    second = manager.speak_to_file("Anything else?", response_language="en-GB")

    assert first.provider == "pocket_tts"
    assert second.provider == "pocket_tts"
    assert len(processes) == 2


def test_pocket_tts_does_not_make_ptpt_available(tmp_path: Path) -> None:
    pocket = fake_pocket_provider(tmp_path, process_factory=FakePocketTTSProcess, runner=FakeChatterboxRunner())
    manager = TTSManager(provider=LocalTTSProvider([pocket]), store=FakeStore())

    assert manager.is_available(response_language="pt-PT") is False
    with pytest.raises(TTSError, match="voz portuguesa"):
        manager.speak_to_file("Olá.", response_language="pt-PT")


def test_chatterbox_provider_conforms_to_tts_provider(tmp_path: Path) -> None:
    provider = fake_chatterbox_provider(tmp_path)

    assert provider.name == "chatterbox"
    assert provider.is_available() is True
    assert provider.list_voices()[0].provider == "chatterbox"


def test_chatterbox_provider_unavailable_without_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ChatterboxLocalProvider(
        env={},
        runtime_python=Path("missing") / "python.exe",
        runner_path=Path("missing") / "runner.py",
    )

    assert provider.is_available() is False
    voices = {voice.locale: voice for voice in provider.list_voices()}
    assert voices["en-GB"].available is False
    assert voices["en-GB"].rejection_reason == "chatterbox_dependency_unavailable"


def test_chatterbox_missing_reference_audio_marks_voice_unavailable(tmp_path: Path) -> None:
    runtime_python = tmp_path / "python.exe"
    runner_path = tmp_path / "runner.py"
    runtime_python.write_text("", encoding="utf-8")
    runner_path.write_text("", encoding="utf-8")
    provider = ChatterboxLocalProvider(
        env={"ECHO_TTS_ENGLISH_REFERENCE_AUDIO": str(tmp_path / "missing.wav")},
        runtime_python=runtime_python,
        runner_path=runner_path,
        command_runner=FakeChatterboxRunner(),
    )

    voices = {voice.locale: voice for voice in provider.list_voices()}

    assert voices["en-GB"].available is False
    assert voices["en-GB"].rejection_reason == "reference_audio_missing"


def test_chatterbox_reference_path_resolution_and_voice_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    reference = Path("refs") / "british.wav"
    reference.parent.mkdir()
    reference.write_bytes(b"reference")
    provider = ChatterboxLocalProvider(
        env={
            "ECHO_TTS_ENGLISH_REFERENCE_AUDIO": str(reference),
            "ECHO_TTS_ENGLISH_EXAGGERATION": "0.67",
            "ECHO_TTS_ENGLISH_CFG_WEIGHT": "0.41",
            "ECHO_TTS_ENGLISH_TEMPERATURE": "0.72",
        },
        runtime_python=tmp_path / "python.exe",
        runner_path=tmp_path / "runner.py",
        command_runner=FakeChatterboxRunner(),
    )
    (tmp_path / "python.exe").write_text("", encoding="utf-8")
    (tmp_path / "runner.py").write_text("", encoding="utf-8")

    voice = provider.list_voices()[0]

    assert Path(voice.reference_audio_path).is_absolute()
    assert Path(voice.reference_audio_path).exists()
    assert voice.synthesis_options == {"exaggeration": 0.67, "cfg_weight": 0.41, "temperature": 0.72, "pace": 1.0}


def test_chatterbox_runner_invocation_uses_isolated_interpreter(tmp_path: Path) -> None:
    runner = FakeChatterboxRunner()
    provider = fake_chatterbox_provider(tmp_path, runner=runner)
    manager = TTSManager(provider=provider, store=FakeStore())

    result = manager.speak_to_file("Good morning.", response_language="en-GB")

    assert result.provider == "chatterbox"
    assert runner.commands[0][0].endswith("python.exe")
    assert runner.commands[0][1].endswith("chatterbox_runner.py")
    assert "--probe" in runner.commands[0]
    assert "--synthesize" in runner.commands[1]
    assert runner.commands[1][runner.commands[1].index("--text") + 1] == "Good morning."


def test_chatterbox_subprocess_bridge_reports_cold_load_per_synthesis(tmp_path: Path) -> None:
    runner = FakeChatterboxRunner()
    provider = fake_chatterbox_provider(tmp_path, runner=runner)
    manager = TTSManager(provider=provider, store=FakeStore())

    first = manager.speak_to_file("Good morning.", response_language="en-GB")
    second = manager.speak_to_file("I found three options.", response_language="en-GB")

    assert len([command for command in runner.commands if "--synthesize" in command]) == 2
    assert first.model_load_ms == 1234.0
    assert second.model_load_ms == 1234.0


def test_chatterbox_persistent_runner_reuses_loaded_model(tmp_path: Path) -> None:
    runner = FakeChatterboxRunner()
    processes: list[FakeChatterboxProcess] = []

    def process_factory(*args, **kwargs):
        process = FakeChatterboxProcess(*args, **kwargs)
        processes.append(process)
        return process

    provider = fake_chatterbox_provider(tmp_path, runner=runner)
    provider.process_factory = process_factory
    provider._use_persistent_server = True
    manager = TTSManager(provider=provider, store=FakeStore())

    first = manager.speak_to_file("Hi. How are you?", response_language="en-GB")
    second = manager.speak_to_file("Good to hear.", response_language="en-GB")

    assert first.model_load_ms == 1111.0
    assert second.model_load_ms == 1111.0
    assert len(processes) == 1
    assert processes[0].synthesis_count == 2


def test_chatterbox_persistent_runner_timeout_clears_process(tmp_path: Path) -> None:
    class SilentStdout(FakeServerStdout):
        def __init__(self) -> None:
            super().__init__()
            self.push({"type": "ready", "ok": True, "model_load_ms": 10.0, "device": "cpu"})

    class SilentProcess(FakeChatterboxProcess):
        def __init__(self, *_args, **_kwargs) -> None:
            self.stdout = SilentStdout()
            class SilentStdin:
                def write(inner_self, value: str) -> int:
                    self.received.append(value)
                    return len(value)

                def flush(inner_self) -> None:
                    return None

            self.stdin = SilentStdin()
            self.stderr = []
            self.received = []
            self.synthesis_count = 0
            self.terminated = False

    provider = fake_chatterbox_provider(tmp_path, runner=FakeChatterboxRunner())
    provider.process_factory = lambda *args, **kwargs: SilentProcess(*args, **kwargs)
    provider._use_persistent_server = True
    provider.synthesis_timeout_seconds = 0.01
    manager = TTSManager(provider=provider, store=FakeStore())

    with pytest.raises(TTSError, match="Chatterbox"):
        manager.speak_to_file("This will time out.", response_language="en-GB")


def test_chatterbox_timeout_does_not_repeat_before_sapi_fallback(tmp_path: Path) -> None:
    class SilentStdout(FakeServerStdout):
        def __init__(self) -> None:
            super().__init__()
            self.push({"type": "ready", "ok": True, "model_load_ms": 10.0, "device": "cpu"})

    class SilentProcess(FakeChatterboxProcess):
        def __init__(self, *_args, **_kwargs) -> None:
            self.stdout = SilentStdout()

            class SilentStdin:
                def write(inner_self, value: str) -> int:
                    self.received.append(value)
                    return len(value)

                def flush(inner_self) -> None:
                    return None

            self.stdin = SilentStdin()
            self.stderr = []
            self.received = []
            self.synthesis_count = 0
            self.terminated = False

    processes: list[SilentProcess] = []

    def process_factory(*args, **kwargs):
        process = SilentProcess(*args, **kwargs)
        processes.append(process)
        return process

    chatterbox = fake_chatterbox_provider(tmp_path, runner=FakeChatterboxRunner())
    chatterbox.process_factory = process_factory
    chatterbox._use_persistent_server = True
    chatterbox.synthesis_timeout_seconds = 0.01
    sapi = FakeProvider([sapi_voice("Microsoft Hazel", "en-GB", provider=WindowsSapiProvider.name)])
    sapi.name = WindowsSapiProvider.name
    manager = TTSManager(provider=LocalTTSProvider([chatterbox, sapi]), store=FakeStore())

    first = manager.speak_to_file("Hello.", response_language="en-GB")
    second = manager.speak_to_file("Again.", response_language="en-GB")

    assert first.provider == WindowsSapiProvider.name
    assert second.provider == WindowsSapiProvider.name
    assert len(processes) == 1


def test_chatterbox_synthesis_returns_playable_result_with_reference_telemetry(tmp_path: Path) -> None:
    provider = fake_chatterbox_provider(tmp_path)
    manager = TTSManager(provider=provider, store=FakeStore())

    result = manager.speak_to_file("Your schedule is quite full today.", response_language="en-GB")

    assert result.audio_path.exists()
    assert result.provider == "chatterbox"
    assert result.voice_id == DEFAULT_ENGLISH_VOICE_ID
    assert result.device == "cpu"
    assert result.model == "ResembleAI/chatterbox"
    assert result.reference_used is True
    runner = provider.command_runner
    synth_command = next(command for command in runner.commands if "--synthesize" in command)
    assert synth_command[synth_command.index("--reference") + 1].endswith("voice.wav")
    options = json.loads(synth_command[synth_command.index("--options-json") + 1])
    assert options["exaggeration"] == 0.62


def test_chatterbox_provider_failure_preserves_text_as_error_without_output(tmp_path: Path) -> None:
    reference = tmp_path / "voice.wav"
    reference.write_bytes(b"reference")

    provider = ChatterboxLocalProvider(
        env={"ECHO_TTS_ENGLISH_REFERENCE_AUDIO": str(reference)},
        runtime_python=tmp_path / "python.exe",
        runner_path=tmp_path / "runner.py",
        command_runner=FakeChatterboxRunner(synth_ok=False),
    )
    (tmp_path / "python.exe").write_text("", encoding="utf-8")
    (tmp_path / "runner.py").write_text("", encoding="utf-8")
    manager = TTSManager(provider=provider, store=FakeStore())

    with pytest.raises(TTSError, match="Chatterbox"):
        manager.speak_to_file("Text remains canonical.", response_language="en-GB")


def test_local_tts_prefers_chatterbox_then_sapi_for_english(tmp_path: Path) -> None:
    chatterbox = fake_chatterbox_provider(tmp_path)
    sapi = FakeProvider([sapi_voice("Microsoft Hazel", "en-GB", provider=WindowsSapiProvider.name)])
    sapi.name = WindowsSapiProvider.name
    manager = TTSManager(provider=LocalTTSProvider([chatterbox, sapi]), store=FakeStore())

    result = manager.speak_to_file("Hello.", response_language="en-GB")

    assert result.provider == "chatterbox"
    assert sapi.synthesized == []


def test_local_tts_falls_back_to_sapi_when_chatterbox_reference_missing(tmp_path: Path) -> None:
    chatterbox = ChatterboxLocalProvider(
        env={"ECHO_TTS_ENGLISH_REFERENCE_AUDIO": str(tmp_path / "missing.wav")},
        runtime_python=tmp_path / "python.exe",
        runner_path=tmp_path / "runner.py",
        command_runner=FakeChatterboxRunner(),
    )
    (tmp_path / "python.exe").write_text("", encoding="utf-8")
    (tmp_path / "runner.py").write_text("", encoding="utf-8")
    sapi = FakeProvider([sapi_voice("Microsoft Hazel", "en-GB", provider=WindowsSapiProvider.name)])
    sapi.name = WindowsSapiProvider.name
    manager = TTSManager(provider=LocalTTSProvider([chatterbox, sapi]), store=FakeStore())

    result = manager.speak_to_file("Hello.", response_language="en-GB")

    assert result.provider == WindowsSapiProvider.name
    assert sapi.synthesized[0][0] == "Hello."


def test_local_tts_falls_back_to_sapi_when_chatterbox_synthesis_fails(tmp_path: Path) -> None:
    chatterbox = fake_chatterbox_provider(tmp_path, runner=FakeChatterboxRunner(synth_ok=False))
    sapi = FakeProvider([sapi_voice("Microsoft Hazel", "en-GB", provider=WindowsSapiProvider.name)])
    sapi.name = WindowsSapiProvider.name
    manager = TTSManager(provider=LocalTTSProvider([chatterbox, sapi]), store=FakeStore())

    result = manager.speak_to_file("Hello.", response_language="en-GB")

    assert result.provider == WindowsSapiProvider.name
    assert sapi.synthesized[0][0] == "Hello."


def test_local_tts_never_uses_english_chatterbox_for_ptpt(tmp_path: Path) -> None:
    chatterbox = fake_chatterbox_provider(tmp_path)
    manager = TTSManager(provider=LocalTTSProvider([chatterbox]), store=FakeStore())

    with pytest.raises(TTSError, match="voz portuguesa"):
        manager.speak_to_file("Olá.", response_language="pt-PT")


def test_local_tts_caches_voice_discovery() -> None:
    provider = FakeProvider([sapi_voice("Microsoft Hazel", "en-GB", provider=WindowsSapiProvider.name)])
    provider.name = WindowsSapiProvider.name
    local = LocalTTSProvider([provider])

    first = local.list_voices()
    second = local.list_voices()

    assert first == second
    assert provider.list_voice_calls == 1


def test_windows_sapi_diagnostics_does_not_relog_unchanged_discovery(capsys) -> None:
    """K. discover_voices() was already cached, but diagnostics() used to
    re-emit a [TTS_DISCOVERY] line per voice on every single call -- the
    real noise source (Phase 7.3.4 section 7), since diagnostics() is
    invoked repeatedly per turn/SPEAK. The underlying state is still
    recomputed fresh every call; only the duplicate identical log lines are
    suppressed."""
    provider = WindowsSapiProvider()
    provider._voice_cache = [sapi_voice("Microsoft Hazel", "en-GB")]

    provider.diagnostics()
    first_log_count = capsys.readouterr().out.count("[TTS_DISCOVERY]")
    provider.diagnostics()
    second_log_count = capsys.readouterr().out.count("[TTS_DISCOVERY]")

    assert first_log_count == 1
    assert second_log_count == 0


def test_tts_manager_availability_does_not_relog_unchanged_state(tmp_path: Path, capsys) -> None:
    """K. TTSManager.is_available() used to print a fresh [TTS_AVAILABILITY]
    line on every call regardless of whether anything changed -- called
    repeatedly per turn/SPEAK (debounced UI poll + speakText()'s own
    eligibility re-check), producing duplicate bursts for one user action."""
    pocket = fake_pocket_provider(tmp_path, process_factory=FakePocketTTSProcess, runner=FakeChatterboxRunner(), predefined_voice="anna")
    manager = TTSManager(provider=LocalTTSProvider([pocket]), store=FakeStore())

    manager.is_available(response_language="en-GB")
    first_log_count = capsys.readouterr().out.count("[TTS_AVAILABILITY]")
    manager.is_available(response_language="en-GB")
    second_log_count = capsys.readouterr().out.count("[TTS_AVAILABILITY]")

    assert first_log_count == 1
    assert second_log_count == 0


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


def test_engb_availability_uses_compatible_sapi_fallback_when_chatterbox_reference_is_missing(tmp_path: Path) -> None:
    chatterbox = ChatterboxLocalProvider(
        env={"ECHO_TTS_ENGLISH_REFERENCE_AUDIO": str(tmp_path / "missing.wav")},
        runtime_python=tmp_path / "python.exe",
        runner_path=tmp_path / "runner.py",
        command_runner=FakeChatterboxRunner(),
    )
    (tmp_path / "python.exe").write_text("", encoding="utf-8")
    (tmp_path / "runner.py").write_text("", encoding="utf-8")
    sapi = FakeProvider([sapi_voice("Microsoft Zira Desktop", "en-US", provider=WindowsSapiProvider.name)])
    sapi.name = WindowsSapiProvider.name
    manager = TTSManager(provider=LocalTTSProvider([chatterbox, sapi]), store=FakeStore())

    diagnostic = manager.availability_diagnostics(response_language="en-GB")

    assert manager.is_available(response_language="en-GB") is True
    assert diagnostic["speak_available"] is True
    assert diagnostic["chatterbox_available"] is False
    assert diagnostic["chatterbox_reference_available"] is False
    assert diagnostic["sapi_available"] is True
    assert diagnostic["compatible_fallback_available"] is True
    assert diagnostic["reason"] == "compatible_english_fallback"


def test_engb_availability_prefers_chatterbox_when_reference_is_available(tmp_path: Path) -> None:
    chatterbox = fake_chatterbox_provider(tmp_path)
    manager = TTSManager(provider=LocalTTSProvider([chatterbox]), store=FakeStore())

    diagnostic = manager.availability_diagnostics(response_language="en-GB")

    assert diagnostic["speak_available"] is True
    assert diagnostic["chatterbox_available"] is True
    assert diagnostic["chatterbox_reference_available"] is True
    assert diagnostic["compatible_fallback_available"] is False
    assert diagnostic["reason"] == "available"


def test_ptpt_availability_requires_playable_ptpt_voice() -> None:
    provider = FakeProvider([sapi_voice("Microsoft Zira Desktop", "en-US")])
    manager = TTSManager(provider=provider, store=FakeStore())

    diagnostic = manager.availability_diagnostics(response_language="pt-PT")

    assert manager.is_available(response_language="pt-PT") is False
    assert diagnostic["speak_available"] is False
    assert diagnostic["compatible_fallback_available"] is False
    assert diagnostic["reason"] == "no_playable_ptpt_voice"


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

from pathlib import Path
import json
import sys
import threading
from types import SimpleNamespace

import pytest

from assistant.voice_input import VoiceTranscriber


def test_settings_select_gpu_without_environment(monkeypatch):
    for key in ("ECHO_STT_PROVIDER", "ECHO_STT_MODEL", "ECHO_STT_PYTHON", "HF_HOME"):
        monkeypatch.delenv(key, raising=False)
    transcriber = VoiceTranscriber(model_name="medium", stt_settings={
        "stt_provider": "faster-whisper", "stt_model": "large-v3-turbo",
        "stt_python": ".venv_stt_bench/Scripts/python.exe", "stt_hf_home": ".hf_cache",
    })
    assert transcriber.stt_provider == "faster-whisper"
    assert transcriber.model_name == "large-v3-turbo"
    root = Path(__file__).resolve().parents[1]
    assert transcriber.stt_config.python == root / ".venv_stt_bench/Scripts/python.exe"
    assert transcriber.stt_config.hf_home == root / ".hf_cache"


def test_environment_overrides_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("ECHO_STT_PROVIDER", "faster-whisper")
    monkeypatch.setenv("ECHO_STT_MODEL", "medium")
    monkeypatch.setenv("ECHO_STT_PYTHON", str(tmp_path / "python.exe"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))
    transcriber = VoiceTranscriber(stt_settings={
        "stt_provider": "openai-whisper", "stt_model": "large-v3-turbo",
    })
    assert transcriber.stt_provider == "faster-whisper"
    assert transcriber.model_name == "medium"
    assert transcriber.stt_config.python == tmp_path / "python.exe"
    assert transcriber.stt_config.hf_home == tmp_path / "cache"


def test_relative_runtime_paths_do_not_depend_on_cwd(monkeypatch, tmp_path):
    for key in ("ECHO_STT_PYTHON", "HF_HOME"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    from assistant.stt_backend import FasterWhisperBackend
    backend = FasterWhisperBackend(python=".venv_stt_bench/Scripts/python.exe", hf_home=".hf_cache")
    root = Path(__file__).resolve().parents[1]
    assert backend.python == root / ".venv_stt_bench/Scripts/python.exe"
    assert backend.hf_home == root / ".hf_cache"


@pytest.mark.parametrize("failure", ["missing_runtime", "cuda_unavailable", "startup_failed"])
def test_gpu_failure_falls_back_with_warning(monkeypatch, tmp_path, capsys, failure):
    from assistant.stt_backend import FasterWhisperBackend
    from tests.test_stt_backend import audio
    calls = []
    cpu = SimpleNamespace(transcribe=lambda *a, **kw: {"text": "O microfone funciona.", "language": "pt"})
    monkeypatch.setitem(sys.modules, "whisper", SimpleNamespace(load_model=lambda name: calls.append(name) or cpu))
    transcriber = VoiceTranscriber(model_name="medium", stt_provider="faster-whisper")
    if failure == "missing_runtime":
        transcriber._stt_backend = FasterWhisperBackend(python=tmp_path / "absent.exe")
    else:
        def fail(*a, **kw):
            raise RuntimeError(failure)
        transcriber._stt_backend = SimpleNamespace(transcribe=fail, close=lambda: None)
    warnings = []
    transcriber.set_stt_status_callback(warnings.append)
    path = audio(tmp_path)
    assert transcriber.transcribe_existing_wav(path) == "O microfone funciona."
    assert calls == ["medium"]
    assert transcriber.stt_provider == "openai-whisper"
    assert transcriber.stt_degraded is True
    assert warnings == ["Voice recognition running in slow mode"]
    assert "degraded=true reason=" in capsys.readouterr().out
    assert path.exists()


def test_fallback_warning_is_delivered_on_gui_thread():
    from PySide6.QtCore import QThread
    from prototype_web_ui.controller import EchoUIController
    from tests.test_web_ui_prototype import _qt_app, _run_until
    app = _qt_app()
    transcriber = VoiceTranscriber(stt_provider="openai-whisper")
    controller = EchoUIController(lambda message: "ok", voice_transcriber=transcriber,
                                  get_response_language=lambda: "en-GB")
    events = []
    controller.uiEvent.connect(lambda event: events.append((json.loads(event), QThread.currentThread())))
    worker = threading.Thread(target=lambda: transcriber._stt_status_callback("Voice recognition running in slow mode"))
    try:
        worker.start()
        worker.join()
        _run_until(lambda: bool(events))
        assert events[0][0] == {"type": "voice_status", "status": "Voice recognition running in slow mode"}
        assert events[0][1] == app.thread()
    finally:
        controller.shutdown()


def test_missing_runtime_during_prewarm_keeps_cpu_ready(monkeypatch, tmp_path):
    calls = []
    model = object()
    monkeypatch.setitem(sys.modules, "whisper", SimpleNamespace(load_model=lambda name: calls.append(name) or model))
    monkeypatch.setenv("ECHO_STT_PYTHON", str(tmp_path / "absent.exe"))
    transcriber = VoiceTranscriber(model_name="medium", stt_provider="faster-whisper")
    warnings = []
    transcriber.set_stt_status_callback(warnings.append)
    assert transcriber.prewarm_stt() is True
    assert transcriber._model is model
    assert calls == ["medium"]
    assert transcriber.stt_degraded
    assert len(warnings) == 1


def test_cancelled_gpu_warmup_does_not_load_cpu(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "whisper", SimpleNamespace(load_model=lambda name: calls.append(name)))
    transcriber = VoiceTranscriber(stt_provider="faster-whisper")
    def warmup():
        transcriber.close()
        raise RuntimeError("cancelled")
    transcriber._stt_backend = SimpleNamespace(warmup=warmup, close=lambda: None)
    with pytest.raises(RuntimeError, match="STT closed"):
        transcriber.prewarm_stt()
    assert calls == []

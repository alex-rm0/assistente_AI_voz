from pathlib import Path

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

"""Resolve local STT settings without depending on the launch directory."""
from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class STTConfig:
    provider: str
    model: str
    python: Path
    hf_home: Path


def resolve_stt_config(settings=None, *, provider=None, cpu_model="base"):
    settings = settings or {}
    selected = provider or os.environ.get("ECHO_STT_PROVIDER") or settings.get("stt_provider") or "openai-whisper"
    model = cpu_model
    if selected == "faster-whisper":
        model = os.environ.get("ECHO_STT_MODEL") or settings.get("stt_model") or "large-v3-turbo"
    return STTConfig(
        selected, model,
        project_path(os.environ.get("ECHO_STT_PYTHON") or settings.get("stt_python") or ".venv_stt_bench/Scripts/python.exe"),
        project_path(os.environ.get("HF_HOME") or settings.get("stt_hf_home") or ".hf_cache"),
    )

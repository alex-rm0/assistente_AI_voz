"""Startup diagnosis: ChatterboxLocalProvider resolved its device in the
constructor (``import torch`` + ``torch.cuda.is_available()``), so every app
start paid ~1.5 s for torch even though Pocket TTS is the active provider.
The device must only be resolved when Chatterbox is really used (server
start / synthesis).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from assistant.tts import ChatterboxLocalProvider

ROOT = Path(__file__).resolve().parents[1]


def test_creating_tts_manager_does_not_import_torch() -> None:
    env = {key: value for key, value in os.environ.items() if key != "ECHO_TTS_CHATTERBOX_DEVICE"}
    code = (
        "import sys\n"
        "from assistant.tts import TTSManager\n"
        "TTSManager(store=None)\n"
        "print('torch' in sys.modules)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert completed.stdout.strip().splitlines()[-1] == "False"


def _provider(tmp_path: Path, **kwargs) -> ChatterboxLocalProvider:
    runtime_python = tmp_path / "python.exe"
    runner_path = tmp_path / "chatterbox_runner.py"
    runtime_python.write_text("", encoding="utf-8")
    runner_path.write_text("", encoding="utf-8")
    return ChatterboxLocalProvider(
        env={},  # no ECHO_TTS_CHATTERBOX_DEVICE -> "auto"
        runtime_python=runtime_python,
        runner_path=runner_path,
        **kwargs,
    )


def test_availability_probe_does_not_resolve_device_in_process(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(ChatterboxLocalProvider, "_select_device", lambda self: calls.append("x") or "cpu")
    commands: list[list[str]] = []

    def runner(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout='{"ok": true, "device": "cpu"}', stderr="")

    provider = _provider(tmp_path, command_runner=runner)
    assert provider.is_available() is True
    assert provider.diagnostics()["device"] == "auto"
    assert calls == []
    # The probe only checks dependencies: an explicit device keeps the
    # runner from importing torch to resolve "auto".
    assert commands[0][-2:] == ["--device", "cpu"]


def test_device_is_resolved_once_when_chatterbox_is_used(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(ChatterboxLocalProvider, "_select_device", lambda self: calls.append("x") or "cuda")
    seen: list[str] = []

    class FakeServer:
        def __init__(self, *, device, **_kwargs) -> None:
            seen.append(device)

        def start(self) -> dict:
            return {"ok": True, "model_load_ms": 1.0}

        def is_alive(self) -> bool:
            return False

    monkeypatch.setattr("assistant.tts._ChatterboxServerProcess", FakeServer)
    provider = _provider(tmp_path)
    assert calls == []
    provider.warmup()
    provider.warmup()
    assert seen == ["cuda", "cuda"]
    assert calls == ["x"]
    assert provider.diagnostics()["device"] == "cuda"

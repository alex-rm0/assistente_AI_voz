from __future__ import annotations

import sys
from types import SimpleNamespace

from assistant import audio_device


class FakeInputStream:
    def __init__(self, module, device=None, **kwargs) -> None:
        self.module = module
        self.device = device

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self, blocksize):
        level = self.module.levels.get(self.device, self.module.levels.get(None, 0.0))
        return [[level] for _ in range(max(1, min(blocksize, 8)))], False


class FakeSoundDevice:
    def __init__(self) -> None:
        self.default = SimpleNamespace(device=[0, None])
        self.levels = {0: 0.0, 1: 0.02, 2: 0.005, None: 0.0}
        self.devices = [
            {"name": "Default Mic", "max_input_channels": 1, "default_samplerate": 44100},
            {"name": "USB Active Mic", "max_input_channels": 2, "default_samplerate": 48000},
            {"name": "Quiet Mic", "max_input_channels": 1, "default_samplerate": 44100},
            {"name": "Speaker", "max_input_channels": 0, "default_samplerate": 44100},
        ]

    def query_devices(self, index=None, kind=None):
        if index is not None:
            return self.devices[int(index)]
        if kind == "input":
            return self.devices[self.default.device[0]]
        return self.devices

    def InputStream(self, **kwargs):
        return FakeInputStream(self, **kwargs)


def install_fake_sounddevice(monkeypatch):
    fake = FakeSoundDevice()
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    return fake


def test_list_input_devices(monkeypatch) -> None:
    install_fake_sounddevice(monkeypatch)

    devices = audio_device.list_input_devices()

    assert [device["name"] for device in devices] == ["Default Mic", "USB Active Mic", "Quiet Mic"]
    assert devices[1]["index"] == 1
    assert devices[1]["channels"] == 2
    assert devices[1]["sample_rate"] == 48000


def test_resolve_input_device(monkeypatch) -> None:
    install_fake_sounddevice(monkeypatch)

    assert audio_device.resolve_input_device("default") == 0
    assert audio_device.resolve_input_device("1") == 1
    assert audio_device.resolve_input_device("active") == 1
    assert audio_device.resolve_input_device("nao existe") is None


def test_probe_microphone_level(monkeypatch) -> None:
    install_fake_sounddevice(monkeypatch)

    level = audio_device.probe_microphone_level(1, duration=0.05, sample_rate=100, channels=1)

    assert level > 0.015


def test_choose_best_input_device_suggests_active_when_default_is_silent(monkeypatch) -> None:
    install_fake_sounddevice(monkeypatch)

    result = audio_device.choose_best_input_device(
        input_device="default",
        auto_select=True,
        silent_rms_threshold=0.001,
        sample_rate=100,
        probe_duration=0.05,
    )

    assert result.device_index == 1
    assert result.device_name == "USB Active Mic"
    assert result.silent is False
    assert "predefinido parece silencioso" in result.message


def test_choose_best_input_device_respects_explicit_device_without_auto_select(monkeypatch) -> None:
    install_fake_sounddevice(monkeypatch)

    result = audio_device.choose_best_input_device(
        input_device="default",
        auto_select=False,
        silent_rms_threshold=0.001,
        sample_rate=100,
        probe_duration=0.05,
    )

    assert result.device_index == 0
    assert result.silent is True


def test_microphone_test_report_contains_rms_and_suggestion(monkeypatch) -> None:
    install_fake_sounddevice(monkeypatch)

    report = audio_device.microphone_test_report(
        input_device="default",
        auto_select=True,
        silent_rms_threshold=0.001,
        sample_rate=100,
        probe_duration=0.05,
    )

    assert "Microfone usado: USB Active Mic" in report
    assert "Nivel RMS:" in report
    assert "Duracao gravada/testada:" in report
    assert "Sample rate: 100 Hz" in report
    assert "Estado: normal" in report

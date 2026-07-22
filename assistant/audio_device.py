from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any


DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CHANNELS = 1
DEFAULT_PROBE_DURATION_SECONDS = 0.5
DEFAULT_SILENT_RMS_THRESHOLD = 0.001


@dataclass(frozen=True)
class MicrophoneProbeResult:
    device_index: int | None
    device_name: str
    rms: float
    silent: bool
    message: str
    suggested_device_index: int | None = None
    suggested_device_name: str = ""
    suggested_rms: float = 0.0


def classify_rms_level(rms: float, silent_threshold: float = DEFAULT_SILENT_RMS_THRESHOLD) -> str:
    """Classify an RMS level into a simple user-facing audio state."""

    if rms < silent_threshold:
        return "silencioso"
    if rms < silent_threshold * 8:
        return "baixo"
    if rms > 0.85:
        return "saturado"
    return "normal"


def list_input_devices() -> list[dict[str, Any]]:
    """Return available input devices without raising when PortAudio is unavailable."""

    try:
        sd = _sounddevice()
        devices = sd.query_devices()
    except Exception:
        return []

    inputs: list[dict[str, Any]] = []
    for index, device in enumerate(devices):
        try:
            channels = int(device.get("max_input_channels", 0))
        except Exception:
            channels = 0
        if channels < 1:
            continue
        inputs.append(
            {
                "index": index,
                "name": str(device.get("name", f"Microfone {index}")),
                "channels": channels,
                "sample_rate": int(float(device.get("default_samplerate", DEFAULT_SAMPLE_RATE) or DEFAULT_SAMPLE_RATE)),
            }
        )
    return inputs


def resolve_input_device(spec: str | int | None = "default") -> int | None:
    """Resolve default, numeric index or name fragment to a sounddevice input index."""

    if spec is None:
        return None

    text = str(spec).strip()
    if not text or text.lower() == "default":
        try:
            default = _sounddevice().default.device[0]
        except Exception:
            return None
        return int(default) if default is not None and int(default) >= 0 else None

    if text.isdigit():
        index = int(text)
        for device in list_input_devices():
            if int(device["index"]) == index:
                return index
        return None

    needle = _normalize(text)
    for device in list_input_devices():
        if needle in _normalize(str(device["name"])):
            return int(device["index"])
    return None


def probe_microphone_level(
    device: str | int | None = None,
    duration: float = DEFAULT_PROBE_DURATION_SECONDS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
) -> float:
    """Measure peak RMS from a microphone for a short duration."""

    sd = _sounddevice()
    device_index = resolve_input_device(device)
    blocksize = max(1, int(sample_rate * 0.04))
    deadline = time.monotonic() + max(0.05, float(duration))
    peak = 0.0
    with sd.InputStream(
        device=device_index,
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        blocksize=blocksize,
    ) as stream:
        while time.monotonic() < deadline:
            data, _overflowed = stream.read(blocksize)
            peak = max(peak, _rms(data))
    return peak


def is_microphone_silent(
    device: str | int | None = None,
    threshold: float = DEFAULT_SILENT_RMS_THRESHOLD,
    duration: float = DEFAULT_PROBE_DURATION_SECONDS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
) -> bool:
    try:
        return probe_microphone_level(device, duration, sample_rate, channels) < threshold
    except Exception:
        return True


def choose_best_input_device(
    input_device: str | int | None = "default",
    auto_select: bool = True,
    silent_rms_threshold: float = DEFAULT_SILENT_RMS_THRESHOLD,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    probe_duration: float = DEFAULT_PROBE_DURATION_SECONDS,
) -> MicrophoneProbeResult:
    """Prefer the configured/default mic; if silent and enabled, scan alternatives."""

    requested_index = resolve_input_device(input_device)
    requested_name = _device_name(requested_index)
    requested_rms = _safe_probe(requested_index, probe_duration, sample_rate, channels)
    requested_active = requested_rms is not None and requested_rms >= silent_rms_threshold

    if requested_active or not auto_select:
        rms = float(requested_rms or 0.0)
        return MicrophoneProbeResult(
            device_index=requested_index,
            device_name=requested_name,
            rms=rms,
            silent=rms < silent_rms_threshold,
            message=_microphone_message(requested_name, rms, silent_rms_threshold),
        )

    best_index = requested_index
    best_name = requested_name
    best_rms = float(requested_rms or 0.0)
    for device in list_input_devices():
        index = int(device["index"])
        if requested_index is not None and index == requested_index:
            continue
        rms = _safe_probe(index, probe_duration, sample_rate, channels)
        if rms is not None and rms > best_rms:
            best_index = index
            best_name = str(device["name"])
            best_rms = float(rms)

    if best_index is not None and best_rms >= silent_rms_threshold:
        return MicrophoneProbeResult(
            device_index=best_index,
            device_name=best_name,
            rms=best_rms,
            silent=False,
            message=(
                f"Microfone ativo: {best_name} (RMS {best_rms:.5f}). "
                f"O microfone predefinido parece silencioso; sugiro usar este."
            ),
            suggested_device_index=best_index,
            suggested_device_name=best_name,
            suggested_rms=best_rms,
        )

    return MicrophoneProbeResult(
        device_index=requested_index,
        device_name=requested_name,
        rms=float(requested_rms or 0.0),
        silent=True,
        message=(
            "Nao encontrei um microfone ativo. "
            f"Ultimo nivel medido: RMS {float(requested_rms or 0.0):.5f}."
        ),
    )


def microphone_test_report(
    input_device: str | int | None = "default",
    auto_select: bool = True,
    silent_rms_threshold: float = DEFAULT_SILENT_RMS_THRESHOLD,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    probe_duration: float = DEFAULT_PROBE_DURATION_SECONDS,
) -> str:
    result = choose_best_input_device(
        input_device=input_device,
        auto_select=auto_select,
        silent_rms_threshold=silent_rms_threshold,
        sample_rate=sample_rate,
        channels=channels,
        probe_duration=probe_duration,
    )
    state = classify_rms_level(result.rms, silent_rms_threshold)
    lines = [
        f"Microfone usado: {result.device_name}",
        f"Indice: {result.device_index if result.device_index is not None else 'default'}",
        f"Nivel RMS: {result.rms:.5f}",
        f"Duracao gravada/testada: {float(probe_duration):.2f}s",
        f"Sample rate: {sample_rate} Hz",
        f"Canais: {channels}",
        f"Estado: {state}",
        result.message,
    ]
    if result.suggested_device_name and result.suggested_device_index != result.device_index:
        lines.append(
            f"Sugestao: {result.suggested_device_name} "
            f"(indice {result.suggested_device_index}, RMS {result.suggested_rms:.5f})."
        )
    return "\n".join(lines)


def _safe_probe(device_index: int | None, duration: float, sample_rate: int, channels: int) -> float | None:
    try:
        return probe_microphone_level(device_index, duration, sample_rate, channels)
    except Exception:
        return None


def _device_name(index: int | None) -> str:
    if index is None:
        return "microfone predefinido"
    for device in list_input_devices():
        if int(device["index"]) == index:
            return str(device["name"])
    return f"microfone {index}"


def _microphone_message(name: str, rms: float, threshold: float) -> str:
    if rms >= threshold:
        return f"Microfone ativo: {name} (RMS {rms:.5f})."
    return f"Microfone silencioso: {name} (RMS {rms:.5f})."


def _rms(data: Any) -> float:
    try:
        import numpy as np

        arr = np.asarray(data, dtype=float)
        if arr.size == 0:
            return 0.0
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        return float(np.sqrt(np.mean(arr**2)))
    except Exception:
        values = _flatten_numeric(data)
        if not values:
            return 0.0
        return math.sqrt(sum(value * value for value in values) / len(values))


def _flatten_numeric(data: Any) -> list[float]:
    if isinstance(data, (int, float)):
        return [float(data)]
    values: list[float] = []
    try:
        iterator = iter(data)
    except TypeError:
        return values
    for item in iterator:
        if isinstance(item, (list, tuple)):
            values.extend(_flatten_numeric(item))
        else:
            try:
                values.append(float(item))
            except (TypeError, ValueError):
                pass
    return values


def _sounddevice():
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("A biblioteca sounddevice nao esta instalada.") from exc
    return sd


def _normalize(text: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))

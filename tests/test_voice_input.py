from __future__ import annotations

from assistant.voice_input import voice_status_report


def test_voice_status_report_ready() -> None:
    result = voice_status_report(
        enabled=True,
        missing_dependencies=[],
        microphone_ok=True,
        microphone_message="Microfone pronto.",
        input_device="default",
        auto_select_input=True,
        model_name="base",
        language="pt",
    )

    assert "whisper: pronto" in result
    assert "ffmpeg: pronto" in result
    assert "voice.enabled: True" in result
    assert "input_device: default" in result
    assert "auto_select_input: True" in result
    assert "modelo whisper: base" in result
    assert "idioma configurado: pt" in result
    assert "A voz esta pronta para uso." in result


def test_voice_status_report_missing_dependencies() -> None:
    result = voice_status_report(
        enabled=True,
        missing_dependencies=["openai-whisper", "ffmpeg"],
    )

    assert "whisper: em falta" in result
    assert "ffmpeg: em falta" in result
    assert "faltam dependencias locais" in result

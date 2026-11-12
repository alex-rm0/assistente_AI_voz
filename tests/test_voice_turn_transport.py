from assistant.turn_input import record_stt_evidence


def test_worker_emits_capture_owned_confidence():
    from prototype_web_ui.controller import EchoVoiceWorker
    from tests.test_web_ui_prototype import _qt_app
    _qt_app()
    class Transcriber:
        def record_and_transcribe(self, *args):
            record_stt_evidence(model="turbo", avg_logprob=-.35, no_speech_probability=.01, compression_ratio=1.1)
            return " hello "
    worker = EchoVoiceWorker(Transcriber(), session_id="session-1")
    captures = []
    worker.transcriptionReady.connect(captures.append)
    worker.run()
    assert captures[0].text == "hello" and captures[0].avg_logprob == -.35
    assert captures[0].session_id == "session-1"


def test_late_capture_cannot_be_delivered_as_the_next_capture():
    from types import SimpleNamespace
    from assistant.turn_input import TurnInput
    from prototype_web_ui.controller import EchoUIController
    from tests.test_web_ui_prototype import _qt_app
    _qt_app()
    controller = EchoUIController(lambda _: "ok")
    delivered = []
    controller.voiceTranscriptionReady.connect(delivered.append)
    controller._voice_worker = SimpleNamespace(capture_id="new-capture")
    try:
        old = TurnInput(text="old text", source="voice", capture_id="old-capture", session_id="0", capture_mode="manual")
        controller._handle_voice_transcription(old)
        assert delivered == []
    finally:
        controller._voice_worker = None
        controller.shutdown()

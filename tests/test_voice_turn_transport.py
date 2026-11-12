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


def test_qthread_controller_engine_preserves_capture_then_creates_clean_typed_turn(tmp_path, monkeypatch):
    from assistant.turn_input import current_turn_input
    from prototype_web_ui.controller import EchoUIController
    from tests.test_automatic_routing_regressions import make_engine
    from tests.test_web_ui_prototype import _qt_app, _run_until
    _qt_app()
    engine, *_ = make_engine(tmp_path, mode="local")
    received = []
    monkeypatch.setattr(engine, "_respond_inner", lambda text: received.append(current_turn_input()) or "ok")
    class Transcriber:
        def record_and_transcribe_vad(self, *a, **kw):
            record_stt_evidence(model="turbo", avg_logprob=-.41, no_speech_probability=.03, compression_ratio=.9)
            return "hello"
    controller = EchoUIController(engine.respond, voice_available=True, voice_transcriber=Transcriber(),
                                  voice_missing_dependencies=[], thinking_min_ms=0)
    controller._voice_conversation_active = True
    controller._voice_session_id = 4
    controller._schedule_voice_rearm = lambda **kw: None
    try:
        controller._begin_listening()
        assert _run_until(lambda: len(received) == 1 and controller._worker is None and controller._voice_worker is None)
        capture = received[0]
        assert capture.text == "hello"
        assert capture.source == "voice" and capture.avg_logprob == -.41
        assert capture.session_id == "4" and capture.capture_id
        controller._voice_conversation_active = False
        controller.submitMessage(" typed after voice ")
        assert _run_until(lambda: len(received) == 2 and controller._worker is None)
        assert received[1].source == "typed" and received[1].avg_logprob is None
        assert received[1].text == "typed after voice"
        assert received[1].capture_id is None
    finally:
        controller.shutdown()


def test_manual_review_does_not_attach_confidence_to_later_typed_submission(tmp_path, monkeypatch):
    from assistant.turn_input import capture_transcription, current_turn_input
    from prototype_web_ui.controller import EchoUIController
    from tests.test_automatic_routing_regressions import make_engine
    from tests.test_web_ui_prototype import _qt_app, _run_until
    _qt_app()
    engine, *_ = make_engine(tmp_path, mode="local")
    received = []
    monkeypatch.setattr(engine, "_respond_inner", lambda _: received.append(current_turn_input()) or "ok")
    controller = EchoUIController(engine.respond)
    def record():
        record_stt_evidence(model="turbo", avg_logprob=-.4)
        return "review this text"
    try:
        controller._handle_voice_transcription(capture_transcription(record))
        assert not received  # MIC still only populates the editable input.
        controller.submitMessage("review this text")
        assert _run_until(lambda: len(received) == 1 and controller._worker is None)
        assert received[0].source == "typed" and received[0].avg_logprob is None
    finally:
        controller.shutdown()

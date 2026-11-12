import pytest

from assistant.speech_clarity import evaluate_speech_clarity, protection_for_turn
from assistant.turn_input import TurnInput


def voice(text, **confidence):
    return TurnInput(text=text, source="voice", capture_mode="active", capture_id="capture", session_id="session", **confidence)


@pytest.mark.parametrize("text,confidence", [
    ("Boa, boa. Queria escrever o mail contigo. Ajudas-me?", -.30688),
    ("Francisca e Karina", -.74510),
    ("Podes escrever o primeiro rascunho, mas o nome da Karina está mal.", -.22159),
    ("Consegues trocar o nome da Karina e o K por um C?", -.27405),
    ("Primeiro guarda no workspace, não fecheira TXT o email, por favor.", -.49697),
    ("Tia, como esta?", -.80690),
    ("Querias que eu ver um mal contigo, mas jostas-me.", None),
])
def test_real_diagnostic_examples_do_not_trigger(text, confidence):
    assert not evaluate_speech_clarity(voice(text, avg_logprob=confidence)).would_ask_repeat


@pytest.mark.parametrize("text", ["Can you the of for the meeting with", "A de para do reunião com"])
def test_two_strong_text_signals_trigger_without_acoustic_metadata(text):
    decision = evaluate_speech_clarity(voice(text))
    assert decision.would_ask_repeat
    assert {"dangling_tail", "function_word_chain"} <= set(decision.signals)


def test_one_text_signal_needs_real_suspicious_confidence():
    text = "Podes dizer a hora do"
    assert not evaluate_speech_clarity(voice(text)).would_ask_repeat
    assert not evaluate_speech_clarity(voice(text, avg_logprob=-.3)).would_ask_repeat
    assert evaluate_speech_clarity(voice(text, avg_logprob=-1.05)).would_ask_repeat
    assert evaluate_speech_clarity(voice(text, no_speech_probability=.6)).would_ask_repeat


def test_confidence_alone_and_typed_input_never_trigger():
    assert not evaluate_speech_clarity(voice("Francisca e Karina", avg_logprob=-2)).would_ask_repeat
    assert not evaluate_speech_clarity(TurnInput.typed("the of for the with")).would_ask_repeat
    assert not evaluate_speech_clarity(voice("Podes dizer a hora do", avg_logprob=float("nan"))).would_ask_repeat


def test_protected_intent_does_not_trigger_and_quotes_are_not_analyzed():
    broken = voice("Can you the of for the meeting with", avg_logprob=-1)
    decision = evaluate_speech_clarity(broken, protected=("canonical_intent",))
    assert not decision.would_ask_repeat and decision.protected == ("canonical_intent",)
    assert not evaluate_speech_clarity(voice('Explain "the of for the meeting with".')).would_ask_repeat
    assert not evaluate_speech_clarity(voice("Preciso de... deixa-me pensar... rever o documento amanhã.")).would_ask_repeat


def test_protection_uses_canonical_commands_and_current_field(tmp_path):
    from tests.test_automatic_routing_regressions import make_engine
    engine, *_ = make_engine(tmp_path, mode="local")
    for text in ("abre o youtube", "acorda", "não", "executa powershell", "vamos comparar Samsung com Apple"):
        assert protection_for_turn(voice(text), engine)
    engine.respond("vamos escrever um mail")
    assert protection_for_turn(voice("Francisca e Karina", avg_logprob=-1.1), engine) == ("recipient_field",)
    assert not protection_for_turn(voice("A de para do reunião com"), engine)


def test_shadow_preserves_response_path_and_llm_payload(tmp_path, capsys):
    from tests.test_automatic_routing_regressions import make_engine
    text = "A de para do reunião com"
    typed, _, typed_llm, _ = make_engine(tmp_path / "typed", mode="local", ollama_replies=["Não percebi. Podes repetir?"])
    spoken, _, voice_llm, _ = make_engine(tmp_path / "voice", mode="local", ollama_replies=["Não percebi. Podes repetir?"])
    expected = typed.respond(text)
    actual = spoken.respond_turn(voice(text, avg_logprob=-1))
    assert actual == expected
    assert spoken.get_last_turn_telemetry()["selected_path"] == typed.get_last_turn_telemetry()["selected_path"]
    assert voice_llm.calls == typed_llm.calls
    assert "[SPEECH_CLARITY]" in capsys.readouterr().out


@pytest.mark.parametrize("language,expected", [
    ("en-GB", "Sorry, I didn't quite catch that — could you say it again?"),
    ("pt-PT", "Desculpa, não percebi bem — podes repetir?"),
])
def test_enforcement_is_explicit_and_local_without_memory_writes(tmp_path, language, expected):
    from tests.test_automatic_routing_regressions import make_engine
    engine, _, local, _ = make_engine(tmp_path, mode="local")
    engine._set_response_language(language)
    engine.speech_clarity_enforce = True
    before = list(engine.memory.load())
    assert engine.respond_turn(voice("A de para do reunião com")) == expected
    assert local.calls == []
    assert list(engine.memory.load()) == before
    assert engine.get_last_turn_telemetry()["llm_calls"] == 0


def test_typed_does_not_log_or_enforce_clarity(tmp_path, capsys):
    from tests.test_automatic_routing_regressions import make_engine
    engine, *_ = make_engine(tmp_path, mode="local")
    engine.speech_clarity_enforce = True
    engine.respond("A de para do reunião com")
    assert engine.get_last_turn_telemetry()["selected_path"] != "SPEECH_CLARITY"
    assert "[SPEECH_CLARITY]" not in capsys.readouterr().out

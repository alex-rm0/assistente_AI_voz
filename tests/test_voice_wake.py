from __future__ import annotations

from assistant.voice_wake import classify_voice_wake_intent, wake_acknowledgement_text


def test_wake_c_portuguese_wake_commands_work() -> None:
    for phrase in ("Echo", "Echo, acorda", "acorda", "podes ouvir-me?", "começa a ouvir", "Podes ouvir-me"):
        assert classify_voice_wake_intent(phrase) == "wake", phrase


def test_wake_d_english_wake_commands_work() -> None:
    for phrase in ("Echo", "Hey Echo", "wake up", "start listening"):
        assert classify_voice_wake_intent(phrase) == "wake", phrase


def test_wake_sleep_commands_recognized_both_languages() -> None:
    for phrase in ("vai dormir", "podes parar de ouvir", "para de ouvir", "go to sleep", "stop listening"):
        assert classify_voice_wake_intent(phrase) == "sleep", phrase


def test_wake_sleep_f_real_failure_phrases_now_recognized() -> None:
    """Phase 7.6.5 Part A (section 1): the exact real STANDBY->ACTIVE sleep
    failure -- "Não, pode ir dormir." and the short STT variant "Dorm."
    both reached the normal conversation pipeline instead of being
    intercepted, because the parser only knew "vai dormir", not any of
    these natural variants."""
    for phrase in (
        "Não, pode ir dormir.",
        "Dorm.",
        "ok, podes ir dormir",
        "pronto, vai dormir",
        "that's all, go to sleep",
        "pode ir dormir",
        "podes ir dormir",
        "pode dormir",
        "podes dormir",
        "dorme",
        "dormir",
        "sleep",
        "you can go to sleep",
        "you can stop listening",
        "já podes parar",
        "pode descansar",
    ):
        assert classify_voice_wake_intent(phrase) == "sleep", phrase


def test_wake_sleep_g_general_mentions_of_sleep_are_not_commands() -> None:
    """Phase 7.6.5 Part A (section 4): false-positive protection -- a bare
    mention of sleep/dormir inside an unrelated sentence must never be
    misread as a command to Echo, even though the same bare words ARE
    recognized as commands on their own (section 3's "dorme"/"dormir"/
    "sleep")."""
    for phrase in (
        "ontem fui dormir tarde",
        "acho que devia dormir mais",
        "o computador entrou em sleep",
        "ele pode ir dormir mais cedo hoje se quiser",
    ):
        assert classify_voice_wake_intent(phrase) == "none", phrase


def test_wake_e_ordinary_speech_does_not_match_wake_or_sleep() -> None:
    for phrase in (
        "que horas são?",
        "What time is it?",
        "levantei-me e acordei tarde",
        "não consegui dormir bem esta noite",
        "podes ajudar-me com uma coisa?",
    ):
        assert classify_voice_wake_intent(phrase) == "none", phrase


def test_wake_classifier_is_case_and_punctuation_insensitive() -> None:
    assert classify_voice_wake_intent("ECHO, ACORDA!!") == "wake"
    assert classify_voice_wake_intent("  vai   dormir...  ") == "sleep"


def test_wake_classifier_handles_empty_or_none_input() -> None:
    assert classify_voice_wake_intent("") == "none"
    assert classify_voice_wake_intent("   ") == "none"


def test_wake_acknowledgement_text_is_deterministic_and_language_specific() -> None:
    assert wake_acknowledgement_text("en-GB") == "I'm listening."
    assert wake_acknowledgement_text("pt-PT") == "Estou a ouvir."
    assert wake_acknowledgement_text("") == "I'm listening."

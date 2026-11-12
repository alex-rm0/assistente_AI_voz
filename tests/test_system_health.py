"""R3 step 4, point 1: "how are you?" is answered from measured component
health, in a short Jarvis-style line, without the LLM. A component that was
not measured is never described.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.system_health import (
    DEGRADED,
    HEALTHY,
    UNAVAILABLE,
    UNKNOWN,
    SystemHealth,
    is_wellbeing_question,
    wellbeing_reply,
)
from assistant.tool_registry import ToolRegistry

ALL_OK = {"ollama": HEALTHY, "stt": HEALTHY, "tts": HEALTHY, "network": HEALTHY}
# Words that describe system state. None may appear when nothing was measured.
STATE_WORDS = ("system", "sistema", "online", "running", "funcionar", "operaciona", "order", "ordem", "model", "voice", "voz", "network", "rede")


def _health(states: dict[str, str]) -> SystemHealth:
    return SystemHealth({name: (lambda value=value: value) for name, value in states.items()})


def _replies(message: str, language: str, states: dict[str, str] | None, count: int = 8) -> list[str]:
    health = _health(states) if states is not None else None
    return [wellbeing_reply(message, language, health, variant=index) for index in range(count)]


@pytest.mark.parametrize("message", [
    "how are you", "How are you?", "Hello, how are you today?", "Evening, Echo. All good?", "all good?",
    "como estás?", "Como é que estás?", "Bom dia, como é que estás?", "Olá, como está?", "tudo bem?", "Echo, estás bem?",
])
def test_wellbeing_questions_are_recognised(message: str) -> None:
    assert is_wellbeing_question(message)


@pytest.mark.parametrize("message", [
    "como está o projeto?", "how are you getting on with the report?", "Bom dia", "tudo bem, vamos continuar o email",
    "como estás a pensar resolver isto?", "all good, thanks", "Porquê que dizes sempre parâmetros normais?",
])
def test_other_messages_are_not_wellbeing_questions(message: str) -> None:
    assert not is_wellbeing_question(message)
    assert wellbeing_reply(message, "en-GB", _health(ALL_OK)) is None


def test_all_measured_and_healthy_says_so_in_several_ways() -> None:
    english = _replies("how are you?", "en-GB", ALL_OK)
    portuguese = _replies("como estás?", "pt-PT", ALL_OK)

    assert "All systems running smoothly, Sir." in english
    assert len(set(english)) >= 3 and len(set(portuguese)) >= 3
    assert english[0] != english[1] and portuguese[0] != portuguese[1]  # consecutive replies differ
    assert all("system" in reply.lower() or "order" in reply.lower() for reply in english)
    assert all("sistema" in reply.lower() or "ordem" in reply.lower() for reply in portuguese)
    assert sum("Sir" in reply for reply in english) <= len(english) // 2
    assert not any("Sir" in reply for reply in portuguese)
    assert all(len(reply) <= 90 and "?" not in reply for reply in english + portuguese)


def test_slow_voice_recognition_is_named() -> None:
    states = dict(ALL_OK, stt=DEGRADED)

    for reply in _replies("how are you?", "en-GB", states):
        assert "voice recognition is a touch slow" in reply
        assert "all systems" not in reply.lower() and "smoothly" not in reply.lower()
    assert "Everything else is in order — voice recognition is a touch slow today, I'm afraid." in _replies("how are you?", "en-GB", states)
    for reply in _replies("como estás?", "pt-PT", states):
        assert "reconhecimento de voz" in reply and "lento" in reply


def test_unavailable_components_are_named_and_never_called_fine() -> None:
    states = dict(ALL_OK, ollama=UNAVAILABLE, network=UNAVAILABLE)

    for reply in _replies("how are you?", "en-GB", states):
        assert "language model" in reply and "network" in reply
        assert "all systems" not in reply.lower()
    for reply in _replies("como estás?", "pt-PT", states):
        assert "modelo de linguagem" in reply and "rede" in reply


@pytest.mark.parametrize("states", [None, {}, {"ollama": UNKNOWN, "stt": UNKNOWN}])
def test_nothing_measured_claims_no_system_state(states) -> None:
    for language, message in (("en-GB", "how are you?"), ("pt-PT", "como estás?")):
        for reply in _replies(message, language, states):
            assert reply and not any(word in reply.lower() for word in STATE_WORDS), reply


def test_partial_measurement_names_only_what_was_measured() -> None:
    states = {"ollama": HEALTHY, "stt": UNKNOWN}

    for reply in _replies("how are you?", "en-GB", states):
        lowered = reply.lower()
        assert "language model" in lowered
        assert "all systems" not in lowered and "everything" not in lowered
        assert "voice" not in lowered and "network" not in lowered and "speech" not in lowered
    # One problem among partly measured components: no claim about the rest.
    for reply in _replies("how are you?", "en-GB", {"stt": DEGRADED}):
        assert "voice recognition" in reply and "everything" not in reply.lower() and "order" not in reply.lower()


def test_greeting_is_returned_in_the_response_language() -> None:
    assert wellbeing_reply("Bom dia, como é que estás?", "en-GB", _health(ALL_OK), variant=0).startswith("Good morning. ")
    assert wellbeing_reply("Bom dia, como é que estás?", "pt-PT", _health(ALL_OK), variant=0).startswith("Bom dia. ")
    assert wellbeing_reply("Evening, Echo. All good?", "en-GB", _health(ALL_OK), variant=0).startswith("Good evening. ")


def test_a_failing_probe_is_unknown_and_results_are_cached() -> None:
    calls = {"ollama": 0}
    now = [0.0]

    def ollama() -> str:
        calls["ollama"] += 1
        return HEALTHY

    def broken() -> str:
        raise OSError("probe failed")

    health = SystemHealth({"ollama": ollama, "tts": broken, "stt": lambda: "nonsense"}, ttl_seconds=30, clock=lambda: now[0])

    assert health.snapshot() == {"ollama": HEALTHY, "tts": UNKNOWN, "stt": UNKNOWN}
    health.snapshot()
    assert calls["ollama"] == 1
    now[0] = 31.0
    health.snapshot()
    assert calls["ollama"] == 2


# --- through the real engine: no LLM call, real route --------------------------


class _CountingLLM:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, user_message, history=None, system_prompt=None, **kwargs):
        self.calls += 1
        return "I'm functioning within normal parameters."

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        return {"tool": None, "arguments": {}, "reason": "none"}

    def embed(self, text: str):
        return None


def _engine(tmp_path: Path, language: str) -> tuple[AssistantEngine, _CountingLLM]:
    data = tmp_path / "data"
    (tmp_path / "workspace").mkdir(exist_ok=True)
    llm = _CountingLLM()
    memory = LongTermMemory(data, embedder=None)
    memory.set_preference("response_language", language)
    engine = AssistantEngine(
        llm=llm,
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=memory,
        tools=ToolRegistry(),
        workspace_path=tmp_path / "workspace",
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
    )
    return engine, llm


@pytest.mark.parametrize("language,message,expected", [
    ("en-GB", "Bom dia, como é que estás?", "voice recognition is a touch slow"),
    ("pt-PT", "Bom dia, como é que estás?", "reconhecimento de voz"),
    ("en-GB", "Evening, Echo. All good?", "voice recognition is a touch slow"),
])
def test_engine_answers_from_measured_health_without_the_llm(tmp_path, language, message, expected) -> None:
    engine, llm = _engine(tmp_path, language)
    engine.system_health = _health(dict(ALL_OK, stt=DEGRADED))

    first = engine.respond(message)
    second = engine.respond(message)

    assert expected in first and expected in second
    assert first != second
    assert llm.calls == 0
    assert engine.get_last_turn_telemetry()["selected_path"] == "SOCIAL_PATH"


def test_engine_without_health_gives_a_neutral_line_without_the_llm(tmp_path) -> None:
    engine, llm = _engine(tmp_path, "en-GB")

    reply = engine.respond("Bom dia, como é que estás?")

    assert reply.startswith("Good morning. ")
    assert not any(word in reply.lower() for word in STATE_WORDS), reply
    assert llm.calls == 0

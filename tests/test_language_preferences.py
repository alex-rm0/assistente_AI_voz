from __future__ import annotations

from pathlib import Path

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.tool_registry import ToolRegistry
from assistant.model_runtime import UserSettingsStore
from assistant.tts import DEFAULT_ENGLISH_VOICE_ID, DEFAULT_PORTUGUESE_VOICE_ID, VoiceCommandResult


class FakeLLM:
    def __init__(self) -> None:
        self.last_system_prompt = ""
        self.chat_calls = 0

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def chat(self, user_message, history=None, system_prompt=None, response_format=None, **_kwargs):
        self.chat_calls += 1
        self.last_system_prompt = system_prompt or ""
        return "resposta"

    def embed(self, text: str):
        return None


class FakeRuntime:
    def __init__(self, store: UserSettingsStore) -> None:
        self.store = store


class FakeTTSManager:
    def __init__(self, store: UserSettingsStore, *, portuguese_available: bool = True) -> None:
        self.store = store
        self.portuguese_available = portuguese_available
        self.set_calls: list[str] = []

    def set_active_voice(self, voice_id: str):
        self.set_calls.append(voice_id)
        self.store.update(active_voice_id=voice_id)
        if voice_id == DEFAULT_PORTUGUESE_VOICE_ID and not self.portuguese_available:
            raise RuntimeError("voz portuguesa indisponível")
        return object()

    def handle_voice_command(self, _message: str):
        return VoiceCommandResult(handled=False)

    def diagnostics(self):
        active = self.store.load().get("active_voice_id", "")
        return {
            "available": active == DEFAULT_ENGLISH_VOICE_ID or self.portuguese_available,
            "active_voice_id": active,
            "resolved_voice_name": active,
            "logical_slots": [],
        }


def make_engine(tmp_path: Path, *, tts_manager=None, store: UserSettingsStore | None = None) -> AssistantEngine:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    data = tmp_path / "data"
    llm = FakeLLM()
    store = store or UserSettingsStore(data / "user_settings.json")
    return AssistantEngine(
        llm=llm,
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=LongTermMemory(data, embedder=llm),
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
        tts_manager=tts_manager,
        model_runtime=FakeRuntime(store),
    )


def test_language_policy_answer_comes_from_internal_state(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    response = engine.respond("Qual é a tua língua base?")

    assert response == "A minha língua base é português de Portugal, mas posso falar inglês quando pedires."
    assert engine.llm.last_system_prompt == ""


def test_user_can_switch_current_language_preference(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    english_response = engine.respond("in English")

    assert english_response == "Sure. I'll respond in English."
    assert engine.response_language == "en-GB"
    assert engine.long_term_memory.get_preference("idioma_base") == "pt-PT"
    assert engine.long_term_memory.get_preference("idioma_atual") == "en"
    assert engine.get_last_turn_telemetry()["llm_calls"] == 0

    portuguese_response = engine.respond("Volta ao português de Portugal.")

    assert portuguese_response == "Claro. Vou responder em português."
    assert engine.response_language == "pt-PT"
    assert engine.long_term_memory.get_preference("idioma_atual") == "pt-PT"


def test_language_instruction_is_sent_to_llm(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    engine.respond("Explica-me rapidamente o que é uma API.")

    assert "idioma_base = pt-PT" in engine.llm.last_system_prompt
    assert "response_language = pt-PT" in engine.llm.last_system_prompt
    assert "aplicações" in engine.llm.last_system_prompt
    assert "ficheiros" in engine.llm.last_system_prompt


def test_response_language_commands_support_portuguese_variants(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    assert engine.respond("fala em inglês") == "Sure. I'll respond in English."
    assert engine.response_language == "en-GB"

    assert engine.respond("responde em português") == "Claro. Vou responder em português."
    assert engine.response_language == "pt-PT"


def test_response_language_preference_persists_in_safe_user_settings(tmp_path: Path) -> None:
    store = UserSettingsStore(tmp_path / "data" / "user_settings.json")
    first = make_engine(tmp_path, store=store)

    first.respond("switch to English")
    second = make_engine(tmp_path, store=store)

    assert second.response_language == "en-GB"
    assert store.load()["response_language"] == "en-GB"


def test_social_fast_path_respects_english_response_language(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    engine.respond("in English")
    response = engine.respond("olá, tudo bem?")

    assert response == "I'm good. How about you?"
    telemetry = engine.get_last_turn_telemetry()
    assert telemetry["selected_path"] == "SOCIAL_PATH"
    assert telemetry["llm_calls"] == 0


def test_social_fast_path_respects_portuguese_response_language(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    engine.respond("responde em português")
    response = engine.respond("how are you?")

    assert response == "Estou bem. E tu?"
    assert engine.get_last_turn_telemetry()["llm_calls"] == 0


def test_english_response_language_adds_high_priority_llm_constraint(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    engine.respond("in English")
    engine.respond("Explain briefly what an API is.")

    assert "High-priority response language" in engine.llm.last_system_prompt
    assert "Respond in English" in engine.llm.last_system_prompt


def test_language_switch_coordinates_active_voice_without_merging_preferences(tmp_path: Path) -> None:
    store = UserSettingsStore(tmp_path / "data" / "user_settings.json")
    tts = FakeTTSManager(store)
    engine = make_engine(tmp_path, store=store, tts_manager=tts)

    engine.respond("fala em inglês")
    engine.respond("volta para português")

    assert tts.set_calls == [DEFAULT_ENGLISH_VOICE_ID, DEFAULT_PORTUGUESE_VOICE_ID]
    assert store.load()["response_language"] == "pt-PT"
    assert store.load()["active_voice_id"] == DEFAULT_PORTUGUESE_VOICE_ID


def test_unavailable_ptpt_voice_does_not_block_portuguese_text_response(tmp_path: Path) -> None:
    store = UserSettingsStore(tmp_path / "data" / "user_settings.json")
    tts = FakeTTSManager(store, portuguese_available=False)
    engine = make_engine(tmp_path, store=store, tts_manager=tts)

    response = engine.respond("responde em português")

    assert response == "Claro. Vou responder em português."
    assert engine.response_language == "pt-PT"


def test_task_dialogue_respects_english_response_language(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    engine.respond("in English")
    response = engine.respond("vamos comparar Samsung com Apple")

    assert response == "Good, we have Samsung and Apple as options. They're already in the workspace."
    telemetry = engine.get_last_turn_telemetry()
    assert telemetry["selected_path"] == "TASK_START"
    assert telemetry["response_language"] == "en-GB"
    assert telemetry["response_language_localized"] is True
    assert telemetry["llm_calls"] == 0


def test_research_unavailable_response_respects_english_response_language(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    engine.respond("in English")
    response = engine.respond("pesquisa sobre Picasso")

    assert "research tool" in response.lower()
    assert "ferramenta de pesquisa" not in response.lower()
    telemetry = engine.get_last_turn_telemetry()
    assert telemetry["selected_path"] == "RESEARCH_REQUEST"
    assert telemetry["response_language"] == "en-GB"
    assert telemetry["llm_calls"] == 0

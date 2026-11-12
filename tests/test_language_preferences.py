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
from assistant.system_health import not_measured_replies


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
        self.stop_calls = 0

    def stop(self) -> None:
        self.stop_calls += 1

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

    assert any(response.startswith(line) for line in not_measured_replies("en-GB"))
    assert response.count("?") <= 1
    telemetry = engine.get_last_turn_telemetry()
    assert telemetry["selected_path"] == "SOCIAL_PATH"
    assert telemetry["llm_calls"] == 0


def test_english_response_language_is_authoritative_for_simple_hello(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    engine.respond("in English")
    response = engine.respond("hello")

    assert response == "Hi. How are you?"
    telemetry = engine.get_last_turn_telemetry()
    assert telemetry["selected_path"] == "SOCIAL_PATH"
    assert telemetry["response_language"] == "en-GB"
    assert telemetry["llm_calls"] == 0


def test_english_response_language_is_authoritative_for_portuguese_greeting(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    engine.respond("in English")
    response = engine.respond("olá")

    assert response == "Hi. How are you?"
    assert engine.get_last_turn_telemetry()["response_language"] == "en-GB"


def test_social_fast_path_respects_portuguese_response_language(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    engine.respond("responde em português")
    response = engine.respond("how are you?")

    assert any(response.startswith(line) for line in not_measured_replies("pt-PT"))
    assert response.count("?") <= 1
    assert engine.get_last_turn_telemetry()["llm_calls"] == 0


def test_portuguese_response_language_is_authoritative_for_simple_hello(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    engine.respond("responde em português")
    response = engine.respond("hello")

    assert response == "Olá! Como estás?"
    telemetry = engine.get_last_turn_telemetry()
    assert telemetry["selected_path"] == "SOCIAL_PATH"
    assert telemetry["response_language"] == "pt-PT"
    assert telemetry["llm_calls"] == 0


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


# --- Phase 7.3.4 (section 1/2/3): response_language authority + contextual
# "translate the previous response" transformation. Root cause of the
# reported regression: response_composer.py's _SYSTEM_PROMPT (and
# context_manager.py's system_prompt()) used to hardcode an unconditional
# "Fala em português de Portugal" persona line, which directly contradicted
# the correct, response_language-aware instruction appended right after it
# -- the model would often obey the earlier, more specific Portuguese line
# regardless of response_language. These tests assert the fix structurally
# (the competing line must never reappear) since a scripted FakeLLM cannot
# itself prove "the model obeys" -- that is validated with the real local
# LLM in the accompanying manual Qt run.

_OLD_HARDCODED_PORTUGUESE_MANDATE = "Fala em português de Portugal e trata o Alexandre por tu"


def test_english_response_language_authoritative_with_portuguese_input(tmp_path: Path) -> None:
    """A. en-GB + Portuguese input -> the system prompt sent to the LLM
    must carry the English instruction and never the old, competing,
    hardcoded Portuguese mandate -- regardless of what language the user
    typed in."""
    engine = make_engine(tmp_path)

    engine.respond("in English")
    engine.respond("estou bem, precisava de ajuda")

    assert "Respond in English" in engine.llm.last_system_prompt
    assert _OLD_HARDCODED_PORTUGUESE_MANDATE not in engine.llm.last_system_prompt


def test_english_response_language_stays_authoritative_across_multiple_turns(tmp_path: Path) -> None:
    """B. en-GB remains authoritative across several GENERAL_CONVERSATION
    turns in a row, all with Portuguese input."""
    engine = make_engine(tmp_path)
    engine.respond("in English")

    for message in (
        "estou bem, precisava de ajuda",
        "consegues escrever um parágrafo sobre o que é um eclipse total solar?",
        "e sobre a lua cheia, sabes alguma coisa?",
    ):
        engine.respond(message)
        assert "Respond in English" in engine.llm.last_system_prompt
        assert _OLD_HARDCODED_PORTUGUESE_MANDATE not in engine.llm.last_system_prompt
        assert engine.response_language == "en-GB"


def test_portuguese_response_language_authoritative_with_english_input(tmp_path: Path) -> None:
    """C. pt-PT (the default) + English input -> the system prompt must
    still carry the pt-PT instruction, and ComposerRequest.response_language
    (previously never populated, always stuck at the dataclass default)
    must reflect the real state."""
    engine = make_engine(tmp_path)

    engine.respond("How are you doing today, my friend?")

    assert "response_language = pt-PT" in engine.llm.last_system_prompt
    assert engine.response_language == "pt-PT"


class _TranslationAwareLLM:
    """A minimal FakeLLM that returns a DIFFERENT canned reply depending on
    whether the system prompt carries the one-off translation instruction
    -- just enough to prove _try_translate_previous_response actually wires
    the previous response, the translation instruction, and the target
    language through to a real composer call, without needing a real LLM."""

    def __init__(self) -> None:
        self.last_system_prompt = ""
        self.chat_calls = 0

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def chat(self, user_message, history=None, system_prompt=None, response_format=None, **_kwargs):
        self.chat_calls += 1
        self.last_system_prompt = system_prompt or ""
        if "Traduz o texto seguinte" in self.last_system_prompt:
            return "A total solar eclipse happens when the Moon fully blocks the Sun."
        return "Um eclipse solar total acontece quando a Lua bloqueia totalmente o Sol."

    def embed(self, text: str):
        return None


def _make_engine_with_translation_llm(tmp_path: Path) -> AssistantEngine:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    data = tmp_path / "data"
    llm = _TranslationAwareLLM()
    return AssistantEngine(
        llm=llm,
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=LongTermMemory(data, embedder=llm),
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
    )


def test_contextual_translate_previous_response_transforms_prior_reply(tmp_path: Path) -> None:
    """D. "quero essa resposta em inglês" must transform the PREVIOUS
    response into English, not merely acknowledge a persistent switch."""
    engine = _make_engine_with_translation_llm(tmp_path)
    engine.respond("explica-me o que é um eclipse solar total")

    response = engine.respond("quero essa resposta em inglês")

    assert response == "A total solar eclipse happens when the Moon fully blocks the Sun."
    assert response != "Sure. I'll respond in English."
    telemetry = engine.get_last_turn_telemetry()
    assert telemetry["selected_path"] == "TRANSLATE_PREVIOUS_RESPONSE"
    assert telemetry["translated_previous_response"] is True
    assert telemetry["translate_previous_target_language"] == "en-GB"


def test_contextual_say_that_in_english_transforms_prior_reply(tmp_path: Path) -> None:
    """E. "say that in English" is equivalent contextual behavior to
    "quero essa resposta em inglês"."""
    engine = _make_engine_with_translation_llm(tmp_path)
    engine.respond("explica-me o que é um eclipse solar total")

    response = engine.respond("say that in English")

    assert response == "A total solar eclipse happens when the Moon fully blocks the Sun."


def test_plain_in_english_still_sets_persistent_response_language(tmp_path: Path) -> None:
    """F. Plain "in English" (no deictic reference to a previous reply)
    must still go through the PERSISTENT language-switch path, not the new
    contextual-translation one."""
    engine = _make_engine_with_translation_llm(tmp_path)
    engine.respond("explica-me o que é um eclipse solar total")

    response = engine.respond("in English")

    assert response == "Sure. I'll respond in English."
    assert engine.response_language == "en-GB"
    telemetry = engine.get_last_turn_telemetry()
    assert telemetry["selected_path"] == "LANGUAGE_COMMAND"


def test_contextual_translation_does_not_change_persistent_response_language(tmp_path: Path) -> None:
    """G. The one-off contextual transformation must never corrupt the
    persistent response_language preference -- only an explicit command
    like "in English"/"fala em inglês" may change it."""
    engine = _make_engine_with_translation_llm(tmp_path)
    engine.respond("explica-me o que é um eclipse solar total")
    assert engine.response_language == "pt-PT"

    engine.respond("quero essa resposta em inglês")

    assert engine.response_language == "pt-PT"
    # The persistent preference is untouched, so a later plain Portuguese
    # turn still gets the pt-PT-flavored prompt, not a leftover English one.
    engine.respond("obrigado")
    assert "response_language = pt-PT" in engine.llm.last_system_prompt or engine.llm.chat_calls == 2


def test_language_command_does_not_stop_tts_manager(tmp_path: Path) -> None:
    """I. Executing "fala em inglês" must never call tts_manager.stop() --
    that call cascades into tearing down the persistent Pocket worker (see
    Phase 7.3.4 section 5's controller.py fix); the language-switch path
    only ever needs to call set_active_voice()."""
    store = UserSettingsStore(tmp_path / "data" / "user_settings.json")
    tts = FakeTTSManager(store)
    engine = make_engine(tmp_path, store=store, tts_manager=tts)

    engine.respond("fala em inglês")

    assert tts.stop_calls == 0
    assert tts.set_calls == [DEFAULT_ENGLISH_VOICE_ID]

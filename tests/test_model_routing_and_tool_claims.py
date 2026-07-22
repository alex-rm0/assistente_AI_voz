from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.tool_registry import ToolRegistry


class ScriptedLLM:
    def __init__(self, model: str, replies: list[str]) -> None:
        self.settings = SimpleNamespace(model=model)
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []
        self._next_source = ""

    def mark_next_call_source(self, source: str) -> None:
        self._next_source = source

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        source = self._next_source or "OTHER"
        self._next_source = ""
        self.calls.append((source, user_message))
        if not self.replies:
            return ""
        return self.replies.pop(0)

    def embed(self, text: str):
        return None

    def calls_from(self, source: str) -> int:
        return sum(1 for call_source, _ in self.calls if call_source == source)


def make_engine(tmp_path: Path, llm: ScriptedLLM, critic_llm: ScriptedLLM | None = None) -> AssistantEngine:
    data = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return AssistantEngine(
        llm=llm,
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=LongTermMemory(data, embedder=llm),
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
        voice_critic_llm=critic_llm,
    )


# --- Teste A: modelo único em todo o pipeline ------------------------------


def test_single_model_is_used_for_optional_regeneration_without_normal_critic(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        "llama3.1:8b",
        [
            "Lamento. Deve ser difícil depois do que passaste.",  # RESPONSE_COMPOSER_REGENERATION
        ],
    )
    engine = make_engine(tmp_path, llm)

    assert engine.response_composer.voice_critic.llm is engine.llm
    assert engine.llm.settings.model == "llama3.1:8b"

    final = engine._finalize_response(
        user_message="Olá, tudo bem?",
        response="Você está bem hoje?",
        source="RESPONSE_COMPOSER",
        selected_path="GENERAL_CONVERSATION",
    )
    assert "Você" not in final
    assert llm.calls_from("VOICE_CRITIC") == 0

    final = engine._finalize_response(
        user_message="Estou mais ou menos, chumbei a um exame importante.",
        response="Que alívio! Estou contente por ter sido convidado para o teu apoio nesse momento.",
        source="RESPONSE_COMPOSER",
        selected_path="GENERAL_CONVERSATION",
    )
    assert final == "Lamento. Deve ser difícil depois do que passaste."
    assert llm.calls_from("RESPONSE_COMPOSER_REGENERATION") == 1


# --- Teste B: resposta normal usa uma única chamada LLM --------------------


def test_normal_safe_reply_uses_a_single_llm_call_and_no_critic(tmp_path: Path) -> None:
    llm = ScriptedLLM("llama3.1:8b", ["não devia ser chamado"])
    engine = make_engine(tmp_path, llm)

    final = engine._finalize_response(
        user_message="Ainda não comecei a estudar porque não tive tempo.",
        response="Percebo que ainda não tiveste tempo para começar.",
        source="RESPONSE_COMPOSER",
        selected_path="GENERAL_CONVERSATION",
    )

    assert final == "Percebo que ainda não tiveste tempo para começar."
    assert llm.calls == []


# --- Teste C: pequena imperfeição tolerada ---------------------------------


def test_exclamation_alone_does_not_trigger_critic(tmp_path: Path) -> None:
    llm = ScriptedLLM("llama3.1:8b", ["não devia ser chamado"])
    engine = make_engine(tmp_path, llm)

    final = engine._finalize_response(
        user_message="Estou finalmente a terminar uma fase do projeto.",
        response="Isso deve saber bem depois de tanto trabalho!",
        source="RESPONSE_COMPOSER",
        selected_path="GENERAL_CONVERSATION",
    )

    assert final == "Isso deve saber bem depois de tanto trabalho!"
    assert llm.calls == []


# --- Teste D: duas perguntas ------------------------------------------------


def test_two_questions_are_reduced_to_one(tmp_path: Path) -> None:
    llm = ScriptedLLM("llama3.1:8b", ["Como correu?"])
    engine = make_engine(tmp_path, llm)

    final = engine._finalize_response(
        user_message="Ainda não comecei a estudar.",
        response="Como correu? Queres falar sobre isso?",
        source="RESPONSE_COMPOSER",
        selected_path="GENERAL_CONVERSATION",
    )

    assert final.count("?") == 1


# --- Teste E: português do Brasil ------------------------------------------


def test_brazilian_portuguese_is_corrected(tmp_path: Path) -> None:
    llm = ScriptedLLM("llama3.1:8b", ["nao devia ser chamado"])
    engine = make_engine(tmp_path, llm)

    final = engine._finalize_response(
        user_message="Não percebi.",
        response="Você pode me explicar isso?",
        source="RESPONSE_COMPOSER",
        selected_path="GENERAL_CONVERSATION",
    )

    assert final == "Podes explicar-me isso?"
    assert llm.calls == []


# --- Teste F: conflito semântico bloqueado e regenerado --------------------


def test_semantic_conflict_is_blocked_and_regenerated(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        "llama3.1:8b",
        [
            "Lamento que tenha corrido mal.",  # regeneration
        ],
    )
    engine = make_engine(tmp_path, llm)

    final = engine._finalize_response(
        user_message="Chumbei a um exame.",
        response="Que alívio!",
        source="RESPONSE_COMPOSER",
        selected_path="GENERAL_CONVERSATION",
    )

    assert final != "Que alívio!"
    assert "alívio" not in final.lower()


# --- Teste G / falsa alegação de pesquisa ----------------------------------


def test_false_search_claim_without_tool_use_is_blocked(tmp_path: Path) -> None:
    llm = ScriptedLLM("llama3.1:8b", [])
    engine = make_engine(tmp_path, llm)

    final = engine._complete_turn(
        "Podes pesquisar no Google sobre pipeline gráfico?",
        "Já encontrei algumas informações no Google.",
        "RESPONSE_COMPOSER",
        selected_path="GENERAL_CONVERSATION",
        tools_used=(),
    )

    assert "encontrei" not in final.lower()
    history_text = " ".join(item.get("content", "") for item in engine.memory.load())
    assert "encontrei" not in history_text.lower()


def test_search_claim_with_real_tool_use_is_allowed(tmp_path: Path) -> None:
    llm = ScriptedLLM("llama3.1:8b", [])
    engine = make_engine(tmp_path, llm)

    final = engine._finalize_response(
        user_message="Podes pesquisar no Google sobre pipeline gráfico?",
        response="Encontrei resultados sobre pipeline gráfico, queres que resuma?",
        source="TOOL_RESULT",
        selected_path="AGENT",
        tools_used=("open_url",),
        technical=True,
    )

    assert "encontrei resultados" in final.lower()


def test_future_promise_without_real_task_is_blocked(tmp_path: Path) -> None:
    llm = ScriptedLLM("llama3.1:8b", [])
    engine = make_engine(tmp_path, llm)

    final = engine._finalize_response(
        user_message="Podes pesquisar no Google sobre pipeline gráfico?",
        response="Vou procurar e depois digo-te.",
        source="RESPONSE_COMPOSER",
        selected_path="GENERAL_CONVERSATION",
        tools_used=(),
    )

    assert "vou procurar" not in final.lower()


# --- Teste J: histórico não contaminado ------------------------------------


def test_rejected_response_never_reaches_history(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        "llama3.1:8b",
        [
            "Que alívio!",
            "Lamento que tenha corrido mal.",
        ],
    )
    engine = make_engine(tmp_path, llm)

    engine._complete_turn(
        "Chumbei a um exame.",
        "Que alívio!",
        "RESPONSE_COMPOSER",
        selected_path="GENERAL_CONVERSATION",
    )

    history_text = " ".join(item.get("content", "") for item in engine.memory.load())
    assert "alívio" not in history_text.lower()

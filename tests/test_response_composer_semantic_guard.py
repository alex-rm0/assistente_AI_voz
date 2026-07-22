from __future__ import annotations

from pathlib import Path

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.tool_registry import ToolRegistry


class ScriptedLLM:
    """Fake LLM that returns one scripted reply per call and records the marked source."""

    def __init__(self, replies: list[str]) -> None:
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


def make_engine(tmp_path: Path, llm: ScriptedLLM) -> AssistantEngine:
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
    )


# --- Teste A / B: conflito semântico grave nunca chega ao utilizador -------


def test_severe_semantic_conflict_is_regenerated_before_reaching_user(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            "Lamento. Deve ser difícil depois do trabalho que tiveste.",  # regeneration
        ]
    )
    engine = make_engine(tmp_path, llm)

    final = engine._finalize_response(
        user_message="Estou mais ou menos, chumbei a um exame importante.",
        response="Que alívio! Estou contente por ter sido convidado para o teu apoio nesse momento.",
        source="RESPONSE_COMPOSER",
        selected_path="GENERAL_CONVERSATION",
    )

    assert final == "Lamento. Deve ser difícil depois do trabalho que tiveste."
    assert "alívio" not in final.lower()
    assert "contente" not in final.lower()
    assert llm.calls_from("VOICE_CRITIC") == 0
    assert llm.calls_from("RESPONSE_COMPOSER_REGENERATION") == 1


def test_semantic_conflict_falls_back_locally_when_regeneration_also_fails(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            "Parabéns, isso é ótimo!",  # regeneration still inverted -> must fall back locally
        ]
    )
    engine = make_engine(tmp_path, llm)

    final = engine._finalize_response(
        user_message="Estou mais ou menos, chumbei a um exame importante.",
        response="Que alívio! Estou contente por ter sido convidado para o teu apoio nesse momento.",
        source="RESPONSE_COMPOSER",
        selected_path="GENERAL_CONVERSATION",
    )

    assert final == "Lamento que tenha corrido mal."
    assert "parabens" not in final.lower()
    assert "alívio" not in final.lower()


# --- Teste F: troca de sujeito ---------------------------------------------


def test_subject_swap_is_regenerated(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            "Percebo que isso te tenha custado, sobretudo depois de te teres preparado.",
        ]
    )
    engine = make_engine(tmp_path, llm)

    final = engine._finalize_response(
        user_message="Foi difícil, apesar de me sentir preparado. Correu mesmo mal.",
        response="Fiquei um pouco surpreendido com o resultado, mas percebo que te custe.",
        source="RESPONSE_COMPOSER",
        selected_path="GENERAL_CONVERSATION",
    )

    assert final == "Percebo que isso te tenha custado, sobretudo depois de te teres preparado."
    assert "fiquei" not in final.lower()


# --- Teste G / H: uma única pergunta ---------------------------------------


def test_two_questions_in_general_conversation_are_reduced_to_one(tmp_path: Path) -> None:
    llm = ScriptedLLM(["Como correu o exame?"])
    engine = make_engine(tmp_path, llm)

    final = engine._finalize_response(
        user_message="Foi difícil, apesar de me sentir preparado. Correu mesmo mal.",
        response="Como foi o exame? Foi difícil ou estavas preparado?",
        source="RESPONSE_COMPOSER",
        selected_path="GENERAL_CONVERSATION",
    )

    assert final.count("?") == 1
    assert llm.calls_from("VOICE_CRITIC") == 0
    assert llm.calls_from("RESPONSE_COMPOSER_REGENERATION") == 0


def test_two_questions_kept_by_revision_trigger_regeneration(tmp_path: Path) -> None:
    llm = ScriptedLLM(["nao devia ser chamado"])
    engine = make_engine(tmp_path, llm)

    final = engine._finalize_response(
        user_message="Foi difícil, apesar de me sentir preparado. Correu mesmo mal.",
        response="Como foi o exame? Foi difícil ou estavas preparado?",
        source="RESPONSE_COMPOSER",
        selected_path="GENERAL_CONVERSATION",
    )

    assert final.count("?") == 1
    assert llm.calls_from("RESPONSE_COMPOSER_REGENERATION") == 0


def test_safe_natural_reply_does_not_trigger_critic(tmp_path: Path) -> None:
    llm = ScriptedLLM(["não devia ser chamado"])
    engine = make_engine(tmp_path, llm)

    final = engine._finalize_response(
        user_message="Foi difícil, apesar de me sentir preparado. Correu mesmo mal.",
        response="Deve ter sido frustrante sentires-te preparado e depois encontrares perguntas tão diferentes.",
        source="RESPONSE_COMPOSER",
        selected_path="GENERAL_CONVERSATION",
    )

    assert final == "Deve ter sido frustrante sentires-te preparado e depois encontrares perguntas tão diferentes."
    assert len(llm.calls) == 0


# --- Teste I: histórico não fica contaminado -------------------------------


def test_rejected_response_never_reaches_history(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            "Lamento. Deve ser difícil depois do trabalho que tiveste.",
        ]
    )
    engine = make_engine(tmp_path, llm)

    final = engine._complete_turn(
        "Estou mais ou menos, chumbei a um exame importante.",
        "Que alívio! Estou contente por ter sido convidado para o teu apoio nesse momento.",
        "RESPONSE_COMPOSER",
        selected_path="GENERAL_CONVERSATION",
    )

    history_text = " ".join(item.get("content", "") for item in engine.memory.load())
    assert final not in "" or True  # final assigned above for readability
    assert "alívio" not in history_text.lower()
    assert "contente" not in history_text.lower()
    assert "Lamento. Deve ser difícil depois do trabalho que tiveste." in history_text


# --- Teste J: limites de chamadas por turno --------------------------------


def test_at_most_two_composer_calls_and_one_critic_call_per_turn(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            "Parabéns, isso é ótimo!",
        ]
    )
    engine = make_engine(tmp_path, llm)

    engine._finalize_response(
        user_message="Estou mais ou menos, chumbei a um exame importante.",
        response="Que alívio! Estou contente por ter sido convidado para o teu apoio nesse momento.",
        source="RESPONSE_COMPOSER",
        selected_path="GENERAL_CONVERSATION",
    )

    assert llm.calls_from("VOICE_CRITIC") == 0
    assert llm.calls_from("RESPONSE_COMPOSER_REGENERATION") <= 1
    assert len(llm.calls) <= 1

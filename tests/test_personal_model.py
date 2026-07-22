from __future__ import annotations

from pathlib import Path

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.personal_model import PersonalModel
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.tool_registry import ToolRegistry


class FakeLLM:
    def __init__(self) -> None:
        self.chat_calls = 0

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        self.chat_calls += 1
        return "resposta"

    def embed(self, text: str):
        return None


def test_create_entry(tmp_path: Path) -> None:
    model = PersonalModel(tmp_path)

    entry = model.add_or_update_entry(
        "preferencias",
        "mapas-mentais",
        "O Alexandre prefere mapas mentais para tomar decisões.",
        confidence=95,
        evidence="Afirmação direta.",
    )

    assert entry.category == "preferencias"
    assert entry.confidence == 95
    assert "mapas mentais" in entry.description


def test_update_entry(tmp_path: Path) -> None:
    model = PersonalModel(tmp_path)

    model.add_or_update_entry("ferramentas", "codex", "O Alexandre usa Codex.", confidence=70)
    updated = model.add_or_update_entry(
        "ferramentas",
        "codex",
        "O Alexandre usa Codex para programação.",
        confidence=90,
        evidence="Uso repetido no projeto.",
    )

    assert updated.confidence == 90
    assert "programação" in updated.description
    assert "Uso repetido" in updated.evidence


def test_search_relevant_context(tmp_path: Path) -> None:
    model = PersonalModel(tmp_path)
    model.add_or_update_entry("estudos", "rvcc", "O Alexandre está a trabalhar em RVCC.", confidence=100)

    context = model.get_relevant_context("ajuda-me com os meus estudos RVCC")

    assert "[personal_model]" in context
    assert "RVCC" in context
    assert "conhecimento forte" in context


def test_list_entries_by_category(tmp_path: Path) -> None:
    model = PersonalModel(tmp_path)
    model.add_or_update_entry("ferramentas", "vscode", "O Alexandre usa VS Code.", confidence=100)

    entries = model.list_entries_by_category("ferramentas")

    assert len(entries) == 1
    assert entries[0].key == "vscode"


def test_delete_entry(tmp_path: Path) -> None:
    model = PersonalModel(tmp_path)
    model.add_or_update_entry("preferencias", "road-trips", "O Alexandre gosta de road trips.", confidence=100)

    deleted = model.delete_entry("road trips")

    assert deleted is not None
    assert model.search_personal_model("road trips") == []


def test_low_and_high_confidence_are_distinguished(tmp_path: Path) -> None:
    model = PersonalModel(tmp_path)
    model.add_or_update_entry("habitos", "manha", "O Alexandre talvez trabalhe melhor de manhã.", confidence=50)
    model.add_or_update_entry("identidade", "nome", "O Alexandre chama-se Alexandre.", confidence=100)

    answer = model.answer_about("", show_details=True)

    assert "hipótese" in answer
    assert "chama-se Alexandre" in answer
    assert "confiança: 50%" in answer
    assert "confiança: 100%" in answer


def test_normal_response_does_not_show_technical_fields(tmp_path: Path) -> None:
    model = PersonalModel(tmp_path)
    model.add_or_update_entry(
        "ferramentas",
        "codex",
        "O Alexandre usa Codex para projetos pessoais e trabalhos da faculdade.",
        confidence=100,
        evidence="Afirmação direta.",
        source="utilizador",
        status="confirmado",
    )

    answer = model.answer_about("ferramentas")

    assert "Codex" in answer
    assert "Personal Model" not in answer
    assert "Categoria:" not in answer
    assert "confian" not in answer.lower()
    assert "origem:" not in answer.lower()
    assert "estado:" not in answer.lower()


def test_details_are_only_shown_when_requested(tmp_path: Path) -> None:
    model = PersonalModel(tmp_path)
    model.add_or_update_entry(
        "ferramentas",
        "codex",
        "O Alexandre usa Codex para programar.",
        confidence=90,
        evidence="Afirmação direta.",
    )

    answer = model.answer_about("codex", show_details=True)

    assert "Detalhes:" in answer
    assert "Categoria:" in answer
    assert "confian" in answer.lower()
    assert "evid" in answer.lower()


def test_conversation_uses_personal_model_for_explicit_memory(tmp_path: Path) -> None:
    data = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    llm = FakeLLM()
    personal_model = PersonalModel(data)
    engine = AssistantEngine(
        llm=llm,
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=LongTermMemory(data, embedder=llm),
        personal_model=personal_model,
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
    )

    remember_response = engine.respond("lembra-te que prefiro mapas mentais")
    answer = engine.respond("o que sabes sobre mim com detalhes?")

    assert "Fico com isso em mente" in remember_response
    assert "Personal Model" not in remember_response
    assert "mapas mentais" in answer
    assert "confiança: 100%" in answer
    assert llm.chat_calls == 0

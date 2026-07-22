from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.personal_model import PersonalModel
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.session_manager import SessionManager
from assistant.tool_registry import ToolRegistry


class FakeLLM:
    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        return user_message

    def embed(self, text: str):
        return None


def make_engine(
    tmp_path: Path,
    memory: LongTermMemory | None = None,
    session_manager: SessionManager | None = None,
) -> AssistantEngine:
    data = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    llm = FakeLLM()
    return AssistantEngine(
        llm=llm,
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=memory or LongTermMemory(data, embedder=llm),
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
        session_manager=session_manager,
    )


def test_startup_without_tasks_or_useful_session_is_quiet(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    assert engine.startup_greeting() == "Olá Alexandre."


def test_startup_with_task_for_today_mentions_task(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path / "data", embedder=FakeLLM())
    memory.create_task("rever o Personal Model", due_date=date.today(), project="AssistenteIA")
    engine = make_engine(tmp_path, memory=memory)

    greeting = engine.startup_greeting()

    assert "Alexandre" in greeting
    assert "Personal Model" in greeting


def test_startup_with_clear_next_step_suggests_resume(tmp_path: Path) -> None:
    data = tmp_path / "data"
    manager = SessionManager(data)
    manager.start_session(datetime(2026, 7, 9, 9, 0, 0))
    manager.record_message_pair(
        "Estamos a melhorar o Personal Model",
        "O próximo passo é melhorar a forma como uso esse contexto nas respostas.",
    )
    manager.end_session(now=datetime(2026, 7, 9, 10, 0, 0))
    engine = make_engine(tmp_path, session_manager=manager)

    greeting = engine.startup_greeting()

    assert greeting.startswith("Olá.")
    assert "Podemos retomar por aqui" not in greeting
    assert "contexto nas respostas" in greeting
    assert greeting.endswith("?")


def test_where_did_we_stop_has_no_technical_logs(tmp_path: Path) -> None:
    data = tmp_path / "data"
    manager = SessionManager(data)
    manager.start_session(datetime(2026, 7, 9, 9, 0, 0))
    manager.record_message_pair(
        "Estamos a trabalhar na arquitetura do Echo",
        "Decidimos melhorar o Session Reflection e o Personal Model.",
    )
    manager.end_session(now=datetime(2026, 7, 9, 10, 0, 0), reason="fecho da aplicacao")
    engine = make_engine(tmp_path, session_manager=manager)

    answer = engine.respond("onde ficámos?")

    assert "Session Reflection" in answer
    assert "sessao iniciada" not in answer.lower()
    assert "sessão iniciada" not in answer.lower()
    assert "sessao terminada" not in answer.lower()
    assert "motivo de fecho" not in answer.lower()
    assert "atividade de trabalho local" not in answer.lower()


def test_what_do_you_know_about_me_has_no_technical_details_by_default(tmp_path: Path) -> None:
    data = tmp_path / "data"
    personal_model = PersonalModel(data)
    personal_model.remember_explicit("uso o Codex para projetos pessoais e trabalhos da faculdade")
    engine = make_engine(tmp_path)
    engine.personal_model = personal_model

    answer = engine.respond("o que sabes sobre mim?")

    assert "Codex" in answer
    assert "confidence" not in answer.lower()
    assert "confiança:" not in answer.lower()
    assert "source" not in answer.lower()
    assert "status" not in answer.lower()


def test_what_do_you_know_about_me_with_empty_model_still_sounds_human(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    engine.personal_model = PersonalModel(tmp_path / "data")

    answer = engine.respond("o que sabes sobre mim?")

    assert "Ainda estamos no início" in answer
    assert "não tenho informação" not in answer.lower()
    assert "nao tenho informacao" not in answer.lower()

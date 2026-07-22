from __future__ import annotations

from datetime import datetime
from pathlib import Path

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.personal_model import PersonalModel
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.response_composer import ComposerRequest, ResponseComposer
from assistant.session_manager import SessionManager
from assistant.tool_registry import ToolRegistry


class FakeLLM:
    def __init__(self, reply: str = "resposta") -> None:
        self.reply = reply
        self.chat_calls = 0
        self.last_system_prompt: str | None = None

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        self.chat_calls += 1
        self.last_system_prompt = system_prompt
        return self.reply

    def embed(self, text: str):
        return None


class RaisingLLM:
    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        raise RuntimeError("Ollama nao esta disponivel.")


# --- Unit tests: ResponseComposer.compose --------------------------------


def test_technical_request_returns_technical_text_without_calling_llm() -> None:
    llm = FakeLLM()
    composer = ResponseComposer(llm)
    request = ComposerRequest(
        intent="personal_model",
        user_message="mostra os detalhes",
        facts=["usa Codex para programar"],
        show_technical=True,
        technical_text="Categoria: ferramentas; confiança: 100%.",
    )

    answer = composer.compose(request)

    assert answer == "Categoria: ferramentas; confiança: 100%."
    assert llm.chat_calls == 0


def test_no_facts_still_calls_llm_for_normal_conversation() -> None:
    llm = FakeLLM("Resposta natural com base na conversa.")
    composer = ResponseComposer(llm)
    request = ComposerRequest(
        intent="session_reflection",
        user_message="onde ficámos?",
        facts=[],
        fallback="Ainda não tenho uma sessão anterior guardada.",
    )

    answer = composer.compose(request)

    assert answer == "Resposta natural com base na conversa."
    assert llm.chat_calls == 1


def test_llm_failure_falls_back_to_deterministic_text() -> None:
    composer = ResponseComposer(RaisingLLM())
    request = ComposerRequest(
        intent="personal_model",
        user_message="o que sabes sobre mim?",
        facts=["usa Codex para programar"],
        fallback="Sei que usas o Codex.",
    )

    answer = composer.compose(request)

    assert answer == "Sei que usas o Codex."


def test_empty_llm_reply_falls_back() -> None:
    composer = ResponseComposer(FakeLLM(reply="   "))
    request = ComposerRequest(
        intent="personal_model",
        user_message="o que sabes sobre mim?",
        facts=["usa Codex para programar"],
        fallback="Sei que usas o Codex.",
    )

    answer = composer.compose(request)

    assert answer == "Sei que usas o Codex."


def test_copied_memory_reply_is_recomposed() -> None:
    composer = ResponseComposer(FakeLLM(reply="usa Codex para programar"))
    request = ComposerRequest(
        intent="personal_model",
        user_message="o que sabes sobre mim?",
        facts=["usa Codex para programar"],
        fallback="Sei que usas o Codex.",
    )

    answer = composer.compose(request)

    assert answer == "Sei que usas o Codex."


def test_composes_from_facts_via_llm_and_never_leaks_technical_terms() -> None:
    llm = FakeLLM(reply="Ainda estou a conhecer-te, mas sei que recorres ao Codex para programar.")
    composer = ResponseComposer(llm)
    request = ComposerRequest(
        intent="personal_model",
        user_message="o que sabes sobre mim?",
        facts=["recorres frequentemente ao Codex para programar"],
        fallback="Sei que usas o Codex.",
    )

    answer = composer.compose(request)

    assert answer == "Ainda estou a conhecer-te, mas sei que recorres ao Codex para programar."
    assert llm.chat_calls == 1
    assert "companheiro digital" in llm.last_system_prompt.lower()


# --- Integration tests: AssistantEngine routes through the composer -------


def _make_engine(tmp_path: Path, personal_model: PersonalModel | None = None, session_manager: SessionManager | None = None):
    data = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    llm = FakeLLM(reply="Da última vez estivemos a evoluir o Echo e o próximo passo era continuar por aí.")
    engine = AssistantEngine(
        llm=llm,
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=LongTermMemory(data, embedder=llm),
        personal_model=personal_model,
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
        session_manager=session_manager,
    )
    return engine, llm


def test_what_do_you_know_about_me_does_not_return_technical_lists(tmp_path: Path) -> None:
    personal_model = PersonalModel(tmp_path / "data")
    personal_model.remember_explicit("uso o Codex para projetos pessoais e trabalhos da faculdade")
    engine, llm = _make_engine(tmp_path, personal_model=personal_model)
    llm.reply = "Ainda estou a conhecer-te, mas já sei que recorres bastante ao Codex para programar."

    answer = engine.respond("o que sabes sobre mim?")

    assert "Codex" in answer
    assert "Categoria" not in answer
    assert "confiança:" not in answer.lower()
    assert "status" not in answer.lower()
    assert "sqlite" not in answer.lower()


def test_where_did_we_stop_produces_a_natural_summary(tmp_path: Path) -> None:
    data = tmp_path / "data"
    manager = SessionManager(data)
    manager.start_session(datetime(2026, 7, 8, 9, 0, 0))
    manager.record_message_pair(
        "Vamos evoluir a arquitetura do Echo",
        "Decidimos criar o Personal Model como próximo passo.",
    )
    manager.end_session(now=datetime(2026, 7, 8, 10, 0, 0), reason="inatividade")
    engine, llm = _make_engine(tmp_path, session_manager=manager)

    answer = engine.respond("onde ficámos?")

    assert answer == llm.reply
    assert "motivo de fecho" not in answer.lower()
    assert "sessao iniciada" not in answer.lower()
    assert "sessão terminada" not in answer.lower()


def test_summarize_last_session_produces_a_short_narrative(tmp_path: Path) -> None:
    data = tmp_path / "data"
    manager = SessionManager(data)
    manager.start_session(datetime(2026, 7, 8, 9, 0, 0))
    manager.record_message_pair(
        "Vamos evoluir a arquitetura do Echo",
        "Decidimos criar o Personal Model como próximo passo.",
    )
    manager.end_session(now=datetime(2026, 7, 8, 10, 0, 0), reason="inatividade")
    engine, llm = _make_engine(tmp_path, session_manager=manager)

    answer = engine.respond("resume a última sessão")

    assert answer == llm.reply
    assert llm.chat_calls == 1
    assert "session_summaries" not in answer.lower()


def test_technical_details_only_appear_when_explicitly_requested(tmp_path: Path) -> None:
    personal_model = PersonalModel(tmp_path / "data")
    personal_model.remember_explicit("uso o Codex para programar")
    engine, llm = _make_engine(tmp_path, personal_model=personal_model)

    natural_answer = engine.respond("o que sabes sobre mim?")
    technical_answer = engine.respond("o que sabes sobre mim com detalhes?")

    assert llm.chat_calls == 0
    assert "Codex" in natural_answer
    assert "Categoria:" not in natural_answer
    assert "Categoria:" in technical_answer
    assert "confiança:" in technical_answer.lower()

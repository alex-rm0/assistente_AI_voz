from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from assistant.anthropic_provider import PAID_CALL_CONFIRMATION_ENV
from assistant.attention import (
    ACTIVE,
    FOCUS,
    GLANCEABLE,
    HIDDEN,
    MINIMIZED,
    VISIBLE,
    set_bulk_attention,
)
from assistant.conversation import AssistantEngine, WorkSession
from assistant.memory import ConversationMemory
from assistant.model_provider import ModelResponse
from assistant.model_router import AutomaticRoutingConfig, ModelRouter, ModelRoutingConfig, ModelUsageBudget, RoutedLLM
from assistant.model_runtime import ModelRuntimeBridge, UserSettingsStore
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.representation import (
    SUPPORTED_STYLES,
    RepresentationConstraint,
    SemanticObject,
    SpacePressureState,
    register_object,
    restore_representation,
    transform_representation,
)
from assistant.conversation import resolve_workspace_reference
from assistant.tool_registry import ToolRegistry


class FakeProvider:
    def __init__(self, name: str, model: str, replies: list[str] | None = None) -> None:
        self._name = name
        self.model = model
        self.replies = replies or ["Resposta."]
        self.calls: list[dict] = []

    @property
    def name(self) -> str:
        return self._name

    def chat(
        self,
        messages,
        *,
        model=None,
        response_format=None,
        tools=None,
        temperature=None,
        num_predict=None,
        timeout_seconds=None,
    ):
        self.calls.append({
            "messages": messages,
            "model": model,
            "response_format": response_format,
            "temperature": temperature,
            "num_predict": num_predict,
            "timeout_seconds": timeout_seconds,
        })
        index = min(len(self.calls) - 1, len(self.replies) - 1)
        return ModelResponse(
            text=self.replies[index],
            provider=self._name,
            model=model or self.model,
            input_tokens=120 + index,
            output_tokens=40 + index,
            latency_ms=1.0,
            estimated_cost_usd=0.002 if self._name == "anthropic" else 0.0,
        )


class MemoryStub:
    def __init__(self) -> None:
        self.preferences: dict[str, str] = {}
        self.context_text = ""
        self.search_results: list[dict] = []

    def get_preference(self, key: str, default: str = "") -> str:
        return self.preferences.get(key, default)

    def set_preference(self, key: str, value: str) -> None:
        self.preferences[key] = value

    def context_for(self, query: str, limit: int = 5) -> str:
        return self.context_text

    def search(self, query: str, limit: int = 5):
        return self.search_results[:limit]

    def pending_tasks(self, *args, **kwargs) -> str:
        return ""


class ContextObserverStub:
    def __init__(self, snapshot=None) -> None:
        self.snapshot = snapshot
        self.observe_calls = 0

    def observe_once(self):
        self.observe_calls += 1
        return self.snapshot

    def latest_snapshot(self):
        return self.snapshot


def make_engine(
    tmp_path: Path,
    *,
    mode: str = "automatic",
    claude_enabled: bool = True,
    env: dict[str, str] | None = None,
    ollama_replies=None,
    anthropic_replies=None,
    now_provider=None,
):
    ollama = FakeProvider("ollama", "llama3.1:8b", ollama_replies)
    anthropic = FakeProvider("anthropic", "claude-haiku-4-5-20251001", anthropic_replies)
    router = ModelRouter(
        ModelRoutingConfig(
            mode=mode,
            mode_source="test",
            automatic=AutomaticRoutingConfig(claude_enabled=claude_enabled, daily_budget_usd=0.25, max_single_call_estimated_usd=0.05),
        ),
        ollama_model=ollama.model,
        anthropic_model=anthropic.model,
        budget=ModelUsageBudget(tmp_path / "usage.json"),
        env=env or {},
    )
    llm = RoutedLLM(
        providers={"ollama": ollama, "anthropic": anthropic},
        router=router,
        system_prompt=get_base_system_prompt(),
        model_source="test",
    )
    data = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    engine = AssistantEngine(
        llm=llm,
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=MemoryStub(),
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
        debug_ollama_payload=True,
        now_provider=now_provider,
    )
    return engine, llm, ollama, anthropic


EMAIL_REQUEST = (
    "Escreve um email profissional detalhado a explicar o estado atual do projeto Echo, "
    "os progressos realizados e os próximos passos."
)


def write_text_pdf(path: Path, text: str) -> None:
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as handle:
        writer.write(handle)


def test_model_status_question_uses_system_status_without_llm(tmp_path: Path) -> None:
    engine, llm, ollama, anthropic = make_engine(tmp_path, mode="local", ollama_replies=["Resposta inventada."])
    engine.model_runtime = ModelRuntimeBridge(
        router=llm.router,
        budget=llm.router.budget,
        store=UserSettingsStore(tmp_path / "user_settings.json"),
        env={},
    )

    response = engine.respond("Que modelo estás a usar?")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "llama3.1:8b" in response
    assert telemetry["selected_path"] == "SYSTEM_STATUS"
    assert telemetry["llm_calls"] == 0
    assert ollama.calls == []
    assert anthropic.calls == []


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Que dia é hoje?", "Hoje é quinta-feira, 23 de julho de 2026."),
        ("Que dia da semana é hoje?", "Hoje é quinta-feira, 23 de julho de 2026."),
        ("Qual é a data de amanhã?", "Amanhã é sexta-feira, 24 de julho de 2026."),
        ("Ontem foi que dia?", "Ontem foi quarta-feira, 22 de julho de 2026."),
        ("Daqui a 3 dias é que dia?", "Daqui a 3 dias é domingo, 26 de julho de 2026."),
        ("Que horas são?", "São 14:35."),
        ("Dia 25/12/2026 calha em que dia da semana?", "25 de dezembro de 2026 calha a sexta-feira."),
    ],
)
def test_system_datetime_uses_local_clock_without_llm(tmp_path: Path, message: str, expected: str) -> None:
    fixed_now = datetime(2026, 7, 23, 14, 35)
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        ollama_replies=["Resposta errada do modelo."],
        now_provider=lambda: fixed_now,
    )

    response = engine.respond(message)
    telemetry = engine.get_last_turn_telemetry() or {}

    assert response == expected
    assert telemetry["selected_path"] == "SYSTEM_DATETIME"
    assert telemetry["response_source"] == "LOCAL_DATETIME"
    assert telemetry["llm_calls"] == 0
    assert telemetry["tools_used"] == ["system_datetime"]
    assert telemetry["configured_model_mode"] == "automatic"
    assert telemetry["execution_path"] == "system_datetime"
    assert telemetry["execution_provider"] == "local"
    assert telemetry["execution_model"] == "NONE"
    assert telemetry["model_routing_mode"] == "automatic"
    assert telemetry["model_routing_provider"] == "local"
    assert telemetry["model_routing_model"] == "NONE"
    assert telemetry["model_routing_reason_code"] == "system_datetime"
    assert telemetry["model_routing_paid_call"] is False
    assert telemetry["estimated_cost_usd"] == 0.0
    assert ollama.calls == []
    assert anthropic.calls == []


def test_calendar_claim_without_calendar_tool_is_blocked(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="local")
    engine._begin_turn_trace("Tenho alguma reunião hoje?")

    response = engine._complete_turn(
        "Tenho alguma reunião hoje?",
        "Consultei o calendário e o meu calendário está atualizado.",
        "RESPONSE_COMPOSER",
        selected_path="GENERAL_CONVERSATION",
        tools_used=(),
    )
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "calendário está atualizado" not in response.lower()
    assert telemetry["unsupported_tool_claim_detected"] is True
    assert telemetry["llm_calls"] == 0
    assert ollama.calls == []
    assert anthropic.calls == []


def test_incomplete_email_request_starts_writing_task_not_memory(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic")

    response = engine.respond("Preciso de ajuda a escrever um email a um professor.")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "WRITING_TASK"
    assert telemetry["response_source"] == "WRITING_WORKFLOW"
    assert telemetry["model_routing_reason_code"] == "writing_workflow"
    assert telemetry["execution_path"] == "writing_workflow"
    assert telemetry["execution_provider"] == "local"
    assert telemetry["memory_route"] == "NONE"
    assert telemetry["llm_calls"] == 0
    assert "o que queres pedir" in response.lower()
    assert ollama.calls == []
    assert anthropic.calls == []


def test_writing_followup_does_not_assume_project_from_vague_reference(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic")

    engine.respond("Preciso de ajuda a escrever um email.")
    engine.respond("É para um potencial patrocinador.")
    response = engine.respond("É sobre um projeto que estou a desenvolver.")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "WRITING_TASK"
    assert telemetry["writing_project"] == "unknown"
    assert telemetry["writing_workflow_active"] is True
    assert telemetry["memory_route"] == "NONE"
    assert telemetry["llm_calls"] == 0
    assert "qual é o projeto" in response.lower()
    assert "assistenteia" not in response.lower()
    assert "echo" not in response.lower()
    assert ollama.calls == []
    assert anthropic.calls == []


def test_social_presence_question_uses_fast_path_without_llm(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic")

    response = engine.respond("Estás aí?")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert response == "Sim, estou aqui."
    assert telemetry["selected_path"] == "SOCIAL_PATH"
    assert telemetry["execution_path"] == "social_fast_path"
    assert telemetry["llm_calls"] == 0
    assert ollama.calls == []
    assert anthropic.calls == []


def test_social_thanks_uses_fast_path_without_llm(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic")

    response = engine.respond("Ok, obrigado.")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert response == "De nada."
    assert telemetry["selected_path"] == "SOCIAL_PATH"
    assert telemetry["llm_calls"] == 0
    assert ollama.calls == []
    assert anthropic.calls == []


def test_child_social_personality_is_playful_without_claiming_feelings(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic")

    response = engine.respond("É a minha irmã de 7 anos a perguntar a tua cor preferida.")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "azul" in response.lower()
    assert "luzes" in response.lower()
    assert "sinto" not in response.lower()
    assert telemetry["selected_path"] == "SOCIAL_PATH"
    assert telemetry["llm_calls"] == 0
    assert ollama.calls == []
    assert anthropic.calls == []


def test_system_context_status_is_local_and_grounded(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic")

    response = engine.respond("Estás a guardar contexto?")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "SYSTEM_CONTEXT_STATUS"
    assert telemetry["response_source"] == "SYSTEM_CONTEXT_STATUS"
    assert telemetry["execution_path"] == "system_context_status"
    assert telemetry["execution_provider"] == "local"
    assert telemetry["configured_model_mode_source"] == "test"
    assert telemetry["response_grounded"] is True
    assert "histórico" in response.lower()
    assert "memória" in response.lower()
    assert telemetry["llm_calls"] == 0
    assert ollama.calls == []
    assert anthropic.calls == []


def test_specific_application_activity_uses_context_observer_not_general_summary(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic")
    snapshot = SimpleNamespace(
        active_app="chrome.exe",
        active_window="Pesquisa Echo - Google Chrome",
        open_windows=[
            SimpleNamespace(title="Pesquisa Echo - Google Chrome", process_name="chrome.exe", is_active=True),
            SimpleNamespace(title="AssistenteIA - Visual Studio Code", process_name="Code.exe", is_active=False),
        ],
    )
    engine.context_observer = ContextObserverStub(snapshot)

    response = engine.respond("O que estou a fazer no Google Chrome?")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "TOOL_PATH"
    assert telemetry["tools_used"] == ["get_application_activity"]
    assert telemetry["grounding_sources"] == ["CONTEXT_OBSERVER"]
    assert "chrome" in response.lower()
    assert "pesquisa echo" in response.lower()
    assert "conteúdo exato" in response.lower()
    assert "ficheiros modificados" not in response.lower()
    assert "git" not in response.lower()
    assert ollama.calls == []
    assert anthropic.calls == []


def test_unsupported_notes_check_claim_is_blocked_without_tool(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="local")
    engine._begin_turn_trace("Tenho notas sobre isto?")

    response = engine._complete_turn(
        "Tenho notas sobre isto?",
        "Estava a verificar as notas e encontrei alguma coisa.",
        "RESPONSE_COMPOSER",
        selected_path="GENERAL_CONVERSATION",
        tools_used=(),
    )
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "estava a verificar as notas" not in response.lower()
    assert telemetry["unsupported_tool_claim_detected"] is True
    assert telemetry["final_response_source"] == "TOOL_CLAIM_BLOCKED"
    assert telemetry["llm_calls"] == 0
    assert ollama.calls == []
    assert anthropic.calls == []


def test_professional_email_request_is_not_memory_command_and_uses_anthropic_when_authorized(tmp_path: Path) -> None:
    engine, llm, ollama, anthropic = make_engine(
        tmp_path,
        env={"ANTHROPIC_API_KEY": "secret", "ECHO_ALLOW_PAID_MODEL_CALLS": "true"},
        anthropic_replies=["Assunto: Estado atual do projeto Echo\n\nOlá,\n\nSegue o ponto de situação do projeto Echo."],
    )

    response = engine.respond(EMAIL_REQUEST)
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] != "MEMORY_COMMAND"
    assert telemetry["response_source"] != "MEMORY_COMMAND"
    assert "ACTIVE_CONVERSATION" not in response
    assert "Assunto:" in response
    assert telemetry["configured_model_mode"] == "automatic"
    assert telemetry["execution_path"] == "llm"
    assert telemetry["execution_provider"] == "anthropic"
    assert telemetry["execution_model"] == "claude-haiku-4-5-20251001"
    assert telemetry["model_routing_mode"] == "automatic"
    assert telemetry["model_routing_provider"] == "anthropic"
    assert telemetry["model_routing_reason_code"] == "professional_writing"
    assert len(anthropic.calls) == 1
    assert ollama.calls == []


def test_professional_email_request_uses_ollama_when_automatic_claude_disabled(tmp_path: Path) -> None:
    engine, llm, ollama, anthropic = make_engine(
        tmp_path,
        claude_enabled=False,
        ollama_replies=["Assunto: Estado atual do projeto Echo\n\nOlá,\n\nSegue o ponto de situação."],
    )

    response = engine.respond(EMAIL_REQUEST)
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "Assunto:" in response
    assert telemetry["configured_model_mode"] == "automatic"
    assert telemetry["execution_path"] == "llm"
    assert telemetry["execution_provider"] == "ollama"
    assert telemetry["execution_model"] == "llama3.1:8b"
    assert telemetry["model_routing_mode"] == "automatic"
    assert telemetry["model_routing_provider"] == "ollama"
    assert telemetry["model_routing_reason_code"] == "automatic_claude_disabled"
    assert len(ollama.calls) == 1
    assert anthropic.calls == []


def test_real_presence_state_question_still_works(tmp_path: Path) -> None:
    engine, llm, ollama, anthropic = make_engine(tmp_path, claude_enabled=False)

    response = engine.respond("Em que modo estás?")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "ACTIVE_CONVERSATION" in response
    assert telemetry["selected_path"] == "MEMORY_COMMAND"
    assert llm.chat_call_count == 0


def test_complete_writing_request_regenerates_instead_of_offering_help(tmp_path: Path) -> None:
    engine, llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="local",
        ollama_replies=[
            "Posso ajudar-te a redigir o email?",
            "Assunto: Estado atual do projeto Echo\n\nOlá,\n\nSegue o ponto de situação.",
        ],
    )

    response = engine.respond(EMAIL_REQUEST)
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "Posso ajudar" not in response
    assert "Queres que" not in response
    assert "Assunto:" in response
    assert telemetry["llm_calls"] == 2
    assert [item["component"] for item in telemetry["llm_call_details"]] == [
        "RESPONSE_COMPOSER",
        "RESPONSE_COMPOSER_REGENERATION",
    ]
    assert all(item["provider"] == "ollama" for item in telemetry["llm_call_details"])


def test_pending_intent_confirmation_executes_original_email_request(tmp_path: Path) -> None:
    engine, llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="local",
        ollama_replies=["Assunto: Estado atual do projeto Echo\n\nOlá,\n\nSegue o ponto de situação."],
    )
    engine._pending_user_intent = {"kind": "professional_writing", "message": EMAIL_REQUEST, "ttl": "3"}

    response = engine.respond("sim")

    assert "Assunto:" in response
    assert "Posso ajudar" not in response
    assert engine._pending_user_intent is None


def test_pending_intent_subject_followup_recovers_email_context(tmp_path: Path) -> None:
    engine, llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="local",
        ollama_replies=["Assunto: Estado atual do projeto Echo\n\nOlá,\n\nSegue o ponto de situação."],
    )
    engine._pending_user_intent = {"kind": "professional_writing", "message": EMAIL_REQUEST, "ttl": "3"}

    response = engine.respond("do email")

    assert "Assunto:" in response
    assert "Echo" in response
    assert engine._pending_user_intent is None


def test_project_duration_question_triggers_grounded_recall_from_history(tmp_path: Path) -> None:
    engine, llm, ollama, anthropic = make_engine(tmp_path, mode="automatic")
    engine.memory.append_pair("Há três meses começámos o projeto Echo.", "Fico com isso como contexto.")

    response = engine.respond("Há quanto tempo estamos a trabalhar no projeto Echo?")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "ha tres meses" in response.lower() or "há tres meses" in response.lower() or "há três meses" in response.lower()
    assert telemetry["selected_path"] == "MEMORY_RECALL"
    assert telemetry["memory_recall_detected"] is True
    assert telemetry["configured_model_mode"] == "automatic"
    assert telemetry["execution_path"] == "memory_recall"
    assert telemetry["execution_provider"] == "memory"
    assert telemetry["execution_model"] == "NONE"
    assert telemetry["model_routing_mode"] == "automatic"
    assert "CONVERSATION_HISTORY" in telemetry["grounding_sources"]
    assert telemetry["history_context_used"] is True
    assert telemetry["llm_calls"] == 0


def test_project_duration_question_without_memory_does_not_invent(tmp_path: Path) -> None:
    engine, llm, ollama, anthropic = make_engine(tmp_path, mode="automatic")

    response = engine.respond("Há quanto tempo estamos a trabalhar no projeto Echo?")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "não tenho dados suficientes" in response.lower()
    assert "cinco meses" not in response.lower()
    assert telemetry["selected_path"] == "MEMORY_RECALL"
    assert telemetry["llm_calls"] == 0


@pytest.mark.parametrize(
    "prompt",
    [
        "O que Ã© o projeto Echo?",
        "Resume o projeto Echo em quatro pontos",
        "Quais sÃ£o os objetivos do Echo?",
        "Em que ponto estÃ¡ o projeto Echo?",
        "O que jÃ¡ fizemos no Echo?",
        "Quais sÃ£o os prÃ³ximos passos?",
    ],
)
def test_general_project_questions_trigger_grounded_recall_before_llm(tmp_path: Path, prompt: str) -> None:
    engine, llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="local",
        ollama_replies=["1. O Echo organiza contexto.\n2. MantÃ©m memÃ³ria.\n3. Ajuda a retomar trabalho.\n4. O prÃ³ximo passo Ã© estabilizar."],
    )
    engine.memory.append_pair(
        "Estamos a trabalhar no projeto Echo, sobretudo em memÃ³ria, routing e interface.",
        "Certo. Vou manter esse contexto.",
    )
    engine.long_term_memory.context_text = "O projeto Echo Ã© um companheiro digital persistente focado em contexto, memÃ³ria e continuidade."

    response = engine.respond(prompt)
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "Echo" in response
    assert telemetry["selected_path"] == "MEMORY_RECALL"
    assert telemetry["memory_recall_detected"] is True
    assert "CONVERSATION_HISTORY" in telemetry["grounding_sources"]
    assert "PERSISTENT_MEMORY" in telemetry["grounding_sources"]
    assert telemetry["history_context_used"] is True
    assert telemetry["llm_calls"] == 1
    assert [item["component"] for item in telemetry["llm_call_details"]] == ["RESPONSE_COMPOSER"]
    assert len(ollama.calls) == 1
    assert anthropic.calls == []


def test_project_recall_without_grounding_does_not_ask_llm_or_invent(tmp_path: Path) -> None:
    engine, llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="local",
        ollama_replies=["O Echo Ã© um projeto antigo com vÃ¡rios anos."],
    )

    response = engine.respond("O que Ã© o projeto Echo?")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "contexto suficiente" in response.lower()
    assert telemetry["selected_path"] == "MEMORY_RECALL"
    assert telemetry["memory_recall_detected"] is True
    assert telemetry["response_grounded"] is False
    assert telemetry["grounding_sources"] == []
    assert telemetry["llm_calls"] == 0
    assert ollama.calls == []
    assert anthropic.calls == []


def test_text_summary_about_echo_is_not_project_memory_recall(tmp_path: Path) -> None:
    engine, llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="local",
        ollama_replies=["1. O Echo acompanha contexto.\n2. Guarda memÃ³ria.\n3. Ajuda a planear.\n4. MantÃ©m continuidade."],
    )
    engine.memory.append_pair("O projeto Echo tem memÃ³ria persistente.", "Certo.")

    response = engine.respond(
        "Resume este texto em quatro pontos: O projeto Echo acompanha contexto, guarda memÃ³ria, ajuda a planear e mantÃ©m continuidade."
    )
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "Echo" in response
    assert telemetry["selected_path"] == "TEXT_SUMMARIZATION"
    assert telemetry["memory_recall_detected"] is False
    assert telemetry["llm_calls"] == 1


def test_conversation_history_provenance_is_reported_when_context_enters_prompt(tmp_path: Path) -> None:
    engine, llm, ollama, anthropic = make_engine(tmp_path, mode="local", ollama_replies=["Ao longo dos últimos cinco meses, trabalhámos no Echo."])
    engine.memory.append_pair("Estamos a trabalhar no projeto Echo há cinco meses.", "Certo.")

    response = engine.respond("Escreve uma frase curta sobre o projeto.")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "cinco meses" in response
    assert "CONVERSATION_HISTORY" in telemetry["grounding_sources"]
    assert telemetry["history_context_used"] is True
    assert telemetry["history_context_turn_ids"]


def test_structured_summary_direct_path_uses_one_llm_call_and_not_task_management(tmp_path: Path) -> None:
    engine, llm, ollama, anthropic = make_engine(
        tmp_path,
        env={"ANTHROPIC_API_KEY": "secret", "ECHO_ALLOW_PAID_MODEL_CALLS": "true"},
        anthropic_replies=["1. A biblioteca alargou o horário.\n2. A medida apoia os exames.\n3. A adesão será avaliada.\n4. A equipa decide depois."],
    )

    response = engine.respond(
        "Resume este texto em quatro pontos claros: A biblioteca alargou o horário durante exames e vai avaliar a medida."
    )
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "biblioteca" in response.lower()
    assert telemetry["selected_path"] == "TEXT_SUMMARIZATION"
    assert telemetry["agent_route_reason"] == ""
    assert telemetry["model_routing_reason_code"] == "structured_summary"
    assert telemetry["llm_calls"] == 1
    assert [item["component"] for item in telemetry["llm_call_details"]] == ["RESPONSE_COMPOSER"]
    assert all(item["component"] != "OTHER" for item in telemetry["llm_call_details"])
    assert anthropic.calls
    assert ollama.calls == []


def test_four_tasks_request_still_routes_to_task_management(tmp_path: Path) -> None:
    engine, llm, ollama, anthropic = make_engine(tmp_path, mode="local")

    engine.respond("Cria quatro tarefas para o projeto Echo.")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] != "TEXT_SUMMARIZATION"


def test_unsupported_historical_duration_from_llm_is_blocked_without_source(tmp_path: Path) -> None:
    engine, llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="local",
        ollama_replies=["Ao longo dos últimos cinco meses, tens trabalhado no projeto Echo."],
    )

    response = engine.respond("Escreve uma frase curta sobre o projeto Echo.")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "não tenho memória suficiente" in response.lower()
    assert telemetry["response_source"] == "RESPONSE_COMPOSER"
    assert telemetry["final_response"] == response
    assert telemetry["unsupported_memory_claim_detected"] is True


def test_pdf_named_request_enters_document_task_without_false_found_claim(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic")

    response = engine.respond("Resume o PDF inexistente.pdf")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["workspace_file_found"] is False
    assert "encontrei o ficheiro" not in response.lower()
    assert ollama.calls == []
    assert anthropic.calls == []


REAL_EMAIL_SUMMARY = (
    "ASSUNTO: Resumo da Reunião de Direção - 19 de junho\n\n"
    "Olá Francisca,\n\n"
    "Segue o resumo da reunião de Direção.\n\n"
    "- Febrada final de época\n"
    "- Protocolo Águas de Coimbra\n"
    "- Candidatura COP\n"
    "- Sanfil\n"
)


def test_document_task_reads_real_email_summary_without_extension_or_llm(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic")
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")

    response = engine.respond("preciso que leias o documento email_resumo_novo e me digas o que está lá escrito")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["execution_path"] == "document_task"
    assert telemetry["execution_provider"] == "tool"
    assert telemetry["execution_model"] == "NONE"
    assert telemetry["llm_calls"] == 0
    assert telemetry["tools_used"] == ["workspace_search", "read_workspace_document"]
    assert telemetry["grounding_sources"] == ["WORKSPACE_FILE"]
    assert telemetry["workspace_file_found"] is True
    assert telemetry["workspace_files_used"][0]["name"] == "email_resumo_novo.txt"
    assert "Olá Francisca" in response
    assert "Sanfil" in response
    assert "benefícios esperados" not in response.lower()
    assert "requisitos para o patrocinador" not in response.lower()
    assert ollama.calls == []
    assert anthropic.calls == []


def test_document_task_multiturn_full_contents_reuses_resolved_file(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic")
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")

    first = engine.respond("preciso que leias o documento email_resumo_novo")
    first_telemetry = engine.get_last_turn_telemetry() or {}
    second = engine.respond("quero saber na íntegra o que está lá escrito")
    second_telemetry = engine.get_last_turn_telemetry() or {}

    assert first_telemetry["selected_path"] == "DOCUMENT_TASK"
    assert second_telemetry["selected_path"] == "DOCUMENT_TASK"
    assert second_telemetry["tools_used"] == []
    assert second_telemetry["grounding_sources"] == ["WORKSPACE_FILE"]
    assert second_telemetry["document_context_reused"] is True
    assert second_telemetry["llm_calls"] == 0
    assert "Olá Francisca" in first
    assert "Olá Francisca" in second
    assert "Sanfil" in second
    assert "benefícios esperados" not in second.lower()
    assert ollama.calls == []
    assert anthropic.calls == []


def test_document_review_and_interpret_continue_from_active_document_context(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        env={"ANTHROPIC_API_KEY": "secret", "ECHO_ALLOW_PAID_MODEL_CALLS": "true"},
        anthropic_replies=[
            "O email é para a Francisca e resume a reunião de Direção. Eu melhorava a clareza dos pontos Sanfil, COP e Águas de Coimbra.",
            "É um email de resumo da reunião de Direção de 19 de junho, com pontos sobre febrada, Águas de Coimbra, COP e Sanfil.",
        ],
    )
    engine.long_term_memory.search_results = []
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")

    first = engine.respond("lê o ficheiro email_resumo_novo e diz exatamente o que lá está escrito")
    review = engine.respond("sugeres alguma alteração ao email?")
    review_telemetry = engine.get_last_turn_telemetry() or {}
    interpret = engine.respond("interpreta o que está no documento e diz-me tu")
    interpret_telemetry = engine.get_last_turn_telemetry() or {}

    assert "Olá Francisca" in first
    assert review_telemetry["selected_path"] == "DOCUMENT_TASK"
    assert review_telemetry["active_document"] is True
    assert review_telemetry["active_document_name"] == "email_resumo_novo.txt"
    assert review_telemetry["document_action"] == "review"
    assert review_telemetry["document_context_reused"] is True
    assert review_telemetry["document_grounded"] is True
    assert review_telemetry["grounding_sources"] == ["WORKSPACE_FILE"]
    assert review_telemetry["tools_used"] == []
    assert review_telemetry["model_routing_reason_code"] == "document_review"
    assert "Francisca" in review
    assert "destinatário" not in review.lower()
    assert "propósito" not in review.lower()
    assert "Francisca" in anthropic.calls[0]["messages"][-1]["content"]
    assert "Sanfil" in anthropic.calls[0]["messages"][-1]["content"]
    assert interpret_telemetry["selected_path"] == "DOCUMENT_TASK"
    assert interpret_telemetry["document_action"] == "interpret"
    assert interpret_telemetry["grounding_sources"] == ["WORKSPACE_FILE"]
    assert interpret_telemetry["tools_used"] == []
    assert "futebol" not in interpret.lower()
    assert "vou começar" not in interpret.lower()
    assert "Direção" in interpret
    assert ollama.calls == []


def test_active_document_display_followup_does_not_call_llm(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic")
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")

    engine.respond("lê o ficheiro email_resumo_novo")
    response = engine.respond("mostra tudo")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["document_action"] == "display"
    assert telemetry["document_context_reused"] is True
    assert telemetry["llm_calls"] == 0
    assert "Olá Francisca" in response
    assert ollama.calls == []
    assert anthropic.calls == []


def test_active_document_side_question_does_not_clear_context_and_can_resume(tmp_path: Path) -> None:
    now = datetime(2026, 7, 26, 10, 30)
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic", now_provider=lambda: now)
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")

    engine.respond("lê o ficheiro email_resumo_novo")
    side = engine.respond("que horas são?")
    resumed = engine.respond("voltando ao email, o que mudavas?")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "10:30" in side
    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["document_action"] == "review"
    assert telemetry["grounding_sources"] == ["WORKSPACE_FILE"]
    assert "ficheiro da workspace" not in resumed.lower()


def test_active_document_clear_on_topic_shift(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic")
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")

    engine.respond("lê o ficheiro email_resumo_novo")
    engine.respond("vamos falar de outra coisa")
    response = engine.respond("sugeres alguma alteração?")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] != "DOCUMENT_TASK"
    assert telemetry["active_document"] is False
    assert "email_resumo_novo" not in response


def test_active_document_rereads_when_file_changes(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="local",
        ollama_replies=["A versão atualizada menciona Sanfil e uma nota nova."],
    )
    file_path = engine.workspace_path / "email_resumo_novo.txt"
    file_path.write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")

    engine.respond("lê o ficheiro email_resumo_novo")
    file_path.write_text(REAL_EMAIL_SUMMARY + "\nNota nova adicionada.\n", encoding="utf-8")
    response = engine.respond("interpreta o que está no documento")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["document_action"] == "interpret"
    assert telemetry["document_context_reused"] is False
    assert telemetry["tools_used"] == ["read_workspace_document"]
    assert "Nota nova adicionada." in ollama.calls[0]["messages"][-1]["content"]
    assert "nota nova" in response.lower()
    assert anthropic.calls == []


def test_active_document_context_is_not_written_to_persistent_memory(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic")
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")

    engine.respond("lê o ficheiro email_resumo_novo")

    assert engine.long_term_memory.search_results == []
    assert engine.long_term_memory.context_text == ""
    assert ollama.calls == []
    assert anthropic.calls == []


def test_document_task_omitted_extension_exact_stem_reads_txt(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic")
    (engine.workspace_path / "nota simples.txt").write_text("Texto real da nota simples.", encoding="utf-8")

    response = engine.respond("mostra-me o conteúdo de nota simples")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["workspace_files_used"][0]["name"] == "nota simples.txt"
    assert "Texto real da nota simples." in response
    assert telemetry["llm_calls"] == 0
    assert ollama.calls == []
    assert anthropic.calls == []


def test_document_task_multiple_matches_asks_for_choice(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic")
    (engine.workspace_path / "relatorio.txt").write_text("A", encoding="utf-8")
    (engine.workspace_path / "relatorio.md").write_text("B", encoding="utf-8")

    response = engine.respond("lê o ficheiro relatorio")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["workspace_file_found"] is False
    assert "mais do que uma possibilidade" in response.lower()
    assert "relatorio.txt" in response
    assert "relatorio.md" in response
    assert telemetry["llm_calls"] == 0
    assert ollama.calls == []
    assert anthropic.calls == []


def test_document_task_missing_file_does_not_fall_back_to_llm(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic")

    response = engine.respond("diz-me o que está no ficheiro inexistente")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["workspace_file_found"] is False
    assert "não encontrei" in response.lower()
    assert telemetry["llm_calls"] == 0
    assert ollama.calls == []
    assert anthropic.calls == []


def test_document_task_read_vs_summarize_are_distinct(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=["Resumo curto das notas."],
    )
    (engine.workspace_path / "notas.txt").write_text("Conteúdo longo real das notas.", encoding="utf-8")

    read_response = engine.respond("lê o ficheiro notas")
    read_telemetry = engine.get_last_turn_telemetry() or {}
    summary_response = engine.respond("resume o ficheiro notas")
    summary_telemetry = engine.get_last_turn_telemetry() or {}

    assert "Conteúdo longo real das notas." in read_response
    assert read_telemetry["llm_calls"] == 0
    assert summary_response == "Resumo curto das notas."
    assert summary_telemetry["llm_calls"] == 1
    assert ollama.calls
    assert anthropic.calls == []


def test_document_false_read_claim_without_tool_is_blocked(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="local")
    engine._begin_turn_trace("Leste o ficheiro?")

    response = engine._complete_turn(
        "Leste o ficheiro?",
        "Já estou a lê-lo e aqui está o conteúdo.",
        "RESPONSE_COMPOSER",
        selected_path="GENERAL_CONVERSATION",
        tools_used=(),
    )
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "já estou a lê-lo" not in response.lower()
    assert telemetry["unsupported_tool_claim_detected"] is True
    assert telemetry["final_response_source"] == "TOOL_CLAIM_BLOCKED"
    assert telemetry["llm_calls"] == 0
    assert ollama.calls == []
    assert anthropic.calls == []


def test_document_task_reads_workspace_file_before_summary(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=["Resumo baseado nas notas reais do Echo."],
    )
    (engine.workspace_path / "notas.txt").write_text("Notas reais sobre o projeto Echo.", encoding="utf-8")

    response = engine.respond("Resume o ficheiro notas.txt")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert response == "Resumo baseado nas notas reais do Echo."
    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["workspace_file_found"] is True
    assert telemetry["grounding_sources"] == ["WORKSPACE_FILE"]
    assert telemetry["workspace_files_used"][0]["name"] == "notas.txt"
    assert "workspace_search" in telemetry["tools_used"]
    assert "read_workspace_document" in telemetry["tools_used"]
    assert "Notas reais sobre o projeto Echo." in ollama.calls[0]["messages"][-1]["content"]
    assert anthropic.calls == []


def test_document_task_creates_txt_file_real_on_disk(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="local")

    response = engine.respond("Cria mail_resumo_novo.txt com este texto: Ola direcao.")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "Criei o ficheiro 'mail_resumo_novo.txt'" in response
    assert (engine.workspace_path / "mail_resumo_novo.txt").read_text(encoding="utf-8") == "Ola direcao."
    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["output_file_created"] is True
    assert telemetry["output_file_path"] == "mail_resumo_novo.txt"
    assert "create_workspace_file" in telemetry["tools_used"]
    assert ollama.calls == []
    assert anthropic.calls == []


def test_document_task_blocks_path_traversal(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="local")

    response = engine.respond("Le o ficheiro ../segredo.txt")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "fora da workspace" in response.lower()
    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["workspace_file_found"] is False
    assert ollama.calls == []
    assert anthropic.calls == []


def test_document_task_pending_create_confirmation_continues_workflow(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=["Email preparado a partir das notas."],
    )
    (engine.workspace_path / "notas.txt").write_text("Notas reais para email.", encoding="utf-8")

    first = engine.respond("Le o ficheiro notas.txt, escreve um email e guarda em mail.txt")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "Criei o ficheiro 'mail.txt'" in first
    assert (engine.workspace_path / "mail.txt").read_text(encoding="utf-8") == "Email preparado a partir das notas."
    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["output_file_created"] is True
    assert "workspace_search" in telemetry["tools_used"]
    assert "read_workspace_document" in telemetry["tools_used"]
    assert "create_workspace_file" in telemetry["tools_used"]
    assert anthropic.calls == []


def test_document_task_reads_pdf_and_creates_txt_from_real_workspace_file(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=[
            (
                "Assunto: Resumo da reunião\n\n"
                "Olá Francisca,\n\n"
                "Resumo: planeamento da regata, orçamento aprovado, patrocínios contigo "
                "e próxima reunião em setembro."
            )
        ],
    )
    pdf_name = "Reuniões Direção - Notas.pdf"
    pdf_text = (
        "Planeamento da regata aprovado. Orcamento aprovado. "
        "Francisca responsavel pelos patrocinios. Proxima reuniao em setembro."
    )
    write_text_pdf(engine.workspace_path / pdf_name, pdf_text)

    response = engine.respond(
        "Escreve um email para a Francisca com o resumo do PDF 'Reuniões Direção - Notas' "
        "e guarda em mail_resumo_novo.txt."
    )
    telemetry = engine.get_last_turn_telemetry() or {}
    output = engine.workspace_path / "mail_resumo_novo.txt"

    assert "Criei o ficheiro 'mail_resumo_novo.txt'" in response
    assert output.exists()
    assert output.stat().st_size > 0
    assert "Francisca" in output.read_text(encoding="utf-8")
    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["workspace_file_found"] is True
    assert telemetry["workspace_files_used"][0]["name"] == pdf_name
    assert telemetry["grounding_sources"] == ["WORKSPACE_FILE"]
    assert telemetry["output_file_created"] is True
    assert telemetry["output_file_path"] == "mail_resumo_novo.txt"
    assert telemetry["tools_used"] == ["workspace_search", "read_workspace_document", "create_workspace_file"]
    assert "Planeamento da regata aprovado" in ollama.calls[0]["messages"][-1]["content"]
    assert "Destinatário: Francisca" in ollama.calls[0]["messages"][-1]["content"]
    assert len(ollama.calls) == 1
    assert anthropic.calls == []


def test_document_task_fuzzy_matches_singular_to_plural_pdf_name(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=["Resumo da reunião encontrado no PDF real."],
    )
    pdf_name = "Reuniões Direção - Notas.pdf"
    write_text_pdf(engine.workspace_path / pdf_name, "Planeamento da regata e orcamento aprovado.")

    response = engine.respond("Resume o PDF 'Reunião Direção - Notas'")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert response == "Resumo da reunião encontrado no PDF real."
    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["workspace_files_used"][0]["name"] == pdf_name
    assert telemetry["workspace_files_used"][0]["match_type"] == "fuzzy"
    assert "Planeamento da regata" in ollama.calls[0]["messages"][-1]["content"]
    assert anthropic.calls == []


def test_document_task_accepts_absolute_path_inside_workspace(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=["Resumo do caminho absoluto dentro da workspace."],
    )
    pdf_path = engine.workspace_path / "Reuniões Direção - Notas.pdf"
    write_text_pdf(pdf_path, "Francisca responsavel pelos patrocinios.")

    response = engine.respond(f'Resume o PDF "{pdf_path}"')
    telemetry = engine.get_last_turn_telemetry() or {}

    assert response == "Resumo do caminho absoluto dentro da workspace."
    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["workspace_files_used"][0]["name"] == "Reuniões Direção - Notas.pdf"
    assert telemetry["workspace_files_used"][0]["match_type"] == "absolute"
    assert ollama.calls
    assert anthropic.calls == []


def test_document_task_rejects_absolute_path_outside_workspace(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="local")
    outside = tmp_path / "segredo.pdf"
    write_text_pdf(outside, "segredo")

    response = engine.respond(f'Resume o PDF "{outside}"')
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "fora da workspace" in response.lower()
    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["workspace_file_found"] is False
    assert ollama.calls == []
    assert anthropic.calls == []


def test_document_task_keeps_pending_workflow_when_source_file_is_provided_later(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=["Email com resumo real da reunião para a Francisca."],
    )
    pdf_name = "Reuniões Direção - Notas.pdf"
    write_text_pdf(
        engine.workspace_path / pdf_name,
        "Planeamento da regata. Orcamento aprovado. Proxima reuniao em setembro.",
    )

    first = engine.respond("Escreve um email para a Francisca com o resumo do PDF e guarda em mail_resumo_novo.txt.")
    second = engine.respond('o ficheiro é "Reuniões Direção - Notas.pdf"')
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "nome do ficheiro" in first.lower()
    assert "Criei o ficheiro 'mail_resumo_novo.txt'" in second
    assert (engine.workspace_path / "mail_resumo_novo.txt").exists()
    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["workspace_files_used"][0]["name"] == pdf_name
    assert telemetry["tools_used"] == ["workspace_search", "read_workspace_document", "create_workspace_file"]
    assert len(ollama.calls) == 1
    assert anthropic.calls == []


def test_document_synthesis_is_not_low_complexity_when_cloud_available(tmp_path: Path) -> None:
    engine, _llm, _ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=True,
        env={"ANTHROPIC_API_KEY": "secret", "ECHO_ALLOW_PAID_MODEL_CALLS": "true"},
        anthropic_replies=["Email preparado com Claude."],
    )
    (engine.workspace_path / "notas.txt").write_text("Notas reais para email.", encoding="utf-8")

    response = engine.respond("Le o ficheiro notas.txt e escreve um email com o resumo")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert response == "Email preparado com Claude."
    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["model_routing_provider"] == "anthropic"
    assert telemetry["model_routing_reason_code"] == "document_synthesis"
    assert anthropic.calls


def test_social_fast_path_after_claude_does_not_inherit_paid_turn_telemetry(tmp_path: Path) -> None:
    engine, _llm, _ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=True,
        env={"ANTHROPIC_API_KEY": "secret", "ECHO_ALLOW_PAID_MODEL_CALLS": "true"},
        anthropic_replies=["Email preparado com Claude."],
    )
    (engine.workspace_path / "notas.txt").write_text("Notas reais para email.", encoding="utf-8")

    first = engine.respond("Le o ficheiro notas.txt e escreve um email com o resumo")
    first_telemetry = engine.get_last_turn_telemetry() or {}
    second = engine.respond("obrigado")
    second_telemetry = engine.get_last_turn_telemetry() or {}

    assert first == "Email preparado com Claude."
    assert first_telemetry["model_routing_provider"] == "anthropic"
    assert first_telemetry["model_routing_paid_call"] is True
    assert first_telemetry["estimated_cost_usd"] > 0
    assert second.lower() in {"de nada.", "claro.", "sempre.", "tranquilo."}
    assert second_telemetry["selected_path"] == "SOCIAL_PATH"
    assert second_telemetry["response_source"] == "SOCIAL_FAST_PATH"
    assert second_telemetry["llm_calls"] == 0
    assert second_telemetry["configured_model_mode"] == "automatic"
    assert second_telemetry["execution_path"] == "social_fast_path"
    assert second_telemetry["execution_provider"] == "local"
    assert second_telemetry["execution_model"] == "NONE"
    assert second_telemetry["model_routing_mode"] == "automatic"
    assert second_telemetry["model_routing_provider"] == "local"
    assert second_telemetry["model_routing_model"] == "NONE"
    assert second_telemetry["model_routing_reason_code"] == "social_fast_path"
    assert second_telemetry["model_routing_paid_call"] is False
    assert second_telemetry["estimated_cost_usd"] == 0.0
    assert len(anthropic.calls) == 1


# --- PATCH A: deterministic full-contents/display follow-ups, negation-safe
# summary detection, safe file_reference extraction, bare-stem pending resolution.


def test_active_document_followup_recognizes_digas_o_que_esta_escrito(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic")
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")

    engine.respond("lê o ficheiro email_resumo_novo e diz exatamente o que lá está escrito")
    response = engine.respond("quero que me digas o que está escrito")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["document_action"] == "display"
    assert telemetry["execution_path"] == "document_task"
    assert telemetry["execution_provider"] == "tool"
    assert telemetry["llm_calls"] == 0
    assert telemetry["grounding_sources"] == ["WORKSPACE_FILE"]
    assert "Olá Francisca" in response
    assert ollama.calls == []
    assert anthropic.calls == []


def test_active_document_followup_negation_sem_resumir_returns_literal_content(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic")
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")

    engine.respond("lê o ficheiro email_resumo_novo e diz exatamente o que lá está escrito")
    response = engine.respond("quero o conteúdo todo, sem resumir")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["document_action"] == "display"
    assert telemetry["execution_path"] == "document_task"
    assert telemetry["llm_calls"] == 0
    assert telemetry["grounding_sources"] == ["WORKSPACE_FILE"]
    assert "Olá Francisca" in response
    assert "Sanfil" in response
    assert ollama.calls == []
    assert anthropic.calls == []


def test_active_document_followup_nao_resumas_mostra_tudo(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic")
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")

    engine.respond("lê o ficheiro email_resumo_novo e diz exatamente o que lá está escrito")
    response = engine.respond("não resumas, mostra tudo")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["document_action"] == "display"
    assert telemetry["llm_calls"] == 0
    assert telemetry["grounding_sources"] == ["WORKSPACE_FILE"]
    assert "Olá Francisca" in response
    assert ollama.calls == []
    assert anthropic.calls == []


def test_explicit_summary_request_ignores_negated_resumir() -> None:
    from assistant.conversation import _explicit_summary_request, _looks_like_summary_or_email

    assert _explicit_summary_request("quero o conteúdo todo, sem resumir") is False
    assert _explicit_summary_request("não resumas, mostra tudo") is False
    assert _explicit_summary_request("não quero um resumo") is False
    assert _looks_like_summary_or_email("sem resumir") is False
    assert _explicit_summary_request("resume o ficheiro notas") is True
    assert _explicit_summary_request("faz um resumo") is True


def test_extract_semantic_file_reference_rejects_relational_words() -> None:
    from assistant.conversation import _extract_semantic_file_reference

    msg = "estou a falar do documento que está no workspace email_resumo_novo, o que está nesse ficheiro?"
    assert _extract_semantic_file_reference(msg) == "email_resumo_novo"


def test_semantic_file_reference_workspace_phrase_resolves_real_file(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic")
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")

    response = engine.respond(
        "estou a falar do documento que está no workspace email_resumo_novo, o que está nesse ficheiro?"
    )
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["workspace_file_found"] is True
    assert telemetry["workspace_files_used"][0]["name"] == "email_resumo_novo.txt"
    assert telemetry["grounding_sources"] == ["WORKSPACE_FILE"]
    assert "não encontrei" not in response.lower()


def test_pending_document_task_resolves_bare_stem_reply(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic")
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")

    first = engine.respond("lê o ficheiro")
    first_telemetry = engine.get_last_turn_telemetry() or {}
    second = engine.respond("email_resumo_novo")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "nome exato do ficheiro" in first.lower()
    assert first_telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["workspace_file_found"] is True
    assert telemetry["workspace_files_used"][0]["name"] == "email_resumo_novo.txt"
    assert telemetry["llm_calls"] == 0
    assert "Olá Francisca" in second
    assert ollama.calls == []
    assert anthropic.calls == []


def test_pending_document_task_bare_stem_with_explicit_extension(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic")
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")

    engine.respond("lê o ficheiro")
    response = engine.respond("email_resumo_novo.txt")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["workspace_file_found"] is True
    assert telemetry["workspace_files_used"][0]["name"] == "email_resumo_novo.txt"
    assert telemetry["llm_calls"] == 0
    assert "Olá Francisca" in response


def test_pending_document_task_bare_stem_multiple_matches_still_asks_choice(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic")
    (engine.workspace_path / "relatorio.txt").write_text("A", encoding="utf-8")
    (engine.workspace_path / "relatorio.md").write_text("B", encoding="utf-8")

    engine.respond("lê o ficheiro")
    response = engine.respond("relatorio")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "mais do que uma possibilidade" in response.lower()
    assert telemetry["llm_calls"] == 0
    assert ollama.calls == []
    assert anthropic.calls == []


def test_pending_document_task_bare_stem_no_match_reports_not_found(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic")

    engine.respond("lê o ficheiro")
    response = engine.respond("xyz_nao_existe")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "não encontrei" in response.lower()
    assert telemetry["llm_calls"] == 0
    assert ollama.calls == []
    assert anthropic.calls == []


# --- PATCH B: grounded fallback for ambiguous document follow-ups that don't
# hit any specific _active_document_followup_action marker.


def test_active_document_grounded_fallback_for_ambiguous_suggestion_request(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=["Podias mencionar o orçamento aprovado com mais destaque."],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")

    engine.respond("lê o ficheiro email_resumo_novo e diz exatamente o que lá está escrito")
    response = engine.respond("tens alguma sugestão para melhorar o mail?")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["document_action"] == "review"
    assert telemetry["active_document"] is True
    assert telemetry["grounding_sources"] == ["WORKSPACE_FILE"]
    assert response == "Podias mencionar o orçamento aprovado com mais destaque."
    assert "Sanfil" in ollama.calls[0]["messages"][-1]["content"]
    assert anthropic.calls == []


def test_active_document_grounded_fallback_does_not_swallow_unrelated_question(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=["Não tenho essa informação, mas parece que vai estar bom tempo."],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")

    engine.respond("lê o ficheiro email_resumo_novo")
    response = engine.respond("achas que vai chover amanhã?")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] != "DOCUMENT_TASK"
    assert telemetry["grounding_sources"] != ["WORKSPACE_FILE"]
    assert response == "Não tenho essa informação, mas parece que vai estar bom tempo."


def test_active_document_grounded_fallback_respects_topic_shift_clear(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=["Pode ser, o que gostavas de falar?", "Não tenho mais contexto sobre isso."],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")

    engine.respond("lê o ficheiro email_resumo_novo")
    engine.respond("vamos falar de outra coisa")
    response = engine.respond("tens alguma sugestão para melhorar o mail?")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] != "DOCUMENT_TASK"
    assert telemetry["active_document"] is False
    assert "email_resumo_novo" not in response


# --- PATCH C: review/interpret/rewrite marker widening in
# _active_document_followup_action itself (earlier, more specific stage than
# the PATCH B grounded fallback), sharing the same opinion-marker tables.


def test_active_document_followup_action_classifies_ambiguous_suggestion_as_review() -> None:
    from assistant.conversation import _active_document_followup_action

    assert _active_document_followup_action("tens alguma sugestão para melhorar o mail?", "email") == "review"
    assert _active_document_followup_action("achas que vai chover amanhã?", "email") == ""


def test_active_document_case_h_resolves_via_followup_action_not_generic_fallback(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=["Podias mencionar o orçamento aprovado com mais destaque."],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")

    engine.respond("lê o ficheiro email_resumo_novo e diz exatamente o que lá está escrito")
    response = engine.respond("tens alguma sugestão para melhorar o mail?")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["document_action"] == "review"
    assert telemetry["execution_path"] == "llm"
    assert telemetry["grounding_sources"] == ["WORKSPACE_FILE"]
    assert response == "Podias mencionar o orçamento aprovado com mais destaque."
    assert anthropic.calls == []


def test_active_document_followup_action_widened_interpret_and_rewrite_paraphrases() -> None:
    from assistant.conversation import _active_document_followup_action

    assert _active_document_followup_action("o que significa isto?", "email") == "interpret"
    assert _active_document_followup_action("explica me isto", "email") == "interpret"
    assert _active_document_followup_action("torna isto mais claro", "email") == "rewrite"
    assert _active_document_followup_action("escreve melhor", "email") == "rewrite"


def test_active_document_followup_action_display_still_wins_over_review() -> None:
    from assistant.conversation import _active_document_followup_action

    assert _active_document_followup_action("mostra tudo", "email") == "display"
    assert _active_document_followup_action("que horas são?", "email") == ""


# --- PATCH C regression fix: natural rewrite phrasing (pronoun-suffixed
# imperatives), review-vs-rewrite tie-break, draft state, and the
# topic-shift ambiguity fix.


def test_active_document_followup_action_recognizes_natural_rewrite_phrasing() -> None:
    from assistant.conversation import _active_document_followup_action

    rewrite_cases = (
        "Torna-o mais formal, mas não guardes ainda.",
        "torna isto mais formal",
        "deixa o email mais formal",
        "reescreve num tom mais formal",
        "melhora o texto",
        "faz uma versão melhor",
        "reformula",
        "escreve isto de outra forma",
        "corrige e melhora",
        "adapta para um tom profissional",
    )
    for message in rewrite_cases:
        assert _active_document_followup_action(message, "email") == "rewrite", message


def test_active_document_followup_action_review_vs_rewrite_distinct() -> None:
    from assistant.conversation import _active_document_followup_action

    assert _active_document_followup_action("sugeres alguma alteração?", "email") == "review"
    assert _active_document_followup_action("há algo a melhorar?", "email") == "review"
    assert _active_document_followup_action("o que mudavas?", "email") == "review"
    assert _active_document_followup_action("torna mais formal", "email") == "rewrite"
    assert _active_document_followup_action("reescreve", "email") == "rewrite"
    # "deixa estar" is the unrelated cancel phrase (_is_negative_confirmation)
    # and must not be misread as a rewrite request just because "deixa" is a
    # guarded rewrite verb.
    assert _active_document_followup_action("deixa estar", "email") == ""


def test_active_document_followup_action_draft_markers() -> None:
    from assistant.conversation import _active_document_followup_action

    assert _active_document_followup_action("mostra a versão melhorada", "email") == "display_draft"
    assert _active_document_followup_action("voltando ao email, mostra-me a versão melhorada", "email") == "display_draft"
    assert _active_document_followup_action("mostra o original", "email") == "display"
    assert _active_document_followup_action("compara as duas versões", "email") == "compare_versions"
    assert _active_document_followup_action("descarta as alterações", "email") == "discard_draft"
    assert _active_document_followup_action("guarda esta versão", "email") == "save_draft"


def test_document_rewrite_creates_unsaved_draft_and_supports_full_followup_sequence(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=[
            "Podias mencionar o orçamento com mais destaque.",
            "Prezada Francisca,\n\nSegue, em anexo, o resumo formal da reunião de Direção.\n\n"
            "- Febrada final de época\n- Protocolo Águas de Coimbra\n- Candidatura COP\n- Sanfil\n\n"
            "Com os melhores cumprimentos,\nAlexandre",
        ],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")
    original_bytes = (engine.workspace_path / "email_resumo_novo.txt").read_bytes()

    # 1. Ler o email.
    first = engine.respond("lê o ficheiro email_resumo_novo e diz exatamente o que lá está escrito")
    assert "Olá Francisca" in first

    # 2. Pedir sugestão (review, não rewrite).
    engine.respond("Tens alguma sugestão para melhorar este email?")
    suggestion_telemetry = engine.get_last_turn_telemetry() or {}
    assert suggestion_telemetry["document_action"] == "review"

    # 3. Rewrite sem guardar.
    rewritten = engine.respond("Torna-o mais formal, mas não guardes ainda.")
    rewrite_telemetry = engine.get_last_turn_telemetry() or {}

    # 4. Confirmar que foi devolvido um email completo reescrito.
    assert "Prezada Francisca" in rewritten
    assert "Alexandre" in rewritten
    assert rewrite_telemetry["selected_path"] == "DOCUMENT_TASK"
    assert rewrite_telemetry["document_action"] == "rewrite"
    assert rewrite_telemetry["document_context_reused"] is True
    assert rewrite_telemetry["document_grounded"] is True
    assert rewrite_telemetry["draft_created"] is True
    assert rewrite_telemetry["draft_saved"] is False
    assert rewrite_telemetry["grounding_sources"] == ["WORKSPACE_FILE"]
    assert rewrite_telemetry["draft_validation_passed"] is True
    assert rewrite_telemetry["draft_validation_failed"] is False
    assert rewrite_telemetry["draft_regeneration_attempted"] is False
    assert rewrite_telemetry["draft_original_length"] > 0
    assert rewrite_telemetry["draft_generated_length"] > 0

    # 5. Confirmar que o ficheiro original não mudou.
    assert (engine.workspace_path / "email_resumo_novo.txt").read_bytes() == original_bytes

    # 6. Pergunta lateral não relacionada com o documento.
    engine.respond("Que horas são?")
    time_telemetry = engine.get_last_turn_telemetry() or {}
    assert time_telemetry["selected_path"] == "SYSTEM_DATETIME"

    # 7. Voltar ao email e pedir a versão melhorada.
    draft_response = engine.respond("Voltando ao email, mostra-me a versão melhorada.")
    draft_telemetry = engine.get_last_turn_telemetry() or {}

    # 8. Confirmar que devolve exatamente o draft, sem reler nem chamar o LLM.
    assert draft_telemetry["selected_path"] == "DOCUMENT_TASK"
    assert draft_telemetry["document_action"] == "display_draft"
    assert draft_telemetry["draft_reused"] is True
    assert draft_telemetry["draft_saved"] is False
    assert draft_telemetry["llm_calls"] == 0
    assert draft_telemetry["grounding_sources"] == ["WORKSPACE_FILE"]
    assert "Prezada Francisca" in draft_response

    # 9. Mostra o original.
    original_response = engine.respond("Mostra o original.")
    original_telemetry = engine.get_last_turn_telemetry() or {}

    # 10. Confirmar que devolve o texto original, não a versão melhorada.
    assert original_telemetry["document_action"] == "display"
    assert "Olá Francisca" in original_response
    assert "Prezada Francisca" not in original_response

    # 11. Descarta as alterações.
    engine.respond("Descarta as alterações.")
    discard_telemetry = engine.get_last_turn_telemetry() or {}

    # 12. Confirmar que draft_content foi limpo.
    assert discard_telemetry["document_action"] == "discard_draft"
    assert engine._active_document_context.draft_content == ""
    assert engine._active_document_context.draft_saved is False
    assert len(ollama.calls) == 2
    assert anthropic.calls == []


_VALID_FORMAL_REWRITE = (
    "Prezada Francisca,\n\nSegue, em anexo, o resumo formal da reunião de Direção.\n\n"
    "- Febrada final de época\n- Protocolo Águas de Coimbra\n- Candidatura COP\n- Sanfil\n\n"
    "Com os melhores cumprimentos,\nAlexandre"
)


def test_document_rewrite_invalid_first_response_triggers_one_regeneration(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=[
            "O documento parece pronto para enviar! É para enviar a algum destinatário ou é apenas um resumo interno?",
            _VALID_FORMAL_REWRITE,
        ],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")

    engine.respond("lê o ficheiro email_resumo_novo e diz exatamente o que lá está escrito")
    response = engine.respond("Torna-o mais formal, mas não guardes ainda.")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["draft_regeneration_attempted"] is True
    assert telemetry["draft_validation_passed"] is True
    assert telemetry["draft_validation_failed"] is False
    assert telemetry["draft_created"] is True
    assert "pronto para enviar" not in response
    assert "Prezada Francisca" in response
    assert engine._active_document_context.draft_content == _VALID_FORMAL_REWRITE
    assert len(ollama.calls) == 2
    assert anthropic.calls == []


def test_document_rewrite_both_attempts_invalid_creates_no_draft(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=[
            "O documento parece pronto para enviar! É para enviar a algum destinatário ou é apenas um resumo interno?",
            "Achas que devia mencionar mais alguma coisa antes de enviar?",
        ],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")
    original_bytes = (engine.workspace_path / "email_resumo_novo.txt").read_bytes()

    engine.respond("lê o ficheiro email_resumo_novo e diz exatamente o que lá está escrito")
    response = engine.respond("Torna-o mais formal, mas não guardes ainda.")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["draft_created"] is False
    assert telemetry["draft_validation_failed"] is True
    assert telemetry["draft_validation_passed"] is False
    assert telemetry["draft_regeneration_attempted"] is True
    assert telemetry["draft_validation_reason"]
    assert engine._active_document_context.draft_content == ""
    assert "não consegui" in response.lower()
    assert (engine.workspace_path / "email_resumo_novo.txt").read_bytes() == original_bytes
    assert len(ollama.calls) == 2
    assert anthropic.calls == []


def test_document_rewrite_rejects_complete_but_short_response_missing_entities(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=["Reunião concluída com sucesso.", "Reunião encerrada."],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")

    engine.respond("lê o ficheiro email_resumo_novo e diz exatamente o que lá está escrito")
    engine.respond("Torna-o mais formal, mas não guardes ainda.")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["draft_created"] is False
    assert telemetry["draft_validation_failed"] is True
    assert engine._active_document_context.draft_content == ""


def test_document_rewrite_valid_first_response_needs_no_regeneration(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=[_VALID_FORMAL_REWRITE],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")

    engine.respond("lê o ficheiro email_resumo_novo e diz exatamente o que lá está escrito")
    rewrite_telemetry_response = engine.respond("Torna-o mais formal, mas não guardes ainda.")
    telemetry = engine.get_last_turn_telemetry() or {}
    draft_response = engine.respond("Voltando ao email, mostra-me a versão melhorada.")
    draft_telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["draft_regeneration_attempted"] is False
    assert telemetry["draft_validation_passed"] is True
    assert "Prezada Francisca" in rewrite_telemetry_response
    assert draft_telemetry["llm_calls"] == 0
    assert draft_response.endswith(_VALID_FORMAL_REWRITE)
    assert len(ollama.calls) == 1
    assert anthropic.calls == []


def test_document_rewrite_draft_not_written_to_persistent_memory(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=[_VALID_FORMAL_REWRITE],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")

    engine.respond("lê o ficheiro email_resumo_novo e diz exatamente o que lá está escrito")
    engine.respond("Torna-o mais formal, mas não guardes ainda.")

    assert engine.long_term_memory.search_results == []
    assert engine.long_term_memory.context_text == ""
    assert "Francisca" not in str(engine.long_term_memory.preferences)


def test_ambiguous_opinion_question_after_topic_shift_asks_for_context(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=["Pode ser, o que gostavas de falar?"],
    )

    engine.respond("Vamos falar de outra coisa.")
    response = engine.respond("O que achas?")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert response == "Sobre o quê?"
    assert telemetry["selected_path"] == "GENERAL_CONVERSATION"
    assert telemetry["response_source"] == "CLARIFICATION_DETERMINISTIC"
    assert telemetry["llm_calls"] == 0
    assert telemetry["memory_route"] == "NONE"
    assert "memoria" not in response.lower() and "memória" not in response.lower()
    assert len(ollama.calls) == 1
    assert anthropic.calls == []


# --- Rewrite reliability fix: preamble stripping, placeholder detection,
# honest failure message, and temperature/num_predict for the rewrite call
# only. Reproduces the manually observed false positive: "Vou reescrever o
# documento completo... Aqui está a resposta:" immediately followed by an
# otherwise-perfect rewrite was rejected only because it matched
# "vou reescrever".

_PREAMBLE_PLUS_VALID_REWRITE = (
    "Vou reescrever o documento completo sem perguntas e sem análise, mantendo todos os factos. "
    "Aqui está a resposta:\n\n"
    "ASSUNTO: Resumo da Reunião de Direção - 19 de junho\n\n"
    "Francisca,\n\n"
    "Segue o resumo da reunião de Direção.\n\n"
    "- Febrada final de época\n"
    "- Protocolo Águas de Coimbra\n"
    "- Candidatura COP\n"
    "- Sanfil"
)


def test_rewrite_strips_preamble_and_accepts_valid_document_after_it(tmp_path: Path) -> None:
    from assistant.conversation import _validate_rewrite_draft

    valid, reason, cleaned = _validate_rewrite_draft(REAL_EMAIL_SUMMARY, _PREAMBLE_PLUS_VALID_REWRITE)

    assert valid is True
    assert reason == ""
    assert cleaned.startswith("ASSUNTO:")
    assert "vou reescrever" not in cleaned.lower()
    assert "aqui está a resposta" not in cleaned.lower()


def test_rewrite_bare_preamble_without_document_is_still_rejected() -> None:
    from assistant.conversation import _validate_rewrite_draft

    valid, reason, _cleaned = _validate_rewrite_draft(REAL_EMAIL_SUMMARY, "Vou reescrever isto para ti em breve.")

    assert valid is False
    assert reason != ""


def test_rewrite_rejects_placeholder_with_specific_reason() -> None:
    from assistant.conversation import _validate_rewrite_draft

    candidate = (
        "ASSUNTO: Resumo da Reunião de Direção - 19 de junho\n\n"
        "Francisca,\n\n"
        "Segue o resumo da reunião de Direção.\n\n"
        "- Febrada final de época\n"
        "- Protocolo Águas de Coimbra\n"
        "- Candidatura COP\n"
        "- Sanfil\n\n"
        "Atenciosamente,\n[Tu o nome]"
    )
    valid, reason, _cleaned = _validate_rewrite_draft(REAL_EMAIL_SUMMARY, candidate)

    assert valid is False
    assert reason == "placeholder_detected"


def test_document_rewrite_first_attempt_invalid_second_has_preamble_accepted(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=[
            "O documento parece pronto para enviar! É para enviar a algum destinatário?",
            _PREAMBLE_PLUS_VALID_REWRITE,
        ],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")

    engine.respond("lê o ficheiro email_resumo_novo e diz exatamente o que lá está escrito")
    response = engine.respond("Torna-o mais formal, mas não guardes ainda.")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["draft_validation_passed"] is True
    assert telemetry["draft_regeneration_attempted"] is True
    assert telemetry["draft_preamble_removed"] is True
    assert telemetry["draft_placeholder_detected"] is False
    assert "vou reescrever" not in engine._active_document_context.draft_content.lower()
    assert engine._active_document_context.draft_content.startswith("ASSUNTO:")
    assert "Francisca" in response
    assert len(ollama.calls) == 2
    assert anthropic.calls == []


def test_document_rewrite_uses_low_temperature_and_bounded_num_predict(tmp_path: Path) -> None:
    from assistant.conversation import _REWRITE_TEMPERATURE, _REWRITE_MAX_OUTPUT_TOKENS

    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=[_VALID_FORMAL_REWRITE, "Podias mencionar o orçamento com mais destaque."],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")

    engine.respond("lê o ficheiro email_resumo_novo e diz exatamente o que lá está escrito")
    engine.respond("Torna-o mais formal, mas não guardes ainda.")
    engine.respond("Tens alguma sugestão para melhorar este email?")

    assert ollama.calls[0]["temperature"] == _REWRITE_TEMPERATURE
    assert ollama.calls[0]["num_predict"] == _REWRITE_MAX_OUTPUT_TOKENS
    # The unrelated "review" call right after must not inherit the rewrite's
    # tighter generation parameters.
    assert ollama.calls[1]["temperature"] is None
    assert ollama.calls[1]["num_predict"] is None


def test_document_rewrite_failure_message_reflects_real_reason(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=[
            "ASSUNTO: Resumo da Reunião de Direção - 19 de junho\n\nFrancisca,\n\nSegue o resumo.\n\n"
            "- Febrada final de época\n- Protocolo Águas de Coimbra\n- Candidatura COP\n- Sanfil\n\n"
            "Atenciosamente,\n[nome]",
            "ASSUNTO: Resumo da Reunião de Direção - 19 de junho\n\nFrancisca,\n\nSegue o resumo.\n\n"
            "- Febrada final de época\n- Protocolo Águas de Coimbra\n- Candidatura COP\n- Sanfil\n\n"
            "Atenciosamente,\n<nome>",
        ],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")

    engine.respond("lê o ficheiro email_resumo_novo e diz exatamente o que lá está escrito")
    response = engine.respond("Torna-o mais formal, mas não guardes ainda.")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["draft_validation_failed"] is True
    assert telemetry["draft_validation_reason"] == "placeholder_detected"
    assert "campos por preencher" in response
    assert "comentário" not in response.lower()
    assert engine._active_document_context.draft_content == ""


# --- Rewrite reliability fix #2: cooperative cancellation, total timeout,
# and progress events. Reproduces a document rewrite no longer looking
# "stuck" during slow Ollama calls: cancellation is checked at fixed points
# instead of trying to interrupt an in-flight HTTP call, and a proactive
# total-timeout budget prevents a second attempt from starting when there's
# not enough time left for it to plausibly finish.


def test_cancel_current_request_returns_false_when_no_active_token(tmp_path: Path) -> None:
    engine, *_ = make_engine(tmp_path)

    assert engine.cancel_current_request() is False
    assert engine.is_cancel_requested() is False


def test_begin_request_resets_cancellation_between_requests(tmp_path: Path) -> None:
    engine, *_ = make_engine(tmp_path)

    engine.begin_request()
    assert engine.cancel_current_request() is True
    assert engine.is_cancel_requested() is True

    engine.begin_request()
    assert engine.is_cancel_requested() is False


def test_document_rewrite_cancelled_before_first_call_makes_no_llm_calls(tmp_path: Path, monkeypatch) -> None:
    import assistant.conversation as conversation_module

    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=[_VALID_FORMAL_REWRITE],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")
    engine.respond("lê o ficheiro email_resumo_novo e diz exatamente o que lá está escrito")

    real_prompt_fn = conversation_module._document_rewrite_prompt

    def _prompt_with_cancel(request, context):
        engine.cancel_current_request()
        return real_prompt_fn(request, context)

    monkeypatch.setattr(conversation_module, "_document_rewrite_prompt", _prompt_with_cancel)

    response = engine.respond("Torna-o mais formal, mas não guardes ainda.")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert response == "Pedido cancelado."
    assert telemetry["cancellation_requested"] is True
    assert telemetry["request_cancelled"] is True
    assert telemetry["draft_created"] is False
    assert engine._active_document_context.draft_content == ""
    assert len(ollama.calls) == 0
    assert anthropic.calls == []


def test_document_rewrite_cancelled_after_first_invalid_attempt_skips_regeneration(tmp_path: Path, monkeypatch) -> None:
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=[
            "O documento parece pronto para enviar! É para enviar a algum destinatário?",
            _VALID_FORMAL_REWRITE,
        ],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")
    engine.respond("lê o ficheiro email_resumo_novo e diz exatamente o que lá está escrito")

    real_chat = ollama.chat

    def _chat_then_cancel(*args, **kwargs):
        result = real_chat(*args, **kwargs)
        engine.cancel_current_request()
        return result

    monkeypatch.setattr(ollama, "chat", _chat_then_cancel)

    response = engine.respond("Torna-o mais formal, mas não guardes ainda.")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert response == "Pedido cancelado."
    assert telemetry["request_cancelled"] is True
    assert engine._active_document_context.draft_content == ""
    assert len(ollama.calls) == 1
    assert anthropic.calls == []


def test_document_rewrite_cancelled_after_valid_generation_creates_no_draft(tmp_path: Path, monkeypatch) -> None:
    import assistant.conversation as conversation_module

    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=[_VALID_FORMAL_REWRITE],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")
    engine.respond("lê o ficheiro email_resumo_novo e diz exatamente o que lá está escrito")

    real_validate = conversation_module._validate_rewrite_draft

    def _validate_then_cancel(original_content, candidate):
        result = real_validate(original_content, candidate)
        if result[0]:
            engine.cancel_current_request()
        return result

    monkeypatch.setattr(conversation_module, "_validate_rewrite_draft", _validate_then_cancel)

    response = engine.respond("Torna-o mais formal, mas não guardes ainda.")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert response == "Pedido cancelado."
    assert telemetry["request_cancelled"] is True
    assert telemetry["draft_created"] is False
    assert engine._active_document_context.draft_content == ""
    assert len(ollama.calls) == 1


def test_document_rewrite_total_timeout_before_second_attempt(tmp_path: Path, monkeypatch) -> None:
    import time as time_module

    import assistant.conversation as conversation_module

    monkeypatch.setattr(conversation_module, "DOCUMENT_REWRITE_TOTAL_TIMEOUT_SECONDS", 0.3)
    monkeypatch.setattr(conversation_module, "_REWRITE_MIN_SECONDS_FOR_ATTEMPT", 0.2)

    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=[
            "O documento parece pronto para enviar! É para enviar a algum destinatário?",
            _VALID_FORMAL_REWRITE,
        ],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")
    original_bytes = (engine.workspace_path / "email_resumo_novo.txt").read_bytes()
    engine.respond("lê o ficheiro email_resumo_novo e diz exatamente o que lá está escrito")

    real_chat = ollama.chat

    def _slow_chat(*args, **kwargs):
        time_module.sleep(0.25)
        return real_chat(*args, **kwargs)

    monkeypatch.setattr(ollama, "chat", _slow_chat)

    response = engine.respond("Torna-o mais formal, mas não guardes ainda.")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["rewrite_timeout"] is True
    assert telemetry["draft_created"] is False
    assert telemetry["request_cancelled"] is False
    assert len(ollama.calls) == 1
    assert "demorou mais do que o previsto" in response
    assert (engine.workspace_path / "email_resumo_novo.txt").read_bytes() == original_bytes


def test_document_rewrite_timeout_seconds_is_bounded_and_other_calls_unaffected(tmp_path: Path) -> None:
    from assistant.conversation import DOCUMENT_REWRITE_TOTAL_TIMEOUT_SECONDS

    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=[_VALID_FORMAL_REWRITE, "Podias mencionar o orçamento com mais destaque."],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")

    engine.respond("lê o ficheiro email_resumo_novo e diz exatamente o que lá está escrito")
    engine.respond("Torna-o mais formal, mas não guardes ainda.")
    engine.respond("Tens alguma sugestão para melhorar este email?")

    assert ollama.calls[0]["timeout_seconds"] is not None
    assert 0 < ollama.calls[0]["timeout_seconds"] <= DOCUMENT_REWRITE_TOTAL_TIMEOUT_SECONDS
    # The unrelated "review" call right after must not inherit a bounded
    # per-call timeout -- it keeps using the provider's own default.
    assert ollama.calls[1]["timeout_seconds"] is None


def test_document_rewrite_success_reports_clean_finish_and_no_cancellation_telemetry(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=[_VALID_FORMAL_REWRITE],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")

    engine.respond("lê o ficheiro email_resumo_novo e diz exatamente o que lá está escrito")
    engine.respond("Torna-o mais formal, mas não guardes ainda.")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["cancellation_requested"] is False
    assert telemetry["request_cancelled"] is False
    assert telemetry["rewrite_timeout"] is False
    assert telemetry["rewrite_regeneration_started"] is False
    assert telemetry["request_finished_cleanly"] is True
    assert len(ollama.calls) == 1


def test_document_rewrite_next_request_after_cancellation_is_not_affected(tmp_path: Path, monkeypatch) -> None:
    import assistant.conversation as conversation_module

    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=[_VALID_FORMAL_REWRITE, _VALID_FORMAL_REWRITE],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")
    engine.respond("lê o ficheiro email_resumo_novo e diz exatamente o que lá está escrito")

    real_prompt_fn = conversation_module._document_rewrite_prompt

    def _prompt_with_cancel(request, context):
        engine.cancel_current_request()
        return real_prompt_fn(request, context)

    monkeypatch.setattr(conversation_module, "_document_rewrite_prompt", _prompt_with_cancel)
    cancelled_response = engine.respond("Torna-o mais formal, mas não guardes ainda.")
    assert cancelled_response == "Pedido cancelado."
    assert len(ollama.calls) == 0

    monkeypatch.setattr(conversation_module, "_document_rewrite_prompt", real_prompt_fn)
    second_response = engine.respond("Torna-o mais formal, mas não guardes ainda.")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert engine.is_cancel_requested() is False
    assert second_response != "Pedido cancelado."
    assert "Prezada Francisca" in second_response
    assert telemetry["draft_created"] is True
    assert len(ollama.calls) == 1
    assert anthropic.calls == []


def test_document_rewrite_cancelling_new_request_preserves_prior_valid_draft(tmp_path: Path, monkeypatch) -> None:
    import assistant.conversation as conversation_module

    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=[_VALID_FORMAL_REWRITE, _VALID_FORMAL_REWRITE],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")
    engine.respond("lê o ficheiro email_resumo_novo e diz exatamente o que lá está escrito")
    engine.respond("Torna-o mais formal, mas não guardes ainda.")

    assert engine._active_document_context.draft_content == _VALID_FORMAL_REWRITE

    # A second rewrite request while a draft already exists is now a
    # refinement of that draft (not a fresh rewrite of the original), so it
    # builds its first-attempt prompt via _document_refinement_prompt.
    real_prompt_fn = conversation_module._document_refinement_prompt

    def _prompt_with_cancel(request, context, base_content):
        engine.cancel_current_request()
        return real_prompt_fn(request, context, base_content)

    monkeypatch.setattr(conversation_module, "_document_refinement_prompt", _prompt_with_cancel)
    response = engine.respond("Torna-o mais formal outra vez, mas não guardes ainda.")

    assert response == "Pedido cancelado."
    assert engine._active_document_context.draft_content == _VALID_FORMAL_REWRITE


def test_document_rewrite_emits_regeneration_started_progress_event(tmp_path: Path) -> None:
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=[
            "O documento parece pronto para enviar! É para enviar a algum destinatário?",
            _VALID_FORMAL_REWRITE,
        ],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")
    engine.respond("lê o ficheiro email_resumo_novo e diz exatamente o que lá está escrito")

    events: list[str] = []
    engine.set_progress_listener(events.append)
    try:
        engine.respond("Torna-o mais formal, mas não guardes ainda.")
    finally:
        engine.set_progress_listener(None)

    assert "rewrite_attempt_started" in events
    assert "rewrite_regeneration_started" in events


def test_document_rewrite_emits_cancelled_progress_event(tmp_path: Path, monkeypatch) -> None:
    import assistant.conversation as conversation_module

    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=[_VALID_FORMAL_REWRITE],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")
    engine.respond("lê o ficheiro email_resumo_novo e diz exatamente o que lá está escrito")

    real_prompt_fn = conversation_module._document_rewrite_prompt

    def _prompt_with_cancel(request, context):
        engine.cancel_current_request()
        return real_prompt_fn(request, context)

    monkeypatch.setattr(conversation_module, "_document_rewrite_prompt", _prompt_with_cancel)

    events: list[str] = []
    engine.set_progress_listener(events.append)
    try:
        response = engine.respond("Torna-o mais formal, mas não guardes ainda.")
    finally:
        engine.set_progress_listener(None)

    assert response == "Pedido cancelado."
    assert "rewrite_cancelled" in events


def test_document_rewrite_progress_events_never_reach_conversation_history(tmp_path: Path) -> None:
    """16.J: progress events (and the frontend's auxiliary progress labels
    derived from them) drive the entity/status-line only -- the actual
    conversation history must contain neither the raw event names nor the
    Portuguese label text a UI would show for them."""
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=[
            "O documento parece pronto para enviar! É para enviar a algum destinatário?",
            _VALID_FORMAL_REWRITE,
        ],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")

    events: list[str] = []
    engine.set_progress_listener(events.append)
    try:
        engine.respond("lê o ficheiro email_resumo_novo e diz exatamente o que lá está escrito")
        engine.respond("Torna-o mais formal, mas não guardes ainda.")
    finally:
        engine.set_progress_listener(None)

    assert "rewrite_regeneration_started" in events

    history_text = " ".join(f"{entry.get('role')}:{entry.get('content')}" for entry in engine.memory.load())
    forbidden_fragments = (
        "rewrite_attempt_started",
        "rewrite_validation_started",
        "rewrite_regeneration_started",
        "rewrite_cancelled",
        "rewrite_timeout",
        "A reescrever",
        "A validar a versão",
        "A tentar novamente",
        "A cancelar",
    )
    for fragment in forbidden_fragments:
        assert fragment not in history_text

# --- Iterative document refinement fix: "ainda consegues melhor" and other
# elliptical/comparative continuations of an open draft thread must be
# recognized as "improve the current draft further", not fall through to
# GENERAL_CONVERSATION. Reproduces the reported regression exactly.

_VALID_REFINED_REWRITE = (
    "Exma. Senhora Francisca,\n\nVenho por este meio apresentar o resumo formal da reunião de Direção, "
    "conforme solicitado anteriormente.\n\n"
    "- Conclusão da febrada final de época\n- Protocolo Águas de Coimbra\n- Candidatura COP\n- Sanfil\n\n"
    "Com os melhores cumprimentos,\nAlexandre"
)


def _setup_engine_with_draft(tmp_path, ollama_replies):
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=ollama_replies,
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")
    engine.respond("lê o ficheiro email_resumo_novo e diz exatamente o que lá está escrito")
    engine.respond("Torna-o mais formal, mas não guardes ainda.")
    first_telemetry = engine.get_last_turn_telemetry() or {}
    assert first_telemetry["draft_revision_number"] == 1
    return engine, ollama, anthropic


def test_draft_refinement_ainda_consegues_melhor_is_document_task(tmp_path: Path) -> None:
    """A: draft ativo + 'ainda consegues melhor' -> DOCUMENT_TASK, usa o
    draft como base, não GENERAL_CONVERSATION, cria revision 2."""
    engine, ollama, anthropic = _setup_engine_with_draft(
        tmp_path, [_VALID_FORMAL_REWRITE, _VALID_REFINED_REWRITE]
    )

    response = engine.respond("ainda consegues melhor")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["response_source"] == "DOCUMENT_TASK"
    assert "WORKSPACE_FILE" in telemetry["grounding_sources"]
    assert telemetry["memory_route"] == "NONE"
    assert telemetry["document_followup_action"] == "iterative_refinement"
    assert telemetry["document_refinement_source"] == "draft"
    assert telemetry["previous_draft_present"] is True
    assert telemetry["draft_revision_number"] == 2
    assert engine._active_document_context.draft_content == _VALID_REFINED_REWRITE
    assert "Alexandre" in response
    assert len(ollama.calls) == 2  # 1 from the initial rewrite (setup) + 1 for this refinement
    assert anthropic.calls == []


def test_draft_refinement_ainda_consegues_melhorar_variant_with_r(tmp_path: Path) -> None:
    """A (variant): the "-ar" verb form used in this task's own manual-test
    wording ("ainda consegues melhorar") must resolve identically to the
    adjective form ("ainda consegues melhor")."""
    engine, ollama, anthropic = _setup_engine_with_draft(
        tmp_path, [_VALID_FORMAL_REWRITE, _VALID_REFINED_REWRITE]
    )

    engine.respond("ainda consegues melhorar")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["document_followup_action"] == "iterative_refinement"
    assert telemetry["draft_revision_number"] == 2


def test_draft_refinement_consegues_fazer_melhor(tmp_path: Path) -> None:
    """B: 'consegues fazer melhor?' is also recognized as refinement."""
    engine, ollama, anthropic = _setup_engine_with_draft(
        tmp_path, [_VALID_FORMAL_REWRITE, _VALID_REFINED_REWRITE]
    )

    telemetry = engine.get_last_turn_telemetry() or {}
    engine.respond("consegues fazer melhor?")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["document_followup_action"] == "iterative_refinement"
    assert telemetry["draft_revision_number"] == 2


def test_draft_refinement_mais_formal_with_draft(tmp_path: Path) -> None:
    """C: 'mais formal' alone is refinement when a draft already exists."""
    engine, ollama, anthropic = _setup_engine_with_draft(
        tmp_path, [_VALID_FORMAL_REWRITE, _VALID_REFINED_REWRITE]
    )

    engine.respond("mais formal")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["document_followup_action"] == "iterative_refinement"


def test_draft_refinement_phrase_without_draft_does_not_invent_context(tmp_path: Path) -> None:
    """D: the same phrase, with no active document/draft, must not invent a
    document context -- it should NOT become DOCUMENT_TASK."""
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=["O Alexandre está sempre a procurar formas de melhoria, não é?"],
    )

    response = engine.respond("ainda consegues melhor")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] != "DOCUMENT_TASK"
    assert telemetry.get("document_followup_action") in ("", None)
    assert engine._active_document_context is None
    assert response


def test_draft_refinement_uses_draft_not_original_as_prompt_base(tmp_path: Path, monkeypatch) -> None:
    """E: the refinement prompt is built from the current draft, and the
    original file on disk is never touched."""
    import assistant.conversation as conversation_module

    engine, ollama, anthropic = _setup_engine_with_draft(
        tmp_path, [_VALID_FORMAL_REWRITE, _VALID_REFINED_REWRITE]
    )
    original_bytes = (engine.workspace_path / "email_resumo_novo.txt").read_bytes()

    seen_base_content = {}
    real_prompt_fn = conversation_module._document_refinement_prompt

    def _capture(request, context, base_content):
        seen_base_content["value"] = base_content
        return real_prompt_fn(request, context, base_content)

    monkeypatch.setattr(conversation_module, "_document_refinement_prompt", _capture)
    engine.respond("ainda consegues melhor")

    assert seen_base_content["value"] == _VALID_FORMAL_REWRITE
    assert (engine.workspace_path / "email_resumo_novo.txt").read_bytes() == original_bytes
    assert engine._active_document_context.content == REAL_EMAIL_SUMMARY


def test_draft_refinement_rejects_noop_and_regenerates_once(tmp_path: Path) -> None:
    """F: a structurally valid but practically identical "improvement" is
    rejected as a no-op; exactly one regeneration is attempted."""
    engine, ollama, anthropic = _setup_engine_with_draft(
        tmp_path, [_VALID_FORMAL_REWRITE, _VALID_FORMAL_REWRITE, _VALID_FORMAL_REWRITE]
    )

    response = engine.respond("ainda consegues melhor")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["draft_validation_failed"] is True
    assert telemetry["draft_validation_reason"] == "no_op_detected"
    assert telemetry["draft_regeneration_attempted"] is True
    assert telemetry["refinement_changed_content"] is False
    assert telemetry["draft_revision_number"] == 1
    assert engine._active_document_context.draft_content == _VALID_FORMAL_REWRITE
    assert "não houve uma melhoria real" in response
    # 1 (first rewrite) + 2 (refinement attempt + its regeneration) = 3 total.
    assert len(ollama.calls) == 3


def test_draft_refinement_accepts_second_attempt_when_genuinely_different(tmp_path: Path) -> None:
    """G: if the first refinement attempt is a no-op but the regenerated
    second attempt is a real change, it is accepted and the draft updates."""
    engine, ollama, anthropic = _setup_engine_with_draft(
        tmp_path, [_VALID_FORMAL_REWRITE, _VALID_FORMAL_REWRITE, _VALID_REFINED_REWRITE]
    )

    response = engine.respond("ainda consegues melhor")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["draft_validation_passed"] is True
    assert telemetry["draft_regeneration_attempted"] is True
    assert telemetry["refinement_regeneration_started"] is True
    assert telemetry["refinement_changed_content"] is True
    assert telemetry["draft_revision_number"] == 2
    assert engine._active_document_context.draft_content == _VALID_REFINED_REWRITE
    assert "Alexandre" in response
    assert len(ollama.calls) == 3  # H: never more than 2 calls per turn (1 + 2)


def test_show_improved_version_returns_revision_2_without_llm_call(tmp_path: Path) -> None:
    """I: 'mostra a versão melhorada' returns the latest revision with 0 LLM
    calls -- pure lookup, no generation."""
    engine, ollama, anthropic = _setup_engine_with_draft(
        tmp_path, [_VALID_FORMAL_REWRITE, _VALID_REFINED_REWRITE]
    )
    engine.respond("ainda consegues melhor")

    response = engine.respond("mostra a versão melhorada")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["llm_calls"] == 0
    assert _VALID_REFINED_REWRITE in response
    assert telemetry["draft_revision_number"] == 2


def test_show_original_returns_untouched_original(tmp_path: Path) -> None:
    """J: 'mostra o original' always returns the real original file,
    unaffected by however many refinements happened."""
    engine, ollama, anthropic = _setup_engine_with_draft(
        tmp_path, [_VALID_FORMAL_REWRITE, _VALID_REFINED_REWRITE]
    )
    engine.respond("ainda consegues melhor")
    original_bytes = (engine.workspace_path / "email_resumo_novo.txt").read_bytes()

    response = engine.respond("mostra o original")

    assert "Febrada final de época" in response
    assert "Olá Francisca" in response
    assert (engine.workspace_path / "email_resumo_novo.txt").read_bytes() == original_bytes


def test_compare_versions_compares_original_with_revision_2(tmp_path: Path) -> None:
    """K: 'compara as versões' compares the true original with the current
    (revision 2) draft."""
    engine, ollama, anthropic = _setup_engine_with_draft(
        tmp_path, [_VALID_FORMAL_REWRITE, _VALID_REFINED_REWRITE]
    )
    engine.respond("ainda consegues melhor")

    response = engine.respond("compara as versões")

    assert "Olá Francisca" in response
    assert _VALID_REFINED_REWRITE in response


def test_draft_refinement_survives_unrelated_side_question(tmp_path: Path) -> None:
    """L: a side question in between still lets a later refinement request
    use the active draft when the reference is clear."""
    engine, ollama, anthropic = _setup_engine_with_draft(
        tmp_path, [_VALID_FORMAL_REWRITE, _VALID_REFINED_REWRITE]
    )

    engine.respond("Que horas são?")
    engine.respond("ainda consegues melhor")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["document_followup_action"] == "iterative_refinement"
    assert telemetry["draft_revision_number"] == 2
    assert engine._active_document_context.draft_content == _VALID_REFINED_REWRITE


def test_draft_refinement_cancellation_still_works(tmp_path: Path, monkeypatch) -> None:
    """M: progress events and cancellation continue to work for a
    refinement turn, not just a plain first-time rewrite."""
    import assistant.conversation as conversation_module

    engine, ollama, anthropic = _setup_engine_with_draft(
        tmp_path, [_VALID_FORMAL_REWRITE, _VALID_REFINED_REWRITE]
    )

    real_prompt_fn = conversation_module._document_refinement_prompt

    def _prompt_with_cancel(request, context, base_content):
        engine.cancel_current_request()
        return real_prompt_fn(request, context, base_content)

    monkeypatch.setattr(conversation_module, "_document_refinement_prompt", _prompt_with_cancel)

    events: list[str] = []
    engine.set_progress_listener(events.append)
    try:
        response = engine.respond("ainda consegues melhor")
    finally:
        engine.set_progress_listener(None)

    telemetry = engine.get_last_turn_telemetry() or {}
    assert response == "Pedido cancelado."
    assert "rewrite_cancelled" in events
    assert telemetry["request_cancelled"] is True
    assert telemetry["document_followup_action"] == "iterative_refinement"
    assert telemetry["draft_revision_number"] == 1  # unchanged -- cancelled before commit
    assert engine._active_document_context.draft_content == _VALID_FORMAL_REWRITE


# --- Automatic-mode escalation for complex document tasks: the router must
# score the RECONSTRUCTED task (draft presence, revision, prior local
# failure), not just the literal length of a short follow-up like "ainda
# consegues melhor". All Anthropic responses here go through FakeProvider --
# no real network call is ever made, even when paid calls are "authorized".

_PAID_ENV = {"ANTHROPIC_API_KEY": "secret", PAID_CALL_CONFIRMATION_ENV: "true"}


def test_escalation_simple_greeting_stays_local_low_complexity(tmp_path: Path) -> None:
    """A: a plain greeting is local/low-complexity even with paid calls on."""
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path, mode="automatic", claude_enabled=True, env=_PAID_ENV, ollama_replies=["Olá! Como posso ajudar?"],
    )

    engine.respond("olá")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry.get("llm_calls", 0) == 0
    assert anthropic.calls == []


def test_escalation_show_original_uses_zero_llm_calls(tmp_path: Path) -> None:
    """B: 'mostra o original' is a deterministic tool lookup -- 0 LLM calls,
    never Claude, regardless of automatic/paid-call configuration."""
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic", claude_enabled=True, env=_PAID_ENV)
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")

    engine.respond("lê o ficheiro email_resumo_novo e mostra tudo sem resumir")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["llm_calls"] == 0
    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert ollama.calls == []
    assert anthropic.calls == []


def test_escalation_simple_short_rewrite_stays_on_ollama(tmp_path: Path) -> None:
    """C: a small, structurally simple first-time rewrite is Ollama-
    preferred (local-first) even with Claude fully authorized."""
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path, mode="automatic", claude_enabled=True, env=_PAID_ENV, ollama_replies=[_VALID_FORMAL_REWRITE],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")
    engine.respond("lê o ficheiro email_resumo_novo e mostra tudo sem resumir")

    engine.respond("Torna-o mais formal, mas não guardes ainda.")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["final_provider"] == "ollama"
    assert telemetry["escalation_selected"] is False
    assert len(ollama.calls) == 1
    assert anthropic.calls == []


def test_escalation_iterative_refinement_selects_claude_when_authorized(tmp_path: Path) -> None:
    """D + M: 'ainda consegues melhor' with an active draft reconstructs a
    high-complexity task (task_complexity_score computed from the profile,
    not the 3-word message) and escalates to Claude -- simulated via
    FakeProvider, no real Anthropic call."""
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=True,
        env=_PAID_ENV,
        ollama_replies=[_VALID_FORMAL_REWRITE],
        anthropic_replies=[_VALID_REFINED_REWRITE],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")
    engine.respond("lê o ficheiro email_resumo_novo e mostra tudo sem resumir")
    engine.respond("Torna-o mais formal, mas não guardes ainda.")

    response = engine.respond("ainda consegues melhor")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "DOCUMENT_TASK"
    assert telemetry["final_provider"] == "anthropic"
    assert telemetry["escalation_selected"] is True
    assert telemetry["escalation_reason"] == "iterative_refinement_high_complexity"
    assert telemetry["task_complexity_score"] > 40.0
    assert telemetry["document_task_type"] == "document_refinement"
    assert telemetry["draft_revision_number"] == 2
    assert "WORKSPACE_FILE" in telemetry["grounding_sources"]
    assert engine._active_document_context.draft_content == _VALID_REFINED_REWRITE
    assert "Alexandre" in response
    assert len(anthropic.calls) == 1
    # Attempt 1 of the refinement itself never touched Ollama -- only the
    # earlier plain rewrite did.
    assert len(ollama.calls) == 1


def test_escalation_blocked_without_paid_calls_confirmed(tmp_path: Path) -> None:
    """E: the same high-complexity refinement, but paid calls are not
    confirmed -- stays on Ollama with a clear blocked reason_code."""
    env = {"ANTHROPIC_API_KEY": "secret"}
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=True,
        env=env,
        ollama_replies=[_VALID_FORMAL_REWRITE, _VALID_REFINED_REWRITE],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")
    engine.respond("lê o ficheiro email_resumo_novo e mostra tudo sem resumir")
    engine.respond("Torna-o mais formal, mas não guardes ainda.")

    engine.respond("ainda consegues melhor")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["final_provider"] == "ollama"
    assert telemetry["escalation_selected"] is False
    assert anthropic.calls == []


def test_escalation_no_real_anthropic_provider_ever_used(tmp_path: Path) -> None:
    """O: across every scenario above, the ONLY Anthropic stand-in is
    FakeProvider -- confirms make_engine never wires a real AnthropicProvider
    capable of a network call."""
    _engine, llm, _ollama, _anthropic = make_engine(tmp_path, mode="automatic", claude_enabled=True, env=_PAID_ENV)

    assert type(llm.providers["anthropic"]).__name__ == "FakeProvider"


# --- Fixed telemetry bug: a real paid test showed a Claude-then-Claude retry
# mislabeled as "Tentativa local falhou a validacao" even though no local
# (Ollama) attempt ever happened that turn. previous_provider is now tracked
# per attempt so the reason code always matches which provider actually
# failed. All Anthropic replies here are FakeProvider -- no real call (I).

def test_escalation_local_failure_then_claude_reports_correct_sequence(tmp_path: Path) -> None:
    """A: a first-time rewrite fails validation on Ollama, then genuinely
    escalates to Claude on attempt 2 -- sequence=[ollama, anthropic],
    reason=document_escalated_after_local_failure (the failure really was
    local this time)."""
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=True,
        env=_PAID_ENV,
        ollama_replies=[
            "O documento parece pronto para enviar! E para enviar a algum destinatario ou e apenas um resumo interno?"
        ],
        anthropic_replies=[_VALID_FORMAL_REWRITE],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")
    engine.respond("lê o ficheiro email_resumo_novo e mostra tudo sem resumir")

    response = engine.respond("Torna-o mais formal, mas não guardes ainda.")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["initial_provider"] == "ollama"
    assert telemetry["final_provider"] == "anthropic"
    assert telemetry["provider_attempt_sequence"] == ["ollama", "anthropic"]
    assert telemetry["attempt_count"] == 2
    assert telemetry["attempt_failure_reasons"] == ["termina_em_pergunta", ""]
    assert telemetry["final_routing_reason_code"] == "document_escalated_after_local_failure"
    assert telemetry["document_routing_features"]["document_previous_provider"] == "ollama"
    assert telemetry["document_routing_features"]["document_previous_local_failure"] is True
    assert len(ollama.calls) == 1
    assert len(anthropic.calls) == 1
    assert "Prezada Francisca" in response


def test_escalation_claude_failure_regenerates_with_claude_not_local(tmp_path: Path) -> None:
    """B: the FIRST refinement attempt already escalated to Claude (high
    complexity is inherent to refinement) and failed validation -- the retry
    stays on Claude, never mislabeled as a local failure. sequence=
    [anthropic, anthropic]; reason=document_regenerated_with_claude;
    previous_local_failure stays false."""
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=True,
        env=_PAID_ENV,
        ollama_replies=[_VALID_FORMAL_REWRITE],
        anthropic_replies=[
            "Isto ja parece uma melhoria bastante razoavel em relacao a versao anterior, mas ainda tenho "
            "duvidas sobre se deves querer ajustar mais algum pormenor antes de ficares satisfeita com o "
            "resultado final?",
            _VALID_REFINED_REWRITE,
        ],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")
    engine.respond("lê o ficheiro email_resumo_novo e mostra tudo sem resumir")
    engine.respond("Torna-o mais formal, mas não guardes ainda.")

    response = engine.respond("ainda consegues melhor")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["initial_provider"] == "anthropic"
    assert telemetry["final_provider"] == "anthropic"
    assert telemetry["provider_attempt_sequence"] == ["anthropic", "anthropic"]
    assert telemetry["attempt_count"] == 2
    assert telemetry["final_routing_reason_code"] == "document_regenerated_with_claude"
    # Preserves the FIRST attempt's own reason too, not just the last one.
    assert telemetry["escalation_reason"] == "iterative_refinement_high_complexity"
    assert telemetry["document_routing_features"]["document_previous_provider"] == "anthropic"
    assert telemetry["document_routing_features"]["document_previous_local_failure"] is False
    assert telemetry["document_routing_features"]["document_previous_validation_failed"] is True
    assert len(ollama.calls) == 1  # only the earlier plain rewrite, never a local retry here
    assert len(anthropic.calls) == 2
    assert "Alexandre" in response


def test_escalation_claude_valid_first_attempt_no_regeneration(tmp_path: Path) -> None:
    """C: Claude gets it right on the first refinement attempt --
    sequence=[anthropic], attempt_count=1, no regeneration reason code."""
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=True,
        env=_PAID_ENV,
        ollama_replies=[_VALID_FORMAL_REWRITE],
        anthropic_replies=[_VALID_REFINED_REWRITE],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")
    engine.respond("lê o ficheiro email_resumo_novo e mostra tudo sem resumir")
    engine.respond("Torna-o mais formal, mas não guardes ainda.")

    engine.respond("ainda consegues melhor")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["provider_attempt_sequence"] == ["anthropic"]
    assert telemetry["attempt_count"] == 1
    assert telemetry["attempt_failure_reasons"] == [""]
    assert telemetry["final_routing_reason_code"] == "iterative_refinement_high_complexity"
    assert len(anthropic.calls) == 1
    assert len(ollama.calls) == 1


def test_local_regeneration_sequence_when_claude_unavailable(tmp_path: Path) -> None:
    """D: a first attempt fails locally and, with Claude unavailable, the
    retry also stays local -- sequence=[ollama, ollama], never touching
    Claude at all."""
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=False,
        ollama_replies=[
            "O documento parece pronto para enviar! E para enviar a algum destinatario ou e apenas um resumo interno?",
            _VALID_FORMAL_REWRITE,
        ],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")
    engine.respond("lê o ficheiro email_resumo_novo e mostra tudo sem resumir")

    engine.respond("Torna-o mais formal, mas não guardes ainda.")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["provider_attempt_sequence"] == ["ollama", "ollama"]
    assert telemetry["attempt_count"] == 2
    assert telemetry["final_provider"] == "ollama"
    assert len(ollama.calls) == 2
    assert anthropic.calls == []


def test_document_previous_local_failure_requires_local_provider(tmp_path: Path) -> None:
    """E: document_previous_local_failure is only true when the previous
    attempt was actually local -- a previous Claude failure never sets it,
    even though both are validation failures."""
    router = ModelRouter(
        ModelRoutingConfig(
            mode="automatic",
            automatic=AutomaticRoutingConfig(claude_enabled=True, daily_budget_usd=0.25, max_single_call_estimated_usd=0.05),
        ),
        env=_PAID_ENV,
        budget=ModelUsageBudget(tmp_path / "u.json"),
    )
    base_profile = {
        "task_type": "document_refinement",
        "document_chars": 200,
        "document_named_entity_count": 2,
        "document_list_item_count": 4,
        "document_regeneration_attempt": 2,
        "document_validation_failure_reason": "termina_em_pergunta",
        "document_previous_validation_failed": True,
        "preferred_provider": "auto",
    }

    from assistant.model_router import ModelRoutingInput

    local_failure_decision = router.decide(
        ModelRoutingInput(
            source="RESPONSE_COMPOSER",
            user_message="ainda consegues melhor",
            task_profile={**base_profile, "document_previous_provider": "ollama", "document_previous_local_failure": True},
        )
    )
    claude_failure_decision = router.decide(
        ModelRoutingInput(
            source="RESPONSE_COMPOSER",
            user_message="ainda consegues melhor",
            task_profile={**base_profile, "document_previous_provider": "anthropic", "document_previous_local_failure": False},
        )
    )

    assert local_failure_decision.reason_code == "document_escalated_after_local_failure"
    assert claude_failure_decision.reason_code == "document_regenerated_with_claude"


def test_escalation_budget_accounted_per_call_not_per_turn(tmp_path: Path) -> None:
    """F: a Claude-then-Claude turn registers TWO calls against the budget
    ledger, not one -- the budget is spent per actual paid call."""
    import json

    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=True,
        env=_PAID_ENV,
        ollama_replies=[_VALID_FORMAL_REWRITE],
        anthropic_replies=[
            "Isto ja parece uma melhoria bastante razoavel em relacao a versao anterior, mas ainda tenho "
            "duvidas sobre se deves querer ajustar mais algum pormenor antes de ficares satisfeita com o "
            "resultado final?",
            _VALID_REFINED_REWRITE,
        ],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")
    engine.respond("lê o ficheiro email_resumo_novo e mostra tudo sem resumir")
    engine.respond("Torna-o mais formal, mas não guardes ainda.")

    engine.respond("ainda consegues melhor")

    usage = json.loads((tmp_path / "usage.json").read_text(encoding="utf-8"))
    assert usage["calls"] == 2
    assert usage["providers"]["anthropic"]["calls"] == 2
    assert len(anthropic.calls) == 2


def test_escalation_never_exceeds_two_calls_per_turn(tmp_path: Path) -> None:
    """G: regardless of how the attempts split across providers, a single
    turn never makes more than 2 LLM calls in total."""
    engine, _llm, ollama, anthropic = make_engine(
        tmp_path,
        mode="automatic",
        claude_enabled=True,
        env=_PAID_ENV,
        ollama_replies=[
            "O documento parece pronto para enviar! E para enviar a algum destinatario ou e apenas um resumo interno?"
        ],
        anthropic_replies=[_VALID_FORMAL_REWRITE],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")
    engine.respond("lê o ficheiro email_resumo_novo e mostra tudo sem resumir")

    engine.respond("Torna-o mais formal, mas não guardes ainda.")

    assert len(ollama.calls) + len(anthropic.calls) == 2


def test_rewrite_and_refinement_prompts_preserve_tense_and_chronology(tmp_path: Path) -> None:
    """H: every rewrite/refinement/correction prompt explicitly instructs the
    model to preserve verb tenses and factual chronology -- a real paid test
    showed a completed action ("contactou") rewritten into the present
    ("contacta")."""
    import assistant.conversation as conversation_module

    request = conversation_module.DocumentTaskRequest(
        source_files=["doc.txt"], actions=["read_source", "rewrite"], instructions="torna-o mais formal",
        action="rewrite", requested_output="",
    )
    context = conversation_module.ActiveDocumentContext(resolved_file="doc.txt", content="Conteúdo original.")

    fidelity_marker = "tempos verbais"
    assert fidelity_marker in conversation_module._document_rewrite_prompt(request, context)
    assert fidelity_marker in conversation_module._document_rewrite_correction_prompt(request, context, "resposta invalida", "termina_em_pergunta")
    assert fidelity_marker in conversation_module._document_refinement_prompt(request, context, "Versão atual.")
    assert fidelity_marker in conversation_module._document_refinement_correction_prompt(
        request, context, "Versão atual.", "resposta invalida", "termina_em_pergunta"
    )


# --- Passo 3: workspace adaptativo e externalização do trabalho. Minimal,
# additive UI events the frontend uses to create/update DOCUMENT/DRAFT/
# COMPARISON workspace objects. Never routing/scoring/memory-affecting --
# these only carry fields already computed for the linear text response.

def test_workspace_document_loaded_event_on_read(tmp_path: Path) -> None:
    """A: reading a workspace document queues a document_loaded UI event
    with a truncated preview, not the full content."""
    import json

    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic", claude_enabled=False)
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")

    engine.respond("lê o ficheiro email_resumo_novo e mostra tudo sem resumir")
    payloads = [json.loads(event) for event in engine.consume_ui_events()]

    loaded = next(p for p in payloads if p["type"] == "document_loaded")
    assert loaded["file_name"] == "email_resumo_novo.txt"
    assert loaded["chars"] > 0
    assert len(loaded["preview"]) <= 280
    assert anthropic.calls == []


def test_workspace_draft_ready_event_on_rewrite(tmp_path: Path) -> None:
    """B: a successful rewrite queues a document_draft_ready event carrying
    the real draft content (not a preview) at revision 1."""
    import json

    engine, _llm, ollama, anthropic = make_engine(
        tmp_path, mode="automatic", claude_enabled=False, ollama_replies=[_VALID_FORMAL_REWRITE],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")
    engine.respond("lê o ficheiro email_resumo_novo e mostra tudo sem resumir")
    engine.consume_ui_events()

    engine.respond("Torna-o mais formal, mas não guardes ainda.")
    payloads = [json.loads(event) for event in engine.consume_ui_events()]

    ready = next(p for p in payloads if p["type"] == "document_draft_ready")
    assert ready["file_name"] == "email_resumo_novo.txt"
    assert ready["revision"] == 1
    assert ready["based_on"] == "original"
    assert ready["content"] == _VALID_FORMAL_REWRITE


def test_workspace_draft_ready_revision_increments_for_same_file(tmp_path: Path) -> None:
    """C: a second successful refinement queues ANOTHER document_draft_ready
    for the SAME file_name with revision incremented -- the backend never
    invents a second draft identity, so the frontend updates one object
    instead of creating an infinite card trail."""
    import json

    engine, _llm, ollama, anthropic = make_engine(
        tmp_path, mode="automatic", claude_enabled=False,
        ollama_replies=[_VALID_FORMAL_REWRITE, _VALID_REFINED_REWRITE],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")
    engine.respond("lê o ficheiro email_resumo_novo e mostra tudo sem resumir")
    engine.respond("Torna-o mais formal, mas não guardes ainda.")
    first_ready = next(json.loads(e) for e in engine.consume_ui_events() if json.loads(e)["type"] == "document_draft_ready")

    engine.respond("ainda consegues melhor")
    payloads = [json.loads(event) for event in engine.consume_ui_events()]
    second_ready = next(p for p in payloads if p["type"] == "document_draft_ready")

    assert first_ready["file_name"] == second_ready["file_name"] == "email_resumo_novo.txt"
    assert first_ready["revision"] == 1
    assert second_ready["revision"] == 2
    assert second_ready["based_on"] == "draft"


def test_workspace_comparison_ready_event_carries_both_versions(tmp_path: Path) -> None:
    """E: 'compara as versões' queues a document_comparison_ready event
    with both the original and the draft content already resolved."""
    import json

    engine, _llm, ollama, anthropic = make_engine(
        tmp_path, mode="automatic", claude_enabled=False, ollama_replies=[_VALID_FORMAL_REWRITE],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")
    engine.respond("lê o ficheiro email_resumo_novo e mostra tudo sem resumir")
    engine.respond("Torna-o mais formal, mas não guardes ainda.")
    engine.consume_ui_events()

    engine.respond("compara as versões")
    payloads = [json.loads(event) for event in engine.consume_ui_events()]

    comparison = next(p for p in payloads if p["type"] == "document_comparison_ready")
    assert comparison["file_name"] == "email_resumo_novo.txt"
    assert "Febrada final de época" in comparison["original_content"]
    assert "Prezada Francisca" in comparison["draft_content"]


def test_workspace_discard_and_restore_events(tmp_path: Path) -> None:
    """discard_draft/restore_previous_draft also queue structured events, so
    the workspace object either disappears or returns to its previous
    revision without the frontend re-deriving any document logic."""
    import json

    engine, _llm, ollama, anthropic = make_engine(
        tmp_path, mode="automatic", claude_enabled=False,
        ollama_replies=[_VALID_FORMAL_REWRITE, _VALID_REFINED_REWRITE],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")
    engine.respond("lê o ficheiro email_resumo_novo e mostra tudo sem resumir")
    engine.respond("Torna-o mais formal, mas não guardes ainda.")
    engine.respond("ainda consegues melhor")  # revision 2, previous_draft = revision 1
    engine.consume_ui_events()

    engine.respond("volta à versão anterior")
    restored = [json.loads(event) for event in engine.consume_ui_events()]
    restore_event = next(p for p in restored if p["type"] == "document_draft_restored")
    assert restore_event["revision"] == 1
    assert restore_event["content"] == _VALID_FORMAL_REWRITE

    engine.respond("descarta as alterações")
    discarded = [json.loads(event) for event in engine.consume_ui_events()]
    assert any(p["type"] == "document_draft_discarded" for p in discarded)


def test_workspace_no_document_events_for_plain_chat(tmp_path: Path) -> None:
    """L: a plain conversational turn queues no document_* workspace event
    -- only documents/drafts/comparisons are ever externalized, never a
    simple reply."""
    import json

    engine, _llm, ollama, anthropic = make_engine(
        tmp_path, mode="automatic", claude_enabled=False, ollama_replies=["Estou bem, e tu?"],
    )

    engine.respond("olá, como estás?")
    payloads = [json.loads(event) for event in engine.consume_ui_events()]

    assert not any(str(p["type"]).startswith("document_") for p in payloads)
    assert anthropic.calls == []


# --- Passo 3.2: voice-first / natural-language workspace control. A new,
# independent detector (_try_workspace_command) recognizes deterministic
# spatial/layout phrases and short-circuits BEFORE document routing with
# llm_calls=0 -- target resolution against the live on-screen scene happens
# entirely client-side (echo_workspace_objects.js), so these tests only
# check what the backend can actually know: the recognized action/target
# phrase, the queued UI event, and that no LLM/provider call ever happens.

def test_workspace_command_side_by_side_is_llm_free(tmp_path: Path) -> None:
    """A/B: 'mostra os dois lado a lado' is recognized deterministically,
    queues one workspace_command event, and never touches any provider."""
    import json

    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic", claude_enabled=True)

    engine.respond("mostra os dois lado a lado")
    payloads = [json.loads(event) for event in engine.consume_ui_events()]
    telemetry = engine.get_last_turn_telemetry() or {}

    command = next(p for p in payloads if p["type"] == "workspace_command")
    assert command["action"] == "place_beside"
    assert command["relation"] == "side_by_side"
    assert command["source"] == "voice_command"
    assert telemetry["llm_calls"] == 0
    assert telemetry["execution_path"] == "workspace_command"
    assert telemetry["workspace_command_detected"] is True
    assert telemetry["workspace_command_action"] == "place_beside"
    assert ollama.calls == []
    assert anthropic.calls == []


def test_workspace_command_minimize_resolves_target_phrase(tmp_path: Path) -> None:
    """C: 'minimiza o original' is recognized with target_phrase='original'
    -- the backend never resolves this to an actual object id, only the
    frontend (which holds the live scene) can do that."""
    import json

    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="local")

    engine.respond("minimiza o original")
    payloads = [json.loads(event) for event in engine.consume_ui_events()]
    telemetry = engine.get_last_turn_telemetry() or {}

    command = next(p for p in payloads if p["type"] == "workspace_command")
    assert command["action"] == "minimize"
    assert command["target_phrase"] == "o original"
    assert telemetry["workspace_command_target"] == "o original"
    assert ollama.calls == []


def test_workspace_command_focus_second_version(tmp_path: Path) -> None:
    """D: 'vamos trabalhar na segunda versão' resolves to action=focus --
    since setPrimary already demotes whatever was primary before, this one
    action alone covers the compound 'minimiza o original, vamos trabalhar
    na segunda versão' scenario without needing a second event."""
    import json

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("ok, já podes minimizar o original, vamos trabalhar na segunda versão")
    payloads = [json.loads(event) for event in engine.consume_ui_events()]

    actions = [p["action"] for p in payloads if p["type"] == "workspace_command"]
    assert "focus" in actions
    assert ollama.calls == []


def test_workspace_command_focus_excerpt_paragraph(tmp_path: Path) -> None:
    """E: 'mostra só o primeiro parágrafo' resolves to action=focus_excerpt
    with selector_index=1."""
    import json

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("mostra só o primeiro parágrafo")
    payloads = [json.loads(event) for event in engine.consume_ui_events()]

    command = next(p for p in payloads if p["type"] == "workspace_command")
    assert command["action"] == "focus_excerpt"
    assert command["selector_index"] == "1"
    assert ollama.calls == []


def test_workspace_command_close_invoice_resolves_temporary_reference_target(tmp_path: Path) -> None:
    """G/H: 'fecha a fatura' resolves to action=close, target_phrase='fatura'."""
    import json

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("fecha a fatura")
    payloads = [json.loads(event) for event in engine.consume_ui_events()]

    command = next(p for p in payloads if p["type"] == "workspace_command")
    assert command["action"] == "close"
    assert command["target_phrase"] == "fatura"
    assert ollama.calls == []


def test_workspace_command_restore_previous_scene(tmp_path: Path) -> None:
    """I: 'volta ao que estávamos a fazer' resolves to
    action=restore_previous_scene -- scene HISTORY restore is executed
    entirely client-side; the backend just recognizes the intent."""
    import json

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("volta ao que estávamos a fazer")
    payloads = [json.loads(event) for event in engine.consume_ui_events()]

    command = next(p for p in payloads if p["type"] == "workspace_command")
    assert command["action"] == "restore_previous_scene"
    assert ollama.calls == []


def test_workspace_command_two_clauses_in_one_utterance(tmp_path: Path) -> None:
    """F: 'chega isto para o lado e mostra-me o segundo ficheiro' carries
    TWO workspace clauses in one turn -- both are recognized and queued as
    separate events (section 2's headline requirement: an utterance is not
    limited to a single command)."""
    import json

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("chega isto para o lado e mostra-me o segundo ficheiro")
    payloads = [json.loads(event) for event in engine.consume_ui_events()]
    actions = {p["action"] for p in payloads if p["type"] == "workspace_command"}

    assert "place" in actions
    assert "show" in actions
    assert ollama.calls == []


def test_workspace_command_defers_when_utterance_also_looks_like_rewrite(tmp_path: Path) -> None:
    """Documented limitation (section 2): a workspace phrase coexisting with
    a genuine content-transformation verb in the SAME utterance defers
    entirely to the existing content pipeline instead of partially
    executing the spatial half and silently dropping the content half."""
    import json

    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local", ollama_replies=["Resposta qualquer."])

    engine.respond("minimiza o original e melhora o texto")
    payloads = [json.loads(event) for event in engine.consume_ui_events()]

    assert not any(p["type"] == "workspace_command" for p in payloads)


def test_workspace_command_does_not_break_pending_action_cancellation(tmp_path: Path) -> None:
    """Regression guard: an early, broader phrasing of the restore-scene
    detector ('volta ao que estavamos', with no continuation) collided with
    the pre-existing pending-action-cancellation phrase 'Cancela e volta ao
    que estávamos a falar.' -- narrowed to require 'a fazer'/'a trabalhar'
    specifically so this keeps working exactly as before."""
    import json

    engine, llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("abre o google")
    llm.chat_calls = 0

    response = engine.respond("Cancela e volta ao que estávamos a falar.")
    payloads = [json.loads(event) for event in engine.consume_ui_events()]

    assert "ação pendente" not in response.lower()
    assert not engine.agent.has_pending_confirmation()
    assert not any(p["type"] == "workspace_command" for p in payloads)


def test_workspace_command_does_not_shadow_existing_draft_followups(tmp_path: Path) -> None:
    """Existing content-navigation phrases ('volta à versão anterior',
    'compara as versões', etc.) must keep routing through the pre-existing
    _active_document_followup_action detector unchanged -- the new
    workspace-command phrase tables are a disjoint vocabulary."""
    import json

    engine, _llm, _ollama, anthropic = make_engine(
        tmp_path, mode="automatic", claude_enabled=False, ollama_replies=[_VALID_FORMAL_REWRITE, _VALID_REFINED_REWRITE],
    )
    (engine.workspace_path / "email_resumo_novo.txt").write_text(REAL_EMAIL_SUMMARY, encoding="utf-8")
    engine.respond("lê o ficheiro email_resumo_novo e mostra tudo sem resumir")
    engine.respond("Torna-o mais formal, mas não guardes ainda.")
    engine.respond("ainda consegues melhor")
    engine.consume_ui_events()

    engine.respond("volta à versão anterior")
    payloads = [json.loads(event) for event in engine.consume_ui_events()]

    assert any(p["type"] == "document_draft_restored" for p in payloads)
    assert not any(p["type"] == "workspace_command" for p in payloads)
    assert anthropic.calls == []


# --- Passo 3.3: Echo-entity voice commands. A real failed test case showed
# "desloca-te para a direita" falling all the way to GENERAL_CONVERSATION
# (llm_calls=1, ~29s latency, Echo never actually moved) -- root cause was
# a pure vocabulary gap: _try_workspace_command already ran at the right
# point in _respond_inner (before document_task/writing_workflow/
# fast_route/topic_shift/general_conversation), but had zero phrases for
# moving Echo itself, only for arranging WorkspaceObjects.

def test_workspace_command_entity_move_right_is_llm_free(tmp_path: Path) -> None:
    """A: 'desloca-te para a direita' resolves to a WORKSPACE_COMMAND with
    action=move, target=echo_entity, region=right, llm_calls=0, latency in
    milliseconds -- never SOCIAL_FAST_PATH/DOCUMENT_TASK/GENERAL_CONVERSATION."""
    import json

    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic", claude_enabled=True)

    response = engine.respond("desloca-te para a direita")
    telemetry = engine.get_last_turn_telemetry() or {}
    payloads = [json.loads(event) for event in engine.consume_ui_events()]

    command = next(p for p in payloads if p["type"] == "workspace_command")
    assert command["action"] == "move"
    assert command["target_phrase"] == "echo_entity"
    assert command["region"] == "right"
    assert telemetry["selected_path"] == "WORKSPACE_COMMAND"
    assert telemetry["execution_path"] == "workspace_command"
    assert telemetry["llm_calls"] == 0
    assert telemetry["workspace_command_action"] == "move"
    assert telemetry["workspace_command_region"] == "right"
    assert telemetry["workspace_command_executed"] is True
    assert telemetry["workspace_command_latency_ms"] < 500  # milliseconds, not the ~29s real Ollama round trip
    assert "está pronto" not in response.lower()
    assert ollama.calls == []
    assert anthropic.calls == []


def test_workspace_command_entity_real_failed_phrase_with_typo(tmp_path: Path) -> None:
    """B: the EXACT real message that failed, typo included ('direta' not
    'direita') -- must resolve to move/right and must NOT fall through to
    GENERAL_CONVERSATION. Passo 3.4 upgrade: the 'vamos escrever um mail'
    remainder now starts a real ActiveTaskContext (compose_email) instead
    of just being acknowledged with a generic placeholder (section 3/4)."""
    import json

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="automatic", claude_enabled=True)

    response = engine.respond("desloca te para a direta, vamos escrever um mail e preciso de espaço")
    telemetry = engine.get_last_turn_telemetry() or {}
    payloads = [json.loads(event) for event in engine.consume_ui_events()]

    assert telemetry["selected_path"] == "WORKSPACE_COMMAND"
    assert telemetry["execution_path"] != "llm"
    assert telemetry["llm_calls"] == 0
    move_commands = [p for p in payloads if p["type"] == "workspace_command" and p["action"] == "move"]
    assert move_commands and move_commands[0]["region"] == "right"
    # A specific direction already satisfies the generic "preciso de
    # espaço" -- only one entity command should fire, not two fighting
    # over where Echo ends up.
    assert len([p for p in payloads if p["type"] == "workspace_command"]) == 1
    assert telemetry["active_task_type"] == "compose_email"
    assert "quem" in response.lower()  # remainder acted on, not just echoed
    assert ollama.calls == []


def test_workspace_command_entity_make_room_without_direction(tmp_path: Path) -> None:
    """C: 'preciso de espaço, chega-te para o lado' -- a make_room phrase
    with no explicit direction still resolves deterministically."""
    import json

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("preciso de espaço, chega-te para o lado")
    telemetry = engine.get_last_turn_telemetry() or {}
    payloads = [json.loads(event) for event in engine.consume_ui_events()]

    command = next(p for p in payloads if p["type"] == "workspace_command")
    assert command["target_phrase"] == "echo_entity"
    assert command["action"] == "make_room"
    assert telemetry["llm_calls"] == 0
    assert ollama.calls == []


def test_workspace_command_entity_move_left(tmp_path: Path) -> None:
    """D: 'vai para a esquerda' -> region=left."""
    import json

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vai para a esquerda")
    payloads = [json.loads(event) for event in engine.consume_ui_events()]

    command = next(p for p in payloads if p["type"] == "workspace_command")
    assert command["action"] == "move"
    assert command["region"] == "left"
    assert ollama.calls == []


def test_workspace_command_entity_center(tmp_path: Path) -> None:
    """E: 'volta para o centro' -> action=center."""
    import json

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("volta para o centro")
    payloads = [json.loads(event) for event in engine.consume_ui_events()]

    command = next(p for p in payloads if p["type"] == "workspace_command")
    assert command["action"] == "center"
    assert command["target_phrase"] == "echo_entity"
    assert ollama.calls == []


def test_workspace_command_entity_move_works_with_empty_workspace(tmp_path: Path) -> None:
    """F: the entity-move command never depends on an active document or
    any WorkspaceObject existing -- it must work in a completely empty
    workspace, which is exactly the real scenario that failed."""
    import json

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    assert engine._active_document_context is None

    engine.respond("desloca-te para a direita")
    payloads = [json.loads(event) for event in engine.consume_ui_events()]

    assert any(p["type"] == "workspace_command" and p["action"] == "move" for p in payloads)
    assert ollama.calls == []


def test_workspace_command_entity_correction_retries_last_command(tmp_path: Path) -> None:
    """I: 'não deslocaste' retries the LAST entity command instead of
    falling through to GENERAL_CONVERSATION -- this is exactly the second
    half of the real failure (the follow-up also went to GENERAL_
    CONVERSATION and produced an incoherent reply)."""
    import json

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("desloca-te para a direita")
    engine.consume_ui_events()

    response = engine.respond("não deslocaste")
    telemetry = engine.get_last_turn_telemetry() or {}
    payloads = [json.loads(event) for event in engine.consume_ui_events()]

    assert telemetry["selected_path"] == "WORKSPACE_COMMAND"
    assert telemetry["llm_calls"] == 0
    command = next(p for p in payloads if p["type"] == "workspace_command")
    assert command["action"] == "move"
    assert command["region"] == "right"
    assert command["source"] == "voice_command_retry"
    assert "?" not in response  # never asks a clarifying question here
    assert ollama.calls == []


def test_workspace_command_entity_correction_without_prior_command_falls_through(tmp_path: Path) -> None:
    """A correction phrase with no prior entity command on record has
    nothing deterministic to retry -- must fall through to normal routing
    rather than fabricating a retry."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local", ollama_replies=["Não percebi."])

    engine.respond("não deslocaste")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] != "WORKSPACE_COMMAND"


def test_workspace_command_entity_acks_never_claim_specific_execution(tmp_path: Path) -> None:
    """J: the response never affirms a UI action happened just because the
    backend recognized it -- acknowledgments are short and neutral
    ("Feito."/"Claro."), never narrating the actual movement ("Vou
    deslocar-me para a direita...") the way the real failed LLM reply did,
    since the backend cannot itself confirm the frontend applied it."""
    import assistant.conversation as conversation_module

    for action, ack in conversation_module._WORKSPACE_COMMAND_ACKS.items():
        assert "direita" not in ack.lower()
        assert "esquerda" not in ack.lower()
        assert "vou" not in ack.lower()
        assert "pixel" not in ack.lower()


def test_workspace_command_entity_clear_no_executive_function_or_composer(tmp_path: Path) -> None:
    """K/L: a clear entity command never reaches ExecutiveFunction/the
    cognitive loop or the Response Composer -- llm_calls==0 already proves
    this indirectly (composer/executive-function calls would show up as
    real chat calls), checked explicitly via response_source too."""
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic", claude_enabled=True)

    engine.respond("desloca-te para a direita")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["response_source"] == "WORKSPACE_COMMAND"
    assert telemetry["selected_path"] == "WORKSPACE_COMMAND"
    assert ollama.calls == []
    assert anthropic.calls == []


def test_workspace_command_entity_latency_is_local_and_fast(tmp_path: Path) -> None:
    """M: latency is deterministic/local -- no provider round trip at all,
    milliseconds not the ~29s the real Ollama call took."""
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic", claude_enabled=True)

    engine.respond("desloca-te para a direita")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["workspace_command_latency_ms"] < 500
    assert ollama.calls == []
    assert anthropic.calls == []


def test_workspace_command_entity_mixed_utterance_preserves_remaining_intent(tmp_path: Path) -> None:
    """N: 'desloca-te para a direita, vamos escrever um mail' preserves the
    'vamos escrever um mail' remainder instead of silently dropping it --
    Passo 3.4 upgrade: it now starts a real ActiveTaskContext (section 3),
    not just echoing the words back."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    response = engine.respond("desloca-te para a direita, vamos escrever um mail")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["active_task_type"] == "compose_email"
    assert "quem" in response.lower()
    assert ollama.calls == []


# --- Passo 3.5 (Problem 1): real-test failure showed "chega para a esquerda"
# falling to GENERAL_CONVERSATION (llm_calls=1) while "desloca-te para a
# esquerda" worked -- root cause was a pure vocabulary gap, not a structural
# one: find_near_pair_span(verbs, direction_words) already disambiguates a
# bare "chega" (stop/finish, no nearby direction word) from "chega" + a
# direction (movement) with zero extra special-casing once "chega" itself is
# added to _ENTITY_MOVEMENT_VERBS. Section 7, A-I.

def test_workspace_language_a_bare_chega_with_direction_is_movement(tmp_path: Path) -> None:
    """A: 'chega para a esquerda' -> move/left, llm_calls=0 -- the exact
    real phrase that previously fell to GENERAL_CONVERSATION."""
    import json

    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic", claude_enabled=True)

    response = engine.respond("chega para a esquerda")
    telemetry = engine.get_last_turn_telemetry() or {}
    payloads = [json.loads(event) for event in engine.consume_ui_events()]

    command = next(p for p in payloads if p["type"] == "workspace_command")
    assert command["action"] == "move"
    assert command["region"] == "left"
    assert telemetry["selected_path"] == "WORKSPACE_COMMAND"
    assert telemetry["llm_calls"] == 0
    assert "chegarei" not in response.lower()
    assert ollama.calls == []
    assert anthropic.calls == []


def test_workspace_language_b_chega_te_direita(tmp_path: Path) -> None:
    """B: 'chega-te para a direita' -> move/right."""
    import json

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("chega-te para a direita")
    payloads = [json.loads(event) for event in engine.consume_ui_events()]

    command = next(p for p in payloads if p["type"] == "workspace_command")
    assert command["action"] == "move"
    assert command["region"] == "right"
    assert ollama.calls == []


def test_workspace_language_c_encosta_te_mais_a_esquerda(tmp_path: Path) -> None:
    """C: 'encosta-te mais à esquerda' -> move/left."""
    import json

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("encosta-te mais à esquerda")
    payloads = [json.loads(event) for event in engine.consume_ui_events()]

    command = next(p for p in payloads if p["type"] == "workspace_command")
    assert command["action"] == "move"
    assert command["region"] == "left"
    assert ollama.calls == []


def test_workspace_language_d_sai_do_meio_is_make_room(tmp_path: Path) -> None:
    """D: 'sai do meio' -> make_room."""
    import json

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("sai do meio")
    payloads = [json.loads(event) for event in engine.consume_ui_events()]

    command = next(p for p in payloads if p["type"] == "workspace_command")
    assert command["action"] == "make_room"
    assert ollama.calls == []


def test_workspace_language_e_da_me_espaco_aqui_is_make_room(tmp_path: Path) -> None:
    """E: 'dá-me espaço aqui' -> make_room."""
    import json

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("dá-me espaço aqui")
    payloads = [json.loads(event) for event in engine.consume_ui_events()]

    command = next(p for p in payloads if p["type"] == "workspace_command")
    assert command["action"] == "make_room"
    assert ollama.calls == []


def test_workspace_language_f_ja_chega_is_not_movement(tmp_path: Path) -> None:
    """F: 'já chega' (stop/finish) must NOT be read as movement -- no
    direction word is anywhere nearby, so find_near_pair_span naturally
    never matches; this is the disambiguation the user asked to verify with
    an explicit test rather than trust implicitly."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local", ollama_replies=["Ok, paro por aqui."])

    engine.respond("já chega")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] != "WORKSPACE_COMMAND"
    assert not telemetry.get("workspace_command_detected")


def test_workspace_language_g_chega_disso_is_not_movement(tmp_path: Path) -> None:
    """G: 'chega disso' (stop/finish) must NOT be read as movement."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local", ollama_replies=["Ok."])

    engine.respond("chega disso")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] != "WORKSPACE_COMMAND"
    assert not telemetry.get("workspace_command_detected")


def test_workspace_language_h_mixed_movement_and_compose_email_no_comma(tmp_path: Path) -> None:
    """H: 'chega para a esquerda para escrevermos um email' -- the EXACT
    real compound sentence, with no comma/' e ' to split on -- must produce
    BOTH the movement command AND start/continue the compose_email
    ActiveTaskContext in the same turn. Neither intent may be dropped."""
    import json

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    response = engine.respond("chega para a esquerda para escrevermos um email")
    telemetry = engine.get_last_turn_telemetry() or {}
    payloads = [json.loads(event) for event in engine.consume_ui_events()]

    command = next(p for p in payloads if p["type"] == "workspace_command")
    assert command["action"] == "move"
    assert command["region"] == "left"
    assert telemetry["selected_path"] == "WORKSPACE_COMMAND"
    assert telemetry["llm_calls"] == 0
    assert telemetry["active_task_type"] == "compose_email"
    assert "quem" in response.lower()
    assert ollama.calls == []


def test_workspace_language_i_clear_command_is_llm_free(tmp_path: Path) -> None:
    """I: a clear, deterministic workspace command never calls any provider,
    regardless of which movement verb/phrasing family it comes from."""
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic", claude_enabled=True)

    engine.respond("vai para a direita")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["llm_calls"] == 0
    assert ollama.calls == []
    assert anthropic.calls == []


def test_workspace_language_telemetry_reports_match_type_and_signals(tmp_path: Path) -> None:
    """Section 9: workspace_language_match_type/signals and the
    previously-dead workspace_command_confidence field are now populated."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("chega para a esquerda")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["workspace_language_match_type"] == "verb_direction_pair"
    assert "chega" in telemetry["workspace_language_signals"]
    assert "esquerda" in telemetry["workspace_language_signals"]
    assert telemetry["workspace_command_confidence"] == "high"


# --- Passo 3.4: ActiveTaskContext -- a generic, session-only "what are we
# building together" layer, compose_email being the first task_type. Real
# failed conversation: turn 1 correctly moved Echo and started a task, but
# turn 2 ("é para um professor... João Cardoso... Computação Gráfica
# Interativa") fell to GENERAL_CONVERSATION and asked "quem vais pedir
# ajuda?" despite the recipient having just been stated; turn 4 had Echo
# explain Geometry Processing instead of treating it as email content.

def test_active_task_exact_real_conversation(tmp_path: Path) -> None:
    """Section 20: the exact 4-turn real conversation, verbatim."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    r1 = engine.respond("desloca te para a direita, vamos escrever um mail")
    t1 = engine.get_last_turn_telemetry() or {}
    assert t1["selected_path"] == "WORKSPACE_COMMAND"
    assert t1["llm_calls"] == 0
    assert t1["active_task_type"] == "compose_email"
    engine.consume_ui_events()

    r2 = engine.respond("é para um professor da faculdade, João Cardoso, da cadeira Computação Gráfica Interativa")
    t2 = engine.get_last_turn_telemetry() or {}
    assert t2["selected_path"] == "TASK_CONTINUATION"
    assert t2["llm_calls"] == 0
    assert t2["task_known_fields"].get("recipient") == "João Cardoso"
    assert t2["task_known_fields"].get("recipientRole") == "professor"
    assert t2["task_known_fields"].get("course") == "Computação Gráfica Interativa"
    assert "quem" not in r2.lower()  # never re-asks who it's for
    engine.consume_ui_events()

    r3 = engine.respond("esclarecimentos sobre uma matéria")
    t3 = engine.get_last_turn_telemetry() or {}
    assert t3["selected_path"] == "TASK_CONTINUATION"
    assert t3["llm_calls"] == 0
    assert t3["task_known_fields"].get("purpose") == "request_clarification"
    engine.consume_ui_events()

    r4 = engine.respond("o pipeline gráfico, a fase de geometry processing")
    t4 = engine.get_last_turn_telemetry() or {}
    assert t4["selected_path"] == "TASK_CONTINUATION"
    assert t4["llm_calls"] == 0
    assert "geometry processing" in t4["task_known_fields"].get("topic", "").lower()
    # Never taught Geometry Processing back to the user -- treated as task content.
    assert "processing e" not in r4.lower() and "e uma fase" not in r4.lower()
    assert t4["active_task_status"] == "ready"
    assert not t4["task_missing_fields"]

    assert ollama.calls == []


def test_active_task_a_plain_trigger_creates_task(tmp_path: Path) -> None:
    """A: 'vamos escrever um email' alone (no workspace command) creates
    the task and asks the first missing question, llm_calls=0."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    response = engine.respond("vamos escrever um email")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "TASK_CONTINUATION"
    assert telemetry["active_task_type"] == "compose_email"
    assert telemetry["llm_calls"] == 0
    assert "quem" in response.lower()
    assert ollama.calls == []


def test_active_task_b_single_name_updates_recipient(tmp_path: Path) -> None:
    """B: 'é para a Maria' -- a single-word name -- updates recipient."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos escrever um email")
    engine.respond("é para a Maria")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["task_known_fields"].get("recipient") == "Maria"
    assert ollama.calls == []


def test_active_task_c_short_fragments_respect_active_task(tmp_path: Path) -> None:
    """C: short voice-style fragments ('o professor', 'CGI') are
    interpreted against the active task, not as isolated GENERAL_
    CONVERSATION turns."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos escrever um email")
    r = engine.respond("o professor")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "TASK_CONTINUATION"
    assert telemetry["task_known_fields"].get("recipientRole") == "professor"
    assert ollama.calls == []


def test_active_task_d_information_in_different_order(tmp_path: Path) -> None:
    """D: giving purpose before recipient works the same as the reverse
    order -- extraction is order-independent."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos escrever um email")
    engine.respond("esclarecimentos sobre uma matéria")
    engine.respond("é para o professor João Cardoso")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["task_known_fields"].get("purpose") == "request_clarification"
    assert telemetry["task_known_fields"].get("recipient") == "João Cardoso"
    assert telemetry["task_known_fields"].get("recipientRole") == "professor"
    assert ollama.calls == []


def test_active_task_e_one_turn_fills_several_fields(tmp_path: Path) -> None:
    """E: 'É para o João Cardoso, professor de CGI, quero perguntar sobre
    geometry processing' fills recipient+role+purpose-adjacent content in
    ONE turn."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos escrever um email")
    engine.respond("É para o João Cardoso, professor de CGI, quero perguntar sobre geometry processing")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["task_known_fields"].get("recipient") == "João Cardoso"
    assert telemetry["task_known_fields"].get("recipientRole") == "professor"
    assert len(telemetry["task_fields_updated"]) >= 2
    assert ollama.calls == []


def test_active_task_f_never_asks_known_fields_again(tmp_path: Path) -> None:
    """F: once recipient is known, the next question is never 'para
    quem?' again."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos escrever um email")
    engine.respond("é para o professor João Cardoso")
    r = engine.respond("da cadeira de Computação Gráfica Interativa")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "para quem" not in r.lower()
    assert "recipient" not in telemetry["task_missing_fields"]


def test_active_task_g_no_accidental_topic_shift(tmp_path: Path) -> None:
    """G: mentioning the task's own subject matter never triggers a
    general-knowledge explanation while the task is active (section 13) --
    covered concretely by turn 4 of the exact-conversation test; this adds
    a second, independent phrase to guard against overfitting to the one
    example."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos escrever um email")
    engine.respond("é para o professor João Cardoso")
    engine.respond("esclarecimentos sobre uma matéria")
    r = engine.respond("rasterization e se acontece antes do clipping")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "TASK_CONTINUATION"
    assert "rasterization" in telemetry["task_known_fields"].get("topic", "").lower()
    assert ollama.calls == []


def test_active_task_h_explicit_teach_me_exits_temporarily(tmp_path: Path) -> None:
    """H: 'já agora explica-me Geometry Processing' is an EXPLICIT signal
    that lets normal conversation answer for that one turn -- the task
    itself is not lost (paused, not cancelled)."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local", ollama_replies=["Explicacao qualquer."])

    engine.respond("vamos escrever um email")
    engine.respond("é para o professor João Cardoso")
    r = engine.respond("já agora explica-me o que é Geometry Processing")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] != "TASK_CONTINUATION"
    assert r == "Explicacao qualquer."
    assert len(ollama.calls) == 1
    assert engine._active_task_context is not None  # not lost, just bypassed for this turn


def test_active_task_i_resume_after_pause(tmp_path: Path) -> None:
    """I: 'volta ao email' resumes a paused task."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos escrever um email")
    engine.respond("espera, deixa o email ai")
    r = engine.respond("volta ao email")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "TASK_RESUME"
    assert telemetry["task_resumed"] is True
    assert telemetry["active_task_status"] in ("collecting", "ready")
    assert "?" in r  # picks the conversation back up with a real question
    assert ollama.calls == []


def test_active_task_j_workspace_movement_does_not_erase_task(tmp_path: Path) -> None:
    """J: a pure workspace command (no task-relevant remainder) while a
    task is active never clears active_task_context."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos escrever um email")
    engine.respond("é para o professor João Cardoso")
    engine.respond("volta para o centro")  # pure entity command, no remainder
    telemetry = engine.get_last_turn_telemetry() or {}

    assert engine._active_task_context is not None
    assert engine._active_task_context.entities.get("recipient") == "João Cardoso"
    assert telemetry["selected_path"] == "WORKSPACE_COMMAND"
    assert ollama.calls == []


def test_active_task_k_task_update_and_workspace_command_same_turn(tmp_path: Path) -> None:
    """K: a workspace command with a task-relevant remainder updates BOTH
    in the same turn (section 18) -- the entity moves AND the task field
    updates, one response."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    import json

    engine.respond("vamos escrever um email")
    payloads_before = [json.loads(e) for e in engine.consume_ui_events()]

    engine.respond("desloca-te para a esquerda, é para o professor João Cardoso")
    telemetry = engine.get_last_turn_telemetry() or {}
    payloads = [json.loads(e) for e in engine.consume_ui_events()]

    move_events = [p for p in payloads if p["type"] == "workspace_command" and p.get("action") == "move"]
    assert move_events and move_events[0]["region"] == "left"
    assert telemetry["task_known_fields"].get("recipient") == "João Cardoso"
    assert ollama.calls == []


def test_active_task_l_never_persisted_as_long_term_memory(tmp_path: Path) -> None:
    """L: ActiveTaskContext is session-only -- nothing about it (recipient
    name, course, topic) is ever written to LongTermMemory (only the
    in-memory ActiveTaskContext dataclass on the engine holds it)."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos escrever um email")
    engine.respond("é para o professor João Cardoso, sobre Geometry Processing")

    assert engine._active_task_context.entities.get("recipient") == "João Cardoso"
    # Unrelated preferences (e.g. detected language) may legitimately exist --
    # only assert that none of THIS feature's data leaked in.
    preferences_text = " ".join(f"{k}={v}" for k, v in engine.long_term_memory.preferences.items())
    assert "João Cardoso" not in preferences_text
    assert "Geometry Processing" not in preferences_text
    assert ollama.calls == []


def test_active_task_m_clear_conversation_clears_task(tmp_path: Path) -> None:
    """M: clear_conversation() wipes the transient ActiveTaskContext, same
    policy as PendingWritingTask/ActiveDocumentContext."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos escrever um email")
    assert engine._active_task_context is not None

    engine.clear_conversation()

    assert engine._active_task_context is None


def test_active_task_generic_trip_email_switch_and_resume_real_conversation(tmp_path: Path) -> None:
    """A generic planning task can be paused by a real email task, the
    workspace command in the middle does not drop the activity, and the
    original trip task can be resumed later."""
    import json

    valid_exam_email = (
        "Caro Professor João Barreto,\n\n"
        "Espero que se encontre bem. Escrevo para perguntar qual será a data do exame "
        "da unidade curricular de Cálculo.\n\n"
        "Com os melhores cumprimentos,\nAlexandre"
    )
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="local", ollama_replies=[valid_exam_email])

    r1 = engine.respond("prepara te, vamos planear umas ferias")
    t1 = engine.get_last_turn_telemetry() or {}
    payloads1 = [json.loads(event) for event in engine.consume_ui_events()]
    assert t1["selected_path"] == "TASK_START"
    assert t1["active_task_type"] == "plan_trip"
    assert t1["activity_family"] == "planning"
    assert t1["activity_domain"] == "travel"
    assert t1["activity_artifact_type"] == "itinerary"
    assert t1["task_started"] is True
    assert t1["llm_calls"] == 0
    assert any(p["type"] == "task_artifact_updated" and p["task_type"] == "plan_trip" for p in payloads1)
    assert "descanso" in r1.lower() or "aventura" in r1.lower()

    engine.respond("desculpa, lembrei me que primeiro preciso de escrever um mail")
    t2 = engine.get_last_turn_telemetry() or {}
    engine.consume_ui_events()
    assert t2["selected_path"] == "TASK_SWITCH"
    assert t2["previous_active_task_type"] == "plan_trip"
    assert t2["new_active_task_type"] == "compose_email"
    assert t2["task_paused"] is True
    assert t2["task_switched"] is True
    assert t2["task_stack_depth"] == 1
    assert t2["llm_calls"] == 0

    engine.respond("ajuda me a escrever, chega te para a esquerda e vamos escrever juntos")
    t3 = engine.get_last_turn_telemetry() or {}
    payloads3 = [json.loads(event) for event in engine.consume_ui_events()]
    assert t3["selected_path"] == "WORKSPACE_COMMAND"
    assert t3["workspace_command_action"] == "move"
    assert t3["workspace_command_region"] == "left"
    assert t3["active_task_type"] == "compose_email"
    assert t3["task_continuation_detected"] is True
    assert t3["llm_calls"] == 0
    assert any(p["type"] == "workspace_command" and p["action"] == "move" for p in payloads3)

    engine.respond("é para o meu professor de cálculo, joão barreto, a perguntar qual será a data do exame")
    t4 = engine.get_last_turn_telemetry() or {}
    fields = t4["task_known_fields"]
    assert t4["selected_path"] == "TASK_CONTINUATION"
    assert fields.get("recipient") == "João Barreto"
    assert fields.get("recipientRole") == "professor"
    assert fields.get("course") == "Cálculo"
    assert fields.get("purpose") == "ask_exam_date"
    assert fields.get("topic") == "data do exame"
    assert t4["active_task_status"] == "ready"
    assert t4["llm_calls"] == 0

    engine.respond("escreve o email completo, e atenção que é um professor de faculdade")
    t5 = engine.get_last_turn_telemetry() or {}
    assert t5["selected_path"] == "TASK_EXECUTION"
    assert t5["active_task_type"] == "compose_email"
    assert t5["active_task_status"] == "reviewing"
    assert t5["llm_calls"] == 1

    engine.respond("agora podemos voltar às férias")
    t6 = engine.get_last_turn_telemetry() or {}
    assert t6["selected_path"] == "TASK_RESUME"
    assert t6["active_task_type"] == "plan_trip"
    assert t6["task_resumed"] is True
    assert t6["previous_active_task_type"] == "compose_email"
    assert t6["new_active_task_type"] == "plan_trip"
    assert t6["llm_calls"] == 0

    assert len(ollama.calls) == 1
    assert anthropic.calls == []


@pytest.mark.parametrize(
    ("message", "task_type", "family", "domain", "artifact_type"),
    [
        ("vamos planear umas férias", "plan_trip", "planning", "travel", "itinerary"),
        ("preciso de decidir entre contratar duas pessoas", "make_decision", "decision", "unknown", "decision_matrix"),
        ("vamos estudar algoritmos de grafos", "study_topic", "study", "university", "plan"),
        ("vamos escrever juntos", "writing", "writing", "unknown", "unknown"),
        ("quero investigar opções de painéis solares", "research_topic", "research", "unknown", "note"),
    ],
)
def test_active_task_generic_activity_starts_without_llm(
    tmp_path: Path,
    message: str,
    task_type: str,
    family: str,
    domain: str,
    artifact_type: str,
) -> None:
    """Generic scenarios start a transient ActiveTaskContext without
    domain-specific tools or a model call."""
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="local")

    engine.respond(message)
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "TASK_START"
    assert telemetry["active_task_type"] == task_type
    assert telemetry["activity_family"] == family
    assert telemetry["activity_domain"] == domain
    assert telemetry["activity_artifact_type"] == artifact_type
    assert telemetry["task_started"] is True
    assert telemetry["task_stack_depth"] == 0
    assert telemetry["llm_calls"] == 0
    assert ollama.calls == []
    assert anthropic.calls == []


# --- Passo 3.5 (Problem 2): real-test failure showed "chegar é o mesmo que
# deslocar?" answered with "...O que queres saber sobre a fase de Geometry
# Processing?" -- true session memory (from an earlier, unrelated topic) but
# NOT relevant to this turn. A Memory Relevance Gate now scores topical
# overlap between the user's message and the persisted-memory candidate
# before it's allowed into the prompt; explicit recall ("lembras-te...")
# always bypasses the score. Section 8, A-F.

GEOMETRY_MEMORY_CONTEXT = (
    "Estivemos a estudar o pipeline gráfico, sobretudo a fase de Geometry Processing "
    "em Computação Gráfica Interativa."
)

# _gate_memory_relevance is only reached through _context_for_agent when the
# executive-function strategy for the turn has use_long_term_memory=True
# (PLANNING/PROBLEM_SOLVING categories) -- ordinary SOCIAL_CONVERSATION/
# GENERAL_INFORMATION turns (like "chegar é o mesmo que deslocar?" itself)
# never call long_term_memory.context_for() at all, so there's nothing for
# the score gate to filter in that specific real-world repro (that leak
# channel is _recent_conversation_history, fixed separately by the
# session-boundary segment reset). These tests exercise the gate directly,
# the same way other tests in this file call private routing methods
# directly (see test_calendar_claim_without_calendar_tool_is_blocked).


def test_memory_relevance_a_irrelevant_memory_is_rejected(tmp_path: Path) -> None:
    """A: an unrelated query against the Geometry Processing memory scores
    below threshold and is rejected, with the reason recorded."""
    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine._begin_turn_trace("chegar é o mesmo que deslocar?")

    result = engine._gate_memory_relevance("chegar é o mesmo que deslocar?", GEOMETRY_MEMORY_CONTEXT)

    assert result == ""
    assert engine._turn_trace["memory_candidate_present"] is True
    assert engine._turn_trace["memory_relevance_gate_passed"] is False
    assert engine._turn_trace["memory_rejected_reason"]


def test_memory_relevance_b_topically_matching_memory_may_be_used(tmp_path: Path) -> None:
    """B: 'o que estávamos a estudar em Computação Gráfica?' shares real
    topical terms with the persisted memory -- the gate should pass on
    relevance score alone, no explicit recall wording needed."""
    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine._begin_turn_trace("o que estávamos a estudar em Computação Gráfica?")

    result = engine._gate_memory_relevance("o que estávamos a estudar em Computação Gráfica?", GEOMETRY_MEMORY_CONTEXT)

    assert result == GEOMETRY_MEMORY_CONTEXT
    assert engine._turn_trace["memory_relevance_gate_passed"] is True
    assert engine._turn_trace["memory_relevance_score"] > 0


def test_memory_relevance_c_explicit_recall_bypasses_score(tmp_path: Path) -> None:
    """C: 'lembras-te da dúvida sobre Geometry Processing?' is an explicit
    recall request -- it must pass the gate even though scored relevance
    isn't what decides it (this is the 'lembrar', not 'mencionar', case)."""
    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine._begin_turn_trace("lembras-te da dúvida sobre Geometry Processing?")

    result = engine._gate_memory_relevance("lembras-te da dúvida sobre Geometry Processing?", GEOMETRY_MEMORY_CONTEXT)

    assert result == GEOMETRY_MEMORY_CONTEXT
    assert engine._turn_trace["memory_relevance_gate_passed"] is True
    assert engine._turn_trace["memory_relevance_reason"] == "explicit_recall"


def test_memory_relevance_d_active_task_outranks_unrelated_memory(tmp_path: Path) -> None:
    """D: with an active compose_email task in progress, an unrelated
    persisted memory must not contaminate the task turn -- current explicit
    intent and active task both outrank stale memory in the priority
    order the user specified."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.long_term_memory.context_text = GEOMETRY_MEMORY_CONTEXT

    engine.respond("vamos escrever um email")
    response = engine.respond("é para a Maria")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "geometry processing" not in response.lower()
    assert telemetry["task_known_fields"].get("recipient") == "Maria"
    assert ollama.calls == []


def test_memory_relevance_e_active_task_doubles_the_threshold(tmp_path: Path) -> None:
    """E: an active task raises the bar for stale memory -- a query whose
    overlap score clears the normal threshold but not the doubled one is
    accepted with no active task, rejected with one."""
    import assistant.conversation as conversation_module

    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")
    query = "estudamos ontem alguma coisa sobre gráfica"  # shares only "grafica" (1/5 terms, score=0.2)

    engine._begin_turn_trace(query)
    without_task = engine._gate_memory_relevance(query, GEOMETRY_MEMORY_CONTEXT)
    score_without_task = engine._turn_trace["memory_relevance_score"]

    engine._active_task_context = conversation_module.ActiveTaskContext(task_type="compose_email")
    engine._begin_turn_trace(query)
    with_task = engine._gate_memory_relevance(query, GEOMETRY_MEMORY_CONTEXT)

    assert score_without_task >= conversation_module._MEMORY_RELEVANCE_THRESHOLD
    assert score_without_task < conversation_module._MEMORY_RELEVANCE_THRESHOLD * 2
    assert without_task == GEOMETRY_MEMORY_CONTEXT
    assert with_task == ""
    assert engine._turn_trace["memory_relevance_gate_passed"] is False


def test_memory_relevance_f_telemetry_reports_rejected_reason(tmp_path: Path) -> None:
    """F: telemetry must indicate memory was rejected for low relevance,
    not just silently drop it."""
    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine._begin_turn_trace("qual é a capital de Portugal?")

    result = engine._gate_memory_relevance("qual é a capital de Portugal?", GEOMETRY_MEMORY_CONTEXT)

    assert result == ""
    assert engine._turn_trace["memory_candidate_present"] is True
    assert engine._turn_trace["memory_relevance_gate_passed"] is False
    assert engine._turn_trace["memory_rejected_reason"] in ("no_shared_terms", "no_comparable_terms")


# --- Passo 3.6 (ActiveTaskContext semantics fix): a real test showed
# continuity working (turns keep updating the SAME task) but the internal
# semantics being wrong -- turn 2 lost the recipient name entirely ("Para: o
# professor"), turn 3 asked for the topic again despite it being given in
# the same sentence as the purpose, and turn 4 ("podes escrever") got
# stored as the email's own subject ("Assunto: podes escrever") instead of
# being recognized as a command to actually generate the email. Section
# 1-14, tests A-M (L/M are frontend-only, covered in test_web_ui_prototype.py).

_BELLMAN_FORD_EMAIL_REPLY = (
    "Caro Professor Luís Paquete,\n\n"
    "Venho por este meio pedir alguns esclarecimentos sobre o algoritmo Bellman-Ford.\n\n"
    "Com os melhores cumprimentos,\nAlexandre"
)


def test_task_semantics_a_data_turn_is_not_misclassified_as_control(tmp_path: Path) -> None:
    """A: a plain TASK_DATA turn (recipient info) is classified as
    task_turn_intent='data', never 'control'."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos escrever um email")
    engine.respond("é para o meu professor de estrategias algoritmicas, luis paquete")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["task_turn_intent"] == "data"
    assert telemetry["task_control_action"] == ""
    assert ollama.calls == []


def test_task_semantics_b_escreve_never_becomes_topic(tmp_path: Path) -> None:
    """B: with recipient+purpose known but topic still missing, a bare
    'escreve' must be classified as TASK_CONTROL (asks for the missing
    topic again) -- never captured as the topic/subject itself. This is
    the exact real bug: 'Assunto: podes escrever'."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos escrever um email")
    engine.respond("é para o professor João Cardoso")
    engine.respond("quero pedir ajuda")
    response = engine.respond("escreve")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["task_turn_intent"] == "control"
    assert telemetry["task_known_fields"].get("topic", "") != "escreve"
    assert "escreve" not in telemetry["task_known_fields"].get("topic", "").lower()
    assert "assunto" in response.lower() or "sobre" in response.lower()
    assert ollama.calls == []


def test_task_semantics_c_podes_escrever_executes_ready_task(tmp_path: Path) -> None:
    """C: 'podes escrever' on a READY task triggers TASK_EXECUTION
    immediately -- no redundant 'queres que escreva?' confirmation."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local", ollama_replies=[_BELLMAN_FORD_EMAIL_REPLY])

    engine.respond("chega te para a esquerda, vamos escrever um mail")
    engine.respond("é para o meu professor de estrategias algoritmicas, luis paquete")
    engine.respond("quero pedir esclarecimentos sobre o algoritmo bellman ford")
    response = engine.respond("podes escrever")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["task_turn_intent"] == "control"
    assert telemetry["task_control_action"] == "execute"
    assert telemetry["task_execution_started"] is True
    assert telemetry["task_execution_completed"] is True
    assert telemetry["llm_calls"] == 1
    assert "queres que escreva" not in response.lower()
    assert len(ollama.calls) == 1


def test_task_semantics_d_name_role_course_same_turn(tmp_path: Path) -> None:
    """D: 'é para o meu professor de estrategias algoritmicas, luis
    paquete' -- all-lowercase, no capitalization -- fills recipient, role
    AND course in the same turn (the exact real bug: only role was
    extracted, recipient came back empty -- 'Para: o professor')."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos escrever um email")
    engine.respond("é para o meu professor de estrategias algoritmicas, luis paquete")
    telemetry = engine.get_last_turn_telemetry() or {}
    fields = telemetry["task_known_fields"]

    assert fields.get("recipientRole") == "professor"
    assert "paquete" in fields.get("recipient", "").lower()
    assert "algoritmicas" in fields.get("course", "").lower()
    assert ollama.calls == []


def test_task_semantics_e_purpose_and_topic_same_turn(tmp_path: Path) -> None:
    """E: 'quero pedir esclarecimentos sobre o algoritmo bellman ford'
    fills BOTH purpose and topic in the same turn -- the exact real bug had
    the 'not updated' guard block topic whenever purpose matched too,
    causing Echo to re-ask 'Sobre que assunto?' with the answer already
    given."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos escrever um email")
    engine.respond("é para o professor João Cardoso")
    response = engine.respond("quero pedir esclarecimentos sobre o algoritmo bellman ford")
    telemetry = engine.get_last_turn_telemetry() or {}
    fields = telemetry["task_known_fields"]

    assert fields.get("purpose") == "request_clarification"
    assert "bellman" in fields.get("topic", "").lower()
    assert "sobre que assunto" not in response.lower()
    assert ollama.calls == []


def test_task_semantics_f_missing_information_correct(tmp_path: Path) -> None:
    """F: right after creating the task with no data at all, missing
    starts at ['recipient'] -- the minimum sufficient set (section 4)."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos escrever um email")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["task_missing_fields"] == ["recipient"]
    assert ollama.calls == []


def test_task_semantics_g_never_reasks_known_field(tmp_path: Path) -> None:
    """G: once recipient/role/course/purpose/topic are all known, the task
    reaches status=ready and stops asking questions -- section 4/7."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("chega te para a esquerda, vamos escrever um mail")
    engine.respond("é para o meu professor de estrategias algoritmicas, luis paquete")
    response = engine.respond("quero pedir esclarecimentos sobre o algoritmo bellman ford")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["active_task_status"] == "ready"
    assert not telemetry["task_missing_fields"]
    assert "?" not in response or "queres que escreva" in response.lower()
    assert ollama.calls == []


def test_task_semantics_h_generation_updates_same_artifact_id(tmp_path: Path) -> None:
    """H: the artifact id created at task-start time is still the SAME one
    after generation -- a real workspace object update, never a fresh card."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local", ollama_replies=[_BELLMAN_FORD_EMAIL_REPLY])

    engine.respond("vamos escrever um email")
    artifact_id_at_start = (engine.get_last_turn_telemetry() or {})["task_artifact_id"]
    engine.respond("é para o meu professor de estrategias algoritmicas, luis paquete")
    engine.respond("quero pedir esclarecimentos sobre o algoritmo bellman ford")
    engine.respond("podes escrever")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert artifact_id_at_start
    assert telemetry["task_artifact_id"] == artifact_id_at_start
    assert telemetry["task_artifact_revision"] >= 4  # one bump per artifact-touching turn


def test_task_semantics_i_general_conversation_never_intercepts_control(tmp_path: Path) -> None:
    """I: 'podes escrever' on a ready task never falls to GENERAL_
    CONVERSATION -- the exact real bug had turn 5 ('podes') fall through
    and have Llama invent email content outside the task entirely."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local", ollama_replies=[_BELLMAN_FORD_EMAIL_REPLY])

    engine.respond("vamos escrever um email")
    engine.respond("é para o meu professor de estrategias algoritmicas, luis paquete")
    engine.respond("quero pedir esclarecimentos sobre o algoritmo bellman ford")
    engine.respond("podes escrever")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] != "GENERAL_CONVERSATION"
    assert telemetry["llm_calls"] == 1


def test_task_semantics_j_prompt_instructs_no_fabrication(tmp_path: Path) -> None:
    """J: the structured generation prompt explicitly forbids inventing
    specifics -- the exact real failure invented 'especialmente com as
    passagens negativas' for a Bellman-Ford question nobody phrased that
    way. Mechanically verifying non-fabrication in the OUTPUT isn't
    feasible without a second LLM judge (documented limitation); this
    verifies the INSTRUCTION the model is actually given."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local", ollama_replies=[_BELLMAN_FORD_EMAIL_REPLY])

    engine.respond("vamos escrever um email")
    engine.respond("é para o meu professor de estrategias algoritmicas, luis paquete")
    engine.respond("quero pedir esclarecimentos sobre o algoritmo bellman ford")
    engine.respond("podes escrever")

    assert len(ollama.calls) == 1
    sent = str(ollama.calls[0]["messages"])
    assert "não inventes" in sent.lower()
    assert "luís paquete" in sent.lower() or "luis paquete" in sent.lower()
    assert "bellman" in sent.lower()


def test_task_semantics_k_ui_never_shows_raw_enum(tmp_path: Path) -> None:
    """K: the progressive artifact shows 'pedir esclarecimentos' in
    natural language, never the raw internal enum 'request_clarification'."""
    import json

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos escrever um email")
    engine.respond("é para o meu professor de estrategias algoritmicas, luis paquete")
    engine.respond("quero pedir esclarecimentos sobre o algoritmo bellman ford")
    payloads = [json.loads(event) for event in engine.consume_ui_events()]
    artifact_events = [p for p in payloads if p["type"] == "task_artifact_updated"]

    assert artifact_events
    last_content = artifact_events[-1]["content"]
    assert "request_clarification" not in last_content
    assert "pedir esclarecimentos" in last_content.lower()
    assert ollama.calls == []


def test_task_semantics_exact_real_conversation(tmp_path: Path) -> None:
    """Section 14: the exact 5-turn real conversation, verbatim (turn 5
    'podes' folded into turn 4's explicit 'podes escrever', which must
    execute immediately without needing a separate confirmation turn)."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local", ollama_replies=[_BELLMAN_FORD_EMAIL_REPLY])

    r1 = engine.respond("chega te para a esquerda, vamos escrever um mail")
    t1 = engine.get_last_turn_telemetry() or {}
    assert t1["selected_path"] == "WORKSPACE_COMMAND"
    assert t1["llm_calls"] == 0
    assert t1["active_task_type"] == "compose_email"
    engine.consume_ui_events()

    r2 = engine.respond("é para o meu professor de estrategias algoritmicas, luis paquete")
    t2 = engine.get_last_turn_telemetry() or {}
    assert "paquete" in t2["task_known_fields"].get("recipient", "").lower()
    assert t2["task_known_fields"].get("recipientRole") == "professor"
    assert "algoritmicas" in t2["task_known_fields"].get("course", "").lower()
    engine.consume_ui_events()

    r3 = engine.respond("quero pedir esclarecimentos sobre o algoritmo bellman ford")
    t3 = engine.get_last_turn_telemetry() or {}
    assert t3["task_known_fields"].get("purpose") == "request_clarification"
    assert "bellman" in t3["task_known_fields"].get("topic", "").lower()
    assert t3["active_task_status"] == "ready"
    assert not t3["task_missing_fields"]
    engine.consume_ui_events()

    r4 = engine.respond("podes escrever")
    t4 = engine.get_last_turn_telemetry() or {}
    assert t4["task_turn_intent"] == "control"
    assert t4["task_control_action"] == "execute"
    assert t4["task_execution_started"] is True
    assert t4["task_execution_completed"] is True
    assert t4["llm_calls"] == 1
    assert t4["active_task_status"] == "reviewing"
    assert "passagens negativas" not in r4.lower()

    assert len(ollama.calls) == 1


# --- Passo 3.7 (output validation + fault attribution): a real test showed
# turn 2 losing the recipient entirely with a comma-less utterance ("para o
# meu professor de estrategias algoritmicas luis paquete"), turns 5/6 falling
# to GENERAL_CONVERSATION after a draft already existed (reviewing wrongly
# read as "task over"), and turn 4's model reply asking for more information
# instead of writing the email -- accepted uncritically before this phase.

_BAD_META_REPLY = (
    "Para redigir o email, preciso saber mais sobre o que o Alexandre gostaria de saber "
    "sobre o algoritmo Bellman Ford. Qual é a dúvida específica?"
)
_GOOD_BELLMAN_REPLY = (
    "Caro Professor Luis Paquete,\n\n"
    "Gostaria de pedir alguns esclarecimentos sobre o algoritmo bellman ford, "
    "nomeadamente sobre o funcionamento e as limitacoes, e tambem sobre as vantagens "
    "e desvantagens em relacao ao dijkstra.\n\n"
    "Com os melhores cumprimentos,\nAlexandre"
)


def test_task_semantics_comma_less_recipient_role_course(tmp_path: Path) -> None:
    """Section 1: 'para o meu professor de estrategias algoritmicas luis
    paquete' -- no commas at all, exactly the real repro -- must still
    extract recipient/role/course, not just role."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos escrever um mail")
    engine.respond("para o meu professor de estrategias algoritmicas luis paquete")
    telemetry = engine.get_last_turn_telemetry() or {}
    fields = telemetry["task_known_fields"]

    assert fields.get("recipientRole") == "professor"
    assert "paquete" in fields.get("recipient", "").lower()
    assert "algoritmicas" in fields.get("course", "").lower()
    assert telemetry["fault_detected"] is False
    assert ollama.calls == []


def test_task_semantics_reviewing_follow_up_stays_in_task(tmp_path: Path) -> None:
    """Section 2/3: once a draft exists (status=reviewing), a content
    fragment ('o funcionamento e as limitações') must stay in
    TASK_CONTINUATION, never fall to GENERAL_CONVERSATION -- the exact
    real bug had turns 5/6 escape as soon as a draft existed."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local", ollama_replies=[_GOOD_BELLMAN_REPLY])

    engine.respond("vamos escrever um mail")
    engine.respond("para o meu professor de estrategias algoritmicas luis paquete")
    engine.respond("quero pedir esclarecimentos sobre o algoritmo bellman ford")
    engine.respond("podes escrever")
    response = engine.respond("o funcionamento e as limitações")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "TASK_CONTINUATION"
    assert telemetry["llm_calls"] == 0
    assert response.strip() == "Certo."
    assert "funcionamento" in engine._active_task_context.key_points[-1].lower()
    assert engine._active_task_context.parameters.get("topic")  # topic never overwritten
    assert "bellman" in engine._active_task_context.parameters.get("topic", "").lower()


def test_task_semantics_key_points_accumulate_without_dropping_topic(tmp_path: Path) -> None:
    """Section 9: two enrichment turns both accumulate into key_points,
    Bellman-Ford stays the topic throughout."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local", ollama_replies=[_GOOD_BELLMAN_REPLY])

    engine.respond("vamos escrever um mail")
    engine.respond("para o meu professor de estrategias algoritmicas luis paquete")
    engine.respond("quero pedir esclarecimentos sobre o algoritmo bellman ford")
    engine.respond("podes escrever")
    engine.respond("o funcionamento e as limitações")
    r6 = engine.respond("as vantagens e desvantagens em relação ao dijkstra")
    telemetry = engine.get_last_turn_telemetry() or {}
    ctx = engine._active_task_context

    assert telemetry["selected_path"] == "TASK_CONTINUATION"
    assert telemetry["llm_calls"] == 0
    assert len(ctx.key_points) == 2
    assert "dijkstra" in ctx.key_points[-1].lower()
    assert "bellman" in ctx.parameters.get("topic", "").lower()
    assert "?" not in r6  # never answered as a tutor


def test_task_semantics_explicit_question_still_exits_task(tmp_path: Path) -> None:
    """Section 3/4: a GENUINE question about the topic ('o que é
    Bellman-Ford?') still uses normal conversation for that turn -- content
    fragments are the exception that stays in-task, not the rule that
    blocks all conversation."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local", ollama_replies=["O Bellman-Ford é um algoritmo de caminho mais curto."])

    engine.respond("vamos escrever um mail")
    engine.respond("para o meu professor de estrategias algoritmicas luis paquete")
    engine.respond("quero pedir esclarecimentos sobre o algoritmo bellman ford")
    telemetry = engine.get_last_turn_telemetry() or {}
    assert telemetry["active_task_status"] == "ready"

    engine.respond("o que e bellman ford")
    telemetry2 = engine.get_last_turn_telemetry() or {}
    assert telemetry2["selected_path"] != "TASK_CONTINUATION"


def test_task_semantics_output_validation_rejects_meta_question(tmp_path: Path) -> None:
    """Section 6/14: the model asking for more information instead of
    writing the email must be REJECTED by validation, not accepted as the
    artifact -- the exact real failure."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local", ollama_replies=[_BAD_META_REPLY, _BAD_META_REPLY])

    engine.respond("vamos escrever um mail")
    engine.respond("para o meu professor de estrategias algoritmicas luis paquete")
    engine.respond("quero pedir esclarecimentos sobre o algoritmo bellman ford")
    engine.respond("podes escrever")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["task_output_validation_passed"] is False
    assert telemetry["task_output_validation_reason"] == "asked_for_more_information"
    assert telemetry["recovery_attempted"] is True
    assert telemetry["recovery_succeeded"] is False
    assert telemetry["active_task_status"] == "ready"  # not left in reviewing with junk content
    assert len(ollama.calls) == 2


def test_task_semantics_artifact_never_shows_invalid_meta_response(tmp_path: Path) -> None:
    """Section 8: even when both generation attempts fail, the artifact
    must never be replaced by the model's meta-response -- the exact real
    bug had the card show 'Para redigir o email, preciso saber mais...'."""
    import json

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local", ollama_replies=[_BAD_META_REPLY, _BAD_META_REPLY])

    engine.respond("vamos escrever um mail")
    engine.respond("para o meu professor de estrategias algoritmicas luis paquete")
    engine.respond("quero pedir esclarecimentos sobre o algoritmo bellman ford")
    engine.respond("podes escrever")
    payloads = [json.loads(event) for event in engine.consume_ui_events()]
    artifact_events = [p for p in payloads if p["type"] == "task_artifact_updated"]

    assert artifact_events
    assert "preciso saber mais" not in artifact_events[-1]["content"].lower()


def test_task_semantics_regeneration_succeeds_on_second_attempt(tmp_path: Path) -> None:
    """Section 7/15: first attempt fails validation, correction prompt
    triggers a second attempt via the normal router, second attempt is a
    valid email -> reviewing, recovery_succeeded=True."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local", ollama_replies=[_BAD_META_REPLY, _GOOD_BELLMAN_REPLY])

    engine.respond("vamos escrever um mail")
    engine.respond("para o meu professor de estrategias algoritmicas luis paquete")
    engine.respond("quero pedir esclarecimentos sobre o algoritmo bellman ford")
    engine.respond("podes escrever")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["task_output_validation_passed"] is True
    assert telemetry["recovery_attempted"] is True
    assert telemetry["recovery_succeeded"] is True
    assert telemetry["active_task_status"] == "reviewing"
    assert telemetry["task_generation_attempt"] == 2
    assert telemetry["task_generation_max_attempts"] == 2
    assert len(ollama.calls) == 2


def test_task_semantics_success_path_fault_fields_explicit(tmp_path: Path) -> None:
    """Section 17: a clean success (valid on the first attempt) must
    explicitly set fault_detected=False/stage=none/owner=none -- absence of
    a fault is never left to be inferred from missing fields."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local", ollama_replies=[_GOOD_BELLMAN_REPLY])

    engine.respond("vamos escrever um mail")
    engine.respond("para o meu professor de estrategias algoritmicas luis paquete")
    engine.respond("quero pedir esclarecimentos sobre o algoritmo bellman ford")
    engine.respond("podes escrever")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["fault_detected"] is False
    assert telemetry["fault_stage"] == "none"
    assert telemetry["fault_owner"] == "none"
    assert telemetry["recovery_attempted"] is False
    assert telemetry["task_output_validation_passed"] is True


def test_task_semantics_exact_real_conversation_six_turns(tmp_path: Path) -> None:
    """Section 15: the exact 6-turn real conversation, verbatim -- attempt
    1 of generation is the bad meta-response, attempt 2 is a valid email,
    then two content-enrichment turns must stay in TASK_CONTINUATION."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local", ollama_replies=[_BAD_META_REPLY, _GOOD_BELLMAN_REPLY])

    r1 = engine.respond("vamos escrever um mail")
    t1 = engine.get_last_turn_telemetry() or {}
    assert t1["selected_path"] == "TASK_CONTINUATION"
    assert t1["active_task_type"] == "compose_email"
    assert t1["llm_calls"] == 0

    r2 = engine.respond("para o meu professor de estrategias algoritmicas luis paquete")
    t2 = engine.get_last_turn_telemetry() or {}
    assert "paquete" in t2["task_known_fields"].get("recipient", "").lower()
    assert t2["task_known_fields"].get("recipientRole") == "professor"
    assert t2["fault_detected"] is False

    r3 = engine.respond("quero pedir esclarecimentos sobre o algoritmo bellman ford")
    t3 = engine.get_last_turn_telemetry() or {}
    assert t3["task_known_fields"].get("purpose") == "request_clarification"
    assert "bellman" in t3["task_known_fields"].get("topic", "").lower()
    assert t3["active_task_status"] == "ready"

    r4 = engine.respond("podes escrever")
    t4 = engine.get_last_turn_telemetry() or {}
    assert t4["task_output_validation_passed"] is True
    assert t4["fault_detected"] is True
    assert t4["fault_stage"] == "model_generation"
    assert t4["fault_owner"] == "model"
    assert t4["recovery_attempted"] is True
    assert t4["recovery_succeeded"] is True
    assert t4["active_task_status"] == "reviewing"

    r5 = engine.respond("o funcionamento e as limitações")
    t5 = engine.get_last_turn_telemetry() or {}
    assert t5["selected_path"] == "TASK_CONTINUATION"
    assert t5["llm_calls"] == 0
    assert len(engine._active_task_context.key_points) == 1

    r6 = engine.respond("as vantagens e desvantagens em relação ao dijkstra")
    t6 = engine.get_last_turn_telemetry() or {}
    assert t6["selected_path"] == "TASK_CONTINUATION"
    assert t6["llm_calls"] == 0
    assert len(engine._active_task_context.key_points) == 2
    assert "?" not in r6

    assert len(ollama.calls) == 2


# --- Section 16: mecanismo vs modelo, explicit attribution tests.

def test_fault_attribution_a_local_parser_no_longer_loses_recipient(tmp_path: Path) -> None:
    """A: the parser bug that used to lose the recipient (llm_calls=0) is
    fixed -- confirms no fault is produced for this exact real input."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos escrever um mail")
    engine.respond("para o meu professor de estrategias algoritmicas luis paquete")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "paquete" in telemetry["task_known_fields"].get("recipient", "").lower()
    assert telemetry["fault_detected"] is False
    assert telemetry["fault_owner"] == "none"


def test_fault_attribution_b_active_task_never_loses_routing(tmp_path: Path) -> None:
    """B: an active task turn must never silently fall to GENERAL_
    CONVERSATION -- confirmed via the cross-cutting expected_route/
    actual_route check in _end_turn_trace."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos escrever um mail")
    engine.respond("para o meu professor de estrategias algoritmicas luis paquete")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "TASK_CONTINUATION"
    assert telemetry["fault_stage"] != "routing"


def test_fault_attribution_c_incomplete_context_blocks_generation(tmp_path: Path) -> None:
    """C: if the mechanism ever tried to execute with missing fields
    (an invariant violation), that is echo_mechanism/prompt_construction,
    never the model's fault -- and no LLM call is even spent on it."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos escrever um mail")
    ctx = engine._active_task_context
    ctx.status = "ready"  # force an invariant violation: "ready" with nothing collected

    engine._begin_turn_trace("podes escrever")
    engine._execute_active_task(ctx)
    telemetry = engine._turn_trace

    assert telemetry["fault_detected"] is True
    assert telemetry["fault_stage"] == "prompt_construction"
    assert telemetry["fault_owner"] == "echo_mechanism"
    assert ollama.calls == []


def test_fault_attribution_d_correct_context_model_still_asks_is_model_fault(tmp_path: Path) -> None:
    """D: context and instruction were both correct (task genuinely ready)
    but the model asked for more information anyway -- fault_owner=model."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local", ollama_replies=[_BAD_META_REPLY, _BAD_META_REPLY])

    engine.respond("vamos escrever um mail")
    engine.respond("para o meu professor de estrategias algoritmicas luis paquete")
    engine.respond("quero pedir esclarecimentos sobre o algoritmo bellman ford")
    engine.respond("podes escrever")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["expected_context_present"] is True
    assert telemetry["model_received_expected_context"] is True
    assert telemetry["fault_owner"] == "model"
    assert telemetry["fault_stage"] == "model_generation"


def test_fault_attribution_e_invalid_output_caught_and_recovery_attempted(tmp_path: Path) -> None:
    """E: the validator catches the invalid output -- fault_detected=True,
    recovery_attempted=True, regardless of whether attempt 2 succeeds."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local", ollama_replies=[_BAD_META_REPLY, _BAD_META_REPLY])

    engine.respond("vamos escrever um mail")
    engine.respond("para o meu professor de estrategias algoritmicas luis paquete")
    engine.respond("quero pedir esclarecimentos sobre o algoritmo bellman ford")
    engine.respond("podes escrever")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["fault_detected"] is True
    assert telemetry["recovery_attempted"] is True


def test_fault_attribution_f_valid_output_no_fault(tmp_path: Path) -> None:
    """F: the model returns a correct output on the first try -- no fault,
    regardless of the fact that the mechanism VALIDATED it (validation
    running is not itself a fault)."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local", ollama_replies=[_GOOD_BELLMAN_REPLY])

    engine.respond("vamos escrever um mail")
    engine.respond("para o meu professor de estrategias algoritmicas luis paquete")
    engine.respond("quero pedir esclarecimentos sobre o algoritmo bellman ford")
    engine.respond("podes escrever")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["fault_detected"] is False
    assert telemetry["recovery_attempted"] is False


def test_fault_attribution_g_ui_event_emitted_after_valid_artifact(tmp_path: Path) -> None:
    """G: backend succeeds -> the UI event for the updated artifact IS
    emitted (checked directly, rather than assumed) -- if this ever failed
    it would be attribution=ui/ui_render, a layer this phase does not
    inject failures into, only verifies stays wired."""
    import json

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local", ollama_replies=[_GOOD_BELLMAN_REPLY])

    engine.respond("vamos escrever um mail")
    engine.respond("para o meu professor de estrategias algoritmicas luis paquete")
    engine.respond("quero pedir esclarecimentos sobre o algoritmo bellman ford")
    engine.respond("podes escrever")
    payloads = [json.loads(event) for event in engine.consume_ui_events()]
    artifact_events = [p for p in payloads if p["type"] == "task_artifact_updated"]

    assert artifact_events
    assert "bellman" in artifact_events[-1]["content"].lower()


# --- Phase 1 (Cognitive Workspace Blueprint v2): WorkSession skeleton. A
# pure state-model refactor -- ActiveTaskContext, ActiveDocumentContext,
# and PendingWritingTask converge into one `engine._work_session`, with
# `_active_task_context`/`_active_document_context`/`_pending_writing_task`
# living on as compatibility properties (read/write, never a second copy).
# No routing/behavior change is expected anywhere else in this file; these
# tests target the migration itself.

def test_work_session_a_active_task_compat_api_writes_through(tmp_path: Path) -> None:
    """A: changing the active task through the compatibility API changes
    WorkSession -- same object, not a copy."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos escrever um mail")

    assert engine._work_session.active_task is not None
    assert engine._active_task_context is engine._work_session.active_task

    engine._active_task_context = None

    assert engine._work_session.active_task is None
    assert ollama.calls == []


def test_work_session_b_document_context_compat_api_writes_through(tmp_path: Path) -> None:
    """B: changing the document context through the compatibility API
    changes WorkSession -- same object, not a copy."""
    import assistant.conversation as conversation_module

    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")
    ctx = conversation_module.ActiveDocumentContext(resolved_file="email_resumo_novo.txt", content="ola")

    engine._active_document_context = ctx

    assert engine._work_session.document_context is ctx

    engine._active_document_context = None

    assert engine._work_session.document_context is None


def test_work_session_c_legacy_writing_context_lives_inside(tmp_path: Path) -> None:
    """C: the legacy PendingWritingTask (kept per the blueprint -- retired
    only in a later phase) lives inside WorkSession, not a separate store."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="automatic")

    engine.respond("Preciso de ajuda a escrever um email a um professor.")

    assert engine._work_session.writing_context is not None
    assert engine._pending_writing_task is engine._work_session.writing_context
    assert engine._work_session.writing_context.task_type == "compose_email"
    assert ollama.calls == []


def test_work_session_d_conversation_clear_resets_all_three(tmp_path: Path) -> None:
    """D: clear_conversation() clears exactly the same state as before --
    active task, document context, legacy writing context, and the paused
    task stack, all together, via WorkSession."""
    import assistant.conversation as conversation_module

    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="automatic")
    engine.respond("vamos escrever um mail")
    engine._active_document_context = conversation_module.ActiveDocumentContext(resolved_file="x.txt")
    engine._work_session.paused_task_stack.append(conversation_module.ActiveTaskContext(id="task-99"))

    assert engine._work_session.active_task is not None
    assert engine._work_session.document_context is not None
    assert engine._work_session.paused_task_stack

    engine.clear_conversation()

    assert engine._work_session.active_task is None
    assert engine._work_session.document_context is None
    assert engine._work_session.writing_context is None
    assert engine._work_session.paused_task_stack == []


def test_work_session_e_topic_shift_resets_equivalently(tmp_path: Path) -> None:
    """E: an explicit topic-shift phrase (_is_topic_shift_request's exact
    vocabulary, e.g. "vamos mudar de tema" -- distinct from the narrower,
    document-only _looks_like_document_context_clear check exercised by
    test_active_document_clear_on_topic_shift) resets the same state
    clear_conversation() does, via the same _reset_transient_context."""
    engine, _llm, ollama, anthropic = make_engine(tmp_path, mode="automatic")

    engine.respond("vamos escrever um mail")
    assert engine._work_session.active_task is not None

    engine.respond("vamos mudar de tema")

    assert engine._work_session.active_task is None


def test_work_session_f_document_ttl_expiry_equivalent(tmp_path: Path) -> None:
    """F: ActiveDocumentContext's own ttl-decrement-then-expire behavior
    (in the per-turn loop) still runs correctly against WorkSession's
    stored copy -- ttl=1 expires to None after exactly one more turn."""
    import assistant.conversation as conversation_module

    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine._active_document_context = conversation_module.ActiveDocumentContext(resolved_file="x.txt", ttl=1)

    engine.respond("ola")

    assert engine._work_session.document_context is None


def test_work_session_g_document_draft_and_revision_state_survives(tmp_path: Path) -> None:
    """G: draft content, previous-draft content, and revision counters
    round-trip through the compatibility property exactly as fields on the
    same WorkSession-held object -- no field dropped or renamed."""
    import assistant.conversation as conversation_module

    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")
    ctx = conversation_module.ActiveDocumentContext(
        resolved_file="email_resumo_novo.txt",
        content="original",
        draft_content="rascunho novo",
        draft_revision=2,
        previous_draft_content="rascunho anterior",
        previous_draft_revision=1,
    )

    engine._active_document_context = ctx

    stored = engine._work_session.document_context
    assert stored.content == "original"
    assert stored.draft_content == "rascunho novo"
    assert stored.draft_revision == 2
    assert stored.previous_draft_content == "rascunho anterior"
    assert stored.previous_draft_revision == 1


def test_work_session_h_email_task_state_survives_real_turns(tmp_path: Path) -> None:
    """H: recipient/role/course/purpose/topic collected across several real
    turns are visible on WorkSession.active_task, not just the compat
    property -- the exact real-conversation scenario from a prior phase."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos escrever um mail")
    engine.respond("para o meu professor de estrategias algoritmicas luis paquete")
    engine.respond("quero pedir esclarecimentos sobre o algoritmo bellman ford")

    task = engine._work_session.active_task
    assert task is not None
    assert task.entities.get("recipientRole") == "professor"
    assert "paquete" in task.entities.get("recipient", "").lower()
    assert task.parameters.get("purpose") == "request_clarification"
    assert "bellman" in task.parameters.get("topic", "").lower()
    assert task.status == "ready"
    assert ollama.calls == []


def test_work_session_i_fault_attribution_sees_correct_task_state(tmp_path: Path) -> None:
    """I: fault attribution (Passo 3.7) still reads the real, current task
    state now that it lives on WorkSession -- a model-generation fault is
    still attributed correctly, and the task state it reasoned about is
    exactly what's on WorkSession.active_task."""
    _BAD_META_REPLY = (
        "Para redigir o email, preciso saber mais sobre o que o Alexandre gostaria de saber "
        "sobre o algoritmo Bellman Ford. Qual é a dúvida específica?"
    )
    engine, _llm, ollama, _anthropic = make_engine(
        tmp_path, mode="local", ollama_replies=[_BAD_META_REPLY, _GOOD_BELLMAN_REPLY]
    )

    engine.respond("vamos escrever um mail")
    engine.respond("para o meu professor de estrategias algoritmicas luis paquete")
    engine.respond("quero pedir esclarecimentos sobre o algoritmo bellman ford")
    engine.respond("podes escrever")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["fault_owner"] == "model"
    assert telemetry["fault_stage"] == "model_generation"
    assert engine._work_session.active_task is not None
    assert engine._work_session.active_task.status == "reviewing"
    assert "bellman" in engine._work_session.active_task.parameters.get("topic", "").lower()


def test_work_session_j_no_duplicated_writable_state(tmp_path: Path) -> None:
    """J: the compatibility names are pure properties -- no separate
    instance attribute shadows them, so there is exactly one writable copy
    of this state (WorkSession), never two."""
    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos escrever um mail")
    instance_dict = vars(engine)

    assert "_active_task_context" not in instance_dict
    assert "_active_document_context" not in instance_dict
    assert "_pending_writing_task" not in instance_dict
    assert "_paused_task_stack" not in instance_dict
    assert "_work_session" in instance_dict
    assert engine._active_task_context is engine._work_session.active_task

    # Migration telemetry (section 12) reports the same live state.
    telemetry = engine.get_last_turn_telemetry() or {}
    assert telemetry["work_session_active"] is True
    assert telemetry["work_session_active_task_present"] is True
    assert telemetry["work_session_document_present"] is False
    assert telemetry["work_session_legacy_writing_present"] is False


# --- Phase 2 (Cognitive Workspace Blueprint v2): Interaction Interpreter,
# SHADOW MODE ONLY. These tests prove the shadow call changes nothing
# about production behavior -- it only ever writes interpreter_* keys
# into telemetry. Classification correctness itself is covered by
# tests/test_interaction_interpreter.py (engine-free unit tests); these
# are the integration-safety proofs section 19 asks for specifically.

def test_shadow_interpreter_populates_telemetry_alongside_real_routing(tmp_path: Path) -> None:
    """Sanity: shadow telemetry is present on an ordinary deterministic
    turn, without changing the real selected_path."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    response = engine.respond("chega-te para a esquerda")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "WORKSPACE_COMMAND"
    assert telemetry["interpreter_shadow_enabled"] is True
    assert telemetry["interpreter_spatial_present"] is True
    assert telemetry["interpreter_spatial_action"] == "move"
    assert telemetry["interpreter_expected_route"] == "WORKSPACE_COMMAND"
    assert telemetry["interpreter_route_match"] == "MATCH"
    assert telemetry["actual_selected_path"] == "WORKSPACE_COMMAND"
    assert telemetry["interpreter_shadow_latency_ms"] >= 0.0
    assert "Feito." in response
    assert ollama.calls == []


def test_shadow_interpreter_does_not_change_work_session(tmp_path: Path) -> None:
    """WorkSession must be bit-for-bit identical before and after the
    shadow call runs -- interpret_shadow is read-only by construction, but
    this proves it end-to-end through the real engine, not just at the
    interpreter's own unit-test level."""
    import copy

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("vamos escrever um mail")
    engine.respond("é para o meu professor de estrategias algoritmicas luis paquete")
    before = copy.deepcopy(engine._work_session)

    engine.respond("chega-te para a esquerda, mantém isto aqui")

    after = engine._work_session
    # The active task's content must be exactly what it was -- the shadow
    # interpreter reading it during classification must not have added,
    # removed, or altered a single field.
    assert after.active_task.entities == before.active_task.entities
    assert after.active_task.parameters == before.active_task.parameters
    assert after.active_task.key_points == before.active_task.key_points
    assert after.active_task.status == before.active_task.status
    assert after.paused_task_stack == before.paused_task_stack
    assert ollama.calls == []


def test_shadow_interpreter_adds_no_provider_call_on_deterministic_turns(tmp_path: Path) -> None:
    """Deterministic interpreter cases must never spend an extra provider
    call -- shadow classification runs entirely without self.llm/
    self.response_composer, on top of routes that were already
    llm_calls=0 before Phase 2 existed."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    for message in ("vamos escrever um mail", "chega-te para a esquerda"):
        engine.respond(message)
        telemetry = engine.get_last_turn_telemetry() or {}
        assert telemetry["interpreter_shadow_enabled"] is True
        assert telemetry["llm_calls"] == 0

    assert ollama.calls == []


def test_shadow_interpreter_does_not_change_ui_events(tmp_path: Path) -> None:
    """The UI event stream for a workspace command is exactly what it was
    before Phase 2 -- shadow telemetry lives only in turn telemetry, never
    in a queued UI event."""
    import json

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("chega-te para a esquerda")
    payloads = [json.loads(event) for event in engine.consume_ui_events()]

    command_events = [p for p in payloads if p["type"] == "workspace_command"]
    assert len(command_events) == 1
    assert command_events[0]["action"] == "move"
    assert command_events[0]["region"] == "left"
    assert not any("interpreter" in str(p).lower() for p in payloads)
    assert ollama.calls == []


def test_shadow_interpreter_multi_intent_telemetry_on_real_mixed_turn(tmp_path: Path) -> None:
    """A real mixed-intent utterance (workspace move + email trigger,
    already handled specially by _try_workspace_command) is ALSO
    classified with multi_intent_count>=2 by the shadow interpreter, and
    the real production route is completely unaffected."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    response = engine.respond("desloca-te para a direita, vamos escrever um mail")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "WORKSPACE_COMMAND"
    assert telemetry["active_task_type"] == "compose_email"
    assert telemetry["interpreter_activity_present"] is True
    assert telemetry["interpreter_spatial_present"] is True
    assert telemetry["interpreter_multi_intent_count"] >= 2
    assert telemetry["interpreter_expected_route"] == "WORKSPACE_COMMAND"
    assert telemetry["interpreter_route_match"] == "MATCH"
    assert "quem" in response.lower()
    assert ollama.calls == []


def test_shadow_interpreter_reports_mismatch_without_altering_fault_attribution(tmp_path: Path) -> None:
    """Section 13: a genuine interpreter/production disagreement is
    reported as interpreter_route_match='MISMATCH', never as
    fault_detected -- shadow evidence and real fault attribution stay
    distinct concepts."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local", ollama_replies=["Resposta qualquer."])

    engine.respond("olá, tudo bem?")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["interpreter_conversational_present"] is True
    assert telemetry["interpreter_expected_route"] == "GENERAL_CONVERSATION"
    assert telemetry["interpreter_route_match"] in ("MATCH", "MISMATCH")
    assert telemetry["fault_detected"] is False
    assert telemetry["fault_stage"] == "none"
    assert telemetry["fault_owner"] == "none"


# --- Phase 3 (Cognitive Workspace Blueprint v2): REPRESENTATION
# TRANSFORMATION. These tests prove representation is a transformable VIEW
# over stable WorkSession data -- the same semantic objects move between
# cards/list/side_by_side/conversation_only without reconstruction,
# re-extraction, or identity loss. Tests A-J exercise assistant.
# representation directly (engine-free, matching the interaction_
# interpreter unit-test convention); K-N go through a real AssistantEngine.

def test_representation_a_default_and_creation(tmp_path: Path) -> None:
    """A: a fresh WorkSession has no representation yet; the first
    transform_representation call creates a real RepresentationState with
    sensible defaults."""
    session = WorkSession()
    assert session.representation is None

    state = transform_representation(session, "cards")
    assert state.style == "cards"
    assert state.style_params == {}
    assert state.visible_object_ids == []
    assert state.previous is None
    assert session.representation is state


def test_representation_b_central_transform_operation(tmp_path: Path) -> None:
    """B: transform_representation is the ONE transition operation -- an
    unsupported style raises rather than silently no-op, and each call
    updates WorkSession.representation in place, chaining .previous."""
    session = WorkSession()
    register_object(session, SemanticObject(id="a", title="A"))

    with pytest.raises(ValueError):
        transform_representation(session, "mind_map")

    first = transform_representation(session, "cards")
    second = transform_representation(session, "list")
    assert session.representation is second
    assert second.previous is first
    assert set(SUPPORTED_STYLES) == {"cards", "list", "side_by_side", "conversation_only"}


def test_representation_c_cards_to_list_preserves_identity(tmp_path: Path) -> None:
    """C: cards -> list keeps the SAME object ids/payloads -- no
    re-extraction, no new ids."""
    session = WorkSession()
    register_object(session, SemanticObject(id="a", title="A", payload={"v": 1}))
    register_object(session, SemanticObject(id="b", title="B", payload={"v": 2}))
    before = dict(session.objects)

    transform_representation(session, "cards")
    transform_representation(session, "list")

    assert session.objects == before
    assert session.objects["a"] is before["a"]
    assert session.representation.style == "list"
    assert set(session.representation.visible_object_ids) == {"a", "b"}


def test_representation_d_list_to_side_by_side_preserves_identity(tmp_path: Path) -> None:
    """D: list -> side_by_side keeps the same three object ids; only two
    become "major", the third stays registered and visible."""
    session = WorkSession()
    register_object(session, SemanticObject(id="a", title="A"))
    register_object(session, SemanticObject(id="b", title="B"))
    register_object(session, SemanticObject(id="c", title="C"))
    before = dict(session.objects)

    transform_representation(session, "list")
    state = transform_representation(session, "side_by_side", major_ids=["a", "c"])

    assert session.objects == before
    assert state.major_object_ids == ["a", "c"]
    assert "b" in state.visible_object_ids
    assert "b" in session.objects


def test_representation_e_conversation_only_preserves_objects_and_knowledge(tmp_path: Path) -> None:
    """E: conversation_only hides everything but deletes nothing -- objects
    and knowledge stay exactly as they were."""
    session = WorkSession()
    register_object(session, SemanticObject(id="a", title="A"))
    register_object(session, SemanticObject(id="b", title="B"))
    session.knowledge["note"] = "preserved marker"
    before_objects = dict(session.objects)

    state = transform_representation(session, "conversation_only")

    assert set(state.hidden_object_ids) == {"a", "b"}
    assert state.visible_object_ids == []
    assert session.objects == before_objects
    assert session.knowledge == {"note": "preserved marker"}


def test_representation_f_restore_previous_representation(tmp_path: Path) -> None:
    """F: restore_representation undoes conversation_only back to exactly
    the representation that was active right before hiding -- no
    reconstruction."""
    session = WorkSession()
    register_object(session, SemanticObject(id="a", title="A"))
    register_object(session, SemanticObject(id="b", title="B"))

    transform_representation(session, "side_by_side", major_ids=["a", "b"])
    transform_representation(session, "conversation_only")
    restored = restore_representation(session)

    assert restored is not None
    assert restored.style == "side_by_side"
    assert restored.major_object_ids == ["a", "b"]
    assert session.representation is restored

    # A no-op when there is nothing further back to restore to.
    session_fresh = WorkSession()
    assert restore_representation(session_fresh) is None


def test_representation_g_source_is_recorded_not_hardcoded(tmp_path: Path) -> None:
    """G: RepresentationState.source reflects whatever the caller passed
    (auto/voice_command/echo_suggestion/pointer), never a fixed value."""
    session = WorkSession()
    register_object(session, SemanticObject(id="a", title="A"))

    auto_state = transform_representation(session, "cards", source="auto")
    assert auto_state.source == "auto"

    voice_state = transform_representation(session, "list", source="voice_command")
    assert voice_state.source == "voice_command"

    restored = restore_representation(session, source="voice_command")
    assert restored.source == "voice_command"


def test_representation_h_style_params_mutate_independently_of_style(tmp_path: Path) -> None:
    """H: style_params is a separate mutable channel from style -- the SAME
    style can carry different params across two transforms."""
    session = WorkSession()
    register_object(session, SemanticObject(id="a", title="A"))

    first = transform_representation(session, "cards", style_params={"detail_level": "compact"})
    second = transform_representation(session, "cards", style_params={"detail_level": "expanded"})

    assert first.style == second.style == "cards"
    assert first.style_params != second.style_params
    assert second.style_params == {"detail_level": "expanded"}


def test_representation_i_ui_payload_has_no_geometry(tmp_path: Path) -> None:
    """I: the representation_changed UI event carries only semantic
    instructions -- never a pixel, coordinate, or transition value."""
    import json

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    register_object(engine._work_session, SemanticObject(id="a", title="Suécia"))
    register_object(engine._work_session, SemanticObject(id="b", title="Itália"))
    transform_representation(engine._work_session, "cards", source="auto")

    engine.respond("põe em lista")
    payloads = [json.loads(event) for event in engine.consume_ui_events()]
    representation_events = [p for p in payloads if p["type"] == "representation_changed"]

    assert len(representation_events) == 1
    event = representation_events[0]
    assert event["style"] == "list"
    forbidden_keys = {"x", "y", "width", "height", "left", "top", "position", "transform", "gap", "viewport"}
    assert forbidden_keys.isdisjoint(event.keys())
    assert ollama.calls == []


def test_representation_j_knowledge_unchanged_across_all_transformations(tmp_path: Path) -> None:
    """J: a full cards->list->side_by_side->conversation_only->cards
    sequence never touches WorkSession.knowledge."""
    session = WorkSession()
    register_object(session, SemanticObject(id="a", title="A"))
    register_object(session, SemanticObject(id="b", title="B"))
    session.knowledge["marker"] = "untouched"

    transform_representation(session, "cards")
    transform_representation(session, "list")
    transform_representation(session, "side_by_side", major_ids=["a", "b"])
    transform_representation(session, "conversation_only")
    transform_representation(session, "cards")

    assert session.knowledge == {"marker": "untouched"}


def test_representation_k_existing_email_task_flow_still_renders(tmp_path: Path) -> None:
    """K: the pre-existing compose_email task_artifact_updated flow is
    completely unaffected by Phase 3 -- same event shape as before, plus
    (bonus, not a behavior change) the same artifact id now also exists in
    the semantic object registry via the compatibility adapter."""
    import json

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos escrever um mail")
    payloads = [json.loads(event) for event in engine.consume_ui_events()]
    artifact_events = [p for p in payloads if p["type"] == "task_artifact_updated"]

    assert len(artifact_events) == 1
    assert artifact_events[0]["task_type"] == "compose_email"
    artifact_id = artifact_events[0]["task_id"]
    ctx = engine._work_session.active_task
    assert ctx is not None
    assert ctx.active_artifact_id in engine._work_session.objects
    assert ollama.calls == []


def test_representation_l_shadow_interpreter_still_non_mutating_with_objects(tmp_path: Path) -> None:
    """L: with a non-empty semantic object registry and a real
    representation in place, the Phase 2 shadow interpreter still never
    mutates WorkSession -- an unrelated utterance leaves .objects and
    .representation exactly as they were."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    register_object(engine._work_session, SemanticObject(id="a", title="Suécia"))
    transform_representation(engine._work_session, "cards", source="auto")
    objects_before = dict(engine._work_session.objects)
    representation_before = engine._work_session.representation

    engine.respond("chega-te para a esquerda")

    assert engine._work_session.objects == objects_before
    assert engine._work_session.representation is representation_before
    assert ollama.calls == []


def test_representation_m_synthetic_abc_acceptance_flow(tmp_path: Path) -> None:
    """M (section 16): the required synthetic acceptance flow. Three
    generic objects -- A=Suécia, B=Itália, C=Açores -- move through
    cards -> list -> side_by_side -> conversation_only -> restore ->
    cards, with stable ids/payloads throughout. This is about
    representation transformability, NOT a travel-domain feature."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    session = engine._work_session
    register_object(session, SemanticObject(id="obj-a", title="Suécia", payload={"note": "A"}))
    register_object(session, SemanticObject(id="obj-b", title="Itália", payload={"note": "B"}))
    register_object(session, SemanticObject(id="obj-c", title="Açores", payload={"note": "C"}))
    original_objects = dict(session.objects)

    # (1) representation=cards, A/B/C ids stable.
    transform_representation(session, "cards", source="auto")
    assert session.representation.style == "cards"
    assert set(session.representation.visible_object_ids) == {"obj-a", "obj-b", "obj-c"}
    assert session.objects == original_objects

    # (2) "põe em lista" -> representation=list, same ids/payloads.
    engine.respond("põe em lista")
    assert session.representation.style == "list"
    assert set(session.representation.visible_object_ids) == {"obj-a", "obj-b", "obj-c"}
    assert session.objects == original_objects

    # (3) "põe Suécia e Açores lado a lado" -> side_by_side, major=[A,C], B preserved.
    engine.respond("põe Suécia e Açores lado a lado")
    assert session.representation.style == "side_by_side"
    assert session.representation.major_object_ids == ["obj-a", "obj-c"]
    assert session.objects == original_objects
    assert "obj-b" in session.objects

    # (4) "não quero ver nada, vamos só conversar" -> conversation_only,
    # A/B/C still registered, knowledge untouched, nothing deleted.
    session.knowledge["trip_note"] = "kept"
    engine.respond("não quero ver nada, vamos só conversar")
    assert session.representation.style == "conversation_only"
    assert set(session.representation.hidden_object_ids) == {"obj-a", "obj-b", "obj-c"}
    assert session.objects == original_objects
    assert session.knowledge == {"trip_note": "kept"}

    # (5) "mostra outra vez" -> restores the side_by_side representation
    # from step 3 (major=[A,C]), not a fresh reconstruction.
    engine.respond("mostra outra vez")
    assert session.representation.style == "side_by_side"
    assert session.representation.major_object_ids == ["obj-a", "obj-c"]

    # (6) "mostra em cartões" -> representation=cards, same original three
    # ids/payloads.
    engine.respond("mostra em cartões")
    assert session.representation.style == "cards"
    assert set(session.representation.visible_object_ids) == {"obj-a", "obj-b", "obj-c"}
    assert session.objects == original_objects
    assert session.objects["obj-a"].payload == {"note": "A"}
    assert session.objects["obj-b"].payload == {"note": "B"}
    assert session.objects["obj-c"].payload == {"note": "C"}

    assert ollama.calls == []


def test_representation_n_explicit_commands_are_deterministic(tmp_path: Path) -> None:
    """N: every explicit representation command in the acceptance flow is
    llm_calls=0 -- the narrow voice bridge never touches self.llm."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    register_object(engine._work_session, SemanticObject(id="obj-a", title="Suécia"))
    register_object(engine._work_session, SemanticObject(id="obj-b", title="Açores"))
    transform_representation(engine._work_session, "cards", source="auto")

    for message in (
        "põe em lista",
        "põe Suécia e Açores lado a lado",
        "não quero ver nada, vamos só conversar",
        "mostra outra vez",
        "mostra em cartões",
    ):
        engine.respond(message)
        telemetry = engine.get_last_turn_telemetry() or {}
        assert telemetry["llm_calls"] == 0

    assert ollama.calls == []


# --- Phase 3B integration fix: a real Qt manual run showed the narrow
# representation bridge declining too eagerly (empty registry, nothing to
# restore), which let genuinely representation-flavored utterances fall
# through into DOCUMENT_TASK's file-search heuristics ("outra vez"/
# "cartões" treated as filenames), and a side_by_side placement claim
# "Aqui tens os dois." even when zero named objects actually resolved.
# These tests exercise the WIDENED _try_representation_command (always
# claims a present RepresentationIntent), the FIX 2 truthful-failure path,
# and the new FIX 3 explicit-alternatives materialization -- through the
# real engine, exactly the surface the Qt run exercised.

def test_integration_fix_representation_commands_beat_document_task(tmp_path: Path) -> None:
    """FIX 1: explicit representation commands must never reach
    DOCUMENT_TASK, even with a completely empty semantic object registry
    (the exact condition that let 'outra vez'/'cartões' get treated as
    filenames in the real Qt run)."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    for message in ("não quero ver nada, vamos só conversar", "mostra outra vez", "mostra em cartões", "põe em lista"):
        engine.respond(message)
        telemetry = engine.get_last_turn_telemetry() or {}
        assert telemetry["selected_path"] == "REPRESENTATION_COMMAND", message
        assert telemetry["selected_path"] != "DOCUMENT_TASK", message

    assert ollama.calls == []


def test_integration_fix_mostra_outra_vez_never_becomes_file_search(tmp_path: Path) -> None:
    """'mostra outra vez' with nothing to restore yet is still claimed by
    the representation bridge (a graceful decline), never DOCUMENT_TASK."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    response = engine.respond("mostra outra vez")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "REPRESENTATION_COMMAND"
    assert telemetry["llm_calls"] == 0
    assert "outra vez" not in response.lower() or "vista anterior" in response.lower()
    assert ollama.calls == []


def test_integration_fix_mostra_em_cartoes_never_becomes_file_search(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("mostra em cartões")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "REPRESENTATION_COMMAND"
    assert telemetry["representation_command_action"] == "requested_style"
    assert engine._work_session.representation is not None
    assert engine._work_session.representation.style == "cards"
    assert ollama.calls == []


def test_integration_fix_nao_quero_ver_nada_never_becomes_file_search(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("não quero ver nada, vamos só conversar")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "REPRESENTATION_COMMAND"
    assert engine._work_session.representation.style == "conversation_only"
    assert ollama.calls == []


def test_integration_fix_side_by_side_cannot_claim_success_without_objects(tmp_path: Path) -> None:
    """FIX 2: naming two alternatives that don't resolve to any known
    semantic object must not produce the optimistic 'Aqui tens os dois.'
    ack, and must set representation_target_resolution_failed=True."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("vamos pensar em três opções: Suécia, Itália e Açores")
    engine.consume_ui_events()

    response = engine.respond("põe Marrocos e Egito lado a lado")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "Aqui tens os dois." not in response
    assert telemetry["representation_target_resolution_failed"] is True
    # Nothing changed -- still whatever it was before this turn (cards,
    # from materialization), no major ids invented.
    assert engine._work_session.representation.style == "cards"
    assert engine._work_session.representation.major_object_ids == []
    assert ollama.calls == []


def test_integration_fix_explicit_alternatives_produce_stable_semantic_objects(tmp_path: Path) -> None:
    """FIX 3: an explicit-alternatives utterance materializes stable,
    generic SemanticObjects (kind='option') -- re-registering the exact
    same titles must never duplicate the registry."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos pensar em três opções: Suécia, Itália e Açores")
    telemetry = engine.get_last_turn_telemetry() or {}
    first_ids = sorted(engine._work_session.objects.keys())

    assert telemetry["alternatives_materialized_count"] == 3
    assert len(first_ids) == 3
    assert all(engine._work_session.objects[oid].kind == "option" for oid in first_ids)
    assert engine._work_session.representation.style == "cards"
    assert sorted(engine._work_session.representation.visible_object_ids) == first_ids

    # Re-mentioning the exact same alternatives must not create new ids.
    engine.respond("vamos pensar em três opções: Suécia, Itália e Açores")
    second_ids = sorted(engine._work_session.objects.keys())
    assert second_ids == first_ids
    assert ollama.calls == []


def test_integration_fix_exact_six_turn_flow(tmp_path: Path) -> None:
    """FIX 5: the exact real Qt flow from the integration-fix report,
    end to end through the real engine."""
    import json

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    # Turn 1
    response1 = engine.respond("vamos pensar em três opções: Suécia, Itália e Açores")
    telemetry1 = engine.get_last_turn_telemetry() or {}
    events1 = [json.loads(e) for e in engine.consume_ui_events()]
    ids = sorted(engine._work_session.objects.keys())
    assert len(ids) == 3
    assert telemetry1["llm_calls"] == 0
    assert engine._work_session.representation.style == "cards"
    assert any(e["type"] == "representation_changed" and e["style"] == "cards" for e in events1)
    suecia_id = next(oid for oid, obj in engine._work_session.objects.items() if obj.title == "Suécia")
    acores_id = next(oid for oid, obj in engine._work_session.objects.items() if obj.title == "Açores")
    italia_id = next(oid for oid, obj in engine._work_session.objects.items() if obj.title == "Itália")

    # Turn 2
    engine.respond("põe em lista")
    telemetry2 = engine.get_last_turn_telemetry() or {}
    events2 = [json.loads(e) for e in engine.consume_ui_events()]
    assert telemetry2["selected_path"] == "REPRESENTATION_COMMAND"
    assert telemetry2["llm_calls"] == 0
    assert engine._work_session.representation.style == "list"
    assert any(e["type"] == "representation_changed" and e["style"] == "list" for e in events2)

    # Turn 3
    engine.respond("põe Suécia e Açores lado a lado")
    telemetry3 = engine.get_last_turn_telemetry() or {}
    events3 = [json.loads(e) for e in engine.consume_ui_events()]
    assert telemetry3["llm_calls"] == 0
    assert engine._work_session.representation.style == "side_by_side"
    assert sorted(engine._work_session.representation.major_object_ids) == sorted([suecia_id, acores_id])
    assert italia_id in engine._work_session.objects
    assert any(e["type"] == "representation_changed" and e["style"] == "side_by_side" for e in events3)

    # Turn 4
    engine.respond("não quero ver nada, vamos só conversar")
    telemetry4 = engine.get_last_turn_telemetry() or {}
    events4 = [json.loads(e) for e in engine.consume_ui_events()]
    assert telemetry4["selected_path"] == "REPRESENTATION_COMMAND"
    assert telemetry4["llm_calls"] == 0
    assert engine._work_session.representation.style == "conversation_only"
    assert sorted(engine._work_session.objects.keys()) == ids  # registry untouched
    assert any(e["type"] == "representation_changed" and e["style"] == "conversation_only" for e in events4)

    # Turn 5
    engine.respond("mostra outra vez")
    telemetry5 = engine.get_last_turn_telemetry() or {}
    events5 = [json.loads(e) for e in engine.consume_ui_events()]
    assert telemetry5["selected_path"] == "REPRESENTATION_COMMAND"
    assert telemetry5["llm_calls"] == 0
    assert engine._work_session.representation.style == "side_by_side"
    assert sorted(engine._work_session.representation.major_object_ids) == sorted([suecia_id, acores_id])
    assert any(e["type"] == "representation_changed" and e["style"] == "side_by_side" for e in events5)

    # Turn 6
    engine.respond("mostra em cartões")
    telemetry6 = engine.get_last_turn_telemetry() or {}
    events6 = [json.loads(e) for e in engine.consume_ui_events()]
    assert telemetry6["selected_path"] == "REPRESENTATION_COMMAND"
    assert telemetry6["llm_calls"] == 0
    assert engine._work_session.representation.style == "cards"
    assert sorted(engine._work_session.representation.visible_object_ids) == ids
    assert any(e["type"] == "representation_changed" and e["style"] == "cards" for e in events6)

    assert ollama.calls == []
    assert response1  # non-empty, sanity


def test_integration_fix_no_travel_specific_code_introduced() -> None:
    """No domain vocabulary (travel or otherwise) leaked into the generic
    alternatives-materialization mechanism -- only generic trigger verbs
    (pensar/comparar/considerar) and structural comma/"e" parsing."""
    import assistant.representation as representation_module
    import assistant.conversation as conversation_module

    with open(representation_module.__file__, encoding="utf-8") as handle:
        representation_src = handle.read()

    forbidden = ("destino", "viagem", "ferias", "hotel", "voo", "pais", "turismo", "roteiro")
    lowered = representation_src.lower()
    for word in forbidden:
        assert word not in lowered, f"found domain-specific word {word!r} in assistant/representation.py"

    assert "_try_explicit_alternatives" in dir(conversation_module.AssistantEngine)


# --- Phase 3B integration fix #2: the real Qt run showed the backend's
# SemanticObject registry has no frontend counterpart -- representation_
# changed named ids the frontend had never heard of. AssistantEngine now
# emits one "semantic_object_upserted" UI event per newly-materialized
# object, queued BEFORE the representation_changed event that references
# it (see _queue_semantic_object_sync in assistant/conversation.py).

def test_object_registration_emits_frontend_sync_event(tmp_path: Path) -> None:
    """Backend object registration (materialize-alternatives path) emits
    one semantic_object_upserted event per created object, carrying the
    exact same id/kind/title as the WorkSession registry."""
    import json

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos pensar em três opções: Suécia, Itália e Açores")
    events = [json.loads(e) for e in engine.consume_ui_events()]
    sync_events = [e for e in events if e["type"] == "semantic_object_upserted"]

    assert len(sync_events) == 3
    synced_ids = {e["id"] for e in sync_events}
    assert synced_ids == set(engine._work_session.objects.keys())
    for event in sync_events:
        obj = engine._work_session.objects[event["id"]]
        assert event["kind"] == obj.kind
        assert event["title"] == obj.title
    assert ollama.calls == []


def test_sync_events_ordered_before_representation_changed(tmp_path: Path) -> None:
    """Section 3: every semantic_object_upserted event for a turn must be
    queued before the representation_changed event that references those
    ids -- the renderer must never see a representation naming an id it
    hasn't been told about yet."""
    import json

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos pensar em três opções: Suécia, Itália e Açores")
    events = [json.loads(e) for e in engine.consume_ui_events()]

    sync_indices = [i for i, e in enumerate(events) if e["type"] == "semantic_object_upserted"]
    representation_indices = [i for i, e in enumerate(events) if e["type"] == "representation_changed"]

    assert len(sync_indices) == 3
    assert len(representation_indices) == 1
    assert max(sync_indices) < min(representation_indices)
    assert ollama.calls == []


def test_email_task_artifact_path_unaffected_by_object_sync(tmp_path: Path) -> None:
    """Section 10: the existing task_artifact_updated/upsertTaskArtifact
    identity pipeline is left alone -- no duplicate semantic_object_upserted
    event is emitted for the SAME id an active task already syncs via its
    own event."""
    import json

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos escrever um mail")
    events = [json.loads(e) for e in engine.consume_ui_events()]

    artifact_events = [e for e in events if e["type"] == "task_artifact_updated"]
    sync_events = [e for e in events if e["type"] == "semantic_object_upserted"]

    assert len(artifact_events) == 1
    assert sync_events == []  # task_artifact_updated is the one identity-sync event for this id
    ctx = engine._work_session.active_task
    assert ctx.active_artifact_id in engine._work_session.objects
    assert ollama.calls == []


# --- Phase 3C: Cognitive Layout Baseline + truthful mixed-intent handling.
# The real Qt manual test found "chega o card dos Açores para o outro
# lado... acrescenta também Madeira" falling through to GENERAL_CONVERSATION,
# where Llama printed a bare "Feito!" with zero real workspace mutation.
# _try_mixed_representation_intent (assistant/conversation.py) now claims
# any utterance that (a) has an active generic option set and (b) names
# one more alternative to add, executes exactly that, and is explicit
# about anything else it could not do -- never an unqualified success ack.

def test_mixed_intent_adds_option_to_existing_set(tmp_path: Path) -> None:
    """FIX 11: 'acrescenta também Madeira' with an active option set
    creates a real fourth SemanticObject, syncs it to the frontend, and
    preserves the current representation style."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("vamos pensar em três opções: Suécia, Itália e Açores")
    engine.consume_ui_events()
    style_before = engine._work_session.representation.style

    response = engine.respond("acrescenta também Madeira")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "REPRESENTATION_COMMAND"
    assert telemetry["llm_calls"] == 0
    titles = sorted(obj.title for obj in engine._work_session.objects.values())
    assert titles == ["Açores", "Itália", "Madeira", "Suécia"]
    assert engine._work_session.representation.style == style_before
    assert "Madeira" in response
    assert ollama.calls == []


def test_mixed_intent_exact_utterance_never_false_success(tmp_path: Path) -> None:
    """FIX 10/12: the exact real-Qt utterance -- combining a named spatial
    move, a grouping request, and an explicit addition -- must never
    reach GENERAL_CONVERSATION/an LLM, and must never claim unqualified
    success ('Feito'/'Aqui tens') for the parts it did not execute."""
    import json

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local", ollama_replies=["Feito!"])
    engine.respond("vamos pensar em três opções: Suécia, Itália e Açores")
    engine.consume_ui_events()

    mixed_utterance = (
        "chega o card dos Açores para o outro lado, para distinguir destinos "
        "portugueses de internacionais. acrescenta também Madeira"
    )
    response = engine.respond(mixed_utterance)
    telemetry = engine.get_last_turn_telemetry() or {}
    events = [json.loads(e) for e in engine.consume_ui_events()]

    assert telemetry["selected_path"] == "REPRESENTATION_COMMAND"
    assert telemetry["selected_path"] != "GENERAL_CONVERSATION"
    assert telemetry["llm_calls"] == 0
    assert response.strip() not in ("Feito!", "Feito.", "Aqui tens.", "Já está.")
    assert "Madeira" in response
    titles = sorted(obj.title for obj in engine._work_session.objects.values())
    assert titles == ["Açores", "Itália", "Madeira", "Suécia"]
    assert any(e["type"] == "semantic_object_upserted" and e["title"] == "Madeira" for e in events)
    assert ollama.calls == []


def test_mixed_intent_reports_partial_execution_truthfully(tmp_path: Path) -> None:
    """FIX 12: requested_intent_count/executed_intent_count/
    failed_intent_count/partial_execution must reflect that only the
    'add' sub-intent executed -- the spatial move and grouping request
    were acknowledged as unaddressed, not silently dropped."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("vamos pensar em três opções: Suécia, Itália e Açores")
    engine.consume_ui_events()

    mixed_utterance = (
        "chega o card dos Açores para o outro lado, para distinguir destinos "
        "portugueses de internacionais. acrescenta também Madeira"
    )
    response = engine.respond(mixed_utterance)
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["requested_intent_count"] >= 2
    assert telemetry["executed_intent_count"] == 1
    assert telemetry["failed_intent_count"] >= 1
    assert telemetry["partial_execution"] is True
    assert "Madeira" in response
    assert ollama.calls == []


def test_mixed_intent_simple_add_is_not_reported_as_partial(tmp_path: Path) -> None:
    """A plain 'acrescenta também X' with nothing else asked is a clean,
    fully-executed turn -- partial_execution must be False, not an
    over-cautious always-partial default."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("vamos pensar em três opções: Suécia, Itália e Açores")
    engine.consume_ui_events()

    engine.respond("acrescenta também Madeira")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["requested_intent_count"] == 1
    assert telemetry["executed_intent_count"] == 1
    assert telemetry["failed_intent_count"] == 0
    assert telemetry["partial_execution"] is False
    assert ollama.calls == []


def test_mixed_intent_fires_as_first_batch_without_a_pre_existing_option_set(tmp_path: Path) -> None:
    """Phase 4 integration fix (section 9): reverses the old expectation --
    'acrescenta também Madeira' with NO pre-existing option set is now a
    perfectly ordinary way to name the FIRST option too (e.g. while a
    plan_trip task is active and nothing has been materialized yet), not
    something to decline. The explicit "add" trigger verb plus a real
    extracted name is already a confident, narrow signal on its own; the old
    has_option_set gate was only ever an accident of this handler being
    written as an "add to" companion to _try_explicit_alternatives, not a
    deliberate safety requirement."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("acrescenta também Madeira")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry.get("representation_command_action") == "mixed_add_option"
    assert any(obj.title == "Madeira" for obj in engine._work_session.objects.values())


# --- Phase 3C final patch: grouping requests must be claimed by the
# representation path (never GENERAL_CONVERSATION/DOCUMENT_TASK) but must
# NEVER mutate style just to produce a success-shaped reply -- the real Qt
# bug was silently reapplying style=cards and saying "Aqui tens em
# cartões." because the utterance happened to contain the word "cards".

def test_grouping_i_unsupported_grouping_is_truthful_and_does_not_mutate(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local", ollama_replies=["Aqui tens em cartões."])
    engine.respond("vamos pensar em três opções: Suécia, Itália e Açores")
    engine.consume_ui_events()
    style_before = engine._work_session.representation.style

    response = engine.respond("separa os cards dos destinos internacionais dos destinos portugueses")
    telemetry = engine.get_last_turn_telemetry() or {}
    events = engine.consume_ui_events()

    assert telemetry["selected_path"] == "REPRESENTATION_COMMAND"
    assert telemetry["selected_path"] != "GENERAL_CONVERSATION"
    assert telemetry["llm_calls"] == 0
    assert telemetry["representation_command_action"] == "grouping_requested"
    assert telemetry["representation_command_executed"] is False
    for false_success_phrase in ("Aqui tens", "Feito", "Já está"):
        assert false_success_phrase not in response
    assert engine._work_session.representation.style == style_before
    assert events == []  # no representation_changed event -- nothing actually changed
    assert ollama.calls == []


def test_grouping_j_mixed_add_and_grouping_still_adds_option_and_is_partial(tmp_path: Path) -> None:
    """The previously-fixed mixed-intent case must not regress now that
    grouping recognition is broader ('distinguir' is now a grouping
    trigger too): Madeira is still added, zero LLM calls, partial
    execution reported truthfully."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("vamos pensar em três opções: Suécia, Itália e Açores")
    engine.consume_ui_events()

    mixed_utterance = (
        "chega o card dos Açores para o outro lado, para distinguir destinos "
        "portugueses de internacionais. acrescenta também Madeira"
    )
    response = engine.respond(mixed_utterance)
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "REPRESENTATION_COMMAND"
    assert telemetry["llm_calls"] == 0
    assert "Madeira" in response
    titles = sorted(obj.title for obj in engine._work_session.objects.values())
    assert titles == ["Açores", "Itália", "Madeira", "Suécia"]
    assert telemetry["partial_execution"] is True
    assert telemetry["executed_intent_count"] == 1
    assert ollama.calls == []


# =============================================================================
# Phase 4 (Cognitive Workspace Blueprint v2): ATTENTION + PROACTIVE
# REPRESENTATION + SPACE PRESSURE
# =============================================================================


def test_phase4_e_auto_cards_proposal_for_three_alternatives(tmp_path: Path) -> None:
    """E: "vamos pensar em tres opcoes: X, Y e Z" -> Echo automatically
    proposes cards (section 7's reason_code proves it went through
    choose_representation, not a hardcoded default), no permission asked,
    zero LLM calls."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    response = engine.respond("vamos pensar em três opções: Suécia, Itália e Açores")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["representation_style"] == "cards"
    assert telemetry["representation_proposal_reason_code"] == "multiple_named_alternatives"
    assert telemetry["representation_command_action"] == "materialize_alternatives"
    for permission_phrase in ("posso ", "queres que", "devo "):
        assert permission_phrase not in response.lower()
    assert ollama.calls == []


def test_phase4_comparison_intent_proposes_side_by_side(tmp_path: Path) -> None:
    """Section 6/7: two alternatives introduced with a comparison verb get
    side_by_side instead of cards -- semantic SHAPE (object count +
    comparison intent), not a domain name."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("quero comparar Suécia e Itália")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["representation_style"] == "side_by_side"
    assert telemetry["representation_proposal_reason_code"] == "two_or_more_objects_being_compared"
    assert engine._work_session.representation.major_object_ids == ["option-suecia", "option-italia"]
    assert ollama.calls == []


def test_phase4_f_explicit_list_blocks_immediate_auto_reversion(tmp_path: Path) -> None:
    """F (section 9/10/22 scenario A): after an explicit correction, a
    LATER auto-representation trigger (a fresh materialize_alternatives
    call, which would otherwise default to cards) must respect the
    recorded constraint instead of reverting."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos pensar em três opções: Suécia, Itália e Açores")
    response2 = engine.respond("não quero cartões, põe em lista")
    telemetry2 = engine.get_last_turn_telemetry() or {}

    assert telemetry2["selected_path"] == "REPRESENTATION_COMMAND"
    assert engine._work_session.representation.style == "list"
    assert engine._work_session.constraints is not None
    assert engine._work_session.constraints.prefer_style == "list"

    response3 = engine.respond("vamos pensar em três opções: Porto, Lisboa e Faro")
    telemetry3 = engine.get_last_turn_telemetry() or {}

    assert telemetry3["representation_style"] == "list"
    assert telemetry3["representation_proposal_reason_code"] == "explicit_prefer_style_constraint"
    assert engine._work_session.representation.style == "list"
    assert ollama.calls == []


def test_phase4_g_low_pressure_no_action(tmp_path: Path) -> None:
    """G: space pressure LOW -> no permission question, no pending action."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    register_object(engine._work_session, SemanticObject(id="email-draft", title="Rascunho do email"))
    register_object(engine._work_session, SemanticObject(id="doc-a", title="Documento A"))
    set_bulk_attention(engine._work_session, {"email-draft": ACTIVE, "doc-a": ACTIVE})
    engine._work_session.capacity = SpacePressureState(pressure="low", visible_object_count=2, major_surface_count=0)

    response = engine.respond("prepara te, vamos planear umas ferias")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["task_started"] is True
    assert telemetry["space_pressure"] == "low"
    assert telemetry["space_pressure_action"] == "none"
    assert telemetry["space_pressure_permission_required"] is False
    assert telemetry.get("space_pressure_candidate_found") is None
    assert engine._pending_workspace_action is None
    assert "minimizar" not in response.lower()


def test_phase4_g2_medium_pressure_no_destructive_minimize(tmp_path: Path) -> None:
    """G2: MEDIUM pressure is conservative in this phase: it can leave
    low-priority context less prominent through attention policy, but it
    never asks for or performs minimization."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    register_object(engine._work_session, SemanticObject(id="email-draft", title="Rascunho do email"))
    register_object(engine._work_session, SemanticObject(id="doc-a", title="Documento A"))
    set_bulk_attention(engine._work_session, {"email-draft": ACTIVE, "doc-a": ACTIVE})
    engine._work_session.capacity = SpacePressureState(pressure="medium", visible_object_count=3, major_surface_count=1)

    response = engine.respond("prepara te, vamos planear umas ferias com base no Documento A")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["space_pressure"] == "medium"
    assert telemetry["space_pressure_action"] == "none"
    assert telemetry["space_pressure_permission_required"] is False
    assert engine._pending_workspace_action is None
    assert engine._work_session.attention.object_states["email-draft"] != "minimized"
    assert "minimizar" not in response.lower()
    assert ollama.calls == []


def test_phase4_h_high_pressure_selects_a_candidate(tmp_path: Path) -> None:
    """H (section 22 scenario B): space pressure HIGH -> a minimization
    candidate is found and Echo asks permission -- never silently
    minimizes."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    register_object(engine._work_session, SemanticObject(id="email-draft", title="Rascunho do email"))
    register_object(engine._work_session, SemanticObject(id="doc-a", title="Documento A"))
    register_object(engine._work_session, SemanticObject(id="doc-b", title="Documento B"))
    set_bulk_attention(engine._work_session, {"email-draft": ACTIVE, "doc-a": ACTIVE, "doc-b": ACTIVE})
    engine._work_session.capacity = SpacePressureState(pressure="high", visible_object_count=5, major_surface_count=0, overflow_risk=True)

    response = engine.respond("prepara te, vamos planear umas ferias com base no Documento A e no Documento B")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["task_started"] is True
    assert telemetry["space_pressure_candidate_found"] is True
    assert telemetry["pending_workspace_action_created"] is True
    assert engine._pending_workspace_action is not None
    assert engine._pending_workspace_action["action"] == "minimize"
    assert engine._pending_workspace_action["object_ids"] == ["email-draft"]
    assert engine._pending_workspace_action["reason"] == "space_pressure"
    assert engine._pending_workspace_action["ttl"] == 2
    assert "capacity_revision" in engine._pending_workspace_action
    assert telemetry["space_pressure"] == "high"
    assert telemetry["space_pressure_candidate_ids"][0] == "email-draft"
    assert "doc-a" not in telemetry["space_pressure_candidate_ids"]
    assert "doc-b" not in telemetry["space_pressure_candidate_ids"]
    assert telemetry["space_pressure_action"] == "propose_minimize"
    assert telemetry["space_pressure_permission_required"] is True
    assert telemetry["pending_workspace_action"] is True
    assert "?" in response
    assert "email" in response.lower() or "rascunho" in response.lower()
    # Referenced objects stay ACTIVE (relevant to the new task); the
    # unreferenced one demotes -- never straight to minimized/hidden.
    assert engine._work_session.attention.object_states["doc-a"] == "active"
    assert engine._work_session.attention.object_states["doc-b"] == "active"
    assert engine._work_session.attention.object_states["email-draft"] == "visible"
    assert ollama.calls == []


def test_phase4_i_focus_object_never_auto_minimized(tmp_path: Path) -> None:
    """I: whatever is currently FOCUS is excluded from the candidate pool
    up front, so even under high pressure it is never the thing Echo
    proposes to minimize."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    register_object(engine._work_session, SemanticObject(id="focus-obj", title="Foco Atual"))
    register_object(engine._work_session, SemanticObject(id="other", title="Outro"))
    set_bulk_attention(engine._work_session, {"focus-obj": FOCUS, "other": GLANCEABLE})
    engine._work_session.capacity = SpacePressureState(pressure="high", visible_object_count=5, major_surface_count=0)

    engine.respond("prepara te, vamos planear umas ferias")

    assert engine._pending_workspace_action is not None
    assert engine._pending_workspace_action["object_ids"] == ["other"]
    assert "focus-obj" not in engine._pending_workspace_action["object_ids"]


def test_phase4_j_do_not_minimize_constraint_respected(tmp_path: Path) -> None:
    """J (section 22 scenario C): an explicit do-not-minimize command sets
    a do_not_minimize constraint; a later high-pressure event must pick
    another candidate (or report it cannot make room), never silently
    minimize the protected object."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    register_object(engine._work_session, SemanticObject(id="email-draft", title="email"))
    set_bulk_attention(engine._work_session, {"email-draft": ACTIVE})

    engine.respond("não minimizes o email")
    telemetry = engine.get_last_turn_telemetry() or {}
    assert telemetry["selected_path"] == "ATTENTION_COMMAND"
    assert engine._work_session.constraints is not None
    assert "email-draft" in engine._work_session.constraints.do_not_minimize_ids

    engine._work_session.capacity = SpacePressureState(pressure="high", visible_object_count=5, major_surface_count=0)
    response = engine.respond("prepara te, vamos planear umas ferias")

    assert "email-draft" not in (engine._pending_workspace_action or {}).get("object_ids", [])
    if engine._pending_workspace_action is None:
        assert "nao encontro" in response.lower().replace("ã", "a").replace("ç", "c") or "nao consigo" in response.lower().replace("ã", "a")
    assert engine._work_session.attention.object_states["email-draft"] != "minimized"


def test_phase4_j2_keep_visible_constraint_respected_by_space_pressure(tmp_path: Path) -> None:
    """J2: keep_visible is a real protection, not just a representation
    preference. High pressure must not propose the kept object."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    register_object(engine._work_session, SemanticObject(id="email-draft", title="email"))
    register_object(engine._work_session, SemanticObject(id="old-note", title="Nota antiga"))
    set_bulk_attention(engine._work_session, {"email-draft": GLANCEABLE, "old-note": GLANCEABLE})

    engine.respond("mantém o email visível")
    assert engine._work_session.constraints is not None
    assert "email-draft" in engine._work_session.constraints.keep_visible_ids

    engine._work_session.capacity = SpacePressureState(pressure="high", visible_object_count=5, major_surface_count=0)
    engine.respond("prepara te, vamos planear umas ferias")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert engine._pending_workspace_action is not None
    assert engine._pending_workspace_action["object_ids"] == ["old-note"]
    assert "email-draft" in telemetry["space_pressure_protected_ids"]
    assert "email-draft" not in telemetry["space_pressure_candidate_ids"]


def test_phase4_k_ask_before_minimize_creates_pending_action(tmp_path: Path) -> None:
    """K: the permission question itself creates a PendingWorkspaceAction
    (already asserted structurally in test H); this test focuses on the
    exact shape of that pending state."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    register_object(engine._work_session, SemanticObject(id="email-draft", title="Rascunho do email"))
    set_bulk_attention(engine._work_session, {"email-draft": GLANCEABLE})
    engine._work_session.capacity = SpacePressureState(pressure="high", visible_object_count=5, major_surface_count=0)

    engine.respond("prepara te, vamos planear umas ferias")

    pending = engine._pending_workspace_action
    assert pending is not None
    assert pending["action"] == "minimize"
    assert pending["object_ids"] == ["email-draft"]
    assert pending["reason"] == "space_pressure"


def test_phase4_l_confirmation_executes_pending_action(tmp_path: Path) -> None:
    """L (section 22 scenario D, part 1): an affirmative confirmation
    executes the pending minimize action -- the object transitions to
    MINIMIZED, same semantic id, task not lost."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    register_object(engine._work_session, SemanticObject(id="email-draft", title="Rascunho do email"))
    set_bulk_attention(engine._work_session, {"email-draft": ACTIVE})
    engine._work_session.capacity = SpacePressureState(pressure="high", visible_object_count=5, major_surface_count=0)
    engine.respond("prepara te, vamos planear umas ferias")
    assert engine._pending_workspace_action is not None

    response = engine.respond("sim")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "ATTENTION_COMMAND"
    assert telemetry["pending_workspace_action_resolved"] == "executed"
    assert telemetry["pending_workspace_action_result"] == "executed"
    assert telemetry["pending_workspace_action_object_ids"] == ["email-draft"]
    assert telemetry["minimization_reason"] == "space_pressure"
    assert engine._pending_workspace_action is None
    assert engine._work_session.attention.object_states["email-draft"] == "minimized"
    assert "email" in response.lower() or "rascunho" in response.lower()
    assert engine._work_session.active_task is not None
    assert engine._work_session.active_task.task_type == "plan_trip"
    assert ollama.calls == []


def test_phase4_m_negative_confirmation_cancels_pending_action_without_losing_task(tmp_path: Path) -> None:
    """M (section 22 scenario D, part 2 / negative branch): declining
    preserves both the object (not minimized) and the active task."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    register_object(engine._work_session, SemanticObject(id="email-draft", title="Rascunho do email"))
    set_bulk_attention(engine._work_session, {"email-draft": ACTIVE})
    engine._work_session.capacity = SpacePressureState(pressure="high", visible_object_count=5, major_surface_count=0)
    engine.respond("prepara te, vamos planear umas ferias")
    assert engine._pending_workspace_action is not None

    response = engine.respond("não")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["pending_workspace_action_resolved"] == "cancelled"
    assert telemetry["pending_workspace_action_result"] == "cancelled"
    assert telemetry["pending_workspace_action_object_ids"] == ["email-draft"]
    assert engine._pending_workspace_action is None
    assert engine._work_session.attention.object_states["email-draft"] != "minimized"
    assert engine._work_session.active_task is not None
    assert engine._work_session.active_task.task_type == "plan_trip"
    assert response  # a real explanatory response, not a silent no-op


def test_phase4_m2_negative_confirmation_suppresses_repeated_space_pressure_prompt(tmp_path: Path) -> None:
    """M2: after the user rejects one space-pressure suggestion, Echo must
    not ask the same question again until the capacity signal materially
    changes."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    register_object(engine._work_session, SemanticObject(id="email-draft", title="Rascunho do email"))
    set_bulk_attention(engine._work_session, {"email-draft": ACTIVE})
    engine._work_session.capacity = SpacePressureState(pressure="high", visible_object_count=5, major_surface_count=0)

    engine.respond("prepara te, vamos planear umas ferias")
    assert engine._pending_workspace_action is not None
    engine.respond("não")

    engine._work_session.active_task = None
    response = engine.respond("vamos preparar outro plano de ferias")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert engine._pending_workspace_action is None
    assert telemetry["space_pressure_action"] == "no_safe_candidate"
    assert telemetry["space_pressure_candidate_found"] is False
    assert response.count("?") == 0
    assert ollama.calls == []


def test_phase4_m3_capacity_bridge_tracks_revision_and_clears_suppression(tmp_path: Path) -> None:
    """M3: frontend capacity reports remain semantic-only but still give
    the backend enough freshness/revision data to reset transient
    rejected-candidate suppression on material change."""
    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine._space_pressure_suppressed_candidate_ids.add("email-draft")

    engine.report_workspace_capacity(
        {
            "pressure": "high",
            "visible_object_count": 5,
            "major_surface_count": 2,
            "overflow_risk": True,
            "x": 100,
            "width": 300,
        }
    )

    capacity = engine._work_session.capacity
    assert capacity is not None
    assert capacity.pressure == "high"
    assert capacity.visible_object_count == 5
    assert capacity.major_surface_count == 2
    assert capacity.overflow_risk is True
    assert capacity.revision == 1
    assert capacity.updated_at > 0
    assert engine._space_pressure_suppressed_candidate_ids == set()


def test_phase4_n_minimized_object_restores_with_same_semantic_id(tmp_path: Path) -> None:
    """N (section 22 scenario D restore / 26 restore phrase): a minimized
    object comes back via natural language with the SAME semantic id and
    content -- no reconstruction."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    original = register_object(
        engine._work_session,
        SemanticObject(id="email-draft", title="Rascunho", payload={"body": "conteudo original do rascunho"}),
    )
    set_bulk_attention(engine._work_session, {"email-draft": MINIMIZED})

    response = engine.respond("mostra o rascunho outra vez")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "ATTENTION_COMMAND"
    assert engine._work_session.attention.object_states["email-draft"] == "active"
    assert engine._work_session.objects["email-draft"] is original  # same identity
    assert engine._work_session.objects["email-draft"].payload == {"body": "conteudo original do rascunho"}
    assert response
    assert ollama.calls == []


def test_phase4_n2_attention_command_wins_over_an_active_task(tmp_path: Path) -> None:
    """Regression found via the mandatory manual real-Ollama test (section
    30): _try_active_task_turn used to run BEFORE _try_attention_command,
    so an explicit attention command ("mostra o email outra vez") arriving
    while a real task (e.g. "vamos planear as ferias") was already active
    got silently swallowed by TASK_CONTINUATION and answered as if it were
    task small talk -- exactly the situation the acceptance scenarios
    always produce, since scenario D's restore command follows scenario
    B's task-starting turn. Fixed by moving _try_attention_command before
    _try_active_task_turn, mirroring _try_workspace_command's existing
    precedence (workspace-visual state beats task-content routing)."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    register_object(engine._work_session, SemanticObject(id="email-draft", title="email"))
    set_bulk_attention(engine._work_session, {"email-draft": ACTIVE})

    engine.respond("prepara te, vamos planear umas ferias")
    assert engine._work_session.active_task is not None  # a real task is now in progress

    engine.respond("minimiza o email")
    response = engine.respond("mostra o email outra vez")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "ATTENTION_COMMAND"
    assert telemetry["selected_path"] != "TASK_CONTINUATION"
    assert engine._work_session.attention.object_states["email-draft"] == "active"
    assert engine._work_session.active_task is not None  # the task itself is untouched
    assert response
    assert ollama.calls == []


def test_phase4_q_conversation_only_then_list_restore_uses_same_objects(tmp_path: Path) -> None:
    """Q (section 26, exact acceptance scenario E): cards with 3 options,
    then hide everything, then a later restore-with-a-different-style
    request -> SAME semantic objects, style=list, no recreation."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos pensar em três opções: Suécia, Itália e Açores")
    original_ids = set(engine._work_session.objects.keys())

    engine.respond("fala só comigo")
    assert engine._work_session.representation.style == "conversation_only"

    response = engine.respond("mostra-me outra vez as opções, mas em lista")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "REPRESENTATION_COMMAND"
    assert engine._work_session.representation.style == "list"
    assert set(engine._work_session.representation.visible_object_ids) == original_ids
    assert set(engine._work_session.objects.keys()) == original_ids  # no recreation
    assert response
    assert ollama.calls == []


def test_phase4_r_authoritative_axes_are_deterministic_no_llm(tmp_path: Path) -> None:
    """R: every clear ATTENTION_COMMAND/REPRESENTATION_COMMAND turn in this
    phase is llm_calls=0 -- both authoritative axes are pure, deterministic
    bridges, same guarantee Phase 3's representation bridge already gave."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    register_object(engine._work_session, SemanticObject(id="doc-a", title="Documento A"))
    set_bulk_attention(engine._work_session, {"doc-a": ACTIVE})

    for message in (
        "minimiza o Documento A",
        "não escondas o Documento A",
        "traz o Documento A para a frente",
        "mostra o Documento A outra vez",
        "põe em lista",
    ):
        engine.respond(message)
        telemetry = engine.get_last_turn_telemetry() or {}
        assert telemetry["llm_calls"] == 0

    assert ollama.calls == []


def test_phase4_s_legacy_general_conversation_remains_compatible(tmp_path: Path) -> None:
    """S: a completely unrelated ordinary conversational turn (no semantic
    objects, no representation/attention vocabulary) is untouched by any
    Phase 4 wiring -- still reaches the SAME pre-existing route (a plain
    greeting hits the existing social fast path, unrelated to and
    unaffected by anything added in this phase) with zero Phase 4 axes
    firing."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local", ollama_replies=["Olá! Tudo bem?"])

    response = engine.respond("olá, tudo bem?")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "SOCIAL_PATH"
    assert telemetry["attention_command_detected"] is False
    assert telemetry["representation_command_detected"] is False
    assert response


def test_phase4_frontend_event_shapes_carry_no_geometry(tmp_path: Path) -> None:
    """Structural companion to O (the frontend-side assertion lives in
    test_web_ui_prototype.py against the JS payload) -- the BACKEND side of
    attention_changed never queues a geometry field either."""
    import json

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    register_object(engine._work_session, SemanticObject(id="doc-a", title="Documento A"))
    set_bulk_attention(engine._work_session, {"doc-a": ACTIVE})

    engine.respond("minimiza o Documento A")
    payloads = [json.loads(event) for event in engine.consume_ui_events()]
    attention_events = [p for p in payloads if p.get("type") == "attention_changed"]

    assert attention_events
    for event in attention_events:
        for geometry_key in ("x", "y", "width", "height", "position", "left", "top"):
            assert geometry_key not in event
        assert "object_states" in event
        assert "focus_object_id" in event


def test_phase4_grouping_regression_still_truthful(tmp_path: Path) -> None:
    """T (companion): Phase 3C's grouping-intent truthfulness must not
    regress now that AttentionIntent/representation constraint handling
    were added to the same waterfall."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("vamos pensar em três opções: Suécia, Itália e Açores")
    style_before = engine._work_session.representation.style

    response = engine.respond("separa os cards dos destinos internacionais dos destinos portugueses")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "REPRESENTATION_COMMAND"
    assert telemetry["representation_command_action"] == "grouping_requested"
    assert telemetry["representation_command_executed"] is False
    assert engine._work_session.representation.style == style_before
    for false_success_phrase in ("Aqui tens", "Feito", "Já está"):
        assert false_success_phrase not in response
    assert ollama.calls == []


# =============================================================================
# Phase 4 manual-test fix: attention language + constraint preservation +
# multi-item addition. All 3 bugs below were exposed by the real Qt manual
# test; the fix moving _try_attention_command before _try_active_task_turn
# (from the PREVIOUS manual-test fix) is preserved untouched -- none of
# these failures are that precedence bug, all were reproduced AFTER it.
# =============================================================================


def test_phase4_fix_1_mantem_x_visivel_is_attention_command(tmp_path: Path) -> None:
    """1: "mantém Açores visível" must be claimed by ATTENTION_COMMAND with
    zero LLM calls and a real mutation (constraint set) -- not fall
    through to GENERAL_CONVERSATION with a fabricated success response and
    no attention_changed event."""
    import json

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    register_object(engine._work_session, SemanticObject(id="opt-acores", title="Açores"))
    set_bulk_attention(engine._work_session, {"opt-acores": ACTIVE})

    response = engine.respond("mantém Açores visível")
    telemetry = engine.get_last_turn_telemetry() or {}
    payloads = [json.loads(event) for event in engine.consume_ui_events()]

    assert telemetry["selected_path"] == "ATTENTION_COMMAND"
    assert telemetry["selected_path"] != "GENERAL_CONVERSATION"
    assert telemetry["llm_calls"] == 0
    assert engine._work_session.constraints is not None
    assert "opt-acores" in engine._work_session.constraints.keep_visible_ids
    assert any(p.get("type") == "attention_changed" for p in payloads)
    assert response
    assert ollama.calls == []


def test_phase4_fix_1_success_response_requires_actual_mutation(tmp_path: Path) -> None:
    """A success-shaped response for an attention command must never be
    returned without the corresponding WorkSession mutation actually
    having happened -- this asserts the mutation directly, not just the
    response text, for the exact real-input phrase from the bug report."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    register_object(engine._work_session, SemanticObject(id="opt-acores", title="Açores"))
    set_bulk_attention(engine._work_session, {"opt-acores": ACTIVE})

    constraint_before = engine._work_session.constraints
    engine.respond("mantém Açores visível")

    assert engine._work_session.constraints is not constraint_before
    assert "opt-acores" in engine._work_session.constraints.keep_visible_ids
    assert ollama.calls == []


def test_phase4_fix_2_do_not_minimize_preserves_focus(tmp_path: Path) -> None:
    """2: Suécia=FOCUS; "não minimizes a Suécia" must add a do_not_minimize
    constraint WITHOUT demoting FOCUS to ACTIVE or clearing
    focus_object_id -- the command only adds a constraint, it never asked
    for an attention-level change."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    register_object(engine._work_session, SemanticObject(id="opt-suecia", title="Suécia"))
    set_bulk_attention(engine._work_session, {"opt-suecia": FOCUS})
    assert engine._work_session.attention.focus_object_id == "opt-suecia"

    response = engine.respond("não minimizes a Suécia")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "ATTENTION_COMMAND"
    assert engine._work_session.attention.object_states["opt-suecia"] == "focus"
    assert engine._work_session.attention.focus_object_id == "opt-suecia"
    assert "opt-suecia" in engine._work_session.constraints.do_not_minimize_ids
    assert response
    assert ollama.calls == []


def test_phase4_fix_2_keep_visible_preserves_focus_where_already_focused(tmp_path: Path) -> None:
    """2 (keep_visible phrasing): "mantém X visível" on an object that is
    ALREADY FOCUS must leave it at FOCUS -- the same invariant as
    do_not_minimize, since both map to the same underlying constraint
    action."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    register_object(engine._work_session, SemanticObject(id="opt-suecia", title="Suécia"))
    set_bulk_attention(engine._work_session, {"opt-suecia": FOCUS})

    engine.respond("mantém a Suécia visível")

    assert engine._work_session.attention.object_states["opt-suecia"] == "focus"
    assert engine._work_session.attention.focus_object_id == "opt-suecia"
    assert ollama.calls == []


def test_phase4_fix_2_keep_visible_preserves_active_level(tmp_path: Path) -> None:
    """2: ACTIVE + "mantém X visível" -> stays ACTIVE (not forced to any
    other level)."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    register_object(engine._work_session, SemanticObject(id="opt-italia", title="Itália"))
    set_bulk_attention(engine._work_session, {"opt-italia": ACTIVE})

    engine.respond("mantém a Itália visível")

    assert engine._work_session.attention.object_states["opt-italia"] == "active"
    assert ollama.calls == []


def test_phase4_fix_2_keep_visible_restores_hidden_object(tmp_path: Path) -> None:
    """2 (the documented exception): if the object was currently HIDDEN or
    MINIMIZED, keep_visible/do_not_minimize must still promote it enough
    to actually satisfy "keep visible" -- the invariant only protects an
    ALREADY-visible level from being touched, it must not trap a hidden
    object in place."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    register_object(engine._work_session, SemanticObject(id="opt-japao", title="Japão"))
    set_bulk_attention(engine._work_session, {"opt-japao": HIDDEN})

    engine.respond("não escondas o Japão")

    assert engine._work_session.attention.object_states["opt-japao"] in ("visible", "active", "focus")
    assert engine._work_session.attention.object_states["opt-japao"] != "hidden"
    assert ollama.calls == []


def test_phase4_fix_3_multi_item_comma_e_enumeration(tmp_path: Path) -> None:
    """3: "acrescenta Madeira, Islândia, Noruega e Japão" must create FOUR
    SemanticObjects (stable generic option ids), preserve the current
    representation style, and cost zero LLM calls -- the real bug only
    created Madeira."""
    import json

    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("vamos pensar em três opções: Suécia, Itália e Açores")
    engine.respond("põe em lista")
    engine.consume_ui_events()

    response = engine.respond("acrescenta Madeira, Islândia, Noruega e Japão")
    telemetry = engine.get_last_turn_telemetry() or {}
    payloads = [json.loads(event) for event in engine.consume_ui_events()]
    upsert_events = [p for p in payloads if p.get("type") == "semantic_object_upserted"]

    titles = sorted(obj.title for obj in engine._work_session.objects.values())
    assert titles == ["Açores", "Islândia", "Itália", "Japão", "Madeira", "Noruega", "Suécia"]
    assert len(upsert_events) == 4
    assert engine._work_session.representation.style == "list"
    assert telemetry["llm_calls"] == 0
    assert response
    assert ollama.calls == []


def test_phase4_fix_4_pure_multi_add_is_not_partial_execution(tmp_path: Path) -> None:
    """4: a pure multi-item addition with no other sub-intent must report
    requested=1/executed=1/failed=0/partial_execution=false, and the
    response must never invent an unsupported "move cards"/"grouping"
    sub-intent the user never asked for."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("vamos pensar em três opções: Suécia, Itália e Açores")

    response = engine.respond("acrescenta Madeira, Islândia, Noruega e Japão")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["requested_intent_count"] == 1
    assert telemetry["executed_intent_count"] == 1
    assert telemetry["failed_intent_count"] == 0
    assert telemetry["partial_execution"] is False
    assert "mover cartões" not in response
    assert "grupos semânticos" not in response
    assert ollama.calls == []


def test_phase4_fix_4_genuine_mixed_add_and_grouping_still_partial(tmp_path: Path) -> None:
    """4 (must NOT regress): the previous real mixed request -- add PLUS a
    genuine spatial/grouping sub-intent -- must still report partial
    execution, since it really does contain more than the supported "add"
    sub-intent."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("vamos pensar em três opções: Suécia, Itália e Açores")

    mixed_utterance = (
        "chega o card dos Açores para o outro lado, para distinguir destinos "
        "portugueses de internacionais. acrescenta também Madeira"
    )
    response = engine.respond(mixed_utterance)
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "Madeira" in response
    assert telemetry["partial_execution"] is True
    assert telemetry["executed_intent_count"] == 1
    assert telemetry["requested_intent_count"] >= 2
    assert ollama.calls == []


def test_phase4_fix_5_accent_safe_target_resolution(tmp_path: Path) -> None:
    """5: accented titles (Suécia, Açores, Itália, Islândia) must resolve
    correctly via normalized comparison, using no LLM call, while the
    ORIGINAL display title (with its accent) is preserved untouched."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    for title in ("Suécia", "Açores", "Itália", "Islândia"):
        object_id = f"opt-{title.lower()}"
        register_object(engine._work_session, SemanticObject(id=object_id, title=title))
    set_bulk_attention(
        engine._work_session,
        {"opt-suécia": ACTIVE, "opt-açores": ACTIVE, "opt-itália": ACTIVE, "opt-islândia": ACTIVE},
    )

    for message, object_id in (
        ("minimiza a Suécia", "opt-suécia"),
        ("mantém Açores visível", "opt-açores"),
        ("não escondas a Itália", "opt-itália"),
        ("minimiza a Islândia", "opt-islândia"),
    ):
        engine.respond(message)
        telemetry = engine.get_last_turn_telemetry() or {}
        assert telemetry["selected_path"] == "ATTENTION_COMMAND", f"{message!r} did not resolve"
        assert telemetry["llm_calls"] == 0

    # Original accented display titles are never mutated by resolution.
    assert engine._work_session.objects["opt-suécia"].title == "Suécia"
    assert engine._work_session.objects["opt-açores"].title == "Açores"
    assert engine._work_session.objects["opt-itália"].title == "Itália"
    assert engine._work_session.objects["opt-islândia"].title == "Islândia"
    assert ollama.calls == []


def test_phase4_fix_6_exact_real_flow_regression(tmp_path: Path) -> None:
    """6: the exact required real-flow regression -- Suécia=FOCUS,
    Itália=ACTIVE, Açores=ACTIVE, representation=list, then A/B/C in
    sequence."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    register_object(engine._work_session, SemanticObject(id="option-suecia", kind="option", title="Suécia"))
    register_object(engine._work_session, SemanticObject(id="option-italia", kind="option", title="Itália"))
    register_object(engine._work_session, SemanticObject(id="option-acores", kind="option", title="Açores"))
    set_bulk_attention(
        engine._work_session,
        {"option-suecia": FOCUS, "option-italia": ACTIVE, "option-acores": ACTIVE},
    )
    transform_representation(engine._work_session, "list", source="auto")

    # A. "mantém Açores visível"
    response_a = engine.respond("mantém Açores visível")
    telemetry_a = engine.get_last_turn_telemetry() or {}
    assert telemetry_a["selected_path"] == "ATTENTION_COMMAND"
    assert telemetry_a["llm_calls"] == 0
    assert engine._work_session.attention.object_states["option-acores"] in ("active", "visible")
    assert "option-acores" in engine._work_session.constraints.keep_visible_ids
    assert engine._work_session.attention.focus_object_id == "option-suecia"
    assert response_a

    # B. "não minimizes a Suécia"
    response_b = engine.respond("não minimizes a Suécia")
    telemetry_b = engine.get_last_turn_telemetry() or {}
    assert telemetry_b["selected_path"] == "ATTENTION_COMMAND"
    assert engine._work_session.attention.object_states["option-suecia"] == "focus"
    assert "option-suecia" in engine._work_session.constraints.do_not_minimize_ids
    assert engine._work_session.attention.focus_object_id == "option-suecia"
    assert response_b

    # C. "acrescenta Madeira, Islândia, Noruega e Japão"
    response_c = engine.respond("acrescenta Madeira, Islândia, Noruega e Japão")
    telemetry_c = engine.get_last_turn_telemetry() or {}
    new_titles = {obj.title for obj in engine._work_session.objects.values()} - {"Suécia", "Itália", "Açores"}
    assert new_titles == {"Madeira", "Islândia", "Noruega", "Japão"}
    assert engine._work_session.representation.style == "list"
    assert "mover cartões" not in response_c
    assert "grupos semânticos" not in response_c
    assert telemetry_c["partial_execution"] is False
    assert telemetry_c["llm_calls"] == 0

    assert ollama.calls == []


def test_phase4_precedence_fix_still_holds(tmp_path: Path) -> None:
    """Guard rail explicitly requested alongside this fix: the PREVIOUS
    manual-test fix (attention commands win over an active task) must
    remain intact -- none of today's fixes touch that precedence."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    register_object(engine._work_session, SemanticObject(id="email-draft", title="email"))
    set_bulk_attention(engine._work_session, {"email-draft": ACTIVE})

    engine.respond("prepara te, vamos planear umas ferias")
    assert engine._work_session.active_task is not None

    engine.respond("minimiza o email")
    response = engine.respond("mostra o email outra vez")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "ATTENTION_COMMAND"
    assert telemetry["selected_path"] != "TASK_CONTINUATION"
    assert engine._work_session.attention.object_states["email-draft"] == "active"
    assert engine._work_session.active_task is not None
    assert response
    assert ollama.calls == []


def test_workspace_open_typo_resolves_existing_object_without_llm(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    register_object(engine._work_session, SemanticObject(id="option-samsung", kind="option", title="Samsung"))
    set_bulk_attention(engine._work_session, {"option-samsung": ACTIVE})

    response = engine.respond("abre samasung")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "ATTENTION_COMMAND"
    assert telemetry["llm_calls"] == 0
    assert telemetry["attention_command_executed"] is True
    assert engine._work_session.attention.object_states["option-samsung"] == FOCUS
    assert "Samsung" in response
    assert ollama.calls == []


def test_workspace_open_ambiguous_typo_asks_clarification_without_llm(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    register_object(engine._work_session, SemanticObject(id="option-samsung", kind="option", title="Samsung"))
    register_object(engine._work_session, SemanticObject(id="option-samsong", kind="option", title="Samsong"))
    set_bulk_attention(engine._work_session, {"option-samsung": ACTIVE, "option-samsong": ACTIVE})

    response = engine.respond("abre samsng")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "ATTENTION_COMMAND"
    assert telemetry["llm_calls"] == 0
    assert telemetry["attention_command_executed"] is False
    assert "Samsung" in response and "Samsong" in response
    assert engine._work_session.attention.object_states["option-samsung"] == ACTIVE
    assert engine._work_session.attention.object_states["option-samsong"] == ACTIVE
    assert ollama.calls == []


def test_generated_operational_success_claim_is_blocked_when_no_workspace_operation_executed(tmp_path: Path) -> None:
    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")

    response = engine._complete_turn(
        "abre samasung",
        "Samsung está a abrir agora.",
        "AGENT_DIRECT",
        selected_path="AGENT",
    )
    telemetry = engine.get_last_turn_telemetry() or {}

    assert response == "Não alterei nada no workspace. Preciso de identificar um objeto real antes de agir."
    assert telemetry == {}


# --- Phase 4 integration fix: workspace activity/object reference
# resolution (section 12, tests A-K). The real Qt manual test showed Echo
# treating each active task as an isolated conversational funnel instead of
# a coherent workspace containing multiple activities/objects -- these
# tests cover the three concrete failures reported plus the generic
# resolver/precedence/ownership behavior the fix introduces.

def _start_plan_trip_with_destination(engine) -> None:
    engine.respond("quero planear umas ferias, mostra-me 10 destinos em Portugal")


def test_workspace_reference_a_open_existing_activity_not_swallowed(tmp_path: Path) -> None:
    """A. active task does not swallow "abre <existing activity>" -- while
    a writing task is active, "abre o planeamento de férias" must not be
    appended to the writing task's notes/collected_facts."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    _start_plan_trip_with_destination(engine)
    engine.respond("vamos escrever também um mail")
    writing_ctx = engine._work_session.active_task
    assert writing_ctx is not None and writing_ctx.task_type == "writing"

    engine.respond("abre o planeamento de férias")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] != "TASK_CONTINUATION"
    assert "abre o planeamento de férias" not in writing_ctx.collected_facts
    assert telemetry["llm_calls"] == 0
    assert ollama.calls == []


def test_workspace_reference_b_as_ferias_resolves_unambiguously(tmp_path: Path) -> None:
    """B. "as férias" resolves to "Planeamento de férias" when unambiguous
    -- exercised directly against resolve_workspace_reference, the single
    generic resolver introduced for section 3."""
    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")
    _start_plan_trip_with_destination(engine)
    plan_trip_ctx = engine._work_session.active_task
    assert plan_trip_ctx is not None

    match = resolve_workspace_reference("as férias", engine._work_session)

    assert match is not None
    assert match.object_id == plan_trip_ctx.active_artifact_id
    assert match.task_id == plan_trip_ctx.id
    assert match.title == "Planeamento de férias"


def test_workspace_reference_c_nao_minimizes_as_ferias(tmp_path: Path) -> None:
    """C. "não minimizes as férias" adds do_not_minimize to the SAME
    existing task-artifact id -- resolve_mentioned_object_ids alone cannot
    do this (it only matches a literal title substring, and "as férias" is
    not a substring of "Planeamento de férias"; the relationship is
    reversed)."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    _start_plan_trip_with_destination(engine)
    plan_trip_ctx = engine._work_session.active_task
    assert plan_trip_ctx is not None
    artifact_id = plan_trip_ctx.active_artifact_id

    response = engine.respond("não minimizes as férias")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "ATTENTION_COMMAND"
    assert artifact_id in engine._work_session.constraints.do_not_minimize_ids
    assert telemetry["llm_calls"] == 0
    assert response
    assert ollama.calls == []


def test_workspace_reference_d_abre_planeamento_focuses_existing_task(tmp_path: Path) -> None:
    """D. "abre o planeamento de férias" focuses the EXISTING task artifact
    -- zero LLM calls, no new task created, active_task stays whatever it
    already was (attention-only operation, never a task switch)."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    _start_plan_trip_with_destination(engine)
    plan_trip_ctx = engine._work_session.active_task
    assert plan_trip_ctx is not None
    artifact_id = plan_trip_ctx.active_artifact_id
    engine.respond("vamos escrever também um mail")
    task_count_before = 1 + len(engine._work_session.paused_task_stack)

    response = engine.respond("abre o planeamento de férias")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "ATTENTION_COMMAND"
    assert telemetry["llm_calls"] == 0
    assert engine._work_session.attention.focus_object_id == artifact_id
    assert engine._work_session.attention.object_states[artifact_id] == FOCUS
    # No new task materialized just by opening an existing one.
    assert 1 + len(engine._work_session.paused_task_stack) == task_count_before
    assert response
    assert ollama.calls == []


def test_workspace_reference_e_writing_task_preserved_when_planning_focused(tmp_path: Path) -> None:
    """E. the current writing task is preserved (not lost/deleted, not
    recreated) when the planning activity is brought back into focus.
    Phase 4.5: task-level "abre X" restores the requested task identity;
    the previous writing task remains preserved in the stack."""
    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")
    _start_plan_trip_with_destination(engine)
    plan_ctx = engine._work_session.active_task
    assert plan_ctx is not None
    engine.respond("vamos escrever também um mail")
    writing_ctx_before = engine._work_session.active_task
    assert writing_ctx_before is not None and writing_ctx_before.task_type == "writing"
    writing_artifact_id = writing_ctx_before.active_artifact_id

    engine.respond("abre o planeamento de férias")

    writing_ctx_after = engine._work_session.active_task
    assert writing_ctx_after is not None
    assert writing_ctx_after.id == plan_ctx.id
    assert writing_ctx_after.task_type == "plan_trip"
    assert engine._work_session.attention.object_states[writing_artifact_id] == ACTIVE
    assert writing_artifact_id in engine._work_session.objects
    assert any(task.id == writing_ctx_before.id for task in engine._work_session.paused_task_stack)


def test_workspace_reference_f_multi_option_addition_reachable_while_task_active(tmp_path: Path) -> None:
    """F. the explicit multi-option addition mechanism remains reachable
    even while a DIFFERENT, unrelated task is active -- proving the fix is
    the general routing-precedence change (section 7), not a plan_trip-
    specific carve-out."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("vamos escrever também um mail")
    assert engine._work_session.active_task is not None
    assert engine._work_session.active_task.task_type == "writing"

    response = engine.respond("acrescenta Coimbra, Aveiro e Braga em cartões")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "REPRESENTATION_COMMAND"
    titles = {obj.title for obj in engine._work_session.objects.values() if obj.kind == "option"}
    assert titles == {"Coimbra", "Aveiro", "Braga"}
    assert engine._work_session.representation.style == "cards"
    assert telemetry["llm_calls"] == 0
    assert response
    assert ollama.calls == []


def test_workspace_reference_g_adiciona_cidades_creates_five_cards(tmp_path: Path) -> None:
    """G. "adiciona Porto, Lisboa, Coimbra, Aveiro e Guarda em cartões"
    creates exactly 5 SemanticObjects (clean titles -- the trailing style
    qualifier must not leak into the last one) and applies cards style,
    with zero LLM calls."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    _start_plan_trip_with_destination(engine)

    response = engine.respond("adiciona Porto, Lisboa, Coimbra, Aveiro e Guarda em cartões")
    telemetry = engine.get_last_turn_telemetry() or {}

    option_titles = {obj.title for obj in engine._work_session.objects.values() if obj.kind == "option"}
    assert option_titles == {"Porto", "Lisboa", "Coimbra", "Aveiro", "Guarda"}
    assert engine._work_session.representation.style == "cards"
    assert telemetry["llm_calls"] == 0
    assert "mover cartões" not in response
    assert ollama.calls == []


def test_workspace_reference_h_task_owned_objects_preserve_task_id(tmp_path: Path) -> None:
    """H. task-owned semantic objects preserve the task_id relation --
    generic ownership via the SAME SemanticObject.task_id field the task
    artifact itself already uses (section 10), not a second/parallel
    relation mechanism."""
    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")
    _start_plan_trip_with_destination(engine)
    plan_trip_ctx = engine._work_session.active_task
    assert plan_trip_ctx is not None

    engine.respond("adiciona Porto, Lisboa, Coimbra, Aveiro e Guarda em cartões")

    option_objects = [obj for obj in engine._work_session.objects.values() if obj.kind == "option"]
    assert len(option_objects) == 5
    assert all(obj.task_id == plan_trip_ctx.id for obj in option_objects)
    artifact = engine._work_session.objects[plan_trip_ctx.active_artifact_id]
    assert artifact.task_id == plan_trip_ctx.id


def test_workspace_reference_i_mostra_cidades_not_silently_noted(tmp_path: Path) -> None:
    """I. "mostra-me cidades em cartões" is not silently appended only to
    Notes -- it must reach a real representation-command handler instead
    of being swallowed by the active task's generic catch-all."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    _start_plan_trip_with_destination(engine)
    plan_trip_ctx = engine._work_session.active_task
    assert plan_trip_ctx is not None

    response = engine.respond("mostra-me cidades em cartões")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "REPRESENTATION_COMMAND"
    assert telemetry["selected_path"] != "TASK_CONTINUATION"
    assert "mostra-me cidades em cartões" not in plan_trip_ctx.collected_facts
    assert engine._work_session.representation.style == "cards"
    assert telemetry["llm_calls"] == 0
    assert response
    assert ollama.calls == []


def test_workspace_reference_j_legacy_compose_email_flow_compatible(tmp_path: Path) -> None:
    """J. legacy compose_email flow remains compatible -- the routing
    reorder must not disturb an ordinary compose_email turn sequence."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos escrever um email")
    r2 = engine.respond("é para a Maria, da equipa de vendas")
    t2 = engine.get_last_turn_telemetry() or {}
    assert t2["selected_path"] == "TASK_CONTINUATION"
    assert t2["task_known_fields"].get("recipient") == "Maria"

    r3 = engine.respond("é para agradecer a reunião de ontem")
    t3 = engine.get_last_turn_telemetry() or {}
    assert t3["selected_path"] == "TASK_CONTINUATION"
    assert t3["llm_calls"] == 0

    assert r2 and r3
    assert ollama.calls == []


def test_workspace_reference_k_attention_and_representation_precedence_kept(tmp_path: Path) -> None:
    """K. attention/representation commands keep their existing precedence
    over active-task continuation -- the reorder inserted new checks but
    must not have moved attention behind representation or behind the
    active task."""
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    register_object(engine._work_session, SemanticObject(id="email-draft", title="email"))
    set_bulk_attention(engine._work_session, {"email-draft": ACTIVE})

    engine.respond("prepara te, vamos planear umas ferias")
    assert engine._work_session.active_task is not None

    engine.respond("minimiza o email")
    response = engine.respond("mostra o email outra vez")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert telemetry["selected_path"] == "ATTENTION_COMMAND"
    assert telemetry["selected_path"] != "TASK_CONTINUATION"
    assert engine._work_session.attention.object_states["email-draft"] == "active"
    assert response
    assert ollama.calls == []


def test_p0_pending_email_does_not_swallow_new_activity_sequence(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    response1 = engine.respond("vamos escrever um mail")
    email_ctx = engine._work_session.active_task
    assert email_ctx is not None
    assert email_ctx.task_type == "compose_email"
    assert email_ctx.missing_information == ["recipient"]
    email_task_id = email_ctx.id
    email_artifact_id = email_ctx.active_artifact_id
    email_fields_before = (dict(email_ctx.entities), dict(email_ctx.parameters), list(email_ctx.collected_facts))
    assert "Para quem" in response1

    engine.respond("vamos também preparar umas férias")
    travel_ctx = engine._work_session.active_task
    telemetry2 = engine.get_last_turn_telemetry() or {}
    assert travel_ctx is not None
    assert travel_ctx.task_type == "plan_trip"
    assert telemetry2["selected_path"] in {"TASK_SWITCH", "TASK_START"}
    assert telemetry2["llm_calls"] == 0
    preserved_email = next(task for task in engine._work_session.paused_task_stack if task.id == email_task_id)
    assert preserved_email.active_artifact_id == email_artifact_id
    assert (dict(preserved_email.entities), dict(preserved_email.parameters), list(preserved_email.collected_facts)) == email_fields_before

    engine.respond("preciso da tua ajuda a planear essas férias")
    telemetry3 = engine.get_last_turn_telemetry() or {}
    assert engine._work_session.active_task is not None
    assert engine._work_session.active_task.id == travel_ctx.id
    assert telemetry3["selected_path"] == "TASK_CONTINUATION"
    preserved_email = next(task for task in engine._work_session.paused_task_stack if task.id == email_task_id)
    assert (dict(preserved_email.entities), dict(preserved_email.parameters), list(preserved_email.collected_facts)) == email_fields_before

    engine.respond("compara samsung com apple")
    telemetry4 = engine.get_last_turn_telemetry() or {}
    comparison_ctx = engine._work_session.active_task
    assert comparison_ctx is not None
    assert comparison_ctx.task_type == "generic_comparison"
    assert telemetry4["selected_path"] in {"REPRESENTATION_COMMAND", "TASK_SWITCH", "TASK_START"}
    assert telemetry4["llm_calls"] == 0
    preserved_email = next(task for task in engine._work_session.paused_task_stack if task.id == email_task_id)
    assert (dict(preserved_email.entities), dict(preserved_email.parameters), list(preserved_email.collected_facts)) == email_fields_before

    response5 = engine.respond("volta ao email")
    telemetry5 = engine.get_last_turn_telemetry() or {}
    resumed_email = engine._work_session.active_task
    assert resumed_email is not None
    assert resumed_email.id == email_task_id
    assert resumed_email.active_artifact_id == email_artifact_id
    assert resumed_email.missing_information == ["recipient"]
    assert telemetry5["selected_path"] == "TASK_RESUME"
    assert "Para quem" in response5

    engine.respond("é para o João")
    telemetry6 = engine.get_last_turn_telemetry() or {}
    assert engine._work_session.active_task is not None
    assert engine._work_session.active_task.id == email_task_id
    assert engine._work_session.active_task.entities.get("recipient") == "João"
    assert telemetry6["selected_path"] == "TASK_CONTINUATION"
    assert telemetry6["llm_calls"] == 0
    assert ollama.calls == []


def test_p0_pending_email_interrupted_by_planning_comparison_and_generic_activity(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("vamos escrever um mail")
    email_ctx = engine._work_session.active_task
    assert email_ctx is not None
    email_task_id = email_ctx.id
    email_fields_before = (dict(email_ctx.entities), dict(email_ctx.parameters), list(email_ctx.collected_facts))

    engine.respond("ajuda-me a planear umas férias")
    assert engine._work_session.active_task is not None
    assert engine._work_session.active_task.task_type == "plan_trip"
    preserved_email = next(task for task in engine._work_session.paused_task_stack if task.id == email_task_id)
    assert (dict(preserved_email.entities), dict(preserved_email.parameters), list(preserved_email.collected_facts)) == email_fields_before

    engine.respond("vamos comparar hotéis A e B")
    assert engine._work_session.active_task is not None
    assert engine._work_session.active_task.task_type == "generic_comparison"
    preserved_email = next(task for task in engine._work_session.paused_task_stack if task.id == email_task_id)
    assert (dict(preserved_email.entities), dict(preserved_email.parameters), list(preserved_email.collected_facts)) == email_fields_before

    engine.respond("prepara uma apresentação")
    assert engine._work_session.active_task is not None
    assert engine._work_session.active_task.task_type == "generic_activity"
    preserved_email = next(task for task in engine._work_session.paused_task_stack if task.id == email_task_id)
    assert (dict(preserved_email.entities), dict(preserved_email.parameters), list(preserved_email.collected_facts)) == email_fields_before
    assert ollama.calls == []


def test_p0_pending_email_interrupted_by_workspace_attention_and_representation_commands(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.respond("vamos escrever um mail")
    email_ctx = engine._work_session.active_task
    assert email_ctx is not None
    email_task_id = email_ctx.id
    email_artifact_id = email_ctx.active_artifact_id
    email_fields_before = (dict(email_ctx.entities), dict(email_ctx.parameters), list(email_ctx.collected_facts))

    engine.respond("minimiza o email")
    telemetry2 = engine.get_last_turn_telemetry() or {}
    assert telemetry2["selected_path"] == "ATTENTION_COMMAND"
    assert engine._work_session.attention.object_states[email_artifact_id] == MINIMIZED

    engine.respond("mostra o email")
    telemetry3 = engine.get_last_turn_telemetry() or {}
    assert telemetry3["selected_path"] == "ATTENTION_COMMAND"
    assert engine._work_session.attention.object_states[email_artifact_id] == FOCUS

    engine.respond("mostra-me cidades em cartões")
    telemetry4 = engine.get_last_turn_telemetry() or {}
    assert telemetry4["selected_path"] == "REPRESENTATION_COMMAND"
    preserved_email = engine._work_session.active_task
    assert preserved_email is not None and preserved_email.id == email_task_id
    assert (dict(preserved_email.entities), dict(preserved_email.parameters), list(preserved_email.collected_facts)) == email_fields_before
    assert ollama.calls == []


def test_p0_pending_email_still_accepts_legitimate_short_field_answers(tmp_path: Path) -> None:
    examples = [
        ("João", "João"),
        ("para o João", "João"),
        ("é para a Maria", "Maria"),
        ("para a minha chefe", "", "manager"),
    ]
    for item in examples:
        utterance = item[0]
        expected_name = item[1]
        expected_role = item[2] if len(item) > 2 else ""
        engine, _llm, ollama, _anthropic = make_engine(tmp_path / utterance.replace(" ", "_"), mode="local")
        engine.respond("vamos escrever um mail")
        engine.respond(utterance)
        telemetry = engine.get_last_turn_telemetry() or {}
        ctx = engine._work_session.active_task
        assert ctx is not None and ctx.task_type == "compose_email"
        if expected_name:
            assert ctx.entities.get("recipient") == expected_name
        if expected_role:
            assert ctx.entities.get("recipientRole") == expected_role
        assert telemetry["selected_path"] == "TASK_CONTINUATION"
        assert telemetry["llm_calls"] == 0
        assert ollama.calls == []

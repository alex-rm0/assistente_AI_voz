from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from assistant.conversation import AssistantEngine
from assistant.memory import ConversationMemory
from assistant.model_provider import ModelResponse
from assistant.model_router import AutomaticRoutingConfig, ModelRouter, ModelRoutingConfig, ModelUsageBudget, RoutedLLM
from assistant.model_runtime import ModelRuntimeBridge, UserSettingsStore
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
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

    def chat(self, messages, *, model=None, response_format=None, tools=None, temperature=None):
        self.calls.append({"messages": messages, "model": model, "response_format": response_format})
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
    assert second_telemetry["model_routing_provider"] == "local"
    assert second_telemetry["model_routing_model"] == "NONE"
    assert second_telemetry["model_routing_reason_code"] == "social_fast_path"
    assert second_telemetry["model_routing_paid_call"] is False
    assert second_telemetry["estimated_cost_usd"] == 0.0
    assert len(anthropic.calls) == 1

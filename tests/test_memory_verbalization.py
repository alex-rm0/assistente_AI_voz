from __future__ import annotations

from pathlib import Path

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.memory_recall import (
    convert_first_person_to_second_person,
    extract_academic_event_candidate,
    extract_task_candidate,
    render_task_list_answer,
)
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.tool_registry import ToolRegistry


class RaisingLLM:
    """Proves a path never calls the LLM (compose() would swallow a raised
    exception, so we count calls explicitly instead of relying on the raise)."""

    def __init__(self) -> None:
        self.call_count = 0

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        self.call_count += 1
        raise AssertionError("o LLM nao devia ter sido chamado")

    def choose_tool(self, *args, **kwargs):
        raise AssertionError("choose_tool nao devia ter sido chamado")

    def embed(self, text: str):
        return None


def make_engine(tmp_path: Path, llm=None) -> AssistantEngine:
    llm = llm or RaisingLLM()
    data = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return AssistantEngine(
        llm=llm,
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=LongTermMemory(data, embedder=llm),
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
    )


# --- Teste A: tarefa simples -------------------------------------------------


def test_teste_a_simple_task_is_extracted_and_answered(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    engine._maybe_extract_structured_memory("Tenho de mandar uma mensagem ao Pedro.")

    answer = engine._try_memory_recall_question("Que tarefas tenho pendentes?")

    assert answer == "Tens de mandar uma mensagem ao Pedro."


def test_task_memory_shape() -> None:
    fields = extract_task_candidate("Tenho de mandar uma mensagem ao Pedro.")
    assert fields["action"] == "mandar uma mensagem ao Pedro"
    assert fields["target"] == "Pedro"
    assert fields["status"] == "pending"
    assert fields["raw_user_text"] == "Tenho de mandar uma mensagem ao Pedro."
    assert "reminder_requested" not in fields


# --- Teste B: não copiar a primeira pessoa ----------------------------------


def test_teste_b_never_echoes_first_person_task_list(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    engine._maybe_extract_structured_memory("Tenho de mandar uma mensagem ao Pedro.")

    answer = engine._try_memory_recall_question("Que tarefas tenho pendentes?")

    assert answer != "Tens as seguintes tarefas: tenho de mandar uma mensagem ao Pedro."
    assert "tenho de" not in answer.lower()


# --- Teste C: pedido explícito de lembrete ----------------------------------


def test_teste_c_explicit_reminder_request_is_acknowledged(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    engine._maybe_extract_structured_memory("Lembra-me de mandar uma mensagem ao Pedro.")

    answer = engine._try_memory_recall_question("Que tarefas tenho pendentes?")

    assert answer == "Pediste-me para te lembrar de mandar uma mensagem ao Pedro."


# --- Teste D: não inventar pedido de lembrete -------------------------------


def test_teste_d_plain_task_never_claims_a_reminder_was_requested(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    engine._maybe_extract_structured_memory("Tenho de mandar uma mensagem ao Pedro.")

    answer = engine._try_memory_recall_question("Que tarefas tenho pendentes?")

    assert "pediste-me" not in answer.lower()


def test_fact_about_a_person_is_not_captured_as_a_task(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    engine._maybe_extract_structured_memory("O Pedro trabalha comigo.")

    facts = engine.long_term_memory.find_structured_facts(fact_type="task")
    assert facts == []


# --- Teste E: várias tarefas -------------------------------------------------


def test_teste_e_three_tasks_are_listed_naturally() -> None:
    from assistant.long_term_memory import StructuredFact

    tasks = [
        StructuredFact(id=1, fact_type="task", action="mandar uma mensagem ao Pedro", status="pending"),
        StructuredFact(id=2, fact_type="task", action="terminar o relatório", status="pending"),
        StructuredFact(id=3, fact_type="task", action="marcar a consulta", status="pending"),
    ]

    answer = render_task_list_answer(tasks)

    assert answer == "Tens três tarefas pendentes: mandar uma mensagem ao Pedro, terminar o relatório e marcar a consulta."


# --- Teste F: disciplina do exame (novos padrões) ---------------------------


def test_teste_f_discipline_extracted_from_e_de_pattern(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    engine._maybe_extract_structured_memory("Vou ter um exame muito importante para a semana.")
    engine._maybe_extract_structured_memory("É de Estratégias Algorítmicas, da faculdade.")

    facts = engine.long_term_memory.find_structured_facts(fact_type="academic_event")
    assert len(facts) == 1
    assert facts[0].discipline == "Estratégias Algorítmicas"
    assert facts[0].date_reference == "para a semana"
    assert facts[0].context == "faculdade"


# --- Teste G: reinício e chat limpo (disciplina) ----------------------------


def test_teste_g_discipline_only_question_after_restart(tmp_path: Path) -> None:
    engine1 = make_engine(tmp_path)
    engine1._maybe_extract_structured_memory("Vou ter um exame muito importante para a semana.")
    engine1._maybe_extract_structured_memory("É de Estratégias Algorítmicas, da faculdade.")

    engine2 = make_engine(tmp_path)
    assert engine2.memory.load() == []
    answer = engine2._try_memory_recall_question("Qual era a disciplina do exame?")

    assert answer == "Era Estratégias Algorítmicas."


# --- Teste H: data do exame --------------------------------------------------


def test_teste_h_date_reference_is_quoted_not_invented(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    engine._maybe_extract_structured_memory("Vou ter um exame muito importante para a semana.")
    engine._maybe_extract_structured_memory("É de Estratégias Algorítmicas, da faculdade.")

    answer = engine._try_memory_recall_question("Quando era o exame?")

    assert answer == "Tinhas dito que era para a semana."


# --- Teste I: resposta curta associada ao evento incompleto ----------------


def test_teste_i_short_reply_updates_the_incomplete_event(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    engine._maybe_extract_structured_memory("Vou ter um exame muito importante para a semana.")
    engine._maybe_extract_structured_memory("É de Estratégias Algorítmicas.")

    facts = engine.long_term_memory.find_structured_facts(fact_type="academic_event")
    assert len(facts) == 1
    assert facts[0].discipline == "Estratégias Algorítmicas"
    assert facts[0].date_reference == "para a semana"


# --- Teste J: não depende da formulação do Echo -----------------------------


def test_teste_j_association_does_not_depend_on_echos_previous_wording(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    # No "Que exame é?" clarifying question at all — a direct pattern match
    # ("é de X") must work independently of what the assistant last said.
    engine.memory.append_pair(
        "Vou ter um exame muito importante para a semana.",
        "Isso parece stressante. Como te sentes?",
    )
    engine._maybe_extract_structured_memory("Vou ter um exame muito importante para a semana.")
    engine._maybe_extract_structured_memory("É de Estratégias Algorítmicas.")

    facts = engine.long_term_memory.find_structured_facts(fact_type="academic_event")
    assert len(facts) == 1
    assert facts[0].discipline == "Estratégias Algorítmicas"


# --- Teste K: correção da pessoa gramatical ---------------------------------


def test_teste_k_first_to_second_person() -> None:
    assert convert_first_person_to_second_person("Preciso de enviar o relatório.") == "Precisas de enviar o relatório."
    assert convert_first_person_to_second_person("Tenho de estudar.") == "Tens de estudar."


# --- Teste L: não inventar entidades -----------------------------------------


def test_teste_l_no_target_is_not_invented(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    engine._maybe_extract_structured_memory("Tenho de estudar.")

    answer = engine._try_memory_recall_question("Que tarefas tenho pendentes?")

    assert answer == "Tens de estudar."
    assert "Pedro" not in answer


# --- Teste M: telemetria determinística -------------------------------------


def test_teste_m_deterministic_response_makes_zero_llm_calls(tmp_path: Path) -> None:
    llm = RaisingLLM()
    engine = make_engine(tmp_path, llm)
    engine._maybe_extract_structured_memory("Tenho de mandar uma mensagem ao Pedro.")

    answer = engine._complete_turn(
        "Que tarefas tenho pendentes?",
        engine._try_memory_recall_question("Que tarefas tenho pendentes?"),
        "MEMORY_RECALL_DETERMINISTIC",
        selected_path="MEMORY_RECALL",
    )

    assert answer == "Tens de mandar uma mensagem ao Pedro."
    assert llm.call_count == 0


# --- Teste N: raw_user_text é evidência, não resposta -----------------------


def test_teste_n_raw_user_text_is_stored_but_never_the_answer(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    engine._maybe_extract_structured_memory("tenho que mandar msg ao pedro")

    facts = engine.long_term_memory.find_structured_facts(fact_type="task")
    assert facts[0].raw_user_text == "tenho que mandar msg ao pedro"

    answer = engine._try_memory_recall_question("Que tarefas tenho pendentes?")
    assert answer != "tenho que mandar msg ao pedro"
    assert "tenho que mandar msg ao pedro" not in answer


# --- Write-side telemetry ----------------------------------------------------


def test_turn_trace_is_disabled_without_debug_flag(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    engine._begin_turn_trace("Vou ter um exame muito importante para a semana.")
    assert engine._turn_trace is None


def test_write_telemetry_reports_created_then_merged(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    engine.debug_ollama_payload = True
    engine._begin_turn_trace("Vou ter um exame muito importante para a semana.")
    engine._maybe_extract_structured_memory("Vou ter um exame muito importante para a semana.")
    assert engine._turn_trace["memory_candidate_detected"] is True
    assert engine._turn_trace["memory_candidate_type"] == "academic_event"
    assert engine._turn_trace["memory_write_action"] == "created"

    engine._begin_turn_trace("É de Estratégias Algorítmicas.")
    engine._maybe_extract_structured_memory("É de Estratégias Algorítmicas.")
    assert engine._turn_trace["memory_write_action"] == "merged"

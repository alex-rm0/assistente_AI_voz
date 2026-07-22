from __future__ import annotations

from pathlib import Path

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.memory_recall import (
    build_memory_retrieval,
    detect_unsupported_memory_claim,
    extract_academic_event_candidate,
)
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.tool_registry import ToolRegistry


class FakeLLM:
    def __init__(self, reply: str = "resposta") -> None:
        self.reply = reply
        self.chat_calls = 0

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        self.chat_calls += 1
        return self.reply

    def embed(self, text: str):
        return None


class RaisingLLM:
    """A LLM that must never be called — proves a path is fully deterministic.

    Tracks call_count explicitly (rather than only raising) because
    ResponseComposer.compose() catches exceptions from llm.chat() and falls
    back gracefully, which would otherwise hide a call that should never
    have happened.
    """

    def __init__(self) -> None:
        self.call_count = 0

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        self.call_count += 1
        raise AssertionError("O LLM nao devia ter sido chamado sem evidencia de memoria.")

    def choose_tool(self, *args, **kwargs):
        raise AssertionError("choose_tool nao devia ter sido chamado.")

    def embed(self, text: str):
        return None


def make_engine(tmp_path: Path, llm, data_dir_name: str = "data") -> AssistantEngine:
    data = tmp_path / data_dir_name
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


# --- Teste A: memória existente ---------------------------------------------


def test_teste_a_existing_memory_answers_directly(tmp_path: Path) -> None:
    # Deterministic template rendering: the answer never touches the LLM.
    llm = RaisingLLM()
    engine = make_engine(tmp_path, llm)
    engine.long_term_memory.remember_structured_fact(
        "academic_event", {"event": "exame", "discipline": "Estratégias Algorítmicas"}, confidence=0.9
    )

    answer = engine._try_memory_recall_question("Qual era a disciplina?")

    assert answer == "Era Estratégias Algorítmicas."
    assert llm.call_count == 0


# --- Teste B: memória inexistente -------------------------------------------


def test_teste_b_no_memory_gives_honest_answer(tmp_path: Path) -> None:
    llm = RaisingLLM()
    engine = make_engine(tmp_path, llm)

    answer = engine._try_memory_recall_question("Qual era a disciplina do exame?")

    assert answer is not None
    assert "não" in answer.lower() or "nao" in answer.lower()
    assert llm.call_count == 0


# --- Teste C: proibir invenção ----------------------------------------------


def test_teste_c_no_evidence_never_reaches_the_llm(tmp_path: Path) -> None:
    llm = RaisingLLM()
    engine = make_engine(tmp_path, llm)

    answer = engine._try_memory_recall_question("Qual era a disciplina do exame?")

    assert answer != "Era Economia."
    assert "economia" not in answer.lower()
    assert llm.call_count == 0, "sem evidencia, o LLM nunca deve ser chamado"


# --- Teste D: lembrança vaga -------------------------------------------------


def test_teste_d_vague_memory_admits_missing_discipline(tmp_path: Path) -> None:
    llm = RaisingLLM()
    engine = make_engine(tmp_path, llm)
    engine.long_term_memory.remember_structured_fact("academic_event", {"event": "exame"}, confidence=0.9)

    answer = engine._try_memory_recall_question("Lembras-te do exame?")

    assert "disciplina" in answer.lower()
    assert "nao tenho" in _strip_accents(answer.lower()) or "não tenho" in answer.lower()
    assert llm.call_count == 0


# --- Teste E: atributo em falta não é inventado -----------------------------


def test_teste_e_missing_attribute_is_not_invented(tmp_path: Path) -> None:
    llm = RaisingLLM()
    engine = make_engine(tmp_path, llm)
    engine.long_term_memory.remember_structured_fact(
        "academic_event", {"event": "exame", "discipline": "Estratégias Algorítmicas"}, confidence=0.9
    )

    answer = engine._try_memory_recall_question("Quando era o exame?")

    assert answer is not None
    assert "data" in answer.lower()
    assert llm.call_count == 0


# --- Teste F: duas memórias possíveis -> desambiguação ----------------------


def test_teste_f_ambiguous_memories_ask_for_disambiguation(tmp_path: Path) -> None:
    llm = RaisingLLM()
    engine = make_engine(tmp_path, llm)
    engine.long_term_memory.remember_structured_fact(
        "academic_event", {"event": "exame", "discipline": "Computação Gráfica"}, confidence=0.9
    )
    engine.long_term_memory.remember_structured_fact(
        "academic_event", {"event": "exame", "discipline": "Estratégias Algorítmicas"}, confidence=0.9
    )

    answer = engine._try_memory_recall_question("Qual era o exame?")

    assert "Computação Gráfica" in answer
    assert "Estratégias Algorítmicas" in answer
    assert "?" in answer
    assert llm.call_count == 0, "ambiguidade deve ser resolvida sem chamar o LLM"


# --- Teste G: resposta anterior do Echo nunca vira facto --------------------


def test_teste_g_echo_own_reply_is_never_extracted_as_fact(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, FakeLLM("ok"))
    engine.memory.append_pair("Qual foi a nota?", "Era Economia.")

    engine._maybe_extract_structured_memory("Não percebi bem.")

    facts = engine.long_term_memory.find_structured_facts()
    assert facts == []


def test_extraction_only_ever_receives_user_text() -> None:
    # A direct, code-level guarantee: the extractor itself has no notion of
    # "assistant said" — it is the caller's job (and the point above) to
    # never feed it anything but the user's own message.
    candidate = extract_academic_event_candidate("Era Economia.")
    assert candidate.get("discipline") != "Economia"


# --- Teste H: reinício com chat limpo ---------------------------------------


def test_teste_h_survives_restart_with_a_clean_chat(tmp_path: Path) -> None:
    llm1 = RaisingLLM()
    engine1 = make_engine(tmp_path, llm1)
    engine1._maybe_extract_structured_memory("Tenho um exame de Estratégias Algorítmicas para a semana.")
    engine1._maybe_extract_structured_memory("É da minha licenciatura em Engenharia Informática.")

    # Simulate closing the app and starting a brand new conversation: a fresh
    # engine, fresh (empty) ConversationMemory, same on-disk data directory.
    llm2 = RaisingLLM()
    engine2 = make_engine(tmp_path, llm2)
    assert engine2.memory.load() == []

    answer = engine2._try_memory_recall_question("Lembras-te do exame de que te falei?")

    assert "Estratégias Algorítmicas" in answer
    assert "Engenharia Informática" in answer
    assert llm1.call_count == 0
    assert llm2.call_count == 0


# --- Teste I: deduplicação ---------------------------------------------------


def test_teste_i_repeated_mentions_update_one_record(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, FakeLLM("ok"))

    engine._maybe_extract_structured_memory("Tenho exame de Estratégias Algorítmicas.")
    engine._maybe_extract_structured_memory("O exame de Estratégias Algorítmicas é para a semana.")
    engine._maybe_extract_structured_memory("Já te disse que tenho exame de Estratégias Algorítmicas.")

    facts = engine.long_term_memory.find_structured_facts(fact_type="academic_event")
    assert len(facts) == 1


def test_vague_exam_emotion_is_not_saved_as_persistent_memory(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, FakeLLM("ok"))

    for message in (
        "Estou nervoso para um exame.",
        "Tenho um exame em breve.",
        "Amanha vai ser um dia complicado.",
    ):
        engine._maybe_extract_structured_memory(message)

    assert engine.long_term_memory.find_structured_facts(fact_type="academic_event") == []


def test_concrete_exam_statements_are_saved(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, FakeLLM("ok"))

    engine._maybe_extract_structured_memory("Tenho exame de Computacao Grafica no dia 28.")
    engine._try_memory_write_command("Lembra-te de que o meu exame de Matematica e sexta-feira.")
    engine._maybe_extract_structured_memory("Chumbei ao exame de Estruturas de Dados.")

    disciplines = {
        _strip_accents(fact.discipline.lower())
        for fact in engine.long_term_memory.find_structured_facts(fact_type="academic_event")
    }

    assert "computacao grafica" in disciplines
    assert "matematica" in disciplines
    assert "estruturas de dados" in disciplines


def test_memory_recall_records_grounding_metadata(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, RaisingLLM())
    engine.debug_ollama_payload = True
    fact = engine.long_term_memory.remember_structured_fact(
        "academic_event",
        {"event": "exame", "discipline": "Estrategias Algoritmicas", "date_reference": "para a semana"},
        confidence=0.9,
    )

    response = engine.respond("Confirma na memoria qual e o exame que vou ter para a semana.")
    telemetry = engine.get_last_turn_telemetry()

    assert "Estrategias Algoritmicas" in _strip_accents(response)
    assert telemetry["memory_recall_detected"] is True
    assert str(fact.id) in telemetry["selected_memory_ids"]
    assert telemetry["response_grounded"] is True
    assert telemetry["grounding_sources"]


# --- Teste J: atualização de estado -----------------------------------------


def test_teste_j_status_transitions_on_the_same_record(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, FakeLLM("ok"))
    engine._maybe_extract_structured_memory("Tenho um exame de Estratégias Algorítmicas para a semana.")

    facts = engine.long_term_memory.find_structured_facts(fact_type="academic_event")
    assert facts[0].status == "upcoming"

    engine._maybe_extract_structured_memory("Chumbei ao exame de Estratégias Algorítmicas.")

    facts = engine.long_term_memory.find_structured_facts(fact_type="academic_event")
    assert len(facts) == 1
    assert facts[0].status == "failed"


# --- Teste K: não dizer "lembro-me" sem fonte -------------------------------


def test_teste_k_memory_claim_without_source_is_flagged() -> None:
    reason = detect_unsupported_memory_claim("Sim, lembro-me.", grounding_sources=[])
    assert reason


def test_grounded_claim_is_not_flagged() -> None:
    reason = detect_unsupported_memory_claim("Era Estratégias Algorítmicas.", grounding_sources=["PERSISTENT_MEMORY:1"])
    assert reason == ""


# --- Teste L: histórico atual válido ----------------------------------------


def test_teste_l_current_history_can_ground_without_persistent_memory() -> None:
    retrieval = build_memory_retrieval(
        requested_attributes={"discipline"},
        history_text="tenho um exame de estrategias algoritmicas para a semana",
        structured_facts=[],
    )

    assert retrieval.grounded
    assert retrieval.sources == ["CURRENT_HISTORY"]


def _strip_accents(text: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))

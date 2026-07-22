from __future__ import annotations

from pathlib import Path

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.memory_recall import (
    canonicalize_action,
    canonicalize_designation,
    extract_academic_event_candidate,
    is_memory_recall_followup,
    is_memory_recall_question,
    is_memory_write_command,
    parse_memory_write_command,
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


def _strip_accents(text: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


# --- Routing: Testes A-E -----------------------------------------------------


def test_teste_a_all_recall_phrasings_are_detected() -> None:
    phrasings = [
        "Ainda te lembras do exame de que te tinha falado?",
        "Lembras-te do exame?",
        "Então, lembras-te?",
        "Já te tinha falado disso, lembras-te?",
        "O que é que te disse sobre o exame?",
        "O que te disse sobre o exame?",
        "Te recordas do exame?",
        "Recordas-te do exame?",
        "Falamos disso, lembras-te?",
        "Tinhas guardado essa informação sobre o exame?",
    ]
    for phrasing in phrasings:
        normalized = _strip_accents(phrasing.lower())
        assert is_memory_recall_question(normalized), f"deveria detetar: {phrasing}"


def test_teste_b_store_command_is_not_a_recall_question() -> None:
    normalized = _strip_accents("lembra-te que prefiro mapas mentais".lower())
    assert not is_memory_recall_question(normalized)


def test_teste_c_unrelated_question_is_not_a_recall_question() -> None:
    normalized = _strip_accents("qual e a tua lingua base?".lower())
    assert not is_memory_recall_question(normalized)


def test_teste_d_pure_recall_question_is_not_new_evidence() -> None:
    candidate = extract_academic_event_candidate("Ainda te lembras do exame?")
    assert candidate == {}


def test_teste_e_mixed_message_extracts_only_the_new_evidence() -> None:
    candidate = extract_academic_event_candidate("Lembras-te do exame? Afinal é na sexta-feira.")
    assert candidate.get("date_reference") == "na sexta-feira"


def test_routing_bug_example_reaches_memory_recall_deterministically(tmp_path: Path) -> None:
    llm = RaisingLLM()
    engine = make_engine(tmp_path, llm)
    engine._maybe_extract_structured_memory("Tenho um exame de Estratégias Algorítmicas para a semana.")

    answer = engine.respond("Ainda te lembras do exame de que te tinha falado?")

    assert "Estratégias Algorítmicas" in answer
    assert llm.call_count == 0


# --- Writing: Testes F-H -----------------------------------------------------


def test_teste_f_register_command_confirms_deterministically(tmp_path: Path) -> None:
    llm = RaisingLLM()
    engine = make_engine(tmp_path, llm)

    answer = engine.respond("Regista que a disciplina é estratégias algoritmicas.")

    assert answer == "Registado. A disciplina é Estratégias Algorítmicas."
    assert llm.call_count == 0

    facts = engine.long_term_memory.find_structured_facts(fact_type="academic_event")
    assert facts[0].discipline == "Estratégias Algorítmicas"


def test_teste_g_update_command_confirms_deterministically(tmp_path: Path) -> None:
    llm = RaisingLLM()
    engine = make_engine(tmp_path, llm)

    answer = engine.respond("Atualiza: o exame é só para a semana.")

    assert answer == "Atualizado. O exame ficou registado para a semana."
    assert llm.call_count == 0


def test_teste_h_task_write_command_confirms_deterministically(tmp_path: Path) -> None:
    llm = RaisingLLM()
    engine = make_engine(tmp_path, llm)

    answer = engine.respond("Guarda que tenho de mandar uma mensagem ao Pedro.")

    assert answer == "Registado. Tens de mandar uma mensagem ao Pedro."
    assert llm.call_count == 0

    facts = engine.long_term_memory.find_structured_facts(fact_type="task")
    assert facts[0].action == "mandar uma mensagem ao Pedro"


def test_explicit_write_never_reaches_the_composer(tmp_path: Path) -> None:
    # Problem C: an explicit write must never also produce an unrelated
    # emotional Composer reply on top of the deterministic confirmation.
    llm = RaisingLLM()
    engine = make_engine(tmp_path, llm)
    engine.debug_ollama_payload = True
    engine._begin_turn_trace("Regista que a disciplina é estratégias algoritmicas.")

    answer = engine.respond("Regista que a disciplina é estratégias algoritmicas.")

    assert answer == "Registado. A disciplina é Estratégias Algorítmicas."
    assert llm.call_count == 0


def test_ambiguous_write_command_asks_to_reformulate(tmp_path: Path) -> None:
    llm = RaisingLLM()
    engine = make_engine(tmp_path, llm)

    answer = engine.respond("Regista que sim.")

    assert "reformular" in answer.lower()
    assert llm.call_count == 0


# --- Normalization: Testes I-N -----------------------------------------------


def test_teste_i_accent_and_capitalization_are_corrected() -> None:
    canonical, status = canonicalize_designation("estratégias algoritmicas")
    assert canonical == "Estratégias Algorítmicas"
    assert status == "SAFE_NORMALIZATION"


def test_teste_j_degree_name_is_corrected() -> None:
    canonical, status = canonicalize_designation("engenharia informatica")
    assert canonical == "Engenharia Informática"
    assert status == "SAFE_NORMALIZATION"


def test_teste_k_short_acronym_is_ambiguous_and_left_unchanged() -> None:
    canonical, status = canonicalize_designation("EA")
    assert canonical == "EA"
    assert status == "AMBIGUOUS"


def test_teste_l_task_action_is_corrected_and_target_recapitalized() -> None:
    canonical, status = canonicalize_action("mandar msg ao pedro", "Pedro")
    assert canonical == "mandar uma mensagem ao Pedro"
    assert status == "SAFE_NORMALIZATION"


def test_teste_m_normalization_never_substitutes_a_different_entity() -> None:
    # A plausible-looking but wrong substitution must never happen: the
    # dictionary only ever fixes spelling/capitalization of the SAME value,
    # it never swaps in a different discipline name.
    canonical, _status = canonicalize_designation("estratégias algoritmicas")
    assert canonical != "Algoritmos e Estruturas de Dados"


def test_teste_n_database_keeps_the_raw_value_verbalizer_never_uses_it(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    engine.respond("Regista que a disciplina é estratégias algoritmicas.")

    facts = engine.long_term_memory.find_structured_facts(fact_type="academic_event")
    assert facts[0].discipline_raw == "estratégias algoritmicas"
    assert facts[0].discipline == "Estratégias Algorítmicas"
    assert facts[0].spoken_value("discipline") == "Estratégias Algorítmicas"


# --- Reading / continuity: Testes O-R ----------------------------------------


def test_teste_o_follow_up_question_uses_the_grounded_topic(tmp_path: Path) -> None:
    llm = RaisingLLM()
    engine = make_engine(tmp_path, llm)
    engine._maybe_extract_structured_memory("Tenho um exame de Estratégias Algorítmicas para a semana.")

    first = engine._try_memory_recall_question("Qual era a disciplina do exame?")
    assert "Estratégias Algorítmicas" in first

    second = engine._try_memory_recall_question("Quando era?")
    assert second is not None
    assert "para a semana" in second


def test_teste_p_continuity_survives_one_intervening_turn(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, RaisingLLM())
    engine._maybe_extract_structured_memory("Tenho um exame de Estratégias Algorítmicas para a semana.")

    engine.respond("Qual era a disciplina do exame?")
    assert engine._active_memory_recall_ttl > 0

    # A turn that starts a new topic still lets a same-topic follow-up land
    # right after, thanks to the TTL window (rather than requiring the
    # follow-up to be the literal very-next turn).
    engine.respond("Bom dia.")
    followup = engine._try_memory_recall_question("Quando era?")
    assert followup is not None
    assert "para a semana" in followup


def test_memory_recall_short_challenge_stays_in_memory_route(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, RaisingLLM())
    engine._maybe_extract_structured_memory("Tenho um exame de Estratégias Algorítmicas para a semana.")

    first = engine.respond("Confirma na memória qual é o exame que vou ter para a semana.")
    second = engine.respond("Não tens?")
    third = engine.respond("E a disciplina?")

    assert first is not None
    assert second is not None
    assert third is not None
    assert "Estratégias Algorítmicas" in third
    assert is_memory_recall_followup(_strip_accents("Não tens?".lower()))
    assert engine._active_memory_recall_ttl > 0


def test_teste_q_continuity_expires_after_the_ttl_window(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, RaisingLLM())
    engine._maybe_extract_structured_memory("Tenho um exame de Estratégias Algorítmicas para a semana.")

    engine.respond("Qual era a disciplina do exame?")
    engine.respond("Bom dia.")
    engine.respond("Como estás?")
    engine.respond("Tudo bem contigo?")

    assert engine._active_memory_recall_ttl == 0


def test_teste_r_write_command_is_not_misdetected_as_a_recall_question() -> None:
    assert not is_memory_recall_question(_strip_accents("regista que a disciplina e estrategias algoritmicas".lower()))
    assert is_memory_write_command("Regista que a disciplina é estratégias algoritmicas.")
    assert parse_memory_write_command("Lembras-te do exame?") is None


# --- Section 21: unsupported-claim guard extended to normal conversation ----


def test_composer_cannot_claim_a_memory_it_never_retrieved(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    blocked = engine._complete_turn(
        "Lembras-te do exame?",
        "Sim, lembro-me bem do exame.",
        "RESPONSE_COMPOSER",
        selected_path="CONVERSATIONAL_REFINEMENT",
    )

    assert blocked == "Não tenho essa informação guardada sobre isso."


def test_ordinary_composer_reply_is_not_touched_by_the_memory_guard(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    reply = engine._complete_turn(
        "Como está o tempo hoje?",
        "Não tenho acesso a essa informação em tempo real.",
        "RESPONSE_COMPOSER",
        selected_path="CONVERSATIONAL_REFINEMENT",
    )

    assert reply == "Não tenho acesso a essa informação em tempo real."

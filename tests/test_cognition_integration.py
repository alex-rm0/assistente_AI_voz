from __future__ import annotations

from pathlib import Path

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
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
        if "Boa ideia. Antes de começarmos" in user_message:
            return (
                "Boa ideia. Antes de começarmos a procurar sítios, deixa-me perceber uma coisa. "
                "Quando pensas em férias, procuras mais descansar, conhecer sítios novos ou alguma aventura?"
            )
        if "ainda me falta perceber como gostas de viajar" in user_message:
            return "Acho que ainda me falta perceber como gostas de viajar antes de sugerir destinos."
        if "uso principal" in user_message and "portátil" in user_message:
            return "Antes de sugerir modelos, preciso de perceber o uso principal. É mais para estudar, trabalhar, programar ou jogar?"
        if "localização" in user_message or "localizacao" in user_message:
            return "Antes de procurar casas, preciso de perceber a localização e o tipo de espaço que tens em mente."
        if "uso principal" in user_message and "carro" in user_message:
            return "Antes de sugerir carros, preciso de perceber o uso principal. É mais cidade, viagens longas ou algo familiar?"
        if "Quando é exatamente o exame?" in user_message:
            return "Quando é exatamente o exame? Já tens apontamentos ou vais estudar pelos slides?"
        if "Que exame é?" in user_message:
            return "Que exame é?"
        if "disciplina" in user_message:
            return "É para que disciplina ou tema?"
        return "Resposta direta do LLM."

    def embed(self, text: str):
        return None


def make_engine(tmp_path: Path, llm: FakeLLM) -> AssistantEngine:
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


def test_vague_study_request_asks_question_before_llm(tmp_path: Path) -> None:
    llm = FakeLLM()
    engine = make_engine(tmp_path, llm)

    answer = engine.respond("Ajuda-me a estudar.")

    assert "disciplina" in answer
    assert llm.chat_calls == 1


def test_exam_anxiety_reacts_without_generic_validation(tmp_path: Path) -> None:
    llm = FakeLLM()
    engine = make_engine(tmp_path, llm)

    answer = engine.respond("Estou um pouco nervoso para um exame.")

    assert answer == "É normal ficares nervoso. Que exame é?"
    assert answer.count("?") == 1
    assert llm.chat_calls == 0


def test_exam_subject_followup_asks_next_context_not_theory(tmp_path: Path) -> None:
    llm = FakeLLM()
    engine = make_engine(tmp_path, llm)

    answer = engine.respond("É um exame de Estratégias Algorítmicas.")

    lowered = answer.lower()
    assert answer.startswith("Quando")
    assert answer.count("?") == 2
    assert "algoritmos" not in lowered
    assert "grafos" not in lowered
    assert llm.chat_calls == 1


def test_social_conversation_does_not_enter_deep_cognitive_path(tmp_path: Path) -> None:
    llm = FakeLLM()
    engine = make_engine(tmp_path, llm)

    answer = engine.respond("Olá, como estás?")

    assert "última sessão" not in answer.lower()
    assert "personal model" not in answer.lower()
    assert "contexto" not in answer.lower()
    assert engine.last_cognitive_strategy is None


def test_cognitive_context_is_available_for_general_agent_path(tmp_path: Path) -> None:
    llm = FakeLLM()
    engine = make_engine(tmp_path, llm)

    engine.respond("Quero falar sobre este projeto.")

    assert engine.last_cognitive_reasoning is not None
    assert engine.last_cognitive_reasoning.intent.intent == "conversa_normal"


def test_travel_planning_discovers_preferences_before_recommending(tmp_path: Path) -> None:
    llm = FakeLLM()
    engine = make_engine(tmp_path, llm)

    answer = engine.respond("Queria planear umas férias.")

    assert "descansar" in answer
    assert "conhecer sítios novos" in answer
    assert "Porto" not in answer
    assert llm.chat_calls == 0


def test_travel_destination_followup_does_not_list_destinations(tmp_path: Path) -> None:
    llm = FakeLLM()
    engine = make_engine(tmp_path, llm)

    engine.respond("Queria planear umas férias.")
    answer = engine.respond("Norte de Portugal.")

    assert "ainda me falta perceber como gostas de viajar" in answer
    assert "Porto" not in answer
    assert "Braga" not in answer
    assert "Guimarães" not in answer
    assert llm.chat_calls == 0


def test_laptop_choice_asks_preference_before_recommending(tmp_path: Path) -> None:
    llm = FakeLLM()
    engine = make_engine(tmp_path, llm)

    answer = engine.respond("Preciso de escolher um portátil.")

    assert "uso principal" in answer
    assert "modelos" in answer
    assert llm.chat_calls == 1


def test_home_choice_asks_preference_before_recommending(tmp_path: Path) -> None:
    llm = FakeLLM()
    engine = make_engine(tmp_path, llm)

    answer = engine.respond("Queria procurar uma casa.")

    assert "localização" in answer
    assert "espaço" in answer
    assert llm.chat_calls == 1


def test_car_choice_asks_preference_before_recommending(tmp_path: Path) -> None:
    llm = FakeLLM()
    engine = make_engine(tmp_path, llm)

    answer = engine.respond("Ajuda-me a escolher um carro.")

    assert "uso principal" in answer
    assert "cidade" in answer
    assert llm.chat_calls == 1

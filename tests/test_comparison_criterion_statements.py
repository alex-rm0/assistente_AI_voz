"""Stabilization battery scenario 11: "Ajuda-me a comparar dois portáteis"
then "o critério é a bateria" kept "Still missing: decision criteria" and
repeated the first answer word for word.

Cause: _extract_criterion_priorities only registered a criterion when the
message had a priority expression ("mais importante", "valorizo mais"...);
a plain declaration ("o critério é X", "critério: X", "decidir pela X")
created no knowledge entry, so infer_unknowns still reported it missing.
These go through the real engine (fake LLM, deterministic task path).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.task_intelligence import infer_unknowns, task_knowledge
from assistant.tool_registry import ToolRegistry

FIRST = "Ajuda-me a comparar dois portáteis"
MISSING_CRITERIA = "critérios de decisão"


class _FakeLLM:
    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        return "resposta inventada"

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def embed(self, text: str):
        return None


def _engine(tmp_path: Path) -> AssistantEngine:
    data = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return AssistantEngine(
        llm=_FakeLLM(),
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=LongTermMemory(data, embedder=None),
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
    )


def _run(tmp_path: Path, second: str):
    engine = _engine(tmp_path)
    first_answer = engine.respond(FIRST)
    task = engine._work_session.active_task
    assert task is not None and task.task_type == "generic_comparison"
    assert MISSING_CRITERIA in infer_unknowns(task, task_knowledge(engine._work_session, task.id))

    second_answer = engine.respond(second)

    task = engine._work_session.active_task
    knowledge = task_knowledge(engine._work_session, task.id)
    return first_answer, second_answer, knowledge, infer_unknowns(task, knowledge)


@pytest.mark.parametrize(
    ("second", "predicate"),
    [
        ("o critério é a bateria", "battery_priority"),
        ("o mais importante é a bateria", "battery_priority"),
        ("critério: preço", "price_priority"),
        ("quero decidir pela autonomia", "battery_priority"),
    ],
)
def test_criterion_statement_is_registered(tmp_path: Path, second: str, predicate: str) -> None:
    first_answer, second_answer, knowledge, unknowns = _run(tmp_path, second)

    assert any(entry.kind == "preference" and entry.predicate == predicate for entry in knowledge), knowledge
    assert MISSING_CRITERIA not in unknowns
    assert second_answer != first_answer
    assert "resposta inventada" not in second_answer


@pytest.mark.xfail(
    strict=True,
    reason=(
        "No generic criterion path reaches task knowledge: _criterion_mentions only "
        "knows camera/battery/support/price/location, the SemanticObject kind='criterion' "
        "path (extract_comparison_criteria) only runs for 'comparar ...' workspace "
        "operations and does not feed infer_unknowns, and this phrase is routed to "
        "GENERAL_CONVERSATION. Supporting it needs a new free-text criterion entry plus "
        "routing changes (out of the stabilization scope)."
    ),
)
def test_out_of_list_criterion_is_registered(tmp_path: Path) -> None:
    _first, _second, knowledge, unknowns = _run(tmp_path, "o critério é o ecrã")

    assert any(entry.kind in {"preference", "criterion"} for entry in knowledge), knowledge
    assert MISSING_CRITERIA not in unknowns


def test_unrelated_battery_sentence_does_not_create_criterion(tmp_path: Path) -> None:
    _first, _second, knowledge, unknowns = _run(tmp_path, "a bateria do meu carro morreu")

    assert not [entry for entry in knowledge if entry.kind in {"preference", "criterion"}], knowledge
    assert MISSING_CRITERIA in unknowns

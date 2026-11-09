"""Policy: response_language is a fixed setting, and the task card follows
it. The card's goal ("comparar opcoes") was the raw stored value in every
language, and a language command in the middle of a task did not re-send
the card, so it kept the old language until the next task turn.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.tool_registry import ToolRegistry

FIRST = "Ajuda-me a comparar dois portáteis"


class _FakeLLM:
    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        return "resposta inventada"

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def embed(self, text: str):
        return None


def _engine(tmp_path: Path, response_language: str = "pt-PT") -> AssistantEngine:
    data = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    engine = AssistantEngine(
        llm=_FakeLLM(),
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=LongTermMemory(data, embedder=None),
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
    )
    engine.response_language = response_language
    return engine


def _task_cards(engine: AssistantEngine) -> list[dict]:
    events = [json.loads(raw) for raw in engine.consume_ui_events()]
    return [event for event in events if event.get("type") == "task_artifact_updated"]


def _card_text(card: dict) -> str:
    return json.dumps({"content": card.get("content"), "presentation": card.get("presentation")}, ensure_ascii=False)


@pytest.mark.parametrize(
    ("response_language", "goal", "other_goal"),
    [("pt-PT", "comparar opções", "compare options"), ("en-GB", "compare options", "comparar opcoes")],
)
def test_task_card_goal_follows_response_language(
    tmp_path: Path, response_language: str, goal: str, other_goal: str
) -> None:
    engine = _engine(tmp_path, response_language)

    engine.respond(FIRST)

    cards = _task_cards(engine)
    assert cards, "no task card emitted"
    text = _card_text(cards[-1])
    assert goal in text
    assert other_goal not in text
    assert engine._work_session.active_task.goal == "comparar opcoes"  # stored value unchanged


def test_language_command_mid_task_resends_card_in_new_language(tmp_path: Path) -> None:
    engine = _engine(tmp_path, "pt-PT")
    engine.respond(FIRST)
    before = _task_cards(engine)[-1]

    engine.respond("responde em inglês")

    cards = _task_cards(engine)
    assert cards, "language command did not re-send the task card"
    after = cards[-1]
    assert after["id"] == before["id"]
    assert after["task_id"] == before["task_id"]
    text = _card_text(after)
    assert "compare options" in text
    # The user's own words ("Ajuda-me a comparar ...") stay as typed.
    assert "comparar opcoes" not in text and "comparar opções" not in text

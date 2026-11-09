"""Policy: response_language is a fixed setting. Three visible strings were
still Portuguese with en-GB:
1. NO_SYSTEM_CONTEXT_MESSAGE, the Context Observer tools' reply when there
   is no data (seen in battery scenario 13);
2. the task card subtitle, which showed the raw internal status ("ready");
3. the compose_email card ("Email para", "Para:", "Assunto:"...).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from assistant import tools
from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.tool_registry import ToolRegistry

PT_NO_DATA = "Consigo tentar observar o computador, mas ainda não tenho dados suficientes."
EN_NO_DATA = "I can try to observe the computer, but I don't have enough data yet."


class _FakeLLM:
    def chat(self, user_message, history=None, system_prompt=None, response_format=None, **kwargs):
        return "resposta inventada"

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def embed(self, text: str):
        return None


def _engine(tmp_path: Path, response_language: str) -> AssistantEngine:
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


# --- 1. Context Observer tools without data ---------------------------------

CONTEXT_TOOLS = ("get_active_window", "get_active_application", "get_open_windows", "get_recent_activity")


@pytest.mark.parametrize("tool_name", CONTEXT_TOOLS)
@pytest.mark.parametrize(("response_language", "expected", "other"), [("pt-PT", PT_NO_DATA, EN_NO_DATA), ("en-GB", EN_NO_DATA, PT_NO_DATA)])
def test_context_tool_without_data_answers_in_response_language(
    tool_name: str, response_language: str, expected: str, other: str
) -> None:
    answer = getattr(tools, tool_name)(context_observer=None, response_language=response_language)

    assert answer.startswith(expected), answer
    assert other not in answer


# --- 2. Task card status subtitle -------------------------------------------


@pytest.mark.parametrize(("response_language", "expected"), [("pt-PT", "pronta"), ("en-GB", "ready")])
def test_task_card_subtitle_is_a_readable_label(tmp_path: Path, response_language: str, expected: str) -> None:
    engine = _engine(tmp_path, response_language)

    engine.respond("Ajuda-me a comparar dois portáteis")

    card = _task_cards(engine)[-1]
    assert card["status"] == "ready"  # internal value unchanged
    assert card["subtitle"] == expected


# --- 3. compose_email card --------------------------------------------------

EMAIL_REQUEST = "vamos escrever um email ao professor Silva sobre o exame"


@pytest.mark.parametrize(
    ("response_language", "expected", "forbidden"),
    [
        ("pt-PT", ("Email para", "Para:"), ("Email to", "To:")),
        ("en-GB", ("Email to", "To:"), ("Email para", "Para:", "Assunto:", "Contexto:", "Objetivo:", "em preparação", "pronto para redigir")),
    ],
)
def test_email_card_follows_response_language(
    tmp_path: Path, response_language: str, expected: tuple[str, ...], forbidden: tuple[str, ...]
) -> None:
    engine = _engine(tmp_path, response_language)

    engine.respond(EMAIL_REQUEST)

    cards = [card for card in _task_cards(engine) if card.get("task_type") == "compose_email"]
    assert cards, "no compose_email card emitted"
    text = f"{cards[-1]['title']}\n{cards[-1]['content']}"
    for fragment in expected:
        assert fragment in text, text
    for fragment in forbidden:
        assert fragment not in text, text

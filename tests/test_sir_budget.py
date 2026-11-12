"""R3: "Sir" is an occasional en-GB touch. The model used it in 10-15 of 26
replies, so the limit is enforced when the reply is finalised: at most one
reply in five keeps a "Sir"; the others lose it without breaking the sentence.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.sir_budget import SIR_EVERY, limit_sir, strip_sir
from assistant.tool_registry import ToolRegistry


@pytest.mark.parametrize("text,expected", [
    ("You play the piano, sir.", "You play the piano."),
    ("You play the piano, Sir!", "You play the piano!"),
    ("Shall I continue, Sir?", "Shall I continue?"),
    ("Sir, the report is ready.", "The report is ready."),
    ("Sir — the report is ready.", "The report is ready."),
    ("Good morning Sir.", "Good morning."),
    ("Yes, Sir, of course.", "Yes, of course."),
    ("Certainly, sir. The meeting is at ten, sir.", "Certainly. The meeting is at ten."),
    ("Very good, Sir; I shall wait.", "Very good; I shall wait."),
    ("Of course, Sir", "Of course"),
    ("Thank you sir, I will.", "Thank you, I will."),
    ("Rome is the capital of Italy, sir.\n\nAnything else, sir?", "Rome is the capital of Italy.\n\nAnything else?"),
    ("Quite so, Sir…", "Quite so…"),
])
def test_sir_is_removed_without_breaking_the_sentence(text: str, expected: str) -> None:
    assert strip_sir(text) == expected


@pytest.mark.parametrize("text", [
    "Sir Isaac Newton formulated the laws of motion.",
    "The football manager was Sir Alex Ferguson.",
    "Subject: Meeting\n\nDear Sir,\n\nCan you attend?",
    "Dear Sir or Madam,",
    "The sirloin is excellent.",
    "He was knighted and became a Sir in 2003, I believe.",
    "Rome is the capital of Italy.",
])
def test_titles_letters_and_other_words_are_left_alone(text: str) -> None:
    assert strip_sir(text) == text


def test_a_reply_that_is_only_sir_is_kept_rather_than_emptied() -> None:
    assert strip_sir("Sir?") == "Sir?"
    assert strip_sir("Sir.") == "Sir."


class _Owner:
    """Anything that can carry the counter; the engine in real use."""


def test_at_most_one_reply_in_five_keeps_sir() -> None:
    owner = _Owner()
    replies = [limit_sir(owner, f"Reply {index}, Sir.", "en-GB") for index in range(12)]

    kept = [index for index, reply in enumerate(replies) if "Sir" in reply]
    assert SIR_EVERY == 5
    assert kept == [0, 5, 10]
    assert replies[1] == "Reply 1."
    for start in range(len(replies) - SIR_EVERY + 1):
        assert sum("Sir" in reply for reply in replies[start:start + SIR_EVERY]) <= 1


def test_replies_without_sir_count_towards_the_gap_and_a_kept_reply_has_only_one() -> None:
    owner = _Owner()

    assert limit_sir(owner, "Very good, Sir.", "en-GB") == "Very good, Sir."
    for _ in range(4):
        assert limit_sir(owner, "Noted.", "en-GB") == "Noted."
    assert limit_sir(owner, "Certainly, Sir. It is done, Sir.", "en-GB") == "Certainly, Sir. It is done."
    assert limit_sir(owner, "Yes, Sir.", "en-GB") == "Yes."


def test_portuguese_replies_are_not_touched_or_counted() -> None:
    owner = _Owner()

    assert limit_sir(owner, "Good evening, Sir.", "en-GB") == "Good evening, Sir."
    for _ in range(10):
        assert limit_sir(owner, "Muito bem, Sir.", "pt-PT") == "Muito bem, Sir."
    assert limit_sir(owner, "Indeed, Sir.", "en-GB") == "Indeed."


# --- through the engine ----------------------------------------------------------


class _SirLLM:
    def chat(self, user_message, history=None, system_prompt=None, **kwargs):
        return "That sounds like a long day, sir."

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        return {"tool": None, "arguments": {}, "reason": "none"}

    def embed(self, text: str):
        return None


def test_engine_limits_sir_in_what_it_returns_stores_and_reports(tmp_path: Path) -> None:
    data = tmp_path / "data"
    (tmp_path / "workspace").mkdir()
    memory = LongTermMemory(data, embedder=None)
    memory.set_preference("response_language", "en-GB")
    engine = AssistantEngine(
        llm=_SirLLM(),
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=memory,
        tools=ToolRegistry(),
        workspace_path=tmp_path / "workspace",
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
    )
    messages = ["I had two meetings today.", "Then I wrote a report.", "And then I trained.", "I am tired now.",
                "Tomorrow will be the same.", "At least it is Friday soon."]

    replies = [engine.respond(message) for message in messages]

    with_sir = [index for index, reply in enumerate(replies) if "sir" in reply.lower()]
    assert with_sir == [0, 5], replies
    assert replies[1] == "That sounds like a long day."
    assert engine.get_last_turn_telemetry()["final_response"] == replies[-1]
    stored = [item["content"] for item in engine.history() if item["role"] == "assistant"]
    assert stored == replies

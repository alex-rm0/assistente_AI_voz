from __future__ import annotations

from pathlib import Path

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.response_composer import ComposerRequest, ResponseComposer
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
    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        raise RuntimeError("indisponivel")


def test_llm_failure_interprets_memory_instead_of_returning_literal_fallback() -> None:
    composer = ResponseComposer(RaisingLLM())

    answer = composer.compose(
        ComposerRequest(
            intent="personal_model",
            user_message="o que sabes sobre mim?",
            facts=["usa Codex para programar"],
            fallback="Sei que usas o Codex.",
        )
    )

    assert answer == "Sei que usas o Codex."


def test_copied_memory_reply_is_recomposed() -> None:
    composer = ResponseComposer(FakeLLM(reply="usa Codex para programar"))

    answer = composer.compose(
        ComposerRequest(
            intent="personal_model",
            user_message="o que sabes sobre mim?",
            facts=["usa Codex para programar"],
            fallback="Sei que usas o Codex.",
        )
    )

    assert answer == "Sei que usas o Codex."


def test_long_term_memory_answer_uses_composer_instead_of_raw_listing(tmp_path: Path) -> None:
    data = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    llm = FakeLLM("Tenho a ideia de que costumas organizar primeiro a arquitetura antes de começares a programar.")
    memory = LongTermMemory(data, embedder=llm)
    memory.remember("O Alexandre prefere planear a arquitetura antes de escrever código.")
    engine = AssistantEngine(
        llm=llm,
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=memory,
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
    )

    answer = engine.respond("o que sabes sobre arquitetura?")

    assert answer == llm.reply
    assert "Sei isto na memoria permanente" not in answer
    assert "[preferencias]" not in answer

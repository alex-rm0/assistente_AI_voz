from __future__ import annotations

from pathlib import Path

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.presence_manager import PresenceManager
from assistant.tool_registry import ToolRegistry


class StableLLM:
    system_prompt = ""

    def __init__(self, reply: str = "Estou bem. Há pedidos em que preciso de ferramentas ou mais contexto.") -> None:
        self.reply = reply
        self.chat_call_count = 0

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        self.chat_call_count += 1
        return self.reply

    def choose_tool(self, *args, **kwargs):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def embed(self, *args, **kwargs):
        return None


def make_engine(tmp_path: Path, llm: StableLLM | None = None) -> tuple[AssistantEngine, StableLLM]:
    data = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    model = llm or StableLLM()
    return (
        AssistantEngine(
            llm=model,
            memory=ConversationMemory(data, "history.json", 20),
            long_term_memory=LongTermMemory(data, embedder=model),
            tools=ToolRegistry(),
            workspace_path=workspace,
            base_system_prompt="",
            presence_manager=PresenceManager(),
        ),
        model,
    )


def test_normal_conversation_does_not_raise_for_common_turns(tmp_path: Path) -> None:
    engine, llm = make_engine(tmp_path)

    assert engine.respond("Tudo bem, e tu?")
    assert engine.respond("Porquê?")
    assert engine.respond("Há pedidos a que não consegues responder?")

    assert llm.chat_call_count >= 1


def test_ola_stays_social_fast_path_without_llm(tmp_path: Path) -> None:
    engine, llm = make_engine(tmp_path)

    response = engine.respond("Olá")

    assert response
    assert llm.chat_call_count == 0

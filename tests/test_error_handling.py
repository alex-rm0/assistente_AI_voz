from __future__ import annotations

from pathlib import Path

import assistant.conversation as conversation_module
from assistant.anthropic_provider import AnthropicProvider
from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.model_provider import ProviderBackedLLM
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.tool_registry import ToolRegistry


class FakeLLM:
    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        return "resposta"

    def choose_tool(self, *args, **kwargs):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def embed(self, text: str):
        return None


def make_engine(tmp_path: Path) -> AssistantEngine:
    llm = FakeLLM()
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


def make_anthropic_engine(tmp_path: Path, *, api_key: str = "", allow_paid_calls: bool = True) -> AssistantEngine:
    provider = AnthropicProvider(
        model="claude-haiku-4-5-20251001",
        api_key=api_key,
        allow_paid_calls=allow_paid_calls,
    )
    llm = ProviderBackedLLM(provider, system_prompt=get_base_system_prompt(), model_source="cli")
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
        debug_ollama_payload=True,
    )


def _force_agent_path(monkeypatch) -> None:
    # `self.agent.run()` (conversation.py, inside respond()) is one of the
    # few call sites whose llm.chat() is NOT wrapped by
    # ResponseComposer.compose()'s own try/except. Forcing every message
    # down that branch (instead of the "answer without agent" shortcut)
    # lets a test reach it deterministically instead of depending on the
    # rule-based intent classifier picking a particular category.
    monkeypatch.setattr(conversation_module, "_should_answer_without_agent", lambda strategy: False)


def test_unhandled_exception_never_reaches_the_caller(tmp_path: Path, monkeypatch) -> None:
    engine = make_engine(tmp_path)
    _force_agent_path(monkeypatch)

    def exploding_run(*args, **kwargs):
        raise RuntimeError("O Ollama nao devolveu texto para esta mensagem.")

    monkeypatch.setattr(engine.agent, "run", exploding_run)

    answer = engine.respond("Mensagem qualquer que acabe por chegar ao agente.")

    assert answer == "Desculpa, tive um problema técnico a processar isso. Podes repetir ou reformular a pergunta?"


def test_unhandled_exception_does_not_poison_conversation_history(tmp_path: Path, monkeypatch) -> None:
    engine = make_engine(tmp_path)
    _force_agent_path(monkeypatch)
    monkeypatch.setattr(engine.agent, "run", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))

    engine.respond("Mensagem que vai falhar internamente.")

    assert engine.memory.load() == []


def test_turn_trace_still_reports_the_exception_when_debugging(tmp_path: Path, monkeypatch, capsys) -> None:
    engine = make_engine(tmp_path)
    engine.debug_ollama_payload = True
    _force_agent_path(monkeypatch)
    monkeypatch.setattr(engine.agent, "run", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("falha simulada")))

    engine.respond("Mensagem que vai falhar internamente.")

    out = capsys.readouterr().out
    assert "[TURN TRACE]" in out
    assert "response_source=INTERNAL_ERROR" in out
    assert "exception_type=RuntimeError" in out
    assert "exception_message=falha simulada" in out


def test_traceback_only_prints_with_echo_debug_errors(tmp_path: Path, monkeypatch, capsys) -> None:
    engine = make_engine(tmp_path)
    _force_agent_path(monkeypatch)
    monkeypatch.setattr(engine.agent, "run", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("falha simulada")))

    monkeypatch.delenv("ECHO_DEBUG_ERRORS", raising=False)
    engine.respond("Mensagem que vai falhar internamente.")
    captured = capsys.readouterr()
    assert "Traceback (most recent call last)" not in captured.err
    assert "[ECHO ERROR]" in captured.out

    monkeypatch.setenv("ECHO_DEBUG_ERRORS", "1")
    engine.respond("Mensagem que vai falhar internamente outra vez.")
    captured = capsys.readouterr()
    assert "Traceback (most recent call last)" in captured.err


def test_normal_conversation_messages_do_not_raise(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    for message in [
        "Olá",
        "Tudo bem, e tu?",
        "Porquê?",
        "Há pedidos a que não consegues responder?",
    ]:
        answer = engine.respond(message)
        assert isinstance(answer, str)
        assert answer.strip()


def test_anthropic_missing_api_key_surfaces_clear_provider_error(tmp_path: Path) -> None:
    engine = make_anthropic_engine(tmp_path, api_key="", allow_paid_calls=True)

    answer = engine.respond("Explica-me em duas frases o que e uma arvore binaria.")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert answer == "O provider Anthropic esta selecionado, mas falta configurar ANTHROPIC_API_KEY."
    assert telemetry["selected_path"] == "PROVIDER_ERROR"
    assert telemetry["response_source"] == "PROVIDER_ERROR"
    assert telemetry["exception_type"] == "ProviderConfigurationError"
    assert telemetry["provider_error_type"] == "missing_api_key"
    assert telemetry["provider"] == "anthropic"
    assert telemetry["llm_calls"] == 0
    assert telemetry["fallback_used"] is False
    assert "problema t" not in answer.lower()


def test_anthropic_missing_paid_confirmation_surfaces_clear_provider_error(tmp_path: Path) -> None:
    engine = make_anthropic_engine(tmp_path, api_key="secret", allow_paid_calls=False)

    answer = engine.respond("Explica-me em duas frases o que e uma arvore binaria.")
    telemetry = engine.get_last_turn_telemetry() or {}

    assert "chamadas pagas estao bloqueadas" in answer
    assert "ECHO_ALLOW_PAID_MODEL_CALLS=true" in answer
    assert telemetry["selected_path"] == "PROVIDER_ERROR"
    assert telemetry["response_source"] == "PROVIDER_ERROR"
    assert telemetry["exception_type"] == "ProviderConfigurationError"
    assert telemetry["provider_error_type"] == "paid_calls_not_confirmed"
    assert telemetry["provider"] == "anthropic"
    assert telemetry["llm_calls"] == 0
    assert telemetry["fallback_used"] is False

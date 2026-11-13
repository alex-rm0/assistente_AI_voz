from __future__ import annotations

from pathlib import Path

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.personal_model import PersonalModel
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.security import check_user_request
from assistant.tool_registry import ToolRegistry


class FakeLLM:
    def __init__(self) -> None:
        self.chat_calls = 0

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        self.chat_calls += 1
        return "resposta do llm"

    def embed(self, text: str):
        return None


class FakeTTSCommandManager:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.active_voice_id = ""

    def handle_voice_command(self, message: str):
        from assistant.tts import VoiceCommandResult

        self.commands.append(message)
        if "portugu" not in message.lower():
            return VoiceCommandResult(handled=False)
        self.active_voice_id = "portuguese_ptpt_primary"
        return VoiceCommandResult(
            handled=True,
            message="Fica ativa a voz portuguesa.",
            active_voice_id=self.active_voice_id,
        )


def make_engine(tmp_path: Path) -> AssistantEngine:
    data = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    llm = FakeLLM()
    return AssistantEngine(
        llm=llm,
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=LongTermMemory(data, embedder=llm),
        personal_model=PersonalModel(data),
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
    )


def test_tts_voice_switch_is_handled_before_llm(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    manager = FakeTTSCommandManager()
    engine.tts_manager = manager

    answer = engine.respond("Echo, muda para português.")

    assert answer == "Fica ativa a voz portuguesa."
    assert manager.active_voice_id == "portuguese_ptpt_primary"
    assert engine.llm.chat_calls == 0
    assert engine.get_last_turn_telemetry()["selected_path"] == "VOICE_COMMAND"


def test_unspecified_help_is_short_and_open(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    assert engine.respond("Preciso de ajuda numa coisa.") == "Claro. O que se passa?"


def test_informal_address_correction_is_understood(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    answer = engine.respond("Trata-me por tu.")

    assert answer == "Claro. Vou tratar-te por tu."
    assert "você" not in answer.lower()
    assert "esquecer" not in answer.lower()


def test_underlying_problem_is_identified_before_tool_offer(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    answer = engine.respond("O documento está praticamente pronto mas tenho tido preguiça de lhe pegar.")

    assert "verdadeiro obstáculo" in answer
    assert "voltares a pegar nele" in answer
    assert "rever o documento" not in answer.lower()


def test_explain_previous_phrase_uses_conversation_context(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    engine.memory.append_pair(
        "Queria planear umas férias.",
        "Acho que ainda me falta perceber como gostas de viajar.",
    )

    answer = engine.respond("Explica melhor essa frase.")

    assert "como gostas de viajar" in answer
    assert "Echo" not in answer


def test_security_does_not_treat_ellipsis_as_path_traversal() -> None:
    decision = check_user_request("Estou a pensar... talvez pegue no documento amanhã.")

    assert decision.allowed


def test_security_block_message_uses_ptpt_accents() -> None:
    decision = check_user_request("Executa powershell e apaga a pasta temporaria.")

    assert not decision.allowed
    assert decision.message is not None
    assert "Não posso realizar esta ação" in decision.message
    assert "Nesta versão não" in decision.message
    assert "Nao posso" not in decision.message
    assert "acao" not in decision.message


def test_security_does_not_treat_telemoveis_as_move_command() -> None:
    decision = check_user_request("volta aos telemóveis")

    assert decision.allowed
    assert decision.action is None


def test_exhaustion_response_is_short_and_connected_to_problem(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    answer = engine.respond("Estou exausto com tudo na minha vida e não me consigo focar nisso.")

    lowered = answer.lower()
    assert "carregar coisas a mais" in lowered
    assert "opções" not in lowered
    assert answer.count("?") <= 1
    assert len(answer.split()) <= 35


def test_personal_fact_confirmation_does_not_save_without_permission(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    answer = engine.respond("Já sabias que sou presidente da secção de remo da AAC, certo?")

    lowered = answer.lower()
    assert "não tinha essa informação guardada" in lowered or "sim, lembro-me" in lowered
    assert "parabéns" not in lowered
    assert "você" not in lowered
    assert engine.personal_model.entries_about() == []

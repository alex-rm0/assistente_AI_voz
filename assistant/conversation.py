from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.security import check_user_request
from assistant.tool_registry import ToolRegistry
from assistant.tools import WorkspaceGuard, read_workspace_file_content


if TYPE_CHECKING:
    from assistant.llm import OllamaClient


MAX_SUMMARY_CHARACTERS = 12000


class AssistantEngine:
    """Coordinates conversation, tools, memory and safety policy."""

    def __init__(
        self,
        llm: OllamaClient,
        memory: ConversationMemory,
        long_term_memory: LongTermMemory,
        tools: ToolRegistry,
        workspace_path: Path,
        base_system_prompt: str,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.long_term_memory = long_term_memory
        self.tools = tools
        self.workspace = WorkspaceGuard(workspace_path)
        self.base_system_prompt = base_system_prompt

    def respond(self, user_message: str) -> str:
        # Local memory questions are answered before tool routing so the LLM
        # cannot confuse "lembras-te da pasta?" with "lista a pasta".
        memory_command_response = self._try_long_term_memory_command(user_message)
        if memory_command_response is not None:
            self.memory.append_pair(user_message, memory_command_response)
            return memory_command_response

        profile_response = self._try_profile_memory(user_message)
        if profile_response is not None:
            self.memory.append_pair(user_message, profile_response)
            return profile_response

        conversation_memory_response = self._try_conversation_memory_question(user_message)
        if conversation_memory_response is not None:
            self.memory.append_pair(user_message, conversation_memory_response)
            return conversation_memory_response

        security = check_user_request(user_message)
        if not security.allowed:
            response = security.message or "Nao posso realizar essa acao por motivos de seguranca."
            self.memory.append_pair(user_message, response)
            return response

        # Summary still uses the LLM over safely-read workspace content.
        summary_response = self._try_summarize(user_message)
        if summary_response is not None:
            return summary_response

        tool_response = self._try_tool(user_message)
        if tool_response is not None:
            return tool_response

        history = self.memory.load()
        recurring_context = self.long_term_memory.context_for(user_message)
        response = self.llm.chat(
            user_message,
            history=history,
            system_prompt=self._system_prompt_with_tools(recurring_context),
        )
        self.memory.append_pair(user_message, response)
        return response

    def clear_history(self) -> None:
        self.memory.clear()

    def history(self) -> list[dict[str, str]]:
        return self.memory.load()

    def _try_tool(self, user_message: str) -> str | None:
        # The LLM only chooses the tool name and arguments. Execution stays in
        # Python, through the registry and the workspace guard.
        decision = self.llm.choose_tool(user_message, self.tools.describe())
        tool_name = decision.get("tool")
        if not tool_name:
            return None

        tool = self.tools.get(str(tool_name))
        if tool is None:
            return None

        arguments = self._prepare_tool_arguments(decision.get("arguments", {}))
        response = tool.run(arguments)
        if tool.remember_result:
            self.memory.append_pair(user_message, response)
        return response

    def _try_summarize(self, user_message: str) -> str | None:
        decision = self.llm.choose_tool(user_message, self.tools.describe())
        if decision.get("tool") != "read_workspace_file":
            return None

        arguments = decision.get("arguments", {})
        filename = arguments.get("filename") if isinstance(arguments, dict) else None
        if not isinstance(filename, str):
            return None

        if not _looks_like_summary_request(user_message):
            return None

        file_content = read_workspace_file_content(filename, self.workspace.resolve())
        if file_content.error is not None:
            return file_content.error
        if len(file_content.content) > MAX_SUMMARY_CHARACTERS:
            return (
                "Este ficheiro e demasiado grande para a versao atual. "
                "Por enquanto, o AssistenteIA so resume ficheiros pequenos."
            )
        if not file_content.content.strip():
            return f"O ficheiro '{file_content.filename}' esta vazio."

        return self.llm.summarize_text(file_content.filename, file_content.content)

    def _prepare_tool_arguments(self, arguments: Any) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            arguments = {}

        prepared = dict(arguments)
        prepared["workspace_path"] = self.workspace.resolve()
        return prepared

    def _try_long_term_memory_command(self, user_message: str) -> str | None:
        lowered = _normalize_text(user_message.strip())
        if lowered.startswith("lembra-te que"):
            content = user_message.strip()[len("lembra-te que") :].strip(" .:")
            return self.long_term_memory.remember(content)

        if lowered.startswith("lembra que"):
            content = user_message.strip()[len("lembra que") :].strip(" .:")
            return self.long_term_memory.remember(content)

        if lowered.startswith("esquece"):
            query = user_message.strip()[len("esquece") :].strip(" .:")
            return self.long_term_memory.forget(query)

        if lowered.startswith("o que sabes sobre"):
            query = user_message.strip()[len("o que sabes sobre") :].strip(" .:?")
            return self.long_term_memory.answer_about(query)

        return None

    def _try_profile_memory(self, user_message: str) -> str | None:
        text = _normalize_text(user_message)
        name_match = re.search(
            r"\b(?:chamo me|chamo-me|o meu nome e|meu nome e)\s+([^.,;:!?\n]{2,60})",
            user_message,
            re.IGNORECASE,
        )
        if name_match:
            name = name_match.group(1).strip(" .,!?:;")
            self.long_term_memory.remember(f"O utilizador chama-se {name}.", category="preferencia")
            return f"Obrigado, {name}. Vou lembrar-me do teu nome."

        if _asks_for_user_name(text):
            name = self._find_known_name()
            if name:
                return f"Chamas-te {name}."
            return "Ainda nao sei como te chamas. Podes dizer-me com: chamo-me Alexandre."

        return None

    def _try_conversation_memory_question(self, user_message: str) -> str | None:
        text = _normalize_text(user_message)
        if _asks_about_previous_conversation(text):
            return self._summarize_conversation_history()

        if "lembras" in text or "lembras-te" in text:
            if any(word in text for word in ("pasta", "ficheiros", "workspace")):
                return self._answer_if_discussed_workspace()

        return None

    def _find_known_name(self) -> str | None:
        for record in self.long_term_memory.search("utilizador nome chama-se", limit=10):
            match = re.search(r"utilizador chama-se\s+(.+?)[.?!]?$", record.content, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        for message in reversed(self.memory.load()):
            if message.get("role") != "user":
                continue
            match = re.search(
                r"\b(?:chamo me|chamo-me|o meu nome e|meu nome e)\s+([^.,;:!?\n]{2,60})",
                message.get("content", ""),
                re.IGNORECASE,
            )
            if match:
                return match.group(1).strip(" .,!?:;")

        return None

    def _summarize_conversation_history(self) -> str:
        history = self.memory.load()
        user_messages = [message["content"] for message in history if message.get("role") == "user"]
        if not user_messages:
            return "Ainda nao temos historico de conversa guardado."

        recent = user_messages[-6:]
        lines = [f"- {message}" for message in recent]
        return "Falamos recentemente sobre:\n" + "\n".join(lines)

    def _answer_if_discussed_workspace(self) -> str:
        for message in reversed(self.memory.load()):
            if message.get("role") != "user":
                continue
            text = _normalize_text(message.get("content", ""))
            if any(word in text for word in ("pasta", "ficheiros", "workspace")):
                return "Sim. Pediste-me para ver/listar a pasta workspace."

        return "Nao encontro no historico recente um pedido teu sobre a pasta workspace."

    def _system_prompt_with_tools(self, recurring_context: str = "") -> str:
        # The normal chat prompt includes tool descriptions and relevant
        # long-term memory, but not raw file contents.
        prompt = (
            f"{self.base_system_prompt}\n\n"
            "Ferramentas disponiveis para a aplicacao:\n"
            f"{self.tools.describe()}\n\n"
            "Quando precisares de uma ferramenta, a aplicacao decide e executa-a antes da resposta final."
        )
        if recurring_context:
            prompt += f"\n\nMemoria permanente relevante:\n{recurring_context}"
        return prompt


def _looks_like_summary_request(message: str) -> bool:
    text = message.lower()
    return any(word in text for word in ("resume", "resumir", "sumariza", "sumarizar", "resumo"))


def _asks_about_previous_conversation(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "o que e que ja falamos",
            "o que ja falamos",
            "sobre o que falamos",
            "que falamos",
            "historico da conversa",
            "resumo da conversa",
        )
    )


def _asks_for_user_name(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "como e que me chamo",
            "como e que eu me chamo",
            "como me chamo",
            "como eu me chamo",
            "qual e o meu nome",
            "sabes o meu nome",
            "diz o meu nome",
        )
    )


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))

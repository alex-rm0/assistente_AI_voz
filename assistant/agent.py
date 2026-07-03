from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from assistant.desktop_actions import ALLOWED_APPLICATIONS, app_label, is_application_open, normalize_app_name
from assistant.tool_registry import ToolRegistry


MAX_AGENT_STEPS = 5
MAX_OBSERVATION_CHARS_FOR_SUMMARY = 12000
CONFIRMATION_REQUIRED_TOOLS = {
    "create_workspace_file",
    "open_application",
    "open_file",
    "open_folder",
    "open_url",
    "open_project",
}
MISSING_MONITORING_TOOL_MESSAGE = "A ferramenta de monitorização ainda não está ligada ao agente."


@dataclass(frozen=True)
class AgentResult:
    response: str
    remember: bool = True


@dataclass(frozen=True)
class AgentStep:
    tool_name: str
    arguments: dict[str, Any]
    reason: str
    observation: str


@dataclass(frozen=True)
class AgentContext:
    system_prompt: str
    history: list[dict[str, str]]
    profile_name: str = "AUTO_CONTEXT"
    active_contexts: list[str] | None = None
    context_debug: str = ""
    recurring_context: str = ""
    pending_tasks: str = ""
    presence_state: str = "ACTIVE_CONVERSATION"
    tools_enabled: bool = True
    suggestions_enabled: bool = True
    confirmations_enabled: bool = True


class Agent:
    """Small agent loop: decide, optionally use one tool, observe, answer."""

    def __init__(
        self,
        llm: Any,
        tools: ToolRegistry,
        workspace_path: Path,
        context_observer: Any | None = None,
        presence_manager: Any | None = None,
        long_term_memory: Any | None = None,
        known_projects: dict[str, str] | None = None,
        desktop_action_runner: Any | None = None,
        debug_agent: bool = False,
        max_steps: int = MAX_AGENT_STEPS,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.workspace_path = workspace_path
        self.context_observer = context_observer
        self.presence_manager = presence_manager
        self.long_term_memory = long_term_memory
        self.known_projects = known_projects or {}
        self.desktop_action_runner = desktop_action_runner
        self.debug_agent = debug_agent
        self.max_steps = max_steps
        self.pending_confirmation: dict[str, Any] | None = None
        self._current_context: AgentContext | None = None

    def run(self, user_message: str, context: AgentContext) -> AgentResult:
        if self.pending_confirmation is not None:
            return self._handle_pending_confirmation(user_message)

        self._current_context = context
        normalized_message = _normalize_text(user_message)
        tools_description = self.tools.describe()
        system_prompt = self._system_prompt_with_tools(context, tools_description)
        if not context.tools_enabled and _looks_like_system_state_request(normalized_message):
            return AgentResult(MISSING_MONITORING_TOOL_MESSAGE)

        if _looks_like_presence_state_question(normalized_message):
            plan = [
                {
                    "tool": "get_presence_state",
                    "arguments": {},
                    "reason": "O utilizador perguntou pelo modo de presenca; a resposta deve vir do PresenceManager.",
                }
            ]
        else:
            plan = (
                self._build_plan(user_message, _context_label(context), tools_description)
                if context.tools_enabled
                else []
            )

        if not plan:
            response = self.llm.chat(
                user_message,
                history=context.history,
                system_prompt=system_prompt,
            )
            return AgentResult(response=response, remember=True)

        observations: list[AgentStep] = []
        plan_index = 0
        while plan_index < len(plan) and len(observations) < self.max_steps:
            decision = plan[plan_index]
            plan_index += 1
            tool_name = str(decision.get("tool") or "")
            reason = str(decision.get("reason") or "Nao foi indicada razao.")
            arguments = self._prepare_tool_arguments(decision.get("arguments", {}))

            if not tool_name:
                continue

            already_open_response = self._already_open_response(tool_name, arguments)
            if already_open_response is not None:
                step = AgentStep(
                    tool_name=tool_name,
                    arguments=arguments,
                    reason=reason,
                    observation=already_open_response,
                )
                return AgentResult(response=self._with_debug(already_open_response, [step]), remember=True)

            if tool_name in CONFIRMATION_REQUIRED_TOOLS:
                if not context.confirmations_enabled:
                    return AgentResult(
                        "Nao posso executar esta acao neste estado de presenca. "
                        "Muda para ACTIVE_CONVERSATION para poderes confirmar esta acao."
                    )
                if tool_name == "create_workspace_file":
                    return self._ask_create_confirmation(arguments, reason, observations)
                return self._ask_desktop_action_confirmation(tool_name, arguments, reason, observations)

            tool = self.tools.get(tool_name)
            if tool is None:
                if tool_name == "get_presence_state":
                    return AgentResult("A ferramenta de presença ainda não está ligada ao agente.", remember=True)
                if tool_name.startswith("get_") and _looks_like_system_state_request(normalized_message):
                    return AgentResult(MISSING_MONITORING_TOOL_MESSAGE, remember=True)
                response = self.llm.chat(
                    user_message,
                    history=context.history,
                    system_prompt=system_prompt,
                )
                return AgentResult(response=response, remember=True)

            # Safety net: never execute a read without a concrete filename.
            if tool_name == "read_workspace_file":
                filename = str(arguments.get("filename", "")).strip().strip("\"'")
                if not filename:
                    continue

            observation = tool.run(arguments)
            observations.append(
                AgentStep(
                    tool_name=tool_name,
                    arguments=arguments,
                    reason=reason,
                    observation=observation,
                )
            )

            next_steps = self._next_steps_from_observation(user_message, observations)
            remaining_slots = self.max_steps - len(observations)
            if next_steps and remaining_slots > 0:
                plan.extend(next_steps[:remaining_slots])

        if not observations:
            response = self.llm.chat(
                user_message,
                history=context.history,
                system_prompt=system_prompt,
            )
            return AgentResult(response=response, remember=True)

        final_response = self._answer_from_observations(
            user_message=user_message,
            context=context,
            observations=observations,
            system_prompt=system_prompt,
        )
        final_response = self._with_debug(final_response, observations)
        return AgentResult(response=final_response, remember=True)

    def has_pending_confirmation(self) -> bool:
        return self.pending_confirmation is not None

    def ask_tool_confirmation(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        reason: str,
        context: AgentContext,
    ) -> AgentResult:
        self._current_context = context
        prepared = self._prepare_tool_arguments(arguments)
        already_open_response = self._already_open_response(tool_name, prepared)
        if already_open_response is not None:
            step = AgentStep(tool_name=tool_name, arguments=prepared, reason=reason, observation=already_open_response)
            return AgentResult(response=self._with_debug(already_open_response, [step]), remember=True)
        return self._ask_desktop_action_confirmation(tool_name, prepared, reason, [])

    def _build_plan(
        self,
        user_message: str,
        profile_name: str,
        tools_description: str,
    ) -> list[dict[str, Any]]:
        deterministic = self._deterministic_plan(user_message)
        if deterministic:
            return deterministic

        decision = self.llm.choose_tool(
            user_message,
            tools_description,
            profile_name=profile_name,
        )
        if decision.get("tool"):
            return [decision]
        return []

    def _deterministic_plan(self, user_message: str) -> list[dict[str, Any]]:
        text = _normalize_text(user_message)

        if _looks_like_presence_state_question(text):
            return [
                {
                    "tool": "get_presence_state",
                    "arguments": {},
                    "reason": "O utilizador perguntou pelo modo de presenca; a resposta deve vir do PresenceManager.",
                }
            ]

        system_state_tool = _system_state_tool_for(text)
        if system_state_tool:
            return [
                {
                    "tool": system_state_tool,
                    "arguments": {},
                    "reason": "O utilizador perguntou sobre o estado do computador; a resposta deve vir do Context Observer.",
                }
            ]

        desktop_action_plan = _desktop_action_plan(user_message, text)
        if desktop_action_plan:
            return desktop_action_plan

        if _asks_to_list_and_summarize_first(text):
            return [
                {
                    "tool": "list_workspace_files",
                    "arguments": {},
                    "reason": "O utilizador pediu para listar ficheiros e resumir o primeiro.",
                }
            ]

        if _asks_to_read_and_create_note(text):
            filename = _extract_readable_filename(user_message)
            if filename:
                return [
                    {
                        "tool": "read_workspace_file",
                        "arguments": {"filename": filename},
                        "reason": "O utilizador pediu para ler um ficheiro antes de criar uma nota.",
                    }
                ]

        if _asks_to_find_relevant_documents(text):
            return [
                {
                    "tool": "list_workspace_files",
                    "arguments": {},
                    "reason": "O utilizador pediu para procurar documentos relevantes na workspace.",
                }
            ]

        if _asks_to_analyze_existing_files(text):
            return [
                {
                    "tool": "list_workspace_files",
                    "arguments": {},
                    "reason": "O utilizador pediu para analisar os ficheiros existentes.",
                }
            ]

        if _looks_like_create_request(text):
            filename, content = _extract_create_request(user_message)
            if filename:
                return [
                    {
                        "tool": "create_workspace_file",
                        "arguments": {"filename": filename, "content": content},
                        "reason": "O utilizador pediu para criar um ficheiro .txt na workspace.",
                    }
                ]

        if _looks_like_list_request(text):
            return [
                {
                    "tool": "list_workspace_files",
                    "arguments": {},
                    "reason": "O utilizador pediu para listar ou mostrar ficheiros da workspace.",
                }
            ]

        if _looks_like_read_or_summary_request(text):
            filename = _extract_readable_filename(user_message)
            if filename:
                return [
                    {
                        "tool": "read_workspace_file",
                        "arguments": {"filename": filename},
                        "reason": "O utilizador pediu para ler ou resumir um ficheiro da workspace.",
                    }
                ]

        return []

    def _next_steps_from_observation(
        self,
        user_message: str,
        observations: list[AgentStep],
    ) -> list[dict[str, Any]]:
        text = _normalize_text(user_message)
        last = observations[-1]

        if last.tool_name == "list_workspace_files":
            files = _extract_files_from_listing(last.observation)
            if _asks_to_list_and_summarize_first(text) and files:
                return [
                    {
                        "tool": "read_workspace_file",
                        "arguments": {"filename": files[0]},
                        "reason": f"O primeiro ficheiro listado foi '{files[0]}'.",
                    }
                ]

            if _asks_to_find_relevant_documents(text) and files:
                readable = [name for name in files if _is_readable_file(name)]
                return [
                    {
                        "tool": "read_workspace_file",
                        "arguments": {"filename": filename},
                        "reason": "Ler documentos candidatos para avaliar relevancia.",
                    }
                    for filename in readable[: self.max_steps - 1]
                ]

        if last.tool_name == "read_workspace_file" and _asks_to_read_and_create_note(text):
            filename = _extract_readable_filename(user_message) or "ficheiro"
            note_filename = _note_filename_for(filename)
            content = self._make_note_content(user_message, observations)
            return [
                {
                    "tool": "create_workspace_file",
                    "arguments": {"filename": note_filename, "content": content},
                    "reason": "Criar uma nota com os pontos principais exige confirmacao do utilizador.",
                }
            ]

        return []

    def _answer_from_observations(
        self,
        user_message: str,
        context: AgentContext,
        observations: list[AgentStep],
        system_prompt: str,
    ) -> str:
        text = _normalize_text(user_message)
        observation_text = _format_observations(observations)

        if _asks_to_analyze_existing_files(text):
            prompt = (
                "Analisa a lista de ficheiros observada e sugere uma organizacao simples para a workspace. "
                "Nao proponhas mover nem apagar ficheiros; apresenta apenas sugestoes.\n\n"
                f"Pedido do utilizador: {user_message}\n\n{observation_text}"
            )
            return self.llm.chat(prompt, history=context.history, system_prompt=system_prompt)

        if _asks_to_find_relevant_documents(text):
            prompt = (
                "Com base nos ficheiros e conteudos observados, indica quais parecem relevantes para o tema pedido. "
                "Justifica de forma curta e nao inventes conteudo que nao apareca nas observacoes.\n\n"
                f"Pedido do utilizador: {user_message}\n\n{observation_text}"
            )
            return self.llm.chat(prompt, history=context.history, system_prompt=system_prompt)

        if any(step.tool_name == "read_workspace_file" for step in observations) and (
            _looks_like_summary_request(user_message) or _asks_to_list_and_summarize_first(text)
        ):
            if len(observation_text) > MAX_OBSERVATION_CHARS_FOR_SUMMARY:
                return (
                    "Este ficheiro e demasiado grande para a versao atual. "
                    "Por enquanto, o AssistenteIA so resume ficheiros pequenos."
                )
            prompt = (
                "Resume em portugues de Portugal o conteudo observado pelas ferramentas. "
                "Mantem o resumo claro, fiel e util para o utilizador.\n\n"
                f"Pedido do utilizador: {user_message}\n\n"
                f"{observation_text}"
            )
            return self.llm.chat(prompt, history=context.history, system_prompt=system_prompt)

        if len(observations) == 1:
            return observations[0].observation

        prompt = (
            "Responde ao utilizador com base apenas nas observacoes das ferramentas. "
            "Apresenta so a resposta final, sem mostrar plano interno.\n\n"
            f"Pedido do utilizador: {user_message}\n\n{observation_text}"
        )
        return self.llm.chat(prompt, history=context.history, system_prompt=system_prompt)

    def _make_note_content(self, user_message: str, observations: list[AgentStep]) -> str:
        prompt = (
            "Cria uma nota curta em portugues de Portugal com os pontos principais do conteudo observado. "
            "Devolve apenas o texto da nota.\n\n"
            f"Pedido do utilizador: {user_message}\n\n{_format_observations(observations)}"
        )
        return self.llm.chat(prompt, history=[], system_prompt="Escreve notas claras e curtas.")

    def _ask_create_confirmation(
        self,
        arguments: dict[str, Any],
        reason: str,
        previous_observations: list[AgentStep],
    ) -> AgentResult:
        filename = str(arguments.get("filename", "")).strip()
        content = str(arguments.get("content", ""))
        self.pending_confirmation = {
            "tool": "create_workspace_file",
            "arguments": {"filename": filename, "content": content},
            "reason": reason,
            "previous_observations": previous_observations,
        }
        preview = content.replace("\n", " ").strip()
        if len(preview) > 300:
            preview = preview[:297] + "..."

        response = (
            f"Para criar o ficheiro '{filename}', preciso da tua confirmacao.\n"
            "Responde 'sim' para criar ou 'nao' para cancelar.\n\n"
            f"Resumo do conteudo a guardar: {preview or '(sem conteudo)'}"
        )
        if self.debug_agent:
            response += (
                "\n\n[DEBUG_AGENT]\n"
                "Passo pendente: create_workspace_file\n"
                f"Razao da escolha: {reason}\n"
                "Resultado resumido: a criacao aguarda confirmacao."
            )
        return AgentResult(response=response, remember=True)

    def _already_open_response(self, tool_name: str, arguments: dict[str, Any]) -> str | None:
        if tool_name == "open_application":
            app_key = normalize_app_name(str(arguments.get("app_name", "")))
            if app_key and is_application_open(app_key, self.context_observer):
                return f"O {app_label(app_key)} ja esta aberto."
        if tool_name == "open_project" and is_application_open("vscode", self.context_observer):
            return "O VS Code ja esta aberto."
        return None

    def _ask_desktop_action_confirmation(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        reason: str,
        previous_observations: list[AgentStep],
    ) -> AgentResult:
        self.pending_confirmation = {
            "tool": tool_name,
            "arguments": dict(arguments),
            "reason": reason,
            "previous_observations": previous_observations,
        }
        question = _desktop_confirmation_question(tool_name, arguments)
        response = f"{question}\nResponde 'sim' para executar ou 'nao' para cancelar."
        if self.debug_agent:
            response += (
                "\n\n[DEBUG_AGENT]\n"
                f"Passo pendente: {tool_name}\n"
                f"Razao da escolha: {reason}\n"
                "Resultado resumido: a acao aguarda confirmacao."
            )
        return AgentResult(response=response, remember=True)

    def _handle_pending_confirmation(self, user_message: str) -> AgentResult:
        text = _normalize_text(user_message)
        if text in {"nao", "não", "cancelar", "cancela"}:
            pending = self.pending_confirmation or {}
            self.pending_confirmation = None
            if str(pending.get("tool") or "") == "create_workspace_file":
                return AgentResult("Operacao cancelada. Nao criei nenhum ficheiro.")
            return AgentResult("Operacao cancelada. Nao executei a acao.")

        if text not in {"sim", "s", "confirmo", "confirma", "podes criar", "podes abrir", "abre"}:
            return AgentResult(
                "Tenho uma acao pendente. Responde 'sim' para executar ou 'nao' para cancelar."
            )

        pending = self.pending_confirmation
        self.pending_confirmation = None
        if pending is None:
            return AgentResult("Nao ha nenhuma acao pendente para confirmar.")

        tool = self.tools.get(str(pending["tool"]))
        if tool is None:
            return AgentResult("Nao consegui concluir a acao pendente: ferramenta indisponivel.")

        arguments = self._prepare_tool_arguments(pending.get("arguments", {}))
        observation = tool.run(arguments)
        response = observation
        if self.debug_agent:
            step = AgentStep(
                tool_name=str(pending["tool"]),
                arguments=arguments,
                reason=str(pending.get("reason") or "Confirmacao do utilizador."),
                observation=observation,
            )
            response = self._with_debug(response, [step])
        return AgentResult(response=response, remember=tool.remember_result)

    def _prepare_tool_arguments(self, arguments: Any) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            arguments = {}

        prepared = dict(arguments)
        prepared["workspace_path"] = self.workspace_path
        prepared["context_observer"] = self.context_observer
        prepared["presence_manager"] = self.presence_manager
        prepared["long_term_memory"] = self.long_term_memory
        prepared["project_root"] = self.workspace_path.parent
        prepared["known_projects"] = self.known_projects
        if self.desktop_action_runner is not None:
            prepared["desktop_action_runner"] = self.desktop_action_runner
        if self._current_context is not None:
            prepared["active_contexts"] = self._current_context.active_contexts or []
            prepared["relevant_memory"] = self._current_context.recurring_context
            prepared["pending_tasks"] = self._current_context.pending_tasks
        return prepared

    def _system_prompt_with_tools(self, context: AgentContext, tools_description: str) -> str:
        prompt = (
            f"{context.system_prompt}\n\n"
            f"Contextos ativos: {_context_label(context)}.\n\n"
            f"Estado de presenca: {context.presence_state}.\n"
            f"Ferramentas ativas: {'sim' if context.tools_enabled else 'nao'}.\n"
            f"Sugestoes ativas: {'sim' if context.suggestions_enabled else 'nao'}.\n\n"
            "Ferramentas disponiveis para a aplicacao:\n"
            f"{tools_description}\n\n"
            "Decide internamente se deves responder diretamente ou usar uma ferramenta. "
            "Nao mostres o teu plano interno ao utilizador.\n"
            "Nunca inventes informacoes sobre o estado do computador, janelas, aplicacoes, programas "
            "ou atividade recente. Se essa informacao nao vier do Context Observer, admite desconhecimento."
        )
        if context.recurring_context:
            prompt += f"\n\nMemoria permanente relevante:\n{context.recurring_context}"
        if context.context_debug:
            prompt += f"\n\nDebug interno de contexto:\n{context.context_debug}"
        return prompt

    def _with_debug(self, response: str, observations: list[AgentStep]) -> str:
        if not self.debug_agent:
            return response

        lines = ["", "", "[DEBUG_AGENT]", "Passos internos:"]
        for index, step in enumerate(observations, start=1):
            summary = step.observation.replace("\n", " ").strip()
            if len(summary) > 300:
                summary = summary[:297] + "..."
            lines.extend(
                [
                    f"{index}. Ferramenta escolhida: {step.tool_name}",
                    f"   Razao da escolha: {step.reason}",
                    f"   Resultado resumido: {summary}",
                ]
            )
        return response + "\n".join(lines)


def _desktop_action_plan(user_message: str, text: str) -> list[dict[str, Any]]:
    if not _looks_like_desktop_open_request(text):
        return []

    url = _extract_url(user_message)
    if url:
        return [
            {
                "tool": "open_url",
                "arguments": {"url": url},
                "reason": "O utilizador pediu para abrir um URL permitido.",
            }
        ]

    if "projeto" in text:
        project_name = _extract_project_name(user_message) or ("AssistenteIA" if "assistenteia" in text else "")
        if project_name:
            return [
                {
                    "tool": "open_project",
                    "arguments": {"project_name": project_name},
                    "reason": "O utilizador pediu para abrir um projeto conhecido.",
                }
            ]

    if any(word in text for word in ("pasta", "folder", "diretorio", "directorio")):
        folder = _extract_folder_target(user_message, text)
        if folder:
            return [
                {
                    "tool": "open_folder",
                    "arguments": {"path": folder},
                    "reason": "O utilizador pediu para abrir uma pasta permitida.",
                }
            ]

    file_path = _extract_open_file_target(user_message)
    if file_path:
        return [
            {
                "tool": "open_file",
                "arguments": {"path": file_path},
                "reason": "O utilizador pediu para abrir um ficheiro permitido.",
            }
        ]

    app_name = _extract_application_name(text)
    if app_name:
        return [
            {
                "tool": "open_application",
                "arguments": {"app_name": app_name},
                "reason": "O utilizador pediu para abrir uma aplicacao permitida.",
            }
        ]

    return []


def _looks_like_desktop_open_request(text: str) -> bool:
    return any(phrase in text for phrase in ("abre", "abrir", "abre me", "abre-me"))


def _extract_url(message: str) -> str | None:
    match = re.search(r"https?://[^\s<>\"]+", message, re.IGNORECASE)
    if not match:
        return None
    return match.group(0).rstrip(".,;)")


def _extract_project_name(message: str) -> str | None:
    match = re.search(r"projeto\s+(.+)$", message, re.IGNORECASE)
    if not match:
        return None
    value = _trim_open_target(match.group(1))
    return value or None


def _extract_folder_target(message: str, text: str) -> str | None:
    if "workspace" in text:
        return "workspace"
    match = re.search(r"(?:pasta|folder|diretorio|directorio)\s+(.+)$", message, re.IGNORECASE)
    if not match:
        return None
    return _trim_open_target(match.group(1)) or None


def _extract_open_file_target(message: str) -> str | None:
    readable = _extract_readable_filename(message)
    if readable:
        return readable
    match = re.search(r"([\w ._\-()\\/]+?\.[A-Za-z0-9]{1,8})\b", message)
    if not match:
        return None
    return _trim_open_target(match.group(1)) or None


def _extract_application_name(text: str) -> str | None:
    aliases: list[str] = []
    for key, info in ALLOWED_APPLICATIONS.items():
        aliases.append(key)
        aliases.extend(str(alias) for alias in info.get("aliases", ()))
    for alias in sorted(set(aliases), key=len, reverse=True):
        normalized_alias = _normalize_text(alias)
        if normalized_alias and normalized_alias in text:
            app_key = normalize_app_name(alias)
            return app_key or alias
    return None


def _trim_open_target(value: str) -> str:
    cleaned = value.strip().strip("\"'")
    cleaned = re.sub(r"^(o|a|os|as|chamado|chamada)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip().strip(".")


def _desktop_confirmation_question(tool_name: str, arguments: dict[str, Any]) -> str:
    if tool_name == "open_application":
        return f"Queres que abra {arguments.get('app_name', 'essa aplicacao')}?"
    if tool_name == "open_project":
        return f"Queres que abra o projeto {arguments.get('project_name', '')}?"
    if tool_name == "open_url":
        return f"Queres que abra este URL? {arguments.get('url', '')}"
    if tool_name == "open_folder":
        return f"Queres que abra a pasta '{arguments.get('path', '')}'?"
    if tool_name == "open_file":
        return f"Queres que abra o ficheiro '{arguments.get('path', '')}'?"
    return "Queres que execute esta acao?"


def _looks_like_summary_request(message: str) -> bool:
    text = _normalize_text(message)
    return any(word in text for word in ("resume", "resumir", "sumariza", "sumarizar", "resumo"))


def _looks_like_read_or_summary_request(text: str) -> bool:
    return _looks_like_summary_request(text) or any(
        phrase in text
        for phrase in (
            "le ",
            "ler ",
            "abre ",
            "abrir ",
            "mostra o ficheiro",
            "mostra ficheiro",
            "conteudo do ficheiro",
            "conteudo de",
        )
    )


def _looks_like_list_request(text: str) -> bool:
    list_words = ("lista", "listar", "mostra", "mostrar", "ver", "ve")
    target_words = ("ficheiros", "documentos", "pasta", "workspace")
    return any(word in text for word in list_words) and any(word in text for word in target_words)


def _looks_like_create_request(text: str) -> bool:
    return any(phrase in text for phrase in ("cria", "criar", "novo ficheiro", "guarda"))


def _asks_to_list_and_summarize_first(text: str) -> bool:
    return (
        any(word in text for word in ("lista", "listar", "mostra", "mostrar"))
        and "primeiro" in text
        and _looks_like_summary_request(text)
    )


def _asks_to_read_and_create_note(text: str) -> bool:
    return _looks_like_read_or_summary_request(text) and any(
        phrase in text
        for phrase in (
            "cria uma nota",
            "criar uma nota",
            "cria nota",
            "nota com os pontos",
            "pontos principais",
        )
    )


def _asks_to_find_relevant_documents(text: str) -> bool:
    return any(word in text for word in ("procura", "procurar", "encontra", "pesquisa")) and any(
        word in text for word in ("documentos", "ficheiros")
    )


def _asks_to_analyze_existing_files(text: str) -> bool:
    return any(word in text for word in ("analisa", "analisar")) and any(
        phrase in text
        for phrase in (
            "ficheiros existentes",
            "documentos existentes",
            "sugere uma organizacao",
            "sugerir uma organizacao",
            "organiza",
        )
    )


def _system_state_tool_for(text: str) -> str | None:
    if not _looks_like_system_state_request(text):
        return None

    if any(
        phrase in text
        for phrase in (
            "o que estou a fazer",
            "que estou a fazer",
            "em que estou a trabalhar",
            "no que estou a trabalhar",
            "resume o contexto atual",
            "resumo do contexto atual",
            "contexto atual",
            "qual parece ser a minha atividade principal",
            "atividade principal",
        )
    ):
        return "get_current_activity_summary"

    if "janela ativa" in text or "qual e a janela" in text or "qual janela" in text:
        return "get_active_window"

    if "aplicacao ativa" in text or "programa ativo" in text or "app ativa" in text:
        return "get_active_application"

    if any(phrase in text for phrase in ("atividade", "monitorizacao", "monitorizacao", "programas estou a usar", "que programas estou a usar")):
        return "get_recent_activity"

    if any(phrase in text for phrase in ("que informacao tens", "que informação tens", "estado do computador", "snapshot")):
        return "get_last_context_snapshot"

    if any(word in text for word in ("janelas", "aplicacoes", "apps", "programas")):
        return "get_open_windows"

    return "get_recent_activity"


def _looks_like_system_state_request(text: str) -> bool:
    return any(
        word in text
        for word in (
            "janela",
            "janelas",
            "aplicacao",
            "aplicacoes",
            "app",
            "apps",
            "programa",
            "programas",
            "atividade",
            "monitorizacao",
            "monitorização",
            "snapshot",
            "computador",
            "informacao",
            "informação",
            "contexto",
            "trabalhar",
            "fazer",
        )
    ) and any(
        word in text
        for word in (
            "abertas",
            "abertos",
            "ativa",
            "ativo",
            "detetadas",
            "detectadas",
            "usar",
            "usando",
            "estou",
            "tens",
            "qual",
            "que",
            "estado",
            "snapshot",
            "resume",
            "resumo",
        )
    )


def _looks_like_presence_state_question(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "em que modo estas",
            "qual e o modo atual",
            "modo atual",
            "estado de presenca",
            "em que estado estas",
        )
    )


def _extract_readable_filename(message: str) -> str | None:
    match = re.search(
        r"([\w ._\-()]+?\.(?:txt|md|docx|pdf))",
        message,
        re.IGNORECASE,
    )
    if match:
        return _clean_filename(match.group(1))
    return None


def _extract_create_request(message: str) -> tuple[str | None, str]:
    match = re.search(
        r"(?:chamado|chamada|nome)\s+(.+?\.txt)\b",
        message,
        re.IGNORECASE,
    )
    if match is None:
        match = re.search(r"([^\s\"']+\.txt)\b", message, re.IGNORECASE)
    if not match:
        return None, ""

    filename = _clean_filename(match.group(1))
    after_filename = message[match.end() :].strip()
    normalized_after = _normalize_text(after_filename)
    markers = (
        "com este texto:",
        "com este texto",
        "com o texto:",
        "com o texto",
        "conteudo:",
        "conteudo",
        "texto:",
        "texto",
    )
    for marker in markers:
        normalized_marker = _normalize_text(marker)
        index = normalized_after.find(normalized_marker)
        if index >= 0:
            return filename, after_filename[index + len(marker) :].strip()

    return filename, after_filename


def _clean_filename(filename: str) -> str:
    cleaned = filename.strip().strip("\"'")
    prefixes = (
        "le o ficheiro ",
        "le ficheiro ",
        "ler o ficheiro ",
        "ler ficheiro ",
        "le o documento ",
        "le documento ",
        "um ficheiro chamado ",
        "um ficheiro ",
        "o ficheiro ",
        "ficheiro ",
        "chamado ",
        "chamada ",
        "nome ",
        "documento ",
        "o documento ",
    )
    normalized = _normalize_text(cleaned)
    for prefix in prefixes:
        if normalized.startswith(prefix):
            return cleaned[len(prefix) :].strip()
    return cleaned


def _extract_files_from_listing(observation: str) -> list[str]:
    files: list[str] = []
    for line in observation.splitlines():
        cleaned = line.strip()
        if not cleaned.startswith("- "):
            continue
        filename = cleaned[2:].strip()
        if filename:
            files.append(filename)
    return files


def _is_readable_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in {".txt", ".md", ".docx", ".pdf"}


def _note_filename_for(filename: str) -> str:
    stem = Path(filename).stem or "ficheiro"
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "ficheiro"
    return f"nota_{safe_stem}.txt"


def _format_observations(observations: list[AgentStep]) -> str:
    chunks = []
    for index, step in enumerate(observations, start=1):
        chunks.append(
            "\n".join(
                (
                    f"Observacao {index}",
                    f"Ferramenta: {step.tool_name}",
                    f"Razao: {step.reason}",
                    f"Resultado:\n{step.observation}",
                )
            )
        )
    return "\n\n".join(chunks)


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _context_label(context: AgentContext) -> str:
    if context.active_contexts:
        return ", ".join(context.active_contexts)
    return context.profile_name

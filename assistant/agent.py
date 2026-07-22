from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from assistant.desktop_actions import ALLOWED_APPLICATIONS, app_label, is_application_open, normalize_app_name
from assistant.planner import PlannerResult, plan_user_request
from assistant.text_matching import contains_any_phrase, find_evidence_span, find_near_pair_span
from assistant.tool_registry import ToolRegistry


MAX_AGENT_STEPS = 5
MAX_OBSERVATION_CHARS_FOR_SUMMARY = 12000
CONFIRMATION_YES = {
    "sim",
    "s",
    "ok",
    "okay",
    "claro",
    "confirmo",
    "confirma",
    "pode ser",
    "executa",
    "abre",
    "podes criar",
    "podes abrir",
}
CONFIRMATION_NO = {
    "nao",
    "n",
    "cancelar",
    "cancela",
    "esquece",
    "deixa estar",
}
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
    tools_used: tuple[str, ...] = ()
    # Internal reasoning trace (chosen tool, reason, summarized observation) —
    # NEVER part of `response`. Only ever surfaces via debug stdout / logs /
    # TurnResult / eval reports (see AssistantEngine._record_agent_debug_trace),
    # never through responseReady to the UI, even with ECHO_DEBUG_* enabled.
    debug_trace: str = ""


@dataclass(frozen=True)
class AgentStep:
    tool_name: str
    arguments: dict[str, Any]
    reason: str
    observation: str


@dataclass(frozen=True)
class PendingActionInterpretation:
    decision: str
    follow_up_intent: str = ""
    cleaned_user_request: str = ""
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class AgentContext:
    system_prompt: str
    history: list[dict[str, str]]
    profile_name: str = "AUTO_CONTEXT"
    active_contexts: list[str] | None = None
    context_debug: str = ""
    recurring_context: str = ""
    pending_tasks: str = ""
    session_summary: str = ""
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
        desktop_config: dict[str, Any] | None = None,
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
        self.desktop_config = desktop_config or {}
        self.desktop_action_runner = desktop_action_runner
        self.debug_agent = debug_agent
        self.max_steps = max_steps
        self.pending_confirmation: dict[str, Any] | None = None
        self._current_context: AgentContext | None = None
        self.last_planner_result: PlannerResult | None = None

    def run(self, user_message: str, context: AgentContext) -> AgentResult:
        if self.pending_confirmation is not None:
            return self._handle_pending_confirmation(user_message)

        self._current_context = context
        normalized_message = _normalize_text(user_message)
        tools_description = self.tools.describe()
        system_prompt = self._system_prompt_with_tools(context, tools_description)
        planner_result = self._plan_request(user_message, context, tools_description)
        self.last_planner_result = planner_result
        if not context.tools_enabled and _looks_like_system_state_request(normalized_message):
            return AgentResult(MISSING_MONITORING_TOOL_MESSAGE)

        planner_action_response = self._try_planner_recommended_action(planner_result, context)
        if planner_action_response is not None:
            return planner_action_response

        planner_direct_result = self._planner_direct_response(planner_result)
        if planner_direct_result is not None:
            return planner_direct_result

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
            _mark_llm_source(self.llm, "AGENT_FINAL_RESPONSE")
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
                return AgentResult(
                    response=already_open_response,
                    remember=True,
                    tools_used=(tool_name,),
                    debug_trace=self._debug_block([step]),
                )

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
                _mark_llm_source(self.llm, "AGENT_FINAL_RESPONSE")
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
            _mark_llm_source(self.llm, "AGENT_FINAL_RESPONSE")
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
        return AgentResult(
            response=final_response,
            remember=True,
            tools_used=tuple(step.tool_name for step in observations),
            debug_trace=self._debug_block(observations),
        )

    def _plan_request(self, user_message: str, context: AgentContext, tools_description: str) -> PlannerResult:
        return plan_user_request(
            user_message=user_message,
            computer_context=context.context_debug,
            pending_tasks=context.pending_tasks,
            relevant_memory=context.recurring_context,
            recent_timeline=context.recurring_context,
            last_session_summary=context.session_summary,
            tools_available=tools_description,
            presence_state=context.presence_state,
        )

    def _try_planner_recommended_action(
        self,
        planner_result: PlannerResult,
        context: AgentContext,
    ) -> AgentResult | None:
        if not planner_result.recommended_actions:
            return None
        if not context.tools_enabled:
            return AgentResult("Neste modo nao posso usar ferramentas para preparar isso.")
        if not context.confirmations_enabled:
            return AgentResult(
                "Tenho um plano, mas nao posso executar acoes neste estado de presenca. "
                "Muda para ACTIVE_CONVERSATION para poderes confirmar."
            )

        action = planner_result.recommended_actions[0]
        if action.tool_name == "open_url" and "example.com" in str(action.arguments.get("url") or "").lower():
            return AgentResult(planner_result.direct_response or "Percebo. Continuamos por aqui.", remember=True)
        if action.tool_name in CONFIRMATION_REQUIRED_TOOLS:
            result = self._ask_desktop_action_confirmation(
                action.tool_name,
                dict(action.arguments),
                f"Planner: {planner_result.intent}. {action.label}",
                [],
            )
            response = result.response
            if planner_result.direct_response:
                response = f"{planner_result.direct_response}\n\n{response}"
            debug_trace = result.debug_trace
            if self.debug_agent:
                planner_debug = f"[DEBUG_AGENT]\n{planner_result.debug_text()}"
                debug_trace = f"{debug_trace}\n\n{planner_debug}" if debug_trace else planner_debug
            return AgentResult(response=response, remember=result.remember, tools_used=result.tools_used, debug_trace=debug_trace)
        return None

    def _planner_direct_response(self, planner_result: PlannerResult) -> AgentResult | None:
        if planner_result.is_normal_conversation or not planner_result.direct_response:
            return None
        debug_trace = f"[DEBUG_AGENT]\n{planner_result.debug_text()}" if self.debug_agent else ""
        return AgentResult(response=planner_result.direct_response, remember=True, debug_trace=debug_trace)

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
            return AgentResult(response=already_open_response, remember=True, debug_trace=self._debug_block([step]))
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

        if _is_vague_technical_complaint(_normalize_text(user_message)):
            # Falha 3 (ferro/erro follow-up): "Tenho um erro no programa." has
            # nothing concrete yet for any tool to act on — asking the model
            # to pick a tool here just burns a call before it says null
            # anyway. A message that also names something concrete (a file,
            # an app, a project) or uses a broader technical marker
            # (arquitetura, refatorar, ...) still goes through choose_tool as
            # before — see test_complex_request_still_reaches_llm_path.
            return []

        decision = self.llm.choose_tool(
            user_message,
            tools_description,
            profile_name=profile_name,
        )
        if decision.get("tool"):
            tool_name = str(decision.get("tool") or "")
            arguments = decision.get("arguments", {})
            if tool_name in CONFIRMATION_REQUIRED_TOOLS and not _allows_desktop_action_from_message(
                tool_name,
                arguments if isinstance(arguments, dict) else {},
                user_message,
            ):
                return []
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

        desktop_action_plan = _desktop_action_plan(user_message, text, self.desktop_config)
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
            _mark_llm_source(self.llm, "AGENT_FINAL_RESPONSE")
            return self.llm.chat(prompt, history=context.history, system_prompt=system_prompt)

        if _asks_to_find_relevant_documents(text):
            prompt = (
                "Com base nos ficheiros e conteudos observados, indica quais parecem relevantes para o tema pedido. "
                "Justifica de forma curta e nao inventes conteudo que nao apareca nas observacoes.\n\n"
                f"Pedido do utilizador: {user_message}\n\n{observation_text}"
            )
            _mark_llm_source(self.llm, "AGENT_FINAL_RESPONSE")
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
            _mark_llm_source(self.llm, "AGENT_FINAL_RESPONSE")
            return self.llm.chat(prompt, history=context.history, system_prompt=system_prompt)

        if len(observations) == 1:
            return observations[0].observation

        prompt = (
            "Responde ao utilizador com base apenas nas observacoes das ferramentas. "
            "Apresenta so a resposta final, sem mostrar plano interno.\n\n"
            f"Pedido do utilizador: {user_message}\n\n{observation_text}"
        )
        _mark_llm_source(self.llm, "AGENT_FINAL_RESPONSE")
        return self.llm.chat(prompt, history=context.history, system_prompt=system_prompt)

    def _make_note_content(self, user_message: str, observations: list[AgentStep]) -> str:
        prompt = (
            "Cria uma nota curta em portugues de Portugal com os pontos principais do conteudo observado. "
            "Devolve apenas o texto da nota.\n\n"
            f"Pedido do utilizador: {user_message}\n\n{_format_observations(observations)}"
        )
        _mark_llm_source(self.llm, "AGENT_FINAL_RESPONSE")
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
        debug_trace = ""
        if self.debug_agent:
            debug_trace = (
                "[DEBUG_AGENT]\n"
                "Passo pendente: create_workspace_file\n"
                f"Razao da escolha: {reason}\n"
                "Resultado resumido: a criacao aguarda confirmacao."
            )
        return AgentResult(response=response, remember=True, debug_trace=debug_trace)

    def _already_open_response(self, tool_name: str, arguments: dict[str, Any]) -> str | None:
        if tool_name == "open_application":
            app_key = normalize_app_name(str(arguments.get("app_name", "")))
            if app_key and is_application_open(app_key, self.context_observer):
                label = app_label(app_key)
                self.pending_confirmation = {
                    "tool": "open_application",
                    "arguments": {"app_name": app_key, "focus_existing": True},
                    "reason": f"O {label} ja esta aberto; o utilizador pode querer traze-lo para a frente.",
                    "previous_observations": [],
                }
                return f"O {label} ja esta aberto. Queres que o traga para a frente?\nResponde 'sim' para executar ou 'nao' para cancelar."
        if tool_name == "open_project" and is_application_open("vscode", self.context_observer):
            self.pending_confirmation = {
                "tool": "open_application",
                "arguments": {"app_name": "vscode", "focus_existing": True},
                "reason": "O VS Code ja esta aberto; o utilizador pode querer traze-lo para a frente.",
                "previous_observations": [],
            }
            return "O VS Code ja esta aberto. Queres que o traga para a frente?\nResponde 'sim' para executar ou 'nao' para cancelar."
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
        debug_trace = ""
        if self.debug_agent:
            debug_trace = (
                "[DEBUG_AGENT]\n"
                f"Passo pendente: {tool_name}\n"
                f"Razao da escolha: {reason}\n"
                "Resultado resumido: a acao aguarda confirmacao."
            )
        return AgentResult(response=response, remember=True, debug_trace=debug_trace)

    def _handle_pending_confirmation(self, user_message: str) -> AgentResult:
        pending = self.pending_confirmation or {}
        interpretation = interpret_pending_action_response(user_message, pending)
        self._debug_pending_action(user_message, interpretation, "PENDING_CONFIRMATION")

        if interpretation.decision == "cancel":
            self.pending_confirmation = None
            self._debug_pending_action(user_message, interpretation, "CANCELLED")
            response = _cancelled_pending_response(pending, interpretation)
            return AgentResult(response)

        if interpretation.decision != "confirm":
            self._debug_pending_action(user_message, interpretation, "PENDING_CONFIRMATION")
            return AgentResult(
                "Ainda não executei nada. Queres que avance com essa ação ou preferes cancelar?"
            )

        pending = self.pending_confirmation
        self.pending_confirmation = None
        if pending is None:
            return AgentResult("Não há nenhuma ação pendente para confirmar.")

        tool = self.tools.get(str(pending["tool"]))
        if tool is None:
            return AgentResult("Não consegui concluir a ação pendente: a ferramenta não está disponível.")

        arguments = self._prepare_tool_arguments(pending.get("arguments", {}))
        observation = tool.run(arguments)
        response = observation
        self._debug_pending_action(user_message, interpretation, "COMPLETED")
        debug_trace = ""
        if self.debug_agent:
            step = AgentStep(
                tool_name=str(pending["tool"]),
                arguments=arguments,
                reason=str(pending.get("reason") or "Confirmação do utilizador."),
                observation=observation,
            )
            debug_trace = self._debug_block([step])
        return AgentResult(
            response=response, remember=tool.remember_result, tools_used=(str(pending["tool"]),), debug_trace=debug_trace
        )

    def _debug_pending_action(
        self,
        user_message: str,
        interpretation: PendingActionInterpretation,
        final_state: str,
    ) -> None:
        if not self.debug_agent:
            return
        pending = self.pending_confirmation or {}
        print(
            "[AssistenteIA DEBUG_AGENT] pending_action | "
            f"estado_final={final_state} | "
            f"acao_anterior={pending.get('tool') or '(nenhuma)'} | "
            f"mensagem={user_message!r} | "
            f"decisao={interpretation.decision} | "
            f"intencao={interpretation.follow_up_intent or '(nenhuma)'} | "
            f"pedido_limpo={interpretation.cleaned_user_request or '(vazio)'} | "
            f"confianca={interpretation.confidence:.2f} | "
            f"razao={interpretation.reason}"
        )

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
            f"Estado de presenca: {context.presence_state}.\n"
            "Responde apenas ao pedido atual, sem anunciar capacidades internas. "
            "Nao menciones ferramentas, workspace, ficheiros, aplicacoes ou monitorizacao "
            "a menos que o utilizador tenha pedido isso explicitamente.\n"
            "Nunca inventes informacoes sobre o estado do computador, janelas, aplicacoes, programas "
            "ou atividade recente. Se essa informacao nao vier do Context Observer, admite desconhecimento."
        )
        if context.recurring_context:
            prompt += f"\n\nMemoria permanente relevante:\n{context.recurring_context}"
        return prompt

    def _debug_block(self, observations: list[AgentStep]) -> str:
        """Internal reasoning trace for these observations — a separate
        string, never appended to the response shown to the user (see
        AgentResult.debug_trace)."""
        if not self.debug_agent:
            return ""

        lines = ["[DEBUG_AGENT]", "Passos internos:"]
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
        return "\n".join(lines)


def _desktop_action_plan(user_message: str, text: str, desktop_config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if not _looks_like_desktop_open_request(text):
        return []
    config = desktop_config or {}

    url = _extract_url(user_message)
    if url:
        return [
            {
                "tool": "open_url",
                "arguments": {"url": url},
                "reason": "O utilizador pediu para abrir um URL permitido.",
            }
        ]

    email_plan = _email_alias_plan(text, config)
    if email_plan:
        return email_plan

    browser_plan = _browser_alias_plan(text, config)
    if browser_plan:
        return browser_plan

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


def _allows_desktop_action_from_message(tool_name: str, arguments: dict[str, Any], user_message: str) -> bool:
    if tool_name not in {"open_application", "open_file", "open_folder", "open_url", "open_project"}:
        return True
    text = _normalize_text(user_message)
    if not text:
        return False

    explicit_action = any(
        phrase in text
        for phrase in (
            "abre",
            "abrir",
            "abre-me",
            "abre me",
            "pesquisa",
            "pesquisar",
            "procura",
            "procurar",
            "consulta",
            "consultar",
            "vai a",
            "vai ao",
        )
    )
    if not explicit_action:
        return False

    if tool_name == "open_url":
        url = str(arguments.get("url") or "")
        if "example.com" in url.lower():
            return False
    return True


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
    if any(phrase in text for phrase in ("documentos", "documents", "os documentos")):
        return "documentos"
    if any(phrase in text for phrase in ("downloads", "transferencias", "transferencias")):
        return "downloads"
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
        app_key = normalize_app_name(str(arguments.get("app_name", "")))
        label = app_label(app_key) if app_key else str(arguments.get("app_name", "essa aplicacao"))
        if arguments.get("focus_existing"):
            return f"Queres que traga o {label} para a frente?"
        return f"Queres que abra o {label}?"
    if tool_name == "open_project":
        return f"Queres que abra o projeto {arguments.get('project_name', '')}?"
    if tool_name == "open_url":
        if arguments.get("display_name"):
            return f"Queres que abra o {arguments.get('display_name')}?"
        if arguments.get("search_engine") and arguments.get("search_query"):
            return f"Queres que pesquise no {arguments.get('search_engine')} por: {arguments.get('search_query')}?"
        return f"Queres que abra este URL? {arguments.get('url', '')}"
    if tool_name == "open_folder":
        return f"Queres que abra a pasta '{arguments.get('path', '')}'?"
    if tool_name == "open_file":
        return f"Queres que abra o ficheiro '{arguments.get('path', '')}'?"
    return "Queres que execute esta acao?"


def _email_alias_plan(text: str, desktop_config: dict[str, Any]) -> list[dict[str, Any]]:
    if not any(phrase in text for phrase in ("mail", "email", "correio", "gmail")):
        return []
    if "gmail" in text or _normalize_text(str(desktop_config.get("default_email", "outlook"))) == "gmail":
        return [
            {
                "tool": "open_url",
                "arguments": {"url": "https://mail.google.com", "display_name": "Gmail"},
                "reason": "O utilizador pediu para abrir o email e o Gmail esta configurado como email predefinido.",
            }
        ]
    return [
        {
            "tool": "open_application",
            "arguments": {"app_name": "outlook"},
            "reason": "O utilizador pediu para abrir o email e o Outlook esta configurado como email predefinido.",
        }
    ]


def _browser_alias_plan(text: str, desktop_config: dict[str, Any]) -> list[dict[str, Any]]:
    if not any(phrase in text for phrase in ("browser", "navegador")):
        return []
    default_browser = _normalize_text(str(desktop_config.get("default_browser", "chrome"))) or "chrome"
    app_name = normalize_app_name(default_browser) or "chrome"
    return [
        {
            "tool": "open_application",
            "arguments": {"app_name": app_name},
            "reason": "O utilizador pediu para abrir o browser predefinido.",
        }
    ]


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


# Explicit, per-tool evidence phrases (Part 3: validate tool intent before
# executing a system-state tool). Deliberately NOT a generic "topic word AND
# action word" gate — that pattern let ordinary words like "que"/"fazer"
# fire a get_recent_activity call on an unrelated sentence ("Quero saber
# quanto estao dispostos a aumentar... vou fazer braco de ferro": "que" is a
# substring of "Quero", and "fazer" alone is too weak a signal on its own).
# Every phrase below is matched with word-boundary safety via
# assistant.text_matching, so "erro" never matches inside "ferro" either.
# A message that doesn't hit one of these specific phrases gets no tool at
# all — better to fall through to general conversation than to guess.
_ACTIVITY_SUMMARY_MARKERS = (
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
_ACTIVE_WINDOW_MARKERS = ("janela ativa", "qual e a janela", "qual janela")
_ACTIVE_APPLICATION_MARKERS = ("aplicacao ativa", "programa ativo", "app ativa")
_RECENT_ACTIVITY_MARKERS = (
    "o que estive a fazer",
    "atividade recente",
    "que programas usei",
    "estado do computador",
    "em que projeto trabalhei",
    "em que projeto estive a trabalhar",
    "o que tenho aberto",
    "que aplicacoes estou a usar",
    "que aplicacao estou a usar",
    "programas estou a usar",
    "que programas estou a usar",
)
_CONTEXT_SNAPSHOT_MARKERS = ("que informacao tens", "snapshot")
_OPEN_WINDOWS_MARKERS = ("janelas abertas", "aplicacoes abertas", "apps abertos", "programas abertos")
# "Que janelas tens detetadas?" / "Que aplicações estão abertas?" put a verb
# between the subject and the state word, so an exact adjacent-phrase check
# would miss them — a bounded proximity match (still word-boundary-safe,
# still narrow: only these specific nouns near only these specific state
# words) covers that without going back to the old "any topic word anywhere
# + any action word anywhere" gate that caused the ferro/erro bug.
_OPEN_WINDOWS_SUBJECTS = ("janela", "janelas", "aplicacao", "aplicacoes", "app", "apps", "programa", "programas")
_OPEN_WINDOWS_STATES = ("abertas", "abertos", "detetadas", "detectadas")

# Order matters: checked top to bottom, first match wins — most specific
# markers (a named tool's exact phrasing) before the more generic ones.
# get_last_context_snapshot is checked before get_recent_activity so an
# explicit "snapshot" mention wins over the broader "estado do computador"
# marker (both can appear in the same sentence, e.g. "mostra o snapshot do
# estado do computador").
_SYSTEM_STATE_TOOL_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("get_current_activity_summary", _ACTIVITY_SUMMARY_MARKERS),
    ("get_active_window", _ACTIVE_WINDOW_MARKERS),
    ("get_active_application", _ACTIVE_APPLICATION_MARKERS),
    ("get_last_context_snapshot", _CONTEXT_SNAPSHOT_MARKERS),
    ("get_recent_activity", _RECENT_ACTIVITY_MARKERS),
)


_VAGUE_TECHNICAL_COMPLAINT_MARKERS = ("erro", "bug", "problema")
_CONCRETE_TOOL_OBJECT_MARKERS = ("ficheiro", "ficheiros", "pasta", "pastas", "documento", "documentos", "workspace", "projeto")
_FILENAME_PATTERN = re.compile(r"\.(txt|md|docx|pdf|py|ps1)\b")


def _is_vague_technical_complaint(text: str) -> bool:
    """A bare "Tenho um erro no programa." has nothing concrete yet for any
    tool to act on — no filename, no app, no project, no activity marker.
    Broader technical markers (arquitetura, refatorar, código, python,
    assistente, ...) are NOT included here on purpose: those tend to come
    with a substantive request (e.g. "ajuda-me a pensar numa arquitetura
    melhor") where consulting available tools is still reasonable.
    """
    if not contains_any_phrase(text, _VAGUE_TECHNICAL_COMPLAINT_MARKERS):
        return False
    if _system_state_tool_for(text) is not None:
        return False
    if _FILENAME_PATTERN.search(text):
        return False
    return not contains_any_phrase(text, _CONCRETE_TOOL_OBJECT_MARKERS)


def _system_state_tool_for(text: str) -> str | None:
    if find_near_pair_span(text, _OPEN_WINDOWS_SUBJECTS, _OPEN_WINDOWS_STATES):
        return "get_open_windows"
    for tool_name, markers in _SYSTEM_STATE_TOOL_MARKERS:
        if contains_any_phrase(text, markers):
            return tool_name
    return None


def _system_state_evidence_span(text: str) -> str:
    """The literal marker phrase that justified a system-state tool call, or
    "" if none matched — see tool_intent_evidence_span in AgentResult."""
    open_windows_span = find_near_pair_span(text, _OPEN_WINDOWS_SUBJECTS, _OPEN_WINDOWS_STATES)
    if open_windows_span:
        return open_windows_span
    for _tool_name, markers in _SYSTEM_STATE_TOOL_MARKERS:
        span = find_evidence_span(text, markers)
        if span:
            return span
    return ""


def _looks_like_system_state_request(text: str) -> bool:
    return _system_state_tool_for(text) is not None


def system_state_tool_intent(normalized_text: str) -> tuple[bool, str, float]:
    """Public read of the same evidence check _deterministic_plan uses to
    decide whether a system-state tool (get_recent_activity and friends) may
    run for this message. Returns (supported, evidence_span, confidence) —
    confidence is binary (1.0/0.0) because this is a deterministic phrase
    match, not a model judgement. Used by conversation.py to expose
    tool_intent_supported_by_current_message / tool_intent_evidence_span /
    tool_selection_confidence in turn telemetry regardless of which path
    (deterministic plan or early tools-disabled check) actually handled the
    turn.
    """
    tool_name = _system_state_tool_for(normalized_text)
    if tool_name is None:
        return False, "", 0.0
    return True, _system_state_evidence_span(normalized_text), 1.0


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


def _normalize_confirmation_response(text: str) -> str:
    normalized = _normalize_text(text).strip()
    normalized = re.sub(r"^[\s\"'`]+|[\s\"'`]+$", "", normalized)
    normalized = re.sub(r"[.!?,;:]+$", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def interpret_pending_action_response(
    message: str,
    pending_action: dict[str, Any] | None = None,
) -> PendingActionInterpretation:
    normalized = _normalize_confirmation_response(message)
    pending_action = pending_action or {}
    if not normalized:
        return PendingActionInterpretation("unclear", confidence=0.2, reason="mensagem vazia")

    if normalized in CONFIRMATION_YES:
        return PendingActionInterpretation("confirm", confidence=1.0, reason="confirmação direta")
    if normalized in CONFIRMATION_NO:
        return PendingActionInterpretation("cancel", confidence=1.0, reason="cancelamento direto")

    cancel_patterns = (
        r"\bnao\b.*\b(?:abras|abrir|executes|executar|pesquisar|pesquises|procures|browser|google|url|nada)\b",
        r"\bnao\b.*\b(?:preciso|quero|e preciso|é preciso)\b.*\b(?:abras|abrir|pesquisar|pesquises|browser|google|url|nada)\b",
        r"\b(?:nao abras|não abras|nao executes|não executes|nao pesquises|não pesquises|nao procures|não procures)\b",
        r"\b(?:esquece isso|cancela isso|cancela e volta|deixa estar|prefiro que me ajudes aqui)\b",
        r"\b(?:quero so falar contigo|quero só falar contigo|quero a tua ajuda apenas|ajuda-me aqui|ajuda me aqui)\b",
        r"\b(?:nao e preciso pesquisar|não é preciso pesquisar|nao quero que abras|não quero que abras)\b",
        r"\b(?:isso nao foi o que pedi|isso não foi o que pedi)\b",
    )
    if any(re.search(pattern, normalized) for pattern in cancel_patterns):
        cleaned = _clean_request_after_cancellation(message)
        return PendingActionInterpretation(
            decision="cancel",
            follow_up_intent=_follow_up_intent_for_cancel(cleaned),
            cleaned_user_request=cleaned,
            confidence=0.95,
            reason="frase contém recusa semântica da ação pendente",
        )

    confirm_patterns = (
        r"\b(?:sim|ok|okay|claro|pode ser|confirma|confirmo)\b.*\b(?:abre|executa|avanca|avança|faz)\b",
        r"\b(?:podes|pode)\s+(?:abrir|executar|avancar|avançar|fazer)\b",
    )
    if any(re.search(pattern, normalized) for pattern in confirm_patterns):
        return PendingActionInterpretation("confirm", confidence=0.9, reason="frase contém confirmação semântica")

    if _looks_like_new_conversation_turn(normalized):
        return PendingActionInterpretation(
            decision="cancel",
            follow_up_intent="new_conversation_turn",
            cleaned_user_request=message.strip(),
            confidence=0.75,
            reason="mensagem parece iniciar outro assunto",
        )

    return PendingActionInterpretation("unclear", confidence=0.4, reason="não consegui confirmar nem cancelar")


def _clean_request_after_cancellation(message: str) -> str:
    cleaned = message.strip()
    cleaned = re.sub(
        r"(?i)\b(n[aã]o preciso que abras nada(?: no google)?|n[aã]o abras nada|n[aã]o quero que abras o browser|n[aã]o [ée] preciso pesquisar|cancela(?: isso)?|esquece isso)\b",
        "",
        cleaned,
    )
    cleaned = re.sub(r"(?i)\b(quero|prefiro)\s+(?:s[oó]\s+)?", "", cleaned, count=1)
    cleaned = cleaned.replace(",", " ").replace(";", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" .")


def _follow_up_intent_for_cancel(cleaned_request: str) -> str:
    text = _normalize_text(cleaned_request)
    if any(phrase in text for phrase in ("ajuda", "a tua ajuda", "aqui", "falar contigo")):
        return "continue_conversation_without_tool"
    if cleaned_request:
        return "continue_with_cleaned_request"
    return "cancel_only"


def _looks_like_new_conversation_turn(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "preciso de ajuda",
            "quero falar",
            "vamos voltar",
            "podemos voltar",
            "outra coisa",
            "muda de assunto",
        )
    )


def _cancelled_pending_response(
    pending_action: dict[str, Any],
    interpretation: PendingActionInterpretation,
) -> str:
    tool_name = str(pending_action.get("tool") or "")
    if interpretation.follow_up_intent == "continue_conversation_without_tool":
        return "Claro. Não abro nada. Continuamos por aqui."
    if interpretation.follow_up_intent == "new_conversation_turn":
        return "Claro. Cancelo isso. Diz-me."
    if tool_name == "create_workspace_file":
        return "Ação cancelada. Não criei nenhum ficheiro."
    return "Ação cancelada. Não executei nada."


def _context_label(context: AgentContext) -> str:
    if context.active_contexts:
        return ", ".join(context.active_contexts)
    return context.profile_name


def _mark_llm_source(llm: Any, source: str) -> None:
    marker = getattr(llm, "mark_next_call_source", None)
    if callable(marker):
        marker(source)
    else:
        try:
            setattr(llm, "_next_call_source", source)
        except Exception:
            pass

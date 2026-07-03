from __future__ import annotations

import re
import time
import unicodedata
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from assistant.agent import Agent, AgentContext
from assistant.briefing import get_last_active_project, summarize_yesterday
from assistant.context_manager import ActiveContext, ContextManager
from assistant.delegation import DelegationManager, DelegationTarget
from assistant.fast_router import route_fast_command
from assistant.long_term_memory import LongTermMemory, MemoryCategory
from assistant.memory import ConversationMemory
from assistant.personal_assistant import (
    generate_daily_briefing,
    generate_session_resume,
    generate_task_summary,
)
from assistant.presence_manager import PresenceManager, PresenceState
from assistant.proactive_suggestions import next_proactive_suggestion
from assistant.security import check_user_request
from assistant.tool_registry import ToolRegistry
from assistant.tools import (
    cancel_task as cancel_task_tool,
    complete_task as complete_task_tool,
    list_pending_tasks as list_pending_tasks_tool,
    postpone_task as postpone_task_tool,
)
from assistant.voice_input import MicrophoneCheckError, check_microphone, voice_status_report
from assistant.workspace import WorkspaceGuard


if TYPE_CHECKING:
    from assistant.context_observer import ContextObserver
    from assistant.llm import OllamaClient


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
        active_profile_name: str = "Geral",
        debug: bool = False,
        debug_agent: bool = False,
        debug_performance: bool = False,
        presence_manager: PresenceManager | None = None,
        context_observer: ContextObserver | None = None,
        known_projects: dict[str, str] | None = None,
        desktop_action_runner=None,
        voice_enabled: bool = False,
        voice_missing_dependencies: list[str] | None = None,
        voice_microphone_ok: bool = False,
        voice_microphone_message: str = "",
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.long_term_memory = long_term_memory
        self.tools = tools
        self.workspace = WorkspaceGuard(workspace_path)
        self.workspace_path = self.workspace.resolve()
        self.base_system_prompt = base_system_prompt
        self.active_profile_name = active_profile_name
        self.debug = debug
        self.debug_performance = debug_performance
        self.presence = presence_manager or PresenceManager()
        self.context_observer = context_observer
        self.context_manager = ContextManager()
        self.active_contexts: list[ActiveContext] = []
        self.last_context_debug = ""
        self.delegation = DelegationManager()
        self.language_base = "pt-PT"
        self.current_language = self._load_language_preference()
        self.voice_enabled = voice_enabled
        self.voice_missing_dependencies = voice_missing_dependencies or []
        self.voice_microphone_ok = voice_microphone_ok
        self.voice_microphone_message = voice_microphone_message
        self._ensure_language_preferences()
        self.agent = Agent(
            llm=llm,
            tools=tools,
            workspace_path=self.workspace_path,
            context_observer=context_observer,
            presence_manager=self.presence,
            long_term_memory=long_term_memory,
            known_projects=known_projects,
            desktop_action_runner=desktop_action_runner,
            debug_agent=debug_agent,
        )

    def set_profile(self, profile_name: str, system_prompt: str) -> None:
        """Deprecated compatibility hook; contexts are now selected automatically."""
        self.active_profile_name = profile_name
        self.base_system_prompt = system_prompt
        self.llm.system_prompt = system_prompt
        self._debug_log("Pedido de perfil ignorado: os contextos sao automaticos.")

    def set_presence_state(self, state: str) -> None:
        self.presence.set_state(state)
        self._debug_log(f"Estado de presenca alterado para: {self.presence.state.value}")

    def presence_state(self) -> str:
        return self.presence.state.value

    def respond(self, user_message: str) -> str:
        request_started_at = time.perf_counter()
        self._perf_log("pedido recebido", request_started_at, request_started_at)

        presence_mode_response = self._try_presence_mode_command(user_message)
        if presence_mode_response is not None:
            self._remember_pair(user_message, presence_mode_response)
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return presence_mode_response

        if not self.presence.can_respond():
            response = self._presence_silent_response()
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return response

        fast_started_at = time.perf_counter()
        fast_response = self._try_fast_route(user_message)
        self._perf_log("router rapido", fast_started_at, time.perf_counter())
        if fast_response is not None:
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return fast_response

        security = check_user_request(user_message)
        if not security.allowed:
            response = security.message or "Nao posso realizar essa acao por motivos de seguranca."
            self._remember_pair(user_message, response)
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return response

        active_contexts = self._activate_contexts(user_message)

        language_response = self._try_language_preference_command(user_message)
        if language_response is not None:
            self._remember_pair(user_message, language_response)
            return language_response

        presence_question_response = self._try_presence_question(user_message)
        if presence_question_response is not None:
            self._remember_pair(user_message, presence_question_response)
            return presence_question_response

        voice_response = self._try_voice_question(user_message)
        if voice_response is not None:
            self._remember_pair(user_message, voice_response)
            return voice_response

        context_question_response = self._try_context_question(user_message)
        if context_question_response is not None:
            self._remember_pair(user_message, context_question_response)
            return context_question_response

        briefing_response = self._try_briefing_question(user_message)
        if briefing_response is not None:
            self._remember_pair(user_message, briefing_response)
            return briefing_response

        proactive_response = self._try_proactive_suggestion_question(user_message)
        if proactive_response is not None:
            self._remember_pair(user_message, proactive_response)
            return proactive_response

        # Local memory questions are answered before tool routing so the LLM
        # cannot confuse "lembras-te da pasta?" with "lista a pasta".
        if self.presence.can_store_memory():
            task_response = self._try_task_command(user_message)
            if task_response is not None:
                self._remember_pair(user_message, task_response)
                return task_response

            timeline_response = self._try_timeline_command(user_message)
            if timeline_response is not None:
                self._remember_pair(user_message, timeline_response)
                return timeline_response

            memory_command_response = self._try_long_term_memory_command(user_message)
            if memory_command_response is not None:
                self._remember_pair(user_message, memory_command_response)
                return memory_command_response

            profile_response = self._try_profile_memory(user_message)
            if profile_response is not None:
                self._remember_pair(user_message, profile_response)
                return profile_response

            conversation_memory_response = self._try_conversation_memory_question(user_message)
            if conversation_memory_response is not None:
                self._remember_pair(user_message, conversation_memory_response)
                return conversation_memory_response

        history = self.memory.load() if self.presence.can_store_memory() else []
        recurring_context = self._context_for_agent(user_message)
        delegation_response = self._try_delegation(user_message, recurring_context)
        if delegation_response is not None:
            self._remember_pair(user_message, delegation_response)
            return delegation_response

        llm_started_at = time.perf_counter()
        result = self.agent.run(
            user_message=user_message,
            context=AgentContext(
                system_prompt=self._system_prompt_for_agent(active_contexts),
                history=history,
                active_contexts=[item.name.value for item in active_contexts],
                context_debug=self.last_context_debug,
                recurring_context=recurring_context,
                pending_tasks=self._pending_tasks_for_agent(),
                presence_state=self.presence.state.value,
                tools_enabled=self.presence.can_use_tools(),
                suggestions_enabled=self.presence.can_make_suggestions(),
                confirmations_enabled=self.presence.can_ask_confirmation(),
            ),
        )
        self._perf_log("agent/LLM quando necessario", llm_started_at, time.perf_counter())
        if result.remember:
            self._remember_pair(user_message, result.response)
        self._perf_log("resposta total", request_started_at, time.perf_counter())
        return result.response

    def _load_language_preference(self) -> str:
        getter = getattr(self.long_term_memory, "get_preference", None)
        if getter is None:
            return "pt-PT"
        value = getter("idioma_atual", "pt-PT")
        return value if value in {"pt-PT", "en"} else "pt-PT"

    def _ensure_language_preferences(self) -> None:
        setter = getattr(self.long_term_memory, "set_preference", None)
        getter = getattr(self.long_term_memory, "get_preference", None)
        if setter is None or getter is None:
            return
        if not getter("idioma_base", ""):
            setter("idioma_base", self.language_base)
        if not getter("idioma_atual", ""):
            setter("idioma_atual", self.current_language)

    def _set_current_language(self, language: str) -> None:
        self.current_language = language
        setter = getattr(self.long_term_memory, "set_preference", None)
        if setter is not None:
            setter("idioma_atual", language)

    def _system_prompt_for_agent(self, active_contexts: list[ActiveContext]) -> str:
        return (
            f"{self.context_manager.system_prompt(active_contexts)}\n\n"
            f"{self._language_instruction()}"
        )

    def _language_instruction(self) -> str:
        if self.current_language == "en":
            current = (
                "O idioma atual é inglês porque o utilizador pediu. "
                "Se o utilizador pedir português, volta imediatamente a pt-PT."
            )
        else:
            current = "O idioma atual é português de Portugal."
        return (
            "Preferências de idioma:\n"
            "- idioma_base = pt-PT.\n"
            f"- idioma_atual = {self.current_language}.\n"
            f"- {current}\n"
            "- Vocabulário pt-PT obrigatório quando responderes em português: "
            "aplicações, ecrã, acompanhar/observar, ficheiros, aceder.\n"
            "- Evita português do Brasil: aplicativos, tela, assistindo, arquivos, acessar."
        )

    def _try_delegation(self, user_message: str, context: str) -> str | None:
        decision = self.delegation.decide(
            user_message=user_message,
            profile_name=", ".join(item.name.value for item in self.active_contexts),
            context=context,
        )
        if decision.target == DelegationTarget.LOCAL:
            return None
        return self.delegation.format_response(decision)

    def clear_history(self) -> None:
        self.memory.clear()

    def history(self) -> list[dict[str, str]]:
        return self.memory.load()

    def pending_tasks_summary(self) -> str:
        task_panel_summary = getattr(self.long_term_memory, "task_panel_summary", None)
        if task_panel_summary is not None:
            return task_panel_summary()
        return self.long_term_memory.pending_tasks()

    def pending_task_count(self) -> int:
        counter = getattr(self.long_term_memory, "pending_task_count", None)
        if counter is None:
            return 0
        return int(counter())

    def tasks_panel_expanded(self) -> bool:
        getter = getattr(self.long_term_memory, "get_preference", None)
        if getter is None:
            return False
        return getter("tasks_panel_expanded", "false") == "true"

    def set_tasks_panel_expanded(self, expanded: bool) -> None:
        setter = getattr(self.long_term_memory, "set_preference", None)
        if setter is not None:
            setter("tasks_panel_expanded", "true" if expanded else "false")

    def startup_greeting(self) -> str:
        from assistant.personal_assistant import generate_greeting

        greeting = generate_greeting(self.long_term_memory, self.context_observer)
        suggestion = next_proactive_suggestion(self.long_term_memory, self.context_observer)
        if not suggestion:
            return greeting
        return f"{greeting}\n\nSugestão: {suggestion}"

    def context_debug(self) -> str:
        return self.last_context_debug

    def _try_fast_route(self, user_message: str) -> str | None:
        if self.agent.has_pending_confirmation():
            result = self.agent.run(user_message, self._fast_agent_context())
            if result.remember:
                self._remember_pair(user_message, result.response)
            return result.response

        route = route_fast_command(user_message)
        if route is None:
            return None

        if route.kind == "clear_conversation":
            self.clear_history()
            return "Conversa limpa."

        if route.kind == "test_microphone":
            return self._test_microphone_now()

        if route.kind == "tool" and route.tool_name:
            result = self.agent.ask_tool_confirmation(
                tool_name=route.tool_name,
                arguments=route.arguments or {},
                reason=route.reason,
                context=self._fast_agent_context(),
            )
            if result.remember:
                self._remember_pair(user_message, result.response)
            return result.response

        return route.response

    def _fast_agent_context(self) -> AgentContext:
        return AgentContext(
            system_prompt=self.base_system_prompt,
            history=[],
            active_contexts=["FAST_ROUTE"],
            presence_state=self.presence.state.value,
            tools_enabled=self.presence.can_use_tools(),
            suggestions_enabled=False,
            confirmations_enabled=self.presence.can_ask_confirmation(),
        )

    def _test_microphone_now(self) -> str:
        if not self.voice_enabled:
            return "A voz esta desligada nas configuracoes."
        if self.voice_missing_dependencies:
            return "Nao consigo testar o microfone porque falta: " + ", ".join(self.voice_missing_dependencies)
        try:
            return check_microphone()
        except MicrophoneCheckError as exc:
            return f"Microfone indisponivel: {exc}"

    def _try_voice_question(self, user_message: str) -> str | None:
        text = _normalize_text(user_message)
        if text.strip(" .,!?:;") not in {
            "estado da voz",
            "estado do microfone",
            "a voz esta pronta",
            "voz pronta",
        }:
            return None
        return voice_status_report(
            enabled=self.voice_enabled,
            missing_dependencies=self.voice_missing_dependencies,
            microphone_ok=self.voice_microphone_ok,
            microphone_message=self.voice_microphone_message,
        )

    def _presence_silent_response(self) -> str:
        if self.presence.state == PresenceState.OFFLINE:
            return "Estou em OFFLINE. Nao vou responder nem executar acoes neste estado."
        return f"Estou em {self.presence.state.value}. {self.presence.description()}"

    def _try_presence_mode_command(self, user_message: str) -> str | None:
        requested_state = self.presence.requested_state_from_message(user_message)
        if requested_state is None:
            return None

        previous_state = self.presence.state
        self._remember_presence_transition(previous_state, requested_state)
        self.presence.set_state(requested_state)
        self._debug_log(
            "Mudanca automatica de presenca | "
            f"estado_anterior={previous_state.value} | "
            f"comando={user_message} | "
            f"estado_seguinte={requested_state.value}"
        )
        return self.presence.confirmation_for(requested_state)

    def _remember_presence_transition(
        self,
        previous_state: PresenceState,
        requested_state: PresenceState,
    ) -> None:
        remember_timeline_event = getattr(self.long_term_memory, "remember_timeline_event", None)
        if remember_timeline_event is None:
            return
        remember_timeline_event(
            "Mudanca automatica de modo de presenca: "
            f"{previous_state.value} -> {requested_state.value}."
        )

    def _remember_pair(self, user_message: str, response: str) -> None:
        if self.presence.can_store_memory():
            self.memory.append_pair(user_message, response)

    def _activate_contexts(self, user_message: str) -> list[ActiveContext]:
        self.active_contexts = self.context_manager.identify(user_message)
        self.last_context_debug = self.context_manager.debug_summary(self.active_contexts)
        self._debug_log(self.last_context_debug.replace("\n", " | "))
        return self.active_contexts

    def _context_for_agent(self, user_message: str) -> str:
        if not self.presence.can_store_memory():
            return ""

        parts: list[str] = []
        active_context_summary = self._active_context_summary()
        if active_context_summary:
            parts.append(active_context_summary)

        memory_context = self.long_term_memory.context_for(user_message)
        if memory_context:
            parts.append(memory_context)

        observed_context = self._observed_context_summary()
        if observed_context:
            parts.append(observed_context)

        return "\n".join(parts)

    def _pending_tasks_for_agent(self) -> str:
        if not self.presence.can_store_memory():
            return ""
        pending_tasks = getattr(self.long_term_memory, "pending_tasks", None)
        if pending_tasks is None:
            return ""
        return pending_tasks()

    def _active_context_summary(self) -> str:
        if not self.active_contexts:
            return ""

        lines = ["[contextos_ativos]"]
        for item in self.active_contexts:
            lines.append(
                f"- {item.name.value}: peso={item.weight:.2f}; memoria={item.memory_category}; "
                f"descricao={item.description}"
            )
        return "\n".join(lines)

    def _observed_context_summary(self) -> str:
        if self.context_observer is None:
            return ""

        snapshot = self.context_observer.latest_snapshot()
        latest_summary = self.context_observer.latest_summary()
        if snapshot is None and latest_summary is None:
            return ""

        lines = ["[contexto_observado]"]
        if latest_summary is not None:
            lines.append(f"Resumo recente: {latest_summary.summary}")
        if snapshot is not None:
            lines.extend(
                (
                    f"Aplicacao ativa: {snapshot.active_app or 'desconhecida'}",
                    f"Janela ativa: {snapshot.active_window or 'desconhecida'}",
                )
            )
            if snapshot.current_project:
                lines.append(f"Projeto aberto: {snapshot.current_project}")
            if snapshot.recent_files:
                lines.append("Ficheiros recentes: " + ", ".join(snapshot.recent_files[:5]))
        return "\n".join(lines)

    def _try_long_term_memory_command(self, user_message: str) -> str | None:
        lowered = _normalize_text(user_message.strip())
        if lowered.startswith("lembra-te que"):
            content = user_message.strip()[len("lembra-te que") :].strip(" .:")
            return self.long_term_memory.remember(content)

        if lowered.startswith("lembra que"):
            content = user_message.strip()[len("lembra que") :].strip(" .:")
            return self.long_term_memory.remember(content)

        if lowered.startswith("nao te esquecas que"):
            content = user_message.strip()[len("nao te esquecas que") :].strip(" .:")
            return self.long_term_memory.remember(content)

        if lowered.startswith("não te esqueças que"):
            content = user_message.strip()[len("não te esqueças que") :].strip(" .:")
            return self.long_term_memory.remember(content)

        if lowered.startswith("guarda isto"):
            content = user_message.strip()[len("guarda isto") :].strip(" .:")
            return self.long_term_memory.remember(content)

        if lowered.startswith("esquece"):
            query = user_message.strip()[len("esquece") :].strip(" .:")
            return self.long_term_memory.forget(query)

        if lowered.startswith("o que sabes sobre"):
            query = user_message.strip()[len("o que sabes sobre") :].strip(" .:?")
            return self.long_term_memory.answer_about(query)

        return None

    def _try_task_command(self, user_message: str) -> str | None:
        text = _normalize_text(user_message)
        show_details = _asks_task_details(text)

        if _asks_tasks_for_today(text):
            if show_details:
                return self.long_term_memory.tasks_for_today(show_details=True)
            return generate_task_summary(self.long_term_memory)

        if _asks_tasks_for_week(text):
            return self.long_term_memory.tasks_for_week(show_details=show_details)

        if _asks_pending_tasks(text):
            if show_details:
                return self._run_task_tool("list_pending_tasks", show_details=True)
            return generate_task_summary(
                pending_tasks=self._run_task_tool("list_pending_tasks"),
            )

        if show_details:
            return self._run_task_tool("list_pending_tasks", show_details=True)

        if _asks_to_complete_task(text):
            return self._run_task_tool("complete_task", query=_extract_task_update_query(user_message))

        if _asks_to_cancel_task(text):
            return self._run_task_tool("cancel_task", query=_extract_task_update_query(user_message))

        if _asks_to_postpone_task(text):
            return self._run_task_tool("postpone_task", query=_extract_task_update_query(user_message))

        if _looks_like_task_request(text):
            task_text = _extract_task_text(user_message)
            if _normalize_text(task_text).startswith("disto"):
                previous = self._previous_user_message()
                if not previous:
                    return "Diz-me primeiro a que te referes com 'disto'."
                task_text = f"{previous} {task_text}"

            return self.long_term_memory.create_task(task_text)

        return None

    def _run_task_tool(self, tool_name: str, **arguments) -> str:
        tool = self.tools.get(tool_name)
        prepared = {"long_term_memory": self.long_term_memory, **arguments}
        if tool is not None:
            return tool.run(prepared)

        fallback_tools = {
            "list_pending_tasks": list_pending_tasks_tool,
            "complete_task": complete_task_tool,
            "cancel_task": cancel_task_tool,
            "postpone_task": postpone_task_tool,
        }
        fallback = fallback_tools.get(tool_name)
        if fallback is None:
            return "Não consegui alterar a tarefa. Ela continua pendente."
        return fallback(**prepared)

    def _previous_user_message(self) -> str:
        for message in reversed(self.memory.load()):
            if message.get("role") == "user":
                return message.get("content", "")
        return ""

    def _try_timeline_command(self, user_message: str) -> str | None:
        text = _normalize_text(user_message)

        if _asks_what_happened_yesterday(text):
            return self.long_term_memory.timeline_for_date(date.today() - timedelta(days=1))

        if _asks_current_work_context(text):
            return self.long_term_memory.current_work_context()

        if _asks_project_start(text):
            project = _extract_project_query(user_message)
            return self.long_term_memory.project_start(project)

        if _looks_like_timeline_event(text):
            return self.long_term_memory.remember_timeline_event(user_message)

        return None

    def _try_presence_question(self, user_message: str) -> str | None:
        text = _normalize_text(user_message)
        if any(
            phrase in text
            for phrase in (
                "em que estado",
                "estado atual",
                "estado de presenca",
                "que modo",
                "em que modo",
                "modo atual",
            )
        ):
            return self.presence.state_report()
        return None

    def _try_briefing_question(self, user_message: str) -> str | None:
        text = _normalize_text(user_message)
        if _asks_daily_briefing(text):
            if _asks_task_details(text):
                return self.long_term_memory.tasks_for_today(show_details=True)
            return generate_daily_briefing(self.long_term_memory, self.context_observer)

        if _asks_session_continuity(text):
            return generate_session_resume(self.long_term_memory, self.context_observer)

        if _asks_yesterday_summary(text):
            return summarize_yesterday(self.long_term_memory)

        if _asks_last_active_project(text):
            return get_last_active_project(self.context_observer)

        return None

    def _try_proactive_suggestion_question(self, user_message: str) -> str | None:
        text = _normalize_text(user_message)
        if not _asks_for_suggestion(text):
            return None
        suggestion = next_proactive_suggestion(self.long_term_memory, self.context_observer)
        if not suggestion:
            return "Neste momento não tenho nenhuma sugestão nova relevante."
        return f"Sugestão: {suggestion}"

    def _try_language_preference_command(self, user_message: str) -> str | None:
        text = _normalize_text(user_message)
        if _asks_language_policy(text):
            return "A minha língua base é português de Portugal, mas posso falar inglês quando pedires."

        if _asks_for_english(text):
            self._set_current_language("en")
            return (
                "Understood. I can speak English when you ask, "
                "but my base language remains Portuguese from Portugal."
            )

        if _asks_for_portuguese(text):
            self._set_current_language("pt-PT")
            return "Claro. Volto ao português de Portugal."

        return None

    def _try_context_question(self, user_message: str) -> str | None:
        text = _normalize_text(user_message)
        if _asks_available_information(text):
            return self._available_information_summary()

        if _asks_about_contexts(text):
            return self.last_context_debug

        if _asks_about_tools(text):
            return self._tools_summary()

        if any(
            phrase in text
            for phrase in (
                "em que perfil",
                "qual perfil",
                "em que perfil estas",
                "qual e o perfil ativo",
                "qual e o perfil",
                "que perfil esta ativo",
                "perfil ativo",
            )
        ):
            return (
                "Ja nao uso perfis manuais. "
                "Identifico contextos automaticamente por mensagem.\n\n"
                f"{self.last_context_debug}"
            )
        return None

    def _available_information_summary(self) -> str:
        sections = ["Tenho acesso a estas fontes de informação:"]

        memory_context = self.long_term_memory.context_for(
            "utilizador projetos preferencias tarefas relacoes contexto",
            limit=5,
        )
        sections.append("\nMemória permanente:")
        sections.append(memory_context or "- Ainda não encontrei memória permanente relevante.")

        sections.append("\nTarefas:")
        sections.append(self.long_term_memory.pending_tasks())

        sections.append("\nTimeline:")
        sections.append(self.long_term_memory.current_work_context())

        sections.append("\nFicheiros da workspace:")
        workspace_tool = self.tools.get("list_workspace_files")
        if workspace_tool is None:
            sections.append("- A ferramenta de listagem da workspace não está ligada.")
        else:
            sections.append(workspace_tool.run({"workspace_path": self.workspace_path}))

        sections.append("\nContexto observado do computador:")
        observed = self._observed_context_summary()
        sections.append(observed or "- Ainda não tenho contexto observado disponível.")

        return "\n".join(sections)

    def _tools_summary(self) -> str:
        tools = self.tools.list()
        if not tools:
            return "Neste momento nao tenho ferramentas disponiveis."

        lines = [
            "Tenho estas ferramentas locais:",
            *[f"- {tool.name}: {tool.description}" for tool in tools],
            "",
            "Todas as ferramentas de ficheiros estao limitadas a pasta workspace.",
        ]
        return "\n".join(lines)

    def _try_profile_memory(self, user_message: str) -> str | None:
        text = _normalize_text(user_message)
        name_match = re.search(
            r"\b(?:chamo me|chamo-me|o meu nome e|meu nome e)\s+([^.,;:!?\n]{2,60})",
            user_message,
            re.IGNORECASE,
        )
        if name_match:
            name = name_match.group(1).strip(" .,!?:;")
            self.long_term_memory.remember(
                f"O utilizador chama-se {name}.",
                category=MemoryCategory.USER_PROFILE,
            )
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
            f"Contextos ativos: {', '.join(item.name.value for item in self.active_contexts)}.\n\n"
            "Ferramentas disponiveis para a aplicacao:\n"
            f"{self.tools.describe()}\n\n"
            "Quando precisares de uma ferramenta, a aplicacao decide e executa-a antes da resposta final."
        )
        if recurring_context:
            prompt += f"\n\nMemoria permanente relevante:\n{recurring_context}"
        self._debug_log("System prompt enviado ao LLM com contextos automaticos.")
        return prompt

    def _debug_log(self, message: str) -> None:
        if self.debug:
            print(f"[AssistenteIA DEBUG] {message}")

    def _perf_log(self, label: str, started_at: float, ended_at: float) -> None:
        if self.debug or self.debug_performance:
            elapsed_ms = (ended_at - started_at) * 1000
            print(f"[AssistenteIA PERF] {label}: {elapsed_ms:.1f} ms")


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


def _asks_about_contexts(text: str) -> bool:
    return "context" in text and any(
        phrase in text
        for phrase in (
            "que contextos",
            "contextos ativos",
            "que contexto",
            "para que serve",
            "o que faz",
            "qual e a funcao",
            "explica este contexto",
            "serve para que",
        )
    )


def _asks_available_information(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "que informacao tens",
            "que informação tens",
            "que dados tens",
            "o que sabes neste momento",
            "que fontes tens",
        )
    )


def _asks_about_tools(text: str) -> bool:
    return "ferrament" in text and any(
        phrase in text
        for phrase in (
            "que ferramentas",
            "ferramentas tens",
            "quais ferramentas",
            "lista as ferramentas",
            "mostra as ferramentas",
            "que consegues fazer",
        )
    )


def _asks_daily_briefing(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "o que tenho para fazer hoje",
            "que tenho para fazer hoje",
            "briefing de hoje",
            "resumo para hoje",
            "prepara o dia",
        )
    )


def _asks_session_continuity(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "onde ficamos",
            "onde ficámos",
            "em que estavamos a trabalhar",
            "em que estávamos a trabalhar",
            "no que estavamos a trabalhar",
            "no que estávamos a trabalhar",
        )
    )


def _asks_yesterday_summary(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "resume o dia de ontem",
            "resumo do dia de ontem",
            "faz um resumo de ontem",
        )
    )


def _asks_last_active_project(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "qual foi o ultimo projeto ativo",
            "qual foi o último projeto ativo",
            "ultimo projeto ativo",
            "último projeto ativo",
        )
    )


def _asks_for_suggestion(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "tens alguma sugestao",
            "tens alguma sugestão",
            "alguma sugestao",
            "alguma sugestão",
            "sugere alguma coisa",
            "o que sugeres",
        )
    )


def _asks_language_policy(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "qual e a tua lingua",
            "qual e o teu idioma",
            "que idioma estas a usar",
            "em que idioma estas",
            "lingua base",
            "idioma base",
            "podes falar ingles",
        )
    )


def _asks_for_english(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "prefiro ingles",
            "fala em ingles",
            "responde em ingles",
            "usa ingles",
            "speak english",
            "answer in english",
            "english please",
        )
    )


def _asks_for_portuguese(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "prefiro portugues",
            "fala em portugues",
            "responde em portugues",
            "volta ao portugues",
            "usa portugues",
            "portugues de portugal",
            "pt-pt",
        )
    )


def _looks_like_timeline_event(text: str) -> bool:
    has_time_reference = any(
        phrase in text
        for phrase in (
            "ontem",
            "anteontem",
            "semana passada",
            "na semana passada",
            "ha ",
            "há ",
        )
    )
    has_event_verb = any(
        word in text
        for word in (
            "estivemos",
            "falamos",
            "falámos",
            "trabalhamos",
            "trabalhámos",
            "comecaste",
            "começaste",
            "comecamos",
            "começámos",
            "iniciamos",
            "iniciámos",
        )
    )
    return has_time_reference and has_event_verb


def _asks_what_happened_yesterday(text: str) -> bool:
    return "ontem" in text and any(
        phrase in text
        for phrase in (
            "o que fizemos",
            "que fizemos",
            "em que trabalhamos",
            "em que estivemos",
            "sobre o que falamos",
        )
    )


def _asks_current_work_context(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "em que estavamos a trabalhar",
            "em que estamos a trabalhar",
            "no que estavamos a trabalhar",
            "no que estamos a trabalhar",
        )
    )


def _asks_project_start(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "quando comecamos este projeto",
            "quando começamos este projeto",
            "quando começámos este projeto",
            "quando iniciou este projeto",
            "quando comecou este projeto",
            "quando começou este projeto",
        )
    )


def _extract_project_query(message: str) -> str | None:
    match = re.search(r"\b(?:projeto|projecto)\s+([^.,;:!?]+)", message, re.IGNORECASE)
    if match:
        value = match.group(1).strip(" .,!?:;")
        if value and _normalize_text(value) not in {"este", "esta"}:
            return value
    if "assistenteia" in _normalize_text(message):
        return "AssistenteIA"
    return None


def _looks_like_task_request(text: str) -> bool:
    return (
        text.startswith("lembra-me")
        or text.startswith("lembra me")
        or text.startswith("tenho de")
        or text.startswith("tenho que")
        or text.startswith("adiciona uma tarefa")
        or text.startswith("cria uma tarefa")
    )


def _asks_tasks_for_today(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "o que tenho para fazer hoje",
            "tarefas para hoje",
            "que tenho para fazer hoje",
            "lembretes para hoje",
        )
    )


def _asks_pending_tasks(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "tarefas pendentes",
            "o que tenho para fazer",
            "que tarefas tenho",
            "que lembretes tenho",
        )
    ) and "hoje" not in text


def _asks_task_details(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "mostra os detalhes",
            "mostrar os detalhes",
            "mostra a tarefa original",
            "tarefa original",
            "com detalhes",
            "detalhes da tarefa",
        )
    )


def _asks_tasks_for_week(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "tarefas esta semana",
            "tarefas para esta semana",
            "que tarefas tenho esta semana",
            "o que tenho para fazer esta semana",
        )
    )


def _asks_to_complete_task(text: str) -> bool:
    direct_phrases = (
        "marca essa tarefa como concluida",
        "marca a tarefa como concluida",
        "marca como concluida",
        "marcar como concluida",
        "conclui essa tarefa",
        "concluir esta tarefa",
        "concluir essa tarefa",
        "ja terminei",
        "terminei esta tarefa",
        "terminei essa tarefa",
        "tarefa feita",
    )
    if any(phrase in text for phrase in direct_phrases):
        return True

    return any(
        phrase in text
        for phrase in (
            "marca esta tarefa como concluida",
            "marca esta tarefa como concluída",
            "conclui esta tarefa",
            "tarefa concluida",
            "tarefa concluída",
        )
    )


def _asks_to_postpone_task(text: str) -> bool:
    direct_phrases = (
        "adia essa tarefa",
        "adia esse lembrete",
        "adiar essa tarefa",
        "passa essa tarefa",
    )
    if any(phrase in text for phrase in direct_phrases):
        return True

    return any(
        phrase in text
        for phrase in (
            "adia esta tarefa",
            "adiar esta tarefa",
            "passa esta tarefa",
        )
    )


def _asks_to_cancel_task(text: str) -> bool:
    direct_phrases = (
        "cancela essa tarefa",
        "cancelar essa tarefa",
        "remove essa tarefa",
        "limpa esta tarefa",
        "limpa essa tarefa",
        "retira esta tarefa",
        "retira essa tarefa",
        "retira este lembrete",
        "retira esse lembrete",
        "ja nao e preciso",
        "ja nao preciso",
        "cancela este lembrete",
        "cancela esse lembrete",
    )
    if any(phrase in text for phrase in direct_phrases):
        return True

    return any(
        phrase in text
        for phrase in (
            "cancela esta tarefa",
            "cancelar esta tarefa",
            "remove esta tarefa",
        )
    )


def _extract_task_text(message: str) -> str:
    cleaned = message.strip()
    normalized = _normalize_text(cleaned)
    prefixes = (
        "lembra-me de ",
        "lembra me de ",
        "lembra-me ",
        "lembra me ",
        "tenho de ",
        "tenho que ",
        "adiciona uma tarefa para ",
        "adiciona uma tarefa ",
        "cria uma tarefa para ",
        "cria uma tarefa ",
    )
    for prefix in prefixes:
        if normalized.startswith(prefix):
            return cleaned[len(prefix) :].strip(" .:")
    return cleaned


def _extract_task_update_query(message: str) -> str:
    text = message.strip()
    normalized = _normalize_text(text)
    cleaned_normalized = normalized.strip(" .,!?:;")
    ambiguous_phrases = (
        "ja terminei",
        "tarefa feita",
        "ja nao e preciso",
        "ja nao preciso",
        "marca como concluida",
        "marcar como concluida",
        "limpa essa tarefa",
        "limpa esta tarefa",
        "retira esse lembrete",
        "retira este lembrete",
        "cancela esse lembrete",
        "cancela este lembrete",
    )
    if cleaned_normalized in ambiguous_phrases:
        return ""

    extra_prefixes = (
        "marca essa tarefa como concluida",
        "marca a tarefa como concluida",
        "conclui essa tarefa",
        "concluir esta tarefa",
        "concluir essa tarefa",
        "terminei esta tarefa",
        "terminei essa tarefa",
        "adia essa tarefa para",
        "adia esse lembrete para",
        "adiar essa tarefa para",
        "adia essa tarefa",
        "adia esse lembrete",
        "adiar essa tarefa",
        "cancela essa tarefa",
        "cancelar essa tarefa",
        "remove essa tarefa",
        "retira esta tarefa",
        "retira essa tarefa",
    )
    for prefix in extra_prefixes:
        if normalized.startswith(prefix):
            return text[len(prefix) :].strip(" .:")

    prefixes = (
        "marca esta tarefa como concluida",
        "marca esta tarefa como concluída",
        "conclui esta tarefa",
        "tarefa concluida",
        "tarefa concluída",
        "adia esta tarefa para",
        "adiar esta tarefa para",
        "adia esta tarefa",
        "adiar esta tarefa",
        "cancela esta tarefa",
        "cancelar esta tarefa",
        "remove esta tarefa",
    )
    for prefix in prefixes:
        if normalized.startswith(prefix):
            return text[len(prefix) :].strip(" .:")
    return text


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))

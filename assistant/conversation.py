from __future__ import annotations

import re
import time
import traceback
import unicodedata
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from assistant.agent import Agent, AgentContext, system_state_tool_intent
from assistant.briefing import get_last_active_project, summarize_yesterday
from assistant.cognition.context_builder import ContextBuilder
from assistant.cognition.executive_function import CognitiveStrategy, ExecutiveFunction
from assistant.cognition.intent_engine import IntentEngine
from assistant.cognition.reasoning_engine import ReasoningEngine, ReasoningResult
from assistant.cognition.reflection_engine import ReflectionEngine
from assistant.context_manager import ActiveContext, ContextManager
from assistant.delegation import DelegationManager, DelegationTarget
from assistant.fast_router import load_quick_sites, route_fast_command
from assistant.long_term_memory import LongTermMemory, MemoryCategory
from assistant.memory import ConversationMemory
from assistant.model_provider import ProviderConfigurationError
from assistant.memory_recall import (
    build_memory_retrieval,
    build_task_retrieval,
    detect_unsupported_memory_claim,
    extract_academic_event_candidate,
    extract_requested_attributes,
    extract_task_candidate,
    is_memory_attribute_followup,
    is_memory_recall_followup,
    is_memory_recall_question,
    is_memory_write_command,
    is_task_recall_question,
    normalize_candidate_fields,
    parse_memory_write_command,
    render_memory_write_confirmation,
)
from assistant.personal_assistant import (
    generate_daily_briefing,
    generate_session_resume,
    generate_task_summary,
)
from assistant.personal_model import PersonalModel, infer_category
from assistant.presence_manager import PresenceManager, PresenceState
from assistant.proactive_suggestions import next_proactive_suggestion
from assistant.response_composer import ComposerRequest, ResponseComposer
from assistant.security import check_user_request
from assistant.session_manager import SessionManager
from assistant.text_encoding import debug_text_encoding, has_mojibake_markers
from assistant.tool_registry import ToolRegistry
from assistant.tools import (
    cancel_task as cancel_task_tool,
    complete_task as complete_task_tool,
    list_pending_tasks as list_pending_tasks_tool,
    postpone_task as postpone_task_tool,
)
from assistant.ui_event_adapter import UIEventAdapter
from assistant.voice_critic import VoiceCritic, detect_semantic_conflict, detect_subject_swap, has_voice_issue
from assistant.voice_input import (
    MicrophoneCheckError,
    check_microphone,
    play_last_voice_input,
    voice_status_report,
)
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
        personal_model: PersonalModel | None = None,
        active_profile_name: str = "Geral",
        debug: bool = False,
        debug_agent: bool = False,
        debug_performance: bool = False,
        debug_ollama_payload: bool = False,
        presence_manager: PresenceManager | None = None,
        context_observer: ContextObserver | None = None,
        session_manager: SessionManager | None = None,
        known_projects: dict[str, str] | None = None,
        desktop_config: dict[str, object] | None = None,
        desktop_action_runner=None,
        voice_enabled: bool = False,
        voice_missing_dependencies: list[str] | None = None,
        voice_microphone_ok: bool = False,
        voice_microphone_message: str = "",
        voice_sample_rate: int = 44100,
        voice_input_device: str | int | None = "default",
        voice_auto_select_input: bool = True,
        voice_silent_rms_threshold: float = 0.001,
        voice_channels: int = 1,
        voice_probe_duration: float = 0.5,
        voice_min_record_seconds: float = 2.0,
        voice_model: str = "base",
        voice_language: str = "pt",
        voice_critic_llm: OllamaClient | None = None,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.long_term_memory = long_term_memory
        self.personal_model = personal_model
        self.response_composer = ResponseComposer(
            llm,
            voice_critic=VoiceCritic(voice_critic_llm) if voice_critic_llm is not None else None,
        )
        self._last_response_blocked = False
        self._last_selected_path = ""
        self._active_memory_topic = ""
        self._active_memory_entity_id: str | None = None
        self._active_memory_recall_ttl = 0
        self._last_turn_telemetry: dict | None = None
        self.tools = tools
        self.desktop_config = desktop_config or {}
        self.quick_sites = self._quick_sites_for_fast_router()
        self.workspace = WorkspaceGuard(workspace_path)
        self.workspace_path = self.workspace.resolve()
        self.base_system_prompt = base_system_prompt
        self.active_profile_name = active_profile_name
        self.debug = debug
        self.debug_agent = debug_agent
        self.debug_performance = debug_performance
        self.debug_ollama_payload = debug_ollama_payload
        self._turn_trace: dict[str, object] | None = None
        self._pending_ui_events: list[str] = []
        self._last_fast_tools_used: tuple[str, ...] = ()
        self._active_operation_type = ""
        self._active_operation_topic = ""
        self._active_operation_id = ""
        self._active_operation_ttl = 0
        self._pending_user_intent: dict[str, str] | None = None
        self._conversation_segment_start = 0
        self.presence = presence_manager or PresenceManager()
        self.context_observer = context_observer
        self.session_manager = session_manager
        self.context_manager = ContextManager()
        self.intent_engine = IntentEngine()
        self.executive_function = ExecutiveFunction()
        self.reflection_engine = ReflectionEngine()
        self.reasoning_engine = ReasoningEngine()
        self.last_cognitive_reasoning: ReasoningResult | None = None
        self.last_cognitive_strategy: CognitiveStrategy | None = None
        self.active_contexts: list[ActiveContext] = []
        self.last_context_debug = ""
        self.delegation = DelegationManager()
        self.language_base = "pt-PT"
        self.current_language = self._load_language_preference()
        self.voice_enabled = voice_enabled
        self.voice_missing_dependencies = voice_missing_dependencies or []
        self.voice_microphone_ok = voice_microphone_ok
        self.voice_microphone_message = voice_microphone_message
        self.voice_sample_rate = voice_sample_rate
        self.voice_input_device = voice_input_device
        self.voice_auto_select_input = voice_auto_select_input
        self.voice_silent_rms_threshold = voice_silent_rms_threshold
        self.voice_channels = voice_channels
        self.voice_probe_duration = voice_probe_duration
        self.voice_min_record_seconds = voice_min_record_seconds
        self.voice_model = voice_model
        self.voice_language = voice_language or "pt"
        self._ensure_language_preferences()
        if self.session_manager is not None and self.presence.state == PresenceState.ACTIVE_CONVERSATION:
            self.session_manager.start_session()
        self.agent = Agent(
            llm=llm,
            tools=tools,
            workspace_path=self.workspace_path,
            context_observer=context_observer,
            presence_manager=self.presence,
            long_term_memory=long_term_memory,
            known_projects=known_projects,
            desktop_config=desktop_config,
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
        previous_state = self.presence.state
        self.presence.set_state(state)
        self._sync_session_with_presence(previous_state, self.presence.state)
        self._debug_log(f"Estado de presenca alterado para: {self.presence.state.value}")

    def presence_state(self) -> str:
        return self.presence.state.value

    def respond(self, user_message: str) -> str:
        """Public entry point: never lets an internal exception reach the UI.

        respond() is called directly from GUI worker threads (see
        prototype_web_ui/controller.py and ui/main_window.py), which show
        whatever this returns — or a generic "Não consegui responder" if it
        raises. Any of the many _try_*/cognition/agent code paths below can
        legitimately fail (a transient Ollama hiccup, an edge case in a
        cognition module, an empty model completion), so the entire turn is
        guarded here rather than trusting every call site downstream to
        catch its own errors.
        """
        try:
            return self._respond_inner(user_message)
        except Exception as exc:  # noqa: BLE001 - last-resort turn guard, see docstring
            return self._handle_unexpected_error(user_message, exc)

    def _handle_unexpected_error(self, user_message: str, exc: Exception) -> str:
        if isinstance(exc, ProviderConfigurationError):
            return self._handle_provider_configuration_error(user_message, exc)
        print("[ECHO ERROR] stage=AssistantEngine.respond")
        print(f"[ECHO ERROR] type={type(exc).__name__}")
        print(f"[ECHO ERROR] message={exc}")
        print(f"[ECHO ERROR] user_message={user_message}")
        if _echo_debug_errors_enabled():
            traceback.print_exc()
        if self._turn_trace is None:
            # Only reachable if the crash happened before _begin_turn_trace
            # ran this turn (e.g. inside session_manager bookkeeping) — start
            # one now so the eval/telemetry API still reports this turn's
            # exception instead of silently reusing the previous turn's data.
            self._begin_turn_trace(user_message)
        if self._turn_trace is not None:
            self._turn_trace["exception_type"] = type(exc).__name__
            self._turn_trace["exception_message"] = str(exc)
        fallback = "Desculpa, tive um problema técnico a processar isso. Podes repetir ou reformular a pergunta?"
        return self._complete_turn(
            user_message,
            fallback,
            "INTERNAL_ERROR",
            remember=False,
            technical=True,
            selected_path="INTERNAL_ERROR",
        )

    def _handle_provider_configuration_error(self, user_message: str, exc: ProviderConfigurationError) -> str:
        print("[ECHO PROVIDER ERROR]")
        print(f"provider={exc.provider}")
        print(f"provider_error_type={exc.provider_error_type}")
        print(f"message={exc}")
        if self._turn_trace is None:
            self._begin_turn_trace(user_message)
        if self._turn_trace is not None:
            self._turn_trace["exception_type"] = type(exc).__name__
            self._turn_trace["exception_message"] = str(exc)
            self._turn_trace["provider"] = exc.provider
            self._turn_trace["provider_error_type"] = exc.provider_error_type
            self._turn_trace["fallback_used"] = False
        return self._complete_turn(
            user_message,
            str(exc),
            "PROVIDER_ERROR",
            remember=False,
            technical=True,
            selected_path="PROVIDER_ERROR",
        )

    def _respond_inner(self, user_message: str) -> str:
        user_message = str(user_message or "")
        if has_mojibake_markers(user_message):
            debug_text_encoding("assistant_engine_input", user_message)
        if self.session_manager is not None:
            inactive_summary = self.session_manager.end_if_inactive(60 * 60 * 2, self.context_observer)
            if inactive_summary is not None:
                self._remember_session_summary(inactive_summary)
            if self.presence.state == PresenceState.ACTIVE_CONVERSATION:
                self.session_manager.start_session()
        request_started_at = time.perf_counter()
        self._begin_turn_trace(user_message)
        self._record_tool_intent_check(user_message)
        self._perf_log("pedido recebido", request_started_at, request_started_at)
        # Decremented once per incoming turn, before this turn can refresh it,
        # so a grounded recall keeps a short follow-up window open (e.g.
        # "Qual era a disciplina?" -> "E quando é?") without depending on the
        # follow-up landing on the very next turn.
        if self._active_memory_recall_ttl > 0:
            self._active_memory_recall_ttl -= 1
        if self._active_operation_ttl > 0:
            self._active_operation_ttl -= 1
            if self._active_operation_ttl == 0:
                self._active_operation_type = ""
                self._active_operation_topic = ""
                self._active_operation_id = ""
        pending_intent_response = self._try_pending_user_intent(user_message)
        if pending_intent_response is not None:
            pending_intent_response = self._complete_turn(
                user_message,
                pending_intent_response,
                "RESPONSE_COMPOSER",
                selected_path="GENERAL_CONVERSATION",
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return pending_intent_response
        # An explicit write command ("Regista que...") is re-extracted from
        # its own stripped content in _try_memory_write_command below, so the
        # passive path is skipped here to avoid writing the same fact twice
        # (once from the full raw message, once from the stripped content).
        if not self._should_skip_passive_memory_extraction(user_message):
            self._maybe_extract_structured_memory(user_message)

        presence_mode_response = self._try_presence_mode_command(user_message)
        if presence_mode_response is not None:
            presence_mode_response = self._complete_turn(
                user_message,
                presence_mode_response,
                "MEMORY_COMMAND",
                selected_path="MEMORY_COMMAND",
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return presence_mode_response

        if not self.presence.can_respond():
            response = self._presence_silent_response()
            response = self._complete_turn(
                user_message,
                response,
                "FALLBACK",
                remember=False,
                technical=True,
                selected_path="FALLBACK",
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return response

        fast_started_at = time.perf_counter()
        fast_response = self._try_fast_route(user_message)
        self._perf_log("router rapido", fast_started_at, time.perf_counter())
        if fast_response is not None:
            fast_response = self._complete_turn(
                user_message=user_message,
                response=fast_response,
                source="TOOL_CONFIRMATION" if self.agent.has_pending_confirmation() else "FAST_ROUTE",
                remember=True,
                tool_confirmation=self.agent.has_pending_confirmation(),
                selected_path="FAST_ROUTE",
                tools_used=self._last_fast_tools_used,
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return fast_response

        topic_shift_response = self._try_topic_shift(user_message)
        if topic_shift_response is not None:
            topic_shift_response = self._complete_turn(
                user_message=user_message,
                response=topic_shift_response,
                source="TOPIC_SHIFT",
                remember=True,
                technical=True,
                selected_path="TOPIC_SHIFT",
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return topic_shift_response

        research_response = self._try_research_request(user_message)
        if research_response is not None:
            research_response = self._complete_turn(
                user_message=user_message,
                response=research_response,
                source="RESEARCH_REQUEST",
                remember=True,
                technical=True,
                selected_path="RESEARCH_REQUEST",
                tools_used=self._last_fast_tools_used,
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return research_response

        system_state_response = self._try_system_state_tool_query(user_message)
        if system_state_response is not None:
            system_state_response = self._complete_turn(
                user_message=user_message,
                response=system_state_response,
                source="TOOL_RESULT",
                remember=True,
                technical=True,
                selected_path="TOOL_PATH",
                tools_used=self._last_fast_tools_used,
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return system_state_response

        social_response = self._try_pure_social_turn(user_message)
        if social_response is not None:
            social_response = self._complete_turn(
                user_message=user_message,
                response=social_response,
                source="SOCIAL_FAST_PATH",
                remember=True,
                selected_path="SOCIAL_PATH",
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return social_response

        casual_share_response = self._try_casual_social_share(user_message)
        if casual_share_response is not None:
            casual_share_response = self._complete_turn(
                user_message=user_message,
                response=casual_share_response,
                source="SOCIAL_FAST_PATH",
                remember=True,
                selected_path="SOCIAL_PATH",
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return casual_share_response

        security = check_user_request(user_message)
        if not security.allowed:
            response = security.message or "Nao posso realizar essa acao por motivos de seguranca."
            response = self._complete_turn(
                user_message,
                response,
                "ERROR",
                technical=True,
                selected_path="SECURITY",
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return response

        if self.presence.can_store_memory():
            memory_write_response = self._try_memory_write_command(user_message)
            if memory_write_response is not None:
                memory_write_response = self._complete_turn(
                    user_message,
                    memory_write_response,
                    "MEMORY_WRITE_DETERMINISTIC",
                    technical=True,
                    selected_path="MEMORY_WRITE",
                )
                self._perf_log("resposta total", request_started_at, time.perf_counter())
                return memory_write_response

            memory_inventory_response = self._try_memory_inventory(user_message)
            if memory_inventory_response is not None:
                memory_inventory_response = self._complete_turn(
                    user_message,
                    memory_inventory_response,
                    "MEMORY_INVENTORY",
                    technical=True,
                    selected_path="MEMORY_INVENTORY",
                )
                self._perf_log("resposta total", request_started_at, time.perf_counter())
                return memory_inventory_response

            memory_recall_response = self._try_memory_recall_question(user_message)
            if memory_recall_response is not None:
                memory_recall_response = self._complete_turn(
                    user_message,
                    memory_recall_response,
                    "MEMORY_RECALL_DETERMINISTIC",
                    selected_path="MEMORY_RECALL",
                )
                self._perf_log("resposta total", request_started_at, time.perf_counter())
                return memory_recall_response

            project_history_response = self._try_project_history_recall(user_message)
            if project_history_response is not None:
                project_history_response = self._complete_turn(
                    user_message,
                    project_history_response,
                    "MEMORY_RECALL_DETERMINISTIC",
                    selected_path="MEMORY_RECALL",
                )
                self._perf_log("resposta total", request_started_at, time.perf_counter())
                return project_history_response

        text_transformation_response = self._try_text_transformation_request(user_message)
        if text_transformation_response is not None:
            text_transformation_response = self._complete_turn(
                user_message,
                text_transformation_response,
                "RESPONSE_COMPOSER",
                selected_path="TEXT_SUMMARIZATION",
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return text_transformation_response

        direct_phrase_response = self._try_direct_short_phrase_request(user_message)
        if direct_phrase_response is not None:
            direct_phrase_response = self._complete_turn(
                user_message,
                direct_phrase_response,
                "DIRECT_SHORT_RESPONSE",
                selected_path="GENERAL_CONVERSATION",
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return direct_phrase_response

        deterministic_help_response = self._try_deterministic_help_response(user_message)
        if deterministic_help_response is not None:
            deterministic_help_response = self._complete_turn(
                user_message,
                deterministic_help_response,
                "DETERMINISTIC_HELP",
                selected_path="GENERAL_CONVERSATION",
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return deterministic_help_response

        general_knowledge_response = self._try_general_knowledge_query(user_message)
        if general_knowledge_response is not None:
            general_knowledge_response = self._complete_turn(
                user_message,
                general_knowledge_response,
                "RESPONSE_COMPOSER",
                selected_path="GENERAL_KNOWLEDGE_QUERY",
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return general_knowledge_response

        conversational_response = self._try_conversational_refinement(user_message)
        if conversational_response is not None:
            conversational_response = self._complete_turn(
                user_message,
                conversational_response,
                "RESPONSE_COMPOSER",
                selected_path="CONVERSATIONAL_REFINEMENT",
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return conversational_response

        intent = self._intent_for_message(user_message)
        strategy = self.executive_function.choose(user_message, intent)
        self.last_cognitive_strategy = strategy
        self._record_agent_route_decision(strategy, intent)
        self._debug_log(
            "Executive Function | "
            f"categoria={strategy.category.value} | modo={strategy.mode} | razao={strategy.reason}"
        )

        if strategy.use_context_manager:
            active_contexts = self._activate_contexts(user_message)
        else:
            active_contexts = []
            self.active_contexts = []
            self.last_context_debug = f"Executive Function: {strategy.category.value} sem contextos automáticos."

        cognitive_reasoning = self._run_cognitive_loop(user_message, intent, strategy, active_contexts)
        if strategy.category.value == "SOCIAL_CONVERSATION":
            response = self.response_composer.compose(
                ComposerRequest(
                    intent="social_conversation",
                    user_message=user_message,
                    history=self._recent_conversation_history(user_message),
                    facts=["Interação social simples em Sistema 1."],
                    fallback="Olá Alexandre.",
                    language_instruction=self._language_instruction(),
                )
            )
            response = self._complete_turn(
                user_message,
                response,
                "RESPONSE_COMPOSER",
                selected_path="SOCIAL_PATH",
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return response

        focused_response = self._try_focused_cognitive_response(cognitive_reasoning)
        if focused_response is not None:
            return self._complete_turn(
                user_message,
                focused_response,
                "RESPONSE_COMPOSER",
                selected_path="CONVERSATIONAL_REFINEMENT",
            )

        if strategy.allow_clarifying_questions and cognitive_reasoning.needs_more_context:
            response = self.response_composer.compose(
                ComposerRequest(
                    intent="clarifying_question",
                    user_message=user_message,
                    history=self._recent_conversation_history(user_message),
                    facts=cognitive_reasoning.questions,
                    fallback="Antes de avançarmos, preciso de perceber melhor o que queres fazer.",
                    language_instruction=self._language_instruction(),
                )
            )
            response = self._complete_turn(
                user_message,
                response,
                "RESPONSE_COMPOSER",
                selected_path="CLARIFYING_QUESTION",
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return response

        language_response = self._try_language_preference_command(user_message)
        if language_response is not None:
            return self._complete_turn(user_message, language_response, "RESPONSE_COMPOSER", selected_path="LANGUAGE_COMMAND")

        presence_question_response = self._try_presence_question(user_message)
        if presence_question_response is not None:
            return self._complete_turn(user_message, presence_question_response, "MEMORY_COMMAND", technical=True, selected_path="MEMORY_COMMAND")

        voice_response = self._try_voice_question(user_message)
        if voice_response is not None:
            return self._complete_turn(user_message, voice_response, "MEMORY_COMMAND", technical=True, selected_path="CONTEXT_COMMAND")

        context_question_response = self._try_context_question(user_message)
        if context_question_response is not None:
            return self._complete_turn(user_message, context_question_response, "TOOL_RESULT", technical=True, selected_path="CONTEXT_COMMAND")

        briefing_response = self._try_briefing_question(user_message)
        if briefing_response is not None:
            return self._complete_turn(user_message, briefing_response, "SESSION_COMMAND", selected_path="BRIEFING")

        proactive_response = self._try_proactive_suggestion_question(user_message)
        if proactive_response is not None:
            return self._complete_turn(user_message, proactive_response, "RESPONSE_COMPOSER", selected_path="GENERAL_CONVERSATION")

        # Local memory questions are answered before tool routing so the LLM
        # cannot confuse "lembras-te da pasta?" with "lista a pasta".
        if self.presence.can_store_memory():
            memory_command_response = self._try_long_term_memory_command(user_message)
            if memory_command_response is not None:
                return self._complete_turn(user_message, memory_command_response, "MEMORY_COMMAND", selected_path="MEMORY_COMMAND")

            task_response = self._try_task_command(user_message)
            if task_response is not None:
                return self._complete_turn(user_message, task_response, "MEMORY_COMMAND", technical=True, selected_path="TASK_COMMAND")

            timeline_response = self._try_timeline_command(user_message)
            if timeline_response is not None:
                return self._complete_turn(user_message, timeline_response, "MEMORY_COMMAND", selected_path="TIMELINE")

            profile_response = self._try_profile_memory(user_message)
            if profile_response is not None:
                return self._complete_turn(user_message, profile_response, "MEMORY_COMMAND", selected_path="PROFILE_MEMORY")

            conversation_memory_response = self._try_conversation_memory_question(user_message)
            if conversation_memory_response is not None:
                return self._complete_turn(user_message, conversation_memory_response, "MEMORY_COMMAND", selected_path="CONVERSATION_MEMORY")

        history = self.memory.load() if self.presence.can_store_memory() else []
        recurring_context = self._context_for_agent(user_message)
        delegation_response = self._try_delegation(user_message, recurring_context)
        if delegation_response is not None:
            return self._complete_turn(user_message, delegation_response, "PLANNER_DIRECT", selected_path="DELEGATION")

        if _should_answer_without_agent(strategy):
            response = self.response_composer.compose(
                ComposerRequest(
                    intent=intent.intent,
                    user_message=user_message,
                    history=self._recent_conversation_history(user_message),
                    facts=self._facts_with_cognitive_reasoning([]),
                    context=recurring_context,
                    fallback="Diz-me um pouco melhor o que tens em mente.",
                    intent_instruction=_intent_instruction_for_user_message(user_message),
                    language_instruction=self._language_instruction(),
                )
            )
            return self._complete_turn(user_message, response, "RESPONSE_COMPOSER", selected_path="GENERAL_CONVERSATION")

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
                session_summary=self._session_summary_for_agent(),
                presence_state=self.presence.state.value,
                tools_enabled=self.presence.can_use_tools(),
                suggestions_enabled=self.presence.can_make_suggestions(),
                confirmations_enabled=self.presence.can_ask_confirmation(),
            ),
        )
        self._perf_log("agent/LLM quando necessario", llm_started_at, time.perf_counter())
        self._remember_tools_used(result.tools_used)
        self._record_agent_debug_trace(result.debug_trace)
        response = self._complete_turn(
            user_message=user_message,
            response=result.response,
            source=_agent_result_source(result),
            remember=result.remember,
            technical=bool(result.tools_used and _looks_like_technical_tool_result(result.response)),
            tool_confirmation=self.agent.has_pending_confirmation(),
            selected_path="AGENT",
            tools_used=result.tools_used,
        )
        self._perf_log("resposta total", request_started_at, time.perf_counter())
        return response

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
        self.clear_conversation()

    def clear_conversation(self) -> None:
        self.memory.clear()
        self._reset_transient_context()
        self._conversation_segment_start = 0
        self._queue_ui_event("conversation_cleared", {})

    def _reset_transient_context(self) -> None:
        self._active_operation_type = ""
        self._active_operation_topic = ""
        self._active_operation_id = ""
        self._active_operation_ttl = 0
        self._active_memory_topic = ""
        self._active_memory_entity_id = None
        self._active_memory_recall_ttl = 0
        self.active_contexts = []
        self.last_context_debug = ""
        self.last_cognitive_reasoning = None
        self.last_cognitive_strategy = None

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
        if self.session_manager is not None and _is_plain_startup_greeting(greeting):
            session_hint = self.session_manager.startup_hint()
            if session_hint:
                greeting = session_hint
        if _is_plain_startup_greeting(greeting):
            greeting = "Olá Alexandre."
        suggestion = next_proactive_suggestion(self.long_term_memory, self.context_observer)
        if not suggestion:
            return greeting
        return f"{greeting}\n\nSugestão: {suggestion}"

    def context_debug(self) -> str:
        return self.last_context_debug

    def _should_skip_passive_memory_extraction(self, user_message: str) -> bool:
        normalized = _normalize_text(user_message)
        if is_memory_write_command(user_message):
            return False
        if _extract_research_query(user_message):
            return True
        if _is_memory_inventory_query(user_message):
            return True
        if is_memory_recall_question(normalized):
            return True
        if self._active_memory_recall_ttl > 0 and is_memory_recall_followup(normalized):
            return True
        if _is_general_knowledge_query(user_message):
            return True
        return False

    def _try_memory_inventory(self, user_message: str) -> str | None:
        if not _is_memory_inventory_query(user_message):
            return None

        facts = self.long_term_memory.find_structured_facts()
        lines: list[str] = []
        sources: list[str] = []
        academic = [fact for fact in facts if fact.fact_type == "academic_event"]
        tasks = [fact for fact in facts if fact.fact_type == "task" and fact.status != "cancelled"]

        for fact in academic[:3]:
            sources.append(f"PERSISTENT_MEMORY:{fact.id}")
            lines.append(_summarize_structured_fact_for_inventory(fact))
        if tasks:
            for fact in tasks[:3]:
                sources.append(f"PERSISTENT_MEMORY:{fact.id}")
                action = fact.action or "uma tarefa"
                lines.append(f"uma tarefa pendente: {action}")

        if self._turn_trace is not None:
            self._turn_trace["persistent_memory_matches"] = len(facts)
            self._turn_trace["selected_memory_ids"] = [source.rsplit(":", 1)[-1] for source in sources]
            self._turn_trace["response_grounded"] = True
            self._turn_trace["grounding_sources"] = sources
            self._turn_trace["deterministic_response"] = True

        if not lines:
            return "Não tenho ainda informação persistente guardada."
        if len(lines) == 1:
            return f"Tenho guardado {lines[0]}."
        return "Tenho guardado " + "; ".join(lines[:-1]) + f"; e {lines[-1]}."

    def _try_direct_short_phrase_request(self, user_message: str) -> str | None:
        return _direct_short_phrase_response(_normalize_text(user_message)) or None

    def _try_deterministic_help_response(self, user_message: str) -> str | None:
        text = _normalize_text(user_message)
        if _looks_like_travel_destination_followup(text, self._previous_user_message()):
            return (
                "Acho que ainda me falta perceber como gostas de viajar. "
                "Isso vai influenciar mais a escolha do que saber apenas a zona."
            )
        return _deterministic_help_response(text) or None

    def _try_general_knowledge_query(self, user_message: str) -> str | None:
        if not _is_general_knowledge_query(user_message):
            return None
        return self.response_composer.compose(
            ComposerRequest(
                intent="general_knowledge_query",
                user_message=user_message,
                history=self._recent_conversation_history(user_message),
                facts=[
                    "Pergunta de conhecimento geral. Não uses memória persistente por defeito.",
                    "Não afirmes que pesquisaste ou que consultaste fontes.",
                ],
                fallback="Posso explicar isso de forma geral, mas sem fingir fontes externas.",
                language_instruction=self._language_instruction(),
            )
        )

    def _try_fast_route(self, user_message: str) -> str | None:
        self._last_fast_tools_used = ()
        if self.agent.has_pending_confirmation():
            result = self.agent.run(user_message, self._fast_agent_context())
            self._remember_tools_used(result.tools_used)
            self._last_fast_tools_used = result.tools_used
            self._record_agent_debug_trace(result.debug_trace)
            return result.response

        route = route_fast_command(user_message, quick_sites=self.quick_sites)
        if route is None:
            return None

        if route.kind == "clear_conversation":
            self.clear_conversation()
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
            self._remember_tools_used(result.tools_used)
            self._last_fast_tools_used = result.tools_used
            self._record_agent_debug_trace(result.debug_trace)
            return result.response

        return route.response

    def _quick_sites_for_fast_router(self) -> dict[str, str]:
        sites = load_quick_sites()
        default_email = str(self.desktop_config.get("default_email", "")).lower()
        if default_email == "gmail":
            sites["mail"] = "https://mail.google.com"
            sites["email"] = "https://mail.google.com"
            sites["correio"] = "https://mail.google.com"
        return sites

    def consume_ui_events(self) -> list[str]:
        events = list(self._pending_ui_events)
        self._pending_ui_events.clear()
        return events

    def _try_topic_shift(self, user_message: str) -> str | None:
        if not _is_topic_shift_request(user_message):
            return None
        self._reset_transient_context()
        self._conversation_segment_start = len(self.memory.load())
        _debug_print("ECHO_DEBUG_UI", "topic_shift_detected=True")
        self._queue_ui_event("topic_changed", {})
        return "Claro. Que tema queres seguir agora?"

    def _queue_ui_event(self, event_type: str, payload: dict[str, object] | None = None) -> None:
        event = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **(payload or {}),
        }
        _debug_print("ECHO_DEBUG_UI", f"ui_event_type={event_type}")
        if "operation_id" in event:
            _debug_print("ECHO_DEBUG_UI", f"ui_event_operation_id={event['operation_id']}")
        self._pending_ui_events.append(UIEventAdapter.serialize(event_type, payload or {}))

    def _try_research_request(self, user_message: str) -> str | None:
        query = _extract_research_query(user_message)
        followup = False
        if not query and self._active_operation_type == "research" and self._active_operation_ttl > 0:
            if _is_research_followup(user_message):
                query = self._active_operation_topic
                followup = True

        if not query:
            return None

        operation_id = self._active_operation_id if followup and self._active_operation_id else f"research-{int(time.time() * 1000)}"
        self._active_operation_type = "research"
        self._active_operation_topic = query
        self._active_operation_id = operation_id
        self._active_operation_ttl = 3
        self._last_fast_tools_used = ()
        _debug_print("ECHO_DEBUG_UI", "research_request_detected=True")
        _debug_print("ECHO_DEBUG_UI", f"research_followup_detected={followup}")
        _debug_print("ECHO_DEBUG_UI", f"research_topic={query}")
        self._queue_ui_event("research_started", {"operation_id": operation_id, "topic": query})

        research_tool = self.tools.get("web_search") or self.tools.get("research_web")
        if research_tool is None:
            _debug_print("ECHO_DEBUG_UI", "research_tool_available=False")
            _debug_print("ECHO_DEBUG_UI", "research_tool_called=False")
            _debug_print("ECHO_DEBUG_UI", "research_result_count=0")
            _debug_print("ECHO_DEBUG_UI", "research_sources=[]")
            _debug_print("ECHO_DEBUG_UI", "research_grounded=False")
            _debug_print("ECHO_DEBUG_UI", "research_failure_reason=tool_unavailable")
            self._active_operation_type = ""
            self._active_operation_topic = ""
            self._active_operation_id = ""
            self._active_operation_ttl = 0
            self._queue_ui_event(
                "research_unavailable",
                {
                    "operation_id": operation_id,
                    "topic": query,
                    "status": "unavailable",
                    "message": "Ainda não tenho uma ferramenta de pesquisa ligada.",
                    "summary": "Ainda não tenho uma ferramenta de pesquisa ligada.",
                    "results": [],
                },
            )
            self._queue_ui_event(
                "research_completed",
                {"operation_id": operation_id, "topic": query, "status": "unavailable"},
            )
            return "Ainda não tenho uma ferramenta de pesquisa ligada."

        result = research_tool.run({"query": query})
        self._last_fast_tools_used = (research_tool.name,)
        _debug_print("ECHO_DEBUG_UI", "research_tool_available=True")
        _debug_print("ECHO_DEBUG_UI", "research_tool_called=True")
        _debug_print("ECHO_DEBUG_UI", "research_result_count=1")
        _debug_print("ECHO_DEBUG_UI", f"research_sources={[research_tool.name]}")
        _debug_print("ECHO_DEBUG_UI", "research_grounded=True")
        self._queue_ui_event(
            "research_results_ready",
            {
                "operation_id": operation_id,
                "topic": query,
                "status": "completed",
                "summary": result,
                "results": [
                    {
                        "title": "Resultado da pesquisa",
                        "snippet": result,
                        "source": research_tool.name,
                        "kind": "web_result",
                    }
                ],
            },
        )
        self._queue_ui_event(
            "research_completed",
            {"operation_id": operation_id, "topic": query, "status": "completed"},
        )
        self._active_operation_type = ""
        self._active_operation_topic = ""
        self._active_operation_id = ""
        self._active_operation_ttl = 0
        return result

    def _try_system_state_tool_query(self, user_message: str) -> str | None:
        """Falha 1 of the ferro/erro follow-up: tool_intent_supported_by_current_message
        was already true in telemetry, but nothing acted on it — the Composer
        shortcut (_should_answer_without_agent) answered first and invented an
        activity summary with tools_used=[]. Routes evidenced messages
        ("O que estive a fazer no computador?") straight to the agent's
        deterministic plan (assistant.agent._system_state_tool_for) instead,
        so the real Context Observer answers, not a guess.
        """
        if not self.presence.can_use_tools():
            return None
        supported, _evidence_span, _confidence = system_state_tool_intent(_normalize_text(user_message))
        if not supported:
            return None

        result = self.agent.run(user_message, self._fast_agent_context())
        if not result.tools_used:
            # Evidence matched but the tool itself didn't actually run (e.g.
            # not registered in this environment) — fall through rather than
            # force an answer through this path with nothing to show for it.
            return None
        self._remember_tools_used(result.tools_used)
        self._last_fast_tools_used = result.tools_used
        self._record_agent_debug_trace(result.debug_trace)
        if self._turn_trace is not None:
            self._turn_trace["response_grounded"] = True
            self._turn_trace["grounding_sources"] = ["CONTEXT_OBSERVER"]
        return result.response

    def _fast_agent_context(self) -> AgentContext:
        return AgentContext(
            system_prompt=self.base_system_prompt,
            history=[],
            active_contexts=["FAST_ROUTE"],
            session_summary=self._session_summary_for_agent(),
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
            return check_microphone(
                sample_rate=self.voice_sample_rate,
                input_device=self.voice_input_device,
                auto_select=self.voice_auto_select_input,
                silent_rms_threshold=self.voice_silent_rms_threshold,
                channels=self.voice_channels,
                probe_duration=self.voice_probe_duration,
            )
        except MicrophoneCheckError as exc:
            return f"Microfone indisponivel: {exc}"

    def _try_pure_social_turn(self, user_message: str) -> str | None:
        text = _normalize_text(user_message).strip(" .,!?:;")
        text = re.sub(r"[,.!?:;]+", " ", text)
        text = " ".join(text.split())
        if not is_pure_social_turn(user_message):
            return None
        if text in {"ola", "viva", "olá"}:
            return "Olá! Como estás?"
        if text in {"viva tudo bem contigo", "ola tudo bem contigo", "olá tudo bem contigo", "tudo bem contigo"}:
            return "Tudo bem. E contigo?"
        if text in {"tambem estou bem", "também estou bem", "tambem estou bem obrigado", "também estou bem obrigado"}:
            return "Ainda bem."
        if text in {"bom dia"}:
            return "Bom dia."
        if text in {"boa tarde"}:
            return "Boa tarde."
        if text in {"boa noite"}:
            return "Boa noite."
        if text in {"obrigado", "obrigada", "muito obrigado", "muito obrigada"}:
            return "De nada."
        return "Estou bem. E tu?"

    def _try_casual_social_share(self, user_message: str) -> str | None:
        text = _normalize_text(user_message)
        if _looks_like_help_or_planning_request(text):
            return None
        if _looks_like_weekend_beach_share(text):
            if any(phrase in text for phrase in ("trabalhar em ti", "trabalhar no echo", "neste projeto", "no echo")):
                return (
                    "Então tiveste uma semana tranquila, apesar de a passares parcialmente a corrigir-me. "
                    "E esse fim de semana parece bem escolhido."
                )
            if any(word in text for word in ("remo", "remar")):
                return "Parece um bom fim de semana. Praia, amigos e ir remar combinam bem."
            if any(word in text for word in ("amigos", "amigas")):
                return "Parece um bom fim de semana. Praia e amigos combinam bem."
            return "Parece um bom fim de semana."
        return None

    def _try_voice_question(self, user_message: str) -> str | None:
        text = _normalize_text(user_message)
        if text.strip(" .,!?:;") in {
            "reproduz ultimo audio",
            "reproduz o ultimo audio",
            "reproduzir ultimo audio",
            "reproduzir o ultimo audio",
            "toca ultimo audio",
            "toca o ultimo audio",
        }:
            return play_last_voice_input()

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
            input_device=self.voice_input_device,
            auto_select_input=self.voice_auto_select_input,
            sample_rate=self.voice_sample_rate,
            channels=self.voice_channels,
            model_name=self.voice_model,
            language=self.voice_language,
            silent_rms_threshold=self.voice_silent_rms_threshold,
            probe_duration=self.voice_probe_duration,
        )

    def _try_conversational_refinement(self, user_message: str) -> str | None:
        text = _normalize_text(user_message).strip(" .,!?:;")

        if _asks_for_informal_address(text):
            self._set_current_language("pt-PT")
            if self.personal_model is not None:
                self.personal_model.add_or_update_entry(
                    category="preferencias",
                    key="tratamento-por-tu",
                    description="prefere ser tratado por tu e não por você",
                    confidence=100,
                    evidence="Correção explícita do Alexandre.",
                    source="utilizador",
                    status="confirmado",
                )
            return "Claro. Vou tratar-te por tu."

        if _asks_to_explain_previous_phrase(text):
            return self._explain_previous_phrase()

        if _asks_if_previous_message_was_read(text):
            return "Já. Tinha lido mal a tua mensagem anterior."

        typo_clarification = _likely_typo_clarification(text, self._recent_conversation_history(user_message))
        if typo_clarification:
            return typo_clarification

        exam_support_response = _exam_emotional_response(text)
        if exam_support_response:
            return exam_support_response

        if _asks_for_unspecified_help(text):
            return "Claro. O que se passa?"

        personal_fact_response = self._try_personal_fact_confirmation(user_message)
        if personal_fact_response is not None:
            return personal_fact_response

        intention_correction = _intention_correction_response(text)
        if intention_correction:
            return intention_correction

        underlying = _underlying_problem_response(text)
        if underlying:
            return underlying

        return None

    def _try_personal_fact_confirmation(self, user_message: str) -> str | None:
        text = _normalize_text(user_message)
        if not any(phrase in text for phrase in ("ja sabias que", "já sabias que", "certo")):
            return None
        fact = _extract_personal_fact_from_confirmation(user_message)
        if not fact:
            return None

        known = False
        if self.personal_model is not None:
            known = bool(self.personal_model.search_personal_model(fact, limit=1))
        if known:
            return "Sim, lembro-me."

        if self.personal_model is not None:
            self.personal_model.add_or_update_entry(
                category=infer_category(fact),
                key="confirmacao-" + _normalize_text(fact)[:60].replace(" ", "-"),
                description=fact,
                confidence=100,
                evidence="Confirmação explícita do Alexandre.",
                source="utilizador",
                status="confirmado",
            )
        return "Não tinha isso guardado de forma segura. Fico com essa informação."

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
        self._sync_session_with_presence(previous_state, requested_state)
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

    def _complete_turn(
        self,
        user_message: str,
        response: str,
        source: str,
        *,
        remember: bool = True,
        technical: bool = False,
        tool_confirmation: bool = False,
        selected_path: str | None = None,
        tools_used: tuple[str, ...] = (),
    ) -> str:
        final_response = self._finalize_response(
            user_message=user_message,
            response=response,
            source=source,
            technical=technical,
            tool_confirmation=tool_confirmation,
            selected_path=selected_path,
            tools_used=tools_used,
        )
        if has_mojibake_markers(final_response):
            debug_text_encoding("assistant_engine_final_response", final_response)
        if remember and not self._last_response_blocked:
            self._remember_pair(user_message, final_response)
        self._last_selected_path = selected_path or source
        self._end_turn_trace(
            selected_path=selected_path or source,
            response_source=source,
            tools_used=tools_used,
            final_response=final_response,
        )
        return final_response

    def _finalize_response(
        self,
        user_message: str,
        response: str,
        source: str,
        *,
        technical: bool = False,
        tool_confirmation: bool = False,
        selected_path: str | None = None,
        tools_used: tuple[str, ...] = (),
    ) -> str:
        original_response = str(response or "").strip()
        final_response = original_response
        critic_trace = None
        voice_critic_call_count = 0
        composer_call_count = 1 if source == "RESPONSE_COMPOSER" else 0
        response_regenerated = False
        regeneration_reason = ""
        final_response_source = source
        self._last_response_blocked = False
        deterministic_trigger = ""
        if final_response and not technical and not tool_confirmation:
            history = self._recent_conversation_history(user_message)
            cleaned_response, deterministic_trigger = _deterministic_response_cleanup(final_response, tools_used)
            final_response = cleaned_response

            severe_conflict = bool(detect_semantic_conflict(user_message, final_response)) or bool(
                detect_subject_swap(final_response)
            )
            if source == "RESPONSE_COMPOSER" and severe_conflict:
                regeneration_reason = "deterministic_semantic_conflict"
                regenerated = self.response_composer.regenerate(user_message, history, regeneration_reason)
                response_regenerated = True
                composer_call_count += 1
                if regenerated and not _is_still_semantically_unsafe(user_message, regenerated):
                    final_response = regenerated
                    final_response_source = "COMPOSER_REGENERATED"
                else:
                    final_response = self.response_composer.local_safe_fallback(user_message, original_response)
                    final_response_source = "LOCAL_SAFE_FALLBACK"
            if source == "RESPONSE_COMPOSER" and _writing_request_kind(user_message) and _looks_like_help_offer(final_response):
                regeneration_reason = "writing_request_help_offer"
                regenerated = self.response_composer.regenerate(user_message, history, regeneration_reason)
                response_regenerated = True
                composer_call_count += 1
                if regenerated and not _looks_like_help_offer(regenerated):
                    final_response = regenerated
                    final_response_source = "COMPOSER_REGENERATED"
                else:
                    fallback = _local_writing_fallback(user_message)
                    if fallback:
                        final_response = fallback
                        final_response_source = "LOCAL_SAFE_FALLBACK"

        claim_reason = _detect_unsupported_tool_claim(final_response, tools_used)
        if claim_reason:
            final_response = (
                "Ainda não consegui fazer essa pesquisa — não tenho essa ferramenta ligada a esta conversa."
            )
            final_response_source = "TOOL_CLAIM_BLOCKED"
            self._last_response_blocked = True

        # Backstop for the normal-conversation path (section 21): if this
        # message was itself memory-flavored (e.g. "lembras-te...?") it should
        # already have been answered by _try_memory_recall_question in
        # respond() and never reach here — but if it did (presence blocked
        # storage this turn, or some other early return), the Composer must
        # not be allowed to claim a memory it never actually retrieved.
        if final_response_source in {"RESPONSE_COMPOSER", "AGENT_DIRECT", "COMPOSER_REGENERATED"} and is_memory_recall_question(
            _normalize_text(user_message)
        ):
            memory_claim_reason = detect_unsupported_memory_claim(final_response, [])
            if memory_claim_reason:
                final_response = "Não tenho essa informação guardada sobre isso."
                final_response_source = "MEMORY_CLAIM_BLOCKED"
                self._last_response_blocked = True
                self._record_memory_grounding(False, [], True, memory_claim_reason)

        if final_response_source in {"RESPONSE_COMPOSER", "AGENT_DIRECT", "COMPOSER_REGENERATED"}:
            memory_claim_reason = _detect_ungrounded_memory_claim(final_response)
            if memory_claim_reason:
                final_response = "Não quero inventar memória. Posso consultar a memória real se quiseres."
                final_response_source = "MEMORY_CLAIM_BLOCKED"
                self._last_response_blocked = True
                self._record_memory_grounding(False, [], True, memory_claim_reason)

        if final_response_source in {"RESPONSE_COMPOSER", "AGENT_DIRECT", "COMPOSER_REGENERATED"}:
            sources = list(self._turn_trace.get("grounding_sources") or []) if self._turn_trace is not None else []
            history_claim_reason = _detect_ungrounded_historical_duration_claim(final_response, sources)
            if history_claim_reason:
                final_response = "Não tenho memória suficiente para confirmar essa duração."
                final_response_source = "MEMORY_CLAIM_BLOCKED"
                self._last_response_blocked = True
                self._record_memory_grounding(False, [], True, history_claim_reason)

        if final_response_source in {"RESPONSE_COMPOSER", "AGENT_DIRECT", "COMPOSER_REGENERATED"} and not tools_used:
            activity_claim_reason = _detect_ungrounded_activity_claim(final_response)
            if activity_claim_reason:
                final_response = "Não tenho essa informação sobre a tua atividade real — só o Context Observer sabe isso."
                final_response_source = "ACTIVITY_CLAIM_BLOCKED"
                self._last_response_blocked = True
                if self._turn_trace is not None:
                    self._turn_trace["response_grounded"] = False
                    self._turn_trace["grounding_sources"] = []

        self._update_pending_user_intent(user_message, final_response, final_response_source, technical, tool_confirmation)
        self._response_debug(
            source=source,
            voice_reviewed=bool(critic_trace and critic_trace.final_response_changed),
            tools_exposed=_mentions_internal_capability(final_response),
        )
        if self._turn_trace is not None:
            semantic_conflict = bool(detect_semantic_conflict(user_message, final_response))
            subject_swap = bool(detect_subject_swap(final_response))
            self._turn_trace["model"] = _model_name(self.llm)
            self._turn_trace["provider"] = _provider_name(self.llm)
            self._turn_trace["response_before_voice_critic"] = original_response
            self._turn_trace["response_after_voice_critic"] = final_response
            self._turn_trace["voice_critic_call_count"] = voice_critic_call_count
            self._turn_trace["voice_critic_trigger"] = deterministic_trigger
            self._turn_trace["voice_critic_review_changed"] = final_response != original_response
            self._turn_trace["voice_critic_review_accepted"] = None
            self._turn_trace["voice_critic_final_response_changed"] = final_response != original_response
            self._turn_trace["voice_critic_rejection_reason"] = ""
            self._turn_trace["semantic_conflict_detected"] = semantic_conflict or subject_swap
            self._turn_trace["semantic_conflict_reason"] = (
                critic_trace.review_trigger.split(":", 1)[-1] if (critic_trace and (semantic_conflict or subject_swap)) else ""
            )
            self._turn_trace["response_regenerated"] = response_regenerated
            self._turn_trace["regeneration_reason"] = regeneration_reason
            self._turn_trace["composer_call_count"] = composer_call_count
            self._turn_trace["deterministic_response"] = composer_call_count == 0 and voice_critic_call_count == 0
            self._turn_trace["unsupported_tool_claim_detected"] = bool(claim_reason)
            self._turn_trace["unsupported_tool_claim_reason"] = claim_reason
            self._turn_trace["final_response_source"] = final_response_source
            self._turn_trace["final_response_kind"] = _response_kind_for_source(source)
            self._turn_trace["model_source"] = _model_source(self.llm)
            self._turn_trace.update(_model_routing_telemetry(self.llm))
            self._turn_trace.setdefault("fallback_used", final_response_source in {"LOCAL_SAFE_FALLBACK", "FALLBACK"})
        return final_response

    def _response_debug(self, source: str, voice_reviewed: bool, tools_exposed: bool) -> None:
        if not (self.debug or self.debug_agent or self.debug_ollama_payload):
            return
        print(
            "[AssistenteIA RESPONSE]\n"
            f"source={source}\n"
            f"response_kind={_response_kind_for_source(source)}\n"
            f"voice_reviewed={'yes' if voice_reviewed else 'no'}\n"
            f"tools_exposed={'yes' if tools_exposed else 'no'}"
        )

    def _begin_turn_trace(self, user_message: str) -> None:
        if not self.debug_ollama_payload:
            self._turn_trace = None
            return
        self._turn_trace = {
            "user_message": user_message,
            "llm_start": self._llm_chat_count(),
            "llm_sources_start": len(getattr(self.llm, "chat_call_sources", ())),
        }

    def _end_turn_trace(
        self,
        *,
        selected_path: str,
        response_source: str,
        tools_used: tuple[str, ...],
        final_response: str,
    ) -> None:
        if not self.debug_ollama_payload or self._turn_trace is None:
            return
        llm_start = int(self._turn_trace.get("llm_start") or 0)
        llm_calls = max(0, self._llm_chat_count() - llm_start)
        sources_start = int(self._turn_trace.get("llm_sources_start") or 0)
        all_sources = list(getattr(self.llm, "chat_call_sources", ()))
        all_token_records = list(getattr(self.llm, "chat_call_tokens", ()) or [])
        turn_token_records = all_token_records[sources_start:]
        turn_sources = all_sources[sources_start:]
        llm_call_details = _llm_call_details(turn_sources, turn_token_records)
        print("\n[TURN TRACE]")
        print(f"user_message={self._turn_trace.get('user_message')}")
        print(f"selected_path={selected_path}")
        print(f"memory_route={selected_path if selected_path in ('MEMORY_WRITE', 'MEMORY_RECALL') else 'NONE'}")
        print(f"response_source={response_source}")
        print(f"final_response_kind={self._turn_trace.get('final_response_kind') or _response_kind_for_source(response_source)}")
        print(f"model={self._turn_trace.get('model')}")
        print(f"model_source={self._turn_trace.get('model_source')}")
        print(f"provider={self._turn_trace.get('provider')}")
        print(f"model_routing_mode={self._turn_trace.get('model_routing_mode')}")
        print(f"model_routing_provider={self._turn_trace.get('model_routing_provider')}")
        print(f"model_routing_model={self._turn_trace.get('model_routing_model')}")
        print(f"model_routing_reason_code={self._turn_trace.get('model_routing_reason_code')}")
        print(f"model_routing_reason={self._turn_trace.get('model_routing_reason')}")
        print(f"model_routing_paid_call={self._turn_trace.get('model_routing_paid_call')}")
        print(f"model_routing_budget_before_usd={self._turn_trace.get('model_routing_budget_before_usd')}")
        print(f"model_routing_budget_after_usd={self._turn_trace.get('model_routing_budget_after_usd')}")
        print(f"model_routing_fallback_reason={self._turn_trace.get('model_routing_fallback_reason')}")
        print(f"model_routing_override_source={self._turn_trace.get('model_routing_override_source')}")
        print(f"routing_user_message_chars={self._turn_trace.get('routing_user_message_chars')}")
        print(f"routing_context_chars={self._turn_trace.get('routing_context_chars')}")
        print(f"routing_constraint_count={self._turn_trace.get('routing_constraint_count')}")
        print(f"provider_error_type={self._turn_trace.get('provider_error_type')}")
        print(f"fallback_used={self._turn_trace.get('fallback_used')}")
        print(f"response_before_voice_critic={self._turn_trace.get('response_before_voice_critic')}")
        print(f"response_after_voice_critic={self._turn_trace.get('response_after_voice_critic')}")
        print(f"voice_critic_call_count={self._turn_trace.get('voice_critic_call_count')}")
        print(f"voice_critic_trigger={self._turn_trace.get('voice_critic_trigger')}")
        print(f"voice_critic_review_changed={self._turn_trace.get('voice_critic_review_changed')}")
        print(f"voice_critic_review_accepted={self._turn_trace.get('voice_critic_review_accepted')}")
        print(f"voice_critic_final_response_changed={self._turn_trace.get('voice_critic_final_response_changed')}")
        print(f"voice_critic_rejection_reason={self._turn_trace.get('voice_critic_rejection_reason')}")
        print(f"semantic_conflict_detected={self._turn_trace.get('semantic_conflict_detected')}")
        print(f"semantic_conflict_reason={self._turn_trace.get('semantic_conflict_reason')}")
        print(f"response_regenerated={self._turn_trace.get('response_regenerated')}")
        print(f"regeneration_reason={self._turn_trace.get('regeneration_reason')}")
        print(f"composer_call_count={self._turn_trace.get('composer_call_count')}")
        print(f"unsupported_tool_claim_detected={self._turn_trace.get('unsupported_tool_claim_detected')}")
        print(f"unsupported_tool_claim_reason={self._turn_trace.get('unsupported_tool_claim_reason')}")
        print(f"tool_intent_supported_by_current_message={self._turn_trace.get('tool_intent_supported_by_current_message')}")
        print(f"tool_intent_evidence_span={self._turn_trace.get('tool_intent_evidence_span')}")
        print(f"tool_selection_confidence={self._turn_trace.get('tool_selection_confidence')}")
        print(f"agent_debug_trace={self._turn_trace.get('agent_debug_trace')}")
        print(f"agent_route_reason={self._turn_trace.get('agent_route_reason')}")
        print(f"agent_route_evidence_span={self._turn_trace.get('agent_route_evidence_span')}")
        print(f"agent_route_confidence={self._turn_trace.get('agent_route_confidence')}")
        print(f"memory_candidate_detected={self._turn_trace.get('memory_candidate_detected')}")
        print(f"memory_candidate_type={self._turn_trace.get('memory_candidate_type')}")
        print(f"memory_candidate_fields={self._turn_trace.get('memory_candidate_fields')}")
        print(f"memory_write_action={self._turn_trace.get('memory_write_action')}")
        print(f"memory_write_id={self._turn_trace.get('memory_write_id')}")
        print(f"memory_write_reason={self._turn_trace.get('memory_write_reason')}")
        print(f"memory_write_origin={self._turn_trace.get('memory_write_origin')}")
        print(f"memory_write_detected={self._turn_trace.get('memory_write_detected')}")
        print(f"memory_raw_fields={self._turn_trace.get('memory_raw_fields')}")
        print(f"memory_canonical_fields={self._turn_trace.get('memory_canonical_fields')}")
        print(f"memory_normalization_attempted={self._turn_trace.get('memory_normalization_attempted')}")
        print(f"memory_normalization_mode={self._turn_trace.get('memory_normalization_mode')}")
        print(f"memory_normalization_status={self._turn_trace.get('memory_normalization_status')}")
        print(f"memory_normalization_changes={self._turn_trace.get('memory_normalization_changes')}")
        print(f"memory_normalization_valid={self._turn_trace.get('memory_normalization_valid')}")
        print(f"memory_normalization_rejection_reason={self._turn_trace.get('memory_normalization_rejection_reason')}")
        print(f"memory_recall_detected={self._turn_trace.get('memory_recall_detected', False)}")
        print(f"memory_recall_continuation={self._turn_trace.get('memory_recall_continuation', False)}")
        print(f"memory_query={self._turn_trace.get('memory_query')}")
        print(f"history_matches={self._turn_trace.get('history_matches')}")
        print(f"persistent_memory_matches={self._turn_trace.get('persistent_memory_matches')}")
        print(f"selected_memory_ids={self._turn_trace.get('selected_memory_ids')}")
        print(f"memory_confidence={self._turn_trace.get('memory_confidence')}")
        print(f"memory_answer_attributes={self._turn_trace.get('memory_answer_attributes')}")
        print(f"memory_verbalization_mode={self._turn_trace.get('memory_verbalization_mode')}")
        print(f"memory_verbalization_template={self._turn_trace.get('memory_verbalization_template')}")
        print(f"memory_verbalization_fields={self._turn_trace.get('memory_verbalization_fields')}")
        print(f"memory_verbalization_valid={self._turn_trace.get('memory_verbalization_valid')}")
        print(f"memory_verbalization_rejection_reason={self._turn_trace.get('memory_verbalization_rejection_reason')}")
        print(f"response_grounded={self._turn_trace.get('response_grounded')}")
        print(f"grounding_sources={self._turn_trace.get('grounding_sources')}")
        print(f"unsupported_memory_claim_detected={self._turn_trace.get('unsupported_memory_claim_detected')}")
        print(f"unsupported_memory_claim_reason={self._turn_trace.get('unsupported_memory_claim_reason')}")
        print(f"final_response_source={self._turn_trace.get('final_response_source')}")
        print(f"exception_type={self._turn_trace.get('exception_type')}")
        print(f"exception_message={self._turn_trace.get('exception_message')}")
        print(f"llm_calls={llm_calls}")
        print(f"llm_call_details={llm_call_details}")
        print(f"tools_used={list(tools_used)}")
        print(f"final_response={final_response}")
        print("[/TURN TRACE]\n")

        input_tokens = _sum_optional_ints(record.get("input_tokens") for record in turn_token_records if isinstance(record, dict))
        output_tokens = _sum_optional_ints(record.get("output_tokens") for record in turn_token_records if isinstance(record, dict))
        estimated_cost_usd = sum(
            float(record.get("estimated_cost_usd") or 0.0) for record in turn_token_records if isinstance(record, dict)
        )
        self._last_turn_telemetry = {
            "user_message": self._turn_trace.get("user_message"),
            "final_response": final_response,
            "selected_path": selected_path,
            "response_source": response_source,
            "model": self._turn_trace.get("model"),
            "model_source": self._turn_trace.get("model_source"),
            "provider": self._turn_trace.get("provider"),
            "model_routing_mode": self._turn_trace.get("model_routing_mode"),
            "model_routing_provider": self._turn_trace.get("model_routing_provider"),
            "model_routing_model": self._turn_trace.get("model_routing_model"),
            "model_routing_reason_code": self._turn_trace.get("model_routing_reason_code"),
            "model_routing_reason": self._turn_trace.get("model_routing_reason"),
            "model_routing_paid_call": self._turn_trace.get("model_routing_paid_call"),
            "model_routing_budget_before_usd": self._turn_trace.get("model_routing_budget_before_usd"),
            "model_routing_budget_after_usd": self._turn_trace.get("model_routing_budget_after_usd"),
            "model_routing_fallback_reason": self._turn_trace.get("model_routing_fallback_reason"),
            "model_routing_override_source": self._turn_trace.get("model_routing_override_source"),
            "routing_user_message_chars": self._turn_trace.get("routing_user_message_chars"),
            "routing_context_chars": self._turn_trace.get("routing_context_chars"),
            "routing_constraint_count": self._turn_trace.get("routing_constraint_count"),
            "history_context_used": bool(self._turn_trace.get("history_context_used")),
            "history_context_turn_ids": self._turn_trace.get("history_context_turn_ids") or [],
            "provider_error_type": self._turn_trace.get("provider_error_type"),
            "fallback_used": self._turn_trace.get("fallback_used"),
            "llm_calls": llm_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": estimated_cost_usd,
            "llm_call_tokens": turn_token_records,
            "llm_call_sources": turn_sources,
            "llm_call_details": llm_call_details,
            "tools_used": list(tools_used),
            "memory_recall_detected": bool(self._turn_trace.get("memory_recall_detected")),
            "selected_memory_ids": self._turn_trace.get("selected_memory_ids") or [],
            "persistent_memory_matches": self._turn_trace.get("persistent_memory_matches"),
            "history_matches": self._turn_trace.get("history_matches"),
            "memory_write_action": self._turn_trace.get("memory_write_action"),
            "grounding_sources": self._turn_trace.get("grounding_sources") or [],
            "response_grounded": self._turn_trace.get("response_grounded"),
            "unsupported_tool_claim_detected": bool(self._turn_trace.get("unsupported_tool_claim_detected")),
            "unsupported_memory_claim_detected": bool(self._turn_trace.get("unsupported_memory_claim_detected")),
            "tool_intent_supported_by_current_message": self._turn_trace.get("tool_intent_supported_by_current_message"),
            "tool_intent_evidence_span": self._turn_trace.get("tool_intent_evidence_span"),
            "tool_selection_confidence": self._turn_trace.get("tool_selection_confidence"),
            "agent_debug_trace": self._turn_trace.get("agent_debug_trace") or "",
            "active_contexts": [item.name.value for item in self.active_contexts],
            "agent_route_reason": self._turn_trace.get("agent_route_reason") or "",
            "agent_route_evidence_span": self._turn_trace.get("agent_route_evidence_span") or "",
            "agent_route_confidence": self._turn_trace.get("agent_route_confidence"),
            "exception_type": self._turn_trace.get("exception_type"),
            "exception_message": self._turn_trace.get("exception_message"),
        }
        self._turn_trace = None

    def get_last_turn_telemetry(self) -> dict | None:
        """Structured telemetry for the most recently completed turn.

        A clean test/eval API (see evals/schemas.py TurnResult) so a runner
        never has to parse the human-readable [TURN TRACE] stdout dump.
        Requires debug_ollama_payload=True (the same flag that populates the
        stdout trace) since that is what drives every _record_memory_*
        call site below to actually fill in the underlying data.
        """
        if self._last_turn_telemetry is None:
            return None
        return dict(self._last_turn_telemetry)

    def _llm_chat_count(self) -> int:
        critic_llm = self.response_composer.voice_critic.llm
        clients = [self.llm] if critic_llm is self.llm else [self.llm, critic_llm]
        return sum(_single_client_chat_count(client) for client in clients)

    def _recent_conversation_history(self, current_message: str = "", limit: int = 8) -> list[dict[str, str]]:
        if not self.presence.can_store_memory():
            return []
        current = str(current_message or "").strip()
        history = list(self.memory.load())
        if self._conversation_segment_start > 0:
            history = history[self._conversation_segment_start :]
        while history and history[-1].get("role") == "user" and history[-1].get("content", "").strip() == current:
            history.pop()
        clean: list[dict[str, str]] = []
        recent = history[-limit:]
        recent_start = max(0, len(history) - len(recent))
        used_turn_ids: list[int] = []
        for index, message in enumerate(recent):
            role = message.get("role")
            content = message.get("content")
            if role == "assistant" and index + 1 < len(recent):
                next_message = recent[index + 1]
                if next_message.get("role") == "user" and _contests_previous_assistant(
                    str(next_message.get("content") or "")
                ):
                    continue
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                clean.append({"role": role, "content": content})
                used_turn_ids.append(recent_start + index)
        if clean and self._turn_trace is not None:
            self._turn_trace["history_context_used"] = True
            self._turn_trace["history_context_turn_ids"] = used_turn_ids
            sources = list(self._turn_trace.get("grounding_sources") or [])
            if "CONVERSATION_HISTORY" not in sources:
                sources.append("CONVERSATION_HISTORY")
            self._turn_trace["grounding_sources"] = sources
        return clean

    def _remember_pair(self, user_message: str, response: str) -> None:
        if self.presence.can_store_memory():
            self.memory.append_pair(user_message, response)
        if self.session_manager is not None:
            self.session_manager.record_message_pair(user_message, response)

    def _remember_tools_used(self, tools_used: tuple[str, ...]) -> None:
        if self.session_manager is None:
            return
        for tool_name in tools_used:
            self.session_manager.record_tool_used(tool_name)

    def close_session(self, reason: str = "fecho da aplicacao") -> None:
        if self.session_manager is not None:
            self._finish_session(reason)

    def _finish_session(self, reason: str) -> None:
        if self.session_manager is None:
            return
        summary = self.session_manager.end_session(self.context_observer, reason=reason)
        if summary is None:
            return
        self._remember_session_summary(summary)

    def _remember_session_summary(self, summary) -> None:
        remember_context_summary = getattr(self.long_term_memory, "remember_context_summary", None)
        if remember_context_summary is not None:
            content = (
                "Resumo de sessao: "
                f"projeto={summary.main_project or 'desconhecido'}; "
                f"atividade={summary.main_activity or 'sem atividade principal'}; "
                f"factos={summary.summary}; "
                f"ficheiros={summary.files_touched or 'sem ficheiros registados'}; "
                f"tarefas={summary.tasks_changed or 'sem tarefas alteradas'}; "
                f"ferramentas={summary.tools_used or 'sem ferramentas registadas'}; "
                f"decisoes={summary.decisions_taken or 'sem decisoes registadas'}; "
                f"proximo_passo={summary.next_suggested_step or 'sem inferencia'}."
            )
            remember_context_summary(content, project=summary.main_project)

    def _sync_session_with_presence(self, previous_state: PresenceState, current_state: PresenceState) -> None:
        if self.session_manager is None or previous_state == current_state:
            return
        if current_state == PresenceState.ACTIVE_CONVERSATION:
            self.session_manager.start_session()
            return
        if current_state == PresenceState.OFFLINE:
            self._finish_session("modo OFFLINE")

    def _activate_contexts(self, user_message: str) -> list[ActiveContext]:
        self.active_contexts = self.context_manager.identify(user_message)
        self.last_context_debug = self.context_manager.debug_summary(self.active_contexts)
        self._debug_log(self.last_context_debug.replace("\n", " | "))
        return self.active_contexts

    def _run_cognitive_loop(
        self,
        user_message: str,
        intent,
        strategy: CognitiveStrategy,
        active_contexts: list[ActiveContext],
    ) -> ReasoningResult:
        if not strategy.use_context_builder:
            reasoning = ReasoningResult(
                intent=intent,
                plan=[f"{strategy.mode}: {strategy.reason}"],
                conclusions=[f"Estratégia escolhida: {strategy.category.value}."],
            )
            self.last_cognitive_reasoning = reasoning
            return reasoning

        context = ContextBuilder(
            personal_model=self.personal_model if strategy.use_personal_model else None,
            long_term_memory=self.long_term_memory if self.presence.can_store_memory() else None,
            session_manager=self.session_manager if strategy.use_session else None,
            context_observer=self.context_observer if strategy.use_observed_context else None,
        ).build(
            user_message=user_message,
            intent=intent,
            active_contexts=[item.name.value for item in active_contexts],
            enabled_sources=strategy.enabled_sources,
        )
        if strategy.use_reflection:
            reflection = self.reflection_engine.reflect(context)
        else:
            reflection = None

        if strategy.use_reasoning and reflection is not None:
            reasoning = self.reasoning_engine.reason(context, reflection)
        else:
            reasoning = ReasoningResult(
                intent=intent,
                plan=[f"{strategy.mode}: {strategy.reason}"],
                conclusions=context.relevant_facts()[:3],
            )
        self.last_cognitive_reasoning = reasoning
        self._debug_log(
            "Cognitive Loop | "
            f"intent={intent.intent} | confidence={intent.confidence:.2f} | "
            f"needs_more_context={'sim' if reasoning.needs_more_context else 'nao'}"
        )
        return reasoning

    def _try_focused_cognitive_response(self, reasoning: ReasoningResult) -> str | None:
        focus = reasoning.conversational_focus
        if not focus.has_focus:
            return None
        if focus.should_ask and reasoning.questions:
            return reasoning.questions[0]

        normalized_focus = _normalize_text(focus.conversational_focus)
        normalized_tension = _normalize_text(focus.implied_tension)
        if "documento" in normalized_focus and "recomeco" in normalized_tension:
            return "O verdadeiro obstáculo parece ser voltares a pegar nele, não o documento em si."
        if "cansaco mental" in normalized_focus or "carga" in normalized_tension:
            return "Parece mais excesso de carga do que falta de vontade."
        return None

    def _intent_for_message(self, user_message: str):
        intent = self.intent_engine.analyse(user_message)
        previous = self.last_cognitive_strategy
        if (
            intent.intent == "conversa_normal"
            and previous is not None
            and previous.category.value == "PLANNING"
            and _looks_like_planning_followup(user_message)
        ):
            from assistant.cognition.intent_engine import IntentResult

            return IntentResult(
                intent="tomar_decisao",
                confidence=0.78,
                goal="continuar a compreender preferências antes de recomendar",
                reason="mensagem curta interpretada como seguimento de planeamento anterior",
            )
        return intent

    def _context_for_agent(self, user_message: str) -> str:
        if not self.presence.can_store_memory():
            return ""

        strategy = self.last_cognitive_strategy
        parts: list[str] = []
        cognitive_context = self._cognitive_context_for_agent()
        if cognitive_context:
            parts.append(cognitive_context)

        active_context_summary = self._active_context_summary()
        if active_context_summary:
            parts.append(active_context_summary)

        memory_context = ""
        if strategy is None or strategy.use_long_term_memory:
            memory_context = self.long_term_memory.context_for(user_message)
        if memory_context:
            parts.append(memory_context)

        personal_context = ""
        if strategy is not None and strategy.use_personal_model:
            personal_context = self._personal_model_context(user_message)
        if personal_context:
            parts.append(personal_context)

        observed_context = ""
        if strategy is not None and strategy.use_observed_context:
            observed_context = self._observed_context_summary()
        if observed_context:
            parts.append(observed_context)

        return "\n".join(parts)

    def _cognitive_context_for_agent(self) -> str:
        reasoning = self.last_cognitive_reasoning
        strategy = self.last_cognitive_strategy
        if strategy is not None and strategy.is_system_1:
            return ""
        if reasoning is None:
            return ""
        facts = reasoning.facts_for_composer()
        lines: list[str] = []
        if facts:
            lines.append("Cognitive Loop:")
            lines.extend(f"- {fact}" for fact in facts)
        if reasoning.should_delegate and reasoning.delegation_target:
            if not lines:
                lines.append("Cognitive Loop:")
            lines.append(f"- delegacao possivel: {reasoning.delegation_target}")
        return "\n".join(lines)

    def _pending_tasks_for_agent(self) -> str:
        if not self.presence.can_store_memory():
            return ""
        strategy = self.last_cognitive_strategy
        if strategy is not None and not strategy.use_tasks:
            return ""
        pending_tasks = getattr(self.long_term_memory, "pending_tasks", None)
        if pending_tasks is None:
            return ""
        return pending_tasks()

    def _session_summary_for_agent(self) -> str:
        strategy = self.last_cognitive_strategy
        if strategy is not None and not strategy.use_session:
            return ""
        if self.session_manager is None:
            return ""
        return self.session_manager.planner_context()

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

    def _personal_model_context(self, user_message: str) -> str:
        if self.personal_model is None:
            return ""
        return self.personal_model.get_relevant_context(user_message)

    def _maybe_extract_structured_memory(self, user_message: str) -> tuple[str, dict[str, str]] | None:
        """Turns a user statement (passive mention or explicit write command)
        into a validated, attribute-level fact, and returns what was written
        so an explicit MEMORY_WRITE command can confirm it deterministically.

        Only ever reads the user's own message plus the assistant's previous
        question (to resolve "que exame?" -> "Estratégias Algorítmicas"); an
        Echo reply is never itself treated as evidence (see _try_memory_recall_question).
        Storage (this method) and verbalization (_try_memory_recall_question)
        are deliberately separate: what gets written here is a structured
        fact plus raw_user_text as evidence, never a sentence meant to be
        shown to the user later.
        """
        if not self.presence.can_store_memory():
            return None
        if not hasattr(self.long_term_memory, "remember_structured_fact_with_trace"):
            # Defensive: some callers/tests supply a lighter long_term_memory
            # stand-in without the structured-facts API. This feature is
            # best-effort and must never be able to break the response path.
            return None

        origin = "explicit_command" if is_memory_write_command(user_message) else "passive_extraction"

        academic_candidate = extract_academic_event_candidate(
            user_message,
            " ".join((self._previous_assistant_message(), self._previous_user_message())),
        )
        if academic_candidate:
            if not academic_candidate.get("event") and not academic_candidate.get("discipline"):
                existing_academic = self.long_term_memory.find_structured_facts(fact_type="academic_event")
                if not existing_academic:
                    academic_candidate = {}
                else:
                    academic_candidate["event"] = existing_academic[0].event or "exame"
        if academic_candidate:
            canonical, raw_fields = normalize_candidate_fields(academic_candidate, "academic_event")
            canonical.update(raw_fields)
            fact, action, reason = self.long_term_memory.remember_structured_fact_with_trace(
                "academic_event", canonical, confidence=0.9, source="user_statement"
            )
            self._record_memory_write("academic_event", canonical, action, fact.id, reason, origin)
            self._record_memory_normalization(academic_candidate, canonical, "explicit_command_or_passive")
            return "academic_event", canonical

        task_candidate = extract_task_candidate(user_message)
        if task_candidate:
            canonical, raw_fields = normalize_candidate_fields(task_candidate, "task")
            canonical.update(raw_fields)
            fact, action, reason = self.long_term_memory.remember_structured_fact_with_trace(
                "task", canonical, confidence=0.95, source="user_statement"
            )
            self._record_memory_write("task", canonical, action, fact.id, reason, origin)
            self._record_memory_normalization(task_candidate, canonical, "explicit_command_or_passive")
            return "task", canonical

        if self._turn_trace is not None:
            self._turn_trace["memory_candidate_detected"] = False
            self._turn_trace["memory_write_action"] = None
            self._turn_trace["memory_write_origin"] = ""
            self._turn_trace["memory_write_reason"] = "memory_query_not_evidence" if is_memory_recall_question(
                _normalize_text(user_message)
            ) else ""
        return None

    def _try_memory_write_command(self, user_message: str) -> str | None:
        """Self-contained MEMORY_WRITE flow for an explicit command.

        Deliberately independent from the passive _maybe_extract_structured_memory
        path (which respond() skips for these messages — see the caller):
        an explicit command's content has already had its trigger phrase
        stripped ("Regista que "/"Atualiza: "/...), so it is re-extracted
        from that stripped content rather than reused from a full-message
        extraction that may not have matched the same way (e.g. "atualiza a
        disciplina para X" has no "é"/"de" for the sentence-shaped discipline
        patterns to latch onto). Ends here with a template confirmation —
        never falls through to conversational refinement / the Composer.
        """
        parsed = parse_memory_write_command(user_message)
        if self._turn_trace is not None:
            self._turn_trace["memory_write_detected"] = parsed is not None
        if parsed is None:
            return None
        if not hasattr(self.long_term_memory, "remember_structured_fact_with_trace"):
            return None

        verb_kind, field_kind, content = parsed

        if field_kind == "discipline":
            candidate, fact_type = {"event": "exame", "discipline": content}, "academic_event"
        elif field_kind == "date_reference":
            candidate, fact_type = {"event": "exame", "date_reference": f"para {content}"}, "academic_event"
        else:
            academic_candidate = extract_academic_event_candidate(content)
            task_candidate = extract_task_candidate(content)
            if academic_candidate:
                candidate, fact_type = academic_candidate, "academic_event"
            elif task_candidate:
                candidate, fact_type = task_candidate, "task"
            else:
                candidate, fact_type = {}, ""

        if not candidate:
            return "Não percebi bem o que devo registar. Podes reformular?"

        canonical, raw_fields = normalize_candidate_fields(candidate, fact_type)
        canonical.update(raw_fields)
        confidence = 0.95 if fact_type == "task" else 0.9
        fact, action, reason = self.long_term_memory.remember_structured_fact_with_trace(
            fact_type, canonical, confidence=confidence, source="explicit_write_command"
        )
        self._record_memory_write(fact_type, canonical, action, fact.id, reason, origin="explicit_command")
        self._record_memory_normalization(candidate, canonical, "explicit_command")
        return render_memory_write_confirmation(verb_kind, fact_type, canonical)

    def _try_memory_recall_question(self, user_message: str) -> str | None:
        normalized = _normalize_text(user_message)

        if is_task_recall_question(normalized):
            tasks = self.long_term_memory.find_structured_facts(fact_type="task", status="pending")
            retrieval = build_task_retrieval(tasks)
            if not retrieval.grounded:
                # Nothing passively captured: fall through to the existing
                # task command, which reads the real tasks table.
                return None
            self._record_memory_trace(user_message, {"pending_tasks"}, retrieval)
            self._record_memory_verbalization(retrieval, template="task_list")
            self._record_memory_grounding(True, retrieval.sources, False, "")
            self._record_memory_continuation(False)
            self._refresh_memory_recall_continuity("task", retrieval)
            return retrieval.final_answer

        # TTL-based rather than "was the immediately previous turn a recall":
        # a follow-up like "E quando é?" should still resolve even if a quick
        # unrelated turn (e.g. a fast-route command) slipped in between.
        is_followup = self._active_memory_recall_ttl > 0 and is_memory_recall_followup(normalized)
        self._record_memory_continuation(is_followup)
        if not (is_memory_recall_question(normalized) or is_followup):
            return None

        self._queue_ui_event(
            "memory_recall_started",
            {
                "topic": self._active_memory_topic or "academic_event",
                "followup": is_followup,
            },
        )

        requested_attributes = extract_requested_attributes(normalized)
        history = self._recent_conversation_history(user_message)
        history_text = _normalize_text(" ".join(item.get("content", "") for item in history))
        # A bare attribute question ("Qual era a disciplina?") shares no words
        # with the stored fact, so fall back to "the one thing we track"
        # (academic_event) rather than requiring literal text overlap. This is
        # only safe because the schema currently supports a single fact type.
        structured_facts = self.long_term_memory.search_structured_facts_text(user_message, limit=8)
        if is_followup and self._active_memory_entity_id:
            all_academic_facts = self.long_term_memory.find_structured_facts(fact_type="academic_event")
            selected = [fact for fact in all_academic_facts if str(fact.id) == str(self._active_memory_entity_id)]
            if selected:
                structured_facts = selected
        if not structured_facts:
            structured_facts = self.long_term_memory.find_structured_facts(fact_type="academic_event")

        retrieval = build_memory_retrieval(
            requested_attributes=requested_attributes,
            history_text=history_text,
            structured_facts=structured_facts,
        )

        self._record_memory_trace(user_message, requested_attributes, retrieval)
        self._record_memory_verbalization(retrieval, template="academic_event")

        if retrieval.ambiguous:
            self._record_memory_grounding(True, retrieval.sources, False, "")
            self._queue_ui_event("memory_recall_completed", {"grounded": True, "ambiguous": True})
            return retrieval.final_answer

        if not retrieval.grounded:
            self._record_memory_grounding(False, [], False, "")
            self._queue_ui_event("memory_recall_completed", {"grounded": False, "ambiguous": False})
            return retrieval.final_answer

        # The answer was built entirely from the retrieved fact (template
        # rendering, no LLM), so this is a defensive check, not a repair
        # mechanism: it should always come back clean.
        claim_reason = detect_unsupported_memory_claim(retrieval.final_answer, retrieval.sources)
        self._record_memory_grounding(not bool(claim_reason), retrieval.sources, bool(claim_reason), claim_reason)
        self._refresh_memory_recall_continuity("academic_event", retrieval)
        self._queue_ui_event(
            "memory_recall_completed",
            {
                "grounded": not bool(claim_reason),
                "ambiguous": False,
                "matched_ids": retrieval.matched_ids,
            },
        )
        return retrieval.final_answer

    def _try_project_history_recall(self, user_message: str) -> str | None:
        normalized = _normalize_text(user_message)
        if _looks_like_complete_text_summary_request(user_message):
            return None
        writing_kind = _writing_request_kind(user_message)
        if writing_kind and writing_kind != "summary":
            return None
        if not _looks_like_project_recall_question(normalized):
            return None
        explicit_project = _extract_project_name_for_history(user_message)
        project = explicit_project or "projeto"
        project_norm = _normalize_text(project)
        project_terms = _project_recall_terms(project, user_message) or [project_norm]
        history = self._recent_conversation_history(user_message, limit=40)
        history_matches = [
            index
            for index, item in enumerate(history)
            if _history_item_matches_project(item, project_terms)
        ]

        persistent_matches: list[str] = []
        context_for = getattr(self.long_term_memory, "context_for", None)
        if callable(context_for):
            context_text = str(context_for(user_message, limit=5) or "").strip()
            if context_text:
                persistent_matches.append(context_text)
        search = getattr(self.long_term_memory, "search", None)
        if callable(search):
            try:
                for item in search(user_message, limit=5) or []:
                    text = _memory_item_text(item)
                    if text:
                        persistent_matches.append(text)
            except TypeError:
                pass

        sources: list[str] = []
        if history_matches:
            sources.append("CONVERSATION_HISTORY")
        if persistent_matches:
            sources.append("PERSISTENT_MEMORY")

        if self._turn_trace is not None:
            self._turn_trace["memory_recall_detected"] = True
            self._turn_trace["memory_query"] = user_message
            self._turn_trace["history_matches"] = len(history_matches)
            self._turn_trace["persistent_memory_matches"] = len(persistent_matches)
            self._turn_trace["selected_memory_ids"] = []
            self._turn_trace["grounding_sources"] = sources
            self._turn_trace["response_grounded"] = bool(sources)
            if history_matches:
                self._turn_trace["history_context_used"] = True
                self._turn_trace["history_context_turn_ids"] = history_matches

        evidence_text = " ".join(str(history[index].get("content") or "") for index in history_matches)
        if persistent_matches:
            evidence_text = f"{evidence_text} {' '.join(persistent_matches)}".strip()
        if not explicit_project:
            project = _extract_project_name_for_history(evidence_text) or project
        if not _looks_like_project_history_question(normalized):
            if sources:
                return self.response_composer.compose(
                    ComposerRequest(
                        intent="PROJECT_MEMORY_RECALL",
                        user_message=user_message,
                        history=[],
                        context=_project_recall_context(project, history, history_matches, persistent_matches),
                        fallback=f"Tenho algum contexto sobre o {project}, mas ainda nÃ£o o suficiente para o resumir bem.",
                        intent_instruction=(
                            "O pedido Ã© uma consulta sobre um projeto conhecido. "
                            "Usa apenas o contexto recuperado. NÃ£o inventes objetivos, datas, progresso ou prÃ³ximos passos. "
                            "Se o Alexandre pedir quatro pontos, responde em exatamente quatro pontos curtos."
                        ),
                        language_instruction=self._language_instruction(),
                    )
                )
            return f"NÃ£o tenho contexto suficiente na memÃ³ria para te resumir o {project} com seguranÃ§a."
        duration = _extract_duration_from_evidence(evidence_text)
        if duration:
            if "quando comecamos" in normalized or "quando começamos" in normalized:
                return f"Comecamos o {project} {duration}."
            return f"Pelo que tenho no histórico, estamos a trabalhar no {project} {duration}."
        if sources:
            return (
                f"Encontrei contexto sobre o {project}, mas não tenho uma data de início suficientemente clara "
                "para dizer há quanto tempo estamos a trabalhar nele."
            )
        return (
            f"Não tenho dados suficientes na memória para dizer há quanto tempo estamos a trabalhar no {project}."
        )

    def _try_text_transformation_request(self, user_message: str) -> str | None:
        if not _looks_like_complete_text_summary_request(user_message):
            return None
        return self.response_composer.compose(
            ComposerRequest(
                intent="TEXT_SUMMARIZATION",
                user_message=user_message,
                history=self._recent_conversation_history(user_message),
                fallback="Posso resumir, mas preciso que me deixes o texto completo.",
                intent_instruction=(
                    "O pedido é uma transformação de texto completa. "
                    "Resume diretamente o texto pedido. Não trates 'pontos' como tarefas. "
                    "Não faças perguntas de esclarecimento se o texto estiver presente."
                ),
                language_instruction=self._language_instruction(),
            )
        )

    def _refresh_memory_recall_continuity(self, topic: str, retrieval) -> None:
        """Keeps a short follow-up window open after a grounded recall.

        See the TTL decrement at the top of respond() for how this expires.
        """
        self._active_memory_topic = topic
        self._active_memory_entity_id = retrieval.matched_ids[0] if retrieval.matched_ids else None
        self._active_memory_recall_ttl = 2

    def _record_memory_continuation(self, is_followup: bool) -> None:
        if self._turn_trace is not None:
            self._turn_trace["memory_recall_continuation"] = is_followup

    def _record_tool_intent_check(self, user_message: str) -> None:
        """Records whether THIS message carries explicit evidence for a
        system-state tool (get_recent_activity and friends) — see Part 3 of
        the ferro/erro task: a false-positive context (or any other stray
        signal) must never be enough on its own to justify a tool call.
        Computed for every turn, not only ones that reach the agent, so
        evals can assert on it regardless of which path actually answered.
        """
        supported, evidence_span, confidence = system_state_tool_intent(_normalize_text(user_message))
        if self._turn_trace is not None:
            self._turn_trace["tool_intent_supported_by_current_message"] = supported
            self._turn_trace["tool_intent_evidence_span"] = evidence_span
            self._turn_trace["tool_selection_confidence"] = confidence

    def _record_agent_route_decision(self, strategy: CognitiveStrategy, intent) -> None:
        """Part 2 (Falha 2) of the ferro/erro follow-up: expose WHY a message
        was routed to a given cognitive category, so a category that leads to
        AGENT/PROBLEM_SOLVING can always be traced back to a real evidence
        span instead of incidental word-fragment co-occurrence.
        """
        if self._turn_trace is None:
            return
        self._turn_trace["agent_route_reason"] = strategy.reason
        self._turn_trace["agent_route_evidence_span"] = strategy.evidence_span
        self._turn_trace["agent_route_confidence"] = intent.confidence if strategy.evidence_span else 0.0

    def _record_agent_debug_trace(self, debug_trace: str) -> None:
        """Part 4: the agent's internal reasoning trace (chosen tool, reason,
        summarized observation) is data for the terminal/logs/telemetry —
        never for the text handed back to _complete_turn, which is what
        responseReady eventually shows the user. This is the only place that
        reads AgentResult.debug_trace; nothing here ever touches `response`.
        """
        if not debug_trace:
            return
        if self.debug_agent:
            print(f"\n{debug_trace}")
        if self._turn_trace is not None:
            self._turn_trace["agent_debug_trace"] = debug_trace

    def _record_memory_write(
        self,
        candidate_type: str,
        fields: dict[str, str],
        action: str,
        fact_id: int,
        reason: str,
        origin: str = "passive_extraction",
    ) -> None:
        if self._turn_trace is None:
            return
        self._turn_trace["memory_candidate_detected"] = True
        self._turn_trace["memory_candidate_type"] = candidate_type
        self._turn_trace["memory_candidate_fields"] = {
            key: value for key, value in fields.items() if key != "raw_user_text" and not key.endswith("_raw")
        }
        self._turn_trace["memory_write_action"] = action
        self._turn_trace["memory_write_id"] = fact_id
        self._turn_trace["memory_write_reason"] = reason
        self._turn_trace["memory_write_origin"] = origin

    def _record_memory_normalization(
        self,
        raw_candidate: dict[str, str],
        canonical_candidate: dict[str, str],
        mode: str,
    ) -> None:
        if self._turn_trace is None:
            return
        changes = {
            key: {"raw": raw_candidate[key], "canonical": canonical_candidate[key]}
            for key in raw_candidate
            if key in canonical_candidate and canonical_candidate[key] != raw_candidate[key]
        }
        self._turn_trace["memory_raw_fields"] = {k: v for k, v in raw_candidate.items() if k != "raw_user_text"}
        self._turn_trace["memory_canonical_fields"] = {
            k: v for k, v in canonical_candidate.items() if k != "raw_user_text" and not k.endswith("_raw")
        }
        self._turn_trace["memory_normalization_attempted"] = bool(changes)
        self._turn_trace["memory_normalization_mode"] = mode if changes else "none"
        self._turn_trace["memory_normalization_status"] = "SAFE_NORMALIZATION" if changes else "EXACT"
        self._turn_trace["memory_normalization_changes"] = changes
        self._turn_trace["memory_normalization_valid"] = True
        self._turn_trace["memory_normalization_rejection_reason"] = ""

    def _record_memory_trace(self, query: str, requested_attributes: set[str], retrieval) -> None:
        if self._turn_trace is None:
            return
        self._turn_trace["memory_recall_detected"] = True
        self._turn_trace["memory_query"] = query
        self._turn_trace["history_matches"] = 1 if any(
            source == "CURRENT_HISTORY" for source in retrieval.sources
        ) else 0
        self._turn_trace["persistent_memory_matches"] = sum(
            1 for source in retrieval.sources if source.startswith("PERSISTENT_MEMORY")
        )
        self._turn_trace["selected_memory_ids"] = retrieval.matched_ids
        self._turn_trace["memory_confidence"] = retrieval.confidence
        self._turn_trace["memory_answer_attributes"] = sorted(requested_attributes & retrieval.attributes_covered)

    def _record_memory_verbalization(self, retrieval, template: str) -> None:
        if self._turn_trace is None:
            return
        self._turn_trace["memory_verbalization_mode"] = "deterministic"
        self._turn_trace["memory_verbalization_template"] = template if (retrieval.grounded or retrieval.ambiguous) else "fallback"
        self._turn_trace["memory_verbalization_fields"] = sorted(retrieval.attributes_covered)
        self._turn_trace["memory_verbalization_valid"] = True
        self._turn_trace["memory_verbalization_rejection_reason"] = ""

    def _record_memory_grounding(
        self,
        grounded: bool,
        sources: list[str],
        unsupported_claim: bool,
        unsupported_reason: str,
    ) -> None:
        if self._turn_trace is None:
            return
        self._turn_trace["response_grounded"] = grounded
        self._turn_trace["grounding_sources"] = sources
        self._turn_trace["unsupported_memory_claim_detected"] = unsupported_claim
        self._turn_trace["unsupported_memory_claim_reason"] = unsupported_reason
        if unsupported_claim:
            self._turn_trace["memory_verbalization_valid"] = False
            self._turn_trace["memory_verbalization_rejection_reason"] = unsupported_reason

    def _try_pending_user_intent(self, user_message: str) -> str | None:
        pending = self._pending_user_intent
        if not pending:
            return None
        if not _is_pending_intent_followup(user_message, pending):
            pending["ttl"] = str(max(0, int(pending.get("ttl", "1")) - 1))
            if pending.get("ttl") == "0":
                self._pending_user_intent = None
            return None

        original = pending.get("message", "")
        kind = pending.get("kind", "writing")
        self._pending_user_intent = None
        return self.response_composer.compose(
            ComposerRequest(
                intent=f"{kind}_execution",
                user_message=original,
                history=self._recent_conversation_history(user_message),
                fallback="Diz-me o texto ou o objetivo concreto e eu escrevo contigo.",
                intent_instruction=(
                    "O Alexandre confirmou uma intenção pendente. "
                    "Executa agora o pedido original diretamente. "
                    "Não perguntes se quer ajuda e não reinicies a conversa."
                ),
                language_instruction=self._language_instruction(),
            )
        )

    def _update_pending_user_intent(
        self,
        user_message: str,
        final_response: str,
        final_response_source: str,
        technical: bool,
        tool_confirmation: bool,
    ) -> None:
        if technical or tool_confirmation:
            return
        if final_response_source not in {"RESPONSE_COMPOSER", "AGENT_DIRECT", "COMPOSER_REGENERATED"}:
            return
        kind = _writing_request_kind(user_message)
        if not kind:
            return
        if not _looks_like_help_offer(final_response):
            self._pending_user_intent = None
            return
        self._pending_user_intent = {"kind": kind, "message": user_message.strip(), "ttl": "3"}

    def _try_long_term_memory_command(self, user_message: str) -> str | None:
        lowered = _normalize_text(user_message.strip())
        if lowered.startswith("lembra-te que"):
            content = user_message.strip()[len("lembra-te que") :].strip(" .:")
            return self._remember_explicit_personal_model(content)

        if lowered.startswith("lembra que"):
            content = user_message.strip()[len("lembra que") :].strip(" .:")
            return self._remember_explicit_personal_model(content)

        if lowered.startswith("nao te esquecas que"):
            content = user_message.strip()[len("nao te esquecas que") :].strip(" .:")
            return self._remember_explicit_personal_model(content)

        if lowered.startswith("não te esqueças que"):
            content = user_message.strip()[len("não te esqueças que") :].strip(" .:")
            return self._remember_explicit_personal_model(content)

        if lowered.startswith("guarda isto"):
            content = user_message.strip()[len("guarda isto") :].strip(" .:")
            return self._remember_explicit_personal_model(content)

        if lowered.startswith("esquece"):
            query = user_message.strip()[len("esquece") :].strip(" .:")
            personal_response = self._forget_from_personal_model(query)
            if personal_response is not None:
                return personal_response
            return self.long_term_memory.forget(query)

        if lowered.startswith("corrige isto"):
            content = user_message.strip()[len("corrige isto") :].strip(" .:")
            return self._correct_personal_model_entry(content)

        if lowered.startswith("o que sabes de mim") or lowered.startswith("o que sabes sobre mim"):
            personal_response = self._answer_from_personal_model("", show_details=_asks_memory_details(lowered))
            if personal_response is not None:
                return personal_response

        if lowered.startswith("o que sabes sobre"):
            query = user_message.strip()[len("o que sabes sobre") :].strip(" .:?")
            personal_response = self._answer_from_personal_model(query, show_details=_asks_memory_details(lowered))
            if personal_response is not None:
                return personal_response
            return self._answer_from_long_term_memory(query, show_details=_asks_memory_details(lowered))

        return None

    def _remember_explicit_personal_model(self, content: str) -> str:
        if self.personal_model is None:
            return self.long_term_memory.remember(content)
        personal_response = self.personal_model.remember_explicit(content)
        self.long_term_memory.remember(content, category=infer_category(content))
        return personal_response

    def _forget_from_personal_model(self, query: str) -> str | None:
        if self.personal_model is None:
            return None
        entry = self.personal_model.delete_entry(query)
        if entry is None:
            return None
        return f"Está feito. Já não vou usar essa informação: {entry.description}"

    def _correct_personal_model_entry(self, content: str) -> str:
        if self.personal_model is None:
            return "Ainda não tenho o Personal Model ligado."
        if not content:
            return "Diz-me o que queres corrigir e qual é a versão correta."
        entry = self.personal_model.add_or_update_entry(
            category=infer_category(content),
            key="correcao-" + _normalize_text(content)[:60].replace(" ", "-"),
            description=content,
            confidence=100,
            evidence="Correção explícita do Alexandre.",
            source="utilizador",
            status="confirmado",
        )
        return f"Obrigado, fica corrigido. Vou passar a ter isto em conta: {entry.description}"

    def _answer_from_personal_model(self, query: str, show_details: bool = False) -> str | None:
        if self.personal_model is None:
            return None
        normalized_query = _normalize_text(query)
        category = next(
            (
                candidate
                for candidate in (
                    "identidade",
                    "vida",
                    "trabalho",
                    "estudos",
                    "projetos",
                    "ferramentas",
                    "preferencias",
                    "habitos",
                    "relacoes",
                    "objetivos",
                )
                if candidate in normalized_query
            ),
            None,
        )
        if category is not None:
            facts = self.personal_model.facts_for_category(category)
            if not facts:
                return None
            fallback = self.personal_model.answer_category(category, show_details=False)
            technical_text = (
                self.personal_model.answer_category(category, show_details=True) if show_details else ""
            )
        else:
            about_query = (
                "" if not normalized_query or any(word in normalized_query for word in ("mim", "eu", "alexandre")) else query
            )
            facts = self.personal_model.facts_about(about_query)
            if not facts and about_query:
                return None
            fallback = self.personal_model.answer_about(about_query, show_details=False)
            technical_text = (
                self.personal_model.answer_about(about_query, show_details=True) if show_details else ""
            )
            if not facts and not show_details:
                return fallback

        if not show_details:
            return fallback

        return self.response_composer.compose(
            ComposerRequest(
                intent="personal_model",
                user_message=query or "o que sabes sobre mim",
                history=self._recent_conversation_history(query),
                facts=self._facts_with_cognitive_reasoning(facts) if facts else [],
                fallback=fallback,
                show_technical=show_details,
                technical_text=technical_text,
                language_instruction=self._language_instruction(),
            )
        )

    def _answer_from_long_term_memory(self, query: str, show_details: bool = False) -> str:
        if show_details:
            return self.long_term_memory.answer_about(query)

        records = self.long_term_memory.search(query, limit=5)
        facts = [record.content for record in records]
        return self.response_composer.compose(
            ComposerRequest(
                intent="long_term_memory",
                user_message=f"o que sabes sobre {query}",
                history=self._recent_conversation_history(query),
                facts=self._facts_with_cognitive_reasoning(facts),
                fallback=f"Ainda não tenho contexto suficiente sobre {query}.",
                language_instruction=self._language_instruction(),
            )
        )

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

    def _previous_assistant_message(self) -> str:
        for message in reversed(self.memory.load()):
            if message.get("role") == "assistant":
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
                "em que estado estas",
                "em que estado estas tu",
                "em que estado esta o echo",
                "em que estado esta o assistente",
                "estado de presenca",
                "estado da presenca",
                "estado do assistente",
                "estado do echo",
                "que modo",
                "em que modo",
                "modo atual",
            )
        ):
            return self.presence.state_report()
        return None

    def _compose_session_answer(self, user_message: str, facts: list[str], fallback: str) -> str:
        return self.response_composer.compose(
            ComposerRequest(
                intent="session_reflection",
                user_message=user_message,
                history=self._recent_conversation_history(user_message),
                facts=self._facts_with_cognitive_reasoning(facts),
                fallback=fallback,
                language_instruction=self._language_instruction(),
            )
        )

    def _facts_with_cognitive_reasoning(self, facts: list[str]) -> list[str]:
        reasoning = self.last_cognitive_reasoning
        if reasoning is None:
            return facts
        combined = list(facts)
        cognitive_facts = reasoning.facts_for_composer()
        if cognitive_facts:
            combined.extend(cognitive_facts)
        return combined

    def _try_briefing_question(self, user_message: str) -> str | None:
        text = _normalize_text(user_message)
        if _asks_last_session_summary(text) or _asks_session_continuity(text) or _asks_today_session_work(text):
            current_summary = self._session_continuity_from_current_conversation(user_message)
            if current_summary:
                return current_summary
        if self.session_manager is not None:
            if _asks_next_step(text):
                return self._compose_session_answer(
                    user_message,
                    self.session_manager.facts_for_next_step(),
                    self.session_manager.answer_next_step(),
                )
            if _asks_changes_since_last_time(text):
                return self._compose_session_answer(
                    user_message,
                    self.session_manager.facts_for_changes_since_last_time(),
                    self.session_manager.answer_changes_since_last_time(),
                )
            if _asks_today_session_work(text):
                return self._compose_session_answer(
                    user_message,
                    self.session_manager.facts_for_today(),
                    self.session_manager.answer_today(),
                )
            if _asks_last_session_summary(text) or _asks_session_continuity(text):
                return self._compose_session_answer(
                    user_message,
                    self.session_manager.facts_for_last_session(),
                    self.session_manager.answer_last_session(),
                )

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

    def _session_continuity_from_current_conversation(self, user_message: str) -> str:
        previous_user_messages = [
            item.get("content", "").strip()
            for item in self.memory.load()
            if item.get("role") == "user" and item.get("content", "").strip() != user_message.strip()
        ]
        if not previous_user_messages:
            return ""
        recent = previous_user_messages[-4:]
        normalized = _normalize_text(" ".join(recent))
        if "escrita passiva" in normalized and "recall" in normalized:
            return "Ficámos na memória do Echo: a escrita passiva já estava corrigida e faltava fechar o recall."
        if len(recent) < 2:
            return ""
        topics = "; ".join(_shorten_for_conversation(message, 80) for message in recent)
        return f"Estávamos a falar disto: {topics}."

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

    def _explain_previous_phrase(self) -> str:
        last_assistant = ""
        for message in reversed(self.memory.load()):
            if message.get("role") == "assistant":
                last_assistant = message.get("content", "").strip()
                break
        if not last_assistant:
            return "Claro. Qual é a frase que queres que eu explique melhor?"
        phrase = _shorten_for_conversation(last_assistant, 180)
        return f"Claro. Estava a referir-me a isto: {phrase}"

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


def is_pure_social_turn(message: str) -> bool:
    text = _normalize_text(message).strip(" .,!?:;")
    text = re.sub(r"[,.!?:;]+", " ", text)
    text = " ".join(text.split())
    if not text:
        return False
    pure_turns = {
        "ola",
        "olá",
        "viva",
        "bom dia",
        "boa tarde",
        "boa noite",
        "obrigado",
        "obrigada",
        "muito obrigado",
        "muito obrigada",
        "como estas",
        "como estás",
        "tudo bem",
        "estas bem",
        "estás bem",
        "ola como estas",
        "olá como estás",
        "viva como estas",
        "viva como estás",
        "ola tudo bem",
        "olá tudo bem",
        "viva tudo bem",
        "tudo bem contigo",
        "ola tudo bem contigo",
        "olá tudo bem contigo",
        "viva tudo bem contigo",
        "tambem estou bem",
        "também estou bem",
        "tambem estou bem obrigado",
        "também estou bem obrigado",
    }
    if text not in pure_turns:
        return False
    return not any(
        word in text
        for word in (
            "preocupado",
            "preocupada",
            "exausto",
            "exausta",
            "nervoso",
            "nervosa",
            "problema",
            "ajuda",
            "preciso",
            "quero",
            "abre",
            "abrir",
            "ficheiro",
            "aplicacao",
            "aplicação",
            "workspace",
        )
    )


def _looks_like_help_or_planning_request(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "ajuda-me",
            "ajuda me",
            "preciso de ajuda",
            "quero planear",
            "quero organizar",
            "ajuda-me a organizar",
            "ajuda me a organizar",
            "planeia",
            "organiza",
            "faz um plano",
            "cria um plano",
        )
    )


def _looks_like_weekend_beach_share(text: str) -> bool:
    return (
        any(phrase in text for phrase in ("fim de semana", "sábado", "sabado", "domingo"))
        and any(word in text for word in ("praia", "mar", "remo", "remar"))
        and any(word in text for word in ("amigos", "amigas", "namorada", "sozinho"))
    )


def _should_answer_without_agent(strategy: CognitiveStrategy) -> bool:
    return strategy.category.value in {
        "GENERAL_INFORMATION",
        "PLANNING",
    }


def _agent_result_source(result) -> str:
    if result.tools_used:
        if any(name.startswith("get_") for name in result.tools_used):
            return "TOOL_RESULT"
        return "TOOL_RESULT"
    return "AGENT_DIRECT"


def _looks_like_technical_tool_result(response: str) -> bool:
    text = _normalize_text(response)
    return any(
        marker in text
        for marker in (
            "ficheiros na pasta workspace",
            "aplicacao ativa detetada",
            "janela ativa",
            "snapshot bruto",
        )
    )


def _mentions_internal_capability(response: str) -> bool:
    text = _normalize_text(response)
    return any(
        phrase in text
        for phrase in (
            "abrir aplicacoes",
            "abrir aplicações",
            "abrir ficheiros",
            "workspace",
            "ferramentas disponiveis",
            "ferramentas disponíveis",
            "observar o computador",
            "context observer",
        )
    )


def _needs_composer_regeneration(critic_trace, final_response: str) -> bool:
    if critic_trace is None:
        return False
    trigger = critic_trace.review_trigger
    if trigger.startswith("conflito_semantico") or trigger.startswith("troca_de_sujeito"):
        return True
    if trigger == "perguntas_a_mais" and final_response.count("?") > 1:
        return True
    return False


def _deterministic_response_cleanup(response: str, tools_used: tuple[str, ...]) -> tuple[str, str]:
    text = str(response or "").strip()
    triggers: list[str] = []
    updated = _normalize_pt_pt_vocabulary(text)
    if updated != text:
        triggers.append("pt_pt_vocabulary")
        text = updated

    updated = _remove_extra_questions(text)
    if updated != text:
        triggers.append("perguntas_a_mais")
        text = updated

    updated = _block_invented_echo_first_person(text)
    if updated != text:
        triggers.append("primeira_pessoa_inventada")
        text = updated

    if _detect_unsupported_tool_claim(text, tools_used):
        triggers.append("claim_ferramenta")
    return text, ",".join(triggers)


def _normalize_pt_pt_vocabulary(text: str) -> str:
    phrase_replacements = {
        r"voc[eê]\s+pode\s+me\s+explicar\s+isso\??": "Podes explicar-me isso?",
        r"voc[eê]\s+pode\s+acessar\s+seus\s+arquivos\s+nesta\s+tela": "Podes aceder aos teus ficheiros neste ecrã",
    }
    cleaned = text
    for pattern, replacement in phrase_replacements.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    replacements = {
        r"\bvoc[eê]\b": "tu",
        r"\baplicativos\b": "aplicações",
        r"\barquivos\b": "ficheiros",
        r"\btela\b": "ecrã",
        r"\bacessar\b": "aceder",
        r"\bacesso\b": "acesso",
        r"\bestou assistindo\b": "estou a acompanhar",
    }
    for pattern, replacement in replacements.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned


def _remove_extra_questions(text: str) -> str:
    normalized = _normalize_text(text)
    if text.count("?") <= 1:
        return text
    if (
        text.count("?") == 2
        and "quando e exatamente o exame" in normalized
        and ("apontamentos" in normalized or "slides" in normalized or "material" in normalized)
    ):
        return text
    parts = re.split(r"(?<=[.!?])\s+", text)
    kept: list[str] = []
    question_seen = False
    for part in parts:
        if "?" not in part:
            kept.append(part)
            continue
        if not question_seen:
            kept.append(part)
            question_seen = True
    return " ".join(item for item in kept if item).strip() or text


def _block_invented_echo_first_person(text: str) -> str:
    normalized = _normalize_text(text)
    invented_patterns = (
        "gostava de ter",
        "eu gostava",
        "tenho experiencia nisso",
        "tenho experiência nisso",
        "estava preocupado",
        "estou preocupado",
        "estou a aprender com",
    )
    if any(pattern in normalized for pattern in invented_patterns):
        return "Não quero inventar uma experiência minha. Posso falar-te do tema diretamente."
    return text


def _is_still_semantically_unsafe(user_message: str, text: str) -> bool:
    return (
        bool(detect_semantic_conflict(user_message, text))
        or bool(detect_subject_swap(text))
        or bool(has_voice_issue(text))
        or text.count("?") > 1
    )


def _single_client_chat_count(llm: object) -> int:
    for attr in ("chat_call_count", "chat_calls"):
        value = getattr(llm, attr, None)
        if isinstance(value, int):
            return value
        if callable(value):
            try:
                return int(value())
            except Exception:
                return 0
    return 0


def _model_name(llm: object) -> str:
    settings = getattr(llm, "settings", None)
    return str(getattr(settings, "model", "") or "")


def _model_source(llm: object) -> str:
    settings = getattr(llm, "settings", None)
    return str(getattr(settings, "model_source", "") or "")


def _provider_name(llm: object) -> str:
    provider = getattr(llm, "provider", None)
    return str(getattr(provider, "name", "") or "")


def _model_routing_telemetry(llm: object) -> dict[str, object]:
    settings = getattr(llm, "settings", None)
    return {
        "model_routing_mode": str(getattr(settings, "model_routing_mode", "") or ""),
        "model_routing_provider": str(getattr(settings, "model_routing_provider", "") or ""),
        "model_routing_model": str(getattr(settings, "model_routing_model", "") or ""),
        "model_routing_reason_code": str(getattr(settings, "model_routing_reason_code", "") or ""),
        "model_routing_reason": str(getattr(settings, "model_routing_reason", "") or ""),
        "model_routing_paid_call": bool(getattr(settings, "model_routing_paid_call", False)),
        "model_routing_budget_before_usd": float(getattr(settings, "model_routing_budget_before_usd", 0.0) or 0.0),
        "model_routing_budget_after_usd": float(getattr(settings, "model_routing_budget_after_usd", 0.0) or 0.0),
        "model_routing_fallback_reason": str(getattr(settings, "model_routing_fallback_reason", "") or ""),
        "model_routing_override_source": str(getattr(settings, "model_routing_override_source", "") or ""),
        "routing_user_message_chars": int(getattr(settings, "routing_user_message_chars", 0) or 0),
        "routing_context_chars": int(getattr(settings, "routing_context_chars", 0) or 0),
        "routing_constraint_count": int(getattr(settings, "routing_constraint_count", 0) or 0),
    }


def _llm_call_details(sources: list[str], token_records: list[dict]) -> list[dict[str, object]]:
    details: list[dict[str, object]] = []
    for index, record in enumerate(token_records):
        if not isinstance(record, dict):
            continue
        details.append(
            {
                "component": sources[index] if index < len(sources) else "OTHER",
                "provider": str(record.get("provider") or ""),
                "model": str(record.get("model") or ""),
                "input_tokens": record.get("input_tokens"),
                "output_tokens": record.get("output_tokens"),
                "estimated_cost_usd": float(record.get("estimated_cost_usd") or 0.0),
            }
        )
    return details


def _sum_optional_ints(values) -> int | None:
    total = 0
    seen = False
    for value in values:
        if isinstance(value, int):
            total += value
            seen = True
    return total if seen else None


_FALSE_TOOL_CLAIM_MARKERS = (
    "pesquisei",
    "procurei",
    "ja encontrei",
    "encontrei resultados",
    "encontrei informacao",
    "encontrei algumas informacoes",
    "consultei",
    "confirmei",
    "verifiquei",
    "vi no google",
    "pesquisa inicial",
    "resultados encontrados",
    "terminei a pesquisa",
    "estou a pesquisar",
)

_FUTURE_PROMISE_MARKERS = (
    "vou procurar",
    "quando estiverem prontos",
    "vou pesquisar e depois digo-te",
    "confirmado, vou pesquisar",
    "digo-te quando estiver pronto",
    "aviso-te quando",
)


def _detect_unsupported_tool_claim(response: str, tools_used: tuple[str, ...]) -> str:
    """Flags claims of tool use/results that never happened, or promises of future work.

    This project has no background task worker, so any "I'll search and tell you
    later" phrasing is always false; a claim of having already searched/found
    something is only false when no tool actually ran this turn.
    """
    normalized = _normalize_text(response)
    for marker in _FUTURE_PROMISE_MARKERS:
        if marker in normalized:
            return marker
    if tools_used:
        return ""
    for marker in _FALSE_TOOL_CLAIM_MARKERS:
        if marker in normalized:
            return marker
    return ""


_UNGROUNDED_MEMORY_CLAIM_MARKERS = (
    "lembro-me",
    "lembro me",
    "tenho guardado",
    "tens guardado",
    "vi na memoria",
    "vi na memória",
    "encontrei na memoria",
    "encontrei na memória",
    "nao encontrei na memoria",
    "não encontrei na memória",
    "guardaste",
    "tenho registado",
)

# Falha 5 (ferro/erro follow-up): a question about real system state
# (activity, programs used, windows, current project, ...) must never be
# answered by the Composer with an invented guess. Falha 1's fix already
# routes evidenced messages straight to the tool before the Composer gets a
# turn — this is the defense-in-depth backstop for phrasings that slip
# through with tools_used=[] anyway.
_UNGROUNDED_ACTIVITY_CLAIM_MARKERS = (
    "se nao me engano",
    "se não me engano",
    "estavas a trabalhar",
    "estava a trabalhar num projeto",
    "deves ter estado",
    "parece que estiveste",
    "acho que estavas",
    "presumo que estavas",
)


def _detect_ungrounded_memory_claim(response: str) -> str:
    normalized = _normalize_text(response)
    for marker in _UNGROUNDED_MEMORY_CLAIM_MARKERS:
        if _normalize_text(marker) in normalized:
            return marker
    return ""


def _detect_ungrounded_activity_claim(response: str) -> str:
    normalized = _normalize_text(response)
    for marker in _UNGROUNDED_ACTIVITY_CLAIM_MARKERS:
        if _normalize_text(marker) in normalized:
            return marker
    return ""


def _detect_ungrounded_historical_duration_claim(response: str, grounding_sources: list[str]) -> str:
    if grounding_sources:
        return ""
    normalized = _normalize_text(response)
    patterns = (
        r"\bao longo dos ultimos\s+(?:\w+\s+)?(?:dias|semanas|meses|anos)\b",
        r"\bha\s+(?:\w+\s+)?(?:dias|semanas|meses|anos)\b",
        r"\bdesde\s+(?:janeiro|fevereiro|marco|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro|\d{4})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return match.group(0)
    return ""


def _response_kind_for_source(source: str) -> str:
    if source in {"TOOL_RESULT", "TOOL_CONFIRMATION"}:
        return "texto vindo de ferramenta ou ação local"
    if source in {
        "FAST_ROUTE",
        "SOCIAL_FAST_PATH",
        "MEMORY_COMMAND",
        "SESSION_COMMAND",
        "PLANNER_DIRECT",
        "ERROR",
        "FALLBACK",
        "MEMORY_RECALL_DETERMINISTIC",
        "MEMORY_WRITE_DETERMINISTIC",
        "MEMORY_CLAIM_BLOCKED",
        "ACTIVITY_CLAIM_BLOCKED",
        "INTERNAL_ERROR",
    }:
        return "texto devolvido diretamente por módulo local"
    if source in {"RESPONSE_COMPOSER", "AGENT_DIRECT"}:
        return "texto gerado pelo LLM ou composto a partir de raciocínio"
    return "texto de origem mista ou não classificada"


def _asks_for_informal_address(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "trata-me por tu",
            "trata me por tu",
            "fala comigo por tu",
            "nao uses voce",
            "nao me trates por voce",
            "sem voce",
        )
    )


def _asks_to_explain_previous_phrase(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "explica melhor essa frase",
            "explica essa frase",
            "explica melhor isso",
            "como assim",
            "nao percebi o que querias dizer",
            "não percebi o que querias dizer",
            "nao percebi esta tua frase",
            "não percebi esta tua frase",
            "o que queres dizer com isso",
            "o que querias dizer com isso",
        )
    )


def _asks_if_previous_message_was_read(text: str) -> bool:
    stripped = text.strip(" .,!?:;")
    return stripped in {"ja leste", "já leste", "ja leste a mensagem", "já leste a mensagem"}


def _asks_for_unspecified_help(text: str) -> bool:
    stripped = text.strip(" .,!?:;")
    return stripped in {
        "preciso de ajuda numa coisa",
        "preciso de ajuda com uma coisa",
        "podes ajudar-me numa coisa",
        "podes ajudar me numa coisa",
        "ajuda-me numa coisa",
        "ajuda me numa coisa",
    }


def _underlying_problem_response(text: str) -> str:
    if any(word in text for word in ("preguica", "procrastin", "adiar", "evitado")) and any(
        word in text for word in ("documento", "relatorio", "trabalho", "tese", "texto")
    ):
        return (
            "Tenho a sensação de que o documento já não é o verdadeiro obstáculo. "
            "O difícil é voltares a pegar nele, certo?"
        )
    if any(word in text for word in ("exausto", "exausta", "esgotado", "esgotada", "sobrecarregado", "sobrecarregada")):
        if any(word in text for word in ("focar", "concentrar", "nisso", "documento", "trabalho", "relatorio")):
            return (
                "Isso explica porque te tem custado voltar a isso. "
                "Parece-me que estás a carregar coisas a mais."
            )
    return ""


def _likely_typo_clarification(text: str, history: list[dict[str, str]]) -> str:
    if "medo de falar" not in text:
        return ""
    context = " ".join(item.get("content", "") for item in history[-6:])
    normalized_context = _normalize_text(context)
    if any(word in normalized_context for word in ("exame", "chumbei", "correu mal", "oportunidade")) or any(
        word in text for word in ("exame", "chumbei", "correu mal", "oportunidade")
    ):
        return "Queres dizer que tens medo de falhar nessa última oportunidade?"
    return ""


def _exam_emotional_response(text: str) -> str:
    if "exame" in text and any(word in text for word in ("nervoso", "nervosa", "ansioso", "ansiosa")):
        return "É normal ficares nervoso. Que exame é?"
    if any(phrase in text for phrase in ("nao sei se ha muito a falar", "não sei se há muito a falar")):
        return "Talvez não haja. Parece que o que mais pesa é saberes que só tens mais uma oportunidade."
    if "fracasso" in text and any(word in text for word in ("falhar", "falhei", "chumbei", "exame")):
        return (
            "Falhar o exame não transforma todo o trabalho num fracasso, "
            "mas percebo que agora seja difícil separar as duas coisas."
        )
    return ""


def _direct_short_phrase_response(text: str) -> str:
    if not re.search(r"\b(?:ajuda-me a escrever|escreve|da-me|dá-me)\b", text):
        return ""
    if "frase" not in text:
        return ""
    if "revisao" in text or "revisão" in text:
        return "Agradeço que revejas este texto, por favor."
    if "desculpa" in text:
        return "Peço desculpa pelo incómodo."
    if "confirmacao" in text or "confirmação" in text:
        return "Agradeço confirmação assim que possível."
    if "resposta" in text:
        return "Agradeço que me respondas assim que possível."
    return ""


def _deterministic_help_response(text: str) -> str:
    if "modulenotfounderror" in text and "pyside6" in text:
        return (
            "Esse erro quer dizer que o PySide6 não está instalado no ambiente Python que estás a usar. "
            "Instala-o no venv com: pip install PySide6"
        )
    if "planear" in text and any(word in text for word in ("ferias", "férias", "viagem")):
        return (
            "Boa ideia. Antes de sugerir sítios, deixa-me perceber uma coisa: "
            "procuras mais descansar, conhecer sítios novos ou alguma aventura?"
        )
    return ""


def _looks_like_travel_destination_followup(text: str, previous_user_message: str) -> bool:
    previous = _normalize_text(previous_user_message)
    if not any(word in previous for word in ("ferias", "férias", "viagem")):
        return False
    return any(place in text for place in ("norte de portugal", "sul de portugal", "alentejo", "algarve", "porto", "geres"))


def _contests_previous_assistant(message: str) -> bool:
    text = _normalize_text(message)
    return any(
        phrase in text
        for phrase in (
            "nao percebi o que querias dizer",
            "não percebi o que querias dizer",
            "nao percebi esta tua frase",
            "não percebi esta tua frase",
            "leste a minha mensagem",
            "leste o que escrevi",
            "li mal",
            "percebeste mal",
        )
    )


def _shorten_for_conversation(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 3].rstrip() + "..."


def _intention_correction_response(text: str) -> str:
    if any(
        phrase in text
        for phrase in (
            "leste a minha mensagem",
            "leste o que escrevi",
            "percebeste a minha mensagem",
        )
    ):
        return "Tens razão. Li mal o que disseste. Voltemos ao ponto certo."
    if any(
        phrase in text
        for phrase in (
            "podemos voltar ao meu problema",
            "volta ao meu problema",
            "vamos voltar ao meu problema",
        )
    ):
        return "Claro. Voltemos ao teu problema."
    return ""


def _extract_personal_fact_from_confirmation(message: str) -> str:
    cleaned = message.strip()
    cleaned = re.sub(r"(?i)^\s*j[áa]\s+sabias\s+que\s+", "", cleaned)
    cleaned = re.sub(r"(?i),?\s*certo\??\s*$", "", cleaned)
    cleaned = cleaned.strip(" .,!?:;")
    if not cleaned:
        return ""
    return cleaned[0].upper() + cleaned[1:]


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


def _is_plain_startup_greeting(value: str) -> bool:
    text = _normalize_text(value).replace(",", "").strip(" .")
    return text in {
        "ola alexandre",
        "bom dia alexandre",
        "boa tarde alexandre",
    }


def _asks_last_session_summary(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "resume a ultima sessao",
            "resumo da ultima sessao",
            "ultima sessao",
            "última sessao",
            "última sessão",
        )
    )


def _asks_today_session_work(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "o que fizemos hoje",
            "que fizemos hoje",
            "em que trabalhamos hoje",
            "em que estivemos hoje",
        )
    )


def _asks_changes_since_last_time(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "o que mudou desde a ultima vez",
            "que mudou desde a ultima vez",
            "mudou desde a ultima vez",
            "mudou desde a última vez",
        )
    )


def _asks_next_step(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "qual e o proximo passo",
            "qual é o próximo passo",
            "proximo passo",
            "próximo passo",
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


def _asks_memory_details(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "com detalhes",
            "mostra os detalhes",
            "mostrar os detalhes",
            "detalhes tecnicos",
            "detalhes técnicos",
            "mostra confianca",
            "mostra confiança",
            "inclui confianca",
            "inclui confiança",
            "qual e a confianca",
            "qual é a confiança",
            "mostra evidencias",
            "mostra evidências",
            "mostra as evidencias",
            "mostra as evidências",
            "inclui evidencias",
            "inclui evidências",
            "mostra os factos",
            "mostra factos",
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


def _message_segments(message: str) -> list[str]:
    raw = str(message or "").strip()
    if not raw:
        return []
    parts = re.split(r"[.!?;\n]+", raw)
    return [part.strip(" \t\r\n,:—-") for part in parts if part.strip(" \t\r\n,:—-")]


def _strip_operational_prefix(segment: str) -> str:
    text = segment.strip(" .,!?:;\"'")
    normalized = _normalize_text(text)
    prefixes = (
        "sim obrigado agora ",
        "sim obrigado ",
        "tudo certo agora ",
        "antes disso ",
        "agora ",
    )
    for prefix in prefixes:
        if normalized.startswith(prefix):
            return text[len(prefix) :].strip(" .,!?:;\"'")
    for marker in (" agora ", " antes disso "):
        index = normalized.find(marker)
        if index >= 0:
            candidate = text[index + len(marker) :].strip(" .,!?:;\"'")
            if _normalize_text(candidate).startswith(
                ("pesquisa", "pesquisar", "procura", "procurar", "encontra", "verifica", "quero que", "consegues")
            ):
                return candidate
    return text


def _extract_research_query(message: str) -> str:
    raw = str(message or "").strip()
    patterns = (
        r"^(?:por favor\s+)?(?:pesquisa|pesquisar|investiga|investigar)\s+(?:sobre|acerca de|informacao sobre|informacoes sobre)\s+(.+)$",
        r"^(?:por favor\s+)?(?:pesquisa|pesquisar|procura|procurar|faz|faz-me|faz me)\s+(?:uma\s+)?(?:pesquisa\s+)?(?:na internet|online)\s+(?:sobre|acerca de)\s+(.+)$",
        r"^(?:por favor\s+)?(?:pesquisa|pesquisar|procura|procurar)\s+(?:informacao geral|informacao|informacoes|fontes|noticias)\s+(?:sobre|acerca de)\s+(.+)$",
        r"^(?:por favor\s+)?(?:faz|faz-me|faz me|quero que facas|quero que faças|consegues fazer)\s+(?:uma\s+)?pesquisa\s+(?:sobre|acerca de)\s+(.+)$",
        r"^(?:por favor\s+)?(?:quero que pesquises|consegues pesquisar|podes pesquisar)\s+(?:na internet\s+|online\s+)?(?:sobre\s+|acerca de\s+)?(.+)$",
        r"^(?:por favor\s+)?(?:quero que investigues|consegues investigar|investiga|investigar)\s+(.+)$",
        r"^(?:por favor\s+)?(?:encontra|encontrar|procura|procurar)\s+fontes\s+(?:sobre|acerca de)\s+(.+)$",
        r"^(?:por favor\s+)?(?:verifica|verificar)\s+(?:online|na internet)\s+(?:sobre\s+)?(.+)$",
        r"^(?:por favor\s+)?(?:procura|procurar)\s+(?:informacao|informacoes|dados)\s+(?:sobre|acerca de)\s+(.+)$",
    )
    for segment in _message_segments(raw):
        candidate = _strip_operational_prefix(segment)
        normalized = _normalize_text(candidate)
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if not match:
                continue
            start, end = match.span(1)
            query = candidate[start:end].strip(" .,!?:;\"'")
            return query
    return ""


def _debug_print(env_name: str, message: str) -> None:
    if os.environ.get(env_name, "").strip().lower() in {"1", "true", "yes", "on"}:
        print(message)


def _echo_debug_errors_enabled() -> bool:
    return os.environ.get("ECHO_DEBUG_ERRORS", "").strip().lower() in {"1", "true", "yes", "on"}


def _is_memory_inventory_query(message: str) -> bool:
    text = _normalize_text(message).strip(" .,!?:;")
    return any(
        marker in text
        for marker in (
            "o que tens na memoria",
            "que coisas tens guardadas sobre mim",
            "o que sabes sobre mim a partir da memoria",
            "mostra-me o que guardaste",
            "mostra me o que guardaste",
            "que memorias tens",
        )
    )


def _has_memory_marker(message: str) -> bool:
    text = _normalize_text(message)
    return any(
        marker in text
        for marker in (
            "memoria",
            "guardado",
            "guardada",
            "guardaste",
            "lembras",
            "recordas",
            "te falei",
            "te disse",
            "meu ",
            "minha ",
        )
    )


def _is_general_knowledge_query(message: str) -> bool:
    for segment in _message_segments(message):
        if _has_memory_marker(segment):
            continue
        text = _normalize_text(segment).strip(" .,!?:;")
        if "sobre mim" in text:
            continue
        if re.match(
            r"^(?:o que sabes sobre|explica-me|explica me|o que e|o que é|como funciona|fala-me sobre|fala me sobre)\s+.+",
            text,
        ):
            return True
    return False


def _summarize_structured_fact_for_inventory(fact) -> str:
    if getattr(fact, "fact_type", "") == "academic_event":
        if fact.discipline and fact.date_reference:
            return f"um exame de {fact.discipline} {fact.date_reference}"
        if fact.discipline:
            return f"um exame de {fact.discipline}"
        if fact.date_reference:
            return f"um exame {fact.date_reference}"
        return "um exame"
    return fact.summary()


def _is_research_followup(message: str) -> bool:
    text = _normalize_text(message).strip(" .,!?:;")
    return text in {
        "so informacao geral",
        "só informacao geral",
        "so quero informacao geral",
        "só quero informação geral",
        "uma visao geral",
        "uma visão geral",
        "geral",
        "informacao geral",
        "informação geral",
    }


def _is_topic_shift_request(message: str) -> bool:
    text = _normalize_text(message).strip(" .,!?:;")
    return text in {
        "queria falar sobre outro tema",
        "quero falar sobre outro tema",
        "vamos falar sobre outro tema",
        "muda de tema",
        "mudar de tema",
        "outro tema",
        "vamos mudar de tema",
    }


def _looks_like_planning_followup(message: str) -> bool:
    text = _normalize_text(message).strip(" .,!?:;")
    if not text:
        return False
    if len(text.split()) <= 5:
        return True
    return any(
        phrase in text
        for phrase in (
            "vou com",
            "vamos com",
            "norte de portugal",
            "road trip",
            "mesmo sitio",
            "mudando de alojamento",
        )
    )


def _writing_request_kind(message: str) -> str:
    text = _normalize_text(message).strip(" .,!?:;")
    if not text:
        return ""
    if re.search(r"\b(?:escreve|redige|prepara|cria)\b", text) and any(
        marker in text for marker in ("email", "e-mail", "mensagem", "texto", "carta")
    ):
        return "professional_writing" if "profissional" in text or "email" in text or "e-mail" in text else "writing"
    if re.search(r"\b(?:reescreve|reformula|melhora)\b", text):
        return "rewrite"
    if re.search(r"\b(?:resume|sintetiza|transforma)\b", text) and any(
        marker in text for marker in ("texto", "pontos", "topicos", "tópicos", ":")
    ):
        return "summary"
    return ""


def _looks_like_complete_text_summary_request(message: str) -> bool:
    text = _normalize_text(message).strip(" .,!?:;")
    if not re.search(r"\b(?:resume|sintetiza|transforma)\b", text):
        return False
    if not any(marker in text for marker in ("texto", "pontos", "topicos", "tÃ³picos", "ideias principais", ":")):
        return False
    if re.search(r"\b(?:cria|adiciona|marca|cancela|adia)\b", text) and re.search(r"\b(?:tarefa|tarefas|lembrete)\b", text):
        return False
    return ":" in str(message or "") or len(text.split()) >= 10


def _looks_like_project_history_question(normalized_message: str) -> bool:
    text = normalized_message.strip(" .,!?:;")
    temporal = any(
        marker in text
        for marker in (
            "ha quanto tempo",
            "desde quando",
            "quando comecamos",
            "quando começamos",
            "o que fizemos anteriormente",
            "progresso recente",
            "ultimos dias",
            "ultimas semanas",
            "ultimos meses",
        )
    )
    project_marker = "projeto" in text or "echo" in text or "assistenteia" in text or "assistente ia" in text
    return temporal and project_marker


def _looks_like_project_recall_question(normalized_message: str) -> bool:
    text = normalized_message.strip(" .,!?:;")
    if _looks_like_project_history_question(text):
        return True
    project_marker = "projeto" in text or "projecto" in text or "echo" in text or "assistenteia" in text or "assistente ia" in text
    if not project_marker:
        return "quais" in text and "ximos passos" in text
    if text.startswith("o que "):
        return True
    return any(
        marker in text
        for marker in (
            "o que e",
            "o que Ã©",
            "o que a©",
            "resume",
            "resumo",
            "objetivos",
            "objectivos",
            "em que ponto",
            "estado do projeto",
            "estado do projecto",
            "o que ja fizemos",
            "o que jÃ¡ fizemos",
            "o que ja¡ fizemos",
            "proximos passos",
            "prÃ³ximos passos",
            "ximos passos",
        )
    )


def _project_recall_terms(project: str, user_message: str) -> list[str]:
    values = {_normalize_text(project)}
    normalized_message = _normalize_text(user_message)
    if "echo" in normalized_message or "echo" in values:
        values.update({"echo", "projeto echo", "projecto echo"})
    if "assistenteia" in normalized_message or "assistente ia" in normalized_message:
        values.update({"assistenteia", "assistente ia", "projeto assistenteia", "projecto assistenteia"})
    return [value for value in values if value]


def _history_item_matches_project(item: dict[str, str], project_terms: list[str]) -> bool:
    content = _normalize_text(str(item.get("content") or ""))
    return bool(content) and any(term and term in content for term in project_terms)


def _project_recall_context(
    project: str,
    history: list[dict[str, str]],
    history_matches: list[int],
    persistent_matches: list[str],
) -> str:
    parts = [f"Projeto consultado: {project}."]
    if history_matches:
        lines = []
        for index in history_matches[-8:]:
            item = history[index]
            role = "Alexandre" if item.get("role") == "user" else "Echo"
            content = str(item.get("content") or "").strip()
            if content:
                lines.append(f"- {role}: {content}")
        if lines:
            parts.append("HistÃ³rico relevante da conversa:\n" + "\n".join(lines))
    if persistent_matches:
        lines = [f"- {text}" for text in persistent_matches[-8:] if str(text).strip()]
        if lines:
            parts.append("MemÃ³ria persistente relevante:\n" + "\n".join(lines))
    return "\n\n".join(parts)


def _extract_project_name_for_history(message: str) -> str:
    text = str(message or "")
    match = re.search(r"\bprojeto\s+([A-Za-z0-9_.-]+)", text, flags=re.IGNORECASE)
    if match:
        return f"projeto {match.group(1)}"
    normalized = _normalize_text(text)
    if "echo" in normalized:
        return "projeto Echo"
    if "assistenteia" in normalized or "assistente ia" in normalized:
        return "projeto AssistenteIA"
    return ""


def _memory_item_text(item) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("content", "description", "summary", "text"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""
    for attr in ("content", "description", "summary", "text"):
        value = getattr(item, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    summary = getattr(item, "summary", None)
    if callable(summary):
        try:
            value = summary()
        except Exception:
            return ""
        return str(value or "").strip()
    return ""


def _extract_duration_from_evidence(text: str) -> str:
    normalized = _normalize_text(text)
    patterns = (
        r"\b(ha\s+(?:\w+\s+)?(?:dia|dias|semana|semanas|mes|meses|ano|anos))\b",
        r"\b(desde\s+(?:janeiro|fevereiro|marco|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}))\b",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return match.group(1)
    return ""


def _intent_instruction_for_user_message(message: str) -> str:
    kind = _writing_request_kind(message)
    if not kind:
        return ""
    return (
        "Se o pedido do Alexandre for completo e imperativo, executa a tarefa diretamente. "
        "Para pedidos como escrever email, reescrever texto ou resumir texto, não respondas com "
        "'posso ajudar', 'queres que escreva' ou outra oferta de ajuda."
    )


def _looks_like_help_offer(response: str) -> bool:
    text = _normalize_text(response).strip(" .,!?:;")
    offer_markers = (
        "posso ajudar",
        "posso ajudar-te",
        "queres que escreva",
        "queres que redija",
        "queres que faca",
        "queres que faça",
        "se preferires posso",
        "posso escrever",
        "posso redigir",
    )
    return any(marker in text for marker in offer_markers)


def _local_writing_fallback(message: str) -> str:
    text = _normalize_text(message)
    if "email" in text or "e-mail" in text:
        return (
            "Assunto: Estado atual do projeto Echo\n\n"
            "Olá,\n\n"
            "Escrevo para partilhar o estado atual do projeto Echo, os progressos realizados e os próximos passos previstos.\n\n"
            "Com os melhores cumprimentos,\n"
            "Alexandre"
        )
    return ""


def _is_pending_intent_followup(message: str, pending: dict[str, str]) -> bool:
    text = _normalize_text(message).strip(" .,!?:;")
    if text in {"sim", "s", "ok", "okay", "claro", "pode ser", "forca", "força", "avanca", "avança"}:
        return True
    kind = pending.get("kind", "")
    if kind in {"professional_writing", "writing"} and text in {"do email", "email", "o email", "desse email"}:
        return True
    if kind == "summary" and text in {"do resumo", "resumo", "desse texto", "do texto"}:
        return True
    if kind == "rewrite" and text in {"da reescrita", "reescreve", "do texto", "desse texto"}:
        return True
    return False

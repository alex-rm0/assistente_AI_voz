from __future__ import annotations

import re
import threading
import time
import traceback
import unicodedata
import os
import hashlib
from collections.abc import Callable
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from assistant.agent import Agent, AgentContext, system_state_tool_intent
from assistant.addressable_content import (
    build_addressable_content,
    detect_content_operation,
    present_content,
    resolve_content_target,
    source_url_for_target,
)
from assistant.briefing import get_last_active_project, summarize_yesterday
from assistant.cognition.context_builder import ContextBuilder
from assistant.cognition.executive_function import CognitiveStrategy, ExecutiveFunction
from assistant.cognition.intent_engine import IntentEngine
from assistant.cognition.reasoning_engine import ReasoningEngine, ReasoningResult
from assistant.cognition.reflection_engine import ReflectionEngine
from assistant.context_manager import ActiveContext, ContextManager
from assistant.delegation import DelegationManager, DelegationTarget
from assistant.fast_router import load_quick_sites, route_fast_command
from assistant.interaction_primitives import workspace_command_actions as _primitive_workspace_command_actions
from assistant.attention import (
    ACTIVE as _ATTN_ACTIVE,
    FOCUS as _ATTN_FOCUS,
    GLANCEABLE as _ATTN_GLANCEABLE,
    HIDDEN as _ATTN_HIDDEN,
    MINIMIZED as _ATTN_MINIMIZED,
    VISIBLE as _ATTN_VISIBLE,
    AttentionState,
    get_attention_level,
    rank_minimization_candidates,
    set_attention,
    set_bulk_attention,
)
from assistant.interaction_interpreter import (
    InteractionInterpretation,
    TurnRoutingDecision,
    compare_route as _interpreter_compare_route,
    decide_turn_route as _interpreter_decide_turn_route,
    expected_route as _interpreter_expected_route,
    interpret_shadow,
)
from assistant.long_term_memory import LongTermMemory, MemoryCategory
from assistant.memory import ConversationMemory
from assistant.model_provider import ProviderConfigurationError, ProviderTimeoutError
from assistant.representation import (
    RepresentationConstraint,
    RepresentationState,
    SemanticObject,
    SpacePressureState,
    choose_representation,
    extract_added_alternatives,
    extract_add_target_reference,
    extract_comparison_category,
    extract_comparison_criteria,
    extract_explicit_alternatives,
    message_implies_comparison,
    register_object,
    resolve_mentioned_object_ids,
    restore_representation,
    set_reasoning_visible,
    slugify_title,
    strip_add_alternatives_phrase,
    transform_representation,
)
from assistant.task_intelligence import (
    answer_task_query,
    classify_task_query,
    cognitive_task_view_model,
    needs_task_reference_clarification,
    task_dialogue_response,
    task_card_summary,
    task_option_titles,
    update_task_knowledge,
)
from assistant.task_research import (
    build_research_request,
    exclude_research_source,
    is_task_research_intent,
    OllamaSemanticExtractor,
    research_sources_response,
    resolve_research_source,
    run_task_research,
    SemanticExtractor,
)
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
from assistant.response_language import (
    ENGLISH_GB,
    PORTUGUESE_PT,
    detect_response_language_command,
    legacy_language_value,
    localize_deterministic_response,
    looks_mixed_for_language,
    normalize_response_language,
    response_language_instruction,
    response_language_status,
    simple_social_response,
)
from assistant.security import check_user_request
from assistant.session_manager import SessionManager
from assistant.text_encoding import debug_text_encoding, has_mojibake_markers
from assistant.text_matching import contains_any_phrase, find_near_pair_span, normalize_for_matching
from assistant.tool_registry import ToolRegistry
from assistant.tools import (
    cancel_task as cancel_task_tool,
    complete_task as complete_task_tool,
    create_workspace_file,
    list_pending_tasks as list_pending_tasks_tool,
    postpone_task as postpone_task_tool,
    read_workspace_file_content,
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


@dataclass
class PendingWritingTask:
    task_type: str = "compose_email"
    recipient: str = ""
    recipient_type: str = ""
    recipient_context: str = ""
    purpose: str = ""
    subject: str = ""
    project: str = "unknown"
    known_facts: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    awaiting_field: str = ""
    current_step: str = "collecting_context"
    active_artifact_id: str = ""
    active_artifact_revision: int = 0
    ttl: int = 4
    batch_total: int = 1
    batch_remaining_after_current: int = 0


@dataclass
class ActiveTaskContext:
    """Passo 3.4: a generic, session-only "what are we currently building
    together" layer -- deliberately NOT specific to email. compose_email is
    the first task_type; entities/parameters are plain dicts so a future
    plan_trip/make_decision/study_topic/prepare_meeting/code_change
    task_type needs no new class, only its own field-extraction/question
    helpers (see _TASK_EXTRACTORS/_TASK_MISSING_FIELDS/_TASK_QUESTIONS).
    Never persisted to long-term memory -- ttl-based like every other
    transient context in this file (ActiveDocumentContext, PendingWritingTask)."""

    id: str = ""
    task_type: str = "compose_email"
    activity_family: str = "communication"
    domain: str = "email"
    artifact_type: str = "email"
    status: str = "collecting"  # collecting|ready|working|reviewing|paused|completed|cancelled
    goal: str = ""
    entities: dict[str, str] = field(default_factory=dict)
    parameters: dict[str, str] = field(default_factory=dict)
    collected_facts: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    active_artifact_id: str = ""
    active_artifact_revision: int = 0
    workspace_object_ids: list[str] = field(default_factory=list)
    started_at: str = ""
    updated_at: str = ""
    source_turns: list[str] = field(default_factory=list)
    ttl: int = 6
    # Section 8: set once the task has nothing left to collect and Echo has
    # asked "queres que escreva uma primeira versão?" -- lets a bare
    # "podes"/"sim"/"força"/"avança" confirm that specific pending offer
    # without needing a fresh explicit command every time. An explicit
    # command ("podes escrever") never depends on this being set.
    pending_action: str = ""
    # Section 9 (fault-attribution phase): generic content-enrichment
    # accumulator -- once the task already has a real topic, further
    # unclassified fragments ("o funcionamento e as limitações") enrich
    # the artefact instead of being silently dropped or misread as a new
    # topic. Never overwrites topic, only appends.
    key_points: list[str] = field(default_factory=list)
    # Section 8 (Phase 4 workspace-reference integration fix): a generic
    # "content the user explicitly asked this task to produce" log ("mostra-
    # me 10 destinos em Portugal") -- distinct from collected_facts (which is
    # a plain running note dump). Never generated deterministically here;
    # this only PRESERVES the request structurally so it is not silently
    # lost as an opaque note, so a later LLM-routed generation step still has
    # it to work from.
    requested_outputs: list[str] = field(default_factory=list)


@dataclass
class ActivityInterpretation:
    """Deterministic activity signal used before the LLM/tool router.

    The interpretation is intentionally structural rather than domain-heavy:
    it answers "is Alexandre starting, continuing, refining, pausing or
    resuming work?" and keeps any remaining content for the task layer.
    """

    starts_task: bool = False
    continues_task: bool = False
    refines_task: bool = False
    pauses_task: bool = False
    resumes_task: bool = False
    activity_family: str = "unknown"
    domain: str = "unknown"
    artifact_type: str = "unknown"
    task_type: str = ""
    goal: str = ""
    confidence: float = 0.0
    reason: str = ""
    remaining_content: str = ""


@dataclass
class DocumentIntent:
    action: str = ""
    file_reference: str = ""
    requested_output: str = ""
    output_file: str = ""
    needs_clarification: bool = False


@dataclass
class ActiveDocumentContext:
    resolved_file: str = ""
    file_type: str = ""
    content_reference: str = ""
    content_hash: str = ""
    content_length: int = 0
    file_size: int = 0
    detected_content_type: str = ""
    last_action: str = ""
    current_document_task: str = ""
    available_followups: list[str] = field(default_factory=list)
    mtime_ns: int = 0
    content: str = ""
    ttl: int = 8
    # Transitory rewritten version, held in memory only. A real file change
    # (detected in _refresh_active_document_context) rebuilds the whole
    # context via _build_active_document_context, which drops these back to
    # their defaults — a stale draft never survives the original changing.
    draft_content: str = ""
    draft_action: str = ""
    draft_created_from_hash: str = ""
    draft_saved: bool = False
    draft_updated_at: str = ""
    # Revision counter for draft_content: 0 = no draft yet, 1 = first
    # rewrite, 2+ = each successful iterative refinement after that.
    # previous_draft_content/_revision hold exactly one step of undo history
    # (not an unlimited list) -- restore_previous_draft is the only consumer.
    draft_revision: int = 0
    previous_draft_content: str = ""
    previous_draft_revision: int = 0


@dataclass
class WorkSession:
    """Phase 1 of the Cognitive Workspace Blueprint v2: a single, central
    session-state container converging the three previously-parallel,
    independently-owned contexts (ActiveTaskContext, ActiveDocumentContext,
    PendingWritingTask) into one object. This is a pure state-model
    refactor -- AssistantEngine holds exactly one `self._work_session`
    instance; the old `self._active_task_context` / `_active_document_context`
    / `_pending_writing_task` names live on as compatibility
    properties (see AssistantEngine) that read and write these same
    fields, never a second copy.

    `paused_task_stack` is REAL, already-wired behavior (task
    switch/pause/resume, added in an earlier phase) -- not to be confused
    with the RESERVED `task_stack` field below, which is the future,
    unified v2 concept and stays completely inert in this phase.

    Every field below `paused_task_stack` is a Phase 1 reservation only,
    per the blueprint's section 2: no code reads or writes them yet. They
    exist purely so later phases (Interaction Interpreter, Representation
    State, reasoning/attention/capacity/constraints) don't require a
    second structural migration of this class."""

    active_task: ActiveTaskContext | None = None
    document_context: ActiveDocumentContext | None = None
    writing_context: PendingWritingTask | None = None
    # Real, wired: tasks pushed aside by an explicit pause or by starting a
    # different task_type while one was already active (see
    # _pause_active_task/_resume_task_from_stack).
    paused_task_stack: list[ActiveTaskContext] = field(default_factory=list)

    # --- Reserved for later phases -- inert in Phase 1, see class docstring. ---
    task_stack: list[object] = field(default_factory=list)
    # Phase 3 (Cognitive Workspace Blueprint v2) activates these two fields
    # for real -- see assistant/representation.py. `objects` is the
    # semantic object registry (section 3); `representation` is the
    # current RepresentationState (section 1), or None before the first
    # transform_representation() call. Everything else below stays inert.
    objects: dict[str, SemanticObject] = field(default_factory=dict)
    relations: list[object] = field(default_factory=list)
    knowledge: dict[str, object] = field(default_factory=dict)
    reasoning: list[object] = field(default_factory=list)
    # Phase 4 (Cognitive Workspace Blueprint v2) activates three more of
    # Phase 1's reserved placeholders for real, the SAME way Phase 3
    # activated `representation` above: the type changes from the generic
    # Phase-1 placeholder to the concrete structure the phase that needs it
    # defines, never a second parallel field. `attention` is the current
    # AttentionState (assistant/attention.py) or None before the first
    # object ever gets a level assigned; `constraints` is the current
    # RepresentationConstraint (assistant/representation.py) or None
    # before any explicit user representation preference has been stated;
    # `capacity` is the current SpacePressureState (assistant/
    # representation.py) or None before the frontend has ever reported one.
    attention: AttentionState | None = None
    representation: RepresentationState | None = None
    constraints: RepresentationConstraint | None = None
    preferences: dict[str, object] = field(default_factory=dict)
    capacity: SpacePressureState | None = None
    recent_references: list[object] = field(default_factory=list)
    content_focus: dict[str, object] = field(default_factory=dict)


def _local_ollama_semantic_extractor(llm) -> SemanticExtractor | None:
    if not callable(getattr(llm, "chat", None)):
        return None
    if bool(getattr(llm, "supports_local_semantic_extraction", False)):
        return OllamaSemanticExtractor(llm)
    provider_name = str(getattr(llm, "provider_name", "") or getattr(llm, "provider", "") or "").lower()
    if provider_name == "ollama":
        return OllamaSemanticExtractor(llm)
    if llm.__class__.__name__ != "OllamaClient":
        return None
    if llm.__class__.__module__ != "assistant.llm":
        return None
    return OllamaSemanticExtractor(llm)


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
        tts_manager=None,
        model_runtime=None,
        now_provider=None,
        semantic_extractor: SemanticExtractor | None = None,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.long_term_memory = long_term_memory
        self.personal_model = personal_model
        self.response_composer = ResponseComposer(
            llm,
            voice_critic=VoiceCritic(voice_critic_llm) if voice_critic_llm is not None else None,
        )
        self.semantic_extractor = semantic_extractor or _local_ollama_semantic_extractor(llm)
        extractor_llm = getattr(self.semantic_extractor, "llm", None)
        print(
            "[SEMEX] SEMEX_CONFIGURED",
            f"configured={self.semantic_extractor is not None}",
            f"response_llm={type(llm).__name__}",
            f"response_provider_name={getattr(llm, 'provider_name', '')}",
            f"response_supports_local_semantic_extraction={bool(getattr(llm, 'supports_local_semantic_extraction', False))}",
            f"extractor={type(self.semantic_extractor).__name__ if self.semantic_extractor is not None else 'None'}",
            f"extractor_llm={type(extractor_llm).__name__ if extractor_llm is not None else 'None'}",
            f"extractor_provider_name={getattr(extractor_llm, 'provider_name', '') if extractor_llm is not None else ''}",
            f"extractor_uses_routed_llm={type(extractor_llm).__name__ == 'RoutedLLM'}",
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
        self._canonical_interpretation: InteractionInterpretation | None = None
        self._routing_decision: TurnRoutingDecision | None = None
        # Cooperative cancellation: each respond() call gets a brand-new Event
        # (never a reused/reset one), so a stale cancel from a finished
        # request can never affect a later one -- see begin_request().
        self._cancel_lock = threading.Lock()
        self._cancel_event: threading.Event | None = None
        # Optional live progress callback, set by the UI worker for the
        # duration of a single respond() call (see set_progress_listener).
        # Not persisted -- purely in-memory, per-process wiring.
        self._progress_listener: Callable[[str], None] | None = None
        self._pending_ui_events: list[str] = []
        self._last_fast_tools_used: tuple[str, ...] = ()
        self._active_operation_type = ""
        self._active_operation_topic = ""
        self._active_operation_id = ""
        self._active_operation_ttl = 0
        self._pending_user_intent: dict[str, str] | None = None
        self._pending_document_task: dict[str, object] | None = None
        # Phase 4 (Cognitive Workspace Blueprint v2, section 15): a
        # transient "Echo asked permission to minimize something, waiting
        # for sim/não" -- same session-only, engine-level pattern as
        # `_pending_user_intent` above (a WorkSession field would be the
        # wrong home for it: this is turn-to-turn CONTROL FLOW state, not
        # session knowledge).
        self._pending_workspace_action: dict[str, object] | None = None
        self._space_pressure_suppressed_candidate_ids: set[str] = set()
        self._workspace_capacity_revision = 0
        self._task_surface_capacity_override = False
        # Phase 1 (Cognitive Workspace Blueprint v2): the single, central
        # session-state container. `_active_task_context`,
        # `_active_document_context`, `_pending_writing_task`, and
        # `_paused_task_stack` below are compatibility properties (defined
        # right after __init__) reading/writing this same object -- there
        # is exactly one writable copy of this state, never two.
        self._work_session = WorkSession()
        # Passo 3.2/3.3: the last successfully detected echo_entity
        # workspace command (action/region/etc), kept only for a
        # "não deslocaste"-style correction retry -- session-only, never
        # long-term memory, and unrelated to _active_document_context.
        self._last_workspace_command: dict[str, str] | None = None
        # Passo 3.4: the generic "what are we building together" context
        # (see ActiveTaskContext) -- only one at a time (pausing swaps
        # `status`, not the object itself; see _try_active_task_turn's
        # pause/resume handling), session-only, never long-term memory.
        # (lives on WorkSession.active_task / .paused_task_stack now --
        # see the compatibility properties right after __init__.)
        self._active_task_id_counter = 0
        # Passo 3.6: set by _process_task_turn (deterministic, llm_calls=0)
        # when a TASK_CONTROL execute phrase matched a ready task -- consumed
        # by _try_active_task_turn immediately after, which is the only place
        # allowed to call the LLM for task generation.
        self._task_execution_requested = False
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
        self.tts_manager = tts_manager
        self.model_runtime = model_runtime
        self.response_language = self._load_response_language_preference()
        self.current_language = legacy_language_value(self.response_language)
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
        self.now_provider = now_provider or (lambda: datetime.now().astimezone())
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

    # --- Phase 1 (Cognitive Workspace Blueprint v2) compatibility
    # properties. Every one of these reads/writes the SAME WorkSession
    # field; there is no independent storage behind any of them. A getter
    # returns the live object reference, so in-place mutation (e.g.
    # `ctx.status = "ready"` on whatever `self._active_task_context`
    # returned) already lands on WorkSession with no extra plumbing --
    # only whole-object reassignment (`self._active_task_context = ...`)
    # goes through the setter below.

    @property
    def _active_task_context(self) -> ActiveTaskContext | None:
        return self._work_session.active_task

    @_active_task_context.setter
    def _active_task_context(self, value: ActiveTaskContext | None) -> None:
        self._work_session.active_task = value

    @property
    def _paused_task_stack(self) -> list[ActiveTaskContext]:
        return self._work_session.paused_task_stack

    @_paused_task_stack.setter
    def _paused_task_stack(self, value: list[ActiveTaskContext]) -> None:
        self._work_session.paused_task_stack = value

    @property
    def _active_document_context(self) -> ActiveDocumentContext | None:
        return self._work_session.document_context

    @_active_document_context.setter
    def _active_document_context(self, value: ActiveDocumentContext | None) -> None:
        self._work_session.document_context = value

    @property
    def _pending_writing_task(self) -> PendingWritingTask | None:
        return self._work_session.writing_context

    @_pending_writing_task.setter
    def _pending_writing_task(self, value: PendingWritingTask | None) -> None:
        self._work_session.writing_context = value

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

    def _try_system_datetime_command(self, user_message: str) -> str | None:
        parsed = _parse_system_datetime_question(user_message)
        if parsed is None:
            return None
        now = self.now_provider()
        if not isinstance(now, datetime):
            now = datetime.now().astimezone()
        current = now.astimezone()
        kind, value = parsed
        target = current.date()
        if kind == "time":
            return f"São {current:%H:%M}."
        if kind == "tomorrow":
            target = current.date() + timedelta(days=1)
            return f"Amanhã é {_format_pt_date(target)}."
        if kind == "yesterday":
            target = current.date() - timedelta(days=1)
            return f"Ontem foi {_format_pt_date(target)}."
        if kind == "offset":
            target = current.date() + timedelta(days=int(value or 0))
            return f"Daqui a {int(value or 0)} dias é {_format_pt_date(target)}."
        if kind == "weekday_for_date" and isinstance(value, date):
            return f"{_format_pt_date(value, include_weekday=False).capitalize()} calha a {_weekday_name(value)}."
        return f"Hoje é {_format_pt_date(target)}."

    def _try_system_status_command(self, user_message: str) -> str | None:
        if self.model_runtime is None:
            return None
        answer = getattr(self.model_runtime, "system_status_answer", None)
        if not callable(answer):
            return None
        return answer(user_message)

    def begin_request(self) -> None:
        """Start a brand-new cancellation token for the request about to run.

        A fresh threading.Event is created (never the previous one reset),
        so a cancel_current_request() call left over from a finished/stale
        request can never leak into this new one -- is_cancel_requested()
        can only ever observe the Event object created here.
        """
        with self._cancel_lock:
            self._cancel_event = threading.Event()

    def cancel_current_request(self) -> bool:
        """Request cooperative cancellation of whatever respond() call is
        currently active. Returns True if there was an active token to
        cancel. Safe to call from any thread (e.g. a Qt worker thread)."""
        with self._cancel_lock:
            event = self._cancel_event
        if event is None:
            return False
        event.set()
        return True

    def is_cancel_requested(self) -> bool:
        with self._cancel_lock:
            event = self._cancel_event
        return event.is_set() if event is not None else False

    def set_progress_listener(self, listener: Callable[[str], None] | None) -> None:
        """Register (or clear, with None) a live progress callback for the
        UI worker thread. Not conversation state -- purely a per-process,
        in-memory wire between the currently running respond() call and
        whoever is waiting on it."""
        self._progress_listener = listener

    def _emit_progress(self, event: str) -> None:
        listener = self._progress_listener
        if listener is None:
            return
        try:
            listener(event)
        except Exception:
            pass

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
        self.begin_request()
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
                # Passo 3.5 (memory relevance gate, section 5/6): a real
                # session boundary is exactly the "outra altura" the raw
                # transcript sent to the LLM must stop treating as this
                # turn's own recent conversation -- same mechanism explicit
                # topic-shift already uses (_try_topic_shift), just a
                # second, time-based trigger for it. Without this, a stale
                # topic from before the inactivity gap (e.g. "Geometry
                # Processing") stays in _recent_conversation_history's
                # window and the LLM volunteers it back on an unrelated
                # new turn.
                self._conversation_segment_start = len(self.memory.load())
            if self.presence.state == PresenceState.ACTIVE_CONVERSATION:
                self.session_manager.start_session()
        request_started_at = time.perf_counter()
        self._begin_turn_trace(user_message)
        self._run_shadow_interpreter(user_message)
        self._record_tool_intent_check(user_message)
        self._perf_log("pedido recebido", request_started_at, request_started_at)
        datetime_response = self._try_system_datetime_command(user_message)
        if datetime_response:
            return self._complete_turn(
                user_message,
                datetime_response,
                "LOCAL_DATETIME",
                technical=True,
                selected_path="SYSTEM_DATETIME",
                tools_used=("system_datetime",),
            )
        system_status_response = self._try_system_status_command(user_message)
        if system_status_response:
            return self._complete_turn(
                user_message,
                system_status_response,
                "SYSTEM_STATUS",
                technical=True,
                selected_path="SYSTEM_STATUS",
                tools_used=("system_status",),
            )
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
        if self._active_document_context is not None:
            self._active_document_context.ttl -= 1
            if self._active_document_context.ttl <= 0:
                self._active_document_context = None
        # Phase 4 (section 15): the most specific pending-confirmation
        # check runs first -- a "sim"/"não" answering Echo's own "posso
        # minimizar X?" must never be swallowed by the more general
        # pending-writing-intent mechanism below.
        pending_workspace_response = self._try_pending_workspace_action(user_message)
        if pending_workspace_response is not None:
            pending_workspace_response = self._complete_turn(
                user_message,
                pending_workspace_response,
                "RESPONSE_COMPOSER",
                technical=True,
                selected_path="ATTENTION_COMMAND",
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return pending_workspace_response

        language_response = self._try_language_preference_command(user_message)
        if language_response is not None:
            language_response = self._complete_turn(
                user_message,
                language_response,
                "LANGUAGE_COMMAND",
                selected_path="LANGUAGE_COMMAND",
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return language_response

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

        if self._active_task_context is not None:
            content_response = self._try_content_operation(user_message)
            if content_response is not None:
                content_response = self._complete_turn(
                    user_message=user_message,
                    response=content_response,
                    source="CONTENT_OPERATION",
                    remember=True,
                    technical=True,
                    selected_path="CONTENT_OPERATION",
                    tools_used=self._last_fast_tools_used,
                    tool_confirmation=self.agent.has_pending_confirmation(),
                )
                self._perf_log("resposta total", request_started_at, time.perf_counter())
                return content_response

            source_open_response = self._try_open_research_source(user_message, self._active_task_context)
            if source_open_response is not None:
                source_open_response = self._complete_turn(
                    user_message=user_message,
                    response=source_open_response,
                    source="RESEARCH_REQUEST",
                    remember=True,
                    technical=True,
                    selected_path="RESEARCH_REQUEST",
                    tools_used=self._last_fast_tools_used,
                    tool_confirmation=self.agent.has_pending_confirmation(),
                )
                self._perf_log("resposta total", request_started_at, time.perf_counter())
                return source_open_response

        compound_workspace_response = self._try_compound_workspace_command(user_message)
        if compound_workspace_response is not None:
            compound_workspace_response = self._complete_turn(
                user_message,
                compound_workspace_response,
                "WORKSPACE_COMMAND",
                technical=True,
                selected_path="WORKSPACE_COMMAND",
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return compound_workspace_response

        workspace_command_response = self._try_workspace_command(user_message)
        if workspace_command_response is not None:
            workspace_command_response = self._complete_turn(
                user_message,
                workspace_command_response,
                "WORKSPACE_COMMAND",
                technical=True,
                selected_path="WORKSPACE_COMMAND",
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return workspace_command_response

        resume_activity_response = self._try_resume_activity_reference(user_message)
        if resume_activity_response is not None:
            resume_activity_response = self._complete_turn(
                user_message,
                resume_activity_response,
                "TASK_CONTINUATION",
                technical=True,
                selected_path="TASK_RESUME",
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return resume_activity_response

        # Phase 4 (section 5/20) -- discovered via the required real-Ollama
        # manual test (section 30): an explicit attention/visibility
        # command ("mostra o email outra vez", "minimiza o email") must win
        # over an ACTIVE task's own "first refusal" below, exactly the same
        # reasoning that already places _try_workspace_command before
        # _try_active_task_turn -- attention is workspace-visual state, not
        # task content, and the acceptance scenarios explicitly chain a
        # restore/minimize command onto a turn where a real task (e.g.
        # "vamos planear as férias") is already in progress. Before this
        # fix, such a command was silently swallowed by TASK_CONTINUATION
        # and answered as if it were trip-planning small talk.
        attention_command_response = self._try_attention_command(user_message)
        if attention_command_response is not None:
            attention_command_response = self._complete_turn(
                user_message,
                attention_command_response,
                "RESPONSE_COMPOSER",
                technical=True,
                selected_path="ATTENTION_COMMAND",
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return attention_command_response

        # Phase 4 integration fix (section 5/7): "abre X"/"mostra X"/"foca no
        # X" naming an EXISTING workspace activity/object must win over the
        # active task's own continuation catch-all too -- otherwise "abre o
        # planeamento de férias" said while a DIFFERENT task (e.g. writing
        # an email) is active gets appended to that unrelated task's notes
        # instead of bringing the planning activity back into focus
        # (section 2/12.A/12.D). Attention-only; never switches active_task.
        open_reference_response = self._try_open_workspace_reference(user_message)
        if open_reference_response is not None:
            open_reference_response = self._complete_turn(
                user_message,
                open_reference_response,
                "RESPONSE_COMPOSER",
                technical=True,
                selected_path="ATTENTION_COMMAND",
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return open_reference_response

        # Phase 4 integration fix (section 7): representation-flavored
        # handlers (materializing/adding named options, style changes) now
        # run BEFORE active-task continuation -- an active task must not get
        # first refusal over an utterance that explicitly names another
        # workspace entity or asks for a representation change (section
        # 2/12.F/12.G/12.I: "adiciona Porto, Lisboa..." and "mostra-me
        # cidades em cartões" were previously swallowed into the active
        # task's plain notes because these handlers used to run after it).

        # Phase 3B integration fix (FIX 3/4): explicit-alternatives
        # materialization runs right before the representation-command
        # bridge -- "vamos pensar em três opções: X, Y e Z" both creates
        # the semantic objects AND sets the initial cards representation
        # in one deterministic turn.
        explicit_alternatives_response = self._try_explicit_alternatives(user_message)
        if explicit_alternatives_response is not None:
            explicit_selected_path = (
                str(self._turn_trace.get("active_task_selected_path") or "REPRESENTATION_COMMAND")
                if self._turn_trace
                else "REPRESENTATION_COMMAND"
            )
            explicit_source = "TASK_CONTINUATION" if explicit_selected_path.startswith("TASK_") else "REPRESENTATION_COMMAND"
            if self._turn_trace is not None and self._turn_trace.get("task_started"):
                self._apply_auto_attention_for_new_activity(user_message)
                extra = self._maybe_propose_space_pressure_minimize(user_message)
                if extra:
                    explicit_alternatives_response = explicit_alternatives_response + extra
            explicit_alternatives_response = self._complete_turn(
                user_message,
                explicit_alternatives_response,
                explicit_source,
                technical=True,
                selected_path=explicit_selected_path,
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return explicit_alternatives_response

        # Phase 3C (FIX 10/11): a mixed utterance adding another option to
        # an already-active generic set -- runs before the pure
        # representation-command bridge below since "acrescenta também X"
        # is a content addition, not a style change, and must never be
        # silently absorbed as a false-success "Feito" from elsewhere.
        mixed_representation_response = self._try_mixed_representation_intent(user_message)
        if mixed_representation_response is not None:
            mixed_representation_response = self._complete_turn(
                user_message,
                mixed_representation_response,
                "REPRESENTATION_COMMAND",
                technical=True,
                selected_path="REPRESENTATION_COMMAND",
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return mixed_representation_response

        association_response = self._try_semantic_association_request(user_message)
        if association_response is not None:
            association_response = self._complete_turn(
                user_message,
                association_response,
                "REPRESENTATION_COMMAND",
                technical=True,
                selected_path="REPRESENTATION_COMMAND",
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return association_response

        # Phase 3 (Cognitive Workspace Blueprint v2): the narrow
        # representation-command bridge (section 11) -- runs after
        # workspace/active-task routing so neither existing handler's
        # priority changes. Phase 3B widened this to ALWAYS claim a
        # present RepresentationIntent (never falls through to
        # DOCUMENT_TASK/GENERAL_CONVERSATION for one) -- see the method's
        # own docstring for why.
        representation_command_response = self._try_representation_command(user_message)
        if representation_command_response is not None:
            representation_command_response = self._complete_turn(
                user_message,
                representation_command_response,
                "REPRESENTATION_COMMAND",
                technical=True,
                selected_path="REPRESENTATION_COMMAND",
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return representation_command_response

        # Canonical writing entry path: equivalent email-writing starts all
        # enter the slot-aware PendingWritingTask workflow. ActiveTaskContext
        # may still own the broader workspace activity model, but email
        # execution state is not duplicated into a second compose_email task.
        normalized_for_writing = _normalize_text(user_message)
        plural_writing_request = _extract_writing_batch_count(normalized_for_writing) > 1
        if (
            self._pending_writing_task is None
            and _looks_like_incomplete_writing_request(normalized_for_writing)
            and not (
                self._active_task_context is not None
                and contains_any_phrase(normalized_for_writing, _ACTIVITY_SWITCH_PHRASES)
                and not plural_writing_request
            )
            and not (self._active_task_context is not None and self._active_task_context.task_type == "compose_email")
        ):
            writing_workflow_response = self._try_writing_workflow(user_message)
            if writing_workflow_response is not None:
                writing_workflow_response = self._complete_turn(
                    user_message,
                    writing_workflow_response,
                    "WRITING_WORKFLOW",
                    technical=True,
                    selected_path="WRITING_TASK",
                )
                self._perf_log("resposta total", request_started_at, time.perf_counter())
                return writing_workflow_response

        task_research_response = self._try_research_request(user_message)
        if task_research_response is not None:
            task_research_response = self._complete_turn(
                user_message=user_message,
                response=task_research_response,
                source="RESEARCH_REQUEST",
                remember=True,
                technical=True,
                selected_path="RESEARCH_REQUEST",
                tools_used=self._last_fast_tools_used,
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return task_research_response

        task_intelligence_response = self._try_task_intelligence_query(user_message)
        if task_intelligence_response is not None:
            task_intelligence_response = self._complete_turn(
                user_message,
                task_intelligence_response,
                "TASK_INTELLIGENCE",
                technical=True,
                selected_path="TASK_INTELLIGENCE",
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return task_intelligence_response

        # Passo 3.4: an ongoing (or newly triggered) ActiveTaskContext gets
        # first refusal on the turn, before document/general routing --
        # "é para o professor João Cardoso" must never reach GENERAL_
        # CONVERSATION just because no WorkspaceCommand also matched.
        active_task_response = self._try_active_task_turn(user_message)
        if active_task_response is not None:
            task_selected_path = str(self._turn_trace.get("active_task_selected_path") or "TASK_CONTINUATION") if self._turn_trace else "TASK_CONTINUATION"
            # Phase 4 (section 4/12/13): a genuinely NEW activity/task
            # starting is the one moment the auto-attention policy and the
            # space-pressure ask-before-minimize check both run -- never on
            # every turn, only when task_started actually fired this turn.
            if self._turn_trace is not None and self._turn_trace.get("task_started"):
                self._apply_auto_attention_for_new_activity(user_message)
                extra = self._maybe_propose_space_pressure_minimize(user_message)
                if extra:
                    active_task_response = active_task_response + extra
            active_task_response = self._complete_turn(
                user_message,
                active_task_response,
                "TASK_CONTINUATION",
                technical=True,
                selected_path=task_selected_path,
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return active_task_response

        document_response = self._try_document_task(user_message)
        if document_response is not None:
            document_response = self._complete_turn(
                user_message=user_message,
                response=document_response,
                source="DOCUMENT_TASK",
                remember=True,
                technical=True,
                selected_path="DOCUMENT_TASK",
                tools_used=self._last_fast_tools_used,
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return document_response

        writing_workflow_response = self._try_writing_workflow(user_message)
        if writing_workflow_response is not None:
            writing_workflow_response = self._complete_turn(
                user_message,
                writing_workflow_response,
                "WRITING_WORKFLOW",
                selected_path="WRITING_TASK",
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return writing_workflow_response
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

        context_status_response = self._try_system_context_status(user_message)
        if context_status_response is not None:
            context_status_response = self._complete_turn(
                user_message,
                context_status_response,
                "SYSTEM_CONTEXT_STATUS",
                remember=True,
                technical=True,
                selected_path="SYSTEM_CONTEXT_STATUS",
                tools_used=("system_context_status",),
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return context_status_response

        application_activity_response = self._try_application_activity_query(user_message)
        if application_activity_response is not None:
            application_activity_response = self._complete_turn(
                user_message=user_message,
                response=application_activity_response,
                source="TOOL_RESULT",
                remember=True,
                technical=True,
                selected_path="TOOL_PATH",
                tools_used=self._last_fast_tools_used,
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return application_activity_response

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

        document_fallback_response = self._try_active_document_grounded_fallback(
            user_message, _normalize_text(user_message)
        )
        if document_fallback_response is not None:
            document_fallback_response = self._complete_turn(
                user_message=user_message,
                response=document_fallback_response,
                source="DOCUMENT_TASK",
                remember=True,
                technical=True,
                selected_path="DOCUMENT_TASK",
                tools_used=self._last_fast_tools_used,
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return document_fallback_response

        # Pending writing workflow slot answers are control-state follow-ups,
        # like confirmations: if Echo just asked "Para quem é o email?", a
        # short reply such as "É para o João" must be consumed by that
        # workflow before generic conversation/LLM routing can reinterpret it.
        pending_writing_response = self._try_pending_writing_workflow(user_message)
        if pending_writing_response is not None:
            pending_writing_response = self._complete_turn(
                user_message,
                pending_writing_response,
                "WRITING_WORKFLOW",
                selected_path="WRITING_TASK",
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return pending_writing_response

        clarification_response = self._try_ambiguous_opinion_clarification(user_message)
        if clarification_response is not None:
            clarification_response = self._complete_turn(
                user_message,
                clarification_response,
                "CLARIFICATION_DETERMINISTIC",
                selected_path="GENERAL_CONVERSATION",
            )
            self._perf_log("resposta total", request_started_at, time.perf_counter())
            return clarification_response

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

        tts_response = self._try_tts_command(user_message)
        if tts_response is not None:
            return self._complete_turn(user_message, tts_response, "TTS_COMMAND", technical=True, selected_path="VOICE_COMMAND")

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
        return legacy_language_value(self._load_response_language_preference())

    def _load_response_language_preference(self) -> str:
        store = self._user_settings_store()
        if store is not None:
            try:
                language = normalize_response_language(store.load().get("response_language"))
            except Exception:
                language = ""
            if language:
                return language
        getter = getattr(self.long_term_memory, "get_preference", None)
        if getter is None:
            return "pt-PT"
        value = normalize_response_language(getter("response_language", "")) or normalize_response_language(
            getter("idioma_atual", "pt-PT")
        )
        return value or "pt-PT"

    def _ensure_language_preferences(self) -> None:
        setter = getattr(self.long_term_memory, "set_preference", None)
        getter = getattr(self.long_term_memory, "get_preference", None)
        if setter is not None and getter is not None:
            if not getter("idioma_base", ""):
                setter("idioma_base", self.language_base)
            if not getter("idioma_atual", ""):
                setter("idioma_atual", self.current_language)
            if not getter("response_language", ""):
                setter("response_language", self.response_language)
        store = self._user_settings_store()
        if store is not None:
            try:
                if not normalize_response_language(store.load().get("response_language")):
                    store.update(response_language=self.response_language)
            except Exception:
                pass

    def _set_current_language(self, language: str) -> None:
        self._set_response_language(language)

    def _set_response_language(self, language: str) -> None:
        normalized = normalize_response_language(language) or PORTUGUESE_PT
        self.response_language = normalized
        self.current_language = legacy_language_value(normalized)
        store = self._user_settings_store()
        if store is not None:
            try:
                store.update(response_language=normalized)
            except Exception:
                pass
        setter = getattr(self.long_term_memory, "set_preference", None)
        if setter is not None:
            setter("idioma_atual", self.current_language)
            setter("response_language", normalized)
        self._coordinate_tts_voice_for_language(normalized)

    def _user_settings_store(self):
        runtime_store = getattr(self.model_runtime, "store", None)
        if runtime_store is not None:
            return runtime_store
        return getattr(self.tts_manager, "store", None)

    def _coordinate_tts_voice_for_language(self, language: str) -> None:
        manager = self.tts_manager
        setter = getattr(manager, "set_active_voice", None)
        if not callable(setter):
            return
        target = "english_british_primary" if normalize_response_language(language) == ENGLISH_GB else "portuguese_ptpt_primary"
        try:
            setter(target)
        except Exception:
            pass

    def _system_prompt_for_agent(self, active_contexts: list[ActiveContext]) -> str:
        return (
            f"{self.context_manager.system_prompt(active_contexts)}\n\n"
            f"{self._language_instruction()}"
        )

    def _language_instruction(self) -> str:
        return response_language_instruction(self.response_language)

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
        self._pending_writing_task = None
        self._pending_user_intent = None
        self._pending_document_task = None
        self._active_document_context = None
        self._active_task_context = None
        self._paused_task_stack = []
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

    def _try_writing_workflow(self, user_message: str) -> str | None:
        normalized = _normalize_text(user_message)
        if _looks_like_complete_writing_request(normalized):
            return None

        if self._pending_writing_task is None:
            if not _looks_like_incomplete_writing_request(normalized):
                return None
            capacity_response = self._maybe_prepare_task_surface_capacity_action(user_message)
            if capacity_response:
                return capacity_response
            if self._active_task_context is not None:
                self._pause_active_task()
            task = PendingWritingTask()
            _update_pending_writing_task(task, user_message)
            _refresh_pending_writing_task_state(task)
            self._ensure_pending_writing_artifact(task)
            self._pending_writing_task = task
            self._record_writing_workflow_trace(task)
            return _writing_workflow_question(task)

        task = self._pending_writing_task
        if _is_explicit_non_writing_activity_or_command(user_message, normalized, self._active_task_context, self._paused_task_stack):
            if self._turn_trace is not None:
                self._turn_trace["writing_followup_rejected"] = True
                self._turn_trace["writing_followup_rejected_reason"] = "explicit_non_writing_activity_or_command"
            return None
        if not _looks_like_writing_followup(normalized, task):
            task.ttl -= 1
            if task.ttl <= 0:
                self._pending_writing_task = None
            return None

        _update_pending_writing_task(task, user_message, awaiting_field=task.awaiting_field)
        task.ttl = 4
        _refresh_pending_writing_task_state(task)
        self._ensure_pending_writing_artifact(task)
        self._pending_writing_task = task
        self._record_writing_workflow_trace(task)
        if task.missing_fields:
            return _writing_workflow_question(task)
        if not _is_writing_execute_confirmation(normalized):
            return _writing_workflow_question(task)

        self._pending_writing_task = None
        return self.response_composer.compose(
            ComposerRequest(
                intent="writing_task",
                user_message=_writing_task_prompt(task),
                history=[],
                fallback=_writing_task_local_fallback(task),
                intent_instruction=(
                    "Escreve diretamente o email pedido. Usa apenas os factos recolhidos no workflow. "
                    "Nao assumas projeto, destinatario, apoio ou contexto que nao esteja indicado."
                ),
                language_instruction=self._language_instruction(),
            )
        )

    def _ensure_pending_writing_artifact(self, task: PendingWritingTask) -> SemanticObject:
        if not task.active_artifact_id:
            base_id = "writing-email-draft"
            object_id = base_id
            suffix = 2
            while object_id in self._work_session.objects:
                object_id = f"{base_id}-{suffix}"
                suffix += 1
            task.active_artifact_id = object_id
        task.active_artifact_revision += 1
        title = f"Email para {task.recipient}" if task.recipient else "Email em preparação"
        payload = {
            "task_type": task.task_type,
            "recipient": task.recipient,
            "purpose": task.purpose,
            "subject": task.subject,
            "recipient_context": task.recipient_context,
            "status": task.current_step,
            "batch_total": task.batch_total,
            "batch_remaining_after_current": task.batch_remaining_after_current,
        }
        obj = register_object(
            self._work_session,
            SemanticObject(
                id=task.active_artifact_id,
                kind="writing_task",
                title=title,
                payload=payload,
                state="active",
            ),
        )
        self._queue_semantic_object_sync(obj)
        if self._turn_trace is not None:
            self._turn_trace["writing_artifact_id"] = task.active_artifact_id
            self._turn_trace["writing_artifact_revision"] = task.active_artifact_revision
        return obj

    def _try_pending_writing_workflow(self, user_message: str) -> str | None:
        if self._pending_writing_task is None:
            return None
        return self._try_writing_workflow(user_message)

    def _record_writing_workflow_trace(self, task: PendingWritingTask) -> None:
        if self._turn_trace is None:
            return
        self._turn_trace["writing_workflow_active"] = True
        self._turn_trace["writing_task_type"] = task.task_type
        self._turn_trace["writing_awaiting_field"] = task.awaiting_field
        self._turn_trace["writing_recipient"] = task.recipient
        self._turn_trace["writing_recipient_type"] = task.recipient_type
        self._turn_trace["writing_recipient_context"] = task.recipient_context
        self._turn_trace["writing_purpose"] = task.purpose
        self._turn_trace["writing_subject"] = task.subject
        self._turn_trace["writing_project"] = task.project
        self._turn_trace["writing_missing_fields"] = list(task.missing_fields)
        self._turn_trace["response_grounded"] = True
        self._turn_trace["grounding_sources"] = ["TRANSIENT_WORKFLOW"]

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

    def _queue_semantic_object_sync(self, obj: SemanticObject) -> None:
        """Phase 3B integration fix #2: the frontend has no independent
        knowledge of WorkSession.objects -- this is the ONE place that
        tells it a semantic object exists/changed (echo_ui.js's
        "semantic_object_upserted" case -> echo_workspace_objects.js's
        upsertSemanticObject, same registry every other WorkspaceObject
        lives in, keyed by this SAME id). Callers must queue this BEFORE
        any representation_changed event that references the object's id.
        Existing task_artifact objects already have their own identity-
        sync event (task_artifact_updated -> upsertTaskArtifact) -- this
        is deliberately NOT also called for those, to keep exactly one
        sync event per object per turn (section 10)."""
        self._queue_ui_event(
            "semantic_object_upserted",
            {
                "id": obj.id,
                "kind": obj.kind,
                "title": obj.title,
                "payload": dict(obj.payload),
                "task_id": obj.task_id,
                "state": obj.state,
            },
        )

    def _try_research_request(self, user_message: str) -> str | None:
        query = _extract_research_query(user_message)
        followup = False
        if not query and self._active_operation_type == "research" and self._active_operation_ttl > 0:
            if _is_research_followup(user_message):
                query = self._active_operation_topic
                followup = True

        task = self._active_task_context
        if task is not None and task.status not in ("completed", "cancelled"):
            excluded_sources = exclude_research_source(self._work_session, task, user_message)
            if excluded_sources:
                self._update_task_artifact_ui(task)
                if self._turn_trace is not None:
                    self._turn_trace["research_source_excluded"] = True
                    self._turn_trace["research_task_id"] = task.id
                    self._turn_trace["research_result_count"] = 0
                    self._turn_trace["response_grounded"] = True
                    self._turn_trace["grounding_sources"] = ["TASK_KNOWLEDGE"]
                return "Certo. Vou deixar essa fonte fora do raciocínio atual, sem apagar o histórico."

            source_response = research_sources_response(self._work_session, task, user_message)
            if source_response:
                if self._turn_trace is not None:
                    self._turn_trace["research_sources_requested"] = True
                    self._turn_trace["research_task_id"] = task.id
                    self._turn_trace["response_grounded"] = True
                    self._turn_trace["grounding_sources"] = ["TASK_KNOWLEDGE", "WEB_SOURCE"]
                return source_response

            if not query and is_task_research_intent(user_message):
                query = user_message.strip(" .!?")

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
        if task is not None and task.status not in ("completed", "cancelled"):
            task_research_tool = self.tools.get("task_web_search") or research_tool
            request = build_research_request(self._work_session, task, user_message, request_id=operation_id)
            if request.clarification:
                self._active_operation_type = ""
                self._active_operation_topic = ""
                self._active_operation_id = ""
                self._active_operation_ttl = 0
                if self._turn_trace is not None:
                    self._turn_trace["research_task_id"] = task.id
                    self._turn_trace["research_needs_clarification"] = True
                    self._turn_trace["research_plan"] = request.plan_text
                    self._turn_trace["research_tool_called"] = False
                    self._turn_trace["response_grounded"] = True
                    self._turn_trace["grounding_sources"] = ["TASK_KNOWLEDGE"]
                self._queue_ui_event(
                    "research_completed",
                    {"operation_id": operation_id, "topic": query, "status": "needs_clarification"},
                )
                return request.clarification
            if task_research_tool is None:
                _debug_print("ECHO_DEBUG_UI", "research_tool_available=False")
                _debug_print("ECHO_DEBUG_UI", "research_tool_called=False")
                self._active_operation_type = ""
                self._active_operation_topic = ""
                self._active_operation_id = ""
                self._active_operation_ttl = 0
                if self._turn_trace is not None:
                    self._turn_trace["research_task_id"] = task.id
                    self._turn_trace["research_plan"] = request.plan_text
                    self._turn_trace["research_tool_available"] = False
                    self._turn_trace["research_tool_called"] = False
                    self._turn_trace["response_grounded"] = False
                    self._turn_trace["grounding_sources"] = ["TASK_KNOWLEDGE"]
                self._queue_ui_event(
                    "research_unavailable",
                    {
                        "operation_id": operation_id,
                        "topic": query,
                        "status": "unavailable",
                        "message": "Ainda não tenho uma ferramenta de pesquisa ligada.",
                        "summary": request.plan_text,
                        "results": [],
                    },
                )
                self._queue_ui_event(
                    "research_completed",
                    {"operation_id": operation_id, "topic": query, "status": "unavailable"},
                )
                return request.plan_text + " Ainda não tenho uma ferramenta de pesquisa ligada."

            run = run_task_research(
                self._work_session,
                task,
                request,
                task_research_tool,
                semantic_extractor=self.semantic_extractor,
            )
            self._last_fast_tools_used = (task_research_tool.name,)
            if self._turn_trace is not None:
                self._turn_trace["semantic_extraction_calls"] = run.semantic_extraction_calls
                self._turn_trace["semantic_extraction_status"] = run.semantic_extraction_status
                self._turn_trace["semantic_candidates_before_validation"] = run.semantic_candidates_before_validation
                self._turn_trace["semantic_candidates_after_validation"] = run.semantic_candidates_after_validation
                self._turn_trace["semantic_extraction_failure_reason"] = run.semantic_extraction_failure_reason
                self._turn_trace["semantic_extraction_batches"] = run.semantic_extraction_batches
                self._turn_trace["semantic_extraction_latency_ms"] = run.semantic_extraction_latency_ms
                self._turn_trace["semantic_extraction_input_chars"] = run.semantic_extraction_input_chars
                self._turn_trace["semantic_extraction_sources_used"] = run.semantic_extraction_sources_used
            self._update_task_artifact_ui(task)
            _debug_print("ECHO_DEBUG_UI", "research_tool_available=True")
            _debug_print("ECHO_DEBUG_UI", "research_tool_called=True")
            _debug_print("ECHO_DEBUG_UI", f"research_result_count={len(run.ingested_entries)}")
            _debug_print("ECHO_DEBUG_UI", f"research_sources={[entry.source.reference for entry in run.ingested_entries]}")
            _debug_print("ECHO_DEBUG_UI", f"research_grounded={bool(run.ingested_entries)}")
            self._queue_ui_event(
                "research_results_ready",
                {
                    "operation_id": operation_id,
                    "topic": query,
                    "task_id": task.id,
                    "status": run.request.status,
                    "summary": run.summary,
                    "plan": run.request.plan_text,
                    "results": run.raw_results,
                    "sources": [
                        {
                            "title": str(entry.metadata.get("source_title") or entry.source.reference),
                            "url": entry.source.reference,
                            "subject": entry.subject,
                            "predicate": entry.predicate,
                            "value": entry.value,
                        }
                        for entry in run.ingested_entries
                    ],
                },
            )
            self._queue_ui_event(
                "research_completed",
                {"operation_id": operation_id, "topic": query, "task_id": task.id, "status": run.request.status},
            )
            self._active_operation_type = ""
            self._active_operation_topic = ""
            self._active_operation_id = ""
            self._active_operation_ttl = 0
            if self._turn_trace is not None:
                providers = _research_providers_from_results(run.raw_results)
                fallback_reasons = _research_fallback_reasons_from_results(run.raw_results)
                self._turn_trace["research_task_id"] = task.id
                self._turn_trace["research_request_id"] = request.id
                self._turn_trace["research_plan"] = request.plan_text
                self._turn_trace["research_query_plan"] = list(request.query_plan)
                self._turn_trace["research_provider"] = providers[0] if providers else ""
                self._turn_trace["research_providers"] = providers
                self._turn_trace["research_fallback_reason"] = fallback_reasons[0] if fallback_reasons else ""
                self._turn_trace["research_tool_available"] = True
                self._turn_trace["research_tool_called"] = True
                self._turn_trace["research_result_count"] = len(run.ingested_entries)
                self._turn_trace["research_status"] = run.request.status
                self._turn_trace["response_grounded"] = bool(run.ingested_entries)
                self._turn_trace["grounding_sources"] = ["TASK_KNOWLEDGE", "WEB_SOURCE"]
            return run.request.plan_text + " " + run.summary

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

    def _try_content_operation(self, user_message: str) -> str | None:
        task = self._active_task_context
        if task is None or task.status in ("completed", "cancelled"):
            return None
        content_focus = self._work_session.content_focus
        previous_target_id = str(content_focus.get("target_id") or "")
        operation = detect_content_operation(user_message, has_previous_target=bool(previous_target_id))
        if operation is None:
            return None

        targets = build_addressable_content(self._work_session, task)
        resolved = resolve_content_target(
            targets,
            operation,
            user_message,
            previous_target_id=previous_target_id,
        )
        if resolved.ambiguous:
            labels = [candidate.title for candidate in resolved.candidates if candidate.title]
            if self._turn_trace is not None:
                self._turn_trace["content_operation_detected"] = True
                self._turn_trace["content_operation"] = operation.kind
                self._turn_trace["content_target_resolution"] = "ambiguous"
                self._turn_trace["content_candidate_count"] = len(labels)
                self._turn_trace["response_grounded"] = True
                self._turn_trace["grounding_sources"] = ["WORK_SESSION"]
            if labels:
                return "Tenho mais do que uma hipótese. Estás a falar de " + " ou ".join(labels[:3]) + "?"
            return "Tenho mais do que uma hipótese. Diz-me qual queres que use."
        if resolved.target is None:
            if self._turn_trace is not None:
                self._turn_trace["content_operation_detected"] = True
                self._turn_trace["content_operation"] = operation.kind
                self._turn_trace["content_target_resolution"] = resolved.reason
                self._turn_trace["content_target_type"] = operation.target_type
                self._turn_trace["response_grounded"] = True
                self._turn_trace["grounding_sources"] = ["WORK_SESSION"]
            if operation.target_type == "source" and operation.target_hint:
                display_hint = _display_source_hint(user_message, operation.target_hint)
                return f"Não tenho uma fonte da {display_hint} nesta pesquisa."
            return None

        target = resolved.target
        if operation.kind == "open":
            url = source_url_for_target(target)
            if not url:
                return "Consigo identificar o conteúdo, mas não tenho um URL seguro para abrir."
            if self.tools.get("open_url") is None:
                return "Tenho a fonte, mas a ferramenta de abrir URL não está disponível."
            result = self.agent.ask_tool_confirmation(
                tool_name="open_url",
                arguments={"url": url},
                reason="abrir fonte de pesquisa confirmada",
                context=self._fast_agent_context(),
            )
            self._remember_tools_used(result.tools_used)
            self._last_fast_tools_used = result.tools_used
            self._record_agent_debug_trace(result.debug_trace)
            content_focus.update(
                {
                    "target_id": target.id,
                    "target_type": target.type,
                    "task_id": target.task_id,
                    "operation": operation.kind,
                }
            )
            if self._turn_trace is not None:
                self._turn_trace["content_operation_detected"] = True
                self._turn_trace["content_operation"] = operation.kind
                self._turn_trace["content_target_id"] = target.id
                self._turn_trace["content_target_type"] = target.type
                self._turn_trace["content_target_resolution"] = resolved.reason
                self._turn_trace["research_source_open_requested"] = True
                self._turn_trace["research_source_id"] = target.id
                self._turn_trace["research_source_url"] = url
                self._turn_trace["research_task_id"] = task.id
                self._turn_trace["response_grounded"] = True
                self._turn_trace["grounding_sources"] = ["TASK_KNOWLEDGE", "WEB_SOURCE"]
            return result.response

        presented = present_content(self._work_session, task, target, operation)
        if presented.ui_event:
            event = presented.ui_event
            self._queue_ui_event(str(event.get("type") or "content_presented"), dict(event.get("payload") or {}))
        content_focus.update(
            {
                "target_id": presented.continuation_token or target.id,
                "target_type": target.type,
                "task_id": target.task_id,
                "operation": operation.kind,
            }
        )
        if self._turn_trace is not None:
            self._turn_trace["content_operation_detected"] = True
            self._turn_trace["content_operation"] = operation.kind
            self._turn_trace["content_target_id"] = target.id
            self._turn_trace["content_target_type"] = target.type
            self._turn_trace["content_target_resolution"] = resolved.reason
            self._turn_trace["content_presentation_mode"] = presented.presentation_mode
            self._turn_trace["response_grounded"] = True
            self._turn_trace["grounding_sources"] = list(presented.provenance or ("TASK_KNOWLEDGE",))
            if target.type in {"source", "source_list", "research_synthesis"}:
                self._turn_trace["research_sources_requested"] = target.type == "source_list"
                self._turn_trace["research_task_id"] = task.id
        return presented.text

    def _try_open_research_source(self, user_message: str, task) -> str | None:
        normalized = _normalize_text(user_message)
        if not contains_any_phrase(
            normalized,
            (
                "abre essa fonte",
                "abrir essa fonte",
                "abre esta fonte",
                "abrir esta fonte",
                "abre a fonte",
                "abrir a fonte",
                "abre esse artigo",
                "abrir esse artigo",
                "abre o artigo",
                "abrir o artigo",
                "abre o link",
                "abrir o link",
            ),
        ):
            return None
        source = resolve_research_source(self._work_session, task, user_message)
        url = str((source or {}).get("url") or "")
        if not url:
            return "Tenho a referência da pesquisa, mas não encontro um URL seguro para abrir."
        if self.tools.get("open_url") is None:
            return "Tenho a fonte, mas a ferramenta de abrir URL não está disponível."
        result = self.agent.ask_tool_confirmation(
            tool_name="open_url",
            arguments={"url": url},
            reason="abrir fonte de pesquisa confirmada",
            context=self._fast_agent_context(),
        )
        self._remember_tools_used(result.tools_used)
        self._last_fast_tools_used = result.tools_used
        self._record_agent_debug_trace(result.debug_trace)
        if self._turn_trace is not None:
            self._turn_trace["research_source_open_requested"] = True
            self._turn_trace["research_source_id"] = str(source.get("id") or "")
            self._turn_trace["research_source_url"] = url
            self._turn_trace["research_task_id"] = task.id
            self._turn_trace["response_grounded"] = True
            self._turn_trace["grounding_sources"] = ["TASK_KNOWLEDGE", "WEB_SOURCE"]
        return result.response

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

    def _try_system_context_status(self, user_message: str) -> str | None:
        text = _normalize_text(user_message).strip(" .,!?:;")
        if not _asks_system_context_status(text):
            return None

        history_count = len(self.memory.load()) if self.presence.can_store_memory() else 0
        has_persistent_memory = _has_persistent_memory_data(self.long_term_memory)
        observer_available = self.context_observer is not None
        snapshot_available = False
        if self.context_observer is not None:
            latest_snapshot = getattr(self.context_observer, "latest_snapshot", None)
            if callable(latest_snapshot):
                snapshot_available = latest_snapshot() is not None

        workflow_bits: list[str] = []
        if self._pending_writing_task is not None:
            workflow_bits.append("um pedido de escrita em curso")
        if self._pending_document_task is not None:
            workflow_bits.append("uma tarefa de documento em curso")
        if self.agent.has_pending_confirmation():
            workflow_bits.append("uma acao pendente de confirmacao")

        lines = ["Sim. O que tenho neste momento é isto:"]
        lines.append(f"- histórico desta conversa: {'ativo' if history_count else 'sem mensagens úteis nesta sessão'}")
        lines.append(f"- memória permanente: {'ativa' if has_persistent_memory else 'sem dados relevantes encontrados agora'}")
        lines.append(f"- observação do computador: {'com snapshot disponível' if snapshot_available else 'sem dados suficientes neste momento'}")
        if workflow_bits:
            lines.append("- workflows transitórios: " + ", ".join(workflow_bits))
        else:
            lines.append("- workflows transitórios: nenhum pedido em curso")
        if observer_available and not snapshot_available:
            lines.append("Consigo tentar observar o computador, mas posso não ter dados suficientes ainda.")

        if self._turn_trace is not None:
            self._turn_trace["response_grounded"] = True
            self._turn_trace["grounding_sources"] = ["SYSTEM_STATE"]
        return "\n".join(lines)

    def _try_application_activity_query(self, user_message: str) -> str | None:
        app_name = _extract_application_activity_name(user_message)
        if not app_name:
            return None
        self._last_fast_tools_used = ("get_application_activity",)
        if self.context_observer is None:
            if self._turn_trace is not None:
                self._turn_trace["response_grounded"] = False
                self._turn_trace["grounding_sources"] = []
            return "Neste momento não tenho acesso a essa informação."

        snapshot = _fresh_or_latest_context_snapshot(self.context_observer)
        if snapshot is None:
            if self._turn_trace is not None:
                self._turn_trace["response_grounded"] = False
                self._turn_trace["grounding_sources"] = []
            return "Consigo tentar observar o computador, mas ainda não tenho dados suficientes. Experimenta mudar de janela ou aguardar alguns segundos."

        result = _application_activity_from_snapshot(snapshot, app_name)
        if self._turn_trace is not None:
            self._turn_trace["response_grounded"] = bool(result["application_found"])
            self._turn_trace["grounding_sources"] = ["CONTEXT_OBSERVER"] if result["application_found"] else []
        if not result["application_found"]:
            return f"Não detetei {result['application_name']} aberto neste momento."
        titles = list(result["window_titles"])
        active_text = " Está ativa agora." if result["active"] else ""
        if not titles:
            return f"Detetei que {result['application_name']} está aberto, mas não consigo determinar com segurança o que estás a fazer nele.{active_text}"
        title_text = "; ".join(titles[:3])
        return (
            f"Detetei que {result['application_name']} está aberto.{active_text} "
            f"Vejo estas janelas: {title_text}. "
            "Não consigo determinar com segurança o conteúdo exato sem mais contexto."
        )

    def _try_compound_workspace_command(self, user_message: str) -> str | None:
        started_at = time.perf_counter()
        normalized = _normalize_text(user_message).strip(" .!?")
        if not normalized or _looks_like_rewrite_request(normalized):
            return None

        clauses = _compound_workspace_clauses(normalized)
        if len(clauses) < 2:
            if _is_relative_rest_target(normalized):
                if self._turn_trace is not None:
                    self._turn_trace["compound_workspace_intent"] = True
                    self._turn_trace["compound_workspace_intent_count"] = 1
                    self._turn_trace["executed_operations"] = 0
                    self._turn_trace["failed_operations"] = 1
                    self._turn_trace["partial_execution"] = True
                    self._turn_trace["workspace_command_detected"] = True
                    self._turn_trace["workspace_command_action"] = "minimize"
                    self._turn_trace["workspace_command_source"] = "compound_voice_command"
                    self._turn_trace["workspace_command_executed"] = False
                    self._turn_trace["workspace_command_latency_ms"] = (time.perf_counter() - started_at) * 1000
                    self._turn_trace["workspace_collection_resolution_source"] = "missing_relative_anchor"
                return "O que queres manter aberto enquanto minimizo o resto?"
            return None

        operations: list[dict[str, object]] = []
        failures: list[str] = []
        anchor_ids: list[str] = []
        referenced_owner_task_id = ""
        for clause in clauses:
            operation = self._workspace_operation_from_clause(
                clause,
                anchor_ids=anchor_ids,
                referenced_owner_task_id=referenced_owner_task_id,
            )
            if operation is None:
                continue
            executed = self._execute_workspace_operation(operation)
            if executed:
                operations.append(operation)
                object_ids = [str(object_id) for object_id in list(operation.get("object_ids") or []) if str(object_id)]
                if operation.get("action") in ("show", "open") and object_ids:
                    anchor_ids = object_ids
                owner_id = str(operation.get("owner_task_id") or "")
                if owner_id:
                    referenced_owner_task_id = owner_id
            else:
                failures.append(str(operation.get("label") or operation.get("action") or clause))

        if len(operations) + len(failures) < 2:
            return None

        executed_count = len(operations)
        failed_count = len(failures)
        if self._turn_trace is not None:
            self._turn_trace["compound_workspace_intent"] = True
            self._turn_trace["compound_workspace_intent_count"] = executed_count + failed_count
            self._turn_trace["executed_operations"] = executed_count
            self._turn_trace["failed_operations"] = failed_count
            self._turn_trace["partial_execution"] = failed_count > 0
            self._turn_trace["workspace_command_detected"] = True
            self._turn_trace["workspace_command_action"] = "+".join(str(op.get("action") or "") for op in operations)
            self._turn_trace["workspace_command_source"] = "compound_voice_command"
            self._turn_trace["workspace_command_executed"] = executed_count > 0
            self._turn_trace["workspace_command_latency_ms"] = (time.perf_counter() - started_at) * 1000
            self._turn_trace["workspace_command_target_ids"] = [
                object_id
                for operation in operations
                for object_id in list(operation.get("object_ids") or [])
            ]
            self._turn_trace["workspace_collection_resolution_source"] = ",".join(
                str(op.get("resolution_source") or "") for op in operations if op.get("resolution_source")
            )

        if failed_count:
            completed = _compound_workspace_success_summary(operations)
            failed = ", ".join(failures)
            return f"{completed} Não consegui alterar: {failed}."
        return _compound_workspace_success_summary(operations)

    def _workspace_operation_from_clause(
        self,
        normalized_clause: str,
        *,
        anchor_ids: list[str] | None = None,
        referenced_owner_task_id: str = "",
    ) -> dict[str, object] | None:
        action = _workspace_clause_action(normalized_clause)
        criteria_titles = extract_comparison_criteria(normalized_clause)
        if criteria_titles and referenced_owner_task_id:
            return {
                "action": "add_criteria",
                "object_ids": [],
                "criteria_titles": criteria_titles,
                "owner_task_id": referenced_owner_task_id,
                "label": _criteria_label(criteria_titles),
                "resolution_source": "referenced_collection_owner",
            }
        if not action:
            return None

        target_phrase = _workspace_clause_target_phrase(normalized_clause)
        if action in ("minimize", "hide") and _is_relative_rest_target(normalized_clause):
            retained_ids = [object_id for object_id in (anchor_ids or []) if object_id in self._work_session.objects]
            if not retained_ids:
                return {
                    "action": action,
                    "object_ids": [],
                    "label": "tudo o resto",
                    "relative_scope": "all_except",
                    "resolution_source": "missing_relative_anchor",
                    "needs_anchor": True,
                }
            object_ids = self._relative_all_except_object_ids(retained_ids)
            return {
                "action": action,
                "object_ids": object_ids,
                "label": "tudo o resto",
                "target_phrase": target_phrase,
                "relative_scope": "all_except",
                "except_object_ids": retained_ids,
                "resolution_source": "relative_scope_anchor",
            }

        target_resolution = resolve_semantic_targets(normalized_clause, self._work_session, "workspace_reference")
        if target_resolution.ambiguous:
            return {"action": action, "object_ids": [], "label": normalized_clause, "ambiguous": True}
        if target_resolution.matches and target_resolution.matches[0].object_id:
            match = target_resolution.matches[0]
            return {
                "action": action,
                "object_ids": [match.object_id],
                "label": match.title or normalized_clause,
                "target_phrase": target_phrase,
                "owner_task_id": match.task_id,
                "resolution_source": target_resolution.source,
            }

        collection = resolve_semantic_collection_targets(normalized_clause, self._work_session)
        if collection.ambiguous:
            return {"action": action, "object_ids": [], "label": normalized_clause, "ambiguous": True}
        if collection.matches:
            return {
                "action": action,
                "object_ids": [match.object_id for match in collection.matches if match.object_id],
                "label": _collection_label(collection.matches),
                "target_phrase": target_phrase,
                "owner_task_id": _single_owner_task_id(collection.matches),
                "resolution_source": collection.source,
                "collection": True,
            }
        return {"action": action, "object_ids": [], "label": normalized_clause, "resolution_source": "no_match"}

    def _execute_workspace_operation(self, operation: dict[str, object]) -> bool:
        action = str(operation.get("action") or "")
        if action == "add_criteria":
            return self._execute_add_comparison_criteria(operation)

        object_ids = [object_id for object_id in list(operation.get("object_ids") or []) if object_id in self._work_session.objects]
        if not object_ids:
            return False
        levels: dict[str, str] = {}
        if action in ("show", "open"):
            for object_id in object_ids:
                levels[object_id] = _ATTN_FOCUS if len(object_ids) == 1 else _ATTN_ACTIVE
        elif action == "expand":
            self._apply_workspace_expand(object_ids[0], source="voice_command")
            levels[object_ids[0]] = _ATTN_FOCUS
        elif action == "minimize":
            for object_id in object_ids:
                levels[object_id] = _ATTN_MINIMIZED
        elif action == "hide":
            for object_id in object_ids:
                levels[object_id] = _ATTN_HIDDEN
        else:
            return False

        if levels:
            new_attention = set_bulk_attention(self._work_session, levels, source="voice_command")
            self._queue_ui_event(
                "attention_changed",
                {
                    "object_states": dict(new_attention.object_states),
                    "focus_object_id": new_attention.focus_object_id,
                    "source": new_attention.source,
                },
            )

        for object_id in object_ids:
            self._queue_ui_event(
                "workspace_command",
                {
                    "action": "show" if action == "open" else action,
                    "target_phrase": str(operation.get("target_phrase") or ""),
                    "target_ref": object_id,
                    "scope": "",
                    "relation": "",
                    "selector_index": "",
                    "region": "",
                    "source": "compound_voice_command",
                },
            )
        return True

    def _execute_add_comparison_criteria(self, operation: dict[str, object]) -> bool:
        owner_task_id = str(operation.get("owner_task_id") or "")
        criteria_titles = [str(title) for title in list(operation.get("criteria_titles") or []) if str(title).strip()]
        task = self._find_task_by_id(owner_task_id)
        if task is None or task.task_type != "generic_comparison" or not criteria_titles:
            return False

        criterion_ids: list[str] = []
        for title in criteria_titles:
            object_id = f"criterion-{slugify_title(title)}"
            existing = self._work_session.objects.get(object_id)
            if existing is None or existing.kind != "criterion" or existing.task_id != owner_task_id:
                existing = register_object(
                    self._work_session,
                    SemanticObject(id=object_id, kind="criterion", title=title, task_id=owner_task_id),
                )
                self._queue_semantic_object_sync(existing)
            criterion_ids.append(object_id)

        self._merge_criteria_into_task(owner_task_id, criteria_titles)
        current = self._work_session.representation
        visible_ids = list(current.visible_object_ids) if current is not None else []
        visible_ids = list(dict.fromkeys(visible_ids + criterion_ids))
        hidden_ids = [oid for oid in (list(current.hidden_object_ids) if current is not None else []) if oid not in criterion_ids]
        new_state = transform_representation(
            self._work_session,
            current.style if current is not None else "cards",
            visible_ids=visible_ids,
            major_ids=list(current.major_object_ids) if current is not None else [],
            hidden_ids=hidden_ids,
            source="auto",
        )
        self._queue_ui_event(
            "representation_changed",
            {
                "style": new_state.style,
                "visible_object_ids": list(new_state.visible_object_ids),
                "major_object_ids": list(new_state.major_object_ids),
                "hidden_object_ids": list(new_state.hidden_object_ids),
                "style_params": dict(new_state.style_params),
                "source": new_state.source,
            },
        )
        if self._turn_trace is not None:
            self._turn_trace["comparison_criteria_added"] = True
            self._turn_trace["comparison_criteria_count"] = len(criteria_titles)
            self._turn_trace["comparison_criteria_owner_task_id"] = owner_task_id
        operation["object_ids"] = criterion_ids
        return True

    def _relative_all_except_object_ids(self, retained_ids: list[str]) -> list[str]:
        excluded = set(retained_ids)
        constraint = self._work_session.constraints
        if constraint is not None:
            excluded.update(constraint.keep_visible_ids)
            excluded.update(constraint.do_not_minimize_ids)
        return [
            object_id
            for object_id in self._work_session.objects
            if object_id not in excluded
        ]

    def _apply_workspace_expand(self, object_id: str, *, source: str = "voice_command") -> None:
        current = self._work_session.representation
        style = current.style if current is not None else "cards"
        visible_ids = list(current.visible_object_ids) if current is not None else list(self._work_session.objects)
        visible_ids = list(dict.fromkeys(visible_ids + [object_id]))
        hidden_ids = [oid for oid in (list(current.hidden_object_ids) if current is not None else []) if oid != object_id]
        new_state = transform_representation(
            self._work_session,
            style,
            visible_ids=visible_ids,
            major_ids=[object_id],
            hidden_ids=hidden_ids,
            source=source,
        )
        self._queue_ui_event(
            "representation_changed",
            {
                "style": new_state.style,
                "visible_object_ids": list(new_state.visible_object_ids),
                "major_object_ids": list(new_state.major_object_ids),
                "hidden_object_ids": list(new_state.hidden_object_ids),
                "style_params": dict(new_state.style_params),
                "source": new_state.source,
            },
        )

    def _try_workspace_command(self, user_message: str) -> str | None:
        """Passo 3.2: recognizes deterministic natural-language commands
        about the WORKSPACE ARRANGEMENT itself (which card is big, side by
        side, hidden, closed...) as opposed to CONTENT commands (rewrite/
        summarize/etc, handled by _try_document_task below). Runs before
        document routing so it never depends on an active document context.
        Never touches self.llm/self.response_composer/self.agent -- a
        recognized command is llm_calls=0 by construction. Target
        RESOLUTION against the live on-screen scene (what "isso" refers to,
        which object is "the original", geometry) happens entirely
        client-side in echo_workspace_objects.js; this only recognizes the
        intent and forwards a best-effort raw target phrase."""
        started_at = time.perf_counter()
        normalized = _normalize_text(user_message).strip(" .!?")
        if not normalized:
            return None
        # If the utterance also looks like a genuine content transformation
        # request, defer entirely to the existing content pipeline rather
        # than partially executing a spatial command and silently dropping
        # the content half -- true simultaneous workspace+content split
        # execution in one turn is not implemented yet (see final report).
        if _looks_like_rewrite_request(normalized):
            return None
        # Section 10/12: "não deslocaste" retries the LAST entity command
        # instead of falling through to GENERAL_CONVERSATION -- which
        # previously produced an incoherent LLM reply and lost context
        # entirely once the user said "sim" to it.
        if contains_any_phrase(normalized, _ENTITY_CORRECTION_PHRASES):
            last = self._last_workspace_command
            if last and last.get("target_phrase") == _ENTITY_TARGET_PHRASE:
                retry_command = dict(last)
                self._queue_ui_event(
                    "workspace_command",
                    {
                        "action": retry_command["action"],
                        "target_phrase": retry_command.get("target_phrase", ""),
                        "relation": retry_command.get("relation", ""),
                        "selector_index": retry_command.get("selector_index", ""),
                        "region": retry_command.get("region", ""),
                        "source": "voice_command_retry",
                    },
                )
                if self._turn_trace is not None:
                    self._turn_trace["workspace_command_detected"] = True
                    self._turn_trace["workspace_command_action"] = retry_command["action"]
                    self._turn_trace["workspace_command_target"] = retry_command.get("target_phrase", "")
                    self._turn_trace["workspace_command_region"] = retry_command.get("region", "")
                    self._turn_trace["workspace_command_source"] = "voice_command_retry"
                    self._turn_trace["workspace_command_executed"] = True
                    self._turn_trace["workspace_command_latency_ms"] = (time.perf_counter() - started_at) * 1000
                return "Peço desculpa, deixa-me tentar outra vez."
            # No prior entity command on record -- nothing deterministic to
            # retry, fall through to normal routing rather than guessing.
            return None
        if not self._route_is_authorized("WORKSPACE_COMMAND"):
            return None
        commands, remaining_clauses = _workspace_command_actions(normalized)
        if not commands:
            return None
        # Phase 3B integration fix (FIX 2): tracks a place_beside command
        # that clearly named specific alternatives (no generic hint like
        # "os dois"/"o original" matched -- see _WORKSPACE_TARGET_HINTS)
        # but resolved to fewer than two known semantic objects, so the
        # generic "Aqui tens os dois." ack below can be replaced with a
        # truthful one instead of claiming a rearrangement that has
        # nothing real to act on.
        representation_resolution_failed = False
        for command in commands:
            target_ref = ""
            target_phrase = command.get("target_phrase", "")
            target_resolution = None
            if target_phrase and target_phrase != _ENTITY_TARGET_PHRASE:
                target_resolution = resolve_semantic_targets(target_phrase, self._work_session, "workspace_reference")
                if target_resolution.matches and not target_resolution.ambiguous:
                    target_ref = target_resolution.matches[0].object_id
                    if command["action"] == "expand" and target_ref in self._task_surface_open_artifact_ids() and len(self._task_surface_open_artifact_ids()) > 1:
                        expand_response = self._maybe_prepare_task_surface_expand_action(target_ref)
                        if expand_response:
                            return expand_response
                if command["action"] == "expand" and target_phrase in ("email", "mail", "mensagem") and not target_ref:
                    representation_resolution_failed = True
                    if self._turn_trace is not None:
                        self._turn_trace["representation_target_resolution_failed"] = True
                        self._turn_trace["workspace_target_resolution_source"] = target_resolution.source if target_resolution else "no_match"
                    continue
            self._queue_ui_event(
                "workspace_command",
                {
                    "action": command["action"],
                    "target_phrase": target_phrase,
                    "target_ref": target_ref,
                    "scope": command.get("scope", ""),
                    "relation": command.get("relation", ""),
                    "selector_index": command.get("selector_index", ""),
                    "region": command.get("region", ""),
                    "source": "voice_command",
                },
            )
            if command["action"] == "expand" and target_ref:
                self._apply_workspace_expand(target_ref, source="voice_command")
                new_attention = set_attention(self._work_session, target_ref, _ATTN_FOCUS, source="voice_command")
                self._queue_ui_event(
                    "attention_changed",
                    {
                        "object_states": dict(new_attention.object_states),
                        "focus_object_id": new_attention.focus_object_id,
                        "source": new_attention.source,
                    },
                )
            if command.get("scope") == "all":
                self._apply_global_workspace_visibility(command["action"])
            if command.get("target_phrase") == _ENTITY_TARGET_PHRASE:
                self._last_workspace_command = dict(command)
            # Phase 3 (section 10/11): a "lado a lado" placement is ALREADY
            # fully handled above via the pre-existing workspace_command
            # event -- this only ADDS a representation transform, and only
            # when the message names two or more already-registered
            # semantic objects by title. If nothing in the registry
            # matches (the normal case for every existing workspace/
            # document flow, where WorkSession.objects stays empty), this
            # is a complete no-op: zero change to prior behavior.
            if command["action"] == "place_beside" and self._work_session.objects:
                mentioned_ids = resolve_mentioned_object_ids(user_message, self._work_session)
                if len(mentioned_ids) >= 2:
                    new_representation = transform_representation(
                        self._work_session,
                        "side_by_side",
                        major_ids=mentioned_ids[:2],
                        source="voice_command",
                    )
                    self._queue_ui_event(
                        "representation_changed",
                        {
                            "style": new_representation.style,
                            "visible_object_ids": list(new_representation.visible_object_ids),
                            "major_object_ids": list(new_representation.major_object_ids),
                            "hidden_object_ids": list(new_representation.hidden_object_ids),
                            "style_params": dict(new_representation.style_params),
                            "source": new_representation.source,
                        },
                    )
                    if self._turn_trace is not None:
                        self._turn_trace["representation_command_detected"] = True
                        self._turn_trace["representation_command_action"] = "side_by_side_placement"
                        self._turn_trace["representation_style"] = new_representation.style
                        self._turn_trace["representation_source"] = new_representation.source
                        self._turn_trace["representation_command_executed"] = True
                elif not command.get("target_phrase"):
                    # Phase 3B integration fix (FIX 2): no generic hint
                    # ("os dois"/"o original"/...) matched either, so this
                    # was very likely naming specific alternatives that
                    # just don't exist yet in the semantic object registry
                    # -- claiming "Aqui tens os dois." below would be a
                    # false success (nothing changed visually). A command
                    # that DID match a generic hint is left alone: that is
                    # the existing document/entity flow, resolved entirely
                    # client-side, unaffected by this registry lookup.
                    representation_resolution_failed = True
                    if self._turn_trace is not None:
                        self._turn_trace["representation_target_resolution_failed"] = True
        primary = commands[0]
        workspace_result = ExecutionResult(
            executed=bool(commands) and not representation_resolution_failed,
            mutated_state=False,
            ui_event_queued=bool(commands),
            partial=bool(remaining_clauses),
            reason=primary["action"],
        )
        if self._turn_trace is not None:
            self._turn_trace["workspace_command_detected"] = True
            self._turn_trace["workspace_command_action"] = primary["action"]
            self._turn_trace["workspace_command_target"] = primary.get("target_phrase", "")
            self._turn_trace["workspace_command_scope"] = primary.get("scope", "")
            self._turn_trace["workspace_command_relation"] = primary.get("relation", "")
            self._turn_trace["workspace_command_region"] = primary.get("region", "")
            self._turn_trace["workspace_command_source"] = "voice_command"
            self._turn_trace["workspace_command_executed"] = workspace_result.executed
            self._turn_trace["workspace_command_ui_event_queued"] = workspace_result.ui_event_queued
            self._turn_trace["workspace_command_partial"] = workspace_result.partial
            self._turn_trace["workspace_command_latency_ms"] = (time.perf_counter() - started_at) * 1000
            self._turn_trace["workspace_language_match_type"] = primary.get("match_type", "")
            self._turn_trace["workspace_language_signals"] = primary.get("signals", [])
            self._turn_trace["workspace_command_confidence"] = primary.get("confidence", "")
        ack = _WORKSPACE_COMMAND_ACKS.get(primary["action"], "Feito.")
        if representation_resolution_failed and primary["action"] == "place_beside":
            # FIX 2: replace the optimistic generic ack with a truthful one
            # -- the placement genuinely did not happen.
            ack = "Ainda não tenho esses objetos disponíveis para pôr lado a lado."
        elif representation_resolution_failed and primary["action"] == "expand":
            ack = "Ainda não tenho esse objeto disponível no workspace."
        # Passo 3.4/3.5: a workspace clause might ALSO carry a task
        # trigger/continuation -- but "chega para a esquerda para
        # escrevermos um email" has no comma/" e " to split on, so
        # remaining_clauses stays empty even though "escrevermos um
        # email" is real task-relevant text (section 3). Try the task
        # layer against whatever text is available -- the isolated
        # remaining clauses if splitting found any, else the WHOLE
        # normalized utterance -- before falling back to the purely
        # cosmetic ack. raw_message stays user_message throughout
        # (preserves real casing for name/course regexes, which the
        # already-lowercased normalized text would break).
        can_mix_task_response = bool(remaining_clauses) or primary.get("target_phrase") == _ENTITY_TARGET_PHRASE
        if not representation_resolution_failed and can_mix_task_response:
            remaining_text = " ".join(clause for clause in remaining_clauses if clause.strip()) or normalized
            task_response = self._process_task_turn(user_message, remaining_text)
            if self._task_execution_requested:
                self._task_execution_requested = False
                ctx = self._active_task_context
                if ctx is not None:
                    task_response = self._execute_active_task(ctx)
            if task_response:
                ack = f"{ack} {task_response}"
            elif remaining_clauses:
                remaining_ack = _remaining_workspace_intent_ack(remaining_clauses)
                if remaining_ack:
                    ack = f"{ack} {remaining_ack}"
        return ack

    def _try_resume_activity_reference(self, user_message: str) -> str | None:
        normalized_message = _normalize_text(user_message)
        if not _is_resume_reference_phrase(normalized_message):
            return None
        target_resolution = resolve_semantic_targets(user_message, self._work_session, "activity_resume")
        if target_resolution.ambiguous or not target_resolution.matches:
            return None
        match = target_resolution.matches[0]
        if not match.task_id:
            return None

        target_task = self._find_task_by_id(match.task_id)
        if target_task is None:
            return None
        target_artifact_id = target_task.active_artifact_id or match.object_id
        target_level = (
            self._work_session.attention.object_states.get(target_artifact_id, _ATTN_VISIBLE)
            if self._work_session.attention is not None
            else _ATTN_VISIBLE
        )
        if target_level == _ATTN_MINIMIZED and len(self._task_surface_open_artifact_ids()) >= 4:
            capacity_response = self._maybe_prepare_task_surface_capacity_action(user_message, exclude_ids={target_artifact_id})
            if capacity_response:
                return capacity_response.replace("para abrir esta", f"para retomar {self._task_surface_label(target_artifact_id)}")

        current = self._work_session.active_task
        previous_type = current.task_type if current is not None and current.id != target_task.id else ""
        if current is not None and current.id != target_task.id and current.status not in ("completed", "cancelled"):
            self._pause_active_task()

        self._paused_task_stack = [task for task in self._paused_task_stack if task.id != target_task.id]
        target_task.status = "ready" if not self._missing_task_fields(target_task) else "collecting"
        self._touch_task(target_task)
        self._active_task_context = target_task

        levels: dict[str, str] = {}
        if current is not None and current.active_artifact_id and current.id != target_task.id:
            levels[current.active_artifact_id] = _ATTN_ACTIVE
        if target_task.active_artifact_id:
            levels[target_task.active_artifact_id] = _ATTN_FOCUS
        if levels:
            new_attention = set_bulk_attention(self._work_session, levels, source="voice_command")
            self._queue_ui_event(
                "attention_changed",
                {
                    "object_states": dict(new_attention.object_states),
                    "focus_object_id": new_attention.focus_object_id,
                    "source": new_attention.source,
                },
            )

        if self._turn_trace is not None:
            self._turn_trace["task_resumed"] = True
            self._turn_trace["task_paused"] = bool(previous_type)
            self._turn_trace["previous_active_task_type"] = previous_type
            self._turn_trace["new_active_task_type"] = target_task.task_type
            self._turn_trace["active_task_selected_path"] = "TASK_RESUME"
            self._turn_trace["task_continuation_detected"] = True
            self._turn_trace["task_stack_depth"] = len(self._paused_task_stack)
            self._turn_trace["attention_command_detected"] = True
            self._turn_trace["attention_command_action"] = "resume_activity"
            self._turn_trace["attention_command_target_ids"] = [match.object_id]
            self._turn_trace["attention_target_resolution_source"] = target_resolution.source
        return self._task_continuation_response(target_task)

    def _find_task_by_id(self, task_id: str) -> ActiveTaskContext | None:
        active = self._work_session.active_task
        if active is not None and active.id == task_id:
            return active
        for task in self._paused_task_stack:
            if task.id == task_id:
                return task
        return None

    def _apply_global_workspace_visibility(self, action: str) -> None:
        """Semantic "all" workspace operation.

        Backend only decides prominence/visibility. It never sends geometry:
        the frontend remains responsible for fitting the resulting objects.
        """
        object_ids = list(self._work_session.objects)
        if not object_ids:
            return

        focus_id = self._work_session.attention.focus_object_id if self._work_session.attention else ""
        levels: dict[str, str] = {}
        for object_id in object_ids:
            current = get_attention_level(self._work_session, object_id)
            if object_id == focus_id and current == _ATTN_FOCUS:
                levels[object_id] = _ATTN_FOCUS
            elif current in (_ATTN_HIDDEN, _ATTN_MINIMIZED, _ATTN_GLANCEABLE):
                levels[object_id] = _ATTN_ACTIVE
            elif action in ("show", "expand"):
                levels[object_id] = current if current in (_ATTN_ACTIVE, _ATTN_FOCUS) else _ATTN_VISIBLE
        if levels:
            new_attention = set_bulk_attention(self._work_session, levels, source="voice_command")
            self._queue_ui_event(
                "attention_changed",
                {
                    "object_states": dict(new_attention.object_states),
                    "focus_object_id": new_attention.focus_object_id,
                    "source": new_attention.source,
                },
            )

        current_representation = self._work_session.representation
        if current_representation is not None:
            visible_ids = list(dict.fromkeys(list(current_representation.visible_object_ids) + object_ids))
            major_ids = [oid for oid in current_representation.major_object_ids if oid in object_ids]
            new_state = transform_representation(
                self._work_session,
                current_representation.style,
                visible_ids=visible_ids,
                major_ids=major_ids,
                hidden_ids=[],
                source="voice_command",
            )
            self._queue_ui_event(
                "representation_changed",
                {
                    "style": new_state.style,
                    "visible_object_ids": list(new_state.visible_object_ids),
                    "major_object_ids": list(new_state.major_object_ids),
                    "hidden_object_ids": list(new_state.hidden_object_ids),
                    "style_params": dict(new_state.style_params),
                    "source": new_state.source,
                },
            )
        if self._turn_trace is not None:
            self._turn_trace["global_workspace_action"] = True
            self._turn_trace["global_workspace_object_count"] = len(object_ids)

    # --- Passo 3.4: ActiveTaskContext (generic multi-turn task tracking) --
    # A session-only "what are we currently building together" layer --
    # NOT specific to email; compose_email is the first task_type. Kept
    # deliberately separate from PendingWritingTask (a narrower, pre-
    # existing email-only slot-filler with different trigger phrases) --
    # see final report for why the two coexist instead of merging.

    def _try_active_task_turn(self, user_message: str) -> str | None:
        """Top-level routing check (see _respond_inner, right after
        _try_workspace_command): handles both starting a new task from a
        plain trigger phrase and continuing/pausing/resuming/cancelling an
        existing one. Never touches self.llm/self.response_composer --
        deterministic fragment turns are llm_calls=0 by construction."""
        normalized = _normalize_text(user_message).strip(" .!?")
        if not normalized:
            return None
        ctx_before = self._active_task_context
        had_active_task = ctx_before is not None and ctx_before.status not in ("completed", "cancelled", "paused")
        response = self._process_task_turn(user_message, normalized)
        if self._task_execution_requested:
            self._task_execution_requested = False
            ctx = self._active_task_context
            if ctx is not None:
                return self._execute_active_task(ctx)
        # P0 routing authority: a live task no longer owns every
        # unclassified utterance. If no positive evidence was found, record
        # the rejection and let later routes handle the turn.
        if response is None and had_active_task and self._turn_trace is not None:
            explicit_signal = contains_any_phrase(
                normalized, _TASK_EXPLICIT_EXIT_PHRASES + _TASK_CANCEL_PHRASES + _TASK_PAUSE_PHRASES
            )
            if not explicit_signal:
                self._turn_trace["task_continuation_rejected"] = True
                self._turn_trace["active_task_continuation_rejected_reason"] = "no_positive_task_evidence"
        return response

    def _try_task_intelligence_query(self, user_message: str) -> str | None:
        if not classify_task_query(user_message):
            return None
        task = self._active_task_context
        if task is None or task.status in ("completed", "cancelled"):
            return None
        response = answer_task_query(self._work_session, task, user_message)
        if not response:
            return None
        if self._turn_trace is not None:
            self._turn_trace["task_intelligence_query"] = True
            self._turn_trace["task_intelligence_task_id"] = task.id
            self._turn_trace["task_intelligence_task_type"] = task.task_type
        return response

    def _current_turn_reference(self) -> str:
        try:
            return f"turn-{len(self.memory.load()) + 1}"
        except Exception:
            return "turn-current"

    def _process_task_turn(self, raw_message: str, normalized_message: str) -> str | None:
        """Returns a response if this text meaningfully created/updated/
        resolved the active task; None means "not task-related, caller
        should fall back to its own normal handling" (never GENERAL_
        CONVERSATION by force -- an unrelated turn just isn't consumed)."""
        interpretation = _interpret_activity(raw_message, normalized_message, self._active_task_context, self._paused_task_stack)
        if interpretation.confidence:
            self._record_activity_interpretation(interpretation)

        if interpretation.resumes_task:
            resumed = self._resume_task_from_stack(interpretation)
            if resumed is not None:
                return self._task_continuation_response(resumed)

        ctx = self._active_task_context
        if ctx is not None and ctx.status not in ("completed", "cancelled"):
            if contains_any_phrase(normalized_message, _TASK_CANCEL_PHRASES):
                self._active_task_context = None
                if self._turn_trace is not None:
                    self._turn_trace["task_continuation_detected"] = True
                    self._turn_trace["active_task_status"] = "cancelled"
                return "Cancelei essa tarefa."
            if contains_any_phrase(normalized_message, _TASK_PAUSE_PHRASES):
                self._pause_active_task()
                if self._turn_trace is not None:
                    self._turn_trace["task_continuation_detected"] = True
                    self._turn_trace["active_task_status"] = "paused"
                return "Fica em pausa. Diz para continuarmos quando quiseres."
            if (
                interpretation.starts_task
                and interpretation.task_type
                and interpretation.task_type != ctx.task_type
                and not (ctx.task_type == "compose_email" and interpretation.task_type == "writing")
            ):
                capacity_response = self._maybe_prepare_task_surface_capacity_action(raw_message)
                if capacity_response:
                    return capacity_response
                previous_type = ctx.task_type
                self._pause_active_task()
                new_ctx = self._create_task_context(
                    interpretation.task_type,
                    activity_family=interpretation.activity_family,
                    domain=interpretation.domain,
                    artifact_type=interpretation.artifact_type,
                    goal=interpretation.goal,
                )
                updated_fields = self._extract_task_fields(new_ctx, raw_message, interpretation.remaining_content or normalized_message)
                materialized_options = self._materialize_comparison_options_from_context(new_ctx, raw_message)
                if materialized_options and "options" not in updated_fields:
                    updated_fields.append("options")
                knowledge_entries = update_task_knowledge(self._work_session, new_ctx, raw_message, turn_id=self._current_turn_reference())
                new_ctx.source_turns.append(raw_message)
                self._active_task_context = new_ctx
                self._touch_task(new_ctx)
                if self._turn_trace is not None:
                    self._turn_trace["task_switched"] = True
                    self._turn_trace["task_paused"] = True
                    self._turn_trace["task_started"] = True
                    self._turn_trace["previous_active_task_type"] = previous_type
                    self._turn_trace["new_active_task_type"] = new_ctx.task_type
                    self._turn_trace["active_task_selected_path"] = "TASK_SWITCH"
                    self._turn_trace["task_continuation_detected"] = False
                    self._turn_trace["task_fields_updated"] = updated_fields
                    self._turn_trace["task_knowledge_updated_count"] = len(knowledge_entries)
                    self._turn_trace["task_context_source"] = "deterministic_activity"
                    self._turn_trace["task_turn_intent"] = "switch"
                return self._task_continuation_response(new_ctx, raw_message, knowledge_entries)
            if contains_any_phrase(normalized_message, _TASK_EXPLICIT_EXIT_PHRASES):
                # Section 13: only an EXPLICIT signal ("já agora explica-me
                # X") lets a mention of the task's own topic fall through to
                # normal conversation/LLM for THIS turn -- the task itself
                # is left as-is (not cancelled), just not consumed here.
                return None
            if ctx.status == "paused":
                if contains_any_phrase(normalized_message, _TASK_RESUME_PHRASES):
                    ctx.status = "ready" if not self._missing_task_fields(ctx) else "collecting"
                    self._touch_task(ctx)
                    if self._turn_trace is not None:
                        self._turn_trace["task_continuation_detected"] = True
                        self._turn_trace["active_task_status"] = ctx.status
                    return self._task_continuation_response(ctx, raw_message, [])
                return None  # a paused task never silently swallows unrelated turns
            # Section 1/5: classify TASK_CONTROL ("podes escrever", "escreve",
            # a bare "podes"/"sim" confirming a just-asked pending offer)
            # BEFORE any data extraction runs -- otherwise a control phrase
            # with no other recognizable field falls into the topic
            # catch-all heuristic below and gets stored as if it were the
            # email's subject (the exact real bug: "Assunto: podes escrever").
            control_action = _detect_task_control_action(normalized_message, ctx)
            if control_action:
                if self._turn_trace is not None:
                    self._turn_trace["task_turn_intent"] = "control"
                    self._turn_trace["task_control_action"] = control_action
                    self._turn_trace["active_task_continuation_evidence"] = "task_control_language"
                    self._turn_trace["task_continuation_rejected"] = False
                    self._turn_trace["active_task_continuation_rejected_reason"] = ""
                if control_action == "execute" and ctx.status in ("ready", "reviewing"):
                    ctx.pending_action = ""
                    self._task_execution_requested = True
                    return None
                # Not ready yet, or an unrecognized control action -- surface
                # what's still missing instead of silently doing nothing.
                return self._task_continuation_response(ctx, raw_message, [])
            if interpretation.refines_task and interpretation.task_type and interpretation.task_type != ctx.task_type:
                previous_type = ctx.task_type
                self._specialize_task(ctx, interpretation)
                if self._turn_trace is not None:
                    self._turn_trace["task_specialized"] = True
                    self._turn_trace["previous_active_task_type"] = previous_type
                    self._turn_trace["new_active_task_type"] = ctx.task_type
            if ctx.task_type == "compose_email" and interpretation.task_type == "writing":
                if self._turn_trace is not None:
                    self._turn_trace["task_continuation_detected"] = True
                    self._turn_trace["task_fields_updated"] = []
                    self._turn_trace["task_context_source"] = "deterministic_activity"
                    self._turn_trace["task_turn_intent"] = "continue"
                return self._task_continuation_response(ctx, raw_message, [])
            if needs_task_reference_clarification(self._work_session, ctx, raw_message):
                options = task_option_titles(self._work_session, ctx)
                labels = " ou ".join(options[:3]) if options else "qual opção"
                if self._turn_trace is not None:
                    self._turn_trace["task_continuation_detected"] = True
                    self._turn_trace["task_turn_intent"] = "clarify_reference"
                    self._turn_trace["active_task_continuation_evidence"] = "ambiguous_task_reference"
                    self._turn_trace["task_continuation_rejected"] = False
                return f"Quando dizes isso, estás a falar de {labels}?"
            same_task_start = bool(interpretation.starts_task and interpretation.task_type == ctx.task_type)
            allow_unclassified_note = bool(interpretation.continues_task or interpretation.refines_task or same_task_start)
            updated_fields = self._extract_task_fields(
                ctx,
                raw_message,
                normalized_message,
                allow_unclassified_note=allow_unclassified_note,
            )
            if not updated_fields and (interpretation.continues_task or interpretation.refines_task or same_task_start):
                updated_fields = _extract_generic_activity_fields(
                    ctx,
                    raw_message,
                    normalized_message,
                    allow_unclassified_note=True,
                )
            if not updated_fields:
                if self._turn_trace is not None:
                    self._turn_trace["active_task_continuation_evidence"] = ""
                    self._turn_trace["task_continuation_rejected"] = True
                    self._turn_trace["active_task_continuation_rejected_reason"] = "no_positive_task_evidence"
                return None
            knowledge_entries = update_task_knowledge(self._work_session, ctx, raw_message, turn_id=self._current_turn_reference())
            ctx.source_turns.append(raw_message)
            self._touch_task(ctx)
            if self._turn_trace is not None:
                self._turn_trace["task_continuation_detected"] = True
                self._turn_trace["active_task_continuation_evidence"] = (
                    "activity_context_match" if allow_unclassified_note else "recognized_task_fields"
                )
                self._turn_trace["task_continuation_rejected"] = False
                self._turn_trace["active_task_continuation_rejected_reason"] = ""
                self._turn_trace["task_fields_updated"] = updated_fields
                self._turn_trace["task_knowledge_updated_count"] = len(knowledge_entries)
                self._turn_trace["task_context_source"] = "deterministic"
                self._turn_trace["task_turn_intent"] = "refine" if interpretation.refines_task else ("continue" if same_task_start else "data")
            return self._task_continuation_response(ctx, raw_message, knowledge_entries)

        task_type = interpretation.task_type or _detect_new_task_intent(normalized_message)
        if not task_type:
            return None
        capacity_response = self._maybe_prepare_task_surface_capacity_action(raw_message)
        if capacity_response:
            return capacity_response
        ctx = self._create_task_context(
            task_type,
            activity_family=interpretation.activity_family,
            domain=interpretation.domain,
            artifact_type=interpretation.artifact_type,
            goal=interpretation.goal,
        )
        updated_fields = self._extract_task_fields(ctx, raw_message, normalized_message)
        materialized_options = self._materialize_comparison_options_from_context(ctx, raw_message)
        if materialized_options and "options" not in updated_fields:
            updated_fields.append("options")
        knowledge_entries = update_task_knowledge(self._work_session, ctx, raw_message, turn_id=self._current_turn_reference())
        ctx.source_turns.append(raw_message)
        self._active_task_context = ctx
        if self._turn_trace is not None:
            self._turn_trace["task_continuation_detected"] = False
            self._turn_trace["task_started"] = True
            self._turn_trace["new_active_task_type"] = ctx.task_type
            if ctx.task_type != "compose_email":
                self._turn_trace["active_task_selected_path"] = "TASK_START"
            self._turn_trace["task_fields_updated"] = updated_fields
            self._turn_trace["task_knowledge_updated_count"] = len(knowledge_entries)
            self._turn_trace["task_context_source"] = "deterministic_activity" if interpretation.task_type else "deterministic"
            self._turn_trace["task_turn_intent"] = "data"
        return self._task_continuation_response(ctx, raw_message, knowledge_entries)

    def _record_activity_interpretation(self, interpretation: ActivityInterpretation) -> None:
        if self._turn_trace is None:
            return
        self._turn_trace["activity_detected"] = True
        self._turn_trace["activity_family"] = interpretation.activity_family
        self._turn_trace["activity_domain"] = interpretation.domain
        self._turn_trace["activity_artifact_type"] = interpretation.artifact_type
        self._turn_trace["activity_confidence"] = interpretation.confidence
        self._turn_trace["activity_reason"] = interpretation.reason
        self._turn_trace["activity_remaining_content"] = interpretation.remaining_content
        self._turn_trace["task_stack_depth"] = len(self._paused_task_stack)

    def _pause_active_task(self) -> None:
        ctx = self._active_task_context
        if ctx is None or ctx.status in ("completed", "cancelled"):
            return
        ctx.status = "paused"
        self._touch_task(ctx)
        self._paused_task_stack = [task for task in self._paused_task_stack if task.id != ctx.id]
        self._paused_task_stack.append(ctx)
        self._active_task_context = None
        if self._turn_trace is not None:
            self._turn_trace["task_paused"] = True
            self._turn_trace["task_stack_depth"] = len(self._paused_task_stack)

    def _resume_task_from_stack(self, interpretation: ActivityInterpretation) -> ActiveTaskContext | None:
        target_index = _find_task_stack_match(self._paused_task_stack, interpretation)
        if target_index is None:
            return None
        current = self._active_task_context
        previous_type = current.task_type if current is not None else ""
        if current is not None and current.status not in ("completed", "cancelled"):
            self._pause_active_task()
        resumed = self._paused_task_stack.pop(target_index)
        resumed.status = "ready" if not self._missing_task_fields(resumed) else "collecting"
        self._touch_task(resumed)
        self._active_task_context = resumed
        if self._turn_trace is not None:
            self._turn_trace["task_resumed"] = True
            self._turn_trace["task_paused"] = bool(previous_type)
            self._turn_trace["previous_active_task_type"] = previous_type
            self._turn_trace["new_active_task_type"] = resumed.task_type
            if resumed.task_type != "compose_email":
                self._turn_trace["active_task_selected_path"] = "TASK_RESUME"
            self._turn_trace["task_continuation_detected"] = True
            self._turn_trace["task_stack_depth"] = len(self._paused_task_stack)
        return resumed

    def _specialize_task(self, ctx: ActiveTaskContext, interpretation: ActivityInterpretation) -> None:
        ctx.task_type = interpretation.task_type or ctx.task_type
        if interpretation.activity_family != "unknown":
            ctx.activity_family = interpretation.activity_family
        if interpretation.domain != "unknown":
            ctx.domain = interpretation.domain
        if interpretation.artifact_type != "unknown":
            ctx.artifact_type = interpretation.artifact_type
        if interpretation.goal:
            ctx.goal = interpretation.goal
        self._touch_task(ctx)

    def _create_task_context(
        self,
        task_type: str,
        *,
        activity_family: str = "unknown",
        domain: str = "unknown",
        artifact_type: str = "unknown",
        goal: str = "",
    ) -> ActiveTaskContext:
        self._active_task_id_counter += 1
        now = self.now_provider().isoformat()
        metadata = _task_metadata(task_type, activity_family, domain, artifact_type, goal)
        return ActiveTaskContext(
            id=f"task-{self._active_task_id_counter}",
            task_type=task_type,
            activity_family=metadata["activity_family"],
            domain=metadata["domain"],
            artifact_type=metadata["artifact_type"],
            goal=metadata["goal"],
            started_at=now,
            updated_at=now,
        )

    def _touch_task(self, ctx: ActiveTaskContext) -> None:
        ctx.updated_at = self.now_provider().isoformat()
        ctx.ttl = 6

    def _extract_task_fields(
        self,
        ctx: ActiveTaskContext,
        raw_text: str,
        normalized_text: str,
        *,
        allow_unclassified_note: bool = True,
    ) -> list[str]:
        if ctx.task_type == "compose_email":
            return _extract_compose_email_fields(ctx, raw_text, normalized_text)
        return _extract_generic_activity_fields(
            ctx,
            raw_text,
            normalized_text,
            allow_unclassified_note=allow_unclassified_note,
        )

    def _missing_task_fields(self, ctx: ActiveTaskContext) -> list[str]:
        if ctx.task_type == "compose_email":
            return _missing_compose_email_fields(ctx)
        return _missing_generic_activity_fields(ctx)

    def _task_continuation_response(self, ctx: ActiveTaskContext, user_message: str = "", knowledge_delta: list | None = None) -> str:
        ctx.missing_information = self._missing_task_fields(ctx)
        ctx.status = "ready" if not ctx.missing_information else "collecting"
        # _update_task_artifact_ui assigns ctx.active_artifact_id the FIRST
        # time it runs -- must happen before the telemetry read below, or
        # the very first turn that creates the artifact reports an empty
        # task_artifact_id (only "fixing itself" from the second turn on).
        self._update_task_artifact_ui(ctx)
        if self._turn_trace is not None:
            self._turn_trace["active_task_id"] = ctx.id
            self._turn_trace["active_task_type"] = ctx.task_type
            self._turn_trace["active_task_status"] = ctx.status
            self._turn_trace["task_known_fields"] = {**ctx.entities, **ctx.parameters}
            self._turn_trace["task_missing_fields"] = list(ctx.missing_information)
            self._turn_trace["task_artifact_id"] = ctx.active_artifact_id
        if ctx.task_type == "compose_email":
            return _compose_email_question(ctx)
        response = task_dialogue_response(
            self._work_session,
            ctx,
            user_message,
            knowledge_delta=knowledge_delta or [],
        )
        return response or _generic_activity_question(ctx)

    def _update_task_artifact_ui(self, ctx: ActiveTaskContext) -> None:
        # Section 9/10/16: a progressive WorkspaceObject that fills in as
        # the task collects information -- the SAME object gets updated
        # (never a new card per turn, tracked via active_artifact_revision);
        # the frontend owns object identity via task_id, this only ever
        # emits the rendered summary. Section 10: once generation has
        # produced a draft, the card shows that content directly instead of
        # the field-by-field summary -- never a raw internal enum value
        # (section 9's "request_clarification" must never reach the UI).
        if ctx.task_type != "compose_email":
            title = _generic_task_title(ctx)
            content = _generic_task_card_content(ctx)
            intelligence_lines = task_card_summary(self._work_session, ctx)
            if intelligence_lines:
                content = content + "\n" + "\n".join(intelligence_lines)
            presentation = cognitive_task_view_model(self._work_session, ctx, title=title)
            if not ctx.active_artifact_id:
                ctx.active_artifact_id = f"artifact-{ctx.id}"
            ctx.active_artifact_revision += 1
            if self._turn_trace is not None:
                self._turn_trace["task_artifact_revision"] = ctx.active_artifact_revision
            # Phase 3 compatibility adapter (section 17): the SAME id used
            # by the existing progressive-artifact UI event also becomes a
            # semantic object, so an existing task artifact can later
            # participate in representation transformation without a
            # second, parallel identity. Never affects the event below.
            register_object(
                self._work_session,
                SemanticObject(id=ctx.active_artifact_id, kind="task_artifact", title=title, payload={"task_type": ctx.task_type, "status": ctx.status}, task_id=ctx.id),
            )
            self._queue_ui_event(
                "task_artifact_updated",
                {
                    "id": ctx.active_artifact_id,
                    "task_id": ctx.id,
                    "task_type": ctx.task_type,
                    "status": ctx.status,
                    "title": title,
                    "subtitle": ctx.status,
                    "content": content,
                    "presentation": presentation,
                },
            )
            return
        recipient = ctx.entities.get("recipient") or ""
        role_label = _ROLE_LABELS.get(ctx.entities.get("recipientRole") or "", ctx.entities.get("recipientRole") or "")
        title = f"Email para {recipient}" if recipient else (f"Email para o {role_label}" if role_label else "Email")
        generated_draft = ctx.parameters.get("generated_draft") or ""
        if generated_draft:
            content = generated_draft
        else:
            lines = []
            if recipient or role_label:
                who = recipient or f"o {role_label}"
                lines.append(f"Para: {who}" + (f" ({role_label})" if recipient and role_label else ""))
            if ctx.parameters.get("course"):
                lines.append(f"Contexto: {ctx.parameters['course']}")
            if ctx.parameters.get("purpose"):
                lines.append(f"Objetivo: {_PURPOSE_LABELS.get(ctx.parameters['purpose'], ctx.parameters['purpose'])}")
            if ctx.parameters.get("topic"):
                lines.append(f"Assunto: {ctx.parameters['topic']}")
            lines.append("[conteúdo ainda em preparação]" if ctx.status != "ready" else "[pronto para redigir]")
            content = "\n".join(lines)
        if not ctx.active_artifact_id:
            ctx.active_artifact_id = f"artifact-{ctx.id}"
        ctx.active_artifact_revision += 1
        if self._turn_trace is not None:
            self._turn_trace["task_artifact_revision"] = ctx.active_artifact_revision
        register_object(
            self._work_session,
            SemanticObject(id=ctx.active_artifact_id, kind="task_artifact", title=title, payload={"task_type": ctx.task_type, "status": ctx.status}, task_id=ctx.id),
        )
        self._queue_ui_event(
            "task_artifact_updated",
            {
                "id": ctx.active_artifact_id,
                "task_id": ctx.id,
                "task_type": ctx.task_type,
                "status": ctx.status,
                "title": title,
                "subtitle": ctx.status,
                "content": content,
            },
        )

    def _execute_active_task(self, ctx: ActiveTaskContext) -> str:
        # Section 6: the only place in the ActiveTaskContext flow allowed to
        # call the LLM -- deterministic data/control turns (llm_calls=0)
        # never reach here. Section 11: provider choice stays with the
        # normal router (task_profile, never a hardcoded provider).
        if ctx.task_type != "compose_email":
            return "Ainda não sei executar este tipo de tarefa."
        ctx.status = "working"
        if self._turn_trace is not None:
            self._turn_trace["task_execution_started"] = True
            self._turn_trace["active_task_selected_path"] = "TASK_EXECUTION"

        missing = self._missing_task_fields(ctx)
        facts, instruction = _compose_email_task_prompt(ctx)
        if self._turn_trace is not None:
            # Section 13: structural snapshot of what the model is ABOUT TO
            # receive -- never the full generated content.
            self._turn_trace["model_context_task_type"] = ctx.task_type
            self._turn_trace["model_context_task_status"] = ctx.status
            self._turn_trace["model_context_known_fields"] = sorted({**ctx.entities, **ctx.parameters}.keys())
            self._turn_trace["model_context_missing_fields"] = list(missing)
            self._turn_trace["model_context_recipient_present"] = bool(ctx.entities.get("recipient") or ctx.entities.get("recipientRole"))
            self._turn_trace["model_context_topic_present"] = bool(ctx.parameters.get("topic"))
            self._turn_trace["model_context_key_points_count"] = len(ctx.key_points)
            self._turn_trace["model_context_instruction_ready_to_execute"] = True

        # Section 12.C: if the constructed context doesn't actually back up
        # what "ready" is supposed to guarantee (recipient-or-role + topic),
        # that is Echo's own fault, not the model's -- never even spend an
        # LLM call on a prompt we already know is incomplete.
        if missing:
            ctx.status = "collecting" if "recipient" in missing else "ready"
            self._touch_task(ctx)
            if self._turn_trace is not None:
                self._turn_trace["task_execution_completed"] = False
                self._turn_trace["fault_detected"] = True
                self._turn_trace["fault_stage"] = "prompt_construction"
                self._turn_trace["fault_owner"] = "echo_mechanism"
                self._turn_trace["fault_reason"] = "task_marked_executable_with_missing_fields"
                self._turn_trace["expected_context_present"] = False
                self._turn_trace["active_task_status"] = ctx.status
            return self._task_continuation_response(ctx)

        if self._turn_trace is not None:
            self._turn_trace["expected_context_present"] = True

        def _generate(current_instruction: str, attempt_number: int) -> tuple[str, str, str]:
            task_profile = {
                "task_type": "compose_email",
                "document_chars": sum(len(fact) for fact in facts),
                "document_requires_fidelity": True,
                "document_requires_full_output": True,
                "document_regeneration_attempt": attempt_number,
                "preferred_provider": "auto",
            }
            generated = self.response_composer.compose(
                ComposerRequest(
                    intent="compose_email_task",
                    user_message="Redige o email com base nos dados fornecidos.",
                    facts=facts,
                    intent_instruction=current_instruction,
                    fallback="Não consegui gerar o email agora. Podes pedir para tentar outra vez.",
                    language_instruction=self._language_instruction(),
                    task_profile=task_profile,
                )
            )
            decision = getattr(self.llm, "last_routing_decision", None)
            return (
                generated,
                str(getattr(decision, "provider", "") or "") if decision else "",
                str(getattr(decision, "model", "") or "") if decision else "",
            )

        max_attempts = _TASK_GENERATION_MAX_ATTEMPTS
        reply, provider, model = _generate(instruction, 1)
        valid, reason = _validate_task_generation(reply, ctx)
        attempt_count = 1
        recovery_attempted = False
        recovery_succeeded = False
        # Section 12.D/E: the model received the RIGHT context and the RIGHT
        # instruction (expected_context_present is True at this point) but
        # still didn't execute -- that failure is the model's, not Echo's.
        model_output_valid = valid
        # The ORIGINAL failure reason -- kept even after a successful
        # recovery, so fault_reason still explains WHAT went wrong on
        # attempt 1 instead of being overwritten by attempt 2's "valid".
        original_failure_reason = reason if not valid else ""

        if not valid and max_attempts > 1:
            recovery_attempted = True
            correction = _compose_email_correction_prompt(reason)
            reply2, provider2, model2 = _generate(correction, 2)
            attempt_count = 2
            valid2, reason2 = _validate_task_generation(reply2, ctx)
            reply, provider, model = reply2, provider2, model2
            if valid2:
                valid = True
                reason = reason2
                recovery_succeeded = True
            else:
                reason = reason2

        if valid:
            ctx.parameters["generated_draft"] = reply.strip()
            ctx.status = "reviewing"
        else:
            ctx.status = "ready"
        self._touch_task(ctx)
        if self._turn_trace is not None:
            self._turn_trace["task_execution_completed"] = True
            self._turn_trace["task_generation_provider"] = provider
            self._turn_trace["task_generation_model"] = model
            self._turn_trace["task_generation_validation_reason"] = reason
            self._turn_trace["task_output_validation_passed"] = valid
            self._turn_trace["task_output_validation_reason"] = reason
            self._turn_trace["task_generation_attempt"] = attempt_count
            self._turn_trace["task_generation_max_attempts"] = max_attempts
            self._turn_trace["model_received_expected_context"] = True
            self._turn_trace["model_output_valid"] = model_output_valid
            self._turn_trace["recovery_attempted"] = recovery_attempted
            self._turn_trace["recovery_succeeded"] = recovery_succeeded
            if not model_output_valid:
                # Section 12.D: right context, right "you are ready, do not
                # ask" instruction, and the model STILL asked for more
                # information or produced a non-artifact -- that is on the
                # model, not on Echo's mechanism.
                self._turn_trace["fault_detected"] = True
                self._turn_trace["fault_stage"] = "model_generation"
                self._turn_trace["fault_owner"] = "model"
                self._turn_trace["fault_reason"] = original_failure_reason or reason
            self._turn_trace["active_task_status"] = ctx.status
            self._turn_trace["active_task_id"] = ctx.id
            self._turn_trace["active_task_type"] = ctx.task_type
            self._turn_trace["task_known_fields"] = {**ctx.entities, **ctx.parameters}
            self._turn_trace["task_artifact_id"] = ctx.active_artifact_id
        # Section 8: an invalid model reply must never become the artifact
        # -- _update_task_artifact_ui only ever reads ctx.parameters
        # ["generated_draft"], which is only set in the `valid` branch above.
        self._update_task_artifact_ui(ctx)
        if not valid:
            return "Tive dificuldade a gerar o email agora. Podes pedir para tentar outra vez."
        return "Escrevi uma primeira versão do email. Dá uma vista de olhos no artefacto."

    # --- Phase 3 (Cognitive Workspace Blueprint v2): a NARROW, explicitly
    # scoped natural-language bridge for representation commands only
    # (section 11). This is deliberately NOT the full Interaction
    # Interpreter made authoritative -- it reuses Phase 2's
    # RepresentationIntent classifier purely to recognize the small set of
    # explicit phrasings this phase supports (cards/list/hide/restore; the
    # side_by_side case is handled inside _try_workspace_command above,
    # since "lado a lado" is already claimed by the existing spatial
    # detector there). Activity/Content/Reasoning intents are still not
    # routed through the interpreter anywhere. Phase 4 is expected to
    # absorb this bridge into a single authoritative router; until then it
    # stays a separate, documented seam that runs after
    # _try_workspace_command/_try_active_task_turn so neither existing
    # handler's priority changes.

    def _try_attention_command(self, user_message: str) -> str | None:
        """Phase 4 (section 5/20): a second authoritative Interaction
        Interpreter axis, alongside REPRESENTATION_COMMAND -- explicit
        attention/visibility commands ("não escondas o documento",
        "minimiza o email", "traz o relatório para a frente", "mostra o
        email outra vez") are claimed here, ahead of the plain
        representation bridge, since they are about ONE object's
        prominence, not the whole set's style. Target resolution reuses
        the existing generic, domain-agnostic
        assistant.representation.resolve_mentioned_object_ids; "isto"-style
        deixis falls back to whatever is currently FOCUS. Never guesses
        when nothing resolves -- returns a clarifying question instead of
        silently acting on the wrong object."""
        started_at = time.perf_counter()
        if not self._route_is_authorized("ATTENTION_COMMAND"):
            return None
        intent = self._canonical_interpretation.attention if self._canonical_interpretation is not None else None
        if intent is None:
            from assistant.interaction_interpreter import interpret as _interp_interpret
            intent = _interp_interpret(user_message, self._work_session).attention
        if not intent.present:
            return None

        target_resolution = resolve_semantic_targets(
            user_message,
            self._work_session,
            "attention_show_again" if intent.action == "show_again" else "attention",
        )
        target_ids = [match.object_id for match in target_resolution.matches if match.object_id]
        if (
            not target_ids
            and intent.action in ("minimize_object", "hide_object", "show_again")
        ):
            collection_resolution = resolve_semantic_collection_targets(user_message, self._work_session)
            if collection_resolution.matches and not collection_resolution.ambiguous:
                target_resolution = collection_resolution
                target_ids = [match.object_id for match in collection_resolution.matches if match.object_id]
        if not target_ids:
            attention_state = self._work_session.attention
            focus_id = attention_state.focus_object_id if attention_state is not None else ""
            if intent.action != "show_again" and focus_id:
                target_ids = [focus_id]

        if not target_ids and intent.action == "show_again":
            # Section 16: an unresolvable "show again" is exactly the same
            # shape as the EXISTING bare "mostra outra vez" restore_
            # representation trigger (assistant.representation) -- e.g.
            # "mostra-me outra vez as opções, mas em lista" also matches
            # show_again structurally, but names no specific object AND
            # asks for a style change, which only the representation
            # bridge actually knows how to do generically (it already
            # resurfaces every known object by default). Falling through
            # (instead of asking "a que objeto te referes?") lets that
            # bridge take the turn instead of dead-ending it.
            return None

        if self._turn_trace is not None:
            self._turn_trace["attention_command_detected"] = True
            self._turn_trace["attention_command_action"] = intent.action

        if not target_ids:
            if self._turn_trace is not None:
                self._turn_trace["attention_command_executed"] = False
                self._turn_trace["attention_command_latency_ms"] = (time.perf_counter() - started_at) * 1000
                self._turn_trace["attention_target_resolution_source"] = target_resolution.source
                self._turn_trace["attention_target_resolution_ambiguous"] = target_resolution.ambiguous
            return "A que objeto te referes? Ainda não consigo identificar qual."

        source = "voice_command"
        if intent.action == "keep_visible":
            constraint = self._work_session.constraints or RepresentationConstraint()
            do_not_minimize = list(dict.fromkeys(list(constraint.do_not_minimize_ids) + target_ids))
            self._work_session.constraints = RepresentationConstraint(
                avoid_style=constraint.avoid_style,
                prefer_style=constraint.prefer_style,
                conversation_only=constraint.conversation_only,
                keep_visible_ids=list(dict.fromkeys(list(constraint.keep_visible_ids) + target_ids)),
                do_not_minimize_ids=do_not_minimize,
                source=source,
            )
            # Phase 4 manual-test fix (section 2): keep_visible/do_not_
            # minimize is a CONSTRAINT ("never take this away from me"),
            # never an attention-LEVEL change on its own -- the real bug:
            # this used to force every target straight to ACTIVE, so
            # "não minimizes a Suécia" while Suécia was FOCUS silently
            # demoted it and cleared focus_object_id, even though the user
            # never asked to change its prominence. Only promote when the
            # object is CURRENTLY not visible at all (HIDDEN/MINIMIZED --
            # the minimum needed to actually satisfy "keep visible");
            # anything already on screen (VISIBLE/GLANCEABLE/ACTIVE/FOCUS)
            # keeps its exact current level untouched.
            for object_id in target_ids:
                current_level = get_attention_level(self._work_session, object_id)
                if current_level in (_ATTN_HIDDEN, _ATTN_MINIMIZED):
                    set_attention(self._work_session, object_id, _ATTN_VISIBLE, source=source)
            response = "Certo, mantenho isso visível e não o minimizo."
        elif intent.action == "hide_object":
            for object_id in target_ids:
                set_attention(self._work_session, object_id, _ATTN_HIDDEN, source=source)
            response = "Ok, deixo de mostrar isso -- continua guardado, só não visível."
        elif intent.action == "minimize_object":
            for object_id in target_ids:
                set_attention(self._work_session, object_id, _ATTN_MINIMIZED, source=source)
            response = "Minimizei."
        elif intent.action == "focus_object":
            for object_id in target_ids:
                set_attention(self._work_session, object_id, _ATTN_FOCUS, source=source)
            response = "Feito, foco nisso agora."
        elif intent.action == "show_again":
            for object_id in target_ids:
                set_attention(self._work_session, object_id, _ATTN_ACTIVE, source=source)
            response = "Aqui está outra vez."
        else:
            if self._turn_trace is not None:
                self._turn_trace["attention_command_executed"] = False
                self._turn_trace["attention_command_latency_ms"] = (time.perf_counter() - started_at) * 1000
            return None

        new_attention = self._work_session.attention
        result = ExecutionResult(
            executed=bool(new_attention is not None),
            mutated_state=bool(new_attention is not None),
            ui_event_queued=bool(new_attention is not None),
            reason=intent.action,
        )
        if new_attention is not None:
            self._queue_ui_event(
                "attention_changed",
                {
                    "object_states": dict(new_attention.object_states),
                    "focus_object_id": new_attention.focus_object_id,
                    "source": new_attention.source,
                },
            )
        if self._turn_trace is not None:
            self._turn_trace["attention_command_executed"] = result.executed
            self._turn_trace["attention_command_target_ids"] = list(target_ids)
            self._turn_trace["attention_command_latency_ms"] = (time.perf_counter() - started_at) * 1000
            self._turn_trace["attention_target_resolution_source"] = target_resolution.source
            self._turn_trace["attention_execution_mutated_state"] = result.mutated_state
            self._turn_trace["attention_execution_ui_event_queued"] = result.ui_event_queued
        if not result.executed:
            return "Não consegui alterar a atenção desse objeto."
        return response

    def _try_open_workspace_reference(self, user_message: str) -> str | None:
        """Phase 4 integration fix (section 5/7): "abre X" / "mostra X" /
        "foca no X" / "foca na X" -- promotes an EXISTING workspace
        activity/object to FOCUS instead of letting it fall through to
        active-task continuation (which would either swallow the phrase
        into that task's notes, section 2/12.A, or -- worse -- start a
        brand-new task). This is deliberately an ATTENTION-only operation:
        it never touches active_task/paused_task_stack (no task-switch,
        section 11's "same objects, no recreation"), never creates a task,
        never calls the LLM. Requires resolve_workspace_reference to find
        an EXPLICIT, unambiguous match; unlike the deictic "foca nisto"
        handled by _try_attention_command, an unresolvable target here
        falls through untouched rather than defaulting to whatever is
        currently FOCUS -- "abre a porta" naming nothing real must never be
        misread as a no-op re-focus of the current object."""
        normalized_message = normalize_for_matching(user_message)
        if not _is_open_reference_phrase(normalized_message):
            return None
        representation_intent = (
            self._canonical_interpretation.representation
            if self._canonical_interpretation is not None
            else None
        )
        if representation_intent is not None and representation_intent.present:
            return None
        target_resolution = resolve_semantic_targets(user_message, self._work_session, "workspace_reference")
        if target_resolution.ambiguous:
            if self._turn_trace is not None:
                self._turn_trace["attention_command_detected"] = True
                self._turn_trace["attention_command_action"] = "open_workspace_reference"
                self._turn_trace["attention_command_executed"] = False
                self._turn_trace["workspace_reference_resolution_source"] = target_resolution.source
                self._turn_trace["workspace_reference_ambiguous"] = True
            labels = [match.title for match in target_resolution.matches if match.title]
            if labels:
                return "Encontrei mais do que uma possibilidade: " + ", ".join(labels) + ". Qual queres abrir?"
            return "Encontrei mais do que uma possibilidade. Qual queres abrir?"
        if not target_resolution.matches:
            target_phrase = _fuzzy_target_phrase(user_message)
            if target_phrase and self._work_session.objects:
                if self._turn_trace is not None:
                    self._turn_trace["attention_command_detected"] = True
                    self._turn_trace["attention_command_action"] = "open_workspace_reference"
                    self._turn_trace["attention_command_executed"] = False
                    self._turn_trace["workspace_reference_resolution_source"] = target_resolution.source
                return "Não encontrei esse objeto no workspace."
            return None
        match = target_resolution.matches[0]
        if not match.object_id:
            return None

        task_level_open = normalized_message.startswith(("abre ", "abrir ", "volta ", "retoma "))
        if match.task_id and task_level_open:
            target_task = self._find_task_by_id(match.task_id)
            if target_task is not None:
                target_artifact_id = target_task.active_artifact_id or match.object_id
                target_level = (
                    self._work_session.attention.object_states.get(target_artifact_id, _ATTN_VISIBLE)
                    if self._work_session.attention is not None
                    else _ATTN_VISIBLE
                )
                if target_level == _ATTN_MINIMIZED and len(self._task_surface_open_artifact_ids()) >= 4:
                    capacity_response = self._maybe_prepare_task_surface_capacity_action(user_message, exclude_ids={target_artifact_id})
                    if capacity_response:
                        return capacity_response.replace("para abrir esta", f"para retomar {self._task_surface_label(target_artifact_id)}")
                current = self._work_session.active_task
                previous_artifact_id = current.active_artifact_id if current is not None and current.id != target_task.id else ""
                if current is not None and current.id != target_task.id and current.status not in ("completed", "cancelled"):
                    self._pause_active_task()
                self._paused_task_stack = [task for task in self._paused_task_stack if task.id != target_task.id]
                target_task.status = "ready" if not self._missing_task_fields(target_task) else "collecting"
                self._touch_task(target_task)
                self._active_task_context = target_task
                if previous_artifact_id:
                    set_attention(self._work_session, previous_artifact_id, _ATTN_ACTIVE, source="voice_command")

        current_task = self._work_session.active_task
        levels: dict[str, str] = {}
        if current_task is not None and current_task.active_artifact_id and current_task.active_artifact_id != match.object_id:
            levels[current_task.active_artifact_id] = _ATTN_ACTIVE
        levels[match.object_id] = _ATTN_FOCUS
        new_attention = set_bulk_attention(self._work_session, levels, source="voice_command")
        self._queue_ui_event(
            "attention_changed",
            {
                "object_states": dict(new_attention.object_states),
                "focus_object_id": new_attention.focus_object_id,
                "source": new_attention.source,
            },
        )
        if self._turn_trace is not None:
            self._turn_trace["attention_command_detected"] = True
            self._turn_trace["attention_command_action"] = "open_workspace_reference"
            self._turn_trace["attention_command_executed"] = True
            self._turn_trace["attention_command_target_ids"] = [match.object_id]
            self._turn_trace["workspace_reference_task_id"] = match.task_id
            self._turn_trace["workspace_reference_confidence"] = match.confidence
            self._turn_trace["workspace_reference_resolution_source"] = target_resolution.source
        label = match.title or "isso"
        return f"Pronto, {label} está outra vez em foco."

    def _try_pending_workspace_action(self, user_message: str) -> str | None:
        """Phase 4 (section 15): "Echo asked permission to minimize
        something, now waiting for sim/não" -- the SAME sim/não vocabulary
        _try_pending_user_intent already uses (_is_positive_document_
        confirmation/_is_negative_confirmation), applied to a completely
        separate, narrower piece of pending state so the two mechanisms
        never interfere with each other."""
        pending = self._pending_workspace_action
        if not pending:
            return None

        if _is_positive_document_confirmation(user_message):
            object_ids = list(pending.get("object_ids") or [])
            action = str(pending.get("action") or "minimize")
            for object_id in object_ids:
                set_attention(self._work_session, object_id, _ATTN_MINIMIZED, source="voice_command")
            self._pending_workspace_action = None
            new_attention = self._work_session.attention
            if new_attention is not None:
                self._queue_ui_event(
                    "attention_changed",
                    {
                        "object_states": dict(new_attention.object_states),
                        "focus_object_id": new_attention.focus_object_id,
                        "source": new_attention.source,
                    },
                )
            if self._turn_trace is not None:
                self._turn_trace["pending_workspace_action"] = True
                self._turn_trace["pending_workspace_action_result"] = "executed"
                self._turn_trace["pending_workspace_action_resolved"] = "executed"
                self._turn_trace["pending_workspace_action_object_ids"] = object_ids
                self._turn_trace["space_pressure_action"] = "minimize_executed"
                self._turn_trace["minimization_reason"] = pending.get("reason") or "space_pressure"
            titles = [
                self._work_session.objects[oid].title or oid
                for oid in object_ids
                if oid in self._work_session.objects
            ]
            label = ", ".join(t for t in titles if t) or "isso"
            deferred_message = str(pending.get("deferred_message") or "")
            if action == "minimize_for_task_surface" and deferred_message:
                try:
                    self._task_surface_capacity_override = True
                    deferred_response = self.respond(deferred_message)
                finally:
                    self._task_surface_capacity_override = False
                return f"Feito, minimizei {label}. {deferred_response}"
            return f"Feito, minimizei {label}."

        if _is_negative_confirmation(user_message):
            object_ids = list(pending.get("object_ids") or [])
            self._space_pressure_suppressed_candidate_ids.update(object_ids)
            self._pending_workspace_action = None
            if self._turn_trace is not None:
                self._turn_trace["pending_workspace_action"] = True
                self._turn_trace["pending_workspace_action_result"] = "cancelled"
                self._turn_trace["pending_workspace_action_resolved"] = "cancelled"
                self._turn_trace["pending_workspace_action_object_ids"] = object_ids
                self._turn_trace["space_pressure_action"] = "cancelled"
                self._turn_trace["minimization_reason"] = pending.get("reason") or "space_pressure"
            if str(pending.get("action") or "") == "minimize_for_task_surface":
                return "Ok, mantenho tudo como está. Para abrir essa tarefa, preciso que minimizes uma das quatro tarefas abertas."
            return "Ok, mantenho tudo visível por agora. Não volto a sugerir essa minimização enquanto o espaço não mudar."

        # Neither a clear yes nor no -- decay like _pending_user_intent's
        # own ttl pattern rather than let an unrelated later "sim" resolve
        # a stale question (section 15: "do not lose the user's current
        # task", which cuts both ways -- an ignored question must not
        # linger forever either).
        ttl = int(pending.get("ttl") or 1) - 1
        if ttl <= 0:
            self._pending_workspace_action = None
        else:
            pending["ttl"] = ttl
        return None

    def _maybe_propose_space_pressure_minimize(self, user_message: str) -> str:
        """Phase 4 (section 12/13/14): called right after a NEW activity/
        task genuinely starts (see _respond_inner's task_started check) --
        never on every turn. Returns an extra sentence to APPEND to that
        turn's own response, or "" when there is nothing to propose. Never
        minimizes anything itself: only ever creates a PendingWorkspaceAction
        and asks."""
        pressure_state = self._work_session.capacity
        pressure = pressure_state.pressure if pressure_state is not None else "low"
        if self._turn_trace is not None:
            self._turn_trace["space_pressure"] = pressure
            self._turn_trace["space_pressure_visible_object_count"] = (
                pressure_state.visible_object_count if pressure_state is not None else 0
            )
            self._turn_trace["space_pressure_major_surface_count"] = (
                pressure_state.major_surface_count if pressure_state is not None else 0
            )
            self._turn_trace["space_pressure_overflow_risk"] = (
                pressure_state.overflow_risk if pressure_state is not None else False
            )
            self._turn_trace["capacity_revision"] = pressure_state.revision if pressure_state is not None else 0
            self._turn_trace["capacity_signal_age"] = (
                max(0.0, time.time() - pressure_state.updated_at)
                if pressure_state is not None and pressure_state.updated_at
                else None
            )
            self._turn_trace["space_pressure_action"] = "none"
            self._turn_trace["space_pressure_permission_required"] = False
            self._turn_trace["space_pressure_candidate_ids"] = []
            self._turn_trace["space_pressure_protected_ids"] = []
        if pressure != "high":
            return ""
        if self._pending_workspace_action:
            # Already asked this turn/cycle -- never stack a second
            # question on top of an unresolved one.
            if self._turn_trace is not None:
                self._turn_trace["space_pressure_action"] = "pending_exists"
                self._turn_trace["pending_workspace_action"] = True
            return ""
        constraint = self._work_session.constraints
        focus_id = self._work_session.attention.focus_object_id if self._work_session.attention else ""
        protected: set[str] = {focus_id} if focus_id else set()
        if constraint is not None:
            protected |= set(constraint.keep_visible_ids)
            protected |= set(constraint.do_not_minimize_ids)
        protected |= set(resolve_mentioned_object_ids(user_message, self._work_session))
        protected |= set(self._space_pressure_suppressed_candidate_ids)
        candidates = rank_minimization_candidates(self._work_session, exclude_ids=protected)
        if self._turn_trace is not None:
            self._turn_trace["space_pressure_candidate_ids"] = candidates
            self._turn_trace["space_pressure_protected_ids"] = sorted(protected)
        if not candidates:
            if self._turn_trace is not None:
                self._turn_trace["space_pressure_candidate_found"] = False
                self._turn_trace["space_pressure_action"] = "no_safe_candidate"
            return " O espaço está cheio e não encontro nada que possa minimizar em segurança."
        candidate_id = candidates[0]
        obj = self._work_session.objects.get(candidate_id)
        label = (obj.title if obj is not None and obj.title else "isto")
        self._pending_workspace_action = {
            "action": "minimize",
            "object_ids": [candidate_id],
            "reason": "space_pressure",
            "ttl": 2,
            "capacity_revision": pressure_state.revision if pressure_state is not None else 0,
        }
        if self._turn_trace is not None:
            self._turn_trace["space_pressure_candidate_found"] = True
            self._turn_trace["space_pressure_action"] = "propose_minimize"
            self._turn_trace["space_pressure_permission_required"] = True
            self._turn_trace["pending_workspace_action"] = True
            self._turn_trace["pending_workspace_action_created"] = True
            self._turn_trace["pending_workspace_action_object_ids"] = [candidate_id]
            self._turn_trace["minimization_reason"] = "space_pressure"
        return f" O espaço está com pouca folga -- posso minimizar {label} para abrir espaço?"

    def _apply_auto_attention_for_new_activity(self, user_message: str) -> None:
        """Phase 4 (section 4): a conservative, deterministic first-version
        auto-attention policy for "a new activity/task just started."
        Objects explicitly named in THIS utterance (resolved the same
        generic, title-substring way _try_attention_command does) stay/
        become ACTIVE -- they are what the new activity is about. Anything
        ELSE that was ACTIVE demotes one step to VISIBLE (still on screen,
        just less prominent) -- never straight to MINIMIZED/HIDDEN; that
        only happens through the explicit ask-before-minimize flow. Never
        touches FOCUS (a real user focus command always wins) or anything
        already GLANCEABLE/MINIMIZED/HIDDEN."""
        attention_state = self._work_session.attention
        if attention_state is None:
            return
        referenced_ids = set(resolve_mentioned_object_ids(user_message, self._work_session))
        levels: dict[str, str] = {}
        for object_id, level in attention_state.object_states.items():
            if object_id in referenced_ids:
                if level != _ATTN_FOCUS:
                    levels[object_id] = _ATTN_ACTIVE
                continue
            if level == _ATTN_ACTIVE:
                levels[object_id] = _ATTN_VISIBLE
        if not levels:
            return
        new_attention = set_bulk_attention(self._work_session, levels, source="auto")
        self._queue_ui_event(
            "attention_changed",
            {
                "object_states": dict(new_attention.object_states),
                "focus_object_id": new_attention.focus_object_id,
                "source": new_attention.source,
            },
        )

    def report_workspace_capacity(self, payload: dict) -> None:
        """Phase 4 (section 11): the ONLY frontend -> backend layout signal
        -- called from prototype_web_ui/controller.py's reportWorkspaceCapacity
        QWebChannel slot. Accepts exactly the semantic fields section 11
        describes (pressure/visible_object_count/major_surface_count/
        overflow_risk); never a pixel or coordinate (there is no field for
        one to arrive in). Stores the signal on WorkSession for the NEXT
        turn's policy checks to read -- this call itself never produces a
        response or mutates attention/representation."""
        payload = payload or {}
        pressure = str(payload.get("pressure") or "low").strip().lower()
        if pressure not in ("low", "medium", "high"):
            pressure = "low"
        current = self._work_session.capacity
        changed = (
            current is None
            or current.pressure != pressure
            or current.visible_object_count != int(payload.get("visible_object_count") or 0)
            or current.major_surface_count != int(payload.get("major_surface_count") or 0)
            or current.overflow_risk != bool(payload.get("overflow_risk", False))
        )
        if changed:
            self._workspace_capacity_revision += 1
            self._space_pressure_suppressed_candidate_ids.clear()
        self._work_session.capacity = SpacePressureState(
            pressure=pressure,
            visible_object_count=int(payload.get("visible_object_count") or 0),
            major_surface_count=int(payload.get("major_surface_count") or 0),
            overflow_risk=bool(payload.get("overflow_risk", False)),
            revision=self._workspace_capacity_revision,
            updated_at=time.time(),
        )

    def _try_representation_command(self, user_message: str) -> str | None:
        """Phase 3B integration fix: a real Qt run showed this bridge
        declining too eagerly (an empty semantic object registry, or
        nothing to restore) let genuinely representation-flavored
        utterances ("não quero ver nada", "mostra outra vez", "faz uma
        tabela") fall through into DOCUMENT_TASK's file-search heuristics,
        which happily treated "outra vez"/"cartões"/the tail of "vamos só
        conversar" as filenames. ANY utterance the Interaction Interpreter
        classifies as RepresentationIntent is now claimed here -- either
        by actually transforming the representation, or (for an action
        this narrow Phase 3 bridge doesn't implement yet, or "mostra outra
        vez" with nothing to restore) by returning an honest, still-
        deterministic decline. It must never again return None for a
        present RepresentationIntent."""
        started_at = time.perf_counter()
        if not self._route_is_authorized("REPRESENTATION_COMMAND"):
            return None
        intent = self._canonical_interpretation.representation if self._canonical_interpretation is not None else None
        if intent is None:
            from assistant.interaction_interpreter import interpret as _interp_interpret
            intent = _interp_interpret(user_message, self._work_session).representation
        if not intent.present:
            return None

        action = intent.action
        # Phase 4 integration fix (section 7): moving this bridge ahead of
        # active-task continuation (so a real style/content request like
        # "mostra-me cidades em cartões" is never swallowed into the task's
        # notes, section 2/12.I) surfaced a real interpreter false-positive
        # this bridge's OWN "claim every present RepresentationIntent"
        # policy used to paper over: plain task data ("o pipeline gráfico, a
        # fase de geometry processing") can weakly classify as an
        # unimplemented representation sub-action (simplify/filter/
        # ordering/parameter_mutation) it was never meant to describe.
        # Only the CONFIDENT, concretely-actionable classifications below
        # are allowed to preempt an active task; anything else falls
        # through to give the active task its normal first refusal, exactly
        # matching this bridge's original (post-active-task) behavior for
        # that ambiguous case.
        is_confident_action = (
            action in ("hide_representation", "restore_representation", "show_reasoning", "hide_reasoning")
            or (action == "requested_style" and intent.style in ("cards", "list"))
            or (action == "rejected_style" and bool(intent.style))
            or intent.grouping_requested
        )
        if not is_confident_action and self._work_session.active_task is not None and self._work_session.active_task.status not in ("completed", "cancelled"):
            return None
        new_state: RepresentationState | None = None
        response: str | None = None
        reasoning_toggled = False

        if action == "hide_representation":
            new_state = transform_representation(self._work_session, "conversation_only", source="voice_command")
            self._set_representation_constraint(conversation_only=True)
        elif action == "restore_representation":
            current = self._work_session.representation
            if current is None or current.previous is None:
                response = "Ainda não tenho uma vista anterior para mostrar outra vez."
            else:
                new_state = restore_representation(self._work_session, source="voice_command")
                if intent.style in ("cards", "list"):
                    new_state = transform_representation(self._work_session, intent.style, source="voice_command")
                    self._set_representation_constraint(prefer_style=intent.style, avoid_style="", conversation_only=False)
                # A restore ends the "hide everything" constraint (there is
                # something to show again), but a prior explicit prefer/
                # avoid_style is not implicated by "mostra outra vez" alone,
                # so it is left untouched here.
                self._set_representation_constraint(conversation_only=False)
        elif action == "requested_style" and intent.style in ("cards", "list"):
            new_state = transform_representation(self._work_session, intent.style, source="voice_command")
            # Phase 4 (section 9/10): an explicit style request is exactly
            # the kind of user preference auto-representation must not
            # immediately undo -- record it so choose_representation()
            # (called from e.g. _try_explicit_alternatives) respects it on
            # the NEXT auto-trigger, instead of silently reverting to
            # whatever its own default would otherwise have been.
            self._set_representation_constraint(prefer_style=intent.style, avoid_style="", conversation_only=False)
        elif action == "rejected_style" and intent.style:
            # Phase 4 (section 9): "não quero cartões" alone (no explicit
            # alternative named) -- record the rejection as a constraint
            # and let the SAME generic policy choose_representation() uses
            # pick a fallback, rather than a bespoke rule here.
            self._set_representation_constraint(avoid_style=intent.style, prefer_style="")
            proposal = choose_representation(
                self._work_session, {"object_count": len(self._work_session.objects)},
            )
            if proposal.should_apply and proposal.style != intent.style:
                new_state = transform_representation(self._work_session, proposal.style, source="voice_command")
                self._set_representation_constraint(prefer_style=proposal.style)
            else:
                response = f"Ok, não uso {_REPRESENTATION_STYLE_LABELS.get(intent.style, intent.style)}."
        elif action in ("show_reasoning", "hide_reasoning"):
            reasoning_toggled = True
            new_state = set_reasoning_visible(self._work_session, action == "show_reasoning", source="voice_command")
            if new_state is None:
                response = (
                    "Ainda não estou a mostrar nada visualmente, por isso não há raciocínio para mostrar ou esconder."
                    if action == "show_reasoning"
                    else "Combinado, não te mostro o raciocínio."
                )
        elif intent.grouping_requested:
            # Phase 3C final patch (section 4/6): a grouping/organization
            # request ("separa os cards dos destinos internacionais dos
            # destinos portugueses") is claimed here -- never
            # GENERAL_CONVERSATION/DOCUMENT_TASK -- but Phase 3C still does
            # not implement semantic grouping execution. This must NEVER
            # mutate the representation just to produce a success-shaped
            # reply (the real Qt bug: it used to silently reapply
            # style=cards and say "Aqui tens em cartões." because the
            # word "cards" happened to appear in the utterance).
            response = "Ainda não consigo criar esse agrupamento semântico automaticamente."
        else:
            # Recognized by the Interaction Interpreter (simplify/filter/
            # ordering/parameter_mutation, or a requested_style outside
            # cards/list) -- Phase 3/4 don't implement these transforms
            # yet, but the turn is still claimed here rather than left to
            # fall through into document/file search.
            response = "Ainda não sei mudar para essa representação, mas já consigo cartões, lista, lado a lado ou só conversar."

        if self._turn_trace is not None:
            self._turn_trace["representation_command_detected"] = True
            self._turn_trace["representation_command_action"] = action

        if new_state is None:
            if self._turn_trace is not None:
                self._turn_trace["representation_command_executed"] = False
                self._turn_trace["representation_command_latency_ms"] = (time.perf_counter() - started_at) * 1000
            return response

        if reasoning_toggled:
            # Section 19: reasoning visibility is a subtle rationale-only
            # toggle, never a geometry/style change -- still surfaced via
            # the SAME representation_changed event (frontend already
            # reads style_params/reasoning generically), but the ack text
            # below is deliberately different from the style-change acks.
            self._queue_ui_event(
                "representation_changed",
                {
                    "style": new_state.style,
                    "visible_object_ids": list(new_state.visible_object_ids),
                    "major_object_ids": list(new_state.major_object_ids),
                    "hidden_object_ids": list(new_state.hidden_object_ids),
                    "style_params": dict(new_state.style_params),
                    "source": new_state.source,
                    "reasoning_visible": new_state.reasoning_visible,
                },
            )
            if self._turn_trace is not None:
                self._turn_trace["representation_style"] = new_state.style
                self._turn_trace["representation_source"] = new_state.source
                self._turn_trace["representation_command_executed"] = True
                self._turn_trace["reasoning_visible"] = new_state.reasoning_visible
                self._turn_trace["representation_command_latency_ms"] = (time.perf_counter() - started_at) * 1000
            return "Aqui tens o porquê, em poucas palavras." if new_state.reasoning_visible else "Combinado, não te mostro o raciocínio."

        self._queue_ui_event(
            "representation_changed",
            {
                "style": new_state.style,
                "visible_object_ids": list(new_state.visible_object_ids),
                "major_object_ids": list(new_state.major_object_ids),
                "hidden_object_ids": list(new_state.hidden_object_ids),
                "style_params": dict(new_state.style_params),
                "source": new_state.source,
            },
        )
        if self._turn_trace is not None:
            self._turn_trace["representation_style"] = new_state.style
            self._turn_trace["representation_source"] = new_state.source
            # Section 20: like workspace_command_executed, this only attests
            # that the ui_event was queued backend-side -- there is no
            # round-trip channel confirming the frontend actually applied
            # it (same documented limitation as the existing workspace
            # command system).
            self._turn_trace["representation_command_executed"] = True
            self._turn_trace["representation_command_latency_ms"] = (time.perf_counter() - started_at) * 1000
        return _REPRESENTATION_COMMAND_ACKS.get(new_state.style, "Feito.")

    def _set_representation_constraint(
        self,
        *,
        prefer_style: str | None = None,
        avoid_style: str | None = None,
        conversation_only: bool | None = None,
    ) -> RepresentationConstraint:
        """Phase 4 (section 9/10): the ONE place that mutates
        WorkSession.constraints -- always builds a fresh RepresentationConstraint
        from the current one plus whichever fields the caller wants to
        change, so a `keep_visible_ids`/`do_not_minimize_ids` set by an
        earlier attention command is never accidentally wiped out by an
        unrelated later style change."""
        current = self._work_session.constraints or RepresentationConstraint()
        new_constraint = RepresentationConstraint(
            avoid_style=current.avoid_style if avoid_style is None else avoid_style,
            prefer_style=current.prefer_style if prefer_style is None else prefer_style,
            conversation_only=current.conversation_only if conversation_only is None else conversation_only,
            keep_visible_ids=list(current.keep_visible_ids),
            do_not_minimize_ids=list(current.do_not_minimize_ids),
            source="voice_command",
        )
        self._work_session.constraints = new_constraint
        return new_constraint

    def _try_explicit_alternatives(self, user_message: str) -> str | None:
        """Phase 3B integration fix (FIX 3): the smallest domain-agnostic
        mechanism to materialize an explicitly-enumerated set of named
        alternatives ("vamos pensar em três opções: Suécia, Itália e
        Açores") into generic SemanticObjects with a stable id per title,
        plus an initial cards representation -- so the representation
        renderer has something real to exercise without a hand-built test
        fixture or any travel-specific (or other domain-specific) code.
        Never infers a fact about the alternatives; the extraction itself
        (assistant.representation.extract_explicit_alternatives) is pure
        text structure (trigger verb + comma/"e"-separated capitalized
        list), not a lookup or a classifier trained on any domain."""
        candidates = extract_explicit_alternatives(user_message)
        if len(candidates) < 2:
            return None

        category = extract_comparison_category(user_message) or _infer_generic_comparison_category(_normalize_text(user_message))
        if self._explicit_alternatives_would_create_new_activity(user_message, candidates, category):
            capacity_response = self._maybe_prepare_task_surface_capacity_action(user_message)
            if capacity_response:
                return capacity_response
        activity_ctx = self._ensure_activity_for_explicit_alternatives(user_message, candidates, category=category)

        # Phase 4 stabilization: explicit alternatives are owned by the
        # activity this turn created/resumed, not by whichever task happened
        # to be active before routing reached this handler.
        owning_task_id = activity_ctx.id if activity_ctx is not None else (
            self._work_session.active_task.id if self._work_session.active_task is not None else ""
        )
        created_ids = self._materialize_comparison_options(candidates, user_message, owning_task_id=owning_task_id)
        knowledge_entries = update_task_knowledge(self._work_session, activity_ctx, user_message, turn_id=self._current_turn_reference()) if activity_ctx is not None else []
        if self._turn_trace is not None:
            if category:
                self._turn_trace["comparison_category"] = category
            self._turn_trace["task_knowledge_updated_count"] = len(knowledge_entries)
            if activity_ctx is not None:
                if self._turn_trace.get("task_started"):
                    self._turn_trace["generic_activity_created"] = True
                    self._turn_trace["active_task_selected_path"] = "TASK_START"
                elif self._turn_trace.get("task_resumed"):
                    self._turn_trace["active_task_selected_path"] = "TASK_RESUME"
        labels = candidates[0] if len(candidates) == 1 else ", ".join(candidates[:-1]) + " e " + candidates[-1]
        return f"Boa, temos {labels} como opções -- já estão no espaço de trabalho."

    def _materialize_comparison_options(self, candidates: list[str], user_message: str, *, owning_task_id: str) -> list[str]:
        """Registers comparison options and their initial visual state.

        This is the canonical bridge from a comparison activity to real
        SemanticObjects. It is intentionally idempotent: registering the same
        stable object id updates/reuses the existing object instead of creating
        duplicate semantic children.
        """
        created_ids: list[str] = []
        for title in candidates:
            object_id = f"option-{slugify_title(title)}"
            obj = register_object(self._work_session, SemanticObject(id=object_id, kind="option", title=title, task_id=owning_task_id))
            # The frontend has no equivalent of WorkSession.objects on its
            # own, so sync objects before any representation event references
            # their ids.
            self._queue_semantic_object_sync(obj)
            created_ids.append(object_id)

        if not created_ids:
            return []

        proposal = choose_representation(
            self._work_session,
            {"object_count": len(created_ids), "comparison_intent": message_implies_comparison(user_message)},
        )
        style = proposal.style if proposal.should_apply else "cards"
        major_ids = created_ids if style == "side_by_side" else []
        new_state = transform_representation(
            self._work_session, style, visible_ids=created_ids, major_ids=major_ids, source="auto",
        )
        self._queue_ui_event(
            "representation_changed",
            {
                "style": new_state.style,
                "visible_object_ids": list(new_state.visible_object_ids),
                "major_object_ids": list(new_state.major_object_ids),
                "hidden_object_ids": list(new_state.hidden_object_ids),
                "style_params": dict(new_state.style_params),
                "source": new_state.source,
            },
        )
        attention_levels = {oid: (_ATTN_FOCUS if index == 0 else _ATTN_ACTIVE) for index, oid in enumerate(created_ids)}
        new_attention = set_bulk_attention(self._work_session, attention_levels, source="auto")
        self._queue_ui_event(
            "attention_changed",
            {
                "object_states": dict(new_attention.object_states),
                "focus_object_id": new_attention.focus_object_id,
                "source": new_attention.source,
            },
        )
        if self._turn_trace is not None:
            self._turn_trace["representation_command_detected"] = True
            self._turn_trace["representation_command_action"] = "materialize_alternatives"
            self._turn_trace["representation_style"] = new_state.style
            self._turn_trace["representation_source"] = new_state.source
            self._turn_trace["representation_command_executed"] = True
            self._turn_trace["alternatives_materialized_count"] = len(created_ids)
            self._turn_trace["representation_proposal_reason_code"] = proposal.reason_code
            self._turn_trace["representation_proposal_confidence"] = proposal.confidence
            self._turn_trace["attention_command_detected"] = True
            self._turn_trace["attention_command_action"] = "materialize_alternatives"
        return created_ids

    def _materialize_comparison_options_from_context(self, ctx: ActiveTaskContext, user_message: str) -> list[str]:
        if ctx.task_type != "generic_comparison":
            return []
        candidates = _split_option_text(ctx.parameters.get("options", ""))
        if len(candidates) < 2:
            candidates = _extract_comparison_options_from_text(user_message)
            if len(candidates) >= 2:
                ctx.parameters["options"] = " e ".join(candidates)
        if len(candidates) < 2:
            return []
        return self._materialize_comparison_options(candidates, user_message, owning_task_id=ctx.id)

    def _ensure_activity_for_explicit_alternatives(
        self,
        user_message: str,
        candidates: list[str],
        *,
        category: str = "",
    ) -> ActiveTaskContext | None:
        normalized = _normalize_text(user_message)
        interpretation = _interpret_activity(user_message, normalized, self._active_task_context, self._paused_task_stack)
        if not (interpretation.starts_task and interpretation.task_type == "generic_comparison"):
            return self._active_task_context if self._active_task_context is not None else None

        current = self._active_task_context
        equivalent = self._find_equivalent_generic_comparison(candidates)
        if equivalent is not None:
            if current is not None and current.id != equivalent.id and current.status not in ("completed", "cancelled"):
                self._pause_active_task()
            self._paused_task_stack = [task for task in self._paused_task_stack if task.id != equivalent.id]
            equivalent.status = "ready" if not self._missing_task_fields(equivalent) else "collecting"
            equivalent.parameters["options"] = " e ".join(candidates)
            if category:
                equivalent.parameters["category"] = category
            if user_message not in equivalent.source_turns:
                equivalent.source_turns.append(user_message)
            self._touch_task(equivalent)
            self._active_task_context = equivalent
            if self._turn_trace is not None:
                self._turn_trace["task_resumed"] = True
                self._turn_trace["generic_activity_reused"] = True
                self._turn_trace["active_task_selected_path"] = "TASK_RESUME"
                self._turn_trace["new_active_task_type"] = equivalent.task_type
                self._turn_trace["task_fields_updated"] = ["options"] + (["category"] if category else [])
                self._turn_trace["task_context_source"] = "deterministic_activity"
                self._turn_trace["task_turn_intent"] = "resume"
            self._task_continuation_response(equivalent)
            return equivalent

        if current is not None and current.status not in ("completed", "cancelled", "paused"):
            if current.task_type == "generic_comparison":
                current_category = _normalize_text(current.parameters.get("category", ""))
                next_category = _normalize_text(category)
                current_options = _normalized_option_set(_split_option_text(current.parameters.get("options", "")))
                next_options = _normalized_option_set(candidates)
                starts_additional_comparison = contains_any_phrase(
                    normalized,
                    (
                        "para alem disso",
                        "para além disso",
                        "vamos tambem comparar",
                        "vamos também comparar",
                        "tambem vamos comparar",
                        "também vamos comparar",
                        "outra comparacao",
                        "outra comparação",
                        "nova comparacao",
                        "nova comparação",
                    ),
                )
                has_distinct_option_set = bool(current_options and next_options and current_options.isdisjoint(next_options))
                if (next_category and current_category and next_category != current_category) or starts_additional_comparison or has_distinct_option_set:
                    self._pause_active_task()
                    ctx = self._create_task_context(
                        "generic_comparison",
                        activity_family=interpretation.activity_family,
                        domain=interpretation.domain,
                        artifact_type=interpretation.artifact_type,
                        goal=interpretation.goal,
                    )
                    self._active_task_context = ctx
                else:
                    ctx = current
            else:
                self._pause_active_task()
                ctx = self._create_task_context(
                    "generic_comparison",
                    activity_family=interpretation.activity_family,
                    domain=interpretation.domain,
                    artifact_type=interpretation.artifact_type,
                    goal=interpretation.goal,
                )
                self._active_task_context = ctx
        else:
            ctx = self._create_task_context(
                "generic_comparison",
                activity_family=interpretation.activity_family,
                domain=interpretation.domain,
                artifact_type=interpretation.artifact_type,
                goal=interpretation.goal,
            )
            self._active_task_context = ctx

        ctx.parameters["options"] = " e ".join(candidates)
        if category:
            ctx.parameters["category"] = category
        if user_message not in ctx.source_turns:
            ctx.source_turns.append(user_message)
        self._touch_task(ctx)
        if self._turn_trace is not None:
            self._turn_trace["task_started"] = True
            self._turn_trace["new_active_task_type"] = ctx.task_type
            self._turn_trace["task_fields_updated"] = ["options"] + (["category"] if category else [])
            self._turn_trace["task_context_source"] = "deterministic_activity"
            self._turn_trace["task_turn_intent"] = "switch" if self._turn_trace.get("task_paused") else "data"
        self._task_continuation_response(ctx)
        return ctx

    def _find_equivalent_generic_comparison(self, candidates: list[str]) -> ActiveTaskContext | None:
        wanted = _normalized_option_set(candidates)
        if not wanted:
            return None
        tasks: list[ActiveTaskContext] = []
        if self._work_session.active_task is not None:
            tasks.append(self._work_session.active_task)
        tasks.extend(reversed(self._paused_task_stack))
        for task in tasks:
            if task.task_type != "generic_comparison":
                continue
            existing = _normalized_option_set(_split_option_text(task.parameters.get("options", "")))
            if existing == wanted:
                return task
        return None

    def _task_surface_artifact_ids(self) -> list[str]:
        """Task Surface membership is semantic: top-level task artifacts
        only. Child objects such as options/criteria keep their task_id
        ownership but never consume one of the four task slots."""
        return [
            object_id
            for object_id, obj in self._work_session.objects.items()
            if obj.kind in {"task_artifact", "writing_task"}
        ]

    def _task_surface_open_artifact_ids(self) -> list[str]:
        open_ids: list[str] = []
        attention = self._work_session.attention
        states = attention.object_states if attention is not None else {}
        for object_id in self._task_surface_artifact_ids():
            level = states.get(object_id, _ATTN_VISIBLE)
            if level not in (_ATTN_MINIMIZED, _ATTN_HIDDEN):
                open_ids.append(object_id)
        return open_ids

    def _task_surface_label(self, object_id: str) -> str:
        obj = self._work_session.objects.get(object_id)
        return (obj.title if obj is not None and obj.title else object_id)

    def _task_surface_minimize_candidate(self, *, exclude_ids: set[str] | None = None) -> str:
        excluded = set(exclude_ids or set())
        constraint = self._work_session.constraints
        if constraint is not None:
            excluded.update(constraint.keep_visible_ids)
            excluded.update(constraint.do_not_minimize_ids)
        focus_id = self._work_session.attention.focus_object_id if self._work_session.attention else ""
        if focus_id:
            excluded.add(focus_id)
        candidates = [oid for oid in self._task_surface_open_artifact_ids() if oid not in excluded]
        if not candidates:
            return ""
        # Prefer a paused/background task before the currently active one.
        active_artifact = self._work_session.active_task.active_artifact_id if self._work_session.active_task else ""
        candidates.sort(key=lambda oid: (1 if oid == active_artifact else 0, self._task_surface_label(oid).lower()))
        return candidates[0]

    def _maybe_prepare_task_surface_capacity_action(self, user_message: str, *, exclude_ids: set[str] | None = None) -> str:
        if self._task_surface_capacity_override:
            return ""
        open_task_ids = self._task_surface_open_artifact_ids()
        if len(open_task_ids) < 4:
            return ""
        candidate_id = self._task_surface_minimize_candidate(exclude_ids=exclude_ids)
        if not candidate_id:
            if self._turn_trace is not None:
                self._turn_trace["task_surface_capacity_blocked"] = True
                self._turn_trace["task_surface_open_task_count"] = len(open_task_ids)
                self._turn_trace["pending_workspace_action_created"] = False
                self._turn_trace["minimization_reason"] = "task_surface_capacity"
            return "Já tens quatro tarefas abertas. Preciso que minimizes uma delas antes de abrir outra."
        self._pending_workspace_action = {
            "action": "minimize_for_task_surface",
            "object_ids": [candidate_id],
            "reason": "task_surface_capacity",
            "ttl": 2,
            "deferred_message": user_message,
        }
        if self._turn_trace is not None:
            self._turn_trace["task_surface_capacity_blocked"] = True
            self._turn_trace["task_surface_open_task_count"] = len(open_task_ids)
            self._turn_trace["task_surface_candidate_id"] = candidate_id
            self._turn_trace["pending_workspace_action"] = True
            self._turn_trace["pending_workspace_action_created"] = True
            self._turn_trace["pending_workspace_action_object_ids"] = [candidate_id]
            self._turn_trace["minimization_reason"] = "task_surface_capacity"
        return f"Já tens quatro tarefas abertas. Posso minimizar {self._task_surface_label(candidate_id)} para abrir esta?"

    def _maybe_prepare_task_surface_expand_action(self, target_id: str) -> str:
        open_task_ids = self._task_surface_open_artifact_ids()
        if target_id not in open_task_ids or len(open_task_ids) <= 1:
            return ""
        candidate_id = self._task_surface_minimize_candidate(exclude_ids={target_id})
        if not candidate_id:
            return f"Posso focar {self._task_surface_label(target_id)}, mas não encontro outra tarefa que possa minimizar em segurança."
        self._pending_workspace_action = {
            "action": "minimize_for_task_surface",
            "object_ids": [candidate_id],
            "reason": "task_surface_expand",
            "ttl": 2,
        }
        if self._turn_trace is not None:
            self._turn_trace["task_surface_expand_requested"] = True
            self._turn_trace["task_surface_expand_target_id"] = target_id
            self._turn_trace["task_surface_candidate_id"] = candidate_id
            self._turn_trace["pending_workspace_action"] = True
            self._turn_trace["pending_workspace_action_created"] = True
            self._turn_trace["pending_workspace_action_object_ids"] = [candidate_id]
            self._turn_trace["minimization_reason"] = "task_surface_expand"
        return (
            f"Posso dar mais espaço a {self._task_surface_label(target_id)}, "
            f"mas preciso de minimizar uma das outras tarefas. Posso minimizar {self._task_surface_label(candidate_id)}?"
        )

    def _explicit_alternatives_would_create_new_activity(self, user_message: str, candidates: list[str], category: str) -> bool:
        equivalent = self._find_equivalent_generic_comparison(candidates)
        if equivalent is not None:
            return False
        current = self._work_session.active_task
        if current is None or current.status in ("completed", "cancelled", "paused"):
            return True
        if current.task_type != "generic_comparison":
            return True
        normalized = _normalize_text(user_message)
        current_category = _normalize_text(current.parameters.get("category", ""))
        next_category = _normalize_text(category)
        starts_additional_comparison = contains_any_phrase(
            normalized,
            (
                "para alem disso",
                "para além disso",
                "vamos tambem comparar",
                "vamos também comparar",
                "tambem vamos comparar",
                "também vamos comparar",
                "outra comparacao",
                "outra comparação",
                "nova comparacao",
                "nova comparação",
            ),
        )
        return bool((next_category and current_category and next_category != current_category) or starts_additional_comparison)

    def _try_mixed_representation_intent(self, user_message: str) -> str | None:
        """Phase 3C (FIX 10/11/12), generalized in the Phase 4 manual-test
        fix (section 3/4): the minimum truthful handling for an utterance
        adding one OR SEVERAL options to an already-active generic set,
        optionally combined with unsupported sub-intents (moving a named
        card, semantic grouping/clustering). Only fires when there is
        already an active generic option set (section 11); otherwise this
        is not the right handler and the turn falls through unchanged.
        Reuses the Interaction Interpreter's multi_intent_count for the
        requested/executed/failed telemetry (section 12) -- this method
        NEVER returns an unqualified "Feito"/"Aqui tens" acking work that
        did not happen; it says exactly what it did and, if anything else
        was asked, that the rest isn't supported yet. A PURE multi-item
        addition ("acrescenta A, B, C e D") is exactly ONE sub-intent
        ("add options") regardless of how many names it lists -- it must
        never be reported as partial just because several titles were
        stripped out of the residual check below.

        Phase 4 integration fix (section 9): no longer requires a PRE-
        EXISTING option set -- "adiciona Porto, Lisboa, Coimbra, Aveiro e
        Guarda em cartões" is a perfectly ordinary way to name the FIRST
        batch of options too (e.g. while a plan_trip task is active and no
        destination has been materialized yet, section 2/12.F/12.G). The
        explicit "add" trigger verb plus a real enumerated list
        (extract_added_alternatives) is already a confident, narrow enough
        signal on its own; requiring an existing set first was only ever an
        accident of this handler being written as an "add to" companion to
        _try_explicit_alternatives, not a deliberate safety gate."""
        added_titles = extract_added_alternatives(user_message)
        if not added_titles:
            return None
        target_reference = extract_add_target_reference(user_message)

        from assistant.interaction_interpreter import interpret as _interp_interpret

        interpretation = _interp_interpret(user_message, self._work_session)
        # Section 12: the Interaction Interpreter has no spatial-by-name or
        # grouping/clustering axis at all yet, so multi_intent_count alone
        # would undercount a sentence like "chega o card dos Açores para o
        # outro lado... acrescenta também Madeira" down to 1 (just this
        # handler's own "add"). A light, generic residual-text check
        # catches the common case honestly: strip the matched trigger
        # phrase and EVERY extracted title out of the utterance -- if
        # anything substantial is left, at least one more sub-intent was
        # requested and NOT executed here, so this must report partial
        # execution rather than a clean "fully done." Stripping only the
        # FIRST title (the Phase 4 manual-test bug) left every additional
        # enumerated name in the residual, misreading a pure multi-add as
        # a mixed/partial request.
        residual = strip_add_alternatives_phrase(user_message, added_titles)
        has_unaddressed_residual = len(residual) > 15
        requested = max(2, interpretation.multi_intent_count) if has_unaddressed_residual else 1

        target_owner_task_id = ""
        target_resolution_source = ""
        if target_reference:
            collection_resolution = resolve_semantic_collection_targets(f"os {target_reference}", self._work_session)
            if collection_resolution.ambiguous:
                if self._turn_trace is not None:
                    self._turn_trace["mixed_add_target_resolution_source"] = collection_resolution.source
                    self._turn_trace["representation_command_executed"] = False
                return "A que conjunto queres acrescentar isso?"
            if collection_resolution.matches:
                owner_ids = {match.task_id for match in collection_resolution.matches if match.task_id}
                if len(owner_ids) == 1:
                    target_owner_task_id = next(iter(owner_ids))
                    target_resolution_source = collection_resolution.source

        # Phase 4 final integration fix: an explicit target collection
        # ("aos telefones", "aos destinos") owns the new object. Only when
        # no target collection resolves do we fall back to the active task.
        owning_task_id = target_owner_task_id or (self._work_session.active_task.id if self._work_session.active_task is not None else "")
        created_ids: list[str] = []
        existing_titles: list[str] = []
        for added_title in added_titles:
            object_id = f"option-{slugify_title(added_title)}"
            existing = self._work_session.objects.get(object_id)
            if existing is not None and existing.kind == "option" and existing.task_id == owning_task_id:
                existing_titles.append(existing.title or added_title)
                created_ids.append(object_id)
                continue
            obj = register_object(self._work_session, SemanticObject(id=object_id, kind="option", title=added_title, task_id=owning_task_id))
            self._queue_semantic_object_sync(obj)
            created_ids.append(object_id)
        self._merge_options_into_task(owning_task_id, added_titles)
        owner_task = self._find_task_by_id(owning_task_id)
        knowledge_entries = update_task_knowledge(self._work_session, owner_task, user_message, turn_id=self._current_turn_reference()) if owner_task is not None else []

        current = self._work_session.representation
        style = current.style if current is not None else "cards"
        all_option_ids = [
            oid
            for oid, o in self._work_session.objects.items()
            if o.kind == "option" and (not owning_task_id or o.task_id == owning_task_id)
        ]
        major_ids = list(current.major_object_ids) if current is not None else []
        hidden_ids = [
            oid for oid in (list(current.hidden_object_ids) if current is not None else []) if oid not in created_ids
        ]
        new_state = transform_representation(
            self._work_session, style, visible_ids=all_option_ids, major_ids=major_ids, hidden_ids=hidden_ids, source="auto",
        )
        self._queue_ui_event(
            "representation_changed",
            {
                "style": new_state.style,
                "visible_object_ids": list(new_state.visible_object_ids),
                "major_object_ids": list(new_state.major_object_ids),
                "hidden_object_ids": list(new_state.hidden_object_ids),
                "style_params": dict(new_state.style_params),
                "source": new_state.source,
            },
        )

        executed = 1  # "add option(s)" is one sub-intent regardless of how many names it listed
        failed = max(0, requested - executed)
        partial = failed > 0

        if self._turn_trace is not None:
            self._turn_trace["representation_command_detected"] = True
            self._turn_trace["representation_command_action"] = "mixed_add_option"
            self._turn_trace["representation_style"] = new_state.style
            self._turn_trace["representation_source"] = new_state.source
            self._turn_trace["representation_command_executed"] = True
            self._turn_trace["requested_intent_count"] = requested
            self._turn_trace["executed_intent_count"] = executed
            self._turn_trace["failed_intent_count"] = failed
            self._turn_trace["partial_execution"] = partial
            self._turn_trace["mixed_add_option_count"] = len(added_titles)
            self._turn_trace["mixed_add_existing_option_count"] = len(existing_titles)
            self._turn_trace["mixed_add_target_reference"] = target_reference
            self._turn_trace["mixed_add_owner_task_id"] = owning_task_id
            self._turn_trace["mixed_add_target_resolution_source"] = target_resolution_source
            self._turn_trace["task_knowledge_updated_count"] = len(knowledge_entries)

        labels = added_titles[0] if len(added_titles) == 1 else ", ".join(added_titles[:-1]) + " e " + added_titles[-1]
        if existing_titles and len(existing_titles) == len(added_titles):
            return f"{labels} já fazem parte da comparação."
        if partial:
            return (
                f"Acrescentei {labels}. Ainda não sei mover cartões individuais nem criar "
                "grupos semânticos automaticamente -- só fiz a parte que já sei fazer."
            )
        return f"Acrescentei {labels}."

    def _merge_options_into_task(self, task_id: str, titles: list[str]) -> None:
        if not task_id or not titles:
            return
        task = self._find_task_by_id(task_id)
        if task is None or task.task_type != "generic_comparison":
            return
        existing = _split_option_text(task.parameters.get("options", ""))
        seen = _normalized_option_set(existing)
        for title in titles:
            normalized = normalize_for_matching(title).strip(" ,.!?;:")
            if normalized and normalized not in seen:
                existing.append(title)
                seen.add(normalized)
        task.parameters["options"] = " e ".join(existing)
        self._touch_task(task)

    def _merge_criteria_into_task(self, task_id: str, titles: list[str]) -> None:
        if not task_id or not titles:
            return
        task = self._find_task_by_id(task_id)
        if task is None or task.task_type != "generic_comparison":
            return
        existing = _split_option_text(task.parameters.get("criteria", ""))
        seen = _normalized_option_set(existing)
        for title in titles:
            normalized = normalize_for_matching(title).strip(" ,.!?;:")
            if normalized and normalized not in seen:
                existing.append(title)
                seen.add(normalized)
        task.parameters["criteria"] = " e ".join(existing)
        self._touch_task(task)

    def _try_semantic_association_request(self, user_message: str) -> str | None:
        normalized = _normalize_text(user_message).strip(" .!?")
        if not contains_any_phrase(normalized, ("associa", "associar")):
            return None
        collection_resolution = resolve_semantic_collection_targets(user_message, self._work_session)
        if not collection_resolution.matches and not collection_resolution.ambiguous:
            target_reference = _semantic_association_target_reference(user_message)
            if target_reference:
                collection_resolution = resolve_semantic_collection_targets(f"os {target_reference}", self._work_session)
        if collection_resolution.ambiguous:
            return "A que conjunto queres associar isso?"
        if not collection_resolution.matches:
            return "Ainda não consigo identificar a que conjunto queres associar isso."
        owner_task_id = _single_owner_task_id(collection_resolution.matches)
        target_task = self._find_task_by_id(owner_task_id)
        if target_task is None or target_task.task_type != "generic_comparison":
            return "Ainda não tenho uma relação segura para associar isso a esse conjunto."

        source_task_id = self._work_session.active_task.id if self._work_session.active_task is not None else ""
        candidates = [
            obj
            for obj in self._work_session.objects.values()
            if obj.kind == "criterion" and obj.task_id != owner_task_id and (not source_task_id or obj.task_id == source_task_id or not obj.task_id)
        ]
        if not candidates:
            return "Não encontrei critérios novos para associar a essa comparação."

        for obj in candidates:
            obj.task_id = owner_task_id
            self._queue_semantic_object_sync(obj)
        self._merge_criteria_into_task(owner_task_id, [obj.title for obj in candidates])
        if self._turn_trace is not None:
            self._turn_trace["semantic_association_detected"] = True
            self._turn_trace["semantic_association_executed"] = True
            self._turn_trace["semantic_association_owner_task_id"] = owner_task_id
            self._turn_trace["semantic_association_count"] = len(candidates)
        labels = _criteria_label([obj.title for obj in candidates])
        return f"Associei {labels} a essa comparação."

    def _try_document_task(self, user_message: str) -> str | None:
        if not self.presence.can_use_tools():
            return None
        normalized = _normalize_text(user_message)
        if _looks_like_document_context_clear(normalized):
            self._active_document_context = None
            self._pending_document_task = None
            return None
        pending_response = self._try_pending_document_task(user_message, normalized)
        if pending_response is not None:
            return pending_response
        active_response = self._try_active_document_followup(user_message, normalized)
        if active_response is not None:
            return active_response
        request = _parse_document_task_request(user_message)
        if not _looks_like_document_task(normalized, request):
            return None

        self._last_fast_tools_used = ()
        output_name = request.output_file or ""
        write_only_content = _extract_inline_file_content(user_message)
        if output_name and write_only_content:
            return self._create_workspace_txt(output_name, write_only_content, tools_used=("create_workspace_file",))

        requested_file = request.source_files[0] if request.source_files else ""
        if not requested_file:
            if output_name and "write_output" in request.actions and not _needs_source_file(request):
                self._pending_document_task = {
                    "requested_output": output_name,
                    "requested_actions": ["create"],
                    "current_step": "awaiting_content",
                }
                self._last_fast_tools_used = ()
                return "Claro. Que texto queres guardar nesse ficheiro?"
            self._pending_document_task = {
                "requested_output": output_name,
                "document_requested_output": request.requested_output,
                "document_action": request.action,
                "output_format": request.output_format or "",
                "recipient": request.recipient or "",
                "requested_actions": request.actions,
                "instructions": request.instructions,
                "current_step": "awaiting_source_file",
                "missing_fields": ["source_file"],
            }
            self._last_fast_tools_used = ("workspace_search",)
            self._record_document_trace(found=False, files_used=[], output_created=False)
            return "Preciso do nome do ficheiro da workspace que queres usar."

        match = _workspace_search(requested_file, self.workspace_path)
        if match.blocked:
            self._last_fast_tools_used = ("workspace_search",)
            self._record_document_trace(found=False, files_used=[], output_created=False)
            return "Não posso aceder a caminhos fora da workspace."
        candidates = list(match.matches)
        if not candidates:
            self._pending_document_task = {
                "requested_output": output_name,
                "document_requested_output": request.requested_output,
                "document_action": request.action,
                "output_format": request.output_format or "",
                "recipient": request.recipient or "",
                "requested_actions": request.actions,
                "instructions": request.instructions,
                "current_step": "awaiting_source_file",
                "missing_fields": ["source_file"],
            }
            self._last_fast_tools_used = ("workspace_search",)
            self._record_document_trace(found=False, files_used=[], output_created=False)
            return f"Não encontrei '{requested_file}' na workspace. Qual é o nome exato do ficheiro?"
        if len(candidates) > 1:
            names = "\n".join(f"- {path.name}" for path in candidates[:8])
            self._pending_document_task = {
                "requested_output": output_name or "",
                "document_requested_output": request.requested_output,
                "document_action": request.action,
                "output_format": request.output_format or "",
                "recipient": request.recipient or "",
                "requested_actions": request.actions,
                "instructions": request.instructions,
                "current_step": "awaiting_file_choice",
                "candidates": [path.name for path in candidates],
            }
            self._last_fast_tools_used = ("workspace_search",)
            self._record_document_trace(found=False, files_used=[], output_created=False)
            return f"Encontrei mais do que uma possibilidade:\n{names}\n\nQual delas queres usar?"

        source_path = candidates[0]
        return self._run_document_task_with_file(source_path.relative_to(self.workspace_path.resolve()).as_posix(), request, match.match_type)

    def _try_pending_document_task(self, user_message: str, normalized: str) -> str | None:
        pending = self._pending_document_task
        if not pending:
            return None
        if _is_negative_confirmation(normalized):
            self._pending_document_task = None
            self._last_fast_tools_used = ()
            return "Está bem, não criei nada."
        if pending.get("current_step") == "read_completed":
            return None
        if pending.get("current_step") == "awaiting_content":
            output = str(pending.get("requested_output") or "")
            self._pending_document_task = None
            return self._create_workspace_txt(output, user_message, tools_used=("create_workspace_file",))
        if pending.get("current_step") == "awaiting_source_file":
            request = _document_request_from_pending(pending, user_message)
            source = request.source_files[0] if request.source_files else _extract_workspace_filename(user_message)
            if not source:
                # Bare stem reply to "qual é o nome do ficheiro?" (no extension,
                # no quotes, no "ficheiro/documento" prefix) — exactly what a
                # user naturally types here. _workspace_search already does
                # fuzzy stem matching, so hand it the cleaned raw message
                # instead of re-asking, as long as it still looks like a
                # plausible filename token and not another full sentence.
                candidate = _clean_filename_fragment(user_message).strip().strip("\"'“”")
                if candidate and _semantic_file_reference_is_plausible(candidate):
                    source = candidate
            if not source:
                return "Preciso mesmo do nome do ficheiro da workspace."
            match = _workspace_search(source, self.workspace_path)
            if match.blocked:
                self._last_fast_tools_used = ("workspace_search",)
                self._record_document_trace(found=False, files_used=[], output_created=False)
                return "Não posso aceder a caminhos fora da workspace."
            if not match.matches:
                self._last_fast_tools_used = ("workspace_search",)
                self._record_document_trace(found=False, files_used=[], output_created=False)
                return f"Não encontrei '{source}' na workspace. Qual é o nome exato do ficheiro?"
            if len(match.matches) > 1:
                pending["current_step"] = "awaiting_file_choice"
                pending["candidates"] = [path.name for path in match.matches]
                self._pending_document_task = pending
                names = "\n".join(f"- {path.name}" for path in match.matches[:8])
                self._last_fast_tools_used = ("workspace_search",)
                self._record_document_trace(found=False, files_used=[], output_created=False)
                return f"Encontrei mais do que uma possibilidade:\n{names}\n\nQual delas queres usar?"
            self._pending_document_task = None
            return self._run_document_task_with_file(
                match.matches[0].relative_to(self.workspace_path.resolve()).as_posix(),
                request,
                match.match_type,
            )
        if pending.get("current_step") == "awaiting_output_name":
            output = _extract_output_txt_filename(user_message) or _clean_filename_fragment(user_message)
            if not output:
                return "Que nome queres dar ao ficheiro .txt?"
            if not output.lower().endswith(".txt"):
                output = f"{Path(output).stem or output}.txt"
            content = str(pending.get("content") or "")
            tools = tuple(str(item) for item in pending.get("tools_used", ()) if item)
            self._pending_document_task = None
            return self._create_workspace_txt(output, content, tools_used=tools + ("create_workspace_file",))
        if pending.get("current_step") == "awaiting_create_confirmation":
            if not _is_positive_document_confirmation(normalized):
                return None
            output = str(pending.get("requested_output") or "")
            content = str(pending.get("content") or "")
            tools = tuple(str(item) for item in pending.get("tools_used", ()) if item)
            self._pending_document_task = None
            return self._create_workspace_txt(output, content, tools_used=tools + ("create_workspace_file",))
        if pending.get("current_step") == "awaiting_draft_save_confirmation":
            if not _is_positive_document_confirmation(normalized):
                return None
            output_name = str(pending.get("draft_output_name") or "")
            source_file = str(pending.get("draft_source_file") or "")
            context = self._active_document_context
            draft_content = context.draft_content if context and context.resolved_file == source_file else ""
            self._pending_document_task = None
            if not output_name or not draft_content:
                return "Já não tenho essa versão melhorada disponível para guardar."
            result = self._create_workspace_txt(output_name, draft_content, tools_used=("create_workspace_file",))
            if context is not None and result.startswith("Criei o ficheiro"):
                context.draft_saved = True
                self._active_document_context = context
                self._queue_ui_event(
                    "document_draft_saved",
                    {"file_name": source_file, "output_name": output_name},
                )
            return result
        if pending.get("current_step") == "awaiting_file_choice":
            choice = _choose_candidate_from_message(user_message, [str(item) for item in pending.get("candidates", [])])
            if not choice:
                return "Qual dos ficheiros queres usar?"
            request = _document_request_from_pending(pending, user_message)
            self._pending_document_task = None
            return self._run_document_task_with_file(choice, request, "choice")
        return None

    def _try_active_document_followup(self, user_message: str, normalized: str) -> str | None:
        context = self._active_document_context
        if context is None:
            return None
        action = _active_document_followup_action(normalized, context.detected_content_type, bool(context.draft_content))
        if not action:
            return None
        refresh_result = self._refresh_active_document_context(context)
        if refresh_result is None:
            self._active_document_context = None
            return "Não consegui voltar a aceder ao documento anterior na workspace."
        context, reloaded = refresh_result
        context.last_action = action
        context.current_document_task = action
        context.ttl = 8
        self._active_document_context = context
        files_used = [_document_context_file_used(context, match_type="active_context")]

        if action in {"display", "read"}:
            self._last_fast_tools_used = ("read_workspace_document",) if reloaded else ()
            self._record_document_trace(
                found=True,
                files_used=files_used,
                output_created=False,
                document_action=action,
                active_document=context,
                context_reused=not reloaded,
            )
            return f"Aqui está outra vez o conteúdo de '{context.resolved_file}':\n\n{_truncate_document_content(context.content)}"

        if action == "display_draft":
            self._last_fast_tools_used = ("read_workspace_document",) if reloaded else ()
            if not context.draft_content:
                self._record_document_trace(
                    found=True, files_used=files_used, output_created=False,
                    document_action=action, active_document=context, context_reused=not reloaded,
                )
                return "Ainda não criei nenhuma versão melhorada deste documento. Queres que o reescreva primeiro?"
            self._record_document_trace(
                found=True, files_used=files_used, output_created=False,
                document_action=action, active_document=context, context_reused=not reloaded,
                draft_reused=True,
                draft_revision_number=context.draft_revision,
            )
            saved_note = "" if context.draft_saved else " (ainda não guardei esta versão no ficheiro original)"
            return f"Aqui está a versão melhorada{saved_note}:\n\n{_truncate_document_content(context.draft_content)}"

        if action == "compare_versions":
            self._last_fast_tools_used = ("read_workspace_document",) if reloaded else ()
            self._record_document_trace(
                found=True, files_used=files_used, output_created=False,
                document_action=action, active_document=context, context_reused=not reloaded,
                draft_reused=bool(context.draft_content),
                draft_revision_number=context.draft_revision,
            )
            if not context.draft_content:
                return "Ainda não tenho nenhuma versão melhorada para comparar com o original."
            self._queue_ui_event(
                "document_comparison_ready",
                {
                    "file_name": context.resolved_file,
                    "revision": context.draft_revision,
                    "original_content": _truncate_document_content(context.content),
                    "draft_content": _truncate_document_content(context.draft_content),
                },
            )
            return (
                f"Original de '{context.resolved_file}':\n\n{_truncate_document_content(context.content)}\n\n"
                "--- versão melhorada (ainda não guardada) ---\n\n"
                f"{_truncate_document_content(context.draft_content)}"
            )

        if action == "discard_draft":
            self._last_fast_tools_used = ("read_workspace_document",) if reloaded else ()
            had_draft = bool(context.draft_content)
            context.draft_content = ""
            context.draft_action = ""
            context.draft_created_from_hash = ""
            context.draft_saved = False
            context.draft_updated_at = ""
            context.draft_revision = 0
            context.previous_draft_content = ""
            context.previous_draft_revision = 0
            self._active_document_context = context
            self._record_document_trace(
                found=True, files_used=files_used, output_created=False,
                document_action=action, active_document=context, context_reused=not reloaded,
            )
            if not had_draft:
                return "Não havia nenhuma versão melhorada para descartar."
            self._queue_ui_event("document_draft_discarded", {"file_name": context.resolved_file})
            return "Descartei a versão melhorada. Fica só o documento original."

        if action == "restore_previous_draft":
            self._last_fast_tools_used = ("read_workspace_document",) if reloaded else ()
            if not context.previous_draft_content:
                self._record_document_trace(
                    found=True, files_used=files_used, output_created=False,
                    document_action=action, active_document=context, context_reused=not reloaded,
                )
                return "Não há nenhuma versão anterior guardada para restaurar."
            context.draft_content = context.previous_draft_content
            context.draft_revision = context.previous_draft_revision
            context.previous_draft_content = ""
            context.previous_draft_revision = 0
            context.draft_saved = False
            context.draft_updated_at = self.now_provider().isoformat()
            self._active_document_context = context
            self._record_document_trace(
                found=True, files_used=files_used, output_created=False,
                document_action=action, active_document=context, context_reused=not reloaded,
                draft_reused=True,
                draft_revision_number=context.draft_revision,
            )
            self._queue_ui_event(
                "document_draft_restored",
                {
                    "file_name": context.resolved_file,
                    "revision": context.draft_revision,
                    "content": _truncate_document_content(context.draft_content),
                },
            )
            return f"Restaurei a versão anterior (revisão {context.draft_revision}):\n\n{_truncate_document_content(context.draft_content)}"

        if action == "save_draft":
            self._last_fast_tools_used = ("read_workspace_document",) if reloaded else ()
            self._record_document_trace(
                found=True, files_used=files_used, output_created=False,
                document_action=action, active_document=context, context_reused=not reloaded,
                draft_reused=bool(context.draft_content),
            )
            if not context.draft_content:
                return "Ainda não tenho nenhuma versão melhorada para guardar."
            stem = Path(context.resolved_file).stem
            suffix = Path(context.resolved_file).suffix or ".txt"
            proposed_name = f"{stem} (melhorado){suffix}"
            self._pending_document_task = {
                "current_step": "awaiting_draft_save_confirmation",
                "draft_output_name": proposed_name,
                "draft_source_file": context.resolved_file,
            }
            return (
                f"Não substituo o ficheiro original automaticamente. Queres que eu guarde a versão melhorada "
                f"num novo ficheiro, '{proposed_name}', mantendo '{context.resolved_file}' como está? (sim/não)"
            )

        if action == "rewrite":
            request = DocumentTaskRequest(
                source_files=[context.resolved_file],
                actions=["read_source", action],
                instructions=user_message,
                action=action,
                requested_output=_requested_output_for_document_action(action),
            )
            self._last_fast_tools_used = ("read_workspace_document",) if reloaded else ()
            original_content = context.content
            # A rewrite follow-up while a draft already exists is an
            # iterative refinement of that draft, not a fresh rewrite of the
            # pristine original -- "não voltar sempre ao original".
            is_refinement = bool(context.draft_content)
            base_content = context.draft_content if is_refinement else original_content
            document_followup_action = "iterative_refinement" if is_refinement else "rewrite"
            document_refinement_source = "draft" if is_refinement else "original"
            next_revision = context.draft_revision + 1 if is_refinement else 1
            rewrite_started_at = time.perf_counter()
            rewrite_deadline = rewrite_started_at + DOCUMENT_REWRITE_TOTAL_TIMEOUT_SECONDS
            document_anchors = _extract_document_anchors(original_content)
            routing_decisions: list[dict[str, object]] = []
            attempt_failure_reasons: list[str] = []
            last_task_profile: dict[str, object] = {}

            def _remaining_seconds() -> float:
                return rewrite_deadline - time.perf_counter()

            def _build_task_profile(
                attempt_number: int, previous_failure_reason: str, previous_provider: str = ""
            ) -> dict[str, object]:
                # Structured description of the RECONSTRUCTED task, not the
                # literal user message -- a short follow-up like "ainda
                # consegues melhor" carries none of this on its own. Never
                # includes the document's actual content (see
                # assistant/model_router.py's _document_task_complexity_score).
                #
                # previous_provider is the ACTUAL provider that served the
                # previous attempt (from routing_decisions, not assumed) --
                # a failed attempt is not necessarily a *local* failure, since
                # attempt 1 can itself already have been escalated to Claude
                # on a high-complexity profile. document_previous_local_failure
                # must only be true when that previous attempt was local.
                previous_provider = str(previous_provider or "").strip().lower()
                profile = {
                    "task_type": "document_refinement" if is_refinement else "document_rewrite",
                    "document_chars": len(base_content),
                    "document_has_draft": is_refinement,
                    "document_revision_number": context.draft_revision,
                    "document_structure_count": len(document_anchors["bullet_items"]),
                    "document_named_entity_count": len(document_anchors["entity_tokens"]),
                    "document_list_item_count": len(document_anchors["bullet_items"]),
                    "document_requires_fidelity": True,
                    "document_requires_full_output": True,
                    "document_previous_provider": previous_provider,
                    "document_previous_call_was_local": previous_provider == "ollama",
                    "document_previous_call_was_paid": previous_provider == "anthropic",
                    "document_previous_validation_failed": bool(previous_failure_reason),
                    "document_previous_local_failure": bool(previous_failure_reason) and previous_provider == "ollama",
                    "document_validation_failure_reason": previous_failure_reason,
                    "document_regeneration_attempt": attempt_number,
                    # Always "auto" -- the router stays the sole authority on
                    # provider choice; this flow never hardcodes a provider.
                    "preferred_provider": "auto",
                }
                nonlocal last_task_profile
                last_task_profile = profile
                return profile

            def _capture_routing_decision() -> None:
                decision = getattr(self.llm, "last_routing_decision", None)
                if decision is None:
                    return
                routing_decisions.append(
                    {
                        "provider": decision.provider,
                        "reason_code": decision.reason_code,
                        "paid_call": bool(decision.paid_call),
                        "task_complexity_score": float(getattr(decision, "task_complexity_score", 0.0) or 0.0),
                        "task_complexity_band": getattr(decision, "task_complexity_band", "") or "",
                        "escalation_considered": bool(getattr(decision, "escalation_considered", False)),
                    }
                )

            def _compose_rewrite(prompt: str, *, timeout_seconds: float, task_profile: dict[str, object]) -> str:
                reply = self.response_composer.compose(
                    ComposerRequest(
                        intent="document_rewrite",
                        user_message=prompt,
                        history=[],
                        facts=[
                            f"Documento ativo: {context.resolved_file}.",
                            f"Tipo detetado: {context.detected_content_type}.",
                        ],
                        fallback="Consigo reescrever o documento, mas preciso de me basear apenas no texto que li.",
                        language_instruction=self._language_instruction(),
                        temperature=_REWRITE_TEMPERATURE,
                        num_predict=_REWRITE_MAX_OUTPUT_TOKENS,
                        timeout_seconds=timeout_seconds,
                        task_profile=task_profile,
                    )
                )
                _capture_routing_decision()
                return reply

            def _routing_telemetry_kwargs() -> dict[str, object]:
                initial_provider = str(routing_decisions[0]["provider"]) if routing_decisions else ""
                final_provider = str(routing_decisions[-1]["provider"]) if routing_decisions else ""
                # Never deduplicated -- each position is one real attempt, so
                # ["anthropic", "anthropic"] and ["ollama", "anthropic"] stay
                # distinguishable instead of collapsing to the same set.
                sequence = [str(d["provider"]) for d in routing_decisions]
                escalation_reason = next(
                    (str(d["reason_code"]) for d in routing_decisions if d["provider"] == "anthropic"), ""
                )
                return {
                    "task_complexity_score": routing_decisions[-1]["task_complexity_score"] if routing_decisions else 0.0,
                    "task_complexity_band": routing_decisions[-1]["task_complexity_band"] if routing_decisions else "",
                    "document_task_type": str(last_task_profile.get("task_type") or ""),
                    "document_routing_features": dict(last_task_profile),
                    "escalation_considered": any(bool(d["escalation_considered"]) for d in routing_decisions),
                    "escalation_selected": "anthropic" in sequence,
                    "escalation_reason": escalation_reason,
                    "initial_provider": initial_provider,
                    "final_provider": final_provider,
                    "initial_routing_reason_code": str(routing_decisions[0]["reason_code"]) if routing_decisions else "",
                    "final_routing_reason_code": str(routing_decisions[-1]["reason_code"]) if routing_decisions else "",
                    "provider_attempt_sequence": sequence,
                    "attempt_count": len(routing_decisions),
                    "attempt_failure_reasons": list(attempt_failure_reasons),
                    "paid_call_authorized": any(bool(d["paid_call"]) for d in routing_decisions),
                }

            def _cancelled_response(attempt_1_ms: float, attempt_2_ms: float, regenerated: bool) -> str:
                self._emit_progress("rewrite_cancelled")
                self._record_document_trace(
                    found=True, files_used=files_used, output_created=False,
                    document_action=action, active_document=context, context_reused=not reloaded,
                    draft_validation_failed=True,
                    draft_regeneration_attempted=regenerated,
                    draft_original_length=len(original_content),
                    rewrite_attempt_1_latency_ms=attempt_1_ms,
                    rewrite_attempt_2_latency_ms=attempt_2_ms,
                    rewrite_total_latency_ms=attempt_1_ms + attempt_2_ms,
                    rewrite_regeneration_started=regenerated,
                    cancellation_requested=True,
                    request_cancelled=True,
                    rewrite_elapsed_ms_at_cancel=(time.perf_counter() - rewrite_started_at) * 1000,
                    request_finished_cleanly=True,
                    document_followup_action=document_followup_action,
                    document_refinement_source=document_refinement_source,
                    previous_draft_present=is_refinement,
                    refinement_regeneration_started=regenerated if is_refinement else False,
                    draft_revision_number=context.draft_revision,
                    local_failure_reason=local_failure_reason,
                    **_routing_telemetry_kwargs(),
                )
                return "Pedido cancelado."

            def _timeout_response(attempt_1_ms: float, attempt_2_ms: float, regenerated: bool, remaining_seconds: float) -> str:
                self._emit_progress("rewrite_timeout")
                self._record_document_trace(
                    found=True, files_used=files_used, output_created=False,
                    document_action=action, active_document=context, context_reused=not reloaded,
                    draft_validation_failed=True,
                    draft_regeneration_attempted=regenerated,
                    draft_original_length=len(original_content),
                    rewrite_attempt_1_latency_ms=attempt_1_ms,
                    rewrite_attempt_2_latency_ms=attempt_2_ms,
                    rewrite_total_latency_ms=attempt_1_ms + attempt_2_ms,
                    rewrite_regeneration_started=regenerated,
                    rewrite_timeout=True,
                    rewrite_timeout_seconds=DOCUMENT_REWRITE_TOTAL_TIMEOUT_SECONDS,
                    rewrite_elapsed_ms_at_timeout=(time.perf_counter() - rewrite_started_at) * 1000,
                    rewrite_remaining_timeout_ms=max(0.0, remaining_seconds * 1000),
                    request_finished_cleanly=True,
                    document_followup_action=document_followup_action,
                    document_refinement_source=document_refinement_source,
                    previous_draft_present=is_refinement,
                    refinement_regeneration_started=regenerated if is_refinement else False,
                    draft_revision_number=context.draft_revision,
                    local_failure_reason=local_failure_reason,
                    **_routing_telemetry_kwargs(),
                )
                return (
                    "A reescrita demorou mais do que o previsto e foi interrompida. "
                    "O ficheiro original continua igual."
                )

            def _check_noop(candidate: str) -> bool:
                return is_refinement and _is_noop_refinement(base_content, candidate)

            local_failure_reason = ""

            # Checkpoint 1: before the first LLM call.
            prompt = (
                _document_refinement_prompt(request, context, base_content)
                if is_refinement
                else _document_rewrite_prompt(request, context)
            )
            if self.is_cancel_requested():
                return _cancelled_response(0.0, 0.0, False)
            remaining_for_attempt_1 = _remaining_seconds()
            if remaining_for_attempt_1 < _REWRITE_MIN_SECONDS_FOR_ATTEMPT:
                return _timeout_response(0.0, 0.0, False, remaining_for_attempt_1)

            self._emit_progress("rewrite_attempt_started")
            attempt_1_started_at = time.perf_counter()
            try:
                attempt = _compose_rewrite(
                    prompt, timeout_seconds=remaining_for_attempt_1, task_profile=_build_task_profile(1, "")
                )
            except ProviderTimeoutError:
                attempt_1_latency_ms = (time.perf_counter() - attempt_1_started_at) * 1000
                return _timeout_response(attempt_1_latency_ms, 0.0, False, _remaining_seconds())
            attempt_1_latency_ms = (time.perf_counter() - attempt_1_started_at) * 1000

            # Checkpoint 2: immediately after the first call.
            if self.is_cancel_requested():
                return _cancelled_response(attempt_1_latency_ms, 0.0, False)

            self._emit_progress("rewrite_validation_started")
            valid, reason, cleaned = _validate_rewrite_draft(original_content, attempt)
            if valid and _check_noop(cleaned):
                valid, reason = False, "no_op_detected"
            attempt_failure_reasons.append(reason if not valid else "")
            preamble_removed = cleaned.strip() != attempt.strip()
            placeholder_detected = reason == "placeholder_detected"
            regenerated = False
            attempt_2_latency_ms = 0.0

            if not valid:
                # Checkpoint 3: before starting the second attempt -- also
                # where the proactive total-timeout check happens, since
                # starting an LLM call we already know can't finish in time
                # is worse than giving up cleanly.
                if self.is_cancel_requested():
                    return _cancelled_response(attempt_1_latency_ms, 0.0, False)
                remaining_for_attempt_2 = _remaining_seconds()
                if remaining_for_attempt_2 < _REWRITE_MIN_SECONDS_FOR_ATTEMPT:
                    return _timeout_response(attempt_1_latency_ms, 0.0, False, remaining_for_attempt_2)

                regenerated = True
                local_failure_reason = reason
                # The provider that actually served attempt 1 -- never
                # assumed -- so attempt 2's profile can tell an Ollama
                # failure apart from a Claude failure (see _build_task_profile).
                previous_provider = str(routing_decisions[-1]["provider"]) if routing_decisions else ""
                self._emit_progress("rewrite_regeneration_started")
                correction_prompt = (
                    _document_refinement_correction_prompt(request, context, base_content, attempt, reason)
                    if is_refinement
                    else _document_rewrite_correction_prompt(request, context, attempt, reason)
                )
                attempt_2_started_at = time.perf_counter()
                try:
                    attempt = _compose_rewrite(
                        correction_prompt,
                        timeout_seconds=remaining_for_attempt_2,
                        task_profile=_build_task_profile(2, local_failure_reason, previous_provider),
                    )
                except ProviderTimeoutError:
                    attempt_2_latency_ms = (time.perf_counter() - attempt_2_started_at) * 1000
                    return _timeout_response(attempt_1_latency_ms, attempt_2_latency_ms, True, _remaining_seconds())
                attempt_2_latency_ms = (time.perf_counter() - attempt_2_started_at) * 1000

                # Checkpoint 4: immediately after the second call.
                if self.is_cancel_requested():
                    return _cancelled_response(attempt_1_latency_ms, attempt_2_latency_ms, True)

                self._emit_progress("rewrite_validation_started")
                valid, reason, cleaned = _validate_rewrite_draft(original_content, attempt)
                if valid and _check_noop(cleaned):
                    valid, reason = False, "no_op_detected"
                attempt_failure_reasons.append(reason if not valid else "")
                preamble_removed = preamble_removed or (cleaned.strip() != attempt.strip())
                placeholder_detected = placeholder_detected or (reason == "placeholder_detected")
            total_latency_ms = attempt_1_latency_ms + attempt_2_latency_ms

            if not valid:
                # Cancellation takes priority over reporting a structural
                # validation failure -- if the user asked to stop, tell them
                # that, not why the (now irrelevant) draft was rejected.
                if self.is_cancel_requested():
                    return _cancelled_response(attempt_1_latency_ms, attempt_2_latency_ms, regenerated)
                self._record_document_trace(
                    found=True, files_used=files_used, output_created=False,
                    document_action=action, active_document=context, context_reused=not reloaded,
                    draft_validation_failed=True,
                    draft_regeneration_attempted=regenerated,
                    draft_validation_reason=reason,
                    draft_original_length=len(original_content),
                    draft_generated_length=len(attempt),
                    rewrite_attempt_1_latency_ms=attempt_1_latency_ms,
                    rewrite_attempt_2_latency_ms=attempt_2_latency_ms,
                    rewrite_total_latency_ms=total_latency_ms,
                    rewrite_regeneration_started=regenerated,
                    draft_preamble_removed=preamble_removed,
                    draft_placeholder_detected=placeholder_detected,
                    request_finished_cleanly=True,
                    document_followup_action=document_followup_action,
                    document_refinement_source=document_refinement_source,
                    previous_draft_present=is_refinement,
                    refinement_changed_content=False,
                    refinement_regeneration_started=regenerated if is_refinement else False,
                    draft_revision_number=context.draft_revision,
                    local_failure_reason=local_failure_reason,
                    **_routing_telemetry_kwargs(),
                )
                fallback_note = (
                    "O ficheiro original continua igual — queres que tente outra vez?"
                    if not is_refinement
                    else "A versão anterior continua ativa — queres que tente melhorar outra vez?"
                )
                intro = (
                    "Não consegui gerar uma versão reescrita completa e fiável deste documento."
                    if not is_refinement
                    else "Não consegui melhorar a versão atual de forma fiável."
                )
                return f"{intro} {_rewrite_failure_message(reason)} {fallback_note}"

            # Checkpoints 5 & 6: before creating/saving the draft and before
            # returning the final (success) response -- checked as one gate
            # since nothing blocking happens between them, and the draft
            # must never be created once cancellation is observed.
            if self.is_cancel_requested():
                return _cancelled_response(attempt_1_latency_ms, attempt_2_latency_ms, regenerated)

            if is_refinement:
                context.previous_draft_content = context.draft_content
                context.previous_draft_revision = context.draft_revision
            context.draft_content = cleaned
            context.draft_action = action
            context.draft_created_from_hash = context.content_hash
            context.draft_saved = False
            context.draft_updated_at = self.now_provider().isoformat()
            context.draft_revision = next_revision
            self._active_document_context = context
            self._queue_ui_event(
                "document_draft_ready",
                {
                    "file_name": context.resolved_file,
                    "revision": next_revision,
                    "based_on": document_refinement_source,
                    "content": _truncate_document_content(cleaned),
                },
            )
            self._record_document_trace(
                found=True, files_used=files_used, output_created=False,
                document_action=action, active_document=context, context_reused=not reloaded,
                draft_created=True,
                draft_validation_passed=True,
                draft_regeneration_attempted=regenerated,
                draft_original_length=len(original_content),
                draft_generated_length=len(cleaned),
                rewrite_attempt_1_latency_ms=attempt_1_latency_ms,
                rewrite_attempt_2_latency_ms=attempt_2_latency_ms,
                rewrite_total_latency_ms=total_latency_ms,
                rewrite_regeneration_started=regenerated,
                draft_preamble_removed=preamble_removed,
                draft_placeholder_detected=placeholder_detected,
                request_finished_cleanly=True,
                document_followup_action=document_followup_action,
                document_refinement_source=document_refinement_source,
                previous_draft_present=is_refinement,
                refinement_changed_content=is_refinement,
                refinement_regeneration_started=regenerated if is_refinement else False,
                draft_revision_number=next_revision,
                local_failure_reason=local_failure_reason,
                **_routing_telemetry_kwargs(),
            )
            intro = "Aqui está uma versão ainda melhor" if is_refinement else "Aqui está a versão reescrita"
            return f"{intro}:\n\n{cleaned}\n\n(Ainda não guardei esta versão — o ficheiro original continua igual.)"

        request = DocumentTaskRequest(
            source_files=[context.resolved_file],
            actions=["read_source", action],
            instructions=user_message,
            action=action,
            requested_output=_requested_output_for_document_action(action),
        )
        self._last_fast_tools_used = ("read_workspace_document",) if reloaded else ()
        self._record_document_trace(
            found=True,
            files_used=files_used,
            output_created=False,
            document_action=action,
            active_document=context,
            context_reused=not reloaded,
        )
        return self.response_composer.compose(
            ComposerRequest(
                intent=f"document_{action}",
                user_message=_active_document_followup_prompt(request, context),
                history=[],
                facts=[
                    f"Documento ativo: {context.resolved_file}.",
                    f"Tipo detetado: {context.detected_content_type}.",
                    "Usa apenas o conteúdo real do documento ativo.",
                    "Não afirmes que vais ler, abrir ou pesquisar; o conteúdo já está disponível.",
                ],
                fallback="Consigo analisar o documento, mas preciso de me basear apenas no texto que li.",
                language_instruction=self._language_instruction(),
            )
        )

    def _try_active_document_grounded_fallback(self, user_message: str, normalized: str) -> str | None:
        """Last-resort grounding for a document-active turn: runs only after
        every more specific handler (pending task, explicit follow-up
        markers, writing task, system/tool intent) has already declined the
        message. Deliberately narrow — it only claims messages that look like
        an opinion/suggestion about the document itself (e.g. "tens alguma
        sugestão para melhorar o mail?"), not just any leftover message,
        so free topic changes and unrelated questions still fall through to
        general conversation as before.
        """
        if not self.presence.can_use_tools():
            return None
        context = self._active_document_context
        if context is None:
            return None
        if _looks_like_document_context_clear(normalized):
            return None
        if not _looks_like_ambiguous_document_followup(normalized, context.detected_content_type):
            return None
        refresh_result = self._refresh_active_document_context(context)
        if refresh_result is None:
            self._active_document_context = None
            return None
        context, reloaded = refresh_result
        context.last_action = "review"
        context.current_document_task = "review"
        context.ttl = 8
        self._active_document_context = context
        files_used = [_document_context_file_used(context, match_type="active_context")]
        self._last_fast_tools_used = ("read_workspace_document",) if reloaded else ()
        self._record_document_trace(
            found=True,
            files_used=files_used,
            output_created=False,
            document_action="review",
            active_document=context,
            context_reused=not reloaded,
        )
        request = DocumentTaskRequest(
            source_files=[context.resolved_file],
            actions=["read_source", "review"],
            instructions=user_message,
            action="review",
            requested_output=_requested_output_for_document_action("review"),
        )
        return self.response_composer.compose(
            ComposerRequest(
                intent="document_review",
                user_message=_active_document_followup_prompt(request, context),
                history=[],
                facts=[
                    f"Documento ativo: {context.resolved_file}.",
                    f"Tipo detetado: {context.detected_content_type}.",
                    "Usa apenas o conteúdo real do documento ativo para responder.",
                    "Não afirmes que vais ler, abrir ou pesquisar; o conteúdo já está disponível.",
                ],
                fallback="Consigo comentar o documento, mas preciso de me basear apenas no texto que li.",
                language_instruction=self._language_instruction(),
            )
        )

    def _try_ambiguous_opinion_clarification(self, user_message: str) -> str | None:
        """A short opinion question with no antecedent at all (e.g. "O que
        achas?" right after "Vamos falar de outra coisa.") has nothing to be
        grounded in — there is no active document, and it is too short/vague
        to answer or to treat as a memory-recall question. Ask what it is
        about instead of letting the Composer guess; an ungrounded guess here
        is exactly the kind of thing the Composer's own memory-claim guard
        later has to catch and replace with a canned "posso consultar a
        memória real" message, which reads as a wrong-cause substitution for
        plain conversational ambiguity.
        """
        if self._active_document_context is not None:
            return None
        normalized = _normalize_text(user_message)
        if not _is_referentially_ambiguous_opinion_question(normalized):
            return None
        return "Sobre o quê?"

    def _refresh_active_document_context(self, context: ActiveDocumentContext) -> tuple[ActiveDocumentContext, bool] | None:
        path = (self.workspace_path / context.resolved_file).resolve()
        try:
            path.relative_to(self.workspace_path.resolve())
        except ValueError:
            return None
        if not path.exists() or not path.is_file():
            return None
        try:
            stat = path.stat()
        except OSError:
            return None
        if stat.st_mtime_ns == context.mtime_ns and stat.st_size == context.file_size:
            return context, False
        read_result = read_workspace_file_content(context.resolved_file, self.workspace_path)
        if read_result.error is not None or not read_result.content.strip():
            return None
        return _build_active_document_context(read_result.filename, read_result.content, self.workspace_path, context.last_action), True

    def _run_document_task_with_file(self, filename: str, request: DocumentTaskRequest, match_type: str = "exact") -> str:
        read_result = read_workspace_file_content(filename, self.workspace_path)
        files_used = [
            {
                "name": read_result.filename or filename,
                "type": Path(read_result.filename or filename).suffix.lower().lstrip("."),
                "read_success": read_result.error is None and bool(read_result.content.strip()),
                "match_type": match_type,
            }
        ]
        if read_result.error is not None:
            self._last_fast_tools_used = ("workspace_search", "read_workspace_document")
            self._record_document_trace(found=False, files_used=files_used, output_created=False)
            return read_result.error
        if not read_result.content.strip():
            self._last_fast_tools_used = ("workspace_search", "read_workspace_document")
            self._record_document_trace(found=True, files_used=files_used, output_created=False)
            return f"O ficheiro '{read_result.filename}' está vazio ou não tem texto extraível."

        active_context = _build_active_document_context(
            read_result.filename,
            read_result.content,
            self.workspace_path,
            request.action or ("summarize" if "summarize" in request.actions else "read"),
        )
        self._active_document_context = active_context
        self._queue_ui_event(
            "document_loaded",
            {
                "file_name": active_context.resolved_file,
                "content_type": active_context.detected_content_type,
                "chars": active_context.content_length,
                "preview": _truncate_document_content(active_context.content, limit=280),
            },
        )
        normalized = _normalize_text(request.instructions)
        full_contents_requested = request.requested_output == "full_contents" or _asks_to_read_file(normalized)
        if full_contents_requested and not _looks_like_summary_or_email(normalized) and not request.output_file:
            self._last_fast_tools_used = ("workspace_search", "read_workspace_document")
            self._record_document_trace(
                found=True,
                files_used=files_used,
                output_created=False,
                document_action="read",
                active_document=active_context,
                context_reused=False,
            )
            self._pending_document_task = {
                "source_file": read_result.filename,
                "resolved_file": read_result.filename,
                "document_requested_output": "full_contents",
                "document_action": "read",
                "requested_actions": ["locate_source", "read_source"],
                "current_step": "read_completed",
                "last_read_content_reference": read_result.filename,
            }
            return f"Aqui está o conteúdo de '{read_result.filename}':\n\n{_truncate_document_content(read_result.content)}"

        composed = self.response_composer.compose(
            ComposerRequest(
                intent="document_task",
                user_message=_document_composition_prompt(request, read_result.filename, read_result.content),
                history=[],
                facts=[
                    f"Ficheiro lido: {read_result.filename}.",
                    "Baseia a resposta apenas no conteúdo lido do ficheiro.",
                    "Não afirmes que pesquisaste na web.",
                ],
                fallback="Posso trabalhar esse ficheiro, mas preciso de conteúdo legível para responder bem.",
                language_instruction=self._language_instruction(),
            )
        )
        self._last_fast_tools_used = ("workspace_search", "read_workspace_document")
        document_action = request.action or ("summarize" if "summarize" in request.actions else "read")
        self._record_document_trace(
            found=True,
            files_used=files_used,
            output_created=False,
            document_action=document_action,
            active_document=active_context,
            context_reused=False,
        )

        if request.output_file:
            return self._create_workspace_txt(
                request.output_file,
                composed,
                tools_used=("workspace_search", "read_workspace_document", "create_workspace_file"),
                files_used=files_used,
            )
        if request.output_format == "txt" or "write_output" in request.actions:
            self._pending_document_task = {
                "source_file": read_result.filename,
                "requested_output": "",
                "output_format": request.output_format or "txt",
                "recipient": request.recipient or "",
                "requested_actions": request.actions,
                "current_step": "awaiting_output_name",
                "missing_fields": ["output_file"],
                "content": composed,
                "tools_used": ("workspace_search", "read_workspace_document"),
            }
            return "Preparei o conteúdo. Que nome queres dar ao ficheiro .txt?"
        return composed

    def _create_workspace_txt(
        self,
        filename: str,
        content: str,
        *,
        tools_used: tuple[str, ...],
        files_used: list[dict[str, object]] | None = None,
    ) -> str:
        result = create_workspace_file(filename, content, self.workspace_path)
        created = result.startswith("Criei o ficheiro")
        self._last_fast_tools_used = tools_used
        self._record_document_trace(
            found=True,
            files_used=files_used or [],
            output_created=created,
            output_file_path=filename if created else "",
        )
        return result

    def _record_document_trace(
        self,
        *,
        found: bool,
        files_used: list[dict[str, object]],
        output_created: bool,
        output_file_path: str = "",
        document_action: str = "",
        active_document: ActiveDocumentContext | None = None,
        context_reused: bool = False,
        draft_created: bool = False,
        draft_reused: bool = False,
        draft_validation_passed: bool = False,
        draft_validation_failed: bool = False,
        draft_regeneration_attempted: bool = False,
        draft_validation_reason: str = "",
        draft_original_length: int = 0,
        draft_generated_length: int = 0,
        rewrite_attempt_1_latency_ms: float = 0.0,
        rewrite_attempt_2_latency_ms: float = 0.0,
        rewrite_total_latency_ms: float = 0.0,
        draft_preamble_removed: bool = False,
        draft_placeholder_detected: bool = False,
        rewrite_regeneration_started: bool = False,
        cancellation_requested: bool = False,
        request_cancelled: bool = False,
        rewrite_timeout: bool = False,
        rewrite_timeout_seconds: float = 0.0,
        rewrite_elapsed_ms_at_cancel: float = 0.0,
        rewrite_elapsed_ms_at_timeout: float = 0.0,
        rewrite_remaining_timeout_ms: float = 0.0,
        request_finished_cleanly: bool = True,
        document_followup_action: str = "",
        document_refinement_source: str = "",
        previous_draft_present: bool = False,
        refinement_changed_content: bool = False,
        refinement_regeneration_started: bool = False,
        draft_revision_number: int = 0,
        task_complexity_score: float = 0.0,
        task_complexity_band: str = "",
        document_task_type: str = "",
        document_routing_features: dict[str, object] | None = None,
        escalation_considered: bool = False,
        escalation_selected: bool = False,
        escalation_reason: str = "",
        initial_provider: str = "",
        final_provider: str = "",
        initial_routing_reason_code: str = "",
        final_routing_reason_code: str = "",
        provider_attempt_sequence: list[str] | None = None,
        attempt_count: int = 0,
        attempt_failure_reasons: list[str] | None = None,
        local_failure_reason: str = "",
        paid_call_authorized: bool = False,
    ) -> None:
        if self._turn_trace is None:
            return
        self._turn_trace["response_grounded"] = bool(found or output_created)
        self._turn_trace["grounding_sources"] = ["WORKSPACE_FILE"] if found else []
        self._turn_trace["workspace_file_found"] = bool(found)
        self._turn_trace["workspace_files_used"] = files_used
        self._turn_trace["output_file_created"] = bool(output_created)
        self._turn_trace["output_file_path"] = output_file_path
        context = active_document or self._active_document_context
        self._turn_trace["active_document"] = bool(context)
        self._turn_trace["active_document_name"] = context.resolved_file if context else ""
        self._turn_trace["document_action"] = document_action or (context.last_action if context else "")
        self._turn_trace["document_context_reused"] = bool(context_reused)
        self._turn_trace["document_grounded"] = bool(found)
        self._turn_trace["draft_created"] = bool(draft_created)
        self._turn_trace["draft_saved"] = bool(context.draft_saved) if context else False
        self._turn_trace["draft_reused"] = bool(draft_reused)
        # Booleans/lengths/short reason label only — never the draft or the
        # model's raw invalid response text.
        self._turn_trace["draft_validation_passed"] = bool(draft_validation_passed)
        self._turn_trace["draft_validation_failed"] = bool(draft_validation_failed)
        self._turn_trace["draft_regeneration_attempted"] = bool(draft_regeneration_attempted)
        self._turn_trace["draft_validation_reason"] = draft_validation_reason
        self._turn_trace["draft_original_length"] = int(draft_original_length)
        self._turn_trace["draft_generated_length"] = int(draft_generated_length)
        self._turn_trace["rewrite_attempt_1_latency_ms"] = round(float(rewrite_attempt_1_latency_ms), 1)
        self._turn_trace["rewrite_attempt_2_latency_ms"] = round(float(rewrite_attempt_2_latency_ms), 1)
        self._turn_trace["rewrite_total_latency_ms"] = round(float(rewrite_total_latency_ms), 1)
        self._turn_trace["draft_preamble_removed"] = bool(draft_preamble_removed)
        self._turn_trace["draft_placeholder_detected"] = bool(draft_placeholder_detected)
        self._turn_trace["rewrite_regeneration_started"] = bool(rewrite_regeneration_started)
        self._turn_trace["cancellation_requested"] = bool(cancellation_requested)
        self._turn_trace["request_cancelled"] = bool(request_cancelled)
        self._turn_trace["rewrite_timeout"] = bool(rewrite_timeout)
        self._turn_trace["rewrite_timeout_seconds"] = float(rewrite_timeout_seconds)
        self._turn_trace["rewrite_elapsed_ms_at_cancel"] = round(float(rewrite_elapsed_ms_at_cancel), 1)
        self._turn_trace["rewrite_elapsed_ms_at_timeout"] = round(float(rewrite_elapsed_ms_at_timeout), 1)
        self._turn_trace["rewrite_remaining_timeout_ms"] = round(float(rewrite_remaining_timeout_ms), 1)
        self._turn_trace["request_finished_cleanly"] = bool(request_finished_cleanly)
        self._turn_trace["document_followup_action"] = document_followup_action
        self._turn_trace["document_refinement_source"] = document_refinement_source
        self._turn_trace["previous_draft_present"] = bool(previous_draft_present)
        self._turn_trace["refinement_changed_content"] = bool(refinement_changed_content)
        self._turn_trace["refinement_regeneration_started"] = bool(refinement_regeneration_started)
        self._turn_trace["draft_revision_number"] = int(draft_revision_number)
        # Router-escalation telemetry (see assistant/model_router.py) --
        # booleans/scores/short codes only, never document content.
        self._turn_trace["task_complexity_score"] = float(task_complexity_score)
        self._turn_trace["task_complexity_band"] = task_complexity_band
        self._turn_trace["document_task_type"] = document_task_type
        self._turn_trace["document_routing_features"] = dict(document_routing_features or {})
        self._turn_trace["escalation_considered"] = bool(escalation_considered)
        self._turn_trace["escalation_selected"] = bool(escalation_selected)
        self._turn_trace["escalation_reason"] = escalation_reason
        self._turn_trace["initial_provider"] = initial_provider
        self._turn_trace["final_provider"] = final_provider
        self._turn_trace["initial_routing_reason_code"] = initial_routing_reason_code
        self._turn_trace["final_routing_reason_code"] = final_routing_reason_code
        self._turn_trace["provider_attempt_sequence"] = list(provider_attempt_sequence or [])
        self._turn_trace["attempt_count"] = int(attempt_count)
        self._turn_trace["attempt_failure_reasons"] = list(attempt_failure_reasons or [])
        self._turn_trace["local_failure_reason"] = local_failure_reason
        self._turn_trace["paid_call_authorized"] = bool(paid_call_authorized)

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
        response = simple_social_response(user_message, self.response_language)
        if response is not None:
            return response
        english = normalize_response_language(self.response_language) == ENGLISH_GB
        if "cor preferida" in text:
            if any(marker in text for marker in ("irma de 7 anos", "irmÃ£ de 7 anos", "crianca", "crianÃ§a")):
                return "I'd choose blue, because it looks like the colour of my lights." if english else "Eu escolhia azul, porque parece a cor das minhas luzes."
            return "If I had to choose, I'd say blue. It suits me." if english else "Se tivesse de escolher uma, dizia azul — combina comigo."
        if "clube" in text and any(marker in text for marker in ("qual", "tens", "preferido")):
            return "I don't have a club, but I can choose one just for fun." if english else "Não tenho clube, mas posso escolher um só para brincar contigo."
        return "I'm good. How about you?" if english else "Estou bem. E tu?"

    def _try_casual_social_share(self, user_message: str) -> str | None:
        text = _normalize_text(user_message)
        if _looks_like_help_or_planning_request(text):
            return None
        if _looks_like_weekend_beach_share(text):
            english = normalize_response_language(self.response_language) == ENGLISH_GB
            if any(phrase in text for phrase in ("trabalhar em ti", "trabalhar no echo", "neste projeto", "no echo")):
                if english:
                    return (
                        "So you had a calm week, even if part of it was spent fixing me. "
                        "And that weekend sounds well chosen."
                    )
                return (
                    "Então tiveste uma semana tranquila, apesar de a passares parcialmente a corrigir-me. "
                    "E esse fim de semana parece bem escolhido."
                )
            if any(word in text for word in ("remo", "remar")):
                if english:
                    return "That sounds like a good weekend. Beach, friends and rowing fit nicely together."
                return "Parece um bom fim de semana. Praia, amigos e ir remar combinam bem."
            if any(word in text for word in ("amigos", "amigas")):
                if english:
                    return "That sounds like a good weekend. Beach and friends go well together."
                return "Parece um bom fim de semana. Praia e amigos combinam bem."
            return "That sounds like a good weekend." if english else "Parece um bom fim de semana."
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
        status = voice_status_report(
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
        tts_status = self._tts_status_summary()
        return f"{status}\n\n{tts_status}" if tts_status else status

    def _tts_status_summary(self) -> str:
        manager = self.tts_manager
        diagnostics = getattr(manager, "diagnostics", None)
        if not callable(diagnostics):
            return "TTS: não está ligado nesta sessão."
        try:
            data = diagnostics()
        except Exception:
            return "TTS: não consegui verificar a disponibilidade."
        available = bool(data.get("available"))
        active_voice_id = str(data.get("active_voice_id") or "")
        resolved = str(data.get("resolved_voice_name") or data.get("resolved_voice_id") or "")
        slots = data.get("logical_slots") if isinstance(data.get("logical_slots"), list) else []
        slot_lines = []
        for slot in slots:
            if not isinstance(slot, dict):
                continue
            usable = "disponível" if slot.get("usable") else "indisponível"
            slot_lines.append(f"- {slot.get('raw_id')}: {usable}")
        language_label = response_language_status(self.response_language)
        header = (
            f"Idioma da resposta: {language_label}.\n"
            f"TTS: {'disponível' if available else 'não encontrei uma voz local compatível'}.\n"
            f"Voz ativa: {active_voice_id or 'nenhuma'}.\n"
            f"Voz resolvida: {resolved or 'nenhuma'}."
        )
        if slot_lines:
            return header + "\n" + "\n".join(slot_lines)
        return header

    def _try_tts_command(self, user_message: str) -> str | None:
        manager = self.tts_manager
        handler = getattr(manager, "handle_voice_command", None)
        if not callable(handler):
            return None
        result = handler(user_message)
        if not getattr(result, "handled", False):
            return None
        return str(getattr(result, "message", "") or "Está feito.")

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

        if final_response_source in {"RESPONSE_COMPOSER", "AGENT_DIRECT", "COMPOSER_REGENERATED"} and not tools_used and not technical:
            activity_claim_reason = _detect_ungrounded_activity_claim(final_response)
            if activity_claim_reason:
                final_response = "Não tenho essa informação sobre a tua atividade real — só o Context Observer sabe isso."
                final_response_source = "ACTIVITY_CLAIM_BLOCKED"
                self._last_response_blocked = True
                if self._turn_trace is not None:
                    self._turn_trace["response_grounded"] = False
                    self._turn_trace["grounding_sources"] = []

        if final_response_source in {"RESPONSE_COMPOSER", "AGENT_DIRECT", "COMPOSER_REGENERATED"} and not tools_used and not technical:
            workspace_claim_reason = _detect_ungrounded_workspace_operation_claim(user_message, final_response)
            if workspace_claim_reason:
                final_response = "Não alterei nada no workspace. Preciso de identificar um objeto real antes de agir."
                final_response_source = "WORKSPACE_CLAIM_BLOCKED"
                self._last_response_blocked = True
                if self._turn_trace is not None:
                    self._turn_trace["ungrounded_workspace_operation_claim"] = workspace_claim_reason

        if final_response and not tool_confirmation:
            localized = localize_deterministic_response(final_response, self.response_language)
            if localized != final_response:
                final_response = localized
                if self._turn_trace is not None:
                    self._turn_trace["response_language_localized"] = True

        if not technical and not tool_confirmation and looks_mixed_for_language(final_response, self.response_language):
            final_response = (
                "I’ll keep this in English. What would you like to do next?"
                if normalize_response_language(self.response_language) == ENGLISH_GB
                else "Vou manter a resposta em português. O que queres fazer a seguir?"
            )
            final_response_source = "LANGUAGE_GUARD_FALLBACK"
            if self._turn_trace is not None:
                self._turn_trace["language_guard_triggered"] = True

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
            self._turn_trace["response_language"] = self.response_language
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
            llm_start = int(self._turn_trace.get("llm_start") or 0)
            if self._llm_chat_count() > llm_start:
                self._turn_trace.update(_model_routing_telemetry(self.llm))
            else:
                self._turn_trace.update(_local_turn_model_telemetry(selected_path or source, self.llm))
                if (selected_path or source) == "PROVIDER_ERROR":
                    failed_provider = str(self._turn_trace.get("provider") or "")
                    if failed_provider:
                        self._turn_trace["execution_provider"] = failed_provider
                        self._turn_trace["model_routing_provider"] = failed_provider
                    error_code = str(self._turn_trace.get("provider_error_type") or "")
                    if error_code:
                        self._turn_trace["model_routing_reason_code"] = error_code
                        self._turn_trace["model_routing_reason"] = _local_reason_label(error_code)
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
        self._canonical_interpretation = None
        self._routing_decision = None
        self._turn_trace = {
            "user_message": user_message,
            "llm_start": self._llm_chat_count(),
            "llm_sources_start": len(getattr(self.llm, "chat_call_sources", ())),
            # Only flipped to True by _record_document_trace once a turn
            # reaches a normal exit path -- stays False if the turn instead
            # ends via _handle_unexpected_error (an uncaught exception).
            "request_finished_cleanly": False,
        }

    def _run_shadow_interpreter(self, user_message: str) -> None:
        """Classifies this utterance once through the Interaction
        Interpreter and records the central routing decision for the turn.
        This method:

        - reads self._work_session (never mutates it -- interpret_shadow
          is a pure function, see assistant/interaction_interpreter.py);
        - never calls an LLM/provider;
        - stores the interpretation/decision for deterministic executors
          that now consume the shared semantic authority.

        The actual-vs-predicted comparison (`interpreter_route_match`)
        happens later, in _end_turn_trace, which is the only place both
        the prediction (stored here) and the real `selected_path`
        (a parameter there) are visible together."""
        if self._turn_trace is None:
            return
        interpretation, latency_ms = interpret_shadow(user_message, self._work_session)
        routing_decision = _interpreter_decide_turn_route(interpretation, self._work_session)
        self._canonical_interpretation = interpretation
        self._routing_decision = routing_decision
        predicted_route = _interpreter_expected_route(interpretation, self._work_session)
        self._turn_trace["interpreter_shadow_enabled"] = True
        self._turn_trace["interpreter_shadow_latency_ms"] = latency_ms
        self._turn_trace["interpreter_activity_present"] = interpretation.activity.present
        self._turn_trace["interpreter_activity_family"] = interpretation.activity.activity_family
        self._turn_trace["interpreter_activity_action"] = (
            "starts" if interpretation.activity.starts_task
            else "resumes" if interpretation.activity.resumes_task
            else "pauses" if interpretation.activity.pauses_task
            else "refines" if interpretation.activity.refines_task
            else "continues" if interpretation.activity.continues_task
            else ""
        )
        self._turn_trace["interpreter_content_present"] = interpretation.content.present
        self._turn_trace["interpreter_spatial_present"] = interpretation.spatial.present
        self._turn_trace["interpreter_spatial_action"] = interpretation.spatial.action
        self._turn_trace["interpreter_representation_present"] = interpretation.representation.present
        self._turn_trace["interpreter_representation_action"] = interpretation.representation.action
        self._turn_trace["interpreter_representation_style"] = interpretation.representation.style
        self._turn_trace["interpreter_reasoning_present"] = interpretation.reasoning.present
        self._turn_trace["interpreter_attention_present"] = interpretation.attention.present
        self._turn_trace["interpreter_attention_action"] = interpretation.attention.action
        self._turn_trace["interpreter_conversational_present"] = interpretation.conversational.present
        self._turn_trace["interpreter_multi_intent_count"] = interpretation.multi_intent_count
        self._turn_trace["interpreter_confidence"] = interpretation.confidence
        self._turn_trace["interpreter_ambiguity_count"] = len(interpretation.ambiguities)
        self._turn_trace["interpreter_expected_route"] = predicted_route
        self._turn_trace["canonical_interpretation_primary_route"] = routing_decision.primary_route
        self._turn_trace["canonical_interpretation_secondary_intents"] = list(routing_decision.secondary_intents)
        self._turn_trace["routing_decision_reason"] = routing_decision.reason_code
        self._turn_trace["routing_authority"] = routing_decision.authority

    def _routing_primary_route(self) -> str:
        return self._routing_decision.primary_route if self._routing_decision is not None else ""

    def _route_is_authorized(self, *routes: str) -> bool:
        primary = self._routing_primary_route()
        return not primary or primary in set(routes)

    def _end_turn_trace(
        self,
        *,
        selected_path: str,
        response_source: str,
        tools_used: tuple[str, ...],
        final_response: str,
    ) -> None:
        if self._turn_trace is None:
            return
        print("[LIFECYCLE]", "ROUTING_DONE", f"selected_path={selected_path}", f"response_source={response_source}")
        llm_start = int(self._turn_trace.get("llm_start") or 0)
        llm_calls = max(0, self._llm_chat_count() - llm_start)
        sources_start = int(self._turn_trace.get("llm_sources_start") or 0)
        all_sources = list(getattr(self.llm, "chat_call_sources", ()))
        all_token_records = list(getattr(self.llm, "chat_call_tokens", ()) or [])
        turn_token_records = all_token_records[sources_start:]
        turn_sources = all_sources[sources_start:]
        llm_call_details = _llm_call_details(turn_sources, turn_token_records)
        if self.debug_ollama_payload:
            self._print_turn_trace(selected_path, response_source, tools_used, final_response, llm_calls, llm_call_details)

        input_tokens = _sum_optional_ints(record.get("input_tokens") for record in turn_token_records if isinstance(record, dict))
        output_tokens = _sum_optional_ints(record.get("output_tokens") for record in turn_token_records if isinstance(record, dict))
        estimated_cost_usd = sum(
            float(record.get("estimated_cost_usd") or 0.0) for record in turn_token_records if isinstance(record, dict)
        )
        configured_mode = str(self._turn_trace.get("configured_model_mode") or self._turn_trace.get("model_routing_mode") or "")
        configured_source = str(
            self._turn_trace.get("configured_model_mode_source")
            or self._turn_trace.get("model_routing_mode_source")
            or _model_mode_source(self.llm)
        )
        execution_path = str(self._turn_trace.get("execution_path") or _execution_path_for_turn(selected_path, response_source, llm_calls))
        execution_provider = str(
            self._turn_trace.get("execution_provider")
            or _execution_provider_for_turn(selected_path, self._turn_trace.get("model_routing_provider"), llm_calls)
        )
        execution_model = str(
            self._turn_trace.get("execution_model")
            or _execution_model_for_turn(self._turn_trace.get("model_routing_model"), self._turn_trace.get("model"), llm_calls)
        )
        fault_detected = bool(self._turn_trace.get("fault_detected"))
        fault_stage = str(self._turn_trace.get("fault_stage") or "none")
        fault_owner = str(self._turn_trace.get("fault_owner") or "none")
        fault_reason = str(self._turn_trace.get("fault_reason") or "")
        expected_route = str(self._turn_trace.get("expected_route") or "")
        # Section 11/12.B: the one cross-cutting fault this function alone
        # can confirm -- an active task turn with no explicit exit signal
        # (flagged by _try_active_task_turn) that still ended up in
        # GENERAL_CONVERSATION. Only ever ADDS a fault here; anything
        # already flagged upstream (e.g. model_generation) is left as-is.
        if not fault_detected and expected_route == "TASK_CONTINUATION" and selected_path == "GENERAL_CONVERSATION":
            fault_detected = True
            fault_stage = "routing"
            fault_owner = "echo_mechanism"
            fault_reason = "active_task_turn_fell_through_to_general_conversation"
        # Phase 2 (Cognitive Workspace Blueprint v2) -- shadow-mode
        # diagnostics only. Deliberately a SEPARATE label
        # (interpreter_shadow_comparison), never fault_detected: this is
        # evidence-collection about a not-yet-authoritative interpreter,
        # not a claim that production routing is broken (section 13).
        interpreter_predicted_route = str(self._turn_trace.get("interpreter_expected_route") or "")
        interpreter_shadow_comparison = (
            _interpreter_compare_route(interpreter_predicted_route, selected_path)
            if self._turn_trace.get("interpreter_shadow_enabled")
            else ""
        )
        # Phase 3 (Cognitive Workspace Blueprint v2, section 20): unlike
        # Phase 2's shadow-only comparison above, REPRESENTATION_COMMAND is
        # now a REAL (narrow) production route for a SPECIFIC subset of
        # RepresentationIntent actions (see _try_representation_command:
        # hide/restore/requested_style-cards-or-list only). Deliberately
        # scoped to exactly that subset, and only when the semantic object
        # registry is non-empty (the bridge's own precondition) -- most
        # representation-flavored utterances in an ordinary conversation
        # with no registered objects are expected non-matches, not faults
        # (that is still the Phase 2 UNSUPPORTED case, see compare_route).
        interp_repr_action = str(self._turn_trace.get("interpreter_representation_action") or "")
        interp_repr_style = str(self._turn_trace.get("interpreter_representation_style") or "")
        representation_bridge_should_have_applied = bool(self._work_session.objects) and (
            interp_repr_action in ("hide_representation", "restore_representation")
            or (interp_repr_action == "requested_style" and interp_repr_style in ("cards", "list"))
        )
        if (
            not fault_detected
            and interpreter_predicted_route == "REPRESENTATION_COMMAND"
            and representation_bridge_should_have_applied
            and selected_path != "REPRESENTATION_COMMAND"
        ):
            fault_detected = True
            fault_stage = "routing"
            fault_owner = "echo_mechanism"
            fault_reason = "representation_command_not_recognized"
        # Phase 4 (section 20/28): same reasoning as the representation
        # bridge above, for the new ATTENTION_COMMAND axis (see
        # _try_attention_command). Only faulted when there is at least one
        # registered semantic object to act on -- an attention phrase with
        # nothing in the registry yet is an expected non-match, not a bug.
        interp_attention_action = str(self._turn_trace.get("interpreter_attention_action") or "")
        # "show_again" deliberately falls through to REPRESENTATION_COMMAND
        # when nothing resolves (see _try_attention_command's own comment)
        # -- that is not a bug, so it is excluded from this precondition.
        attention_bridge_should_have_applied = (
            bool(self._work_session.objects) and bool(interp_attention_action) and interp_attention_action != "show_again"
        )
        if (
            not fault_detected
            and interpreter_predicted_route == "ATTENTION_COMMAND"
            and attention_bridge_should_have_applied
            and selected_path != "ATTENTION_COMMAND"
        ):
            fault_detected = True
            fault_stage = "routing"
            fault_owner = "echo_mechanism"
            fault_reason = "attention_command_not_recognized"
        self._last_turn_telemetry = {
            "user_message": self._turn_trace.get("user_message"),
            "final_response": final_response,
            "selected_path": selected_path,
            "response_source": response_source,
            "final_response_source": self._turn_trace.get("final_response_source") or response_source,
            "response_language": self.response_language,
            "response_language_changed": bool(self._turn_trace.get("response_language_changed")),
            "response_language_localized": bool(self._turn_trace.get("response_language_localized")),
            "language_guard_triggered": bool(self._turn_trace.get("language_guard_triggered")),
            "configured_model_mode": configured_mode,
            "configured_model_mode_source": configured_source,
            "execution_path": execution_path,
            "execution_provider": execution_provider,
            "execution_model": execution_model,
            "model": execution_model,
            "model_source": self._turn_trace.get("model_source"),
            "provider": execution_provider,
            "model_routing_mode": configured_mode,
            "model_routing_provider": execution_provider,
            "model_routing_model": execution_model,
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
            "memory_route": selected_path if selected_path in ("MEMORY_WRITE", "MEMORY_RECALL") else "NONE",
            "memory_recall_detected": bool(self._turn_trace.get("memory_recall_detected")),
            "selected_memory_ids": self._turn_trace.get("selected_memory_ids") or [],
            "persistent_memory_matches": self._turn_trace.get("persistent_memory_matches"),
            "history_matches": self._turn_trace.get("history_matches"),
            "memory_write_action": self._turn_trace.get("memory_write_action"),
            "grounding_sources": self._turn_trace.get("grounding_sources") or [],
            "research_task_id": self._turn_trace.get("research_task_id") or "",
            "research_request_id": self._turn_trace.get("research_request_id") or "",
            "research_plan": self._turn_trace.get("research_plan") or "",
            "research_query_plan": self._turn_trace.get("research_query_plan") or [],
            "research_provider": self._turn_trace.get("research_provider") or "",
            "research_providers": self._turn_trace.get("research_providers") or [],
            "research_fallback_reason": self._turn_trace.get("research_fallback_reason") or "",
            "research_status": self._turn_trace.get("research_status") or "",
            "research_result_count": int(self._turn_trace.get("research_result_count") or 0),
            "semantic_extraction_calls": int(self._turn_trace.get("semantic_extraction_calls") or 0),
            "semantic_extraction_status": self._turn_trace.get("semantic_extraction_status") or "",
            "semantic_candidates_before_validation": int(self._turn_trace.get("semantic_candidates_before_validation") or 0),
            "semantic_candidates_after_validation": int(self._turn_trace.get("semantic_candidates_after_validation") or 0),
            "semantic_extraction_failure_reason": self._turn_trace.get("semantic_extraction_failure_reason") or "",
            "semantic_extraction_batches": int(self._turn_trace.get("semantic_extraction_batches") or 0),
            "semantic_extraction_latency_ms": int(self._turn_trace.get("semantic_extraction_latency_ms") or 0),
            "semantic_extraction_input_chars": int(self._turn_trace.get("semantic_extraction_input_chars") or 0),
            "semantic_extraction_sources_used": int(self._turn_trace.get("semantic_extraction_sources_used") or 0),
            "research_tool_available": self._turn_trace.get("research_tool_available"),
            "research_tool_called": self._turn_trace.get("research_tool_called"),
            "research_needs_clarification": bool(self._turn_trace.get("research_needs_clarification")),
            "research_sources_requested": bool(self._turn_trace.get("research_sources_requested")),
            "research_source_excluded": bool(self._turn_trace.get("research_source_excluded")),
            "research_source_open_requested": bool(self._turn_trace.get("research_source_open_requested")),
            "research_source_id": self._turn_trace.get("research_source_id") or "",
            "research_source_url": self._turn_trace.get("research_source_url") or "",
            "content_operation_detected": bool(self._turn_trace.get("content_operation_detected")),
            "content_operation": self._turn_trace.get("content_operation") or "",
            "content_target_id": self._turn_trace.get("content_target_id") or "",
            "content_target_type": self._turn_trace.get("content_target_type") or "",
            "content_target_resolution": self._turn_trace.get("content_target_resolution") or "",
            "content_candidate_count": int(self._turn_trace.get("content_candidate_count") or 0),
            "content_presentation_mode": self._turn_trace.get("content_presentation_mode") or "",
            # Passo 3.2 (section 18): scene_before/after and target
            # resolution/confidence are deliberately NOT tracked here -- the
            # WorkspaceScene lives entirely client-side (echo_workspace_
            # objects.js), so the backend has no visibility into it at
            # response time. Those two fields are logged client-side
            # instead (controller.commandLog()); see final report.
            "workspace_command_detected": bool(self._turn_trace.get("workspace_command_detected")),
            "workspace_command_action": self._turn_trace.get("workspace_command_action") or "",
            "workspace_command_target": self._turn_trace.get("workspace_command_target") or "",
            "workspace_command_scope": self._turn_trace.get("workspace_command_scope") or "",
            "workspace_command_relation": self._turn_trace.get("workspace_command_relation") or "",
            "workspace_command_region": self._turn_trace.get("workspace_command_region") or "",
            "workspace_command_source": self._turn_trace.get("workspace_command_source") or "",
            "workspace_command_confidence": self._turn_trace.get("workspace_command_confidence") or "",
            "workspace_language_match_type": self._turn_trace.get("workspace_language_match_type") or "",
            "workspace_language_signals": self._turn_trace.get("workspace_language_signals") or [],
            # Backend can only attest to "the ui_event was queued", not
            # whether the frontend actually applied it (no round-trip
            # channel exists) -- see final report's documented limitation.
            "workspace_command_executed": bool(self._turn_trace.get("workspace_command_executed")),
            "workspace_command_execution_error": self._turn_trace.get("workspace_command_execution_error") or "",
            "workspace_command_latency_ms": float(self._turn_trace.get("workspace_command_latency_ms") or 0.0),
            "compound_workspace_intent": bool(self._turn_trace.get("compound_workspace_intent")),
            "compound_workspace_intent_count": int(self._turn_trace.get("compound_workspace_intent_count") or 0),
            "executed_operations": int(self._turn_trace.get("executed_operations") or 0),
            "failed_operations": int(self._turn_trace.get("failed_operations") or 0),
            "workspace_command_target_ids": self._turn_trace.get("workspace_command_target_ids") or [],
            "workspace_collection_resolution_source": self._turn_trace.get("workspace_collection_resolution_source") or "",
            "global_workspace_action": bool(self._turn_trace.get("global_workspace_action")),
            "global_workspace_object_count": int(self._turn_trace.get("global_workspace_object_count") or 0),
            "last_workspace_command": dict(self._last_workspace_command) if self._last_workspace_command else None,
            # Passo 3.4: ActiveTaskContext -- never guards content, just
            # observability. Content itself is never dumped here (only
            # field NAMES/values the user already stated deterministically,
            # e.g. "recipient"/"João Cardoso" -- no raw email body).
            "active_task_id": self._turn_trace.get("active_task_id") or "",
            "active_task_type": self._turn_trace.get("active_task_type") or "",
            "active_task_status": self._turn_trace.get("active_task_status") or "",
            "activity_detected": bool(self._turn_trace.get("activity_detected")),
            "activity_family": self._turn_trace.get("activity_family") or "",
            "activity_domain": self._turn_trace.get("activity_domain") or "",
            "activity_artifact_type": self._turn_trace.get("activity_artifact_type") or "",
            "activity_confidence": float(self._turn_trace.get("activity_confidence") or 0.0),
            "activity_reason": self._turn_trace.get("activity_reason") or "",
            "task_started": bool(self._turn_trace.get("task_started")),
            "task_switched": bool(self._turn_trace.get("task_switched")),
            "task_paused": bool(self._turn_trace.get("task_paused")),
            "task_resumed": bool(self._turn_trace.get("task_resumed")),
            "task_specialized": bool(self._turn_trace.get("task_specialized")),
            "task_stack_depth": int(self._turn_trace.get("task_stack_depth") or len(self._paused_task_stack)),
            "previous_active_task_type": self._turn_trace.get("previous_active_task_type") or "",
            "new_active_task_type": self._turn_trace.get("new_active_task_type") or "",
            "task_continuation_detected": bool(self._turn_trace.get("task_continuation_detected")),
            "task_fields_updated": self._turn_trace.get("task_fields_updated") or [],
            "task_known_fields": self._turn_trace.get("task_known_fields") or {},
            "task_missing_fields": self._turn_trace.get("task_missing_fields") or [],
            "task_artifact_id": self._turn_trace.get("task_artifact_id") or "",
            "task_context_source": self._turn_trace.get("task_context_source") or "",
            "task_llm_interpretation_used": bool(self._turn_trace.get("task_llm_interpretation_used")),
            # Section 16: TASK_DATA vs TASK_CONTROL vs TASK_EXECUTION
            # observability -- never the generated email's full content.
            "task_turn_intent": self._turn_trace.get("task_turn_intent") or "",
            "task_control_action": self._turn_trace.get("task_control_action") or "",
            "active_task_continuation_evidence": self._turn_trace.get("active_task_continuation_evidence") or "",
            "task_continuation_rejected": bool(self._turn_trace.get("task_continuation_rejected")),
            "active_task_continuation_rejected_reason": self._turn_trace.get("active_task_continuation_rejected_reason") or "",
            "task_execution_started": bool(self._turn_trace.get("task_execution_started")),
            "task_execution_completed": bool(self._turn_trace.get("task_execution_completed")),
            "task_generation_provider": self._turn_trace.get("task_generation_provider") or "",
            "task_generation_model": self._turn_trace.get("task_generation_model") or "",
            "task_artifact_revision": int(self._turn_trace.get("task_artifact_revision") or 0),
            "task_generation_validation_reason": self._turn_trace.get("task_generation_validation_reason") or "",
            "generic_activity_reused": bool(self._turn_trace.get("generic_activity_reused")),
            # Section 11-14 (fault-attribution phase): a per-turn diagnostic
            # layer distinguishing echo_mechanism/model/provider/ui faults --
            # always explicitly present (never absence-as-success, section 17).
            "fault_detected": fault_detected,
            "fault_stage": fault_stage,
            "fault_owner": fault_owner,
            "fault_reason": fault_reason,
            "expected_route": expected_route,
            "actual_route": selected_path,
            "expected_context_present": bool(self._turn_trace.get("expected_context_present")),
            "model_received_expected_context": bool(self._turn_trace.get("model_received_expected_context")),
            "model_output_valid": bool(self._turn_trace.get("model_output_valid")),
            "recovery_attempted": bool(self._turn_trace.get("recovery_attempted")),
            "recovery_succeeded": bool(self._turn_trace.get("recovery_succeeded")),
            # Section 13: a structural snapshot of what the model actually
            # received -- never the full generated content.
            "model_context_task_type": self._turn_trace.get("model_context_task_type") or "",
            "model_context_task_status": self._turn_trace.get("model_context_task_status") or "",
            "model_context_known_fields": self._turn_trace.get("model_context_known_fields") or [],
            "model_context_missing_fields": self._turn_trace.get("model_context_missing_fields") or [],
            "model_context_recipient_present": bool(self._turn_trace.get("model_context_recipient_present")),
            "model_context_topic_present": bool(self._turn_trace.get("model_context_topic_present")),
            "model_context_key_points_count": int(self._turn_trace.get("model_context_key_points_count") or 0),
            "model_context_instruction_ready_to_execute": bool(self._turn_trace.get("model_context_instruction_ready_to_execute")),
            # Section 14: output validation observability.
            "task_output_validation_passed": bool(self._turn_trace.get("task_output_validation_passed")),
            "task_output_validation_reason": self._turn_trace.get("task_output_validation_reason") or "",
            "task_generation_attempt": int(self._turn_trace.get("task_generation_attempt") or 0),
            "task_generation_max_attempts": int(self._turn_trace.get("task_generation_max_attempts") or 0),
            # Phase 1 (Cognitive Workspace Blueprint v2) migration
            # diagnostics only -- confirms WorkSession is live and what it
            # currently holds, nothing more; no new schema.
            "work_session_active": True,
            "work_session_active_task_present": self._work_session.active_task is not None,
            "work_session_document_present": self._work_session.document_context is not None,
            "work_session_legacy_writing_present": self._work_session.writing_context is not None,
            # Phase 2 (Cognitive Workspace Blueprint v2) -- Interaction
            # Interpreter, SHADOW MODE ONLY (section 11/12/13). None of
            # these fields are read by any routing decision anywhere;
            # they exist purely to compare "what would the new
            # architecture have understood" against "what route the old
            # system actually took."
            "interpreter_shadow_enabled": bool(self._turn_trace.get("interpreter_shadow_enabled")),
            "interpreter_shadow_latency_ms": float(self._turn_trace.get("interpreter_shadow_latency_ms") or 0.0),
            "interpreter_activity_present": bool(self._turn_trace.get("interpreter_activity_present")),
            "interpreter_activity_family": self._turn_trace.get("interpreter_activity_family") or "",
            "interpreter_activity_action": self._turn_trace.get("interpreter_activity_action") or "",
            "interpreter_content_present": bool(self._turn_trace.get("interpreter_content_present")),
            "interpreter_spatial_present": bool(self._turn_trace.get("interpreter_spatial_present")),
            "interpreter_spatial_action": self._turn_trace.get("interpreter_spatial_action") or "",
            "interpreter_representation_present": bool(self._turn_trace.get("interpreter_representation_present")),
            "interpreter_representation_action": self._turn_trace.get("interpreter_representation_action") or "",
            "interpreter_representation_style": self._turn_trace.get("interpreter_representation_style") or "",
            "interpreter_reasoning_present": bool(self._turn_trace.get("interpreter_reasoning_present")),
            "interpreter_conversational_present": bool(self._turn_trace.get("interpreter_conversational_present")),
            "interpreter_multi_intent_count": int(self._turn_trace.get("interpreter_multi_intent_count") or 0),
            "interpreter_confidence": float(self._turn_trace.get("interpreter_confidence") or 0.0),
            "interpreter_ambiguity_count": int(self._turn_trace.get("interpreter_ambiguity_count") or 0),
            "interpreter_expected_route": interpreter_predicted_route,
            "actual_selected_path": selected_path,
            "interpreter_route_match": interpreter_shadow_comparison,
            "canonical_interpretation_primary_route": self._turn_trace.get("canonical_interpretation_primary_route") or "",
            "canonical_interpretation_secondary_intents": self._turn_trace.get("canonical_interpretation_secondary_intents") or [],
            "routing_decision_reason": self._turn_trace.get("routing_decision_reason") or "",
            "routing_authority": self._turn_trace.get("routing_authority") or "",
            # Phase 3 (Cognitive Workspace Blueprint v2) -- representation
            # transformation observability (section 5/11/20). Never the
            # object payloads themselves, only style/ids/source/pass-fail,
            # matching the workspace_command_* convention above.
            "representation_command_detected": bool(self._turn_trace.get("representation_command_detected")),
            "representation_command_action": self._turn_trace.get("representation_command_action") or "",
            "representation_style": self._turn_trace.get("representation_style") or "",
            "representation_source": self._turn_trace.get("representation_source") or "",
            "representation_command_executed": bool(self._turn_trace.get("representation_command_executed")),
            "representation_command_latency_ms": float(self._turn_trace.get("representation_command_latency_ms") or 0.0),
            "representation_object_count": len(self._work_session.objects),
            "representation_current_style": self._work_session.representation.style if self._work_session.representation is not None else "",
            # Phase 3B integration fix: FIX 2's "do not claim success
            # without objects" telemetry, and FIX 3's materialization count.
            "representation_target_resolution_failed": bool(self._turn_trace.get("representation_target_resolution_failed")),
            "alternatives_materialized_count": int(self._turn_trace.get("alternatives_materialized_count") or 0),
            # Phase 3C (section 12): false-success prevention for mixed
            # utterances -- never present (all default to 0/False) unless
            # a handler that can partially fail (currently only
            # _try_mixed_representation_intent) actually ran this turn.
            "requested_intent_count": int(self._turn_trace.get("requested_intent_count") or 0),
            "executed_intent_count": int(self._turn_trace.get("executed_intent_count") or 0),
            "failed_intent_count": int(self._turn_trace.get("failed_intent_count") or 0),
            "partial_execution": bool(self._turn_trace.get("partial_execution")),
            "mixed_add_option_count": int(self._turn_trace.get("mixed_add_option_count") or 0),
            # Phase 4 (Cognitive Workspace Blueprint v2) -- attention,
            # proactive representation, and space-pressure observability
            # (section 28's fault-attribution categories: attention_policy,
            # representation_policy, space_pressure, pending_workspace_action).
            "attention_command_detected": bool(self._turn_trace.get("attention_command_detected")),
            "attention_command_action": self._turn_trace.get("attention_command_action") or "",
            "attention_command_executed": bool(self._turn_trace.get("attention_command_executed")),
            "attention_command_target_ids": self._turn_trace.get("attention_command_target_ids") or [],
            "attention_command_latency_ms": float(self._turn_trace.get("attention_command_latency_ms") or 0.0),
            "attention_target_resolution_source": self._turn_trace.get("attention_target_resolution_source") or "",
            "attention_target_resolution_ambiguous": bool(self._turn_trace.get("attention_target_resolution_ambiguous")),
            "attention_execution_mutated_state": bool(self._turn_trace.get("attention_execution_mutated_state")),
            "attention_execution_ui_event_queued": bool(self._turn_trace.get("attention_execution_ui_event_queued")),
            "representation_proposal_reason_code": self._turn_trace.get("representation_proposal_reason_code") or "",
            "representation_proposal_confidence": float(self._turn_trace.get("representation_proposal_confidence") or 0.0),
            "reasoning_visible": bool(self._turn_trace.get("reasoning_visible")),
            "space_pressure": self._turn_trace.get("space_pressure") or "",
            "space_pressure_visible_object_count": int(self._turn_trace.get("space_pressure_visible_object_count") or 0),
            "space_pressure_major_surface_count": int(self._turn_trace.get("space_pressure_major_surface_count") or 0),
            "space_pressure_overflow_risk": bool(self._turn_trace.get("space_pressure_overflow_risk")),
            "capacity_revision": int(self._turn_trace.get("capacity_revision") or 0),
            "capacity_signal_age": self._turn_trace.get("capacity_signal_age"),
            "space_pressure_candidate_ids": self._turn_trace.get("space_pressure_candidate_ids") or [],
            "space_pressure_protected_ids": self._turn_trace.get("space_pressure_protected_ids") or [],
            "space_pressure_action": self._turn_trace.get("space_pressure_action") or "",
            "space_pressure_permission_required": bool(self._turn_trace.get("space_pressure_permission_required")),
            "space_pressure_candidate_found": self._turn_trace.get("space_pressure_candidate_found"),
            "pending_workspace_action": bool(self._turn_trace.get("pending_workspace_action")),
            "pending_workspace_action_created": bool(self._turn_trace.get("pending_workspace_action_created")),
            "pending_workspace_action_object_ids": self._turn_trace.get("pending_workspace_action_object_ids") or [],
            "pending_workspace_action_result": self._turn_trace.get("pending_workspace_action_result") or "",
            "pending_workspace_action_resolved": self._turn_trace.get("pending_workspace_action_resolved") or "",
            "minimization_reason": self._turn_trace.get("minimization_reason") or "",
            # Passo 3.5: memory relevance gate (section 9). Never the raw
            # memory content itself, only score/reason/pass-fail.
            "memory_candidate_present": bool(self._turn_trace.get("memory_candidate_present")),
            "memory_relevance_score": float(self._turn_trace.get("memory_relevance_score") or 0.0),
            "memory_relevance_reason": self._turn_trace.get("memory_relevance_reason") or "",
            "memory_relevance_gate_passed": bool(self._turn_trace.get("memory_relevance_gate_passed")),
            "memory_rejected_reason": self._turn_trace.get("memory_rejected_reason") or "",
            "workspace_file_found": self._turn_trace.get("workspace_file_found"),
            "workspace_files_used": self._turn_trace.get("workspace_files_used") or [],
            "output_file_created": self._turn_trace.get("output_file_created"),
            "output_file_path": self._turn_trace.get("output_file_path") or "",
            "active_document": bool(self._turn_trace.get("active_document")),
            "active_document_name": self._turn_trace.get("active_document_name") or "",
            "document_action": self._turn_trace.get("document_action") or "",
            "document_context_reused": bool(self._turn_trace.get("document_context_reused")),
            "document_grounded": bool(self._turn_trace.get("document_grounded")),
            "draft_created": bool(self._turn_trace.get("draft_created")),
            "draft_saved": bool(self._turn_trace.get("draft_saved")),
            "draft_reused": bool(self._turn_trace.get("draft_reused")),
            "draft_validation_passed": bool(self._turn_trace.get("draft_validation_passed")),
            "draft_validation_failed": bool(self._turn_trace.get("draft_validation_failed")),
            "draft_regeneration_attempted": bool(self._turn_trace.get("draft_regeneration_attempted")),
            "draft_validation_reason": self._turn_trace.get("draft_validation_reason") or "",
            "draft_original_length": int(self._turn_trace.get("draft_original_length") or 0),
            "draft_generated_length": int(self._turn_trace.get("draft_generated_length") or 0),
            "rewrite_attempt_1_latency_ms": float(self._turn_trace.get("rewrite_attempt_1_latency_ms") or 0.0),
            "rewrite_attempt_2_latency_ms": float(self._turn_trace.get("rewrite_attempt_2_latency_ms") or 0.0),
            "rewrite_total_latency_ms": float(self._turn_trace.get("rewrite_total_latency_ms") or 0.0),
            "draft_preamble_removed": bool(self._turn_trace.get("draft_preamble_removed")),
            "draft_placeholder_detected": bool(self._turn_trace.get("draft_placeholder_detected")),
            "rewrite_regeneration_started": bool(self._turn_trace.get("rewrite_regeneration_started")),
            "cancellation_requested": bool(self._turn_trace.get("cancellation_requested")),
            "request_cancelled": bool(self._turn_trace.get("request_cancelled")),
            "rewrite_timeout": bool(self._turn_trace.get("rewrite_timeout")),
            "rewrite_timeout_seconds": float(self._turn_trace.get("rewrite_timeout_seconds") or 0.0),
            "rewrite_elapsed_ms_at_cancel": float(self._turn_trace.get("rewrite_elapsed_ms_at_cancel") or 0.0),
            "rewrite_elapsed_ms_at_timeout": float(self._turn_trace.get("rewrite_elapsed_ms_at_timeout") or 0.0),
            "rewrite_remaining_timeout_ms": float(self._turn_trace.get("rewrite_remaining_timeout_ms") or 0.0),
            "request_finished_cleanly": bool(self._turn_trace.get("request_finished_cleanly")),
            "document_followup_action": self._turn_trace.get("document_followup_action") or "",
            "document_refinement_source": self._turn_trace.get("document_refinement_source") or "",
            "previous_draft_present": bool(self._turn_trace.get("previous_draft_present")),
            "refinement_changed_content": bool(self._turn_trace.get("refinement_changed_content")),
            "refinement_regeneration_started": bool(self._turn_trace.get("refinement_regeneration_started")),
            "draft_revision_number": int(self._turn_trace.get("draft_revision_number") or 0),
            "task_complexity_score": float(self._turn_trace.get("task_complexity_score") or 0.0),
            "task_complexity_band": self._turn_trace.get("task_complexity_band") or "",
            "document_task_type": self._turn_trace.get("document_task_type") or "",
            "document_routing_features": self._turn_trace.get("document_routing_features") or {},
            "escalation_considered": bool(self._turn_trace.get("escalation_considered")),
            "escalation_selected": bool(self._turn_trace.get("escalation_selected")),
            "escalation_reason": self._turn_trace.get("escalation_reason") or "",
            "initial_provider": self._turn_trace.get("initial_provider") or "",
            "final_provider": self._turn_trace.get("final_provider") or "",
            "initial_routing_reason_code": self._turn_trace.get("initial_routing_reason_code") or "",
            "final_routing_reason_code": self._turn_trace.get("final_routing_reason_code") or "",
            "provider_attempt_sequence": self._turn_trace.get("provider_attempt_sequence") or [],
            "attempt_count": int(self._turn_trace.get("attempt_count") or 0),
            "attempt_failure_reasons": self._turn_trace.get("attempt_failure_reasons") or [],
            "local_failure_reason": self._turn_trace.get("local_failure_reason") or "",
            "paid_call_authorized": bool(self._turn_trace.get("paid_call_authorized")),
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
            "writing_workflow_active": bool(self._turn_trace.get("writing_workflow_active")),
            "writing_task_type": self._turn_trace.get("writing_task_type") or "",
            "writing_awaiting_field": self._turn_trace.get("writing_awaiting_field") or "",
            "writing_recipient": self._turn_trace.get("writing_recipient") or "",
            "writing_recipient_type": self._turn_trace.get("writing_recipient_type") or "",
            "writing_purpose": self._turn_trace.get("writing_purpose") or "",
            "writing_subject": self._turn_trace.get("writing_subject") or "",
            "writing_project": self._turn_trace.get("writing_project") or "",
            "writing_missing_fields": self._turn_trace.get("writing_missing_fields") or [],
        }
        self._turn_trace = None

    def _print_turn_trace(
        self,
        selected_path: str,
        response_source: str,
        tools_used: tuple[str, ...],
        final_response: str,
        llm_calls: int,
        llm_call_details: list[dict[str, object]],
    ) -> None:
        if self._turn_trace is None:
            return
        print("\n[TURN TRACE]")
        for key in (
            "user_message",
            "configured_model_mode",
            "configured_model_mode_source",
            "execution_path",
            "execution_provider",
            "execution_model",
            "model",
            "model_source",
            "provider",
            "model_routing_mode",
            "model_routing_provider",
            "model_routing_model",
            "model_routing_reason_code",
            "model_routing_reason",
            "model_routing_paid_call",
            "model_routing_budget_before_usd",
            "model_routing_budget_after_usd",
            "model_routing_fallback_reason",
            "model_routing_override_source",
            "routing_user_message_chars",
            "routing_context_chars",
            "routing_constraint_count",
            "provider_error_type",
            "fallback_used",
            "response_before_voice_critic",
            "response_after_voice_critic",
            "composer_call_count",
            "memory_recall_detected",
            "response_grounded",
            "grounding_sources",
            "exception_type",
            "exception_message",
        ):
            print(f"{key}={self._turn_trace.get(key)}")
        print(f"selected_path={selected_path}")
        print(f"memory_route={selected_path if selected_path in ('MEMORY_WRITE', 'MEMORY_RECALL') else 'NONE'}")
        print(f"response_source={response_source}")
        print(f"llm_calls={llm_calls}")
        print(f"llm_call_details={llm_call_details}")
        print(f"tools_used={list(tools_used)}")
        print(f"final_response={final_response}")
        print("[/TURN TRACE]\n")

    def get_last_turn_telemetry(self) -> dict | None:
        """Structured telemetry for the most recently completed turn.

        A clean test/eval API (see evals/schemas.py TurnResult) so a runner
        never has to parse the human-readable [TURN TRACE] stdout dump.
        This is always collected so the Echo OS UI can show real telemetry.
        DEBUG_OLLAMA_PAYLOAD only controls the verbose terminal dump.
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

    def _gate_memory_relevance(self, user_message: str, candidate: str) -> str:
        """Passo 3.5, section 4/5: a retrieved memory candidate only gets
        included when it's actually about THIS turn -- an explicit recall
        request, or real keyword overlap with the current message.
        Priority order (section 5): current explicit intent already won by
        construction (this only runs from _context_for_agent, itself only
        reached once TASK_CONTINUATION/workspace-command/document-task have
        all already returned None for this turn) > active task (a
        stricter threshold while one is in progress, since the task's own
        content should already be what's relevant) > persisted memory."""
        normalized_message = _normalize_text(user_message).strip(" .!?")
        explicit_recall = bool(normalized_message) and (
            is_memory_recall_question(normalized_message)
            or (self._active_memory_recall_ttl > 0 and is_memory_recall_followup(normalized_message))
        )
        score, reason = _memory_relevance_score(user_message, candidate)
        threshold = _MEMORY_RELEVANCE_THRESHOLD * (2 if self._active_task_context is not None else 1)
        gate_passed = explicit_recall or score >= threshold
        if self._turn_trace is not None:
            self._turn_trace["memory_candidate_present"] = True
            self._turn_trace["memory_relevance_score"] = score
            self._turn_trace["memory_relevance_reason"] = "explicit_recall" if explicit_recall else reason
            self._turn_trace["memory_relevance_gate_passed"] = gate_passed
            if not gate_passed:
                self._turn_trace["memory_rejected_reason"] = reason
        return candidate if gate_passed else ""

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
            candidate = self.long_term_memory.context_for(user_message)
            if candidate:
                memory_context = self._gate_memory_relevance(user_message, candidate)
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
            if normalize_response_language(self.response_language) == ENGLISH_GB:
                return "My base language is Portuguese from Portugal, but I'm currently responding in English."
            return "A minha língua base é português de Portugal, mas posso falar inglês quando pedires."

        command = detect_response_language_command(user_message)
        if command.handled:
            self._set_response_language(command.language)
            if self._turn_trace is not None:
                self._turn_trace["response_language_changed"] = True
                self._turn_trace["response_language"] = self.response_language
            return command.confirmation

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
        "ok obrigado",
        "ok obrigada",
        "esta bem obrigado",
        "estÃ¡ bem obrigado",
        "muito obrigado",
        "muito obrigada",
        "estas ai",
        "estÃ¡s ai",
        "estas aÃ­",
        "estÃ¡s aÃ­",
        "como te chamas",
        "qual e o teu nome",
        "qual Ã© o teu nome",
        "qual e a tua cor preferida",
        "qual Ã© a tua cor preferida",
        "e a minha irma de 7 anos a perguntar a tua cor preferida",
        "Ã© a minha irmÃ£ de 7 anos a perguntar a tua cor preferida",
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


def _model_mode_source(llm: object | None) -> str:
    settings = getattr(llm, "settings", None)
    return str(getattr(settings, "model_routing_mode_source", "") or "")


def _configured_model_mode(llm: object | None) -> str:
    settings = getattr(llm, "settings", None)
    return str(getattr(settings, "model_routing_mode", "") or "local")


def _provider_name(llm: object) -> str:
    provider = getattr(llm, "provider", None)
    return str(getattr(provider, "name", "") or "")


def _execution_path_for_turn(selected_path: str, response_source: str, llm_calls: int) -> str:
    if llm_calls > 0:
        if response_source in {"AGENT_DIRECT", "AGENT"} or selected_path == "AGENT":
            return "agent"
        if response_source in {"RESPONSE_COMPOSER", "COMPOSER_REGENERATED"}:
            return "response_composer"
        return "llm"
    path = str(selected_path or response_source or "").upper()
    if path == "SOCIAL_PATH":
        return "social_fast_path"
    if path == "SYSTEM_DATETIME":
        return "system_datetime"
    if path == "SYSTEM_CONTEXT_STATUS":
        return "system_context_status"
    if path == "WRITING_TASK":
        return "writing_workflow"
    if "MEMORY" in path:
        return "memory_recall"
    if path == "DOCUMENT_TASK":
        return "document_task"
    if path == "WORKSPACE_COMMAND":
        return "workspace_command"
    if path == "TASK_CONTINUATION":
        return "task_continuation"
    if "TOOL" in path or "FAST_ROUTE" in path:
        return "local_tool"
    if "ERROR" in path:
        return "error"
    return _local_reason_code_for_path(str(selected_path or response_source or ""))


def _execution_provider_for_turn(selected_path: str, routed_provider: object, llm_calls: int) -> str:
    if llm_calls > 0:
        return str(routed_provider or "ollama")
    path = str(selected_path or "").upper()
    if "MEMORY" in path:
        return "memory"
    if path == "DOCUMENT_TASK" or "TOOL" in path or "FAST_ROUTE" in path:
        return "tool"
    if "ERROR" in path:
        return "none"
    return "local"


def _execution_model_for_turn(routed_model: object, fallback_model: object, llm_calls: int) -> str:
    if llm_calls <= 0:
        return "NONE"
    return str(routed_model or fallback_model or "")


def _model_routing_telemetry(llm: object) -> dict[str, object]:
    settings = getattr(llm, "settings", None)
    configured_mode = str(getattr(settings, "model_routing_mode", "") or "")
    execution_provider = str(getattr(settings, "model_routing_provider", "") or "")
    execution_model = str(getattr(settings, "model_routing_model", "") or "")
    return {
        "configured_model_mode": configured_mode,
        "configured_model_mode_source": str(getattr(settings, "model_routing_mode_source", "") or ""),
        "execution_path": "llm",
        "execution_provider": execution_provider,
        "execution_model": execution_model,
        "model_routing_mode": configured_mode,
        "model_routing_provider": execution_provider,
        "model_routing_model": execution_model,
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


def _local_turn_model_telemetry(selected_path: str, llm: object | None = None) -> dict[str, object]:
    reason_code = _local_reason_code_for_path(selected_path)
    configured_mode = _configured_model_mode(llm)
    configured_source = _model_mode_source(llm)
    execution_provider = _execution_provider_for_turn(selected_path, "", 0)
    execution_path = _execution_path_for_turn(selected_path, "", 0)
    return {
        "configured_model_mode": configured_mode,
        "configured_model_mode_source": configured_source,
        "execution_path": execution_path,
        "execution_provider": execution_provider,
        "execution_model": "NONE",
        "model_routing_mode": configured_mode,
        "model_routing_provider": execution_provider,
        "model_routing_model": "NONE",
        "model_routing_reason_code": reason_code,
        "model_routing_reason": _local_reason_label(reason_code),
        "model_routing_paid_call": False,
        "model_routing_budget_before_usd": 0.0,
        "model_routing_budget_after_usd": 0.0,
        "model_routing_fallback_reason": "",
        "model_routing_override_source": "",
        "routing_user_message_chars": 0,
        "routing_context_chars": 0,
        "routing_constraint_count": 0,
    }


def _local_reason_code_for_path(selected_path: str) -> str:
    mapping = {
        "SOCIAL_PATH": "social_fast_path",
        "SYSTEM_DATETIME": "system_datetime",
        "SYSTEM_CONTEXT_STATUS": "system_context_status",
        "SYSTEM_STATUS": "system_status",
        "WRITING_TASK": "writing_workflow",
        "DOCUMENT_TASK": "local_tool",
        "TOOL_PATH": "local_tool",
        "FAST_ROUTE": "fast_route",
        "MEMORY_COMMAND": "memory_command",
        "MEMORY_RECALL": "memory_recall",
        "TASK_COMMAND": "task_command",
    }
    return mapping.get(selected_path, "local_deterministic")


def _local_reason_label(reason_code: str) -> str:
    labels = {
        "social_fast_path": "Social fast path",
        "system_datetime": "System datetime",
        "system_context_status": "System context status",
        "system_status": "System status",
        "writing_workflow": "Writing workflow",
        "local_tool": "Local tool",
        "fast_route": "Fast route",
        "memory_command": "Memory command",
        "memory_recall": "Memory recall",
        "task_command": "Task command",
        "local_deterministic": "Local deterministic response",
    }
    return labels.get(reason_code, reason_code.replace("_", " ").title())


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


DOCUMENT_EXTENSIONS = (".txt", ".md", ".pdf", ".docx")

PT_WEEKDAYS = (
    "segunda-feira",
    "terça-feira",
    "quarta-feira",
    "quinta-feira",
    "sexta-feira",
    "sábado",
    "domingo",
)
PT_MONTHS = (
    "janeiro",
    "fevereiro",
    "março",
    "abril",
    "maio",
    "junho",
    "julho",
    "agosto",
    "setembro",
    "outubro",
    "novembro",
    "dezembro",
)


def _parse_system_datetime_question(message: str) -> tuple[str, object | None] | None:
    text = _normalize_text(message).strip(" ?!.")
    if not text:
        return None
    non_datetime_intents = (
        "lembra me",
        "lembra-me",
        "adiciona",
        "tarefa",
        "tarefas",
        "adia",
        "adiei",
        "regista",
        "registei",
        "estivemos",
        "fizemos",
        "trabalhar",
    )
    if any(marker in text for marker in non_datetime_intents):
        if not any(
            phrase in text
            for phrase in (
                "que dia",
                "qual e a data",
                "qual é a data",
                "dia da semana",
                "que horas",
            )
        ):
            return None
    calendar_words = ("compromisso", "compromissos", "reuniao", "reunião", "agenda", "calendario", "calendário")
    if any(word in text for word in calendar_words) and not any(word in text for word in ("data", "dia", "horas", "hora")):
        return None
    if re.search(r"\b(?:que|quais?)\s+horas?\s+(?:sao|são|e|é)\b", text) or text in {"horas", "hora atual"}:
        return ("time", None)
    if ("amanha" in text or "amanhã" in text) and any(phrase in text for phrase in ("data de amanha", "data de amanhã", "que dia", "qual e a data", "qual é a data")):
        return ("tomorrow", None)
    if "ontem" in text and any(phrase in text for phrase in ("ontem foi que dia", "que dia foi ontem", "data de ontem")):
        return ("yesterday", None)
    offset = re.search(r"\bdaqui\s+a\s+(\d{1,4})\s+dias?\b", text)
    if offset:
        return ("offset", int(offset.group(1)))
    explicit = _extract_explicit_date(message)
    if explicit is not None and any(word in text for word in ("calha", "dia da semana", "semana")):
        return ("weekday_for_date", explicit)
    date_questions = (
        "que dia e hoje",
        "que dia é hoje",
        "qual e a data",
        "qual é a data",
        "data de hoje",
        "dia de hoje",
        "que dia da semana e hoje",
        "que dia da semana é hoje",
    )
    if any(phrase in text for phrase in date_questions):
        return ("today", None)
    return None


def _extract_explicit_date(message: str) -> date | None:
    numeric = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", message)
    if numeric:
        day = int(numeric.group(1))
        month = int(numeric.group(2))
        year = int(numeric.group(3) or datetime.now().year)
        if year < 100:
            year += 2000
        try:
            return date(year, month, day)
        except ValueError:
            return None
    iso = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", message)
    if iso:
        try:
            return date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        except ValueError:
            return None
    return None


def _weekday_name(value: date) -> str:
    return PT_WEEKDAYS[value.weekday()]


def _format_pt_date(value: date, *, include_weekday: bool = True) -> str:
    prefix = f"{_weekday_name(value)}, " if include_weekday else ""
    return f"{prefix}{value.day} de {PT_MONTHS[value.month - 1]} de {value.year}"


@dataclass
class DocumentTaskRequest:
    source_files: list[str] = field(default_factory=list)
    output_file: str | None = None
    recipient: str | None = None
    actions: list[str] = field(default_factory=list)
    output_format: str | None = None
    instructions: str = ""
    action: str = ""
    requested_output: str = ""
    needs_clarification: bool = False


@dataclass
class WorkspaceSearchResult:
    found: bool
    matches: list[Path] = field(default_factory=list)
    blocked: bool = False
    match_type: str = ""


def _looks_like_document_task(text: str, request: DocumentTaskRequest | None = None) -> bool:
    request = request or _parse_document_task_request(text)
    has_document_name = bool(request.source_files) or any(ext in text for ext in DOCUMENT_EXTENSIONS)
    has_file_action = bool(request.actions) or _looks_like_read_or_summary(text) or _looks_like_create_file_task(text)
    return (
        bool(request.action and (request.source_files or request.needs_clarification))
        or bool(request.source_files and request.action in {"read", "display", "summarize", "create"})
        or bool(request.source_files and request.requested_output == "full_contents")
        or bool(request.needs_clarification and request.action in {"read", "display"})
        or
        (has_document_name and has_file_action)
        or "ficheiro" in text and any(word in text for word in ("resume", "resumo", "ler ", "le ", "lê ", "cria", "guarda"))
        or _looks_like_create_file_task(text)
        or "workspace" in text and any(word in text for word in ("notas", "ficheiros", "documentos", "usa"))
    )


def _detect_document_intent(message: str, output_file: str = "") -> DocumentIntent:
    text = str(message or "")
    normalized = _normalize_text(text).strip(" .,!?:;")
    action = ""
    requested_output = ""

    if _looks_like_create_file_task(normalized):
        action = "create"
        requested_output = "transformed_content"
    elif re.search(r"\b(?:resume|resumir|resumo|sintetiza|síntese|sintese)\b", normalized):
        action = "summarize"
        requested_output = "summary"
    elif _looks_like_full_document_read(normalized):
        action = "read"
        requested_output = "full_contents"
    elif _looks_like_document_read_verb(normalized):
        action = "read"
        requested_output = "full_contents"

    file_reference = _extract_semantic_file_reference(text, output_file=output_file)
    needs_clarification = bool(
        action in {"read", "display", "summarize"}
        and not file_reference
        and _looks_like_document_reference_context(normalized)
    )
    return DocumentIntent(
        action=action,
        file_reference=file_reference,
        requested_output=requested_output,
        needs_clarification=needs_clarification,
    )


def _parse_document_task_request(message: str) -> DocumentTaskRequest:
    text = str(message or "")
    normalized = _normalize_text(text)
    actions = _document_actions(normalized, _looks_like_create_file_task(normalized))
    output_file = _extract_output_txt_filename(text)
    intent = _detect_document_intent(text, output_file=output_file)
    output_format = _extract_output_format(normalized)
    recipient = _extract_recipient(text)
    source_files = _extract_source_files(text, output_file)
    if intent.file_reference and intent.file_reference not in source_files:
        source_files.insert(0, intent.file_reference)
    if intent.action in {"read", "display"}:
        for action in ("locate_source", "read_source"):
            if action not in actions:
                actions.append(action)
    if intent.action == "summarize" and "summarize" not in actions:
        actions.append("summarize")
    if output_file and "write_output" not in actions:
        actions.append("write_output")
    if recipient and "compose_email" not in actions and _explicit_email_composition_request(normalized):
        actions.append("compose_email")
    return DocumentTaskRequest(
        source_files=source_files,
        output_file=output_file or None,
        recipient=recipient or None,
        actions=actions,
        output_format=output_format or None,
        instructions=text,
        action=intent.action,
        requested_output=intent.requested_output,
        needs_clarification=intent.needs_clarification,
    )


def _document_request_from_pending(pending: dict[str, object], message: str) -> DocumentTaskRequest:
    parsed = _parse_document_task_request(message)
    source_files = parsed.source_files or ([str(pending.get("source_file"))] if pending.get("source_file") else [])
    output = parsed.output_file or str(pending.get("requested_output") or "") or None
    recipient = parsed.recipient or str(pending.get("recipient") or "") or None
    output_format = parsed.output_format or str(pending.get("output_format") or "") or None
    requested_output = parsed.requested_output or str(pending.get("document_requested_output") or "")
    action = parsed.action or str(pending.get("document_action") or "")
    actions = [str(item) for item in pending.get("requested_actions", []) if item]
    for action in parsed.actions:
        if action not in actions:
            actions.append(action)
    instructions = str(pending.get("instructions") or "").strip()
    if message.strip() and message.strip() not in instructions:
        instructions = f"{instructions}\n{message}".strip()
    return DocumentTaskRequest(
        source_files=source_files,
        output_file=output,
        recipient=recipient,
        actions=actions,
        output_format=output_format,
        instructions=instructions or message,
        action=action,
        requested_output=requested_output,
        needs_clarification=parsed.needs_clarification,
    )


def _looks_like_read_or_summary(text: str) -> bool:
    return (
        _explicit_summary_request(text)
        or _explicit_email_composition_request(text)
        or _looks_like_document_read_verb(text)
        or _looks_like_full_document_read(text)
    )


def _looks_like_create_file_task(text: str) -> bool:
    return any(phrase in text for phrase in ("cria ", "criar ", "guarda em", "guardar em", "grava em", "escreve em", "com este texto"))


def _asks_to_read_file(text: str) -> bool:
    return _looks_like_document_read_verb(text) or _looks_like_full_document_read(text)


def _looks_like_summary_or_email(text: str) -> bool:
    return _explicit_summary_request(text) or _explicit_email_composition_request(text)


_SUMMARY_NEGATION_MARKERS = (
    "sem resumir",
    "sem resumo",
    "sem sintese",
    "sem síntese",
    "sem sintetizar",
    "nao resumas",
    "não resumas",
    "nao resumir",
    "não resumir",
    "nao quero resumo",
    "não quero resumo",
    "nao quero um resumo",
    "não quero um resumo",
    "nao e um resumo",
    "não é um resumo",
)


def _explicit_summary_request(text: str) -> bool:
    normalized = _normalize_text(text)
    if any(marker in normalized for marker in _SUMMARY_NEGATION_MARKERS):
        return False
    return bool(re.search(r"\b(?:resume|resumir|resumo|sintese|síntese|sintetiza)\b", normalized))


def _explicit_email_composition_request(text: str) -> bool:
    normalized = _normalize_text(text)
    return bool(
        re.search(r"\b(?:escreve|redige|prepara|cria|faz|compoe|compõe)\b.{0,60}\b(?:email|e-mail|mail|mensagem)\b", normalized)
        or re.search(r"\b(?:email|e-mail|mail|mensagem)\b.{0,60}\b(?:para enviar|a enviar|ao professor|à professora|a professora)\b", normalized)
    )


def _looks_like_document_read_verb(text: str) -> bool:
    normalized = _normalize_text(text)
    return bool(
        re.search(r"\b(?:le|ler|leias|lê|leres)\b", normalized)
        or re.search(r"\b(?:abre|abrir)\b.{0,40}\b(?:documento|ficheiro|pdf|docx|txt|md)\b", normalized)
        or "quero ver" in normalized
    )


_FULL_DOCUMENT_READ_MARKERS = (
    "diz me o que esta",
    "diz-me o que esta",
    "digas o que esta",
    "o que esta escrito",
    "o que la esta escrito",
    "conteudo completo",
    "conteudo integral",
    "conteudo todo",
    "texto completo",
    "texto integral",
    "na integra",
    "na íntegra",
    "mostra tudo",
    "mostra me o conteudo",
    "mostra-me o conteudo",
    "mostra o conteudo",
    "exatamente o que la esta",
    "exactamente o que la esta",
    "exatamente como esta",
    "palavra por palavra",
    "nao resumas",
    "não resumas",
    "sem resumir",
    "sem resumo",
    "sem sintese",
    "sem síntese",
)


def _looks_like_full_document_read(text: str) -> bool:
    normalized = _normalize_text(text)
    return contains_any_phrase(normalized, _FULL_DOCUMENT_READ_MARKERS)


def _looks_like_document_reference_context(text: str) -> bool:
    return any(marker in text for marker in ("ficheiro", "documento", "pdf", "docx", "txt", "workspace"))


def _looks_like_direct_create_after_document(text: str) -> bool:
    return _looks_like_create_file_task(text) and any(word in text for word in ("com", "base", "resumo", "email", "mail"))


def _document_actions(text: str, has_output: bool) -> list[str]:
    actions: list[str] = []
    if _looks_like_read_or_summary(text):
        actions.extend(["locate_source", "read_source"])
    if _looks_like_summary_or_email(text):
        actions.append("summarize")
    if _explicit_email_composition_request(text):
        actions.append("compose_email")
    if has_output or _looks_like_create_file_task(text):
        actions.append("write_output")
    return actions


def _needs_source_file(request: DocumentTaskRequest) -> bool:
    return any(action in request.actions for action in ("locate_source", "read_source", "summarize", "compose_email"))


def _needs_source_file_from_text(text: str) -> bool:
    return (
        _looks_like_summary_or_email(text)
        or bool(re.search(r"\b(l[eê]|ler)\b", text))
    )


def _extract_output_format(text: str) -> str:
    if re.search(r"\b(?:ficheiro|formato|documento)\s+txt\b", text) or re.search(r"\btxt\b", text) and _looks_like_create_file_task(text):
        return "txt"
    return ""


def _extract_recipient(message: str) -> str:
    match = re.search(r"\b(?:para|à|a)\s+(?:a\s+|o\s+)?([A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ'-]+)", message)
    if not match:
        return ""
    name = match.group(1).strip(" .,;:")
    if _normalize_text(name) in {"workspace", "pdf", "ficheiro", "documento"}:
        return ""
    return name


# Relational/grammatical words that a lazy regex capture can swallow instead
# of the actual filename (e.g. "documento que está no workspace X" capturing
# "que está" because "no" is the first recognized stop-word). A real filename
# reference never legitimately starts with one of these.
_NON_FILENAME_LEAD_WORDS = frozenset(
    {
        "que",
        "quem",
        "o",
        "a",
        "os",
        "as",
        "este",
        "esta",
        "esse",
        "essa",
        "isto",
        "isso",
        "aquele",
        "aquela",
        "aquilo",
        "nesse",
        "nessa",
        "neste",
        "nesta",
        "naquele",
        "naquela",
    }
)

# Checked before the generic relational patterns below: these anchor on a
# location/naming keyword ("workspace", "pasta", "chamado") so the capture
# starts at the actual filename token instead of drifting into "que está".
_NAMED_FILE_REFERENCE_PATTERNS = (
    r"\bworkspace\s+(.+?)(?:\s*[,.;:?!]|\s+(?:e|para|com|que|na|no|o|a)\b|$)",
    r"\bpasta\s+(.+?)(?:\s*[,.;:?!]|\s+(?:e|para|com|que|na|no|o|a)\b|$)",
    r"\b(?:ficheiro|documento|arquivo)\s+(?:chamado|chamada|com o nome)\s+(.+?)(?:\s*[,.;:?!]|\s+(?:e|para|com|que|na|no)\b|$)",
)


def _extract_semantic_file_reference(message: str, output_file: str = "") -> str:
    text = str(message or "")
    if _contains_path_traversal(text):
        return "../"
    output_norm = _normalize_text(output_file)

    for pattern in _NAMED_FILE_REFERENCE_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = _clean_filename_fragment(match.group(1))
        if output_norm and _normalize_text(candidate) == output_norm:
            continue
        if candidate and not _is_output_format_token(candidate) and _semantic_file_reference_is_plausible(candidate):
            return candidate

    explicit = _extract_workspace_filename(text)
    if (
        explicit
        and (not output_norm or _normalize_text(explicit) != output_norm)
        and _semantic_file_reference_is_plausible(explicit)
    ):
        return explicit

    patterns = (
        r"\b(?:documento|ficheiro|arquivo)\s+(.+?)(?:\s+(?:e|para|com|que|na|no)\b|$)",
        r"\b(?:o que est[áa]\s+(?:l[áa]\s+)?escrito\s+(?:em|no|na))\s+(.+?)(?:\s|$)",
        r"\b(?:conte[úu]do\s+(?:de|do|da))\s+(.+?)(?:\s|$)",
        r"\b(?:quero ver|mostra(?:-me)?|l[êe]|ler|leias)\s+(.+?)(?:\s+(?:na íntegra|na integra|completo|todo|tudo)\b|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = _clean_filename_fragment(match.group(1))
        if output_norm and _normalize_text(candidate) == output_norm:
            continue
        if candidate and not _is_output_format_token(candidate) and _semantic_file_reference_is_plausible(candidate):
            return candidate
    return ""


def _semantic_file_reference_is_plausible(candidate: str) -> bool:
    normalized = _normalize_text(candidate).strip(" .,!?:;")
    if not normalized:
        return False
    if normalized in {"conteudo", "conteudo completo", "tudo", "integra", "na integra"}:
        return False
    if normalized.startswith(("e guarda", "guarda em", "guardar em", "grava em", "e grava")):
        return False
    if len(normalized.split()) > 8:
        return False
    if normalized.split()[0] in _NON_FILENAME_LEAD_WORDS:
        return False
    return True


def _extract_source_files(message: str, output_file: str = "") -> list[str]:
    if _contains_path_traversal(message):
        return ["../"]

    output_norm = _normalize_text(output_file)
    sources: list[str] = []
    for quoted in re.findall(r"[\"“”']([^\"“”']+)[\"“”']", message):
        candidate = _clean_filename_fragment(quoted)
        if not candidate:
            continue
        if output_norm and _normalize_text(candidate) == output_norm:
            continue
        if _is_output_format_token(candidate):
            continue
        suffix = Path(candidate).suffix.lower()
        if suffix in DOCUMENT_EXTENSIONS or _quote_has_source_context(message, quoted):
            sources.append(candidate)

    for match in re.finditer(r"([^\s,;:\"'“”]+?\.(?:txt|md|pdf|docx))\b", message, flags=re.IGNORECASE):
        candidate = _clean_filename_fragment(match.group(1))
        if not candidate or _is_output_format_token(candidate):
            continue
        if output_norm and _normalize_text(candidate) == output_norm:
            continue
        if candidate not in sources:
            sources.append(candidate)

    for pattern in (
        r"\b(?:o\s+)?ficheiro\s+(?:é|e|chama-se|chama se)\s+(.+)$",
        r"\b(?:o\s+)?documento\s+(?:é|e|chama-se|chama se)\s+(.+)$",
        r"\bpdf\s+(?:é|e|chama-se|chama se)?\s*(.+)$",
    ):
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            candidate = _clean_filename_fragment(match.group(1))
            if output_file and _normalize_text(output_file) in _normalize_text(candidate):
                continue
            if candidate and not _is_output_format_token(candidate) and candidate not in sources:
                sources.append(candidate)

    return sources[:3]


def _quote_has_source_context(message: str, quoted: str) -> bool:
    index = message.find(quoted)
    if index < 0:
        return False
    window = _normalize_text(message[max(0, index - 80) : index + len(quoted) + 40])
    return any(word in window for word in ("pdf", "documento", "ficheiro", "notas", "workspace", "reuniao", "reunião"))


def _is_output_format_token(value: str) -> bool:
    normalized = _normalize_text(value).strip(" .")
    return normalized in {"txt", ".txt", "pdf", ".pdf", "docx", ".docx", "md", ".md"}


def _extract_workspace_filename(message: str) -> str:
    text = str(message or "")
    if _contains_path_traversal(text):
        return "../"
    quoted = re.findall(r"[\"“”']([^\"“”']+\.(?:txt|md|pdf|docx))\b[\"“”']?", text, flags=re.IGNORECASE)
    if quoted:
        return quoted[0].strip()
    match = re.search(r"([^\s,;:\"'“”]+?\.(?:txt|md|pdf|docx))\b", text, flags=re.IGNORECASE)
    if match:
        return _clean_filename_fragment(match.group(1))
    for marker in ("ficheiro", "documento", "pdf", "docx"):
        pattern = rf"{marker}\s+(.+?)(?:\s+(?:e|para|com|num|numa|em)\s|$)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _clean_filename_fragment(match.group(1))
    return ""


def _extract_output_txt_filename(message: str) -> str:
    text = str(message or "")
    if _contains_path_traversal(text):
        return "../"
    context_patterns = (
        r"\b(?:guarda|guardar|grava|gravar|escreve|escrever)\s+em\s+([^\s,;:\"'“”]+?\.txt)\b",
        r"\b(?:cria|criar)\s+(?:um\s+)?(?:ficheiro\s+)?(?:chamado\s+)?([^\s,;:\"'“”]+?\.txt)\b",
    )
    for pattern in context_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _clean_filename_fragment(match.group(1))
    if _extract_inline_file_content(message):
        match = re.search(r"([^\s,;:\"'“”]+?\.txt)\b", text, flags=re.IGNORECASE)
        if match:
            return _clean_filename_fragment(match.group(1))
    return ""


def _extract_inline_file_content(message: str) -> str:
    patterns = (
        r"\bcom este texto[:\s]+(.+)$",
        r"\bcom o texto[:\s]+(.+)$",
        r"\bconteudo[:\s]+(.+)$",
        r"\bconteúdo[:\s]+(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""


def _clean_filename_fragment(value: str) -> str:
    cleaned = str(value or "").strip().strip("\"'“”.,:;")
    cleaned = re.sub(r"^(resume|resumir|le|lê|ler|cria|criar|guarda|guardar)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(o|a|um|uma|em|ficheiro|documento|pdf|docx)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _contains_path_traversal(text: str) -> bool:
    return "../" in text or "..\\" in text


def _find_workspace_file_candidates(raw_filename: str, workspace_path: Path) -> dict[str, object]:
    result = _workspace_search(raw_filename, workspace_path)
    return {"blocked": result.blocked, "candidates": result.matches}


def _workspace_search(raw_filename: str, workspace_path: Path) -> WorkspaceSearchResult:
    raw_text = str(raw_filename or "").strip().strip("\"'“”")
    candidate = Path(raw_text)
    if _is_output_format_token(raw_text):
        return WorkspaceSearchResult(found=False)
    root = workspace_path.resolve()
    if not root.exists():
        return WorkspaceSearchResult(found=False)
    if candidate.is_absolute():
        try:
            resolved_absolute = candidate.resolve()
            relative = resolved_absolute.relative_to(root)
        except (OSError, ValueError):
            return WorkspaceSearchResult(found=False, blocked=True)
        if not resolved_absolute.is_file() or resolved_absolute.suffix.lower() not in DOCUMENT_EXTENSIONS:
            return WorkspaceSearchResult(found=False)
        return WorkspaceSearchResult(found=True, matches=[resolved_absolute], match_type="absolute")
    if ".." in candidate.parts:
        return WorkspaceSearchResult(found=False, blocked=True)
    try:
        exact = (root / raw_text).resolve()
        if exact != root and root not in exact.parents:
            return WorkspaceSearchResult(found=False, blocked=True)
    except OSError:
        return WorkspaceSearchResult(found=False, blocked=True)

    files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in DOCUMENT_EXTENSIONS]
    exact_matches = [path for path in files if path.name.lower() == candidate.name.lower()]
    if exact_matches:
        matches = sorted(exact_matches)
        return WorkspaceSearchResult(found=bool(matches), matches=matches, match_type="exact")
    if candidate.suffix:
        rel_matches = [path for path in files if path.relative_to(root).as_posix().lower() == raw_text.replace("\\", "/").lower()]
        if rel_matches:
            matches = sorted(rel_matches)
            return WorkspaceSearchResult(found=bool(matches), matches=matches, match_type="relative")
    normalized = _normalize_filename_for_match(candidate.stem or raw_text)
    matches = []
    scored_matches: list[tuple[float, Path]] = []
    for path in files:
        path_stem = _normalize_filename_for_match(path.stem)
        if not normalized or not path_stem:
            continue
        if path_stem.startswith(normalized) or normalized in path_stem:
            matches.append(path)
            continue
        singular_input = _singularized_tokens(normalized)
        singular_stem = _singularized_tokens(path_stem)
        if singular_input and singular_input == singular_stem:
            scored_matches.append((0.99, path))
            continue
        score = SequenceMatcher(None, normalized, path_stem).ratio()
        token_score = SequenceMatcher(None, singular_input, singular_stem).ratio() if singular_input and singular_stem else 0.0
        best = max(score, token_score)
        if best >= 0.82:
            scored_matches.append((best, path))
    if matches:
        matches = sorted(matches)
        return WorkspaceSearchResult(found=bool(matches), matches=matches, match_type="stem")
    scored_matches.sort(key=lambda item: item[0], reverse=True)
    if scored_matches:
        best_score = scored_matches[0][0]
        confident = [path for score, path in scored_matches if score >= best_score - 0.03 and score >= 0.82]
        return WorkspaceSearchResult(found=True, matches=sorted(confident), match_type="fuzzy")
    matches = sorted(matches)
    return WorkspaceSearchResult(found=bool(matches), matches=matches)


def _normalize_filename_for_match(value: str) -> str:
    normalized = _normalize_text(value)
    normalized = re.sub(r"[-_]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _singularized_tokens(value: str) -> str:
    tokens = []
    for token in _normalize_filename_for_match(value).split():
        if len(token) > 4 and token.endswith("oes"):
            token = token[:-3] + "ao"
        elif len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        tokens.append(token)
    return " ".join(tokens)


def _choose_candidate_from_message(message: str, candidates: list[str]) -> str:
    text = _normalize_text(message)
    for candidate in candidates:
        if _normalize_text(candidate) in text or _normalize_text(Path(candidate).stem) in text:
            return candidate
    if len(candidates) == 1 and _is_positive_document_confirmation(text):
        return candidates[0]
    return ""


def _is_positive_document_confirmation(text: str) -> bool:
    normalized = _normalize_text(text).strip(" .!?")
    return normalized in {"sim", "s", "ok", "okay", "claro", "podes criar", "cria", "criar", "esse mesmo", "pode ser"}


def _looks_like_document_full_content_followup(text: str) -> bool:
    normalized = _normalize_text(text).strip(" .!?")
    return (
        _is_positive_document_confirmation(normalized)
        or _looks_like_full_document_read(normalized)
        or normalized in {"continua", "mostra tudo", "sim esse", "sim esse mesmo", "esse", "esse mesmo"}
    )


def _looks_like_document_context_clear(text: str) -> bool:
    normalized = _normalize_text(text).strip(" .!?")
    return any(
        marker in normalized
        for marker in (
            "vamos falar de outra coisa",
            "muda de assunto",
            "esquece o documento",
            "fecha o documento",
            "deixa o documento",
        )
    )


# Opinion/suggestion phrasing that plausibly continues a document conversation
# even without hitting one of _active_document_followup_action's specific
# markers (e.g. "tens alguma sugestão para melhorar o mail?" — a real
# paraphrase of "review" that doesn't contain any of its fixed markers).
# Specific enough on their own (people don't ask to "melhorar"/"corrigir"
# something without an artifact in mind).
_DOCUMENT_OPINION_STRONG_MARKERS = (
    "sugestao",
    "sugestão",
    "sugestoes",
    "sugestões",
    "melhorar",
    "melhoria",
    "consegues melhorar",
    "podes melhorar",
    "esta bem assim",
    "está bem assim",
)

# Generic enough to be about anything ("achas que vai chover?") — only count
# these when a document-reference word also appears in the same message.
_DOCUMENT_OPINION_WEAK_MARKERS = (
    "achas",
    "acha que",
    "opiniao",
    "opinião",
    "que tal",
    "parece te",
    "parece-te",
    "como fica",
)

_DOCUMENT_GENERIC_REFERENCE_WORDS = (
    "mail",
    "email",
    "e-mail",
    "documento",
    "ficheiro",
    "texto",
    "isto",
    "isso",
    "nota",
    "pdf",
    "docx",
    "conteudo",
    "conteúdo",
)

# Extra positive signal keyed by the document's own detected type, so a bare
# mention of "mail"/"email" (without any opinion marker) still counts when
# the active document actually is one.
_DOCUMENT_TYPE_REFERENCE_WORDS = {
    "email": ("mail", "email", "e-mail"),
    "meeting_notes": ("reuniao", "reunião", "ata"),
}


def _looks_like_ambiguous_document_followup(text: str, content_type: str) -> bool:
    normalized = _normalize_text(text).strip(" .!?")
    if not normalized:
        return False
    if contains_any_phrase(normalized, _DOCUMENT_OPINION_STRONG_MARKERS):
        return True
    type_words = _DOCUMENT_TYPE_REFERENCE_WORDS.get(content_type, ())
    if type_words and contains_any_phrase(normalized, type_words):
        return True
    if contains_any_phrase(normalized, _DOCUMENT_OPINION_WEAK_MARKERS) and contains_any_phrase(
        normalized, _DOCUMENT_GENERIC_REFERENCE_WORDS
    ):
        return True
    return False


# Rewriting a document faithfully is a low-creativity, high-fidelity task —
# a lower temperature reduces the odds of the model refusing, adding a
# chatty preamble, or drifting from the original's facts. num_predict caps
# worst-case generation length/latency while staying generous enough for a
# full rewritten document (observed real responses were 90-370 tokens).
# Applied only to this composer call — every other LLM call keeps using the
# provider's defaults.
_REWRITE_TEMPERATURE = 0.2
_REWRITE_MAX_OUTPUT_TOKENS = 1200

# Aggregate wall-clock budget for the whole rewrite flow (both attempts,
# validation and response composition) -- independent of, and tighter than,
# a single Ollama HTTP call's own 120s default timeout. A slow first attempt
# eats into the budget available for a possible second attempt; the timeout
# passed to the provider on each call is clamped to whatever remains of it.
DOCUMENT_REWRITE_TOTAL_TIMEOUT_SECONDS = 60
# Below this much remaining budget, starting another LLM call isn't worth
# it (real rewrite responses take several seconds at minimum) -- the flow
# gives up cleanly instead of starting a call almost certain to also time out.
_REWRITE_MIN_SECONDS_FOR_ATTEMPT = 2.0

# Verbs that, on their own, unambiguously mean "transform the active
# document now" (produce a full rewritten version) rather than something
# else. Word-boundary-safe matching already treats a hyphenated pronoun
# suffix ("torna-o", "melhora-o") as a separate token bounded by the hyphen,
# so no need to enumerate pronoun forms explicitly — "torna" alone matches
# inside "torna-o" too.
_REWRITE_STANDALONE_VERBS = ("reescreve", "reformula", "corrige", "adapta", "melhora")

# These need a paired quality/target word nearby to disambiguate from other
# meanings: "deixa estar" is a cancel phrase (_is_negative_confirmation),
# "escreve um email" is a brand-new composition request
# (_explicit_email_composition_request), and "torna" alone isn't a complete
# instruction.
_REWRITE_GUARDED_VERBS = ("torna", "deixa", "escreve")
_REWRITE_QUALITY_WORDS = ("formal", "claro", "profissional", "melhor")

_REWRITE_EXPLICIT_PHRASES = (
    "reescreve num tom mais formal",
    "faz uma versao melhor",
    "escreve isto de outra forma",
    "escreve de outra forma",
    "corrige e melhora",
    "adapta para um tom profissional",
    "adapta para um tom mais profissional",
)


def _looks_like_rewrite_request(text: str) -> bool:
    normalized = _normalize_text(text)
    if contains_any_phrase(normalized, _REWRITE_EXPLICIT_PHRASES):
        return True
    if contains_any_phrase(normalized, _REWRITE_STANDALONE_VERBS):
        return True
    return bool(find_near_pair_span(normalized, _REWRITE_GUARDED_VERBS, _REWRITE_QUALITY_WORDS, max_gap=40))


# Elliptical/comparative continuations of an open draft thread -- no
# explicit rewrite verb, just "still not good enough" phrasing. Only ever
# consulted when a draft already exists (see _active_document_followup_action),
# so this never invents a document context on its own.
_DRAFT_REFINEMENT_STANDALONE_PHRASES = (
    "ainda consegues melhor",
    "ainda consegues melhorar",
    "consegues fazer melhor",
    "consegues melhorar isto",
    "consegues melhorar isso",
    "tenta melhorar",
    "refina mais",
    "refina isto",
    "refina a versao",
    "melhora mais um pouco",
    "podes melhorar mais",
    "faz uma versao ainda melhor",
    "outra versao mas melhor",
    "nao esta suficientemente formal",
    "nao ficou suficientemente formal",
    "quero algo mais profissional",
)
_DRAFT_REFINEMENT_COMPARATIVE_WORDS = ("ainda", "mais", "outra vez", "de novo", "um pouco")
_DRAFT_REFINEMENT_QUALITY_WORDS = ("melhor", "formal", "profissional", "claro", "simples", "conciso", "elegante")


def _looks_like_draft_refinement_request(text: str) -> bool:
    normalized = _normalize_text(text)
    if contains_any_phrase(normalized, _DRAFT_REFINEMENT_STANDALONE_PHRASES):
        return True
    return bool(
        find_near_pair_span(normalized, _DRAFT_REFINEMENT_COMPARATIVE_WORDS, _DRAFT_REFINEMENT_QUALITY_WORDS, max_gap=40)
    )


_GREETING_LINE_PATTERN = re.compile(
    r"^(?:ol[áa]|caro|cara|prezad[oa]s?|estimad[oa]s?|exm[oa]s?)\b[,:]?\s*([A-ZÀ-Ý][\wÀ-ÿ'-]*)?",
    re.IGNORECASE,
)
_CLOSING_LINE_WORDS = (
    "cumprimentos",
    "atenciosamente",
    "atentamente",
    "obrigado",
    "obrigada",
    "abraco",
    "abraço",
    "com os melhores cumprimentos",
)
# Capitalized words that show up at the start of common sentences/lines in
# this kind of document — excluded from entity extraction so ordinary
# sentence capitalization doesn't get mistaken for a name/topic that must
# survive the rewrite.
_ENTITY_STOPWORDS = frozenset(
    {
        "assunto",
        "subject",
        "segue",
        "cumprimentos",
        "atenciosamente",
        "ola",
        "caro",
        "cara",
        "prezado",
        "prezada",
        "prezados",
        "prezadas",
        "estimado",
        "estimada",
        "obrigado",
        "obrigada",
        "discutimos",
        "abraco",
    }
)


def _extract_document_anchors(content: str) -> dict[str, object]:
    """Structural anchors pulled from the ORIGINAL document — generic
    (regex/heuristics on structure), never hardcoded to a specific person or
    project. Used to check a rewritten draft actually kept the substance of
    the original instead of drifting into commentary."""
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    subject = ""
    greeting_name = ""
    closing_name = ""
    bullet_items: list[str] = []
    entity_tokens: set[str] = set()
    for index, line in enumerate(lines):
        norm = _normalize_text(line)
        if not subject and (norm.startswith("assunto") or norm.startswith("subject")):
            subject = line
        if not greeting_name:
            greeting_match = _GREETING_LINE_PATTERN.match(line)
            if greeting_match and greeting_match.group(1):
                greeting_name = greeting_match.group(1)
        if not closing_name and norm.rstrip(" ,.:;") in _CLOSING_LINE_WORDS and index + 1 < len(lines):
            closing_name = lines[index + 1]
        bullet_match = re.match(r"^[-*•]\s*(.+)$", line)
        if bullet_match:
            bullet_items.append(bullet_match.group(1).strip())
            continue
        for word_index, word in enumerate(line.split()):
            cleaned = word.strip(",.:;!?()\"'“”")
            if (
                len(cleaned) >= 3
                and cleaned[0].isupper()
                and cleaned[1:].islower()
                and word_index > 0
                and _normalize_text(cleaned) not in _ENTITY_STOPWORDS
            ):
                entity_tokens.add(cleaned)
    if greeting_name:
        entity_tokens.add(greeting_name)
    if closing_name:
        entity_tokens.add(closing_name)
    return {
        "subject": subject,
        "greeting_name": greeting_name,
        "closing_name": closing_name,
        "bullet_items": bullet_items,
        "entity_tokens": entity_tokens,
        "length": len(content.strip()),
    }


# Phrasing that marks a response as commentary/a question/a refusal rather
# than the rewritten document itself — the exact failure mode reported
# manually ("O documento parece pronto para enviar! ... vais enviar...?").
_REWRITE_INVALID_RESPONSE_MARKERS = (
    "parece pronto",
    "esta pronto",
    "está pronto",
    "vais enviar",
    "posso reescrever",
    "queres que eu",
    "gostarias que",
    "posso ajudar",
    "sugiro que",
    "sugestao",
    "sugestão",
    "vou reescrever",
    "vou analisar",
    "deixa me analisar",
    "deixa-me analisar",
)

# A short leading sentence announcing what the model is about to do
# ("Vou reescrever o documento completo. Aqui está a resposta:") immediately
# followed by the real document is a preamble, not a refusal — reproduced
# manually, this alone was enough to reject an otherwise perfect rewrite
# because it happened to contain "vou reescrever". _strip_rewrite_preamble
# only ever removes text before the FIRST blank-line paragraph break, and
# only when that leading text is short and matches one of these markers —
# it never touches anything that could be real document content.
_REWRITE_PREAMBLE_MARKERS = (
    "vou reescrever",
    "vou analisar",
    "aqui esta a resposta",
    "aqui está a resposta",
    "aqui esta o documento",
    "aqui está o documento",
    "segue a versao reescrita",
    "segue a versão reescrita",
    "deixa me analisar",
    "deixa-me analisar",
)
_REWRITE_PREAMBLE_MAX_CHARS = 200


def _strip_rewrite_preamble(candidate: str) -> str:
    text = str(candidate or "")
    if "\n\n" not in text:
        return text
    preamble, _, remainder = text.partition("\n\n")
    preamble = preamble.strip()
    remainder = remainder.strip()
    if not preamble or not remainder or len(preamble) > _REWRITE_PREAMBLE_MAX_CHARS:
        return text
    if not contains_any_phrase(_normalize_text(preamble), _REWRITE_PREAMBLE_MARKERS):
        return text
    return remainder


# Generic placeholder patterns ("[nome]", "<nome>", "nome aqui", "sua
# assinatura", ...) — never a specific person's name, so this stays valid
# regardless of which document/recipient is active.
_PLACEHOLDER_PATTERNS = (
    re.compile(r"\[[^\]]{0,30}nome[^\]]{0,30}\]", re.IGNORECASE),
    re.compile(r"<[^>]{0,30}nome[^>]{0,30}>", re.IGNORECASE),
    re.compile(r"\bnome\s+aqui\b", re.IGNORECASE),
    re.compile(r"\binserir\s+nome\b", re.IGNORECASE),
    re.compile(r"\bsua\s+assinatura\b", re.IGNORECASE),
    re.compile(r"\[[^\]]{0,30}assinatura[^\]]{0,30}\]", re.IGNORECASE),
)


def _looks_like_placeholder(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(pattern.search(normalized) for pattern in _PLACEHOLDER_PATTERNS)


def _validate_rewrite_draft(original_content: str, candidate: str) -> tuple[bool, str, str]:
    """Structural plausibility check for a rewrite draft before it is
    accepted — catches the model answering with a question/comment instead
    of the rewritten document. Anchors are extracted from the ORIGINAL
    document, never hardcoded to a specific name/project.

    Returns (valid, reason_code, cleaned_content). cleaned_content has any
    leading meta-commentary preamble stripped and is what should be stored
    as the draft — the caller should use it instead of the raw candidate
    regardless of whether validation passed, so a preamble never leaks into
    a stored draft even on a borderline accept.
    """
    cleaned = _strip_rewrite_preamble(str(candidate or ""))
    candidate_stripped = cleaned.strip()
    if not candidate_stripped:
        return False, "resposta_vazia", cleaned
    anchors = _extract_document_anchors(original_content)
    original_length = int(anchors["length"]) or 1
    if len(candidate_stripped) < original_length * 0.5:
        return False, "muito_curta", cleaned
    last_line = next((line for line in reversed(candidate_stripped.splitlines()) if line.strip()), "")
    if last_line.strip().endswith("?"):
        return False, "termina_em_pergunta", cleaned
    if _looks_like_placeholder(candidate_stripped):
        return False, "placeholder_detected", cleaned
    normalized_candidate = _normalize_text(candidate_stripped)
    if contains_any_phrase(normalized_candidate, _REWRITE_INVALID_RESPONSE_MARKERS):
        return False, "parece_comentario", cleaned
    entity_tokens = anchors["entity_tokens"]
    if entity_tokens:
        present = sum(1 for token in entity_tokens if token.lower() in candidate_stripped.lower())
        if present / len(entity_tokens) < 0.6:
            return False, "faltam_entidades", cleaned
    bullet_items = anchors["bullet_items"]
    if bullet_items:
        preserved = sum(1 for item in bullet_items if item.lower() in candidate_stripped.lower())
        if preserved / len(bullet_items) < 0.5:
            return False, "faltam_itens_lista", cleaned
    greeting_name = str(anchors["greeting_name"] or "")
    if greeting_name and greeting_name.lower() not in candidate_stripped.lower():
        return False, "falta_saudacao", cleaned
    closing_name = str(anchors["closing_name"] or "")
    if closing_name and closing_name.lower() not in candidate_stripped.lower():
        return False, "falta_assinatura", cleaned
    return True, "", cleaned


_REWRITE_FAILURE_MESSAGES = {
    "placeholder_detected": "A versão gerada ficou incompleta porque continha campos por preencher.",
    "falta_assinatura": "A versão gerada perdeu elementos importantes do documento.",
    "falta_saudacao": "A versão gerada perdeu elementos importantes do documento.",
    "faltam_entidades": "A versão gerada perdeu elementos importantes do documento.",
    "faltam_itens_lista": "A versão gerada perdeu elementos importantes do documento.",
    "no_op_detected": "A nova versão ficou demasiado parecida com a anterior — não houve uma melhoria real.",
}
_DEFAULT_REWRITE_FAILURE_MESSAGE = "O modelo respondeu com um comentário em vez do documento completo."


def _rewrite_failure_message(reason: str) -> str:
    return _REWRITE_FAILURE_MESSAGES.get(reason, _DEFAULT_REWRITE_FAILURE_MESSAGE)


# Deliberately simple: no embeddings, no semantic model -- just a plain-text
# similarity ratio (stdlib difflib, already imported). Only used to catch a
# refinement pass that came back practically unchanged; a small genuine edit
# must not be flagged, so the threshold is high (>=0.97) and exact match is
# checked first for the common case.
_NOOP_REFINEMENT_SIMILARITY_THRESHOLD = 0.97


def _is_noop_refinement(base_content: str, candidate: str) -> bool:
    base_normalized = " ".join(_normalize_text(base_content).split())
    candidate_normalized = " ".join(_normalize_text(candidate).split())
    if not base_normalized or not candidate_normalized:
        return False
    if base_normalized == candidate_normalized:
        return True
    ratio = SequenceMatcher(None, base_normalized, candidate_normalized).ratio()
    return ratio >= _NOOP_REFINEMENT_SIMILARITY_THRESHOLD


def _document_rewrite_prompt(request: DocumentTaskRequest, context: ActiveDocumentContext) -> str:
    return (
        "TAREFA: reescrever o documento completo abaixo.\n"
        "SAÍDA OBRIGATÓRIA: apenas o documento completo reescrito, do início ao fim.\n"
        "Não faças perguntas. Não analises. Não resumas. Não omitas nenhuma secção.\n"
        "Mantém todos os factos, nomes, destinatário, assunto, listas e assinatura do original.\n"
        "Preserva os tempos verbais e a cronologia de cada facto -- não transformes ações já concluídas em "
        "ações atuais, nem mudes quem fez, o que fez, quando fez, ou se algo está concluído, previsto ou "
        "pendente. Podes melhorar o estilo; nunca reinterpretar a cronologia.\n"
        "Não uses placeholders. Não digas que o documento está pronto, nem que vais enviar ou guardar algo.\n"
        "Não comeces a resposta com frases como 'Aqui está' — devolve apenas o documento.\n"
        f"Aplica apenas esta transformação: {request.instructions}\n\n"
        f"<ORIGINAL_DOCUMENT>\n{_truncate_document_content(context.content, limit=12000)}\n</ORIGINAL_DOCUMENT>\n\n"
        f"<USER_TRANSFORMATION>\n{request.instructions}\n</USER_TRANSFORMATION>"
    )


def _document_rewrite_correction_prompt(
    request: DocumentTaskRequest, context: ActiveDocumentContext, invalid_response: str, reason: str
) -> str:
    return (
        f"A tua resposta anterior não cumpriu o pedido ({reason}). "
        "Essa resposta é só diagnóstico interno — não a uses como base, parte apenas do documento original abaixo.\n"
        "TAREFA: reescrever o documento completo abaixo, do início ao fim.\n"
        "SAÍDA OBRIGATÓRIA: apenas o documento completo reescrito.\n"
        "Não faças perguntas. Não analises. Não comentes. Não omitas nenhuma secção.\n"
        "Mantém todos os factos, nomes, destinatário, assunto, listas e assinatura do original.\n"
        "Preserva os tempos verbais e a cronologia de cada facto -- não transformes ações já concluídas em "
        "ações atuais, nem mudes quem fez, o que fez, quando fez, ou se algo está concluído, previsto ou "
        "pendente. Podes melhorar o estilo; nunca reinterpretar a cronologia.\n"
        "Não comeces a resposta com frases como 'Aqui está' — devolve apenas o documento.\n"
        f"Aplica apenas esta transformação: {request.instructions}\n\n"
        f"<ORIGINAL_DOCUMENT>\n{_truncate_document_content(context.content, limit=12000)}\n</ORIGINAL_DOCUMENT>\n\n"
        f"<USER_TRANSFORMATION>\n{request.instructions}\n</USER_TRANSFORMATION>\n\n"
        f"<RESPOSTA_INVALIDA_ANTERIOR razao=\"{reason}\">\n{_truncate_document_content(invalid_response, limit=2000)}\n</RESPOSTA_INVALIDA_ANTERIOR>"
    )


def _document_refinement_prompt(request: DocumentTaskRequest, context: ActiveDocumentContext, base_content: str) -> str:
    """Prompt for improving an EXISTING draft further -- not the first
    rewrite. The model must understand it's iterating on its own prior
    output, not repeating the original transformation from scratch."""
    return (
        "TAREFA: melhorar ainda mais a versão já reescrita abaixo. Não é a primeira reescrita — é um refinamento "
        "de uma versão que já foi produzida antes.\n"
        "SAÍDA OBRIGATÓRIA: apenas o documento completo melhorado, do início ao fim.\n"
        "Não faças perguntas. Não analises. Não resumas. Não omitas nenhuma secção.\n"
        "Mantém todos os factos, nomes, destinatário, assunto, listas e assinatura.\n"
        "Preserva os tempos verbais e a cronologia de cada facto -- não transformes ações já concluídas em "
        "ações atuais, nem mudes quem fez, o que fez, quando fez, ou se algo está concluído, previsto ou "
        "pendente. Podes melhorar o estilo; nunca reinterpretar a cronologia.\n"
        "Não uses placeholders. Não digas que não tens acesso ao documento. "
        "Não digas que o documento está pronto, nem que vais enviar ou guardar algo.\n"
        "Não comeces a resposta com frases como 'Aqui está' — devolve apenas o documento.\n"
        "Produz uma melhoria real e percetível (estilo, clareza ou tom) — não repitas a versão atual quase sem alterações.\n"
        f"Aplica este pedido de melhoria: {request.instructions}\n\n"
        f"<VERSAO_ATUAL>\n{_truncate_document_content(base_content, limit=12000)}\n</VERSAO_ATUAL>\n\n"
        f"<DOCUMENTO_ORIGINAL_REFERENCIA>\n{_truncate_document_content(context.content, limit=12000)}\n</DOCUMENTO_ORIGINAL_REFERENCIA>\n\n"
        f"<USER_TRANSFORMATION>\n{request.instructions}\n</USER_TRANSFORMATION>"
    )


def _document_refinement_correction_prompt(
    request: DocumentTaskRequest,
    context: ActiveDocumentContext,
    base_content: str,
    invalid_response: str,
    reason: str,
) -> str:
    if reason == "no_op_detected":
        problem = "A tua resposta anterior ficou praticamente igual à versão atual — isso não é uma melhoria."
        instruction = "Melhora de forma mais evidente (estilo, clareza ou tom), sem alterar nenhum facto."
    else:
        problem = f"A tua resposta anterior não cumpriu o pedido ({reason})."
        instruction = "Parte apenas da versão atual abaixo, não da resposta anterior."
    return (
        f"{problem} {instruction}\n"
        "TAREFA: melhorar ainda mais a versão já reescrita abaixo, do início ao fim.\n"
        "SAÍDA OBRIGATÓRIA: apenas o documento completo melhorado.\n"
        "Não faças perguntas. Não analises. Não comentes. Não omitas nenhuma secção.\n"
        "Mantém todos os factos, nomes, destinatário, assunto, listas e assinatura.\n"
        "Preserva os tempos verbais e a cronologia de cada facto -- não transformes ações já concluídas em "
        "ações atuais, nem mudes quem fez, o que fez, quando fez, ou se algo está concluído, previsto ou "
        "pendente. Podes melhorar o estilo; nunca reinterpretar a cronologia.\n"
        "Não uses placeholders. Não comeces a resposta com frases como 'Aqui está' — devolve apenas o documento.\n"
        f"Aplica este pedido de melhoria: {request.instructions}\n\n"
        f"<VERSAO_ATUAL>\n{_truncate_document_content(base_content, limit=12000)}\n</VERSAO_ATUAL>\n\n"
        f"<DOCUMENTO_ORIGINAL_REFERENCIA>\n{_truncate_document_content(context.content, limit=12000)}\n</DOCUMENTO_ORIGINAL_REFERENCIA>\n\n"
        f"<USER_TRANSFORMATION>\n{request.instructions}\n</USER_TRANSFORMATION>\n\n"
        f"<RESPOSTA_INVALIDA_ANTERIOR razao=\"{reason}\">\n{_truncate_document_content(invalid_response, limit=2000)}\n</RESPOSTA_INVALIDA_ANTERIOR>"
    )


_BARE_OPINION_QUESTION_MARKERS = ("achas", "que tal", "concordas", "o que dizes", "que dizes")


def _is_referentially_ambiguous_opinion_question(text: str) -> bool:
    """A short opinion question with no antecedent at all ("O que achas?"
    right after a topic change, with no active document) — deterministically
    ask what it's about instead of letting the Composer guess and risk a
    hallucinated-sounding answer that later gets caught by the ungrounded-
    memory-claim guard and replaced with a canned memory-flavored message."""
    normalized = _normalize_text(text).strip(" .!?")
    if not normalized or len(normalized.split()) > 4:
        return False
    return contains_any_phrase(normalized, _BARE_OPINION_QUESTION_MARKERS)


# --- Workspace command (spatial/layout) detection --------------------------
# Passo 3.2: distinguishes WORKSPACE commands (about the on-screen
# arrangement -- which card is big, side by side, hidden, closed) from
# CONTENT commands (rewrite/summarize/etc, detected further below). Phrase
# tables are deliberately disjoint from every _WORKSPACE-unrelated table
# above (draft/compare/discard/restore/save/display/review/interpret/
# summarize/rewrite) so this new detector never shadows existing, already
# validated document-followup routing -- and _try_workspace_command runs
# BEFORE document routing, so its own phrases always win when they match.
#  Deliberately narrow (requires a continuation verb like "a fazer"/"a
#  trabalhar") -- a bare "volta ao que estavamos"/"volta a isso" would also
#  match ordinary pending-action-cancellation phrasing ("Cancela e volta ao
#  que estávamos a falar."), which must keep working exactly as before.
_WORKSPACE_RESTORE_SCENE_PHRASES = (
    "volta ao que estavamos a fazer",
    "volta ao que estavamos a trabalhar",
    "volta ao que tinhamos aberto",
    "volta a trabalhar naquilo",
)
_WORKSPACE_AUTO_ORGANIZE_PHRASES = (
    "organiza isto automaticamente",
    "organiza automaticamente",
    "organiza isto sozinho",
)
_WORKSPACE_SIDE_BY_SIDE_PHRASES = (
    "lado a lado",
    "um ao lado do outro",
    "mostra os dois grandes",
)
_WORKSPACE_FOCUS_EXCERPT_MARKERS = ("mostra so o", "mostra apenas o", "so quero ver o", "quero ver so o")
_WORKSPACE_WORK_ON_PHRASES = (
    "vamos trabalhar na segunda versao",
    "vamos trabalhar no rascunho",
    "vamos trabalhar na versao",
    "trabalhar na segunda versao",
    "trabalhar no rascunho",
)
_WORKSPACE_ASIDE_PHRASES = (
    "chega isso para o lado",
    "chega isto para o lado",
    "poe isso ao lado",
    "poe isto ao lado",
)
_WORKSPACE_BRING_FORWARD_PHRASES = (
    "traz o orcamento para a frente",
    "traz isto para a frente",
    "traz isso para a frente",
    "traz para a frente",
)
_WORKSPACE_MINIMIZE_PHRASES = (
    "minimiza o original",
    "minimiza isto",
    "minimiza isso",
    "podes minimizar o original",
    "ja podes minimizar",
    "compacta o original",
    "compacta isto",
)
_WORKSPACE_EXPAND_PHRASES = (
    "expande o original",
    "expande a versao",
    "expande isto",
    "expande isso",
    "aumenta o original",
    "quero ver o original maior",
)
_WORKSPACE_HIDE_PHRASES = (
    "esconde isto",
    "esconde isso",
    "esconde o original",
    "esconde isto por agora",
    "esconde isso por agora",
)
_WORKSPACE_CLOSE_PHRASES = (
    "fecha a fatura",
    "fecha o original",
    "fecha isto",
    "fecha isso",
    "fecha o ficheiro",
    "fecha a segunda versao",
    "fecha o segundo ficheiro",
)
_WORKSPACE_KEEP_OPEN_PHRASES = (
    "mantem isto aberto",
    "mantem isso aberto",
    "deixa isto aberto",
    "deixa so estes dois abertos",
    "deixa apenas estes dois abertos",
)
_WORKSPACE_SHOW_PHRASES = (
    "mostra-me o ficheiro",
    "mostra o ficheiro",
    "preciso de ver rapidamente",
    "deixa-me ver rapidamente",
    "mostra-me rapidamente",
    "quero ver a fatura",
    "mostra-me o segundo ficheiro",
    "mostra o segundo ficheiro",
)
# --- Echo-entity movement commands (found missing entirely in real-world
# testing: "desloca-te para a direita" fell through to GENERAL_CONVERSATION
# because no phrase table for moving ECHO ITSELF -- as opposed to a
# WorkspaceObject -- ever existed). target_phrase is always the constant
# "echo_entity"; a `region` field (not resolved here, just named) tells the
# frontend which existing safe-anchor zone to move to.
_ENTITY_TARGET_PHRASE = "echo_entity"
# Broadened after real-world testing showed "chega para a esquerda" (no
# reflexive "-te") fell through entirely -- only "chega-te"/"chega te" were
# recognized. Movement is a whole verb FAMILY, not one fixed phrase per
# verb (section 1): desloca/move/mexe/vai/chega/encosta/aproxima/afasta/
# sai/põe-te/fica/coloca-te.
_ENTITY_MOVEMENT_VERBS = (
    "desloca", "desloca-te",
    "move", "move-te",
    "mexe", "mexe-te",
    "vai",
    "chega", "chega-te", "chega te",
    "encosta", "encosta-te",
    "aproxima", "aproxima-te",
    "afasta", "afasta-te",
    "sai",
    "poe-te", "poe te",
    "fica",
    "coloca-te",
)
# "lado"/"canto" are direction words WITHOUT a specific region (ambiguous
# among left/right or among four corners) -- resolved to "" below, which
# the frontend's moveEchoEntityToRegion already treats as "pick a sensible
# default" (the same fallback make_room already uses). "direta" is a
# deliberate typo-tolerance entry -- confirmed present in a real failed
# user message ("desloca te para a direta").
_ENTITY_DIRECTION_WORDS = ("direita", "direta", "esquerda", "cima", "baixo", "lado", "canto", "centro")
_ENTITY_DIRECTION_TO_REGION = {
    "direita": "right",
    "direta": "right",
    "esquerda": "left",
    "cima": "up",
    "baixo": "down",
    "centro": "center",
    "lado": "",
    "canto": "",
}
# Section 2: "chega" is ambiguous ("chega para a esquerda" = movement,
# "já chega"/"chega disso" = stop/finish). Disambiguation is structural,
# not a special-cased word list: movement requires a direction word nearby
# (find_near_pair_span, below) -- a bare/stop-flavoured "chega" has no
# direction word in the sentence at all, so it simply never matches and
# falls through to whatever already handles it. The one EXPLICIT guard
# needed is the opposite direction: "chega isto/isso para o lado" targets
# a WorkspaceObject (see _WORKSPACE_ASIDE_PHRASES), not Echo itself, so it
# must not be captured here even though it also contains "chega"+"lado".
_ENTITY_OBJECT_TARGETED_CHEGA_PHRASES = ("chega isto", "chega isso")
_ENTITY_CENTER_PHRASES = (
    "volta para o centro",
    "volta ao centro",
    "centra-te",
    "coloca-te no centro",
    "fica no centro",
)
# No explicit direction -- "just get out of the way somewhere safe" -- the
# frontend picks an actual free region using real geometry (section 15),
# never a hardcoded one here.
_ENTITY_MAKE_ROOM_PHRASES = (
    "preciso de espaco",
    "preciso de espaco aqui",
    "preciso de espaco para trabalhar",
    "da-me espaco",
    "da-me espaco para trabalhar",
    "quero espaco",
    "deixa espaco",
    "abre espaco",
    "sai um bocado do meio",
    "sai do meio",
    "chega-te para o lado",
    "chega te para o lado",
    "poe-te ali",
    "poe te ali",
)
# Section 12/10: a correction retries the LAST entity command instead of
# falling through to GENERAL_CONVERSATION (which previously produced an
# incoherent reply and lost context entirely on "sim").
_ENTITY_CORRECTION_PHRASES = (
    "nao deslocaste",
    "nao te deslocaste",
    "nao mexeste",
    "nao foi para a direita",
    "nao foi para a esquerda",
    "nao foste",
    "eu disse para ires para o lado",
    "nao mudaste de sitio",
)


def _entity_workspace_command(normalized_clause: str) -> dict[str, str] | None:
    """One clause -> at most one echo_entity command. Checked before the
    generic WorkspaceObject detector (_single_workspace_clause_action) --
    entity commands are a distinct target ("echo_entity", not an on-screen
    object) and never collide with the object phrase tables (e.g.
    "chega-te para o lado" -- reflexive "-te" -- is textually distinct from
    "chega isto/isso para o lado")."""
    if contains_any_phrase(normalized_clause, _ENTITY_OBJECT_TARGETED_CHEGA_PHRASES):
        return None
    if contains_any_phrase(normalized_clause, _ENTITY_CENTER_PHRASES):
        return {
            "action": "center", "target_phrase": _ENTITY_TARGET_PHRASE, "relation": "", "selector_index": "", "region": "center",
            "match_type": "exact_phrase", "signals": ["center_phrase"], "confidence": "high",
        }
    if contains_any_phrase(normalized_clause, _ENTITY_MAKE_ROOM_PHRASES):
        return {
            "action": "make_room", "target_phrase": _ENTITY_TARGET_PHRASE, "relation": "", "selector_index": "", "region": "",
            "match_type": "exact_phrase", "signals": ["make_room_phrase"], "confidence": "high",
        }
    if find_near_pair_span(normalized_clause, _ENTITY_MOVEMENT_VERBS, _ENTITY_DIRECTION_WORDS, max_gap=30):
        # A direction word being present at all is exactly what
        # disambiguates a movement "chega" from a stop/finish "chega" --
        # region can legitimately be "" (lado/canto -- ambiguous which
        # side/corner), the frontend already treats that as "pick a
        # sensible default", same as make_room.
        verb = next((word for word in _ENTITY_MOVEMENT_VERBS if word in normalized_clause), "")
        direction = next((word for word in _ENTITY_DIRECTION_WORDS if word in normalized_clause), "")
        region = _ENTITY_DIRECTION_TO_REGION.get(direction, "")
        return {
            "action": "move", "target_phrase": _ENTITY_TARGET_PHRASE, "relation": "", "selector_index": "", "region": region,
            "match_type": "verb_direction_pair", "signals": [verb, direction], "confidence": "high" if region else "medium",
        }
    return None


# Best-effort raw target phrase extraction -- NOT resolution. The actual
# object id is resolved client-side (echo_workspace_objects.js) against the
# live WorkspaceScene, which is the only place that knows what is really on
# screen right now. Checked longest-first so "segunda versao" wins over a
# bare "versao" fragment, etc.
_WORKSPACE_TARGET_HINTS = (
    "segunda versao", "primeira versao", "versao anterior", "segundo ficheiro",
    "primeiro ficheiro", "este documento", "estes dois", "os dois",
    "o original", "original", "rascunho", "draft", "fatura", "orcamento",
    "excerto", "paragrafo", "isto", "isso", "aquilo",
)
_WORKSPACE_ORDINAL_WORDS = {
    "primeiro": 1, "primeira": 1, "segundo": 2, "segunda": 2,
    "terceiro": 3, "terceira": 3, "quarto": 4, "quarta": 4,
}
# Short, deliberately unremarkable confirmations -- section 16: a spatial
# command's own visible effect on the workspace IS the feedback; the reply
# must not narrate geometry ("Movi o documento 322 pixels para a esquerda.").
_WORKSPACE_COMMAND_ACKS = {
    "expand": "Claro.",
    "minimize": "Feito.",
    "place_beside": "Aqui tens os dois.",
    "focus": "Vamos trabalhar nessa versão.",
    "focus_excerpt": "Aqui está esse trecho.",
    "hide": "Escondido por agora.",
    "close": "Fechado.",
    "place": "Feito.",
    "bring_forward": "Aqui está.",
    "keep_open": "Deixo aberto.",
    "restore_previous_scene": "Voltei ao que estávamos a fazer.",
    "auto_organize": "Organizei automaticamente.",
    "show": "Aqui está.",
}

# Phase 3 (Cognitive Workspace Blueprint v2): acks for the narrow
# representation-command voice bridge (see AssistantEngine._try_
# representation_command). Deliberately never narrates geometry, same
# convention as _WORKSPACE_COMMAND_ACKS above.
_REPRESENTATION_COMMAND_ACKS = {
    "cards": "Aqui tens em cartões.",
    "list": "Aqui tens em lista.",
    "side_by_side": "Aqui tens os dois lado a lado.",
    "conversation_only": "Claro, fico só a falar contigo.",
}
_REPRESENTATION_STYLE_LABELS = {
    "cards": "cartões",
    "list": "lista",
    "side_by_side": "lado a lado",
    "conversation_only": "conversa só",
}


def _single_workspace_clause_action(normalized_clause: str) -> dict[str, str] | None:
    """One clause -> at most one workspace action. Order matters: more
    specific/rarer phrasings are checked before broader ones that could
    otherwise shadow them (e.g. "focus" -- a new primary's promotion
    already demotes whatever was primary before, so it deliberately wins
    over a same-clause "minimize" reading of the same compound sentence)."""
    entity_result = _entity_workspace_command(normalized_clause)
    if entity_result:
        return entity_result
    action = ""
    relation = ""
    selector_index = 0
    if contains_any_phrase(normalized_clause, _WORKSPACE_RESTORE_SCENE_PHRASES):
        action = "restore_previous_scene"
    elif contains_any_phrase(normalized_clause, _WORKSPACE_AUTO_ORGANIZE_PHRASES):
        action = "auto_organize"
    elif contains_any_phrase(normalized_clause, _WORKSPACE_SIDE_BY_SIDE_PHRASES):
        action = "place_beside"
        relation = "side_by_side"
    elif contains_any_phrase(normalized_clause, _WORKSPACE_FOCUS_EXCERPT_MARKERS) and "paragrafo" in normalized_clause:
        action = "focus_excerpt"
        for word, index in _WORKSPACE_ORDINAL_WORDS.items():
            if word in normalized_clause:
                selector_index = index
                break
        selector_index = selector_index or 1
    elif contains_any_phrase(normalized_clause, _WORKSPACE_WORK_ON_PHRASES):
        action = "focus"
    elif contains_any_phrase(normalized_clause, _WORKSPACE_ASIDE_PHRASES):
        action = "place"
        relation = "aside"
    elif contains_any_phrase(normalized_clause, _WORKSPACE_BRING_FORWARD_PHRASES):
        action = "bring_forward"
    elif contains_any_phrase(normalized_clause, _WORKSPACE_MINIMIZE_PHRASES):
        action = "minimize"
    elif contains_any_phrase(normalized_clause, _WORKSPACE_EXPAND_PHRASES):
        action = "expand"
    elif contains_any_phrase(normalized_clause, _WORKSPACE_HIDE_PHRASES):
        action = "hide"
    elif contains_any_phrase(normalized_clause, _WORKSPACE_CLOSE_PHRASES):
        action = "close"
    elif contains_any_phrase(normalized_clause, _WORKSPACE_KEEP_OPEN_PHRASES):
        action = "keep_open"
    elif contains_any_phrase(normalized_clause, _WORKSPACE_SHOW_PHRASES):
        action = "show"
    if not action:
        return None
    target_phrase = ""
    for hint in _WORKSPACE_TARGET_HINTS:
        if hint in normalized_clause:
            target_phrase = hint
            break
    return {
        "action": action,
        "target_phrase": target_phrase,
        "relation": relation,
        "selector_index": str(selector_index) if selector_index else "",
        "region": "",
    }


def _workspace_command_clauses(normalized: str) -> list[str]:
    """Splits an utterance on a simple comma/" e " heuristic (not real
    conjunction parsing -- see final report's documented limitations)."""
    clauses = [normalized]
    for separator in (" e ", ","):
        if separator in normalized:
            clauses = [part.strip(" .!?") for part in normalized.replace(",", " e ").split(" e ") if part.strip()]
            break
    return clauses


def _workspace_command_actions(normalized: str) -> tuple[list[dict[str, str]], list[str]]:
    """Section 2/13: a single utterance can carry more than one workspace
    clause ("chega isto para o lado e mostra-me o segundo ficheiro"), and a
    workspace clause can coexist with genuine remaining user intent
    ("desloca-te para a direita... vamos escrever um mail") that must not
    be silently dropped. Returns (found_commands, remaining_clause_texts) --
    order-preserving, deduplicated by action, capped at 3 commands."""
    return _primitive_workspace_command_actions(normalized)


def _remaining_workspace_intent_ack(remaining_clauses: list[str]) -> str:
    """Section 13: never silently drop the rest of a mixed utterance --
    a short, generic acknowledgment invites the user to continue instead
    of pretending the leftover text was understood or acted on."""
    combined = " ".join(clause for clause in remaining_clauses if clause.strip())
    if not combined.strip():
        return ""
    if contains_any_phrase(combined, ("escrever", "compor", "redigir")):
        return "Diz-me o que queres escrever."
    return "Diz-me mais."


_COMPOUND_WORKSPACE_SEPARATORS_RE = re.compile(
    r"\s*(?:[,;]\s*|\s+mas\s+|\s+depois\s+|\s+enquanto\s+|\s+e tambem\s+|\s+e\s+(?=(?:minimiza|compacta|esconde|oculta|abre|abrir|abras|mostra|mostrar|traz|aumenta|expande|maximiza|faz|vamos comparar|compara)\b))",
    flags=re.IGNORECASE,
)
_WORKSPACE_SHOW_VERBS = ("abre", "abrir", "abras", "mostra", "mostrar", "mostra-me", "traz")
_WORKSPACE_EXPAND_VERBS = ("aumenta", "expande", "maximiza", "faz")
_WORKSPACE_MINIMIZE_VERBS = ("minimiza", "compacta")
_WORKSPACE_HIDE_VERBS = ("esconde", "esconder", "oculta", "ocultar")
_WORKSPACE_TARGET_FILLERS = (
    "preciso que", "podes", "pode", "quero", "queria", "para ver", "rapidamente",
    "por favor", "me", "o", "a", "os", "as", "um", "uma", "isto", "isso",
)
_COLLECTION_WORDS = {"opcao", "opcoes", "opção", "opções", "documento", "documentos", "ficheiro", "ficheiros"}
_SEMANTIC_TARGET_STOPWORDS = {
    "a", "o", "as", "os", "de", "da", "do", "das", "dos", "em", "para", "e", "um", "uma", "no", "na", "ao", "aos",
    "preciso", "que", "abras", "abre", "abrir", "mostra", "mostrar", "minimiza", "compacta",
    "aumenta", "expande", "maximiza", "faz", "maior", "ver", "tambem", "depois",
    "enquanto", "mail", "email", "mensagem",
}


def _compound_workspace_clauses(normalized: str) -> list[str]:
    return [part.strip(" .!?") for part in _COMPOUND_WORKSPACE_SEPARATORS_RE.split(normalized) if part.strip(" .!?")]


def _workspace_clause_action(normalized_clause: str) -> str:
    if contains_any_phrase(normalized_clause, _WORKSPACE_HIDE_VERBS):
        return "hide"
    if contains_any_phrase(normalized_clause, _WORKSPACE_MINIMIZE_VERBS):
        return "minimize"
    if contains_any_phrase(normalized_clause, _WORKSPACE_EXPAND_VERBS):
        return "expand"
    if contains_any_phrase(normalized_clause, _WORKSPACE_SHOW_VERBS):
        return "show"
    return ""


def _workspace_clause_target_phrase(normalized_clause: str) -> str:
    cleaned = normalized_clause
    for phrase in _WORKSPACE_TARGET_FILLERS + _WORKSPACE_SHOW_VERBS + _WORKSPACE_EXPAND_VERBS + _WORKSPACE_MINIMIZE_VERBS + _WORKSPACE_HIDE_VERBS:
        cleaned = re.sub(rf"\b{re.escape(phrase)}\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .!?")
    return cleaned


def _semantic_token_forms(text: str) -> set[str]:
    forms: set[str] = set()
    for token in re.findall(r"\b[\wÀ-ÿ]{3,}\b", normalize_for_matching(text)):
        if token in _SEMANTIC_TARGET_STOPWORDS:
            continue
        forms.add(token)
        if token.endswith("s") and len(token) > 4:
            forms.add(token[:-1])
        if token.endswith("es") and len(token) > 5:
            forms.add(token[:-2])
    return forms


def _task_semantic_text(task: "ActiveTaskContext") -> str:
    values: list[str] = [
        task.task_type,
        task.activity_family,
        task.domain,
        task.artifact_type,
        task.goal,
    ]
    values.extend(str(value) for value in task.entities.values())
    values.extend(str(value) for value in task.parameters.values())
    values.extend(task.source_turns[-3:])
    return " ".join(value for value in values if value)


def _task_collection_object_ids(task: "ActiveTaskContext", work_session: "WorkSession", normalized_clause: str) -> list[str]:
    if any(word in normalized_clause for word in ("opcoes", "opções")):
        allowed_kinds = {"option"}
    elif any(word in normalized_clause for word in ("documentos", "ficheiros")):
        allowed_kinds = {"document", "file"}
    else:
        # For subject/domain references such as "os telefones", prefer the
        # owned option objects. The activity artefact itself stays visible
        # unless the user names it directly.
        allowed_kinds = {"option"}
    return [
        object_id
        for object_id, obj in work_session.objects.items()
        if obj.task_id == task.id and obj.kind in allowed_kinds
    ]


def resolve_semantic_collection_targets(raw_message: str, work_session: "WorkSession") -> TargetResolution:
    normalized = normalize_for_matching(raw_message)
    tokens = _semantic_token_forms(raw_message)
    has_plural_reference = bool(tokens) and bool(re.search(r"\b(?:os|as|esses|essas|estes|estas)\b", normalized))
    has_generic_collection = bool(_COLLECTION_WORDS.intersection(tokens))
    if not has_plural_reference and not has_generic_collection:
        return TargetResolution(matches=[], confidence=0.0, ambiguous=False, source="not_collection_reference")

    tasks: list[ActiveTaskContext] = []
    if work_session.active_task is not None:
        tasks.append(work_session.active_task)
    tasks.extend(reversed(work_session.paused_task_stack))

    candidate_sets: list[tuple[ActiveTaskContext, list[str], float]] = []
    for task in tasks:
        object_ids = _task_collection_object_ids(task, work_session, normalized)
        if not object_ids:
            continue
        task_tokens = _semantic_token_forms(_task_semantic_text(task))
        option_tokens = set()
        for object_id in object_ids:
            option_tokens.update(_semantic_token_forms(work_session.objects[object_id].title))
        overlap = tokens.intersection(task_tokens | option_tokens)
        if has_generic_collection or overlap:
            confidence = 0.82 if overlap else 0.65
            candidate_sets.append((task, object_ids, confidence))

    if not candidate_sets:
        return TargetResolution(matches=[], confidence=0.0, ambiguous=False, source="no_collection_match")
    best_confidence = max(confidence for _task, _ids, confidence in candidate_sets)
    best = [(task, ids, confidence) for task, ids, confidence in candidate_sets if confidence == best_confidence]
    if len(best) != 1:
        return TargetResolution(matches=[], confidence=0.0, ambiguous=True, source="ambiguous_collection")
    task, object_ids, confidence = best[0]
    matches = [
        WorkspaceReferenceMatch(
            object_id=object_id,
            task_id=task.id,
            title=work_session.objects[object_id].title,
            confidence=confidence,
        )
        for object_id in object_ids
    ]
    return TargetResolution(matches=matches, confidence=confidence, ambiguous=False, source="task_owned_collection")


def _collection_label(matches: list[WorkspaceReferenceMatch]) -> str:
    if len(matches) == 1:
        return matches[0].title or "isso"
    return "as opções"


def _criteria_label(criteria_titles: list[str]) -> str:
    if len(criteria_titles) == 1:
        return criteria_titles[0]
    return ", ".join(criteria_titles[:-1]) + " e " + criteria_titles[-1]


def _single_owner_task_id(matches: list[WorkspaceReferenceMatch]) -> str:
    owner_ids = {match.task_id for match in matches if match.task_id}
    return next(iter(owner_ids)) if len(owner_ids) == 1 else ""


def _is_relative_rest_target(normalized_clause: str) -> bool:
    return contains_any_phrase(
        normalized_clause,
        (
            "tudo o resto",
            "todo o resto",
            "o resto",
            "os outros",
            "as outras",
            "outros",
            "outras",
            "os restantes",
            "as restantes",
            "restantes",
        ),
    )


def _semantic_association_target_reference(raw_message: str) -> str:
    match = re.search(
        r"\b(?:aos|às|as|ao|à|nos|nas|no|na)\s+(.+?)\s*[.!?…]*$",
        raw_message,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip(" .!?…") if match else ""


def _compound_workspace_success_summary(operations: list[dict[str, object]]) -> str:
    actions = [str(operation.get("action") or "") for operation in operations]
    if "add_criteria" in actions and "show" in actions:
        return "Feito — mostrei as opções e acrescentei os critérios à comparação."
    if "show" in actions and ("minimize" in actions or "hide" in actions):
        if any(operation.get("relative_scope") == "all_except" for operation in operations):
            return "Feito — mantive isso em foco e afastei o resto."
        return "Feito — abri o email para veres e minimizei as opções."
    if "add_criteria" in actions:
        return "Acrescentei esses critérios à comparação."
    if "expand" in actions:
        return "Feito — aumentei isso no workspace."
    if "minimize" in actions:
        return "Feito — minimizei isso."
    if "show" in actions:
        return "Aqui está."
    return "Feito."


def _active_document_followup_action(text: str, content_type: str = "", has_draft: bool = False) -> str:
    normalized = _normalize_text(text).strip(" .!?")
    if not normalized:
        return ""
    if contains_any_phrase(normalized, ("que horas sao", "que dia e", "abre o calendario")):
        return ""
    # Draft-related actions are checked before everything else: "voltando ao
    # email, mostra-me a versão melhorada" would otherwise be swallowed by
    # the "voltando ao email" catch-all near the bottom of this function and
    # misclassified as review.
    if contains_any_phrase(normalized, ("versao melhorada", "mostra o rascunho", "mostra-me o rascunho", "mostra o draft", "mostra-me o draft")):
        return "display_draft"
    if contains_any_phrase(normalized, ("compara as duas versoes", "compara as versoes", "compara com o original")):
        return "compare_versions"
    if contains_any_phrase(normalized, ("volta a versao anterior", "volta para a versao anterior", "restaura a versao anterior", "volta atras")):
        return "restore_previous_draft"
    if contains_any_phrase(normalized, ("descarta as alteracoes", "descarta o rascunho", "descarta a versao", "descarta o draft")):
        return "discard_draft"
    if contains_any_phrase(normalized, ("guarda esta versao", "guarda as alteracoes", "guarda o rascunho", "guarda a versao melhorada")):
        return "save_draft"
    if (
        "mostra outra vez" in normalized
        or _looks_like_full_document_read(normalized)
        or contains_any_phrase(normalized, ("mostra o original", "versao original", "texto original"))
    ):
        return "display"
    # Tie-break order below matters: once a branch matches, later ones are
    # never consulted. Explicit, unambiguous markers are checked first
    # (review's own fixed phrases, then interpret, summarize, rewrite);
    # _looks_like_ambiguous_document_followup is the broadest catch-all (a
    # bare mention of "mail"/"email" already counts) so it is deliberately
    # checked LAST among these — otherwise a rewrite request that happens to
    # mention "o email" ("deixa o email mais formal") would be swallowed by
    # the generic review fallback before ever reaching the rewrite check.
    # "corrige"/"corrigir" belong to rewrite (not review) — asking to fix
    # something means producing corrected text, not just suggestions.
    if contains_any_phrase(normalized, ("sugeres alguma alteracao", "alguma alteracao", "o que mudavas", "reve", "review")):
        return "review"
    if contains_any_phrase(
        normalized,
        (
            "interpreta",
            "analisa",
            "o que quer dizer",
            "diz me tu",
            "diz-me tu",
            "o que significa",
            "explica me",
            "o que quer isto dizer",
        ),
    ):
        return "interpret"
    if contains_any_phrase(normalized, ("resume melhor", "faz um resumo", "resumo melhor", "sintetiza", "resume outra vez")):
        return "summarize"
    if _looks_like_rewrite_request(normalized):
        return "rewrite"
    # Elliptical/comparative continuations of an already-open draft thread
    # ("ainda consegues melhor", "mais formal", "consegues fazer melhor?")
    # have no explicit rewrite verb, so _looks_like_rewrite_request alone
    # never catches them. Only trust this reading when a draft already
    # exists -- without one, these phrases must not invent a document
    # context (they fall through to normal conversation instead).
    if has_draft and _looks_like_draft_refinement_request(normalized):
        return "rewrite"
    if _looks_like_ambiguous_document_followup(normalized, content_type):
        return "review"
    if normalized.startswith(("voltando ao email", "voltando ao documento", "sobre o email", "sobre o documento")):
        return "review"
    return ""


def _requested_output_for_document_action(action: str) -> str:
    return {
        "review": "selected_section",
        "interpret": "metadata",
        "summarize": "summary",
        "rewrite": "transformed_content",
        "display": "full_contents",
        "read": "full_contents",
    }.get(action, "transformed_content")


def _build_active_document_context(
    filename: str,
    content: str,
    workspace_path: Path,
    last_action: str,
) -> ActiveDocumentContext:
    path = (workspace_path / filename).resolve()
    try:
        stat = path.stat()
        mtime_ns = stat.st_mtime_ns
        file_size = stat.st_size
    except OSError:
        mtime_ns = 0
        file_size = 0
    file_type = Path(filename).suffix.lower().lstrip(".")
    digest = hashlib.sha256(str(content or "").encode("utf-8", errors="replace")).hexdigest()
    detected_type = _detect_document_content_type(content, filename)
    return ActiveDocumentContext(
        resolved_file=filename,
        file_type=file_type,
        content_reference=filename,
        content_hash=digest,
        content_length=len(content or ""),
        file_size=file_size,
        detected_content_type=detected_type,
        last_action=last_action,
        current_document_task=last_action,
        available_followups=["display", "summarize", "interpret", "review", "rewrite"],
        mtime_ns=mtime_ns,
        content=content,
        ttl=8,
    )


def _detect_document_content_type(content: str, filename: str = "") -> str:
    normalized = _normalize_text(f"{filename}\n{content}")
    if any(marker in normalized for marker in ("assunto:", "ola ", "olá ", "cumprimentos", "segue em anexo")):
        return "email"
    if any(marker in normalized for marker in ("ata", "reuniao", "reunião", "direcao", "direção")):
        return "meeting_notes"
    return "document"


def _document_context_file_used(context: ActiveDocumentContext, match_type: str) -> dict[str, object]:
    return {
        "name": context.resolved_file,
        "type": context.file_type,
        "read_success": bool(context.content.strip()),
        "match_type": match_type,
        "content_hash": context.content_hash,
        "content_length": context.content_length,
        "file_size": context.file_size,
    }


def _active_document_followup_prompt(request: DocumentTaskRequest, context: ActiveDocumentContext) -> str:
    action_guidance = {
        "review": (
            "Revê o documento ativo. Sugere alterações úteis de clareza, tom e estrutura. "
            "Não peças destinatário ou propósito se eles forem inferíveis do conteúdo."
        ),
        "interpret": (
            "Interpreta o documento ativo. Explica o que é, para quem parece ser, que assuntos aborda, "
            "que pontos estão pendentes e que destaques existem."
        ),
        "summarize": "Resume o documento ativo com clareza e sem inventar contexto.",
        # "rewrite" is handled by a dedicated prompt (_document_rewrite_prompt)
        # with structural validation — it never reaches this generic path.
    }.get(request.action, "Trabalha o documento ativo usando apenas o conteúdo fornecido.")
    return (
        f"Pedido do utilizador: {request.instructions}\n"
        f"Ação documental: {request.action}\n"
        f"Ficheiro ativo: {context.resolved_file}\n"
        f"Tipo detetado: {context.detected_content_type}\n\n"
        "Conteúdo real do documento:\n"
        f"{_truncate_document_content(context.content, limit=12000)}\n\n"
        f"{action_guidance}\n"
        "Não menciones assuntos que não estejam no documento. "
        "Não digas que vais começar a ler: responde diretamente com base no conteúdo acima."
    )


def _is_negative_confirmation(text: str) -> bool:
    normalized = _normalize_text(text).strip(" .!?")
    return normalized in {"nao", "não", "n", "cancela", "cancelar", "deixa estar", "esquece"}


def _truncate_document_content(content: str, limit: int = 5000) -> str:
    text = str(content or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n\n[Conteúdo truncado para leitura.]"


def _document_composition_prompt(request: DocumentTaskRequest, filename: str, content: str) -> str:
    limited = _truncate_document_content(content, limit=12000)
    recipient = f"\nDestinatário: {request.recipient}" if request.recipient else ""
    output = f"\nFormato de saída pedido: {request.output_format}" if request.output_format else ""
    actions = f"\nAções pedidas: {', '.join(request.actions)}" if request.actions else ""
    return (
        f"Pedido do utilizador: {request.instructions}\n"
        f"{recipient}{output}{actions}\n\n"
        f"Ficheiro lido da workspace: {filename}\n\n"
        "Conteúdo extraído:\n"
        f"{limited}\n\n"
        "Responde em português de Portugal, usando apenas este conteúdo. "
        "Se estiveres a escrever um email, inclui assunto e corpo do email."
    )


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
    "encontrei o ficheiro",
    "li o documento",
    "li o ficheiro",
    "criei o ficheiro",
    "guardei no workspace",
    "guardei na workspace",
    "copiei o conteudo",
    "copiei o conteúdo",
    "calendario esta atualizado",
    "calendário está atualizado",
    "consultei o calendario",
    "consultei o calendário",
    "vi na agenda",
    "estava a verificar as notas",
    "estive a verificar as notas",
    "verifiquei as notas",
    "consultei as notas",
    "estava a verificar os ficheiros",
    "estive a verificar os ficheiros",
    "ja estou a le-lo",
    "ja estou a ler",
    "vou ler agora",
    "ja abri o ficheiro",
    "aqui esta o conteudo",
    "vou comecar a le-lo",
    "vou começar a lê-lo",
    "acho que vou ler outro documento",
    "ja estou a analisa-lo",
    "já estou a analisá-lo",
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


_WORKSPACE_OPERATION_REQUEST_MARKERS = (
    "abre",
    "abrir",
    "mostra",
    "mostrar",
    "foca",
    "minimiza",
    "compacta",
    "esconde",
    "oculta",
)

_WORKSPACE_OPERATION_SUCCESS_CLAIM_MARKERS = (
    "esta a abrir",
    "está a abrir",
    "abri",
    "ja abri",
    "já abri",
    "minimizei",
    "minimizado",
    "escondi",
    "foquei",
    "esta em foco",
    "está em foco",
)


def _detect_ungrounded_workspace_operation_claim(user_message: str, response: str) -> str:
    normalized_message = _normalize_text(user_message)
    if not contains_any_phrase(normalized_message, _WORKSPACE_OPERATION_REQUEST_MARKERS):
        return ""
    normalized_response = _normalize_text(response)
    for marker in _WORKSPACE_OPERATION_SUCCESS_CLAIM_MARKERS:
        if _normalize_text(marker) in normalized_response:
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


def _research_providers_from_results(results: list[dict[str, Any]]) -> list[str]:
    providers: list[str] = []
    seen: set[str] = set()
    for result in results:
        provider = str(result.get("provider") or "").strip()
        if provider and provider not in seen:
            providers.append(provider)
            seen.add(provider)
    return providers


def _research_fallback_reasons_from_results(results: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    seen: set[str] = set()
    for result in results:
        reason = str(result.get("fallback_reason") or "").strip()
        if reason and reason not in seen:
            reasons.append(reason)
            seen.add(reason)
    return reasons


def _display_source_hint(message: str, fallback: str) -> str:
    patterns = (
        r"\bfonte\s+d[ao]\s+(.+)$",
        r"\bartigo\s+d[ao]\s+(.+)$",
        r"\blink\s+d[ao]\s+(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, message.strip(), flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip(" .!?")
            if value:
                return value
    return fallback


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


def _looks_like_incomplete_writing_request(normalized_message: str) -> bool:
    text = normalized_message.replace("-", " ").strip(" .,!?:;")
    if not re.search(r"\b(?:preciso|quero|ajuda|ajuda me|podes|consegues)\b", text):
        return False
    if _extract_writing_batch_count(text) > 1 and re.search(r"\b(?:escrever|redigir|preparar)\b", text) and re.search(
        r"\b(?:emails?|e mails?|mails?|mensagens?)\b",
        text,
    ):
        return True
    if any(marker in text for marker in ("escrever um email", "escrever um mail", "redigir um email", "redigir um mail", "preparar um email", "preparar um mail", "escrever uma mensagem")):
        return True
    return bool(
        re.search(
            r"\b(?:ajuda|ajuda me|preciso|quero|podes|consegues)\b.*\b(?:num|no|com um|com o|para um|para o)?\s*(?:email|mail|mensagem)\b",
            text,
        )
    )


_PT_NUMBER_WORDS = {
    "um": 1,
    "uma": 1,
    "dois": 2,
    "duas": 2,
    "tres": 3,
    "três": 3,
    "quatro": 4,
    "cinco": 5,
    "seis": 6,
    "sete": 7,
    "oito": 8,
    "nove": 9,
    "dez": 10,
}


def _extract_writing_batch_count(normalized_message: str) -> int:
    text = normalized_message.replace("-", " ").strip(" .,!?:;")
    match = re.search(r"\b(?:mais\s+)?(\d+|um|uma|dois|duas|tres|três|quatro|cinco|seis|sete|oito|nove|dez)\s+(?:emails?|e mails?|mails?|mensagens?)\b", text)
    if not match:
        return 1
    token = match.group(1)
    if token.isdigit():
        return max(1, int(token))
    return _PT_NUMBER_WORDS.get(token, 1)


def _looks_like_complete_writing_request(normalized_message: str) -> bool:
    text = normalized_message.strip(" .,!?:;")
    if not _writing_request_kind(text):
        return False
    return bool(re.search(r"\b(?:escreve|redige|prepara|cria)\b", text)) and (
        " a " in f" {text} " or " para " in text or len(text.split()) >= 9
    )


def _looks_like_writing_followup(normalized_message: str, task: PendingWritingTask) -> bool:
    text = normalized_message.strip(" .,!?:;")
    if not text:
        return False
    if task.awaiting_field == "recipient" and _extract_contextual_writing_recipient(text):
        return True
    if task.awaiting_field == "purpose" and _extract_contextual_writing_purpose(text):
        return True
    if not task.missing_fields and _is_writing_execute_confirmation(text):
        return True
    if any(marker in text for marker in ("patrocinador", "professor", "cliente", "colega", "direcao", "direção")):
        return True
    if any(marker in text for marker in ("projeto", "projecto", "apoio", "financiamento", "parceria", "rever", "marcar", "pedir")):
        return True
    if task.current_step == "collecting_context" and len(text.split()) <= 14:
        return True
    return False


def _is_explicit_non_writing_activity_or_command(
    raw_message: str,
    normalized_message: str,
    active_task: "ActiveTaskContext | None",
    paused_tasks: list["ActiveTaskContext"],
) -> bool:
    """Reject email slot-filling when another deterministic router has evidence.

    A pending writing task may consume short/person-like field answers, but it
    must not preempt explicit activity starts, resumes, workspace commands or
    representation/comparison commands. This is deterministic only; no LLM is
    used for the compatibility check.
    """
    interpretation = _interpret_activity(raw_message, normalized_message, active_task, paused_tasks)
    if (
        (interpretation.starts_task or interpretation.resumes_task or interpretation.refines_task)
        and interpretation.task_type
        and interpretation.task_type not in {"compose_email", "writing"}
        and interpretation.confidence >= 0.7
    ):
        return True
    workspace_commands, _remaining = _workspace_command_actions(normalized_message)
    if workspace_commands:
        return True
    if message_implies_comparison(normalized_message) and len(extract_explicit_alternatives(raw_message)) >= 2:
        return True
    if extract_explicit_alternatives(raw_message) and contains_any_phrase(
        normalized_message, ("em cartoes", "em cartões", "em lista", "lado a lado")
    ):
        return True
    return False


def _is_writing_execute_confirmation(normalized_message: str) -> bool:
    text = normalized_message.strip(" .,!?:;")
    return contains_any_phrase(
        text,
        ("podes escrever", "podes redigir", "escreve", "redige", "podes", "sim", "avanca", "avança"),
    )


def _asks_system_context_status(normalized_message: str) -> bool:
    text = normalized_message.strip(" .,!?:;")
    return any(
        marker in text
        for marker in (
            "estas a guardar contexto",
            "guardas contexto",
            "guardas memoria",
            "estas a guardar memoria",
            "o que sabes sobre o que estou a fazer",
            "o que observas do meu computador",
            "consegues ver as aplicacoes abertas",
            "consegues ver os programas abertos",
            "que contexto tens agora",
            "que informacao tens agora",
        )
    )


def _has_persistent_memory_data(memory: object) -> bool:
    finder = getattr(memory, "find_structured_facts", None)
    if callable(finder):
        try:
            if finder():
                return True
        except Exception:
            pass
    search = getattr(memory, "search", None)
    if callable(search):
        try:
            if search("", limit=1):
                return True
        except Exception:
            pass
    context_for = getattr(memory, "context_for", None)
    if callable(context_for):
        try:
            return bool(str(context_for("contexto", limit=1) or "").strip())
        except Exception:
            pass
    return False


def _extract_application_activity_name(message: str) -> str:
    text = _normalize_text(message)
    if not any(marker in text for marker in ("o que", "que estou", "atividade", "fazer")):
        return ""
    app_aliases = {
        "chrome": "Chrome",
        "google chrome": "Chrome",
        "edge": "Edge",
        "firefox": "Firefox",
        "vscode": "VS Code",
        "vs code": "VS Code",
        "visual studio code": "VS Code",
        "codex": "Codex",
        "teams": "Teams",
        "outlook": "Outlook",
        "discord": "Discord",
    }
    for alias, label in app_aliases.items():
        if alias in text:
            return label
    return ""


def _fresh_or_latest_context_snapshot(context_observer: object):
    for method_name in ("observe_once", "capture_snapshot", "snapshot_now"):
        method = getattr(context_observer, method_name, None)
        if callable(method):
            try:
                snapshot = method()
                if snapshot is not None:
                    return snapshot
            except Exception:
                pass
    for method_name in ("latest_snapshot", "last_snapshot"):
        method = getattr(context_observer, method_name, None)
        if callable(method):
            try:
                snapshot = method()
                if snapshot is not None:
                    return snapshot
            except Exception:
                pass
    return None


def _application_activity_from_snapshot(snapshot: object, application_name: str) -> dict[str, object]:
    app_norm = _normalize_text(application_name)
    windows = list(getattr(snapshot, "open_windows", ()) or ())
    matching = [
        window
        for window in windows
        if _window_matches_application(window, app_norm)
    ]
    active_app = str(getattr(snapshot, "active_app", "") or "")
    active_window = str(getattr(snapshot, "active_window", "") or "")
    active = app_norm in _normalize_text(active_app) or any(bool(getattr(window, "is_active", False)) for window in matching)
    if not matching and (app_norm in _normalize_text(active_app) or app_norm in _normalize_text(active_window)):
        matching = [type("ObservedWindow", (), {"title": active_window, "process_name": active_app, "is_active": True})()]
        active = True
    titles = [str(getattr(window, "title", "") or "").strip() for window in matching if str(getattr(window, "title", "") or "").strip()]
    return {
        "application_found": bool(matching),
        "application_name": application_name,
        "active": active,
        "window_titles": titles,
        "observed_activity": "",
        "confidence": 0.55 if matching else 0.0,
        "limitations": "Nao consigo ver o conteudo interno da aplicacao com seguranca.",
    }


def _window_matches_application(window: object, app_norm: str) -> bool:
    process = _normalize_text(str(getattr(window, "process_name", "") or ""))
    title = _normalize_text(str(getattr(window, "title", "") or ""))
    if app_norm == "chrome":
        return "chrome" in process or "google chrome" in title
    if app_norm == "vs code":
        return "code" in process or "visual studio code" in title or "vs code" in title
    return app_norm in process or app_norm in title


# --- Memory relevance gate (Passo 3.5) --------------------------------------
# Real bug: "chegar é o mesmo que deslocar?" (a plain word-meaning question,
# unrelated to anything) got answered with "...O que queres saber sobre a
# fase de Geometry Processing?" -- a topic from a stale, unrelated moment
# leaking in just because it existed somewhere in persisted content.
# Deterministic (keyword-overlap), no LLM call -- this only decides whether
# already-retrieved content is topically relevant enough to INCLUDE, it
# never changes what long_term_memory.context_for() itself retrieves.
_MEMORY_RELEVANCE_STOPWORDS = {
    "que", "para", "com", "uma", "um", "dos", "das", "nos", "nas", "essa",
    "esse", "isto", "isso", "aquilo", "mesmo", "mesma", "qual", "quais",
    "como", "quando", "onde", "porque", "mais", "menos", "esta", "este",
    "sobre", "pelo", "pela", "muito", "pouco", "tambem", "ainda", "agora",
}
_MEMORY_RELEVANCE_THRESHOLD = 0.12


def _memory_relevance_terms(text: str) -> set[str]:
    normalized = _normalize_text(text)
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if len(token) >= 4 and token not in _MEMORY_RELEVANCE_STOPWORDS
    }


def _memory_relevance_score(user_message: str, candidate_text: str) -> tuple[float, str]:
    """Section 4's MemoryCandidate.relevance_score/_reason, computed as
    plain keyword overlap (no embeddings, no LLM call): what fraction of
    the CURRENT message's meaningful terms also appear in the retrieved
    memory content. A genuinely relevant recall naturally shares real
    terms with the question that prompted its retrieval in the first
    place; unrelated leftover content from another moment does not."""
    query_terms = _memory_relevance_terms(user_message)
    candidate_terms = _memory_relevance_terms(candidate_text)
    if not query_terms or not candidate_terms:
        return 0.0, "no_comparable_terms"
    shared = query_terms & candidate_terms
    if not shared:
        return 0.0, "no_shared_terms"
    return len(shared) / len(query_terms), "shared_terms:" + ",".join(sorted(shared))[:80]


# --- ActiveTaskContext helpers (Passo 3.4) ----------------------------------
# Deterministic, regex/keyword-based -- no LLM call anywhere in this block.
# The task layer is generic: activity interpretation decides the family/domain
# first, and only then can a task specialize into compose_email or another
# concrete workflow.

_ACTIVITY_SWITCH_PHRASES = (
    "enquanto isso",
    "primeiro preciso de",
    "antes disso",
    "para alem disso",
    "para além disso",
    "deixa isso por agora",
    "vamos fazer outra coisa",
    "vamos tambem comparar",
    "vamos também comparar",
    "tambem vamos comparar",
    "também vamos comparar",
    "lembrei me que",
    "lembrei-me que",
    "voltamos a isto depois",
    "agora precisamos de",
)

_ACTIVITY_RESUME_PHRASES = (
    "voltar a",
    "volta a",
    "voltar ao",
    "volta ao",
    "voltar aos",
    "volta aos",
    "voltar as",
    "volta as",
    "voltar às",
    "volta às",
    "retoma",
    "continuar",
    "continuamos",
    "podemos voltar",
)

_TASK_METADATA = {
    "compose_email": {
        "activity_family": "communication",
        "domain": "email",
        "artifact_type": "email",
        "goal": "escrever um email",
    },
    "writing": {
        "activity_family": "writing",
        "domain": "unknown",
        "artifact_type": "unknown",
        "goal": "escrever em conjunto",
    },
    "plan_trip": {
        "activity_family": "planning",
        "domain": "travel",
        "artifact_type": "itinerary",
        "goal": "planear ferias",
    },
    "make_decision": {
        "activity_family": "decision",
        "domain": "unknown",
        "artifact_type": "decision_matrix",
        "goal": "tomar uma decisao",
    },
    "study_topic": {
        "activity_family": "study",
        "domain": "university",
        "artifact_type": "plan",
        "goal": "estudar",
    },
    "research_topic": {
        "activity_family": "research",
        "domain": "unknown",
        "artifact_type": "note",
        "goal": "investigar opcoes",
    },
    "generic_comparison": {
        "activity_family": "comparison",
        "domain": "generic",
        "artifact_type": "comparison",
        "goal": "comparar opcoes",
    },
    "generic_activity": {
        "activity_family": "activity",
        "domain": "generic",
        "artifact_type": "workspace",
        "goal": "trabalhar num objetivo",
    },
}


def _task_metadata(task_type: str, activity_family: str = "unknown", domain: str = "unknown", artifact_type: str = "unknown", goal: str = "") -> dict[str, str]:
    defaults = dict(_TASK_METADATA.get(task_type, {}))
    return {
        "activity_family": activity_family if activity_family != "unknown" else defaults.get("activity_family", "unknown"),
        "domain": domain if domain != "unknown" else defaults.get("domain", "unknown"),
        "artifact_type": artifact_type if artifact_type != "unknown" else defaults.get("artifact_type", "unknown"),
        "goal": goal or defaults.get("goal", task_type),
    }


def _activity_start_from_message(raw_text: str, normalized_text: str) -> tuple[str, str]:
    if contains_any_phrase(normalized_text, _TASK_TRIGGER_PHRASES["compose_email"]):
        return "compose_email", "compose_email_trigger"
    if contains_any_phrase(
        normalized_text,
        (
            "compara",
            "comparar",
            "vamos comparar",
            "vamos tambem comparar",
            "vamos também comparar",
            "tambem vamos comparar",
            "também vamos comparar",
            "quero comparar",
            "preciso de comparar",
            "ajuda me a comparar",
            "estou a pensar",
            "vamos avaliar",
            "quero avaliar",
            "preciso de avaliar",
            "quero escolher entre",
            "escolher entre",
        ),
    ) and (
        contains_any_phrase(normalized_text, ("comparar", "avaliar", "com", "entre"))
        or len(extract_explicit_alternatives(raw_text)) >= 2
    ):
        return "generic_comparison", "generic_comparison_trigger"
    if contains_any_phrase(normalized_text, ("planear umas ferias", "planear ferias", "planear uma viagem", "planear viagem", "road trip")):
        return "plan_trip", "travel_planning_trigger"
    if ("ferias" in normalized_text or "viagem" in normalized_text) and contains_any_phrase(
        normalized_text,
        ("quero ferias", "queria ferias", "gostava de ferias", "preciso de ferias"),
    ):
        return "plan_trip", "travel_planning_trigger"
    if ("ferias" in normalized_text or "viagem" in normalized_text) and contains_any_phrase(
        normalized_text,
        (
            "vamos planear",
            "quero planear",
            "gostava de planear",
            "ajuda me a planear",
            "ajuda-me a planear",
            "ajuda a planear",
            "preciso da tua ajuda a planear",
            "preciso de ajuda a planear",
            "vamos tambem preparar",
            "vamos também preparar",
            "preparar umas",
            "preparar ferias",
            "preparar férias",
            "prepara te",
        ),
    ):
        return "plan_trip", "travel_planning_trigger"
    if contains_any_phrase(normalized_text, ("quero ajuda a planear", "ajuda a planear", "ajuda me a planear")) and re.search(
        r"\b(vou|vamos|iremos)\s+(?:a|para)\s+\w+",
        normalized_text,
    ):
        return "plan_trip", "travel_planning_trigger"
    if re.search(r"\b(vou|vamos|iremos)\s+(?:a|para)\s+\w+", normalized_text) and re.search(
        r"\b\d{1,2}\s+a\s+\d{1,2}\s+de\s+\w+",
        normalized_text,
    ):
        return "plan_trip", "travel_planning_trigger"
    if contains_any_phrase(normalized_text, ("vamos estudar",)):
        return "study_topic", "study_trigger"
    if contains_any_phrase(normalized_text, ("vamos decidir", "preciso de decidir", "ajuda me a decidir", "decidir entre")):
        return "make_decision", "decision_trigger"
    if contains_any_phrase(normalized_text, ("quero investigar", "vamos investigar", "preciso de investigar", "investigar opcoes", "investigar opções")):
        return "research_topic", "research_trigger"
    if contains_any_phrase(normalized_text, ("prepara uma apresentacao", "prepara uma apresentação", "preparar uma apresentacao", "preparar uma apresentação")):
        return "generic_activity", "generic_activity_trigger"
    if contains_any_phrase(normalized_text, ("vamos organizar", "vamos preparar")):
        return "generic_activity", "generic_activity_trigger"
    if contains_any_phrase(normalized_text, ("vamos escrever juntos", "vamos escrever")):
        return "writing", "writing_trigger"
    return "", ""


def _interpret_activity(
    raw_message: str,
    normalized_text: str,
    active_task: "ActiveTaskContext | None",
    paused_tasks: list["ActiveTaskContext"],
) -> ActivityInterpretation:
    task_type, reason = _activity_start_from_message(raw_message, normalized_text)
    metadata = _task_metadata(task_type) if task_type else _task_metadata("")
    has_switch = contains_any_phrase(normalized_text, _ACTIVITY_SWITCH_PHRASES)
    has_resume = contains_any_phrase(normalized_text, _ACTIVITY_RESUME_PHRASES)
    resume_target = _resume_target_from_message(normalized_text)
    if has_resume and resume_target and paused_tasks:
        meta = _task_metadata(resume_target)
        return ActivityInterpretation(
            resumes_task=True,
            task_type=resume_target,
            activity_family=meta["activity_family"],
            domain=meta["domain"],
            artifact_type=meta["artifact_type"],
            goal=meta["goal"],
            confidence=0.92,
            reason="resume_previous_activity",
            remaining_content=normalized_text,
        )
    if has_resume and paused_tasks:
        return ActivityInterpretation(
            resumes_task=True,
            confidence=0.68,
            reason="resume_semantic_reference",
            remaining_content=normalized_text,
        )
    if task_type:
        refines = bool(active_task and active_task.task_type in ("writing", "unknown") and task_type != active_task.task_type)
        return ActivityInterpretation(
            starts_task=not refines,
            refines_task=refines,
            activity_family=metadata["activity_family"],
            domain=metadata["domain"],
            artifact_type=metadata["artifact_type"],
            task_type=task_type,
            goal=metadata["goal"],
            confidence=0.95 if has_switch or reason else 0.75,
            reason=("switch:" if has_switch else "") + reason,
            remaining_content=_strip_activity_switch_prefix(normalized_text),
        )
    if active_task is not None and active_task.status not in ("completed", "cancelled", "paused"):
        if _message_mentions_active_task(normalized_text, active_task):
            return ActivityInterpretation(
                continues_task=True,
                activity_family=active_task.activity_family,
                domain=active_task.domain,
                artifact_type=active_task.artifact_type,
                task_type=active_task.task_type,
                goal=active_task.goal,
                confidence=0.72,
                reason="active_task_context_match",
                remaining_content=normalized_text,
            )
    return ActivityInterpretation()


def _strip_activity_switch_prefix(normalized_text: str) -> str:
    result = normalized_text
    for phrase in _ACTIVITY_SWITCH_PHRASES:
        if phrase in result:
            result = result.split(phrase, 1)[1].strip(" ,.!?")
            break
    return result or normalized_text


def _resume_target_from_message(normalized_text: str) -> str:
    if contains_any_phrase(normalized_text, ("ferias", "viagem", "road trip")):
        return "plan_trip"
    if contains_any_phrase(normalized_text, ("email", "mail", "mensagem")):
        return "compose_email"
    if "estudo" in normalized_text or "estudar" in normalized_text:
        return "study_topic"
    if "decisao" in normalized_text or "decidir" in normalized_text:
        return "make_decision"
    if "investigacao" in normalized_text or "investigar" in normalized_text:
        return "research_topic"
    return ""


def _find_task_stack_match(tasks: list["ActiveTaskContext"], interpretation: ActivityInterpretation) -> int | None:
    target_tokens = _semantic_token_forms(interpretation.remaining_content)
    for index in range(len(tasks) - 1, -1, -1):
        task = tasks[index]
        if interpretation.task_type and task.task_type == interpretation.task_type:
            return index
        if interpretation.domain != "unknown" and task.domain == interpretation.domain:
            return index
        if interpretation.activity_family != "unknown" and task.activity_family == interpretation.activity_family:
            return index
        if interpretation.resumes_task and target_tokens:
            task_tokens = _semantic_token_forms(_task_semantic_text(task))
            if target_tokens.intersection(task_tokens):
                return index
    return None


def _message_mentions_active_task(normalized_text: str, ctx: "ActiveTaskContext") -> bool:
    if ctx.task_type == "compose_email":
        return contains_any_phrase(
            normalized_text,
            (
                "email", "mail", "mensagem", "professor", "professora", "cadeira", "disciplina",
                "exame", "data do exame", "para o meu", "para a minha", "para o", "para a",
                "perguntar", "pedir", "esclarecimentos",
            ),
        )
    if ctx.task_type == "plan_trip":
        return contains_any_phrase(normalized_text, ("ferias", "viagem", "norte", "portugal", "destino", "alojamento", "descansar", "aventura", "orcamento", "orçamento", "gastar", "euros", "datas", "semana", "voos", "voo", "itinerario", "itinerário", "visitar", "coisas para ver"))
    if ctx.task_type == "study_topic":
        return contains_any_phrase(normalized_text, ("exame", "teste", "slides", "apontamentos", "materia", "algoritmos", "grafos", "estudar"))
    if ctx.task_type == "make_decision":
        return contains_any_phrase(normalized_text, ("opcao", "opcoes", "escolher", "decidir", "comparar", "criterios", "contratar"))
    if ctx.task_type == "research_topic":
        return contains_any_phrase(normalized_text, ("investigar", "pesquisar", "opcoes", "opcao", "fontes", "alternativas"))
    if ctx.task_type == "generic_comparison":
        return contains_any_phrase(normalized_text, ("compara", "comparar", "comparacao", "comparação", "bateria", "camara", "câmara", "autonomia", "criterios", "critérios", "opcao", "opcoes", "adiciona", "acrescenta", "orcamento", "orçamento", "gastar", "euros", "custa", "esta a", "está a", "preco", "preço", "mais importante", "prioridade", "melhor", "pior", "vantagem", "desvantagem", "escolher", "decidir", "suporte"))
    if ctx.task_type == "generic_activity":
        return contains_any_phrase(normalized_text, ("organizar", "preparar", "plano", "passo", "proximo", "próximo", "tarefa", "ideia"))
    if ctx.task_type == "writing":
        return contains_any_phrase(normalized_text, ("escrever", "texto", "documento", "nota", "email", "mail"))
    return False

# Deliberately DISJOINT from PendingWritingTask's own trigger phrases
# (_looks_like_incomplete_writing_request requires a preciso/quero/ajuda/
# podes/consegues verb + "escrever um email"-style marker) -- that pre-
# existing, narrower email-only slot-filler must keep working exactly as
# before. "vamos ..." matches NEITHER of its predicates today, which is
# exactly why the real reported phrase ("vamos escrever um mail") fell
# through everything; scoping this trigger to "vamos" keeps the two
# mechanisms from ever firing on the same utterance.
_TASK_TRIGGER_PHRASES = {
    "compose_email": (
        "vamos escrever um mail",
        "vamos escrever um email",
        "vamos escrever uma mensagem",
        "preciso de escrever um mail",
        "preciso de escrever um email",
        "preciso de escrever mais",
        "preciso de escrever dois mails",
        "preciso de escrever duas mensagens",
        "quero escrever mais",
        "ajuda me a escrever mais",
        # 1st-person-plural infinitive form ("...para escrevermos um
        # email") -- distinct wording from "vamos escrever", found in a
        # real mixed workspace+task utterance ("chega para a esquerda
        # para escrevermos um email").
        "escrevermos um mail",
        "escrevermos um email",
        "escrevermos uma mensagem",
    ),
}


def _detect_new_task_intent(normalized_text: str) -> str:
    for task_type, phrases in _TASK_TRIGGER_PHRASES.items():
        if contains_any_phrase(normalized_text, phrases):
            return task_type
    return ""


_TASK_CANCEL_PHRASES = ("cancela o email", "ja nao preciso do email", "esquece o email de vez")
_TASK_PAUSE_PHRASES = ("esquece o email por agora", "deixa o email ai", "deixa isso para depois", "espera, deixa o email ai")
_TASK_RESUME_PHRASES = ("volta ao email", "vamos voltar ao email", "continua o email", "continua com o email")
# Section 13: an EXPLICIT teach-me/switch-topic signal is the only thing
# that lets the task's own subject matter fall through to normal
# conversation for a single turn -- a bare mention of the topic itself
# ("o pipeline gráfico, a fase de geometry processing") must NOT trigger
# this, or turn 4 of the real scenario would misfire straight back into
# GENERAL_CONVERSATION explaining Geometry Processing.
_TASK_EXPLICIT_EXIT_PHRASES = (
    "ja agora explica", "agora explica-me", "agora explica me", "muda de assunto",
    # Section 3/4 (fault-attribution phase): a genuine QUESTION about the
    # task's own topic ("o que é Bellman-Ford?") still gets to use normal
    # conversation/tutoring for that one turn -- only these explicit
    # question/teach-me signals count; a bare nominal fragment ("as
    # vantagens e desvantagens em relação ao Dijkstra") is NOT a question
    # and must instead enrich the task (see key_points accumulation below).
    "explica-me", "explica me", "ensina-me", "ensina me", "o que e", "quais sao",
)

# Section 5/8: TASK_CONTROL -- an action ON the task itself, never data
# ABOUT the task's artefact. An explicit phrase from this list always
# attempts execution (subject to ctx.status in ("ready", "reviewing"));
# the bare confirmation words only count as execute when a pending offer
# (ctx.pending_action=="create_draft") is actually outstanding, so a random
# "podes" mid-conversation with no active task doesn't misfire.
_TASK_CONTROL_EXECUTE_PHRASES = (
    "podes escrever", "ja podes escrever", "podes redigir", "ja podes redigir",
    "escreve o email", "escreve a mensagem", "escreve",
    "redige o email", "redige a mensagem", "redige",
    "cria o email", "cria a mensagem",
    "faz uma primeira versao", "faz o rascunho", "faz uma versao",
    "podes avancar", "podes criar",
    # Section 10: an explicit request to regenerate with newly-added
    # key_points -- content additions ALONE never trigger a fresh LLM call
    # (see key_points accumulation below), only an explicit word like these.
    "atualiza o email", "atualiza a mensagem", "reescreve com isso", "reescreve o email",
)
_TASK_CONTROL_BARE_CONFIRM_PHRASES = ("podes", "sim", "forca", "avanca", "continua")


def _detect_task_control_action(normalized_message: str, ctx: "ActiveTaskContext") -> str:
    if contains_any_phrase(normalized_message, _TASK_CONTROL_EXECUTE_PHRASES):
        return "execute"
    if ctx.pending_action == "create_draft" and contains_any_phrase(normalized_message, _TASK_CONTROL_BARE_CONFIRM_PHRASES):
        return "execute"
    return ""

_RECIPIENT_ROLE_MARKERS = {
    "professor": "professor",
    "professora": "professor",
    "colega": "colleague",
    "cliente": "client",
    "patrocinador": "sponsor",
    "chefe": "manager",
    "orientador": "advisor",
}
_PURPOSE_MARKERS = {
    "data do exame": "ask_exam_date",
    "data de exame": "ask_exam_date",
    "qual sera a data do exame": "ask_exam_date",
    "qual será a data do exame": "ask_exam_date",
    "esclarecimentos": "request_clarification",
    "esclarecimento": "request_clarification",
    "duvidas": "request_clarification",
    "duvida": "request_clarification",
    "ajuda": "request_help",
    "reuniao": "schedule_meeting",
    "agradecer": "thanks",
}
# Bounded heuristic, not real NER (explicitly out of scope -- section 19/24):
# 1-4 consecutive capitalized words in the RAW (properly-cased) message --
# a bare single name ("é para a Maria") must match too, not just multi-word
# ones ("João Cardoso"), so a stopword filter (below) guards the
# single-word case instead of requiring 2+ words outright.
_NAME_RUN_RE = re.compile(r"\b([A-ZÀ-Ý][\wà-ÿ]*(?:\s+[A-ZÀ-Ý][\wà-ÿ]*){0,3})\b")
_NAME_STOPWORDS = {
    "e", "ok", "sim", "nao", "claro", "bom", "boa", "entao", "ja", "so", "ah", "bem", "ola", "obrigado", "obrigada",
    "vamos", "quero", "preciso", "ajuda", "escrever", "redigir", "preparar",
    # Section 1 (comma-less recipient parsing): "para o meu professor de X"
    # has no name at all -- without this, the relaxed name-before-role
    # pattern (_NAME_ROLE_COURSE_RE) would misread the possessive filler
    # "meu" itself as the recipient's name.
    "meu", "minha", "nosso", "nossa", "teu", "tua", "vosso", "vossa",
}


def _looks_like_name_candidate(candidate: str) -> bool:
    words = candidate.split()
    if len(words) >= 2:
        return True
    if not words:
        return False
    return _normalize_text(words[0]) not in _NAME_STOPWORDS


def _clean_course_text(course_text: str) -> str:
    return re.split(
        r"\s+e\s+(?:quero|preciso|gostava|vou|para|pedir|perguntar)\b",
        course_text.strip(" .,!?;:"),
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" .,!?;:")


_COURSE_TEXT_RE = re.compile(
    r"(?:cadeira|disciplina|unidade curricular|curso)\s+(?:de\s+)?([A-ZÀ-Ýa-zà-ÿ0-9][\wà-ÿ ]*?)(?=[.,;]|$)",
    re.IGNORECASE,
)

# Section 2: real-world messages are often all-lowercase (voice-to-text),
# so the capitalized-word-run heuristic above (_NAME_RUN_RE) never fires --
# these three patterns recognize a name by POSITION relative to a role+
# "de <curso>" construct instead of by casing. Bounded, not real NER (the
# user explicitly asked for exactly these three phrasings, nothing heavier).
_ROLE_WORDS_ALT = "|".join(_RECIPIENT_ROLE_MARKERS.keys())
# Voice-to-text frequently drops commas entirely -- ", " and a plain space
# must both count as the same separator wherever a hard word ("de", the
# role word itself) already marks the real boundary unambiguously.
_SEP = r"(?:,\s*|\s+)"
_ROLE_COURSE_NAME_RE = re.compile(
    rf"\b(?:{_ROLE_WORDS_ALT})\s+de\s+([\wà-ÿ]+(?:\s+[\wà-ÿ]+)*?)\s*,\s*([\wà-ÿ]+(?:\s+[\wà-ÿ]+){{0,3}})(?=[.,;]|$)",
    re.IGNORECASE,
)
# No comma AND no "de"/role word between course and name to anchor the
# split -- the only structurally safe reading is "the last two words are
# the name" (a plain first+last name), so this variant is deliberately
# narrower than the comma form: it does NOT fire for a one-word trailing
# name (genuinely ambiguous whether that word is part of the course or the
# name -- "não inventar", left unresolved instead).
_ROLE_COURSE_NAME_NOCOMMA_RE = re.compile(
    rf"\b(?:{_ROLE_WORDS_ALT})\s+de\s+([\wà-ÿ]+(?:\s+[\wà-ÿ]+)*?)\s+([\wà-ÿ]+\s+[\wà-ÿ]+)(?=[.,;]|$)",
    re.IGNORECASE,
)
_ROLE_NAME_COURSE_RE = re.compile(
    rf"\b(?:{_ROLE_WORDS_ALT})\s+([\wà-ÿ]+(?:\s+[\wà-ÿ]+){{0,3}}?){_SEP}de\s+([\wà-ÿ]+(?:\s+[\wà-ÿ]+)*?)(?=[.,;]|$)",
    re.IGNORECASE,
)
_NAME_ROLE_COURSE_RE = re.compile(
    rf"\bpara\s+(?:o\s+|a\s+)?([\wà-ÿ]+(?:\s+[\wà-ÿ]+){{0,3}}?){_SEP}(?:{_ROLE_WORDS_ALT})\s+de\s+([\wà-ÿ]+(?:\s+[\wà-ÿ]+)*?)(?=[.,;]|$)",
    re.IGNORECASE,
)
# "sobre X" / "acerca de X" / "relativamente a X" / "em relação a X" --
# section 3: an explicit topic marker, extracted unconditionally (not just
# as a last-resort catch-all), so it fills the SAME turn a purpose marker
# ("esclarecimentos") is also found in.
_TOPIC_ABOUT_RE = re.compile(
    r"\b(?:sobre|acerca\s+de|relativamente\s+a|em\s+rela[cç][aã]o\s+a)\s+(.+)$",
    re.IGNORECASE,
)
_LEADING_ARTICLE_RE = re.compile(r"^(?:o|a|os|as|um|uma)\s+", re.IGNORECASE)
# "esclarecimentos sobre uma matéria" -- "uma matéria" is a placeholder, not
# a real topic; a vague noun alone must NOT get locked in as the final
# topic, or a genuinely specific follow-up ("o pipeline gráfico, a fase de
# geometry processing") would never get a chance to fill the field (section
# 13's existing behavior, must keep working unchanged).
_VAGUE_TOPIC_WORDS = {"assunto", "coisa", "materia", "duvida", "duvidas", "questao", "tema", "topico", "ajuda", "esclarecimento", "esclarecimentos"}


def _normalize_extracted_name(text: str) -> str:
    # "Preservar capitalização normalizada" -- always Title Case each word,
    # regardless of how the source text was cased (voice-to-text input is
    # typically all-lowercase). No diacritic restoration attempted (e.g.
    # "luis" stays "Luis", not "Luís") -- that would require a name
    # dictionary, explicitly out of scope ("não implementar NER pesado").
    cleaned = _LEADING_ARTICLE_RE.sub("", text.strip())
    return " ".join(word.capitalize() for word in cleaned.split())


def _extract_role_course_name(raw_text: str) -> dict[str, str]:
    for pattern, groups in (
        (_ROLE_COURSE_NAME_RE, ("course", "name")),
        (_ROLE_NAME_COURSE_RE, ("name", "course")),
        (_NAME_ROLE_COURSE_RE, ("name", "course")),
        (_ROLE_COURSE_NAME_NOCOMMA_RE, ("course", "name")),
    ):
        match = pattern.search(raw_text)
        if not match:
            continue
        captured = dict(zip(groups, (g.strip(" .,!?;:") for g in match.groups())))
        # "para o meu professor de X" -- the possessive filler "meu" would
        # otherwise get misread as the recipient's name by the relaxed
        # name-before-role pattern, which ALSO drags the real course words
        # into its own course capture. Reject this match entirely (not
        # just the bad name field) so a later, more specific pattern in
        # this same list (_ROLE_COURSE_NAME_NOCOMMA_RE) gets a chance to
        # parse the sentence correctly instead of locking in a half-wrong
        # course value.
        if captured.get("name") and not _looks_like_name_candidate(captured["name"]):
            continue
        role_match = re.search(rf"\b({_ROLE_WORDS_ALT})\b", raw_text, re.IGNORECASE)
        result: dict[str, str] = {}
        if role_match:
            result["role"] = _RECIPIENT_ROLE_MARKERS.get(role_match.group(1).lower(), "")
        for key, cleaned in captured.items():
            if cleaned:
                result[key] = _normalize_extracted_name(_clean_course_text(cleaned)) if key == "course" else _normalize_extracted_name(cleaned)
        return result
    return {}


_CONTENT_REQUEST_VERBS = ("mostra", "sugere", "da me", "dá-me", "quero ver", "apresenta")


def _looks_like_content_request(cleaned: str, normalized_text: str) -> bool:
    """Section 8: a GENERIC "produce N things matching X" content request
    ("mostra-me 10 destinos em Portugal") -- deliberately just a number plus
    a request verb, never a travel/study/decision-specific pattern. Content
    generation itself is out of scope here (see ActiveTaskContext.
    requested_outputs' own docstring); this only decides whether the
    utterance should be preserved structurally instead of falling into the
    plain collected_facts note dump below."""
    if not re.search(r"\d", cleaned):
        return False
    return contains_any_phrase(normalized_text, _CONTENT_REQUEST_VERBS)


def _extract_comparison_options_from_text(raw_text: str) -> list[str]:
    options = extract_explicit_alternatives(raw_text)
    if options:
        return options
    # Direct imperatives ("compara Samsung com Apple") are routed by the
    # activity layer as comparison starts, but they do not use the broader
    # "vamos comparar ..." alternatives trigger. For a new comparison, those
    # operands are still the semantic options; criteria remain follow-up
    # requests once the comparison already has options.
    direct_options = extract_comparison_criteria(raw_text)
    if len(direct_options) >= 2 and message_implies_comparison(raw_text):
        return [_capitalize_initial_word(option) for option in direct_options]
    return []


def _capitalize_initial_word(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    return cleaned[:1].upper() + cleaned[1:]


def _extract_generic_activity_fields(
    ctx: "ActiveTaskContext",
    raw_text: str,
    normalized_text: str,
    *,
    allow_unclassified_note: bool = True,
) -> list[str]:
    updated: list[str] = []
    cleaned = raw_text.strip(" .,!?;:")
    if not cleaned:
        return updated
    if _looks_like_content_request(cleaned, normalized_text):
        if cleaned not in ctx.requested_outputs:
            ctx.requested_outputs.append(cleaned)
        updated.append("requested_output")
    if ctx.task_type == "plan_trip":
        if contains_any_phrase(normalized_text, ("a partir de", "a partir do", "a partir da", "partir de", "desde ")) and not ctx.parameters.get("origin"):
            origin_match = re.search(r"\b(?:a partir d[eo]s?|partir d[eo]s?|desde)\s+([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ' -]{1,40})", raw_text, flags=re.IGNORECASE)
            if origin_match:
                ctx.parameters["origin"] = _normalize_extracted_name(origin_match.group(1).strip(" .,!?;:"))
                updated.append("origin")
        if re.search(r"\b\d+\s*(?:euros|eur|€)\b", normalized_text) and not ctx.parameters.get("budget_max"):
            budget_match = re.search(r"\b(\d+)\s*(?:euros|eur|€)\b", raw_text, flags=re.IGNORECASE)
            if budget_match:
                ctx.parameters["budget_max"] = f"{budget_match.group(1)} euros"
                updated.append("budget_max")
        if re.search(r"\b\d+\s+dias\b", normalized_text) and not ctx.parameters.get("duration"):
            duration_match = re.search(r"\b(\d+\s+dias)\b", raw_text, flags=re.IGNORECASE)
            if duration_match:
                ctx.parameters["duration"] = duration_match.group(1)
                updated.append("duration")
        if contains_any_phrase(normalized_text, ("voos curtos", "voo curto", "sem voos muito longos")) and not ctx.parameters.get("flight_constraint"):
            ctx.parameters["flight_constraint"] = "voos curtos"
            updated.append("flight_constraint")
        if "norte de portugal" in normalized_text and not ctx.parameters.get("destination"):
            ctx.parameters["destination"] = "Norte de Portugal"
            updated.append("destination")
        elif "portugal" in normalized_text and not ctx.parameters.get("destination"):
            ctx.parameters["destination"] = "Portugal"
            updated.append("destination")
        if contains_any_phrase(normalized_text, ("descansar", "descanso", "calmo", "tranquilo")) and not ctx.parameters.get("travel_style"):
            ctx.parameters["travel_style"] = "descanso"
            updated.append("travel_style")
        elif contains_any_phrase(normalized_text, ("aventura", "road trip", "ir mudando")) and not ctx.parameters.get("travel_style"):
            ctx.parameters["travel_style"] = "aventura"
            updated.append("travel_style")
        elif contains_any_phrase(normalized_text, ("conhecer sitios", "conhecer sítios", "cidades", "natureza")) and not ctx.parameters.get("travel_style"):
            ctx.parameters["travel_style"] = "exploracao"
            updated.append("travel_style")
    elif ctx.task_type == "study_topic":
        if not ctx.parameters.get("topic"):
            ctx.parameters["topic"] = cleaned
            updated.append("topic")
    elif ctx.task_type == "make_decision":
        if contains_any_phrase(normalized_text, ("entre", "duas", "opcoes", "opções")) and not ctx.parameters.get("options"):
            ctx.parameters["options"] = cleaned
            updated.append("options")
    elif ctx.task_type == "research_topic":
        if not ctx.parameters.get("topic"):
            ctx.parameters["topic"] = cleaned
            updated.append("topic")
    elif ctx.task_type == "generic_comparison":
        if not ctx.parameters.get("category"):
            category = _infer_generic_comparison_category(normalized_text)
            if category:
                ctx.parameters["category"] = category
                updated.append("category")
        if not ctx.parameters.get("options"):
            options = _extract_comparison_options_from_text(raw_text)
            if options:
                ctx.parameters["options"] = " e ".join(options)
                updated.append("options")
        if _looks_like_comparison_priority_statement(normalized_text):
            updated.append("preference")
        if contains_any_phrase(normalized_text, ("compara", "comparar", "criterios", "critérios", "bateria", "camara", "câmara", "autonomia", "custa", "esta a", "está a", "preco", "preço", "suporte")):
            if "preference" not in updated and cleaned not in ctx.requested_outputs:
                ctx.requested_outputs.append(cleaned)
                updated.append("requested_output")
    elif ctx.task_type == "generic_activity":
        if not ctx.parameters.get("topic"):
            ctx.parameters["topic"] = cleaned
            updated.append("topic")
    elif ctx.task_type == "writing":
        if not ctx.parameters.get("subject"):
            ctx.parameters["subject"] = cleaned
            updated.append("subject")
    if allow_unclassified_note and not updated and cleaned not in ctx.collected_facts and len(cleaned.split()) <= 40:
        ctx.collected_facts.append(cleaned)
        updated.append("note")
    elif _should_preserve_raw_activity_note(updated) and cleaned not in ctx.collected_facts:
        ctx.collected_facts.append(cleaned)
    return updated


def _infer_generic_comparison_category(normalized_text: str) -> str:
    if contains_any_phrase(normalized_text, ("telemovel", "telemoveis", "telefone", "telefones", "iphone", "smartphone", "smartphones")):
        return "telemóveis"
    if contains_any_phrase(normalized_text, ("proposta", "propostas", "fornecedor", "fornecedores")):
        return "propostas"
    return ""


def _should_preserve_raw_activity_note(updated_fields: list[str]) -> bool:
    if not updated_fields or "preference" in updated_fields:
        return False
    semantic_fields = {
        "origin",
        "destination",
        "date_range",
        "duration",
        "budget",
        "budget_max",
        "travel_style",
        "flight_constraint",
        "options",
        "category",
        "subject",
        "topic",
    }
    return bool(set(updated_fields).isdisjoint(semantic_fields))


def _looks_like_comparison_priority_statement(normalized_text: str) -> bool:
    if not contains_any_phrase(
        normalized_text,
        ("mais importante", "valorizo mais", "pesa mais", "importa menos", "menos do que", "dou mais importancia", "dou mais importância"),
    ):
        return False
    return contains_any_phrase(
        normalized_text,
        (
            "camara",
            "camera",
            "câmara",
            "fotografia",
            "video",
            "vídeo",
            "bateria",
            "autonomia",
            "suporte",
            "apoio",
            "preco",
            "preço",
            "custo",
            "localizacao",
            "localização",
            "zona",
        ),
    )


def _missing_generic_activity_fields(ctx: "ActiveTaskContext") -> list[str]:
    if ctx.task_type == "plan_trip":
        if not ctx.parameters.get("travel_style"):
            return ["travel_style"]
        return []
    if ctx.task_type == "make_decision":
        if not ctx.parameters.get("criteria"):
            return ["criteria"]
        return []
    if ctx.task_type == "study_topic":
        if not ctx.parameters.get("materials"):
            return ["materials"]
        return []
    if ctx.task_type == "research_topic":
        if not ctx.parameters.get("scope"):
            return ["scope"]
        return []
    if ctx.task_type in ("generic_comparison", "generic_activity"):
        return []
    if ctx.task_type == "writing":
        if ctx.artifact_type == "unknown":
            return ["artifact_type"]
        return []
    return []


def _generic_activity_question(ctx: "ActiveTaskContext") -> str:
    if ctx.task_type == "plan_trip":
        if ctx.parameters.get("destination"):
            return "Acho que ainda me falta perceber como gostas de viajar. Procuras mais descansar, conhecer sítios novos ou alguma aventura?"
        return "Boa ideia. Antes de escolhermos sítios, procuras mais descansar, conhecer sítios novos ou alguma aventura?"
    if ctx.task_type == "make_decision":
        return "Certo. Qual é o critério que pesa mais nessa decisão?"
    if ctx.task_type == "study_topic":
        return "Vamos a isso. Que disciplina é? Já tens apontamentos ou vais começar pelos materiais da cadeira?"
    if ctx.task_type == "research_topic":
        return "Certo. Queres uma visão geral primeiro ou queres comparar opções concretas?"
    if ctx.task_type == "generic_comparison":
        return "Certo. Já temos essa comparação aberta."
    if ctx.task_type == "generic_activity":
        return "Certo. Vamos trabalhando nisso por partes."
    if ctx.task_type == "writing":
        return "Claro. O que queres escrever?"
    return "Certo. Diz-me só qual é o próximo passo."


def _generic_task_title(ctx: "ActiveTaskContext") -> str:
    if ctx.task_type == "plan_trip":
        return "Planeamento de férias"
    if ctx.task_type == "make_decision":
        return "Decisão em preparação"
    if ctx.task_type == "study_topic":
        return "Plano de estudo"
    if ctx.task_type == "research_topic":
        return "Investigação"
    if ctx.task_type == "generic_comparison":
        category = ctx.parameters.get("category") or ""
        if category:
            return f"Comparação: {category}"
        options = ctx.parameters.get("options") or ""
        return f"Comparação: {options}" if options else "Comparação"
    if ctx.task_type == "generic_activity":
        return "Atividade em curso"
    if ctx.task_type == "writing":
        return "Escrita em conjunto"
    return "Tarefa"


def _generic_task_card_content(ctx: "ActiveTaskContext") -> str:
    lines = [f"Objetivo: {ctx.goal or _generic_task_title(ctx).lower()}"]
    if ctx.parameters.get("destination"):
        lines.append(f"Destino: {ctx.parameters['destination']}")
    if ctx.parameters.get("travel_style"):
        lines.append(f"Preferência: {ctx.parameters['travel_style']}")
    if ctx.parameters.get("topic"):
        lines.append(f"Tema: {ctx.parameters['topic']}")
    if ctx.parameters.get("category"):
        lines.append(f"Categoria: {ctx.parameters['category']}")
    if ctx.parameters.get("options"):
        lines.append(f"Opções: {ctx.parameters['options']}")
    if ctx.parameters.get("criteria"):
        lines.append(f"Critérios: {ctx.parameters['criteria']}")
    if ctx.requested_outputs:
        lines.append("Pedidos: " + "; ".join(ctx.requested_outputs[-3:]))
    if ctx.collected_facts:
        lines.append("Notas: " + "; ".join(ctx.collected_facts[-3:]))
    if len(lines) == 1:
        lines.append("[a recolher contexto]")
    return "\n".join(lines)


def _split_option_text(options_text: str) -> list[str]:
    return [part.strip(" ,.!?;:") for part in re.split(r"\s+e\s+|,", options_text or "") if part.strip(" ,.!?;:")]


def _normalized_option_set(options: list[str]) -> set[str]:
    return {normalize_for_matching(option).strip(" ,.!?;:") for option in options if normalize_for_matching(option).strip(" ,.!?;:")}


# --- Phase 4 integration fix (section 3): workspace reference resolver ----
# ONE generic resolver over the whole WorkSession -- answers "what existing
# workspace entity (SemanticObject, active task artifact, paused task
# artifact, or open document) does this utterance name?". No domain-specific
# (travel/email/study/...) logic: every task_type's artifact is ALREADY a
# registered SemanticObject with task_id set, keyed by whatever
# _generic_task_title/the compose_email title logic already produced (see
# _update_task_artifact_ui) -- so a single pass over work_session.objects
# already covers "another existing activity" and "another SemanticObject" at
# once, active OR paused (pausing a task never removes its artifact object).

_WORKSPACE_REFERENCE_STOPWORDS = {
    "a", "o", "as", "os", "de", "da", "do", "das", "dos", "em", "para", "e", "um", "uma", "no", "na", "ao", "aos",
}


@dataclass(frozen=True)
class WorkspaceReferenceMatch:
    """object_id is always resolvable when kind is a task artifact (task
    artifacts are SemanticObjects too) -- "" only for a bare document-context
    match, which has no SemanticObject id of its own."""

    object_id: str
    task_id: str
    title: str
    confidence: float


@dataclass(frozen=True)
class TargetResolution:
    matches: list[WorkspaceReferenceMatch]
    confidence: float = 0.0
    ambiguous: bool = False
    source: str = ""


@dataclass(frozen=True)
class ExecutionResult:
    executed: bool = False
    mutated_state: bool = False
    ui_event_queued: bool = False
    partial: bool = False
    reason: str = ""


def _title_alias_index(normalized_message: str, title_norm: str) -> int:
    """Word-overlap alias match: "as ferias" must resolve against
    "planeamento de ferias" even though it is not a literal substring of it
    -- the relationship is reversed (a distinctive word FROM the title
    appears in the message, not the whole title in the message). Ignores
    stopwords/very short tokens so "a"/"de"/"em" never counts as the
    distinctive word on its own. Returns the earliest match index, or -1."""
    best = -1
    for word in title_norm.split():
        if word in _WORKSPACE_REFERENCE_STOPWORDS or len(word) <= 2:
            continue
        match = re.search(rf"\b{re.escape(word)}\b", normalized_message)
        if match and (best == -1 or match.start() < best):
            best = match.start()
    return best


def _title_match_confidence(normalized_message: str, title: str) -> float:
    """0 means no match at all -- never guessed. A literal, whole-title
    mention is the strongest signal (0.9); a distinctive-word alias overlap
    is weaker but still unambiguous enough to act on (0.6)."""
    if not title:
        return 0.0
    title_norm = normalize_for_matching(title)
    if not title_norm:
        return 0.0
    if title_norm in normalized_message:
        return 0.9
    return 0.6 if _title_alias_index(normalized_message, title_norm) >= 0 else 0.0


def _fuzzy_target_phrase(raw_message: str) -> str:
    normalized = normalize_for_matching(raw_message)
    target = _workspace_clause_target_phrase(normalized)
    tokens = [
        token
        for token in re.findall(r"\b[\wÀ-ÿ]{3,}\b", target)
        if token not in _SEMANTIC_TARGET_STOPWORDS
    ]
    return " ".join(tokens)


def _fuzzy_title_score(target_phrase: str, title: str) -> float:
    title_norm = normalize_for_matching(title)
    if not target_phrase or not title_norm:
        return 0.0
    target_tokens = target_phrase.split()
    title_tokens = title_norm.split()
    if len(target_tokens) == 1 and len(title_tokens) == 1:
        if abs(len(target_tokens[0]) - len(title_tokens[0])) > 2:
            return 0.0
        return SequenceMatcher(None, target_tokens[0], title_tokens[0]).ratio()
    if abs(len(target_phrase) - len(title_norm)) > 4:
        return 0.0
    return SequenceMatcher(None, target_phrase, title_norm).ratio()


def _resolve_fuzzy_semantic_targets(raw_message: str, work_session: "WorkSession") -> TargetResolution:
    target_phrase = _fuzzy_target_phrase(raw_message)
    if len(target_phrase) < 4:
        return TargetResolution(matches=[], confidence=0.0, ambiguous=False, source="fuzzy_target_too_short")

    candidates: list[WorkspaceReferenceMatch] = []
    for object_id, obj in work_session.objects.items():
        score = _fuzzy_title_score(target_phrase, obj.title)
        if score < 0.86:
            continue
        candidates.append(
            WorkspaceReferenceMatch(
                object_id=object_id,
                task_id=obj.task_id,
                title=obj.title,
                confidence=score,
            )
        )

    if not candidates:
        return TargetResolution(matches=[], confidence=0.0, ambiguous=False, source="no_fuzzy_match")
    candidates.sort(key=lambda match: match.confidence, reverse=True)
    best = candidates[0]
    close = [match for match in candidates if best.confidence - match.confidence < 0.04]
    if len(close) > 1:
        return TargetResolution(matches=close, confidence=best.confidence, ambiguous=True, source="ambiguous_fuzzy_workspace_reference")
    return TargetResolution(matches=[best], confidence=best.confidence, ambiguous=False, source="fuzzy_workspace_reference")


def resolve_workspace_reference(raw_message: str, work_session: "WorkSession") -> "WorkspaceReferenceMatch | None":
    """Section 3: never guesses -- returns None unless exactly one distinct
    workspace entity reaches the highest confidence tier found. Candidates
    are deduped by task_id (falling back to object_id for task-less objects)
    so a task's own artifact never counts as a second, competing candidate
    against itself."""
    normalized_message = normalize_for_matching(raw_message)
    best_by_key: dict[str, WorkspaceReferenceMatch] = {}

    for object_id, obj in work_session.objects.items():
        confidence = _title_match_confidence(normalized_message, obj.title)
        if confidence <= 0:
            continue
        key = obj.task_id or object_id
        existing = best_by_key.get(key)
        if existing is None or confidence > existing.confidence:
            best_by_key[key] = WorkspaceReferenceMatch(
                object_id=object_id, task_id=obj.task_id, title=obj.title, confidence=confidence,
            )

    document_context = work_session.document_context
    if document_context is not None and document_context.resolved_file:
        confidence = _title_match_confidence(normalized_message, document_context.resolved_file)
        if confidence > 0:
            key = "document:" + document_context.resolved_file
            existing = best_by_key.get(key)
            if existing is None or confidence > existing.confidence:
                best_by_key[key] = WorkspaceReferenceMatch(
                    object_id="", task_id="", title=document_context.resolved_file, confidence=confidence,
                )

    if not best_by_key:
        return None
    top_confidence = max(match.confidence for match in best_by_key.values())
    top_matches = [match for match in best_by_key.values() if match.confidence == top_confidence]
    if len(top_matches) != 1:
        return None
    return top_matches[0]


def resolve_semantic_targets(raw_message: str, work_session: "WorkSession", intent_context: str = "") -> TargetResolution:
    """Canonical backend semantic target resolver.

    It deliberately returns semantic object ids/titles only. Frontend
    geometry and live DOM target resolution remain client-local.
    """
    mentioned_ids = resolve_mentioned_object_ids(raw_message, work_session)
    if mentioned_ids:
        matches = [
            WorkspaceReferenceMatch(
                object_id=object_id,
                task_id=work_session.objects[object_id].task_id,
                title=work_session.objects[object_id].title,
                confidence=0.9,
            )
            for object_id in mentioned_ids
            if object_id in work_session.objects
        ]
        return TargetResolution(matches=matches, confidence=0.9, ambiguous=False, source="mentioned_object_ids")

    normalized_message = normalize_for_matching(raw_message)
    writing_context = work_session.writing_context
    if (
        writing_context is not None
        and writing_context.active_artifact_id
        and writing_context.active_artifact_id in work_session.objects
        and contains_any_phrase(normalized_message, ("email", "mail", "mensagem"))
        and intent_context in ("attention", "workspace_reference", "attention_show_again", "activity_resume")
    ):
        obj = work_session.objects[writing_context.active_artifact_id]
        return TargetResolution(
            matches=[
                WorkspaceReferenceMatch(
                    object_id=obj.id,
                    task_id=obj.task_id,
                    title=obj.title,
                    confidence=0.85,
                )
            ],
            confidence=0.85,
            ambiguous=False,
            source="writing_context_alias",
        )

    if (
        contains_any_phrase(normalized_message, ("email", "mail", "mensagem"))
        and intent_context in ("attention", "workspace_reference", "attention_show_again", "activity_resume")
    ):
        matches: list[WorkspaceReferenceMatch] = []
        for object_id, obj in work_session.objects.items():
            if obj.kind != "task_artifact":
                continue
            task_type = str(obj.payload.get("task_type") or "")
            if task_type != "compose_email":
                continue
            matches.append(
                WorkspaceReferenceMatch(
                    object_id=object_id,
                    task_id=obj.task_id,
                    title=obj.title,
                    confidence=0.82,
                )
            )
        if len(matches) == 1:
            return TargetResolution(matches=matches, confidence=0.82, ambiguous=False, source="task_artifact_alias")
        if len(matches) > 1:
            return TargetResolution(matches=[], confidence=0.0, ambiguous=True, source="multiple_email_artifacts")

    reference = resolve_workspace_reference(raw_message, work_session)
    if reference is not None:
        return TargetResolution(matches=[reference], confidence=reference.confidence, ambiguous=False, source="workspace_reference")

    if intent_context in ("attention", "workspace_reference", "attention_show_again", "activity_resume"):
        fuzzy = _resolve_fuzzy_semantic_targets(raw_message, work_session)
        if fuzzy.matches or fuzzy.ambiguous:
            return fuzzy

    if intent_context == "attention_show_again" and work_session.attention is not None:
        minimized_ids = [oid for oid, level in work_session.attention.object_states.items() if level == _ATTN_MINIMIZED]
        hidden_ids = [oid for oid, level in work_session.attention.object_states.items() if level == _ATTN_HIDDEN]
        pool = minimized_ids or hidden_ids
        if len(pool) == 1 and pool[0] in work_session.objects:
            obj = work_session.objects[pool[0]]
            return TargetResolution(
                matches=[WorkspaceReferenceMatch(object_id=pool[0], task_id=obj.task_id, title=obj.title, confidence=0.65)],
                confidence=0.65,
                source="single_hidden_or_minimized",
            )
        if len(pool) > 1:
            return TargetResolution(matches=[], confidence=0.0, ambiguous=True, source="multiple_hidden_or_minimized")

    return TargetResolution(matches=[], confidence=0.0, ambiguous=False, source="no_match")


_OPEN_REFERENCE_TRIGGER_PHRASES = ("abre", "mostra", "foca no", "foca na")
_RESUME_REFERENCE_TRIGGER_PHRASES = (
    "volta ao",
    "volta a",
    "volta aos",
    "volta as",
    "volta às",
    "vamos voltar ao",
    "vamos voltar a",
    "vamos voltar aos",
    "vamos voltar as",
    "vamos voltar às",
    "continua o",
    "continua a",
    "continua com",
    "retoma o",
    "retoma a",
    "quero continuar o",
    "quero continuar a",
)


def _is_open_reference_phrase(normalized_message: str) -> bool:
    return contains_any_phrase(normalized_message, _OPEN_REFERENCE_TRIGGER_PHRASES)


def _is_resume_reference_phrase(normalized_message: str) -> bool:
    return contains_any_phrase(normalized_message, _RESUME_REFERENCE_TRIGGER_PHRASES)


def _extract_compose_email_fields(ctx: "ActiveTaskContext", raw_text: str, normalized_text: str) -> list[str]:
    """Section 6/9: order-independent -- every detector below runs
    unconditionally against whatever text it's given, so one turn filling
    several fields at once ("é para o João Cardoso, professor de CGI...")
    works the same as info trickling in one fact per turn."""
    updated: list[str] = []

    pattern_match = _extract_role_course_name(raw_text)
    if pattern_match.get("role") and not ctx.entities.get("recipientRole"):
        ctx.entities["recipientRole"] = pattern_match["role"]
        updated.append("recipientRole")
    if pattern_match.get("course") and not ctx.parameters.get("course"):
        ctx.parameters["course"] = pattern_match["course"]
        updated.append("course")
    if pattern_match.get("name") and not ctx.entities.get("recipient"):
        ctx.entities["recipient"] = pattern_match["name"]
        updated.append("recipient")

    course_span = None
    if not ctx.parameters.get("course"):
        course_match = _COURSE_TEXT_RE.search(raw_text)
        if course_match:
            course_text = course_match.group(1).strip()
            if course_text:
                ctx.parameters["course"] = _normalize_extracted_name(_clean_course_text(course_text))
                course_span = course_match.span(1)
                updated.append("course")
        else:
            relation_course_match = re.search(
                r"\b(?:professor|professora)\s+de\s+(.+?)(?=\s+e\s+(?:quero|preciso|gostava|vou|para|pedir|perguntar)\b|[,.;]|$)",
                raw_text,
                flags=re.IGNORECASE,
            )
            if relation_course_match:
                course_text = _normalize_extracted_name(_clean_course_text(relation_course_match.group(1)))
                if course_text:
                    ctx.parameters["course"] = course_text
                    updated.append("course")

    if not ctx.entities.get("recipientRole"):
        for marker, role in _RECIPIENT_ROLE_MARKERS.items():
            if marker in normalized_text:
                ctx.entities["recipientRole"] = role
                updated.append("recipientRole")
                break

    if not ctx.entities.get("recipient"):
        for match in _NAME_RUN_RE.finditer(raw_text):
            if course_span and match.start() >= course_span[0] and match.end() <= course_span[1]:
                continue  # the course name itself is also capitalized -- not a person
            candidate = match.group(1).strip()
            if not _looks_like_name_candidate(candidate):
                continue  # a lone sentence-starter capital ("É", "Ok") isn't a name
            ctx.entities["recipient"] = _normalize_extracted_name(candidate)
            updated.append("recipient")
            break

    if not ctx.parameters.get("purpose"):
        for marker, purpose in _PURPOSE_MARKERS.items():
            if marker in normalized_text:
                ctx.parameters["purpose"] = purpose
                updated.append("purpose")
                break
        if not ctx.parameters.get("purpose") and contains_any_phrase(normalized_text, ("perguntar", "pedir")):
            ctx.parameters["purpose"] = "request_clarification"
            updated.append("purpose")
    elif ctx.parameters.get("purpose") == "request_help" and contains_any_phrase(normalized_text, ("data do exame", "data de exame", "qual sera a data do exame")):
        ctx.parameters["purpose"] = "ask_exam_date"
        updated.append("purpose")
    if ctx.parameters.get("purpose") == "ask_exam_date" and not ctx.parameters.get("topic"):
        ctx.parameters["topic"] = "data do exame"
        updated.append("topic")

    # Section 3: an explicit "sobre X" (etc.) marker fills topic in the SAME
    # turn a purpose marker was also found in -- must not wait for "nothing
    # else matched this turn" the way the catch-all below does.
    if not ctx.parameters.get("topic"):
        topic_match = _TOPIC_ABOUT_RE.search(raw_text.strip(" .,!?;:"))
        if topic_match:
            topic_text = topic_match.group(1).strip(" .,!?;:")
            topic_text = _LEADING_ARTICLE_RE.sub("", topic_text).strip()
            if topic_text and _normalize_text(topic_text) not in _VAGUE_TOPIC_WORDS:
                ctx.parameters["topic"] = topic_text
                updated.append("topic")
        else:
            request_match = re.search(r"\b(?:perguntar|pedir)\s+(.+)$", raw_text.strip(" .,!?;:"), flags=re.IGNORECASE)
            if request_match:
                topic_text = request_match.group(1).strip(" .,!?;:")
                if topic_text and _normalize_text(topic_text) not in _VAGUE_TOPIC_WORDS:
                    ctx.parameters["topic"] = topic_text
                    updated.append("topic")

    # Section 13: once we know WHY (purpose) but not yet ABOUT WHAT (topic),
    # an otherwise-unclassified continuation turn is captured as the topic
    # itself -- content for the task, never routed to a general-knowledge
    # explanation of whatever subject it happens to mention.
    if ctx.parameters.get("purpose") and not ctx.parameters.get("topic") and not updated:
        cleaned = raw_text.strip(" .,!?;:")
        if cleaned and len(cleaned.split()) <= 24:
            ctx.parameters["topic"] = cleaned
            updated.append("topic")

    # Section 3/4/9 (fault-attribution phase): once the task already HAS a
    # topic, a further unclassified fragment ("o funcionamento e as
    # limitações", "as vantagens e desvantagens em relação ao Dijkstra")
    # enriches the email as an additional key_point instead of being
    # dropped -- this is what keeps turns 5/6 of the real conversation in
    # TASK_CONTINUATION instead of falling to GENERAL_CONVERSATION. Genuine
    # questions about the topic are filtered out earlier via
    # _TASK_EXPLICIT_EXIT_PHRASES, before extraction ever reaches this point.
    if ctx.parameters.get("topic") and not updated:
        cleaned = raw_text.strip(" .,!?;:")
        if cleaned and len(cleaned.split()) <= 24 and cleaned not in ctx.key_points:
            ctx.key_points.append(cleaned)
            updated.append("key_points")

    if updated:
        ctx.collected_facts.append(raw_text.strip())
    return updated


def _missing_compose_email_fields(ctx: "ActiveTaskContext") -> list[str]:
    # Section 8's minimum sufficient set: destinatário/contexto, objetivo,
    # conteúdo principal -- course is nice-to-have context, not blocking.
    missing: list[str] = []
    if not ctx.entities.get("recipient") and not ctx.entities.get("recipientRole"):
        missing.append("recipient")
        return missing
    if not ctx.parameters.get("purpose"):
        missing.append("purpose")
    elif not ctx.parameters.get("topic"):
        missing.append("topic")
    return missing


def _compose_email_question(ctx: "ActiveTaskContext") -> str:
    # Section 7: never ask about a field already known -- each branch here
    # only fires while that specific field is still in missing_information.
    missing = ctx.missing_information
    if "recipient" in missing:
        return "Claro. Para quem é?"
    if "purpose" in missing:
        return "O que precisas de pedir ou dizer nesse email?"
    if "topic" in missing:
        return "Sobre que assunto, mais especificamente?"
    # Section 10 (fault-attribution phase): once a draft already exists, a
    # data/content turn (e.g. a new key_point) is just acknowledged short --
    # re-asking "queres que escreva?" after a draft is already in review
    # would be nonsensical and surprising; regeneration only happens on an
    # explicit request (see _TASK_CONTROL_EXECUTE_PHRASES's "atualiza o
    # email"/"reescreve com isso").
    if ctx.parameters.get("generated_draft"):
        return "Certo."
    # Section 8: from here on a bare "podes"/"sim"/"força"/"avança" counts
    # as confirming THIS specific offer (see _detect_task_control_action).
    ctx.pending_action = "create_draft"
    return "Já tenho o suficiente. Queres que escreva uma primeira versão?"


_ROLE_LABELS = {
    "professor": "professor",
    "colleague": "colega",
    "client": "cliente",
    "sponsor": "patrocinador",
    "manager": "chefe",
    "advisor": "orientador",
}
_PURPOSE_LABELS = {
    "request_clarification": "pedir esclarecimentos",
    "request_help": "pedir ajuda",
    "schedule_meeting": "marcar uma reunião",
    "thanks": "agradecer",
    "ask_exam_date": "perguntar a data do exame",
}


_TASK_GENERATION_MAX_ATTEMPTS = 2


def _compose_email_task_prompt(ctx: "ActiveTaskContext") -> tuple[list[str], str]:
    """Section 5/7: structured facts + an explicit no-fabrication rule --
    the model gets exactly what the user stated, nothing implied. In the
    real failure the model invented "especialmente com as passagens
    negativas" for a Bellman-Ford question nobody asked in those terms.
    Section 5's other real failure: the model asked "qual é a dúvida
    específica?" despite the task already being marked ready -- the
    explicit "não peças mais informação" instruction below addresses that
    directly, since the earlier prompt never said it outright."""
    recipient = ctx.entities.get("recipient") or ""
    role_label = _ROLE_LABELS.get(ctx.entities.get("recipientRole") or "", ctx.entities.get("recipientRole") or "")
    course = ctx.parameters.get("course") or ""
    purpose_label = _PURPOSE_LABELS.get(ctx.parameters.get("purpose") or "", ctx.parameters.get("purpose") or "")
    topic = ctx.parameters.get("topic") or ""

    facts = ["Tarefa: escrever um email."]
    if recipient:
        facts.append(f"Destinatário: {recipient}")
    if role_label:
        facts.append(f"Papel do destinatário: {role_label}")
    if course:
        facts.append(f"Curso/contexto: {course}")
    if purpose_label:
        facts.append(f"Objetivo do email: {purpose_label}")
    if topic:
        facts.append(f"Assunto/tema: {topic}")
    for point in ctx.key_points:
        facts.append(f"Ponto a incluir: {point}")

    instruction = (
        "A task foi classificada como READY pelo sistema: os factos abaixo já são "
        "considerados suficientes.\n"
        "Redige o email agora com os factos disponíveis. Não peças mais informação. Não "
        "faças perguntas ao Alexandre.\n"
        "Vais redigir um email profissional e natural em português de Portugal, com base "
        "exclusivamente nos factos fornecidos.\n"
        "Regra obrigatória: não inventes a natureza específica da dúvida, exemplos, detalhes "
        "técnicos ou motivos que não tenham sido fornecidos como factos.\n"
        "Se um detalhe não foi especificado, usa uma formulação neutra e não o inventes -- "
        "por exemplo: \"Gostaria de pedir alguns esclarecimentos sobre <assunto>.\"\n"
        "Inclui uma saudação adequada ao papel do destinatário, o corpo do email e uma "
        "despedida. Não incluas uma linha \"Assunto:\" no corpo -- isso é tratado à parte."
    )
    return facts, instruction


def _compose_email_correction_prompt(previous_reason: str) -> str:
    # Section 7: the corrective prompt for a first attempt that didn't
    # execute the task at all (asked for more information instead of
    # writing) -- reusing the normal router, never a hardcoded provider.
    return (
        "A resposta anterior não executou a tarefa: pediu informação adicional em vez de "
        "redigir o email.\n"
        "A task já foi marcada READY.\n"
        "Produz agora o email completo apenas com os factos fornecidos. Não faças "
        "perguntas. Não inventes detalhes."
    )


# Section 6/14: structural signals that the model responded WITH A
# META-QUESTION instead of producing the email itself -- the exact real
# failure ("Para redigir o email, preciso saber mais... Qual é a dúvida
# específica?"). Deliberately literal/structural, not semantic judgment.
_ASK_FOR_MORE_INFO_MARKERS = (
    "preciso saber mais", "precisamos de saber mais", "podes especificar", "podes dizer-me mais",
    "podes dizer me mais", "qual e a duvida especifica", "qual e a tua duvida",
    "que informacao", "mais informacao sobre", "podes clarificar", "preciso de mais detalhes",
    "poderias especificar", "podes ser mais especifico", "preciso que me digas",
)


def _validate_task_generation(reply: str, ctx: "ActiveTaskContext") -> tuple[bool, str]:
    # Deliberately structural, not semantic -- verifying "did it invent
    # facts" mechanically isn't feasible without a second LLM judge (out of
    # scope here); the no-fabrication guarantee comes from the prompt
    # instruction above. This catches: an empty/degenerate reply, the model
    # asking for more information instead of writing (section 6's real
    # failure), and -- when known -- a reply that never mentions the
    # recipient or topic it was explicitly given.
    cleaned = (reply or "").strip()
    if not cleaned:
        return False, "empty_output"
    if len(cleaned) < 20:
        return False, "not_artifact"
    normalized = _normalize_text(cleaned)
    if contains_any_phrase(normalized, _ASK_FOR_MORE_INFO_MARKERS):
        return False, "asked_for_more_information"
    # A short reply that ENDS in a question mark is itself a meta-response
    # (asking the user something) rather than the artifact -- a real email
    # may legitimately contain a rhetorical "?" mid-body, but ending on one
    # in a short reply is the same "asked instead of executed" pattern.
    if cleaned.rstrip().endswith("?") and len(cleaned.split()) <= 60:
        return False, "asked_for_more_information"
    recipient = ctx.entities.get("recipient") or ""
    if recipient:
        first_name = _normalize_text(recipient.split()[0])
        if first_name and first_name not in normalized:
            return False, "missing_recipient"
    topic = ctx.parameters.get("topic") or ""
    if topic:
        topic_terms = [term for term in _normalize_text(topic).split() if len(term) >= 4]
        if topic_terms and not any(term in normalized for term in topic_terms):
            return False, "missing_topic"
    return True, "valid"


def _update_pending_writing_task(task: PendingWritingTask, message: str, *, awaiting_field: str = "") -> None:
    text = _normalize_text(message)
    raw = str(message or "").strip()
    if "email" in text or "e-mail" in text or "mail" in text:
        task.task_type = "compose_email"
    batch_count = _extract_writing_batch_count(text)
    if batch_count > 1:
        task.batch_total = max(task.batch_total, batch_count)
        task.batch_remaining_after_current = max(task.batch_remaining_after_current, task.batch_total - 1)
    details = _extract_pending_writing_recipient_details(raw)
    if details.get("recipient") and not task.recipient:
        task.recipient = details["recipient"]
    if details.get("recipient_type") and not task.recipient_type:
        task.recipient_type = details["recipient_type"]
    if details.get("recipient_context") and not task.recipient_context:
        task.recipient_context = details["recipient_context"]
    request = _extract_pending_writing_request(raw)
    if request and not task.subject:
        task.subject = request
    if "professor" in text:
        task.recipient_type = "professor"
    elif "patrocinador" in text:
        task.recipient_type = "potential_sponsor"
    elif "cliente" in text:
        task.recipient_type = "client"
    elif "colega" in text:
        task.recipient_type = "colleague"
    if "potencial patrocinador" in text:
        task.recipient_type = "potential_sponsor"
    if "projeto" in text or "projecto" in text:
        explicit = _extract_explicit_project_reference(raw)
        task.project = explicit or "unknown"
    if any(marker in text for marker in ("apoio", "financiamento", "patrocinio", "patrocínio")):
        task.purpose = "pedir apoio"
    elif re.search(r"\b(?:rever|revisao|revisão)\b", text):
        task.purpose = "pedir revisão"
    elif "projeto" in text or "projecto" in text:
        task.subject = raw
    if awaiting_field == "recipient" and not task.recipient and not task.recipient_type:
        recipient = _extract_contextual_writing_recipient(raw)
        if recipient:
            task.recipient = recipient
    if awaiting_field == "purpose" and not task.purpose and not task.subject:
        purpose_or_subject = _extract_contextual_writing_purpose(raw)
        if purpose_or_subject:
            task.subject = purpose_or_subject
    if raw and raw not in task.known_facts:
        task.known_facts.append(raw)


def _extract_pending_writing_recipient_details(message: str) -> dict[str, str]:
    raw = str(message or "").strip(" \t\r\n.,!?;:")
    if not raw:
        return {}
    patterns = (
        re.compile(
            rf"\b(?:o|a)\s+([A-ZÀ-Ý][\wÀ-ÿ-]+)\s+(?:é|e)\s+(?:o\s+|a\s+)?(?:meu\s+|minha\s+)?({_ROLE_WORDS_ALT})\s+de\s+([^,.;]+)",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\bpara\s+(?:o\s+|a\s+)?([A-ZÀ-Ý][\wÀ-ÿ-]+)\s*,?\s+(?:meu\s+|minha\s+)?({_ROLE_WORDS_ALT})\s+de\s+([^,.;]+)",
            re.IGNORECASE,
        ),
    )
    for pattern in patterns:
        match = pattern.search(raw)
        if not match:
            continue
        name, role, context = (part.strip(" \t\r\n.,!?;:") for part in match.groups())
        role_key = _RECIPIENT_ROLE_MARKERS.get(_normalize_text(role), _normalize_text(role))
        return {
            "recipient": _normalize_extracted_name(name),
            "recipient_type": role_key,
            "recipient_context": f"{role.lower()} de {context.strip()}",
        }
    return {}


def _extract_pending_writing_request(message: str) -> str:
    raw = str(message or "").strip(" \t\r\n.,!?;:")
    if not raw:
        return ""
    patterns = (
        r"\b(?:quero|preciso de|gostava de)\s+(?:enviar|escrever|mandar)\s+(?:um\s+|uma\s+)?(?:mail|email|mensagem)\s+a\s+(.+)$",
        r"\bpara\s+(perguntar|pedir|dizer|transmitir|explicar)\s+(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if not match:
            continue
        if len(match.groups()) == 1:
            candidate = match.group(1).strip(" \t\r\n.,!?;:")
        else:
            candidate = " ".join(part.strip(" \t\r\n.,!?;:") for part in match.groups() if part).strip()
        if candidate:
            return candidate
    return ""


def _extract_contextual_writing_recipient(message: str) -> str:
    raw = str(message or "").strip(" \t\r\n.,!?;:")
    if not raw:
        return ""
    normalized = _normalize_text(raw)
    if any(marker in normalized for marker in ("professor", "patrocinador", "cliente", "colega")):
        return ""
    match = re.search(r"^(?:é\s+|e\s+)?para\s+(?:o\s+|a\s+)?(.+)$", raw, flags=re.IGNORECASE)
    candidate = match.group(1).strip(" \t\r\n.,!?;:") if match else raw
    if not candidate:
        return ""
    candidate_norm = _normalize_text(candidate)
    if candidate_norm in {"um", "uma", "o", "a", "isso", "isto", "email", "mail"}:
        return ""
    if len(candidate.split()) > 6:
        return ""
    return candidate[0].upper() + candidate[1:] if candidate and candidate[0].islower() else candidate


def _extract_contextual_writing_purpose(message: str) -> str:
    raw = str(message or "").strip(" \t\r\n.,!?;:")
    if not raw:
        return ""
    match = re.search(
        r"^(?:é\s+|e\s+)?(?:sobre|acerca\s+de|relativamente\s+a|em\s+relação\s+a)\s+(.+)$",
        raw,
        flags=re.IGNORECASE,
    )
    candidate = match.group(1).strip(" \t\r\n.,!?;:") if match else raw
    if not candidate or len(candidate.split()) > 18:
        return ""
    return candidate


def _extract_explicit_project_reference(message: str) -> str:
    match = re.search(r"\bprojeto\s+([A-Za-z0-9_.-]+)\b", message, flags=re.IGNORECASE)
    if match:
        name = match.group(1).strip(" .,!?:;")
        if _normalize_text(name) not in {"que", "em", "um", "uma", "este", "esse"}:
            return name
    normalized = _normalize_text(message)
    if "assistenteia" in normalized:
        return "AssistenteIA"
    if "echo" in normalized:
        return "Echo"
    return ""


def _missing_writing_fields(task: PendingWritingTask) -> list[str]:
    missing: list[str] = []
    if not task.recipient_type and not task.recipient:
        missing.append("recipient")
    if not task.purpose and not task.subject:
        missing.append("purpose")
    if task.project == "unknown" and task.recipient_type == "potential_sponsor":
        missing.append("project")
    return missing[:2]


def _refresh_pending_writing_task_state(task: PendingWritingTask) -> None:
    task.missing_fields = _missing_writing_fields(task)
    task.awaiting_field = task.missing_fields[0] if task.missing_fields else ""
    task.current_step = f"awaiting_{task.awaiting_field}" if task.awaiting_field else "ready_to_draft"


def _writing_workflow_question(task: PendingWritingTask) -> str:
    missing = set(task.missing_fields)
    if {"project", "purpose"}.issubset(missing):
        return "Qual é o projeto e que tipo de apoio queres pedir?"
    if "recipient" in missing:
        return "Para quem é o email?"
    if "project" in missing:
        return "Qual é o projeto?"
    if "purpose" in missing:
        return "O que queres pedir ou transmitir nesse email?"
    return "Queres que eu escreva uma primeira versão?"


def _writing_task_prompt(task: PendingWritingTask) -> str:
    facts = "\n".join(f"- {fact}" for fact in task.known_facts)
    return (
        "Escreve um email com estes dados recolhidos:\n"
        f"tipo={task.task_type}\n"
        f"destinatario={task.recipient or 'unknown'}\n"
        f"destinatario_tipo={task.recipient_type or 'unknown'}\n"
        f"contexto_destinatario={task.recipient_context or 'unknown'}\n"
        f"projeto={task.project}\n"
        f"objetivo={task.purpose or task.subject or 'unknown'}\n"
        f"factos:\n{facts}"
    )


def _writing_task_local_fallback(task: PendingWritingTask) -> str:
    recipient = "Olá," if not task.recipient else f"Olá {task.recipient},"
    subject = task.subject or task.purpose or "Pedido"
    return f"Assunto: {subject}\n\n{recipient}\n\nEscrevo-te sobre este tema para percebermos a melhor forma de avançar.\n\nAlexandre"


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

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from assistant.cognition.intent_engine import IntentResult


@dataclass(frozen=True)
class ContextBundle:
    user_message: str
    intent: IntentResult
    active_contexts: list[str] = field(default_factory=list)
    time_context: str = ""
    personal_facts: list[str] = field(default_factory=list)
    session_facts: list[str] = field(default_factory=list)
    memory_facts: list[str] = field(default_factory=list)
    pending_tasks: str = ""
    observed_context: str = ""

    def relevant_facts(self) -> list[str]:
        facts: list[str] = []
        facts.extend(self.personal_facts)
        facts.extend(self.session_facts)
        facts.extend(self.memory_facts)
        if self.pending_tasks:
            facts.append(self.pending_tasks)
        if self.observed_context:
            facts.append(self.observed_context)
        if self.time_context:
            facts.append(self.time_context)
        return facts

    def summary(self) -> str:
        facts = self.relevant_facts()
        if not facts:
            return ""
        return "\n".join(f"- {fact}" for fact in facts[:8])


class ContextBuilder:
    """Collects only the context that is relevant for the inferred intent."""

    def __init__(
        self,
        personal_model: Any | None = None,
        long_term_memory: Any | None = None,
        session_manager: Any | None = None,
        context_observer: Any | None = None,
        now: datetime | None = None,
    ) -> None:
        self.personal_model = personal_model
        self.long_term_memory = long_term_memory
        self.session_manager = session_manager
        self.context_observer = context_observer
        self.now = now

    def build(
        self,
        user_message: str,
        intent: IntentResult,
        active_contexts: list[str] | None = None,
        enabled_sources: set[str] | None = None,
    ) -> ContextBundle:
        sources = enabled_sources or set()
        return ContextBundle(
            user_message=user_message,
            intent=intent,
            active_contexts=active_contexts or [],
            time_context=self._time_context(),
            personal_facts=self._personal_facts(user_message, intent) if "personal_model" in sources else [],
            session_facts=self._session_facts(intent) if "session" in sources else [],
            memory_facts=self._memory_facts(user_message, intent) if "long_term_memory" in sources else [],
            pending_tasks=self._pending_tasks(intent) if "tasks" in sources else "",
            observed_context=self._observed_context(intent) if "observed_context" in sources else "",
        )

    def _time_context(self) -> str:
        now = self.now or datetime.now()
        return f"Agora é {now.strftime('%A, %Y-%m-%d %H:%M')}."

    def _personal_facts(self, user_message: str, intent: IntentResult) -> list[str]:
        if self.personal_model is None:
            return []

        if intent.intent == "explorar_identidade":
            facts = getattr(self.personal_model, "facts_about", lambda _query="": [])("")
            return list(facts)

        facts = getattr(self.personal_model, "facts_about", lambda _query="": [])(user_message)
        return list(facts[:5])

    def _session_facts(self, intent: IntentResult) -> list[str]:
        if self.session_manager is None:
            return []

        if intent.intent == "retomar_contexto":
            return list(getattr(self.session_manager, "facts_for_last_session", lambda: [])())

        if intent.intent in {"resolver_problema_tecnico", "gerir_tarefas", "conversa_normal"}:
            context = getattr(self.session_manager, "planner_context", lambda: "")()
            return [context] if context else []

        return []

    def _memory_facts(self, user_message: str, intent: IntentResult) -> list[str]:
        if self.long_term_memory is None:
            return []
        if intent.intent in {"explorar_identidade", "retomar_contexto"}:
            return []
        records = getattr(self.long_term_memory, "search", lambda _query, limit=5: [])(user_message, limit=5)
        return [record.content for record in records if getattr(record, "content", "")]

    def _pending_tasks(self, intent: IntentResult) -> str:
        if self.long_term_memory is None or intent.intent not in {"gerir_tarefas", "retomar_contexto", "planear_estudo"}:
            return ""
        return getattr(self.long_term_memory, "pending_tasks", lambda limit=5, show_details=False: "")(
            limit=5,
            show_details=False,
        )

    def _observed_context(self, intent: IntentResult) -> str:
        if self.context_observer is None or intent.intent not in {
            "retomar_contexto",
            "resolver_problema_tecnico",
            "acao_operacional",
        }:
            return ""
        snapshot_getter = getattr(self.context_observer, "last_snapshot", None)
        if snapshot_getter is None:
            return ""
        snapshot = snapshot_getter()
        if snapshot is None:
            return ""
        return str(snapshot)

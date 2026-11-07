from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

from assistant.response_language import localize
from assistant.text_matching import contains_any_phrase, normalize_for_matching


@dataclass
class KnowledgeSource:
    source_type: str = "user"
    reference: str = ""
    original_text: str = ""
    created_at: str = ""


@dataclass
class KnowledgeEntry:
    id: str
    task_id: str
    kind: str
    subject: str
    predicate: str
    value: str
    source: KnowledgeSource
    confidence: float = 1.0
    status: str = "active"
    replaces: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskReasoningState:
    task_id: str
    stage: str = "explore"
    current_recommendation: str = ""
    recommendation_confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    stale: bool = False
    updated_at: str = ""


@dataclass(frozen=True)
class CognitiveTaskSection:
    role: str
    title: str
    priority: str
    items: list[dict[str, Any]] = field(default_factory=list)
    text: str = ""


@dataclass(frozen=True)
class CognitiveTaskViewModel:
    task_id: str
    task_type: str
    title: str
    stage: str
    sections: list[CognitiveTaskSection] = field(default_factory=list)


@dataclass(frozen=True)
class TaskDialoguePlan:
    acknowledgement: str = ""
    implication: str = ""
    next_question: str = ""
    recommendation: str = ""
    uncertainty: str = ""
    next_step: str = ""

    def render(self) -> str:
        parts = [
            self.acknowledgement,
            self.implication,
            self.recommendation,
            self.uncertainty,
            self.next_step,
            self.next_question,
        ]
        return " ".join(part.strip() for part in parts if part and part.strip()).strip()


def task_knowledge(work_session, task_id: str) -> list[KnowledgeEntry]:
    bucket = work_session.knowledge.setdefault("task_knowledge", {})
    return bucket.setdefault(task_id, [])


def task_reasoning(work_session, task_id: str) -> TaskReasoningState:
    bucket = work_session.knowledge.setdefault("task_reasoning", {})
    state = bucket.get(task_id)
    if isinstance(state, TaskReasoningState):
        return state
    state = TaskReasoningState(task_id=task_id)
    bucket[task_id] = state
    return state


def update_task_knowledge(work_session, task, message: str, *, turn_id: str = "") -> list[KnowledgeEntry]:
    if task is None:
        return []
    entries: list[KnowledgeEntry] = []
    _ensure_options_from_task(work_session, task, entries, message, turn_id)
    if task.task_type == "generic_comparison":
        entries.extend(_extract_comparison_entries(work_session, task, message, turn_id))
    elif task.task_type == "plan_trip":
        entries.extend(_extract_travel_entries(work_session, task, message, turn_id))
    else:
        entries.extend(_extract_generic_entries(work_session, task, message, turn_id))
    if entries:
        _refresh_reasoning(work_session, task)
    return entries


def classify_task_query(message: str) -> str:
    normalized = normalize_for_matching(message).strip(" .!?")
    if contains_any_phrase(
        normalized,
        (
            "com estas fontes",
            "com as fontes",
            "com o que encontraste",
            "o que encontraste",
            "o que concluis",
            "estas fontes chegam",
            "as fontes chegam",
            "ja chega",
            "já chega",
            "ha evidencia suficiente",
            "há evidência suficiente",
            "o que e que estas fontes nos dizem",
            "o que é que estas fontes nos dizem",
            "com isto ja conseguimos comparar",
            "com isto já conseguimos comparar",
            "ja consegues dizer",
            "já consegues dizer",
            "isto prova",
        ),
    ):
        return "evidence"
    if contains_any_phrase(
        normalized,
        (
            "resume me so o importante",
            "resume-me só o importante",
            "resume so o importante",
            "resume só o importante",
            "so o importante",
            "só o importante",
        ),
    ):
        return "important"
    if contains_any_phrase(
        normalized,
        (
            "o que sabemos",
            "que sabemos",
            "o que ja sabes",
            "o que já sabes",
            "o que sabes sobre",
        ),
    ):
        return "known"
    if contains_any_phrase(normalized, ("o que falta decidir", "que falta decidir", "decisoes em aberto", "decisões em aberto")):
        return "decisions"
    if contains_any_phrase(normalized, ("o que ainda falta", "o que falta saber", "ainda nao sabemos", "ainda não sabemos", "precisas de saber")):
        return "unknowns"
    if contains_any_phrase(
        normalized,
        (
            "qual parece melhor",
            "qual escolherias",
            "qual e melhor",
            "qual é melhor",
            "e agora",
            "consegues sugerir destinos",
            "podes sugerir destinos",
            "sugere destinos",
        ),
    ):
        return "recommendation"
    if (
        normalized in {"porque", "porquê", "por que"}
        or normalized.startswith("porque ")
        or normalized.startswith("porquê ")
        or contains_any_phrase(normalized, ("porque?", "porquê?", "porque e que", "porquê é que"))
    ):
        return "why"
    if contains_any_phrase(normalized, ("de onde veio", "qual e a fonte", "qual é a fonte", "fonte desse", "origem desse")):
        return "source"
    if contains_any_phrase(normalized, ("informacao contraditoria", "informação contraditória", "contradicoes", "contradições", "fontes discordam", "fontes que discordem")):
        return "conflicts"
    if contains_any_phrase(normalized, ("proximo passo", "próximo passo", "o que temos de organizar", "que temos de organizar")):
        return "next_step"
    return ""


def answer_task_query(work_session, task, message: str, response_language: str = "pt-PT") -> str:
    """Phase 7.7.1: status/missing/next-step/summary/recommendation/why
    queries now render through response_language. _format_important/
    _format_sources/_format_evidence_answer/_format_conflicts (evidence
    and cross-option comparison prose) are a separate, deeper sentence-
    generation surface intentionally left for a future pass -- see the
    Phase 7.7.1 report."""
    if task is None:
        return ""
    query = classify_task_query(message)
    if not query:
        return ""
    knowledge = _active_entries(task_knowledge(work_session, task.id))
    if query == "known":
        return _format_known(task, knowledge, response_language)
    if query == "important":
        return _format_important(task, knowledge, task_reasoning(work_session, task.id), response_language)
    if query == "unknowns":
        return _format_unknowns(task, knowledge, response_language)
    if query == "decisions":
        return _format_decisions(task, knowledge, response_language)
    if query == "recommendation":
        if task.task_type == "plan_trip":
            return _format_travel_recommendation(task, knowledge, response_language)
        reasoning = _refresh_reasoning(work_session, task, response_language)
        return _format_recommendation(reasoning, response_language)
    if query == "why":
        reasoning = task_reasoning(work_session, task.id)
        if not reasoning.reasons:
            reasoning = _refresh_reasoning(work_session, task, response_language)
        return _format_why(reasoning, response_language)
    if query == "source":
        return _format_sources(knowledge, message, response_language)
    if query == "evidence":
        return _format_evidence_answer(task, knowledge, message, response_language)
    if query == "conflicts":
        return _format_conflicts(knowledge, response_language)
    if query == "next_step":
        return _format_next_step(task, knowledge, response_language)
    return ""


def task_dialogue_response(
    work_session,
    task,
    message: str = "",
    *,
    knowledge_delta: list[KnowledgeEntry] | None = None,
    response_language: str = "pt-PT",
) -> str:
    """Natural task dialogue grounded in the current structured state.

    The UI projection and this text both derive from the same TaskKnowledge
    and TaskReasoningState. The delta only explains what changed this turn.

    Phase 7.7.1: response_language is threaded through the plan_trip and
    generic dialogue paths (the ones this phase closes); the
    generic_comparison path (_plan_comparison_dialogue) is a separate,
    larger sentence-generation subsystem intentionally left for a future
    pass -- see the Phase 7.7.1 report's "known remaining inconsistencies".
    """
    if task is None:
        return ""
    knowledge = _knowledge_with_task_fields(task, _active_entries(task_knowledge(work_session, task.id)))
    reasoning = _refresh_reasoning(work_session, task, response_language)
    delta = [entry for entry in (knowledge_delta or []) if entry and entry.status == "active"]
    if task.task_type == "plan_trip":
        plan = _plan_travel_dialogue(task, knowledge, reasoning, delta, message, response_language)
    elif task.task_type == "generic_comparison":
        plan = _plan_comparison_dialogue(task, knowledge, reasoning, delta, response_language)
    else:
        plan = _plan_generic_dialogue(task, knowledge, reasoning, delta, response_language)
    rendered = plan.render()
    return rendered or _fallback_task_dialogue(task, knowledge, reasoning, response_language)


def task_card_summary(work_session, task, response_language: str = "pt-PT") -> list[str]:
    if task is None:
        return []
    knowledge = _active_entries(task_knowledge(work_session, task.id))
    if not knowledge:
        return []
    lines: list[str] = []
    if task.task_type == "generic_comparison":
        options = _subjects_with_kind(knowledge, "option")
        if options:
            lines.append(localize("task.card_summary.options", response_language, options=_join_pt(options[:4], response_language)))
        constraints = [e for e in knowledge if e.kind == "constraint"]
        preferences = [e for e in knowledge if e.kind == "preference"]
        if constraints:
            lines.append(
                localize("task.card_summary.limit", response_language, value=_entry_phrase(constraints[-1], response_language))
            )
        if preferences:
            lines.append(
                localize(
                    "task.card_summary.priority", response_language, value=_entry_phrase(preferences[-1], response_language)
                )
            )
    elif task.task_type == "plan_trip":
        essentials = [e for e in knowledge if e.predicate in {"destination", "date_range", "duration", "budget_max", "travel_style", "flight_constraint"}]
        for entry in essentials[:4]:
            lines.append(_entry_phrase(entry, response_language))
    else:
        for entry in knowledge[:4]:
            lines.append(_entry_phrase(entry, response_language))
    unknowns = infer_unknowns(task, knowledge, response_language)
    if unknowns:
        lines.append(localize("task.card_summary.open", response_language, items=_join_pt(unknowns[:3], response_language)))
    return lines


def cognitive_task_view_model(
    work_session, task, *, title: str = "", response_language: str = "pt-PT"
) -> dict[str, Any]:
    """Read-only projection for the Task Surface.

    This deliberately derives from task knowledge/reasoning every time. It
    does not create a second mutable source of truth for the UI.

    Phase 7.7 (section 1/7/16): every section TITLE/label here is
    Echo-authored presentation text and localizes via response_language
    (the canonical section `role` -- "overview"/"options"/"decisions"/etc.
    -- is unchanged and remains the real semantic key; only its rendered
    `title` changes with language, matching section 16's "canonical
    semantics, localized presentation" split). The recommendation/next-
    step SENTENCE CONTENT (_next_step_label, reasoning.current_
    recommendation's own text) is a deeper, free-text dialogue-generation
    gap this phase does not attempt -- see the Phase 7.7 report's "known
    remaining language inconsistencies".
    """
    if task is None:
        return {}
    knowledge = _knowledge_with_task_fields(task, _active_entries(task_knowledge(work_session, task.id)))
    reasoning = _refresh_reasoning(work_session, task, response_language)
    stage = infer_task_stage(task, knowledge) or reasoning.stage
    sections: list[CognitiveTaskSection] = []

    overview_items = _overview_items(task, knowledge, response_language)
    if overview_items:
        sections.append(
            CognitiveTaskSection("overview", _overview_title(task, response_language), "high", overview_items)
        )

    if task.task_type == "generic_comparison":
        option_items = _option_view_items(knowledge, response_language)
        if option_items:
            sections.append(
                CognitiveTaskSection("options", localize("task.section.options", response_language), "high", option_items)
            )
    elif task.task_type == "plan_trip":
        found_items = _travel_found_items(knowledge)
        if found_items:
            sections.append(
                CognitiveTaskSection(
                    "found", localize("task.section.found", response_language), "high", found_items[:5]
                )
            )

    constraints = _entries_for_roles(knowledge, {"constraint"})
    if constraints:
        sections.append(
            CognitiveTaskSection(
                "constraints",
                localize("task.section.limits", response_language),
                "high",
                _entry_view_items(constraints[:3], response_language),
            )
        )

    preferences = _entries_for_roles(knowledge, {"preference"})
    if preferences:
        sections.append(
            CognitiveTaskSection(
                "priorities",
                localize("task.section.priorities", response_language),
                "medium",
                _entry_view_items(preferences[:4], response_language),
            )
        )

    decisions = _entries_for_roles(knowledge, {"decision"})
    if decisions:
        sections.append(
            CognitiveTaskSection(
                "decisions",
                localize("task.section.decisions", response_language),
                "high",
                _entry_view_items(decisions[:3], response_language),
            )
        )

    facts = _non_duplicated_fact_items(task, knowledge, response_language)
    if facts:
        sections.append(
            CognitiveTaskSection(
                "key_information", localize("task.section.known", response_language), "medium", facts[:6]
            )
        )

    if reasoning.current_recommendation:
        sections.append(
            CognitiveTaskSection(
                "recommendation",
                localize("task.section.now", response_language),
                "high",
                [
                    {
                        "label": localize(
                            "task.recommendation.seems_best", response_language, recommendation=reasoning.current_recommendation
                        ),
                        "details": reasoning.reasons[:3],
                    }
                ],
            )
        )
    elif task.task_type == "plan_trip" and stage == "plan":
        sections.append(
            CognitiveTaskSection(
                "plan_areas",
                localize("task.section.organizing", response_language),
                "medium",
                [
                    {"label": item, "details": []}
                    for item in (
                        localize("task.item.accommodation", response_language),
                        localize("task.item.transport", response_language),
                        localize("task.item.itinerary", response_language),
                        localize("task.item.costs", response_language),
                    )
                ],
            )
        )

    conflicts = _conflict_items(knowledge)
    if conflicts:
        sections.append(
            CognitiveTaskSection("conflicts", localize("task.section.attention", response_language), "high", conflicts[:3])
        )

    unknowns = _prioritized_unknowns(task, knowledge, reasoning, response_language)
    if unknowns:
        sections.append(
            CognitiveTaskSection(
                "unknowns",
                localize("task.section.still_missing", response_language),
                "medium",
                [{"label": item, "details": []} for item in unknowns[:4]],
            )
        )

    next_step = _next_step_label(task, knowledge, reasoning, response_language)
    if next_step:
        sections.append(
            CognitiveTaskSection("next_step", localize("task.section.next_step", response_language), "high", text=next_step)
        )

    view = CognitiveTaskViewModel(
        task_id=task.id,
        task_type=task.task_type,
        title=title or getattr(task, "goal", "") or task.task_type,
        stage=stage,
        sections=sections,
    )
    return _view_model_to_dict(view)


def _view_model_to_dict(view: CognitiveTaskViewModel) -> dict[str, Any]:
    return {
        "version": 1,
        "task_id": view.task_id,
        "task_type": view.task_type,
        "title": view.title,
        "stage": view.stage,
        "sections": [
            {
                "role": section.role,
                "title": section.title,
                "priority": section.priority,
                "text": section.text,
                "items": [
                    {
                        "label": str(item.get("label", "")),
                        "details": [str(detail) for detail in item.get("details", []) if str(detail)],
                    }
                    for item in section.items
                    if str(item.get("label", ""))
                ],
            }
            for section in view.sections
            if section.text or section.items
        ],
    }


def _overview_title(task, response_language: str = "pt-PT") -> str:
    if task.task_type == "generic_comparison":
        return localize("task.overview.comparison", response_language)
    if task.task_type == "plan_trip":
        return localize("task.overview.planning", response_language)
    return localize("task.overview.goal", response_language)


def _overview_items(task, knowledge: list[KnowledgeEntry], response_language: str = "pt-PT") -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    category = getattr(task, "parameters", {}).get("category", "")
    if category and task.task_type == "generic_comparison":
        items.append({"label": f"{localize('task.card.category', response_language)} {category}", "details": []})
    if getattr(task, "goal", ""):
        items.append({"label": task.goal, "details": []})
    for predicate in ("destination", "origin", "date_range"):
        value = _latest_value(knowledge, "trip", predicate)
        if value:
            items.append(
                {
                    "label": _entry_phrase(
                        KnowledgeEntry("", "", "fact", "trip", predicate, value, KnowledgeSource()), response_language
                    ),
                    "details": [],
                }
            )
    return _dedupe_view_items(items)


def _knowledge_with_task_fields(task, knowledge: list[KnowledgeEntry]) -> list[KnowledgeEntry]:
    projected = list(knowledge)
    if task.task_type == "plan_trip":
        existing = {(normalize_for_matching(entry.subject), normalize_for_matching(entry.predicate)) for entry in projected}
        for predicate in ("destination", "travel_style"):
            value = getattr(task, "parameters", {}).get(predicate, "")
            key = (normalize_for_matching("trip" if predicate == "destination" else "user"), normalize_for_matching(predicate))
            if value and key not in existing:
                projected.append(
                    KnowledgeEntry(
                        id=f"projected-{task.id}-{predicate}",
                        task_id=task.id,
                        kind="fact" if predicate == "destination" else "preference",
                        subject="trip" if predicate == "destination" else "user",
                        predicate=predicate,
                        value=value,
                        source=KnowledgeSource(source_type="task_field"),
                    )
                )
        return projected
    if task.task_type != "generic_comparison":
        return projected
    existing_options = {normalize_for_matching(entry.subject) for entry in projected if entry.kind == "option"}
    for option in _split_task_options(getattr(task, "parameters", {}).get("options", "")):
        if normalize_for_matching(option) in existing_options:
            continue
        projected.append(
            KnowledgeEntry(
                id=f"projected-{task.id}-{normalize_for_matching(option)}",
                task_id=task.id,
                kind="option",
                subject=option,
                predicate="is_option",
                value="true",
                source=KnowledgeSource(source_type="task_field"),
            )
        )
    return projected


def _option_view_items(knowledge: list[KnowledgeEntry], response_language: str = "pt-PT") -> list[dict[str, Any]]:
    options = _subjects_with_kind(knowledge, "option")
    items: list[dict[str, Any]] = []
    for option in options:
        details: list[str] = []
        price = _latest_value(knowledge, option, "price")
        if price:
            details.append(f"{price} €")
        details.extend(_quality_view_details(knowledge, option, response_language))
        items.append({"label": option, "details": _dedupe(details)[:3]})
    return items


def _travel_found_items(knowledge: list[KnowledgeEntry]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for entry in knowledge:
        if entry.status != "active":
            continue
        if entry.predicate == "destination_candidate" and entry.subject:
            details = []
            if entry.value and not _looks_like_raw_research_text(entry.value):
                details.append(entry.value)
            items.append({"label": entry.subject, "details": details[:1]})
        elif entry.predicate == "activities":
            label = entry.value if entry.value and not _looks_like_raw_research_text(entry.value) else entry.subject
            if label and not _looks_like_raw_research_text(label):
                items.append({"label": label, "details": []})
    return _dedupe_view_items(items)


def _looks_like_raw_research_text(value: str) -> bool:
    text = str(value or "")
    normalized = normalize_for_matching(text)
    if len(text) > 120:
        return True
    if text.strip().startswith("#") or "\n" in text or ">" in text:
        return True
    if normalized.startswith("trip:") or normalized == "trip":
        return True
    return contains_any_phrase(
        normalized,
        (
            "http",
            "best places",
            "things to do",
            "travel guide",
            "day itinerary",
            "article",
            "snippet",
            "what is included",
            "in addition to",
            "search result",
            "while porto",
            "opening hours -",
        ),
    )


def _entries_for_roles(knowledge: list[KnowledgeEntry], roles: set[str]) -> list[KnowledgeEntry]:
    return [entry for entry in knowledge if entry.kind in roles]


def _entry_view_items(entries: list[KnowledgeEntry], response_language: str = "pt-PT") -> list[dict[str, Any]]:
    return _dedupe_view_items([{"label": _entry_phrase(entry, response_language), "details": []} for entry in entries])


def _non_duplicated_fact_items(
    task, knowledge: list[KnowledgeEntry], response_language: str = "pt-PT"
) -> list[dict[str, Any]]:
    skipped_predicates = {
        "is_option",
        "price",
        "budget_max",
        "camera_priority",
        "battery_priority",
        "support_priority",
        "price_priority",
        "location_priority",
    }
    if task.task_type == "plan_trip":
        skipped_predicates |= {"destination", "date_range", "destination_candidate", "activities"}
    if task.task_type == "generic_comparison":
        skipped_predicates |= {"battery", "camera_quality", "support_quality"}
    facts = [
        {"label": _entry_phrase(entry, response_language), "details": []}
        for entry in knowledge
        if entry.kind in {"fact", "claim", "opinion", "derived_insight"} and entry.predicate not in skipped_predicates
    ]
    return _dedupe_view_items(facts)


def _conflict_items(knowledge: list[KnowledgeEntry]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], set[str]] = {}
    for entry in knowledge:
        if not _entry_can_conflict(entry):
            continue
        grouped.setdefault((entry.subject, entry.predicate), set()).add(entry.value)
    items = []
    for (subject, predicate), values in grouped.items():
        if _values_are_semantically_conflicting(values, predicate):
            items.append({"label": f"{subject}: {_join_pt(sorted(values))}", "details": [predicate]})
    return items


def _prioritized_unknowns(
    task, knowledge: list[KnowledgeEntry], reasoning: TaskReasoningState, response_language: str = "pt-PT"
) -> list[str]:
    # Phase 7.7.2: the priority MARKERS below are matched by substring
    # against the (now-localized) unknowns text via _priority_index, so
    # they must be built in the SAME response_language or sorting silently
    # degrades to an arbitrary order once infer_unknowns stopped always
    # returning Portuguese text.
    unknowns = infer_unknowns(task, knowledge, response_language) or reasoning.uncertainties
    if task.task_type == "generic_comparison":
        predicates = {entry.predicate: entry.value for entry in knowledge if entry.kind == "preference"}
        camera_label = localize("task.quality.label.camera", response_language).lower()
        battery_label = localize("task.quality.label.battery", response_language).lower()
        support_label = localize("task.quality.label.support", response_language).lower()
        price_label = localize("task.predicate_label.price_bare", response_language).lower()
        ordered = [localize("task.unknown.decision_criteria", response_language)]
        if predicates.get("camera_priority") == "high":
            ordered.append(camera_label)
        if predicates.get("battery_priority") == "high":
            ordered.append(battery_label)
        if predicates.get("support_priority") == "high":
            ordered.append(support_label)
        if predicates.get("price_priority") == "high":
            ordered.append(price_label)
        ordered.extend([price_label, camera_label, battery_label, support_label])
        priority = tuple(_dedupe(ordered))
    elif task.task_type == "plan_trip":
        priority = tuple(
            localize(key, response_language)
            for key in (
                "task.unknown.destination_type",
                "task.unknown.dates_or_duration",
                "task.unknown.budget",
                "task.unknown.travel_pace",
                "task.unknown.accommodation",
                "task.unknown.transport",
            )
        )
    else:
        priority = (localize("task.unknown.next_concrete_step", response_language),)
    return sorted(_dedupe(unknowns), key=lambda item: _priority_index(item, priority))


def _priority_index(item: str, priority: tuple[str, ...]) -> int:
    normalized = normalize_for_matching(item)
    for index, marker in enumerate(priority):
        if marker in normalized:
            return index
    return len(priority)


def _next_step_label(
    task, knowledge: list[KnowledgeEntry], reasoning: TaskReasoningState, response_language: str = "pt-PT"
) -> str:
    unknowns = _prioritized_unknowns(task, knowledge, reasoning, response_language)
    if unknowns:
        return localize("task.next_step.close", response_language, gaps=_join_pt(unknowns[:2], response_language))
    if task.task_type == "plan_trip" and reasoning.stage == "plan":
        return localize("task.next_step.organize_trip", response_language)
    if reasoning.current_recommendation:
        return localize("task.next_step.confirm_before_deciding", response_language)
    return ""


def _plan_travel_dialogue(
    task,
    knowledge: list[KnowledgeEntry],
    reasoning: TaskReasoningState,
    delta: list[KnowledgeEntry],
    message: str,
    response_language: str = "pt-PT",
) -> TaskDialoguePlan:
    # Phase 7.7.1 (section 3/4): the SAME extracted semantic values
    # (destination, date_range, budget, travel_style, flight_constraint --
    # all canonical, never translated) now render through localize()
    # instead of hardcoded Portuguese sentence templates. The branching
    # logic itself (which state the dialogue is in) is completely unchanged.
    predicates = {entry.predicate for entry in knowledge}
    delta_predicates = {entry.predicate for entry in delta}
    stage = infer_task_stage(task, knowledge)
    destination = _latest_value(knowledge, "trip", "destination")
    date_range = _latest_value(knowledge, "trip", "date_range")
    duration = _latest_value(knowledge, "trip", "duration")
    budget = _latest_value(knowledge, "trip", "budget_max")
    travel_style = _latest_value(knowledge, "user", "travel_style")
    flight_constraint = _latest_value(knowledge, "trip", "flight_constraint")
    normalized = normalize_for_matching(message)

    if stage == "plan":
        if {"destination", "date_range"}.intersection(delta_predicates):
            ack = localize("task.travel.great_have_destination", response_language, destination=destination)
            if date_range:
                ack += localize("task.travel.and_dates", response_language, date_range=date_range)
            ack += "."
            return TaskDialoguePlan(
                acknowledgement=ack,
                implication=localize("task.travel.now_organize", response_language),
                next_question=localize("task.travel.where_to_start", response_language),
            )
        if "budget_max" in delta_predicates:
            scope = (
                localize("task.travel.budget_scope_excl_flights", response_language)
                if contains_any_phrase(normalized, ("sem contar os voos", "sem voos"))
                else ""
            )
            return TaskDialoguePlan(
                acknowledgement=localize("task.travel.budget_set", response_language, budget=budget, scope=scope),
                next_step=localize("task.travel.next_step_accommodation_type", response_language),
            )
        return TaskDialoguePlan(
            acknowledgement=localize("task.travel.dest_dates_set", response_language),
            next_step=localize("task.travel.now_organize_full", response_language),
        )

    if {"duration", "budget_max"}.intersection(delta_predicates):
        known = []
        if duration:
            known.append(duration)
        if budget:
            known.append(localize("task.travel.until_budget", response_language, budget=budget))
        return TaskDialoguePlan(
            acknowledgement=localize("task.travel.great_have", response_language, known=_join_pt(known, response_language)),
            implication=localize("task.travel.missing_travel_style", response_language) if "travel_style" not in predicates else "",
            next_question=_travel_next_question(knowledge, reasoning, response_language),
        )
    if "travel_style" in delta_predicates:
        return TaskDialoguePlan(
            acknowledgement=localize("task.travel.looking_for_style", response_language, travel_style=travel_style),
            next_question=_travel_next_question(knowledge, reasoning, response_language),
        )
    if "flight_constraint" in delta_predicates:
        summary = _travel_basis_summary(duration, budget, travel_style, flight_constraint, response_language)
        return TaskDialoguePlan(
            acknowledgement=localize("task.travel.perfect_summary", response_language, summary=summary),
            next_step=localize("task.travel.can_search_destinations", response_language),
        )
    if not delta:
        if destination and "travel_style" not in predicates:
            return TaskDialoguePlan(
                acknowledgement=localize("task.travel.missing_travel_preference", response_language),
                next_question=_travel_next_question(knowledge, reasoning, response_language),
            )
        return TaskDialoguePlan(
            acknowledgement=localize("task.travel.sure", response_language),
            next_question=_travel_next_question(knowledge, reasoning, response_language),
        )
    return TaskDialoguePlan(
        acknowledgement=localize("task.travel.right_that_helps", response_language),
        next_question=_travel_next_question(knowledge, reasoning, response_language),
    )


def _plan_comparison_dialogue(
    task,
    knowledge: list[KnowledgeEntry],
    reasoning: TaskReasoningState,
    delta: list[KnowledgeEntry],
    response_language: str = "pt-PT",
) -> TaskDialoguePlan:
    # Phase 7.7.2 (section 2/3): canonical values (option names, budget
    # numbers) unchanged; only the Echo-authored sentence scaffolding
    # localizes. Branching logic itself untouched.
    options = _subjects_with_kind(knowledge, "option")
    delta_predicates = {entry.predicate for entry in delta}
    budget = _latest_value(knowledge, "current_task", "budget_max")

    if not delta:
        if options:
            return TaskDialoguePlan(
                acknowledgement=localize(
                    "task.comparison.active", response_language, options=_join_pt(options, response_language)
                ),
                next_question=_comparison_next_question(task, knowledge, reasoning, response_language),
            )
        return TaskDialoguePlan(next_question=localize("task.comparison.which_options", response_language))

    if "budget_max" in delta_predicates and "battery_priority" in delta_predicates:
        return TaskDialoguePlan(
            acknowledgement=localize("task.comparison.budget_and_battery_change", response_language, budget=budget),
            implication=_comparison_budget_implication(knowledge, budget, response_language),
            recommendation=_comparison_recommendation_sentence(reasoning, response_language),
            uncertainty=_comparison_uncertainty_sentence(reasoning, response_language),
        )
    if "budget_max" in delta_predicates:
        return TaskDialoguePlan(
            acknowledgement=localize("task.comparison.budget_set", response_language, budget=budget),
            implication=_comparison_budget_implication(knowledge, budget, response_language),
            next_question=_comparison_next_question(task, knowledge, reasoning, response_language),
        )
    if {"camera_priority", "battery_priority", "support_priority", "price_priority", "location_priority"}.intersection(delta_predicates):
        return TaskDialoguePlan(
            acknowledgement=_comparison_preference_ack(delta, response_language),
            implication=_comparison_current_implication(knowledge, reasoning, response_language),
            next_question=_comparison_next_question(task, knowledge, reasoning, response_language),
        )
    if "price" in delta_predicates:
        return TaskDialoguePlan(
            acknowledgement=_comparison_prices_ack(knowledge, response_language),
            implication=_comparison_budget_implication(knowledge, budget, response_language),
            next_question=_comparison_next_question(task, knowledge, reasoning, response_language),
        )
    if {"camera_quality", "battery", "support_quality"}.intersection(delta_predicates):
        return TaskDialoguePlan(
            acknowledgement=_comparison_tradeoff_ack(delta, response_language),
            recommendation=_comparison_recommendation_sentence(reasoning, response_language),
            uncertainty=_comparison_uncertainty_sentence(reasoning, response_language),
            next_question=_comparison_next_question(task, knowledge, reasoning, response_language),
        )
    return TaskDialoguePlan(
        acknowledgement=localize("task.comparison.default_ack", response_language),
        next_question=_comparison_next_question(task, knowledge, reasoning, response_language),
    )


def _plan_generic_dialogue(
    task,
    knowledge: list[KnowledgeEntry],
    reasoning: TaskReasoningState,
    delta: list[KnowledgeEntry],
    response_language: str = "pt-PT",
) -> TaskDialoguePlan:
    if delta:
        return TaskDialoguePlan(
            acknowledgement=localize("task.generic.that_helps", response_language),
            next_question=_generic_next_question(reasoning, response_language),
        )
    return TaskDialoguePlan(
        acknowledgement=localize("task.generic.sure", response_language),
        next_question=_generic_next_question(reasoning, response_language),
    )


def _fallback_task_dialogue(
    task, knowledge: list[KnowledgeEntry], reasoning: TaskReasoningState, response_language: str = "pt-PT"
) -> str:
    if task.task_type == "plan_trip":
        return _travel_next_question(knowledge, reasoning, response_language)
    if task.task_type == "generic_comparison":
        return _comparison_next_question(task, knowledge, reasoning) or localize(
            "task.comparison.continue_parts", response_language
        )
    return _generic_next_question(reasoning, response_language) or localize(
        "task.generic.continue_parts", response_language
    )


def _travel_next_question(
    knowledge: list[KnowledgeEntry], reasoning: TaskReasoningState, response_language: str = "pt-PT"
) -> str:
    predicates = {entry.predicate for entry in knowledge}
    stage = infer_task_stage(type("Task", (), {"task_type": "plan_trip"})(), knowledge)
    if stage == "plan":
        return localize("task.travel.where_to_start", response_language)
    if "travel_style" not in predicates:
        return localize("task.travel.q.style", response_language)
    if "duration" not in predicates and "date_range" not in predicates:
        return localize("task.travel.q.days", response_language)
    if "budget_max" not in predicates:
        return localize("task.travel.q.budget", response_language)
    if "flight_constraint" not in predicates:
        return localize("task.travel.q.flight_constraint", response_language)
    return ""


def _travel_basis_summary(
    duration: str, budget: str, travel_style: str, flight_constraint: str, response_language: str = "pt-PT"
) -> str:
    parts = []
    if duration:
        parts.append(duration)
    if budget:
        parts.append(localize("task.travel.until_budget", response_language, budget=budget))
    if travel_style:
        parts.append(travel_style)
    if flight_constraint:
        parts.append(localize("task.travel.basis_flight_pref", response_language))
    if parts:
        return localize("task.travel.basis_have", response_language, parts=_join_pt(parts, response_language))
    return localize("task.travel.basis_initial", response_language)


def _comparison_next_question(
    task, knowledge: list[KnowledgeEntry], reasoning: TaskReasoningState, response_language: str = "pt-PT"
) -> str:
    # Phase 7.7.2: infer_unknowns() now returns response_language-rendered
    # text -- classification below checks BOTH the Portuguese AND English
    # forms of each category, since matching only the Portuguese substring
    # (as before) would silently stop recognizing the category once the
    # text itself is English, falling through to the generic question.
    unknowns = infer_unknowns(task, knowledge, response_language)
    if not unknowns:
        return ""
    first = unknowns[0]
    normalized = normalize_for_matching(first)
    if any(marker in normalized for marker in ("criterios", "critérios", "criteria")):
        return localize("task.comparison.q.criteria", response_language)
    if normalized.startswith(("preco", "preço", "price")):
        return localize("task.comparison.q.know_value", response_language, value=first)
    if normalized.startswith(("camara", "câmara", "camera")):
        return localize("task.comparison.q.missing_reference", response_language, value=first)
    if normalized.startswith(("autonomia", "battery")):
        return localize("task.comparison.q.missing_decisive", response_language, value=first)
    if any(marker in normalized for marker in ("suporte", "support")):
        return localize("task.comparison.q.missing_compare", response_language, value=first)
    return localize("task.comparison.q.missing_generic", response_language, value=first)


def _generic_next_question(reasoning: TaskReasoningState, response_language: str = "pt-PT") -> str:
    if reasoning.uncertainties:
        return localize("task.generic.next_step_prompt", response_language)
    return ""


def _comparison_budget_implication(knowledge: list[KnowledgeEntry], budget: str, response_language: str = "pt-PT") -> str:
    if not budget:
        return ""
    try:
        budget_num = float(budget)
    except ValueError:
        return ""
    inside: list[str] = []
    outside: list[str] = []
    for option in _subjects_with_kind(knowledge, "option"):
        price = _latest_value(knowledge, option, "price")
        if not price:
            continue
        try:
            price_num = float(price)
        except ValueError:
            continue
        if price_num <= budget_num:
            inside.append(option)
        else:
            outside.append(
                localize("task.comparison.budget_over", response_language, option=option, amount=int(price_num - budget_num))
            )
    inside_key = "task.comparison.budget_inside_plural" if len(inside) > 1 else "task.comparison.budget_inside_singular"
    inside_phrase = (
        localize(inside_key, response_language, options=_join_pt(inside, response_language), option=_join_pt(inside, response_language))
        if inside
        else ""
    )
    if inside and outside:
        return f"{inside_phrase}; {_join_pt(outside, response_language)}."
    if inside:
        return f"{inside_phrase}."
    if outside:
        return _join_pt(outside, response_language) + "."
    return ""


def _comparison_preference_ack(delta: list[KnowledgeEntry], response_language: str = "pt-PT") -> str:
    predicates = {entry.predicate for entry in delta}
    if "camera_priority" in predicates and "battery_priority" in predicates:
        return localize("task.comparison.preference_camera_and_battery", response_language)
    if "camera_priority" in predicates:
        return localize("task.comparison.preference_camera", response_language)
    if "battery_priority" in predicates:
        value = _latest_delta_value(delta, "battery_priority")
        if value == "low":
            return localize("task.comparison.preference_battery_low", response_language)
        return localize("task.comparison.preference_battery_other", response_language)
    if "support_priority" in predicates:
        return localize("task.comparison.preference_support", response_language)
    return localize("task.comparison.preference_generic", response_language)


def _comparison_current_implication(
    knowledge: list[KnowledgeEntry], reasoning: TaskReasoningState, response_language: str = "pt-PT"
) -> str:
    if reasoning.current_recommendation:
        return localize(
            "task.comparison.gaining_ground", response_language, recommendation=reasoning.current_recommendation
        )
    options = _subjects_with_kind(knowledge, "option")
    if options:
        return localize("task.comparison.need_compare_priority", response_language)
    return ""


def _comparison_prices_ack(knowledge: list[KnowledgeEntry], response_language: str = "pt-PT") -> str:
    prices = []
    for option in _subjects_with_kind(knowledge, "option"):
        price = _latest_value(knowledge, option, "price")
        if price:
            prices.append((option, int(price) if price.isdigit() else price))
    if not prices:
        return localize("task.comparison.more_price_data", response_language)
    numeric = [(option, price) for option, price in prices if isinstance(price, int)]
    if len(numeric) >= 2:
        ranked = sorted(numeric, key=lambda item: item[1])
        return localize(
            "task.comparison.prices_known_plural",
            response_language,
            list=", ".join(f"{option} {price} €" for option, price in ranked),
        )
    return localize(
        "task.comparison.prices_known_singular",
        response_language,
        list=", ".join(f"{option} {price} €" for option, price in prices),
    )


def _comparison_tradeoff_ack(delta: list[KnowledgeEntry], response_language: str = "pt-PT") -> str:
    quality = [entry for entry in delta if entry.predicate in {"camera_quality", "battery", "support_quality"}]
    if len(quality) >= 2:
        return localize(
            "task.comparison.tradeoff_clearer",
            response_language,
            items=_join_pt([_quality_delta_phrase(entry, response_language) for entry in quality[:3]], response_language),
        )
    if quality:
        prefix = localize("task.generic.sure", response_language)
        return f"{prefix} {_entry_phrase(quality[0], response_language).capitalize()}."
    return localize("task.comparison.tradeoff_helps", response_language)


def _quality_delta_phrase(entry: KnowledgeEntry, response_language: str = "pt-PT") -> str:
    if entry.predicate == "support_quality" and entry.value == "limited":
        return f"{entry.subject}: {localize('task.quality.support_limited', response_language).lower()}"
    return _short_detail(entry, response_language).replace("+ ", f"{entry.subject}: ")


def _comparison_recommendation_sentence(reasoning: TaskReasoningState, response_language: str = "pt-PT") -> str:
    if not reasoning.current_recommendation:
        return ""
    reasons = (
        "; ".join(reasoning.reasons[:2])
        if reasoning.reasons
        else localize("task.recommendation.default_reason", response_language)
    )
    return localize(
        "task.comparison.recommendation_gaining",
        response_language,
        recommendation=reasoning.current_recommendation,
        reasons=reasons,
    )


def _comparison_uncertainty_sentence(reasoning: TaskReasoningState, response_language: str = "pt-PT") -> str:
    if not reasoning.current_recommendation or not reasoning.uncertainties:
        return ""
    return localize(
        "task.comparison.still_provisional", response_language, gaps=_join_pt(reasoning.uncertainties[:2], response_language)
    )


def _latest_delta_value(delta: list[KnowledgeEntry], predicate: str) -> str:
    for entry in reversed(delta):
        if entry.predicate == predicate:
            return entry.value
    return ""


def _short_detail(entry: KnowledgeEntry, response_language: str = "pt-PT") -> str:
    if entry.predicate == "battery":
        return (
            localize("task.quality.battery_advantage", response_language)
            if entry.value == "better"
            else localize("task.quality.battery_value", response_language, value=entry.value)
        )
    if entry.predicate == "camera_quality":
        if entry.value == "better":
            return localize("task.quality.camera_better", response_language)
        if entry.value == "good":
            return localize("task.quality.camera_good", response_language)
        if entry.value == "mixed":
            return localize("task.quality.camera_mixed", response_language)
        return localize("task.quality.camera_little_evidence", response_language)
    if entry.predicate == "support_quality":
        return (
            localize("task.quality.support_better", response_language)
            if entry.value == "better"
            else localize("task.quality.support_limited", response_language)
        )
    return _entry_phrase(entry, response_language)


def _quality_view_details(knowledge: list[KnowledgeEntry], option: str, response_language: str = "pt-PT") -> list[str]:
    details: list[str] = []
    for predicate, label_key in (
        ("camera_quality", "task.quality.label.camera"),
        ("battery", "task.quality.label.battery"),
        ("support_quality", "task.quality.label.support"),
    ):
        entries = [
            entry
            for entry in knowledge
            if normalize_for_matching(entry.subject) == normalize_for_matching(option)
            and entry.predicate == predicate
            and entry.status == "active"
        ]
        summary = _quality_summary_label(entries, localize(label_key, response_language), response_language)
        if summary:
            details.append(summary)
    return details


def _quality_summary_label(entries: list[KnowledgeEntry], label: str, response_language: str = "pt-PT") -> str:
    if not entries:
        return ""
    values = {normalize_for_matching(entry.value) for entry in entries}
    strengths = {str((entry.metadata or {}).get("source_strength_for_claim") or "") for entry in entries}
    if "weak" in strengths and not {"strong", "useful"}.intersection(strengths):
        return localize("task.quality.summary.weak_positive", response_language, label=label)
    if "better" in values:
        return localize("task.quality.summary.advantage", response_language, label=label)
    if "good" in values and "mixed" in values:
        return localize("task.quality.summary.good_mixed", response_language, label=label)
    if "good" in values:
        return localize("task.quality.summary.good", response_language, label=label)
    if "mixed" in values:
        return localize("task.quality.summary.mixed", response_language, label=label)
    return ""


def _quality_score_and_reason(
    knowledge: list[KnowledgeEntry], option: str, predicate: str, label: str, response_language: str = "pt-PT"
) -> tuple[float, str]:
    entries = [
        entry
        for entry in knowledge
        if entry.status == "active"
        and normalize_for_matching(entry.subject) == normalize_for_matching(option)
        and entry.predicate == predicate
    ]
    if not entries:
        return 0.0, ""
    values = {normalize_for_matching(entry.value) for entry in entries}
    if "better" not in values:
        return 0.0, ""
    best_confidence = max(float(entry.confidence or 0.0) for entry in entries)
    strengths = {str((entry.metadata or {}).get("source_strength_for_claim") or "") for entry in entries}
    if "strong" in strengths:
        return 1.0, localize("task.quality.reason.strong", response_language, label=label)
    if "useful" in strengths:
        return 0.7, localize("task.quality.reason.useful", response_language, label=label)
    if "weak" in strengths or best_confidence < 0.55:
        return 0.25, localize("task.quality.reason.weak", response_language, label=label)
    return 0.5, localize("task.quality.reason.some", response_language, label=label)


def _dedupe_view_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        label = str(item.get("label", ""))
        key = _ascii_key(label)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _split_task_options(options_text: str) -> list[str]:
    return [part.strip(" ,.!?;:") for part in re.split(r"\s+e\s+|,", options_text or "") if part.strip(" ,.!?;:")]


def needs_task_reference_clarification(work_session, task, message: str) -> bool:
    if task is None or task.task_type != "generic_comparison":
        return False
    normalized = normalize_for_matching(message)
    if not contains_any_phrase(normalized, ("esse", "essa", "isso", "ele", "ela")):
        return False
    if not contains_any_phrase(normalized, ("melhor", "pior", "vale a pena", "escolher")):
        return False
    options = _task_option_titles(work_session, task)
    return not any(_subject_index(normalized, option, options) != -1 for option in options)


def task_option_titles(work_session, task) -> list[str]:
    return _task_option_titles(work_session, task)


def infer_unknowns(task, knowledge: list[KnowledgeEntry], response_language: str = "pt-PT") -> list[str]:
    # Phase 7.7.2: found via a real mid-comparison scenario run -- this
    # feeds EVERY travel/comparison/generic dialogue and query formatter
    # above, so leaving it hardcoded silently reintroduced Portuguese
    # fragments into sentences that were otherwise already localized.
    # option (canonical, user-provided) is never translated.
    active = _active_entries(knowledge)
    predicates = {entry.predicate for entry in active}
    if task.task_type == "generic_comparison":
        options = _subjects_with_kind(active, "option")
        unknowns: list[str] = []
        if len(options) < 2:
            unknowns.append(localize("task.reason.options_to_compare", response_language))
        has_price = {entry.subject for entry in active if entry.predicate == "price"}
        has_battery = {entry.subject for entry in active if entry.predicate == "battery"}
        has_camera = {entry.subject for entry in active if entry.predicate == "camera_quality"}
        for option in options:
            if option not in has_price:
                unknowns.append(localize("task.unknown.price_of", response_language, option=option))
            if "camera_priority" in predicates and option not in has_camera:
                unknowns.append(localize("task.unknown.camera_of", response_language, option=option))
            if "battery_priority" in predicates and option not in has_battery:
                unknowns.append(localize("task.unknown.battery_of", response_language, option=option))
        if not any(entry.kind == "criterion" for entry in active) and not any(entry.kind == "preference" for entry in active):
            unknowns.append(localize("task.unknown.decision_criteria", response_language))
        return _dedupe(unknowns)
    if task.task_type == "plan_trip":
        unknowns = []
        stage = infer_task_stage(task, active)
        if stage == "explore" and "destination" not in predicates:
            unknowns.append(localize("task.unknown.destination_type", response_language))
        if "date_range" not in predicates and "duration" not in predicates:
            unknowns.append(localize("task.unknown.dates_or_duration", response_language))
        if "budget_max" not in predicates:
            unknowns.append(localize("task.unknown.budget", response_language))
        if "travel_style" not in predicates:
            unknowns.append(localize("task.unknown.travel_pace", response_language))
        if stage == "plan":
            unknowns.extend(
                [
                    localize("task.unknown.accommodation", response_language),
                    localize("task.unknown.transport", response_language),
                    localize("task.unknown.itinerary", response_language),
                    localize("task.unknown.costs", response_language),
                ]
            )
        return _dedupe(unknowns)
    return [localize("task.unknown.next_concrete_step", response_language)]


def infer_task_stage(task, knowledge: list[KnowledgeEntry]) -> str:
    predicates = {entry.predicate for entry in _active_entries(knowledge)}
    if task.task_type == "plan_trip":
        if "destination" in predicates and ("date_range" in predicates or "duration" in predicates):
            return "plan"
        return "explore"
    if task.task_type == "generic_comparison":
        if any(entry.kind == "decision" for entry in knowledge):
            return "decide"
        if any(entry.predicate in {"price", "camera_quality", "battery"} for entry in knowledge):
            return "narrow"
        return "explore"
    return "explore"


def _ensure_options_from_task(work_session, task, added: list[KnowledgeEntry], message: str, turn_id: str) -> None:
    if task.task_type != "generic_comparison":
        return
    for obj in work_session.objects.values():
        if obj.kind == "option" and obj.task_id == task.id:
            entry = _upsert_entry(
                work_session,
                task.id,
                kind="option",
                subject=obj.title,
                predicate="is_option",
                value="true",
                message=message,
                turn_id=turn_id,
                replace_active=False,
            )
            if entry:
                added.append(entry)


def _extract_comparison_entries(work_session, task, message: str, turn_id: str) -> list[KnowledgeEntry]:
    entries: list[KnowledgeEntry] = []
    normalized = normalize_for_matching(message)
    correction = _is_correction(normalized)

    budget = _extract_budget(normalized)
    if budget:
        entries.append(_upsert_entry(work_session, task.id, "constraint", "current_task", "budget_max", budget, message, turn_id, replace_active=correction or True))

    for predicate, value in _extract_criterion_priorities(normalized):
        entries.append(_upsert_entry(work_session, task.id, "preference", "user", predicate, value, message, turn_id, replace_active=True))
    if contains_any_phrase(normalized, ("valorizo bastante a bateria", "tambem valorizo bastante a bateria", "também valorizo bastante a bateria")):
        entries.append(_upsert_entry(work_session, task.id, "preference", "user", "battery_priority", "medium", message, turn_id, replace_active=True))
    if contains_any_phrase(normalized, ("bateria ja nao e tao importante", "bateria já não é tão importante", "nao me importo tanto com bateria", "não me importo tanto com bateria", "nao dou tanta importancia a bateria", "não dou tanta importância à bateria", "nao dou tanta importancia à bateria")):
        entries.append(_upsert_entry(work_session, task.id, "preference", "user", "battery_priority", "low", message, turn_id, replace_active=True))
    if contains_any_phrase(normalized, ("melhor suporte", "suporte e importante", "suporte é importante", "prefiro pagar um pouco mais")):
        entries.append(_upsert_entry(work_session, task.id, "preference", "user", "support_priority", "high", message, turn_id, replace_active=True))

    options = _task_option_titles(work_session, task)
    for option in options:
        price = _extract_price_for_subject(message, option, options)
        if price:
            entries.append(_upsert_entry(work_session, task.id, "fact", option, "price", price, message, turn_id, replace_active=True))
        subject_fragment = _fragment_after_subject(message, option, options)
        subject_norm = normalize_for_matching(subject_fragment)
        if contains_any_phrase(subject_norm, ("melhor bateria", "melhor na bateria", "dura mais", "melhor autonomia", "aguenta mais tempo", "aguenta mais")):
            entries.append(_upsert_entry(work_session, task.id, "claim", option, "battery", "better", message, turn_id, replace_active=True, confidence=0.78))
        if contains_any_phrase(subject_norm, ("melhor camara", "melhor câmara")):
            kind = "opinion" if contains_any_phrase(normalized, ("acho", "parece", "parece me", "parece-me")) else "claim"
            entries.append(_upsert_entry(work_session, task.id, kind, option, "camera_quality", "better", message, turn_id, replace_active=True, confidence=0.7 if kind == "opinion" else 0.78))
        if contains_any_phrase(subject_norm, ("suporte 24h", "suporte 24", "suporte vinte e quatro", "melhor suporte")):
            entries.append(_upsert_entry(work_session, task.id, "claim", option, "support_quality", "better", message, turn_id, replace_active=True, confidence=0.78))
        if contains_any_phrase(subject_norm, ("horario laboral", "horário laboral", "so em horario", "só em horário")):
            entries.append(_upsert_entry(work_session, task.id, "claim", option, "support_quality", "limited", message, turn_id, replace_active=True, confidence=0.78))
    return [entry for entry in entries if entry is not None]


def _extract_criterion_priorities(normalized: str) -> list[tuple[str, str]]:
    priorities: list[tuple[str, str]] = []
    criteria = _criterion_mentions(normalized)
    if not criteria:
        return priorities

    if contains_any_phrase(
        normalized,
        (
            "mais importante",
            "parte mais importante",
            "valorizo mais",
            "pesa mais",
            "pesam mais",
            "dou mais importancia",
            "dou mais importância",
        ),
    ):
        for predicate in criteria[:1]:
            priorities.append((predicate, "high"))
    # Battery scenario 11: a plain declaration ("o critério é a bateria",
    # "critério: preço", "decidir pela autonomia") is a criterion too.
    if re.search(r"\bcriterios?\s*:", normalized) or contains_any_phrase(
        normalized,
        ("criterio e", "criterios sao", "decidir pela", "decidir pelo", "decidir por"),
    ):
        priorities.extend((predicate, "high") for predicate in criteria)
    less_than_match = re.search(r"\b(.+?)\s+importa(?:m)?(?:,)?\s+mas\s+menos\s+do\s+que\s+(.+)$", normalized)
    if not less_than_match:
        less_than_match = re.search(r"\b(.+?)\s+importa(?:m)?\s+menos\s+do\s+que\s+(.+)$", normalized)
    if less_than_match:
        lower = _criterion_mentions(less_than_match.group(1))
        higher = _criterion_mentions(less_than_match.group(2))
        priorities.extend((predicate, "low") for predicate in lower[:1])
        priorities.extend((predicate, "high") for predicate in higher[:1])
    if contains_any_phrase(normalized, ("mais importante que", "mais importante do que")) and len(criteria) >= 2:
        priorities.append((criteria[0], "high"))
        priorities.append((criteria[1], "medium"))
    return _dedupe_priority_pairs(priorities)


def _criterion_mentions(text: str) -> list[str]:
    mapping = (
        ("camera_priority", ("camara", "camera", "câmara", "fotografia", "video", "vídeo")),
        ("battery_priority", ("bateria", "autonomia")),
        ("support_priority", ("suporte", "apoio")),
        ("price_priority", ("preco", "preço", "custo")),
        ("location_priority", ("localizacao", "localização", "localizaçao", "zona", "localizacao")),
    )
    found: list[str] = []
    for predicate, phrases in mapping:
        if contains_any_phrase(text, phrases):
            found.append(predicate)
    return found


def _dedupe_priority_pairs(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    result: list[tuple[str, str]] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _extract_travel_entries(work_session, task, message: str, turn_id: str) -> list[KnowledgeEntry]:
    entries: list[KnowledgeEntry] = []
    normalized = normalize_for_matching(message)
    destination = _extract_destination(message)
    if destination:
        entries.append(_upsert_entry(work_session, task.id, "fact", "trip", "destination", destination, message, turn_id, replace_active=True))
    origin = _extract_origin(message)
    if origin:
        entries.append(_upsert_entry(work_session, task.id, "fact", "trip", "origin", origin, message, turn_id, replace_active=True))
    date_range = _extract_date_range(message)
    if date_range:
        entries.append(_upsert_entry(work_session, task.id, "fact", "trip", "date_range", date_range, message, turn_id, replace_active=True))
    duration = _extract_trip_duration(normalized)
    if duration:
        entries.append(_upsert_entry(work_session, task.id, "constraint", "trip", "duration", duration, message, turn_id, replace_active=True))
    budget = _extract_budget(normalized)
    if budget:
        entries.append(_upsert_entry(work_session, task.id, "constraint", "trip", "budget_max", budget, message, turn_id, replace_active=True))
    styles: list[str] = []
    if contains_any_phrase(normalized, ("descansar", "descanso")):
        styles.append("descanso")
    if contains_any_phrase(normalized, ("coisas para ver", "conhecer sitios", "conhecer sítios", "visitar")):
        styles.append("ter coisas para ver")
    if styles:
        entries.append(_upsert_entry(work_session, task.id, "preference", "user", "travel_style", " + ".join(styles), message, turn_id, replace_active=True))
    if contains_any_phrase(normalized, ("evitar voos muito longos", "voos muito longos", "voos curtos", "voo curto")):
        entries.append(_upsert_entry(work_session, task.id, "constraint", "trip", "flight_constraint", "voos curtos", message, turn_id, replace_active=True))
    return [entry for entry in entries if entry is not None]


def _extract_generic_entries(work_session, task, message: str, turn_id: str) -> list[KnowledgeEntry]:
    entries: list[KnowledgeEntry] = []
    budget = _extract_budget(normalize_for_matching(message))
    if budget:
        entries.append(_upsert_entry(work_session, task.id, "constraint", "current_task", "budget_max", budget, message, turn_id, replace_active=True))
    return [entry for entry in entries if entry is not None]


def _upsert_entry(
    work_session,
    task_id: str,
    kind: str,
    subject: str,
    predicate: str,
    value: str,
    message: str,
    turn_id: str,
    *,
    replace_active: bool = True,
    confidence: float = 1.0,
    source: KnowledgeSource | None = None,
    metadata: dict[str, Any] | None = None,
) -> KnowledgeEntry | None:
    if not task_id or not subject or not predicate or value == "":
        return None
    entries = task_knowledge(work_session, task_id)
    subject_norm = normalize_for_matching(subject)
    predicate_norm = normalize_for_matching(predicate)
    value_norm = normalize_for_matching(str(value))
    for existing in entries:
        if (
            existing.status == "active"
            and normalize_for_matching(existing.subject) == subject_norm
            and normalize_for_matching(existing.predicate) == predicate_norm
            and normalize_for_matching(str(existing.value)) == value_norm
        ):
            return existing
    replaced: list[str] = []
    if replace_active:
        for existing in entries:
            if (
                existing.status == "active"
                and normalize_for_matching(existing.subject) == subject_norm
                and normalize_for_matching(existing.predicate) == predicate_norm
                and normalize_for_matching(str(existing.value)) != value_norm
            ):
                existing.status = "superseded"
                replaced.append(existing.id)
    entry = KnowledgeEntry(
        id=f"k-{task_id}-{len(entries) + 1}",
        task_id=task_id,
        kind=kind,
        subject=subject,
        predicate=predicate,
        value=str(value),
        source=source
        or KnowledgeSource(
            source_type="user",
            reference=turn_id or f"turn-{len(entries) + 1}",
            original_text=message,
            created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        ),
        confidence=confidence,
        replaces=replaced,
        metadata=dict(metadata or {}),
    )
    entries.append(entry)
    work_session.relations.append(entry)
    return entry


def _refresh_reasoning(work_session, task, response_language: str = "pt-PT") -> TaskReasoningState:
    knowledge = _active_entries(task_knowledge(work_session, task.id))
    state = task_reasoning(work_session, task.id)
    state.stage = infer_task_stage(task, knowledge)
    state.updated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    state.stale = False
    if task.task_type == "generic_comparison":
        _reason_comparison(state, knowledge, response_language)
    else:
        state.current_recommendation = ""
        state.recommendation_confidence = 0.0
        state.reasons = []
        state.uncertainties = infer_unknowns(task, knowledge, response_language)
    return state


def _reason_comparison(state: TaskReasoningState, knowledge: list[KnowledgeEntry], response_language: str = "pt-PT") -> None:
    options = _subjects_with_kind(knowledge, "option")
    budget = _latest_value(knowledge, "current_task", "budget_max")
    priorities = {
        "camera_quality": _latest_value(knowledge, "user", "camera_priority"),
        "battery": _latest_value(knowledge, "user", "battery_priority"),
        "support_quality": _latest_value(knowledge, "user", "support_priority"),
        "price": _latest_value(knowledge, "user", "price_priority"),
    }
    camera_label = localize("task.quality.label.camera", response_language)
    battery_label = localize("task.quality.label.battery", response_language)
    support_label = localize("task.quality.label.support", response_language)
    scores = {option: 0.0 for option in options}
    reasons = {option: [] for option in options}
    for option in options:
        price = _latest_value(knowledge, option, "price")
        if price and budget:
            try:
                price_num = float(price)
                budget_num = float(budget)
                if price_num <= budget_num:
                    scores[option] += 2.0
                    reasons[option].append(
                        localize(
                            "task.reason.within_budget", response_language, price=int(price_num), budget=int(budget_num)
                        )
                    )
                else:
                    scores[option] -= 2.0
                    reasons[option].append(
                        localize(
                            "task.reason.over_budget", response_language, price=int(price_num), budget=int(budget_num)
                        )
                    )
            except ValueError:
                pass
        camera_score, camera_reason = _quality_score_and_reason(
            knowledge, option, "camera_quality", camera_label, response_language
        )
        if camera_score > 0:
            weight = 2.0 if priorities.get("camera_quality") == "high" else 1.0
            scores[option] += camera_score * weight
            reasons[option].append(camera_reason)
        battery_score, battery_reason = _quality_score_and_reason(
            knowledge, option, "battery", battery_label, response_language
        )
        if battery_score > 0:
            weight = 2.0 if priorities.get("battery") == "high" else (0.4 if priorities.get("battery") == "low" else 1.0)
            scores[option] += battery_score * weight
            reasons[option].append(battery_reason)
        support_quality = _latest_value(knowledge, option, "support_quality")
        support_score, support_reason = _quality_score_and_reason(
            knowledge, option, "support_quality", support_label, response_language
        )
        if support_score > 0:
            weight = 2.0 if priorities.get("support_quality") == "high" else 1.0
            scores[option] += support_score * weight
            reasons[option].append(support_reason)
        elif support_quality == "limited":
            scores[option] -= 0.8
            reasons[option].append(localize("task.reason.limited_support", response_language))
    if not scores:
        state.current_recommendation = ""
        state.reasons = []
        state.uncertainties = [localize("task.reason.options_to_compare", response_language)]
        return
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    blocking_unknown = _blocking_comparison_unknown(options, knowledge, priorities)
    if len(ranked) < 2 or ranked[0][1] <= 0 or abs(ranked[0][1] - ranked[1][1]) < 0.5 or blocking_unknown:
        state.current_recommendation = ""
        state.recommendation_confidence = 0.35
        state.reasons = []
    else:
        state.current_recommendation = ranked[0][0]
        state.recommendation_confidence = min(0.85, 0.45 + (ranked[0][1] - ranked[1][1]) / 6)
        state.reasons = reasons[ranked[0][0]][:3]
    state.uncertainties = infer_unknowns(
        type("Task", (), {"task_type": "generic_comparison"})(), knowledge, response_language
    )


def _blocking_comparison_unknown(options: list[str], knowledge: list[KnowledgeEntry], priorities: dict[str, str]) -> str:
    for predicate in ("camera_quality", "battery", "support_quality"):
        if priorities.get(predicate) != "high":
            continue
        subjects_with_evidence = {
            entry.subject
            for entry in knowledge
            if entry.status == "active" and entry.predicate == predicate and entry.source.source_type in {"web", "user"}
        }
        web_subjects_with_evidence = {
            entry.subject
            for entry in knowledge
            if entry.status == "active" and entry.predicate == predicate and entry.source.source_type == "web"
        }
        if not web_subjects_with_evidence:
            continue
        missing = [option for option in options if option not in subjects_with_evidence]
        if missing:
            return predicate
    return ""


def _format_known(task, knowledge: list[KnowledgeEntry], response_language: str = "pt-PT") -> str:
    if not knowledge:
        return localize("task.known.empty", response_language)
    stage = infer_task_stage(task, knowledge)
    lines = []
    if task.task_type == "generic_comparison":
        options = _subjects_with_kind(knowledge, "option")
        if options:
            lines.append(localize("task.known.comparing", response_language, options=_join_pt(options, response_language)))
    elif task.task_type == "plan_trip":
        mode_key = "task.known.mode_planning" if stage == "plan" else "task.known.mode_discovery"
        lines.append(localize("task.known.travel_mode", response_language, mode=localize(mode_key, response_language)))
    else:
        lines.append(localize("task.known.generic_context", response_language))
    for entry in [e for e in knowledge if e.kind != "option"][:6]:
        lines.append(_entry_phrase(entry, response_language) + ".")
    return " ".join(lines)


def _format_important(
    task, knowledge: list[KnowledgeEntry], reasoning: TaskReasoningState, response_language: str = "pt-PT"
) -> str:
    if not knowledge:
        return localize("task.important.empty", response_language)
    if task.task_type == "generic_comparison":
        parts: list[str] = []
        budget = _latest_value(knowledge, "current_task", "budget_max")
        if budget:
            parts.append(localize("task.important.budget_limit", response_language, budget=budget))
        if reasoning.current_recommendation:
            parts.append(
                localize(
                    "task.important.recommendation_now", response_language, recommendation=reasoning.current_recommendation
                )
            )
        priorities = [entry for entry in knowledge if entry.kind == "preference"]
        if priorities:
            parts.append(_entry_phrase(priorities[-1], response_language))
        unknowns = infer_unknowns(task, knowledge, response_language)
        if unknowns:
            parts.append(localize("task.important.biggest_gap", response_language, gap=unknowns[0]))
        return localize("task.important.comparison_header", response_language, parts="; ".join(parts[:4]))
    if task.task_type == "plan_trip":
        essentials = []
        for subject, predicate in (
            ("trip", "destination"),
            ("trip", "date_range"),
            ("trip", "duration"),
            ("trip", "budget_max"),
            ("user", "travel_style"),
            ("trip", "flight_constraint"),
        ):
            value = _latest_value(knowledge, subject, predicate)
            if value:
                essentials.append(
                    _entry_phrase(
                        KnowledgeEntry("", "", "fact", subject, predicate, value, KnowledgeSource()), response_language
                    )
                )
        unknowns = infer_unknowns(task, knowledge, response_language)
        if unknowns:
            essentials.append(localize("task.important.travel_missing", response_language, gap=unknowns[0]))
        return localize("task.important.header", response_language, parts="; ".join(essentials[:4]))
    return localize(
        "task.important.header",
        response_language,
        parts="; ".join(_entry_phrase(entry, response_language) for entry in knowledge[:3]),
    )


def _format_unknowns(task, knowledge: list[KnowledgeEntry], response_language: str = "pt-PT") -> str:
    unknowns = infer_unknowns(task, knowledge, response_language)
    if not unknowns:
        return localize("task.unknowns.none", response_language)
    return localize("task.unknowns.missing", response_language, gaps=_join_pt(unknowns[:4], response_language))


def _format_decisions(task, knowledge: list[KnowledgeEntry], response_language: str = "pt-PT") -> str:
    decisions = [entry for entry in knowledge if entry.kind == "decision"]
    if decisions:
        return localize(
            "task.decisions.registered",
            response_language,
            decisions=_join_pt([_entry_phrase(entry, response_language) for entry in decisions[:3]], response_language),
        )
    if task.task_type == "generic_comparison":
        unknowns = infer_unknowns(task, knowledge, response_language)
        if unknowns:
            return localize(
                "task.decisions.comparison_missing", response_language, gaps=_join_pt(unknowns[:2], response_language)
            )
        return localize("task.decisions.comparison_pending", response_language)
    if task.task_type == "plan_trip":
        stage = infer_task_stage(task, knowledge)
        if stage == "plan":
            return localize("task.decisions.travel_plan", response_language)
        return localize("task.decisions.travel_destination", response_language)
    return localize("task.decisions.generic", response_language)


def _format_recommendation(reasoning: TaskReasoningState, response_language: str = "pt-PT") -> str:
    if not reasoning.current_recommendation:
        gaps = (
            _join_pt(reasoning.uncertainties[:3], response_language)
            if reasoning.uncertainties
            else localize("task.recommendation.more_data", response_language)
        )
        return localize("task.recommendation.no_confidence", response_language, gaps=gaps)
    reasons = "; ".join(reasoning.reasons) if reasoning.reasons else localize("task.recommendation.default_reason", response_language)
    return localize(
        "task.recommendation.provisional",
        response_language,
        recommendation=reasoning.current_recommendation,
        reasons=reasons,
    )


def _format_travel_recommendation(task, knowledge: list[KnowledgeEntry], response_language: str = "pt-PT") -> str:
    unknowns = infer_unknowns(task, knowledge, response_language)
    stage = infer_task_stage(task, knowledge)
    if stage == "explore" and unknowns:
        return localize("task.travel_rec.no_confidence", response_language, gaps=_join_pt(unknowns[:2], response_language))
    if unknowns:
        return localize(
            "task.travel_rec.can_start_planning", response_language, gaps=_join_pt(unknowns[:2], response_language)
        )
    return localize("task.travel_rec.ready", response_language)


def _format_why(reasoning: TaskReasoningState, response_language: str = "pt-PT") -> str:
    if not reasoning.current_recommendation:
        return localize("task.why.no_data", response_language)
    reasons = "; ".join(reasoning.reasons) if reasoning.reasons else localize("task.why.default_reason", response_language)
    return localize("task.why.because", response_language, reasons=reasons)


def _format_sources(knowledge: list[KnowledgeEntry], message: str, response_language: str = "pt-PT") -> str:
    # Phase 7.7.2 (section 5/6): "Veio de.../It came from..." is Echo-
    # authored and localizes; title/evidence/original_text are source-
    # derived content and are NEVER translated -- only the fallback label
    # ("web source") used when a fetched entry has no title at all is
    # Echo-authored and localizes too.
    normalized = normalize_for_matching(message)
    matching = [entry for entry in knowledge if normalize_for_matching(entry.subject) in normalized or normalize_for_matching(entry.predicate) in normalized]
    if not matching:
        matching = knowledge[-3:]
    if not matching:
        return localize("task.sources.none", response_language)
    entry = matching[-1]
    if entry.source.source_type == "web":
        title = str(entry.metadata.get("source_title") or entry.source.reference or localize("task.sources.default_web_label", response_language))
        evidence = entry.source.original_text or entry.value
        suffix = (
            localize("task.sources.web_reference_suffix", response_language, reference=entry.source.reference)
            if entry.source.reference
            else "."
        )
        return localize("task.sources.from_web", response_language, title=title, evidence=evidence) + suffix
    return localize("task.sources.from_user", response_language, text=entry.source.original_text)


def _format_evidence_answer(task, knowledge: list[KnowledgeEntry], message: str, response_language: str = "pt-PT") -> str:
    # Phase 7.7.2 (section 5/6): connective sentences/confidence phrasing
    # localize; subject/option names (canonical, user-provided) are never
    # translated.
    web_entries = [
        entry for entry in knowledge
        if entry.source.source_type == "web" and entry.status == "active"
    ]
    if not web_entries:
        return localize("task.evidence.no_sources", response_language)

    predicate = _predicate_from_evidence_question(message)
    options = _subjects_with_kind(knowledge, "option")
    subjects = _subjects_mentioned_in_message(options or [entry.subject for entry in web_entries], message)
    scoped_entries = web_entries
    if subjects:
        scoped_entries = [entry for entry in web_entries if _entry_subject_matches_any(entry, subjects)]
    if predicate:
        scoped_entries = [entry for entry in scoped_entries if entry.predicate == predicate]

    if _asks_comparative_proof(message) and len(options) >= 2:
        subjects_with_evidence = {
            option for option in options
            if any(_entry_subject_matches_any(entry, [option]) and (not predicate or entry.predicate == predicate) for entry in web_entries)
        }
        missing = [option for option in options if option not in subjects_with_evidence]
        if missing:
            known = (
                _join_pt(sorted(subjects_with_evidence), response_language)
                if subjects_with_evidence
                else localize("task.evidence.one_of_the_options", response_language)
            )
            return localize(
                "task.evidence.comparative_no_proof",
                response_language,
                known=known,
                missing=_join_pt(missing[:2], response_language),
            )
    if not scoped_entries:
        target = _join_pt(subjects, response_language) if subjects else localize("task.evidence.that_point", response_language)
        return localize("task.evidence.no_direct_evidence", response_language, target=target)

    strong = [entry for entry in scoped_entries if _entry_evidence_strength(entry) == "strong"]
    useful = [entry for entry in scoped_entries if _entry_evidence_strength(entry) == "useful"]
    weak = [entry for entry in scoped_entries if _entry_evidence_strength(entry) == "weak"]
    positive = [entry for entry in scoped_entries if normalize_for_matching(entry.value) in {"good", "better", "boa", "bom", "superior", "melhor"}]
    mixed = [entry for entry in scoped_entries if normalize_for_matching(entry.value) == "mixed"]
    target = (
        _join_pt(subjects, response_language)
        if subjects
        else _join_pt(_dedupe([entry.subject for entry in scoped_entries])[:2], response_language)
    )
    criterion = _predicate_natural_label(predicate or scoped_entries[-1].predicate, response_language)

    if _asks_sufficiency(message):
        if len(options) >= 2:
            subjects_with_evidence = {
                option for option in options
                if any(_entry_subject_matches_any(entry, [option]) and (not predicate or entry.predicate == predicate) for entry in web_entries)
            }
            missing = [option for option in options if option not in subjects_with_evidence]
            if missing:
                return localize(
                    "task.evidence.sufficiency_missing",
                    response_language,
                    target=target,
                    missing=_join_pt(missing[:2], response_language),
                )
        if strong or useful:
            return localize("task.evidence.sufficiency_yes", response_language)
        return localize("task.evidence.sufficiency_no", response_language)

    if strong or useful:
        if positive and not mixed:
            return localize("task.evidence.strong_positive", response_language, target=target, criterion=criterion)
        if mixed:
            return localize("task.evidence.strong_mixed", response_language, target=target, criterion=criterion)
        return localize("task.evidence.strong_useful_base", response_language, target=target)
    if weak:
        return localize("task.evidence.weak_signals", response_language, target=target)
    return localize("task.evidence.not_enough", response_language)


def _predicate_from_evidence_question(message: str) -> str:
    normalized = normalize_for_matching(message)
    if contains_any_phrase(normalized, ("camara", "camera", "fotografia", "video", "vídeo")):
        return "camera_quality"
    if contains_any_phrase(normalized, ("bateria", "autonomia")):
        return "battery"
    if contains_any_phrase(normalized, ("preco", "preço", "custa", "valor")):
        return "price"
    if contains_any_phrase(normalized, ("suporte", "apoio")):
        return "support_quality"
    return ""


def _subjects_mentioned_in_message(subjects: list[str], message: str) -> list[str]:
    normalized = normalize_for_matching(message)
    found: list[str] = []
    for subject in _dedupe(subjects):
        subject_norm = normalize_for_matching(subject)
        if not subject_norm:
            continue
        subject_tokens = [token for token in subject_norm.split() if len(token) >= 3]
        direct = subject_norm in normalized or normalized in subject_norm
        token_match = any(re.search(rf"\b{re.escape(token)}\b", normalized) for token in subject_tokens)
        if direct or token_match:
            found.append(subject)
    return found


def _entry_subject_matches_any(entry: KnowledgeEntry, subjects: list[str]) -> bool:
    entry_subject = normalize_for_matching(entry.subject)
    for subject in subjects:
        subject_norm = normalize_for_matching(subject)
        if subject_norm and (subject_norm in entry_subject or entry_subject in subject_norm):
            return True
    return False


def _entry_evidence_strength(entry: KnowledgeEntry) -> str:
    strength = str((entry.metadata or {}).get("source_strength_for_claim") or "").strip().lower()
    return strength if strength in {"strong", "useful", "weak"} else "unknown"


def _asks_comparative_proof(message: str) -> bool:
    normalized = normalize_for_matching(message)
    return contains_any_phrase(normalized, ("prova", "melhor que", "superior a", "comparar"))


def _asks_sufficiency(message: str) -> bool:
    normalized = normalize_for_matching(message)
    return contains_any_phrase(normalized, ("ja chega", "já chega", "chegam", "suficiente", "decidir", "comparar"))


def _predicate_natural_label(predicate: str, response_language: str = "pt-PT") -> str:
    keys = {
        "camera_quality": "task.predicate_label.camera_quality",
        "battery": "task.predicate_label.battery",
        "price": "task.predicate_label.price",
        "support_quality": "task.predicate_label.support_quality",
    }
    key = keys.get(predicate)
    return localize(key, response_language) if key else predicate.replace("_", " ")


def _format_conflicts(knowledge: list[KnowledgeEntry], response_language: str = "pt-PT") -> str:
    # subject/predicate/value below are canonical extracted data (never
    # translated); only the wrapper sentences localize (section 5/6).
    conflicts = []
    grouped: dict[tuple[str, str], set[str]] = {}
    for entry in knowledge:
        if not _entry_can_conflict(entry):
            continue
        grouped.setdefault((entry.subject, entry.predicate), set()).add(entry.value)
    for (subject, predicate), values in grouped.items():
        if _values_are_semantically_conflicting(values, predicate):
            conflicts.append(f"{subject} / {predicate}: {_join_pt(sorted(values), response_language)}")
    if not conflicts:
        return localize("task.conflicts.none", response_language)
    return localize("task.conflicts.found", response_language, items="; ".join(conflicts[:3]))


def _entry_can_conflict(entry: KnowledgeEntry) -> bool:
    metadata = getattr(entry, "metadata", {}) or {}
    if str(metadata.get("relevance") or "") == "irrelevant":
        return False
    if str(metadata.get("applicability") or "") in {"family_related", "adjacent"}:
        return False
    if entry.predicate in {"price", "cost"} and str(metadata.get("semantic_role") or "") == "destination":
        return False
    return entry.predicate in {"price", "opening_hours", "availability", "transport", "battery"}


def _values_are_semantically_conflicting(values: set[str], predicate: str) -> bool:
    normalized = {normalize_for_matching(value) for value in values if value}
    if len(normalized) < 2:
        return False
    if predicate == "price":
        numeric_prices = {
            match.group(1).replace(",", ".")
            for value in normalized
            for match in re.finditer(r"\b(\d{2,5}(?:[,.]\d+)?)\b", value)
        }
        return len(numeric_prices) > 1
    joined = " ".join(normalized)
    if contains_any_phrase(joined, ("open", "aberto")) and contains_any_phrase(joined, ("closed", "fechado")):
        return True
    prices = {
        match.group(1).replace(",", ".")
        for value in normalized
        for match in re.finditer(r"\b(\d{2,5}(?:[,.]\d+)?)\s*(?:eur|euros|€)\b", value)
    }
    if len(prices) > 1:
        return True
    return False


def _format_next_step(task, knowledge: list[KnowledgeEntry], response_language: str = "pt-PT") -> str:
    stage = infer_task_stage(task, knowledge)
    if task.task_type == "plan_trip" and stage == "plan":
        return localize("task.next_step.travel_organize", response_language)
    unknowns = infer_unknowns(task, knowledge, response_language)
    if unknowns:
        return localize("task.next_step.close_gaps", response_language, gaps=_join_pt(unknowns[:2], response_language))
    return localize("task.next_step.choose_direction", response_language)


def _active_entries(entries: list[KnowledgeEntry]) -> list[KnowledgeEntry]:
    return [entry for entry in entries if entry.status == "active"]


def _subjects_with_kind(entries: list[KnowledgeEntry], kind: str) -> list[str]:
    return _dedupe([entry.subject for entry in entries if entry.kind == kind])


def _latest_value(entries: list[KnowledgeEntry], subject: str, predicate: str) -> str:
    subject_norm = normalize_for_matching(subject)
    predicate_norm = normalize_for_matching(predicate)
    for entry in reversed(entries):
        if normalize_for_matching(entry.subject) == subject_norm and normalize_for_matching(entry.predicate) == predicate_norm:
            return entry.value
    return ""


def _task_option_titles(work_session, task) -> list[str]:
    return [
        obj.title
        for obj in work_session.objects.values()
        if obj.kind == "option" and obj.task_id == task.id and obj.title
    ]


def _extract_budget(normalized: str) -> str:
    match = re.search(r"(?:mais de|ate(?:\s+aos?)?|até(?:\s+aos?)?|maximo|maxímo|max)\s+(\d{2,5})\s*(?:euros|eur|€)?", normalized)
    if match:
        return match.group(1)
    match = re.search(r"orcamento\s+(?:de\s+)?(\d{2,5})", normalized)
    return match.group(1) if match else ""


def _extract_price_for_subject(message: str, subject: str, subjects: list[str] | None = None) -> str:
    normalized_message = normalize_for_matching(message)
    index = _subject_index(normalized_message, subject, subjects)
    if index == -1:
        return ""
    matched = _matched_subject_text(normalized_message, subject, index)
    fragment = normalized_message[index + len(matched): index + len(matched) + 60]
    match = re.search(r"(?:custa|esta a|está a|por|pelos?)?\s*(\d{2,5})\s*(?:euros|eur|€)?", fragment)
    if not match:
        return ""
    before = fragment[: match.start(1)]
    after = fragment[match.end(1): match.end(1) + 2].lower()
    has_price_marker = bool(re.search(r"(?:custa|esta a|está a|por|pelos?|a)\s*$", before))
    has_currency = bool(re.match(r"\s*(?:euros|eur|€)", fragment[match.end(1):]))
    previous_context = normalized_message[max(0, index - 60):index]
    has_contextual_price_marker = bool(re.search(r"\b(?:custa|esta a|está a|preco|preço)\b", previous_context))
    if not (has_price_marker or has_currency or has_contextual_price_marker):
        return ""
    if after.startswith("h") and not (has_price_marker or has_currency):
        return ""
    return match.group(1)


def _fragment_after_subject(message: str, subject: str, subjects: list[str] | None = None) -> str:
    normalized_message = normalize_for_matching(message)
    index = _subject_index(normalized_message, subject, subjects)
    if index == -1:
        return ""
    end = index + 100
    for other in subjects or []:
        other_index = _subject_index(normalized_message, other, subjects)
        matched = _matched_subject_text(normalized_message, subject, index)
        if other_index != -1 and other_index <= index + len(matched):
            continue
        if other_index != -1:
            end = min(end, other_index)
    return normalized_message[index:end]


def _subject_index(normalized_message: str, subject: str, subjects: list[str] | None = None) -> int:
    candidates = _subject_aliases(subject)
    exact_matches = [normalized_message.find(candidate) for candidate in candidates if candidate and normalized_message.find(candidate) != -1]
    if exact_matches:
        return min(exact_matches)
    subject_key = normalize_for_matching(subject)
    words = re.findall(r"\b[\wÀ-ÿ]{4,}\b", normalized_message)
    matches = [
        (word, SequenceMatcher(None, word, subject_key).ratio())
        for word in words
        if abs(len(word) - len(subject_key)) <= 2
    ]
    plausible = [(word, score) for word, score in matches if score >= 0.86]
    if not plausible:
        return -1
    plausible.sort(key=lambda item: item[1], reverse=True)
    if len(plausible) > 1 and plausible[0][1] - plausible[1][1] < 0.08:
        return -1
    if subjects:
        competing_subjects = [normalize_for_matching(item) for item in subjects if normalize_for_matching(item) != subject_key]
        for word, score in plausible[:1]:
            for other in competing_subjects:
                if abs(len(word) - len(other)) <= 2 and SequenceMatcher(None, word, other).ratio() >= score - 0.03:
                    return -1
    return normalized_message.find(plausible[0][0])


def _matched_subject_text(normalized_message: str, subject: str, index: int) -> str:
    for candidate in _subject_aliases(subject):
        if candidate and normalized_message.startswith(candidate, index):
            return candidate
    match = re.match(r"[\wÀ-ÿ]+", normalized_message[index:])
    return match.group(0) if match else normalize_for_matching(subject)


def _subject_aliases(subject: str) -> list[str]:
    normalized = normalize_for_matching(subject)
    aliases = [normalized]
    if normalized == "apple":
        aliases.append("iphone")
    return aliases


def _extract_destination(message: str) -> str:
    match = re.search(r"\b(?:vou\s+a|vou\s+para|destino\s+e|destino\s+é)\s+(?:o|a|os|as)?\s*([A-ZÀ-Ý][\wÀ-ÿ]+(?:\s+[A-ZÀ-Ý][\wÀ-ÿ]+){0,3})", message)
    if match:
        return match.group(1).strip(" .,!?:;")
    normalized = normalize_for_matching(message)
    if "estocolmo" in normalized:
        return "Estocolmo"
    return ""


def _extract_origin(message: str) -> str:
    match = re.search(r"\b(?:a partir(?:\s+de|\s+do|\s+da)?|partir\s+de|desde(?:\s+de|\s+do|\s+da)?)\s+([A-ZÀ-Ý][\wÀ-ÿ]+(?:\s+[A-ZÀ-Ý][\wÀ-ÿ]+){0,3})", message)
    if match:
        return match.group(1).strip(" .,!?:;")
    return ""


def _extract_date_range(message: str) -> str:
    match = re.search(r"\b(\d{1,2})\s+a\s+(\d{1,2})\s+de\s+([A-Za-zÀ-ÿ]+)", message, flags=re.IGNORECASE)
    if match:
        return f"{match.group(1)} a {match.group(2)} de {match.group(3)}"
    return ""


def _extract_trip_duration(normalized: str) -> str:
    if contains_any_phrase(normalized, ("uma semana", "cerca de uma semana")):
        return "cerca de uma semana"
    match = re.search(r"\b(?:uns?|cerca de|aproximadamente)?\s*(\d{1,2})\s+dias\b", normalized)
    if match:
        return f"{match.group(1)} dias"
    return ""


def _is_correction(normalized: str) -> bool:
    return contains_any_phrase(normalized, ("afinal", "corrige", "nao,", "não,", "enganei me", "enganei-me"))


def _entry_phrase(entry: KnowledgeEntry, response_language: str = "pt-PT") -> str:
    # Phase 7.7.1 (section 4/18): the LABEL per predicate localizes; entry.
    # subject/value (destinations, prices, option names, free-text values)
    # are canonical extracted data and are never translated.
    pred = entry.predicate
    if pred == "budget_max":
        return localize("task.entry.budget_max", response_language, value=entry.value)
    if pred == "price":
        return localize("task.entry.price", response_language, subject=entry.subject, value=entry.value)
    if pred == "camera_priority":
        return localize("task.entry.camera_priority", response_language)
    if pred == "battery_priority":
        if entry.value == "low":
            return localize("task.entry.battery_priority.low", response_language)
        if entry.value == "medium":
            return localize("task.entry.battery_priority.medium", response_language)
        return localize("task.entry.battery_priority.high", response_language)
    if pred == "camera_quality":
        return localize("task.entry.camera_quality", response_language, subject=entry.subject)
    if pred == "battery":
        return localize("task.entry.battery", response_language, subject=entry.subject)
    if pred == "support_priority":
        return localize("task.entry.support_priority", response_language)
    if pred == "price_priority":
        if entry.value == "low":
            return localize("task.entry.price_priority.low", response_language)
        return localize("task.entry.price_priority.high", response_language)
    if pred == "location_priority":
        return localize("task.entry.location_priority", response_language)
    if pred == "support_quality":
        if entry.value == "better":
            return localize("task.entry.support_quality.better", response_language, subject=entry.subject)
        if entry.value == "limited":
            return localize("task.entry.support_quality.limited", response_language, subject=entry.subject)
        return localize("task.entry.support_quality.other", response_language, subject=entry.subject, value=entry.value)
    if pred == "destination":
        return localize("task.entry.destination", response_language, value=entry.value)
    if pred == "origin":
        return localize("task.entry.origin", response_language, value=entry.value)
    if pred == "date_range":
        return localize("task.entry.date_range", response_language, value=entry.value)
    if pred == "duration":
        return localize("task.entry.duration", response_language, value=entry.value)
    if pred == "travel_style":
        return localize("task.entry.travel_style", response_language, value=entry.value)
    if pred == "flight_constraint":
        return entry.value
    if pred == "destination_candidate":
        return f"{entry.subject}: {entry.value}"
    if pred == "activities":
        return localize("task.entry.activities", response_language, value=entry.value, subject=entry.subject)
    return f"{entry.subject}: {entry.value}"


def _join_pt(items: list[str], response_language: str = "pt-PT") -> str:
    clean = [item for item in items if item]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    return ", ".join(clean[:-1]) + f" {localize('task.join.and', response_language)} " + clean[-1]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = _ascii_key(item)
        if key and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _ascii_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).casefold().strip()

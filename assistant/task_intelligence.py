from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

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


def answer_task_query(work_session, task, message: str) -> str:
    if task is None:
        return ""
    query = classify_task_query(message)
    if not query:
        return ""
    knowledge = _active_entries(task_knowledge(work_session, task.id))
    if query == "known":
        return _format_known(task, knowledge)
    if query == "important":
        return _format_important(task, knowledge, task_reasoning(work_session, task.id))
    if query == "unknowns":
        return _format_unknowns(task, knowledge)
    if query == "decisions":
        return _format_decisions(task, knowledge)
    if query == "recommendation":
        if task.task_type == "plan_trip":
            return _format_travel_recommendation(task, knowledge)
        reasoning = _refresh_reasoning(work_session, task)
        return _format_recommendation(reasoning)
    if query == "why":
        reasoning = task_reasoning(work_session, task.id)
        if not reasoning.reasons:
            reasoning = _refresh_reasoning(work_session, task)
        return _format_why(reasoning)
    if query == "source":
        return _format_sources(knowledge, message)
    if query == "evidence":
        return _format_evidence_answer(task, knowledge, message)
    if query == "conflicts":
        return _format_conflicts(knowledge)
    if query == "next_step":
        return _format_next_step(task, knowledge)
    return ""


def task_dialogue_response(
    work_session,
    task,
    message: str = "",
    *,
    knowledge_delta: list[KnowledgeEntry] | None = None,
) -> str:
    """Natural task dialogue grounded in the current structured state.

    The UI projection and this text both derive from the same TaskKnowledge
    and TaskReasoningState. The delta only explains what changed this turn.
    """
    if task is None:
        return ""
    knowledge = _knowledge_with_task_fields(task, _active_entries(task_knowledge(work_session, task.id)))
    reasoning = _refresh_reasoning(work_session, task)
    delta = [entry for entry in (knowledge_delta or []) if entry and entry.status == "active"]
    if task.task_type == "plan_trip":
        plan = _plan_travel_dialogue(task, knowledge, reasoning, delta, message)
    elif task.task_type == "generic_comparison":
        plan = _plan_comparison_dialogue(task, knowledge, reasoning, delta)
    else:
        plan = _plan_generic_dialogue(task, knowledge, reasoning, delta)
    rendered = plan.render()
    return rendered or _fallback_task_dialogue(task, knowledge, reasoning)


def task_card_summary(work_session, task) -> list[str]:
    if task is None:
        return []
    knowledge = _active_entries(task_knowledge(work_session, task.id))
    if not knowledge:
        return []
    lines: list[str] = []
    if task.task_type == "generic_comparison":
        options = _subjects_with_kind(knowledge, "option")
        if options:
            lines.append("Opções: " + _join_pt(options[:4]))
        constraints = [e for e in knowledge if e.kind == "constraint"]
        preferences = [e for e in knowledge if e.kind == "preference"]
        if constraints:
            lines.append("Limite: " + _entry_phrase(constraints[-1]))
        if preferences:
            lines.append("Prioridade: " + _entry_phrase(preferences[-1]))
    elif task.task_type == "plan_trip":
        essentials = [e for e in knowledge if e.predicate in {"destination", "date_range", "duration", "budget_max", "travel_style", "flight_constraint"}]
        for entry in essentials[:4]:
            lines.append(_entry_phrase(entry))
    else:
        for entry in knowledge[:4]:
            lines.append(_entry_phrase(entry))
    unknowns = infer_unknowns(task, knowledge)
    if unknowns:
        lines.append("Em aberto: " + _join_pt(unknowns[:3]))
    return lines


def cognitive_task_view_model(work_session, task, *, title: str = "") -> dict[str, Any]:
    """Read-only projection for the Task Surface.

    This deliberately derives from task knowledge/reasoning every time. It
    does not create a second mutable source of truth for the UI.
    """
    if task is None:
        return {}
    knowledge = _knowledge_with_task_fields(task, _active_entries(task_knowledge(work_session, task.id)))
    reasoning = _refresh_reasoning(work_session, task)
    stage = infer_task_stage(task, knowledge) or reasoning.stage
    sections: list[CognitiveTaskSection] = []

    overview_items = _overview_items(task, knowledge)
    if overview_items:
        sections.append(CognitiveTaskSection("overview", _overview_title(task), "high", overview_items))

    if task.task_type == "generic_comparison":
        option_items = _option_view_items(knowledge)
        if option_items:
            sections.append(CognitiveTaskSection("options", "Opções", "high", option_items))
    elif task.task_type == "plan_trip":
        found_items = _travel_found_items(knowledge)
        if found_items:
            sections.append(CognitiveTaskSection("found", "Encontrado", "high", found_items[:5]))

    constraints = _entries_for_roles(knowledge, {"constraint"})
    if constraints:
        sections.append(CognitiveTaskSection("constraints", "Limites", "high", _entry_view_items(constraints[:3])))

    preferences = _entries_for_roles(knowledge, {"preference"})
    if preferences:
        sections.append(CognitiveTaskSection("priorities", "Prioridades", "medium", _entry_view_items(preferences[:4])))

    decisions = _entries_for_roles(knowledge, {"decision"})
    if decisions:
        sections.append(CognitiveTaskSection("decisions", "Decisões", "high", _entry_view_items(decisions[:3])))

    facts = _non_duplicated_fact_items(task, knowledge)
    if facts:
        sections.append(CognitiveTaskSection("key_information", "O que já sabemos", "medium", facts[:6]))

    if reasoning.current_recommendation:
        sections.append(
            CognitiveTaskSection(
                "recommendation",
                "Neste momento",
                "high",
                [{"label": f"{reasoning.current_recommendation} parece a melhor opção", "details": reasoning.reasons[:3]}],
            )
        )
    elif task.task_type == "plan_trip" and stage == "plan":
        sections.append(
            CognitiveTaskSection(
                "plan_areas",
                "A organizar",
                "medium",
                [{"label": item, "details": []} for item in ("Alojamento", "Transportes", "Itinerário", "Custos")],
            )
        )

    conflicts = _conflict_items(knowledge)
    if conflicts:
        sections.append(CognitiveTaskSection("conflicts", "Atenção", "high", conflicts[:3]))

    unknowns = _prioritized_unknowns(task, knowledge, reasoning)
    if unknowns:
        sections.append(CognitiveTaskSection("unknowns", "Ainda falta", "medium", [{"label": item, "details": []} for item in unknowns[:4]]))

    next_step = _next_step_label(task, knowledge, reasoning)
    if next_step:
        sections.append(CognitiveTaskSection("next_step", "Próximo passo", "high", text=next_step))

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


def _overview_title(task) -> str:
    if task.task_type == "generic_comparison":
        return "Comparação"
    if task.task_type == "plan_trip":
        return "Planeamento"
    return "Objetivo"


def _overview_items(task, knowledge: list[KnowledgeEntry]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    category = getattr(task, "parameters", {}).get("category", "")
    if category and task.task_type == "generic_comparison":
        items.append({"label": f"Categoria: {category}", "details": []})
    if getattr(task, "goal", ""):
        items.append({"label": task.goal, "details": []})
    for predicate in ("destination", "origin", "date_range"):
        value = _latest_value(knowledge, "trip", predicate)
        if value:
            items.append({"label": _entry_phrase(KnowledgeEntry("", "", "fact", "trip", predicate, value, KnowledgeSource())), "details": []})
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


def _option_view_items(knowledge: list[KnowledgeEntry]) -> list[dict[str, Any]]:
    options = _subjects_with_kind(knowledge, "option")
    items: list[dict[str, Any]] = []
    for option in options:
        details: list[str] = []
        price = _latest_value(knowledge, option, "price")
        if price:
            details.append(f"{price} €")
        details.extend(_quality_view_details(knowledge, option))
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


def _entry_view_items(entries: list[KnowledgeEntry]) -> list[dict[str, Any]]:
    return _dedupe_view_items([{"label": _entry_phrase(entry), "details": []} for entry in entries])


def _non_duplicated_fact_items(task, knowledge: list[KnowledgeEntry]) -> list[dict[str, Any]]:
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
        {"label": _entry_phrase(entry), "details": []}
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


def _prioritized_unknowns(task, knowledge: list[KnowledgeEntry], reasoning: TaskReasoningState) -> list[str]:
    unknowns = infer_unknowns(task, knowledge) or reasoning.uncertainties
    if task.task_type == "generic_comparison":
        predicates = {entry.predicate: entry.value for entry in knowledge if entry.kind == "preference"}
        ordered = ["critérios de decisão"]
        if predicates.get("camera_priority") == "high":
            ordered.append("câmara")
        if predicates.get("battery_priority") == "high":
            ordered.append("autonomia")
        if predicates.get("support_priority") == "high":
            ordered.append("suporte")
        if predicates.get("price_priority") == "high":
            ordered.append("preço")
        ordered.extend(["preço", "câmara", "autonomia", "suporte"])
        priority = tuple(_dedupe(ordered))
    elif task.task_type == "plan_trip":
        priority = ("tipo de destino", "datas", "duração", "orçamento", "ritmo", "alojamento", "transportes")
    else:
        priority = ("próximo passo",)
    return sorted(_dedupe(unknowns), key=lambda item: _priority_index(item, priority))


def _priority_index(item: str, priority: tuple[str, ...]) -> int:
    normalized = normalize_for_matching(item)
    for index, marker in enumerate(priority):
        if marker in normalized:
            return index
    return len(priority)


def _next_step_label(task, knowledge: list[KnowledgeEntry], reasoning: TaskReasoningState) -> str:
    unknowns = _prioritized_unknowns(task, knowledge, reasoning)
    if unknowns:
        return "Fechar " + _join_pt(unknowns[:2])
    if task.task_type == "plan_trip" and reasoning.stage == "plan":
        return "Organizar alojamento, transportes, itinerário e custos"
    if reasoning.current_recommendation:
        return "Confirmar os pontos em aberto antes de decidir"
    return ""


def _plan_travel_dialogue(
    task,
    knowledge: list[KnowledgeEntry],
    reasoning: TaskReasoningState,
    delta: list[KnowledgeEntry],
    message: str,
) -> TaskDialoguePlan:
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
            ack = f"Ótimo. Já temos {destination}"
            if date_range:
                ack += f" e as datas, {date_range}"
            ack += "."
            return TaskDialoguePlan(
                acknowledgement=ack,
                implication="Agora faz sentido organizar alojamento, transportes, itinerário e orçamento.",
                next_question="Por onde queres começar?",
            )
        if "budget_max" in delta_predicates:
            scope = " para alojamento, transportes, alimentação e atividades, sem contar os voos" if contains_any_phrase(normalized, ("sem contar os voos", "sem voos")) else ""
            return TaskDialoguePlan(
                acknowledgement=f"Ficamos então com até {budget} €{scope}.",
                next_step="O próximo passo mais útil é perceber o tipo de alojamento e a zona onde faz sentido ficar.",
            )
        return TaskDialoguePlan(
            acknowledgement="Temos destino e datas definidos.",
            next_step="Agora precisamos de organizar alojamento, transportes, itinerário e orçamento.",
        )

    if {"duration", "budget_max"}.intersection(delta_predicates):
        known = []
        if duration:
            known.append(duration)
        if budget:
            known.append(f"até {budget} €")
        return TaskDialoguePlan(
            acknowledgement=f"Ótimo, já temos {_join_pt(known)}.",
            implication="Falta-me perceber melhor o estilo da viagem." if "travel_style" not in predicates else "",
            next_question=_travel_next_question(knowledge, reasoning),
        )
    if "travel_style" in delta_predicates:
        return TaskDialoguePlan(
            acknowledgement=f"Então procuramos {travel_style}.",
            next_question=_travel_next_question(knowledge, reasoning),
        )
    if "flight_constraint" in delta_predicates:
        summary = _travel_basis_summary(duration, budget, travel_style, flight_constraint)
        return TaskDialoguePlan(
            acknowledgement=f"Perfeito. {summary}.",
            next_step="Já consigo começar a procurar destinos adequados.",
        )
    if not delta:
        if destination and "travel_style" not in predicates:
            return TaskDialoguePlan(
                acknowledgement="Acho que ainda me falta perceber como gostas de viajar.",
                next_question=_travel_next_question(knowledge, reasoning),
            )
        return TaskDialoguePlan(
            acknowledgement="Claro.",
            next_question=_travel_next_question(knowledge, reasoning),
        )
    return TaskDialoguePlan(
        acknowledgement="Certo, isso já ajuda.",
        next_question=_travel_next_question(knowledge, reasoning),
    )


def _plan_comparison_dialogue(
    task,
    knowledge: list[KnowledgeEntry],
    reasoning: TaskReasoningState,
    delta: list[KnowledgeEntry],
) -> TaskDialoguePlan:
    options = _subjects_with_kind(knowledge, "option")
    delta_predicates = {entry.predicate for entry in delta}
    budget = _latest_value(knowledge, "current_task", "budget_max")

    if not delta:
        if options:
            return TaskDialoguePlan(
                acknowledgement=f"Certo. Temos a comparação entre {_join_pt(options)} ativa.",
                next_question=_comparison_next_question(task, knowledge, reasoning),
            )
        return TaskDialoguePlan(next_question="Certo. Que opções queres comparar?")

    if "budget_max" in delta_predicates and "battery_priority" in delta_predicates:
        return TaskDialoguePlan(
            acknowledgement=f"Isso muda a comparação. O limite passa para {budget} € e a bateria passa a pesar menos.",
            implication=_comparison_budget_implication(knowledge, budget),
            recommendation=_comparison_recommendation_sentence(reasoning),
            uncertainty=_comparison_uncertainty_sentence(reasoning),
        )
    if "budget_max" in delta_predicates:
        return TaskDialoguePlan(
            acknowledgement=f"Perfeito. Fica definido um limite de {budget} €.",
            implication=_comparison_budget_implication(knowledge, budget),
            next_question=_comparison_next_question(task, knowledge, reasoning),
        )
    if {"camera_priority", "battery_priority", "support_priority"}.intersection(delta_predicates):
        return TaskDialoguePlan(
            acknowledgement=_comparison_preference_ack(delta),
            implication=_comparison_current_implication(knowledge, reasoning),
            next_question=_comparison_next_question(task, knowledge, reasoning),
        )
    if "price" in delta_predicates:
        return TaskDialoguePlan(
            acknowledgement=_comparison_prices_ack(knowledge),
            implication=_comparison_budget_implication(knowledge, budget),
            next_question=_comparison_next_question(task, knowledge, reasoning),
        )
    if {"camera_quality", "battery", "support_quality"}.intersection(delta_predicates):
        return TaskDialoguePlan(
            acknowledgement=_comparison_tradeoff_ack(delta),
            recommendation=_comparison_recommendation_sentence(reasoning),
            uncertainty=_comparison_uncertainty_sentence(reasoning),
            next_question=_comparison_next_question(task, knowledge, reasoning),
        )
    return TaskDialoguePlan(
        acknowledgement="Certo, isso fica como contexto da comparação.",
        next_question=_comparison_next_question(task, knowledge, reasoning),
    )


def _plan_generic_dialogue(
    task,
    knowledge: list[KnowledgeEntry],
    reasoning: TaskReasoningState,
    delta: list[KnowledgeEntry],
) -> TaskDialoguePlan:
    if delta:
        return TaskDialoguePlan(acknowledgement="Certo, isso já ajuda.", next_question=_generic_next_question(reasoning))
    return TaskDialoguePlan(acknowledgement="Certo.", next_question=_generic_next_question(reasoning))


def _fallback_task_dialogue(task, knowledge: list[KnowledgeEntry], reasoning: TaskReasoningState) -> str:
    if task.task_type == "plan_trip":
        return _travel_next_question(knowledge, reasoning)
    if task.task_type == "generic_comparison":
        return _comparison_next_question(task, knowledge, reasoning) or "Certo. Continuamos a comparação por partes."
    return _generic_next_question(reasoning) or "Certo. Continuamos por partes."


def _travel_next_question(knowledge: list[KnowledgeEntry], reasoning: TaskReasoningState) -> str:
    predicates = {entry.predicate for entry in knowledge}
    stage = infer_task_stage(type("Task", (), {"task_type": "plan_trip"})(), knowledge)
    if stage == "plan":
        return "Por onde queres começar?"
    if "travel_style" not in predicates:
        return "Procuras mais descanso, conhecer sítios novos ou um equilíbrio para descansar e explorar?"
    if "duration" not in predicates and "date_range" not in predicates:
        return "Quantos dias tens para a viagem?"
    if "budget_max" not in predicates:
        return "Tens algum limite de orçamento?"
    if "flight_constraint" not in predicates:
        return "Tens alguma limitação importante em relação à duração dos voos?"
    return ""


def _travel_basis_summary(duration: str, budget: str, travel_style: str, flight_constraint: str) -> str:
    parts = []
    if duration:
        parts.append(duration)
    if budget:
        parts.append(f"até {budget} €")
    if travel_style:
        parts.append(travel_style)
    if flight_constraint:
        parts.append("preferência por voos curtos")
    return "Temos " + _join_pt(parts) if parts else "Temos uma base inicial"


def _comparison_next_question(task, knowledge: list[KnowledgeEntry], reasoning: TaskReasoningState) -> str:
    unknowns = infer_unknowns(task, knowledge)
    if not unknowns:
        return ""
    first = unknowns[0]
    normalized = normalize_for_matching(first)
    if "criterios" in normalized or "critérios" in normalized:
        return "Que fator pesa mais para ti nesta escolha?"
    if normalized.startswith("preco") or normalized.startswith("preço"):
        return f"Sabes o {first}?"
    if normalized.startswith("camara") or normalized.startswith("câmara"):
        return f"Falta perceber a {first}. Tens alguma referência?"
    if normalized.startswith("autonomia"):
        return f"Falta perceber a {first}. Isso é decisivo para ti?"
    if "suporte" in normalized:
        return f"Falta comparar o {first}."
    return f"Falta perceber {first}."


def _generic_next_question(reasoning: TaskReasoningState) -> str:
    if reasoning.uncertainties:
        return "Diz-me o próximo passo concreto quando quiseres."
    return ""


def _comparison_budget_implication(knowledge: list[KnowledgeEntry], budget: str) -> str:
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
            outside.append(f"{option} fica {int(price_num - budget_num)} € acima")
    if inside and outside:
        verb = "ficam" if len(inside) > 1 else "fica"
        return f"{_join_pt(inside)} {verb} dentro do limite; {_join_pt(outside)}."
    if inside:
        verb = "ficam" if len(inside) > 1 else "fica"
        return f"{_join_pt(inside)} {verb} dentro do limite."
    if outside:
        return _join_pt(outside) + "."
    return ""


def _comparison_preference_ack(delta: list[KnowledgeEntry]) -> str:
    predicates = {entry.predicate for entry in delta}
    if "camera_priority" in predicates and "battery_priority" in predicates:
        return "Percebi: a câmara pesa mais, e a bateria fica como fator secundário."
    if "camera_priority" in predicates:
        return "Percebi: a câmara passa a pesar mais na decisão."
    if "battery_priority" in predicates:
        value = _latest_delta_value(delta, "battery_priority")
        if value == "low":
            return "Percebi: a bateria passa a pesar menos na decisão."
        return "Percebi: a bateria também conta na decisão."
    if "support_priority" in predicates:
        return "Percebi: suporte passa a pesar mais do que escolher apenas pelo preço."
    return "Percebi a prioridade."


def _comparison_current_implication(knowledge: list[KnowledgeEntry], reasoning: TaskReasoningState) -> str:
    if reasoning.current_recommendation:
        return f"Com os dados atuais, {reasoning.current_recommendation} começa a ganhar força."
    options = _subjects_with_kind(knowledge, "option")
    if options:
        return "Agora precisamos de comparar essa prioridade nas opções."
    return ""


def _comparison_prices_ack(knowledge: list[KnowledgeEntry]) -> str:
    prices = []
    for option in _subjects_with_kind(knowledge, "option"):
        price = _latest_value(knowledge, option, "price")
        if price:
            prices.append((option, int(price) if price.isdigit() else price))
    if not prices:
        return "Certo, já temos mais dados de preço."
    numeric = [(option, price) for option, price in prices if isinstance(price, int)]
    if len(numeric) >= 2:
        ranked = sorted(numeric, key=lambda item: item[1])
        return "Já temos os preços: " + ", ".join(f"{option} {price} €" for option, price in ranked) + "."
    return "Já temos o preço: " + ", ".join(f"{option} {price} €" for option, price in prices) + "."


def _comparison_tradeoff_ack(delta: list[KnowledgeEntry]) -> str:
    quality = [entry for entry in delta if entry.predicate in {"camera_quality", "battery", "support_quality"}]
    if len(quality) >= 2:
        return "Boa, já temos um tradeoff mais claro: " + _join_pt([_quality_delta_phrase(entry) for entry in quality[:3]]) + "."
    if quality:
        return "Certo. " + _entry_phrase(quality[0]).capitalize() + "."
    return "Certo, isso ajuda a comparar melhor."


def _quality_delta_phrase(entry: KnowledgeEntry) -> str:
    if entry.predicate == "support_quality" and entry.value == "limited":
        return f"{entry.subject}: suporte limitado"
    return _short_detail(entry).replace("+ ", f"{entry.subject}: ")


def _comparison_recommendation_sentence(reasoning: TaskReasoningState) -> str:
    if not reasoning.current_recommendation:
        return ""
    reasons = "; ".join(reasoning.reasons[:2]) if reasoning.reasons else "encaixa melhor nos dados atuais"
    return f"Neste momento, {reasoning.current_recommendation} parece mais forte, porque {reasons}."


def _comparison_uncertainty_sentence(reasoning: TaskReasoningState) -> str:
    if not reasoning.current_recommendation or not reasoning.uncertainties:
        return ""
    return "Ainda é provisório: falta " + _join_pt(reasoning.uncertainties[:2]) + "."


def _latest_delta_value(delta: list[KnowledgeEntry], predicate: str) -> str:
    for entry in reversed(delta):
        if entry.predicate == predicate:
            return entry.value
    return ""


def _short_detail(entry: KnowledgeEntry) -> str:
    if entry.predicate == "battery":
        return "Vantagem de autonomia" if entry.value == "better" else f"Autonomia: {entry.value}"
    if entry.predicate == "camera_quality":
        if entry.value == "better":
            return "Parece ter vantagem na câmara"
        if entry.value == "good":
            return "Boa prestação de câmara"
        if entry.value == "mixed":
            return "Evidência mista sobre a câmara"
        return "Ainda há pouca evidência sobre a câmara"
    if entry.predicate == "support_quality":
        return "Melhor suporte" if entry.value == "better" else "Suporte limitado"
    return _entry_phrase(entry)


def _quality_view_details(knowledge: list[KnowledgeEntry], option: str) -> list[str]:
    details: list[str] = []
    for predicate, label in (
        ("camera_quality", "Câmara"),
        ("battery", "Autonomia"),
        ("support_quality", "Suporte"),
    ):
        entries = [
            entry
            for entry in knowledge
            if normalize_for_matching(entry.subject) == normalize_for_matching(option)
            and entry.predicate == predicate
            and entry.status == "active"
        ]
        summary = _quality_summary_label(entries, label)
        if summary:
            details.append(summary)
    return details


def _quality_summary_label(entries: list[KnowledgeEntry], label: str) -> str:
    if not entries:
        return ""
    values = {normalize_for_matching(entry.value) for entry in entries}
    strengths = {str((entry.metadata or {}).get("source_strength_for_claim") or "") for entry in entries}
    if "weak" in strengths and not {"strong", "useful"}.intersection(strengths):
        return f"{label}: sinais positivos, ainda com evidência limitada."
    if "better" in values:
        return f"{label}: parece haver vantagem."
    if "good" in values and "mixed" in values:
        return f"{label}: resultados positivos, mas com alguma incerteza entre fontes."
    if "good" in values:
        return f"{label}: boa prestação."
    if "mixed" in values:
        return f"{label}: evidência mista."
    return ""


def _quality_score_and_reason(knowledge: list[KnowledgeEntry], option: str, predicate: str, label: str) -> tuple[float, str]:
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
        return 1.0, f"tem vantagem de {label} em fonte técnica forte"
    if "useful" in strengths:
        return 0.7, f"tem sinais úteis de vantagem em {label}"
    if "weak" in strengths or best_confidence < 0.55:
        return 0.25, f"tem sinais fracos de vantagem em {label}"
    return 0.5, f"tem sinais de vantagem em {label}"


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


def infer_unknowns(task, knowledge: list[KnowledgeEntry]) -> list[str]:
    active = _active_entries(knowledge)
    predicates = {entry.predicate for entry in active}
    if task.task_type == "generic_comparison":
        options = _subjects_with_kind(active, "option")
        unknowns: list[str] = []
        if len(options) < 2:
            unknowns.append("opções a comparar")
        has_price = {entry.subject for entry in active if entry.predicate == "price"}
        has_battery = {entry.subject for entry in active if entry.predicate == "battery"}
        has_camera = {entry.subject for entry in active if entry.predicate == "camera_quality"}
        for option in options:
            if option not in has_price:
                unknowns.append(f"preço de {option}")
            if "camera_priority" in predicates and option not in has_camera:
                unknowns.append(f"câmara de {option}")
            if "battery_priority" in predicates and option not in has_battery:
                unknowns.append(f"autonomia de {option}")
        if not any(entry.kind == "criterion" for entry in active) and not any(entry.kind == "preference" for entry in active):
            unknowns.append("critérios de decisão")
        return _dedupe(unknowns)
    if task.task_type == "plan_trip":
        unknowns = []
        stage = infer_task_stage(task, active)
        if stage == "explore" and "destination" not in predicates:
            unknowns.append("tipo de destino")
        if "date_range" not in predicates and "duration" not in predicates:
            unknowns.append("datas ou duração")
        if "budget_max" not in predicates:
            unknowns.append("orçamento")
        if "travel_style" not in predicates:
            unknowns.append("ritmo da viagem")
        if stage == "plan":
            unknowns.extend(["alojamento", "transportes", "itinerário", "custos"])
        return _dedupe(unknowns)
    return ["próximo passo concreto"]


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


def _refresh_reasoning(work_session, task) -> TaskReasoningState:
    knowledge = _active_entries(task_knowledge(work_session, task.id))
    state = task_reasoning(work_session, task.id)
    state.stage = infer_task_stage(task, knowledge)
    state.updated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    state.stale = False
    if task.task_type == "generic_comparison":
        _reason_comparison(state, knowledge)
    else:
        state.current_recommendation = ""
        state.recommendation_confidence = 0.0
        state.reasons = []
        state.uncertainties = infer_unknowns(task, knowledge)
    return state


def _reason_comparison(state: TaskReasoningState, knowledge: list[KnowledgeEntry]) -> None:
    options = _subjects_with_kind(knowledge, "option")
    budget = _latest_value(knowledge, "current_task", "budget_max")
    priorities = {
        "camera_quality": _latest_value(knowledge, "user", "camera_priority"),
        "battery": _latest_value(knowledge, "user", "battery_priority"),
        "support_quality": _latest_value(knowledge, "user", "support_priority"),
        "price": _latest_value(knowledge, "user", "price_priority"),
    }
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
                    reasons[option].append(f"fica dentro do orçamento ({int(price_num)} <= {int(budget_num)})")
                else:
                    scores[option] -= 2.0
                    reasons[option].append(f"passa o orçamento ({int(price_num)} > {int(budget_num)})")
            except ValueError:
                pass
        camera_score, camera_reason = _quality_score_and_reason(knowledge, option, "camera_quality", "câmara")
        if camera_score > 0:
            weight = 2.0 if priorities.get("camera_quality") == "high" else 1.0
            scores[option] += camera_score * weight
            reasons[option].append(camera_reason)
        battery_score, battery_reason = _quality_score_and_reason(knowledge, option, "battery", "bateria")
        if battery_score > 0:
            weight = 2.0 if priorities.get("battery") == "high" else (0.4 if priorities.get("battery") == "low" else 1.0)
            scores[option] += battery_score * weight
            reasons[option].append(battery_reason)
        support_quality = _latest_value(knowledge, option, "support_quality")
        support_score, support_reason = _quality_score_and_reason(knowledge, option, "support_quality", "suporte")
        if support_score > 0:
            weight = 2.0 if priorities.get("support_quality") == "high" else 1.0
            scores[option] += support_score * weight
            reasons[option].append(support_reason)
        elif support_quality == "limited":
            scores[option] -= 0.8
            reasons[option].append("tem suporte mais limitado")
    if not scores:
        state.current_recommendation = ""
        state.reasons = []
        state.uncertainties = ["opções a comparar"]
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
    state.uncertainties = infer_unknowns(type("Task", (), {"task_type": "generic_comparison"})(), knowledge)


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


def _format_known(task, knowledge: list[KnowledgeEntry]) -> str:
    if not knowledge:
        return "Ainda estamos no início. Para já só sei que estamos a trabalhar nesta tarefa."
    stage = infer_task_stage(task, knowledge)
    lines = []
    if task.task_type == "generic_comparison":
        options = _subjects_with_kind(knowledge, "option")
        if options:
            lines.append(f"Estamos a comparar {_join_pt(options)}.")
    elif task.task_type == "plan_trip":
        lines.append("Até agora, a viagem está em modo de " + ("planeamento" if stage == "plan" else "descoberta") + ".")
    else:
        lines.append("Até agora, temos este contexto principal.")
    for entry in [e for e in knowledge if e.kind != "option"][:6]:
        lines.append(_entry_phrase(entry) + ".")
    return " ".join(lines)


def _format_important(task, knowledge: list[KnowledgeEntry], reasoning: TaskReasoningState) -> str:
    if not knowledge:
        return "O essencial é simples: ainda não temos dados suficientes para decidir bem."
    if task.task_type == "generic_comparison":
        parts: list[str] = []
        budget = _latest_value(knowledge, "current_task", "budget_max")
        if budget:
            parts.append(f"o limite é {budget} €")
        if reasoning.current_recommendation:
            parts.append(f"{reasoning.current_recommendation} parece melhor neste momento")
        priorities = [entry for entry in knowledge if entry.kind == "preference"]
        if priorities:
            parts.append(_entry_phrase(priorities[-1]))
        unknowns = infer_unknowns(task, knowledge)
        if unknowns:
            parts.append("a maior lacuna é " + unknowns[0])
        return "O mais importante: " + "; ".join(parts[:4]) + "."
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
                essentials.append(_entry_phrase(KnowledgeEntry("", "", "fact", subject, predicate, value, KnowledgeSource())))
        unknowns = infer_unknowns(task, knowledge)
        if unknowns:
            essentials.append("falta " + unknowns[0])
        return "O essencial: " + "; ".join(essentials[:4]) + "."
    return "O essencial: " + "; ".join(_entry_phrase(entry) for entry in knowledge[:3]) + "."


def _format_unknowns(task, knowledge: list[KnowledgeEntry]) -> str:
    unknowns = infer_unknowns(task, knowledge)
    if not unknowns:
        return "Neste momento não vejo nenhuma lacuna essencial. Podemos começar a decidir ou aprofundar um ponto específico."
    return "Ainda falta perceber sobretudo " + _join_pt(unknowns[:4]) + "."


def _format_decisions(task, knowledge: list[KnowledgeEntry]) -> str:
    decisions = [entry for entry in knowledge if entry.kind == "decision"]
    if decisions:
        return "Já há uma decisão registada: " + _join_pt([_entry_phrase(entry) for entry in decisions[:3]]) + "."
    if task.task_type == "generic_comparison":
        unknowns = infer_unknowns(task, knowledge)
        if unknowns:
            return "Ainda falta decidir a opção final. Antes disso, convém fechar " + _join_pt(unknowns[:2]) + "."
        return "Falta decidir qual opção escolhes."
    if task.task_type == "plan_trip":
        stage = infer_task_stage(task, knowledge)
        if stage == "plan":
            return "Ainda falta decidir alojamento, transportes, itinerário e custos."
        return "Ainda falta decidir o destino ou o tipo de destino."
    return "Ainda falta transformar isto numa decisão concreta."


def _format_recommendation(reasoning: TaskReasoningState) -> str:
    if not reasoning.current_recommendation:
        gaps = _join_pt(reasoning.uncertainties[:3]) if reasoning.uncertainties else "mais alguns dados"
        return f"Ainda não escolheria uma opção com confiança. Falta-nos sobretudo {gaps}."
    reasons = "; ".join(reasoning.reasons) if reasoning.reasons else "encaixa melhor nos dados atuais"
    return f"Neste momento, escolheria {reasoning.current_recommendation}, de forma provisória, porque {reasons}. Isto pode mudar se mudares prioridades ou aparecerem dados novos."


def _format_travel_recommendation(task, knowledge: list[KnowledgeEntry]) -> str:
    unknowns = infer_unknowns(task, knowledge)
    stage = infer_task_stage(task, knowledge)
    if stage == "explore" and unknowns:
        return "Ainda não sugeria destinos com confiança. Falta perceber sobretudo " + _join_pt(unknowns[:2]) + "."
    if unknowns:
        return "Já dá para começar a planear, mas antes fechava " + _join_pt(unknowns[:2]) + "."
    return "Já temos base para começar. Eu compararia alojamento, transportes, itinerário e custos antes de fechar o plano."


def _format_why(reasoning: TaskReasoningState) -> str:
    if not reasoning.current_recommendation:
        return "Porque ainda não há dados suficientes para justificar uma escolha com segurança."
    reasons = "; ".join(reasoning.reasons) if reasoning.reasons else "é a opção com melhor equilíbrio nos dados atuais"
    return f"Porque {reasons}. A conclusão continua provisória."


def _format_sources(knowledge: list[KnowledgeEntry], message: str) -> str:
    normalized = normalize_for_matching(message)
    matching = [entry for entry in knowledge if normalize_for_matching(entry.subject) in normalized or normalize_for_matching(entry.predicate) in normalized]
    if not matching:
        matching = knowledge[-3:]
    if not matching:
        return "Ainda não tenho uma fonte concreta para isso."
    entry = matching[-1]
    if entry.source.source_type == "web":
        title = str(entry.metadata.get("source_title") or entry.source.reference or "fonte web")
        evidence = entry.source.original_text or entry.value
        return f"Veio de {title}: “{evidence}”" + (f" ({entry.source.reference})." if entry.source.reference else ".")
    return f"Veio do que me disseste: “{entry.source.original_text}”."


def _format_evidence_answer(task, knowledge: list[KnowledgeEntry], message: str) -> str:
    web_entries = [
        entry for entry in knowledge
        if entry.source.source_type == "web" and entry.status == "active"
    ]
    if not web_entries:
        return "Ainda não tenho fontes suficientes para responder a isso com segurança."

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
            known = _join_pt(sorted(subjects_with_evidence)) if subjects_with_evidence else "uma das opções"
            return (
                f"Não. Estas fontes ajudam a avaliar {known}, mas ainda não provam que seja melhor "
                f"do que {_join_pt(missing[:2])}."
            )
    if not scoped_entries:
        target = _join_pt(subjects) if subjects else "esse ponto"
        return f"Ainda não tenho evidência direta suficiente sobre {target}."

    strong = [entry for entry in scoped_entries if _entry_evidence_strength(entry) == "strong"]
    useful = [entry for entry in scoped_entries if _entry_evidence_strength(entry) == "useful"]
    weak = [entry for entry in scoped_entries if _entry_evidence_strength(entry) == "weak"]
    positive = [entry for entry in scoped_entries if normalize_for_matching(entry.value) in {"good", "better", "boa", "bom", "superior", "melhor"}]
    mixed = [entry for entry in scoped_entries if normalize_for_matching(entry.value) == "mixed"]
    target = _join_pt(subjects) if subjects else _join_pt(_dedupe([entry.subject for entry in scoped_entries])[:2])
    criterion = _predicate_natural_label(predicate or scoped_entries[-1].predicate)

    if _asks_sufficiency(message):
        if len(options) >= 2:
            subjects_with_evidence = {
                option for option in options
                if any(_entry_subject_matches_any(entry, [option]) and (not predicate or entry.predicate == predicate) for entry in web_entries)
            }
            missing = [option for option in options if option not in subjects_with_evidence]
            if missing:
                return (
                    f"Chega para ter uma primeira leitura sobre {target}, mas ainda não para fechar a comparação. "
                    f"Falta evidência equivalente sobre {_join_pt(missing[:2])}."
                )
        if strong or useful:
            return "Sim, já chega para uma conclusão provisória. Eu ainda a trataria como provisória, porque podemos reforçar com mais uma fonte independente."
        return "Ainda não. As fontes que temos são demasiado fracas para decidir com segurança."

    if strong or useful:
        if positive and not mixed:
            return f"Sim. Com estas fontes, já há boa evidência de que {target} tem {criterion}."
        if mixed:
            return f"As fontes sugerem que {target} tem pontos fortes em {criterion}, mas a conclusão ainda é mista."
        return f"Estas fontes já dão uma base útil sobre {target}, mas eu manteria a conclusão como provisória."
    if weak:
        return f"Há sinais sobre {target}, mas a evidência ainda é fraca. Eu não fecharia uma decisão só com isto."
    return "Ainda não tenho evidência suficiente para concluir isso com segurança."


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


def _predicate_natural_label(predicate: str) -> str:
    labels = {
        "camera_quality": "boa câmara",
        "battery": "boa autonomia",
        "price": "um preço definido",
        "support_quality": "bom suporte",
    }
    return labels.get(predicate, predicate.replace("_", " "))


def _format_conflicts(knowledge: list[KnowledgeEntry]) -> str:
    conflicts = []
    grouped: dict[tuple[str, str], set[str]] = {}
    for entry in knowledge:
        if not _entry_can_conflict(entry):
            continue
        grouped.setdefault((entry.subject, entry.predicate), set()).add(entry.value)
    for (subject, predicate), values in grouped.items():
        if _values_are_semantically_conflicting(values, predicate):
            conflicts.append(f"{subject} / {predicate}: {_join_pt(sorted(values))}")
    if not conflicts:
        return "Não vejo informação contraditória ativa neste momento."
    return "Há valores diferentes nestes pontos: " + "; ".join(conflicts[:3]) + "."


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


def _format_next_step(task, knowledge: list[KnowledgeEntry]) -> str:
    stage = infer_task_stage(task, knowledge)
    if task.task_type == "plan_trip" and stage == "plan":
        return "Agora temos de organizar alojamento, transportes, itinerário e custos."
    unknowns = infer_unknowns(task, knowledge)
    if unknowns:
        return "O próximo passo é fechar " + _join_pt(unknowns[:2]) + "."
    return "O próximo passo é escolher uma direção e transformar isto em plano."


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


def _entry_phrase(entry: KnowledgeEntry) -> str:
    pred = entry.predicate
    if pred == "budget_max":
        return f"limite de orçamento: {entry.value} euros"
    if pred == "price":
        return f"{entry.subject} custa {entry.value} euros"
    if pred == "camera_priority":
        return "a câmara pesa mais na decisão"
    if pred == "battery_priority":
        if entry.value == "low":
            return "a bateria pesa menos na decisão"
        if entry.value == "medium":
            return "a bateria fica como fator secundário"
        return "a bateria pesa bastante na decisão"
    if pred == "camera_quality":
        return f"{entry.subject} parece melhor em câmara"
    if pred == "battery":
        return f"{entry.subject} parece melhor em bateria"
    if pred == "support_priority":
        return "o suporte pesa bastante na decisão"
    if pred == "price_priority":
        if entry.value == "low":
            return "o preço pesa menos na decisão"
        return "o preço pesa bastante na decisão"
    if pred == "location_priority":
        return "a localização pesa bastante na decisão"
    if pred == "support_quality":
        if entry.value == "better":
            return f"{entry.subject} tem melhor suporte"
        if entry.value == "limited":
            return f"{entry.subject} tem suporte limitado"
        return f"{entry.subject}: suporte {entry.value}"
    if pred == "destination":
        return f"destino: {entry.value}"
    if pred == "origin":
        return f"partida: {entry.value}"
    if pred == "date_range":
        return f"datas: {entry.value}"
    if pred == "duration":
        return f"duração: {entry.value}"
    if pred == "travel_style":
        return f"preferência: {entry.value}"
    if pred == "flight_constraint":
        return entry.value
    if pred == "destination_candidate":
        return f"{entry.subject}: {entry.value}"
    if pred == "activities":
        return f"{entry.value} em {entry.subject}"
    return f"{entry.subject}: {entry.value}"


def _join_pt(items: list[str]) -> str:
    clean = [item for item in items if item]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    return ", ".join(clean[:-1]) + " e " + clean[-1]


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

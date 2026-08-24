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
            "o que sabemos",
            "que sabemos",
            "o que ja sabes",
            "o que já sabes",
            "o que sabes sobre",
            "resume me so o importante",
            "resume-me só o importante",
        ),
    ):
        return "known"
    if contains_any_phrase(normalized, ("o que ainda falta", "o que falta saber", "o que falta decidir", "ainda nao sabemos", "ainda não sabemos", "precisas de saber")):
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
    if normalized in {"porque", "porquê", "por que"} or contains_any_phrase(normalized, ("porque?", "porquê?", "porque e que", "porquê é que")):
        return "why"
    if contains_any_phrase(normalized, ("de onde veio", "qual e a fonte", "qual é a fonte", "fonte desse", "origem desse")):
        return "source"
    if contains_any_phrase(normalized, ("informacao contraditoria", "informação contraditória", "contradicoes", "contradições", "fontes discordam")):
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
    if query == "unknowns":
        return _format_unknowns(task, knowledge)
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
    if query == "conflicts":
        return _format_conflicts(knowledge)
    if query == "next_step":
        return _format_next_step(task, knowledge)
    return ""


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

    if contains_any_phrase(normalized, ("camara e mais importante", "câmara é mais importante", "dou mais importancia a camara", "dou mais importância à câmara")):
        entries.append(_upsert_entry(work_session, task.id, "preference", "user", "camera_priority", "high", message, turn_id, replace_active=True))
        entries.append(_upsert_entry(work_session, task.id, "preference", "user", "battery_priority", "medium", message, turn_id, replace_active=True))
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
        if contains_any_phrase(subject_norm, ("melhor bateria", "melhor na bateria", "dura mais", "melhor autonomia", "aguenta mais tempo")):
            entries.append(_upsert_entry(work_session, task.id, "claim", option, "battery", "better", message, turn_id, replace_active=True, confidence=0.78))
        if contains_any_phrase(subject_norm, ("melhor camara", "melhor câmara")):
            kind = "opinion" if contains_any_phrase(normalized, ("acho", "parece", "parece me", "parece-me")) else "claim"
            entries.append(_upsert_entry(work_session, task.id, kind, option, "camera_quality", "better", message, turn_id, replace_active=True, confidence=0.7 if kind == "opinion" else 0.78))
        if contains_any_phrase(subject_norm, ("suporte 24", "suporte vinte e quatro", "melhor suporte")):
            entries.append(_upsert_entry(work_session, task.id, "claim", option, "support_quality", "better", message, turn_id, replace_active=True, confidence=0.78))
        if contains_any_phrase(subject_norm, ("horario laboral", "horário laboral", "so em horario", "só em horário")):
            entries.append(_upsert_entry(work_session, task.id, "claim", option, "support_quality", "limited", message, turn_id, replace_active=True, confidence=0.78))
    return [entry for entry in entries if entry is not None]


def _extract_travel_entries(work_session, task, message: str, turn_id: str) -> list[KnowledgeEntry]:
    entries: list[KnowledgeEntry] = []
    normalized = normalize_for_matching(message)
    destination = _extract_destination(message)
    if destination:
        entries.append(_upsert_entry(work_session, task.id, "fact", "trip", "destination", destination, message, turn_id, replace_active=True))
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
    if contains_any_phrase(normalized, ("evitar voos muito longos", "voos muito longos")):
        entries.append(_upsert_entry(work_session, task.id, "constraint", "trip", "flight_constraint", "evitar voos muito longos", message, turn_id, replace_active=True))
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
        source=KnowledgeSource(
            source_type="user",
            reference=turn_id or f"turn-{len(entries) + 1}",
            original_text=message,
            created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        ),
        confidence=confidence,
        replaces=replaced,
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
        if _latest_value(knowledge, option, "camera_quality") == "better":
            weight = 2.0 if priorities.get("camera_quality") == "high" else 1.0
            scores[option] += weight
            reasons[option].append("tem vantagem de câmara nos dados atuais")
        if _latest_value(knowledge, option, "battery") == "better":
            weight = 2.0 if priorities.get("battery") == "high" else (0.4 if priorities.get("battery") == "low" else 1.0)
            scores[option] += weight
            reasons[option].append("tem vantagem de bateria nos dados atuais")
        support_quality = _latest_value(knowledge, option, "support_quality")
        if support_quality == "better":
            weight = 2.0 if priorities.get("support_quality") == "high" else 1.0
            scores[option] += weight
            reasons[option].append("tem melhor suporte nos dados atuais")
        elif support_quality == "limited":
            scores[option] -= 0.8
            reasons[option].append("tem suporte mais limitado")
    if not scores:
        state.current_recommendation = ""
        state.reasons = []
        state.uncertainties = ["opções a comparar"]
        return
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) < 2 or ranked[0][1] <= 0 or abs(ranked[0][1] - ranked[1][1]) < 0.5:
        state.current_recommendation = ""
        state.recommendation_confidence = 0.35
        state.reasons = []
    else:
        state.current_recommendation = ranked[0][0]
        state.recommendation_confidence = min(0.85, 0.45 + (ranked[0][1] - ranked[1][1]) / 6)
        state.reasons = reasons[ranked[0][0]][:3]
    state.uncertainties = infer_unknowns(type("Task", (), {"task_type": "generic_comparison"})(), knowledge)


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


def _format_unknowns(task, knowledge: list[KnowledgeEntry]) -> str:
    unknowns = infer_unknowns(task, knowledge)
    if not unknowns:
        return "Neste momento não vejo nenhuma lacuna essencial. Podemos começar a decidir ou aprofundar um ponto específico."
    return "Ainda falta perceber sobretudo " + _join_pt(unknowns[:4]) + "."


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
    return f"Veio do que me disseste: “{entry.source.original_text}”."


def _format_conflicts(knowledge: list[KnowledgeEntry]) -> str:
    conflicts = []
    grouped: dict[tuple[str, str], set[str]] = {}
    for entry in knowledge:
        grouped.setdefault((entry.subject, entry.predicate), set()).add(entry.value)
    for (subject, predicate), values in grouped.items():
        if len(values) > 1:
            conflicts.append(f"{subject} / {predicate}: {_join_pt(sorted(values))}")
    if not conflicts:
        return "Não vejo informação contraditória ativa neste momento."
    return "Há valores diferentes nestes pontos: " + "; ".join(conflicts[:3]) + "."


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
    return match.group(1) if match else ""


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
    match = re.search(r"\b(?:vou a|vou para|destino e|destino é)\s+([A-ZÀ-Ý][\wÀ-ÿ]+(?:\s+[A-ZÀ-Ý][\wÀ-ÿ]+){0,3})", message)
    if match:
        return match.group(1).strip(" .,!?:;")
    normalized = normalize_for_matching(message)
    if "estocolmo" in normalized:
        return "Estocolmo"
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
        return f"orçamento máximo de {entry.value} euros"
    if pred == "price":
        return f"{entry.subject} custa {entry.value} euros"
    if pred == "camera_priority":
        return "a câmara pesa mais na decisão"
    if pred == "battery_priority":
        return f"a bateria tem prioridade {entry.value}"
    if pred == "camera_quality":
        return f"{entry.subject} parece melhor em câmara"
    if pred == "battery":
        return f"{entry.subject} parece melhor em bateria"
    if pred == "support_priority":
        return "o suporte pesa bastante na decisão"
    if pred == "support_quality":
        return f"{entry.subject} tem suporte {entry.value}"
    if pred == "destination":
        return f"destino: {entry.value}"
    if pred == "date_range":
        return f"datas: {entry.value}"
    if pred == "duration":
        return f"duração: {entry.value}"
    if pred == "travel_style":
        return f"preferência: {entry.value}"
    if pred == "flight_constraint":
        return entry.value
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

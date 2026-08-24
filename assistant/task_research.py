from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from assistant.task_intelligence import (
    KnowledgeEntry,
    KnowledgeSource,
    _refresh_reasoning,
    _upsert_entry,
    task_knowledge,
    task_option_titles,
    task_reasoning,
)
from assistant.text_matching import contains_any_phrase, normalize_for_matching


@dataclass(frozen=True)
class ResearchQuestion:
    subject: str
    predicate: str
    question: str


class ResearchProvider(Protocol):
    def run(self, arguments: dict[str, Any]) -> str:
        ...


@dataclass
class ResearchRequest:
    id: str
    task_id: str
    target_object_ids: list[str] = field(default_factory=list)
    questions: list[ResearchQuestion] = field(default_factory=list)
    criteria: list[str] = field(default_factory=list)
    query_plan: list[str] = field(default_factory=list)
    status: str = "planned"
    created_at: str = ""
    source: str = "user"
    result_ids: list[str] = field(default_factory=list)
    clarification: str = ""
    plan_text: str = ""


@dataclass(frozen=True)
class ResearchSourceRecord:
    title: str
    url: str
    domain: str = ""
    source_quality: str = "unknown"
    retrieved_at: str = ""


@dataclass(frozen=True)
class ResearchClaim:
    subject: str
    predicate: str
    value: str
    kind: str = "claim"
    confidence: float = 0.65
    epistemic_type: str = "claim"
    evidence: str = ""
    source: ResearchSourceRecord = field(default_factory=lambda: ResearchSourceRecord("", ""))


@dataclass(frozen=True)
class ResearchRunResult:
    request: ResearchRequest
    claims: list[ResearchClaim] = field(default_factory=list)
    raw_results: list[dict[str, Any]] = field(default_factory=list)
    ingested_entries: list[KnowledgeEntry] = field(default_factory=list)
    summary: str = ""
    failure_reason: str = ""


def is_task_research_intent(message: str) -> bool:
    text = normalize_for_matching(message)
    return contains_any_phrase(
        text,
        (
            "pesquisa",
            "procura",
            "procurar",
            "ve na internet",
            "vê na internet",
            "verifica",
            "confirma",
            "ve se",
            "vê se",
            "usando informacao atual",
            "usando informação atual",
            "mais fontes",
            "opinioes de utilizadores",
            "opiniões de utilizadores",
        ),
    )


def build_research_request(work_session, task, message: str, *, request_id: str) -> ResearchRequest:
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    request = ResearchRequest(id=request_id, task_id=getattr(task, "id", ""), created_at=now)
    if task is None:
        request.clarification = "Preciso de uma tarefa ativa para ligar a pesquisa ao contexto certo."
        return request

    knowledge = task_knowledge(work_session, task.id)
    if task.task_type == "generic_comparison":
        _build_comparison_request(work_session, task, message, request, knowledge)
    elif task.task_type == "plan_trip":
        _build_travel_request(task, message, request, knowledge)
    else:
        _build_generic_request(task, message, request)
    request.plan_text = _render_plan_text(task, request)
    return request


def run_task_research(work_session, task, request: ResearchRequest, provider: ResearchProvider | None) -> ResearchRunResult:
    if request.clarification:
        request.status = "failed"
        return ResearchRunResult(request=request, failure_reason="needs_clarification")
    if provider is None:
        request.status = "failed"
        return ResearchRunResult(request=request, failure_reason="tool_unavailable")
    if not request.query_plan:
        request.status = "failed"
        return ResearchRunResult(request=request, failure_reason="empty_query_plan")

    raw_results: list[dict[str, Any]] = []
    claims: list[ResearchClaim] = []
    for query in request.query_plan:
        raw = provider.run({"query": query})
        parsed_results, parsed_claims = _parse_provider_payload(raw, query)
        raw_results.extend(parsed_results)
        claims.extend(parsed_claims)

    entries: list[KnowledgeEntry] = []
    for claim in claims:
        entry = _ingest_claim(work_session, request, claim)
        if entry is not None:
            entries.append(entry)

    if entries:
        reasoning = task_reasoning(work_session, task.id)
        reasoning.stale = True
        _refresh_reasoning(work_session, task)
        request.result_ids = [entry.id for entry in entries]
        request.status = "completed"
    else:
        request.status = "partial" if raw_results else "failed"

    summary = synthesize_research_update(work_session, task, request, entries, raw_results)
    return ResearchRunResult(request=request, claims=claims, raw_results=raw_results, ingested_entries=entries, summary=summary)


def synthesize_research_update(work_session, task, request: ResearchRequest, entries: list[KnowledgeEntry], raw_results: list[dict[str, Any]]) -> str:
    if not entries and raw_results:
        return "Encontrei resultados, mas ainda não consegui transformá-los em conhecimento seguro para esta tarefa."
    if not entries:
        return "Não encontrei informação suficientemente estruturada para atualizar esta tarefa."

    subjects = _dedupe([entry.subject for entry in entries])
    source_count = len(_dedupe([entry.source.reference for entry in entries if entry.source.reference]))
    conflict_text = _research_conflict_sentence(entries)
    reasoning = task_reasoning(work_session, task.id)
    update = f"Depois da pesquisa, atualizei esta tarefa com informação sobre {_join_pt(subjects)}."
    if source_count:
        update += f" Guardei {source_count} fonte" + ("s" if source_count != 1 else "") + "."
    if conflict_text:
        update += " " + conflict_text
    if reasoning.current_recommendation:
        update += f" Neste momento, {reasoning.current_recommendation} parece ganhar força, mas continua uma conclusão provisória."
    return update


def research_sources_response(work_session, task, message: str) -> str:
    if task is None:
        return ""
    text = normalize_for_matching(message)
    if contains_any_phrase(
        text,
        (
            "fontes que discordem",
            "fontes discordam",
            "informacao contraditoria",
            "informação contraditória",
            "contradicoes",
            "contradições",
        ),
    ):
        return ""
    if not contains_any_phrase(
        text,
        (
            "de onde tiraste",
            "de onde veio",
            "qual e a fonte",
            "qual é a fonte",
            "fontes usaste",
            "que fontes",
            "mostra as fontes",
            "origem disso",
        ),
    ):
        return ""
    entries = [
        entry
        for entry in task_knowledge(work_session, task.id)
        if entry.status == "active" and entry.source.source_type == "web"
    ]
    if not entries:
        return ""
    matching = [
        entry
        for entry in entries
        if normalize_for_matching(entry.subject) in text or normalize_for_matching(entry.predicate) in text
    ] or entries[-5:]
    lines = []
    for entry in matching[-5:]:
        title = str(entry.metadata.get("source_title") or entry.source.reference or "fonte web")
        url = entry.source.reference
        evidence = entry.source.original_text or entry.value
        lines.append(f"{entry.subject}: {evidence} — {title}" + (f" ({url})" if url else ""))
    return "Estas são as fontes que tenho ligadas a esta tarefa: " + "; ".join(lines) + "."


def exclude_research_source(work_session, task, message: str) -> list[KnowledgeEntry]:
    if task is None:
        return []
    text = normalize_for_matching(message)
    if not contains_any_phrase(text, ("ignora", "exclui", "desatualizada", "desactualizada")):
        return []
    entries = task_knowledge(work_session, task.id)
    candidates = [entry for entry in entries if entry.source.source_type == "web" and entry.status == "active"]
    if not candidates:
        return []
    target = candidates[-1]
    for entry in reversed(candidates):
        haystack = normalize_for_matching(" ".join([entry.source.reference, str(entry.metadata.get("source_title", "")), entry.source.original_text]))
        if any(token in haystack for token in _meaningful_tokens(text)):
            target = entry
            break
    target.status = "excluded"
    _refresh_reasoning(work_session, task)
    return [target]


def _build_comparison_request(work_session, task, message: str, request: ResearchRequest, knowledge: list[KnowledgeEntry]) -> None:
    options = task_option_titles(work_session, task)
    criterion = _criterion_from_message(message)
    if not criterion:
        criterion = _first_missing_comparison_criterion(knowledge)
    if not criterion:
        criterion = "general_quality"
    request.criteria = [criterion]
    request.target_object_ids = [
        obj.id for obj in work_session.objects.values() if obj.task_id == task.id and obj.kind == "option"
    ]
    if criterion in {"camera_quality", "battery"} and _needs_exact_models(task, options):
        label = "câmara" if criterion == "camera_quality" else "bateria"
        request.clarification = f"Antes de pesquisar {label}, preciso de saber os modelos exatos. Samsung e Apple podem significar muitos telemóveis diferentes."
        return
    for option in options:
        request.questions.append(ResearchQuestion(option, criterion, _question_for_criterion(option, criterion)))
        request.query_plan.append(_query_for_criterion(option, criterion))


def _build_travel_request(task, message: str, request: ResearchRequest, knowledge: list[KnowledgeEntry]) -> None:
    destination = _latest_value(knowledge, "trip", "destination") or task.parameters.get("destination", "")
    date_range = _latest_value(knowledge, "trip", "date_range")
    duration = _latest_value(knowledge, "trip", "duration")
    budget = _latest_value(knowledge, "trip", "budget_max")
    origin = _latest_value(knowledge, "trip", "origin") or task.parameters.get("origin", "")
    text = normalize_for_matching(message)
    if contains_any_phrase(text, ("destinos", "destino")) and not origin:
        request.clarification = "Antes de pesquisar destinos, preciso de saber de onde vais partir. A duração e o custo mudam bastante com a cidade de partida."
        return
    if destination:
        criteria = ["activities", "opening_hours", "transport", "cost"]
        request.criteria = criteria
        for criterion in criteria:
            request.questions.append(ResearchQuestion(destination, criterion, f"{criterion} em {destination}"))
            request.query_plan.append(f"{destination} {criterion} {date_range}".strip())
        return
    style = _latest_value(knowledge, "user", "travel_style") or task.parameters.get("travel_style", "")
    flight_constraint = _latest_value(knowledge, "trip", "flight_constraint") or task.parameters.get("flight_constraint", "")
    basis = " ".join(part for part in (origin, duration, budget, style, flight_constraint) if part)
    request.criteria = ["flight_duration", "lodging_cost", "daily_cost", "things_to_visit", "fit_for_trip"]
    request.query_plan = [
        f"short flight holiday destinations from {origin or 'Porto'} {duration} {budget} relaxing sightseeing".strip(),
        f"European city breaks from {origin or 'Porto'} {duration} {budget} short flights things to visit".strip(),
    ]
    request.questions = [ResearchQuestion("trip", criterion, criterion) for criterion in request.criteria]


def _build_generic_request(task, message: str, request: ResearchRequest) -> None:
    topic = task.parameters.get("topic") or task.goal or message
    criterion = _criterion_from_message(message) or "reputation"
    request.criteria = [criterion]
    request.questions = [ResearchQuestion(topic, criterion, message.strip(" .!?"))]
    request.query_plan = [f"{topic} {criterion}".strip()]


def _render_plan_text(task, request: ResearchRequest) -> str:
    if request.clarification:
        return request.clarification
    if task.task_type == "generic_comparison":
        labels = {
            "camera_quality": "qualidade da câmara",
            "battery": "autonomia, carregamento e testes independentes",
            "support_quality": "suporte oficial e feedback de utilizadores",
        }
        target = labels.get(request.criteria[0], request.criteria[0] if request.criteria else "os pontos em falta")
        subjects = _join_pt([q.subject for q in request.questions]) or "as opções"
        return f"Vou verificar {target} para {subjects} e ligar as fontes a esta comparação."
    if task.task_type == "plan_trip":
        if any("destinations" in query or "destinos" in query for query in request.query_plan):
            return "Vou verificar destinos que encaixem nos teus critérios e comparar duração, custos, coisas para visitar e adequação à viagem."
        subject = request.questions[0].subject if request.questions else "a viagem"
        return f"Vou verificar atividades, custos e transportes para {subject}, mantendo tudo nesta tarefa."
    return "Vou procurar informação relevante e anexar só o que conseguir validar a esta tarefa."


def _parse_provider_payload(raw: str, query: str) -> tuple[list[dict[str, Any]], list[ResearchClaim]]:
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        payload = json.loads(str(raw))
    except Exception:
        result = {"title": "Resultado da pesquisa", "url": "", "snippet": str(raw), "query": query}
        return [result], []
    result_items = payload.get("results", []) if isinstance(payload, dict) else []
    parsed_results: list[dict[str, Any]] = []
    claims: list[ResearchClaim] = []
    for item in result_items:
        if not isinstance(item, dict):
            continue
        source = ResearchSourceRecord(
            title=str(item.get("title", "")),
            url=str(item.get("url", "")),
            domain=str(item.get("domain", "")),
            source_quality=str(item.get("source_quality", "unknown")),
            retrieved_at=str(item.get("retrieved_at") or now),
        )
        parsed_results.append({
            "title": source.title,
            "url": source.url,
            "snippet": str(item.get("snippet", "")),
            "source_quality": source.source_quality,
            "query": query,
        })
        for claim in item.get("claims", []):
            if not isinstance(claim, dict):
                continue
            subject = str(claim.get("subject", "")).strip()
            predicate = str(claim.get("predicate", "")).strip()
            value = str(claim.get("value", "")).strip()
            if not subject or not predicate or not value:
                continue
            claims.append(
                ResearchClaim(
                    subject=subject,
                    predicate=predicate,
                    value=value,
                    kind=str(claim.get("kind", "claim")),
                    confidence=float(claim.get("confidence", 0.65)),
                    epistemic_type=str(claim.get("epistemic_type", "claim")),
                    evidence=str(claim.get("evidence") or item.get("snippet") or value),
                    source=source,
                )
            )
    return parsed_results, claims


def _ingest_claim(work_session, request: ResearchRequest, claim: ResearchClaim) -> KnowledgeEntry | None:
    entry = _upsert_entry(
        work_session,
        request.task_id,
        claim.kind,
        claim.subject,
        claim.predicate,
        claim.value,
        claim.evidence,
        claim.source.url or claim.source.title or request.id,
        replace_active=False,
        confidence=claim.confidence,
        source=KnowledgeSource(
            source_type="web",
            reference=claim.source.url,
            original_text=claim.evidence,
            created_at=claim.source.retrieved_at,
        ),
        metadata={
            "research_request_id": request.id,
            "source_title": claim.source.title,
            "source_domain": claim.source.domain,
            "source_quality": claim.source.source_quality,
            "epistemic_type": claim.epistemic_type,
        },
    )
    return entry


def _criterion_from_message(message: str) -> str:
    text = normalize_for_matching(message)
    if contains_any_phrase(text, ("camara", "câmara", "fotografia", "video", "zoom")):
        return "camera_quality"
    if contains_any_phrase(text, ("bateria", "autonomia", "carregamento")):
        return "battery"
    if contains_any_phrase(text, ("suporte", "reputacao", "reputação", "opinioes", "opiniões")):
        return "support_quality"
    if contains_any_phrase(text, ("preco", "preço", "custo")):
        return "price"
    return ""


def _first_missing_comparison_criterion(knowledge: list[KnowledgeEntry]) -> str:
    options = _dedupe([entry.subject for entry in knowledge if entry.kind == "option"])
    for predicate in ("camera_quality", "battery", "support_quality", "price"):
        if options and any(not _latest_value(knowledge, option, predicate) for option in options):
            return predicate
    return ""


def _needs_exact_models(task, options: list[str]) -> bool:
    category = normalize_for_matching(getattr(task, "parameters", {}).get("category", ""))
    if not contains_any_phrase(category, ("telemovel", "telemoveis", "telefone", "telefones", "smartphone", "smartphones")):
        return False
    if len(options) < 2:
        return False
    return any(not re.search(r"\d|pro|max|ultra|plus|mini|se\b", normalize_for_matching(option)) for option in options)


def _query_for_criterion(option: str, criterion: str) -> str:
    labels = {
        "camera_quality": "camera review independent test low light video",
        "battery": "battery life real world test charging review",
        "support_quality": "customer support reputation user reviews",
        "price": "current price",
    }
    return f"{option} {labels.get(criterion, criterion)}".strip()


def _question_for_criterion(option: str, criterion: str) -> str:
    labels = {
        "camera_quality": "qualidade real da câmara",
        "battery": "autonomia real e carregamento",
        "support_quality": "qualidade do suporte",
        "price": "preço atual",
    }
    return f"{option}: {labels.get(criterion, criterion)}"


def _research_conflict_sentence(entries: list[KnowledgeEntry]) -> str:
    grouped: dict[tuple[str, str], set[str]] = {}
    for entry in entries:
        grouped.setdefault((entry.subject, entry.predicate), set()).add(entry.value)
    conflicts = [
        f"{subject} tem valores diferentes para {predicate}: {_join_pt(sorted(values))}"
        for (subject, predicate), values in grouped.items()
        if len(values) > 1
    ]
    if not conflicts:
        return ""
    return "Há conflito nas fontes: " + "; ".join(conflicts[:2]) + "."


def _latest_value(entries: list[KnowledgeEntry], subject: str, predicate: str) -> str:
    subject_norm = normalize_for_matching(subject)
    predicate_norm = normalize_for_matching(predicate)
    for entry in reversed(entries):
        if entry.status == "active" and normalize_for_matching(entry.subject) == subject_norm and normalize_for_matching(entry.predicate) == predicate_norm:
            return entry.value
    return ""


def _meaningful_tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9à-ÿ]{4,}", text) if token not in {"essa", "fonte", "esta", "isso"}]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = normalize_for_matching(value)
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _join_pt(values: list[str]) -> str:
    values = [value for value in values if value]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + " e " + values[-1]

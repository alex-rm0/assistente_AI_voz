from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol
from urllib.parse import urlparse

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


def _semex_trace(event: str, **metadata: Any) -> None:
    parts = [f"{key}={value}" for key, value in metadata.items() if value is not None and value != ""]
    print("[SEMEX]", event, " ".join(parts))


@dataclass(frozen=True)
class ResearchQuestion:
    subject: str
    predicate: str
    question: str
    semantic_role: str = "target"
    required_entity_precision: str = "unknown"
    freshness: str = "current"
    desired_source_type: str = ""


class ResearchProvider(Protocol):
    def run(self, arguments: dict[str, Any]) -> str:
        ...


@dataclass
class ResearchRequest:
    id: str
    task_id: str
    task_type: str = ""
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
    id: str = ""
    domain: str = ""
    source_quality: str = "unknown"
    source_kind: str = "unknown"
    source_strength_for_claim: str = "unknown"
    retrieved_at: str = ""
    relevance: str = "unknown"
    relevance_reason: str = ""


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
    metadata: dict[str, Any] = field(default_factory=dict)
    applicability: str = "unknown"
    relevance: str = "unknown"
    rejection_reason: str = ""


@dataclass(frozen=True)
class SemanticExtractionRequest:
    task_type: str
    extraction_goal: str
    known_context: dict[str, Any] = field(default_factory=dict)
    excluded_entities: list[str] = field(default_factory=list)
    evidence: str = ""
    source_title: str = ""


@dataclass(frozen=True)
class SemanticCandidate:
    canonical_name: str
    entity_type: str
    evidence_quote_or_fragment: str
    confidence: float = 0.0
    aliases: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticExtractionResult:
    candidates: list[SemanticCandidate] = field(default_factory=list)
    status: str = "not_requested"
    failure_reason: str = ""


class SemanticExtractor(Protocol):
    def extract(self, request: SemanticExtractionRequest) -> list[SemanticCandidate] | SemanticExtractionResult:
        ...


class OllamaSemanticExtractor:
    """Narrow local-only extractor: it suggests candidates, never writes knowledge."""

    def __init__(self, llm, timeout_seconds: float = 45) -> None:
        self.llm = llm
        self.timeout_seconds = timeout_seconds

    def extract(self, request: SemanticExtractionRequest) -> SemanticExtractionResult:
        prompt = _semantic_extraction_prompt(request)
        try:
            raw = self.llm.chat(
                prompt,
                history=[],
                system_prompt=(
                    "És um extractor local e restrito. Devolve apenas JSON válido. "
                    "Não escolhas recomendações, não expliques, não inventes."
                ),
                response_format="json",
                source="SEMANTIC_EXTRACTOR_LOCAL",
                timeout_seconds=self.timeout_seconds,
            )
        except Exception as exc:
            return SemanticExtractionResult(status="call_failed", failure_reason=type(exc).__name__)
        return _parse_semantic_candidates_with_status(raw)


@dataclass(frozen=True)
class ResearchRunResult:
    request: ResearchRequest
    claims: list[ResearchClaim] = field(default_factory=list)
    raw_results: list[dict[str, Any]] = field(default_factory=list)
    rejected_results: list[dict[str, Any]] = field(default_factory=list)
    ingested_entries: list[KnowledgeEntry] = field(default_factory=list)
    summary: str = ""
    failure_reason: str = ""
    semantic_extraction_calls: int = 0
    semantic_extraction_status: str = "not_needed"
    semantic_candidates_before_validation: int = 0
    semantic_candidates_after_validation: int = 0
    semantic_extraction_failure_reason: str = ""
    semantic_extraction_batches: int = 0
    semantic_extraction_latency_ms: int = 0
    semantic_extraction_input_chars: int = 0
    semantic_extraction_sources_used: int = 0


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
    request = ResearchRequest(
        id=request_id,
        task_id=getattr(task, "id", ""),
        task_type=getattr(task, "task_type", ""),
        created_at=now,
    )
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


def run_task_research(
    work_session,
    task,
    request: ResearchRequest,
    provider: ResearchProvider | None,
    semantic_extractor: SemanticExtractor | None = None,
) -> ResearchRunResult:
    _semex_trace(
        "SEMEX_CONFIGURED",
        configured=semantic_extractor is not None,
        extractor=type(semantic_extractor).__name__ if semantic_extractor is not None else "None",
    )
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
    rejected_results: list[dict[str, Any]] = []
    claims: list[ResearchClaim] = []
    semantic_extraction_calls = 0
    semantic_extraction_batches = 0
    semantic_extraction_latency_ms = 0
    semantic_extraction_input_chars = 0
    semantic_extraction_sources_used = 0
    semantic_candidates_before_validation = 0
    semantic_candidates_after_validation = 0
    semantic_extraction_statuses: list[str] = []
    semantic_extraction_failure_reasons: list[str] = []
    parsed_claims: list[ResearchClaim] = []
    for query in request.query_plan:
        raw = provider.run({"query": query})
        parsed_results, parsed_query_claims = _parse_provider_payload(raw, query)
        raw_results.extend(parsed_results)
        parsed_claims.extend(parsed_query_claims)

    semantic_batches: dict[tuple[str, str], list[ResearchClaim]] = {}
    for claim in parsed_claims:
        job = semantic_extraction_job(request, claim)
        if job is not None:
            semantic_batches.setdefault((job["goal"], job["origin"]), []).append(claim)
            continue
        (
            semantic_claims,
            extraction_calls,
            before_count,
            after_count,
            extraction_status,
            extraction_failure_reason,
        ) = _semantic_claims_for_request(
            request, claim, None
        )
        semantic_extraction_calls += extraction_calls
        semantic_candidates_before_validation += before_count
        semantic_candidates_after_validation += after_count
        if extraction_status:
            semantic_extraction_statuses.append(extraction_status)
        if extraction_failure_reason:
            semantic_extraction_failure_reasons.append(extraction_failure_reason)
        for semantic_claim in semantic_claims:
            evaluated = _evaluate_claim_for_request(request, semantic_claim)
            result = _result_for_claim(raw_results, claim)
            if evaluated.relevance == "irrelevant" or (
                evaluated.applicability in {"family_related", "adjacent", "unknown"} and _requires_exact(evaluated)
            ):
                rejected_results.append(_annotate_rejected_result(result, evaluated))
                continue
            claims.append(evaluated)

    for (goal, origin), batch_claims in semantic_batches.items():
        allowed_types = _semantic_allowed_types(goal)
        batch_result = _semantic_claims_for_batch(
            request,
            batch_claims,
            extraction_goal=goal,
            allowed_types=allowed_types,
            semantic_extractor=semantic_extractor,
            origin=origin,
        )
        semantic_extraction_calls += batch_result["calls"]
        semantic_extraction_batches += batch_result["batches"]
        semantic_extraction_latency_ms += batch_result["latency_ms"]
        semantic_extraction_input_chars += batch_result["input_chars"]
        semantic_extraction_sources_used += batch_result["sources_used"]
        semantic_candidates_before_validation += batch_result["before"]
        semantic_candidates_after_validation += batch_result["after"]
        if batch_result["status"]:
            semantic_extraction_statuses.append(batch_result["status"])
        if batch_result["failure_reason"]:
            semantic_extraction_failure_reasons.append(batch_result["failure_reason"])
        for semantic_claim in batch_result["claims"]:
            evaluated = _evaluate_claim_for_request(request, semantic_claim)
            result = _result_for_claim(raw_results, semantic_claim)
            if evaluated.relevance == "irrelevant" or (
                evaluated.applicability in {"family_related", "adjacent", "unknown"} and _requires_exact(evaluated)
            ):
                rejected_results.append(_annotate_rejected_result(result, evaluated))
                continue
            claims.append(evaluated)

    entries: list[KnowledgeEntry] = []
    for claim in claims:
        entry = _ingest_claim(work_session, request, claim)
        if entry is not None:
            entries.append(entry)
    _ingest_relevant_source_results(work_session, request, raw_results, entries, rejected_results)
    _store_research_sources(work_session, request, entries, raw_results, rejected_results)

    if entries:
        reasoning = task_reasoning(work_session, task.id)
        reasoning.stale = True
        _refresh_reasoning(work_session, task)
        request.result_ids = [entry.id for entry in entries]
        request.status = "completed"
    else:
        request.status = "partial" if raw_results else "failed"

    semantic_extraction_status = _semantic_extraction_status(
        semantic_extraction_calls,
        semantic_candidates_before_validation,
        semantic_candidates_after_validation,
        semantic_extraction_statuses,
    )
    semantic_extraction_failure_reason = "; ".join(_dedupe(semantic_extraction_failure_reasons))
    if semantic_extraction_status == "success":
        semantic_extraction_failure_reason = ""
    _semex_trace(
        "SEMEX_TELEMETRY_COMMITTED",
        task_id=request.task_id,
        calls=semantic_extraction_calls,
        status=semantic_extraction_status,
        before=semantic_candidates_before_validation,
        after=semantic_candidates_after_validation,
        batches=semantic_extraction_batches,
        latency_ms=semantic_extraction_latency_ms,
        input_chars=semantic_extraction_input_chars,
        sources_used=semantic_extraction_sources_used,
    )
    summary = synthesize_research_update(work_session, task, request, entries, raw_results, rejected_results)
    _store_research_summary(work_session, request, summary)
    return ResearchRunResult(
        request=request,
        claims=claims,
        raw_results=_compact_results_for_ui(raw_results, entries, rejected_results),
        rejected_results=rejected_results,
        ingested_entries=entries,
        summary=summary,
        semantic_extraction_calls=semantic_extraction_calls,
        semantic_extraction_status=semantic_extraction_status,
        semantic_candidates_before_validation=semantic_candidates_before_validation,
        semantic_candidates_after_validation=semantic_candidates_after_validation,
        semantic_extraction_failure_reason=semantic_extraction_failure_reason,
        semantic_extraction_batches=semantic_extraction_batches,
        semantic_extraction_latency_ms=semantic_extraction_latency_ms,
        semantic_extraction_input_chars=semantic_extraction_input_chars,
        semantic_extraction_sources_used=semantic_extraction_sources_used,
    )


def synthesize_research_update(
    work_session,
    task,
    request: ResearchRequest,
    entries: list[KnowledgeEntry],
    raw_results: list[dict[str, Any]],
    rejected_results: list[dict[str, Any]] | None = None,
) -> str:
    rejected_results = rejected_results or []
    related_count = len([item for item in rejected_results if item.get("applicability") in {"family_related", "adjacent"}])
    irrelevant_count = len([item for item in rejected_results if item.get("relevance") == "irrelevant"])
    semantic_entries = [entry for entry in entries if entry.kind != "source"]
    source_entries = [entry for entry in entries if entry.kind == "source"]
    if not entries and raw_results:
        if related_count:
            return "Encontrei resultados, mas eram sobre modelos relacionados. Separei-os para não os tratar como prova direta."
        if irrelevant_count:
            return "Encontrei resultados, mas descartei-os por não terem relação suficiente com esta tarefa."
        return "Encontrei resultados, mas ainda não consegui transformá-los em conhecimento seguro para esta tarefa."
    if not entries:
        return "Não encontrei informação suficientemente estruturada para atualizar esta tarefa."
    if source_entries and not semantic_entries and getattr(task, "task_type", "") == "plan_trip":
        roles = {question.semantic_role for question in request.questions}
        predicates = {question.predicate for question in request.questions}
        if "destination_discovery" in roles:
            return "Encontrei fontes sobre city breaks, mas ainda não consegui extrair destinos fiáveis para te sugerir."
        if "destination" in roles and "activities" in predicates:
            destination = request.questions[0].subject if request.questions else getattr(task, "title", "esse destino")
            return f"Encontrei fontes relevantes sobre {destination}, mas ainda não consegui extrair atividades fiáveis."

    subjects = _dedupe([entry.subject for entry in semantic_entries])
    source_count = len(_dedupe([entry.source.reference for entry in entries if entry.source.reference]))
    strong_count = len([entry for entry in semantic_entries if str(entry.metadata.get("source_strength_for_claim") or "") == "strong"])
    weak_count = len([entry for entry in semantic_entries if str(entry.metadata.get("source_strength_for_claim") or "") == "weak"])
    conflict_text = _research_conflict_sentence(entries)
    reasoning = task_reasoning(work_session, task.id)
    destination_candidates = [entry.subject for entry in semantic_entries if entry.predicate == "destination_candidate"]
    activities = [entry.value for entry in semantic_entries if entry.predicate == "activities" and entry.value]
    if getattr(task, "task_type", "") == "plan_trip" and destination_candidates:
        visible_candidates = _dedupe(destination_candidates)[:5]
        candidate_count = len(visible_candidates)
        plausible_word = "plausível" if candidate_count == 1 else "plausíveis"
        update = (
            f"Encontrei {candidate_count} destino"
            f"{'s' if candidate_count != 1 else ''} {plausible_word}: "
            f"{_join_pt(visible_candidates)}."
        )
        if source_count:
            update += f" Usei {source_count} fonte" + ("s" if source_count != 1 else "") + " relevante" + ("s" if source_count != 1 else "") + "."
        if related_count:
            update += " Separei resultados relacionados para não os tratar como prova direta."
        if irrelevant_count:
            update += f" Descartei {irrelevant_count} resultado" + ("s" if irrelevant_count != 1 else "") + " sem relação suficiente."
        if conflict_text:
            update += " " + conflict_text
        update += " Ainda falta validar preços reais de voo e alojamento."
        return update
    if getattr(task, "task_type", "") == "plan_trip" and activities:
        update = (
            f"Encontrei {len(_dedupe(activities))} atividade"
            f"{'s' if len(_dedupe(activities)) != 1 else ''} relevante"
            f"{'s' if len(_dedupe(activities)) != 1 else ''}: "
            f"{_join_pt(_dedupe(activities)[:5])}."
        )
        if source_count:
            update += f" Usei {source_count} fonte" + ("s" if source_count != 1 else "") + " relevante" + ("s" if source_count != 1 else "") + "."
        if related_count:
            update += " Separei resultados relacionados para não os tratar como prova direta."
        if irrelevant_count:
            update += f" Descartei {irrelevant_count} resultado" + ("s" if irrelevant_count != 1 else "") + " sem relação suficiente."
        if conflict_text:
            update += " " + conflict_text
        update += " Ainda falta confirmar horários para as tuas datas."
        return update
    update = f"Pesquisa concluída: encontrei informação útil sobre {_join_pt(subjects)}."
    if source_count:
        update += f" Usei {source_count} fonte" + ("s" if source_count != 1 else "") + " relevante" + ("s" if source_count != 1 else "") + "."
    if related_count:
        update += " Separei resultados sobre modelos relacionados para não os tratar como prova direta."
    if irrelevant_count:
        update += f" Descartei {irrelevant_count} resultado" + ("s" if irrelevant_count != 1 else "") + " sem relação suficiente."
    if conflict_text:
        update += " " + conflict_text
    if source_count and not strong_count and weak_count >= source_count:
        update += " A evidência ainda é limitada, por isso eu trataria a conclusão como provisória."
    elif reasoning.current_recommendation:
        update += f" Neste momento, {reasoning.current_recommendation} parece ganhar força, ainda de forma provisória."
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
    sources = research_source_records(work_session, task.id)
    entries = [
        entry
        for entry in task_knowledge(work_session, task.id)
        if entry.status == "active" and entry.source.source_type == "web"
    ]
    if not entries and not sources:
        return ""
    if sources:
        matching_sources = _matching_research_sources(sources, message) or sources[-5:]
        lines = []
        for source in matching_sources[-5:]:
            role = str(source.get("relevance_role") or source.get("source_quality") or "fonte")
            title = str(source.get("title") or "Fonte web")
            domain = str(source.get("domain") or "")
            url = str(source.get("url") or "")
            lines.append(f"{title}" + (f" ({domain})" if domain else "") + f" — {role}" + (f" — {url}" if url else ""))
        return "Estas são as fontes mais úteis que tenho nesta pesquisa: " + "; ".join(lines) + "."
    matching_entries = [
        entry
        for entry in entries
        if normalize_for_matching(entry.subject) in text or normalize_for_matching(entry.predicate) in text
    ] or entries[-5:]
    lines = []
    for entry in matching_entries[-5:]:
        title = str(entry.metadata.get("source_title") or entry.source.reference or "fonte web")
        url = entry.source.reference
        lines.append(f"{entry.subject}: {entry.predicate} — {title}" + (f" ({url})" if url else ""))
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
    explicit_targets = _explicit_research_option_targets(options, message)
    criterion = _criterion_from_message(message)
    if not criterion:
        criterion = _first_missing_comparison_criterion(knowledge)
    if not criterion:
        criterion = "general_quality"
    request.criteria = [criterion]
    target_options = explicit_targets or options
    target_norms = {normalize_for_matching(option) for option in target_options}
    request.target_object_ids = [
        obj.id
        for obj in work_session.objects.values()
        if obj.task_id == task.id and obj.kind == "option" and normalize_for_matching(obj.title) in target_norms
    ]
    if criterion in {"camera_quality", "battery"} and _needs_exact_models(task, target_options):
        label = "câmara" if criterion == "camera_quality" else "bateria"
        request.clarification = f"Antes de pesquisar {label}, preciso de saber os modelos exatos. Samsung e Apple podem significar muitos telemóveis diferentes."
        return
    for option in target_options:
        request.questions.append(
            ResearchQuestion(
                option,
                criterion,
                _question_for_criterion(option, criterion),
                semantic_role="comparison_option",
                required_entity_precision="exact" if criterion in {"camera_quality", "battery", "price"} else "family_ok",
                desired_source_type="independent_review" if criterion in {"camera_quality", "battery"} else "",
            )
        )
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
            request.questions.append(
                ResearchQuestion(
                    destination,
                    criterion,
                    f"{criterion} em {destination}",
                    semantic_role="destination",
                    required_entity_precision="family_ok",
                    desired_source_type="official" if criterion in {"opening_hours", "transport"} else "",
                )
            )
            request.query_plan.append(_travel_destination_query(destination, criterion, date_range))
        return
    style = _latest_value(knowledge, "user", "travel_style") or task.parameters.get("travel_style", "")
    flight_constraint = _latest_value(knowledge, "trip", "flight_constraint") or task.parameters.get("flight_constraint", "")
    basis = " ".join(part for part in (origin, duration, budget, style, flight_constraint) if part)
    request.criteria = ["flight_duration", "lodging_cost", "daily_cost", "things_to_visit", "fit_for_trip"]
    request.query_plan = [
        _travel_discovery_query(origin, duration, budget, style, flight_constraint),
        f"European city breaks FROM {origin or 'specified origin'} destination ideas {duration} {budget} short direct flights rest sightseeing".strip(),
    ]
    request.questions = [
        ResearchQuestion("trip", criterion, criterion, semantic_role="origin" if criterion == "flight_duration" else "destination_discovery")
        for criterion in request.criteria
    ]


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
        if any("destination" in query or "destino" in query for query in request.query_plan):
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
            id=_research_source_id(str(item.get("url", "")), str(item.get("title", ""))),
            domain=str(item.get("domain", "")),
            source_quality=_source_quality_label(item),
            source_kind=_source_kind(item),
            retrieved_at=str(item.get("retrieved_at") or now),
        )
        parsed_results.append({
            "title": source.title,
            "url": source.url,
            "domain": source.domain,
            "snippet": str(item.get("snippet", "")),
            "evidence": str(item.get("content") or item.get("raw_content") or item.get("snippet") or ""),
            "source_quality": source.source_quality,
            "source_kind": source.source_kind,
            "query": query,
            "provider": str(item.get("provider") or ""),
            "fallback_reason": str(item.get("fallback_reason") or ""),
            "score": item.get("score"),
        })
        for claim in item.get("claims", []):
            if not isinstance(claim, dict):
                continue
            subject = str(claim.get("subject", "")).strip()
            predicate = str(claim.get("predicate", "")).strip()
            value = str(claim.get("value", "")).strip()
            if not subject or not predicate or not value:
                continue
            raw_evidence = str(claim.get("evidence") or item.get("content") or item.get("raw_content") or item.get("snippet") or value)
            claims.append(
                ResearchClaim(
                    subject=subject,
                    predicate=predicate,
                    value=value,
                    kind=str(claim.get("kind", "claim")),
                    confidence=float(claim.get("confidence", 0.65)),
                    epistemic_type=str(claim.get("epistemic_type", "claim")),
                    evidence=_clean_evidence_text(raw_evidence, source.title),
                    source=source,
                    metadata={
                        "provider": str(item.get("provider") or ""),
                        "fallback_reason": str(item.get("fallback_reason") or ""),
                        "score": item.get("score"),
                        "query": query,
                        "result_title": source.title,
                        "result_snippet": str(item.get("snippet", "")),
                        "raw_evidence": raw_evidence,
                    },
                )
            )
    return parsed_results, claims


SEMANTIC_BATCH_CHAR_CAP = 9000
SEMANTIC_BATCH_SOURCE_CAP = 10
SEMANTIC_FIRST_BATCH_TARGET_SOURCES = 5
SEMANTIC_SOURCE_FRAGMENT_CHAR_CAP = 520


def semantic_extraction_job(request: ResearchRequest, claim: ResearchClaim) -> dict[str, str] | None:
    """Return the semantic extraction job for a claim, or None for deterministic handling."""

    question = _best_question_for_claim(request, claim)
    if question and question.semantic_role == "destination_discovery":
        if claim.predicate not in {"web_evidence", "fit_for_trip", "destination_candidate"}:
            return None
        origin = _origin_from_discovery_request(request)
        if _direct_destination_candidate(claim, origin=origin):
            return None
        return {"goal": "destination_discovery", "origin": origin}
    if question and question.semantic_role == "destination" and question.predicate == "activities":
        if claim.predicate in {"price", "cost", "transport", "opening_hours", "availability"}:
            return None
        if claim.predicate == "web_evidence" and not _web_evidence_supports_activity_extraction(request, claim):
            return None
        if _direct_place_candidate(claim):
            return None
        if not _has_activity_semantic_need(request, claim):
            return None
        return {"goal": "stockholm_places", "origin": ""}
    return None


def _semantic_allowed_types(goal: str) -> set[str]:
    if goal == "destination_discovery":
        return {"destination"}
    return {"place", "attraction", "activity"}


def _has_activity_semantic_need(request: ResearchRequest, claim: ResearchClaim) -> bool:
    text = normalize_for_matching(
        " ".join(
            [
                claim.subject,
                claim.predicate,
                claim.value,
                claim.evidence,
                claim.source.title,
                claim.source.domain,
                str(claim.metadata.get("query") or ""),
            ]
        )
    )
    if claim.predicate == "activities":
        return True
    if "activities" not in set(request.criteria):
        return False
    return contains_any_phrase(
        text,
        (
            "things to do",
            "what to do",
            "attraction",
            "attractions",
            "museum",
            "museu",
            "visitar",
            "visit stockholm",
            "visit sweden",
            "tourism",
            "turismo",
            "skansen",
            "vasa",
            "abba",
            "grona",
            "gröna",
            "gamla stan",
        ),
    )


def _web_evidence_supports_activity_extraction(request: ResearchRequest, claim: ResearchClaim) -> bool:
    query = normalize_for_matching(str(claim.metadata.get("query") or ""))
    if not contains_any_phrase(query, ("things to do", "what to do", "activities", "attractions", "visitar")):
        return False
    text = normalize_for_matching(" ".join([claim.subject, claim.predicate, claim.value, claim.evidence, claim.source.title, claim.source.domain]))
    return contains_any_phrase(
        text,
        (
            "things to do",
            "what to do",
            "attraction",
            "attractions",
            "museum",
            "museu",
            "visitar",
            "visit stockholm",
            "visit sweden",
            "tourism",
            "turismo",
            "skansen",
            "vasa",
            "abba",
            "grona",
            "gröna",
            "gamla stan",
        ),
    )


def _semantic_claims_for_batch(
    request: ResearchRequest,
    batch_claims: list[ResearchClaim],
    *,
    extraction_goal: str,
    allowed_types: set[str],
    semantic_extractor: SemanticExtractor | None,
    origin: str = "",
) -> dict[str, Any]:
    if semantic_extractor is None:
        return {
            "claims": [],
            "calls": 0,
            "batches": 0,
            "latency_ms": 0,
            "input_chars": 0,
            "sources_used": 0,
            "before": 0,
            "after": 0,
            "status": "not_configured",
            "failure_reason": "semantic_extractor_unavailable",
        }
    accepted_claims: list[ResearchClaim] = []
    total_calls = 0
    total_batches = 0
    total_latency_ms = 0
    total_input_chars = 0
    total_sources_used = 0
    total_before = 0
    statuses: list[str] = []
    failure_reasons: list[str] = []
    sorted_claims = _select_semantic_batch_claims(
        _sort_claims_for_semantic_batch(batch_claims),
        extraction_goal=extraction_goal,
    )
    for batch in _semantic_extraction_batches(sorted_claims):
        if len(_dedupe([claim.value for claim in accepted_claims])) >= _semantic_sufficiency_threshold(extraction_goal):
            break
        started_at = time.perf_counter()
        _semex_trace(
            "batch_started",
            goal=extraction_goal,
            sources=len(batch),
            evidence_chars=sum(len(_claim_semantic_evidence(claim)) for claim in batch),
        )
        candidates, calls, before_count, status, failure_reason, input_chars = _validated_semantic_batch_candidates(
            request,
            batch,
            extraction_goal=extraction_goal,
            allowed_types=allowed_types,
            semantic_extractor=semantic_extractor,
            origin=origin,
        )
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        total_latency_ms += latency_ms
        total_input_chars += input_chars
        total_sources_used += len(batch) if calls else 0
        total_calls += calls
        total_batches += 1 if calls else 0
        total_before += before_count
        statuses.append(status)
        if failure_reason:
            failure_reasons.append(failure_reason)
        produced = _claims_from_batch_candidates(request, batch, candidates, extraction_goal=extraction_goal, origin=origin)
        accepted_claims.extend(produced)
        _semex_trace(
            "batch_finished",
            goal=extraction_goal,
            latency_ms=latency_ms,
            before=before_count,
            after=len(produced),
        )
    accepted_values = _dedupe([claim.value for claim in accepted_claims])
    return {
        "claims": accepted_claims,
        "calls": total_calls,
        "batches": total_batches,
        "latency_ms": total_latency_ms,
        "input_chars": total_input_chars,
        "sources_used": total_sources_used,
        "before": total_before,
        "after": len(accepted_values),
        "status": _semantic_extraction_status(total_calls, total_before, len(accepted_values), statuses),
        "failure_reason": "; ".join(_dedupe(failure_reasons)),
    }


def _semantic_extraction_batches(claims: list[ResearchClaim]) -> list[list[ResearchClaim]]:
    batches: list[list[ResearchClaim]] = []
    current: list[ResearchClaim] = []
    current_chars = 0
    for claim in claims:
        evidence_chars = len(_claim_semantic_evidence(claim))
        if current and (
            current_chars + evidence_chars > SEMANTIC_BATCH_CHAR_CAP
            or len(current) >= SEMANTIC_BATCH_SOURCE_CAP
        ):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(claim)
        current_chars += evidence_chars
    if current:
        batches.append(current)
    return batches[:3]


def _select_semantic_batch_claims(claims: list[ResearchClaim], *, extraction_goal: str) -> list[ResearchClaim]:
    if extraction_goal != "stockholm_places" or len(claims) <= SEMANTIC_FIRST_BATCH_TARGET_SOURCES:
        return claims
    selected: list[ResearchClaim] = []
    for claim in claims:
        selected.append(claim)
        if len(selected) >= SEMANTIC_FIRST_BATCH_TARGET_SOURCES and _batch_has_rich_place_evidence(selected):
            return selected
    return selected


def _batch_has_rich_place_evidence(claims: list[ResearchClaim]) -> bool:
    text = normalize_for_matching(" ".join(_claim_semantic_evidence(claim) for claim in claims))
    signals = (
        "museum",
        "museu",
        "skansen",
        "vasa",
        "abba",
        "grona",
        "gröna",
        "djurgarden",
        "djurgården",
        "gamla stan",
        "attraction",
        "attractions",
    )
    return sum(1 for signal in signals if signal in text) >= 4


def _sort_claims_for_semantic_batch(claims: list[ResearchClaim]) -> list[ResearchClaim]:
    priority = {"official": 0, "specialist_media": 1, "independent_test": 1, "independent_review": 2, "community": 3, "video": 3, "social": 4}
    return sorted(
        claims,
        key=lambda claim: (
            priority.get(claim.source.source_kind or _source_kind(claim.source), 5),
            -float(claim.metadata.get("score") or 0.0),
            claim.source.title,
        ),
    )


def _semantic_sufficiency_threshold(goal: str) -> int:
    return 5 if goal == "destination_discovery" else 6


def _claim_semantic_evidence(claim: ResearchClaim) -> str:
    return _clean_evidence_text(
        " ".join(part for part in (claim.value, claim.evidence) if part),
        claim.source.title,
    )


def _validated_semantic_batch_candidates(
    request: ResearchRequest,
    claims: list[ResearchClaim],
    *,
    extraction_goal: str,
    allowed_types: set[str],
    semantic_extractor: SemanticExtractor,
    origin: str = "",
) -> tuple[list[SemanticCandidate], int, int, str, str, int]:
    evidence = _batch_evidence_payload(claims)
    if not evidence:
        return [], 0, 0, "not_requested", "empty_evidence", 0
    _semex_trace(
        "SEMEX_REQUEST_CREATED",
        task_id=request.task_id,
        goal=extraction_goal,
        extractor=type(semantic_extractor).__name__,
        sources=len(claims),
        evidence_chars=len(evidence),
    )
    try:
        _semex_trace("SEMEX_REQUEST_BEGIN", task_id=request.task_id, goal=extraction_goal, sources=len(claims))
        _semex_trace("SEMEX_CALL_STARTED", task_id=request.task_id, goal=extraction_goal, sources=len(claims))
        extraction_result = _coerce_semantic_extraction_result(semantic_extractor.extract(
            SemanticExtractionRequest(
                task_type=request.task_type or request.task_id,
                extraction_goal=extraction_goal,
                known_context={
                    "criteria": list(request.criteria),
                    "questions": [question.question for question in request.questions],
                    "query_plan": list(request.query_plan),
                    "batch_source_ids": [claim.source.id or claim.source.url or claim.source.title for claim in claims],
                },
                excluded_entities=[origin] if origin else [],
                evidence=evidence,
                source_title=f"Batch com {len(claims)} fontes",
            )
        ))
    except Exception as exc:
        _semex_trace("SEMEX_CALL_RETURNED", task_id=request.task_id, status="call_failed", candidates=0)
        _semex_trace("SEMEX_REQUEST_END", task_id=request.task_id, status="call_failed", before=0, after=0, failure_reason=type(exc).__name__)
        return [], 1, 0, "call_failed", type(exc).__name__, len(evidence)
    raw_candidates = extraction_result.candidates
    accepted: list[SemanticCandidate] = []
    rejection_reasons: list[str] = []
    for candidate in raw_candidates:
        normalized_type = normalize_for_matching(candidate.entity_type)
        if normalized_type not in allowed_types:
            rejection_reasons.append("unsupported_entity_type")
            continue
        name = _normalize_place_name(candidate.canonical_name) if extraction_goal == "stockholm_places" else _normalize_destination_name(candidate.canonical_name)
        if extraction_goal == "destination_discovery":
            name = _canonical_destination_identity(name)
        normalized_candidate = SemanticCandidate(
            name,
            normalized_type,
            candidate.evidence_quote_or_fragment,
            candidate.confidence,
            candidate.aliases,
            candidate.source_ids,
        )
        support = _supporting_claim_for_candidate(normalized_candidate, claims)
        if support is None:
            rejection_reasons.append("not_grounded_in_evidence")
            continue
        rejection_reason = _semantic_candidate_rejection_reason(normalized_candidate, support, extraction_goal=extraction_goal, origin=origin)
        if rejection_reason:
            rejection_reasons.append(rejection_reason)
            continue
        accepted.append(normalized_candidate)
    if accepted:
        status = "success"
    elif extraction_result.status in {"call_failed", "invalid_json", "empty_result"}:
        status = extraction_result.status
    elif raw_candidates:
        status = "all_rejected"
    else:
        status = "empty_result"
    failure_reason = extraction_result.failure_reason
    if not failure_reason and status == "all_rejected":
        failure_reason = "; ".join(_dedupe(rejection_reasons))
    _semex_trace("SEMEX_CALL_RETURNED", task_id=request.task_id, status=extraction_result.status, candidates=len(raw_candidates))
    _semex_trace("SEMEX_PARSE_RESULT", task_id=request.task_id, status=extraction_result.status, candidates=len(raw_candidates))
    _semex_trace("SEMEX_VALIDATION_RESULT", task_id=request.task_id, status=status, before=len(raw_candidates), after=len(accepted))
    _semex_trace("SEMEX_REQUEST_END", task_id=request.task_id, status=status, before=len(raw_candidates), after=len(accepted), failure_reason=failure_reason)
    return accepted, 1, len(raw_candidates), status, failure_reason, len(evidence)


def _batch_evidence_payload(claims: list[ResearchClaim]) -> str:
    sources = []
    for claim in claims:
        evidence = _claim_semantic_evidence(claim)
        if not evidence:
            continue
        sources.append(
            {
                "source_id": claim.source.id or claim.source.url or claim.source.title,
                "title": claim.source.title,
                "source_kind": claim.source.source_kind,
                "predicate": claim.predicate,
                "evidence": _compact_semantic_evidence_fragment(evidence),
            }
        )
    return json.dumps({"sources": sources}, ensure_ascii=False)


def _compact_semantic_evidence_fragment(evidence: str) -> str:
    text = re.sub(r"\s+", " ", str(evidence or "")).strip()
    if len(text) <= SEMANTIC_SOURCE_FRAGMENT_CHAR_CAP:
        return text
    candidates = re.split(r"(?<=[.!?])\s+", text)
    scored: list[tuple[int, int, str]] = []
    for index, sentence in enumerate(candidates):
        normalized = normalize_for_matching(sentence)
        score = sum(
            1
            for signal in (
                "museum",
                "museu",
                "attraction",
                "attractions",
                "things to do",
                "visit",
                "visitar",
                "skansen",
                "vasa",
                "abba",
                "grona",
                "gröna",
                "djurgarden",
                "djurgården",
                "gamla stan",
                "park",
            )
            if signal in normalized
        )
        if score:
            scored.append((score, -index, _semantic_signal_window(sentence.strip())))
    fragments = [item[2] for item in sorted(scored, reverse=True)]
    if not fragments:
        fragments = candidates[:2]
    compact = " ".join(fragments)
    return compact[:SEMANTIC_SOURCE_FRAGMENT_CHAR_CAP].strip()


def _semantic_signal_window(text: str) -> str:
    if len(text) <= SEMANTIC_SOURCE_FRAGMENT_CHAR_CAP:
        return text
    normalized = normalize_for_matching(text)
    signals = (
        "vasa",
        "skansen",
        "abba",
        "grona",
        "gröna",
        "djurgarden",
        "djurgården",
        "gamla stan",
        "museum",
        "museu",
        "attraction",
        "attractions",
        "things to do",
        "visit",
        "visitar",
        "park",
    )
    center = -1
    for signal in signals:
        position = normalized.find(signal)
        if position >= 0:
            center = position
            break
    if center < 0:
        return text[:SEMANTIC_SOURCE_FRAGMENT_CHAR_CAP].strip()
    start = max(0, center - 180)
    end = min(len(text), start + SEMANTIC_SOURCE_FRAGMENT_CHAR_CAP)
    if end - start < SEMANTIC_SOURCE_FRAGMENT_CHAR_CAP:
        start = max(0, end - SEMANTIC_SOURCE_FRAGMENT_CHAR_CAP)
    return text[start:end].strip()


def _claims_from_batch_candidates(
    request: ResearchRequest,
    claims: list[ResearchClaim],
    candidates: list[SemanticCandidate],
    *,
    extraction_goal: str,
    origin: str = "",
) -> list[ResearchClaim]:
    produced: list[ResearchClaim] = []
    seen: set[str] = set()
    for candidate in candidates:
        support = _supporting_claim_for_candidate(candidate, claims)
        if support is None:
            continue
        key = normalize_for_matching(candidate.canonical_name)
        if not key or key in seen:
            continue
        seen.add(key)
        question = _best_question_for_claim(request, support)
        if extraction_goal == "destination_discovery":
            produced.append(
                _clone_claim(
                    support,
                    subject=candidate.canonical_name,
                    value=_destination_fit_value(candidate.canonical_name, support),
                    kind="candidate",
                    metadata_extra={"semantic_extraction": "destination_candidate", "origin_excluded": origin},
                )
            )
        else:
            produced.append(
                _clone_claim(
                    support,
                    subject=question.subject if question else support.subject,
                    value=candidate.canonical_name,
                    metadata_extra={
                        "semantic_extraction": "named_place",
                        "original_value": support.value,
                        "semantic_batch_source_id": support.source.id,
                    },
                )
            )
    return produced


def _supporting_claim_for_candidate(candidate: SemanticCandidate, claims: list[ResearchClaim]) -> ResearchClaim | None:
    candidate_names = [candidate.canonical_name, *candidate.aliases]
    fragment = normalize_for_matching(candidate.evidence_quote_or_fragment)
    source_ids = {normalize_for_matching(source_id) for source_id in candidate.source_ids if source_id}
    for claim in claims:
        claim_source_ids = {
            normalize_for_matching(claim.source.id),
            normalize_for_matching(claim.source.url),
            normalize_for_matching(claim.source.title),
        }
        if source_ids and source_ids.intersection(claim_source_ids):
            return claim
    for claim in claims:
        haystack = normalize_for_matching(" ".join([claim.value, claim.evidence, claim.source.title]))
        if fragment and fragment in haystack:
            return claim
        if any(name and normalize_for_matching(name) in haystack for name in candidate_names):
            return claim
    return None


def _semantic_claims_for_request(
    request: ResearchRequest,
    claim: ResearchClaim,
    semantic_extractor: SemanticExtractor | None = None,
) -> tuple[list[ResearchClaim], int, int, int, str, str]:
    question = _best_question_for_claim(request, claim)
    if question and question.semantic_role == "destination_discovery":
        origin = _origin_from_discovery_request(request)
        direct = _direct_destination_candidate(claim, origin=origin)
        if direct:
            candidates = [direct]
            extraction_calls = 0
            before_count = 1
            extraction_status = "not_requested"
            extraction_failure_reason = ""
        else:
            candidates, extraction_calls, before_count, extraction_status, extraction_failure_reason = _validated_semantic_candidates(
                request,
                claim,
                extraction_goal="destination_discovery",
                allowed_types={"destination"},
                semantic_extractor=semantic_extractor,
                origin=origin,
            )
        if candidates:
            return [
                _clone_claim(
                    claim,
                    subject=candidate,
                    value=_destination_fit_value(candidate, claim),
                    kind="candidate",
                    metadata_extra={"semantic_extraction": "destination_candidate", "origin_excluded": origin},
                )
                for candidate in candidates[:5]
            ], extraction_calls, before_count, len(candidates), extraction_status, extraction_failure_reason
        return [claim], extraction_calls, before_count, 0, extraction_status, extraction_failure_reason
    if question and question.semantic_role == "destination" and question.predicate == "activities":
        direct = _direct_place_candidate(claim)
        if direct:
            places = [direct]
            extraction_calls = 0
            before_count = 1
            extraction_status = "not_requested"
            extraction_failure_reason = ""
        else:
            places, extraction_calls, before_count, extraction_status, extraction_failure_reason = _validated_semantic_candidates(
                request,
                claim,
                extraction_goal="stockholm_places",
                allowed_types={"place", "attraction", "activity"},
                semantic_extractor=semantic_extractor,
            )
        if places:
            return [
                _clone_claim(
                    claim,
                    subject=question.subject,
                    value=place,
                    metadata_extra={"semantic_extraction": "named_place", "original_value": claim.value},
                )
                for place in places[:6]
            ], extraction_calls, before_count, len(places), extraction_status, extraction_failure_reason
        return [claim], extraction_calls, before_count, 0, extraction_status, extraction_failure_reason
    if question and question.semantic_role == "destination" and question.predicate == "transport":
        transport = _extract_transport_claim(claim)
        if transport:
            return [_clone_claim(claim, value=transport, metadata_extra={"semantic_extraction": "transport_summary"})], 0, 0, 0, "not_requested", ""
    return [claim], 0, 0, 0, "not_requested", ""


def _clone_claim(
    claim: ResearchClaim,
    *,
    subject: str | None = None,
    predicate: str | None = None,
    value: str | None = None,
    kind: str | None = None,
    metadata_extra: dict[str, Any] | None = None,
) -> ResearchClaim:
    metadata = dict(claim.metadata)
    metadata.update(metadata_extra or {})
    return ResearchClaim(
        subject=subject if subject is not None else claim.subject,
        predicate=predicate if predicate is not None else claim.predicate,
        value=value if value is not None else claim.value,
        kind=kind if kind is not None else claim.kind,
        confidence=claim.confidence,
        epistemic_type=claim.epistemic_type,
        evidence=claim.evidence,
        source=claim.source,
        metadata=metadata,
        applicability=claim.applicability,
        relevance=claim.relevance,
        rejection_reason=claim.rejection_reason,
    )


def _evaluate_claim_for_request(request: ResearchRequest, claim: ResearchClaim) -> ResearchClaim:
    question = _best_question_for_claim(request, claim)
    relevance, relevance_reason = _claim_relevance(request, question, claim)
    applicability = _claim_applicability(question, claim)
    predicate = question.predicate if question else claim.predicate
    source_strength = _source_strength_for_claim(predicate, claim.source, claim.evidence)
    if relevance == "relevant" and not _has_substantive_evidence(predicate, claim):
        relevance = "irrelevant"
        relevance_reason = "insufficient_substantive_evidence"
    subject = question.subject if question and applicability == "exact" else claim.subject
    kind = claim.kind
    output_predicate = claim.predicate
    if question and question.semantic_role == "destination_discovery":
        subject = claim.subject
        output_predicate = "destination_candidate"
        kind = "candidate"
        applicability = "exact" if relevance == "relevant" else applicability
    elif (
        question
        and question.predicate == "activities"
        and claim.predicate not in {"opening_hours", "price", "transport", "availability"}
        and claim.value
        and not _looks_like_article_title(claim.value)
    ):
        output_predicate = "activities"
    value = _semantic_value_for_claim(question, claim)
    metadata = dict(claim.metadata)
    metadata.update(
        {
            "applicability": applicability,
            "relevance": relevance,
            "relevance_reason": relevance_reason,
            "claimed_subject": claim.subject,
            "semantic_role": question.semantic_role if question else "",
            "required_entity_precision": question.required_entity_precision if question else "unknown",
            "source_kind": claim.source.source_kind,
            "source_strength_for_claim": source_strength,
        }
    )
    source = ResearchSourceRecord(
        title=claim.source.title,
        url=claim.source.url,
        id=claim.source.id,
        domain=claim.source.domain,
        source_quality=claim.source.source_quality,
        source_kind=claim.source.source_kind,
        source_strength_for_claim=source_strength,
        retrieved_at=claim.source.retrieved_at,
        relevance=claim.source.relevance,
        relevance_reason=claim.source.relevance_reason,
    )
    return ResearchClaim(
        subject=subject,
        predicate=output_predicate,
        value=value,
        kind=kind,
        confidence=_adjusted_confidence(claim.confidence, relevance, applicability, source_strength),
        epistemic_type="semantic_claim" if relevance == "relevant" and applicability == "exact" else claim.epistemic_type,
        evidence=claim.evidence,
        source=source,
        metadata=metadata,
        applicability=applicability,
        relevance=relevance,
        rejection_reason="" if relevance == "relevant" else relevance_reason,
    )


def _best_question_for_claim(request: ResearchRequest, claim: ResearchClaim) -> ResearchQuestion | None:
    if not request.questions:
        return None
    subject_norm = normalize_for_matching(claim.subject)
    discovery_questions = [question for question in request.questions if question.semantic_role == "destination_discovery"]
    if discovery_questions and claim.predicate in {"web_evidence", "fit_for_trip", "destination_candidate"}:
        return discovery_questions[0]
    exact = [question for question in request.questions if normalize_for_matching(question.subject) == subject_norm]
    exact_predicate = [question for question in exact if question.predicate == claim.predicate]
    if exact_predicate:
        return exact_predicate[0]
    if exact:
        return exact[0]
    haystack = normalize_for_matching(" ".join([claim.subject, claim.evidence, str(claim.metadata.get("query") or "")]))
    mentioned = [
        question
        for question in request.questions
        if question.predicate == claim.predicate and normalize_for_matching(question.subject) in haystack
    ]
    if mentioned:
        mentioned.sort(key=lambda question: len(normalize_for_matching(question.subject)), reverse=True)
        return mentioned[0]
    for question in request.questions:
        if question.predicate == claim.predicate:
            return question
    return request.questions[0]


def _claim_relevance(request: ResearchRequest, question: ResearchQuestion | None, claim: ResearchClaim) -> tuple[str, str]:
    haystack = normalize_for_matching(" ".join([
        claim.subject,
        claim.predicate,
        claim.value,
        claim.evidence,
        claim.source.title,
        claim.source.domain,
        str(claim.metadata.get("query") or ""),
    ]))
    predicate = question.predicate if question else claim.predicate
    if _looks_like_irrelevant_same_keyword_result(predicate, haystack):
        return "irrelevant", "keyword_match_not_answering_question"
    if normalize_for_matching(claim.subject) in {"trip", "viagem", "travel"} and claim.predicate == "web_evidence":
        return "irrelevant", "raw_trip_evidence_not_candidate"
    if predicate == "activities" and str(claim.metadata.get("semantic_extraction") or "") != "named_place":
        compact_place = _normalize_place_name(claim.value)
        if not (_is_valid_place_candidate(compact_place) and _candidate_grounded(compact_place, claim)):
            return "irrelevant", "activity_requires_named_entity"
    if predicate == "activities" and any(token in haystack for token in ("museum", "museu", "palace", "visit stockholm", "tourism", "attraction", "gamla stan")):
        return "relevant", "tourism_activity_match"
    if predicate == "transport" and _source_kind(claim.source) == "social" and not _extract_transport_claim(claim):
        return "irrelevant", "weak_social_transport_noise"
    if predicate == "transport" and any(token in haystack for token in ("public transport", "metro", "sl ", "airport train", "transport authority")):
        return "relevant", "public_transport_match"
    if predicate in {"camera_quality", "battery", "support_quality", "price"}:
        return "relevant", "comparison_criterion_match"
    if question and question.semantic_role == "destination_discovery":
        candidate = _canonical_destination_identity(claim.subject)
        origin = _origin_from_discovery_request(request)
        if normalize_for_matching(claim.subject) in {"trip", "viagem", "travel", "user"}:
            return "irrelevant", "raw_trip_evidence_not_candidate"
        if not _is_valid_destination_candidate(candidate, origin=origin):
            return "irrelevant", "invalid_destination_identity"
        if _looks_like_article_title(claim.subject) or _looks_like_article_title(claim.value):
            return "irrelevant", "article_title_not_candidate"
        return "relevant", "destination_candidate"
    return "relevant", "default"


def _looks_like_irrelevant_same_keyword_result(predicate: str, haystack: str) -> bool:
    if predicate == "activities" and any(token in haystack for token in ("cooperation protocol", "university", "camões", "camoes", "archive", "historia archive")):
        return True
    if predicate == "transport" and any(token in haystack for token in ("road freight", "haulage", "logistics", "a/s road")):
        return True
    return False


def _claim_applicability(question: ResearchQuestion | None, claim: ResearchClaim) -> str:
    if question is None:
        return "unknown"
    target = question.subject
    if normalize_for_matching(target) in {"trip", "user"}:
        return "exact"
    text = normalize_for_matching(" ".join([claim.subject, claim.source.title, claim.evidence, str(claim.metadata.get("result_snippet") or "")]))
    target_norm = normalize_for_matching(target)
    if not target_norm:
        return "unknown"
    subject_norm = normalize_for_matching(claim.subject)
    if _mentions_related_variant_without_exact_base(target_norm, text) and not _mentions_explicit_base_model(target_norm, claim.evidence):
        return "family_related"
    if subject_norm == target_norm:
        return "exact"
    if subject_norm.startswith(target_norm + " ") and _mentions_related_variant_without_exact_base(target_norm, subject_norm):
        return "family_related"
    claim_text = normalize_for_matching(" ".join([claim.subject, claim.evidence, str(claim.metadata.get("result_snippet") or "")]))
    if _mentions_related_variant_without_exact_base(target_norm, claim_text) and not _mentions_explicit_base_model(target_norm, claim.evidence):
        return "family_related"
    if target_norm in claim_text:
        return "exact"
    if _mentions_related_variant_without_exact_base(target_norm, text):
        return "family_related"
    if subject_norm.startswith(target_norm + " ") or target_norm.startswith(subject_norm + " "):
        return "family_related"
    return "unknown"


def _mentions_related_variant_without_exact_base(target_norm: str, text: str) -> bool:
    variant_markers = (" ultra", " pro max", " pro", " plus", " max")
    for marker in variant_markers:
        if target_norm + marker in text:
            return True
    return False


def _mentions_explicit_base_model(target_norm: str, text: str) -> bool:
    normalized = normalize_for_matching(text)
    if not target_norm or target_norm not in normalized:
        return False
    return any(
        pattern in normalized
        for pattern in (
            f"base {target_norm}",
            f"{target_norm} base",
            f"standard {target_norm}",
            f"{target_norm} standard",
            f"modelo base {target_norm}",
            f"{target_norm} modelo base",
        )
    )


def _requires_exact(claim: ResearchClaim) -> bool:
    return str(claim.metadata.get("required_entity_precision") or "") == "exact"


def _semantic_value_for_claim(question: ResearchQuestion | None, claim: ResearchClaim) -> str:
    if question is None:
        return _short_semantic_text(claim.value)
    predicate = question.predicate
    text = normalize_for_matching(" ".join([claim.value, claim.evidence]))
    if predicate == "camera_quality":
        if contains_any_phrase(normalize_for_matching(claim.value), ("mixed", "mista", "variavel", "variável")):
            return "mixed"
        if contains_any_phrase(text, ("better", "superior", "more consistent", "mais consistente", "best", "melhor")):
            return "better"
        if contains_any_phrase(text, ("good", "strong", "boa", "forte")):
            return "good"
        return "mixed"
    if predicate == "battery":
        hour = re.search(r"\b(\d{1,2})\s*h\b", claim.value, flags=re.IGNORECASE)
        if hour:
            return f"{hour.group(1)} h"
        if contains_any_phrase(text, ("better", "superior", "longer", "melhor", "dura mais")):
            return "better"
        return _short_semantic_text(claim.value)
    if predicate == "support_quality":
        if contains_any_phrase(text, ("24h", "24 h", "strong", "better", "bom", "melhor")):
            return "better"
        if contains_any_phrase(text, ("limited", "lento", "slow", "horario laboral", "horário laboral")):
            return "limited"
        return _short_semantic_text(claim.value)
    if predicate == "transport":
        semantic_transport = _extract_transport_claim(claim)
        if semantic_transport:
            return semantic_transport
        return _short_semantic_text(claim.value)
    return _short_semantic_text(claim.value)


def _research_source_id(url: str, title: str) -> str:
    basis = url or title or "source"
    parsed = urlparse(url)
    if parsed.netloc:
        basis = parsed.netloc + parsed.path
    slug = re.sub(r"[^a-z0-9]+", "-", normalize_for_matching(basis)).strip("-")
    return ("source-" + slug[:80]).rstrip("-") or "source-web"


def _travel_destination_query(destination: str, criterion: str, date_range: str | None = "") -> str:
    labels = {
        "activities": "official tourism things to do attractions",
        "opening_hours": "official opening hours museums attractions",
        "transport": "official public transport airport city pass",
        "cost": "travel cost restaurants attractions prices",
    }
    parts = [destination, labels.get(criterion, criterion)]
    if date_range:
        parts.append(str(date_range))
    return " ".join(part for part in parts if part).strip()


def _travel_discovery_query(
    origin: str | None,
    duration: str | None,
    budget: str | None,
    style: str | None,
    flight_constraint: str | None,
) -> str:
    parts = [
        "European city break destination ideas",
        f"from {origin}" if origin else "from specified origin",
        duration or "",
        f"budget {budget}" if budget else "",
        style or "",
        flight_constraint or "",
    ]
    return " ".join(part for part in parts if part).strip()


def _origin_from_discovery_request(request: ResearchRequest) -> str:
    for query in request.query_plan:
        match = re.search(
            r"\bfrom\s+([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ' -]{1,40}?)(?=\s+\d|\s+budget|\s+short|\s+direct|\s+rest|\s+sightseeing|$)",
            query,
        )
        if match:
            return match.group(1).strip()
    return ""


def _semantic_extraction_prompt(request: SemanticExtractionRequest) -> str:
    if request.extraction_goal == "destination_discovery":
        instruction = (
            "Extrai apenas destinos geográficos reais explicitamente mencionados na evidência. "
            "Não devolvas adjectivos, títulos de artigo, pessoas, organizações, termos genéricos "
            "de viagem ou conceitos históricos."
        )
        entity_types = "destination"
    else:
        instruction = (
            "Extrai apenas lugares nomeados, museus, atrações ou atividades claramente nomeadas "
            "no destino referido. Não devolvas adjectivos, dias, meses, categorias genéricas ou títulos."
        )
        entity_types = "place, attraction, activity"
    payload = {
        "task_type": request.task_type,
        "extraction_goal": request.extraction_goal,
        "known_context": request.known_context,
        "excluded_entities": request.excluded_entities,
        "source_title": request.source_title,
        "evidence": request.evidence[:5000],
    }
    return (
        instruction
        + "\n\nFormato obrigatório: {\"candidates\":[{\"canonical_name\":\"...\","
        + "\"entity_type\":\""
        + entity_types
        + "\",\"source_ids\":[\"...\"],\"evidence_quote_or_fragment\":\"...\",\"confidence\":0.0,\"aliases\":[]}]}"
        + "\n\nEvidência:\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def _parse_semantic_candidates(raw: str) -> list[SemanticCandidate]:
    return _parse_semantic_candidates_with_status(raw).candidates


def _parse_semantic_candidates_with_status(raw: str) -> SemanticExtractionResult:
    text = str(raw or "").strip()
    if not text:
        return SemanticExtractionResult(status="empty_result")
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        decoded = json.loads(text)
    except Exception as exc:
        return SemanticExtractionResult(status="invalid_json", failure_reason=type(exc).__name__)
    if isinstance(decoded, dict):
        items = decoded.get("candidates")
    else:
        items = decoded
    if not isinstance(items, list):
        return SemanticExtractionResult(status="invalid_json", failure_reason="missing_candidates")
    candidates: list[SemanticCandidate] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = _clean_entity_name(str(item.get("canonical_name") or ""))
        entity_type = normalize_for_matching(str(item.get("entity_type") or ""))
        evidence = str(item.get("evidence_quote_or_fragment") or "")
        aliases_raw = item.get("aliases") or []
        aliases = tuple(str(alias) for alias in aliases_raw if isinstance(alias, str))
        source_ids_raw = item.get("source_ids") or []
        source_ids = tuple(str(source_id) for source_id in source_ids_raw if isinstance(source_id, str))
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if name and entity_type:
            candidates.append(
                SemanticCandidate(
                    canonical_name=name,
                    entity_type=entity_type,
                    evidence_quote_or_fragment=evidence,
                    confidence=confidence,
                    aliases=aliases,
                    source_ids=source_ids,
                )
            )
    status = "success" if candidates else "empty_result"
    return SemanticExtractionResult(candidates=candidates, status=status)


def _coerce_semantic_extraction_result(
    value: list[SemanticCandidate] | SemanticExtractionResult,
) -> SemanticExtractionResult:
    if isinstance(value, SemanticExtractionResult):
        return value
    candidates = list(value or [])
    return SemanticExtractionResult(
        candidates=candidates,
        status="success" if candidates else "empty_result",
    )


def _direct_destination_candidate(claim: ResearchClaim, *, origin: str = "") -> str:
    if claim.subject and normalize_for_matching(claim.subject) not in {"trip", "viagem", "travel", "user"}:
        candidate = _normalize_destination_name(claim.subject)
        candidate = _canonical_destination_identity(candidate)
        if _is_valid_destination_candidate(candidate, origin=origin) and _candidate_grounded(candidate, claim):
            return candidate
    return ""


def _direct_place_candidate(claim: ResearchClaim) -> str:
    candidate = _normalize_place_name(claim.value)
    if _is_valid_place_candidate(candidate) and _candidate_grounded(candidate, claim):
        return candidate
    return ""


def _validated_semantic_candidates(
    request: ResearchRequest,
    claim: ResearchClaim,
    *,
    extraction_goal: str,
    allowed_types: set[str],
    semantic_extractor: SemanticExtractor | None,
    origin: str = "",
) -> tuple[list[str], int, int, str, str]:
    if semantic_extractor is None:
        return [], 0, 0, "not_configured", "semantic_extractor_unavailable"
    evidence = _clean_evidence_text(
        " ".join(part for part in (claim.value, claim.evidence) if part),
        claim.source.title,
    )
    if not evidence:
        return [], 0, 0, "not_requested", "empty_evidence"
    _semex_trace(
        "SEMEX_REQUEST_CREATED",
        task_id=request.task_id,
        predicate=claim.predicate,
        goal=extraction_goal,
        extractor=type(semantic_extractor).__name__,
        evidence_chars=len(evidence),
    )
    status = "call_failed"
    failure_reason = ""
    before_count = 0
    after_count = 0
    try:
        _semex_trace("SEMEX_REQUEST_BEGIN", task_id=request.task_id, predicate=claim.predicate, goal=extraction_goal)
        _semex_trace("SEMEX_CALL_STARTED", task_id=request.task_id, predicate=claim.predicate, goal=extraction_goal)
        extraction_result = _coerce_semantic_extraction_result(semantic_extractor.extract(
            SemanticExtractionRequest(
                task_type=request.task_type or request.task_id,
                extraction_goal=extraction_goal,
                known_context={
                    "criteria": list(request.criteria),
                    "questions": [question.question for question in request.questions],
                    "query_plan": list(request.query_plan),
                },
                excluded_entities=[origin] if origin else [],
                evidence=evidence,
                source_title=claim.source.title,
            )
        ))
    except Exception as exc:
        _semex_trace("SEMEX_CALL_RETURNED", task_id=request.task_id, status="call_failed", candidates=0)
        _semex_trace(
            "SEMEX_REQUEST_END",
            task_id=request.task_id,
            status="call_failed",
            before=0,
            after=0,
            failure_reason=type(exc).__name__,
        )
        return [], 1, 0, "call_failed", type(exc).__name__
    _semex_trace(
        "SEMEX_CALL_RETURNED",
        task_id=request.task_id,
        status=extraction_result.status,
        candidates=len(extraction_result.candidates),
    )
    _semex_trace(
        "SEMEX_PARSE_RESULT",
        task_id=request.task_id,
        status=extraction_result.status,
        candidates=len(extraction_result.candidates),
    )
    raw_candidates = extraction_result.candidates
    before_count = len(raw_candidates)
    accepted: list[str] = []
    rejection_reasons: list[str] = []
    for candidate in raw_candidates:
        normalized_type = normalize_for_matching(candidate.entity_type)
        if normalized_type not in allowed_types:
            rejection_reasons.append("unsupported_entity_type")
            continue
        name = _normalize_place_name(candidate.canonical_name) if extraction_goal == "stockholm_places" else _normalize_destination_name(candidate.canonical_name)
        if extraction_goal == "destination_discovery":
            name = _canonical_destination_identity(name)
        candidate = SemanticCandidate(name, normalized_type, candidate.evidence_quote_or_fragment, candidate.confidence, candidate.aliases)
        rejection_reason = _semantic_candidate_rejection_reason(candidate, claim, extraction_goal=extraction_goal, origin=origin)
        if not rejection_reason:
            accepted.append(candidate.canonical_name)
        else:
            rejection_reasons.append(rejection_reason)
    if accepted:
        status = "success"
    elif extraction_result.status in {"call_failed", "invalid_json", "empty_result"}:
        status = extraction_result.status
    elif raw_candidates:
        status = "all_rejected"
    else:
        status = "empty_result"
    failure_reason = extraction_result.failure_reason
    if not failure_reason and status == "all_rejected":
        failure_reason = "; ".join(_dedupe(rejection_reasons))
    accepted_names = _dedupe(accepted)
    after_count = len(accepted_names)
    _semex_trace(
        "SEMEX_VALIDATION_RESULT",
        task_id=request.task_id,
        predicate=claim.predicate,
        status=status,
        before=before_count,
        after=after_count,
    )
    _semex_trace(
        "SEMEX_REQUEST_END",
        task_id=request.task_id,
        status=status,
        before=before_count,
        after=after_count,
        failure_reason=failure_reason,
    )
    return accepted_names, 1, before_count, status, failure_reason


def _semantic_candidate_is_valid(
    candidate: SemanticCandidate,
    claim: ResearchClaim,
    *,
    extraction_goal: str,
    origin: str = "",
) -> bool:
    return not _semantic_candidate_rejection_reason(candidate, claim, extraction_goal=extraction_goal, origin=origin)


def _semantic_candidate_rejection_reason(
    candidate: SemanticCandidate,
    claim: ResearchClaim,
    *,
    extraction_goal: str,
    origin: str = "",
) -> str:
    name = candidate.canonical_name
    if candidate.confidence and candidate.confidence < 0.45:
        return "low_confidence"
    if extraction_goal == "destination_discovery":
        if not _is_valid_destination_candidate(name, origin=origin):
            return "invalid_destination_candidate"
    elif not _is_valid_place_candidate(name):
        return "invalid_place_candidate"
    normalized = normalize_for_matching(name)
    if normalized in _INVALID_SEMANTIC_ENTITY_TERMS:
        return "invalid_generic_term"
    if _looks_like_person_or_blogger(name, claim):
        return "person_or_blogger"
    if not _candidate_grounded(name, claim, candidate.aliases, candidate.evidence_quote_or_fragment):
        return "not_grounded_in_evidence"
    return ""


def _candidate_grounded(
    name: str,
    claim: ResearchClaim,
    aliases: tuple[str, ...] = (),
    fragment: str = "",
) -> bool:
    haystack = normalize_for_matching(" ".join([claim.value, claim.evidence]))
    fragment_norm = normalize_for_matching(fragment)
    variants = [name, *aliases]
    if normalize_for_matching(name) == "vasa museum":
        variants.extend(["Vasa Museet"])
    for variant in variants:
        variant_norm = normalize_for_matching(_clean_entity_name(variant))
        if variant_norm and (variant_norm in haystack or (fragment_norm in haystack and variant_norm in fragment_norm)):
            return True
    return False


_INVALID_SEMANTIC_ENTITY_TERMS = {
    "uncover",
    "roman empire",
    "roman",
    "empire",
    "northern portuguese",
    "cheap",
    "city",
    "breaks",
    "city breaks",
    "jeanne",
    "learning",
    "breathe",
    "finding",
    "perfect",
    "vasa",
    "swedish",
    "science",
    "museums",
    "tourist",
    "tuesday",
    "search",
    "welcome",
    "spring",
    "october",
    "shop",
    "booking",
    "monday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "november",
    "december",
}


def _looks_like_person_or_blogger(name: str, claim: ResearchClaim) -> bool:
    normalized = normalize_for_matching(name)
    haystack = normalize_for_matching(" ".join([claim.source.title, claim.evidence]))
    if normalized in {"jeanne", "rick steves", "lonely planet"}:
        return True
    return bool(normalized and re.search(rf"\bby\s+{re.escape(normalized)}\b", haystack))


def _extract_destination_candidates(claim: ResearchClaim, *, origin: str = "") -> list[str]:
    text = " ".join(
        part
        for part in (
            claim.subject,
            claim.value,
            claim.evidence,
        )
        if part
    )
    if not _has_destination_context(text):
        return []
    candidates: list[str] = []
    for raw in _entity_spans(text):
        candidate = _normalize_destination_name(raw)
        candidate = _canonical_destination_identity(candidate)
        if _is_valid_destination_candidate(candidate, origin=origin):
            candidates.append(candidate)
    return _dedupe(candidates)


def _extract_place_candidates(claim: ResearchClaim) -> list[str]:
    text = " ".join(part for part in (claim.value, claim.evidence) if part)
    candidates: list[str] = []
    patterns = (
        r"\bisland of\s+([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ' -]{2,45}?)(?=,|\.|\s+home\b|\s+with\b|\s+and\b|$)",
        r"\bfamous\s+([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ' -]{2,45}?)(?=,|\.|\s+is\b|\s+with\b|\s+and\b|$)",
        r"\b(?:Opening hours|Horários?)\s*[-:]\s*([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ' -]{2,45}?)(?=,|\.|\s+-\s+|$)",
        r"\b(?:visit|visitar|ver|explorar|explore)\s+(?:the\s+)?([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ' -]{2,45}?)(?=,|\.|\s+with\b|\s+and\b|$)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            candidate = _normalize_place_name(match.group(1))
            if _is_valid_place_candidate(candidate):
                candidates.append(candidate)
    for raw in _entity_spans(text):
        candidate = _normalize_place_name(raw)
        if _is_valid_place_candidate(candidate):
            candidates.append(candidate)
    return _dedupe(candidates)


def _extract_transport_claim(claim: ResearchClaim) -> str:
    text = normalize_for_matching(" ".join([claim.value, claim.evidence]))
    if contains_any_phrase(text, ("transit fee", "airport train", "arlanda")):
        return "Pode existir uma taxa adicional no percurso ferroviário de/para Arlanda."
    if contains_any_phrase(text, ("metro", "bus", "autocarro", "tram", "comboio", "commuter rail", "public transport", "transportes publicos", "transportes públicos")):
        if "sl" in text:
            return "Rede SL cobre metro, autocarros e comboios suburbanos."
        return "Rede de transportes públicos relevante para deslocações locais."
    return ""


def _has_destination_context(text: str) -> bool:
    normalized = normalize_for_matching(text)
    return contains_any_phrase(
        normalized,
        (
            "destination",
            "destino",
            "city break",
            "city breaks",
            "from porto",
            "from lisbon",
            "from madrid",
            "from paris",
            "voo curto",
            "short flight",
            "short flights",
            "beach",
            "praia",
            "historic",
            "histórico",
            "cultural",
            "sightseeing",
            "holiday",
            "férias",
            "ferias",
        ),
    )


def _entity_spans(text: str) -> list[str]:
    spans: list[str] = []
    pattern = r"\b[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’.-]*(?:\s+[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’.-]*){0,3}"
    for match in re.finditer(pattern, text):
        raw = match.group(0).strip(" .,:;!?()[]{}\"'")
        if raw:
            spans.append(raw)
            parts = raw.split()
            if len(parts) > 1 and not any(normalize_for_matching(raw).endswith(suffix) for suffix in ("museum", "museet", "palace", "nationalmuseum")):
                spans.extend(part.strip(" .,:;!?()[]{}\"'") for part in parts if len(part) >= 3)
    return spans


def _normalize_destination_name(value: str) -> str:
    return _clean_entity_name(value)


def _canonical_destination_identity(value: str) -> str:
    name = _clean_entity_name(value)
    if "|" in name:
        parts = [_clean_entity_name(part) for part in name.split("|") if _clean_entity_name(part)]
        name = parts[-1] if parts else name
    name = re.sub(
        r"\b(?:city\s+breaks?|holiday(?:s)?|package(?:s)?|travel\s+guide|itinerary|things\s+to\s+do|guide|tourism|tourist\s+guide)\b",
        " ",
        name,
        flags=re.IGNORECASE,
    )
    name = re.sub(r"\s+", " ", name).strip(" -,:;.!?\"'()[]{}")
    return _clean_entity_name(name)


def _normalize_place_name(value: str) -> str:
    name = _clean_entity_name(value)
    if normalize_for_matching(name) == "vasa museet":
        return "Vasa Museum"
    return name


def _clean_entity_name(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" -,:;.!?\"'()[]{}")
    text = re.sub(r"['’]s\b", "", text)
    text = re.sub(r"\b(?:home|with|where|including|and|or|near|for|from|to|of|the|a|an)$", "", text, flags=re.IGNORECASE).strip()
    words = text.split()
    if len(words) >= 2 and normalize_for_matching(words[-1]) == normalize_for_matching(words[-2]):
        text = " ".join(words[:-1])
    return text


def _is_valid_destination_candidate(value: str, *, origin: str = "") -> bool:
    if not _is_entity_shaped(value):
        return False
    normalized = normalize_for_matching(value)
    if normalized.split(" ", 1)[0] in {"while", "from", "near", "visit", "best", "top", "explore"}:
        return False
    if len(value.split()) > 2:
        return False
    if origin and normalized == normalize_for_matching(origin):
        return False
    if normalized in {
        "trip",
        "viagem",
        "travel",
        "europe",
        "europa",
        "european",
        "city break",
        "city breaks",
        "portugal",
        "specified origin",
        "porto prose",
        "fonte",
        "dados",
    }:
        return False
    return not _looks_like_article_title(value)


def _is_valid_place_candidate(value: str) -> bool:
    if not _is_entity_shaped(value):
        return False
    normalized = normalize_for_matching(value)
    if normalized in {"stockholm", "estocolmo", "visit stockholm", "science museums", "categories", "opening hours", "what is included", "explore", "passing", "opening", "visit", "visitar", "fonte", "dados"}:
        return False
    if contains_any_phrase(normalized, ("what is included", "in addition", "buying tickets", "categories", "road freight", "cooperation protocol")):
        return False
    return not _looks_like_article_title(value)


def _is_entity_shaped(value: str) -> bool:
    text = str(value or "").strip()
    if not text or len(text) > 55:
        return False
    if text.startswith("#") or "\n" in text or ":" in text or ">" in text or "..." in text:
        return False
    if len(text.split()) > 4:
        return False
    normalized = normalize_for_matching(text)
    if contains_any_phrase(normalized, ("http", "snippet", "description", "search result", "subscribe", "cookie")):
        return False
    if re.search(r"\b(?:is|are|was|were|tem|tens|fica|inclui|includes|home to|passing by|explore)\b", normalized):
        return False
    return bool(re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", text))


def _destination_fit_value(candidate: str, claim: ResearchClaim) -> str:
    evidence = _clean_evidence_text(claim.evidence, claim.source.title)
    normalized = normalize_for_matching(evidence)
    details: list[str] = []
    if contains_any_phrase(normalized, ("voo curto", "short flight", "short flights", "direct flight", "direct flights")):
        details.append("acessível por voo curto")
    if contains_any_phrase(normalized, ("beach", "praia", "descanso", "rest")):
        details.append("bom equilíbrio entre descanso e passeio")
    if contains_any_phrase(normalized, ("historic", "histórico", "historico", "cultural", "sightseeing", "monument")):
        details.append("com visitas culturais")
    return ", ".join(_dedupe(details)[:2]) or _short_semantic_text(claim.value)


def _looks_like_article_title(text: str) -> bool:
    normalized = normalize_for_matching(text)
    if re.search(r"\b\d+\s+(?:dias|days|day|coisas|things|destinos|places|itinerary|roteiro)\b", normalized):
        return True
    return contains_any_phrase(
        normalized,
        (
            "best places",
            "top places",
            "best european city breaks",
            "city breaks",
            "things to do",
            "day itinerary",
            "travel guide",
            "guia de viagem",
            "roteiro",
            "melhores destinos",
            "lugares a visitar",
        ),
    )


def _clean_display_text(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text.strip(" \t\r\n\"'")


def _source_quality_label(item: dict[str, Any]) -> str:
    explicit = str(item.get("source_quality") or "").strip()
    kind = _source_kind(item)
    if kind in {"community", "social"}:
        return kind
    if explicit and explicit not in {"unknown", "web"}:
        return explicit
    if kind == "independent_test":
        return "independent_test"
    if kind == "specialist_media":
        return "independent_review"
    if kind == "official":
        return "official"
    return explicit or "unknown"


def _source_kind(item: dict[str, Any] | ResearchSourceRecord) -> str:
    if isinstance(item, ResearchSourceRecord):
        title = item.title
        domain = item.domain or urlparse(item.url).netloc
        quality = item.source_quality
    else:
        title = str(item.get("title") or "")
        url = str(item.get("url") or "")
        domain = str(item.get("domain") or "") or urlparse(url).netloc
        quality = str(item.get("source_quality") or "")
    haystack = normalize_for_matching(" ".join([domain, title, quality]))
    if any(token in haystack for token in ("reddit", "forum", "community", "comunidade", "discussions.apple.com", "community.samsung.com")):
        return "community"
    if any(token in haystack for token in ("dxomark", "rtings", "rting")):
        return "independent_test"
    if any(token in haystack for token in ("dpreview", "gsmarena", "notebookcheck", "the verge", "android authority", "tom's guide", "tom s guide", "trusted reviews")):
        return "specialist_media"
    if any(token in haystack for token in ("samsung.com", "apple.com", "google.com", "microsoft.com", "official")):
        return "official"
    if any(token in haystack for token in ("tiktok", "instagram", "facebook", "x.com", "twitter")):
        return "social"
    if "youtube" in haystack or "youtu.be" in haystack:
        return "video"
    if "independent_review" in haystack or "review" in haystack:
        return "specialist_media"
    return "general_web"


def _source_strength_for_claim(predicate: str, source: ResearchSourceRecord, evidence: str) -> str:
    kind = source.source_kind or _source_kind(source)
    if predicate in {"price", "opening_hours", "availability"}:
        if kind == "official":
            return "strong"
        if kind in {"specialist_media", "independent_test"}:
            return "useful"
    if predicate in {"camera_quality", "battery", "support_quality"}:
        if kind in {"independent_test", "specialist_media"}:
            return "strong"
        if kind == "official":
            return "useful"
        if kind == "video":
            return "useful" if _looks_like_test_evidence(evidence) else "weak"
        if kind == "community":
            return "useful" if predicate == "support_quality" else "weak"
        if kind == "social":
            return "weak"
    if kind in {"independent_test", "specialist_media", "official"}:
        return "useful"
    if kind in {"community", "video"}:
        return "weak"
    return "unknown"


def _looks_like_test_evidence(evidence: str) -> bool:
    text = normalize_for_matching(evidence)
    return contains_any_phrase(
        text,
        (
            "test",
            "teste",
            "low light",
            "baixa luz",
            "pouca luz",
            "4k",
            "fps",
            "autofocus",
            "estabilizacao",
            "estabilização",
            "dynamic range",
            "alcance dinamico",
            "alcance dinâmico",
            "comparacao",
            "comparação",
        ),
    )


def _has_substantive_evidence(predicate: str, claim: ResearchClaim) -> bool:
    evidence = _clean_evidence_text(claim.evidence, claim.source.title)
    text = normalize_for_matching(evidence)
    if not text:
        return False
    if contains_any_phrase(
        text,
        (
            "having trouble playing this video",
            "speech not recognized",
            "sign in to confirm",
            "login",
            "cookie",
            "subscribers",
            "likes",
        ),
    ):
        return False
    title_norm = normalize_for_matching(claim.source.title)
    if text and text == title_norm:
        return False
    if predicate in {"camera_quality", "battery", "support_quality"}:
        return len(text) >= 10 and (
            contains_any_phrase(text, ("camara", "câmara", "camera", "video", "vídeo", "foto", "battery", "bateria", "autonomia", "support", "suporte"))
            or _looks_like_test_evidence(text)
            or contains_any_phrase(text, ("analise", "análise", "review", "observacao", "observação"))
            or contains_any_phrase(text, ("24h", "24 h", "horario", "horário", "resposta lenta", "avaliacoes", "avaliações"))
        )
    return len(text) >= 6


def _clean_evidence_text(value: str, title: str = "") -> str:
    raw = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    noise_patterns = (
        r"^\s*(?:description|descrição|search|login|log in|sign in|cookie|cookies|subscribe|subscribers|likes?|views?|share|save|following|for you)\b",
        r"\b(?:we're having trouble playing this video|speech not recognized|enable javascript|accept cookies)\b",
        r"^\s*#\w+",
    )
    lines: list[str] = []
    seen_title = False
    title_norm = normalize_for_matching(title)
    for line in raw.split("\n"):
        clean = re.sub(r"\s+", " ", line).strip()
        if not clean:
            continue
        clean_norm = normalize_for_matching(clean)
        if title_norm and clean_norm == title_norm:
            if seen_title:
                continue
            seen_title = True
        if any(re.search(pattern, clean, flags=re.IGNORECASE) for pattern in noise_patterns):
            continue
        if re.fullmatch(r"[\d,.]+\s*[kmb]?\s*(?:subscribers|likes|views?)", clean, flags=re.IGNORECASE):
            continue
        lines.append(clean)
    text = " ".join(lines) if lines else re.sub(r"\s+", " ", raw).strip()
    text = re.sub(r"\b\d[\d,.]*\s*[kmb]?\s*(?:subscribers|likes|views?)\b", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip(" -|")


def _predicate_label(predicate: str) -> str:
    labels = {
        "camera_quality": "câmara",
        "battery": "bateria",
        "support_quality": "suporte",
        "opening_hours": "horários",
        "transport": "transportes",
        "price": "preço",
    }
    return labels.get(predicate, predicate.replace("_", " "))


def _claim_dimension(entry: KnowledgeEntry) -> str:
    metadata = getattr(entry, "metadata", {}) or {}
    explicit = str(metadata.get("claim_dimension") or "").strip()
    if explicit:
        return explicit
    value = normalize_for_matching(entry.value)
    if re.search(r"\b\d+(?:[,.]\d+)?\s*h\b", value):
        return "duration_hours"
    if re.search(r"\b\d+(?:[,.]\d+)?\s*(?:eur|euros|€)\b", value):
        return "money"
    if contains_any_phrase(value, ("open", "aberto", "closed", "fechado")):
        return "availability"
    return "qualitative"


def _claim_can_conflict(entry: KnowledgeEntry) -> bool:
    metadata = getattr(entry, "metadata", {}) or {}
    if str(metadata.get("relevance") or "") == "irrelevant":
        return False
    if str(metadata.get("applicability") or "") in {"family_related", "adjacent"}:
        return False
    if entry.predicate in {"price", "cost"} and str(metadata.get("semantic_role") or "") == "destination":
        return False
    return entry.predicate in {"price", "opening_hours", "availability", "transport", "battery"}


def _values_are_conflicting(values: set[str]) -> bool:
    normalized = {normalize_for_matching(value) for value in values if value}
    if len(normalized) < 2:
        return False
    joined = " ".join(normalized)
    if contains_any_phrase(joined, ("open", "aberto")) and contains_any_phrase(joined, ("closed", "fechado")):
        return True
    money_values = {
        match.group(1).replace(",", ".")
        for value in normalized
        for match in re.finditer(r"\b(\d{2,5}(?:[,.]\d+)?)\s*(?:eur|euros|€)\b", value)
    }
    if len(money_values) > 1:
        return True
    return False


def _short_semantic_text(value: str) -> str:
    text = _clean_display_text(value)
    if len(text) <= 90:
        return text
    return text[:87].rsplit(" ", 1)[0].strip() + "..."


def _adjusted_confidence(confidence: float, relevance: str, applicability: str, source_strength: str = "unknown") -> float:
    value = max(0.0, min(1.0, float(confidence or 0.0)))
    if relevance == "irrelevant":
        return min(value, 0.2)
    if applicability == "exact":
        if source_strength == "strong":
            return min(0.95, max(value, 0.72))
        if source_strength == "useful":
            return min(value, 0.7)
        if source_strength == "weak":
            return min(value, 0.45)
        return min(value, 0.62)
    if applicability == "family_related":
        return min(value, 0.45)
    return min(value, 0.35)


def _result_for_claim(results: list[dict[str, Any]], claim: ResearchClaim) -> dict[str, Any]:
    for result in results:
        if result.get("url") == claim.source.url and result.get("title") == claim.source.title:
            return dict(result)
    return {"title": claim.source.title, "url": claim.source.url, "snippet": claim.evidence, "source_quality": claim.source.source_quality}


def _annotate_rejected_result(result: dict[str, Any], claim: ResearchClaim) -> dict[str, Any]:
    annotated = dict(result)
    annotated["relevance"] = claim.relevance
    annotated["applicability"] = claim.applicability
    annotated["rejection_reason"] = claim.rejection_reason or claim.metadata.get("relevance_reason", "")
    annotated["source_kind"] = claim.metadata.get("source_kind", "")
    annotated["source_strength_for_claim"] = claim.metadata.get("source_strength_for_claim", "")
    annotated["claimed_subject"] = claim.metadata.get("claimed_subject", claim.subject)
    return annotated


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
            "source_kind": claim.source.source_kind,
            "source_strength_for_claim": claim.source.source_strength_for_claim,
            "evidence_quality": claim.source.source_strength_for_claim,
            "epistemic_type": claim.epistemic_type,
            "provider": claim.metadata.get("provider", ""),
            "fallback_reason": claim.metadata.get("fallback_reason", ""),
            "score": claim.metadata.get("score"),
            "applicability": claim.applicability,
            "relevance": claim.relevance,
            "raw_evidence": claim.metadata.get("raw_evidence", ""),
            "claimed_subject": claim.metadata.get("claimed_subject", claim.subject),
            "semantic_role": claim.metadata.get("semantic_role", ""),
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


def _explicit_research_option_targets(options: list[str], message: str) -> list[str]:
    text = normalize_for_matching(message)
    if contains_any_phrase(text, ("ambos", "os dois", "as duas", "todos", "todas", "comparar")):
        return []
    matches: list[str] = []
    for option in options:
        option_norm = normalize_for_matching(option)
        option_tokens = [token for token in re.findall(r"[a-z0-9]+", option_norm) if len(token) >= 4]
        if option_norm and option_norm in text:
            matches.append(option)
            continue
        # A unique strong brand/entity token is enough for "câmara do iPhone"
        # to target "iPhone 15", but only when it does not collide with
        # another option in the same comparison.
        unique_tokens = [
            token
            for token in option_tokens
            if sum(1 for other in options if token in normalize_for_matching(other)) == 1
        ]
        if unique_tokens and any(re.search(rf"\b{re.escape(token)}\b", text) for token in unique_tokens):
            matches.append(option)
    return _dedupe(matches)


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
    return f"{option} base model {labels.get(criterion, criterion)}".strip()


def _question_for_criterion(option: str, criterion: str) -> str:
    labels = {
        "camera_quality": "qualidade real da câmara",
        "battery": "autonomia real e carregamento",
        "support_quality": "qualidade do suporte",
        "price": "preço atual",
    }
    return f"{option}: {labels.get(criterion, criterion)}"


def _research_conflict_sentence(entries: list[KnowledgeEntry]) -> str:
    grouped: dict[tuple[str, str, str], set[str]] = {}
    for entry in entries:
        if not _claim_can_conflict(entry):
            continue
        grouped.setdefault((entry.subject, entry.predicate, _claim_dimension(entry)), set()).add(entry.value)
    conflicts = [
        f"{subject} tem dados incompatíveis sobre {_predicate_label(predicate)}"
        for (subject, predicate, _dimension), values in grouped.items()
        if _values_are_conflicting(values)
    ]
    if not conflicts:
        return ""
    return "Há conflito nas fontes: " + "; ".join(conflicts[:2]) + "."


def _semantic_extraction_status(calls: int, before_count: int, after_count: int, statuses: list[str] | None = None) -> str:
    statuses = [status for status in (statuses or []) if status and status != "not_requested"]
    if calls <= 0:
        if "not_configured" in statuses:
            return "not_configured"
        return "not_requested"
    if after_count > 0:
        return "success"
    for status in ("call_failed", "invalid_json"):
        if status in statuses:
            return status
    if before_count > 0:
        return "all_rejected"
    if "not_configured" in statuses:
        return "not_configured"
    return "empty_result"


def _ingest_relevant_source_results(
    work_session,
    request: ResearchRequest,
    raw_results: list[dict[str, Any]],
    entries: list[KnowledgeEntry],
    rejected_results: list[dict[str, Any]],
) -> None:
    accepted_urls = {entry.source.reference for entry in entries if entry.source.reference}
    rejected_by_url = {str(item.get("url") or ""): item for item in rejected_results}
    for result in raw_results:
        url = str(result.get("url") or "")
        if not url or url in accepted_urls:
            continue
        relevant, reason = _result_relevance_for_request(request, result)
        if not relevant:
            continue
        rejected = rejected_by_url.get(url, {})
        title = str(result.get("title") or "Fonte web")
        snippet = _clean_evidence_text(str(result.get("snippet") or ""), title)
        source = KnowledgeSource(
            source_type="web",
            reference=url,
            original_text=snippet or title,
            created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        )
        entry = _upsert_entry(
            work_session,
            request.task_id,
            "source",
            _source_subject_for_request(request, result),
            "research_evidence",
            title,
            snippet or title,
            request.id,
            replace_active=False,
            confidence=0.55,
            source=source,
            metadata={
                "applicability": "source_only",
                "relevance": "relevant",
                "relevance_reason": reason,
                "semantic_role": _source_semantic_role_for_request(request),
                "source_kind": str(result.get("source_kind") or ""),
                "source_strength_for_claim": "source_only",
                "title": title,
                "domain": str(result.get("domain") or ""),
                "query": str(result.get("query") or ""),
                "semantic_extraction": "no_semantic_entity",
            },
        )
        if entry is not None:
            entries.append(entry)
            accepted_urls.add(url)


def _source_subject_for_request(request: ResearchRequest, result: dict[str, Any]) -> str:
    for question in request.questions:
        if question.semantic_role == "destination":
            return question.subject
    return "pesquisa"


def _source_semantic_role_for_request(request: ResearchRequest) -> str:
    if not request.questions:
        return ""
    roles = _dedupe([question.semantic_role for question in request.questions if question.semantic_role])
    return roles[0] if roles else ""


def _result_relevance_for_request(request: ResearchRequest, result: dict[str, Any]) -> tuple[bool, str]:
    text = normalize_for_matching(
        " ".join(
            str(result.get(key) or "")
            for key in ("title", "snippet", "domain", "query")
        )
    )
    roles = {question.semantic_role for question in request.questions}
    predicates = {question.predicate for question in request.questions}
    if "destination_discovery" in roles:
        if contains_any_phrase(text, ("city break", "city breaks", "destination", "destino", "travel", "holiday", "férias", "ferias")):
            return True, "travel_discovery_source"
    if "destination" in roles and "activities" in predicates:
        destinations = []
        for question in request.questions:
            if question.semantic_role != "destination":
                continue
            destination = normalize_for_matching(question.subject)
            destinations.append(destination)
            if destination == "estocolmo":
                destinations.append("stockholm")
        if any(destination and destination in text for destination in destinations) and contains_any_phrase(
            text,
            (
                "things to do",
                "what to do",
                "attraction",
                "attractions",
                "museum",
                "museums",
                "museu",
                "museus",
                "visit",
                "visitar",
                "activities",
                "atividades",
            ),
        ):
            return True, "destination_activity_source"
    return False, ""


def _store_research_sources(
    work_session,
    request: ResearchRequest,
    entries: list[KnowledgeEntry],
    raw_results: list[dict[str, Any]],
    rejected_results: list[dict[str, Any]],
) -> None:
    bucket = work_session.knowledge.setdefault("research_sources", [])
    if not isinstance(bucket, list):
        bucket = []
        work_session.knowledge["research_sources"] = bucket
    accepted_urls = {entry.source.reference for entry in entries if entry.source.reference}
    accepted_lookup = {entry.source.reference: entry for entry in entries if entry.source.reference}
    rejected_lookup = {item.get("url"): item for item in rejected_results if item.get("url")}
    for visible_order, result in enumerate(raw_results, start=1):
        url = str(result.get("url") or "")
        title = str(result.get("title") or "Fonte web")
        if not url:
            continue
        existing = next((item for item in bucket if isinstance(item, dict) and item.get("url") == url and item.get("task_id") == request.task_id), None)
        record = existing if existing is not None else {}
        source_id = _research_source_id(url, title)
        accepted = url in accepted_urls
        rejected = rejected_lookup.get(url)
        entry = accepted_lookup.get(url)
        record.update(
            {
                "id": source_id,
                "title": title,
                "url": url,
                "domain": str(result.get("domain") or ""),
                "task_id": request.task_id,
                "research_request_id": request.id,
                "status": "accepted" if accepted else "rejected",
                "source_quality": str(result.get("source_quality") or "unknown"),
                "source_kind": str(result.get("source_kind") or ""),
                "source_strength_for_claim": str((entry.metadata.get("source_strength_for_claim") if entry else "") or (rejected or {}).get("source_strength_for_claim") or "unknown"),
                "relevance": "relevant" if accepted else str((rejected or {}).get("relevance") or "unknown"),
                "relevance_role": _source_relevance_role(entry, rejected),
                "applicability": str((entry.metadata.get("applicability") if entry else "") or (rejected or {}).get("applicability") or "unknown"),
                "snippet": _clean_evidence_text(str(result.get("snippet") or ""), title),
                "visible_order": visible_order,
            }
        )
        if existing is None:
            bucket.append(record)


def research_source_records(work_session, task_id: str = "") -> list[dict[str, Any]]:
    records = work_session.knowledge.get("research_sources", [])
    if not isinstance(records, list):
        return []
    result = [dict(item) for item in records if isinstance(item, dict)]
    if task_id:
        result = [item for item in result if item.get("task_id") == task_id]
    return result


def resolve_research_source(work_session, task, message: str) -> dict[str, Any] | None:
    if task is None:
        return None
    sources = research_source_records(work_session, task.id)
    if not sources:
        return None
    matches = _matching_research_sources(sources, message)
    if matches:
        return matches[0]
    normalized = normalize_for_matching(message)
    if _explicit_source_reference_hint(message):
        return None
    if contains_any_phrase(normalized, ("essa fonte", "esta fonte", "a fonte", "isso", "artigo")):
        accepted = [item for item in sources if item.get("status") == "accepted"]
        return (accepted or sources)[-1]
    return None


def _matching_research_sources(sources: list[dict[str, Any]], message: str) -> list[dict[str, Any]]:
    text = normalize_for_matching(message)
    tokens = [token for token in _meaningful_tokens(text) if token not in {"abre", "abrir", "fonte", "fontes", "artigo", "link"}]
    if not tokens:
        return []
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for source in sources:
        title = str(source.get("title") or "")
        domain = str(source.get("domain") or "")
        url = str(source.get("url") or "")
        aliases = _source_aliases(title, domain, url)
        haystack = normalize_for_matching(" ".join([title, domain, url, " ".join(aliases)]))
        score = 0
        if any(alias and alias in text for alias in aliases):
            score = 100
        elif any(alias and text in alias and len(text) >= 4 for alias in aliases):
            score = 90
        elif all(token in haystack for token in tokens):
            score = 75
        elif len(tokens) > 1 and any(len(token) >= 5 and token in haystack for token in tokens):
            score = 55
        if score:
            scored.append((score, -int(source.get("visible_order") or 0), source))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [source for _score, _order, source in scored]


def _explicit_source_reference_hint(message: str) -> str:
    text = normalize_for_matching(message)
    text = re.sub(r"\b(?:abre|abrir|le|lê|le-me|lê-me|diz-me|diz|mostra|resume|fonte|fontes|artigo|link|da|do|das|dos|a|o|as|os|essa|esta|esse|este|isso|qual|quais|de|onde|veio|tiraste)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [token for token in re.findall(r"[a-z0-9]+", text) if len(token) >= 4]
    return " ".join(tokens)


def latest_research_summary(work_session, task_id: str = "") -> dict[str, Any]:
    records = work_session.knowledge.get("research_summaries", [])
    if not isinstance(records, list):
        return {}
    summaries = [item for item in records if isinstance(item, dict)]
    if task_id:
        summaries = [item for item in summaries if item.get("task_id") == task_id]
    return dict(summaries[-1]) if summaries else {}


def _store_research_summary(work_session, request: ResearchRequest, summary: str) -> None:
    bucket = work_session.knowledge.setdefault("research_summaries", [])
    if not isinstance(bucket, list):
        bucket = []
        work_session.knowledge["research_summaries"] = bucket
    bucket.append(
        {
            "task_id": request.task_id,
            "research_request_id": request.id,
            "summary": summary,
            "criteria": list(request.criteria),
            "query_plan": list(request.query_plan),
            "status": request.status,
            "created_at": request.created_at,
        }
    )


def _source_aliases(title: str, domain: str, url: str) -> list[str]:
    aliases = [title, domain, domain.replace("www.", "")]
    parsed = urlparse(url)
    if parsed.netloc:
        aliases.append(parsed.netloc)
        aliases.append(parsed.netloc.replace("www.", ""))
    for part in re.split(r"[^A-Za-z0-9]+", title):
        if len(part) >= 4 and not part.isdigit():
            aliases.append(part)
    return [normalize_for_matching(alias) for alias in aliases if alias]


def _source_relevance_role(entry: KnowledgeEntry | None, rejected: dict[str, Any] | None) -> str:
    if entry is not None:
        applicability = str(entry.metadata.get("applicability") or "")
        if applicability == "exact":
            return "evidência direta"
        return "evidência útil"
    if rejected:
        if rejected.get("applicability") in {"family_related", "adjacent"}:
            return "relacionada, não direta"
        if rejected.get("relevance") == "irrelevant":
            return "descartada"
    return "fonte analisada"


def _compact_results_for_ui(
    raw_results: list[dict[str, Any]],
    entries: list[KnowledgeEntry],
    rejected_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    accepted_urls = {entry.source.reference for entry in entries if entry.source.reference}
    accepted_by_url = {entry.source.reference: entry for entry in entries if entry.source.reference}
    rejected_by_url = {str(item.get("url") or ""): item for item in rejected_results}
    compact: list[dict[str, Any]] = []
    for result in raw_results:
        url = str(result.get("url") or "")
        rejected = rejected_by_url.get(url, {})
        entry = accepted_by_url.get(url)
        compact.append(
            {
                "id": _research_source_id(url, str(result.get("title") or "Fonte web")),
                "title": str(result.get("title") or "Fonte web"),
                "url": url,
                "domain": str(result.get("domain") or ""),
                "source_quality": str(result.get("source_quality") or "unknown"),
                "source_kind": str(result.get("source_kind") or ""),
                "provider": str(result.get("provider") or ""),
                "status": "accepted" if url in accepted_urls else "rejected",
                "relevance": "relevant" if url in accepted_urls else str(rejected.get("relevance") or "unknown"),
                "applicability": str(rejected.get("applicability") or ("exact" if url in accepted_urls else "unknown")),
                "source_strength_for_claim": str((entry.metadata.get("source_strength_for_claim") if entry else "") or rejected.get("source_strength_for_claim") or "unknown"),
                "relevance_role": "evidência direta" if url in accepted_urls else _source_relevance_role(None, rejected),
                "snippet": "",
                "has_detail": bool(result.get("snippet")),
            }
        )
    return compact[:8]


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

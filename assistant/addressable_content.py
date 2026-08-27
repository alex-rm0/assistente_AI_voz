from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from assistant.task_intelligence import (
    answer_task_query,
    cognitive_task_view_model,
    infer_unknowns,
    task_knowledge,
    task_option_titles,
    task_reasoning,
)
from assistant.task_research import latest_research_summary, research_source_records
from assistant.text_matching import contains_any_phrase, normalize_for_matching


@dataclass(frozen=True)
class ContentOperation:
    kind: str
    target_type: str = ""
    target_hint: str = ""
    verbosity: str = "summary"
    raw_excerpt: bool = False
    filter_hint: str = ""


@dataclass(frozen=True)
class AddressableContent:
    id: str
    type: str
    task_id: str = ""
    parent_id: str = ""
    title: str = ""
    aliases: tuple[str, ...] = ()
    ordinal: int = 0
    visible: bool = True
    focused: bool = False
    source_reference: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContentResolveResult:
    target: AddressableContent | None
    ambiguous: bool = False
    candidates: tuple[AddressableContent, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class ContentPresentationResult:
    text: str
    presentation_mode: str = "both"
    target_id: str = ""
    continuation_token: str = ""
    provenance: tuple[str, ...] = ()
    ui_event: dict[str, Any] | None = None


_SOURCE_WORDS = ("fonte", "fontes", "artigo", "link", "resultado")
_TASK_WORDS = ("task", "tarefa")
_COMPARISON_WORDS = ("comparacao", "comparação")
_RESEARCH_WORDS = ("pesquisa", "resultado")
_DEICTIC_WORDS = ("isto", "isso", "esta", "essa", "este", "esse", "selecionado", "seleccionado")
_TOKEN_STOPWORDS = {
    "a",
    "as",
    "o",
    "os",
    "de",
    "da",
    "do",
    "das",
    "dos",
    "em",
    "para",
    "por",
    "que",
    "qual",
    "quais",
    "fonte",
    "fontes",
    "artigo",
    "link",
    "abre",
    "abrir",
    "le",
    "lê",
    "me",
    "diz",
    "diz-me",
    "resume",
    "mostra",
    "mostra-me",
    "essa",
    "esta",
    "esse",
    "este",
    "isso",
}


def detect_content_operation(message: str, *, has_previous_target: bool = False) -> ContentOperation | None:
    text = normalize_for_matching(message).strip(" .!?")
    if not text:
        return None
    if text.startswith("porque ") or text in {"porque", "porquê", "por que"}:
        return None

    verbosity = _verbosity_from_text(text)
    raw_excerpt = contains_any_phrase(
        text,
        ("le exatamente", "lê exatamente", "le exactamente", "lê exactamente", "exatamente o excerto", "exactamente o excerto", "mostra o excerto", "excerto original", "texto original"),
    )
    if raw_excerpt and has_previous_target:
        return ContentOperation("read", "previous", verbosity="full", raw_excerpt=True)
    if raw_excerpt:
        return ContentOperation("read", "source", verbosity="full", raw_excerpt=True)

    if contains_any_phrase(
        text,
        (
            "mostra as fontes",
            "mostra-me as fontes",
            "quero ver as fontes",
            "abre a lista de fontes",
            "quais foram as fontes",
            "que fontes",
            "fontes usaste",
        ),
    ):
        return ContentOperation("show", "source_list", verbosity=verbosity)

    if _is_open_source_command(text):
        return ContentOperation("open", "source", target_hint=_target_hint(message), verbosity=verbosity)

    section = _section_from_text(text)
    if section:
        return ContentOperation("read", "section", target_hint=section, verbosity=verbosity, raw_excerpt=raw_excerpt)

    if contains_any_phrase(text, ("o que encontraste", "resumo da pesquisa", "conclusao da pesquisa", "conclusão da pesquisa", "lê só o resultado", "le so o resultado")):
        return ContentOperation("read", "research_synthesis", verbosity=verbosity)

    ordinal = _ordinal_index(text)
    if ordinal and contains_any_phrase(text, _SOURCE_WORDS + ("segunda", "primeira", "terceira", "ultima", "última")):
        kind = "summarize" if _is_summarize_command(text) else "read"
        return ContentOperation(kind, "source", target_hint=_target_hint(message), verbosity=verbosity, raw_excerpt=raw_excerpt)

    if _is_summarize_command(text):
        target_type = _target_type_from_text(text, default="previous" if has_previous_target else "")
        if target_type:
            return ContentOperation("summarize", target_type, target_hint=_target_hint(message), verbosity=verbosity, raw_excerpt=raw_excerpt)

    if _is_read_command(text):
        target_type = _target_type_from_text(text, default="previous" if has_previous_target and _is_bare_followup(text) else "")
        if target_type:
            return ContentOperation("read", target_type, target_hint=_target_hint(message), verbosity=verbosity, raw_excerpt=raw_excerpt)
        hint = _target_hint(message)
        if hint:
            return ContentOperation("read", "source", target_hint=hint, verbosity=verbosity, raw_excerpt=raw_excerpt)

    if has_previous_target and contains_any_phrase(text, ("mais detalhe", "com mais detalhe", "continua", "repete", "agora a segunda", "agora a terceira", "agora a primeira", "abre")):
        kind = "open" if text in {"abre", "abrir"} else ("summarize" if _is_summarize_command(text) else "read")
        return ContentOperation(kind, "previous", target_hint=_target_hint(message), verbosity="detailed" if "detalhe" in text or "continua" in text else verbosity)

    return None


def build_addressable_content(work_session, task) -> list[AddressableContent]:
    if task is None:
        return []
    targets: list[AddressableContent] = []
    title = getattr(task, "goal", "") or _task_type_label(getattr(task, "task_type", ""))
    targets.append(
        AddressableContent(
            id=f"task:{task.id}",
            type="task",
            task_id=task.id,
            title=title,
            aliases=(title, "esta task", "esta tarefa", "o que esta selecionado", "o que está selecionado"),
            focused=True,
            payload={"task": task},
        )
    )

    if getattr(task, "task_type", "") == "generic_comparison":
        targets.append(
            AddressableContent(
                id=f"comparison:{task.id}",
                type="comparison",
                task_id=task.id,
                title=title or "Comparação",
                aliases=("comparacao", "comparação", "esta comparacao", "esta comparação"),
                focused=True,
                payload={"task": task},
            )
        )

    presentation = cognitive_task_view_model(work_session, task, title=title)
    for section in presentation.get("sections", []):
        role = str(section.get("role") or "")
        section_title = str(section.get("title") or role)
        targets.append(
            AddressableContent(
                id=f"section:{task.id}:{role}",
                type="section",
                task_id=task.id,
                parent_id=f"task:{task.id}",
                title=section_title,
                aliases=tuple(_section_aliases(role, section_title)),
                payload={"section": section, "task": task},
            )
        )

    for index, option in enumerate(task_option_titles(work_session, task), start=1):
        targets.append(
            AddressableContent(
                id=f"option:{task.id}:{normalize_for_matching(option).replace(' ', '-')}",
                type="option",
                task_id=task.id,
                parent_id=f"task:{task.id}",
                title=option,
                aliases=(option,),
                ordinal=index,
                payload={"option": option},
            )
        )

    summary = latest_research_summary(work_session, task.id)
    if summary:
        targets.append(
            AddressableContent(
                id=f"research_synthesis:{task.id}:{summary.get('research_request_id', 'latest')}",
                type="research_synthesis",
                task_id=task.id,
                parent_id=f"task:{task.id}",
                title="Resumo da pesquisa",
                aliases=("pesquisa", "resumo da pesquisa", "resultado da pesquisa", "conclusao da pesquisa", "conclusão da pesquisa"),
                payload={"summary": summary},
            )
        )

    for index, source in enumerate(_ordered_user_sources(research_source_records(work_session, task.id)), start=1):
        title = str(source.get("title") or "Fonte web")
        domain = str(source.get("domain") or _domain(source.get("url", "")))
        targets.append(
            AddressableContent(
                id=str(source.get("id") or f"source:{task.id}:{index}"),
                type="source",
                task_id=task.id,
                parent_id=f"research_synthesis:{task.id}",
                title=title,
                aliases=tuple(_source_aliases(title, domain, str(source.get("url") or ""))),
                ordinal=index,
                source_reference=str(source.get("url") or ""),
                payload={"source": source},
            )
        )
    return targets


def resolve_content_target(
    targets: list[AddressableContent],
    operation: ContentOperation,
    message: str,
    *,
    previous_target_id: str = "",
) -> ContentResolveResult:
    if operation.target_type == "source_list":
        return ContentResolveResult(_virtual_source_list(targets), reason="source_list")
    if operation.target_type == "previous" and previous_target_id:
        previous = next((target for target in targets if target.id == previous_target_id), None)
        if previous is not None:
            hinted = _resolve_ordinal(targets, message, previous.type)
            return ContentResolveResult(hinted or previous, reason="previous")

    target_type = _effective_target_type(operation, message, previous_target_id)
    ordinal_target = _resolve_ordinal(targets, message, target_type)
    if ordinal_target is not None:
        return ContentResolveResult(ordinal_target, reason="ordinal")

    if operation.target_type == "section":
        target = next((item for item in targets if item.type == "section" and operation.target_hint in item.aliases), None)
        return ContentResolveResult(target, reason="section" if target else "not_found")

    if target_type in {"task", "comparison", "research_synthesis"}:
        target = _focused_target(targets, target_type)
        if target is not None:
            return ContentResolveResult(target, reason="active_task")

    if target_type == "source":
        if previous_target_id:
            previous = next((target for target in targets if target.id == previous_target_id and target.type == "source"), None)
            if previous is not None and not operation.target_hint and not _ordinal_index(normalize_for_matching(message)):
                return ContentResolveResult(previous, reason="previous_source")
        exact = _resolve_named_source(targets, operation.target_hint or message)
        if exact.ambiguous or exact.target is not None:
            return exact
        sources = [target for target in targets if target.type == "source" and str(target.payload.get("source", {}).get("status") or "") == "accepted"]
        if not sources:
            sources = [target for target in targets if target.type == "source"]
        if operation.raw_excerpt and sources:
            return ContentResolveResult(sources[-1], reason="latest_source")
        if _is_deictic(message) and previous_target_id:
            previous = next((target for target in targets if target.id == previous_target_id and target.type == "source"), None)
            if previous is not None:
                return ContentResolveResult(previous, reason="deictic_previous")
        if _is_deictic(message) and sources:
            return ContentResolveResult(sources[-1], reason="deictic_latest")
    return ContentResolveResult(None, reason="not_found")


def present_content(work_session, task, target: AddressableContent, operation: ContentOperation) -> ContentPresentationResult:
    if target.type == "source_list":
        sources = [item for item in build_addressable_content(work_session, task) if item.type == "source"]
        if not sources:
            return ContentPresentationResult("Ainda não tenho fontes ligadas a esta tarefa.", target_id=target.id)
        event_results = [dict(source.payload.get("source") or {}) for source in sources]
        analysed_count = len(event_results)
        used_count = len([source for source in event_results if source.get("status") == "accepted"])
        if used_count == analysed_count:
            text = f"Claro — usei {used_count} fonte" + ("s" if used_count != 1 else "") + " relevante" + ("s" if used_count != 1 else "") + " nesta pesquisa."
        else:
            text = (
                f"Analisei {analysed_count} resultado" + ("s" if analysed_count != 1 else "")
                + f" e usei {used_count} fonte" + ("s" if used_count != 1 else "")
                + " relevante" + ("s" if used_count != 1 else "") + "."
            )
        return ContentPresentationResult(
            text=text,
            target_id=target.id,
            continuation_token=target.id,
            provenance=("TASK_KNOWLEDGE", "WEB_SOURCE"),
            ui_event={"type": "research_results_ready", "payload": {"task_id": getattr(task, "id", ""), "status": "completed", "summary": text, "results": event_results}},
        )
    if target.type == "source":
        return _present_source(work_session, task, target, operation)
    if target.type == "research_synthesis":
        summary = target.payload.get("summary") or {}
        text = str(summary.get("summary") or "Ainda não tenho uma síntese útil desta pesquisa.")
        if operation.kind == "summarize":
            text = _first_sentence(text)
        return ContentPresentationResult(text, target_id=target.id, continuation_token=target.id, provenance=("TASK_KNOWLEDGE", "WEB_SOURCE"))
    if target.type == "section":
        return _present_section(target, operation)
    if target.type in {"task", "comparison"}:
        return _present_task(work_session, task, target, operation)
    if target.type == "option":
        option = str(target.payload.get("option") or target.title)
        entries = [entry for entry in task_knowledge(work_session, getattr(task, "id", "")) if entry.status == "active" and normalize_for_matching(entry.subject) == normalize_for_matching(option)]
        if not entries:
            text = f"Tenho {option} como opção, mas ainda não tenho detalhes suficientes sobre ela."
        else:
            text = f"Sobre {option}, tenho isto: " + "; ".join(_entry_phrase(entry) for entry in entries[:4]) + "."
        return ContentPresentationResult(text, target_id=target.id, continuation_token=target.id, provenance=("TASK_KNOWLEDGE",))
    return ContentPresentationResult("", target_id=target.id)


def source_url_for_target(target: AddressableContent) -> str:
    if target.type != "source":
        return ""
    return str(target.source_reference or target.payload.get("source", {}).get("url") or "")


def _present_source(work_session, task, target: AddressableContent, operation: ContentOperation) -> ContentPresentationResult:
    source = dict(target.payload.get("source") or {})
    title = str(source.get("title") or target.title or "Fonte")
    domain = str(source.get("domain") or _domain(source.get("url", "")))
    entries = [
        entry
        for entry in task_knowledge(work_session, getattr(task, "id", ""))
        if entry.status == "active" and entry.source.reference == source.get("url")
    ]
    snippet = str(source.get("snippet") or "")
    if operation.raw_excerpt:
        excerpt = _clean_source_excerpt(entries[-1].source.original_text if entries else snippet)
        text = excerpt or "Não tenho um excerto guardado para essa fonte."
        return ContentPresentationResult(text, target_id=target.id, continuation_token=target.id, provenance=("WEB_SOURCE",))
    if operation.kind == "show":
        url = str(source.get("url") or "")
        text = f"Veio de {title}" + (f" ({domain})" if domain else "") + (f": {url}" if url else ".")
        return ContentPresentationResult(text, target_id=target.id, continuation_token=target.id, provenance=("WEB_SOURCE",))
    if operation.kind == "summarize" or operation.verbosity == "brief":
        facts = [_natural_entry_phrase(entry, brief=True) for entry in entries[:2]]
        if facts:
            text = f"Em resumo, {title}" + (f" ({domain})" if domain else "") + " " + "; ".join(facts) + "."
        else:
            text = f"{title}" + (f" ({domain})" if domain else "") + " está relacionada com a pesquisa, mas ainda não tenho uma conclusão estruturada a partir dela."
    elif operation.verbosity == "detailed":
        facts = [_natural_entry_phrase(entry) for entry in entries[:4]]
        detail = "; ".join(facts) if facts else "não tenho factos estruturados aceites a partir dela"
        excerpt = _short_excerpt(_clean_source_excerpt(entries[-1].source.original_text if entries else snippet), 260)
        text = f"Com mais detalhe, {title}" + (f" ({domain})" if domain else "") + f" indica que {detail}."
        if excerpt:
            text += f" O ponto útil do excerto é: {excerpt}"
    else:
        facts = [_natural_entry_phrase(entry, brief=True) for entry in entries[:2]]
        if facts:
            text = f"{title}" + (f" ({domain})" if domain else "") + " " + "; ".join(facts) + "."
        elif snippet:
            text = f"Tenho a fonte {title}" + (f" ({domain})" if domain else "") + ", mas ainda sem factos estruturados aceites para ler em voz alta."
        else:
            text = f"Tenho a fonte {title}" + (f" ({domain})" if domain else "") + ", mas sem excerto útil guardado."
    return ContentPresentationResult(text, target_id=target.id, continuation_token=target.id, provenance=("WEB_SOURCE", "TASK_KNOWLEDGE"))


def _present_section(target: AddressableContent, operation: ContentOperation) -> ContentPresentationResult:
    section = target.payload.get("section") or {}
    title = str(section.get("title") or target.title)
    items = [str(item.get("label") or "") for item in section.get("items", []) if str(item.get("label") or "")]
    text_value = str(section.get("text") or "")
    if text_value:
        body = text_value
    elif items:
        body = _join_pt(items[:4]) if operation.verbosity != "full" else "; ".join(items)
    else:
        body = "não tenho nada estruturado nessa secção."
    if operation.kind == "summarize":
        return ContentPresentationResult(f"{title}: {body}", target_id=target.id, continuation_token=target.id, provenance=("TASK_KNOWLEDGE",))
    return ContentPresentationResult(f"{title}: {body}", target_id=target.id, continuation_token=target.id, provenance=("TASK_KNOWLEDGE",))


def _present_task(work_session, task, target: AddressableContent, operation: ContentOperation) -> ContentPresentationResult:
    if getattr(task, "task_type", "") == "generic_comparison":
        options = task_option_titles(work_session, task)
        reasoning = task_reasoning(work_session, task.id)
        unknowns = infer_unknowns(task, task_knowledge(work_session, task.id))
        if operation.kind == "summarize":
            if reasoning.current_recommendation:
                text = f"Estamos a comparar {_join_pt(options)}. Neste momento, {reasoning.current_recommendation} parece melhor, mas a conclusão ainda é provisória."
            else:
                text = f"Estamos a comparar {_join_pt(options)}. Ainda falta fechar {_join_pt(unknowns[:2]) if unknowns else 'a decisão final'}."
        else:
            parts = [f"Estamos a comparar {_join_pt(options)}."] if options else ["Estamos a construir uma comparação."]
            if reasoning.current_recommendation:
                parts.append(f"Neste momento, {reasoning.current_recommendation} parece melhor porque {'; '.join(reasoning.reasons) if reasoning.reasons else 'encaixa melhor nos dados atuais'}.")
            if unknowns:
                parts.append("Ainda falta perceber " + _join_pt(unknowns[:3]) + ".")
            text = " ".join(parts)
        return ContentPresentationResult(text, target_id=target.id, continuation_token=target.id, provenance=("TASK_KNOWLEDGE",))
    if getattr(task, "task_type", "") == "plan_trip":
        response = answer_task_query(work_session, task, "resume-me só o importante")
        return ContentPresentationResult(response or "Ainda estou a reunir o essencial desta viagem.", target_id=target.id, continuation_token=target.id, provenance=("TASK_KNOWLEDGE",))
    response = answer_task_query(work_session, task, "o que sabemos")
    return ContentPresentationResult(response or "Ainda não tenho muita informação estruturada nesta tarefa.", target_id=target.id, continuation_token=target.id, provenance=("TASK_KNOWLEDGE",))


def _effective_target_type(operation: ContentOperation, message: str, previous_target_id: str) -> str:
    if operation.target_type and operation.target_type != "previous":
        return operation.target_type
    text = normalize_for_matching(message)
    if contains_any_phrase(text, _SOURCE_WORDS):
        return "source"
    if contains_any_phrase(text, _COMPARISON_WORDS):
        return "comparison"
    if contains_any_phrase(text, _RESEARCH_WORDS):
        return "research_synthesis"
    if contains_any_phrase(text, _TASK_WORDS):
        return "task"
    return "source" if previous_target_id.startswith("source-") else "task"


def _resolve_ordinal(targets: list[AddressableContent], message: str, target_type: str) -> AddressableContent | None:
    ordinal = _ordinal_index(normalize_for_matching(message))
    if not ordinal:
        return None
    candidates = [target for target in targets if target.type == (target_type or "source")]
    if ordinal == -1:
        return candidates[-1] if candidates else None
    if 1 <= ordinal <= len(candidates):
        return candidates[ordinal - 1]
    return None


def _resolve_named_source(targets: list[AddressableContent], hint: str) -> ContentResolveResult:
    sources = [target for target in targets if target.type == "source"]
    tokens = _meaningful_tokens(hint)
    if not tokens:
        return ContentResolveResult(None, reason="no_hint")
    scores: list[tuple[int, AddressableContent]] = []
    hint_norm = normalize_for_matching(hint)
    for source in sources:
        aliases = [normalize_for_matching(alias) for alias in source.aliases if alias]
        haystack = normalize_for_matching(" ".join([source.title, " ".join(source.aliases)]))
        score = 0
        if hint_norm in aliases:
            score = 100
        elif any(alias and alias in hint_norm for alias in aliases):
            score = 95
        elif any(alias and hint_norm in alias and len(hint_norm) >= 4 for alias in aliases):
            score = 85
        elif all(token in haystack for token in tokens):
            score = 70
        elif len(tokens) > 1 and any(token in haystack for token in tokens if len(token) >= 5):
            score = 55
        if score:
            scores.append((score, source))
    if not scores:
        return ContentResolveResult(None, reason="no_named_match")
    scores.sort(key=lambda item: item[0], reverse=True)
    best_score = scores[0][0]
    best = [source for score, source in scores if score == best_score]
    if len(best) > 1 and best_score < 95:
        return ContentResolveResult(None, ambiguous=True, candidates=tuple(best), reason="ambiguous")
    return ContentResolveResult(best[0], reason="named_source")


def _ordered_user_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def rank(source: dict[str, Any]) -> tuple[int, int]:
        status = str(source.get("status") or "")
        applicability = str(source.get("applicability") or "")
        strength = str(source.get("source_strength_for_claim") or "")
        strength_rank = {"strong": 0, "useful": 1, "weak": 2}.get(strength, 3)
        if status == "accepted" and applicability in {"exact", ""}:
            bucket = 0
        elif status == "accepted":
            bucket = 1
        else:
            bucket = 2
        return bucket, strength_rank, int(source.get("visible_order") or 9999)

    return sorted([dict(source) for source in sources], key=rank)


def _focused_target(targets: list[AddressableContent], target_type: str) -> AddressableContent | None:
    candidates = [target for target in targets if target.type == target_type]
    return candidates[0] if candidates else None


def _virtual_source_list(targets: list[AddressableContent]) -> AddressableContent:
    task_id = next((target.task_id for target in targets if target.type == "source"), "")
    return AddressableContent(id=f"source_list:{task_id}", type="source_list", task_id=task_id, title="Fontes")


def _is_open_source_command(text: str) -> bool:
    if not (text == "abre" or text.startswith("abre ") or text.startswith("abrir ")):
        return False
    return contains_any_phrase(text, _SOURCE_WORDS) or text in {"abre", "abrir"}


def _is_read_command(text: str) -> bool:
    return (
        text.startswith("le ")
        or text.startswith("lê ")
        or text.startswith("le-me ")
        or text.startswith("lê-me ")
        or text.startswith("diz-me ")
        or text.startswith("diz me ")
        or text.startswith("o que diz ")
        or text.startswith("conta-me ")
    )


def _is_summarize_command(text: str) -> bool:
    return text == "resume" or text.startswith("resume ") or text.startswith("resume-me ") or text.startswith("resume me ")


def _is_bare_followup(text: str) -> bool:
    return text in {"continua", "mais detalhe", "com mais detalhe", "repete", "resume", "abre"}


def _target_type_from_text(text: str, *, default: str = "") -> str:
    if contains_any_phrase(text, _SOURCE_WORDS):
        return "source"
    if contains_any_phrase(text, _COMPARISON_WORDS):
        return "comparison"
    if contains_any_phrase(text, _RESEARCH_WORDS):
        return "research_synthesis"
    if contains_any_phrase(text, _TASK_WORDS):
        return "task"
    return default


def _section_from_text(text: str) -> str:
    if not (
        text.startswith("le ")
        or text.startswith("lê ")
        or text.startswith("le-me ")
        or text.startswith("lê-me ")
        or text.startswith("diz-me ")
        or text.startswith("diz me ")
        or text.startswith("mostra ")
    ):
        return ""
    mapping = {
        "unknowns": ("o que falta", "o que ainda falta", "que falta", "lacunas", "falta saber"),
        "recommendation": ("recomendacao", "recomendação", "conclusao", "conclusão", "escolherias"),
        "constraints": ("limites", "restricoes", "restrições"),
        "priorities": ("prioridades", "criterios", "critérios"),
        "next_step": ("proximo passo", "próximo passo"),
        "key_information": ("o que sabemos", "o que ja sabemos", "o que já sabemos"),
    }
    for role, phrases in mapping.items():
        if contains_any_phrase(text, tuple("le " + phrase for phrase in phrases) + tuple("lê " + phrase for phrase in phrases) + tuple("diz-me " + phrase for phrase in phrases) + tuple("diz me " + phrase for phrase in phrases) + tuple("mostra " + phrase for phrase in phrases)):
            return role
    return ""


def _section_aliases(role: str, title: str) -> list[str]:
    aliases = [role, title]
    aliases.extend(
        {
            "overview": ["overview", "objetivo", "planeamento", "comparacao", "comparação"],
            "options": ["opcoes", "opções", "alternativas"],
            "constraints": ["limites", "restricoes", "restrições"],
            "priorities": ["prioridades", "criterios", "critérios"],
            "decisions": ["decisoes", "decisões"],
            "key_information": ["o que sabemos", "informacao principal", "informação principal"],
            "recommendation": ["recomendacao", "recomendação", "conclusao", "conclusão"],
            "conflicts": ["conflitos", "contradicoes", "contradições"],
            "unknowns": ["unknowns", "o que falta", "ainda falta", "lacunas"],
            "next_step": ["proximo passo", "próximo passo"],
        }.get(role, [])
    )
    return [normalize_for_matching(alias) for alias in aliases if alias]


def _source_aliases(title: str, domain: str, url: str) -> list[str]:
    aliases = [title, domain, domain.replace("www.", "")]
    parsed = urlparse(url)
    if parsed.netloc:
        aliases.append(parsed.netloc)
        aliases.append(parsed.netloc.replace("www.", ""))
    for part in re.split(r"[^A-Za-z0-9]+", title):
        if len(part) >= 4 and not part.isdigit():
            aliases.append(part)
    return [alias for alias in aliases if alias]


def _target_hint(message: str) -> str:
    text = normalize_for_matching(message)
    text = re.sub(r"\b(?:abre|abrir|le|lê|le-me|lê-me|diz-me|diz me|resume|mostra|mostra-me|o que diz|da|do|das|dos|a|o|as|os|fonte|fontes|artigo|link|resultado|primeira|segunda|terceira|ultima|última|agora|exatamente|exactamente|excerto|com|mais|detalhe|so|só)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip(" .!?")


def _verbosity_from_text(text: str) -> str:
    if contains_any_phrase(text, ("le tudo", "lê tudo", "com mais detalhe", "mais detalhe", "continua")):
        return "detailed"
    if contains_any_phrase(text, ("resume muito", "so o importante", "só o importante", "so a conclusao", "só a conclusão")):
        return "brief"
    return "summary"


def _ordinal_index(text: str) -> int:
    if contains_any_phrase(text, ("primeira", "primeiro", "1a", "1º", "1 ")):
        return 1
    if contains_any_phrase(text, ("segunda", "segundo", "2a", "2º", "2 ")):
        return 2
    if contains_any_phrase(text, ("terceira", "terceiro", "3a", "3º", "3 ")):
        return 3
    if contains_any_phrase(text, ("ultima", "última", "ultimo", "último")):
        return -1
    return 0


def _is_deictic(message: str) -> bool:
    return contains_any_phrase(normalize_for_matching(message), _DEICTIC_WORDS)


def _meaningful_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", normalize_for_matching(text))
        if len(token) >= 3 and token not in _TOKEN_STOPWORDS
    ]


def _entry_phrase(entry) -> str:
    predicate = str(entry.predicate).replace("_", " ")
    return f"{entry.subject}: {predicate} = {entry.value}"


def _natural_entry_phrase(entry, *, brief: bool = False) -> str:
    subject = str(getattr(entry, "subject", "") or "a opção")
    predicate = str(getattr(entry, "predicate", "") or "")
    value = normalize_for_matching(str(getattr(entry, "value", "") or ""))
    weak = str((getattr(entry, "metadata", {}) or {}).get("source_strength_for_claim") or "") == "weak"
    verb = "sugere" if weak else "mostra"
    if predicate == "camera_quality":
        if value == "better":
            return f"{verb} uma vantagem de câmara para {subject}"
        if value == "good":
            return f"{verb} bom desempenho de câmara no {subject}"
        return f"deixa a avaliação da câmara do {subject} ainda mista"
    if predicate == "battery":
        if value == "better":
            return f"{subject} parece ter melhor autonomia"
        return f"a autonomia de {subject} surge como {getattr(entry, 'value', '')}"
    if predicate == "support_quality":
        if value == "better":
            return f"{subject} aparece com melhor suporte"
        if value == "limited":
            return f"{subject} aparece com suporte mais limitado"
    if predicate == "price":
        return f"{subject} aparece com preço de {getattr(entry, 'value', '')} €"
    if predicate == "activities":
        return f"{subject} tem {getattr(entry, 'value', '')} como ponto relevante"
    label = predicate.replace("_", " ")
    text = f"{subject}: {label} {getattr(entry, 'value', '')}".strip()
    return text if brief else text + _evidence_suffix(entry)


def _evidence_suffix(entry) -> str:
    evidence = _short_excerpt(_clean_source_excerpt(getattr(getattr(entry, "source", None), "original_text", "")), 220)
    return f" — {evidence}" if evidence else ""


def _short_excerpt(value: str, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 3].rsplit(" ", 1)[0].strip() + "..."


def _clean_source_excerpt(value: str) -> str:
    text = str(value or "")
    lines: list[str] = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        clean = re.sub(r"\s+", " ", line).strip()
        if not clean:
            continue
        if re.search(r"\b(?:subscribers|likes|views?|description|login|sign in|cookies?|share|save)\b", clean, flags=re.IGNORECASE):
            continue
        lines.append(clean)
    compact = " ".join(lines)
    compact = re.sub(r"\b\d[\d,.]*\s*[kmb]?\s*(?:subscribers|likes|views?)\b", "", compact, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", compact).strip()


def _join_pt(items: list[str]) -> str:
    clean = [str(item) for item in items if str(item)]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    return ", ".join(clean[:-1]) + " e " + clean[-1]


def _domain(url: str) -> str:
    try:
        return urlparse(str(url)).netloc
    except Exception:
        return ""


def _first_sentence(text: str) -> str:
    match = re.search(r"^(.+?[.!?])(?:\s|$)", str(text).strip())
    return match.group(1) if match else str(text).strip()


def _task_type_label(task_type: str) -> str:
    return {
        "generic_comparison": "Comparação",
        "plan_trip": "Planeamento de viagem",
        "compose_email": "Email",
    }.get(task_type, "Tarefa")

"""Pure lexical primitives shared by conversation routing and the interpreter.

This module deliberately contains no execution code and no WorkSession
mutation. It exists to keep `assistant.interaction_interpreter` from
importing private helpers from `assistant.conversation`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from assistant.text_matching import contains_any_phrase, find_near_pair_span


@dataclass(frozen=True)
class PrimitiveActivityInterpretation:
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


_ACTIVITY_SWITCH_PHRASES = (
    "enquanto isso",
    "primeiro preciso de",
    "antes disso",
    "deixa isso por agora",
    "vamos fazer outra coisa",
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

_TASK_TRIGGER_PHRASES = {
    "compose_email": (
        "vamos escrever um mail",
        "vamos escrever um email",
        "vamos escrever uma mensagem",
        "ajuda-me num mail",
        "ajuda-me num email",
        "ajuda-me no mail",
        "ajuda-me no email",
        "ajuda-me com um mail",
        "ajuda-me com um email",
        "ajuda-me com uma mensagem",
        "ajuda me num mail",
        "ajuda me num email",
        "ajuda me no mail",
        "ajuda me no email",
        "ajuda me com um mail",
        "ajuda me com um email",
        "ajuda me com uma mensagem",
        "preciso de escrever um mail",
        "preciso de escrever um email",
        "escrevermos um mail",
        "escrevermos um email",
        "escrevermos uma mensagem",
    ),
}

_TASK_CONTROL_EXECUTE_PHRASES = (
    "podes escrever", "ja podes escrever", "podes redigir", "ja podes redigir",
    "escreve o email", "escreve a mensagem", "escreve",
    "redige o email", "redige a mensagem", "redige",
    "cria o email", "cria a mensagem",
    "faz uma primeira versao", "faz o rascunho", "faz uma versao",
    "podes avancar", "podes criar",
    "atualiza o email", "atualiza a mensagem", "reescreve com isso", "reescreve o email",
)
_TASK_CONTROL_BARE_CONFIRM_PHRASES = ("podes", "sim", "forca", "avanca", "continua")

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

_ROLE_WORDS_ALT = "|".join(_RECIPIENT_ROLE_MARKERS.keys())
_SEP = r"(?:,\s*|\s+)"
_ROLE_COURSE_NAME_RE = re.compile(
    rf"\b(?:{_ROLE_WORDS_ALT})\s+de\s+([\wà-ÿ]+(?:\s+[\wà-ÿ]+)*?)\s*,\s*([\wà-ÿ]+(?:\s+[\wà-ÿ]+){{0,3}})(?=[.,;]|$)",
    re.IGNORECASE,
)
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
_TOPIC_ABOUT_RE = re.compile(
    r"\b(?:sobre|acerca\s+de|relativamente\s+a|em\s+rela[cç][aã]o\s+a)\s+(.+)$",
    re.IGNORECASE,
)
_NAME_STOPWORDS = {
    "e", "ok", "sim", "nao", "claro", "bom", "boa", "entao", "ja", "so", "ah", "bem", "ola", "obrigado", "obrigada",
    "vamos", "quero", "preciso", "ajuda", "escrever", "redigir", "preparar",
    "meu", "minha", "nosso", "nossa", "teu", "tua", "vosso", "vossa",
}


def task_metadata(task_type: str, activity_family: str = "unknown", domain: str = "unknown", artifact_type: str = "unknown", goal: str = "") -> dict[str, str]:
    defaults = dict(_TASK_METADATA.get(task_type, {}))
    return {
        "activity_family": activity_family if activity_family != "unknown" else defaults.get("activity_family", "unknown"),
        "domain": domain if domain != "unknown" else defaults.get("domain", "unknown"),
        "artifact_type": artifact_type if artifact_type != "unknown" else defaults.get("artifact_type", "unknown"),
        "goal": goal or defaults.get("goal", task_type),
    }


def detect_new_task_intent(normalized_text: str) -> str:
    for task_type, phrases in _TASK_TRIGGER_PHRASES.items():
        if contains_any_phrase(normalized_text, phrases):
            return task_type
    return ""


def _activity_start_from_message(normalized_text: str) -> tuple[str, str]:
    if contains_any_phrase(normalized_text, _TASK_TRIGGER_PHRASES["compose_email"]):
        return "compose_email", "compose_email_trigger"
    if contains_any_phrase(normalized_text, ("vamos comparar", "quero comparar", "preciso de comparar", "ajuda me a comparar", "estou a pensar")) and contains_any_phrase(normalized_text, ("comparar", "com", "entre")):
        return "generic_comparison", "generic_comparison_trigger"
    if contains_any_phrase(normalized_text, ("planear umas ferias", "planear ferias", "planear uma viagem", "planear viagem", "road trip")):
        return "plan_trip", "travel_planning_trigger"
    if "ferias" in normalized_text and contains_any_phrase(normalized_text, ("vamos planear", "quero planear", "gostava de planear", "prepara te")):
        return "plan_trip", "travel_planning_trigger"
    if contains_any_phrase(normalized_text, ("vamos estudar",)):
        return "study_topic", "study_trigger"
    if contains_any_phrase(normalized_text, ("vamos decidir", "preciso de decidir", "ajuda me a decidir", "decidir entre")):
        return "make_decision", "decision_trigger"
    if contains_any_phrase(normalized_text, ("quero investigar", "vamos investigar", "preciso de investigar", "investigar opcoes", "investigar opções")):
        return "research_topic", "research_trigger"
    if contains_any_phrase(normalized_text, ("vamos organizar", "vamos preparar")):
        return "generic_activity", "generic_activity_trigger"
    if contains_any_phrase(normalized_text, ("vamos escrever juntos", "vamos escrever")):
        return "writing", "writing_trigger"
    return "", ""


def interpret_activity(raw_message: str, normalized_text: str, active_task, paused_tasks: list) -> PrimitiveActivityInterpretation:
    task_type, reason = _activity_start_from_message(normalized_text)
    metadata = task_metadata(task_type) if task_type else task_metadata("")
    has_switch = contains_any_phrase(normalized_text, _ACTIVITY_SWITCH_PHRASES)
    has_resume = contains_any_phrase(normalized_text, _ACTIVITY_RESUME_PHRASES)
    resume_target = resume_target_from_message(normalized_text)
    if has_resume and resume_target and paused_tasks:
        meta = task_metadata(resume_target)
        return PrimitiveActivityInterpretation(
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
    if task_type:
        refines = bool(active_task and active_task.task_type in ("writing", "unknown") and task_type != active_task.task_type)
        return PrimitiveActivityInterpretation(
            starts_task=not refines,
            refines_task=refines,
            activity_family=metadata["activity_family"],
            domain=metadata["domain"],
            artifact_type=metadata["artifact_type"],
            task_type=task_type,
            goal=metadata["goal"],
            confidence=0.95 if has_switch or reason else 0.75,
            reason=("switch:" if has_switch else "") + reason,
            remaining_content=strip_activity_switch_prefix(normalized_text),
        )
    if active_task is not None and active_task.status not in ("completed", "cancelled", "paused"):
        if message_mentions_active_task(normalized_text, active_task):
            return PrimitiveActivityInterpretation(
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
    return PrimitiveActivityInterpretation()


def strip_activity_switch_prefix(normalized_text: str) -> str:
    result = normalized_text
    for phrase in _ACTIVITY_SWITCH_PHRASES:
        if phrase in result:
            result = result.split(phrase, 1)[1].strip(" ,.!?")
            break
    return result or normalized_text


def resume_target_from_message(normalized_text: str) -> str:
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


def message_mentions_active_task(normalized_text: str, ctx) -> bool:
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
        return contains_any_phrase(normalized_text, ("ferias", "viagem", "norte", "portugal", "destino", "alojamento", "descansar", "aventura"))
    if ctx.task_type == "study_topic":
        return contains_any_phrase(normalized_text, ("exame", "teste", "slides", "apontamentos", "materia", "algoritmos", "grafos", "estudar"))
    if ctx.task_type == "make_decision":
        return contains_any_phrase(normalized_text, ("opcao", "opcoes", "escolher", "decidir", "comparar", "criterios", "contratar"))
    if ctx.task_type == "research_topic":
        return contains_any_phrase(normalized_text, ("investigar", "pesquisar", "opcoes", "opcao", "fontes", "alternativas"))
    if ctx.task_type == "generic_comparison":
        return contains_any_phrase(normalized_text, ("compara", "comparar", "comparacao", "comparação", "bateria", "camara", "câmara", "autonomia", "criterios", "critérios", "opcao", "opcoes", "adiciona", "acrescenta"))
    if ctx.task_type == "generic_activity":
        return contains_any_phrase(normalized_text, ("organizar", "preparar", "plano", "passo", "proximo", "próximo", "tarefa", "ideia"))
    if ctx.task_type == "writing":
        return contains_any_phrase(normalized_text, ("escrever", "texto", "documento", "nota", "email", "mail"))
    return False


def detect_task_control_action(normalized_message: str, ctx) -> str:
    if contains_any_phrase(normalized_message, _TASK_CONTROL_EXECUTE_PHRASES):
        return "execute"
    if getattr(ctx, "pending_action", "") == "create_draft" and contains_any_phrase(normalized_message, _TASK_CONTROL_BARE_CONFIRM_PHRASES):
        return "execute"
    return ""


def normalize_extracted_name(text: str) -> str:
    cleaned = re.sub(r"^(?:o|a|os|as|um|uma)\s+", "", text.strip(), flags=re.IGNORECASE)
    return " ".join(word.capitalize() for word in cleaned.split())


def looks_like_name_candidate(candidate: str) -> bool:
    words = candidate.split()
    if len(words) >= 2:
        return True
    if not words:
        return False
    return words[0].lower() not in _NAME_STOPWORDS


def extract_role_course_name(raw_text: str) -> dict[str, str]:
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
        if captured.get("name") and not looks_like_name_candidate(captured["name"]):
            continue
        role_match = re.search(rf"\b({_ROLE_WORDS_ALT})\b", raw_text, re.IGNORECASE)
        result: dict[str, str] = {}
        if role_match:
            result["role"] = _RECIPIENT_ROLE_MARKERS.get(role_match.group(1).lower(), "")
        for key, cleaned in captured.items():
            if cleaned:
                result[key] = normalize_extracted_name(cleaned)
        return result
    return {}


def purpose_marker_keys() -> tuple[str, ...]:
    return tuple(_PURPOSE_MARKERS.keys())


def topic_about_re() -> re.Pattern:
    return _TOPIC_ABOUT_RE


_WORKSPACE_RESTORE_SCENE_PHRASES = (
    "volta ao que estavamos a fazer",
    "volta ao que estavamos a trabalhar",
    "volta ao que tinhamos aberto",
    "volta a trabalhar naquilo",
)
_WORKSPACE_AUTO_ORGANIZE_PHRASES = ("organiza isto automaticamente", "organiza automaticamente", "organiza isto sozinho")
_WORKSPACE_SIDE_BY_SIDE_PHRASES = ("lado a lado", "um ao lado do outro", "mostra os dois grandes")
_WORKSPACE_FOCUS_EXCERPT_MARKERS = ("mostra so o", "mostra apenas o", "so quero ver o", "quero ver so o")
_WORKSPACE_WORK_ON_PHRASES = ("vamos trabalhar na segunda versao", "vamos trabalhar no rascunho", "vamos trabalhar na versao", "trabalhar na segunda versao", "trabalhar no rascunho")
_WORKSPACE_ASIDE_PHRASES = ("chega isso para o lado", "chega isto para o lado", "poe isso ao lado", "poe isto ao lado")
_WORKSPACE_BRING_FORWARD_PHRASES = ("traz o orcamento para a frente", "traz isto para a frente", "traz isso para a frente", "traz para a frente")
_WORKSPACE_MINIMIZE_PHRASES = ("minimiza o original", "minimiza isto", "minimiza isso", "podes minimizar o original", "ja podes minimizar", "compacta o original", "compacta isto")
_WORKSPACE_EXPAND_PHRASES = (
    "expande o original",
    "expande a versao",
    "expande isto",
    "expande isso",
    "aumenta o original",
    "aumenta o email",
    "aumenta o mail",
    "expande o email",
    "expande o mail",
    "maximiza o email",
    "maximiza o mail",
    "faz o email maior",
    "faz o mail maior",
    "quero ver o original maior",
)
_WORKSPACE_HIDE_PHRASES = ("esconde isto", "esconde isso", "esconde o original", "esconde isto por agora", "esconde isso por agora")
_WORKSPACE_CLOSE_PHRASES = ("fecha a fatura", "fecha o original", "fecha isto", "fecha isso", "fecha o ficheiro", "fecha a segunda versao", "fecha o segundo ficheiro")
_WORKSPACE_KEEP_OPEN_PHRASES = ("mantem isto aberto", "mantem isso aberto", "deixa isto aberto", "deixa so estes dois abertos", "deixa apenas estes dois abertos")
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
_WORKSPACE_GLOBAL_ALL_PHRASES = (
    "maximiza todos",
    "maximiza tudo",
    "abre todos",
    "abre tudo",
    "mostra todos",
    "expande todos",
    "expande tudo",
    "traz tudo para cima",
)
_ENTITY_TARGET_PHRASE = "echo_entity"
_ENTITY_MOVEMENT_VERBS = ("desloca", "desloca-te", "move", "move-te", "mexe", "mexe-te", "vai", "chega", "chega-te", "chega te", "encosta", "encosta-te", "aproxima", "aproxima-te", "afasta", "afasta-te", "sai", "poe-te", "poe te", "fica", "coloca-te")
_ENTITY_DIRECTION_WORDS = ("direita", "direta", "esquerda", "cima", "baixo", "lado", "canto", "centro")
_ENTITY_DIRECTION_TO_REGION = {"direita": "right", "direta": "right", "esquerda": "left", "cima": "up", "baixo": "down", "centro": "center", "lado": "", "canto": ""}
_ENTITY_OBJECT_TARGETED_CHEGA_PHRASES = ("chega isto", "chega isso")
_ENTITY_CENTER_PHRASES = ("volta para o centro", "volta ao centro", "centra-te", "coloca-te no centro", "fica no centro")
_ENTITY_MAKE_ROOM_PHRASES = ("preciso de espaco", "preciso de espaco aqui", "preciso de espaco para trabalhar", "da-me espaco", "da-me espaco para trabalhar", "quero espaco", "deixa espaco", "abre espaco", "sai um bocado do meio", "sai do meio", "chega-te para o lado", "chega te para o lado", "poe-te ali", "poe te ali")
_WORKSPACE_TARGET_HINTS = ("segunda versao", "primeira versao", "versao anterior", "segundo ficheiro", "primeiro ficheiro", "este documento", "estes dois", "os dois", "o original", "original", "email", "mail", "mensagem", "rascunho", "draft", "fatura", "orcamento", "excerto", "paragrafo", "isto", "isso", "aquilo")
_WORKSPACE_ORDINAL_WORDS = {"primeiro": 1, "primeira": 1, "segundo": 2, "segunda": 2, "terceiro": 3, "terceira": 3, "quarto": 4, "quarta": 4}


def entity_target_phrase() -> str:
    return _ENTITY_TARGET_PHRASE


def _entity_workspace_command(normalized_clause: str) -> dict[str, str] | None:
    if contains_any_phrase(normalized_clause, _ENTITY_OBJECT_TARGETED_CHEGA_PHRASES):
        return None
    if contains_any_phrase(normalized_clause, _ENTITY_CENTER_PHRASES):
        return {"action": "center", "target_phrase": _ENTITY_TARGET_PHRASE, "relation": "", "selector_index": "", "region": "center", "match_type": "exact_phrase", "signals": ["center_phrase"], "confidence": "high"}
    if contains_any_phrase(normalized_clause, _ENTITY_MAKE_ROOM_PHRASES):
        return {"action": "make_room", "target_phrase": _ENTITY_TARGET_PHRASE, "relation": "", "selector_index": "", "region": "", "match_type": "exact_phrase", "signals": ["make_room_phrase"], "confidence": "high"}
    if find_near_pair_span(normalized_clause, _ENTITY_MOVEMENT_VERBS, _ENTITY_DIRECTION_WORDS, max_gap=30):
        verb = next((word for word in _ENTITY_MOVEMENT_VERBS if word in normalized_clause), "")
        direction = next((word for word in _ENTITY_DIRECTION_WORDS if word in normalized_clause), "")
        region = _ENTITY_DIRECTION_TO_REGION.get(direction, "")
        return {"action": "move", "target_phrase": _ENTITY_TARGET_PHRASE, "relation": "", "selector_index": "", "region": region, "match_type": "verb_direction_pair", "signals": [verb, direction], "confidence": "high" if region else "medium"}
    return None


def _single_workspace_clause_action(normalized_clause: str) -> dict[str, str] | None:
    entity_result = _entity_workspace_command(normalized_clause)
    if entity_result:
        return entity_result
    action = ""
    relation = ""
    selector_index = 0
    if contains_any_phrase(normalized_clause, _WORKSPACE_GLOBAL_ALL_PHRASES):
        verb_action = "show" if contains_any_phrase(normalized_clause, ("mostra", "abre", "traz")) else "expand"
        return {
            "action": verb_action,
            "target_phrase": "",
            "scope": "all",
            "relation": "",
            "selector_index": "",
            "region": "",
            "match_type": "global_scope_phrase",
            "signals": ["all"],
            "confidence": "high",
        }
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
    return {"action": action, "target_phrase": target_phrase, "scope": "", "relation": relation, "selector_index": str(selector_index) if selector_index else "", "region": ""}


def workspace_command_clauses(normalized: str) -> list[str]:
    clauses = [normalized]
    for separator in (" e ", ","):
        if separator in normalized:
            clauses = [part.strip(" .!?") for part in normalized.replace(",", " e ").split(" e ") if part.strip()]
            break
    return clauses


def workspace_command_actions(normalized: str) -> tuple[list[dict[str, str]], list[str]]:
    clauses = workspace_command_clauses(normalized)
    found: list[dict[str, str]] = []
    remaining: list[str] = []
    seen_actions: set[str] = set()
    for clause in clauses:
        result = _single_workspace_clause_action(clause)
        if result and result["action"] not in seen_actions:
            found.append(result)
            seen_actions.add(result["action"])
        elif not result:
            remaining.append(clause)
    if not found:
        whole = _single_workspace_clause_action(normalized)
        if whole:
            return [whole], []
        return [], remaining
    if any(c["action"] == "move" for c in found) and any(c["action"] == "make_room" for c in found):
        found = [c for c in found if c["action"] != "make_room"]
    return found[:3], remaining

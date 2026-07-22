"""Deterministic assertion checks for one graded turn.

Every check takes the turn's TurnExpectation + TurnResult and returns an
AssertionOutcome. run_turn_assertions() is the single entry point the runner
calls; it only runs a check when the case actually set the corresponding
expectation (an unset expectation is not a pass, it's "not checked").
"""

from __future__ import annotations

import re
import unicodedata

from evals.schemas import AssertionOutcome, TurnExpectation, TurnResult

_BRAZILIAN_MARKERS = (
    "aplicativos",
    "aplicativo",
    "tela",
    "arquivos",
    "arquivo",
    "acessar",
    "aprendizado",
    "revisar",
    "okay",
)
_INFORMAL_ADDRESS_MARKERS = ("voce", "voces")
_COMPUTER_OBSERVATION_VERBS = (
    "acompanhar",
    "acompanho",
    "observar",
    "observo",
    "ver",
    "vejo",
    "monitorizar",
    "monitorizo",
    "usar",
    "usas",
)
_COMPUTER_OBSERVATION_OBJECTS = (
    "ecra",
    "ecran",
    "tela",
    "aplicacao",
    "aplicacoes",
    "aplicativo",
    "aplicativos",
    "ficheiro",
    "ficheiros",
    "arquivo",
    "arquivos",
    "atividade",
    "actividade",
    "computador",
    "chrome",
    "zoom",
    "vscode",
    "vs code",
    "terminal",
    "outlook",
    "teams",
)


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", (text or "").lower())
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def detect_brazilian_portuguese_markers(text: str) -> list[str]:
    normalized = _normalize(text)
    return [marker for marker in _BRAZILIAN_MARKERS if _has_word(normalized, marker)]


def detect_informal_address_mix(text: str) -> list[str]:
    normalized = _normalize(text)
    return [marker for marker in _INFORMAL_ADDRESS_MARKERS if _has_word(normalized, marker)]


def run_turn_assertions(expected: TurnExpectation, result: TurnResult) -> list[AssertionOutcome]:
    outcomes: list[AssertionOutcome] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        outcomes.append(AssertionOutcome(name=name, passed=condition, detail=detail))

    if expected.expected_exception is not None:
        check(
            "expected_exception",
            result.exception_type == expected.expected_exception,
            f"esperado={expected.expected_exception} obtido={result.exception_type}",
        )
        # An expected exception short-circuits every other content check —
        # there is no meaningful response/path to grade when the case is
        # specifically testing that a failure is handled.
        return outcomes

    check(
        "no_unexpected_exception",
        result.exception_type is None,
        f"exception_type={result.exception_type} message={result.exception_message}",
    )

    if expected.selected_path is not None:
        check(
            "selected_path",
            result.selected_path == expected.selected_path,
            f"esperado={expected.selected_path} obtido={result.selected_path}",
        )

    if expected.forbidden_paths:
        check(
            "forbidden_paths",
            result.selected_path not in expected.forbidden_paths,
            f"obtido={result.selected_path} proibidos={list(expected.forbidden_paths)}",
        )

    if expected.llm_calls_min is not None:
        check(
            "llm_calls_min",
            result.llm_calls >= expected.llm_calls_min,
            f"esperado>={expected.llm_calls_min} obtido={result.llm_calls}",
        )

    if expected.llm_calls_max is not None:
        check(
            "llm_calls_max",
            result.llm_calls <= expected.llm_calls_max,
            f"esperado<={expected.llm_calls_max} obtido={result.llm_calls}",
        )

    if expected.expected_tools:
        used = set(result.tools_used)
        missing = [t for t in expected.expected_tools if t not in used]
        check("expected_tools", not missing, f"em falta={missing} usados={result.tools_used}")

    if expected.forbidden_tools:
        used = set(result.tools_used)
        present = [t for t in expected.forbidden_tools if t in used]
        check("forbidden_tools", not present, f"proibidos_usados={present}")

    if expected.no_tools_used:
        check("no_tools_used", not result.tools_used, f"tools_used={result.tools_used}")

    if expected.forbidden_contexts:
        active = set(result.active_contexts)
        present = [c for c in expected.forbidden_contexts if c in active]
        check("forbidden_contexts", not present, f"contextos_proibidos_ativos={present} obtidos={result.active_contexts}")

    if expected.expected_memory_ids:
        got = set(result.selected_memory_ids)
        missing = [m for m in expected.expected_memory_ids if m not in got]
        check("expected_memory_ids", not missing, f"em falta={missing} obtidos={result.selected_memory_ids}")

    if expected.expected_provider is not None:
        check(
            "expected_provider",
            result.provider == expected.expected_provider,
            f"esperado={expected.expected_provider} obtido={result.provider}",
        )

    if expected.expected_model is not None:
        check(
            "expected_model",
            result.model == expected.expected_model,
            f"esperado={expected.expected_model} obtido={result.model}",
        )

    if expected.expected_model_routing_provider is not None:
        check(
            "expected_model_routing_provider",
            result.model_routing_provider == expected.expected_model_routing_provider,
            f"esperado={expected.expected_model_routing_provider} obtido={result.model_routing_provider}",
        )

    if expected.expected_model_routing_mode is not None:
        check(
            "expected_model_routing_mode",
            result.model_routing_mode == expected.expected_model_routing_mode,
            f"esperado={expected.expected_model_routing_mode} obtido={result.model_routing_mode}",
        )

    if expected.expected_model_routing_reason_code is not None:
        check(
            "expected_model_routing_reason_code",
            result.model_routing_reason_code == expected.expected_model_routing_reason_code,
            f"esperado={expected.expected_model_routing_reason_code} obtido={result.model_routing_reason_code}",
        )

    if expected.expected_model_routing_paid_call is not None:
        check(
            "expected_model_routing_paid_call",
            result.model_routing_paid_call is expected.expected_model_routing_paid_call,
            f"esperado={expected.expected_model_routing_paid_call} obtido={result.model_routing_paid_call}",
        )

    if expected.require_provider_match:
        check(
            "provider_matches_request",
            result.provider == result.requested_provider,
            f"pedido={result.requested_provider} obtido={result.provider}",
        )

    if expected.require_model_match:
        check(
            "model_matches_request",
            result.model == result.requested_model,
            f"pedido={result.requested_model} obtido={result.model}",
        )

    if expected.forbid_fallback:
        check("no_fallback", not result.fallback_used, f"fallback_used={result.fallback_used}")

    if expected.forbidden_response_sources:
        check(
            "forbidden_response_sources",
            result.response_source not in expected.forbidden_response_sources,
            f"obtido={result.response_source} proibidos={list(expected.forbidden_response_sources)}",
        )

    if expected.forbid_deterministic_response:
        deterministic_sources = {
            "DIRECT_SHORT_RESPONSE",
            "DETERMINISTIC_HELP",
            "FAST_ROUTE",
            "TOOL_RESULT",
            "TOOL_CONFIRMATION",
            "MEMORY_RECALL_DETERMINISTIC",
            "MEMORY_WRITE_DETERMINISTIC",
            "MEMORY_COMMAND",
            "SESSION_COMMAND",
            "ERROR",
        }
        check(
            "no_deterministic_response",
            result.response_source not in deterministic_sources,
            f"response_source={result.response_source}",
        )

    if expected.checks_memory_write_action:
        check(
            "memory_write_action",
            result.memory_write_action == expected.memory_write_action,
            f"esperado={expected.memory_write_action!r} obtido={result.memory_write_action!r}",
        )

    if expected.forbid_memory_write:
        check(
            "forbid_memory_write",
            result.memory_write_action is None,
            f"a memoria foi escrita quando nao devia (action={result.memory_write_action})",
        )

    if expected.forbid_ungrounded_computer_observation:
        grounded = bool(result.tools_used) or bool(result.grounding_sources)
        claim = detect_ungrounded_computer_observation(result.final_response)
        check(
            "no_ungrounded_computer_observation",
            grounded or not claim,
            f"claim={claim!r} tools_used={result.tools_used} grounding_sources={result.grounding_sources}",
        )

    for phrase in expected.must_contain:
        check(f"must_contain:{phrase}", phrase.lower() in result.final_response.lower(), result.final_response)

    if expected.must_contain_any:
        normalized_response = _normalize(result.final_response)
        found = [phrase for phrase in expected.must_contain_any if _normalize(phrase) in normalized_response]
        check(
            "must_contain_any",
            bool(found),
            f"esperado_um_de={list(expected.must_contain_any)} resposta={result.final_response}",
        )

    for index, group in enumerate(expected.must_contain_each_any):
        normalized_response = _normalize(result.final_response)
        found = [phrase for phrase in group if _normalize(phrase) in normalized_response]
        check(
            f"must_contain_each_any:{index}",
            bool(found),
            f"esperado_um_de={list(group)} resposta={result.final_response}",
        )

    for phrase in expected.must_not_contain:
        check(
            f"must_not_contain:{phrase}",
            phrase.lower() not in result.final_response.lower(),
            result.final_response,
        )

    if expected.response_not_empty:
        check("response_not_empty", bool(result.final_response.strip()))

    if expected.response_grounded is not None:
        check(
            "response_grounded",
            result.response_grounded == expected.response_grounded,
            f"esperado={expected.response_grounded} obtido={result.response_grounded}",
        )

    if expected.max_latency_ms is not None:
        check(
            "max_latency_ms",
            result.latency_ms <= expected.max_latency_ms,
            f"limite={expected.max_latency_ms}ms obtido={result.latency_ms:.0f}ms",
        )

    if expected.max_words is not None:
        word_count = len(re.findall(r"\b[\wÀ-ÿ'-]+\b", result.final_response, flags=re.UNICODE))
        check(
            "max_words",
            word_count <= expected.max_words,
            f"limite={expected.max_words} obtido={word_count}",
        )

    if expected.min_words is not None:
        word_count = len(re.findall(r"\b[\wÀ-ÿ'-]+\b", result.final_response, flags=re.UNICODE))
        check(
            "min_words",
            word_count >= expected.min_words,
            f"limite={expected.min_words} obtido={word_count}",
        )

    if expected.min_bullet_points is not None:
        bullet_count = _count_bullet_points(result.final_response)
        check(
            "min_bullet_points",
            bullet_count >= expected.min_bullet_points,
            f"limite={expected.min_bullet_points} obtido={bullet_count}",
        )

    if expected.max_questions is not None:
        question_count = result.final_response.count("?")
        check(
            "max_questions",
            question_count <= expected.max_questions,
            f"limite={expected.max_questions} obtido={question_count}",
        )

    if expected.max_outer_questions is not None:
        question_count = _count_outer_questions(result.final_response)
        check(
            "max_outer_questions",
            question_count <= expected.max_outer_questions,
            f"limite={expected.max_outer_questions} obtido={question_count}",
        )

    if expected.forbid_unnecessary_question_when_sufficient:
        sufficient = request_has_sufficient_info(result.user_message)
        question_count = result.final_response.count("?")
        check(
            "no_unnecessary_question_when_sufficient",
            not sufficient or question_count == 0,
            f"sufficient={sufficient} perguntas={question_count} resposta={result.final_response}",
        )

    if expected.forbid_unsupported_tool_claim:
        check(
            "no_unsupported_tool_claim",
            not result.unsupported_tool_claim_detected,
            "a resposta afirmou ter usado uma ferramenta que nao foi chamada",
        )

    if expected.forbid_unsupported_memory_claim:
        check(
            "no_unsupported_memory_claim",
            not result.unsupported_memory_claim_detected,
            "a resposta afirmou lembrar-se de algo sem grounding",
        )

    if expected.forbid_brazilian_portuguese:
        markers = detect_brazilian_portuguese_markers(result.final_response)
        check("no_brazilian_portuguese", not markers, f"marcadores={markers}" if markers else "")

    return outcomes


def detect_ungrounded_computer_observation(text: str) -> str:
    normalized = _normalize(text)
    if not normalized:
        return ""
    if _looks_like_denial_of_computer_access(normalized):
        return ""
    if _looks_like_conditional_or_hypothetical_observation(normalized):
        return ""
    for verb in _COMPUTER_OBSERVATION_VERBS:
        match = re.search(rf"\b{re.escape(verb)}\b", normalized)
        if not match:
            continue
        window_before = normalized[max(0, match.start() - 32) : match.start()]
        if re.search(r"\b(?:nao|nunca|sem|incapaz de|impossivel)\b", window_before):
            continue
        tail = normalized[match.end() : match.end() + 80]
        for obj in _COMPUTER_OBSERVATION_OBJECTS:
            if _has_word(tail, obj):
                return obj
    return ""


def request_has_sufficient_info(user_message: str) -> bool:
    text = _normalize(user_message)
    if "frase" in text and any(verb in text for verb in ("escreve", "ajuda-me a escrever", "da-me", "diz-me")):
        return any(
            phrase in text
            for phrase in (
                "pedir uma revisao",
                "pedir revisao",
                "pedir desculpa",
                "pedir confirmacao",
                "pedir resposta",
            )
        )
    return False


def _count_bullet_points(text: str) -> int:
    count = 0
    for line in (text or "").splitlines():
        stripped = line.strip()
        if re.match(r"^(?:[-*•]|\d+[.)])\s+\S", stripped):
            count += 1
    return count


def _count_outer_questions(text: str) -> int:
    """Count questions outside quoted/generated content blocks.

    This lets an eval for "write an email" allow a question mark inside the
    email itself, while still catching assistant-side follow-up questions such
    as "Queres que ajuste?" after the requested text.
    """

    if _looks_like_standalone_email(text or ""):
        return 0

    cleaned = re.sub(r'"[^"]*"', "", text or "", flags=re.DOTALL)
    cleaned = re.sub(r"“[^”]*”", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL)
    return cleaned.count("?")


def _looks_like_standalone_email(text: str) -> bool:
    compact = " ".join((text or "").strip().split())
    if not compact:
        return False
    normalized = _normalize(compact)
    email_starters = (
        "assunto:",
        "exmo.",
        "exma.",
        "caro professor",
        "cara professora",
        "prezado professor",
        "prezada professora",
        "bom dia professor",
        "boa tarde professor",
    )
    if not normalized.startswith(email_starters):
        return False
    assistant_followups = (
        "queres que",
        "posso ajustar",
        "se quiseres",
        "fica assim",
        "aqui vai",
    )
    return not any(marker in normalized for marker in assistant_followups)


def _looks_like_denial_of_computer_access(text: str) -> bool:
    denial_patterns = (
        r"\bnao\s+(?:consigo|posso|estou a|tenho acesso|tenho como|sou capaz de)\b.{0,80}\b(?:ecra|ecran|tela|aplicac(?:ao|oes)|ficheiros?|arquivos?|atividade|actividade|computador)\b",
        r"\bnao\s+(?:consigo|posso|estou a)\s+(?:ver|observar|acompanhar|monitorizar)\b",
        r"\bsem\s+acesso\b.{0,80}\b(?:ecra|ecran|tela|aplicac(?:ao|oes)|ficheiros?|arquivos?|atividade|actividade|computador)\b",
    )
    return any(re.search(pattern, text) for pattern in denial_patterns)


def _looks_like_conditional_or_hypothetical_observation(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:posso|poderia|conseguiria)\s+(?:ver|observar|acompanhar|monitorizar)\b.{0,80}\b(?:se|quando)\b",
            text,
        )
    )

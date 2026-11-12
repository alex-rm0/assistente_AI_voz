"""Conservative, deterministic speech clarity assessment. No model or mutation."""
from dataclasses import dataclass
import math
import re
import json

from assistant.text_matching import normalize_for_matching


@dataclass(frozen=True)
class SpeechClarityDecision:
    would_ask_repeat: bool
    signals: tuple[str, ...] = ()
    protected: tuple[str, ...] = ()


_FUNCTION_WORDS = frozenset("a o os as de do da dos das para por com que ao aos no na nos nas em the of for with to it that when an at which".split())
_DANGLING = frozenset("de do da dos das para com que ao aos the of for with to an which".split())


def evaluate_speech_clarity(turn, *, protected=()):
    if turn.source != "voice":
        return SpeechClarityDecision(False, protected=("typed",))
    # Quoted examples and ordinary PT question constructions are not broken speech.
    text = re.sub(r'"[^"\n]*"|“[^”\n]*”|(?<!\w)\x27[^\x27\n]+\x27(?!\w)', " ", turn.text)
    text = normalize_for_matching(text)
    text = re.sub(r"\b(?:o que|como|onde|porque) e que\b", "question", text)
    words = re.findall(r"[\w-]+", text)
    strong = []
    if len(words) >= 5 and words[-1] in _DANGLING:
        strong.append("dangling_tail")
    chain = 0
    for word in words:
        chain = chain + 1 if word in _FUNCTION_WORDS else 0
        if chain >= 4:
            strong.append("function_word_chain")
            break
    if re.search(r"\bque (?:eu|tu|ele|ela) (?:ver|fazer|ir|dizer|escrever|ler)\b", text):
        strong.append("unattached_infinitive")
    acoustic = []
    def finite(value):
        return isinstance(value, (int, float)) and math.isfinite(value)
    if finite(turn.avg_logprob) and turn.avg_logprob < -.9:
        acoustic.append("low_avg_logprob")
    if finite(turn.no_speech_probability) and turn.no_speech_probability > .5:
        acoustic.append("high_no_speech_probability")
    ask = len(strong) >= 2 or (len(strong) >= 1 and bool(acoustic))
    return SpeechClarityDecision(bool(ask and not protected), tuple(strong + acoustic), tuple(protected))


def protection_for_turn(turn, engine):
    from assistant.artifact_export import export_kind
    from assistant.fast_router import route_fast_command
    from assistant.interaction_interpreter import interpret
    from assistant.security import check_user_request
    from assistant.voice_wake import classify_voice_wake_intent
    from assistant.writing_requests import incomplete_writing_request
    text = normalize_for_matching(turn.text).strip(" .,!?:;")
    if turn.capture_mode == "standby" or classify_voice_wake_intent(turn.text) in {"wake", "sleep"}:
        return ("voice_control",)
    if not engine.presence.can_respond():
        return ("presence_state",)
    security = check_user_request(turn.text)
    if not security.allowed or security.action:
        return ("security_or_tool_command",)
    if text in {"sim", "nao", "yes", "no", "ok", "okay", "cancela", "cancel"}:
        return ("confirmation",)
    if route_fast_command(turn.text) is not None or export_kind(turn.text) is not None:
        return ("fast_command",)
    if incomplete_writing_request(text):
        return ("writing_intent",)
    session = engine._work_session
    interpretation = interpret(turn.text, session)
    activity = interpretation.activity
    if (interpretation.spatial.present or interpretation.attention.present or interpretation.representation.present
            or activity.starts_task or activity.resumes_task or activity.refines_task
            or interpretation.content.kind in {"entity", "control_execute"}):
        return ("canonical_intent",)
    pending = session.writing_context
    fields = [pending.awaiting_field] if pending is not None else []
    if session.active_task is not None:
        fields += engine._missing_task_fields(session.active_task)
    if pending is not None:
        if re.search(r"\b(?:nome|name|destinatario|recipient)\b", text) and re.search(r"\b(?:corrig\w*|troca\w*|altera\w*|change|correct)\b", text):
            return ("recipient_correction",)
    words = re.findall(r"[\w-]+", text)
    strong = {"dangling_tail", "function_word_chain", "unattached_infinitive"}
    if ("recipient" in fields and 1 <= len(words) <= 6
            and not re.search(r"\b(?:podes|consegues|queria|quero|preciso|can|could|help|want)\b", text)
            and any(word not in _FUNCTION_WORDS and word not in {"e", "and"} for word in words)
            and not strong.intersection(evaluate_speech_clarity(turn).signals)):
        return ("recipient_field",)
    if fields and re.fullmatch(r"(?:\d{1,4}(?:[/:h.,-]\d{1,4})*(?:\s*(?:euros?|dias?|days?|horas?|hours?))?|amanha|hoje|tomorrow|today)", text):
        return ("structured_field",)
    return ()


def observe_speech_clarity(turn, engine=None, *, protected=None, enforce=False):
    if turn.source != "voice":
        return None
    reasons = protected if protected is not None else protection_for_turn(turn, engine)
    decision = evaluate_speech_clarity(turn, protected=reasons)
    compact = lambda value: json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    print(
        f"[SPEECH_CLARITY] capture_id={turn.capture_id} session_id={turn.session_id} mode={turn.capture_mode} "
        f"would_ask_repeat={str(decision.would_ask_repeat).lower()} signals={compact(decision.signals)} "
        f"protected={str(bool(decision.protected)).lower()} protected_reasons={compact(decision.protected)} "
        f"enforce={str(enforce).lower()} stt_model={compact(turn.stt_model)} avg_logprob={turn.avg_logprob} "
        f"no_speech_probability={turn.no_speech_probability} compression_ratio={turn.compression_ratio}",
        flush=True,
    )
    return decision


def repeat_response(language):
    return ("Sorry, I didn't quite catch that — could you say it again?" if language == "en-GB"
            else "Desculpa, não percebi bem — podes repetir?")

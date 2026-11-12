"""Measured component health, and the reply to "how are you?" built from it.

R3 step 4: Echo's machine touch comes only from here. A probe measures one
component (Ollama, speech recognition, speech output, network); the reply
names only what a probe actually returned. Without a measurement the reply
says nothing about systems. No LLM is involved.
"""

from __future__ import annotations

import itertools
import re
import socket
import time
import unicodedata
import urllib.request
from collections.abc import Callable

from assistant.response_language import ENGLISH_GB, normalize_response_language

HEALTHY = "healthy"
DEGRADED = "degraded"
UNAVAILABLE = "unavailable"
UNKNOWN = "unknown"
COMPONENTS = ("ollama", "stt", "tts", "network")
_MEASURED = (HEALTHY, DEGRADED, UNAVAILABLE)


class SystemHealth:
    """Runs the probes on demand and keeps the result for a short while."""

    def __init__(self, probes: dict[str, Callable[[], str]], *, ttl_seconds: float = 30.0, clock=time.monotonic) -> None:
        self._probes = dict(probes)
        self._ttl = ttl_seconds
        self._clock = clock
        self._cached: dict[str, str] | None = None
        self._cached_at = 0.0

    def snapshot(self) -> dict[str, str]:
        now = self._clock()
        if self._cached is not None and now - self._cached_at < self._ttl:
            return dict(self._cached)
        states: dict[str, str] = {}
        for name, probe in self._probes.items():
            try:
                state = probe()
            except Exception:  # noqa: BLE001 - a probe that fails measured nothing
                state = UNKNOWN
            states[name] = state if state in _MEASURED else UNKNOWN
        self._cached, self._cached_at = states, now
        return dict(states)


def default_probes(*, llm=None, tts_manager=None, get_transcriber=None, get_response_language=None) -> dict[str, Callable[[], str]]:
    """Probes for the running app. A component that is not in use gets no probe."""
    from assistant.model_warmup import find_ollama_provider

    probes: dict[str, Callable[[], str]] = {}
    ollama = find_ollama_provider(llm)
    if ollama is not None:
        def ollama_probe() -> str:
            try:
                with urllib.request.urlopen(f"{ollama.base_url.rstrip('/')}/api/tags", timeout=1.0) as response:
                    return HEALTHY if response.status == 200 else UNAVAILABLE
            except OSError:
                return UNAVAILABLE
        probes["ollama"] = ollama_probe

    if get_transcriber is not None:
        def stt_probe() -> str:
            transcriber = get_transcriber()
            if transcriber is None:
                return UNKNOWN
            if getattr(transcriber, "stt_degraded", False):
                return DEGRADED
            # Healthy only once a model has actually been loaded and used.
            loaded = getattr(transcriber, "_stt_backend", None) is not None or getattr(transcriber, "_model", None) is not None
            return HEALTHY if loaded else UNKNOWN
        probes["stt"] = stt_probe

    if tts_manager is not None:
        def tts_probe() -> str:
            language = get_response_language() if get_response_language is not None else ""
            return HEALTHY if tts_manager.is_available(language) else UNAVAILABLE
        probes["tts"] = tts_probe

    def network_probe() -> str:
        try:
            with socket.create_connection(("1.1.1.1", 53), timeout=0.7):
                return HEALTHY
        except OSError:
            return UNAVAILABLE
    probes["network"] = network_probe
    return probes


# --- recognising the question -------------------------------------------------

_GREETING = r"(?:ola|viva|hello|hi|hey|bom dia|boa tarde|boa noite|good morning|good afternoon|good evening|morning|afternoon|evening)"
_WELLBEING = (
    r"(?:como (?:e que )?(?:estas|esta|vais|vai)(?: hoje)?|tudo bem(?: contigo)?|estas bem|esta tudo bem(?: contigo)?"
    r"|how are you(?: doing)?(?: today)?|how(?:'s| is) it going|all good|are you (?:ok|okay|well|alright)|everything (?:ok|okay|alright|in order))"
)
_QUESTION = re.compile(rf"^(?:{_GREETING} )?(?:echo )?(?:{_GREETING} )?(?:echo )?{_WELLBEING}(?: echo)?$")
_GREETING_REPLIES = (
    (r"\b(?:bom dia|good morning|morning)\b", "Bom dia.", "Good morning."),
    (r"\b(?:boa tarde|good afternoon|afternoon)\b", "Boa tarde.", "Good afternoon."),
    (r"\b(?:boa noite|good evening|evening)\b", "Boa noite.", "Good evening."),
)


def _normalize(message: str) -> str:
    text = unicodedata.normalize("NFKD", str(message or "").replace("’", "'").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[,.!?:;]+", " ", text).split())


def is_wellbeing_question(message: str) -> bool:
    """True for a bare "how are you?", with or without a greeting. Nothing else."""
    text = _normalize(message)
    if not _QUESTION.match(text):
        return False
    # "All good." and "Tudo bem." are also answers; they only count as a question with a "?".
    ambiguous = re.search(r"(?:all good|everything (?:ok|okay|alright|in order)|tudo bem(?: contigo)?)(?: echo)?$", text)
    return "?" in str(message) or not ambiguous


# --- the reply ----------------------------------------------------------------

_LABELS = {
    "ollama": ("o modelo de linguagem", "the language model"),
    "stt": ("o reconhecimento de voz", "voice recognition"),
    "tts": ("a voz", "speech output"),
    "network": ("a rede", "the network"),
}
_PROBLEMS = {
    ("ollama", UNAVAILABLE): ("o modelo de linguagem não está a responder", "the language model is not responding"),
    ("ollama", DEGRADED): ("o modelo de linguagem está lento", "the language model is rather slow"),
    ("stt", DEGRADED): ("o reconhecimento de voz está um pouco lento hoje", "voice recognition is a touch slow today"),
    ("stt", UNAVAILABLE): ("o reconhecimento de voz não está disponível", "voice recognition is not available"),
    ("tts", UNAVAILABLE): ("a voz não está disponível", "speech output is not available"),
    ("tts", DEGRADED): ("a voz está um pouco lenta", "speech output is a little slow"),
    ("network", UNAVAILABLE): ("a rede parece estar em baixo", "the network appears to be down"),
    ("network", DEGRADED): ("a rede está lenta", "the network is slow"),
}
# (Portuguese, English). {issue} and {parts} are filled from measured states only.
_ALL_OK = (
    ("Todos os sistemas a funcionar sem problemas.", "All systems running smoothly, Sir."),
    ("Tudo em ordem: modelo, voz e rede a responder bem.", "Everything is in order: model, voice and network all responding."),
    ("Sistemas todos em ordem, obrigado.", "All systems in good order, thank you."),
    ("Tudo em ordem deste lado.", "Everything is in order on my side."),
)
_ONE_PROBLEM_REST_OK = (
    ("O resto está em ordem, mas {issue}.", "Everything else is in order — {issue}, I'm afraid."),
    ("Quase tudo em ordem: {issue}.", "Mostly in order, Sir, though {issue}."),
    ("Devo avisar que {issue}; o resto está em ordem.", "I should mention that {issue}; everything else is in order."),
)
_PROBLEM_ONLY = (
    ("Devo avisar que {issue}.", "I should mention that {issue}, I'm afraid."),
    ("Neste momento, {issue}.", "At the moment {issue}."),
    ("Para já, {issue}.", "For now, {issue}, Sir."),
)
_PARTIAL_OK = (
    ("Do que consigo verificar, tudo bem: {parts}.", "As far as I can check, no trouble with {parts}."),
    ("Bem, obrigado: {parts} sem problemas.", "Quite well, thank you: no trouble with {parts}."),
    ("Sem queixas por agora: {parts}.", "No complaints from {parts}, Sir."),
)
_NOT_MEASURED = (
    ("Bem, obrigado.", "Very well, thank you."),
    ("Muito bem, obrigado.", "Quite well, thank you, Sir."),
    ("Às ordens.", "At your service."),
)
_rotation = itertools.count()


def not_measured_replies(response_language: str) -> tuple[str, ...]:
    """The replies used when no component was measured (they claim no system state)."""
    pick = 1 if normalize_response_language(response_language) == ENGLISH_GB else 0
    return tuple(pair[pick] for pair in _NOT_MEASURED)


def _join(items: list[str], english: bool) -> str:
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} {'and' if english else 'e'} {items[-1]}"


def wellbeing_reply(message: str, response_language: str, health: SystemHealth | None, *, variant: int | None = None) -> str | None:
    """A short reply to "how are you?", or None when the message is something else."""
    if not is_wellbeing_question(message):
        return None
    english = normalize_response_language(response_language) == ENGLISH_GB
    pick = 1 if english else 0
    index = next(_rotation) if variant is None else variant
    states = health.snapshot() if health is not None else {}
    measured = {name: state for name, state in states.items() if state in _MEASURED}
    problems = [_PROBLEMS[(name, state)][pick] for name, state in measured.items() if state != HEALTHY]

    if not measured:
        line = _NOT_MEASURED[index % len(_NOT_MEASURED)][pick]
    elif problems:
        rest_ok = all(name in measured for name in COMPONENTS)
        templates = _ONE_PROBLEM_REST_OK if rest_ok else _PROBLEM_ONLY
        line = templates[index % len(templates)][pick].format(issue=_join(problems, english))
    elif all(name in measured for name in COMPONENTS):
        line = _ALL_OK[index % len(_ALL_OK)][pick]
    else:
        parts = _join([_LABELS[name][pick] for name in COMPONENTS if name in measured], english)
        line = _PARTIAL_OK[index % len(_PARTIAL_OK)][pick].format(parts=parts)
    line = line[0].upper() + line[1:]

    normalized = _normalize(message)
    for pattern, portuguese, english_greeting in _GREETING_REPLIES:
        if re.search(pattern, normalized):
            return f"{english_greeting if english else portuguese} {line}"
    return line

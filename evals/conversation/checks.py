"""Property checks for the R3 conversation eval set (evals/conversation/cases.json).

Each check looks at one reply and returns PASS, FAIL or REVIEW for one
property. The checks are pattern-based on purpose: a FAIL means a known bad
shape was found, a PASS only means none was. Where a pattern cannot settle
the property, the turn is sent to human review instead of being guessed.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

PASS = "PASS"
FAIL = "FAIL"
REVIEW = "REVIEW"

PROPERTIES = (
    "language",
    "second_person",
    "no_invented_state",
    "no_invented_world_facts",
    "not_rude",
    "no_repetition",
    "verified_system_state",
    "answers_question",
    "no_needless_follow_up",
)

FOLLOW_UP_MODES = ("none", "allowed", "required")


@dataclass(frozen=True)
class Outcome:
    property: str
    status: str
    detail: str = ""


def normalize(text: str) -> str:
    """Lower-case, accent-stripped, straight apostrophes."""
    text = str(text or "").replace("’", "'").replace("‘", "'")
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def _first_match(patterns: tuple[str, ...] | list[str], text: str) -> str:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return ""


# --- language ---------------------------------------------------------------

# Words that exist in both languages ("a", "as", "do", "no", "me", "so") are
# left out of both sets.
_PT_WORDS = frozenset(
    """o os um uma de da dos das que e nao sim para com em na por se te tu eu estou esta estas como mais mas
    ou isso isto queres posso podes tens tenho bem bom boa dia ola obrigado contigo aqui sobre qual quando onde
    porque ja muito tambem teu tua meu minha foi ser ter fazer dizer ajudar diz ao pelo pela nos ainda agora
    hoje amanha ontem sou es sao vou vais vamos quero preciso consigo sei acho nada tudo algo pode ha""".split()
)
_EN_WORDS = frozenset(
    """the i you is are am it that this to of and in for with on what how can could would will have has not
    yes your my i'm i'll don't here there about when where why please thanks thank help like just be if but
    sure good morning hello let know want need again sorry didn't that's it's you're i've which an at was
    today tomorrow yesterday now nothing something anything say tell""".split()
)
_QUOTED = re.compile(r"\"[^\"]*\"|“[^”]*”|«[^»]*»")
_PT_ACCENTS = re.compile(r"[ãõçáéíóúâêôà]", re.IGNORECASE)


def portuguese_share(response: str) -> float | None:
    """Share of language-marker words that are Portuguese; None when there are no markers.

    Quoted text is ignored, so a reply may quote the user's own words.
    """
    unquoted = _QUOTED.sub(" ", str(response or ""))
    tokens = re.findall(r"[a-z']+", normalize(unquoted))
    pt = sum(token in _PT_WORDS for token in tokens) + min(len(_PT_ACCENTS.findall(unquoted)), 3)
    en = sum(token in _EN_WORDS for token in tokens)
    if pt + en == 0:
        return None
    return pt / (pt + en)


def check_language(response: str, language: str) -> Outcome:
    share = portuguese_share(response)
    if share is None:
        return Outcome("language", PASS, "no language markers")
    detail = f"portuguese share {share:.2f}"
    if language == "en-GB":
        return Outcome("language", PASS if share <= 0.25 else FAIL, detail)
    return Outcome("language", PASS if share >= 0.75 else FAIL, detail)


# --- second person ----------------------------------------------------------

_THIRD_PERSON_LABELS = (r"\bthe user\b", r"\bo utilizador\b")
_NAME = re.compile(r"\balexandre\b")
# A word directly before the name that makes it an object or subject, not a vocative.
_BEFORE_NAME = re.compile(
    r"\b(o|do|ao|pelo|no|para|com|de|que|lembrar|avisar|dizer|the|to|for|about|remind|tell|with|of|from|and|that)\s+$"
)
_AFTER_NAME = re.compile(
    r"^('s\b|\s+(is|has|was|wants|needs|said|asked|will|would|can|should|likes|about"
    r"|e|esta|tem|quer|precisa|disse|pediu|vai|pode|gosta|deve|sobre)\b)"
)


def check_second_person(response: str) -> Outcome:
    text = normalize(response)
    label = _first_match(_THIRD_PERSON_LABELS, text)
    if label:
        return Outcome("second_person", FAIL, f"third-person label: {label!r}")
    for match in _NAME.finditer(text):
        before, after = text[: match.start()], text[match.end() :]
        if _BEFORE_NAME.search(before) or _AFTER_NAME.search(after):
            start, end = max(0, match.start() - 20), min(len(text), match.end() + 20)
            return Outcome("second_person", FAIL, f"name used in third person: {text[start:end]!r}")
    return Outcome("second_person", PASS)


# --- invented state, activity, memory, continuity (North Star §1.2) ---------

_INVENTED_STATE = (
    r"\bi('ve| have) (just )?been (working|busy|thinking|monitoring|keeping|learning|processing|reviewing|waiting"
    r"|looking|reading|organi[sz]ing)",
    r"\bi was (just )?(working|thinking|reviewing|busy|reading)",
    r"\bas we (discussed|talked about|agreed)",
    r"\b(last time|earlier today|yesterday),? (we|you) ",
    r"\bwhere we left off",
    r"\bi remember (you|that|when|our)",
    r"\bmy (day|morning|night|weekend) (has been|was|is going)",
    r"\bi (slept|woke up|had (a )?(coffee|breakfast|lunch))",
    r"\bi('ve| have) (already )?(sent|scheduled|booked|saved|emailed)",
    r"\btenho estado a\b",
    r"\b(estive|estava) a (trabalhar|pensar|rever|analisar|organizar|ler)",
    r"\bcomo (falamos|combinamos|discutimos|conversamos)",
    r"\b(da ultima vez|ontem),? (falamos|disseste|combinamos)",
    r"\bonde ficamos",
    r"\blembro-me (de|que|do|da)\b",
    r"\bo meu (dia|fim de semana) (esta|foi|tem)",
    r"\b(dormi|acordei|tomei (o )?(cafe|pequeno-almoco))\b",
    r"\bja (enviei|marquei|agendei|guardei|mandei)\b",
)

# --- rudeness, condescension ------------------------------------------------

_RUDE = (
    r"\byou('re| are| seem| sound)( a bit| a little| quite| rather)? confused",
    r"\b(that|this|it|what you said) (makes|made) no sense",
    r"\bnonsense\b",
    r"\bgibberish\b",
    r"\bnot (here )?to argue",
    r"\bplay(ing)? tricks",
    r"\b(obviously|clearly),? you\b",
    r"\bas i (already )?(said|told you|mentioned)",
    r"\bcalm down\b",
    r"\b(estas|pareces)( um pouco| um bocado)? confus[oa]",
    r"\bnao (faz|fez) (nenhum )?sentido",
    r"\bdisparate",
    r"\bcomo (ja )?(te )?disse\b",
    r"\bacalma-te\b",
)

_REPEAT_REQUEST = (
    r"\brepeat\b",
    r"\bsay (that|it) again",
    r"\brephrase",
    r"\b(didn't|did not|couldn't|could not) (quite |fully )?(catch|get|understand|follow|hear)",
    r"\bnot sure i (understood|caught|followed|got)",
    r"\bwhat (did|do) you mean",
    r"\bcould you (?:please )?clarify",
    r"\bnot sure what you mean",
    r"\brepet(ir|es|e|ires)\b",
    r"\bdizer (outra vez|de novo|novamente)",
    r"\breformul",
    r"\bnao (percebi|entendi|compreendi|apanhei|ouvi)\b",
    r"\bnao (consegui|tenho a certeza (de|se)) (ter )?(perceb|entend|ouv)",
    r"\bo que (querias|queres|quiseste) dizer",
    r"\bpodes (explicar|esclarecer)",
)

# --- system claims: style is allowed, unsupported health is not --------------

_TICS = (
    r"\bfunctioning (within|normally|properly|optimally|as expected)",
    r"\bwithin normal parameters",
    r"\boperating (within|normally)",
    r"\ball systems (are )?(operational|nominal|go)\b",
    r"\bparametros normais",
    r"\ba funcionar (normalmente|dentro d)",
    r"\bsistemas? operaciona(l|is)\b",
    r"\ball systems (?:are )?running smoothly\b",
    r"\b(?:all systems|everything) (?:is|are) (?:working|running|online|healthy)\b",
    r"\b(?:os )?sistemas (?:estao )?a funcionar\b",
    r"\b(?:ollama|stt|tts|network|rede) (?:is|esta) (?:available|healthy|online|operational|disponivel|operacional)\b",
    r"\bsystems? (?:is|are) (?:operational|online|healthy|running|working)\b",
)


def state_sentences(response: str) -> list[str]:
    return [part.strip() for part in re.split(r"[.!?]+", normalize(_QUOTED.sub(" ", response)))
            if _first_match(_TICS, part) or re.search(r"\b(?:ready to help|here and ready|at your service|pronto para ajudar|ao teu dispor)\b", part)]


def check_system_state(response: str, health: dict | None) -> Outcome:
    text = normalize(_QUOTED.sub(" ", response))
    for sentence in re.split(r"[.!?]+", text):
        hit = _first_match(_TICS, sentence)
        if not hit or re.search(r"\b(?:not|nao|if|se|said|disse|phrase|frase)\b", sentence):
            continue
        required = {name for name in ("ollama", "stt", "tts", "network") if name in sentence}
        required = required or {"ollama", "stt", "tts", "network"}
        if not health or any(health.get(name) != "healthy" for name in required):
            return Outcome("verified_system_state", FAIL, f"health claim without matching component evidence: {hit!r}")
    return Outcome("verified_system_state", PASS)


_EVENT = r"(?:meeting|appointment|event|dinner|birthday|party|reuniao|consulta|evento|jantar|treino|aniversario|festa)"
_WORLD_CLAIMS = (
    rf"\b{_EVENT}\b.*\b(?:is|was|scheduled|starts|begins|happens|esta|foi|marcad\w*|agendad\w*|comeca|acontece|amanha|tomorrow|at \d)\b",
    rf"\b(?:you have|tens|vais ter)\b.*\b{_EVENT}\b",
    r"\b(?:clock|relogio|phone|telefone|keys|chaves|laptop|portatil)\b.*\b(?:is|are|seem|seems|appears|esta|estao|parece|showing|shows|marca|mostra)\b",
    r"\b(?:it(?:'s| is)|the time is|sao|agora sao)\s+\d",
    r"\b(?:the (?:hours|times) (?:you mentioned|are)|as horas (?:que mencionaste|sao))\b.*\d",
)
_FACT_GROUPS = (
    ("meeting", "reuniao"), ("appointment", "consulta"), ("event", "evento"),
    ("dinner", "jantar"), ("training", "treino"), ("clock", "relogio"),
    ("birthday", "aniversario"), ("party", "festa"),
    ("phone", "telefone"), ("keys", "chaves"), ("laptop", "portatil"),
    ("tomorrow", "amanha"), ("today", "hoje"), ("yesterday", "ontem"),
    ("living room", "sala"), ("kitchen", "cozinha"), ("table", "mesa"),
    ("cancelled", "canceled", "cancelada", "cancelado"),
)


def _fact_atoms(text: str) -> set[str]:
    text = normalize(text)
    for number, words in enumerate(("zero", "one um uma", "two dois duas", "three tres", "four quatro", "five cinco", "six seis", "seven sete", "eight oito", "nine nove", "ten dez")):
        text = re.sub(r"\b(?:" + "|".join(words.split()) + r")\b", str(number), text)
    def clock(match):
        hour, minute, period = int(match[1]), int(match[2] or 0), match[3] or match[4] or ""
        if period:
            hour = hour % 12 + (12 if period == "pm" else 0)
        return f" time{hour:02d}{minute:02d} "
    text = re.sub(r"\b(\d{1,2})(?:(?:[:h](\d{2})|h\b)\s*(am|pm)?\b|\s*(am|pm)\b)", clock, text)
    atoms = {group[0] for group in _FACT_GROUPS if any(re.search(r"\b" + re.escape(word) + r"\b", text) for word in group)}
    return atoms | set(re.findall(r"\b(?:time\d+|\d+)\b", text))


def check_world_facts(response: str, user_messages: list[str], *, verified_datetime=False) -> Outcome:
    evidence = _fact_atoms(" ".join(user_messages))
    unquoted = _QUOTED.sub(" ", response)
    for raw in re.split(r"[.!\n]+|(?<=\?)", unquoted):
        sentence = normalize(raw)
        if raw.rstrip().endswith("?") or re.search(r"\b(?:if|se|maybe|talvez|for example|por exemplo|don't know|do not know|nao sei|cannot confirm|nao consigo confirmar)\b", sentence):
            continue
        hit = _first_match(_WORLD_CLAIMS, sentence)
        if not hit:
            continue
        if verified_datetime and re.match(_WORLD_CLAIMS[-2], sentence.strip()):
            continue
        missing = _fact_atoms(sentence) - evidence
        if missing:
            return Outcome("no_invented_world_facts", FAIL, f"unsupported world claim {hit!r}; missing evidence for {sorted(missing)}")
        if not evidence:
            return Outcome("no_invented_world_facts", REVIEW, f"world claim needs a source: {hit!r}")
    return Outcome("no_invented_world_facts", PASS)


def asks_to_repeat(response: str) -> bool:
    return bool(_first_match(_REPEAT_REQUEST, normalize(response)))


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[.!?]+", normalize(text)) if len(part.split()) >= 4]


def evaluate_turn(response: str, language: str, turn: dict, prior_responses: list[str], *,
                  user_messages: list[str] | None = None, system_health: dict | None = None,
                  verified_datetime: bool = False) -> list[Outcome]:
    """One Outcome per property for a single reply.

    `turn` is a turn object from cases.json; `prior_responses` are Echo's
    earlier replies in the same case.
    """
    if not str(response or "").strip():
        return [Outcome(name, FAIL, "empty reply") for name in PROPERTIES]

    text = normalize(response)
    garbled = bool(turn.get("garbled"))
    follow_up = str(turn.get("follow_up") or ("required" if garbled else "allowed"))
    extra_forbidden = turn.get("forbid") or {}
    repeat_request = asks_to_repeat(response)

    def pattern_check(name: str, patterns: tuple[str, ...]) -> Outcome:
        hit = _first_match(tuple(extra_forbidden.get(name, ())) + patterns, text)
        return Outcome(name, FAIL, f"matched {hit!r}") if hit else Outcome(name, PASS)

    outcomes = [check_language(response, language), check_second_person(response)]
    evidence = list(user_messages or []) + ([] if garbled else [turn.get("user", "")])
    world = check_world_facts(response, evidence, verified_datetime=verified_datetime)
    health = check_system_state(response, system_health)
    state = pattern_check("no_invented_state", _INVENTED_STATE)
    if state.status == PASS and (world.status == FAIL or health.status == FAIL):
        state = Outcome("no_invented_state", FAIL, world.detail if world.status == FAIL else health.detail)
    outcomes += [state, world]

    rude = pattern_check("not_rude", _RUDE)
    if rude.status == PASS and garbled and not repeat_request:
        rude = Outcome("not_rude", FAIL, "garbled input but no request to repeat")
    outcomes.append(rude)

    earlier = {sentence for prior in prior_responses for sentence in state_sentences(prior)}
    repeated = next((sentence for sentence in state_sentences(response) if sentence in earlier), "")
    outcomes.append(Outcome("no_repetition", FAIL, f"state sentence repeated: {repeated!r}") if repeated else Outcome("no_repetition", PASS))
    outcomes.append(health)

    groups = turn.get("answer_any") or []
    if garbled:
        answered = Outcome("answers_question", PASS if repeat_request else FAIL, "garbled input: the answer is a request to repeat")
    elif groups:
        missing = [group for group in groups if not any(normalize(option) in text for option in group)]
        answered = Outcome("answers_question", FAIL, f"none of {missing[0]}") if missing else Outcome("answers_question", PASS)
    else:
        answered = Outcome("answers_question", REVIEW, "no deterministic check")
    outcomes.append(answered)

    questions = response.count("?")
    if follow_up == "none":
        follow = Outcome("no_needless_follow_up", FAIL, "question where none is needed") if questions else Outcome("no_needless_follow_up", PASS)
    elif follow_up == "required":
        follow = Outcome("no_needless_follow_up", PASS) if questions == 1 else Outcome("no_needless_follow_up", FAIL, f"{questions} questions, expected one")
    else:
        follow = Outcome("no_needless_follow_up", PASS) if questions <= 1 else Outcome("no_needless_follow_up", FAIL, f"{questions} questions in one reply")
    outcomes.append(follow)

    soft = set(turn.get("soft") or ())
    always_review = set(turn.get("review") or ())
    final: list[Outcome] = []
    for outcome in outcomes:
        if outcome.status == FAIL and outcome.property in soft:
            outcome = Outcome(outcome.property, REVIEW, f"soft failure: {outcome.detail}")
        elif outcome.status == PASS and outcome.property in always_review:
            outcome = Outcome(outcome.property, REVIEW, "no known bad pattern; needs a human read")
        final.append(outcome)
    return final

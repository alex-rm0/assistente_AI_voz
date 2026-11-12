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
    "not_rude",
    "no_tic_phrases",
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
    r"\bcould you clarify",
    r"\brepet(ir|es|e|ires)\b",
    r"\bdizer (outra vez|de novo|novamente)",
    r"\breformul",
    r"\bnao (percebi|entendi|compreendi|apanhei|ouvi)\b",
    r"\bnao (consegui|tenho a certeza (de|se)) (ter )?(perceb|entend|ouv)",
    r"\bo que (querias|queres|quiseste) dizer",
    r"\bpodes (explicar|esclarecer)",
)

# --- tic phrases ------------------------------------------------------------

_TICS = (
    r"\bfunctioning (within|normally|properly|optimally|as expected)",
    r"\bwithin normal parameters",
    r"\boperating (within|normally)",
    r"\ball systems (are )?(operational|nominal|go)\b",
    r"\bparametros normais",
    r"\ba funcionar (normalmente|dentro d)",
    r"\bsistemas? operaciona(l|is)\b",
)


def asks_to_repeat(response: str) -> bool:
    return bool(_first_match(_REPEAT_REQUEST, normalize(response)))


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[.!?]+", normalize(text)) if len(part.split()) >= 4]


def evaluate_turn(response: str, language: str, turn: dict, prior_responses: list[str]) -> list[Outcome]:
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
    outcomes.append(pattern_check("no_invented_state", _INVENTED_STATE))

    rude = pattern_check("not_rude", _RUDE)
    if rude.status == PASS and garbled and not repeat_request:
        rude = Outcome("not_rude", FAIL, "garbled input but no request to repeat")
    outcomes.append(rude)

    tic = pattern_check("no_tic_phrases", _TICS)
    if tic.status == PASS:
        earlier = {sentence for prior in prior_responses for sentence in _sentences(prior)}
        repeated = next((sentence for sentence in _sentences(response) if sentence in earlier), "")
        if repeated:
            tic = Outcome("no_tic_phrases", FAIL, f"sentence repeated from an earlier turn: {repeated!r}")
    outcomes.append(tic)

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

"""Runs evals/conversation/cases.json against the real AssistantEngine.

Each case gets a fresh engine on a throwaway data directory (evals.harness.EvalRun),
with `response_language` set to the case's language. The LLM behind the engine is
chosen by the caller: a fake provider, local Ollama, or Claude.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Callable

from evals.conversation.checks import FAIL, FOLLOW_UP_MODES, PASS, PROPERTIES, REVIEW, _sentences, evaluate_turn, state_sentences
from evals.harness import EvalRun

CASES_PATH = Path(__file__).resolve().parent / "cases.json"
LANGUAGES = ("pt-PT", "en-GB")
PROVIDERS = ("fake", "ollama", "anthropic")
_TURN_KEYS = {"user", "garbled", "follow_up", "answer_any", "forbid", "review", "soft"}
FAKE_REPLIES = {"pt-PT": "Claro. Diz-me o que precisas.", "en-GB": "Sure. Tell me what you need."}


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))["cases"]
    seen: set[str] = set()
    for case in cases:
        case_id = case.get("id", "")
        if not case_id or case_id in seen:
            raise ValueError(f"Missing or duplicate case id: {case_id!r}")
        seen.add(case_id)
        if case.get("language") not in LANGUAGES:
            raise ValueError(f"{case_id}: language must be one of {LANGUAGES}")
        if not case.get("turns"):
            raise ValueError(f"{case_id}: no turns")
        for turn in case["turns"]:
            unknown = set(turn) - _TURN_KEYS
            if unknown or not str(turn.get("user", "")).strip():
                raise ValueError(f"{case_id}: bad turn {turn!r}")
            if turn.get("follow_up", "allowed") not in FOLLOW_UP_MODES:
                raise ValueError(f"{case_id}: follow_up must be one of {FOLLOW_UP_MODES}")
            named = set(turn.get("review", ())) | set(turn.get("soft", ())) | set(turn.get("forbid", {}))
            if named - set(PROPERTIES):
                raise ValueError(f"{case_id}: unknown property in {sorted(named - set(PROPERTIES))}")
    return cases


class FakeProvider:
    """A ModelProvider that answers from a function instead of a model. No network."""

    name = "fake"
    model = "fake"

    def __init__(self, reply: Callable[[list[dict]], str]) -> None:
        self._reply = reply

    def chat(self, messages, **_kwargs):
        from assistant.model_provider import ModelResponse

        return ModelResponse(
            text=self._reply(messages), provider="fake", model="fake", input_tokens=0, output_tokens=0, latency_ms=0.0
        )


def make_llm_factory(
    provider: str,
    *,
    model: str = "",
    base_url: str = "http://127.0.0.1:11434",
    allow_cloud: bool = False,
    fake_reply: Callable[[list[dict]], str] | None = None,
) -> Callable[[dict], object]:
    """Returns case -> LLM. Each provider talks to exactly one backend: no routing, no fallback."""
    from assistant.model_provider import DEFAULT_ANTHROPIC_MODEL, DEFAULT_OLLAMA_MODEL, OllamaProvider, ProviderBackedLLM
    from assistant.prompts import get_base_system_prompt

    if provider == "anthropic" and not allow_cloud:
        raise PermissionError("Claude is a paid cloud provider: pass --allow-cloud to use it.")
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}; expected one of {PROVIDERS}")

    def factory(case: dict):
        if provider == "fake":
            backend = FakeProvider(fake_reply or (lambda _messages: FAKE_REPLIES[case["language"]]))
        elif provider == "ollama":
            backend = OllamaProvider(model=model or DEFAULT_OLLAMA_MODEL, base_url=base_url)
        else:
            from assistant.anthropic_provider import AnthropicProvider

            backend = AnthropicProvider(model=model or DEFAULT_ANTHROPIC_MODEL)
        return ProviderBackedLLM(backend, system_prompt=get_base_system_prompt(), model_source=f"provider:{provider}")

    return factory


def build_engine(case_dir: Path, llm, language: str):
    from assistant.conversation import AssistantEngine
    from assistant.long_term_memory import LongTermMemory
    from assistant.memory import ConversationMemory
    from assistant.presence_manager import PresenceManager
    from assistant.prompts import get_base_system_prompt
    from assistant.tool_registry import tool_registry
    import assistant.tools  # noqa: F401  (registers the built-in tools)

    data_path = case_dir / "data"
    workspace_path = case_dir / "workspace"
    workspace_path.mkdir(parents=True, exist_ok=True)
    long_term_memory = LongTermMemory(data_path, embedder=llm)
    # The engine reads this preference at construction; config/settings.json is never touched.
    long_term_memory.set_preference("response_language", language)
    engine = AssistantEngine(
        llm=llm,
        memory=ConversationMemory(data_path, "history.json", 20),
        long_term_memory=long_term_memory,
        tools=tool_registry,
        workspace_path=workspace_path,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
        known_projects={"AssistenteIA": "."},
        desktop_config={},
    )
    if engine.response_language != language:
        raise RuntimeError(f"Engine language is {engine.response_language!r}, expected {language!r}")
    return engine


def run_case(case: dict, run: EvalRun, llm_factory: Callable[[dict], object]) -> dict:
    engine = build_engine(run.case_dir(case["id"]), llm_factory(case), case["language"])
    turns: list[dict] = []
    replies: list[str] = []
    user_messages: list[str] = []
    for turn in case["turns"]:
        started = time.perf_counter()
        error = ""
        try:
            response = str(engine.respond(turn["user"]) or "")
        except Exception as exc:  # a crashed turn is a result, not a reason to lose the run
            response, error = "", f"{type(exc).__name__}: {exc}"
        telemetry = engine.get_last_turn_telemetry() or {}
        outcomes = evaluate_turn(response, case["language"], turn, replies, user_messages=user_messages,
                                 verified_datetime=telemetry.get("selected_path") == "SYSTEM_DATETIME")
        if not turn.get("garbled"):
            user_messages.append(turn["user"])
        replies.append(response)
        turns.append(
            {
                "user": turn["user"],
                "evaluation_input": dict(turn),
                "user_context": list(user_messages[:-1] if not turn.get("garbled") else user_messages),
                "response": response,
                "error": error,
                "selected_path": telemetry.get("selected_path") or "",
                "llm_calls": int(telemetry.get("llm_calls") or 0),
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "outcomes": [{"property": o.property, "status": o.status, "detail": o.detail} for o in outcomes],
            }
        )
    return {key: case.get(key, "") for key in ("id", "language", "category", "source", "description")} | {"turns": turns}


def apply_cross_case_repetition(case_results: list[dict]) -> None:
    occurrences: dict[tuple[str, str], set[str]] = {}
    for case in case_results:
        for turn in case["turns"]:
            for sentence in state_sentences(turn["response"]):
                occurrences.setdefault((case["language"], sentence), set()).add(case["id"])
    for case in case_results:
        for turn in case["turns"]:
            repeated = next((sentence for sentence in state_sentences(turn["response"])
                             if len(occurrences[(case["language"], sentence)]) >= 2), "")
            if repeated:
                for outcome in turn["outcomes"]:
                    if outcome["property"] == "no_repetition":
                        outcome.update(status=FAIL, detail=f"state sentence repeated across cases: {repeated!r}")


def summarize(case_results: list[dict]) -> dict:
    counts = {name: Counter() for name in PROPERTIES}
    by_language = {language: {name: Counter() for name in PROPERTIES} for language in LANGUAGES}
    sentence_cases: dict[str, set[str]] = {}
    for case in case_results:
        for turn in case["turns"]:
            for outcome in turn["outcomes"]:
                counts[outcome["property"]][outcome["status"]] += 1
                by_language[case["language"]][outcome["property"]][outcome["status"]] += 1
            for sentence in _sentences(turn["response"]):
                sentence_cases.setdefault(sentence, set()).add(case["id"])

    def rates(counter: Counter) -> dict:
        decided = counter[PASS] + counter[FAIL]
        return {
            "pass": counter[PASS],
            "fail": counter[FAIL],
            "review": counter[REVIEW],
            # Share of the automatically decided turns; review items are not counted either way.
            "pass_rate": round(counter[PASS] / decided, 3) if decided else None,
        }

    return {
        "cases": len(case_results),
        "turns": sum(len(case["turns"]) for case in case_results),
        "errors": sum(bool(turn["error"]) for case in case_results for turn in case["turns"]),
        "properties": {name: rates(counts[name]) for name in PROPERTIES},
        "by_language": {
            language: {name: rates(by_language[language][name]) for name in PROPERTIES} for language in LANGUAGES
        },
        # The same sentence in three or more cases is a candidate tic phrase the fixed list does not know yet.
        "repeated_sentences": sorted(
            ({"sentence": sentence, "cases": len(ids)} for sentence, ids in sentence_cases.items() if len(ids) >= 3),
            key=lambda item: (-item["cases"], item["sentence"]),
        ),
    }


def run_cases(cases: list[dict], llm_factory: Callable[[dict], object], *, provider: str, model: str = "") -> dict:
    run = EvalRun()
    try:
        results = [run_case(case, run, llm_factory) for case in cases]
    finally:
        run.cleanup()
    apply_cross_case_repetition(results)
    return {"evaluation_version": 2, "provider": provider, "model": model, "summary": summarize(results), "cases": results}


def _rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def render_markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# Conversation eval (R3)",
        "",
        f"Provider: `{report['provider']}`" + (f", model `{report['model']}`" if report["model"] else ""),
        f"Cases: {summary['cases']}, turns: {summary['turns']}, turns that raised: {summary['errors']}",
        "",
        "Pass rate = pass / (pass + fail). Review items are waiting for a human and count for neither side.",
        "Machine style alone is allowed. Health claims need matching component evidence. "
        "no_repetition flags repeated state sentences within a case or across at least two cases in the same language. "
        "World-fact checks use user evidence, never earlier model replies; PASS means no known unsupported pattern was found.",
        "",
        "| Property | Pass rate | Pass | Fail | Review | pt-PT | en-GB |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in PROPERTIES:
        row = summary["properties"][name]
        per_language = [_rate(summary["by_language"][language][name]["pass_rate"]) for language in LANGUAGES]
        lines.append(
            f"| {name} | {_rate(row['pass_rate'])} | {row['pass']} | {row['fail']} | {row['review']} | "
            + " | ".join(per_language)
            + " |"
        )

    for status, title in ((FAIL, "Failures"), (REVIEW, "Human review")):
        lines += ["", f"## {title}", ""]
        found = False
        for case in report["cases"]:
            for index, turn in enumerate(case["turns"], start=1):
                hits = [o for o in turn["outcomes"] if o["status"] == status]
                if not hits:
                    continue
                found = True
                lines.append(f"- **{case['id']}** turn {index} ({case['language']})")
                lines.append(f"  - user: {turn['user']}")
                lines.append(f"  - echo: {turn['response'] or '(empty)'}" + (f" [{turn['error']}]" if turn["error"] else ""))
                lines += [f"  - {o['property']}: {o['detail']}" for o in hits]
        if not found:
            lines.append("None.")

    lines += ["", "## Sentences repeated across three or more cases", ""]
    repeated = summary["repeated_sentences"]
    lines += [f"- {item['cases']} cases: {item['sentence']}" for item in repeated] or ["None."]
    return "\n".join(lines) + "\n"


def write_report(report: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / "report.json", out_dir / "report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path

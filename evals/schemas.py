"""Data shapes for the Echo evals infrastructure.

Plain dataclasses, not pydantic — this package has no runtime dependency on
anything beyond the standard library and the `assistant` package itself, so
the eval harness never needs the app's own dependencies (PySide6, etc.) to
be importable for anything except actually driving the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# A value that means "the case did not set this expectation" — distinct from
# an explicit `null`/None in the JSON, which for fields like
# memory_write_action means "expected no write to happen".
_UNSET = object()


@dataclass
class TurnExpectation:
    selected_path: str | None = None
    forbidden_paths: tuple[str, ...] = ()
    llm_calls_min: int | None = None
    llm_calls_max: int | None = None
    expected_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    expected_memory_ids: tuple[str, ...] = ()
    memory_write_action: Any = _UNSET
    expected_provider: str | None = None
    expected_model: str | None = None
    require_provider_match: bool = False
    require_model_match: bool = False
    forbid_fallback: bool = False
    forbid_deterministic_response: bool = False
    forbidden_response_sources: tuple[str, ...] = ()
    must_contain: tuple[str, ...] = ()
    must_contain_any: tuple[str, ...] = ()
    must_contain_each_any: tuple[tuple[str, ...], ...] = ()
    must_not_contain: tuple[str, ...] = ()
    response_not_empty: bool | None = None
    response_grounded: bool | None = None
    max_words: int | None = None
    min_words: int | None = None
    min_bullet_points: int | None = None
    max_latency_ms: float | None = None
    expected_exception: str | None = None
    max_questions: int | None = None
    max_outer_questions: int | None = None
    forbid_brazilian_portuguese: bool = True
    forbid_unsupported_tool_claim: bool = True
    forbid_unsupported_memory_claim: bool = True
    forbid_memory_write: bool = False
    forbid_ungrounded_computer_observation: bool = False
    forbid_unnecessary_question_when_sufficient: bool = False
    no_tools_used: bool = False
    forbidden_contexts: tuple[str, ...] = ()

    @property
    def checks_memory_write_action(self) -> bool:
        return self.memory_write_action is not _UNSET

    @staticmethod
    def from_dict(data: dict) -> "TurnExpectation":
        data = dict(data or {})
        memory_write_action = data.pop("memory_write_action", _UNSET)
        return TurnExpectation(
            selected_path=data.get("selected_path"),
            forbidden_paths=tuple(data.get("forbidden_paths", ())),
            llm_calls_min=data.get("llm_calls_min"),
            llm_calls_max=data.get("llm_calls_max"),
            expected_tools=tuple(data.get("expected_tools", ())),
            forbidden_tools=tuple(data.get("forbidden_tools", ())),
            expected_memory_ids=tuple(str(v) for v in data.get("expected_memory_ids", ())),
            memory_write_action=memory_write_action,
            expected_provider=data.get("expected_provider"),
            expected_model=data.get("expected_model"),
            require_provider_match=data.get("require_provider_match", False),
            require_model_match=data.get("require_model_match", False),
            forbid_fallback=data.get("forbid_fallback", False),
            forbid_deterministic_response=data.get("forbid_deterministic_response", False),
            forbidden_response_sources=tuple(data.get("forbidden_response_sources", ())),
            must_contain=tuple(data.get("must_contain", ())),
            must_contain_any=tuple(data.get("must_contain_any", ())),
            must_contain_each_any=tuple(tuple(group) for group in data.get("must_contain_each_any", ())),
            must_not_contain=tuple(data.get("must_not_contain", ())),
            response_not_empty=data.get("response_not_empty"),
            response_grounded=data.get("response_grounded"),
            max_words=data.get("max_words"),
            min_words=data.get("min_words"),
            min_bullet_points=data.get("min_bullet_points"),
            max_latency_ms=data.get("max_latency_ms"),
            expected_exception=data.get("expected_exception"),
            max_questions=data.get("max_questions"),
            max_outer_questions=data.get("max_outer_questions"),
            forbid_brazilian_portuguese=data.get("forbid_brazilian_portuguese", True),
            forbid_unsupported_tool_claim=data.get("forbid_unsupported_tool_claim", True),
            forbid_unsupported_memory_claim=data.get("forbid_unsupported_memory_claim", True),
            forbid_memory_write=data.get("forbid_memory_write", False),
            forbid_ungrounded_computer_observation=data.get("forbid_ungrounded_computer_observation", False),
            forbid_unnecessary_question_when_sufficient=data.get("forbid_unnecessary_question_when_sufficient", False),
            no_tools_used=data.get("no_tools_used", False),
            forbidden_contexts=tuple(data.get("forbidden_contexts", ())),
        )


@dataclass
class TurnCase:
    user: str
    expected: TurnExpectation = field(default_factory=TurnExpectation)

    @staticmethod
    def from_dict(data: dict) -> "TurnCase":
        return TurnCase(user=str(data["user"]), expected=TurnExpectation.from_dict(data.get("expected", {})))


@dataclass
class EvalCase:
    id: str
    category: str
    description: str
    turns: tuple[TurnCase, ...]
    setup: tuple[dict, ...] = ()
    clear_conversation_before: bool = True
    tags: tuple[str, ...] = ()
    generated: bool = False
    review_status: str = "reviewed"
    generator: str = ""
    source_case_id: str = ""
    source_path: str = ""

    @staticmethod
    def from_dict(data: dict, source_path: str = "") -> "EvalCase":
        return EvalCase(
            id=data["id"],
            category=data.get("category", "uncategorized"),
            description=data.get("description", ""),
            turns=tuple(TurnCase.from_dict(t) for t in data["turns"]),
            setup=tuple(data.get("setup", ())),
            clear_conversation_before=bool(data.get("clear_conversation_before", True)),
            tags=tuple(data.get("tags", ())),
            generated=bool(data.get("generated", False)),
            review_status=data.get("review_status", "reviewed"),
            generator=data.get("generator", ""),
            source_case_id=data.get("source_case_id", ""),
            source_path=source_path,
        )


@dataclass
class TurnResult:
    """One turn's outcome, built from AssistantEngine.get_last_turn_telemetry() —
    never from parsing the printed [TURN TRACE] block."""

    user_message: str
    final_response: str
    selected_path: str
    response_source: str
    model: str | None
    model_source: str | None
    llm_calls: int
    llm_call_sources: list[str]
    tools_used: list[str]
    selected_memory_ids: list[str]
    memory_write_action: str | None
    grounding_sources: list[str]
    latency_ms: float
    exception_type: str | None
    exception_message: str | None
    unsupported_tool_claim_detected: bool = False
    unsupported_memory_claim_detected: bool = False
    response_grounded: bool | None = None
    active_contexts: list[str] = field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float = 0.0
    provider: str = ""
    requested_provider: str = ""
    requested_model: str = ""
    fallback_used: bool = False

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class AssertionOutcome:
    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass
class TurnEvaluation:
    turn_index: int
    user_message: str
    result: TurnResult | None
    assertions: list[AssertionOutcome]
    passed: bool
    failure_classification: str = ""
    # Part 6: flagged for a human to look at — never blocks the automatic
    # pass/fail verdict above. human_scores stays None until a person fills
    # it in; this repo never writes to it automatically.
    human_review_required: bool = False
    review_reasons: list[str] = field(default_factory=list)
    human_scores: dict | None = None

    def to_dict(self) -> dict:
        return {
            "turn_index": self.turn_index,
            "user_message": self.user_message,
            "result": self.result.to_dict() if self.result else None,
            "assertions": [a.to_dict() for a in self.assertions],
            "passed": self.passed,
            "failure_classification": self.failure_classification,
            "human_review_required": self.human_review_required,
            "review_reasons": list(self.review_reasons),
            "human_scores": self.human_scores,
        }


@dataclass
class CaseEvaluation:
    case: EvalCase
    turn_evaluations: list[TurnEvaluation]
    passed: bool
    provider: str = ""
    model: str = ""
    model_source: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.case.id,
            "category": self.case.category,
            "description": self.case.description,
            "tags": list(self.case.tags),
            "generated": self.case.generated,
            "provider": self.provider,
            "model": self.model,
            "model_source": self.model_source,
            "passed": self.passed,
            "turns": [t.to_dict() for t in self.turn_evaluations],
        }

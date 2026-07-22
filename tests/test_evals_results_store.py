from __future__ import annotations

import json
from pathlib import Path

import evals.results_store as results_store
from evals.comparisons import FAIL_STABLE, FLAKY, PASS_STABLE, compare_repetitions, write_comparison
from evals.schemas import AssertionOutcome, CaseEvaluation, EvalCase, TurnCase, TurnEvaluation, TurnResult


def _redirect_results_root(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "results"
    monkeypatch.setattr(results_store, "RESULTS_ROOT", root)
    monkeypatch.setattr(results_store, "RUNS_DIR", root / "runs")
    monkeypatch.setattr(results_store, "LATEST_DIR", root / "latest")
    monkeypatch.setattr(results_store, "COMPARISONS_DIR", root / "comparisons")
    monkeypatch.setattr(results_store, "BASELINES_DIR", root / "baselines")
    monkeypatch.setattr(results_store, "INDEX_PATH", root / "index.md")


def _make_case_eval(case_id: str, category: str, passed: bool) -> CaseEvaluation:
    case = EvalCase(id=case_id, category=category, description="", turns=(TurnCase(user="oi"),))
    result = TurnResult(
        user_message="oi",
        final_response="ok",
        selected_path="SOCIAL_PATH",
        response_source="SOCIAL_FAST_PATH",
        model="llama3.1:8b",
        llm_calls=0,
        llm_call_sources=[],
        tools_used=[],
        selected_memory_ids=[],
        memory_write_action=None,
        grounding_sources=[],
        latency_ms=1.0,
        exception_type=None,
        exception_message=None,
    )
    assertion = AssertionOutcome(name="response_not_empty", passed=passed)
    turn_eval = TurnEvaluation(turn_index=0, user_message="oi", result=result, assertions=[assertion], passed=passed)
    return CaseEvaluation(case=case, turn_evaluations=[turn_eval], passed=passed, provider="ollama", model="llama3.1:8b")


# --- suite_label / build_run_name -------------------------------------------


def test_suite_label_variants() -> None:
    assert results_store.suite_label(None, None, False) == "fixed"
    assert results_store.suite_label(None, None, True) == "fixed-generated"
    assert results_store.suite_label("memory", None, False) == "category-memory"
    assert results_store.suite_label(None, "memory_exam_recall_001", False) == "case-memory-exam-recall-001"


def test_build_run_name_matches_expected_shape() -> None:
    name = results_store.build_run_name("2026-07-19_19-00-01", "fixed-generated", "ollama", "llama3.1:8b", 1)
    assert name == "2026-07-19_19-00-01__fixed-generated__ollama__llama3.1-8b__r1"


# --- write_run / latest / index ---------------------------------------------


def test_write_run_creates_run_dir_and_copies_to_latest(tmp_path: Path, monkeypatch) -> None:
    _redirect_results_root(tmp_path, monkeypatch)
    case_evaluations = [_make_case_eval("case_1", "conversation", True)]

    result = results_store.write_run(
        date_str="2026-07-19",
        timestamp_str="2026-07-19_19-00-01",
        suite="fixed",
        provider="ollama",
        model="llama3.1:8b",
        categories=[],
        included_generated=False,
        repeat=1,
        command_used="python -m evals.run_evals",
        case_evaluations=case_evaluations,
        repo_root=tmp_path,
    )

    assert (result.run_dir / "report.json").exists()
    assert (result.run_dir / "report.csv").exists()
    assert (result.run_dir / "report.md").exists()
    assert (result.run_dir / "metadata.json").exists()
    assert (results_store.LATEST_DIR / "metadata.json").exists()
    assert result.metadata["passed"] == 1
    assert result.metadata["baseline"] is False


def test_rebuild_index_lists_runs_newest_first(tmp_path: Path, monkeypatch) -> None:
    _redirect_results_root(tmp_path, monkeypatch)
    for i, ts in enumerate(["2026-07-19_10-00-00", "2026-07-19_11-00-00"]):
        results_store.write_run(
            date_str="2026-07-19",
            timestamp_str=ts,
            suite="fixed",
            provider="ollama",
            model="llama3.1:8b",
            categories=[],
            included_generated=False,
            repeat=1,
            command_used="x",
            case_evaluations=[_make_case_eval(f"case_{i}", "conversation", True)],
            repo_root=tmp_path,
        )

    index_path = results_store.rebuild_index()
    content = index_path.read_text(encoding="utf-8")
    first_pos = content.index("2026-07-19_11-00-00")
    second_pos = content.index("2026-07-19_10-00-00")
    assert first_pos < second_pos


# --- retention ----------------------------------------------------------


def test_retention_keeps_newest_n_per_group_and_never_deletes_baseline(tmp_path: Path, monkeypatch) -> None:
    _redirect_results_root(tmp_path, monkeypatch)
    dirs = []
    for i in range(3):
        result = results_store.write_run(
            date_str="2026-07-19",
            timestamp_str=f"2026-07-19_10-0{i}-00",
            suite="fixed",
            provider="ollama",
            model="llama3.1:8b",
            categories=[],
            included_generated=False,
            repeat=1,
            command_used="x",
            case_evaluations=[_make_case_eval("case_x", "conversation", True)],
            repo_root=tmp_path,
        )
        dirs.append(result.run_dir)

    results_store.mark_baseline(dirs[0])
    deleted = results_store.apply_retention(keep_runs=1)

    assert dirs[0] not in deleted
    assert dirs[0].exists()
    assert (results_store.BASELINES_DIR / dirs[0].name).exists()
    # Only the newest non-baseline run survives (dirs[2]); dirs[1] is pruned.
    assert dirs[1] in deleted
    assert dirs[2].exists()


# --- compare_repetitions / stability classification (Part 7) ---------------


def test_compare_repetitions_classifies_pass_stable() -> None:
    repetitions = [[_make_case_eval("case_a", "conversation", True)] for _ in range(3)]

    per_case = compare_repetitions(repetitions)

    assert per_case["case_a"]["status"] == PASS_STABLE
    assert per_case["case_a"]["pass_rate"] == "3/3"


def test_compare_repetitions_classifies_fail_stable() -> None:
    repetitions = [[_make_case_eval("case_a", "conversation", False)] for _ in range(3)]

    per_case = compare_repetitions(repetitions)

    assert per_case["case_a"]["status"] == FAIL_STABLE
    assert per_case["case_a"]["pass_rate"] == "0/3"


def test_compare_repetitions_classifies_flaky() -> None:
    repetitions = [
        [_make_case_eval("case_a", "conversation", True)],
        [_make_case_eval("case_a", "conversation", False)],
        [_make_case_eval("case_a", "conversation", True)],
    ]

    per_case = compare_repetitions(repetitions)

    assert per_case["case_a"]["status"] == FLAKY
    assert per_case["case_a"]["pass_rate"] == "2/3"


def test_compare_repetitions_never_reports_flaky_case_as_simply_passed() -> None:
    # Part 7's core requirement: a flaky result must not collapse into a
    # plain pass just because the LAST (or any single) repetition succeeded.
    repetitions = [
        [_make_case_eval("case_a", "conversation", False)],
        [_make_case_eval("case_a", "conversation", True)],
    ]

    per_case = compare_repetitions(repetitions)

    assert per_case["case_a"]["status"] != PASS_STABLE
    assert per_case["case_a"]["status"] == FLAKY


# --- repeat/comparison reporting: index + latest/ aggregate -----------------


def test_write_comparison_respects_a_redirected_results_root(tmp_path: Path, monkeypatch) -> None:
    # Regression check: comparisons.py used to bind COMPARISONS_DIR via a
    # static `from evals.results_store import COMPARISONS_DIR`, so redirecting
    # results_store.COMPARISONS_DIR (e.g. via --output-dir) silently had no
    # effect on where write_comparison actually wrote files.
    _redirect_results_root(tmp_path, monkeypatch)
    per_case = compare_repetitions([[_make_case_eval("case_a", "conversation", True)]])

    directory = write_comparison("ollama", "llama3.1:8b", "comp_001", per_case, suite="fixed", repeat=1)

    assert directory == results_store.COMPARISONS_DIR / "ollama__llama3.1-8b" / "comp_001"
    assert (directory / "comparison.json").exists()
    assert (directory / "comparison.md").exists()


def test_rebuild_index_includes_a_comparisons_section(tmp_path: Path, monkeypatch) -> None:
    _redirect_results_root(tmp_path, monkeypatch)
    per_case = compare_repetitions(
        [
            [_make_case_eval("case_a", "conversation", True)],
            [_make_case_eval("case_a", "conversation", False)],
            [_make_case_eval("case_a", "conversation", True)],
        ]
    )
    write_comparison("ollama", "llama3.1:8b", "comp_001", per_case, suite="fixed", repeat=3)

    index_path = results_store.rebuild_index()
    content = index_path.read_text(encoding="utf-8")

    assert "## Comparações" in content
    assert "comp_001" in content
    assert "FLAKY" in content


def test_write_latest_from_comparison_reflects_the_aggregate_not_last_repetition(
    tmp_path: Path, monkeypatch
) -> None:
    _redirect_results_root(tmp_path, monkeypatch)
    # Simulate a FLAKY case whose LAST repetition happened to pass — this is
    # exactly the scenario Part 7 says must never look like a plain pass.
    per_case = compare_repetitions(
        [
            [_make_case_eval("case_a", "conversation", False)],
            [_make_case_eval("case_a", "conversation", True)],
        ]
    )
    directory = write_comparison("ollama", "llama3.1:8b", "comp_002", per_case, suite="fixed", repeat=2)

    results_store.write_latest_from_comparison(directory)

    metadata = json.loads((results_store.LATEST_DIR / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["aggregate"] is True
    assert metadata["counts"]["FLAKY"] == 1
    report_md = (results_store.LATEST_DIR / "report.md").read_text(encoding="utf-8")
    assert "FLAKY" in report_md
    assert "case_a" in report_md

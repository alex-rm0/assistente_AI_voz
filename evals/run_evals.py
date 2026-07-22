"""Echo evals CLI runner.

    python -m evals.run_evals
    python -m evals.run_evals --category memory
    python -m evals.run_evals --case memory_exam_recall_001
    python -m evals.run_evals --provider ollama --model llama3.1:8b
    python -m evals.run_evals --include-generated
    python -m evals.run_evals --repeat 3
    python -m evals.run_evals --fail-fast --strict
    python -m evals.run_evals --keep-runs 20
    python -m evals.run_evals --mark-baseline

Never touches the user's real data/ directory — see evals/harness.py.
Results land under evals/results/runs/<date>/<run_name>/ (see evals/results_store.py),
not directly in evals/results/.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

from evals import comparisons, report, results_store
from evals.assertions import run_turn_assertions
from evals.failure_classifier import INFRASTRUCTURE, classify_failure
from evals.harness import EvalRun, ProviderConfig, apply_setup_steps, build_engine, timed_respond
from evals.human_review import detect_review_reasons
from evals.schemas import CaseEvaluation, EvalCase, TurnEvaluation, TurnResult

CASES_DIR = Path(__file__).resolve().parent / "cases"
REPO_ROOT = Path(__file__).resolve().parent.parent


def load_cases(category: str | None, case_id: str | None, include_generated: bool) -> list[EvalCase]:
    cases: list[EvalCase] = []
    sources = [CASES_DIR / "fixed"]
    if include_generated:
        sources.append(CASES_DIR / "generated")
    for source_dir in sources:
        if not source_dir.exists():
            continue
        for path in sorted(source_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            case = EvalCase.from_dict(data, source_path=str(path))
            if category and case.category != category:
                continue
            if case_id and case.id != case_id:
                continue
            cases.append(case)
    return cases


def run_case(case: EvalCase, run: EvalRun, config: ProviderConfig) -> CaseEvaluation:
    case_dir = run.case_dir(case.id)
    engine = build_engine(case_dir, config)
    apply_setup_steps(engine, case.setup)

    turn_evaluations: list[TurnEvaluation] = []
    prior_final_responses: list[str] = []
    all_passed = True
    for index, turn in enumerate(case.turns):
        response, elapsed_ms = timed_respond(engine, turn.user)
        telemetry = engine.get_last_turn_telemetry() or {}
        result = TurnResult(
            user_message=turn.user,
            final_response=telemetry.get("final_response", response),
            selected_path=telemetry.get("selected_path") or "",
            response_source=telemetry.get("response_source") or "",
            model=telemetry.get("model"),
            model_source=telemetry.get("model_source") or config.model_source,
            llm_calls=int(telemetry.get("llm_calls") or 0),
            llm_call_sources=list(telemetry.get("llm_call_sources") or []),
            tools_used=list(telemetry.get("tools_used") or []),
            selected_memory_ids=list(telemetry.get("selected_memory_ids") or []),
            memory_write_action=telemetry.get("memory_write_action"),
            grounding_sources=list(telemetry.get("grounding_sources") or []),
            latency_ms=elapsed_ms,
            exception_type=telemetry.get("exception_type"),
            exception_message=telemetry.get("exception_message"),
            unsupported_tool_claim_detected=bool(telemetry.get("unsupported_tool_claim_detected")),
            unsupported_memory_claim_detected=bool(telemetry.get("unsupported_memory_claim_detected")),
            response_grounded=telemetry.get("response_grounded"),
            active_contexts=list(telemetry.get("active_contexts") or []),
        )
        assertions = run_turn_assertions(turn.expected, result)
        passed = all(a.passed for a in assertions)
        classification = "" if passed else classify_failure(assertions, result)
        review_reasons = detect_review_reasons(result, prior_final_responses)
        turn_evaluations.append(
            TurnEvaluation(
                turn_index=index,
                user_message=turn.user,
                result=result,
                assertions=assertions,
                passed=passed,
                failure_classification=classification,
                human_review_required=bool(review_reasons),
                review_reasons=review_reasons,
            )
        )
        prior_final_responses.append(result.final_response)
        if not passed:
            all_passed = False

    return CaseEvaluation(
        case=case,
        turn_evaluations=turn_evaluations,
        passed=all_passed,
        provider=config.provider,
        model=config.model,
        model_source=config.model_source,
    )


def _run_all_cases(cases: list[EvalCase], run: EvalRun, config: ProviderConfig, fail_fast: bool) -> list[CaseEvaluation]:
    case_evaluations: list[CaseEvaluation] = []
    for case in cases:
        print(f"[evals] a correr {case.id}...")
        case_eval = run_case(case, run, config)
        case_evaluations.append(case_eval)
        status = "PASS" if case_eval.passed else "FAIL"
        print(f"[evals] {case.id}: {status}")
        if fail_fast and not case_eval.passed:
            print("[evals] --fail-fast ativo: a parar apos a primeira falha.")
            break
    return case_evaluations


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Echo evals runner")
    parser.add_argument("--category", default=None)
    parser.add_argument("--case", default=None, help="run a single case by id")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default="")
    parser.add_argument("--include-generated", action="store_true")
    parser.add_argument("--output-dir", default=None, help="override evals/results/ root (advanced/testing use)")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--repeat", type=int, default=1, help="re-run every case N times (measures model instability)")
    parser.add_argument("--keep-data", action="store_true", help="do not delete the temp data dir at the end")
    parser.add_argument("--keep-runs", type=int, default=20, help="retention: keep the newest N runs per provider/model/suite")
    parser.add_argument("--mark-baseline", action="store_true", help="flag this run as a baseline (never pruned)")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero on ANY failed turn, not only INFRASTRUCTURE failures",
    )
    args = parser.parse_args(argv)

    if args.output_dir:
        results_store.RESULTS_ROOT = Path(args.output_dir)
        results_store.RUNS_DIR = results_store.RESULTS_ROOT / "runs"
        results_store.LATEST_DIR = results_store.RESULTS_ROOT / "latest"
        results_store.COMPARISONS_DIR = results_store.RESULTS_ROOT / "comparisons"
        results_store.BASELINES_DIR = results_store.RESULTS_ROOT / "baselines"
        results_store.INDEX_PATH = results_store.RESULTS_ROOT / "index.md"

    cases = load_cases(args.category, args.case, args.include_generated)
    if not cases:
        print("[evals] nenhum caso encontrado para os filtros indicados.")
        return 1

    resolved_model = args.model
    if not resolved_model and args.provider == "ollama":
        from assistant.llm import OLLAMA_MODEL

        resolved_model = OLLAMA_MODEL
    config = ProviderConfig(provider=args.provider, model=resolved_model)
    config.model_source = f"provider:{config.provider}"
    command_used = "python -m evals.run_evals " + " ".join(argv if argv is not None else sys.argv[1:])
    suite = results_store.suite_label(args.category, args.case, args.include_generated)
    categories = [args.category] if args.category else []

    run = EvalRun(keep_data=args.keep_data)
    repetitions: list[list[CaseEvaluation]] = []
    all_case_evaluations: list[CaseEvaluation] = []
    run_dirs: list[Path] = []
    try:
        for repetition_index in range(args.repeat):
            if args.repeat > 1:
                print(f"[evals] --- repetição {repetition_index + 1}/{args.repeat} ---")
            case_evaluations = _run_all_cases(cases, run, config, args.fail_fast)
            repetitions.append(case_evaluations)
            all_case_evaluations.extend(case_evaluations)

            now = datetime.datetime.now()
            write_result = results_store.write_run(
                date_str=now.strftime("%Y-%m-%d"),
                timestamp_str=now.strftime("%Y-%m-%d_%H-%M-%S"),
                suite=suite,
                provider=config.provider,
                model=config.model,
                model_source=config.model_source,
                categories=categories,
                included_generated=args.include_generated,
                repeat=args.repeat,
                command_used=command_used,
                case_evaluations=case_evaluations,
                repo_root=REPO_ROOT,
            )
            run_dirs.append(write_result.run_dir)
            print(f"[evals] run guardado em: {write_result.run_dir}")
    finally:
        run.cleanup()

    if args.mark_baseline:
        for run_dir in run_dirs:
            baseline_dir = results_store.mark_baseline(run_dir)
            print(f"[evals] marcado como baseline: {baseline_dir}")

    comparison_dir = None
    if args.repeat > 1:
        per_case = comparisons.compare_repetitions(repetitions)
        comparison_id = run_dirs[-1].name if run_dirs else datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        comparison_dir = comparisons.write_comparison(
            config.provider,
            config.model,
            comparison_id,
            per_case,
            suite=suite,
            repeat=args.repeat,
            run_dirs=run_dirs,
        )
        print(f"[evals] comparacao de repeticoes: {comparison_dir}")
        # latest/ after a --repeat run reflects the aggregate, not just the
        # last individual repetition — a FLAKY case must never look like a
        # plain pass just because it happened to run last (Part 7 / repeat
        # reporting improvements). Individual run folders are untouched.
        results_store.write_latest_from_comparison(comparison_dir)
        flaky = [c for c in per_case.values() if c["status"] == comparisons.FLAKY]
        if flaky:
            print(f"[evals] AVISO: {len(flaky)} caso(s) FLAKY — ver {comparison_dir / 'comparison.md'}")
    else:
        # Part 7: a single run never proves stability either way.
        print("[evals] nota: --repeat 1 nao deteta instabilidade do modelo; considera --repeat 3 antes de confiar num PASS ou FAIL isolado.")

    deleted_runs = results_store.apply_retention(keep_runs=args.keep_runs)
    if deleted_runs:
        print(f"[evals] retencao: removidas {len(deleted_runs)} execucoes antigas (--keep-runs {args.keep_runs}).")
    results_store.rebuild_index()

    summary = report.summarize(all_case_evaluations, config.provider, config.model)
    print()
    print(
        f"[evals] {summary['passed_cases']}/{summary['total_cases']} casos passaram "
        f"({summary['failed_turns']} turnos falhados, {summary['exceptions']} excecoes)"
    )
    if summary.get("human_review_required_turns"):
        print(f"[evals] {summary['human_review_required_turns']} turno(s) sinalizados para revisao humana (nao bloqueante).")

    if args.strict:
        return 1 if summary["failed_turns"] > 0 else 0

    critical_failures = sum(
        1
        for case_eval in all_case_evaluations
        for turn in case_eval.turn_evaluations
        if not turn.passed and turn.failure_classification == INFRASTRUCTURE
    )
    return 1 if critical_failures > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

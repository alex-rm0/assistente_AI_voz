"""Run a set of tests N times, each run in a fresh pytest process, and report
failures and exit codes -- for crashes that only show up intermittently
(e.g. a native fail-fast, 0xC0000409, that kills pytest with no summary).

Default set: the 9 wake tests that crashed after the R2 model warm-up
(see KNOWN_ISSUES.md).

Run (from the repo root):

    .venv\\Scripts\\python.exe scripts\\flaky_check.py
    .venv\\Scripts\\python.exe scripts\\flaky_check.py --runs 40
    .venv\\Scripts\\python.exe scripts\\flaky_check.py --repo ..\\some-worktree --runs 40
    .venv\\Scripts\\python.exe scripts\\flaky_check.py tests/test_foo.py::test_bar --runs 10

--repo runs the tests in another checkout (e.g. a temporary worktree of an
older commit) with this interpreter. Exit code: 0 if every run passed, 1
otherwise.
"""
from __future__ import annotations

import argparse
import collections
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_WAKE = "tests/test_web_ui_prototype.py::"
DEFAULT_TESTS = [
    _WAKE + "test_wake_fast_path_a_standby_uses_wake_recognizer_not_medium_vad_path",
    _WAKE + "test_wake_fast_path_b_active_still_uses_medium_vad_path_not_wake_recognizer",
    _WAKE + "test_wake_fast_path_j_single_wake_transcript_triggers_active_transition_exactly_once",
    _WAKE + "test_wake_fast_path_m_unrelated_wake_recognizer_result_does_not_activate",
    _WAKE + "test_wake_fast_path_prewarm_triggered_asynchronously_on_standby_entry",
    _WAKE + "test_wake_fast_path_timing_log_includes_recognizer_and_model_fields",
    _WAKE + "test_wake_prewarm_a_shutdown_while_prewarm_still_running_waits_safely",
    _WAKE + "test_wake_prewarm_b_shutdown_after_prewarm_already_finished_is_a_noop",
    _WAKE + "test_wake_prewarm_g_normal_wake_fast_path_still_works",
]


def _describe(code: int) -> str:
    if code < 0:
        code &= 0xFFFFFFFF
    return f"{code} (0x{code:08X})" if code > 255 else str(code)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tests", nargs="*", help="pytest node ids (default: the 9 wake tests)")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--repo", type=Path, default=ROOT, help="checkout to run the tests in")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    tests = args.tests or DEFAULT_TESTS
    codes: collections.Counter[str] = collections.Counter()
    failed_runs: list[int] = []
    started = time.perf_counter()
    for run in range(1, args.runs + 1):
        result = subprocess.run(
            [args.python, "-m", "pytest", *tests, "-q", "-p", "no:cacheprovider"],
            cwd=str(args.repo), capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        lines = result.stdout.strip().splitlines()
        summary = lines[-1] if lines else ""
        passed = result.returncode == 0 and " passed" in summary
        codes[_describe(result.returncode)] += 1
        if not passed:
            failed_runs.append(run)
        print(f"run {run:3d}: {'ok  ' if passed else 'FAIL'} exit={_describe(result.returncode)} "
              f"summary={summary[:70] if summary else '(none -- process died)'}", flush=True)

    total = args.runs
    print(f"\nrepo={args.repo} tests={len(tests)} runs={total} failed={len(failed_runs)} "
          f"rate={len(failed_runs) / total:.1%} elapsed={time.perf_counter() - started:.0f}s")
    print("exit codes:", dict(codes))
    return 1 if failed_runs else 0


if __name__ == "__main__":
    raise SystemExit(main())

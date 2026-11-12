"""Conversation eval for R3: run evals/conversation/cases.json and report a pass rate per property.

    .venv\\Scripts\\python.exe scripts\\eval_conversation.py --provider fake
    .venv\\Scripts\\python.exe scripts\\eval_conversation.py --provider ollama --model llama3.1:8b
    .venv\\Scripts\\python.exe scripts\\eval_conversation.py --provider ollama --model qwen3.5:9b --think off
    .venv\\Scripts\\python.exe scripts\\eval_conversation.py --provider anthropic --allow-cloud --max-cost-usd 1

`fake` answers every LLM call with one fixed sentence: it checks the runner, not the conversation.
`ollama` uses the local GPU: check `nvidia-smi` first (CLAUDE.md). `anthropic` is paid: it refuses
to run without --allow-cloud and --max-cost-usd, needs ECHO_ALLOW_PAID_MODEL_CALLS=true in the
environment, and stops as soon as the estimated spend passes the ceiling. Never touches data/ or config/settings.json: every case runs on a
temporary data directory. The report goes to evals/results/conversation/<timestamp>__<provider>/.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evals.conversation import runner  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", choices=runner.PROVIDERS, default="fake")
    parser.add_argument("--model", default="", help="Model name; defaults to the provider's default.")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434", help="Ollama base URL.")
    parser.add_argument("--allow-cloud", action="store_true", help="Required for --provider anthropic (paid).")
    parser.add_argument("--cases", type=Path, default=runner.CASES_PATH)
    parser.add_argument("--case", action="append", default=[], help="Run only this case id (repeatable).")
    parser.add_argument("--language", choices=runner.LANGUAGES, help="Run only cases configured for this language.")
    parser.add_argument("--out", type=Path, help="Report directory.")
    parser.add_argument("--verbose", action="store_true", help="Show the engine's own trace output.")
    parser.add_argument("--max-cost-usd", type=float, help="Stop the run once the estimated spend passes this. Required for anthropic.")
    parser.add_argument("--think", choices=("on", "off"), help="Ollama thinking models: send think=true/false. Default: not sent.")
    args = parser.parse_args(argv)

    try:
        llm_factory = runner.make_llm_factory(
            args.provider, model=args.model, base_url=args.base_url, allow_cloud=args.allow_cloud,
            max_cost_usd=args.max_cost_usd, think={"on": True, "off": False}.get(args.think),
        )
    except (PermissionError, ValueError) as exc:
        parser.error(str(exc))

    cases = runner.load_cases(args.cases)
    if args.case:
        unknown = set(args.case) - {case["id"] for case in cases}
        if unknown:
            parser.error(f"Unknown case id(s): {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case["id"] in args.case]
    if args.language:
        cases = [case for case in cases if case["language"] == args.language]

    # The engine prints a trace for every turn; keep it out of the way unless asked for.
    with contextlib.redirect_stdout(sys.stdout if args.verbose else io.StringIO()):
        report = runner.run_cases(cases, llm_factory, provider=args.provider, model=args.model, max_cost_usd=args.max_cost_usd)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = args.out or REPO_ROOT / "evals" / "results" / "conversation" / f"{stamp}__{args.provider}"
    _json_path, md_path = runner.write_report(report, out_dir)

    for name, row in report["summary"]["properties"].items():
        print(f"{name:24} {runner._rate(row['pass_rate']):>5}  pass={row['pass']} fail={row['fail']} review={row['review']}")
    usage = report["summary"]["usage"]
    print(f"latency mean={usage['latency_ms_mean']:.0f} ms p90={usage['latency_ms_p90']:.0f} ms  cost={usage['cost_usd']:.4f} USD  calls={usage['llm_calls']}")
    if report["stopped"]:
        print(f"STOPPED: cost ceiling of {args.max_cost_usd} USD exceeded after {report['summary']['turns']} turns.")
    print(f"Report: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

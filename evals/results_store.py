"""Results directory layout, metadata, retention, baselines, index (Part 1).

    evals/results/
    ├── latest/                     copy of the most recent run
    ├── runs/YYYY-MM-DD/<run_name>/ one folder per execution
    ├── comparisons/<provider>__<model>/<comparison_id>/   --repeat aggregates
    ├── baselines/<run_name>/       runs explicitly marked with --mark-baseline
    └── index.md                    one row per run, newest first
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from evals import report
from evals.schemas import CaseEvaluation

RESULTS_ROOT = Path(__file__).resolve().parent / "results"
RUNS_DIR = RESULTS_ROOT / "runs"
LATEST_DIR = RESULTS_ROOT / "latest"
COMPARISONS_DIR = RESULTS_ROOT / "comparisons"
BASELINES_DIR = RESULTS_ROOT / "baselines"
INDEX_PATH = RESULTS_ROOT / "index.md"

_REPORT_FILES = ("report.json", "report.csv", "report.md", "metadata.json")


def safe_token(value: str) -> str:
    value = str(value or "").strip()
    cleaned = re.sub(r"[^A-Za-z0-9.\-]+", "-", value).strip("-")
    return cleaned or "unknown"


def suite_label(category: str | None, case_id: str | None, include_generated: bool) -> str:
    if case_id:
        base = f"case-{safe_token(case_id)}"
    elif category:
        base = f"category-{safe_token(category)}"
    else:
        base = "fixed"
    if include_generated:
        base += "-generated"
    return base


def build_run_name(timestamp_str: str, suite: str, provider: str, model: str, repeat: int) -> str:
    return f"{timestamp_str}__{suite}__{safe_token(provider)}__{safe_token(model)}__r{repeat}"


def git_info(repo_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        commit = "unknown"
    try:
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except Exception:
        dirty = False
    return commit, dirty


@dataclass
class RunWriteResult:
    run_dir: Path
    metadata: dict


def write_run(
    *,
    date_str: str,
    timestamp_str: str,
    suite: str,
    provider: str,
    model: str,
    categories: list[str],
    included_generated: bool,
    repeat: int,
    command_used: str,
    case_evaluations: list[CaseEvaluation],
    repo_root: Path,
) -> RunWriteResult:
    run_name = build_run_name(timestamp_str, suite, provider, model, repeat)
    run_dir = RUNS_DIR / date_str / run_name
    # Disambiguate the rare case where two repetitions of an all-deterministic
    # (zero-latency) suite finish within the same second.
    candidate = run_dir
    suffix = 2
    while candidate.exists():
        candidate = Path(f"{run_dir}-{suffix}")
        suffix += 1
    run_dir = candidate
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = report.summarize(case_evaluations, provider, model)
    (run_dir / "report.json").write_text(report.render_json(run_name, summary, case_evaluations), encoding="utf-8")
    (run_dir / "report.csv").write_text(report.render_csv(case_evaluations), encoding="utf-8")
    (run_dir / "report.md").write_text(report.render_markdown(run_name, summary, case_evaluations), encoding="utf-8")

    git_commit, git_dirty = git_info(repo_root)
    metadata = {
        "run_id": run_name,
        "timestamp": timestamp_str,
        "provider": provider,
        "model": model,
        "categories": categories,
        "included_generated": included_generated,
        "repeat": repeat,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "total_cases": summary["total_cases"],
        "passed": summary["passed_cases"],
        "failed": summary["failed_cases"],
        "exceptions": summary["exceptions"],
        "average_latency_ms": summary["average_latency_ms"],
        "command_used": command_used,
        "baseline": False,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    _copy_to_latest(run_dir)
    return RunWriteResult(run_dir=run_dir, metadata=metadata)


def _copy_to_latest(run_dir: Path) -> None:
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    for name in _REPORT_FILES:
        shutil.copyfile(run_dir / name, LATEST_DIR / name)


def mark_baseline(run_dir: Path) -> Path:
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["baseline"] = True
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    baseline_dir = BASELINES_DIR / run_dir.name
    baseline_dir.parent.mkdir(parents=True, exist_ok=True)
    if baseline_dir.exists():
        shutil.rmtree(baseline_dir)
    shutil.copytree(run_dir, baseline_dir)
    return baseline_dir


def _all_run_dirs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted({p.parent for p in RUNS_DIR.glob("*/*/metadata.json")})


def _load_metadata(run_dir: Path) -> dict | None:
    try:
        return json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    except Exception:
        return None


def apply_retention(keep_runs: int = 20) -> list[Path]:
    """Keeps the newest `keep_runs` runs per (provider, model, suite) group.
    Never deletes a run marked baseline=true — see mark_baseline()."""
    groups: dict[tuple[str, str, str], list[tuple[str, Path]]] = {}
    for run_dir in _all_run_dirs():
        metadata = _load_metadata(run_dir)
        if metadata is None or metadata.get("baseline"):
            continue
        suite = run_dir.name.split("__")[1] if "__" in run_dir.name else "unknown"
        key = (metadata.get("provider", ""), metadata.get("model", ""), suite)
        groups.setdefault(key, []).append((metadata.get("timestamp", ""), run_dir))

    deleted: list[Path] = []
    for entries in groups.values():
        entries.sort(key=lambda pair: pair[0], reverse=True)
        for _timestamp, run_dir in entries[keep_runs:]:
            shutil.rmtree(run_dir, ignore_errors=True)
            deleted.append(run_dir)
    return deleted


def _all_comparison_dirs() -> list[Path]:
    if not COMPARISONS_DIR.exists():
        return []
    return sorted({p.parent for p in COMPARISONS_DIR.glob("*/*/comparison.json")})


def _load_comparison(comparison_dir: Path) -> dict | None:
    try:
        return json.loads((comparison_dir / "comparison.json").read_text(encoding="utf-8"))
    except Exception:
        return None


def rebuild_index() -> Path:
    rows = []
    for run_dir in _all_run_dirs():
        metadata = _load_metadata(run_dir)
        if metadata is None:
            continue
        rows.append((metadata, run_dir))
    rows.sort(key=lambda pair: pair[0].get("timestamp", ""), reverse=True)

    comparison_rows = []
    for comparison_dir in _all_comparison_dirs():
        payload = _load_comparison(comparison_dir)
        if payload is None:
            continue
        comparison_rows.append((payload, comparison_dir))
    comparison_rows.sort(key=lambda pair: pair[0].get("comparison_id", ""), reverse=True)

    lines = ["# Índice de execuções de evals", ""]

    if comparison_rows:
        lines.append("## Comparações (--repeat) — agregado, nunca esconde instabilidade")
        lines.append("")
        lines.append("| Comparação | Modelo | Suite | Repetição | PASS_STABLE | FLAKY | FAIL_STABLE | Pasta |")
        lines.append("|---|---|---|---:|---:|---:|---:|---|")
        for payload, comparison_dir in comparison_rows:
            counts = payload.get("counts", {})
            rel_path = comparison_dir.relative_to(RESULTS_ROOT)
            lines.append(
                f"| {payload.get('comparison_id', '?')} | {payload.get('model', '?')} | {payload.get('suite', '?')} | "
                f"{payload.get('repeat', '?')} | {counts.get('PASS_STABLE', 0)} | {counts.get('FLAKY', 0)} | "
                f"{counts.get('FAIL_STABLE', 0)} | `{rel_path}` |"
            )
        lines.append("")

    lines.append("## Execuções individuais" if comparison_rows else "## Execuções")
    lines.append("")
    lines.append("| Data | Modelo | Suite | Repetição | Passaram | Total | Falhas | Pasta |")
    lines.append("|---|---|---|---:|---:|---:|---:|---|")
    for metadata, run_dir in rows:
        suite = run_dir.name.split("__")[1] if "__" in run_dir.name else "?"
        rel_path = run_dir.relative_to(RESULTS_ROOT)
        baseline_marker = " (baseline)" if metadata.get("baseline") else ""
        lines.append(
            f"| {metadata.get('timestamp', '?')} | {metadata.get('model', '?')} | {suite}{baseline_marker} | "
            f"{metadata.get('repeat', '?')} | {metadata.get('passed', '?')} | {metadata.get('total_cases', '?')} | "
            f"{metadata.get('failed', '?')} | `{rel_path}` |"
        )
    content = "\n".join(lines) + "\n"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(content, encoding="utf-8")
    return INDEX_PATH


def write_latest_from_comparison(comparison_dir: Path) -> None:
    """After a --repeat>1 run, `latest/` should reflect the aggregate
    comparison — not just the last individual repetition — so a person (or
    script) reading `latest/report.md` sees stability, not a lucky pass."""
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    comparison_json = comparison_dir / "comparison.json"
    comparison_md = comparison_dir / "comparison.md"
    (LATEST_DIR / "report.json").write_text(comparison_json.read_text(encoding="utf-8"), encoding="utf-8")
    (LATEST_DIR / "report.md").write_text(comparison_md.read_text(encoding="utf-8"), encoding="utf-8")
    payload = json.loads(comparison_json.read_text(encoding="utf-8"))
    metadata = {
        "aggregate": True,
        "comparison_id": payload.get("comparison_id"),
        "provider": payload.get("provider"),
        "model": payload.get("model"),
        "suite": payload.get("suite"),
        "repeat": payload.get("repeat"),
        "counts": payload.get("counts"),
        "run_dirs": payload.get("run_dirs"),
        "comparison_dir": str(comparison_dir),
    }
    (LATEST_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    # report.csv has no natural aggregate shape across repetitions (a CSV row
    # is one turn in one repetition) — keep the one from the last repetition
    # rather than inventing a lossy aggregate format; report.json/.md above
    # are the authoritative aggregate view.

"""Remove automatic noise from long-term memory (R3 step 5).

Three categories, all written automatically and never by the user:

1. memories "O utilizador usa frequentemente <alvo> para <acao>." -- written
   by desktop_actions.remember_desktop_action every time Echo opened a
   link/app (stopped in the same R3 step). The timeline events
   "Acao no Windows: ..." are kept.
2. structured_facts of type task with every attribute empty (no action,
   target, date, project, ...): nothing left to recall.
3. preferences "proactive_suggestion:<data>:<chave>" = shown for days
   before today: the per-day "already shown" marker of
   proactive_suggestions.py, only ever read for the current day.

Only with --include-request-tasks (decision of 12/11): structured_facts of
type task that are requests to Echo, not tasks of the user ("quero que me
digas o que esta escrito") -- pending, with no date, priority or reminder.
The dry run shows what the user said for each one.

How to run (from the repo root, with the app closed):

    .venv\\Scripts\\python.exe scripts\\cleanup_memory_noise.py            # dry run (default)
    .venv\\Scripts\\python.exe scripts\\cleanup_memory_noise.py --apply    # really delete
    .venv\\Scripts\\python.exe scripts\\cleanup_memory_noise.py --include-request-tasks [--apply]

Both modes first copy the database to <nome>.backup-<timestamp>.sqlite next
to it. The dry run never writes to the database.
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from assistant.long_term_memory import STRUCTURED_FACT_ATTRIBUTES  # noqa: E402

FREQUENT_USE = "O utilizador usa frequentemente % para %"
SUGGESTION_PREFIX = "proactive_suggestion:"


@dataclass
class Plan:
    frequent_use: list[tuple[int, str]] = field(default_factory=list)
    empty_tasks: list[tuple[int, str]] = field(default_factory=list)
    old_suggestions: list[str] = field(default_factory=list)
    request_tasks: list[tuple[int, str]] = field(default_factory=list)


def build_plan(connection: sqlite3.Connection, today: date, *, include_request_tasks: bool = False) -> Plan:
    plan = Plan()
    plan.frequent_use = connection.execute(
        "SELECT id, content FROM memories WHERE content LIKE ? ORDER BY id", (FREQUENT_USE,)
    ).fetchall()

    empty = " AND ".join(f"TRIM({attribute}) = ''" for attribute in STRUCTURED_FACT_ATTRIBUTES)
    plan.empty_tasks = connection.execute(
        f"SELECT id, created_at FROM structured_facts WHERE fact_type = 'task' AND {empty} ORDER BY id"
    ).fetchall()

    for key, value in connection.execute(
        "SELECT key, value FROM preferences WHERE key LIKE ? ORDER BY key", (SUGGESTION_PREFIX + "%",)
    ):
        day = key[len(SUGGESTION_PREFIX) :].split(":", 1)[0]
        try:
            shown_on = date.fromisoformat(day)
        except ValueError:
            continue
        if value == "shown" and shown_on < today:
            plan.old_suggestions.append(key)

    if include_request_tasks:
        empty_ids = {fact_id for fact_id, _created in plan.empty_tasks}
        plan.request_tasks = [
            (fact_id, text)
            for fact_id, text in connection.execute(
                "SELECT id, COALESCE(NULLIF(TRIM(raw_user_text), ''), action) FROM structured_facts "
                "WHERE fact_type = 'task' AND status = 'pending' AND TRIM(date_reference) = '' "
                "AND TRIM(priority) = '' AND TRIM(reminder_requested) = '' ORDER BY id"
            )
            if fact_id not in empty_ids
        ]
    return plan


def _report(connection: sqlite3.Connection, plan: Plan) -> None:
    memories = connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    tasks = connection.execute("SELECT COUNT(*) FROM structured_facts WHERE fact_type = 'task'").fetchone()[0]
    suggestions = connection.execute(
        "SELECT COUNT(*) FROM preferences WHERE key LIKE ?", (SUGGESTION_PREFIX + "%",)
    ).fetchone()[0]

    print(f"memorias: {memories} no total")
    print(f"  a remover 'usa frequentemente ...': {len(plan.frequent_use)}")
    for memory_id, content in plan.frequent_use:
        print(f"    [{memory_id}] {content[:160]}")
    print(f"  ficam: {memories - len(plan.frequent_use)}")
    for category, count in connection.execute(
        "SELECT category, COUNT(*) FROM memories WHERE content NOT LIKE ? GROUP BY category ORDER BY category",
        (FREQUENT_USE,),
    ):
        print(f"    {category}: {count}")

    print(f"factos estruturados task: {tasks} no total")
    print(f"  a remover (todos os atributos vazios): {len(plan.empty_tasks)}")
    for fact_id, created_at in plan.empty_tasks:
        print(f"    [{fact_id}] criado {created_at}")
    if plan.request_tasks:
        print(f"  a remover (--include-request-tasks: pedidos ao Echo, pendentes, sem data/prioridade/lembrete): {len(plan.request_tasks)}")
        for fact_id, text in plan.request_tasks:
            print(f"    [{fact_id}] {text[:160]}")
    print(f"  ficam: {tasks - len(plan.empty_tasks) - len(plan.request_tasks)}")

    print(f"preferencias proactive_suggestion: {suggestions} no total")
    print(f"  a remover (ja mostradas em dias anteriores): {len(plan.old_suggestions)}")
    for key in plan.old_suggestions:
        print(f"    {key}")
    print(f"  ficam: {suggestions - len(plan.old_suggestions)}")
    others = [row[0] for row in connection.execute(
        "SELECT key FROM preferences WHERE key NOT LIKE ? ORDER BY key", (SUGGESTION_PREFIX + "%",)
    )]
    print(f"outras preferencias (ficam): {', '.join(others) or 'nenhuma'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=REPO / "data" / "long_term_memory.sqlite")
    parser.add_argument("--apply", action="store_true", help="delete (default is a dry run)")
    parser.add_argument("--include-request-tasks", action="store_true", help="also delete task facts that are requests to Echo")
    parser.add_argument("--today", type=date.fromisoformat, default=date.today(), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    db = args.db
    if not db.exists():
        print(f"Base de dados nao encontrada: {db}")
        return 1
    backup = db.with_name(f"{db.stem}.backup-{datetime.now():%Y%m%d-%H%M%S}.sqlite")
    shutil.copyfile(db, backup)
    print(f"backup: {backup}")

    with sqlite3.connect(db) as connection:
        plan = build_plan(connection, args.today, include_request_tasks=args.include_request_tasks)
        _report(connection, plan)
        if not args.apply:
            print("dry run: nada foi alterado. Usa --apply para remover.")
            return 0
        connection.executemany("DELETE FROM memories WHERE id = ?", [(i,) for i, _c in plan.frequent_use])
        connection.executemany(
            "DELETE FROM structured_facts WHERE id = ?", [(i,) for i, _c in plan.empty_tasks + plan.request_tasks]
        )
        connection.executemany("DELETE FROM preferences WHERE key = ?", [(k,) for k in plan.old_suggestions])
    print(
        f"removidas: {len(plan.frequent_use)} memorias, {len(plan.empty_tasks) + len(plan.request_tasks)} factos task, "
        f"{len(plan.old_suggestions)} preferencias proactive_suggestion."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

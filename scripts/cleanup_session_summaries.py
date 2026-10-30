"""Remove content-less "Resumo de sessao" entries from long-term memory
(KNOWN_ISSUES §5).

Uses the same rule as AssistantEngine._remember_session_summary
(assistant.conversation.session_summary_has_content): an entry is removed
only when tasks, tools AND decisions are all empty (files do not count:
they are the Context Observer's recently modified project files). Every summary is
written by LongTermMemory.remember_context_summary both to `memories` and
to `timeline_events` with identical content, so both copies are handled.
The summaries themselves stay in session_manager.sqlite.

How to run (from the repo root, with the app closed):

    .venv\\Scripts\\python.exe scripts\\cleanup_session_summaries.py            # dry run (default)
    .venv\\Scripts\\python.exe scripts\\cleanup_session_summaries.py --apply    # really delete

Both modes first copy data/long_term_memory.sqlite to
data/long_term_memory.backup-<timestamp>.sqlite. The dry run never writes
to the database.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from assistant.conversation import session_summary_has_content  # noqa: E402

PREFIX = "Resumo de sessao"
# Placeholders written by _remember_session_summary for empty fields.
_FIELDS = re.compile(
    r"; ficheiros=(?P<files_touched>.*?); tarefas=(?P<tasks_changed>.*?); "
    r"ferramentas=(?P<tools_used>.*?); decisoes=(?P<decisions_taken>.*?); proximo_passo="
)
_EMPTY = {
    "files_touched": "sem ficheiros registados",
    "tasks_changed": "sem tarefas alteradas",
    "tools_used": "sem ferramentas registadas",
    "decisions_taken": "sem decisoes registadas",
}


def parse_summary(content: str) -> SimpleNamespace | None:
    match = _FIELDS.search(content)
    if match is None:
        return None
    values = {name: "" if value == _EMPTY[name] else value for name, value in match.groupdict().items()}
    return SimpleNamespace(**values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=REPO / "data" / "long_term_memory.sqlite")
    parser.add_argument("--apply", action="store_true", help="delete (default is a dry run)")
    args = parser.parse_args()

    db = args.db
    if not db.exists():
        print(f"Base de dados nao encontrada: {db}")
        return 1
    backup = db.with_name(f"{db.stem}.backup-{datetime.now():%Y%m%d-%H%M%S}.sqlite")
    shutil.copyfile(db, backup)
    print(f"backup: {backup}")

    with sqlite3.connect(db) as connection:
        memories = connection.execute(
            "SELECT id, content FROM memories WHERE content LIKE ? ORDER BY id", (PREFIX + "%",)
        ).fetchall()
        total = connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

        remove: list[tuple[int, str]] = []
        keep: list[tuple[int, str, SimpleNamespace]] = []
        unparsed = 0
        for memory_id, content in memories:
            parsed = parse_summary(content)
            if parsed is None:
                unparsed += 1
                continue
            if session_summary_has_content(parsed):
                keep.append((memory_id, content, parsed))
            else:
                remove.append((memory_id, content))

        contents = [content for _id, content in remove]
        timeline_matches = 0
        for content in contents:
            timeline_matches += connection.execute(
                "SELECT COUNT(*) FROM timeline_events WHERE content = ?", (content,)
            ).fetchone()[0]

        print(f"memorias no total: {total}")
        print(f"memorias '{PREFIX}': {len(memories)}")
        print(f"  a remover (sem tarefas, ferramentas nem decisoes): {len(remove)}")
        print(f"  a manter: {len(keep)}")
        print(f"  nao reconhecidas (mantidas): {unparsed}")
        print(f"timeline_events com o mesmo conteudo a remover: {timeline_matches}")
        print(f"memorias que ficam no total: {total - len(remove)}")
        print("exemplos que ficam:")
        for memory_id, content, _parsed in keep[:3]:
            print(f"  [{memory_id}] {content[:400]}")

        if not args.apply:
            print("dry run: nada foi alterado. Usa --apply para remover.")
            return 0

        connection.executemany("DELETE FROM memories WHERE id = ?", [(memory_id,) for memory_id, _c in remove])
        connection.executemany("DELETE FROM timeline_events WHERE content = ?", [(content,) for content in contents])
    print(f"removidas: {len(remove)} memorias e {timeline_matches} timeline_events.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

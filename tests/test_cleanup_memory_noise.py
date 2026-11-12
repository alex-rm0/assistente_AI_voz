import sqlite3
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import cleanup_memory_noise as cleanup  # noqa: E402
from assistant.long_term_memory import LongTermMemory  # noqa: E402


def _seed(tmp_path: Path) -> Path:
    memory = LongTermMemory(tmp_path / "data", embeddings_enabled=False)
    memory.remember("O utilizador usa frequentemente https://www.google.com para abriu URL.", category="preferencias")
    memory.remember("Prefiro respostas curtas.", category="preferencias")
    memory.set_preference("proactive_suggestion:2026-07-10:timeline_assistenteia", "shown")
    memory.set_preference("proactive_suggestion:2026-11-12:timeline_assistenteia", "shown")
    memory.set_preference("response_language", "en-GB")
    memory.remember_structured_fact("task", {"action": "", "target": ""})
    memory.remember_structured_fact("task", {"action": "escrever o relatorio"})
    return memory.db_path


def test_plan_lists_only_noise(tmp_path: Path) -> None:
    db = _seed(tmp_path)

    with sqlite3.connect(db) as connection:
        plan = cleanup.build_plan(connection, today=date(2026, 11, 12))

    assert [content for _id, content in plan.frequent_use] == [
        "O utilizador usa frequentemente https://www.google.com para abriu URL."
    ]
    assert plan.old_suggestions == ["proactive_suggestion:2026-07-10:timeline_assistenteia"]
    assert len(plan.empty_tasks) == 1


def test_dry_run_is_default_and_writes_backup_only(tmp_path: Path, capsys) -> None:
    db = _seed(tmp_path)

    assert cleanup.main(["--db", str(db), "--today", "2026-11-12"]) == 0

    assert list(db.parent.glob("long_term_memory.backup-*.sqlite"))
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM preferences").fetchone()[0] == 3
        assert connection.execute("SELECT COUNT(*) FROM structured_facts").fetchone()[0] == 2
    assert "dry run" in capsys.readouterr().out


def test_apply_removes_noise_and_keeps_the_rest(tmp_path: Path) -> None:
    db = _seed(tmp_path)

    assert cleanup.main(["--db", str(db), "--today", "2026-11-12", "--apply"]) == 0

    with sqlite3.connect(db) as connection:
        memories = [row[0] for row in connection.execute("SELECT content FROM memories")]
        preferences = sorted(row[0] for row in connection.execute("SELECT key FROM preferences"))
        tasks = [row[0] for row in connection.execute("SELECT action FROM structured_facts")]
    assert memories == ["Prefiro respostas curtas."]
    assert preferences == ["proactive_suggestion:2026-11-12:timeline_assistenteia", "response_language"]
    assert tasks == ["escrever o relatorio"]

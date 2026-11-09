"""Policy: response_language is a fixed setting, and the page follows it.
Before: no signal told the page the response language, so <html lang>
stayed "pt-PT" and the task card kept "TAREFA"/"Fechar"/"comparar opcoes"
after "responde em inglês" in the middle of a task. Real QWebEngine run
(tests/_webchannel_language_driver.py).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = Path(__file__).resolve().parent / "_webchannel_language_driver.py"


@pytest.fixture(scope="module")
def run() -> dict:
    if find_spec("PySide6.QtWebEngineWidgets") is None:
        pytest.skip("QtWebEngine not available")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONIOENCODING="utf-8")
    completed = subprocess.run(
        [sys.executable, str(DRIVER)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    match = re.search(r"^DRIVER_RESULT (\{.*\})$", completed.stdout, re.MULTILINE)
    assert match is not None, (completed.stdout + completed.stderr)[-4000:]
    result = json.loads(match.group(1))
    assert result["errors"] == []
    return result


def test_page_starts_in_portuguese_with_portuguese_card(run) -> None:
    task = run["snapshots"]["task_pt"]

    assert run["snapshots"]["ready"]["lang"] == "pt-PT"
    assert task["lang"] == "pt-PT"
    assert task["card_count"] == 1
    assert task["type_label"] == "TAREFA"
    assert task["close_label"] == "Fechar"
    assert "comparar opções" in task["text"]


def test_language_switch_mid_task_relabels_page_and_same_card(run) -> None:
    before = run["snapshots"]["task_pt"]
    after = run["snapshots"]["after_switch_en"]

    assert run["response_language"] == "en-GB"
    assert after["lang"] == "en-GB"
    assert after["card_count"] == 1
    assert after["card_id"] == before["card_id"]
    assert after["type_label"] == "TASK"
    assert after["close_label"] == "Close"
    assert "compare options" in after["text"]
    assert "comparar opcoes" not in after["text"] and "comparar opções" not in after["text"]
    assert "Ainda falta" not in after["text"]

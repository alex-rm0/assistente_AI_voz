"""Response-language page/card driver: real EchoOSWindow + QWebEngine +
QWebChannel with the REAL AssistantEngine (fake LLM, deterministic task
and language-command paths).

Run as a subprocess by tests/test_task_card_language_ui.py. Starts in
pt-PT, opens a comparison task, then says "responde em inglês" in the
middle of it. Prints one final `DRIVER_RESULT {json}` line with DOM
snapshots (<html lang>, task card type label, buttons, text, id).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication([sys.argv[0]])

from assistant.conversation import AssistantEngine  # noqa: E402
from assistant.long_term_memory import LongTermMemory  # noqa: E402
from assistant.memory import ConversationMemory  # noqa: E402
from assistant.presence_manager import PresenceManager  # noqa: E402
from assistant.prompts import get_base_system_prompt  # noqa: E402
from assistant.tool_registry import ToolRegistry  # noqa: E402
from prototype_web_ui.window import EchoOSWindow  # noqa: E402


class _FakeLLM:
    def chat(self, user_message, history=None, system_prompt=None, **kwargs):
        return "resposta inventada"

    def choose_tool(self, *args, **kwargs):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def embed(self, text):
        return None


class _ModelRuntime:
    """Minimal stand-in for ModelRuntimeBridge: only the payload methods the
    controller calls."""

    def current_payload(self, state: str = "idle_local") -> dict:
        return {"state": state, "mode": "local", "provider": "ollama", "model": "fake"}

    def request_started_payload(self) -> dict:
        return self.current_payload("thinking_local")

    def telemetry_payload(self, telemetry, *, state: str = "response_ready") -> dict:
        return self.current_payload(state)

    def error_payload(self, message: str, provider_error_type: str = "") -> dict:
        return {"state": "error", "note": message}

    def execute_command(self, command, source: str = "ui") -> dict:
        return self.current_payload("idle_local")


tmp = Path(tempfile.mkdtemp())
(tmp / "workspace").mkdir()
engine = AssistantEngine(
    llm=_FakeLLM(),
    memory=ConversationMemory(tmp / "data", "history.json", 20),
    long_term_memory=LongTermMemory(tmp / "data", embedder=None),
    tools=ToolRegistry(),
    workspace_path=tmp / "workspace",
    base_system_prompt=get_base_system_prompt(),
    presence_manager=PresenceManager(),
)
engine.response_language = "pt-PT"

window = EchoOSWindow(
    responder=engine.respond,
    title="language-driver",
    get_telemetry=engine.get_last_turn_telemetry,
    model_runtime=_ModelRuntime(),
    voice_available=False,
    get_response_language=lambda: engine.response_language,
)
controller = window.controller
controller.thinking_min_ms = 0
window.show()

result: dict[str, object] = {"snapshots": {}, "errors": []}
DOM_JS = """(() => {
  const cards = Array.from(document.querySelectorAll('#workspaceObjects .workspace-object'));
  const card = cards[cards.length - 1] || null;
  return JSON.stringify({
    lang: document.documentElement.lang,
    card_count: cards.length,
    card_id: card ? card.dataset.id : null,
    type_label: card ? card.querySelector('.workspace-object__type').textContent : null,
    close_label: card ? card.querySelector('.workspace-object__close').getAttribute('aria-label') : null,
    buttons: card ? Array.from(card.querySelectorAll('.workspace-object__actions button, .workspace-object__actions-secondary button')).map((b) => b.textContent) : [],
    text: card ? card.textContent : '',
  });
})()"""


def _wait(predicate, then, *, timeout_ms: int = 10000, label: str = "") -> None:
    deadline = time.perf_counter() + timeout_ms / 1000

    def poll() -> None:
        if predicate():
            then()
        elif time.perf_counter() > deadline:
            result["errors"].append(f"timeout:{label}")
            finish()
        else:
            QTimer.singleShot(20, poll)

    poll()


def _snapshot(name: str, then) -> None:
    def store(raw) -> None:
        result["snapshots"][name] = json.loads(raw)
        then()

    window.page().runJavaScript(DOM_JS, store)


turns = {"done": 0}
_original_finished = controller.requestFinished


def _type(text: str) -> None:
    window.page().runJavaScript(
        "(() => { const i = document.getElementById('voiceIn'); i.value = %s; i.form.requestSubmit(); })()"
        % json.dumps(text)
    )


controller.requestFinished.connect(lambda: turns.__setitem__("done", turns["done"] + 1))


def step_wait_ready() -> None:
    state = {"ready": False}

    def check(value) -> None:
        state["ready"] = bool(value)

    def poll() -> None:
        window.page().runJavaScript("Boolean(window.echoController)", check)
        if state["ready"]:
            QTimer.singleShot(300, lambda: _snapshot("ready", step_task))
        else:
            QTimer.singleShot(50, poll)

    poll()


def step_task() -> None:
    _type("Ajuda-me a comparar dois portáteis")
    _wait(lambda: turns["done"] >= 1, lambda: QTimer.singleShot(500, lambda: _snapshot("task_pt", step_switch)), label="task")


def step_switch() -> None:
    _type("responde em inglês")
    _wait(lambda: turns["done"] >= 2, lambda: QTimer.singleShot(500, lambda: _snapshot("after_switch_en", finish)), label="switch")


def finish() -> None:
    result["response_language"] = engine.response_language
    print("DRIVER_RESULT " + json.dumps(result, ensure_ascii=True), flush=True)
    window.close()


QTimer.singleShot(0, step_wait_ready)
QTimer.singleShot(45000, lambda: (print("DRIVER_TIMEOUT", flush=True), app.exit(3)))
sys.exit(app.exec())

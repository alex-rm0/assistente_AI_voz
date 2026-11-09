"""Stabilization battery driver (docs/STABILIZATION_BATTERY.md, scenarios
1-3 and 11-14).

Runs the REAL app (`app.main()` with `--ui echo-os`: real engine, real
Ollama, real ContextObserver) inside QWebEngine offscreen, and types each
message through the page's own form, the same way
tests/_webchannel_ui_driver.py does with a fake responder. Prints
`BATTERY {json}` marker lines (send / request_finished / DOM snapshots /
close timings) interleaved with the app's normal log output.

How to run (from the repo root, with Ollama running):

    .venv\\Scripts\\python.exe scripts\\battery_driver.py > battery_run.txt 2>&1

Optionally pass the repo root as the first argument (defaults to the
parent of this scripts/ folder). Paid model calls are kept off:
ECHO_ALLOW_PAID_MODEL_CALLS is removed from the environment.

Scenario 12 starts from pt-PT: before the app starts, the driver sets
`response_language` to "pt-PT" in data/user_settings.json, and "responde
em inglês" is then sent while scenario 11's task is still open.

WARNING: this drives the real app, so it WRITES TO data/ (history.json,
long_term_memory.sqlite -- one more session summary per run --,
session_manager.sqlite, user_settings.json, and a task created by scenario
11). Back up data/ first, e.g.:

    Copy-Item data\\*.json, data\\*.sqlite <backup-folder>

Then read the BATTERY lines plus the trace/[UI_RESPONSE]/[SHUTDOWN] lines,
and Ollama's own %LOCALAPPDATA%\\Ollama\\server.log for per-request timings.
Offscreen QtWebEngine also prints thousands of Chromium GPU errors
(`Context lost during MakeCurrent`); filter them out with
`grep -v ':ERROR:../../../qtwebengine'`.
"""
from __future__ import annotations

import json
import os
import sys
import time

T0 = time.perf_counter()
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.pop("ECHO_ALLOW_PAID_MODEL_CALLS", None)
REPO = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
sys.argv = [os.path.join(REPO, "app.py"), "--ui", "echo-os"]

from PySide6.QtCore import QTimer  # noqa: E402

import prototype_web_ui.window as window_module  # noqa: E402

TURN_TIMEOUT_S = 240
DOM_JS = """(() => {
  const e = document.getElementById('echoResponse');
  const s = getComputedStyle(e);
  const w = document.getElementById('workspaceObjects');
  const active = document.activeElement;
  return JSON.stringify({
    text: e.textContent.trim().slice(0, 400),
    visible: !e.hidden && s.display !== 'none' && s.visibility !== 'hidden' && (e.style.opacity === '1' || s.opacity !== '0'),
    active_element: active ? (active.id || active.tagName) : null,
    input_disabled: document.getElementById('voiceIn').disabled,
    task_surface: w ? (w.dataset.taskSurface || null) : null,
    open_task_count: w ? (w.dataset.openTaskCount || null) : null,
    workspace_text: w ? w.innerText.replace(/\\s+/g, ' ').trim().slice(0, 300) : null,
    lang: document.documentElement.lang || null
  });
})()"""

state: dict[str, object] = {"window": None, "finished": 0}


def mark(**fields: object) -> None:
    fields["t_ms"] = int((time.perf_counter() - T0) * 1000)
    print("BATTERY " + json.dumps(fields, ensure_ascii=False), flush=True)


_original_show = window_module.EchoOSWindow.show


def _hooked_show(self) -> None:
    _original_show(self)
    state["window"] = self
    self.controller.requestFinished.connect(lambda: state.__setitem__("finished", int(state["finished"]) + 1))
    mark(event="window_shown")
    QTimer.singleShot(0, wait_ready)


window_module.EchoOSWindow.show = _hooked_show


def page():
    return state["window"].page()


def wait_ready() -> None:
    flag = {"ready": False}

    def poll() -> None:
        page().runJavaScript("Boolean(window.echoController)", lambda v: flag.__setitem__("ready", bool(v)))
        if flag["ready"]:
            mark(scenario=1, event="ui_ready")
            QTimer.singleShot(1500, lambda: snapshot("1_after_ready", run_next))
        else:
            QTimer.singleShot(50, poll)

    poll()


def snapshot(name: str, then) -> None:
    def store(raw) -> None:
        mark(event="dom", name=name, dom=json.loads(raw))
        then()

    page().runJavaScript(DOM_JS, store)


def type_and_wait(scenario: int, label: str, text: str, then) -> None:
    before = int(state["finished"])
    mark(scenario=scenario, event="send", label=label, text=text)
    page().runJavaScript(
        "(() => { const i = document.getElementById('voiceIn'); i.value = %s; i.form.requestSubmit(); })()"
        % json.dumps(text)
    )
    deadline = time.perf_counter() + TURN_TIMEOUT_S
    sent_at = time.perf_counter()

    def poll() -> None:
        if int(state["finished"]) > before:
            mark(scenario=scenario, event="request_finished", label=label, elapsed_ms=int((time.perf_counter() - sent_at) * 1000))
            QTimer.singleShot(700, lambda: snapshot(f"{scenario}_{label}", then))
        elif time.perf_counter() > deadline:
            mark(scenario=scenario, event="timeout", label=label)
            snapshot(f"{scenario}_{label}_timeout", then)
        else:
            QTimer.singleShot(100, poll)

    poll()


STEPS = [
    (2, "ola", "Olá"),
    (3, "capital", "Qual é a capital de França?"),
    (11, "task_1", "Ajuda-me a comparar dois portáteis"),
    (11, "task_2", "o critério é a bateria"),
    (12, "lang_1", "responde em inglês"),
    (12, "lang_2", "olá"),
    (13, "tool", "Que janela tenho aberta?"),
]


def run_next() -> None:
    if not STEPS:
        QTimer.singleShot(500, close_window)
        return
    scenario, label, text = STEPS.pop(0)
    type_and_wait(scenario, label, text, run_next)


def close_window() -> None:
    mark(scenario=14, event="close_requested")
    started = time.perf_counter()
    state["window"].close()
    mark(scenario=14, event="close_call_returned", call_ms=int((time.perf_counter() - started) * 1000),
         visible_after=state["window"].isVisible())


import threading  # noqa: E402

import app  # noqa: E402

def reset_response_language() -> None:
    # Scenario 12 must show a real switch: start every run from pt-PT, so
    # "responde em inglês" (sent while scenario 11's task is still open)
    # changes the language in the middle of a task.
    path = os.path.join(REPO, "data", "user_settings.json")
    settings: dict[str, object] = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            settings = json.load(handle)
    previous = settings.get("response_language")
    settings["response_language"] = "pt-PT"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(settings, handle, ensure_ascii=False, indent=2)
    mark(scenario=12, event="response_language_reset", previous=previous, value="pt-PT")


reset_response_language()
_guard = threading.Timer(TURN_TIMEOUT_S * 9, lambda: (mark(event="global_timeout"), os._exit(3)))
_guard.daemon = True
_guard.start()
code = app.main()
mark(scenario=14, event="app_exec_returned", code=code)

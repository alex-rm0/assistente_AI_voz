"""Real WebEngine regression: a draft has one readable canonical surface."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_email_draft_is_not_duplicated_over_its_task_card(tmp_path):
    script = r'''
import json, sys
from pathlib import Path
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
app = QApplication([])
from tests.test_live_writing_regressions import english_engine
from tests.test_voice_email_stabilization import DRAFT
from prototype_web_ui.window import EchoOSWindow
engine, local = english_engine(Path(sys.argv[1]), [DRAFT])
window = EchoOSWindow(engine.respond)
messages = iter(['Queria escrever um mail contigo.', 'Francisca e Karina',
                 'Queria perguntar se podem vir à reunião desta semana que é às 9h30 da noite.',
                 'Podes escrever o primeiro rascunho.'])
def finish():
    query = """JSON.stringify((() => {
      const response = document.getElementById('echoResponse');
      const card = document.querySelector('#workspaceObjects [data-type="writing_task"]');
      const a = response.getBoundingClientRect(), b = card.getBoundingClientRect();
      return {response: response.textContent, card: card.textContent,
        intersects: a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top};
    })())"""
    def result(raw):
        print('DRAFT_UI ' + raw, flush=True)
        window.close()
    window.page().runJavaScript(query, result)
def send_next():
    try:
        message = next(messages)
    except StopIteration:
        QTimer.singleShot(700, finish)
        return
    window.page().runJavaScript('document.getElementById("voiceIn").value=' + json.dumps(message) + '; document.getElementById("echoForm").requestSubmit();')
def ready():
    def received(value):
        if value:
            window.page().runJavaScript('document.documentElement.lang="en-GB";')
            send_next()
        else:
            QTimer.singleShot(50, ready)
    window.page().runJavaScript('Boolean(window.echoController)', received)
window.controller.requestFinished.connect(lambda: QTimer.singleShot(200, send_next))
window.show()
QTimer.singleShot(100, ready)
QTimer.singleShot(15000, app.quit)
app.exec()
'''
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)],
                            cwd=Path(__file__).resolve().parents[1], env=env,
                            capture_output=True, text=True, encoding="utf-8", timeout=25)
    line = next((line for line in result.stdout.splitlines() if line.startswith("DRAFT_UI ")), None)
    assert result.returncode == 0 and line, result.stdout[-1500:] + result.stderr[-1500:]
    snapshot = json.loads(line.removeprefix("DRAFT_UI "))
    assert "Subject: Meeting" in snapshot["card"]
    assert "Subject: Meeting" not in snapshot["response"]
    assert not snapshot["intersects"]

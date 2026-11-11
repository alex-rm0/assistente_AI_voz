import ast
import inspect
from types import SimpleNamespace

import app
from prototype_web_ui.window import EchoOSWindow


def test_main_does_not_probe_microphone_before_ui_startup():
    tree = ast.parse(inspect.getsource(app.main))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(isinstance(node.func, ast.Name) and node.func.id == "check_microphone" for node in calls)


def test_window_notifies_controller_only_when_webchannel_ready():
    events = []
    window = SimpleNamespace(
        _close_in_progress=False,
        controller=SimpleNamespace(notify_ui_ready=lambda: events.append("ready")),
        _ui_ready_timer=SimpleNamespace(start=lambda delay: events.append(delay)),
    )
    EchoOSWindow._on_ui_ready_checked(window, False)
    assert events == [50]
    EchoOSWindow._on_ui_ready_checked(window, True)
    assert events == [50, "ready"]
    window._close_in_progress = True
    EchoOSWindow._on_ui_ready_checked(window, True)
    assert events == [50, "ready"]

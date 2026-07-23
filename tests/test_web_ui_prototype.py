from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOTYPE = ROOT / "prototype_web_ui"
WEB = PROTOTYPE / "web"


def test_web_ui_prototype_files_exist() -> None:
    expected = (
        PROTOTYPE / "__init__.py",
        PROTOTYPE / "run_prototype.py",
        PROTOTYPE / "controller.py",
        PROTOTYPE / "window.py",
        WEB / "index.html",
        WEB / "styles.css",
        WEB / "echo_entity.js",
        WEB / "echo_ui.js",
        PROTOTYPE / "README.md",
    )

    assert all(path.exists() for path in expected)


def test_web_ui_prototype_is_independent_from_design_runtime() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    scripts = (WEB / "echo_entity.js").read_text(encoding="utf-8") + (WEB / "echo_ui.js").read_text(encoding="utf-8")

    assert "support.js" not in html
    assert "DCLogic" not in scripts
    assert "<x-dc" not in html
    assert "{{" not in html
    assert "fonts.googleapis.com" not in html
    assert "unpkg.com" not in html


def test_web_ui_controller_api_is_exposed() -> None:
    controller = (PROTOTYPE / "controller.py").read_text(encoding="utf-8")
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")
    window = (PROTOTYPE / "window.py").read_text(encoding="utf-8")

    for name in ("submitMessage", "cancelCurrentRequest", "receiveMessage", "setState", "clearConversation"):
        assert name in controller
    for signal in ("responseReady", "stateChanged", "errorOccurred", "uiEvent", "requestStarted", "requestFinished"):
        assert signal in controller
        assert signal in ui
    assert 'registerObject("echoController", self.controller)' in window
    assert "channel.objects.echoController" in ui
    assert "channel.objects.echoBridge" not in ui


def test_web_ui_entity_exposes_state_api() -> None:
    entity = (WEB / "echo_entity.js").read_text(encoding="utf-8")
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    assert "setState(state)" in entity
    assert "window.echoEntity.setState(value)" in ui
    for state in ("idle", "thinking", "speaking", "error"):
        assert state in ui


def test_web_ui_app_entrypoints_are_available() -> None:
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    window = (PROTOTYPE / "window.py").read_text(encoding="utf-8")

    assert "--ui" in app
    assert "echo-os" in app
    assert "classic" in app
    assert "EchoOSWindow" in window


def test_web_ui_scaling_allows_upscaling() -> None:
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    assert "stage-scaler" in css
    assert "available_width / 1300" not in ui
    assert "window.innerWidth" in ui
    assert "scale(" in ui


def test_echo_response_has_dedicated_visible_layer() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    assert 'id="echoResponse"' in html
    assert 'aria-live="polite"' in html
    assert ".echo-response" in css
    assert "z-index: 7" in css
    assert 'const element = byId("echoResponse")' in ui
    assert "function renderEchoResponse(text)" in ui
    assert 'element.textContent = value' in ui
    assert 'element.style.display = "block"' in ui
    assert 'element.classList.add("visible")' in ui
    assert 'console.log("[Echo UI JS] resposta aplicada:"' in ui


def test_research_workspace_is_available() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    assert 'id="researchWorkspace"' in html
    assert ".research-workspace" in css
    assert 'stage[data-workspace="research"]' in css
    assert "function createWorkspaceController()" in ui
    assert "handleUiEventPayload" in ui
    assert "activeController.uiEvent.connect" in ui
    assert "research_results_ready" in ui
    assert "research_unavailable" in ui
    assert "conversation_cleared" in ui


def test_cognitive_telemetry_panel_is_available_with_mock_data() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    assert 'id="telemetryCompact"' in html
    assert 'id="telemetryPanel"' in html
    assert 'aria-controls="telemetryPanel"' in html
    assert 'aria-hidden="true"' in html
    assert "MODEL BEHAVIOUR" in html
    for mode in ("local", "claude", "automatic"):
        assert f'data-telemetry-mode="{mode}"' in html

    assert ".telemetry-compact" in css
    assert ".telemetry-panel" in css
    assert ".telemetry-panel.open" in css
    assert "backdrop-filter: blur" in css
    assert "@media (prefers-reduced-motion: reduce)" in css

    assert "const telemetryMocks" in ui
    assert "function applyTelemetryMock" in ui
    assert "function setTelemetryPanel" in ui
    assert "function setTelemetryMode" in ui
    assert "window.echoTelemetryDemo = applyTelemetryMock" in ui


def test_cognitive_telemetry_is_mocked_and_keyboard_accessible() -> None:
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    assert "routing_automatic" in ui
    assert "thinking_cloud" in ui
    assert "response_ready" in ui
    assert 'event.key === "Escape" && telemetryPanelOpen' in ui
    assert 'setAttribute("aria-expanded"' in ui
    assert 'setAttribute("aria-hidden"' in ui
    assert "runAutomaticRoutingMock()" in ui
    assert "controller.submitMessage(message)" in ui
    assert "ModelRouter" not in ui
    assert "budget" not in ui.lower()


def test_cognitive_telemetry_maps_reason_codes_to_human_labels() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    expected_labels = (
        "Professional writing",
        "Structured summary",
        "Technical explanation",
        "Local response",
        "Memory recall",
        "Local tool",
    )
    for label in expected_labels:
        assert label in ui or label in html

    assert "professional_writing" not in html
    assert "structured_summary" not in html
    assert "technical_explanation" not in html
    assert 'elements.routing.textContent = data.routing' in ui


def test_cognitive_telemetry_compact_line_sits_above_input_zone() -> None:
    css = (WEB / "styles.css").read_text(encoding="utf-8")

    compact_block = css[css.index(".telemetry-compact {"):css.index(".telemetry-compact.visible")]
    visible_block = css[css.index(".telemetry-compact.visible"):css.index(".telemetry-compact::before")]
    input_block = css[css.index(".input-zone {"):css.index(".input-pill {")]

    assert "bottom: 124px;" in compact_block
    assert "bottom: 30px;" in input_block
    assert "opacity: .62;" in visible_block
    assert '.stage[data-telemetry-state="thinking_local"] .telemetry-compact' in css
    assert '.stage[data-telemetry-state="response_ready"] .telemetry-compact' in css
    assert "opacity: .9;" in css


def test_cognitive_telemetry_panel_resize_layout_is_explicit() -> None:
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    assert "function scaleStage()" in ui
    assert 'window.addEventListener("resize", () => {' in ui
    assert "window.requestAnimationFrame" in ui
    assert "renderEchoResponse" not in ui[ui.index("function scaleStage"):ui.index("function setInputEnabled")]
    assert 'elements.stage.dataset.telemetryPanel = telemetryPanelOpen ? "open" : "closed"' in ui

    assert '.stage[data-telemetry-panel="open"] .echo-main' not in css
    assert "left: 380px;" not in css
    assert 'style.setProperty("--workspace-x"' in ui
    assert 'style.setProperty("--workspace-y"' in ui
    assert "@media (max-width: 760px), (max-height: 620px)" in css
    assert "max-height: calc(100% - 190px)" in css
    assert '.stage[data-telemetry-panel="open"] #mind' not in css


def test_cognitive_telemetry_panel_keeps_response_and_input_safe_on_compact_windows() -> None:
    css = (WEB / "styles.css").read_text(encoding="utf-8")

    compact_block = css[css.index("@media (max-width: 760px), (max-height: 620px)"):css.index("@media (max-height: 560px)")]
    input_block = css[css.index(".input-zone {"):css.index(".input-pill {")]

    assert "bottom: 30px;" in input_block
    assert '.stage[data-telemetry-panel="open"] .echo-response' not in compact_block
    assert "inputSafeRect()" in (WEB / "echo_ui.js").read_text(encoding="utf-8")
    assert "rectsIntersect(itemRect(item), inputRect)" in (WEB / "echo_ui.js").read_text(encoding="utf-8")


def test_web_ui_removes_try_suggestions_and_numbered_debug_states() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    assert 'id="suggestions"' not in html
    assert "<span>TRY</span>" not in html
    assert "data-message" not in html
    assert 'class="debug-states"' not in html
    assert "data-state=" not in html
    assert ".suggestions" not in css
    assert ".debug-states" not in css
    assert "closest(\"button[data-message]\")" not in ui
    assert "closest(\"button[data-state]\")" not in ui


def test_web_ui_header_has_three_independent_zones() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    assert 'class="topbar-left"' in html
    assert 'id="topbarClock"' in html
    assert 'id="appVersion"' in html
    assert "ECHO OS · v0.4.0" in html
    assert "grid-template-columns: 1fr auto 1fr;" in css
    assert ".topbar-clock" in css
    assert ".topbar-version" in css
    assert 'const APP_VERSION = "v0.4.0";' in ui
    assert "function updateClock()" in ui
    assert "window.setInterval(updateClock, 60000)" in ui


def test_web_ui_entity_uses_user_controlled_workspace_positioning() -> None:
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")
    entity = (WEB / "echo_entity.js").read_text(encoding="utf-8")

    assert ".echo-main" in css
    assert ".stage[data-telemetry-panel=\"open\"] .echo-main" not in css
    assert 'elements.stage.dataset.telemetryPanel = telemetryPanelOpen ? "open" : "closed"' in ui
    assert "stage.dataset.viewport" not in ui
    assert "window.innerWidth < 760" not in ui
    assert "function createFreeWorkspaceController()" in ui
    assert "createWorkspaceItem" in ui
    assert "window.echoFreeWorkspace" in ui
    assert "setCenter(x, y, immediate = false)" in entity
    assert "clearCustomCenter()" in entity


def test_web_ui_free_workspace_has_drag_snap_persistence_and_locking() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    assert 'tabindex="0"' in html
    assert 'data-workspace-drag-handle="telemetryPanel"' in html
    assert 'id="recenterEcho"' in html
    assert 'id="resetWorkspace"' in html
    assert 'id="toggleWorkspaceLock"' in html
    assert 'aria-pressed="false"' in html

    assert "WORKSPACE_STORAGE_KEY" in ui
    assert "window.localStorage.setItem(WORKSPACE_STORAGE_KEY" in ui
    assert "window.localStorage.getItem(WORKSPACE_STORAGE_KEY)" in ui
    assert "SNAP_ZONES" in ui
    for zone in ("center", "top-center", "upper-left", "upper-right", "center-left", "center-right"):
      assert zone in ui
    assert "pointerdown" in ui
    assert "pointermove" in ui
    assert "pointerup" in ui
    assert "setPointerCapture" in ui
    assert "releasePointerCapture" in ui
    assert "toggleLock" in ui
    assert "recenterEcho" in ui
    assert "resetWorkspace" in ui
    assert 'event.key === "Escape") cancelDrag()' in ui
    assert "clampAfterResize" in ui

    assert "#mind {" in css
    assert "cursor: grab;" in css
    assert "body.workspace-dragging" in css
    assert ".telemetry-workspace-controls" in css


def test_web_ui_workspace_drag_does_not_call_backend_or_move_input() -> None:
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    workspace_block = ui[ui.index("function createFreeWorkspaceController()"):ui.index("function applyEchoState")]
    input_block = css[css.index(".input-zone {"):css.index(".input-pill {")]

    assert "controller.submitMessage" not in workspace_block
    assert "echoController" not in workspace_block
    assert "ModelRouter" not in ui
    assert "bottom: 30px;" in input_block
    assert ".stage[data-telemetry-panel=\"open\"] .input-zone" not in css
    assert ".stage[data-telemetry-panel=\"open\"] .echo-main" not in css


def test_web_ui_initializes_qwebchannel_once() -> None:
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")
    window = (PROTOTYPE / "window.py").read_text(encoding="utf-8")

    assert "function initializeEchoChannel()" in ui
    assert "window.__echoChannelInitialized" in ui
    assert 'console.log("[Echo UI JS] QWebChannel disponivel:", typeof QWebChannel)' in ui
    assert 'console.log("[Echo UI JS] qt disponivel:", typeof qt)' in ui
    assert '"[Echo UI JS] transport disponivel:"' in ui
    assert "function connectControllerSignals(activeController)" in ui
    assert 'console.log("[Echo UI JS] signals ligados")' in ui
    assert "registered object name=echoController" in window


def test_idle_state_does_not_hide_rendered_response() -> None:
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")
    set_state_block = ui[ui.index("function applyEchoState"):ui.index("function focusInput")]

    assert "echoSays.classList.remove(\"visible\")" in set_state_block
    assert "echoResponse" not in set_state_block


def test_thinking_animation_is_not_overactive() -> None:
    entity = (WEB / "echo_entity.js").read_text(encoding="utf-8")

    assert "sparkRandom(5)" not in entity
    assert "this.cxTarget = this.w * 0.42" not in entity
    assert "dt * 9" not in entity
    assert "dt * 5" not in entity
    assert "0.42 * glow" not in entity
    assert "0.7 + glow" not in entity
    assert "0.16 * glow" in entity
    assert "0.7 + 0.35 * glow" in entity
    assert "this.spark(this.thinkingFocus, 1)" in entity


def test_controller_keeps_visual_minimum_durations() -> None:
    controller = (PROTOTYPE / "controller.py").read_text(encoding="utf-8")

    assert "thinking_min_ms: int = 300" in controller
    assert "speaking_duration_ms: int = 1000" in controller
    assert "response_text = str(response or \"\")" in controller
    assert "def _emit_response_ready" in controller
    assert "self.responseReady.emit(response_text)" in controller
    assert "delay_ms = max(0, self.thinking_min_ms - elapsed_ms)" in controller
    assert "def _enter_speaking" in controller
    assert "def _finish_speaking" in controller


def test_worker_results_are_queued_back_to_controller_thread() -> None:
    controller = (PROTOTYPE / "controller.py").read_text(encoding="utf-8")

    assert "from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot" in controller
    assert "self._worker.finished.connect(self._handle_response, Qt.ConnectionType.QueuedConnection)" in controller
    assert "self._worker.failed.connect(self._handle_error, Qt.ConnectionType.QueuedConnection)" in controller
    assert "self._worker.finished.connect(self._cleanup_worker)" not in controller
    assert "self._worker.failed.connect(self._cleanup_worker)" not in controller
    assert "@Slot(str)\n    def _handle_response" in controller
    assert "@Slot(str)\n    def _handle_error" in controller
    assert "[Echo UI THREAD]" in controller


def test_web_ui_forwards_javascript_console_to_terminal() -> None:
    window = (PROTOTYPE / "window.py").read_text(encoding="utf-8")

    assert "class EchoWebPage(QWebEnginePage)" in window
    assert "def javaScriptConsoleMessage" in window
    assert 'print(' in window
    assert "self.setPage(EchoWebPage(self))" in window

from __future__ import annotations

import json
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
    for signal in ("telemetryUpdated", "modelModeChanged"):
        assert signal in controller
        assert signal in ui
    for slot in ("getModelTelemetry", "setModelMode", "setAutomaticClaudeEnabled", "setModelBudget", "executeSystemCommand"):
        assert slot in controller
    assert "executeSystemCommand" in ui
    assert 'registerObject("echoController", self.controller)' in window
    assert "channel.objects.echoController" in ui
    assert "channel.objects.echoBridge" not in ui


def test_web_ui_entity_exposes_state_api() -> None:
    entity = (WEB / "echo_entity.js").read_text(encoding="utf-8")
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    assert "setState(state, options = {})" in entity
    assert "window.echoEntity.setState(value, {intensityBoost});" in ui
    for state in ("idle", "listening", "thinking", "reading", "working", "speaking", "error"):
        assert state in ui


def test_web_ui_app_entrypoints_are_available() -> None:
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    window = (PROTOTYPE / "window.py").read_text(encoding="utf-8")

    assert "--ui" in app
    assert "echo-os" in app
    assert "classic" in app
    assert "EchoOSWindow" in window


def test_web_ui_uses_adaptive_layout_instead_of_global_scaling() -> None:
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")
    html = (WEB / "index.html").read_text(encoding="utf-8")

    assert "stage-scaler" in css
    assert "--shell-inline-margin" in css
    assert "--shell-block-margin" in css
    assert "--content-max-width" in css
    assert "--reading-max-width" in css
    assert "--entity-size" in css
    assert "--safe-gap" in css
    assert "transform: none;" in css
    assert "scale(" not in ui[ui.index("function updateAdaptiveLayout"):ui.index("function setInputEnabled")]
    assert 'data-layout-density="standard"' in html
    assert "function updateAdaptiveLayout()" in ui
    assert "window.requestAnimationFrame" in ui


def test_web_ui_has_structural_adaptive_regions_for_future_modules() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    css = (WEB / "styles.css").read_text(encoding="utf-8")

    for region in (
        'class="layout-region layout-region--left"',
        'class="layout-region layout-region--stage"',
        'class="layout-region layout-region--right"',
        'class="layout-region layout-region--response"',
        'class="layout-region layout-region--composer"',
    ):
        assert region in html

    assert ".adaptive-layout-grid" in css
    assert "grid-template-areas:" in css
    assert '"left stage right"' in css
    assert '"left response right"' in css
    assert '"left composer right"' in css
    assert ".layout-region--left:empty" in css
    assert ".layout-region--right:empty" in css


def test_web_ui_removes_nonfunctional_decorative_corner_ticks() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    css = (WEB / "styles.css").read_text(encoding="utf-8")

    assert 'class="ticks"' not in html
    assert "<path d=\"M28 44 h20 M28 44 v20\"" not in html
    assert ".ticks" not in css
    assert ".stage {" in css
    assert "border: 1px solid var(--hairline);" in css


def test_web_ui_declares_experience_breakpoints_and_density_attributes() -> None:
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")
    entity = (WEB / "echo_entity.js").read_text(encoding="utf-8")

    for density in ("compact", "standard", "wide", "ultrawide"):
        assert f'return "{density}"' in ui or f'=== "{density}"' in ui
        assert f'.stage[data-layout-density="{density}"]' in css

    assert "chooseLayoutDensity" in ui
    assert "dataset.entityScale" in ui
    assert "dataset.responseSize" in ui
    assert "--entity-size-actual" in css
    assert 'typeof window.echoEntity.resize === "function"' in ui
    assert "window.echoEntity.resize();" in ui
    assert "responsiveRadius()" in entity
    assert 'getPropertyValue("--entity-size-actual")' in entity


def test_web_ui_response_and_composer_expand_safely_on_large_viewports() -> None:
    css = (WEB / "styles.css").read_text(encoding="utf-8")

    assert "width: min(var(--reading-max-width), calc(100% - var(--safe-gap) * 2));" in css
    assert "max-height: var(--response-max-height);" in css
    assert ".stage[data-layout-density=\"wide\"]" in css
    assert "--reading-max-width: min(1120px, 72vw);" in css
    assert ".stage[data-layout-density=\"ultrawide\"]" in css
    assert "--content-max-width: min(1880px, calc(100vw - var(--shell-inline-margin) * 2));" in css
    assert "#voiceIn" in css
    assert "width: clamp(260px, 42vw, 780px);" in css


def test_web_ui_telemetry_is_slightly_larger_only_on_wide_layouts() -> None:
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    root_block = css[css.index(":root {"):css.index("html,")]
    wide_block = css[css.index('.stage[data-layout-density="wide"]'):css.index('.stage[data-layout-density="ultrawide"]')]
    ultrawide_block = css[css.index('.stage[data-layout-density="ultrawide"]'):css.index(".stage[data-entity-scale")]
    compact_block = css[css.index('.stage[data-layout-density="compact"]'):css.index('.stage[data-layout-density="standard"]')]
    standard_block = css[css.index('.stage[data-layout-density="standard"]'):css.index('.stage[data-layout-density="wide"]')]

    assert "--telemetry-panel-width: 318px;" in root_block
    assert "--telemetry-panel-font-size: 11px;" in root_block
    assert "--telemetry-panel-width: 354px;" in wide_block
    assert "--telemetry-panel-width: 366px;" in ultrawide_block
    assert "--telemetry-panel-font-size: 12px;" in wide_block
    assert "--telemetry-panel-font-size: 12px;" in ultrawide_block
    assert "--telemetry-panel-width" not in compact_block
    assert "--telemetry-panel-width" not in standard_block
    assert 'cssPixelNumber(stage, "--telemetry-panel-width", PANEL_ITEM_SIZE.width)' in ui


def test_web_ui_response_layouts_use_real_metrics_and_anchors() -> None:
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")
    html = (WEB / "index.html").read_text(encoding="utf-8")

    stacked_block = css[css.index('.stage[data-response-layout="stacked"] .echo-response'):css.index(".echo-response p,")]
    focus_block = css[css.index('.stage[data-response-layout="focus"] .echo-response {'):css.index('.stage[data-response-layout="stacked"] .echo-response {')]
    workspace_defaults_block = ui[ui.index("function workspaceDefaults"):ui.index("function createFreeWorkspaceController")]

    assert 'data-response-layout="inline"' in html
    assert 'data-entity-anchor="center"' in html
    assert "const RESPONSE_LAYOUT_CONFIG" in ui
    assert "function collectResponseMetrics" in ui
    assert "function determineResponseLayout(metrics" in ui
    assert "renderedHeight" in ui
    assert "availableStageHeight" in ui
    assert "renderedRatio" in ui
    assert "contentType" in ui
    assert 'return "inline"' in ui
    assert 'return "stacked"' in ui
    assert 'return "focus"' in ui
    assert 'stage.dataset.responseLayout = layout' in ui
    assert 'stage.dataset.entityAnchor = anchorResult.anchor' in ui
    assert "top: var(--response-top);" in stacked_block
    assert "max-height: min(var(--response-max-height), var(--response-max-height-current));" in stacked_block
    assert "width: min(980px" in focus_block
    assert "const entityRadius = Math.max(70, Math.min(220, entityDiameter / 2));" in workspace_defaults_block
    assert "const topCenterY = headerHeight + entityRadius + safeGap;" in workspace_defaults_block
    assert '"top-right": {x:' in workspace_defaults_block
    assert "function updateResponseLayoutBounds(stage, layout)" in ui
    assert 'stage.style.setProperty("--response-top"' in ui
    assert 'return "top-center"' in ui
    assert 'point: zones.echo["top-center"]' in ui
    assert 'responseAnchorForLayout(stage, layout)' in ui


def test_web_ui_response_layout_modes_position_routing_and_composer() -> None:
    css = (WEB / "styles.css").read_text(encoding="utf-8")

    assert '.stage[data-response-layout="inline"] .echo-response' in css
    assert '.stage[data-response-layout="stacked"] .telemetry-compact' in css
    assert "bottom: calc(var(--input-height) + var(--safe-gap) + 28px);" in css
    assert '.stage[data-response-layout="stacked"] .input-zone' in css
    assert "bottom: calc(var(--safe-gap) + 24px);" in css
    assert '.stage[data-response-layout="focus"] .telemetry-compact' in css
    assert "bottom: calc(var(--input-height) + var(--safe-gap) + 20px);" in css
    assert '.stage[data-response-layout="focus"] .input-zone' in css
    assert "bottom: calc(var(--safe-gap) + 18px);" in css


def test_web_ui_focus_layout_reserves_horizontal_space_for_echo() -> None:
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    bounds_block = ui[ui.index("function updateResponseLayoutBounds"):ui.index("function createFreeWorkspaceController")]
    focus_block = css[css.index('.stage[data-response-layout="focus"] .echo-response {'):css.index('.stage[data-response-layout="focus"] .echo-response.visible')]

    assert "const minFocusResponseWidth = 540;" in bounds_block
    assert "const echoRect = echoRectForPoint(anchorResult.point, entityRadius);" in bounds_block
    assert "const reservedRightEdge = echoRect.left - safeGap;" in bounds_block
    assert "const availableWidth = reservedRightEdge - availableLeft;" in bounds_block
    assert "focusResponseWidth = Math.min(maxFocusResponseWidth, availableWidth);" in bounds_block
    assert "focusResponseLeft = availableLeft + Math.max(0, (availableWidth - focusResponseWidth) / 2);" in bounds_block
    assert 'stage.style.setProperty("--focus-response-left"' in bounds_block
    assert 'stage.style.setProperty("--focus-response-width"' in bounds_block
    assert "left: var(--focus-response-left);" in focus_block
    assert "width: var(--focus-response-width);" in focus_block
    assert "transform: translateY(6px);" in focus_block


def test_web_ui_focus_layout_falls_back_to_top_center_when_space_is_tight() -> None:
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    bounds_block = ui[ui.index("function updateResponseLayoutBounds"):ui.index("function createFreeWorkspaceController")]
    apply_block = ui[ui.index("function applyResponseLayout"):ui.index("function updateResponseLayout(element, text)")]

    assert 'anchorResult = {anchor: "top-center", point: zones.echo["top-center"]};' in bounds_block
    assert "const finalAnchor = stage.dataset.entityAnchor || responseAnchorForLayout(stage, layout);" in apply_block
    assert "resolveSafeEchoAnchor(stage, finalAnchor)" in apply_block
    assert '.stage[data-response-layout="focus"][data-entity-anchor="top-center"] .echo-response' in css
    assert "width: min(980px, calc(100% - var(--safe-gap) * 2));" in css
    assert "transform: translateX(-50%) translateY(6px);" in css


def test_web_ui_focus_layout_avoids_telemetry_echo_collision() -> None:
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    safe_anchor_block = ui[ui.index("function resolveSafeEchoAnchor"):ui.index("function updateResponseLayoutBounds")]

    assert 'requestedAnchor === "top-right" && telemetryPanelOpen' in safe_anchor_block
    assert "localRectFromElement(stage, byId(\"telemetryPanel\"))" in safe_anchor_block
    assert "boxesIntersect(echoRect, panelRect)" in safe_anchor_block
    assert 'return {anchor: "center", point: zones.echo.center};' in safe_anchor_block


def test_web_ui_entity_placement_avoids_primary_surface_collision() -> None:
    """L/M (ActiveTaskContext semantics fix, section 13): a voice-driven
    move ('desloca-te para a esquerda') must never land Echo on top of the
    primary WorkspaceObject surface -- resolveSafeEchoAnchor now also
    checks getBoundingClientRect() collision against
    '#workspaceObjects [data-role="primary"]' for ANY requested anchor
    (not just top-right/telemetryPanel), trying same-side alternates
    before falling back to center. entityPlacementOverride itself is never
    cleared by a collision -- only the resolved point changes, so a later
    re-render re-checks fresh geometry instead of losing the voice intent."""
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    safe_anchor_block = ui[ui.index("function resolveSafeEchoAnchor"):ui.index("function updateResponseLayoutBounds")]
    move_block = ui[ui.index("function moveEchoEntityToRegion"):ui.index("function setWorkspaceObjectFocusActive")]

    assert 'document.querySelector(\'#workspaceObjects [data-role="primary"]\')' in ui
    assert "primarySurfaceRect(stage)" in safe_anchor_block
    assert "boxesIntersect(echoRect, surfaceRect)" in safe_anchor_block
    assert "ANCHOR_SAME_SIDE_ALTERNATES" in safe_anchor_block
    # The override object is only ever written by moveEchoEntityToRegion
    # itself (voice command) or cleared by center/auto-organize -- never as
    # a side effect of a collision resolution inside resolveSafeEchoAnchor.
    assert "entityPlacementOverride" not in safe_anchor_block
    assert "entityPlacementOverride = effectiveRegion" in move_block


def test_web_ui_renders_safe_markdown_without_inner_html() -> None:
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")
    css = (WEB / "styles.css").read_text(encoding="utf-8")

    markdown_block = ui[ui.index("function appendInlineMarkdown"):ui.index("const WORKSPACE_STORAGE_KEY")]
    render_block = ui[ui.index("function renderEchoResponse"):ui.index("function applyResponseLayout")]

    assert "function renderSafeMarkdown" in markdown_block
    assert "document.createElement(\"strong\")" in markdown_block
    assert "document.createElement(\"em\")" in markdown_block
    assert "document.createElement(\"ul\")" in markdown_block
    assert "document.createElement(\"li\")" in markdown_block
    assert "document.createElement(\"p\")" in markdown_block
    assert "document.createElement(\"br\")" in markdown_block
    assert "document.createElement(\"code\")" in markdown_block
    assert "document.createTextNode" in markdown_block
    assert ".textContent =" in markdown_block
    assert "innerHTML" not in markdown_block
    assert "renderSafeMarkdown(element, value)" in render_block
    assert "element.textContent = value" not in render_block
    assert "container.dataset.rawText = source" in markdown_block
    assert ".echo-response strong" in css
    assert ".echo-response code" in css
    assert "user-select: text;" in css


def test_web_ui_workspace_resize_clamps_without_destroying_persistent_positions() -> None:
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    assert "layoutFrame" in ui
    assert "updateAdaptiveLayout()" in ui
    assert "clampAfterResize({persist: false, restoreResponseLayout: true})" in ui
    resize_block = ui[ui.index("function clampAfterResize"):ui.index("function bind")]
    assert "if (options.persist) saveWorkspaceState();" in resize_block
    assert "stageSize(stage)" in ui
    assert "workspaceDefaults(stage)" in ui
    assert "resizeWorkspaceItems()" in ui
    assert "safeAreaRect()" in ui


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
    assert 'renderSafeMarkdown(element, value)' in ui
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


def test_cognitive_telemetry_panel_is_available_with_real_runtime_controls() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    assert 'id="telemetryCompact"' in html
    assert 'id="telemetryPanel"' in html
    assert 'aria-controls="telemetryPanel"' in html
    assert 'aria-hidden="true"' in html
    assert "MODEL BEHAVIOUR" in html
    assert "AUTOMATIC CLOUD ROUTING" in html
    assert 'id="automaticClaudeStatus"' in html
    assert 'data-auto-claude="false"' in html
    assert 'data-auto-claude="true"' in html
    assert 'id="dailyBudgetUsd"' in html
    assert 'id="singleCallBudgetUsd"' in html
    assert 'id="saveModelBudget"' in html
    for section in ("status", "models", "workspace", "system"):
        assert f'data-panel-section="{section}"' in html
        assert f'data-panel-page="{section}"' in html
    for mode in ("local", "claude", "automatic"):
        assert f'data-telemetry-mode="{mode}"' in html

    assert ".telemetry-compact" in css
    assert ".telemetry-panel" in css
    assert ".telemetry-panel.open" in css
    assert "backdrop-filter: blur" in css
    assert "@media (prefers-reduced-motion: reduce)" in css

    assert "function applyRealTelemetry" in ui
    assert "configured_model_mode" in ui
    assert "execution_path" in ui
    assert "execution_provider" in ui
    assert "execution_model" in ui
    assert "data.configured_model_mode || data.mode" in ui
    assert "data.execution_provider || data.provider" in ui
    assert "data.execution_model || data.model" in ui
    assert "const telemetryMocks" in ui
    assert "function applyTelemetryMock" in ui
    assert "function setTelemetryPanel" in ui
    assert "function setTelemetryMode" in ui
    assert "window.echoTelemetryDemo = applyTelemetryMock" in ui
    assert 'telemetryDebug") === "1"' in ui
    assert '"set_model_mode"' in ui
    assert '"set_automatic_claude_enabled"' in ui
    assert '"set_model_budget"' in ui
    assert "executeSystemCommand" in ui
    assert "buildSystemCommand" in ui


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
    assert "ANTHROPIC_API_KEY" not in ui
    assert "api_key_source" in ui
    assert "secret_storage_available" in ui


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
        "Social fast path",
        "System datetime",
        "API key required",
        "Cloud routing disabled",
    )
    for label in expected_labels:
        assert label in ui or label in html

    assert "professional_writing" not in html
    assert "structured_summary" not in html
    assert "technical_explanation" not in html
    assert "humanReasonLabel" in ui
    assert "reason_label" in ui


def test_cognitive_telemetry_panel_sections_keep_concerns_separate() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    status_block = html[html.index('data-panel-page="status"'):html.index('data-panel-page="models"')]
    models_block = html[html.index('data-panel-page="models"'):html.index('data-panel-page="workspace"')]
    workspace_block = html[html.index('data-panel-page="workspace"'):html.index('data-panel-page="system"')]
    system_block = html[html.index('data-panel-page="system"'):html.index('</aside>')]

    assert "telemetryMode" in status_block
    assert "telemetrySource" in status_block
    assert "telemetryFallback" in status_block
    assert "data-telemetry-mode" not in status_block
    assert "dailyBudgetUsd" not in status_block
    assert "resetWorkspace" not in status_block

    assert "data-telemetry-mode" in models_block
    assert "data-auto-claude" in models_block
    assert "automaticClaudeStatus" in models_block
    assert "anthropicApiKey" in models_block
    assert "resetWorkspace" not in models_block

    assert "recenterEcho" in workspace_block
    assert "resetWorkspace" in workspace_block
    assert "toggleWorkspaceLock" in workspace_block
    assert "data-telemetry-mode" not in workspace_block

    assert "systemVersion" in system_block
    assert "systemClaude" in system_block
    assert "systemCliOverride" in system_block
    assert "systemModelModeSource" in system_block
    assert "systemAutomaticClaudeSource" in system_block
    assert "systemApiKeySource" in system_block

    assert "openPanelSection" in ui
    assert "ArrowLeft" in ui and "ArrowRight" in ui
    assert "anthropicKeyPanel" in ui
    assert "CONFIGURED BY ENVIRONMENT" in ui
    assert "CONFIGURED SECURELY" in ui
    assert "window.confirm" not in ui
    assert "alert(" not in ui


def test_cognitive_telemetry_compact_line_sits_above_input_zone() -> None:
    css = (WEB / "styles.css").read_text(encoding="utf-8")

    compact_block = css[css.index(".telemetry-compact {"):css.index(".telemetry-compact.visible")]
    visible_block = css[css.index(".telemetry-compact.visible"):css.index(".telemetry-compact::before")]
    input_block = css[css.index(".input-zone {"):css.index(".input-pill {")]

    assert "bottom: calc(var(--input-height) + var(--safe-gap) + 10px);" in compact_block
    assert "bottom: var(--safe-gap);" in input_block
    assert "opacity: .62;" in visible_block
    assert '.stage[data-telemetry-state="thinking_local"] .telemetry-compact' in css
    assert '.stage[data-telemetry-state="response_ready"] .telemetry-compact' in css
    assert "opacity: .9;" in css


def test_cognitive_telemetry_panel_resize_layout_is_explicit() -> None:
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    assert "function updateAdaptiveLayout()" in ui
    assert 'window.addEventListener("resize", () => {' in ui
    assert "window.requestAnimationFrame" in ui
    assert "renderEchoResponse" not in ui[ui.index("function updateAdaptiveLayout"):ui.index("function setInputEnabled")]
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

    assert "bottom: var(--safe-gap);" in input_block
    assert '.stage[data-telemetry-panel="open"] .echo-response' not in compact_block
    assert "inputSafeRect()" in (WEB / "echo_ui.js").read_text(encoding="utf-8")
    assert "rectsIntersect(itemRect(item), inputRect)" in (WEB / "echo_ui.js").read_text(encoding="utf-8")


def test_focus_response_layout_is_available_without_persisting_echo_position() -> None:
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    assert 'stage.dataset.responseLayout = layout' in ui
    assert "function updateResponseLayout(element, text)" in ui
    assert "function applyResponseLayout(element, text" in ui
    assert "function determineResponseLayout(metrics" in ui
    assert 'return "top-right"' in ui
    assert 'return "top-center"' in ui
    assert "restoreEchoPosition" in ui
    assert "applyWorkspaceItem(state.items.echo, false)" in ui
    assert "saveWorkspaceState()" not in ui[ui.index("function applyResponseLayout"):ui.index("function updateResponseLayout")]

    assert '.stage[data-response-layout="focus"] .echo-response' in css
    assert "top: var(--response-top" in css
    assert "max-height: min(var(--response-max-height)" in css
    assert "overflow-y: auto;" in css
    assert "scrollbar-width: thin;" in css
    assert "::-webkit-scrollbar-thumb" in css
    assert "user-select: text;" in css
    assert '.stage[data-response-layout="focus"] .telemetry-compact' in css
    assert "@media (max-height: 720px)" in css


def test_response_layout_hysteresis_and_content_type_forces_focus() -> None:
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    config_block = ui[ui.index("const RESPONSE_LAYOUT_CONFIG"):ui.index("let activeResponseLayout")]
    determine_block = ui[ui.index("function determineResponseLayout"):ui.index("function createWorkspaceItem")]
    classifier_block = ui[ui.index("function classifyResponseContent"):ui.index("function collectResponseMetrics")]

    assert "enterRenderedRatio: 0.40" in config_block
    assert "exitRenderedRatio: 0.34" in config_block
    assert "exitMaxRenderedRatio: 0.22" in config_block
    assert "metrics.renderedRatio > focus.exitRenderedRatio" in determine_block
    assert "metrics.renderedRatio >= focus.enterRenderedRatio" in determine_block
    assert "metrics.codeBlockCount > 0" in determine_block
    assert "metrics.tableLineCount >= 2" in determine_block
    assert '["email", "document", "code", "table"].includes(type)' in determine_block
    assert 'return "email"' in classifier_block
    assert 'return "table"' in classifier_block
    assert 'return "code"' in classifier_block


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
    assert "bottom: var(--safe-gap);" in input_block
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


def test_cancel_current_request_calls_engine_cooperatively_without_terminate() -> None:
    controller = (PROTOTYPE / "controller.py").read_text(encoding="utf-8")

    assert ".terminate()" not in controller
    cancel_block = controller[controller.index("def cancelCurrentRequest"):controller.index("def setState")]
    assert 'getattr(self.responder, "__self__", None)' in cancel_block
    assert 'getattr(owner, "cancel_current_request", None)' in cancel_block
    assert "cancel()" in cancel_block


def test_worker_wires_and_clears_progress_listener_around_responder_call() -> None:
    controller = (PROTOTYPE / "controller.py").read_text(encoding="utf-8")

    assert "progress = Signal(str)" in controller
    run_block = controller[controller.index("def run(self) -> None:"):controller.index("class EchoUIController")]
    assert 'getattr(self.responder, "__self__", None)' in run_block
    assert 'getattr(owner, "set_progress_listener", None)' in run_block
    assert "set_listener(self.progress.emit)" in run_block
    assert "finally:" in run_block
    assert "set_listener(None)" in run_block
    assert "self._worker.progress.connect(self._handle_progress, Qt.ConnectionType.QueuedConnection)" in controller


class _FakeEngine:
    """Stands in for AssistantEngine.respond bound to __self__, matching how
    EchoRequestWorker/EchoUIController look up cancel_current_request and
    set_progress_listener via getattr(responder, '__self__')."""

    def __init__(self, reply: str = "Resposta.", progress_events: tuple[str, ...] = ()) -> None:
        self.reply = reply
        self.progress_events = progress_events
        self.cancelled = False
        self._progress_listener = None

    def set_progress_listener(self, listener) -> None:
        self._progress_listener = listener

    def cancel_current_request(self) -> bool:
        self.cancelled = True
        return True

    def respond(self, message: str) -> str:
        for event in self.progress_events:
            if self._progress_listener is not None:
                self._progress_listener(event)
        return self.reply


def _qt_app():
    from PySide6.QtCore import QCoreApplication

    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    return app


def test_cancel_current_request_invokes_fake_engine_cancellation() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine()
    controller = EchoUIController(engine.respond)
    controller._active = True

    controller.cancelCurrentRequest()

    assert engine.cancelled is True


def test_cancel_current_request_is_a_noop_when_no_request_is_active() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine()
    controller = EchoUIController(engine.respond)

    controller.cancelCurrentRequest()

    assert engine.cancelled is False


def test_submit_message_reaches_idle_cleans_worker_and_delivers_progress_event() -> None:
    from PySide6.QtCore import QEventLoop, QTimer

    from prototype_web_ui.controller import EchoUIController

    app = _qt_app()
    engine = _FakeEngine(reply="Aqui está a versão reescrita.", progress_events=("rewrite_regeneration_started",))
    controller = EchoUIController(engine.respond, thinking_min_ms=0, speaking_duration_ms=0)

    states: list[str] = []
    ui_events: list[str] = []
    finished = {"count": 0}
    controller.stateChanged.connect(states.append)
    controller.uiEvent.connect(ui_events.append)
    controller.requestFinished.connect(lambda: finished.__setitem__("count", finished["count"] + 1))

    # requestFinished (worker/thread cleanup) fires as soon as the response
    # is handled -- well before the thinking->speaking->idle visual timers
    # complete -- so the loop must keep running until "idle" actually shows
    # up in stateChanged, not just quit on the first requestFinished.
    loop = QEventLoop()
    controller.stateChanged.connect(lambda state: loop.quit() if state == "idle" else None)
    controller.submitMessage("Torna-o mais formal, mas não guardes ainda.")
    QTimer.singleShot(5000, loop.quit)
    loop.exec()

    assert finished["count"] == 1
    assert "idle" in states
    assert controller._thread is None
    assert controller._worker is None
    assert controller.has_active_request() is False
    assert any("rewrite_regeneration_started" in payload for payload in ui_events)

    # A new request right after must work normally -- no leftover state from
    # the previous one should block or corrupt it.
    finished["count"] = 0
    states.clear()
    loop2 = QEventLoop()
    controller.stateChanged.connect(lambda state: loop2.quit() if state == "idle" else None)
    controller.submitMessage("Outra mensagem.")
    QTimer.singleShot(5000, loop2.quit)
    loop2.exec()

    assert finished["count"] == 1
    assert "idle" in states
    assert controller.has_active_request() is False


# --- Echo visual-state redesign (Passo 2): idle/listening/thinking/reading/
# working/speaking/compact/focus/error. echo_entity.js exposes
# window.EchoEntity + window.ECHO_STATE_CONFIG, so these run the real
# animation code in Node rather than pattern-matching source text.

ENTITY_JS_PATH = WEB / "echo_entity.js"


def _run_entity_harness(js_tail: str) -> dict:
    import json
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        import pytest

        pytest.skip("node is not available on PATH")

    script = f"""
    global.window = global;
    global.__reducedMotion = false;
    global.performance = {{ now: () => Date.now() }};
    global.requestAnimationFrame = () => 0;
    global.addEventListener = () => {{}};
    global.getComputedStyle = () => ({{ getPropertyValue: () => "" }});
    window.matchMedia = () => ({{
      get matches() {{ return global.__reducedMotion === true; }},
      addEventListener() {{}},
      addListener() {{}}
    }});

    function fakeCtx() {{
      return {{
        setTransform() {{}}, clearRect() {{}}, beginPath() {{}}, arc() {{}}, fill() {{}}, stroke() {{}},
        moveTo() {{}}, lineTo() {{}},
        createRadialGradient() {{ return {{ addColorStop() {{}} }}; }}
      }};
    }}
    function fakeCanvas() {{
      return {{
        getContext: () => fakeCtx(),
        getBoundingClientRect: () => ({{ width: 1300, height: 812 }}),
        closest: () => null,
        addEventListener() {{}},
        width: 0, height: 0
      }};
    }}

    require({json.dumps(str(ENTITY_JS_PATH))});

    {js_tail}
    """
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, f"node harness failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    return json.loads(result.stdout.strip().splitlines()[-1])


SEMANTIC_ENTITY_STATES = {"idle", "listening", "thinking", "reading", "working", "speaking", "error"}
REQUIRED_STATE_CONFIG_KEYS = (
    "spin", "statePulse", "nodeDecay", "edgeDecay", "pulseChance", "driftScale",
    "accent", "outerAlpha", "coreAlpha", "sparkOnEnter", "ambientSparkRate", "periodicFocusSpark",
)


def test_echo_entity_state_config_covers_all_semantic_states() -> None:
    """16.A: each semantic state produces its own configured visual state,
    and an unrecognised state name falls back to idle instead of silently
    rendering nothing special."""
    result = _run_entity_harness(
        """
        const REQUIRED = %s;
        const configKeys = Object.keys(window.ECHO_STATE_CONFIG).sort();
        const missingKeys = {};
        for (const k of configKeys) {
          missingKeys[k] = REQUIRED.filter((r) => !(r in window.ECHO_STATE_CONFIG[k]));
        }
        const entity = new window.EchoEntity(fakeCanvas());
        const observedStates = {};
        for (const name of configKeys) {
          entity.setState(name);
          observedStates[name] = entity.state;
        }
        entity.setState("nonexistent_state_xyz");
        console.log(JSON.stringify({configKeys, missingKeys, observedStates, fallbackState: entity.state}));
        """
        % (list(REQUIRED_STATE_CONFIG_KEYS),)
    )

    assert set(result["configKeys"]) == SEMANTIC_ENTITY_STATES
    assert all(not missing for missing in result["missingKeys"].values()), result["missingKeys"]
    assert all(result["observedStates"][name] == name for name in SEMANTIC_ENTITY_STATES)
    assert result["fallbackState"] == "idle"


def test_echo_entity_layout_role_combines_with_cognitive_state() -> None:
    """16.B: compact/focus are spatial roles orthogonal to the cognitive
    state -- setting one must never overwrite the other."""
    result = _run_entity_harness(
        """
        const entity = new window.EchoEntity(fakeCanvas());
        const combos = [];
        for (const mode of ["idle", "thinking", "working", "speaking"]) {
          for (const role of ["normal", "compact", "focus"]) {
            entity.setState(mode);
            entity.setLayoutRole(role);
            combos.push({mode, role, observedState: entity.state, observedRole: entity.layoutRole});
          }
        }
        console.log(JSON.stringify({combos}));
        """
    )

    for combo in result["combos"]:
        assert combo["observedState"] == combo["mode"], combo
        assert combo["observedRole"] == combo["role"], combo


def test_echo_entity_reduced_motion_lowers_live_motion_params() -> None:
    """16.I: prefers-reduced-motion must reduce (not necessarily eliminate)
    canvas-side motion parameters -- this loop is outside CSS's reach."""
    result = _run_entity_harness(
        """
        function buildAndStep(reduced) {
          global.__reducedMotion = reduced;
          const entity = new window.EchoEntity(fakeCanvas());
          entity.setState("thinking");
          for (let i = 0; i < 240; i += 1) entity.stepLiveParams(0.05);
          return {spin: entity.liveParams.spin, pulseChance: entity.liveParams.pulseChance, driftScale: entity.liveParams.driftScale};
        }
        const normal = buildAndStep(false);
        const reduced = buildAndStep(true);
        console.log(JSON.stringify({normal, reduced}));
        """
    )

    assert result["reduced"]["spin"] < result["normal"]["spin"]
    assert result["reduced"]["pulseChance"] < result["normal"]["pulseChance"]
    assert result["reduced"]["driftScale"] < result["normal"]["driftScale"]


def test_echo_entity_regeneration_intensity_boost_raises_live_params() -> None:
    """16.D (entity side): rewrite_regeneration_started keeps the "working"
    mode but must read as slightly more intense than a plain working tick."""
    result = _run_entity_harness(
        """
        function buildAndStep(boost) {
          const entity = new window.EchoEntity(fakeCanvas());
          entity.setState("working", {intensityBoost: boost});
          for (let i = 0; i < 240; i += 1) entity.stepLiveParams(0.05);
          return {coreAlpha: entity.liveParams.coreAlpha, statePulse: entity.liveParams.statePulse};
        }
        const base = buildAndStep(1);
        const boosted = buildAndStep(1.15);
        console.log(JSON.stringify({base, boosted}));
        """
    )

    assert result["boosted"]["coreAlpha"] > result["base"]["coreAlpha"]
    assert result["boosted"]["statePulse"] > result["base"]["statePulse"]


def test_progress_events_map_to_states_and_labels_without_new_layout() -> None:
    """16.D (UI side): the live rewrite progress events drive mode +
    progressLabel through the single PROGRESS_EVENT_STATE_MAP table."""
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")
    map_block = ui[ui.index("const PROGRESS_EVENT_STATE_MAP"):ui.index("let currentProgressLabel")]

    assert '{mode: "working", progressLabel: "A reescrever…"}' in map_block.replace("\n", " ").replace("  ", " ") or "rewrite_attempt_started" in map_block
    assert "rewrite_validation_started" in map_block and "A validar a versão" in map_block
    assert "rewrite_regeneration_started" in map_block and "intensityBoost: 1.15" in map_block
    assert "rewrite_cancelled" in map_block and '{mode: "idle", progressLabel: ""}' in map_block
    assert "rewrite_timeout" in map_block and '{mode: "error", progressLabel: ""}' in map_block
    assert "cancel_requested" in map_block and "mode: null" in map_block
    # Not a new DOM element / new layout region -- reuses the existing
    # #statusLine element via updateStatusLabel.
    assert "function applyEchoState" in ui
    assert "updateStatusLabel(value, currentProgressLabel)" in ui


def test_progress_events_are_handled_before_workspace_delegation() -> None:
    """Progress events (working/idle/error) are not research/memory
    workspace events -- they must be intercepted before
    window.echoWorkspace.handleEvent, not silently misrouted into it."""
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")
    dispatch_block = ui[ui.index("function handleUiEventPayload"):ui.index("function fadeCurrentReply")]

    assert "if (applyProgressEvent(event)) return;" in dispatch_block
    assert dispatch_block.index("applyProgressEvent(event)") < dispatch_block.index("echoWorkspace.handleEvent")


def test_focus_role_never_hides_echo_entity() -> None:
    """16.C: focus lowers Echo's prominence but never hides it -- no rule
    may set display:none/visibility:hidden on the entity for focus/compact."""
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    role_block = css[css.index('.stage[data-echo-role="compact"] #mind'):css.index('.stage[data-echo-role="compact"] #mind') + 400]

    assert "display: none" not in role_block
    assert "visibility: hidden" not in role_block
    assert "opacity: .92" in role_block or "opacity: .9" in role_block


def test_controller_known_states_include_new_modes_and_skip_speaking_on_cancel_timeout() -> None:
    """16.E/F: cancellation goes straight to idle and timeout flashes error
    then idle -- skipping the normal thinking->speaking animation, driven
    off the same progress-event tracking used for telemetry."""
    controller = (PROTOTYPE / "controller.py").read_text(encoding="utf-8")

    for state in ("idle", "listening", "thinking", "reading", "working", "speaking", "error"):
        assert f'"{state}"' in controller

    assert "self._last_progress_event = str(event or \"\")" in controller
    response_block = controller[controller.index("def _handle_response"):controller.index("def _handle_error")]
    assert 'last_progress_event == "rewrite_cancelled"' in response_block
    assert 'last_progress_event == "rewrite_timeout"' in response_block
    assert 'self._emit_state("idle")' in response_block
    assert 'self._emit_state("error")' in response_block


# --- Passo 3: workspace adaptativo e externalização do trabalho. A new
# centralized WorkspaceObject model/renderer (echo_workspace_objects.js) --
# these are static-content assertions matching this file's existing
# convention (no jsdom/browser execution), since a real interactive check
# happens separately in the manual test.

def test_workspace_objects_module_exists_and_is_wired() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    assert (WEB / "echo_workspace_objects.js").exists()
    assert '<script defer src="./echo_workspace_objects.js"></script>' in html
    assert html.index("echo_entity.js") < html.index("echo_workspace_objects.js") < html.index("echo_ui.js")
    assert 'id="workspaceObjects"' in html
    assert 'id="workspaceTethers"' in html
    assert "window.EchoWorkspaceObjects.create(" in ui
    assert "window.echoWorkspaceObjects = " in ui


def test_workspace_object_model_has_generic_registry_not_per_type_logic() -> None:
    """3: one centralized model/renderer -- type-specific differences live
    in small lookup tables, not scattered per-case branches."""
    objects_js = (WEB / "echo_workspace_objects.js").read_text(encoding="utf-8")

    for name in ("TYPE_LABELS", "ACTION_PHRASES", "ACTION_LABELS", "PRIMARY_ACTIONS", "REFERENCE_ACTIONS"):
        assert name in objects_js
    for fn in (
        "function documentLoaded", "function draftReady", "function comparisonReady",
        "function removeObject", "function render(", "function renderTethers",
    ):
        assert fn in objects_js
    # Action buttons resend the exact backend-recognized phrase -- never a
    # bespoke API call -- so no document logic is duplicated in JS.
    assert '"ainda consegues melhorar?"' in objects_js
    assert '"compara as versões"' in objects_js
    assert '"guarda esta versão"' in objects_js
    assert '"volta à versão anterior"' in objects_js
    assert '"descarta as alterações"' in objects_js


def test_workspace_scene_primitives_are_type_agnostic() -> None:
    """I/J: setPrimary/demoteToReference/openTransientView/closeTransientView/
    recomputeLayoutIntent never mention "document" or "draft" -- only the
    small event-handler functions below them (named after actual backend
    events) do. A future itinerary/decision-matrix scene would reuse these
    same primitives untouched."""
    objects_js = (WEB / "echo_workspace_objects.js").read_text(encoding="utf-8")

    scene_primitives_block = objects_js[
        objects_js.index("function setPrimary"):objects_js.index("// --- Document / draft / comparison lifecycle")
    ]
    for forbidden in ("document", "draft", "comparison"):
        assert forbidden not in scene_primitives_block.lower()
    for name in ("setPrimary", "demoteToReference", "openTransientView", "closeTransientView", "recomputeLayoutIntent"):
        assert f"function {name}" in scene_primitives_block

    assert "getScene" in objects_js
    assert "primaryObjectId" in objects_js
    assert "referenceObjectIds" in objects_js
    assert "supportingObjectIds" in objects_js
    assert "transientObjectId" in objects_js
    assert "layoutIntent" in objects_js
    for intent in ('"single"', '"primary_with_references"', '"compare"'):
        assert intent in objects_js


def test_workspace_document_and_draft_stay_related() -> None:
    """D: a draft created for a document is linked to it via relatedTo, and
    a comparison links back to both."""
    objects_js = (WEB / "echo_workspace_objects.js").read_text(encoding="utf-8")

    draft_ready_block = objects_js[objects_js.index("function draftReady"):objects_js.index("function draftCancelled")]
    assert "relatedTo: doc ? doc.id : null" in draft_ready_block or "draft.relatedTo = doc.id" in draft_ready_block

    comparison_block = objects_js[objects_js.index("function comparisonReady"):objects_js.index("function removeObject")]
    assert "relatedTo: [doc ? doc.id : null, draft ? draft.id : null]" in comparison_block


def test_workspace_revision_updates_existing_draft_not_a_new_card() -> None:
    """C: draftReady looks up an existing draft by file name before ever
    creating a new one -- a second revision updates the same object."""
    objects_js = (WEB / "echo_workspace_objects.js").read_text(encoding="utf-8")

    draft_ready_block = objects_js[objects_js.index("function draftReady"):objects_js.index("function draftCancelled")]
    assert "findByFileName(\"draft\", fields.fileName)" in draft_ready_block
    assert "draft.revision = fields.revision || draft.revision + 1" in draft_ready_block


def test_workspace_cancellation_preserves_valid_draft_timeout_never_removes() -> None:
    """I/J: rewrite_cancelled only removes a draft that never had real
    content; rewrite_timeout never removes anything, it only flags the
    failed attempt -- neither destroys a previously valid draft/original."""
    objects_js = (WEB / "echo_workspace_objects.js").read_text(encoding="utf-8")

    cancelled_block = objects_js[objects_js.index("function draftCancelled"):objects_js.index("function draftTimedOut")]
    assert "hasValidContent" in cancelled_block
    assert "removeObject(draft.id)" in cancelled_block

    timeout_block = objects_js[objects_js.index("function draftTimedOut"):objects_js.index("function draftDiscarded")]
    assert "removeObject" not in timeout_block
    assert "metadata.failed = true" in timeout_block


def test_workspace_drag_updates_position_and_tethers() -> None:
    """H: dragging a card updates its stored position and re-renders
    tethers so lines keep following the object being moved."""
    objects_js = (WEB / "echo_workspace_objects.js").read_text(encoding="utf-8")

    drag_block = objects_js[objects_js.index("function continueDrag"):objects_js.index("function endDrag")]
    assert "obj.position = {" in drag_block
    assert "applyPosition(dragState.id)" in drag_block
    assert "renderTethers()" in drag_block


def test_workspace_manual_reference_position_survives_rail_auto_layout() -> None:
    """J: positionReferenceRail (which measures the primary surface's rect
    every render to place references just below it) explicitly skips any
    reference the user has dragged -- draftReady/a revision update never
    touches obj.position either, so a manually placed reference keeps its
    spot across a revision and a primary/reference role swap."""
    objects_js = (WEB / "echo_workspace_objects.js").read_text(encoding="utf-8")

    rail_block = objects_js[objects_js.index("function positionReferenceRail"):objects_js.index("function scheduleRender")]
    assert 'if (surfaceIds.includes(obj.id) || obj.position || obj.visibility === "hidden") return;' in rail_block
    assert "node.getBoundingClientRect().bottom - hostRect.top" in rail_block
    # Every non-surface object is repositioned, not just
    # scene.referenceObjectIds -- an eclipsed major (a draft while a
    # transient view is open, or a co-major from a side_by_side command)
    # must land in the rail too.
    assert "listObjects().forEach" in rail_block

    apply_position_block = objects_js[objects_js.index("function applyPosition"):objects_js.index("function positionReferenceRail")]
    assert "if (!node || !obj || !obj.position) return;" in apply_position_block

    draft_ready_block = objects_js[objects_js.index("function draftReady"):objects_js.index("function draftCancelled")]
    assert "draft.position" not in draft_ready_block


def test_workspace_one_primary_surface_at_a_time() -> None:
    """F: setPrimary demotes whatever was primary before to a reference --
    there is never more than one object with workspaceRole "primary"."""
    objects_js = (WEB / "echo_workspace_objects.js").read_text(encoding="utf-8")

    set_primary_block = objects_js[objects_js.index("function setPrimary"):objects_js.index("function demoteToReference")]
    assert "demoteToReference(previous)" in set_primary_block
    assert 'objects.get(id).workspaceRole = "primary"' in set_primary_block

    demote_block = objects_js[objects_js.index("function demoteToReference"):objects_js.index("function openTransientView")]
    assert 'obj.workspaceRole = "reference"' in demote_block


def test_workspace_entity_never_hidden_dampened_via_layout_role() -> None:
    """K: the entity is dampened via the existing layoutRole mechanism
    (never hidden, per Passo 2's intensity floor) whenever a surface is
    showing, and restored to "normal" once the workspace is empty."""
    objects_js = (WEB / "echo_workspace_objects.js").read_text(encoding="utf-8")

    presentation_block = objects_js[objects_js.index("function updatePresentation"):objects_js.index("function runAction")]
    assert 'setEntityLayoutRole(hasSurface ? "focus" : "normal")' in presentation_block
    assert "display: none" not in objects_js
    assert "visibility: hidden" not in objects_js


def test_workspace_events_wired_without_duplicating_document_logic() -> None:
    """16: document_loaded/document_draft_ready/document_comparison_ready/
    document_draft_discarded/document_draft_restored/document_draft_saved
    and the existing rewrite_* progress events all drive the workspace
    object model -- through a thin field-forwarding dispatcher, never by
    re-deriving document state in JavaScript."""
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    dispatch_block = ui[ui.index("function handleDocumentWorkspaceEvent"):ui.index("function createWorkspaceController")]
    for event_type in (
        "document_loaded", "document_draft_ready", "document_comparison_ready",
        "document_draft_discarded", "document_draft_restored", "document_draft_saved",
    ):
        assert event_type in dispatch_block

    progress_block = ui[ui.index("function applyWorkspaceObjectProgress"):ui.index("function createWorkspaceController")]
    assert "rewrite_attempt_started" in progress_block
    assert "rewrite_regeneration_started" in progress_block
    assert "rewrite_cancelled" in progress_block
    assert "rewrite_timeout" in progress_block
    assert "draftWorkingStarted" in progress_block
    assert "draftCancelled" in progress_block
    assert "draftTimedOut" in progress_block


def test_workspace_conversation_cleared_clears_workspace_objects() -> None:
    """K: conversation_cleared (and the CLEAR button) also wipe the
    workspace object model, not just the research workspace."""
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    handle_event_block = ui[ui.index("handleEvent(event)"):ui.index("handleUiEventPayload")]
    assert 'event.type === "conversation_cleared"' in handle_event_block
    assert "window.echoWorkspaceObjects.clear()" in handle_event_block


def test_workspace_objects_respect_reduced_motion_and_adaptive_layout() -> None:
    """N/O: workspace-object transitions fall under the site-wide
    prefers-reduced-motion rule (no bespoke override), and the new markup
    does not touch the existing adaptive-layout grid regions."""
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    html = (WEB / "index.html").read_text(encoding="utf-8")

    assert "@media (prefers-reduced-motion: reduce)" in css
    reduced_motion_block = css[css.index("@media (prefers-reduced-motion: reduce)"):]
    assert "transition-duration: .01ms !important;" in reduced_motion_block
    assert ".workspace-object {" in css

    grid_index = html.index("adaptive-layout-grid")
    objects_index = html.index('id="workspaceObjects"')
    tethers_index = html.index('id="workspaceTethers"')
    echo_main_index = html.index('class="echo-main"')
    # workspace objects/tethers live inside .echo-main, alongside #mind and
    # #echoResponse, not inside the (still-empty) adaptive-layout-grid.
    assert grid_index < echo_main_index < min(objects_index, tethers_index)


# --- Focus/comparison visual hierarchy fix: a real manual test found Echo
# and the response panel competing with the focused surface for the same
# screen space. Root causes fixed: wrong safe anchor (top-center instead of
# top-right), a fixed max-width instead of a real left+right reservation for
# Echo's corner, #mind painting below .workspace-objects, and the plain
# response panel never being suppressed while a workspace object is
# protagonist. Static assertions only, no pixel-perfect coordinates.

def test_workspace_document_alone_is_primary() -> None:
    """A: reading a document with no draft yet makes the document object
    the primary surface -- setPrimary only runs once, when nothing else is
    primary yet."""
    objects_js = (WEB / "echo_workspace_objects.js").read_text(encoding="utf-8")

    document_loaded_block = objects_js[objects_js.index("function documentLoaded"):objects_js.index("function draftWorkingStarted")]
    assert "if (!scene.primaryObjectId) setPrimary(existing.id);" in document_loaded_block
    assert "if (!scene.primaryObjectId) setPrimary(obj.id);" in document_loaded_block


def test_workspace_draft_becomes_primary_document_becomes_reference() -> None:
    """B: as soon as a draft exists (even just "working"), it becomes the
    primary surface; the document it came from is demoted to a reference
    generically by setPrimary (see test_workspace_one_primary_surface_at_a_time),
    not by a document-specific branch here."""
    objects_js = (WEB / "echo_workspace_objects.js").read_text(encoding="utf-8")

    working_block = objects_js[objects_js.index("function draftWorkingStarted"):objects_js.index("function activeDocumentFileName")]
    assert "setPrimary(draft.id);" in working_block

    ready_block = objects_js[objects_js.index("function draftReady"):objects_js.index("function draftCancelled")]
    assert "setPrimary(draft.id);" in ready_block  # idempotent if this revision's draft is already primary


def test_workspace_comparison_is_transient_never_a_permanent_card() -> None:
    """D/E: comparisonReady opens a transient view (borrows the surface
    slot without ever promoting the comparison to primary); closing it
    removes the object outright instead of leaving a third permanent card,
    and the scene falls back to the primary (the draft) plus its
    reference."""
    objects_js = (WEB / "echo_workspace_objects.js").read_text(encoding="utf-8")

    comparison_block = objects_js[objects_js.index("function comparisonReady"):objects_js.index("function removeObject")]
    assert "openTransientView(comparison.id);" in comparison_block
    assert "setPrimary(comparison.id)" not in comparison_block

    open_block = objects_js[objects_js.index("function openTransientView"):objects_js.index("function closeTransientView")]
    assert 'scene.layoutIntent = "compare";' in open_block

    close_block = objects_js[objects_js.index("function closeTransientView"):objects_js.index("function recomputeLayoutIntent")]
    assert "removeObject(id);" in close_block
    assert "recomputeLayoutIntent();" in close_block

    recompute_block = objects_js[objects_js.index("function recomputeLayoutIntent"):objects_js.index("// --- Document")]
    assert 'scene.layoutIntent = scene.referenceObjectIds.length > 0 ? "primary_with_references" : "single";' in recompute_block


def test_workspace_reference_cards_have_one_action_primary_has_a_few() -> None:
    """G: a reference object gets a single action ("Abrir"); the primary
    surface carries the 2-4 contextual actions (melhorar/comparar/guardar
    for a draft) -- never the same 5-action panel on every card."""
    objects_js = (WEB / "echo_workspace_objects.js").read_text(encoding="utf-8")
    css = (WEB / "styles.css").read_text(encoding="utf-8")

    assert 'document: ["open"]' in objects_js
    assert 'draft: ["improve", "compare", "save"]' in objects_js
    actions_for_block = objects_js[objects_js.index("function actionsFor"):objects_js.index("function buildNode")]
    assert "REFERENCE_ACTIONS[obj.type] || [\"open\"]" in actions_for_block

    reference_css_index = css.index('.workspace-object[data-role="reference"] {')
    reference_block = css[reference_css_index:reference_css_index + 150]
    assert "width: 190px;" in reference_block


def test_workspace_response_panel_remains_visible_above_active_surfaces() -> None:
    """A conversational reply must remain perceptible while a workspace
    surface is active. The placeholder hint can disappear, but the real
    response panel must not be opacity-hidden by workspace focus."""
    objects_js = (WEB / "echo_workspace_objects.js").read_text(encoding="utf-8")
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")
    css = (WEB / "styles.css").read_text(encoding="utf-8")

    presentation_block = objects_js[objects_js.index("function updatePresentation"):objects_js.index("function runAction")]
    assert "const hasSurface = Boolean(activeSurfaceId());" in presentation_block
    assert "onFocusChange(hasSurface);" in presentation_block

    schedule_render_block = objects_js[objects_js.index("function scheduleRender"):objects_js.index("function buildBody")]
    assert "updatePresentation();" in schedule_render_block

    assert "onFocusChange: setWorkspaceObjectFocusActive" in ui
    assert "setEntityPoint: setWorkspaceObjectEntityPoint" in ui
    assert '.stage[data-workspace-object-focus="true"] .workspace-hint' in css
    assert '.stage[data-workspace-object-focus="true"] .echo-response.visible' in css
    focus_block = css[css.index('.stage[data-workspace-object-focus="true"] .echo-response.visible'):]
    focus_block = focus_block[:focus_block.index("}") + 1]
    assert "opacity: 0" not in focus_block
    assert "z-index: 8;" in focus_block


def test_workspace_echo_positioned_from_measured_primary_surface_rect() -> None:
    """K: Echo's point is computed from the ACTUAL rendered primary/
    transient surface (getBoundingClientRect), not a hardcoded corner --
    section 12's "não usar dimensões hardcoded como arquitetura final". A
    374px CSS reservation still exists for the very first paint, before
    anything has rendered to measure."""
    objects_js = (WEB / "echo_workspace_objects.js").read_text(encoding="utf-8")
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    proximity_block = objects_js[objects_js.index("function updateEchoProximity("):objects_js.index("function centerOf")]
    # Generalized for multi-major layouts (Passo 3.2, section 14): anchors
    # off whichever measured major surface extends furthest right, so Echo
    # clears ALL of them in a side_by_side/split/stack arrangement, not
    # just a single hardcoded one -- still real getBoundingClientRect
    # geometry, never a hardcoded corner. Phase 3C final patch factored the
    # actual point computation out into the shared placeEchoAdjacentTo()
    # (reused by updateEchoProximityForGroup and the compact side_by_side
    # layout too) -- this function's own job is just the measurement.
    assert "node.getBoundingClientRect();" in proximity_block
    assert "nodeRect.right > rect.right" in proximity_block
    assert "placeEchoAdjacentTo(" in proximity_block

    # The shared placement function: still the same "try the right side,
    # fall back below, never overlap" logic, and still the only path that
    # ever calls setEntityPoint for this.
    place_block = objects_js[objects_js.index("function placeEchoAdjacentTo"):objects_js.index("function updateEchoProximityForGroup")]
    assert "setEntityPoint(point);" in place_block
    assert "point.x + entityRadius > hostRect.width" in place_block

    assert "function setWorkspaceObjectEntityPoint(point)" in ui
    assert "window.echoEntity.setCenter(point.x, point.y, false);" in ui


def test_workspace_tethers_hidden_during_compare_restored_after() -> None:
    """A transient comparison is self-contained -- no relation lines are
    drawn while one is open; closing it removes the object (which
    schedules a render) so renderTethers can draw the entity<->primary and
    primary<->reference tethers again."""
    objects_js = (WEB / "echo_workspace_objects.js").read_text(encoding="utf-8")

    tethers_block = objects_js[objects_js.index("function renderTethers"):objects_js.index("function appendTether")]
    assert 'if (scene.layoutIntent === "compare") return;' in tethers_block

    close_block = objects_js[objects_js.index("function closeTransientView"):objects_js.index("function recomputeLayoutIntent")]
    assert "removeObject(id);" in close_block  # removeObject itself calls scheduleRender -> render -> renderTethers


def test_workspace_narrow_layout_reclaims_primary_width_and_stacks_comparison() -> None:
    """L: below 860px, the primary surface no longer reserves Echo's lane
    (the entity is already compact there) and the comparison grid stacks to
    a single column."""
    css = (WEB / "styles.css").read_text(encoding="utf-8")

    narrow_block = css[css.index("@media (max-width: 860px)"):]
    assert "grid-template-columns: 1fr;" in narrow_block
    assert '.workspace-object[data-role="primary"] {' in narrow_block
    assert "right: var(--safe-gap);" in narrow_block


def test_workspace_close_button_shown_only_for_closable_surface() -> None:
    """A closable surface (today: a transient comparison) shows the header
    × button; a primary DOCUMENT/DRAFT has nothing to "close" so it stays
    hidden -- same header-close convention as #researchWorkspace/
    #telemetryPanel."""
    objects_js = (WEB / "echo_workspace_objects.js").read_text(encoding="utf-8")
    css = (WEB / "styles.css").read_text(encoding="utf-8")

    build_node_block = objects_js[objects_js.index("function buildNode"):objects_js.index("function fillActions")]
    assert 'closeButton.dataset.action = "close"' in build_node_block
    # The button lives inside the header, whose own pointerdown starts a
    # drag -- it must be excluded, or a click on it would be swallowed by
    # drag-start instead of firing the "close" action.
    header_pointerdown_block = build_node_block[build_node_block.index('header.addEventListener("pointerdown"'):]
    assert 'event.target.closest("[data-action]")' in header_pointerdown_block[:300]

    update_node_block = objects_js[objects_js.index("function updateNode"):objects_js.index("function render()")]
    assert '(PRIMARY_ACTIONS[obj.type] || []).includes("close")' in update_node_block

    assert ".workspace-object__close {" in css
    assert "display: none;" in css[css.index(".workspace-object__close {"):css.index(".workspace-object__close {") + 300]
    assert '.workspace-object[data-closable="true"] .workspace-object__close' in css


def test_workspace_update_node_writes_dataset_state_for_css_reveal() -> None:
    """Regression: updateNode used to only ever set data-role/data-closable/
    data-working, never data-state -- so the CSS enter-transition rule
    (.workspace-object[data-state="created"/...] { opacity: 1; ... }) never
    matched anything, and every workspace-object stayed at the base rule's
    opacity:0/scale(.98) forever. It was masked only for [data-role=
    "reference"], which happens to carry its own hardcoded opacity:1 -- the
    primary/transient surface (a lone document, or a draft mid-rewrite) had
    no such override and rendered permanently invisible while still
    occupying its measured layout slot ("workspace vazio" + a tether
    pointing at a real-but-invisible center)."""
    objects_js = (WEB / "echo_workspace_objects.js").read_text(encoding="utf-8")
    css = (WEB / "styles.css").read_text(encoding="utf-8")

    # Phase 3B extracted the title/subtitle/body/actions/state-setting
    # logic into a shared fillNodeContent() helper (reused by the new flat
    # representation render path too) -- updateNode still ends up setting
    # data-state on every call, just by delegating to that helper instead
    # of inlining it.
    fill_node_content_block = objects_js[objects_js.index("function fillNodeContent"):objects_js.index("function updateNode")]
    update_node_block = objects_js[objects_js.index("function updateNode"):objects_js.index("function render()")]
    assert "node.dataset.state = obj.state" in fill_node_content_block
    assert "fillNodeContent(" in update_node_block

    base_rule = css[css.index(".workspace-object {"):css.index(".workspace-object {") + 400]
    assert "opacity: 0;" in base_rule

    reveal_rule = css[css.index('.workspace-object[data-state="created"]'):]
    reveal_rule = reveal_rule[:reveal_rule.index("}") + 1]
    assert "opacity: 1;" in reveal_rule
    assert "transform: translateY(0" in reveal_rule

    # The primary role rule must NOT need its own opacity override -- it
    # relies entirely on data-state now being populated by updateNode.
    primary_rule = css[css.index('.workspace-object[data-role="primary"] {'):]
    primary_rule = primary_rule[:primary_rule.index("}") + 1]
    assert "opacity" not in primary_rule


def test_workspace_tethers_skip_invisible_or_disconnected_endpoints() -> None:
    """Section 5: a tether must never be drawn to a surface that isn't
    actually connected/visible -- centerOf (used for both the entity tether
    and relation tethers) checks isConnected, a non-zero rendered rect, and
    effective visibility (display/visibility/opacity) before returning a
    point; callers already skip drawing when either endpoint is null."""
    objects_js = (WEB / "echo_workspace_objects.js").read_text(encoding="utf-8")

    visible_block = objects_js[objects_js.index("function isNodeVisible"):objects_js.index("function centerOf")]
    assert "node.isConnected" in visible_block
    assert "rect.width <= 0 || rect.height <= 0" in visible_block
    assert 'style.display === "none"' in visible_block
    assert 'style.visibility === "hidden"' in visible_block
    assert "Number(style.opacity) <= 0" in visible_block

    center_of_block = objects_js[objects_js.index("function centerOf"):objects_js.index("function tetherPath")]
    assert "isNodeVisible(node)" in center_of_block

    render_tethers_block = objects_js[objects_js.index("function renderTethers"):objects_js.index("function appendTether")]
    assert "if (entityPoint && surfacePoint)" in render_tethers_block
    assert "if (a && b) appendTether" in render_tethers_block


def test_workspace_ensure_primary_surface_is_generic_safety_net() -> None:
    """Section 9: if the scene ever ends up with visible objects but no
    valid primary (an invariant violation, not the normal path), render()
    promotes the most recently touched non-transient object instead of
    silently leaving the workspace empty. Generic: no document/draft/
    comparison-specific logic, reused by any future object type."""
    objects_js = (WEB / "echo_workspace_objects.js").read_text(encoding="utf-8")

    ensure_block = objects_js[
        objects_js.index("function ensurePrimarySurface"):objects_js.index("function sceneInvariantHolds")
    ]
    assert "scene.primaryObjectId && objects.has(scene.primaryObjectId)" in ensure_block
    assert 'workspaceRole !== "transient"' in ensure_block
    assert "metadata.updatedAt > a.metadata.updatedAt" in ensure_block
    assert "setPrimary(next.id)" in ensure_block
    assert "console.warn" in ensure_block
    for forbidden in ("document", "draft", "comparison"):
        assert forbidden not in ensure_block.lower()

    invariant_block = objects_js[
        objects_js.index("function sceneInvariantHolds"):objects_js.index("function render()")
    ]
    assert "scene.primaryObjectId === \"\"" in invariant_block
    assert "objects.has(scene.primaryObjectId)" in invariant_block

    render_block = objects_js[objects_js.index("function render()"):objects_js.index("function updateEchoProximity(")]
    assert render_block.index("ensurePrimarySurface()") < render_block.index("listObjects()")

    assert "sceneInvariantHolds" in objects_js[objects_js.index("return {"):]


def test_workspace_document_loaded_and_draft_working_started_yield_visible_primary() -> None:
    """Cases A and B from the regression report: a lone document becomes
    primary immediately on document_loaded, and a rewrite's working
    placeholder becomes primary immediately on rewrite_attempt_started --
    both go through setPrimary, and since updateNode now always stamps
    data-state, both are guaranteed to hit the CSS reveal rule instead of
    staying at the invisible base state."""
    objects_js = (WEB / "echo_workspace_objects.js").read_text(encoding="utf-8")

    document_loaded_block = objects_js[objects_js.index("function documentLoaded"):objects_js.index("function draftWorkingStarted")]
    assert "setPrimary(obj.id)" in document_loaded_block
    assert "setPrimary(existing.id)" in document_loaded_block

    draft_working_block = objects_js[objects_js.index("function draftWorkingStarted"):objects_js.index("function activeDocumentFileName")]
    assert "setPrimary(draft.id)" in draft_working_block


def test_workspace_remove_object_does_not_orphan_its_dom_node() -> None:
    """Regression found while re-verifying the transient-comparison flow:
    removeObject used to delete the id from BOTH `objects` and `nodes`
    synchronously, before the next render() ran -- render()'s own cleanup
    loop (which detaches a stale node via node.remove()) walks the `nodes`
    map, so an id already missing from it was skipped entirely and its
    DOM element stayed connected forever. In practice this meant closing a
    comparison left its card node in the DOM (isConnected: true) even
    though the object was gone from the registry -- exactly the permanent
    third card this model must never produce. Confirmed live: dom node
    count went 3 -> 2 only after removing the premature nodes.delete(id)."""
    objects_js = (WEB / "echo_workspace_objects.js").read_text(encoding="utf-8")

    remove_object_block = objects_js[objects_js.index("function removeObject"):objects_js.index("function clear()")]
    assert "objects.delete(id);" in remove_object_block
    assert "nodes.delete(id)" not in remove_object_block

    render_block = objects_js[objects_js.index("function render()"):objects_js.index("function updateEchoProximity(")]
    cleanup_block = render_block[render_block.index("nodes.forEach"):]
    assert "node.remove();" in cleanup_block
    assert "nodes.delete(id);" in cleanup_block


# --- Passo 3.2: voice-first / natural-language workspace control. The
# backend (_try_workspace_command in assistant/conversation.py, see
# tests/test_automatic_routing_regressions.py) only recognizes the intent
# and a raw target phrase; everything else -- target RESOLUTION against
# the live scene, scene history, multi-object layout, excerpts -- is pure,
# DOM-free logic in echo_workspace_objects.js (it never touches `document`
# when `container` is null), so these run the REAL code in Node instead of
# pattern-matching source text, the same way test_echo_visual_states.py's
# _run_entity_harness already does for echo_entity.js.

WORKSPACE_JS_PATH = WEB / "echo_workspace_objects.js"


def _run_workspace_harness(js_tail: str) -> dict:
    import json
    import shutil
    import subprocess

    import pytest

    node = shutil.which("node")
    if not node:
        pytest.skip("node is not available on PATH")

    script = f"""
    global.window = global;
    require({json.dumps(str(WORKSPACE_JS_PATH))});

    const submitted = [];
    const focusChanges = [];
    const controller = EchoWorkspaceObjects.create({{
      container: null,
      tetherLayer: null,
      getEntityPoint: () => null,
      setEntityLayoutRole: () => {{}},
      setEntityAnchor: () => {{}},
      setEntityPoint: () => {{}},
      onFocusChange: (hasSurface) => focusChanges.push(hasSurface),
      submit: (text) => submitted.push(text),
    }});

    {js_tail}
    """
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, f"node harness failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    return json.loads(result.stdout.strip().splitlines()[-1])


# --- Phase 3C final patch: the Echo-placement fix (layoutCompactMajorSurfaces
# computing Echo's point directly from known geometry, never by
# re-measuring getBoundingClientRect() a moment after setting it) needs a
# REAL container to exercise -- _run_workspace_harness's container: null
# makes render()/layoutMajorSurfaces bail immediately (see that function's
# own `if (!container) return;` guard), so it can't observe this fix at
# all. This harness provides the smallest DOM shim that satisfies exactly
# what buildNode/updateNode/layoutCompactMajorSurfaces touch (element
# creation, dataset, style get/set, appendChild, a class-name-only
# querySelector) -- not a general jsdom replacement, just enough for these
# geometry-focused tests. hostRect is fully controllable so tests assert
# on deterministic numbers, not a real screen.

def _run_workspace_harness_with_dom(js_tail: str, host_rect: dict) -> dict:
    import json
    import shutil
    import subprocess

    import pytest

    node = shutil.which("node")
    if not node:
        pytest.skip("node is not available on PATH")

    dom_shim = """
    function makeStyle() {
      const store = {};
      return new Proxy(store, {
        get(target, prop) {
          if (prop === "setProperty") return (k, v) => { target[k] = v; };
          if (prop === "removeProperty") return (k) => { delete target[k]; };
          return target[prop];
        },
        set(target, prop, value) { target[prop] = value; return true; },
      });
    }
    function makeElement(tag) {
      const el = {
        tagName: tag,
        className: "",
        dataset: {},
        style: makeStyle(),
        children: [],
        parentNode: null,
        textContent: "",
        innerHTML: "",
        type: "",
        appendChild(child) { child.parentNode = el; el.children.push(child); return child; },
        removeChild(child) { el.children = el.children.filter((c) => c !== child); child.parentNode = null; },
        remove() { if (el.parentNode) el.parentNode.removeChild(el); },
        addEventListener() {},
        setAttribute() {},
        setPointerCapture() {},
        releasePointerCapture() {},
        get isConnected() { return true; },
        querySelector(selector) {
          const className = selector.replace(".", "");
          function find(node) {
            for (const c of node.children) {
              if (String(c.className).split(" ").includes(className)) return c;
              const found = find(c);
              if (found) return found;
            }
            return null;
          }
          return find(el);
        },
        getBoundingClientRect() { return el._rect || {left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0}; },
      };
      return el;
    }
    global.document = {
      createElement: (tag) => makeElement(tag),
      createElementNS: (ns, tag) => makeElement(tag),
      addEventListener: () => {},
      getElementById: () => null,
    };
    global.window = global;
    """

    script = f"""
    {dom_shim}
    require({json.dumps(str(WORKSPACE_JS_PATH))});

    const hostRect = {json.dumps(host_rect)};
    const containerEl = {{
      children: [],
      appendChild(child) {{ child.parentNode = containerEl; containerEl.children.push(child); return child; }},
      removeChild(child) {{ containerEl.children = containerEl.children.filter((c) => c !== child); child.parentNode = null; }},
      querySelector(selector) {{
        const className = selector.replace(":scope > .", "").replace(".", "");
        return containerEl.children.find((c) => String(c.className).split(" ").includes(className)) || null;
      }},
      getBoundingClientRect() {{ return hostRect; }},
      get dataset() {{ return containerEl._dataset || (containerEl._dataset = {{}}); }},
    }};
    const tetherLayerEl = {{ children: [], innerHTML: "", appendChild(child) {{ tetherLayerEl.children.push(child); return child; }} }};

    const entityPointCalls = [];
    const entityAnchorCalls = [];
    const controller = EchoWorkspaceObjects.create({{
      container: containerEl,
      tetherLayer: tetherLayerEl,
      getEntityPoint: () => null,
      setEntityLayoutRole: () => {{}},
      setEntityAnchor: (a) => entityAnchorCalls.push(a),
      setEntityPoint: (p) => entityPointCalls.push(p),
      onFocusChange: () => {{}},
      submit: () => {{}},
    }});

    {js_tail}
    """
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, f"node DOM harness failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_workspace_command_expand_promotes_target_to_primary() -> None:
    """A: 'expande o original' -- resolveTarget's document-type hint finds
    the original among visible objects; applyWorkspaceCommand's "expand"
    case promotes it to primary via the generic setPrimary primitive."""
    result = _run_workspace_harness(
        """
        const doc = controller.createObject({type: "document", title: "email.txt"});
        const draft = controller.createObject({type: "draft", title: "email.txt - draft", primary: true});
        controller.applyWorkspaceCommand({action: "expand", target: "o original"});
        const scene = controller.getScene();
        console.log(JSON.stringify({primaryObjectId: scene.primaryObjectId, docId: doc.id, draftId: draft.id}));
        """
    )
    assert result["primaryObjectId"] == result["docId"]
    assert result["primaryObjectId"] != result["draftId"]


def test_workspace_command_expand_accepts_backend_target_ref() -> None:
    result = _run_workspace_harness(
        """
        const email = controller.createObject({type: "email", title: "Email para Joao"});
        const doc = controller.createObject({type: "document", title: "notas.txt", primary: true});
        controller.applyWorkspaceCommand({action: "expand", target: "mail", targetRef: email.id});
        const scene = controller.getScene();
        console.log(JSON.stringify({primaryObjectId: scene.primaryObjectId, emailId: email.id, docId: doc.id}));
        """
    )
    assert result["primaryObjectId"] == result["emailId"]
    assert result["primaryObjectId"] != result["docId"]


def test_workspace_command_place_beside_creates_side_by_side() -> None:
    """B: 'mostra os dois lado a lado' -- resolveTarget's 'os dois' branch
    resolves to the two most recently focused objects (session-local
    recentObjectIds, never long-term memory); setMajorObjects promotes
    both to major simultaneously and marks the arrangement an explicit
    layoutOverride (section 12/13)."""
    result = _run_workspace_harness(
        """
        const a = controller.createObject({type: "document", title: "email.txt"});
        const b = controller.createObject({type: "draft", title: "email.txt - draft"});
        controller.applyWorkspaceCommand({action: "focus", targetRef: a.id});
        controller.applyWorkspaceCommand({action: "focus", targetRef: b.id});
        controller.applyWorkspaceCommand({action: "place_beside", target: "os dois", relation: "side_by_side"});
        const scene = controller.getScene();
        console.log(JSON.stringify({
          layoutIntent: scene.layoutIntent,
          layoutOverride: scene.layoutOverride,
          layoutSource: scene.layoutSource,
          majors: [scene.primaryObjectId].concat(scene.coObjectIds).sort(),
          aId: a.id, bId: b.id,
        }));
        """
    )
    assert result["layoutIntent"] == "side_by_side"
    assert result["layoutOverride"] is True
    assert result["layoutSource"] == "voice_command"
    assert result["majors"] == sorted([result["aId"], result["bId"]])


def test_workspace_command_minimize_demotes_current_primary() -> None:
    """C: 'minimiza o original' -- minimizeToReference demotes the target
    even when it IS the current primary (unlike plain demoteToReference,
    which only ever ran on the object setPrimary was about to replace)."""
    result = _run_workspace_harness(
        """
        const doc = controller.createObject({type: "document", title: "email.txt", primary: true});
        controller.applyWorkspaceCommand({action: "minimize", target: "o original"});
        const obj = controller.getObject(doc.id);
        const scene = controller.getScene();
        console.log(JSON.stringify({role: obj.workspaceRole, primaryObjectId: scene.primaryObjectId, docId: doc.id}));
        """
    )
    assert result["role"] == "reference"
    assert result["primaryObjectId"] != result["docId"]


def test_workspace_command_scope_all_restores_existing_objects_without_target_lookup() -> None:
    result = _run_workspace_harness(
        """
        const doc = controller.createObject({type: "document", title: "email.txt", primary: true});
        const draft = controller.createObject({type: "draft", title: "email.txt - draft"});
        controller.applyWorkspaceCommand({action: "hide", targetRef: draft.id});
        controller.applyWorkspaceCommand({action: "show", scope: "all"});
        const draftObj = controller.getObject(draft.id);
        const log = controller.commandLog().slice(-1)[0];
        console.log(JSON.stringify({
          docVisible: controller.getObject(doc.id).visibility,
          draftVisible: draftObj.visibility,
          resolution: log.resolution,
          scope: log.command.scope,
          primaryObjectId: log.sceneAfter.primaryObjectId,
        }));
        """
    )
    assert result["docVisible"] == "visible"
    assert result["draftVisible"] == "visible"
    assert result["resolution"] == "resolved"
    assert result["scope"] == "all"
    assert result["primaryObjectId"]


def test_workspace_command_focus_second_version_promotes_draft() -> None:
    """D: 'vamos trabalhar na segunda versão' -- resolveTarget's draft-type
    hint resolves 'segunda versao'/'draft'/'rascunho' to the draft object,
    and focus promotes it (demoting the document that was primary)."""
    result = _run_workspace_harness(
        """
        const doc = controller.createObject({type: "document", title: "email.txt", primary: true});
        const draft = controller.createObject({type: "draft", title: "email.txt - draft"});
        controller.applyWorkspaceCommand({action: "focus", target: "segunda versao"});
        const scene = controller.getScene();
        const docObj = controller.getObject(doc.id);
        console.log(JSON.stringify({primaryObjectId: scene.primaryObjectId, draftId: draft.id, docRole: docObj.workspaceRole}));
        """
    )
    assert result["primaryObjectId"] == result["draftId"]
    assert result["docRole"] == "reference"


def test_workspace_command_focus_excerpt_creates_paragraph_object() -> None:
    """E: 'mostra só o primeiro parágrafo' -- focusExcerpt creates a
    type=excerpt object with parentObjectId/selector (section 9), and it
    becomes primary; the parent is not touched/duplicated."""
    result = _run_workspace_harness(
        """
        const draft = controller.createObject({
          type: "draft", title: "email.txt - draft", primary: true,
          content: "Primeiro paragrafo aqui.\\n\\nSegundo paragrafo aqui.",
        });
        controller.applyWorkspaceCommand({action: "focus_excerpt", selector: {index: 1}});
        const scene = controller.getScene();
        const excerpt = controller.getObject(scene.primaryObjectId);
        console.log(JSON.stringify({
          type: excerpt.type, parentObjectId: excerpt.parentObjectId, draftId: draft.id,
          content: excerpt.content, selectorIndex: excerpt.selector.index,
        }));
        """
    )
    assert result["type"] == "excerpt"
    assert result["parentObjectId"] == result["draftId"]
    assert result["content"] == "Primeiro paragrafo aqui."
    assert result["selectorIndex"] == 1


def test_workspace_command_deictic_resolves_to_active_object() -> None:
    """F: 'chega isso para o lado' -- resolveTarget resolves the bare
    deictic 'isso' (also 'isto'/'aquilo') to whatever was last focused,
    without any textual match against the object's own title."""
    result = _run_workspace_harness(
        """
        const a = controller.createObject({type: "note", title: "Nota qualquer"});
        controller.applyWorkspaceCommand({action: "focus", targetRef: a.id});
        const resolved = controller.resolveTarget("isso");
        console.log(JSON.stringify({resolvedId: resolved ? resolved.id : null, aId: a.id}));
        """
    )
    assert result["resolvedId"] == result["aId"]


def test_workspace_command_show_tags_temporary_reference() -> None:
    """G: a 'show'-style quick-look command ('mostra-me o ficheiro da
    fatura'/'preciso de ver rapidamente...') tags the target as temporary
    via a metadata flag (not workspaceRole, which setPrimary always
    overwrites to "primary" while an object is the active surface) so a
    later bare 'fecha a fatura' knows to restore the previous scene."""
    result = _run_workspace_harness(
        """
        const draft = controller.createObject({type: "draft", title: "email.txt - draft", primary: true});
        const invoice = controller.createObject({type: "document", title: "fatura.pdf"});
        controller.applyWorkspaceCommand({action: "show", target: "fatura"});
        const invoiceObj = controller.getObject(invoice.id);
        const scene = controller.getScene();
        console.log(JSON.stringify({
          primaryObjectId: scene.primaryObjectId, invoiceId: invoice.id,
          role: invoiceObj.workspaceRole, temporary: Boolean(invoiceObj.metadata.temporary),
        }));
        """
    )
    assert result["primaryObjectId"] == result["invoiceId"]
    assert result["role"] == "primary"
    assert result["temporary"] is True


def test_workspace_command_close_temporary_restores_previous_scene() -> None:
    """H: 'fecha a fatura' alone (no explicit 'volta ao que estávamos a
    fazer') restores the previous scene automatically, because the invoice
    was tagged temporary when it was shown -- generalizes section 8-G's
    two-sentence example down to the single sentence the user actually
    said in section 1's example list."""
    result = _run_workspace_harness(
        """
        const draft = controller.createObject({type: "draft", title: "email.txt - draft", primary: true});
        const invoice = controller.createObject({type: "document", title: "fatura.pdf"});
        controller.applyWorkspaceCommand({action: "show", target: "fatura"});
        controller.applyWorkspaceCommand({action: "close", target: "fatura"});
        const scene = controller.getScene();
        console.log(JSON.stringify({
          primaryObjectId: scene.primaryObjectId, draftId: draft.id,
          invoiceStillExists: Boolean(controller.getObject(invoice.id)),
        }));
        """
    )
    assert result["primaryObjectId"] == result["draftId"]
    assert result["invoiceStillExists"] is False


def test_workspace_command_restore_previous_scene_pops_history() -> None:
    """I: 'volta ao que estávamos a fazer' restores the most recent
    meaningfully-different scene snapshot (section 6) -- and degrades
    gracefully (skips an entry, tries the next-older one) if an object it
    referenced no longer exists, instead of restoring a scene pointing at
    nothing."""
    result = _run_workspace_harness(
        """
        const a = controller.createObject({type: "document", title: "primeiro.txt", primary: true});
        const b = controller.createObject({type: "document", title: "segundo.txt"});
        controller.applyWorkspaceCommand({action: "focus", targetRef: b.id}); // pushes history (a was primary)
        controller.applyWorkspaceCommand({action: "restore_previous_scene"});
        const scene = controller.getScene();
        console.log(JSON.stringify({primaryObjectId: scene.primaryObjectId, aId: a.id, bId: b.id}));
        """
    )
    assert result["primaryObjectId"] == result["aId"]


def test_workspace_command_layout_override_survives_refocus_within_set() -> None:
    """J: re-focusing one of the two objects already in an explicit
    side_by_side arrangement does not collapse it back to a single primary
    -- the voice-driven layoutOverride only ends on a genuinely new,
    unrelated target (clearLayoutOverride), never on ordinary interaction
    with the same arrangement."""
    result = _run_workspace_harness(
        """
        const a = controller.createObject({type: "document", title: "a.txt"});
        const b = controller.createObject({type: "document", title: "b.txt"});
        controller.setMajorObjects([a.id, b.id], {source: "voice_command", relation: "side_by_side"});
        controller.applyWorkspaceCommand({action: "focus", targetRef: b.id});
        const scene = controller.getScene();
        console.log(JSON.stringify({
          layoutIntent: scene.layoutIntent, layoutOverride: scene.layoutOverride,
          majors: [scene.primaryObjectId].concat(scene.coObjectIds).sort(), aId: a.id, bId: b.id,
        }));
        """
    )
    assert result["layoutIntent"] == "side_by_side"
    assert result["layoutOverride"] is True
    assert result["majors"] == sorted([result["aId"], result["bId"]])


def test_workspace_command_auto_organize_clears_override() -> None:
    """K: 'organiza isto automaticamente' is the explicit escape hatch
    (besides an object closing or a genuinely new focus target) that ends
    a voice-driven layoutOverride and lets automatic re-layout resume."""
    result = _run_workspace_harness(
        """
        const a = controller.createObject({type: "document", title: "a.txt"});
        const b = controller.createObject({type: "document", title: "b.txt"});
        controller.setMajorObjects([a.id, b.id], {source: "voice_command", relation: "side_by_side"});
        controller.applyWorkspaceCommand({action: "auto_organize"});
        const scene = controller.getScene();
        console.log(JSON.stringify({layoutOverride: scene.layoutOverride, layoutSource: scene.layoutSource, layoutIntent: scene.layoutIntent}));
        """
    )
    assert result["layoutOverride"] is False
    assert result["layoutSource"] == "auto"
    assert result["layoutIntent"] == "primary_with_references"


def test_workspace_command_model_is_type_agnostic() -> None:
    """L: makeWorkspaceCommand/applyWorkspaceCommand's dispatch is a plain
    action->behavior switch with no per-object-type branch (beyond
    resolveTarget's optional document/draft convenience hints, which are
    layered on top of -- not required for -- the generic title/source
    match every other type automatically gets)."""
    objects_js = (WEB / "echo_workspace_objects.js").read_text(encoding="utf-8")

    dispatch_block = objects_js[objects_js.index("function applyWorkspaceCommand"):objects_js.index("// --- Drag")]
    for forbidden in ('obj.type === "document"', 'obj.type === "draft"', 'obj.type === "comparison"'):
        assert forbidden not in dispatch_block

    assert "WORKSPACE_COMMAND_DEFAULTS" in objects_js
    for field in ("action", "target", "targetRef", "relation", "size", "visibility", "region", "selector", "preserveCurrent", "restorePolicy", "confidence"):
        assert f"{field}:" in objects_js[objects_js.index("WORKSPACE_COMMAND_DEFAULTS"):objects_js.index("function makeWorkspaceCommand")]


def test_workspace_command_side_by_side_works_for_simulated_future_types() -> None:
    """M: an itinerary + a budget (simulated future object types --
    section 19's proof, deliberately NOT wired to any real backend
    scenario) can go side_by_side through the exact same setMajorObjects
    primitive a document + draft use -- no per-type code was needed."""
    result = _run_workspace_harness(
        """
        const itinerary = controller.createObject({type: "itinerary", title: "Viagem a Lisboa"});
        const budget = controller.createObject({type: "budget", title: "Orcamento da viagem"});
        controller.applyWorkspaceCommand({action: "focus", targetRef: itinerary.id});
        controller.applyWorkspaceCommand({action: "focus", targetRef: budget.id});
        controller.applyWorkspaceCommand({action: "place_beside", target: "estes dois"});
        const scene = controller.getScene();
        console.log(JSON.stringify({
          layoutIntent: scene.layoutIntent,
          majors: [scene.primaryObjectId].concat(scene.coObjectIds).sort(),
          itineraryId: itinerary.id, budgetId: budget.id,
        }));
        """
    )
    assert result["layoutIntent"] == "side_by_side"
    assert result["majors"] == sorted([result["itineraryId"], result["budgetId"]])


def test_workspace_command_decision_option_can_become_primary() -> None:
    """N: a decision_option (simulated future type) can become primary via
    the generic title-match fallback in resolveTarget -- no dedicated
    'decision_option' branch anywhere in the resolution or dispatch code."""
    result = _run_workspace_harness(
        """
        const note = controller.createObject({type: "note", title: "Notas da reuniao", primary: true});
        const optionA = controller.createObject({type: "decision_option", title: "Opcao A: continuar local"});
        controller.applyWorkspaceCommand({action: "focus", target: "Opcao A"});
        const scene = controller.getScene();
        console.log(JSON.stringify({primaryObjectId: scene.primaryObjectId, optionAId: optionA.id}));
        """
    )
    assert result["primaryObjectId"] == result["optionAId"]


def test_workspace_command_scheduling_keeps_response_panel_suppression_wired() -> None:
    """P: a spatial command's own visible effect is the feedback (section
    16) -- applyWorkspaceCommand ends in scheduleRender(), which still
    drives the same onFocusChange/response-panel-suppression path any
    other scene change already used (Passo 3's continuous suppression),
    so a workspace command doesn't leave a stale conversational response
    fighting for attention with the new arrangement."""
    result = _run_workspace_harness(
        """
        const focusChangesBefore = focusChanges.length;
        const doc = controller.createObject({type: "document", title: "a.txt"});
        controller.applyWorkspaceCommand({action: "expand", targetRef: doc.id});
        console.log(JSON.stringify({
          focusChangesAfter: focusChanges.slice(focusChangesBefore),
        }));
        """
    )
    assert result["focusChangesAfter"] == [True]


# --- Phase 3B (Cognitive Workspace Blueprint v2): REPRESENTATION RENDERER.
# Realizes the backend's RepresentationState (assistant/representation.py,
# the representation_changed UI event) visually. Central mechanism:
# echo_workspace_objects.js's applyRepresentation()/renderRepresentation() --
# side_by_side reuses the EXISTING setMajorObjects/layoutMajorSurfaces
# primitives above unchanged; cards/list/conversation_only are a new flat
# (no primary/reference) render path. The Node harness runs with
# container: null (render() bails immediately, see render()'s own guard),
# so these tests exercise the REAL registry/scene-level logic
# (applyRepresentation, getRepresentation, getScene, visibility, object
# identity) the same way the side_by_side/workspace-command tests above
# do -- DOM/CSS-level claims (visible projection difference, no debug
# chrome, reduced-motion inheritance) are verified structurally against
# the actual source below instead, and were ALSO manually verified against
# a real rendered DOM via a live browser session (see final report).

REPRESENTATION_STYLES = ("cards", "list", "side_by_side", "conversation_only")


def test_representation_a_cards_applied() -> None:
    """A: a representation_changed-shaped payload for style=cards is
    applied and reported back via getRepresentation()."""
    result = _run_workspace_harness(
        """
        const a = controller.createObject({id: "obj-a", type: "option", title: "Suécia"});
        const b = controller.createObject({id: "obj-b", type: "option", title: "Itália"});
        const applied = controller.applyRepresentation({
          style: "cards", visible_object_ids: [a.id, b.id], major_object_ids: [], hidden_object_ids: [], source: "auto",
        });
        console.log(JSON.stringify({applied, representation: controller.getRepresentation()}));
        """
    )
    assert result["applied"] is True
    assert result["representation"]["style"] == "cards"
    assert set(result["representation"]["visibleIds"]) == {"obj-a", "obj-b"}


def test_representation_b_cards_and_list_are_distinct_projections() -> None:
    """B (structural): cards and list are genuinely different projections,
    not the same cards nudged into a column -- list removes card chrome
    (border/shadow/body/actions) and wraps everything in one bordered
    panel, which cards does not."""
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    list_group_block = css[css.index('[data-representation-style="list"] .workspace-representation-group'):]
    list_group_block = list_group_block[:list_group_block.index("}") + 1]

    assert "border:" in list_group_block and "background:" in list_group_block
    assert "border-bottom" in css  # per-row divider, on the list's [data-representation-role] rule
    assert 'data-representation-style="list"] .workspace-object__body' in css
    assert 'data-representation-style="cards"] .workspace-object__body' not in css


def test_representation_c_list_rows_preserve_original_ids() -> None:
    """C: switching cards -> list keeps the exact same object ids/registry
    -- no copied 'list item' semantic objects are ever created."""
    result = _run_workspace_harness(
        """
        const ids = ["obj-a", "obj-b", "obj-c"];
        ids.forEach((id) => controller.createObject({id, type: "option", title: id}));
        controller.applyRepresentation({style: "cards", visible_object_ids: ids, major_object_ids: [], hidden_object_ids: [], source: "auto"});
        const beforeIds = controller.listObjects().map((o) => o.id).sort();
        controller.applyRepresentation({style: "list", visible_object_ids: ids, major_object_ids: [], hidden_object_ids: [], source: "voice_command"});
        const afterIds = controller.listObjects().map((o) => o.id).sort();
        console.log(JSON.stringify({beforeIds, afterIds, visibleInList: controller.getRepresentation().visibleIds.sort()}));
        """
    )
    assert result["beforeIds"] == ["obj-a", "obj-b", "obj-c"]
    assert result["afterIds"] == result["beforeIds"]
    assert result["visibleInList"] == result["beforeIds"]


def test_representation_d_side_by_side_uses_major_object_ids() -> None:
    """D: side_by_side reuses setMajorObjects with EXACTLY the backend-
    resolved major_object_ids -- not whatever the legacy workspace_command
    target-phrase resolver would have guessed."""
    result = _run_workspace_harness(
        """
        const a = controller.createObject({id: "obj-a", title: "Suécia"});
        const b = controller.createObject({id: "obj-b", title: "Itália"});
        const c = controller.createObject({id: "obj-c", title: "Açores"});
        controller.applyRepresentation({
          style: "side_by_side", visible_object_ids: [a.id, b.id, c.id], major_object_ids: [a.id, c.id], hidden_object_ids: [], source: "voice_command",
        });
        const scene = controller.getScene();
        console.log(JSON.stringify({majors: [scene.primaryObjectId].concat(scene.coObjectIds).sort(), layoutIntent: scene.layoutIntent, layoutSource: scene.layoutSource}));
        """
    )
    assert result["majors"] == ["obj-a", "obj-c"]
    assert result["layoutIntent"] == "side_by_side"
    assert result["layoutSource"] == "voice_command"


def test_representation_e_non_major_object_preserved() -> None:
    """E: the object left out of major_object_ids (Itália) is neither
    deleted nor hidden -- it stays registered and visible, just not
    'major'."""
    result = _run_workspace_harness(
        """
        const a = controller.createObject({id: "obj-a", title: "Suécia"});
        const b = controller.createObject({id: "obj-b", title: "Itália"});
        const c = controller.createObject({id: "obj-c", title: "Açores"});
        controller.applyRepresentation({
          style: "side_by_side", visible_object_ids: [a.id, b.id, c.id], major_object_ids: [a.id, c.id], hidden_object_ids: [], source: "voice_command",
        });
        console.log(JSON.stringify({
          bStillRegistered: Boolean(controller.getObject(b.id)),
          bVisibility: controller.getObject(b.id).visibility,
        }));
        """
    )
    assert result["bStillRegistered"] is True
    assert result["bVisibility"] != "hidden"


def test_representation_f_conversation_only_hides_without_deleting() -> None:
    """F: conversation_only hides every object (visibility='hidden') but
    the registry (WorkSession.objects' frontend mirror) keeps all three."""
    result = _run_workspace_harness(
        """
        const ids = ["obj-a", "obj-b", "obj-c"];
        ids.forEach((id) => controller.createObject({id, title: id}));
        controller.applyRepresentation({style: "cards", visible_object_ids: ids, major_object_ids: [], hidden_object_ids: [], source: "auto"});
        controller.applyRepresentation({style: "conversation_only", visible_object_ids: [], major_object_ids: [], hidden_object_ids: ids, source: "voice_command"});
        console.log(JSON.stringify({
          registeredIds: controller.listObjects().map((o) => o.id).sort(),
          visibilities: ids.map((id) => controller.getObject(id).visibility),
        }));
        """
    )
    assert result["registeredIds"] == ["obj-a", "obj-b", "obj-c"]
    assert result["visibilities"] == ["hidden", "hidden", "hidden"]


def test_representation_g_restore_reproduces_previous_layout_state() -> None:
    """G: 'mostra outra vez' is backend-driven (assistant/representation.py's
    restore_representation) -- the frontend just re-applies whatever
    representation_changed event arrives. Re-applying the exact side_by_side
    payload from before conversation_only reproduces the identical scene,
    with no reconstruction needed frontend-side."""
    result = _run_workspace_harness(
        """
        const a = controller.createObject({id: "obj-a", title: "Suécia"});
        const b = controller.createObject({id: "obj-b", title: "Itália"});
        const c = controller.createObject({id: "obj-c", title: "Açores"});
        const sideBySidePayload = {style: "side_by_side", visible_object_ids: [a.id, b.id, c.id], major_object_ids: [a.id, c.id], hidden_object_ids: [], source: "voice_command"};
        controller.applyRepresentation(sideBySidePayload);
        const before = controller.getScene();
        controller.applyRepresentation({style: "conversation_only", visible_object_ids: [], major_object_ids: [], hidden_object_ids: [a.id, b.id, c.id], source: "voice_command"});
        controller.applyRepresentation(sideBySidePayload); // the backend's "restore" result
        const restored = controller.getScene();
        console.log(JSON.stringify({
          beforeMajors: [before.primaryObjectId].concat(before.coObjectIds).sort(),
          restoredMajors: [restored.primaryObjectId].concat(restored.coObjectIds).sort(),
        }));
        """
    )
    assert result["restoredMajors"] == result["beforeMajors"] == ["obj-a", "obj-c"]


def test_representation_h_cards_restored_with_same_ids() -> None:
    """H: the full cards -> list -> side_by_side -> conversation_only ->
    side_by_side -> cards cycle ends with the exact original three ids,
    nothing added or dropped."""
    result = _run_workspace_harness(
        """
        const ids = ["obj-a", "obj-b", "obj-c"];
        ids.forEach((id) => controller.createObject({id, title: id}));
        controller.applyRepresentation({style: "cards", visible_object_ids: ids, major_object_ids: [], hidden_object_ids: [], source: "auto"});
        controller.applyRepresentation({style: "list", visible_object_ids: ids, major_object_ids: [], hidden_object_ids: [], source: "voice_command"});
        controller.applyRepresentation({style: "side_by_side", visible_object_ids: ids, major_object_ids: ["obj-a", "obj-c"], hidden_object_ids: [], source: "voice_command"});
        controller.applyRepresentation({style: "conversation_only", visible_object_ids: [], major_object_ids: [], hidden_object_ids: ids, source: "voice_command"});
        controller.applyRepresentation({style: "side_by_side", visible_object_ids: ids, major_object_ids: ["obj-a", "obj-c"], hidden_object_ids: [], source: "voice_command"});
        controller.applyRepresentation({style: "cards", visible_object_ids: ids, major_object_ids: [], hidden_object_ids: [], source: "voice_command"});
        console.log(JSON.stringify({
          finalStyle: controller.getRepresentation().style,
          finalIds: controller.listObjects().map((o) => o.id).sort(),
          finalVisible: controller.getRepresentation().visibleIds.sort(),
        }));
        """
    )
    assert result["finalStyle"] == "cards"
    assert result["finalIds"] == ["obj-a", "obj-b", "obj-c"]
    assert result["finalVisible"] == ["obj-a", "obj-b", "obj-c"]


def test_representation_i_style_params_update_without_style_change() -> None:
    """I: style_params is a separate mutable channel -- applying the SAME
    style twice with different style_params updates styleParams while
    style itself stays put."""
    result = _run_workspace_harness(
        """
        const a = controller.createObject({id: "obj-a", title: "Suécia"});
        controller.applyRepresentation({style: "cards", visible_object_ids: [a.id], major_object_ids: [], hidden_object_ids: [], style_params: {detail_level: "compact"}, source: "auto"});
        const first = controller.getRepresentation();
        controller.applyRepresentation({style: "cards", visible_object_ids: [a.id], major_object_ids: [], hidden_object_ids: [], style_params: {detail_level: "expanded"}, source: "voice_command"});
        const second = controller.getRepresentation();
        console.log(JSON.stringify({firstStyle: first.style, firstParams: first.styleParams, secondStyle: second.style, secondParams: second.styleParams}));
        """
    )
    assert result["firstStyle"] == result["secondStyle"] == "cards"
    assert result["firstParams"] != result["secondParams"]
    assert result["secondParams"] == {"detail_level": "expanded"}


def test_representation_j_no_geometry_read_from_payload() -> None:
    """J (structural): applyRepresentation only ever reads style/
    visible_object_ids/major_object_ids/hidden_object_ids/style_params/
    source off the incoming event -- never a pixel/coordinate field, since
    the backend never sends one (assistant/representation.py's
    RepresentationState has no x/y/width/height field either)."""
    objects_js = (WEB / "echo_workspace_objects.js").read_text(encoding="utf-8")
    fn_block = objects_js[objects_js.index("function applyRepresentation"):objects_js.index("function visibleIdsForRepresentation")]
    for forbidden in ("p.x", "p.y", "p.width", "p.height", "p.left", "p.top", ".position", ".viewport", ".transform"):
        assert forbidden not in fn_block


def test_representation_k_echo_proximity_reuses_existing_machinery() -> None:
    """K (structural): the flat render path evaluates Echo collision
    against the group's ACTUAL measured geometry (section 7 of the
    cognitive-layout fix -- necessarily its own getBoundingClientRect
    call, since there is no single "primary surface" in a flat style to
    reuse the original function's id-based lookup for), but still only
    ever calls the SAME setEntityPoint/setEntityAnchor frontend callbacks
    the original updateEchoProximity() used -- never a direct
    window.echoEntity.setCenter call of its own.

    Phase 3C micro-fix: the actual call to updateEchoProximityForGroup
    moved out of renderRepresentation() itself and into render()'s shared
    finalizeEchoPlacement() tail (section 1 -- ONE place recalculates Echo
    for every representation, flat or not), so this now checks
    finalizeEchoPlacement instead of renderRepresentation's own body."""
    objects_js = (WEB / "echo_workspace_objects.js").read_text(encoding="utf-8")
    finalize_block = objects_js[
        objects_js.index("function finalizeEchoPlacement"):objects_js.index("function updateEchoProximity(")
    ]
    assert "updateEchoProximityForGroup(ensureRepresentationGroup())" in finalize_block
    assert "setCenter" not in finalize_block

    render_block = objects_js[objects_js.index("function render()"):objects_js.index("function finalizeEchoPlacement")]
    assert "renderRepresentation()" in render_block
    assert "finalizeEchoPlacement(majorLayoutResult)" in render_block
    assert "renderTethers();" in render_block
    assert "setCenter" not in render_block

    group_proximity_block = objects_js[
        objects_js.index("function updateEchoProximityForGroup"):objects_js.index("function renderRepresentation")
    ]
    # Phase 3C final patch: delegates the actual point computation to the
    # shared placeEchoAdjacentTo() (also used by updateEchoProximity() and
    # the compact side_by_side layout) -- still never a direct setCenter.
    assert "placeEchoAdjacentTo(" in group_proximity_block
    assert 'setEntityAnchor("center")' in group_proximity_block
    assert "setCenter" not in group_proximity_block


def test_representation_l_entity_placement_override_path_untouched() -> None:
    """L (structural): echo_ui.js's representation_changed handler forwards
    the event verbatim into applyRepresentation -- it never calls
    moveEchoEntityToRegion or touches entityPlacementOverride itself, so
    an explicit voice placement (section 9) is exactly as protected as it
    already was for every other workspace event (setWorkspaceObjectEntityPoint's
    existing entityPlacementOverride guard, untouched by this phase)."""
    ui_js = (WEB / "echo_ui.js").read_text(encoding="utf-8")
    case_block = ui_js[ui_js.index('case "representation_changed"'):ui_js.index("default:", ui_js.index('case "representation_changed"'))]
    assert "entityPlacementOverride" not in case_block
    assert "moveEchoEntityToRegion" not in case_block
    assert "objects.applyRepresentation(" in case_block


def test_representation_m_email_task_artifact_flow_untouched() -> None:
    """M: the compose_email task_artifact path (upsertTaskArtifact/
    createObject) behaves exactly as before Phase 3B when no representation
    has ever been applied -- primary auto-promotion still works."""
    result = _run_workspace_harness(
        """
        const artifact = controller.upsertTaskArtifact({taskId: "task-1", title: "Email para Joao", subtitle: "ready", content: "Ola"});
        const scene = controller.getScene();
        console.log(JSON.stringify({
          representation: controller.getRepresentation(),
          primaryObjectId: scene.primaryObjectId,
          artifactId: artifact.id,
        }));
        """
    )
    assert result["representation"] is None
    assert result["primaryObjectId"] == result["artifactId"]


def test_representation_n_document_flow_untouched() -> None:
    """N: documentLoaded's primary-promotion behaves exactly as before
    Phase 3B when representation stays null (default, section 1's own
    'render() branch never taps unless a representation_changed event has
    arrived')."""
    result = _run_workspace_harness(
        """
        const doc = controller.documentLoaded({fileName: "a.txt", contentType: "text/plain", chars: 10, preview: "ola"});
        const scene = controller.getScene();
        console.log(JSON.stringify({representation: controller.getRepresentation(), primaryObjectId: scene.primaryObjectId, docId: doc.id}));
        """
    )
    assert result["representation"] is None
    assert result["primaryObjectId"] == result["docId"]


def test_representation_o_reduced_motion_inherited_not_reintroduced() -> None:
    """O (structural): the new cards/list CSS block declares no `transition`
    or `animation` property of its own -- it relies entirely on
    .workspace-object's existing transition (and the global
    prefers-reduced-motion override at the bottom of this file), so no new
    motion needs its own reduced-motion handling."""
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    block_start = css.index("/* --- Phase 3B/3C: Representation Renderer")
    block_end = css.index("@media (max-width: 860px)")
    representation_block = css[block_start:block_end]
    assert "transition:" not in representation_block
    assert "animation:" not in representation_block
    assert "prefers-reduced-motion" in css


def test_representation_p_no_duplicate_semantic_objects_after_toggling() -> None:
    """P: repeated cards -> list -> cards -> list toggling never grows the
    semantic object registry -- exactly 3 ids throughout. DOM-node-level
    de-duplication cannot be observed by this Node harness (container is
    null, see this section's header comment); it was verified against a
    real rendered DOM in a live browser session instead (final report)."""
    result = _run_workspace_harness(
        """
        const ids = ["obj-a", "obj-b", "obj-c"];
        ids.forEach((id) => controller.createObject({id, title: id}));
        const counts = [];
        const styles = ["cards", "list", "cards", "list"];
        styles.forEach((style) => {
          controller.applyRepresentation({style, visible_object_ids: ids, major_object_ids: [], hidden_object_ids: [], source: "voice_command"});
          counts.push(controller.listObjects().length);
        });
        console.log(JSON.stringify({counts, finalIds: controller.listObjects().map((o) => o.id).sort()}));
        """
    )
    assert result["counts"] == [3, 3, 3, 3]
    assert result["finalIds"] == ["obj-a", "obj-b", "obj-c"]


def test_representation_unsupported_style_reports_diagnostic_failure() -> None:
    """Section 18: an unrecognized style is a clear, diagnosable failure
    (representation_render_success=false), not a silent no-op or a crash --
    and the representation mirror is left unchanged."""
    result = _run_workspace_harness(
        """
        const a = controller.createObject({id: "obj-a", title: "Suécia"});
        controller.applyRepresentation({style: "cards", visible_object_ids: [a.id], major_object_ids: [], hidden_object_ids: [], source: "auto"});
        const applied = controller.applyRepresentation({style: "mind_map", visible_object_ids: [a.id], major_object_ids: [], hidden_object_ids: [], source: "voice_command"});
        const log = controller.representationLog();
        console.log(JSON.stringify({applied, stillCards: controller.getRepresentation().style, lastLogEntry: log[log.length - 1]}));
        """
    )
    assert result["applied"] is False
    assert result["stillCards"] == "cards"
    assert result["lastLogEntry"]["representation_render_success"] is False


def test_representation_side_by_side_single_major_expands_one_object() -> None:
    """A single major id is the representation shape produced by
    "aumenta X": it must promote that one object instead of failing the
    renderer just because this is not a comparison pair."""
    result = _run_workspace_harness(
        """
        const a = controller.createObject({id: "obj-a", title: "Suécia"});
        const applied = controller.applyRepresentation({style: "side_by_side", visible_object_ids: [a.id], major_object_ids: [a.id], hidden_object_ids: [], source: "voice_command"});
        const log = controller.representationLog();
        console.log(JSON.stringify({
          applied,
          primaryObjectId: controller.getScene().primaryObjectId,
          lastLogEntry: log[log.length - 1],
        }));
        """
    )
    assert result["applied"] is True
    assert result["primaryObjectId"] == "obj-a"
    assert result["lastLogEntry"]["representation_render_success"] is True


def test_task_artifact_id_is_shared_with_representation_renderer() -> None:
    result = _run_workspace_harness(
        """
        const artifact = controller.upsertTaskArtifact({
          id: "artifact-task-2",
          taskId: "task-2",
          title: "Email para Joao",
          subtitle: "collecting",
          content: "Para quem?"
        });
        const applied = controller.applyRepresentation({
          style: "side_by_side",
          visible_object_ids: ["artifact-task-2"],
          major_object_ids: ["artifact-task-2"],
          hidden_object_ids: [],
          source: "voice_command"
        });
        const log = controller.representationLog();
        console.log(JSON.stringify({
          artifactId: artifact.id,
          primaryObjectId: controller.getScene().primaryObjectId,
          applied,
          lastLogEntry: log[log.length - 1],
        }));
        """
    )
    assert result["artifactId"] == "artifact-task-2"
    assert result["primaryObjectId"] == "artifact-task-2"
    assert result["applied"] is True
    assert result["lastLogEntry"]["representation_render_success"] is True


def test_task_artifact_update_reuses_backend_identity_without_duplicate_dom_object() -> None:
    result = _run_workspace_harness(
        """
        controller.upsertTaskArtifact({id: "artifact-task-2", taskId: "task-2", title: "Email", content: "A"});
        controller.upsertTaskArtifact({id: "artifact-task-2", taskId: "task-2", title: "Email para Joao", content: "B"});
        console.log(JSON.stringify({
          ids: controller.listObjects().map((obj) => obj.id),
          title: controller.getObject("artifact-task-2").title,
          content: controller.getObject("artifact-task-2").content,
        }));
        """
    )
    assert result["ids"] == ["artifact-task-2"]
    assert result["title"] == "Email para Joao"
    assert result["content"] == "B"


# --- Phase 3B integration fix #2: the real Qt run showed the backend's
# SemanticObject registry (assistant/representation.py's WorkSession.
# objects) has no frontend equivalent of its own -- representation_changed
# arrived naming ids the frontend had never heard of, so nothing rendered
# even though the backend reported success. echo_workspace_objects.js
# gained upsertSemanticObject() (the SAME registry every other
# WorkspaceObject lives in, keyed by the backend's own id) and
# applyRepresentation() now resolves every requested id against that
# registry BEFORE claiming success.

def test_semantic_object_sync_registers_into_same_registry() -> None:
    """Backend object registration must land in the EXACT SAME `objects`
    registry applyRepresentation/render() already use -- not a second,
    parallel frontend object system."""
    result = _run_workspace_harness(
        """
        const created = controller.upsertSemanticObject({id: "option-sweden", kind: "option", title: "Sweden"});
        const fromRegistry = controller.getObject("option-sweden");
        console.log(JSON.stringify({
          createdId: created.id,
          sameObject: fromRegistry === created,
          title: fromRegistry.title,
          type: fromRegistry.type,
          inListObjects: controller.listObjects().map((o) => o.id).includes("option-sweden"),
        }));
        """
    )
    assert result["createdId"] == "option-sweden"
    assert result["sameObject"] is True
    assert result["title"] == "Sweden"
    assert result["type"] == "option"
    assert result["inListObjects"] is True


def test_semantic_object_sync_is_idempotent_by_id() -> None:
    """Re-syncing the same backend id updates the existing object rather
    than creating a duplicate -- the semantic id is authoritative."""
    result = _run_workspace_harness(
        """
        controller.upsertSemanticObject({id: "option-sweden", kind: "option", title: "Sweden"});
        controller.upsertSemanticObject({id: "option-sweden", kind: "option", title: "Sweden (updated)"});
        console.log(JSON.stringify({
          count: controller.listObjects().length,
          title: controller.getObject("option-sweden").title,
        }));
        """
    )
    assert result["count"] == 1
    assert result["title"] == "Sweden (updated)"


def test_cards_render_actual_synced_objects_not_only_layout_state() -> None:
    """Section 5: cards must reflect REAL synced objects -- resolved/
    rendered counts equal what was actually registered, not just what the
    representation payload claimed."""
    result = _run_workspace_harness(
        """
        ["option-suecia", "option-italia", "option-acores"].forEach((id) => {
          controller.upsertSemanticObject({id, kind: "option", title: id});
        });
        const applied = controller.applyRepresentation({
          style: "cards",
          visible_object_ids: ["option-suecia", "option-italia", "option-acores"],
          major_object_ids: [], hidden_object_ids: [], source: "auto",
        });
        const log = controller.representationLog();
        console.log(JSON.stringify({applied, diag: log[log.length - 1]}));
        """
    )
    assert result["applied"] is True
    diag = result["diag"]
    assert diag["semantic_objects_requested"] == 3
    assert diag["semantic_objects_resolved"] == 3
    assert diag["semantic_objects_rendered"] == 3
    assert diag["semantic_object_missing_ids"] == []


def test_list_reports_non_empty_resolved_rows() -> None:
    result = _run_workspace_harness(
        """
        ["option-suecia", "option-italia", "option-acores"].forEach((id) => {
          controller.upsertSemanticObject({id, kind: "option", title: id});
        });
        controller.applyRepresentation({style: "cards", visible_object_ids: ["option-suecia","option-italia","option-acores"], major_object_ids: [], hidden_object_ids: [], source: "auto"});
        const applied = controller.applyRepresentation({
          style: "list",
          visible_object_ids: ["option-suecia", "option-italia", "option-acores"],
          major_object_ids: [], hidden_object_ids: [], source: "voice_command",
        });
        const log = controller.representationLog();
        console.log(JSON.stringify({applied, diag: log[log.length - 1]}));
        """
    )
    assert result["applied"] is True
    assert result["diag"]["semantic_objects_rendered"] == 3


def test_side_by_side_resolves_real_synced_nodes() -> None:
    result = _run_workspace_harness(
        """
        ["option-suecia", "option-italia", "option-acores"].forEach((id) => {
          controller.upsertSemanticObject({id, kind: "option", title: id});
        });
        const applied = controller.applyRepresentation({
          style: "side_by_side",
          visible_object_ids: ["option-suecia", "option-italia", "option-acores"],
          major_object_ids: ["option-suecia", "option-acores"], hidden_object_ids: [], source: "voice_command",
        });
        const scene = controller.getScene();
        console.log(JSON.stringify({applied, majors: [scene.primaryObjectId].concat(scene.coObjectIds).sort()}));
        """
    )
    assert result["applied"] is True
    assert result["majors"] == ["option-acores", "option-suecia"]


def test_missing_semantic_ids_cause_render_failure_never_false_success() -> None:
    """The exact regression from the real Qt run: representation_changed
    naming ids that were never synced must report failure, with the
    missing ids named -- not a silent/false success."""
    result = _run_workspace_harness(
        """
        // Deliberately NO upsertSemanticObject calls -- simulates the
        // missing bridge.
        const applied = controller.applyRepresentation({
          style: "cards",
          visible_object_ids: ["option-x", "option-y", "option-z"],
          major_object_ids: [], hidden_object_ids: [], source: "auto",
        });
        const log = controller.representationLog();
        console.log(JSON.stringify({applied, diag: log[log.length - 1]}));
        """
    )
    assert result["applied"] is False
    diag = result["diag"]
    assert diag["representation_render_success"] is False
    assert diag["semantic_objects_requested"] == 3
    assert diag["semantic_objects_resolved"] == 0
    assert diag["semantic_objects_rendered"] == 0
    assert sorted(diag["semantic_object_missing_ids"]) == ["option-x", "option-y", "option-z"]


def test_same_ids_survive_full_representation_cycle_after_sync() -> None:
    """Same semantic ids stay stable and resolved across cards -> list ->
    side_by_side -> conversation_only -> cards, once properly synced."""
    result = _run_workspace_harness(
        """
        const ids = ["option-suecia", "option-italia", "option-acores"];
        ids.forEach((id) => controller.upsertSemanticObject({id, kind: "option", title: id}));
        const diags = [];
        function apply(style, visible, major, hidden) {
          controller.applyRepresentation({style, visible_object_ids: visible, major_object_ids: major, hidden_object_ids: hidden, source: "voice_command"});
          const log = controller.representationLog();
          diags.push(log[log.length - 1].representation_render_success);
        }
        apply("cards", ids, [], []);
        apply("list", ids, [], []);
        apply("side_by_side", ids, ["option-suecia", "option-acores"], []);
        apply("conversation_only", [], [], ids);
        apply("cards", ids, [], []);
        console.log(JSON.stringify({diags, finalIds: controller.listObjects().map((o) => o.id).sort()}));
        """
    )
    assert result["diags"] == [True, True, True, True, True]
    assert result["finalIds"] == ["option-acores", "option-italia", "option-suecia"]


def test_semantic_object_upserted_event_wired_in_echo_ui() -> None:
    """Structural: echo_ui.js's dispatcher forwards semantic_object_upserted
    into upsertSemanticObject verbatim (id/kind/title/task_id) -- reusing
    the same object-identity pipeline task_artifact_updated already has,
    not a second channel."""
    ui_js = (WEB / "echo_ui.js").read_text(encoding="utf-8")
    case_block = ui_js[ui_js.index('case "semantic_object_upserted"'):ui_js.index('case "representation_changed"')]
    assert "objects.upsertSemanticObject(" in case_block
    assert "event.id" in case_block and "event.kind" in case_block and "event.title" in case_block


# --- Phase 3C: Cognitive Layout Baseline. The real Qt manual test showed
# cards pinned to the top-left corner, a list rendered as a full-height
# sidebar, and side_by_side surfaces stretched across the whole viewport
# with an unnecessary tether -- all because the flat representation styles
# were laid out by styling the FULL-VIEWPORT #workspaceObjects container
# directly. echo_workspace_objects.js now hosts flat-style nodes inside a
# shrink-to-fit `.workspace-representation-group` wrapper the container
# only CENTERS (CSS grid place-items), and side_by_side gets a
# content-aware compact layout path (layoutCompactMajorSurfaces) instead
# of always stretching to the full document-editing width. Geometry
# itself (does the group actually end up centered/bounded on screen) was
# additionally verified live against a real rendered DOM in a browser
# session (see final report) -- the Node harness here runs with
# container: null (render() bails immediately), so these are structural/
# registry-level checks, consistent with this file's existing convention
# for claims the harness cannot directly observe.

def test_compact_card_sizing_is_content_driven_not_viewport_relative() -> None:
    """Section 3/4: a title-only card sizes to its own content (fit-content
    within calm min/max bounds), not a fixed viewport-relative box that
    would either stay tiny or balloon just because space exists."""
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    cards_group_rule = css[css.index('[data-representation-style="cards"] .workspace-object[data-representation-role]'):]
    cards_group_rule = cards_group_rule[:cards_group_rule.index("}") + 1]
    assert "width: fit-content;" in cards_group_rule
    assert "min-width:" in cards_group_rule
    assert "max-width:" in cards_group_rule
    # The wrapper group itself is bounded, not viewport-filling.
    group_cards_rule = css[css.index('[data-representation-style="cards"] .workspace-representation-group'):]
    group_cards_rule = group_cards_rule[:group_cards_rule.index("}") + 1]
    assert "max-width:" in group_cards_rule


def test_compact_list_is_content_sized_not_a_full_height_sidebar() -> None:
    """Section 5: the list panel grows with content up to a bounded
    max-height/width -- NOT the old bug of styling the full-viewport
    #workspaceObjects container directly (which painted a border/
    background across the whole screen regardless of 3 short rows)."""
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    list_group_rule = css[css.index('[data-representation-style="list"] .workspace-representation-group'):]
    list_group_rule = list_group_rule[:list_group_rule.index("}") + 1]
    assert "max-height:" in list_group_rule
    assert "width:" in list_group_rule
    assert "height: 100%" not in list_group_rule
    assert "inset:" not in list_group_rule
    # The container itself only ever centers -- border/background/sizing
    # live on the group wrapper, never on the full-bleed container.
    container_rule = css[css.index('.workspace-objects[data-representation-style="cards"],'):]
    container_rule = container_rule[:container_rule.index("}") + 1]
    assert "border" not in container_rule
    assert "background" not in container_rule
    assert "display: grid" in container_rule
    assert "place-items: center" in container_rule


def test_side_by_side_has_bounded_compact_layout_path() -> None:
    """Section 6: two short-content majors get a small, bounded, centered
    pair of surfaces (fixed cardWidth/cardHeight) instead of stretching
    across the whole reserved document-editing width -- gated on content
    length (isCompactObject), never on object kind (section 9)."""
    objects_js = (WEB / "echo_workspace_objects.js").read_text(encoding="utf-8")

    compact_check_block = objects_js[objects_js.index("function isCompactObject"):objects_js.index("let representationGroupEl")]
    assert "obj.title" in compact_check_block and "obj.content" in compact_check_block
    assert "obj.type" not in compact_check_block  # content length decides, never object kind

    layout_major_block = objects_js[objects_js.index("function layoutMajorSurfaces"):objects_js.index("function layoutCompactMajorSurfaces")]
    assert "layoutCompactMajorSurfaces(surfaceIds, hostRect)" in layout_major_block
    assert "isCompactObject(obj)" in layout_major_block

    compact_layout_block = objects_js[objects_js.index("function layoutCompactMajorSurfaces"):objects_js.index("function positionReferenceRail")]
    assert "cardWidth" in compact_layout_block and "cardHeight" in compact_layout_block
    # Bounded constants, not a fraction of the measured host width like
    # the original (non-compact) split below still correctly uses.
    assert "hostRect.width -" not in compact_layout_block.split("usableWidth")[0]


def test_representation_side_by_side_suppresses_meaningless_tether() -> None:
    """Section 6: a representation-driven side_by_side of generic option
    objects carries no real 'Echo is editing this' relationship to show,
    unlike the legacy document/entity side_by_side flow (where
    `representation` stays null and this tether keeps its original
    meaning) -- verified live (no tether drawn) in a real browser session
    (final report); this is the structural guard for that behavior."""
    objects_js = (WEB / "echo_workspace_objects.js").read_text(encoding="utf-8")
    tethers_block = objects_js[objects_js.index("function renderTethers"):objects_js.index("function appendTether")]
    assert 'representation.style === "side_by_side"' in tethers_block
    assert "suppressEntityTether" in tethers_block


def test_echo_recovers_to_safe_anchor_when_group_is_empty_or_degenerate() -> None:
    """Section 7: conversation_only (or any flat style with nothing
    visible) must recover Echo to the safe 'center' anchor -- checking
    BOTH dimensions independently (a collapsed flex group commonly reports
    zero HEIGHT while still inheriting its parent's full WIDTH; checking
    only "both are zero" was the actual bug caught during manual
    verification, see final report)."""
    objects_js = (WEB / "echo_workspace_objects.js").read_text(encoding="utf-8")
    group_proximity_block = objects_js[
        objects_js.index("function updateEchoProximityForGroup"):objects_js.index("function renderRepresentation")
    ]
    assert 'rect.width === 0 || rect.height === 0' in group_proximity_block
    assert 'rect.width === 0 && rect.height === 0' not in group_proximity_block
    assert 'setEntityAnchor("center")' in group_proximity_block


def test_existing_document_side_by_side_layout_path_unaffected() -> None:
    """Section 10: real documents/drafts (their usual, longer content)
    still get the ORIGINAL full-width measured split -- compact treatment
    only applies when EVERY major is short-content, never touching the
    non-compact branch's own geometry math."""
    result = _run_workspace_harness(
        """
        const doc = controller.createObject({type: "document", title: "Relatorio", content: "x".repeat(200)});
        const draft = controller.createObject({type: "draft", title: "Relatorio - rascunho", content: "y".repeat(200)});
        controller.applyWorkspaceCommand({action: "focus", targetRef: doc.id});
        controller.applyWorkspaceCommand({action: "focus", targetRef: draft.id});
        controller.applyWorkspaceCommand({action: "place_beside", target: "estes dois"});
        const scene = controller.getScene();
        console.log(JSON.stringify({
          layoutIntent: scene.layoutIntent,
          majors: [scene.primaryObjectId].concat(scene.coObjectIds).sort(),
          docId: doc.id, draftId: draft.id,
        }));
        """
    )
    assert result["layoutIntent"] == "side_by_side"
    assert result["majors"] == sorted([result["docId"], result["draftId"]])


# --- Phase 3C final patch: Echo safe placement in side_by_side + restore.
# The real Qt manual test showed Echo drifting to the top-left and being
# partially clipped in side_by_side, and restoring reproducing the same
# bad placement. Root cause: updateEchoProximity() re-measured
# getBoundingClientRect() on the two major nodes a moment after
# layoutMajorSurfaces() had just repositioned them via inline styles --
# that measurement cannot be trusted to already reflect the change within
# the same synchronous pass in every runtime, so Echo was anchored off a
# stale (sometimes overlapping) rect. Fixed by having the compact
# side_by_side layout (layoutCompactMajorSurfaces) place Echo directly
# from the SAME geometry it just computed, plus a final viewport clamp in
# the shared placeEchoAdjacentTo(). These tests use
# _run_workspace_harness_with_dom (a real, controllable container) since
# _run_workspace_harness's container: null can't exercise render()/layout
# at all.

_HOST_RECT = {"left": 0, "top": 0, "right": 1220, "bottom": 640, "width": 1220, "height": 640}
_ENTITY_RADIUS = 160

_SIDE_BY_SIDE_SETUP = """
      const a = controller.upsertSemanticObject({id: "opt-a", kind: "option", title: "Suecia"});
      const b = controller.upsertSemanticObject({id: "opt-b", kind: "option", title: "Italia"});
      const c = controller.upsertSemanticObject({id: "opt-c", kind: "option", title: "Acores"});
"""


def test_echo_placement_a_compact_side_by_side_stays_within_viewport() -> None:
    result = _run_workspace_harness_with_dom(
        _SIDE_BY_SIDE_SETUP
        + """
        controller.setMajorObjects(["opt-a", "opt-c"], {source: "voice_command", relation: "side_by_side"});
        controller.render();
        console.log(JSON.stringify({point: entityPointCalls[entityPointCalls.length - 1]}));
        """,
        _HOST_RECT,
    )
    point = result["point"]
    assert point is not None
    assert _ENTITY_RADIUS <= point["x"] <= _HOST_RECT["width"] - _ENTITY_RADIUS
    assert _ENTITY_RADIUS <= point["y"] <= _HOST_RECT["height"] - _ENTITY_RADIUS


def test_echo_placement_b_does_not_overlap_major_surfaces() -> None:
    result = _run_workspace_harness_with_dom(
        _SIDE_BY_SIDE_SETUP
        + """
        controller.setMajorObjects(["opt-a", "opt-c"], {source: "voice_command", relation: "side_by_side"});
        controller.render();
        console.log(JSON.stringify({
          point: entityPointCalls[entityPointCalls.length - 1],
        }));
        """,
        _HOST_RECT,
    )
    point = result["point"]
    # The compact layout centers two 240px-wide, gap-20px cards within
    # (hostRect.width - 374 - 48), starting no earlier than x=24 -- Echo's
    # circle (radius 160) must clear that whole span, either to the right
    # of it or below both cards entirely (never inside their vertical
    # band while also horizontally inside their span).
    card_width, gap, echo_reserve = 240, 20, 374
    usable_width = max(card_width, _HOST_RECT["width"] - echo_reserve - 48)
    total_width = card_width * 2 + gap
    start_x = max(24, (usable_width - total_width) / 2 + 24)
    right_edge = start_x + total_width
    center_y = max(72, _HOST_RECT["height"] * 0.38)
    bottom_edge = center_y + 160
    clears_horizontally = (point["x"] - _ENTITY_RADIUS) >= right_edge
    clears_vertically = (point["y"] - _ENTITY_RADIUS) >= bottom_edge
    assert clears_horizontally or clears_vertically


def test_echo_placement_c_restore_reproduces_safe_placement() -> None:
    result = _run_workspace_harness_with_dom(
        _SIDE_BY_SIDE_SETUP
        + """
        controller.setMajorObjects(["opt-a", "opt-c"], {source: "voice_command", relation: "side_by_side"});
        controller.render();
        const first = entityPointCalls[entityPointCalls.length - 1];
        entityPointCalls.length = 0;
        // Simulate conversation_only then a backend-driven restore back to
        // the exact same major pair -- the frontend just re-applies
        // whatever the backend resolved, same as a real restore event.
        controller.applyRepresentation({style: "conversation_only", visible_object_ids: [], major_object_ids: [], hidden_object_ids: ["opt-a","opt-b","opt-c"], source: "voice_command"});
        controller.render();
        controller.applyRepresentation({style: "side_by_side", visible_object_ids: ["opt-a","opt-b","opt-c"], major_object_ids: ["opt-a","opt-c"], hidden_object_ids: [], source: "voice_command"});
        controller.render();
        const restored = entityPointCalls[entityPointCalls.length - 1];
        console.log(JSON.stringify({first, restored}));
        """,
        _HOST_RECT,
    )
    assert result["restored"] == result["first"]


def test_echo_placement_d_full_cycle_never_clips() -> None:
    """cards -> side_by_side -> conversation_only -> restore never leaves
    Echo outside the viewport at any step."""
    result = _run_workspace_harness_with_dom(
        _SIDE_BY_SIDE_SETUP
        + """
        const points = [];
        const anchors = [];
        function snapshot() {
          points.push(entityPointCalls.slice());
          anchors.push(entityAnchorCalls.slice());
          entityPointCalls.length = 0;
          entityAnchorCalls.length = 0;
        }
        controller.applyRepresentation({style: "cards", visible_object_ids: ["opt-a","opt-b","opt-c"], major_object_ids: [], hidden_object_ids: [], source: "auto"});
        controller.render();
        snapshot();
        controller.applyRepresentation({style: "side_by_side", visible_object_ids: ["opt-a","opt-b","opt-c"], major_object_ids: ["opt-a","opt-c"], hidden_object_ids: [], source: "voice_command"});
        controller.render();
        snapshot();
        controller.applyRepresentation({style: "conversation_only", visible_object_ids: [], major_object_ids: [], hidden_object_ids: ["opt-a","opt-b","opt-c"], source: "voice_command"});
        controller.render();
        snapshot();
        controller.applyRepresentation({style: "side_by_side", visible_object_ids: ["opt-a","opt-b","opt-c"], major_object_ids: ["opt-a","opt-c"], hidden_object_ids: [], source: "voice_command"});
        controller.render();
        snapshot();
        console.log(JSON.stringify({points}));
        """,
        _HOST_RECT,
    )
    all_points = [p for step in result["points"] for p in step]
    assert all_points  # at least the two side_by_side steps produced a point
    for point in all_points:
        assert _ENTITY_RADIUS <= point["x"] <= _HOST_RECT["width"] - _ENTITY_RADIUS
        assert _ENTITY_RADIUS <= point["y"] <= _HOST_RECT["height"] - _ENTITY_RADIUS


def test_echo_placement_e_only_uses_the_override_gated_callbacks() -> None:
    """The compact layout/placement path must only ever call
    setEntityPoint/setEntityAnchor -- the SAME two callbacks echo_ui.js's
    entityPlacementOverride check already gates (setWorkspaceObjectEntityPoint
    no-ops when an explicit override is set) -- never a direct
    window.echoEntity/setCenter call that would bypass that check and
    silently overwrite a valid explicit placement."""
    objects_js = (WEB / "echo_workspace_objects.js").read_text(encoding="utf-8")
    compact_block = objects_js[
        objects_js.index("function layoutCompactMajorSurfaces"):objects_js.index("function positionReferenceRail")
    ]
    place_block = objects_js[
        objects_js.index("function placeEchoAdjacentTo"):objects_js.index("function updateEchoProximityForGroup")
    ]
    for block in (compact_block, place_block):
        assert "window.echoEntity" not in block
        assert "setCenter" not in block
    assert "placeEchoAdjacentTo(" in compact_block
    assert "setEntityPoint(point);" in place_block


# --- Phase 3C micro-fix: recalculate Echo placement on EVERY representation
# change, including transitions back to cards/list after a restore. The real
# Qt bug: after cards -> side_by_side -> conversation_only -> restore(side_
# by_side) -> cards, Echo ended up overlapping/behind the cards group. Two
# confirmed contributing defects, both fixed:
#   (1) updateRepresentationNode() cleared left/top/width/right but never
#       max-height/dataset.split/dataset.compactMajor left behind by a prior
#       compact side_by_side layout, so a card that used to be a major
#       stayed artificially capped/tagged after returning to cards/list.
#   (2) the non-flat render() path only ever appended a BRAND NEW node to
#       `container` -- a node already built under a flat style's
#       `.workspace-representation-group` (or fully detached by
#       conversation_only) was never re-attached, so a restored side_by_side
#       could silently render its majors nowhere at all.
# Both are fixed by (a) clearing the residual state in
# updateRepresentationNode and (b) always reconciling node.parentNode in
# render()'s non-flat branch, plus centralizing Echo placement into ONE
# finalizeEchoPlacement() tail every render() call reaches unconditionally
# (section 1) instead of two divergent per-branch tails.
#
# _run_workspace_harness_with_dom's fake DOM has no real layout engine --
# getBoundingClientRect() only ever returns a manually-stubbed `_rect`
# (defaulting to all-zero). These tests stub a plausible settled cards-row
# rect on `.workspace-representation-group` after each cards/list render to
# exercise updateEchoProximityForGroup's real measurement path, mirroring
# how a real browser's layout would have already settled by the time
# finalizeEchoPlacement measures it.

_CARDS_RECT = {"left": 350, "top": 200, "right": 870, "bottom": 311, "width": 520, "height": 111}
_EMPTY_RECT = {"left": 0, "top": 0, "right": 0, "bottom": 0, "width": 0, "height": 0}

_CARDS_HARNESS_PRELUDE = """
      function setGroupRect(rect) {
        const group = containerEl.querySelector(".workspace-representation-group");
        if (group) group._rect = rect;
        return group;
      }
      const CARDS_RECT = %s;
      const EMPTY_RECT = %s;
      const a = controller.upsertSemanticObject({id: "opt-a", kind: "option", title: "Suecia"});
      const b = controller.upsertSemanticObject({id: "opt-b", kind: "option", title: "Italia"});
      const c = controller.upsertSemanticObject({id: "opt-c", kind: "option", title: "Acores"});
""" % (json.dumps(_CARDS_RECT), json.dumps(_EMPTY_RECT))


def _clears_bounds(point: dict, bounds: dict) -> bool:
    """Mirrors placeEchoAdjacentTo's own "adjacent, not overlapping" check:
    Echo's circle (radius _ENTITY_RADIUS) must clear `bounds` either to the
    right of it or below it entirely."""
    clears_horizontally = (point["x"] - _ENTITY_RADIUS) >= bounds["right"]
    clears_vertically = (point["y"] - _ENTITY_RADIUS) >= bounds["bottom"]
    return clears_horizontally or clears_vertically


def _within_viewport(point: dict) -> bool:
    return (
        _ENTITY_RADIUS <= point["x"] <= _HOST_RECT["width"] - _ENTITY_RADIUS
        and _ENTITY_RADIUS <= point["y"] <= _HOST_RECT["height"] - _ENTITY_RADIUS
    )


def test_echo_placement_micro_a_initial_cards_is_safe() -> None:
    """A: a fresh cards representation (no prior side_by_side/restore in
    its history) places Echo adjacent to the settled group, never
    overlapping it -- the baseline this micro-fix must not regress."""
    result = _run_workspace_harness_with_dom(
        _CARDS_HARNESS_PRELUDE
        + """
        controller.applyRepresentation({style: "cards", visible_object_ids: ["opt-a","opt-b","opt-c"], major_object_ids: [], hidden_object_ids: [], source: "auto"});
        controller.render();
        setGroupRect(CARDS_RECT);
        controller.render();
        console.log(JSON.stringify({point: entityPointCalls[entityPointCalls.length - 1]}));
        """,
        _HOST_RECT,
    )
    point = result["point"]
    assert point is not None
    assert _within_viewport(point)
    assert _clears_bounds(point, _CARDS_RECT)


def test_echo_placement_micro_b_side_by_side_then_cards_clears_residual_state() -> None:
    """B: side_by_side -> cards. Confirms both halves of the fix at once:
    the two previously-major nodes must have their leftover inline
    max-height/dataset.split/dataset.compactMajor cleared (defect 1), and
    Echo's placement must be recalculated safely against the settled
    cards group rather than any stale side_by_side geometry."""
    result = _run_workspace_harness_with_dom(
        _CARDS_HARNESS_PRELUDE
        + """
        controller.applyRepresentation({style: "side_by_side", visible_object_ids: ["opt-a","opt-b","opt-c"], major_object_ids: ["opt-a","opt-c"], hidden_object_ids: [], source: "voice_command"});
        controller.render();
        controller.applyRepresentation({style: "cards", visible_object_ids: ["opt-a","opt-b","opt-c"], major_object_ids: [], hidden_object_ids: [], source: "voice_command"});
        controller.render();
        setGroupRect(CARDS_RECT);
        controller.render();
        const group = containerEl.querySelector(".workspace-representation-group");
        const nodeA = group.children.find((n) => n.dataset.id === "opt-a");
        const nodeC = group.children.find((n) => n.dataset.id === "opt-c");
        console.log(JSON.stringify({
          point: entityPointCalls[entityPointCalls.length - 1],
          aMaxHeight: nodeA.style["max-height"] || null,
          aSplit: nodeA.dataset.split || null,
          aCompactMajor: nodeA.dataset.compactMajor || null,
          cMaxHeight: nodeC.style["max-height"] || null,
          cSplit: nodeC.dataset.split || null,
          cCompactMajor: nodeC.dataset.compactMajor || null,
        }));
        """,
        _HOST_RECT,
    )
    assert result["aMaxHeight"] is None
    assert result["aSplit"] is None
    assert result["aCompactMajor"] is None
    assert result["cMaxHeight"] is None
    assert result["cSplit"] is None
    assert result["cCompactMajor"] is None
    point = result["point"]
    assert point is not None
    assert _within_viewport(point)
    assert _clears_bounds(point, _CARDS_RECT)


def test_echo_placement_micro_c_cards_then_conversation_only_then_cards_is_safe() -> None:
    """C: cards -> conversation_only -> cards. Echo must recover to
    "center" while hidden, then recalculate a safe position again once
    cards come back -- and the cards nodes must actually be re-attached
    to the group (not silently left detached from the conversation_only
    hide)."""
    result = _run_workspace_harness_with_dom(
        _CARDS_HARNESS_PRELUDE
        + """
        controller.applyRepresentation({style: "cards", visible_object_ids: ["opt-a","opt-b","opt-c"], major_object_ids: [], hidden_object_ids: [], source: "auto"});
        controller.render();
        setGroupRect(CARDS_RECT);
        controller.render();
        const firstPoint = entityPointCalls[entityPointCalls.length - 1];

        controller.applyRepresentation({style: "conversation_only", visible_object_ids: [], major_object_ids: [], hidden_object_ids: ["opt-a","opt-b","opt-c"], source: "voice_command"});
        setGroupRect(EMPTY_RECT);
        controller.render();
        const anchorWhileHidden = entityAnchorCalls[entityAnchorCalls.length - 1];

        controller.applyRepresentation({style: "cards", visible_object_ids: ["opt-a","opt-b","opt-c"], major_object_ids: [], hidden_object_ids: [], source: "voice_command"});
        controller.render();
        setGroupRect(CARDS_RECT);
        controller.render();
        const secondPoint = entityPointCalls[entityPointCalls.length - 1];
        const group = containerEl.querySelector(".workspace-representation-group");
        const nodeA = group.children.find((n) => n.dataset.id === "opt-a");
        console.log(JSON.stringify({
          firstPoint, anchorWhileHidden, secondPoint,
          aReattached: nodeA.parentNode === group,
        }));
        """,
        _HOST_RECT,
    )
    assert _within_viewport(result["firstPoint"])
    assert _clears_bounds(result["firstPoint"], _CARDS_RECT)
    assert result["anchorWhileHidden"] == "center"
    assert _within_viewport(result["secondPoint"])
    assert _clears_bounds(result["secondPoint"], _CARDS_RECT)
    assert result["aReattached"] is True


def test_echo_placement_micro_d_full_restore_cycle_never_overlaps_cards() -> None:
    """D (the literal reported bug): cards -> side_by_side ->
    conversation_only -> restore(side_by_side) -> cards must leave Echo
    safely clear of the FINAL cards group -- not overlapping/behind it --
    and the restored-then-returned nodes must carry no leftover side_by_side
    state and must be genuinely attached to the group, not orphaned."""
    result = _run_workspace_harness_with_dom(
        _CARDS_HARNESS_PRELUDE
        + """
        controller.applyRepresentation({style: "cards", visible_object_ids: ["opt-a","opt-b","opt-c"], major_object_ids: [], hidden_object_ids: [], source: "auto"});
        controller.render();
        setGroupRect(CARDS_RECT);
        controller.render();

        controller.applyRepresentation({style: "side_by_side", visible_object_ids: ["opt-a","opt-b","opt-c"], major_object_ids: ["opt-a","opt-c"], hidden_object_ids: [], source: "voice_command"});
        controller.render();

        controller.applyRepresentation({style: "conversation_only", visible_object_ids: [], major_object_ids: [], hidden_object_ids: ["opt-a","opt-b","opt-c"], source: "voice_command"});
        setGroupRect(EMPTY_RECT);
        controller.render();

        // restore: backend re-sends the exact same side_by_side payload.
        controller.applyRepresentation({style: "side_by_side", visible_object_ids: ["opt-a","opt-b","opt-c"], major_object_ids: ["opt-a","opt-c"], hidden_object_ids: [], source: "voice_command"});
        controller.render();

        controller.applyRepresentation({style: "cards", visible_object_ids: ["opt-a","opt-b","opt-c"], major_object_ids: [], hidden_object_ids: [], source: "voice_command"});
        controller.render();
        setGroupRect(CARDS_RECT);
        controller.render();

        const group = containerEl.querySelector(".workspace-representation-group");
        const nodeA = group.children.find((n) => n.dataset.id === "opt-a");
        const nodeB = group.children.find((n) => n.dataset.id === "opt-b");
        const nodeC = group.children.find((n) => n.dataset.id === "opt-c");
        console.log(JSON.stringify({
          point: entityPointCalls[entityPointCalls.length - 1],
          aAttached: nodeA.parentNode === group,
          bAttached: nodeB.parentNode === group,
          cAttached: nodeC.parentNode === group,
          aMaxHeight: nodeA.style["max-height"] || null,
          aSplit: nodeA.dataset.split || null,
          cMaxHeight: nodeC.style["max-height"] || null,
          cSplit: nodeC.dataset.split || null,
        }));
        """,
        _HOST_RECT,
    )
    assert result["aAttached"] is True
    assert result["bAttached"] is True
    assert result["cAttached"] is True
    assert result["aMaxHeight"] is None
    assert result["aSplit"] is None
    assert result["cMaxHeight"] is None
    assert result["cSplit"] is None
    point = result["point"]
    assert point is not None
    assert _within_viewport(point)
    assert _clears_bounds(point, _CARDS_RECT), (
        f"Echo point {point} overlaps the final settled cards group {_CARDS_RECT} "
        "-- the exact reported bug (Echo behind/overlapping the cards after a "
        "full restore cycle)"
    )


def test_echo_placement_micro_e_list_then_cards_then_list_is_safe_in_both() -> None:
    """E: list -> cards -> list. The flat<->flat switch must reach the
    SAME finalizeEchoPlacement recalculation both times (never assume
    the previous flat style already got it right)."""
    result = _run_workspace_harness_with_dom(
        _CARDS_HARNESS_PRELUDE
        + """
        controller.applyRepresentation({style: "list", visible_object_ids: ["opt-a","opt-b","opt-c"], major_object_ids: [], hidden_object_ids: [], source: "auto"});
        controller.render();
        setGroupRect(CARDS_RECT);
        controller.render();
        const listPoint = entityPointCalls[entityPointCalls.length - 1];

        controller.applyRepresentation({style: "cards", visible_object_ids: ["opt-a","opt-b","opt-c"], major_object_ids: [], hidden_object_ids: [], source: "voice_command"});
        controller.render();
        setGroupRect(CARDS_RECT);
        controller.render();
        const cardsPoint = entityPointCalls[entityPointCalls.length - 1];

        controller.applyRepresentation({style: "list", visible_object_ids: ["opt-a","opt-b","opt-c"], major_object_ids: [], hidden_object_ids: [], source: "voice_command"});
        controller.render();
        setGroupRect(CARDS_RECT);
        controller.render();
        const secondListPoint = entityPointCalls[entityPointCalls.length - 1];

        console.log(JSON.stringify({listPoint, cardsPoint, secondListPoint}));
        """,
        _HOST_RECT,
    )
    for point in (result["listPoint"], result["cardsPoint"], result["secondListPoint"]):
        assert _within_viewport(point)
        assert _clears_bounds(point, _CARDS_RECT)


def test_echo_placement_micro_f_explicit_override_survives_representation_changes() -> None:
    """F: setWorkspaceObjectEntityPoint's entityPlacementOverride guard
    (echo_ui.js) is untouched by this micro-fix -- finalizeEchoPlacement
    still only ever calls the SAME injected setEntityPoint/setEntityAnchor
    callbacks, never window.echoEntity.setCenter directly, so an active
    voice-driven placement override continues to silently swallow every
    one of them across a full cards/side_by_side/conversation_only cycle
    (more representation-change events now trigger recalculation than
    before this fix -- the override must still win every single time)."""
    ui_js = (WEB / "echo_ui.js").read_text(encoding="utf-8")
    point_fn_block = ui_js[
        ui_js.index("function setWorkspaceObjectEntityPoint"):ui_js.index("function moveEchoEntityToRegion")
    ]
    assert "if (entityPlacementOverride) return;" in point_fn_block

    result = _run_workspace_harness_with_dom(
        """
        let overrideActive = false;
        const guardedPoints = [];
        // Mirrors echo_ui.js's setWorkspaceObjectEntityPoint guard exactly --
        // a real override silently drops every subsequent point.
        function guardedSetEntityPoint(p) { if (!overrideActive) guardedPoints.push(p); }
        const guardedController = EchoWorkspaceObjects.create({
          container: containerEl, tetherLayer: tetherLayerEl,
          getEntityPoint: () => null, setEntityLayoutRole: () => {},
          setEntityAnchor: () => {}, setEntityPoint: guardedSetEntityPoint,
          onFocusChange: () => {}, submit: () => {},
        });
        guardedController.upsertSemanticObject({id: "opt-a", kind: "option", title: "Suecia"});
        guardedController.upsertSemanticObject({id: "opt-b", kind: "option", title: "Italia"});
        guardedController.upsertSemanticObject({id: "opt-c", kind: "option", title: "Acores"});
        guardedController.applyRepresentation({style: "cards", visible_object_ids: ["opt-a","opt-b","opt-c"], major_object_ids: [], hidden_object_ids: [], source: "auto"});
        guardedController.render();

        overrideActive = true;
        guardedPoints.length = 0;
        guardedController.applyRepresentation({style: "side_by_side", visible_object_ids: ["opt-a","opt-b","opt-c"], major_object_ids: ["opt-a","opt-c"], hidden_object_ids: [], source: "voice_command"});
        guardedController.render();
        guardedController.applyRepresentation({style: "conversation_only", visible_object_ids: [], major_object_ids: [], hidden_object_ids: ["opt-a","opt-b","opt-c"], source: "voice_command"});
        guardedController.render();
        guardedController.applyRepresentation({style: "cards", visible_object_ids: ["opt-a","opt-b","opt-c"], major_object_ids: [], hidden_object_ids: [], source: "voice_command"});
        guardedController.render();
        console.log(JSON.stringify({ pointsWhileOverridden: guardedPoints.length }));
        """,
        _HOST_RECT,
    )
    assert result["pointsWhileOverridden"] == 0


# =============================================================================
# Phase 4 (Cognitive Workspace Blueprint v2): ATTENTION + SPACE PRESSURE
# (frontend half -- echo_workspace_objects.js's applyAttention/
# computeSpacePressure, and echo_ui.js's attention_changed dispatch).
# =============================================================================


def test_phase4_attention_stamps_data_attention_on_flat_nodes() -> None:
    """Section 17/18: applyAttention() -- a completely separate mirror
    from `representation` -- stamps data-attention on the SAME nodes the
    (unrelated) representation style already renders, never touching
    data-representation-role."""
    result = _run_workspace_harness_with_dom(
        _CARDS_HARNESS_PRELUDE
        + """
        controller.applyRepresentation({style: "cards", visible_object_ids: ["opt-a","opt-b","opt-c"], major_object_ids: [], hidden_object_ids: [], source: "auto"});
        controller.render();
        controller.applyAttention({object_states: {"opt-a": "focus", "opt-b": "active", "opt-c": "glanceable"}, focus_object_id: "opt-a", source: "auto"});
        controller.render();
        const group = containerEl.querySelector(".workspace-representation-group");
        const nodeA = group.children.find((n) => n.dataset.id === "opt-a");
        const nodeB = group.children.find((n) => n.dataset.id === "opt-b");
        const nodeC = group.children.find((n) => n.dataset.id === "opt-c");
        console.log(JSON.stringify({
          aAttention: nodeA.dataset.attention,
          bAttention: nodeB.dataset.attention,
          cAttention: nodeC.dataset.attention,
          aRepresentationRole: nodeA.dataset.representationRole,
        }));
        """,
        _HOST_RECT,
    )
    assert result["aAttention"] == "focus"
    assert result["bAttention"] == "active"
    assert result["cAttention"] == "glanceable"
    assert result["aRepresentationRole"] == "peer"  # untouched by attention


def test_phase4_attention_defaults_to_visible_for_unknown_objects() -> None:
    result = _run_workspace_harness_with_dom(
        _CARDS_HARNESS_PRELUDE
        + """
        controller.applyRepresentation({style: "cards", visible_object_ids: ["opt-a","opt-b","opt-c"], major_object_ids: [], hidden_object_ids: [], source: "auto"});
        controller.render();
        const group = containerEl.querySelector(".workspace-representation-group");
        const nodeA = group.children.find((n) => n.dataset.id === "opt-a");
        console.log(JSON.stringify({ aAttention: nodeA.dataset.attention }));
        """,
        _HOST_RECT,
    )
    assert result["aAttention"] == "visible"


def test_phase4_minimized_has_lightweight_css_rule() -> None:
    """P: MINIMIZED collapses to a compact chip -- header/body/actions all
    hidden, no heavy dock/taskbar element introduced."""
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    minimized_block = css[css.index('[data-attention="minimized"]'):]
    assert "display: none" in minimized_block[: minimized_block.index("@media")]
    assert '.workspace-object[data-attention="minimized"] .workspace-object__header' in css
    assert '.workspace-object[data-attention="minimized"] .workspace-object__body' in css
    assert '.workspace-object[data-attention="minimized"] .workspace-object__actions' in css


def test_phase4_hidden_attention_is_not_rendered() -> None:
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    assert '.workspace-object[data-attention="hidden"]' in css
    hidden_rule = css[css.index('.workspace-object[data-attention="hidden"]'):]
    assert "display: none" in hidden_rule[:120]


def test_phase4_attention_changed_event_dispatch_carries_no_geometry() -> None:
    """O: echo_ui.js's attention_changed handler forwards object_states/
    focus_object_id/source verbatim -- structurally, no x/y/width/height
    field is ever read off the event."""
    ui_js = (WEB / "echo_ui.js").read_text(encoding="utf-8")
    case_block = ui_js[ui_js.index('case "attention_changed"'):ui_js.index("default:", ui_js.index('case "attention_changed"'))]
    assert "objects.applyAttention(" in case_block
    for geometry_key in ("event.x", "event.y", "event.width", "event.height", "event.position"):
        assert geometry_key not in case_block


def test_phase4_compute_space_pressure_returns_only_semantic_fields() -> None:
    """Section 11: computeSpacePressure() -- the frontend -> backend
    layout signal -- returns exactly the semantic fields the phase spec
    lists, never a pixel/coordinate."""
    result = _run_workspace_harness_with_dom(
        _CARDS_HARNESS_PRELUDE
        + """
        controller.applyRepresentation({style: "cards", visible_object_ids: ["opt-a","opt-b","opt-c"], major_object_ids: [], hidden_object_ids: [], source: "auto"});
        controller.render();
        const pressure = controller.computeSpacePressure();
        console.log(JSON.stringify({ pressure, keys: Object.keys(pressure).sort() }));
        """,
        _HOST_RECT,
    )
    assert result["keys"] == ["major_surface_count", "overflow_risk", "pressure", "visible_object_count"]
    assert result["pressure"]["pressure"] in ("low", "medium", "high")
    assert isinstance(result["pressure"]["visible_object_count"], int)
    assert isinstance(result["pressure"]["overflow_risk"], bool)


def test_phase4_report_capacity_callback_only_fires_on_change() -> None:
    """Section 11/12: reportCapacity is only invoked when the computed
    signal actually differs from the previous report -- a steady-state
    workspace does not spam the backend on every render. The shared
    _CARDS_HARNESS_PRELUDE controller has no reportCapacity hook, so this
    test builds its own, wired to a counting callback."""
    result = _run_workspace_harness_with_dom(
        """
        const reports = [];
        const countingController = EchoWorkspaceObjects.create({
          container: containerEl, tetherLayer: tetherLayerEl,
          getEntityPoint: () => null, setEntityLayoutRole: () => {},
          setEntityAnchor: () => {}, setEntityPoint: () => {},
          onFocusChange: () => {}, submit: () => {},
          reportCapacity: (payload) => reports.push(payload),
        });
        countingController.upsertSemanticObject({id: "opt-a", kind: "option", title: "Suecia"});
        countingController.upsertSemanticObject({id: "opt-b", kind: "option", title: "Italia"});
        countingController.applyRepresentation({style: "cards", visible_object_ids: ["opt-a","opt-b"], major_object_ids: [], hidden_object_ids: [], source: "auto"});
        countingController.render();
        countingController.render();
        countingController.render();
        console.log(JSON.stringify({ reportCount: reports.length, lastReport: reports[reports.length - 1] }));
        """,
        _HOST_RECT,
    )
    assert result["reportCount"] == 1  # only the first render's report differs from "nothing yet"
    for geometry_key in ("x", "y", "width", "height", "position"):
        assert geometry_key not in result["lastReport"]

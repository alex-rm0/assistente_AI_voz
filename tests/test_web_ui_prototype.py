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

    assert "requestedAnchor !== \"top-right\" || !telemetryPanelOpen" in safe_anchor_block
    assert 'const panelRect = localRectFromElement(stage, byId("telemetryPanel"));' in safe_anchor_block
    assert "if (!boxesIntersect(echoRect, panelRect)) return {anchor: requestedAnchor, point};" in safe_anchor_block
    assert 'return {anchor: "top-center", point: zones.echo["top-center"]};' in safe_anchor_block


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

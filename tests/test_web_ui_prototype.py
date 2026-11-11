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


def test_web_ui_echo_workspace_point_is_clamped_against_persistent_surfaces() -> None:
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    for name in (
        "function occupiedWorkspaceRects(stage)",
        "function safeEchoWorkspaceRect(stage, radius)",
        "function clampEchoPointToSafeWorkspace(stage, point)",
    ):
        assert name in ui

    occupied_block = ui[ui.index("function occupiedWorkspaceRects"):ui.index("function safeEchoWorkspaceRect")]
    point_block = ui[ui.index("function setWorkspaceObjectEntityPoint"):ui.index("function setWorkspaceObjectFocusActive")]

    assert '#workspaceObjects [data-task-surface-slot="true"]' in occupied_block
    assert '#workspaceObjects [data-minimized-rail="true"]' in occupied_block
    assert '#workspaceObjects [data-role="primary"]' in occupied_block
    assert 'byId("researchWorkspace")' in occupied_block
    assert 'researchWorkspace.classList.contains("visible")' in occupied_block
    assert 'byId("telemetryPanel")' in occupied_block
    assert "telemetryPanelOpen" in occupied_block
    assert "clampEchoPointToSafeWorkspace(stage, point)" in point_block
    assert "window.echoEntity.setCenter(safePoint.x, safePoint.y, false);" in point_block


def test_web_ui_workspace_objects_receive_real_echo_entity_point() -> None:
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")
    creation_block = ui[ui.index("window.echoWorkspaceObjects = window.EchoWorkspaceObjects.create"):ui.index("window.echoFreeWorkspace =")]
    point_block = ui[ui.index("function getWorkspaceObjectEntityPoint"):ui.index("function moveEchoEntityToRegion")]

    assert "getEntityPoint: getWorkspaceObjectEntityPoint" in creation_block
    assert "window.echoEntity.getCenter()" in point_block
    assert 'window.echoFreeWorkspace.getItem("echo")' in point_block
    assert "getItem(itemId)" in ui


def test_web_ui_echo_safe_rect_is_frontend_only_and_uses_minimal_correction() -> None:
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")
    objects_js = (WEB / "echo_workspace_objects.js").read_text(encoding="utf-8")

    safe_block = ui[ui.index("function safeEchoWorkspaceRect"):ui.index("function clampEchoPointToSafeWorkspace")]
    clamp_block = ui[ui.index("function clampEchoPointToSafeWorkspace"):ui.index("const ANCHOR_SAME_SIDE_ALTERNATES")]

    assert "occupiedWorkspaceRects(stage)" in safe_block
    assert "rect.right + safeGap" in safe_block
    assert "rect.left - safeGap" in safe_block
    assert "const candidates = [base];" in clamp_block
    assert "bestDistance" in clamp_block
    assert "boxesIntersect(echoRect, rect)" in clamp_block
    assert "safeEchoWorkspaceRect" not in objects_js
    assert "clampEchoPointToSafeWorkspace" not in objects_js


def test_web_ui_echo_safe_rect_covers_task_surface_research_panel_and_repeated_recompute() -> None:
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    occupied_block = ui[ui.index("function occupiedWorkspaceRects"):ui.index("function safeEchoWorkspaceRect")]
    safe_block = ui[ui.index("function safeEchoWorkspaceRect"):ui.index("function clampEchoPointToSafeWorkspace")]
    point_block = ui[ui.index("function setWorkspaceObjectEntityPoint"):ui.index("function setWorkspaceObjectFocusActive")]

    # Task Surface only and resized Task Surface: every call reads the real
    # current DOM boxes, including task-surface slots and minimized rail.
    assert "document.querySelectorAll(selector).forEach" in occupied_block
    assert 'data-task-surface-slot="true"' in occupied_block
    assert 'data-minimized-rail="true"' in occupied_block
    assert "localRectFromElement(stage, node)" in occupied_block

    # Task Surface + Research Inspector: the inspector is treated as another
    # occupied region only while it is visibly open.
    assert 'researchWorkspace.classList.contains("visible")' in occupied_block
    assert "rects.push(rect)" in occupied_block

    # Viewport resize / open-close stability: no previous safe rect is cached;
    # the safe area starts from current stageSize and current occupied rects.
    assert "const size = stageSize(stage);" in safe_block
    assert "for (const rect of occupiedWorkspaceRects(stage))" in safe_block
    assert "state.items" not in safe_block
    assert "localStorage" not in safe_block
    assert "clampEchoPointToSafeWorkspace(stage, point)" in point_block


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


def test_research_cards_are_progressive_and_safe() -> None:
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")
    css = (WEB / "styles.css").read_text(encoding="utf-8")

    render_block = ui[ui.index("function renderCards(items)") : ui.index("return {", ui.index("function renderCards(items)"))]
    assert "document.createElement(\"article\")" in render_block
    assert "textContent" in render_block
    assert "innerHTML" not in render_block
    assert "research-card-detail" in render_block
    assert "item.relevance_role || item.source_quality" in render_block
    assert "card.dataset.expanded" in render_block
    assert ".research-card-detail" in css


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
    assert 'id="telemetrySemanticExtraction"' in html
    assert "semantic_extraction_status" in ui
    assert "semantic_candidates_before_validation" in ui
    assert "semantic_candidates_after_validation" in ui
    assert "semantic_extraction_failure_reason" in ui
    assert "semantic_extraction_batches" in ui
    assert "semantic_extraction_latency_ms" in ui
    assert "semantic_extraction_input_chars" in ui
    assert "semantic_extraction_sources_used" in ui
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
    assert "telemetrySemanticExtraction" in status_block
    assert "telemetryFallback" in status_block
    assert "data-telemetry-mode" not in status_block
    assert "dailyBudgetUsd" not in status_block
    assert "resetWorkspace" not in status_block

    assert "data-telemetry-mode" in models_block
    assert "data-auto-claude" in models_block
    assert "automaticClaudeStatus" in models_block
    assert "anthropicApiKey" in models_block
    assert "WEB RESEARCH · TAVILY" in models_block
    assert "tavilyApiKey" in models_block
    assert "saveTavilyKey" in models_block
    assert "removeTavilyKey" in models_block
    assert "testTavilyKey" in models_block
    assert "resetWorkspace" not in models_block

    assert "recenterEcho" in workspace_block
    assert "resetWorkspace" in workspace_block
    assert "toggleWorkspaceLock" in workspace_block
    assert "data-telemetry-mode" not in workspace_block

    assert "systemVersion" in system_block
    assert "systemClaude" in system_block
    assert "systemTavily" in system_block
    assert "systemCliOverride" in system_block
    assert "systemModelModeSource" in system_block
    assert "systemAutomaticClaudeSource" in system_block
    assert "systemApiKeySource" in system_block

    assert "openPanelSection" in ui
    assert "ArrowLeft" in ui and "ArrowRight" in ui
    assert "anthropicKeyPanel" in ui
    assert "tavilyKeyPanel" in ui
    assert "save_tavily_key" in ui
    assert "remove_tavily_key" in ui
    assert "test_tavily_connection" in ui
    assert "tavily_key_configured" in ui
    assert "tavily_key_source" in ui
    assert "CONFIGURED BY ENVIRONMENT" in ui
    assert "CONFIGURED SECURELY" in ui
    assert "CONNECTION OK" in ui
    assert "window.confirm" not in ui
    assert "alert(" not in ui


def test_tavily_settings_ui_never_renders_a_secret_value() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")

    tavily_block = html[html.index('id="tavilyKeyPanel"'):html.index('data-panel-page="workspace"')]
    update_block = ui[ui.index("function updateModelConfigState"):ui.index("function renderConfigurationError")]
    listener_block = ui[ui.index("if (saveTavilyKey)"):ui.index("if (removeTavilyKey)")]

    assert 'type="password"' in tavily_block
    assert "tvly-" in tavily_block
    assert "value=" not in tavily_block
    assert "elements.tavilyKeyInput.value = \"\";" in update_block
    assert "elements.tavilyKeyInput.hidden = !storageAvailable || tavilyConfigured;" in update_block
    assert "elements.tavilyKeyStatus.textContent" in update_block
    assert "keyInput.value = \"\";" in listener_block
    assert "localStorage" not in listener_block
    assert "tvly-test-secret" not in html
    assert "tvly-test-secret" not in ui


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
    assert "response_text = str(response or \"\")" in controller
    assert "def _emit_response_ready" in controller
    assert "self.responseReady.emit(response_text)" in controller
    assert "delay_ms = max(0, self.thinking_min_ms - elapsed_ms)" in controller
    assert "lambda: self._emit_state(\"idle\")" in controller
    assert "def _handle_tts_started" in controller
    assert "def _handle_tts_finished" in controller


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
        self.capacity_reports: list[dict] = []
        # Phase 7.6.5 (section 16 D/E/F): lets tests assert a sleep command
        # never reached the LLM -- respond() is the ONLY thing that stands
        # in for an actual model/task call in this fake.
        self.calls: list[str] = []

    def set_progress_listener(self, listener) -> None:
        self._progress_listener = listener

    def cancel_current_request(self) -> bool:
        self.cancelled = True
        return True

    def respond(self, message: str) -> str:
        self.calls.append(message)
        for event in self.progress_events:
            if self._progress_listener is not None:
                self._progress_listener(event)
        return self.reply

    def report_workspace_capacity(self, payload: dict) -> None:
        self.capacity_reports.append(payload)


class _SlowFakeEngine(_FakeEngine):
    """Phase 7.4: adds a real, short sleep before responding -- gives tests
    a reliable window to observe/act on the controller while it is
    genuinely "thinking" (a plain _FakeEngine can resolve faster than the
    test's own polling notices it started)."""

    def __init__(self, *, delay_seconds: float = 0.25, **kwargs) -> None:
        super().__init__(**kwargs)
        self.delay_seconds = delay_seconds

    def respond(self, message: str) -> str:
        import time

        time.sleep(self.delay_seconds)
        return super().respond(message)


class _FailingTelemetryRuntime:
    def request_started_payload(self) -> dict:
        return {"state": "request_started"}

    def telemetry_payload(self, telemetry: dict, *, state: str) -> dict:
        raise RuntimeError("telemetry serialization failed")

    def error_payload(self, message: str, *, provider_error_type: str) -> dict:
        raise RuntimeError("telemetry error serialization failed")

    def current_payload(self, *, state: str) -> dict:
        return {"state": state}


class _FakeVoiceTranscriber:
    def __init__(self, result: str = "vou a Estocolmo de 10 a 15 de outubro", error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.thread_checks: list[bool] = []

    def record_and_transcribe(self, stop_event, status_callback=None) -> str:
        from PySide6.QtCore import QCoreApplication, QThread

        app = QCoreApplication.instance()
        self.thread_checks.append(QThread.currentThread() != app.thread())
        if status_callback is not None:
            status_callback("A ouvir...")
            status_callback("A processar...")
        stop_event.set()
        if self.error is not None:
            raise self.error
        return self.result


class _NoSpeechVoiceTranscriber(_FakeVoiceTranscriber):
    def record_and_transcribe(self, stop_event, status_callback=None) -> str:
        from assistant.voice_input import VoiceNoSpeechDetected

        if status_callback is not None:
            status_callback("A ouvir...")
        stop_event.set()
        raise VoiceNoSpeechDetected("Nao detetei fala.")


class _SequentialVoiceTranscriber:
    """Phase 7.4: a different scripted result/error on each successive
    call -- used to test multi-cycle conversation-loop behavior (listening
    restarts, TTS-manager reuse across turns) without needing a real
    microphone. The LAST step repeats if record_and_transcribe is called
    more times than there are steps."""

    def __init__(self, steps: list[tuple[str | None, Exception | None]]) -> None:
        self.steps = list(steps)
        self.call_count = 0

    def record_and_transcribe(self, stop_event, status_callback=None) -> str:
        index = min(self.call_count, len(self.steps) - 1)
        self.call_count += 1
        result, error = self.steps[index]
        if status_callback is not None:
            status_callback("A ouvir...")
        stop_event.set()
        if error is not None:
            raise error
        return result or ""


class _BlockingVoiceTranscriber:
    """Phase 7.4: blocks until stop_event is externally set (simulating a
    still-open listening cycle), then raises VoiceNoSpeechDetected -- for
    tests that need to observe/act on the controller WHILE a listening
    cycle is genuinely in flight (stop-during-listening, typed-turn-during-
    listening)."""

    def record_and_transcribe(self, stop_event, status_callback=None) -> str:
        import time

        from assistant.voice_input import VoiceNoSpeechDetected

        if status_callback is not None:
            status_callback("A ouvir...")
        while not stop_event.is_set():
            time.sleep(0.01)
        raise VoiceNoSpeechDetected("cancelled")


def _run_until(predicate, timeout_ms: int = 3000) -> bool:
    """Phase 7.4: pumps the Qt event loop until `predicate()` is true or the
    timeout elapses -- used instead of nested QEventLoop/QTimer pairs for
    the multi-hop voice-conversation-loop tests (listen -> transcript ->
    submit -> respond -> speak -> finished -> rearm -> listen again)."""
    import time

    from PySide6.QtCore import QCoreApplication

    app = QCoreApplication.instance()
    deadline = time.perf_counter() + (timeout_ms / 1000)
    while time.perf_counter() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return False


def _stop_and_drain(controller, timeout_ms: int = 3000) -> None:
    """Phase 7.4: MUST be called at the end of every voice-conversation-loop
    test. Without it, a still-active listening/rearm/speaking cycle leaves
    a live QThread behind that nothing will ever pump again once the test
    function returns -- pytest's own process/session teardown then hangs
    waiting for that thread to finish, which it never will."""
    controller.stopVoiceConversation()
    _run_until(
        lambda: controller._voice_worker is None and controller._tts_worker is None and controller._worker is None,
        timeout_ms=timeout_ms,
    )


def _qt_app():
    from PySide6.QtCore import QCoreApplication

    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    return app


def test_voice_status_uses_current_response_language_without_changing_state() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    language = ["en-GB"]
    controller = EchoUIController(lambda _: "", get_response_language=lambda: language[0])
    events = []
    states = []
    controller.uiEvent.connect(lambda payload: events.append(json.loads(payload)))
    controller.stateChanged.connect(states.append)
    for status in ("Preparar...", "A ouvir...", "A processar...", "Concluído."):
        controller._handle_voice_status(status)
    assert [event["status"] for event in events] == ["Getting ready...", "Listening...", "Processing...", "Done."]
    assert states == ["listening", "thinking", "thinking"]
    language[0] = "pt-PT"
    controller._handle_voice_status("A ouvir...")
    assert events[-1]["status"] == "A ouvir..."


def test_cancel_current_request_invokes_fake_engine_cancellation() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine()
    controller = EchoUIController(engine.respond)
    controller._active = True

    controller.cancelCurrentRequest()

    assert engine.cancelled is True


def test_report_workspace_capacity_bridge_calls_engine_with_semantic_payload() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine()
    controller = EchoUIController(engine.respond)

    controller.reportWorkspaceCapacity(
        '{"pressure":"high","visible_object_count":5,"major_surface_count":2,"overflow_risk":true}'
    )

    assert engine.capacity_reports == [
        {
            "pressure": "high",
            "visible_object_count": 5,
            "major_surface_count": 2,
            "overflow_risk": True,
        }
    ]


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
    controller = EchoUIController(engine.respond, thinking_min_ms=0)

    states: list[str] = []
    ui_events: list[str] = []
    finished = {"count": 0}
    controller.stateChanged.connect(states.append)
    controller.uiEvent.connect(ui_events.append)
    controller.requestFinished.connect(lambda: finished.__setitem__("count", finished["count"] + 1))

    # requestFinished fires as soon as the worker is cleaned up; idle is a
    # separate GUI-thread timer preserving the minimum thinking duration.
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


def test_echo_os_controller_exposes_voice_api_and_transcribes_off_ui_thread() -> None:
    from PySide6.QtCore import QEventLoop, QTimer

    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine()
    transcriber = _FakeVoiceTranscriber()
    controller = EchoUIController(engine.respond, voice_available=True, voice_transcriber=transcriber)

    states: list[str] = []
    transcripts: list[str] = []
    finished = {"count": 0}
    controller.stateChanged.connect(states.append)
    controller.voiceTranscriptionReady.connect(transcripts.append)
    controller.voiceInputFinished.connect(lambda: finished.__setitem__("count", finished["count"] + 1))

    loop = QEventLoop()
    controller.voiceInputFinished.connect(loop.quit)
    controller.startVoiceInput()
    QTimer.singleShot(5000, loop.quit)
    loop.exec()

    assert transcripts == ["vou a Estocolmo de 10 a 15 de outubro"]
    assert finished["count"] == 1
    assert "listening" in states
    assert states[-1] == "idle"
    assert transcriber.thread_checks == [True]
    assert controller._voice_thread is None
    assert controller._voice_worker is None
    assert engine._progress_listener is None


def test_echo_os_voice_disabled_fails_safely_to_idle() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    controller = EchoUIController(_FakeEngine().respond, voice_available=False, voice_transcriber=None)
    errors: list[str] = []
    states: list[str] = []
    controller.voiceInputError.connect(errors.append)
    controller.stateChanged.connect(states.append)

    controller.startVoiceInput()

    assert errors == ["A voz não está disponível nesta sessão."]
    assert states[-1] == "idle"
    assert controller.isVoiceInputAvailable() is False


def test_echo_os_voice_runtime_missing_fails_safely_to_idle() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    controller = EchoUIController(
        _FakeEngine().respond,
        voice_available=True,
        voice_transcriber=_FakeVoiceTranscriber(),
        voice_missing_dependencies=["ffmpeg"],
    )
    errors: list[str] = []
    states: list[str] = []
    controller.voiceInputError.connect(errors.append)
    controller.stateChanged.connect(states.append)

    controller.startVoiceInput()

    assert errors == ["A voz não está pronta porque falta: ffmpeg"]
    assert states[-1] == "idle"
    assert controller.isVoiceInputAvailable() is False


def test_echo_os_voice_failure_returns_error_then_idle() -> None:
    from PySide6.QtCore import QEventLoop, QTimer

    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    controller = EchoUIController(
        _FakeEngine().respond,
        voice_available=True,
        voice_transcriber=_FakeVoiceTranscriber(error=RuntimeError("microfone indisponível")),
    )
    errors: list[str] = []
    states: list[str] = []
    controller.voiceInputError.connect(errors.append)
    controller.stateChanged.connect(states.append)

    loop = QEventLoop()
    controller.voiceInputFinished.connect(loop.quit)
    controller.startVoiceInput()
    QTimer.singleShot(5000, loop.quit)
    loop.exec()

    assert errors == ["Não consegui usar a voz: microfone indisponível"]
    assert "error" in states
    assert controller._voice_thread is None
    assert controller._voice_worker is None


def test_echo_os_voice_no_speech_returns_idle_without_error() -> None:
    from PySide6.QtCore import QEventLoop, QTimer

    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    controller = EchoUIController(
        _FakeEngine().respond,
        voice_available=True,
        voice_transcriber=_NoSpeechVoiceTranscriber(),
    )
    errors: list[str] = []
    states: list[str] = []
    transcripts: list[str] = []
    controller.voiceInputError.connect(errors.append)
    controller.stateChanged.connect(states.append)
    controller.voiceTranscriptionReady.connect(transcripts.append)

    loop = QEventLoop()
    controller.voiceInputFinished.connect(loop.quit)
    controller.startVoiceInput()
    QTimer.singleShot(5000, loop.quit)
    loop.exec()

    assert errors == []
    assert transcripts == [""]
    assert "listening" in states
    assert states[-1] == "idle"
    assert controller._voice_thread is None
    assert controller._voice_worker is None


def test_echo_os_voice_frontend_inserts_text_without_auto_send() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")
    controller = (PROTOTYPE / "controller.py").read_text(encoding="utf-8")

    assert 'id="micButton"' in html
    assert "startVoiceInput" in controller
    assert "cancelVoiceInput" in controller
    assert "voiceTranscriptionReady = Signal(str)" in controller
    assert "voiceInputError = Signal(str)" in controller
    assert "function insertVoiceTranscription(text)" in ui
    insert_block = ui[ui.index("function insertVoiceTranscription"):ui.index("function startOrStopVoiceInput")]
    assert "if (!transcription) return;" in insert_block
    assert "input.value = current ? `${current} ${transcription}` : transcription;" in insert_block
    assert "sendMessage" not in insert_block
    assert "controller.submitMessage" not in insert_block
    assert 'event.type === "voice_status"' in ui
    assert "updateStatusLabel(document.body.dataset.echoState || \"idle\", currentProgressLabel)" in ui


def test_echo_os_voice_frontend_never_uses_speaking_for_stt() -> None:
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")
    voice_block = ui[ui.index("function startOrStopVoiceInput"):ui.index("function updateStatusLabel")]

    assert 'applyEchoState("listening")' in voice_block
    assert "controller.startVoiceInput()" in voice_block
    assert "controller.cancelVoiceInput()" in voice_block
    assert "speaking" not in voice_block


def test_submit_message_reaches_idle_even_if_response_telemetry_fails() -> None:
    from PySide6.QtCore import QEventLoop, QTimer

    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Resposta pronta.")
    controller = EchoUIController(
        engine.respond,
        get_telemetry=lambda: {"selected_path": "RESEARCH_REQUEST", "llm_calls": 0},
        model_runtime=_FailingTelemetryRuntime(),
        thinking_min_ms=0,
    )

    states: list[str] = []
    responses: list[str] = []
    finished = {"count": 0}
    controller.stateChanged.connect(states.append)
    controller.responseReady.connect(responses.append)
    controller.requestFinished.connect(lambda: finished.__setitem__("count", finished["count"] + 1))

    loop = QEventLoop()
    controller.stateChanged.connect(lambda state: loop.quit() if state == "idle" else None)
    controller.submitMessage("olá")
    QTimer.singleShot(5000, loop.quit)
    loop.exec()

    assert responses == ["Resposta pronta."]
    assert finished["count"] == 1
    assert "idle" in states
    assert controller._thread is None
    assert controller._worker is None
    assert controller.has_active_request() is False


class _FakeTTSManager:
    def __init__(
        self,
        error: Exception | None = None,
        available: bool = True,
        pt_available: bool = True,
        delay_seconds: float = 0.0,
        provider_voice: str = "",
    ) -> None:
        from pathlib import Path

        from assistant.tts import TTSResult

        self.error = error
        self.available = available
        self.pt_available = pt_available
        self.delay_seconds = delay_seconds
        self.stopped = False
        self.calls: list[str] = []
        self.response_languages: list[str] = []
        self.result = TTSResult(
            audio_path=Path("fake.wav"),
            provider="fake",
            voice_id="english_british_primary",
            language="en",
            synthesis_ms=4.0,
            text_chars=0,
            provider_voice=provider_voice,
        )

    @property
    def provider(self):
        available = self.available

        class Provider:
            def is_available(self) -> bool:
                return available

        return Provider()

    def is_available(self, response_language: str = "") -> bool:
        if response_language == "pt-PT" and not self.pt_available:
            return False
        return self.available

    def speak_to_file(self, text: str, *, response_language: str = ""):
        if self.delay_seconds:
            import time

            time.sleep(self.delay_seconds)
        if self.error is not None:
            raise self.error
        self.calls.append(text)
        self.response_languages.append(response_language)
        return self.result

    def stop(self) -> None:
        self.stopped = True


def test_echo_os_tts_speaks_current_response_manually_off_ui_thread() -> None:
    import json

    from PySide6.QtCore import QCoreApplication, QEventLoop, QThread, QTimer

    from prototype_web_ui.controller import EchoUIController

    app = _qt_app()
    tts_manager = _FakeTTSManager()
    player_checks: list[bool] = []

    def fake_player(_path, _stop_checker=None) -> float:
        player_checks.append(QThread.currentThread() != app.thread())
        return 7.0

    controller = EchoUIController(
        _FakeEngine().respond,
        tts_manager=tts_manager,
        tts_player=fake_player,
        get_response_language=lambda: "en-GB",
    )
    states: list[str] = []
    started: list[str] = []
    finished: list[str] = []
    controller.stateChanged.connect(states.append)
    controller.ttsStarted.connect(started.append)
    controller.ttsFinished.connect(finished.append)

    loop = QEventLoop()
    controller.ttsFinished.connect(loop.quit)
    controller.speakText("Olá, Alexandre.")
    QTimer.singleShot(5000, loop.quit)
    loop.exec()

    assert tts_manager.calls == ["Olá, Alexandre."]
    assert tts_manager.response_languages == ["en-GB"]
    assert player_checks == [True]
    assert "speaking" in states
    assert states[-1] == "idle"
    assert started and "english_british_primary" in started[0]
    started_payload = json.loads(started[0])
    assert started_payload["response_language"] == "en-GB"
    assert started_payload["voice_language"] == "en"
    assert finished and "playback_ms" in finished[0]
    assert controller._tts_thread is None
    assert controller._tts_worker is None


def test_speak_text_requires_interruptible_playback_during_voice_conversation_mode() -> None:
    """Phase 7.6.4 (section 7): Voice Conversation Mode must ask
    play_audio_file() for interruptible-only playback (allow_winsound_
    fallback=False) -- a manual (non-voice-mode) SPEAK must not."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Spoken reply.")
    transcriber = _FakeVoiceTranscriber(result="ola echo")
    tts = _FakeTTSManager()
    player_calls: list[bool] = []

    def recording_player(_path, _stop_checker=None, *, on_playback_event=None, allow_winsound_fallback=True) -> float:
        player_calls.append(allow_winsound_fallback)
        return 0.01

    controller = EchoUIController(
        engine.respond,
        voice_available=True,
        voice_transcriber=transcriber,
        tts_manager=tts,
        tts_player=recording_player,
        thinking_min_ms=0,
    )

    controller.startVoiceConversation()

    assert _run_until(lambda: player_calls, timeout_ms=3000)
    assert player_calls == [False]
    _stop_and_drain(controller)


def test_speak_text_allows_winsound_fallback_outside_voice_conversation_mode() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Typed reply.")
    tts = _FakeTTSManager()
    player_calls: list[bool] = []

    def recording_player(_path, _stop_checker=None, *, on_playback_event=None, allow_winsound_fallback=True) -> float:
        player_calls.append(allow_winsound_fallback)
        return 0.01

    controller = EchoUIController(engine.respond, tts_manager=tts, tts_player=recording_player, thinking_min_ms=0)

    controller.speakText("Typed reply.")

    assert _run_until(lambda: player_calls, timeout_ms=3000)
    assert player_calls == [True]


def test_voice_mode_playback_backend_failure_keeps_text_visible_and_recovers_loop() -> None:
    """Phase 7.6.4 (section 7): if the interruptible sounddevice backend
    genuinely cannot be opened while voice mode is active, play_audio_file
    raises TTSError instead of silently degrading to winsound. This must
    surface as a bounded ttsError while the ALREADY-SHOWN text response
    stays visible, and the conversation loop must recover (listen again)
    rather than getting stuck."""
    from assistant.tts import TTSError
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Spoken reply.")
    transcriber = _SequentialVoiceTranscriber([("ola echo", None), ("outra vez", None)])
    tts = _FakeTTSManager()

    def failing_when_interruptible_required(
        _path, _stop_checker=None, *, on_playback_event=None, allow_winsound_fallback=True
    ) -> float:
        if not allow_winsound_fallback:
            raise TTSError("Reprodução interrompível indisponível (RuntimeError: no output device)")
        return 0.01

    controller = EchoUIController(
        engine.respond,
        voice_available=True,
        voice_transcriber=transcriber,
        tts_manager=tts,
        tts_player=failing_when_interruptible_required,
        thinking_min_ms=0,
    )
    errors: list[str] = []
    controller.ttsError.connect(errors.append)
    responses: list[str] = []
    controller.responseReady.connect(responses.append)

    controller.startVoiceConversation()

    assert _run_until(lambda: errors, timeout_ms=3000)
    assert "indisponível" in errors[0]
    assert responses == ["Spoken reply."]  # text response never retracted
    assert _run_until(lambda: transcriber.call_count >= 2, timeout_ms=5000)  # loop recovered, listened again
    _stop_and_drain(controller)


def test_echo_os_tts_telemetry_reports_pocket_anna_correctly() -> None:
    """Phase 7.3.3 (section 6): a normal Anna synthesis' ttsStarted payload
    must surface provider=pocket_tts, voice_id=english_british_primary,
    provider_voice=anna, response_language=en-GB, fallback_used=false."""
    import json

    from PySide6.QtCore import QEventLoop, QTimer

    from assistant.tts import TTSResult
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    tts_manager = _FakeTTSManager(provider_voice="anna")
    tts_manager.result = TTSResult(
        audio_path=tts_manager.result.audio_path,
        provider="pocket_tts",
        voice_id="english_british_primary",
        language="en",
        synthesis_ms=4.0,
        text_chars=0,
        provider_voice="anna",
        fallback_used=False,
    )
    controller = EchoUIController(
        _FakeEngine().respond,
        tts_manager=tts_manager,
        tts_player=lambda *_a, **_k: 1.0,
        get_response_language=lambda: "en-GB",
    )
    started: list[str] = []
    controller.ttsStarted.connect(started.append)

    loop = QEventLoop()
    controller.ttsFinished.connect(loop.quit)
    controller.speakText("Good morning.")
    QTimer.singleShot(5000, loop.quit)
    loop.exec()

    assert started
    payload = json.loads(started[0])
    assert payload["provider"] == "pocket_tts"
    assert payload["voice_id"] == "english_british_primary"
    assert payload["provider_voice"] == "anna"
    assert payload["response_language"] == "en-GB"
    assert payload["fallback_used"] is False


def test_echo_os_tts_duplicate_speak_click_is_ignored_while_synthesizing() -> None:
    from PySide6.QtCore import QEventLoop, QTimer

    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    manager = _FakeTTSManager(delay_seconds=0.15)
    controller = EchoUIController(_FakeEngine().respond, tts_manager=manager, get_response_language=lambda: "en-GB")
    states: list[str] = []
    controller.stateChanged.connect(states.append)

    working_loop = QEventLoop()
    controller.stateChanged.connect(lambda state: working_loop.quit() if state == "working" else None)
    controller.speakText("Hi. How are you?")
    QTimer.singleShot(1000, working_loop.quit)
    working_loop.exec()

    controller.speakText("Hi. How are you?")

    finished_loop = QEventLoop()
    controller.ttsFinished.connect(finished_loop.quit)
    QTimer.singleShot(5000, finished_loop.quit)
    finished_loop.exec()

    assert manager.calls == ["Hi. How are you?"]
    assert states[0] == "working"
    assert "speaking" in states
    assert states[-1] == "idle"


def test_echo_os_tts_failure_keeps_textual_response_canonical() -> None:
    from PySide6.QtCore import QEventLoop, QTimer

    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    controller = EchoUIController(_FakeEngine().respond, tts_manager=_FakeTTSManager(error=RuntimeError("sem voz")))
    errors: list[str] = []
    states: list[str] = []
    controller.ttsError.connect(errors.append)
    controller.stateChanged.connect(states.append)

    loop = QEventLoop()
    controller.ttsError.connect(loop.quit)
    controller.speakText("Resposta já visível.")
    QTimer.singleShot(5000, loop.quit)
    loop.exec()

    assert errors == ["Não consegui falar a resposta: sem voz"]
    assert "error" in states
    assert controller._tts_thread is None
    assert controller._tts_worker is None


def test_echo_os_tts_unavailable_stays_idle_and_does_not_start_worker() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    controller = EchoUIController(_FakeEngine().respond, tts_manager=_FakeTTSManager(available=False))
    errors: list[str] = []
    states: list[str] = []
    controller.ttsError.connect(errors.append)
    controller.stateChanged.connect(states.append)

    assert controller.isTtsAvailable() is False

    controller.speakText("Resposta já visível.")

    assert errors == ["Não encontrei uma voz TTS local compatível."]
    assert states[-1] == "idle"
    assert controller._tts_thread is None
    assert controller._tts_worker is None


def test_echo_os_tts_unavailable_for_ptpt_stays_idle_and_preserves_text() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    manager = _FakeTTSManager(available=True, pt_available=False)
    controller = EchoUIController(
        _FakeEngine().respond,
        tts_manager=manager,
        get_response_language=lambda: "pt-PT",
    )
    errors: list[str] = []
    states: list[str] = []
    controller.ttsError.connect(errors.append)
    controller.stateChanged.connect(states.append)

    assert controller.isTtsAvailable() is False

    controller.speakText("Resposta em português.")

    assert errors == ["A voz portuguesa ainda não está disponível."]
    assert states[-1] == "idle"
    assert manager.calls == []
    assert controller._tts_thread is None
    assert controller._tts_worker is None


def test_echo_os_tts_switching_language_changes_speak_availability() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    current_language = {"value": "en-GB"}
    controller = EchoUIController(
        _FakeEngine().respond,
        tts_manager=_FakeTTSManager(available=True, pt_available=False),
        get_response_language=lambda: current_language["value"],
    )

    assert controller.isTtsAvailable() is True

    current_language["value"] = "pt-PT"
    assert controller.isTtsAvailable() is False

    current_language["value"] = "en-GB"
    assert controller.isTtsAvailable() is True


def test_echo_os_tts_playback_failure_returns_to_idle() -> None:
    from PySide6.QtCore import QEventLoop, QTimer

    from prototype_web_ui.controller import EchoUIController

    _qt_app()

    def failing_player(_path, _stop_checker=None) -> float:
        raise RuntimeError("coluna indisponível")

    controller = EchoUIController(_FakeEngine().respond, tts_manager=_FakeTTSManager(), tts_player=failing_player)
    errors: list[str] = []
    states: list[str] = []
    controller.ttsError.connect(errors.append)
    controller.stateChanged.connect(states.append)

    loop = QEventLoop()
    controller.stateChanged.connect(lambda state: loop.quit() if state == "idle" else None)
    controller.speakText("Resposta já visível.")
    QTimer.singleShot(5000, loop.quit)
    loop.exec()

    assert errors == ["Não consegui falar a resposta: coluna indisponível"]
    assert "error" in states
    assert states[-1] == "idle"
    assert controller._tts_thread is None
    assert controller._tts_worker is None


def test_echo_os_tts_stop_speaking_returns_idle_and_preserves_controller() -> None:
    """Phase 7.3.4 (section 5): stopSpeaking() must stop THIS playback
    (the player's own stop_checker already handles that) without tearing
    down the shared, persistent TTS manager/worker -- manager.stopped used
    to assert True here, which was exactly the bug: an ordinary mid-session
    Stop/next-message action was silently forcing a full Pocket server
    restart on the next SPEAK. teardown_manager is reserved for real
    shutdown (see test_echo_os_controller_shutdown_stops_active_tts_worker)."""
    import time

    from PySide6.QtCore import QEventLoop, QTimer

    from prototype_web_ui.controller import EchoUIController

    _qt_app()

    def slow_player(_path, stop_checker=None) -> float:
        started = time.perf_counter()
        while time.perf_counter() - started < 2.0:
            if stop_checker and stop_checker():
                return 0.0
            time.sleep(0.01)
        return 2000.0

    manager = _FakeTTSManager()
    controller = EchoUIController(_FakeEngine().respond, tts_manager=manager, tts_player=slow_player)
    states: list[str] = []
    controller.stateChanged.connect(states.append)

    speaking_loop = QEventLoop()
    controller.stateChanged.connect(lambda state: speaking_loop.quit() if state == "speaking" else None)
    controller.speakText("Resposta longa.")
    QTimer.singleShot(5000, speaking_loop.quit)
    speaking_loop.exec()

    finished_loop = QEventLoop()
    controller.ttsFinished.connect(finished_loop.quit)
    controller.stopSpeaking()
    QTimer.singleShot(5000, finished_loop.quit)
    finished_loop.exec()

    assert manager.stopped is False
    assert "speaking" in states
    assert states[-1] == "idle"
    assert controller._tts_thread is None
    assert controller._tts_worker is None


def test_echo_os_controller_shutdown_stops_active_tts_worker() -> None:
    import time

    from PySide6.QtCore import QEventLoop, QTimer

    from prototype_web_ui.controller import EchoUIController

    _qt_app()

    def slow_player(_path, stop_checker=None) -> float:
        while not (stop_checker and stop_checker()):
            time.sleep(0.01)
        return 0.0

    manager = _FakeTTSManager()
    controller = EchoUIController(_FakeEngine().respond, tts_manager=manager, tts_player=slow_player)

    loop = QEventLoop()
    controller.stateChanged.connect(lambda state: loop.quit() if state == "speaking" else None)
    controller.speakText("Resposta longa.")
    QTimer.singleShot(5000, loop.quit)
    loop.exec()

    controller.shutdown()

    assert manager.stopped is True
    assert controller._tts_thread is None
    assert controller._tts_worker is None


def test_echo_os_tts_frontend_has_manual_speak_controls_without_auto_speak() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")
    controller = (PROTOTYPE / "controller.py").read_text(encoding="utf-8")

    assert 'id="speakButton"' in html
    assert 'id="stopSpeechButton"' in html
    assert "speakText" in controller
    assert "stopSpeaking" in controller
    assert "isTtsAvailable" in controller
    assert "function speakCurrentResponse()" in ui
    assert "function refreshTtsAvailability()" in ui
    assert "function performTtsAvailabilityRefresh()" in ui
    assert "ttsAvailabilityRefreshTimer" in ui
    render_block = ui[ui.index("function renderEchoResponse"):ui.index("function applyResponseLayout")]
    assert "currentResponseText = value" in render_block
    assert "refreshTtsAvailability();" in render_block
    speak_block = ui[ui.index("function speakCurrentResponse"):ui.index("function stopSpeaking")]
    assert "ttsActive = true" in speak_block
    assert 'applyEchoState("working"' in speak_block
    assert "controller.speakText" not in render_block
    assert "activeController.ttsStarted" in ui
    assert "activeController.ttsFinished" in ui
    assert "activeController.ttsError" in ui
    telemetry_block = ui[ui.index("activeController.telemetryUpdated.connect"):ui.index("activeController.requestStarted.connect")]
    assert "refreshTtsAvailability();" in telemetry_block
    channel_block = ui[ui.index("function initializeEchoChannel"):ui.index("function bindDomEvents")]
    assert "controller.isTtsAvailable((available)" not in channel_block
    assert "refreshTtsAvailability();" in channel_block
    tts_error_block = ui[ui.index("activeController.ttsError.connect"):ui.index("activeController.uiEvent.connect")]
    assert "renderEchoError" not in tts_error_block
    assert 'applyEchoState("error"' in tts_error_block


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
    assert "const taskState = taskSurfaceState();" in presentation_block
    assert "Boolean(activeSurfaceId())" in presentation_block
    assert "taskState.openTasks.length > 0" in presentation_block
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
    assert "const safePoint = stage ? clampEchoPointToSafeWorkspace(stage, point) : point;" in ui
    assert "window.echoEntity.setCenter(safePoint.x, safePoint.y, false);" in ui


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

def _run_workspace_harness_with_dom(js_tail: str, host_rect: dict, pre_controller_js: str = "") -> dict:
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
        _listeners: {},
        appendChild(child) {
          if (child.parentNode && child.parentNode.children) {
            child.parentNode.children = child.parentNode.children.filter((c) => c !== child);
          }
          el.children = el.children.filter((c) => c !== child);
          child.parentNode = el;
          el.children.push(child);
          return child;
        },
        removeChild(child) { el.children = el.children.filter((c) => c !== child); child.parentNode = null; },
        remove() { if (el.parentNode) el.parentNode.removeChild(el); },
        addEventListener(type, fn) { el._listeners[type] = el._listeners[type] || []; el._listeners[type].push(fn); },
        setAttribute(name, value) { el[name] = value; },
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
    const documentListeners = {};
    global.document = {
      createElement: (tag) => makeElement(tag),
      createElementNS: (ns, tag) => makeElement(tag),
      _listeners: documentListeners,
      addEventListener: (type, fn) => {
        documentListeners[type] = documentListeners[type] || [];
        documentListeners[type].push(fn);
      },
      getElementById: () => null,
    };
    global.window = global;
    global.localStorage = {
      _store: {},
      getItem(key) { return Object.prototype.hasOwnProperty.call(this._store, key) ? this._store[key] : null; },
      setItem(key, value) { this._store[key] = String(value); },
      removeItem(key) { delete this._store[key]; },
      clear() { this._store = {}; },
    };
    """

    script = f"""
    {dom_shim}
    require({json.dumps(str(WORKSPACE_JS_PATH))});

    const hostRect = {json.dumps(host_rect)};
    const containerEl = {{
      children: [],
      style: makeStyle(),
      appendChild(child) {{
        if (child.parentNode && child.parentNode.children) {{
          child.parentNode.children = child.parentNode.children.filter((c) => c !== child);
        }}
        containerEl.children = containerEl.children.filter((c) => c !== child);
        child.parentNode = containerEl;
        containerEl.children.push(child);
        return child;
      }},
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
    {pre_controller_js}
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
    assert "payload: event.payload" in case_block


def test_writing_semantic_payload_updates_same_card_without_losing_content() -> None:
    result = _run_workspace_harness(
        """
        const first = controller.upsertSemanticObject({id: "email-one", kind: "writing_task", title: "Email", taskId: "task-one",
          payload: {recipient: "Francisca and Carina", purpose: "Meeting this week", status: "collecting"}});
        const collecting = first.content;
        controller.upsertSemanticObject({id: "email-one", kind: "writing_task", title: "Email", taskId: "task-one",
          payload: {recipient: "Francisca and Carina", draft_content: "Subject: Meeting\\n\\nHello Carina,", saved_file: "workspace/meeting.txt"}});
        console.log(JSON.stringify({same: first === controller.getObject("email-one"), count: controller.listObjects().length,
          collecting, draft: first.content, payload: first.metadata.semanticPayload, task: first.metadata.taskId}));
        """
    )
    assert result["same"] and result["count"] == 1
    assert "Meeting this week" in result["collecting"]
    assert result["draft"] == "Subject: Meeting\n\nHello Carina,"
    assert result["payload"]["saved_file"] == "workspace/meeting.txt"
    assert result["task"] == "task-one"


def test_writing_card_renders_payload_as_text_not_html() -> None:
    source = WORKSPACE_JS_PATH.read_text(encoding="utf-8")
    body = source[source.index("function fillBody("):source.index("function buildCognitiveTaskProjection(")]
    assert 'obj.type === "writing_task"' in body
    assert "semanticPayload" in body and "textContent" in body


def test_writing_card_relabels_existing_body_on_language_switch() -> None:
    result = _run_workspace_harness_with_dom(
        """
        controller.upsertSemanticObject({id: "email-lang", kind: "writing_task", title: "Email", payload: {recipient: ""}});
        controller.render();
        function cards(root) {
          return (root.children || []).flatMap((child) => [
            ...(String(child.className).split(" ").includes("workspace-object") ? [child] : []), ...cards(child)]);
        }
        const nodes = cards(containerEl);
        nodes.forEach((node) => { node.querySelectorAll = () => []; });
        containerEl.querySelectorAll = () => nodes;
        const card = nodes.find((node) => node.dataset.id === "email-lang");
        const body = card.querySelector(".workspace-object__body");
        const before = body.children.map((node) => node.textContent).join(" ");
        controller.setLanguage("en-GB");
        const after = body.children.map((node) => node.textContent).join(" ");
        console.log(JSON.stringify({before, after, count: controller.listObjects().length, id: card.dataset.id}));
        """,
        {"left": 0, "top": 0, "right": 1328, "bottom": 860, "width": 1328, "height": 860},
    )
    assert result["before"].startswith("A aguardar ")
    assert result["after"] == "Waiting for a recipient."
    assert result["count"] == 1 and result["id"] == "email-lang"


def test_document_labels_and_saved_receipt_follow_language_without_changing_draft() -> None:
    result = _run_workspace_harness(
        """
        controller.setLanguage("en-GB");
        controller.documentLoaded({fileName: "notes.txt", preview: "Original"});
        const working = controller.draftWorkingStarted("notes.txt");
        const workingLabel = working.subtitle;
        const draft = controller.draftReady({fileName: "notes.txt", revision: 2, content: "Canonical text"});
        controller.draftSaved({fileName: "notes.txt", outputName: "notes_revised.txt"});
        const english = {title: draft.title, subtitle: draft.subtitle};
        controller.setLanguage("pt-PT");
        console.log(JSON.stringify({workingLabel, english, portuguese: {title: draft.title, subtitle: draft.subtitle}, content: draft.content, saved: draft.metadata.savedAs}));
        """
    )
    assert result["workingLabel"] == "Rewriting..."
    assert "draft" in result["english"]["title"] and result["english"]["subtitle"] == "revision 2"
    assert "rascunho" in result["portuguese"]["title"]
    assert "revision" not in result["portuguese"]["subtitle"]
    assert result["content"] == "Canonical text" and result["saved"] == "notes_revised.txt"


def test_task_export_receipt_reaches_existing_card_without_duplicate_identity() -> None:
    ui_js = (WEB / "echo_ui.js").read_text(encoding="utf-8")
    case = ui_js[ui_js.index('case "task_artifact_updated"'):ui_js.index('case "semantic_object_upserted"')]
    assert "savedFile: event.saved_file" in case
    result = _run_workspace_harness(
        """
        const first = controller.upsertTaskArtifact({id: "trip", taskId: "task-trip", title: "Trip", content: "Current plan"});
        const second = controller.upsertTaskArtifact({id: "trip", taskId: "task-trip", title: "Trip", content: "Current plan", savedFile: "workspace/trip.txt"});
        console.log(JSON.stringify({same: first === second, count: controller.listObjects().length, saved: second.metadata.savedFile}));
        """
    )
    assert result == {"same": True, "count": 1, "saved": "workspace/trip.txt"}


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


def test_phase4_workspace_open_actions_are_layout_idempotent() -> None:
    """Repeatedly opening the same semantic objects must not accumulate
    inline geometry from previous renders. The same final semantic state
    should leave the same layout-owned properties every time."""
    result = _run_workspace_harness_with_dom(
        """
        const samsung = controller.createObject({id: "option-samsung", type: "option", title: "Samsung"});
        const apple = controller.createObject({id: "option-apple", type: "option", title: "Apple"});
        const pixel = controller.createObject({id: "option-pixel", type: "option", title: "Pixel"});
        controller.applyRepresentation({
          style: "side_by_side",
          visible_object_ids: [samsung.id, apple.id, pixel.id],
          major_object_ids: [samsung.id, apple.id],
          hidden_object_ids: [],
          source: "test",
        });
        controller.render();

        function nodeFor(id) {
          return containerEl.children.find((child) => child.dataset && child.dataset.id === id);
        }
        function snapshot() {
          const ids = [samsung.id, apple.id, pixel.id];
          return ids.map((id) => {
            const node = nodeFor(id);
            return {
              id,
              parentIsContainer: node ? node.parentNode === containerEl : false,
              role: node ? (node.dataset.role || "") : "",
              representationRole: node ? (node.dataset.representationRole || "") : "",
              split: node ? (node.dataset.split || "") : "",
              compactMajor: node ? (node.dataset.compactMajor || "") : "",
              attention: node ? (node.dataset.attention || "") : "",
              left: node ? (node.style.left || "") : "",
              top: node ? (node.style.top || "") : "",
              right: node ? (node.style.right || "") : "",
              width: node ? (node.style.width || "") : "",
              maxHeight: node ? (node.style["max-height"] || "") : "",
              transform: node ? (node.style.transform || "") : "",
            };
          });
        }
        function open(id) {
          controller.runAction(id, "open");
          controller.render();
        }

        open(pixel.id);
        open(samsung.id);
        const firstSamsung = JSON.stringify(snapshot());
        open(pixel.id);
        open(samsung.id);
        const secondSamsung = JSON.stringify(snapshot());
        open(pixel.id);
        const firstPixel = JSON.stringify(snapshot());
        open(samsung.id);
        open(pixel.id);
        const secondPixel = JSON.stringify(snapshot());

        const semanticIds = controller.listObjects().map((obj) => obj.id).sort();
        console.log(JSON.stringify({
          samsungStable: firstSamsung === secondSamsung,
          pixelStable: firstPixel === secondPixel,
          semanticIds,
          firstSamsung: JSON.parse(firstSamsung),
          secondSamsung: JSON.parse(secondSamsung),
          firstPixel: JSON.parse(firstPixel),
          secondPixel: JSON.parse(secondPixel),
        }));
        """,
        _HOST_RECT,
    )
    assert result["samsungStable"] is True
    assert result["pixelStable"] is True
    assert result["semanticIds"] == ["option-apple", "option-pixel", "option-samsung"]
    for snapshot_name in ("firstSamsung", "secondSamsung", "firstPixel", "secondPixel"):
        for node in result[snapshot_name]:
            assert "translate" not in node["transform"]
            assert node["id"] in result["semanticIds"]


def test_workspace_entity_tether_anchors_to_edges_in_common_coordinates() -> None:
    result = _run_workspace_harness_with_dom(
        """
        global.__mind = makeElement("div");
        global.__mind._rect = {left: 700, top: 220, right: 860, bottom: 380, width: 160, height: 160};
        document.getElementById = (id) => id === "mind" ? global.__mind : null;

        const edgeController = EchoWorkspaceObjects.create({
          container: containerEl,
          tetherLayer: tetherLayerEl,
          getEntityPoint: () => ({x: 780, y: 300}),
          setEntityLayoutRole: () => {},
          setEntityAnchor: () => {},
          setEntityPoint: () => {},
          onFocusChange: () => {},
          submit: () => {},
        });
        const surface = edgeController.createObject({id: "artifact-task", type: "task_artifact", title: "Task ativa", primary: true});
        edgeController.render();
        const node = containerEl.children.find((child) => child.dataset && child.dataset.id === surface.id);
        node._rect = {left: 120, top: 210, right: 420, bottom: 390, width: 300, height: 180};
        edgeController.renderTethers();

        const path = tetherLayerEl.children[0];
        const d = path ? path.d : "";
        const nums = d.match(/-?\\d+(?:\\.\\d+)?/g).map(Number);
        console.log(JSON.stringify({
          d,
          className: path ? path.class : "",
          finite: nums.every(Number.isFinite),
          startX: nums[0],
          startY: nums[1],
          endX: nums[6],
          endY: nums[7],
          command: d.includes(" C "),
        }));
        """,
        _HOST_RECT,
    )
    assert result["finite"] is True
    assert result["command"] is True
    assert "workspace-tether--entity" in result["className"]
    assert 700 <= result["startX"] < 780  # Echo boundary, not center.
    assert result["endX"] == 420  # right edge of the active surface.
    assert 210 < result["endY"] < 390
    assert "NaN" not in result["d"]
    assert "Infinity" not in result["d"]


def test_phase45_task_surface_renders_task_slots_without_child_slots() -> None:
    result = _run_workspace_harness_with_dom(
        """
        controller.upsertTaskArtifact({id: "artifact-task-1", taskId: "task-1", title: "Email", subtitle: "collecting", content: "Para quem?"});
        controller.upsertTaskArtifact({id: "artifact-task-2", taskId: "task-2", title: "Comparação: telefones", subtitle: "ready", content: "Samsung\\nApple\\nPixel"});
        controller.upsertSemanticObject({id: "option-samsung", kind: "option", title: "Samsung", taskId: "task-2"});
        controller.upsertSemanticObject({id: "option-apple", kind: "option", title: "Apple", taskId: "task-2"});
        controller.applyRepresentation({style: "cards", visible_object_ids: ["option-samsung","option-apple"], major_object_ids: [], hidden_object_ids: [], source: "auto"});
        controller.render();

        const directNodes = containerEl.children.filter((child) => child.dataset && child.dataset.id);
        const taskSlots = directNodes.filter((node) => node.dataset.taskSurfaceSlot === "true");
        const childNodes = directNodes.filter((node) => String(node.dataset.id).startsWith("option-"));
        console.log(JSON.stringify({
          taskSurface: containerEl.dataset.taskSurface,
          openTaskCount: containerEl.dataset.openTaskCount,
          taskSlotIds: taskSlots.map((node) => node.dataset.id).sort(),
          childNodeCount: childNodes.length,
          entityPointCount: entityPointCalls.length,
          lastEntityPoint: entityPointCalls[entityPointCalls.length - 1],
        }));
        """,
        _HOST_RECT,
    )
    assert result["taskSurface"] == "active"
    assert result["openTaskCount"] == "2"
    assert result["taskSlotIds"] == ["artifact-task-1", "artifact-task-2"]
    assert result["childNodeCount"] == 0
    assert result["entityPointCount"] >= 1
    assert result["lastEntityPoint"]["x"] > 500


def test_phase45_task_surface_four_slot_layout_is_repeat_render_idempotent() -> None:
    result = _run_workspace_harness_with_dom(
        """
        ["Email", "Comparação: telefones", "Comparação: carros", "Planeamento de férias"].forEach((title, index) => {
          controller.upsertTaskArtifact({id: `artifact-task-${index + 1}`, taskId: `task-${index + 1}`, title, subtitle: "ready", content: title});
        });
        controller.render();
        function snapshot() {
          return containerEl.children
            .filter((child) => child.dataset && child.dataset.taskSurfaceSlot === "true")
            .map((node) => ({
              id: node.dataset.id,
              index: node.dataset.taskSlotIndex,
              count: node.dataset.taskCount,
              left: node.style.left || "",
              top: node.style.top || "",
              width: node.style.width || "",
              height: node.style.height || "",
              transform: node.style.transform || "",
            }))
            .sort((a, b) => a.id.localeCompare(b.id));
        }
        const first = JSON.stringify(snapshot());
        controller.render();
        controller.render();
        const second = JSON.stringify(snapshot());
        console.log(JSON.stringify({stable: first === second, slots: JSON.parse(second)}));
        """,
        _HOST_RECT,
    )
    assert result["stable"] is True
    assert len(result["slots"]) == 4
    assert {slot["count"] for slot in result["slots"]} == {"4"}
    assert all("translate" not in slot["transform"] for slot in result["slots"])


def test_phase451_task_surface_default_width_is_slightly_wider_and_css_bound() -> None:
    result = _run_workspace_harness_with_dom(
        """
        controller.upsertTaskArtifact({id: "artifact-task-1", taskId: "task-1", title: "Email", subtitle: "ready", content: "Email"});
        controller.render();
        const slot = containerEl.children.find((child) => child.dataset && child.dataset.taskSurfaceSlot === "true");
        const handle = containerEl.children.find((child) => child.dataset && child.dataset.taskSurfaceResizeHandle === "true");
        console.log(JSON.stringify({
          width: slot.style.width,
          cssWidth: containerEl.style["--task-surface-width"],
          handleLeft: handle.style.left,
          handleCount: containerEl.children.filter((child) => child.dataset && child.dataset.taskSurfaceResizeHandle === "true").length,
        }));
        """,
        _HOST_RECT,
    )
    assert result["width"] == "439px"
    assert int(result["width"].removesuffix("px")) > 414
    assert result["cssWidth"] == "439px"
    assert result["handleLeft"] == "458px"
    assert result["handleCount"] == 1


def test_phase451_task_surface_width_clamps_invalid_or_extreme_preferences() -> None:
    tiny = _run_workspace_harness_with_dom(
        """
        controller.upsertTaskArtifact({id: "artifact-task-1", taskId: "task-1", title: "Email", subtitle: "ready", content: "Email"});
        controller.render();
        const slot = containerEl.children.find((child) => child.dataset && child.dataset.taskSurfaceSlot === "true");
        console.log(JSON.stringify({width: slot.style.width}));
        """,
        _HOST_RECT,
        pre_controller_js='window.localStorage.setItem("echo_os.task_surface.v1", JSON.stringify({widthRatio: 0.01}));',
    )
    huge = _run_workspace_harness_with_dom(
        """
        controller.upsertTaskArtifact({id: "artifact-task-1", taskId: "task-1", title: "Email", subtitle: "ready", content: "Email"});
        controller.render();
        const slot = containerEl.children.find((child) => child.dataset && child.dataset.taskSurfaceSlot === "true");
        console.log(JSON.stringify({width: slot.style.width}));
        """,
        _HOST_RECT,
        pre_controller_js='window.localStorage.setItem("echo_os.task_surface.v1", JSON.stringify({widthRatio: 0.99}));',
    )
    invalid = _run_workspace_harness_with_dom(
        """
        controller.upsertTaskArtifact({id: "artifact-task-1", taskId: "task-1", title: "Email", subtitle: "ready", content: "Email"});
        controller.render();
        const slot = containerEl.children.find((child) => child.dataset && child.dataset.taskSurfaceSlot === "true");
        console.log(JSON.stringify({width: slot.style.width}));
        """,
        _HOST_RECT,
        pre_controller_js='window.localStorage.setItem("echo_os.task_surface.v1", "{bad json");',
    )
    assert tiny["width"] == "317px"
    assert huge["width"] == "536px"
    assert invalid["width"] == "439px"


def test_phase451_manual_resize_persists_and_next_render_is_idempotent() -> None:
    result = _run_workspace_harness_with_dom(
        """
        ["Email", "Comparação: telefones"].forEach((title, index) => {
          controller.upsertTaskArtifact({id: `artifact-task-${index + 1}`, taskId: `task-${index + 1}`, title, subtitle: "ready", content: title});
        });
        controller.render();
        const handle = containerEl.children.find((child) => child.dataset && child.dataset.taskSurfaceResizeHandle === "true");
        handle._listeners.pointerdown[0]({pointerId: 4, clientX: 488, preventDefault() {}});
        document._listeners.pointermove.forEach((fn) => fn({pointerId: 4, clientX: 548}));
        document._listeners.pointerup.forEach((fn) => fn({pointerId: 4}));
        function snapshot() {
          return containerEl.children
            .filter((child) => child.dataset && child.dataset.taskSurfaceSlot === "true")
            .map((node) => ({id: node.dataset.id, left: node.style.left, top: node.style.top, width: node.style.width, height: node.style.height}))
            .sort((a, b) => a.id.localeCompare(b.id));
        }
        const afterDrag = JSON.stringify(snapshot());
        controller.render();
        controller.render();
        const afterRender = JSON.stringify(snapshot());
        console.log(JSON.stringify({
          stable: afterDrag === afterRender,
          slots: JSON.parse(afterRender),
          stored: JSON.parse(window.localStorage.getItem("echo_os.task_surface.v1")),
          handleCount: containerEl.children.filter((child) => child.dataset && child.dataset.taskSurfaceResizeHandle === "true").length,
        }));
        """,
        _HOST_RECT,
    )
    assert result["stable"] is True
    assert {slot["width"] for slot in result["slots"]} == {"499px"}
    assert abs(result["stored"]["widthRatio"] - (499 / 1220)) < 0.000001
    assert result["handleCount"] == 1


def test_phase451_persisted_preference_restores_and_viewport_resize_does_not_drift() -> None:
    result = _run_workspace_harness_with_dom(
        """
        ["Email", "Comparação: telefones", "Comparação: carros", "Plano"].forEach((title, index) => {
          controller.upsertTaskArtifact({id: `artifact-task-${index + 1}`, taskId: `task-${index + 1}`, title, subtitle: "ready", content: title});
        });
        controller.render();
        function snapshot() {
          return {
            width: containerEl.style["--task-surface-width"],
            handleLeft: containerEl.children.find((child) => child.dataset && child.dataset.taskSurfaceResizeHandle === "true").style.left,
            slots: containerEl.children
              .filter((child) => child.dataset && child.dataset.taskSurfaceSlot === "true")
              .map((node) => ({id: node.dataset.id, width: node.style.width, index: node.dataset.taskSlotIndex}))
              .sort((a, b) => a.id.localeCompare(b.id)),
          };
        }
        const first = JSON.stringify(snapshot());
        hostRect.width = 1000;
        controller.render();
        const narrowA = JSON.stringify(snapshot());
        controller.render();
        const narrowB = JSON.stringify(snapshot());
        hostRect.width = 1220;
        controller.render();
        const restored = JSON.stringify(snapshot());
        const semanticIds = controller.listObjects().map((obj) => obj.id).sort();
        console.log(JSON.stringify({
          first: JSON.parse(first),
          narrowStable: narrowA === narrowB,
          restoredSameAsFirst: restored === first,
          semanticIds,
        }));
        """,
        dict(_HOST_RECT),
        pre_controller_js='window.localStorage.setItem("echo_os.task_surface.v1", JSON.stringify({widthRatio: 0.4}));',
    )
    assert result["first"]["width"] == "488px"
    assert result["narrowStable"] is True
    assert result["restoredSameAsFirst"] is True
    assert result["semanticIds"] == ["artifact-task-1", "artifact-task-2", "artifact-task-3", "artifact-task-4"]


def test_phase451_echo_and_minimized_rail_follow_resized_task_surface() -> None:
    result = _run_workspace_harness_with_dom(
        """
        controller.upsertTaskArtifact({id: "artifact-task-1", taskId: "task-1", title: "Email", subtitle: "ready", content: "Email"});
        controller.upsertTaskArtifact({id: "artifact-task-2", taskId: "task-2", title: "Comparação: carros", subtitle: "ready", content: "Audi\\nBMW"});
        controller.applyAttention({object_states: {"artifact-task-2": "minimized"}, focus_object_id: "", source: "voice_command"});
        controller.render();
        const beforePoint = entityPointCalls[entityPointCalls.length - 1];
        const beforeRail = containerEl.children.find((child) => child.dataset && child.dataset.minimizedRail === "true").style.left;
        const handle = containerEl.children.find((child) => child.dataset && child.dataset.taskSurfaceResizeHandle === "true");
        handle._listeners.pointerdown[0]({pointerId: 9, clientX: 488, preventDefault() {}});
        document._listeners.pointermove.forEach((fn) => fn({pointerId: 9, clientX: 588}));
        document._listeners.pointerup.forEach((fn) => fn({pointerId: 9}));
        const afterPoint = entityPointCalls[entityPointCalls.length - 1];
        const afterRail = containerEl.children.find((child) => child.dataset && child.dataset.minimizedRail === "true").style.left;
        console.log(JSON.stringify({
          beforePoint,
          afterPoint,
          beforeRail,
          afterRail,
          railCount: containerEl.children.filter((child) => child.dataset && child.dataset.minimizedRail === "true").length,
        }));
        """,
        _HOST_RECT,
    )
    assert result["afterPoint"]["x"] > result["beforePoint"]["x"]
    assert result["beforeRail"] == result["afterRail"] == "24px"
    assert result["railCount"] == 1


def test_phase51_task_surface_renders_cognitive_projection_sections() -> None:
    result = _run_workspace_harness_with_dom(
        """
        controller.upsertTaskArtifact({
          id: "artifact-task-1",
          taskId: "task-1",
          title: "Comparação: telemóveis",
          subtitle: "narrow",
          content: "Notas brutas que não devem dominar",
          presentation: {
            version: 1,
            stage: "narrow",
            sections: [
              {role: "options", title: "Opções", priority: "high", items: [
                {label: "Samsung", details: ["799 €", "+ autonomia"]},
                {label: "Apple", details: ["899 €", "+ câmara"]},
              ]},
              {role: "constraints", title: "Limites", priority: "high", items: [
                {label: "orçamento máximo de 850 euros", details: []},
              ]},
              {role: "recommendation", title: "Neste momento", priority: "high", items: [
                {label: "Samsung parece a melhor opção", details: ["fica dentro do orçamento"]},
              ]},
            ],
          },
        });
        controller.render();
        const slot = containerEl.children.find((child) => child.dataset && child.dataset.taskSurfaceSlot === "true");
        const body = slot.querySelector(".workspace-object__body");
        const projection = body.children.find((child) => child.className === "workspace-object__cognitive");
        const sections = projection.children.map((section) => ({
          role: section.dataset.role,
          priority: section.dataset.priority,
          title: section.children[0].textContent,
          itemLabels: section.children.find((child) => String(child.tagName).toLowerCase() === "ul").children.map((item) => item.children[0].textContent),
        }));
        console.log(JSON.stringify({
          hasProjection: Boolean(projection),
          bodyInnerHtml: body.innerHTML || "",
          sectionRoles: sections.map((section) => section.role),
          sections,
        }));
        """,
        _HOST_RECT,
    )
    assert result["hasProjection"] is True
    assert result["bodyInnerHtml"] == ""
    assert result["sectionRoles"] == ["options", "constraints", "recommendation"]
    assert result["sections"][0]["itemLabels"] == ["Samsung", "Apple"]


def test_phase51_cognitive_projection_density_and_idempotence_for_four_tasks() -> None:
    result = _run_workspace_harness_with_dom(
        """
        const presentation = {
          version: 1,
          stage: "narrow",
          sections: [
            {role: "options", title: "Opções", priority: "high", items: [
              {label: "Samsung", details: ["799 €"]},
              {label: "Apple", details: ["899 €"]},
            ]},
            {role: "key_information", title: "O que já sabemos", priority: "medium", items: [
              {label: "detalhe menos importante", details: []},
            ]},
            {role: "next_step", title: "Próximo passo", priority: "high", text: "confirmar câmara"},
          ],
        };
        ["Telefones", "Férias", "Estocolmo", "Propostas"].forEach((title, index) => {
          controller.upsertTaskArtifact({
            id: `artifact-task-${index + 1}`,
            taskId: `task-${index + 1}`,
            title,
            subtitle: "ready",
            content: title,
            presentation,
          });
        });
        controller.render();
        function snapshot() {
          return containerEl.children
            .filter((child) => child.dataset && child.dataset.taskSurfaceSlot === "true")
            .map((node) => {
              const projection = node.querySelector(".workspace-object__cognitive");
              return {
                id: node.dataset.id,
                taskCount: node.dataset.taskCount,
                left: node.style.left,
                top: node.style.top,
                width: node.style.width,
                roles: projection.children.map((section) => section.dataset.role),
              };
            })
            .sort((a, b) => a.id.localeCompare(b.id));
        }
        const first = JSON.stringify(snapshot());
        controller.render();
        controller.render();
        const second = JSON.stringify(snapshot());
        console.log(JSON.stringify({stable: first === second, slots: JSON.parse(second)}));
        """,
        _HOST_RECT,
    )
    assert result["stable"] is True
    assert len(result["slots"]) == 4
    assert {slot["taskCount"] for slot in result["slots"]} == {"4"}
    for slot in result["slots"]:
        assert "options" in slot["roles"]
        assert "next_step" in slot["roles"]


def test_phase51_cognitive_projection_css_uses_slot_density_not_geometry() -> None:
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    assert ".workspace-object__cognitive" in css
    assert '.workspace-object[data-task-count="4"] .workspace-object__cognitive-section[data-role="key_information"]' in css
    assert "--task-surface-width" not in css[css.index(".workspace-object__cognitive"): css.index("@keyframes workspace-object-pulse")]


def test_phase45_minimized_task_uses_rail_not_open_slot() -> None:
    result = _run_workspace_harness_with_dom(
        """
        controller.upsertTaskArtifact({id: "artifact-task-1", taskId: "task-1", title: "Email", subtitle: "ready", content: "Email"});
        controller.upsertTaskArtifact({id: "artifact-task-2", taskId: "task-2", title: "Comparação: carros", subtitle: "ready", content: "Audi\\nBMW"});
        controller.applyAttention({object_states: {"artifact-task-2": "minimized"}, focus_object_id: "", source: "voice_command"});
        controller.render();
        const directNodes = containerEl.children.filter((child) => child.dataset && child.dataset.id);
        const openSlots = directNodes.filter((node) => node.dataset.taskSurfaceSlot === "true").map((node) => node.dataset.id);
        const rail = directNodes.filter((node) => node.dataset.minimizedRail === "true").map((node) => node.dataset.id);
        console.log(JSON.stringify({
          openTaskCount: containerEl.dataset.openTaskCount,
          minimizedTaskCount: containerEl.dataset.minimizedTaskCount,
          openSlots,
          rail,
        }));
        """,
        _HOST_RECT,
    )
    assert result["openTaskCount"] == "1"
    assert result["minimizedTaskCount"] == "1"
    assert result["openSlots"] == ["artifact-task-1"]
    assert result["rail"] == ["artifact-task-2"]


# --- Phase 7.4: explicit Voice Conversation Mode -----------------------
# LISTENING -> TRANSCRIBING -> THINKING -> SYNTHESIZING -> SPEAKING ->
# LISTENING, all driven from ONE backend state (EchoUIController.
# _voice_conversation_active + _voice_session_id), reusing submitMessage()/
# speakText() exactly as manual turns do (section 4) -- never a separate
# "voice answer" path. Letters below match the task's own lettered list.


def test_voice_conversation_a_start_begins_listening() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    controller = EchoUIController(_FakeEngine().respond, voice_available=True, voice_transcriber=_FakeVoiceTranscriber())
    states: list[str] = []
    conversation_events: list[bool] = []
    controller.stateChanged.connect(states.append)
    controller.voiceConversationChanged.connect(conversation_events.append)

    result = controller.startVoiceConversation()

    assert result is True
    assert controller.isVoiceConversationActive() is True
    assert conversation_events == [True]
    assert states[-1] == "listening"
    assert controller._voice_worker is not None
    _stop_and_drain(controller)


def test_voice_conversation_b_accepted_transcript_auto_submits_only_in_mode() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Auto reply.")
    transcriber = _FakeVoiceTranscriber(result="ola echo")
    controller = EchoUIController(engine.respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)
    responses: list[str] = []
    controller.responseReady.connect(responses.append)

    controller.startVoiceConversation()

    assert _run_until(lambda: responses, timeout_ms=3000)
    assert responses == ["Auto reply."]
    _stop_and_drain(controller)


def test_voice_conversation_c_manual_mic_does_not_auto_submit_outside_mode() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Should not appear.")
    transcriber = _FakeVoiceTranscriber(result="ola echo")
    controller = EchoUIController(engine.respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)
    responses: list[str] = []
    transcripts: list[str] = []
    controller.responseReady.connect(responses.append)
    controller.voiceTranscriptionReady.connect(transcripts.append)

    controller.startVoiceInput()

    assert _run_until(lambda: transcripts, timeout_ms=3000)
    _run_until(lambda: False, timeout_ms=300)  # grace window: prove nothing auto-submits afterwards
    assert transcripts == ["ola echo"]
    assert responses == []
    assert controller.isVoiceConversationActive() is False


def test_voice_conversation_d_response_auto_speaks_only_in_mode() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Spoken reply.")
    transcriber = _FakeVoiceTranscriber(result="ola echo")
    tts = _FakeTTSManager()
    controller = EchoUIController(
        engine.respond,
        voice_available=True,
        voice_transcriber=transcriber,
        tts_manager=tts,
        tts_player=lambda *_a, **_k: 0.01,
        thinking_min_ms=0,
    )

    controller.startVoiceConversation()

    assert _run_until(lambda: tts.calls, timeout_ms=3000)
    assert tts.calls == ["Spoken reply."]
    _stop_and_drain(controller)


def test_typed_turn_does_not_auto_speak_outside_conversation_mode() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Typed reply.")
    tts = _FakeTTSManager()
    controller = EchoUIController(engine.respond, tts_manager=tts, tts_player=lambda *_a, **_k: 0.01, thinking_min_ms=0)
    responses: list[str] = []
    controller.responseReady.connect(responses.append)

    controller.submitMessage("Hello")

    assert _run_until(lambda: responses, timeout_ms=3000)
    _run_until(lambda: False, timeout_ms=300)
    assert responses == ["Typed reply."]
    assert tts.calls == []


def test_voice_conversation_e_tts_finished_restarts_listening() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Spoken reply.")
    transcriber = _SequentialVoiceTranscriber([("ola echo", None), ("outra vez", None)])
    tts = _FakeTTSManager()
    controller = EchoUIController(
        engine.respond,
        voice_available=True,
        voice_transcriber=transcriber,
        tts_manager=tts,
        tts_player=lambda *_a, **_k: 0.01,
        thinking_min_ms=0,
    )

    controller.startVoiceConversation()

    assert _run_until(lambda: transcriber.call_count >= 2, timeout_ms=5000)
    assert tts.calls == ["Spoken reply."]
    _stop_and_drain(controller)


def test_voice_conversation_f_mic_not_active_while_speaking() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Spoken reply.")
    transcriber = _FakeVoiceTranscriber(result="ola echo")
    tts = _FakeTTSManager()

    def slow_player(_path, stop_checker=None) -> float:
        import time

        deadline = time.perf_counter() + 0.3
        while time.perf_counter() < deadline:
            if stop_checker and stop_checker():
                break
            time.sleep(0.01)
        return 0.0

    controller = EchoUIController(
        engine.respond,
        voice_available=True,
        voice_transcriber=transcriber,
        tts_manager=tts,
        tts_player=slow_player,
        thinking_min_ms=0,
    )

    controller.startVoiceConversation()

    assert _run_until(lambda: controller._tts_worker is not None, timeout_ms=3000)
    assert controller._voice_worker is None
    assert _run_until(lambda: controller._voice_worker is not None, timeout_ms=3000)
    _stop_and_drain(controller)


def test_voice_conversation_g_no_speech_rearms_without_submitting() -> None:
    from assistant.voice_input import VoiceNoSpeechDetected
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Only the second cycle should reach this.")
    transcriber = _SequentialVoiceTranscriber([(None, VoiceNoSpeechDetected("Nao detetei fala.")), ("ola echo", None)])
    controller = EchoUIController(engine.respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)
    responses: list[str] = []
    controller.responseReady.connect(responses.append)

    controller.startVoiceConversation()

    assert _run_until(lambda: responses, timeout_ms=5000)
    assert responses == ["Only the second cycle should reach this."]
    assert transcriber.call_count == 2
    _stop_and_drain(controller)


def test_voice_conversation_h_too_short_capture_rearms_without_fatal_error() -> None:
    from assistant.voice_input import VoiceInputError
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Second reply.")
    transcriber = _SequentialVoiceTranscriber(
        [
            (None, VoiceInputError("A gravacao foi demasiado curta. Gravaste 0.30s; o minimo configurado e 1.20s.")),
            ("ola echo", None),
        ]
    )
    controller = EchoUIController(engine.respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)
    errors: list[str] = []
    responses: list[str] = []
    controller.voiceInputError.connect(errors.append)
    controller.responseReady.connect(responses.append)

    controller.startVoiceConversation()

    assert _run_until(lambda: responses, timeout_ms=5000)
    assert errors == []
    assert responses == ["Second reply."]
    assert transcriber.call_count == 2
    _stop_and_drain(controller)


def test_voice_conversation_i_stop_during_listening() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    controller = EchoUIController(_FakeEngine().respond, voice_available=True, voice_transcriber=_BlockingVoiceTranscriber())
    conversation_events: list[bool] = []
    controller.voiceConversationChanged.connect(conversation_events.append)

    controller.startVoiceConversation()
    assert _run_until(lambda: controller._voice_worker is not None, timeout_ms=2000)

    controller.stopVoiceConversation()

    assert _run_until(lambda: controller._voice_worker is None, timeout_ms=3000)
    assert controller.isVoiceConversationActive() is False
    assert conversation_events == [True, False]


def test_voice_conversation_j_stop_during_transcribing() -> None:
    """From the controller's perspective "transcribing" is the SAME
    EchoVoiceWorker lifecycle as "listening" -- record_and_transcribe is one
    blocking call with no separate sub-state exposed. This exercises
    stopping once the worker's status has already advanced past the
    initial "A ouvir..." point."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    controller = EchoUIController(_FakeEngine().respond, voice_available=True, voice_transcriber=_BlockingVoiceTranscriber())

    controller.startVoiceConversation()
    assert _run_until(lambda: controller._voice_worker is not None, timeout_ms=2000)

    controller.stopVoiceConversation()

    assert _run_until(lambda: controller._voice_worker is None, timeout_ms=3000)
    assert controller.isVoiceConversationActive() is False


def test_voice_conversation_k_stop_during_thinking_does_not_auto_speak_or_rearm() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _SlowFakeEngine(reply="Late reply.", delay_seconds=0.25)
    transcriber = _FakeVoiceTranscriber(result="ola echo")
    tts = _FakeTTSManager()
    controller = EchoUIController(
        engine.respond,
        voice_available=True,
        voice_transcriber=transcriber,
        tts_manager=tts,
        tts_player=lambda *_a, **_k: 0.01,
        thinking_min_ms=0,
    )
    responses: list[str] = []
    controller.responseReady.connect(responses.append)

    controller.startVoiceConversation()
    assert _run_until(lambda: controller._active, timeout_ms=3000)

    controller.stopVoiceConversation()

    assert _run_until(lambda: responses, timeout_ms=3000)
    _run_until(lambda: False, timeout_ms=400)
    assert responses == ["Late reply."]
    assert tts.calls == []
    assert controller._voice_worker is None
    _stop_and_drain(controller)


def test_voice_conversation_l_stop_during_synthesis() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Reply.")
    transcriber = _FakeVoiceTranscriber(result="ola echo")
    tts = _FakeTTSManager(delay_seconds=0.3)
    controller = EchoUIController(
        engine.respond,
        voice_available=True,
        voice_transcriber=transcriber,
        tts_manager=tts,
        tts_player=lambda *_a, **_k: 0.0,
        thinking_min_ms=0,
    )

    controller.startVoiceConversation()
    assert _run_until(lambda: controller._tts_worker is not None, timeout_ms=3000)

    controller.stopVoiceConversation()

    assert controller.isVoiceConversationActive() is False
    assert _run_until(lambda: controller._tts_worker is None, timeout_ms=3000)
    _stop_and_drain(controller)


def test_voice_conversation_m_stop_during_speaking_returns_to_listening() -> None:
    """M. Preferred v1 behavior (section 14): STOP pressed while actively
    speaking interrupts playback only -- voice conversation mode stays ON
    and rearms listening. A separate stopVoiceConversation() call is the
    only action that exits the loop."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Reply.")
    transcriber = _FakeVoiceTranscriber(result="ola echo")
    tts = _FakeTTSManager()

    def slow_player(_path, stop_checker=None) -> float:
        import time

        for _ in range(50):
            if stop_checker and stop_checker():
                return 0.0
            time.sleep(0.01)
        return 0.0

    controller = EchoUIController(
        engine.respond,
        voice_available=True,
        voice_transcriber=transcriber,
        tts_manager=tts,
        tts_player=slow_player,
        thinking_min_ms=0,
    )
    states: list[str] = []
    controller.stateChanged.connect(states.append)

    controller.startVoiceConversation()
    assert _run_until(lambda: "speaking" in states, timeout_ms=3000)

    controller.stopSpeaking()

    assert controller.isVoiceConversationActive() is True
    assert _run_until(lambda: controller._voice_worker is not None, timeout_ms=3000)
    _stop_and_drain(controller)


def test_voice_conversation_n_stale_stt_callback_ignored_after_stop() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()

    class _StopControlledTranscriber:
        def __init__(self) -> None:
            import threading

            self.release_event = threading.Event()

        def record_and_transcribe(self, stop_event, status_callback=None) -> str:
            self.release_event.wait(timeout=5)
            return "late transcript"

    engine = _FakeEngine(reply="Should not be used.")
    transcriber = _StopControlledTranscriber()
    controller = EchoUIController(engine.respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)
    responses: list[str] = []
    controller.responseReady.connect(responses.append)

    controller.startVoiceConversation()
    assert _run_until(lambda: controller._voice_worker is not None, timeout_ms=2000)
    controller.stopVoiceConversation()
    transcriber.release_event.set()

    _run_until(lambda: False, timeout_ms=800)
    assert responses == []


def test_voice_conversation_o_stale_rearm_and_auto_speak_ignored_after_new_session() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    tts = _FakeTTSManager()
    controller = EchoUIController(_FakeEngine().respond, tts_manager=tts, voice_available=True, voice_transcriber=_FakeVoiceTranscriber())

    controller._voice_conversation_active = True
    controller._voice_session_id = 1
    stale_session = controller._voice_session_id
    # Mode stopped, then restarted -- a NEW session, distinct from the one
    # any in-flight callback captured.
    controller._voice_session_id = 2

    controller._rearm_voice_listening(stale_session)
    assert controller._voice_worker is None  # stale rearm must not start listening

    controller._auto_speak_response("stale response", stale_session)
    assert tts.calls == []  # stale auto-speak must not fire


def test_voice_conversation_p_duplicate_listening_cycles_cannot_start() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    controller = EchoUIController(_FakeEngine().respond, voice_available=True, voice_transcriber=_BlockingVoiceTranscriber())

    controller.startVoiceConversation()
    first_worker = controller._voice_worker
    assert first_worker is not None

    controller._rearm_voice_listening(controller._voice_session_id)

    assert controller._voice_worker is first_worker
    controller.stopVoiceConversation()
    _run_until(lambda: controller._voice_worker is None, timeout_ms=2000)


def test_voice_conversation_q_typed_turn_during_mode_stops_listening_and_rearms() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Typed while listening.")
    tts = _FakeTTSManager()
    controller = EchoUIController(
        engine.respond,
        voice_available=True,
        voice_transcriber=_BlockingVoiceTranscriber(),
        tts_manager=tts,
        tts_player=lambda *_a, **_k: 0.01,
        thinking_min_ms=0,
    )

    controller.startVoiceConversation()
    assert _run_until(lambda: controller._voice_worker is not None, timeout_ms=2000)

    controller.submitMessage("typed message")

    assert _run_until(lambda: tts.calls, timeout_ms=5000)
    assert tts.calls == ["Typed while listening."]
    assert controller.isVoiceConversationActive() is True
    assert _run_until(lambda: controller._voice_worker is not None, timeout_ms=3000)
    _stop_and_drain(controller)


def test_voice_conversation_r_ptpt_unavailable_tts_preserves_text_and_survives() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Resposta em português.")
    transcriber = _FakeVoiceTranscriber(result="ola echo")
    tts = _FakeTTSManager(available=True, pt_available=False)
    controller = EchoUIController(
        engine.respond,
        voice_available=True,
        voice_transcriber=transcriber,
        tts_manager=tts,
        tts_player=lambda *_a, **_k: 0.01,
        get_response_language=lambda: "pt-PT",
        thinking_min_ms=0,
    )
    responses: list[str] = []
    tts_errors: list[str] = []
    controller.responseReady.connect(responses.append)
    controller.ttsError.connect(tts_errors.append)

    controller.startVoiceConversation()

    assert _run_until(lambda: responses, timeout_ms=3000)
    assert _run_until(lambda: tts_errors, timeout_ms=3000)
    assert responses == ["Resposta em português."]
    assert tts.calls == []
    assert controller.isVoiceConversationActive() is True
    assert _run_until(lambda: controller._voice_worker is not None, timeout_ms=3000)
    _stop_and_drain(controller)


def test_voice_conversation_s_tts_provider_failure_recovers_loop() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Reply despite TTS failure.")
    transcriber = _FakeVoiceTranscriber(result="ola echo")
    tts = _FakeTTSManager(error=RuntimeError("pocket crashed"))
    controller = EchoUIController(engine.respond, voice_available=True, voice_transcriber=transcriber, tts_manager=tts, thinking_min_ms=0)
    tts_errors: list[str] = []
    controller.ttsError.connect(tts_errors.append)

    controller.startVoiceConversation()

    assert _run_until(lambda: tts_errors, timeout_ms=3000)
    assert controller.isVoiceConversationActive() is True
    assert _run_until(lambda: controller._voice_worker is not None, timeout_ms=3000)
    _stop_and_drain(controller)


def test_voice_conversation_t_tts_manager_reused_across_multiple_voice_turns() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Reply.")
    transcriber = _SequentialVoiceTranscriber([("turno um", None), ("turno dois", None)])
    tts = _FakeTTSManager()
    controller = EchoUIController(
        engine.respond,
        voice_available=True,
        voice_transcriber=transcriber,
        tts_manager=tts,
        tts_player=lambda *_a, **_k: 0.01,
        thinking_min_ms=0,
    )

    controller.startVoiceConversation()

    assert _run_until(lambda: len(tts.calls) >= 2, timeout_ms=6000)
    assert tts.calls == ["Reply.", "Reply."]
    assert tts.stopped is False
    _stop_and_drain(controller)


def test_voice_conversation_stop_from_idle_is_a_safe_noop() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    controller = EchoUIController(_FakeEngine().respond, voice_available=True, voice_transcriber=_FakeVoiceTranscriber())

    controller.stopVoiceConversation()

    assert controller.isVoiceConversationActive() is False


def test_voice_conversation_start_refuses_while_busy() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    controller = EchoUIController(_FakeEngine().respond, voice_available=True, voice_transcriber=_FakeVoiceTranscriber())
    controller._active = True
    errors: list[str] = []
    controller.voiceInputError.connect(errors.append)

    result = controller.startVoiceConversation()

    assert result is False
    assert controller.isVoiceConversationActive() is False
    assert errors


# --- Phase 7.4.1: real-use UX/lifecycle polish ------------------------------
# A corrective pass after real manual use of the Phase 7.4 voice-conversation
# loop surfaced problems automated tests hadn't caught: an accepted voice
# transcript stayed visible in the composer after being auto-submitted; a
# late benign STT result from an already-stopped session rendered as the
# main Echo response; and no backoff existed for repeated no-content voice
# cycles. See controller.py's _is_voice_worker_stale/_voice_rearm_delay and
# echo_ui.js's voiceTranscriptionReady handler for the fixes.


def test_voice_ux_polish_ab_composer_fill_gated_by_conversation_mode() -> None:
    """A+B (structural -- pytest has no DOM): echo_ui.js must only fill the
    composer from an accepted transcript when Voice Conversation Mode is
    OFF (manual MIC's existing fill-but-don't-send behavior, section 1
    "outside mode" case); it must skip the fill entirely when the mode is
    ON, since the controller already auto-submitted that transcript through
    a path that never clears the composer itself."""
    ui = (WEB / "echo_ui.js").read_text(encoding="utf-8")
    start = ui.index("activeController.voiceTranscriptionReady.connect")
    end = ui.index("if (activeController.voiceInputError)")
    handler = ui[start:end]

    assert "insertVoiceTranscription(text)" in handler
    assert "if (!voiceConversationActive)" in handler
    assert handler.index("if (!voiceConversationActive)") < handler.index("insertVoiceTranscription(text)")


def test_voice_ux_polish_a_accepted_transcript_auto_submits_in_mode() -> None:
    """A (behavioral): inside Voice Conversation Mode an accepted transcript
    still auto-submits and the loop still produces a spoken response --
    unchanged Phase 7.4 contract, re-verified alongside the composer fix."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Resposta falada.")
    transcriber = _FakeVoiceTranscriber(result="ola echo")
    tts = _FakeTTSManager()
    controller = EchoUIController(
        engine.respond,
        voice_available=True,
        voice_transcriber=transcriber,
        tts_manager=tts,
        tts_player=lambda *_a, **_k: 0.0,
        thinking_min_ms=0,
    )
    responses: list[str] = []
    controller.responseReady.connect(responses.append)

    controller.startVoiceConversation()

    assert _run_until(lambda: responses, timeout_ms=3000)
    assert responses == ["Resposta falada."]
    _stop_and_drain(controller)


def test_voice_ux_polish_b_manual_mic_still_fills_composer_path() -> None:
    """B (behavioral half): outside Voice Conversation Mode, manual MIC
    still emits the transcript (which the JS side -- verified structurally
    above -- fills into the composer) and never auto-submits. Same
    contract as Phase 7.4's test_voice_conversation_c, re-asserted here
    under this phase's naming for section 7 traceability."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Should not appear.")
    transcriber = _FakeVoiceTranscriber(result="ola echo")
    controller = EchoUIController(engine.respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)
    responses: list[str] = []
    transcripts: list[str] = []
    controller.responseReady.connect(responses.append)
    controller.voiceTranscriptionReady.connect(transcripts.append)

    controller.startVoiceInput()

    assert _run_until(lambda: transcripts, timeout_ms=3000)
    _run_until(lambda: False, timeout_ms=300)
    assert transcripts == ["ola echo"]
    assert responses == []
    assert controller.isVoiceConversationActive() is False


def test_voice_ux_polish_c_stop_during_recording_then_late_too_short_does_not_alter_response() -> None:
    """C (regression for the exact reported bug): Voice Mode is turned OFF
    while the mic worker is still open; it later finishes with the benign
    "recording too short" VoiceInputError. That must never replace the
    main Echo response, never flip to the error state, and never restart
    the loop."""
    import threading

    from assistant.voice_input import VoiceInputError
    from prototype_web_ui.controller import EchoUIController

    class _StopControlledTooShortTranscriber:
        def __init__(self) -> None:
            self.release_event = threading.Event()

        def record_and_transcribe(self, stop_event, status_callback=None) -> str:
            self.release_event.wait(timeout=5)
            raise VoiceInputError("A gravacao foi demasiado curta. Gravaste 0.96s; o minimo configurado e 2.00s.")

    _qt_app()
    engine = _FakeEngine(reply="Should not be used.")
    transcriber = _StopControlledTooShortTranscriber()
    controller = EchoUIController(engine.respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)
    responses: list[str] = []
    errors: list[str] = []
    states: list[str] = []
    controller.responseReady.connect(responses.append)
    controller.errorOccurred.connect(errors.append)
    controller.stateChanged.connect(states.append)

    controller.startVoiceConversation()
    assert _run_until(lambda: controller._voice_worker is not None, timeout_ms=2000)

    controller.stopVoiceConversation()
    transcriber.release_event.set()

    _run_until(lambda: False, timeout_ms=800)  # grace window for the late callback to actually arrive
    assert responses == []
    assert errors == []
    assert "error" not in states
    assert states[-1] == "idle"  # settles cleanly instead of staying stuck on "thinking" (cancelVoiceInput's flash)
    assert controller.isVoiceConversationActive() is False
    _stop_and_drain(controller)


def test_voice_ux_polish_d_stale_genuine_voice_error_ignored_after_stop() -> None:
    """D: a GENUINE (non-benign) STT error from a worker whose session was
    already stopped must be dropped exactly like a benign one -- the
    staleness check runs before the benign/genuine branch."""
    import threading

    from assistant.voice_input import VoiceInputError
    from prototype_web_ui.controller import EchoUIController

    class _StopControlledFailingTranscriber:
        def __init__(self) -> None:
            self.release_event = threading.Event()

        def record_and_transcribe(self, stop_event, status_callback=None) -> str:
            self.release_event.wait(timeout=5)
            raise VoiceInputError("Falha real do microfone durante a gravacao.")

    _qt_app()
    engine = _FakeEngine(reply="Should not be used.")
    transcriber = _StopControlledFailingTranscriber()
    controller = EchoUIController(engine.respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)
    errors: list[str] = []
    states: list[str] = []
    controller.voiceInputError.connect(errors.append)
    controller.stateChanged.connect(states.append)

    controller.startVoiceConversation()
    assert _run_until(lambda: controller._voice_worker is not None, timeout_ms=2000)

    controller.stopVoiceConversation()
    transcriber.release_event.set()

    _run_until(lambda: False, timeout_ms=800)
    assert errors == []
    assert "error" not in states
    assert states[-1] == "idle"
    assert controller._voice_conversation_failures == 0
    assert controller.isVoiceConversationActive() is False
    _stop_and_drain(controller)


def test_voice_ux_polish_e_stale_transcription_ready_signal_suppressed() -> None:
    """E: a late but otherwise VALID transcript from a stopped session's
    worker must not even reach the voiceTranscriptionReady signal -- not
    just "not auto-submitted" (already covered by Phase 7.4's test n), but
    genuinely suppressed so it cannot linger anywhere in the UI either."""
    import threading

    from prototype_web_ui.controller import EchoUIController

    class _StopControlledTranscriber:
        def __init__(self) -> None:
            self.release_event = threading.Event()

        def record_and_transcribe(self, stop_event, status_callback=None) -> str:
            self.release_event.wait(timeout=5)
            return "late transcript"

    _qt_app()
    engine = _FakeEngine(reply="Should not be used.")
    transcriber = _StopControlledTranscriber()
    controller = EchoUIController(engine.respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)
    transcripts: list[str] = []
    responses: list[str] = []
    controller.voiceTranscriptionReady.connect(transcripts.append)
    controller.responseReady.connect(responses.append)

    controller.startVoiceConversation()
    assert _run_until(lambda: controller._voice_worker is not None, timeout_ms=2000)

    controller.stopVoiceConversation()
    transcriber.release_event.set()

    _run_until(lambda: False, timeout_ms=800)
    assert transcripts == []
    assert responses == []
    _stop_and_drain(controller)


def test_voice_ux_polish_f_stale_voice_input_finished_does_not_restart_listening() -> None:
    """F: voiceInputFinished (thread-cleanup) still fires for a stale
    worker -- its own QThread must still be released -- but that cleanup
    must never resurrect the loop: mode stays off and no new listening
    cycle begins afterward."""
    import threading

    from prototype_web_ui.controller import EchoUIController

    class _StopControlledTranscriber:
        def __init__(self) -> None:
            self.release_event = threading.Event()

        def record_and_transcribe(self, stop_event, status_callback=None) -> str:
            self.release_event.wait(timeout=5)
            return "late transcript"

    _qt_app()
    transcriber = _StopControlledTranscriber()
    controller = EchoUIController(_FakeEngine().respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)
    finished_events: list[bool] = []
    controller.voiceInputFinished.connect(lambda: finished_events.append(True))

    controller.startVoiceConversation()
    assert _run_until(lambda: controller._voice_worker is not None, timeout_ms=2000)

    controller.stopVoiceConversation()
    transcriber.release_event.set()

    assert _run_until(lambda: finished_events, timeout_ms=2000)  # cleanup still runs
    _run_until(lambda: False, timeout_ms=500)  # grace window: no restart afterwards
    assert controller.isVoiceConversationActive() is False
    assert controller._voice_worker is None


def test_voice_ux_polish_i_genuine_microphone_failure_still_surfaces_error() -> None:
    """I: a real, non-benign, NON-stale microphone/dependency failure must
    still surface a concise error (section 4 explicitly preserves this) and
    still count toward the existing 3-strikes stop-the-mode limit."""
    from assistant.voice_input import MicrophoneCheckError
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Should not be reached.")
    transcriber = _FakeVoiceTranscriber(error=MicrophoneCheckError("Nao consegui abrir o microfone: dispositivo ocupado."))
    controller = EchoUIController(engine.respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)
    errors: list[str] = []
    controller.voiceInputError.connect(errors.append)

    controller.startVoiceConversation()

    assert _run_until(lambda: controller.isVoiceConversationActive() is False, timeout_ms=5000)
    assert len(errors) == 3  # one per consecutive genuine failure before the 3-strikes stop
    assert all("microfone" in message.lower() for message in errors)


def test_voice_ux_polish_j_portuguese_transcript_with_english_response_language_unaffected() -> None:
    """J: input language (whatever the user actually spoke) and
    response_language (Echo's reply language) are independent -- the
    controller passes the raw transcript through unchanged and speaks
    whatever get_response_language_callback reports, regardless of what
    language the transcript happens to be in."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Sure, here is the answer in English.")
    transcriber = _FakeVoiceTranscriber(result="Hoje fui almoçar com a minha família.")
    tts = _FakeTTSManager()
    controller = EchoUIController(
        engine.respond,
        voice_available=True,
        voice_transcriber=transcriber,
        tts_manager=tts,
        tts_player=lambda *_a, **_k: 0.0,
        get_response_language=lambda: "en-GB",
        thinking_min_ms=0,
    )
    responses: list[str] = []
    controller.responseReady.connect(responses.append)

    controller.startVoiceConversation()

    assert _run_until(lambda: tts.calls, timeout_ms=3000)
    assert responses == ["Sure, here is the answer in English."]
    assert tts.response_languages == ["en-GB"]
    _stop_and_drain(controller)


def test_voice_ux_polish_k_input_language_not_coupled_to_response_language() -> None:
    """K: the voice-transcription/listening path must never read or branch
    on response_language -- STT input language is a separate, pre-
    configured concern (assistant/voice_input.py's VoiceTranscriber.language),
    untouched by this phase and not coupled to Echo's reply language."""
    import inspect

    from prototype_web_ui.controller import EchoUIController

    source = inspect.getsource(EchoUIController._handle_voice_transcription) + inspect.getsource(
        EchoUIController._begin_listening
    )
    assert "response_language" not in source
    assert "get_response_language_callback" not in source


def test_voice_conversation_mode_retires_phase_741_idle_backoff() -> None:
    """Phase 7.5 (section 9): the Phase 7.4.1 no-content-cycle backoff
    (_voice_idle_cycles/_VOICE_IDLE_BACKOFF_*) is retired -- VAD now bounds
    listening-cycle length structurally (trailing-silence window below,
    max-utterance ceiling above), so the artificial widening backoff that
    guarded against rapid repeated Whisper calls is redundant. Benign
    cycles rearm at the plain fixed cadence again."""
    from prototype_web_ui.controller import EchoUIController

    controller_attrs = dir(EchoUIController)
    assert "_voice_idle_cycles" not in controller_attrs or not hasattr(EchoUIController, "_voice_idle_cycles")
    assert not hasattr(EchoUIController, "_VOICE_IDLE_BACKOFF_THRESHOLD")
    assert not hasattr(EchoUIController, "_VOICE_IDLE_BACKOFF_DELAY_MS")
    assert not hasattr(EchoUIController, "_voice_rearm_delay")


# --- Phase 7.5: VAD routing at the controller level --------------------------
# The VAD mechanism itself (state machine, thresholds, Whisper-avoidance) is
# unit-tested in tests/test_voice_input.py. These tests cover the piece that
# lives in the controller: EchoVoiceWorker(use_vad=...) must route Voice
# Conversation Mode to record_and_transcribe_vad and manual MIC to the
# unchanged record_and_transcribe -- and every existing session/staleness
# guarantee must keep holding regardless of which path was used.


class _DualPathVoiceTranscriber:
    """Implements BOTH the legacy manual-mode method and the new VAD
    method, each independently counted -- lets a test assert which one
    EchoVoiceWorker actually called instead of inferring it from side
    effects."""

    def __init__(self, result: str = "ola echo") -> None:
        self.result = result
        self.vad_calls = 0
        self.manual_calls = 0

    def record_and_transcribe_vad(self, stop_event, status_callback=None) -> str:
        self.vad_calls += 1
        if status_callback is not None:
            status_callback("A ouvir...")
        stop_event.set()
        return self.result

    def record_and_transcribe(self, stop_event, status_callback=None) -> str:
        self.manual_calls += 1
        if status_callback is not None:
            status_callback("A ouvir...")
        stop_event.set()
        return self.result


def test_voice_vad_conversation_mode_routes_to_vad_path() -> None:
    """E/F (controller level): Voice Conversation Mode calls
    record_and_transcribe_vad exactly once per turn, never the legacy
    method -- and never needs an external stop_event.set() from the test
    itself (the fake sets it only because a real transcriber's VAD path
    would have already stopped capture by the time it returns)."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Resposta.")
    transcriber = _DualPathVoiceTranscriber(result="ola echo")
    tts = _FakeTTSManager()
    controller = EchoUIController(
        engine.respond,
        voice_available=True,
        voice_transcriber=transcriber,
        tts_manager=tts,
        tts_player=lambda *_a, **_k: 0.0,
        thinking_min_ms=0,
    )
    responses: list[str] = []
    controller.responseReady.connect(responses.append)

    controller.startVoiceConversation()

    assert _run_until(lambda: responses, timeout_ms=3000)
    assert transcriber.vad_calls == 1
    assert transcriber.manual_calls == 0
    _stop_and_drain(controller)


def test_voice_vad_r_manual_mic_uses_legacy_path_not_vad() -> None:
    """R: manual MIC (outside Voice Conversation Mode) is untouched by
    Phase 7.5 -- it must still call the plain record_and_transcribe(),
    never record_and_transcribe_vad."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    transcriber = _DualPathVoiceTranscriber()
    controller = EchoUIController(_FakeEngine().respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)
    transcripts: list[str] = []
    controller.voiceTranscriptionReady.connect(transcripts.append)

    controller.startVoiceInput()

    assert _run_until(lambda: transcripts, timeout_ms=3000)
    assert transcriber.manual_calls == 1
    assert transcriber.vad_calls == 0


def test_voice_vad_j_stale_callback_ignored_with_vad_transcriber() -> None:
    """J: the Phase 7.4.1 session-staleness guard works identically
    whether the worker used the legacy or the new VAD capture path -- a
    late benign error from a VAD-based worker whose session was already
    stopped must still be dropped, not surfaced as the main response."""
    import threading

    from assistant.voice_input import VoiceInputError
    from prototype_web_ui.controller import EchoUIController

    class _StopControlledVadTranscriber:
        def __init__(self) -> None:
            self.release_event = threading.Event()

        def record_and_transcribe_vad(self, stop_event, status_callback=None) -> str:
            self.release_event.wait(timeout=5)
            raise VoiceInputError("A fala detetada foi demasiado curta. Fala detetada: 40ms; o minimo e 300ms.")

    _qt_app()
    transcriber = _StopControlledVadTranscriber()
    controller = EchoUIController(_FakeEngine().respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)
    errors: list[str] = []
    states: list[str] = []
    controller.voiceInputError.connect(errors.append)
    controller.stateChanged.connect(states.append)

    controller.startVoiceConversation()
    assert _run_until(lambda: controller._voice_worker is not None, timeout_ms=2000)

    controller.stopVoiceConversation()
    transcriber.release_event.set()

    _run_until(lambda: False, timeout_ms=800)
    assert errors == []
    assert "error" not in states
    assert states[-1] == "idle"


def test_voice_vad_q_english_speech_with_portuguese_response_supported() -> None:
    """Q: the reverse direction of Phase 7.4.1's Portuguese-in/English-out
    test -- English speech transcribed, response_language=pt-PT -- proving
    the architecture genuinely doesn't care about direction."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Resposta em português.")
    transcriber = _FakeVoiceTranscriber(result="I went to lunch with my family today.")
    tts = _FakeTTSManager()
    controller = EchoUIController(
        engine.respond,
        voice_available=True,
        voice_transcriber=transcriber,
        tts_manager=tts,
        tts_player=lambda *_a, **_k: 0.0,
        get_response_language=lambda: "pt-PT",
        thinking_min_ms=0,
    )
    responses: list[str] = []
    controller.responseReady.connect(responses.append)

    controller.startVoiceConversation()

    assert _run_until(lambda: tts.calls, timeout_ms=3000)
    assert responses == ["Resposta em português."]
    assert tts.response_languages == ["pt-PT"]
    _stop_and_drain(controller)


# --- Phase 7.6: Voice Standby Mode (wake/sleep) ------------------------------
# Wake/sleep phrase PARSING is unit-tested in tests/test_voice_wake.py
# (covers C/D from section 19 directly). These tests cover the controller-
# level state machine: standby is a genuinely separate mode from active
# (never both True), a captured standby transcript is a wake/ignore
# decision rather than conversational content, and a sleep command during
# active mode exits cleanly back to standby rather than being routed as a
# task/query.


def test_voice_standby_a_state_distinct_from_active() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    controller = EchoUIController(_FakeEngine().respond, voice_available=True, voice_transcriber=_BlockingVoiceTranscriber())

    result = controller.startVoiceStandby()

    assert result is True
    assert controller.isVoiceStandbyActive() is True
    assert controller.isVoiceConversationActive() is False
    assert controller.voiceMode() == "standby"
    controller.stopVoiceStandby()
    _run_until(lambda: controller._voice_worker is None, timeout_ms=2000)


def test_voice_standby_b_wake_intent_enters_active_mode() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Normal reply.")
    transcriber = _SequentialVoiceTranscriber([("Echo", None), ("ola echo", None)])
    tts = _FakeTTSManager()
    controller = EchoUIController(
        engine.respond,
        voice_available=True,
        voice_transcriber=transcriber,
        tts_manager=tts,
        tts_player=lambda *_a, **_k: 0.0,
        get_response_language=lambda: "en-GB",
        thinking_min_ms=0,
    )
    responses: list[str] = []
    controller.responseReady.connect(responses.append)

    controller.startVoiceStandby()

    assert _run_until(lambda: controller.isVoiceConversationActive(), timeout_ms=3000)
    assert controller.isVoiceStandbyActive() is False
    # Section 6: a deterministic acknowledgement is spoken + shown -- not an
    # LLM-composed reply -- before the mic ever reopens.
    assert _run_until(lambda: "I'm listening." in responses, timeout_ms=3000)
    # A normal turn afterward still works fully hands-free.
    assert _run_until(lambda: "Normal reply." in responses, timeout_ms=5000)
    _stop_and_drain(controller)


def test_voice_standby_e_non_wake_speech_does_not_enter_pipeline() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Should not be used.")
    transcriber = _SequentialVoiceTranscriber([("que horas sao", None), ("outra coisa qualquer", None)])
    controller = EchoUIController(engine.respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)
    responses: list[str] = []
    controller.responseReady.connect(responses.append)

    controller.startVoiceStandby()

    assert _run_until(lambda: transcriber.call_count >= 2, timeout_ms=5000)
    assert responses == []
    assert controller.isVoiceStandbyActive() is True
    controller.stopVoiceStandby()
    _run_until(lambda: controller._voice_worker is None, timeout_ms=2000)


def test_voice_standby_fg_sleep_intent_exits_active_without_routing_as_content() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Should not be used for the sleep phrase.")
    transcriber = _SequentialVoiceTranscriber([("vai dormir", None)])
    controller = EchoUIController(engine.respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)
    responses: list[str] = []
    controller.responseReady.connect(responses.append)

    controller.startVoiceConversation()

    assert _run_until(lambda: controller.isVoiceStandbyActive(), timeout_ms=5000)
    assert controller.isVoiceConversationActive() is False
    assert responses == []  # the sleep phrase never reached the engine/response pipeline
    controller.stopVoiceStandby()
    _run_until(lambda: controller._voice_worker is None, timeout_ms=2000)


def test_voice_conversation_sleep_a_vai_dormir_enters_standby() -> None:
    """Phase 7.6.5 (section 16.A): the baseline supported phrase must still
    work end to end -- ACTIVE -> STANDBY, no LLM call."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Should not be used.")
    transcriber = _SequentialVoiceTranscriber([("vai dormir", None)])
    controller = EchoUIController(engine.respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)
    responses: list[str] = []
    standby_states: list[bool] = []
    active_states: list[bool] = []
    controller.responseReady.connect(responses.append)
    controller.voiceStandbyChanged.connect(standby_states.append)
    controller.voiceConversationChanged.connect(active_states.append)

    controller.startVoiceConversation()

    assert _run_until(lambda: controller.isVoiceStandbyActive(), timeout_ms=5000)
    assert controller.isVoiceConversationActive() is False
    assert responses == []
    assert engine.calls == []
    assert True in active_states and False in active_states  # entered ACTIVE, then left it
    assert True in standby_states
    controller.stopVoiceStandby()
    _run_until(lambda: controller._voice_worker is None, timeout_ms=2000)


def test_voice_conversation_sleep_b_real_reported_phrase_nao_pode_ir_dormir() -> None:
    """Phase 7.6.5 (section 16.B): the EXACT real failure -- "Não, pode ir
    dormir." reached the normal conversation pipeline instead of being
    intercepted before this phase's parser fix."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Should not be used.")
    transcriber = _SequentialVoiceTranscriber([("Não, pode ir dormir.", None)])
    controller = EchoUIController(engine.respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)
    responses: list[str] = []
    controller.responseReady.connect(responses.append)

    controller.startVoiceConversation()

    assert _run_until(lambda: controller.isVoiceStandbyActive(), timeout_ms=5000)
    assert controller.isVoiceConversationActive() is False
    assert responses == []
    assert engine.calls == []
    controller.stopVoiceStandby()
    _run_until(lambda: controller._voice_worker is None, timeout_ms=2000)


def test_voice_conversation_sleep_c_go_to_sleep_enters_standby() -> None:
    """Phase 7.6.5 (section 16.C): the English form must work identically."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Should not be used.")
    transcriber = _SequentialVoiceTranscriber([("go to sleep", None)])
    controller = EchoUIController(engine.respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)
    responses: list[str] = []
    controller.responseReady.connect(responses.append)

    controller.startVoiceConversation()

    assert _run_until(lambda: controller.isVoiceStandbyActive(), timeout_ms=5000)
    assert controller.isVoiceConversationActive() is False
    assert responses == []
    assert engine.calls == []
    controller.stopVoiceStandby()
    _run_until(lambda: controller._voice_worker is None, timeout_ms=2000)


def test_voice_conversation_sleep_defg_short_stt_variant_and_false_positive() -> None:
    """Phase 7.6.5 (sections 16.D/E/F/G, combined): the short truncated STT
    variant ("Dorm.") is intercepted (no LLM call, no task/response
    content) exactly like the full phrase, while a genuine mention of sleep
    in ordinary conversation ("ontem fui dormir tarde") is left completely
    alone -- still reaches the engine and is spoken back as a normal turn."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Normal reply about yesterday.")
    transcriber = _SequentialVoiceTranscriber(
        [("Dorm.", None), ("ontem fui dormir tarde", None)]
    )
    tts = _FakeTTSManager()
    controller = EchoUIController(
        engine.respond,
        voice_available=True,
        voice_transcriber=transcriber,
        tts_manager=tts,
        tts_player=lambda *_a, **_k: 0.0,
        thinking_min_ms=0,
    )
    responses: list[str] = []
    controller.responseReady.connect(responses.append)

    controller.startVoiceConversation()

    # D/E/F: "Dorm." intercepted before reaching the engine -- ACTIVE exits
    # to STANDBY without any LLM call and without a task/response turn.
    assert _run_until(lambda: controller.isVoiceStandbyActive(), timeout_ms=5000)
    assert controller.isVoiceConversationActive() is False
    assert responses == []
    assert engine.calls == []

    # G: the false-positive sentence, said next, wakes Echo (it contains no
    # wake word though, so it must NOT wake by itself) -- re-enter ACTIVE
    # manually to verify the sentence itself still routes as normal
    # conversation when Echo IS listening for turns, i.e. it is not treated
    # as a sleep command.
    controller.stopVoiceStandby()
    _run_until(lambda: controller._voice_worker is None, timeout_ms=2000)
    controller.startVoiceConversation()
    assert _run_until(lambda: "Normal reply about yesterday." in responses, timeout_ms=5000)
    assert engine.calls == ["ontem fui dormir tarde"]
    _stop_and_drain(controller)


# --- Phase 7.6.5 Part B: reduce wake latency (section 17) -------------------


def test_voice_wake_j_wake_word_itself_never_reaches_the_engine() -> None:
    """J: no unnecessary conversation routing on wake -- the wake word
    transcript ("Echo") must never be treated as conversational content; a
    real turn spoken right after wake is what reaches the engine."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Real reply.")
    transcriber = _SequentialVoiceTranscriber([("Echo", None), ("uma pergunta real", None)])
    tts = _FakeTTSManager()
    controller = EchoUIController(
        engine.respond,
        voice_available=True,
        voice_transcriber=transcriber,
        tts_manager=tts,
        tts_player=lambda *_a, **_k: 0.0,
        thinking_min_ms=0,
    )

    controller.startVoiceStandby()

    assert _run_until(lambda: engine.calls == ["uma pergunta real"], timeout_ms=5000)
    _stop_and_drain(controller)


def test_voice_wake_ik_timing_diagnostic_reports_small_classification_and_transition_ms(capsys) -> None:
    """I/K: [VOICE_WAKE_TIMING] must show classification (deterministic
    text matching, not an LLM call) and the ACTIVE-flip itself both taking
    a negligible, bounded amount of time -- no redundant sleep/timer sits
    between STT finishing and ACTIVE becoming true."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Reply.")
    transcriber = _SequentialVoiceTranscriber([("Echo", None)])
    tts = _FakeTTSManager()
    controller = EchoUIController(
        engine.respond,
        voice_available=True,
        voice_transcriber=transcriber,
        tts_manager=tts,
        tts_player=lambda *_a, **_k: 0.0,
        thinking_min_ms=0,
    )

    captured = ""

    def _seen() -> bool:
        nonlocal captured
        captured += capsys.readouterr().out
        return "[VOICE_WAKE_TIMING]" in captured

    controller.startVoiceStandby()

    assert _run_until(_seen, timeout_ms=5000)
    line = next(line for line in captured.splitlines() if "[VOICE_WAKE_TIMING]" in line)
    fields = dict(part.split("=", 1) for part in line.split(" ")[1:])
    assert float(fields["classification_ms"]) < 50.0
    assert float(fields["wake_to_active_ms"]) < 50.0
    _stop_and_drain(controller)


def test_voice_wake_l_acknowledgement_does_not_block_active_transition() -> None:
    """L: ACTIVE must flip to True as soon as the wake intent is matched --
    even while the acknowledgement TTS call is still in flight (a slow
    player standing in for real synthesis+playback time)."""
    import time as _time

    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Reply.")
    transcriber = _SequentialVoiceTranscriber([("Echo", None)])
    tts = _FakeTTSManager()
    active_when_player_called: list[bool] = []

    def slow_player(_path, stop_checker=None, **_kwargs) -> float:
        active_when_player_called.append(controller.isVoiceConversationActive())
        deadline = _time.perf_counter() + 0.2
        while _time.perf_counter() < deadline:
            if stop_checker and stop_checker():
                break
            _time.sleep(0.01)
        return 0.2

    controller = EchoUIController(
        engine.respond,
        voice_available=True,
        voice_transcriber=transcriber,
        tts_manager=tts,
        tts_player=slow_player,
        thinking_min_ms=0,
    )

    controller.startVoiceStandby()

    assert _run_until(lambda: active_when_player_called, timeout_ms=5000)
    # ACTIVE was ALREADY true by the time the (slow) acknowledgement
    # playback started -- the transition never waited on it.
    assert active_when_player_called == [True]
    _stop_and_drain(controller)


def test_voice_wake_m_ignored_standby_wake_rearms_faster_than_active(capsys) -> None:
    """M: an ignored/non-wake utterance heard during STANDBY must rearm
    listening with the faster 150ms standby-only delay, not ACTIVE's
    350ms -- confirmed here by reading the actual [VOICE_LOOP]
    event=rearm_listening delay_ms=... diagnostic line."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Should not be used.")
    transcriber = _SequentialVoiceTranscriber([("que horas sao", None), ("outra coisa qualquer", None)])
    controller = EchoUIController(engine.respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)

    captured = ""

    def _seen() -> bool:
        nonlocal captured
        captured += capsys.readouterr().out
        return "event=rearm_listening" in captured

    controller.startVoiceStandby()

    assert _run_until(_seen, timeout_ms=5000)
    line = next(line for line in captured.splitlines() if "event=rearm_listening" in line)
    assert "delay_ms=150" in line
    controller.stopVoiceStandby()
    _run_until(lambda: controller._voice_worker is None, timeout_ms=2000)


def test_voice_wake_m_active_rearm_delay_remains_unchanged(capsys) -> None:
    """M (counterpart): the SAME rearm path during ACTIVE conversation must
    still use the original 350ms delay -- only STANDBY got faster."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Reply.")
    transcriber = _SequentialVoiceTranscriber([("", None), ("outra coisa qualquer", None)])
    controller = EchoUIController(engine.respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)

    captured = ""

    def _seen() -> bool:
        nonlocal captured
        captured += capsys.readouterr().out
        return "event=rearm_listening" in captured

    controller.startVoiceConversation()

    assert _run_until(_seen, timeout_ms=5000)
    line = next(line for line in captured.splitlines() if "event=rearm_listening" in line)
    assert "delay_ms=350" in line
    _stop_and_drain(controller)


def test_voice_standby_h_wake_sleep_does_not_touch_tts_manager_teardown() -> None:
    """H: transitioning standby<->active must never call the TTS manager's
    own teardown/stop -- only a worker's own per-request stop_event, which
    never touches the shared persistent Pocket process."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Reply.")
    transcriber = _SequentialVoiceTranscriber([("Echo", None)])
    tts = _FakeTTSManager()
    controller = EchoUIController(
        engine.respond,
        voice_available=True,
        voice_transcriber=transcriber,
        tts_manager=tts,
        tts_player=lambda *_a, **_k: 0.0,
        thinking_min_ms=0,
    )

    controller.startVoiceStandby()

    assert _run_until(lambda: controller.isVoiceConversationActive(), timeout_ms=3000)
    assert tts.stopped is False
    _stop_and_drain(controller)
    assert tts.stopped is False


def test_voice_standby_i_stale_callback_after_sleep_cannot_restart_active() -> None:
    import threading

    from prototype_web_ui.controller import EchoUIController

    class _StopControlledTranscriber:
        def __init__(self) -> None:
            self.release_event = threading.Event()
            self.first_call = True

        def record_and_transcribe(self, stop_event, status_callback=None) -> str:
            if self.first_call:
                self.first_call = False
                return "vai dormir"
            self.release_event.wait(timeout=5)
            return "late transcript"

    _qt_app()
    engine = _FakeEngine(reply="Should not be used.")
    transcriber = _StopControlledTranscriber()
    controller = EchoUIController(engine.respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)
    responses: list[str] = []
    controller.responseReady.connect(responses.append)

    controller.startVoiceConversation()
    assert _run_until(lambda: controller.isVoiceStandbyActive(), timeout_ms=5000)
    # The standby loop already opened a second, now-blocked listening cycle.
    assert _run_until(lambda: controller._voice_worker is not None, timeout_ms=2000)

    controller.stopVoiceStandby()
    transcriber.release_event.set()

    _run_until(lambda: False, timeout_ms=800)
    assert responses == []
    assert controller.isVoiceConversationActive() is False
    assert controller.isVoiceStandbyActive() is False


def test_voice_standby_j_off_stays_off_without_manual_action() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    controller = EchoUIController(_FakeEngine().respond, voice_available=True, voice_transcriber=_FakeVoiceTranscriber())

    assert controller.voiceMode() == "off"
    assert controller.isVoiceConversationActive() is False
    assert controller.isVoiceStandbyActive() is False


# --- Phase 7.6.1: fix short wake commands in standby -------------------------
# Real-mic validation showed standby was reusing ACTIVE's vad_config
# (min_speech_ms~300ms), so a short wake word like "Echo" (measured at
# 200-280ms) was routinely rejected as a benign "too short" retry before
# Whisper ever ran. The VAD-level fix (a separate, more permissive
# standby_vad_config) is unit-tested in tests/test_voice_input.py; these
# tests cover the controller-level wiring: standby actually requests its
# own config, short wake words still correctly enter ACTIVE, and short
# non-wake noise still doesn't.


class _VadConfigCapturingTranscriber:
    """Records which vad_config was actually passed to
    record_and_transcribe_vad on each call -- proves standby requests its
    OWN, more permissive config while active does not request one at all
    (falling back to the transcriber's normal default)."""

    def __init__(self, result: str = "Echo") -> None:
        self.result = result
        self.received_vad_configs: list[object] = []
        # Sentinels distinct from any real VadConfig instance, so the test
        # can assert exactly which one (if any) was passed through.
        self.standby_vad_config = object()
        self.vad_config = object()

    def record_and_transcribe_vad(
        self,
        stop_event,
        status_callback=None,
        *,
        input_language=None,
        vad_config=None,
        on_speech_confirmed=None,
        require_confirmation_before_recording=False,
    ) -> str:
        self.received_vad_configs.append(vad_config)
        stop_event.set()
        return self.result

    def record_and_transcribe(self, stop_event, status_callback=None) -> str:
        raise AssertionError("manual (legacy) path must not be used for standby/active workers")


def test_voice_standby_f_standby_requests_its_own_vad_config() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    transcriber = _VadConfigCapturingTranscriber(result="Echo")
    controller = EchoUIController(_FakeEngine().respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)

    controller.startVoiceStandby()

    assert _run_until(lambda: transcriber.received_vad_configs, timeout_ms=3000)
    assert transcriber.received_vad_configs[0] is transcriber.standby_vad_config
    _stop_and_drain(controller)


# --- Phase 7.6.6: STANDBY wake fast path (section 16 A/B/J/M + prewarm/logging) --
# EchoVoiceWorker.run() duck-types record_and_transcribe_wake the same way
# it already duck-types record_and_transcribe_vad -- a transcriber double
# without it (every OLDER fake, including _VadConfigCapturingTranscriber
# above) safely falls through to the pre-7.6.6 standby path, which is why
# none of those existing tests needed updating for this phase.


class _WakeMethodCapturingTranscriber:
    """Records which method EchoVoiceWorker actually invoked for each mode
    -- proves STANDBY uses the new fast wake path (record_and_transcribe_
    wake) while ACTIVE keeps using the Medium-backed vad path
    (record_and_transcribe_vad), never the other way around."""

    def __init__(self, wake_result: str = "Echo", active_result: str = "reply") -> None:
        self.wake_result = wake_result
        self.active_result = active_result
        self.wake_calls = 0
        self.vad_calls: list[object] = []
        self.standby_vad_config = object()
        self.vad_config = object()
        self.model_name = "medium"

    def record_and_transcribe_wake(self, stop_event, status_callback=None, *, on_speech_confirmed=None) -> str:
        self.wake_calls += 1
        stop_event.set()
        return self.wake_result

    def record_and_transcribe_vad(
        self,
        stop_event,
        status_callback=None,
        *,
        input_language=None,
        vad_config=None,
        on_speech_confirmed=None,
        require_confirmation_before_recording=False,
    ) -> str:
        self.vad_calls.append(vad_config)
        stop_event.set()
        return self.active_result

    def record_and_transcribe(self, stop_event, status_callback=None) -> str:
        raise AssertionError("manual (legacy) path must not be used for standby/active workers")


def test_wake_fast_path_a_standby_uses_wake_recognizer_not_medium_vad_path() -> None:
    """A: STANDBY must call the new record_and_transcribe_wake method --
    never the old Medium-backed record_and_transcribe_vad path."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    transcriber = _WakeMethodCapturingTranscriber(wake_result="Echo")
    controller = EchoUIController(_FakeEngine().respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)

    controller.startVoiceStandby()

    assert _run_until(lambda: controller.isVoiceConversationActive(), timeout_ms=3000)
    assert transcriber.wake_calls == 1
    assert transcriber.vad_calls == []
    _stop_and_drain(controller)


def test_wake_fast_path_b_active_still_uses_medium_vad_path_not_wake_recognizer() -> None:
    """B: ACTIVE conversation transcription must keep using the Medium-
    backed record_and_transcribe_vad path -- the fast wake recognizer must
    never be touched for a real conversational turn."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Reply.")
    transcriber = _WakeMethodCapturingTranscriber(active_result="uma pergunta real")
    tts = _FakeTTSManager()
    controller = EchoUIController(
        engine.respond,
        voice_available=True,
        voice_transcriber=transcriber,
        tts_manager=tts,
        tts_player=lambda *_a, **_k: 0.0,
        thinking_min_ms=0,
    )

    controller.startVoiceConversation()

    assert _run_until(lambda: engine.calls == ["uma pergunta real"], timeout_ms=5000)
    assert transcriber.wake_calls == 0
    _stop_and_drain(controller)


def test_wake_fast_path_j_single_wake_transcript_triggers_active_transition_exactly_once() -> None:
    """J: one wake transcript must flip ACTIVE exactly once -- no duplicate
    transition."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    transcriber = _WakeMethodCapturingTranscriber(wake_result="Echo")
    controller = EchoUIController(_FakeEngine().respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)
    active_true_count = 0

    def _on_active_changed(active: bool) -> None:
        nonlocal active_true_count
        if active:
            active_true_count += 1

    controller.voiceConversationChanged.connect(_on_active_changed)

    controller.startVoiceStandby()

    assert _run_until(lambda: controller.isVoiceConversationActive(), timeout_ms=3000)
    assert active_true_count == 1
    _stop_and_drain(controller)


def test_wake_fast_path_m_unrelated_wake_recognizer_result_does_not_activate() -> None:
    """M: the deterministic wake parser remains the sole authority --
    whatever text the fast recognizer produces, it still must classify as
    "wake" before ACTIVE is entered."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    transcriber = _WakeMethodCapturingTranscriber(wake_result="ola, tudo bem")
    controller = EchoUIController(_FakeEngine().respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)

    controller.startVoiceStandby()

    assert _run_until(lambda: transcriber.wake_calls >= 1, timeout_ms=3000)
    _run_until(lambda: False, timeout_ms=300)
    assert controller.isVoiceConversationActive() is False
    assert controller.isVoiceStandbyActive() is True
    controller.stopVoiceStandby()
    _run_until(lambda: controller._voice_worker is None, timeout_ms=2000)


class _PrewarmCapturingWakeRecognizer:
    def __init__(self) -> None:
        import threading

        self.prewarm_calls = 0
        self._lock = threading.Lock()

    def prewarm(self) -> None:
        with self._lock:
            self.prewarm_calls += 1


def test_wake_fast_path_prewarm_triggered_asynchronously_on_standby_entry() -> None:
    """Section 6: entering STANDBY must kick off the wake recognizer's
    prewarm() -- without blocking the Slot/UI thread."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    transcriber = _WakeMethodCapturingTranscriber(wake_result="Echo")
    fake_wake_recognizer = _PrewarmCapturingWakeRecognizer()
    transcriber.wake_recognizer = fake_wake_recognizer
    controller = EchoUIController(_FakeEngine().respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)

    controller.startVoiceStandby()

    assert _run_until(lambda: fake_wake_recognizer.prewarm_calls >= 1, timeout_ms=3000)
    # The prewarm QThread must be genuinely finished (not just its side
    # effect observed) before this test returns -- letting a still-running
    # QThread's Python wrapper go out of scope and get garbage collected is
    # a real PySide6 crash risk (the same class of bug _stop_and_drain's
    # own docstring warns about for the other worker threads).
    prewarm_thread = getattr(controller, "_wake_prewarm_thread", None)
    if prewarm_thread is not None:
        prewarm_thread.wait(2000)
    _stop_and_drain(controller)


def test_wake_fast_path_timing_log_includes_recognizer_and_model_fields(capsys) -> None:
    """Section 13: [VOICE_WAKE_TIMING] must include recognizer=/model=/
    model_load_ms=/fallback_to_medium= so a real log makes it obvious which
    recognizer actually produced the transcript."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    transcriber = _WakeMethodCapturingTranscriber(wake_result="Echo")
    controller = EchoUIController(_FakeEngine().respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)

    captured = ""

    def _seen() -> bool:
        nonlocal captured
        captured += capsys.readouterr().out
        return "[VOICE_WAKE_TIMING]" in captured

    controller.startVoiceStandby()

    assert _run_until(_seen, timeout_ms=3000)
    line = next(line for line in captured.splitlines() if "[VOICE_WAKE_TIMING]" in line)
    assert "recognizer=" in line
    assert "model=" in line
    assert "model_load_ms=" in line
    assert "fallback_to_medium=" in line
    _stop_and_drain(controller)


# --- Phase 7.6.6.1: wake prewarm QThread shutdown lifetime ------------------
# Real runtime crash: `RuntimeError: libshiboken: Internal C++ object
# (_WakeRecognizerPrewarmThread) already deleted` inside shutdown()'s
# `.wait()` call, plus a `QThread: Destroyed while thread '' is still
# running` warning. Root cause: the prewarm thread's `finished` signal was
# connected straight to `thread.deleteLater()`, but nothing ever cleared the
# controller's own `self._wake_prewarm_thread` reference -- so once Qt
# actually deleted the C++ object (asynchronously, on a later event-loop
# turn), that attribute became a dangling wrapper. shutdown() calling
# `.wait()` on it was a genuine use-after-delete. The fix: the controller
# owns the reference for the thread's entire valid lifetime and nowhere
# else -- `_on_wake_prewarm_finished` clears it FIRST, before requesting
# deletion -- with `shiboken6.isValid()` as a secondary, defensive guard in
# `_shutdown_wake_prewarm_thread`, never the primary fix.


class _SlowPrewarmWakeRecognizer:
    """A fake whose prewarm() genuinely blocks for a bit -- lets a test
    observe/act on the controller WHILE the QThread is still running,
    instead of racing a near-instantaneous fake."""

    def __init__(self, delay_seconds: float = 0.3) -> None:
        self.delay_seconds = delay_seconds
        self.prewarm_calls = 0

    def prewarm(self) -> None:
        import time

        time.sleep(self.delay_seconds)
        self.prewarm_calls += 1


def test_wake_prewarm_a_shutdown_while_prewarm_still_running_waits_safely() -> None:
    """A: closing the app WHILE the prewarm thread is still mid-flight must
    wait for it (bounded) and never raise/crash."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    transcriber = _WakeMethodCapturingTranscriber(wake_result="Echo")
    fake_wake_recognizer = _SlowPrewarmWakeRecognizer(delay_seconds=0.3)
    transcriber.wake_recognizer = fake_wake_recognizer
    controller = EchoUIController(_FakeEngine().respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)

    controller.startVoiceStandby()
    thread = controller._wake_prewarm_thread
    assert thread is not None
    assert thread.isRunning() is True

    controller.shutdown()  # must block until prewarm finishes, never raise

    assert fake_wake_recognizer.prewarm_calls == 1
    assert controller._wake_prewarm_thread is None
    assert thread.isRunning() is False


def test_wake_prewarm_b_shutdown_after_prewarm_already_finished_is_a_noop() -> None:
    """B: closing the app AFTER the prewarm thread already finished (and
    its finished->clear->deleteLater flow already ran) must be a safe
    no-op -- nothing left to wait on or delete."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    transcriber = _WakeMethodCapturingTranscriber(wake_result="Echo")
    fake_wake_recognizer = _PrewarmCapturingWakeRecognizer()
    transcriber.wake_recognizer = fake_wake_recognizer
    controller = EchoUIController(_FakeEngine().respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)

    controller.startVoiceStandby()
    assert _run_until(lambda: controller._wake_prewarm_thread is None, timeout_ms=3000)

    controller.shutdown()  # nothing to do -- must not raise

    assert fake_wake_recognizer.prewarm_calls == 1


def test_wake_prewarm_c_shutdown_after_qt_object_already_deleted_does_not_crash() -> None:
    """C/E/F (the exact real bug): a stale Python wrapper whose underlying
    C++ object was ALREADY deleted must never reach .wait() -- this
    reproduces the real crash's precondition directly (a dangling
    _WakeRecognizerPrewarmThread reference), which the fixed code detects
    via shiboken6.isValid() and handles gracefully instead of raising
    `RuntimeError: libshiboken: ... already deleted`."""
    import shiboken6

    from prototype_web_ui.controller import EchoUIController, _WakeRecognizerPrewarmThread

    _qt_app()
    transcriber = _WakeMethodCapturingTranscriber(wake_result="Echo")
    controller = EchoUIController(_FakeEngine().respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)

    dangling = _WakeRecognizerPrewarmThread(lambda: None, controller)
    dangling.start()
    dangling.wait(2000)
    # Force the C++ object gone deterministically (deleteLater()'s actual
    # deferred-delete timing isn't guaranteed to have run after a bounded
    # number of processEvents() calls in a script, not a real exec() loop)
    # -- shiboken6.delete() is the direct, synchronous equivalent for
    # reproducing "the underlying object is already gone" precisely.
    shiboken6.delete(dangling)
    assert not shiboken6.isValid(dangling)

    # Simulate the real bug's exact precondition: the controller's own
    # attribute still points at the now-deleted wrapper.
    controller._wake_prewarm_thread = dangling

    controller.shutdown()  # must NOT raise RuntimeError

    assert controller._wake_prewarm_thread is None


def test_wake_prewarm_d_shutdown_called_twice_is_safe() -> None:
    """D: calling shutdown() twice must never double-wait or double-delete."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    transcriber = _WakeMethodCapturingTranscriber(wake_result="Echo")
    fake_wake_recognizer = _PrewarmCapturingWakeRecognizer()
    transcriber.wake_recognizer = fake_wake_recognizer
    controller = EchoUIController(_FakeEngine().respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)

    controller.startVoiceStandby()
    controller.shutdown()
    controller.shutdown()  # must be a safe no-op the second time

    assert controller._wake_prewarm_thread is None


def test_wake_prewarm_g_normal_wake_fast_path_still_works() -> None:
    """G: none of the shutdown/ownership fixes changed normal wake
    recognition behavior -- a clear "Echo" still activates."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    transcriber = _WakeMethodCapturingTranscriber(wake_result="Echo")
    controller = EchoUIController(_FakeEngine().respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)

    controller.startVoiceStandby()

    assert _run_until(lambda: controller.isVoiceConversationActive(), timeout_ms=3000)
    _stop_and_drain(controller)


def test_wake_prewarm_h_never_starts_a_second_prewarm_while_one_in_flight() -> None:
    """H: re-entering standby-prewarm logic while a prewarm is already
    running must never spawn a second QThread -- the existing one already
    warms the SAME shared WakeRecognizer, and starting a second (or
    reloading the model twice) is exactly what section 5 forbids."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    transcriber = _WakeMethodCapturingTranscriber(wake_result="Echo")
    fake_wake_recognizer = _SlowPrewarmWakeRecognizer(delay_seconds=0.3)
    transcriber.wake_recognizer = fake_wake_recognizer
    controller = EchoUIController(_FakeEngine().respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)

    controller._begin_voice_standby(reason="user_request")
    first_thread = controller._wake_prewarm_thread
    controller._prewarm_wake_recognizer()
    second_thread = controller._wake_prewarm_thread

    assert first_thread is second_thread
    controller.shutdown()
    assert fake_wake_recognizer.prewarm_calls == 1


def test_wake_prewarm_real_lifecycle_all_three_close_timings_are_safe() -> None:
    """Reproduces the real reported sequence as closely as an automated
    test reasonably can: a REAL _WakeRecognizerPrewarmThread driving a REAL
    WakeRecognizer (only the underlying `whisper` module is faked, to avoid
    a real model load/download), closed at the three real manual-
    validation timings from the task -- immediately, mid-flight, and after
    the prewarm has already finished. None may raise or crash."""
    import sys
    import time

    from assistant.wake_recognizer import WakeRecognizer
    from prototype_web_ui.controller import EchoUIController

    class _FakeModel:
        def transcribe(self, *_args: object, **_kwargs: object) -> dict[str, str]:
            return {"text": "Echo"}

    class _FakeWhisperModuleForPrewarm:
        def __init__(self, load_delay_seconds: float) -> None:
            self.load_delay_seconds = load_delay_seconds

        def load_model(self, _model_name: str) -> _FakeModel:
            time.sleep(self.load_delay_seconds)
            return _FakeModel()

    for close_timing, load_delay_seconds in (
        ("immediately", 0.3),
        ("mid_flight", 0.3),
        ("after_finished", 0.0),
    ):
        original_whisper = sys.modules.get("whisper")
        sys.modules["whisper"] = _FakeWhisperModuleForPrewarm(load_delay_seconds)
        try:
            app = _qt_app()
            transcriber = _WakeMethodCapturingTranscriber(wake_result="Echo")
            transcriber.wake_recognizer = WakeRecognizer(model_name="tiny")
            controller = EchoUIController(
                _FakeEngine().respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0
            )

            controller._begin_voice_standby(reason="user_request")
            if close_timing == "after_finished":
                for _ in range(50):
                    app.processEvents()
                    if controller._wake_prewarm_thread is None:
                        break
                    time.sleep(0.01)

            controller.shutdown()  # must never raise, regardless of timing
        finally:
            if original_whisper is not None:
                sys.modules["whisper"] = original_whisper
            else:
                sys.modules.pop("whisper", None)


def test_voice_standby_d_short_wake_word_never_submitted_as_turn() -> None:
    """D: a short wake-like transcript ("Echo") is never routed through
    submitMessage()/the engine as conversational content -- only the
    deterministic wake acknowledgement appears."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Should not be a reply to Echo itself.")
    transcriber = _SequentialVoiceTranscriber([("Echo", None)])
    controller = EchoUIController(engine.respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)
    responses: list[str] = []
    controller.responseReady.connect(responses.append)

    controller.startVoiceStandby()

    assert _run_until(lambda: controller.isVoiceConversationActive(), timeout_ms=3000)
    assert responses == ["I'm listening."]
    _stop_and_drain(controller)


def test_voice_standby_g_short_echo_enters_active() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    transcriber = _SequentialVoiceTranscriber([("Echo", None)])
    controller = EchoUIController(_FakeEngine().respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)

    controller.startVoiceStandby()

    assert _run_until(lambda: controller.isVoiceConversationActive(), timeout_ms=3000)
    _stop_and_drain(controller)


def test_voice_standby_h_short_acorda_enters_active() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    transcriber = _SequentialVoiceTranscriber([("acorda", None)])
    controller = EchoUIController(
        _FakeEngine().respond,
        voice_available=True,
        voice_transcriber=transcriber,
        get_response_language=lambda: "pt-PT",
        thinking_min_ms=0,
    )

    controller.startVoiceStandby()

    assert _run_until(lambda: controller.isVoiceConversationActive(), timeout_ms=3000)
    _stop_and_drain(controller)


def test_voice_standby_i_non_wake_short_noise_does_not_enter_active() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Should not be used.")
    transcriber = _SequentialVoiceTranscriber([("uh", None), ("Echo", None)])
    controller = EchoUIController(engine.respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)
    responses: list[str] = []
    controller.responseReady.connect(responses.append)

    controller.startVoiceStandby()

    assert _run_until(lambda: controller.isVoiceConversationActive(), timeout_ms=5000)
    assert responses == ["I'm listening."]  # "uh" never reached the engine as content
    _stop_and_drain(controller)


def test_voice_standby_j_no_duplicate_rearm() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    controller = EchoUIController(_FakeEngine().respond, voice_available=True, voice_transcriber=_BlockingVoiceTranscriber())

    controller.startVoiceStandby()
    first_worker = controller._voice_worker
    assert first_worker is not None

    controller._rearm_voice_listening(controller._voice_session_id)

    assert controller._voice_worker is first_worker
    controller.stopVoiceStandby()
    _run_until(lambda: controller._voice_worker is None, timeout_ms=2000)


# --- Phase 7.6: barge-in -----------------------------------------------------
# The low-level VAD mechanics (discard-until-confirmed, short noise spikes,
# sustained speech) are unit-tested in tests/test_voice_input.py. These
# tests cover the controller-level wiring: the detector only runs while
# genuinely speaking in ACTIVE mode, a confirmed interruption stops
# playback and hands off into a normal capture that becomes the next turn,
# races with STOP/sleep/a stale ttsFinished are all handled safely, and no
# duplicate stop/rearm ever happens.


class _BargeInCapableTranscriber:
    """Implements the FULL record_and_transcribe_vad signature (vad_config/
    on_speech_confirmed/require_confirmation_before_recording) so
    controller-level tests can exercise _begin_barge_in_monitor's duck-
    typed capability check and the resulting stop-TTS/capture-handoff flow
    without real audio hardware. Confirms speech exactly once ever (real
    interruption), unless cancelled (stop_event set) before that point --
    mirroring how a real cancelled barge-in monitor never confirms."""

    def __init__(self, active_result: str = "ola echo", barge_in_result: str = "espera nao era isso") -> None:
        self.active_result = active_result
        self.barge_in_result = barge_in_result
        self.active_calls = 0
        self.barge_in_calls = 0
        self.confirmed_count = 0
        self._confirmed_once = False

    def record_and_transcribe_vad(
        self,
        stop_event,
        status_callback=None,
        *,
        input_language=None,
        vad_config=None,
        on_speech_confirmed=None,
        require_confirmation_before_recording=False,
    ) -> str:
        import time

        from assistant.voice_input import VoiceNoSpeechDetected

        if require_confirmation_before_recording:
            self.barge_in_calls += 1
            if not self._confirmed_once:
                for step in range(50):
                    if stop_event.is_set():
                        raise VoiceNoSpeechDetected("Nao detetei fala.")
                    if step == 4:  # confirm at ~50ms if not cancelled first
                        self._confirmed_once = True
                        self.confirmed_count += 1
                        if on_speech_confirmed is not None:
                            on_speech_confirmed()
                        stop_event.set()
                        return self.barge_in_result
                    time.sleep(0.01)
                raise VoiceNoSpeechDetected("Nao detetei fala.")
            while not stop_event.is_set():
                time.sleep(0.01)
            raise VoiceNoSpeechDetected("Nao detetei fala.")
        self.active_calls += 1
        stop_event.set()
        return self.active_result

    def record_and_transcribe(self, stop_event, status_callback=None) -> str:
        raise AssertionError("manual (legacy) path must not be used for active/barge-in workers")


def _slow_player(_path, stop_checker=None) -> float:
    import time

    for _ in range(50):
        if stop_checker and stop_checker():
            return 0.0
        time.sleep(0.01)
    return 0.0


def test_barge_in_k_detector_only_runs_while_active() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    transcriber = _BargeInCapableTranscriber()
    controller = EchoUIController(_FakeEngine().respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)

    controller._begin_barge_in_monitor()  # not in active mode -- must no-op
    assert controller._voice_worker is None

    controller._voice_conversation_active = True
    controller._begin_barge_in_monitor()
    assert controller._voice_worker is not None
    assert controller._voice_worker_is_barge_in is True
    _stop_and_drain(controller)


def test_barge_in_nrstu_confirmed_interruption_becomes_next_turn() -> None:
    """N/R/S/T/U (and implicitly O/Q): sustained speech during playback
    stops TTS, hands off into a normal VAD capture, trailing silence
    completes the interrupted utterance, it transcribes/submits exactly
    once, and the resulting response is the NEXT turn."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Reply.")
    transcriber = _BargeInCapableTranscriber(active_result="ola echo", barge_in_result="espera nao era isso")
    tts = _FakeTTSManager()
    controller = EchoUIController(
        engine.respond,
        voice_available=True,
        voice_transcriber=transcriber,
        tts_manager=tts,
        tts_player=_slow_player,
        thinking_min_ms=0,
    )
    states: list[str] = []
    responses: list[str] = []
    controller.stateChanged.connect(states.append)
    controller.responseReady.connect(responses.append)

    controller.startVoiceConversation()

    assert _run_until(lambda: "speaking" in states, timeout_ms=3000)
    assert _run_until(lambda: transcriber.confirmed_count >= 1, timeout_ms=3000)
    assert _run_until(lambda: responses.count("Reply.") >= 2, timeout_ms=8000)
    assert transcriber.confirmed_count == 1  # confirmed exactly once (P/S)
    assert transcriber.active_calls >= 1
    _stop_and_drain(controller)


def test_barge_in_t_stale_tts_finished_after_confirmed_does_not_duplicate_listener() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    transcriber = _BargeInCapableTranscriber()
    controller = EchoUIController(_FakeEngine().respond, voice_available=True, voice_transcriber=transcriber, thinking_min_ms=0)
    controller._voice_conversation_active = True
    controller._voice_session_id = 1

    # Simulate: a barge-in worker already confirmed and is now capturing.
    sentinel_worker = object()
    controller._voice_worker = sentinel_worker
    controller._voice_worker_is_barge_in = True
    controller._voice_worker_barge_in_confirmed = True
    controller._voice_worker_session_id = 1
    controller._voice_worker_mode = "active"

    # The (now redundant) ttsFinished for the interrupted playback arrives.
    controller._handle_tts_finished('{"playback_ms": 120, "voice_id": "x"}')

    # Must not have touched/replaced the still-capturing worker.
    assert controller._voice_worker is sentinel_worker
    controller._voice_conversation_active = False


def test_barge_in_v_interrupted_response_text_stays_visible() -> None:
    """V: nothing in the barge-in confirmation/interruption path clears or
    touches the already-shown response text."""
    import inspect

    from prototype_web_ui.controller import EchoUIController

    source = inspect.getsource(EchoUIController._handle_barge_in_confirmed)
    assert "responseReady" not in source
    assert "_emit_response_ready" not in source


def test_barge_in_w_no_resume_mechanism_exists() -> None:
    """W: abandoned/interrupted audio is never resumed -- there is no
    resume-style method anywhere on the TTS worker to even call."""
    from prototype_web_ui.controller import EchoTTSWorker

    assert not hasattr(EchoTTSWorker, "resume")


def test_barge_in_x_manual_stop_cancels_unconfirmed_monitor_too() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Reply.")
    transcriber = _BargeInCapableTranscriber()
    tts = _FakeTTSManager()
    controller = EchoUIController(
        engine.respond,
        voice_available=True,
        voice_transcriber=transcriber,
        tts_manager=tts,
        tts_player=_slow_player,
        thinking_min_ms=0,
    )
    states: list[str] = []
    controller.stateChanged.connect(states.append)

    controller.startVoiceConversation()
    assert _run_until(lambda: "speaking" in states, timeout_ms=3000)
    assert _run_until(lambda: transcriber.barge_in_calls >= 1, timeout_ms=2000)

    controller.stopSpeaking()  # STOP pressed before the monitor ever confirms

    assert controller.isVoiceConversationActive() is True
    assert _run_until(lambda: controller._voice_worker is not None, timeout_ms=3000)  # rearmed to a fresh listening cycle
    assert transcriber.confirmed_count == 0
    _stop_and_drain(controller)


def test_barge_in_y_sleep_said_as_interruption_exits_to_standby() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Reply.")
    transcriber = _BargeInCapableTranscriber(barge_in_result="vai dormir")
    tts = _FakeTTSManager()
    controller = EchoUIController(
        engine.respond,
        voice_available=True,
        voice_transcriber=transcriber,
        tts_manager=tts,
        tts_player=_slow_player,
        thinking_min_ms=0,
    )

    controller.startVoiceConversation()

    assert _run_until(lambda: controller.isVoiceStandbyActive(), timeout_ms=8000)
    assert controller.isVoiceConversationActive() is False
    controller.stopVoiceStandby()
    # Drain fully -- including the interrupted TTS worker's OWN eventual
    # finished signal, which the earlier assert's _run_until stopped
    # pumping for the instant isVoiceStandbyActive() went true. Without
    # this, the TTS QThread never gets a chance to actually quit and
    # orphans, hanging pytest's own teardown.
    assert _run_until(lambda: controller._voice_worker is None and controller._tts_worker is None, timeout_ms=3000)


# --- Phase 7.6.2: STOP-during-speaking -----------------------------------
# Real-mic validation showed STOP appeared to have no effect during a long
# Pocket/Anna playback -- the controller ROUTING here was already correct
# (stopSpeaking() always called _tts_worker.stop()); the actual root cause
# was assistant/tts.py's play_audio_file() ignoring the stop signal mid-
# playback (fixed + unit-tested separately in tests/test_tts.py). These
# tests re-confirm the controller-level state machine around STOP: it
# reaches the TTS worker, cancels an unconfirmed barge-in monitor, never
# exits ACTIVE mode, and rearms exactly once. B/C/D/E from section 9 are
# already covered by the existing test_barge_in_x_manual_stop_cancels_
# unconfirmed_monitor_too, test_barge_in_w_no_resume_mechanism_exists, and
# test_barge_in_t_stale_tts_finished_after_confirmed_does_not_duplicate_listener.


def test_voice_stop_a_stop_while_speaking_calls_tts_stop() -> None:
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Reply.")
    transcriber = _FakeVoiceTranscriber(result="ola echo")
    tts = _FakeTTSManager()
    controller = EchoUIController(
        engine.respond,
        voice_available=True,
        voice_transcriber=transcriber,
        tts_manager=tts,
        tts_player=_slow_player,
        thinking_min_ms=0,
    )
    states: list[str] = []
    controller.stateChanged.connect(states.append)

    controller.startVoiceConversation()
    assert _run_until(lambda: "speaking" in states, timeout_ms=3000)
    tts_worker = controller._tts_worker
    assert tts_worker is not None

    controller.stopSpeaking()

    assert tts_worker.stop_event.is_set()
    _stop_and_drain(controller)


def test_voice_stop_f_stop_while_listening_leaves_capture_untouched() -> None:
    """F: STOP called while merely LISTENING (no TTS active, no barge-in
    monitor) has no effect on the ongoing capture -- preserves existing
    semantics."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    controller = EchoUIController(_FakeEngine().respond, voice_available=True, voice_transcriber=_BlockingVoiceTranscriber())

    controller.startVoiceConversation()
    assert _run_until(lambda: controller._voice_worker is not None, timeout_ms=2000)
    listening_worker = controller._voice_worker

    controller.stopSpeaking()

    assert controller._voice_worker is listening_worker
    assert controller.isVoiceConversationActive() is True
    _stop_and_drain(controller)


def test_voice_stop_g_pocket_manager_not_torn_down() -> None:
    """G: stopSpeaking() must never call the TTS MANAGER's own teardown/
    stop -- only the per-request worker's stop_event."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Reply.")
    transcriber = _FakeVoiceTranscriber(result="ola echo")
    tts = _FakeTTSManager()
    controller = EchoUIController(
        engine.respond,
        voice_available=True,
        voice_transcriber=transcriber,
        tts_manager=tts,
        tts_player=_slow_player,
        thinking_min_ms=0,
    )
    states: list[str] = []
    controller.stateChanged.connect(states.append)

    controller.startVoiceConversation()
    assert _run_until(lambda: "speaking" in states, timeout_ms=3000)

    controller.stopSpeaking()

    assert tts.stopped is False
    _stop_and_drain(controller)


def test_voice_stop_logs_bounded_state_aware_events(capsys) -> None:
    """Section 3: [VOICE_STOP] requested/tts_stop_requested/
    returned_to_listening all fire for a STOP-during-speaking sequence."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Reply.")
    transcriber = _FakeVoiceTranscriber(result="ola echo")
    tts = _FakeTTSManager()
    controller = EchoUIController(
        engine.respond,
        voice_available=True,
        voice_transcriber=transcriber,
        tts_manager=tts,
        tts_player=_slow_player,
        thinking_min_ms=0,
    )
    states: list[str] = []
    controller.stateChanged.connect(states.append)

    controller.startVoiceConversation()
    assert _run_until(lambda: "speaking" in states, timeout_ms=3000)
    capsys.readouterr()  # discard startup noise

    controller.stopSpeaking()

    assert _run_until(lambda: controller._voice_worker is not None, timeout_ms=3000)
    captured = capsys.readouterr().out
    assert "[VOICE_STOP] event=requested" in captured and "state=speaking" in captured
    assert "[VOICE_STOP] event=tts_stop_requested" in captured
    assert "[VOICE_STOP] event=returned_to_listening" in captured
    _stop_and_drain(controller)


def test_voice_stop_h_shutdown_stops_playback_and_tears_down_manager() -> None:
    """H (section 15): shutdown() must stop in-progress playback AND
    (unlike a normal STOP) tear down the TTS manager -- teardown_manager=True
    is reserved for genuine session-ending shutdown, unlike stopSpeaking()."""
    from prototype_web_ui.controller import EchoUIController

    _qt_app()
    engine = _FakeEngine(reply="Reply.")
    transcriber = _FakeVoiceTranscriber(result="ola echo")
    tts = _FakeTTSManager()
    controller = EchoUIController(
        engine.respond,
        voice_available=True,
        voice_transcriber=transcriber,
        tts_manager=tts,
        tts_player=_slow_player,
        thinking_min_ms=0,
    )
    states: list[str] = []
    controller.stateChanged.connect(states.append)

    controller.startVoiceConversation()
    assert _run_until(lambda: "speaking" in states, timeout_ms=3000)
    tts_worker = controller._tts_worker
    assert tts_worker is not None

    controller.shutdown()

    assert tts_worker.stop_event.is_set()
    assert tts.stopped is True


def test_tts_playback_stop_latency_bounded_end_to_end(monkeypatch, capsys) -> None:
    """C (section 15): end-to-end through the REAL play_audio_file (with a
    mocked sounddevice backend) -- [TTS_PLAYBACK] event=stopped reports a
    small stop_latency_ms, proving the sounddevice-based backend
    interrupts promptly rather than running to the file's full nominal
    (here: 30s) duration."""
    import dataclasses
    import re
    import sys
    import threading as _threading
    import time as _time
    import wave

    from prototype_web_ui.controller import EchoUIController

    _qt_app()

    class _FakeCallbackStop(Exception):
        pass

    class _FakeOutputStream:
        def __init__(self, **kwargs) -> None:
            self.channels = kwargs["channels"]
            self.dtype = kwargs["dtype"]
            self.callback = kwargs["callback"]
            self.finished_callback = kwargs.get("finished_callback")
            self.aborted = False

        def start(self) -> None:
            def feed() -> None:
                import numpy as np

                chunk_frames = 64
                while not self.aborted:
                    outdata = np.zeros((chunk_frames, self.channels), dtype=self.dtype)
                    try:
                        self.callback(outdata, chunk_frames, None, None)
                    except _FakeCallbackStop:
                        break
                    _time.sleep(0.01)
                if self.finished_callback is not None:
                    self.finished_callback()

            self._thread = _threading.Thread(target=feed, daemon=True)
            self._thread.start()

        def abort(self) -> None:
            self.aborted = True

        def close(self) -> None:
            pass

        def __enter__(self) -> "_FakeOutputStream":
            self.start()
            return self

        def __exit__(self, *exc_info: object) -> None:
            self.close()

    class _FakeSd:
        CallbackStop = _FakeCallbackStop

        def OutputStream(self, **kwargs) -> _FakeOutputStream:
            return _FakeOutputStream(**kwargs)

    monkeypatch.setitem(sys.modules, "sounddevice", _FakeSd())

    import tempfile

    wav_path = Path(tempfile.NamedTemporaryFile(prefix="echo_test_playback_", suffix=".wav", delete=False).name)
    with wave.open(str(wav_path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(8000)
        f.writeframes(b"\x00\x00" * (8000 * 30))  # 30s nominal duration

    engine = _FakeEngine(reply="Reply.")
    transcriber = _FakeVoiceTranscriber(result="ola echo")
    tts = _FakeTTSManager()
    tts.result = dataclasses.replace(tts.result, audio_path=wav_path)
    controller = EchoUIController(
        engine.respond,
        voice_available=True,
        voice_transcriber=transcriber,
        voice_missing_dependencies=[],  # avoid check_voice_runtime() probing the faked sounddevice module
        tts_manager=tts,
        thinking_min_ms=0,
    )
    states: list[str] = []
    controller.stateChanged.connect(states.append)

    try:
        controller.startVoiceConversation()
        assert _run_until(lambda: "speaking" in states, timeout_ms=3000)
        capsys.readouterr()

        controller.stopSpeaking()

        assert _run_until(lambda: controller._voice_worker is not None, timeout_ms=3000)
        captured = capsys.readouterr().out
        match = re.search(r"\[TTS_PLAYBACK\] event=stopped.*stop_latency_ms=(\d+)", captured)
        assert match is not None, captured
        assert int(match.group(1)) < 500
        _stop_and_drain(controller)
    finally:
        wav_path.unlink(missing_ok=True)

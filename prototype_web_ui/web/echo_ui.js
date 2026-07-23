(function () {
  let controller = null;
  let controllerReady = false;
  let requestActive = false;
  let replyFadeToken = 0;
  let telemetryPanelOpen = false;
  let telemetryMode = "local";
  let telemetryState = "idle_local";
  let telemetryRoutingTimer = null;
  let telemetryDebugMode = false;
  let latestRealTelemetry = null;
  let activeRuntimeState = null;
  let configurationError = null;

  const APP_VERSION = "v0.4.0";

  const stateLabels = {
    active: ["ACTIVE", "#8fd0c4"],
    idle: ["IDLE", "#5a5d6b"],
    thinking: ["THINKING", "#6ea8ff"],
    speaking: ["SPEAKING", "#8fd0c4"],
    error: ["ERROR", "#f08a8a"]
  };

  const routingLabels = {
    professional_writing: "Professional writing",
    structured_summary: "Structured summary",
    technical_explanation: "Technical explanation",
    low_complexity: "Local response",
    memory_recall: "Memory recall",
    project_memory_recall: "Project memory",
    local_mode: "Local mode",
    claude_mode: "Claude mode",
    low_complexity: "Local response",
    automatic_claude_disabled: "Cloud routing disabled",
    paid_calls_disabled: "Cloud disabled",
    paid_calls_not_confirmed: "Cloud disabled",
    missing_api_key: "API key required",
    daily_budget_exceeded: "Daily budget reached",
    single_call_limit_exceeded: "Call limit reached",
    single_call_budget_exceeded: "Call limit reached",
    budget_state_unavailable: "Budget unavailable",
    tool_result: "Local tool",
    local_tool: "Local tool"
  };

  const telemetryMocks = {
    idle_local: {
      compact: "LOCAL · LLAMA · 0 COST",
      mode: "LOCAL",
      model: "LLAMA 3.1 8B",
      routing: routingLabels.low_complexity,
      latency: "620 ms",
      tokens: "430 → 72",
      cost: "$0",
      daily: "$0.000 / $0.25",
      note: "Stable local response.",
      behaviour: "local"
    },
    thinking_local: {
      compact: "ANALYSING LOCALLY",
      mode: "LOCAL",
      model: "LLAMA 3.1 8B",
      routing: routingLabels.low_complexity,
      latency: "analysing",
      tokens: "estimating",
      cost: "$0",
      daily: "$0.000 / $0.25",
      note: "Local model is handling this turn.",
      behaviour: "local"
    },
    routing_automatic: {
      compact: "ROUTING REQUEST",
      mode: "AUTOMATIC",
      model: "SELECTING",
      routing: "Analysing complexity",
      latency: "routing",
      tokens: "pending",
      cost: "pending",
      daily: "$0.015 / $0.25",
      note: "Echo selects the model.",
      behaviour: "automatic"
    },
    thinking_cloud: {
      compact: "CLOUD MODEL SELECTED",
      mode: "AUTOMATIC",
      model: "CLAUDE HAIKU",
      routing: routingLabels.structured_summary,
      latency: "thinking",
      tokens: "1309 → ...",
      cost: "$0.0019 est.",
      daily: "$0.015 / $0.25",
      note: "Cloud model selected for this mock turn.",
      behaviour: "claude"
    },
    response_ready: {
      compact: "AUTO · CLAUDE · 1.8s",
      mode: "AUTOMATIC",
      model: "CLAUDE HAIKU",
      routing: routingLabels.structured_summary,
      latency: "1.83 s",
      tokens: "1309 → 111",
      cost: "$0.0019",
      daily: "$0.015 / $0.25",
      note: "Response ready.",
      behaviour: "automatic"
    }
  };

  function byId(id) {
    return document.getElementById(id);
  }

  function scaleStage() {
    const scaler = byId("stageScaler");
    if (!scaler) {
      console.error("[Echo UI JS] #stageScaler nao encontrado");
      return;
    }

    const margin = 24;
    const scale = Math.min(
      (window.innerWidth - margin * 2) / 1300,
      (window.innerHeight - margin * 2) / 812
    );
    const cleanScale = Math.max(0.32, scale);
    scaler.style.transform = `translate(-50%, -50%) scale(${cleanScale})`;
  }

  function setInputEnabled(enabled) {
    const input = byId("voiceIn");
    const form = byId("echoForm");
    if (!input || !form) {
      console.error("[Echo UI JS] input/form indisponivel");
      return;
    }
    input.disabled = !enabled;
    form.classList.toggle("busy", !enabled);
  }

  function updateStatusLabel(state) {
    const stage = document.querySelector(".stage");
    const statusLine = byId("statusLine");
    const label = stateLabels[state] || stateLabels.error;

    if (!stage || !statusLine) {
      console.error("[Echo UI JS] stage/statusLine indisponivel");
      return;
    }

    stage.dataset.state = state;
    statusLine.textContent = label[0];
    statusLine.style.color = label[1];
  }

  function telemetryElements() {
    return {
      stage: document.querySelector(".stage"),
      compact: byId("telemetryCompact"),
      compactText: byId("telemetryCompactText"),
      panel: byId("telemetryPanel"),
      mode: byId("telemetryMode"),
      model: byId("telemetryModel"),
      routing: byId("telemetryRouting"),
      latency: byId("telemetryLatency"),
      tokens: byId("telemetryTokens"),
      cost: byId("telemetryCost"),
      daily: byId("telemetryDaily"),
      source: byId("telemetrySource"),
      fallback: byId("telemetryFallback"),
      fallbackRow: document.querySelector(".telemetry-fallback-row"),
      note: byId("telemetryNote"),
      configError: byId("modelConfigError"),
      configureClaude: byId("configureClaude"),
      keyPanel: byId("anthropicKeyPanel"),
      keyInput: byId("anthropicApiKey"),
      keyStatus: byId("anthropicKeyStatus"),
      saveAnthropicKey: byId("saveAnthropicKey"),
      removeAnthropicKey: byId("removeAnthropicKey"),
      testAnthropicKey: byId("testAnthropicKey"),
      autoClaudeButtons: Array.from(document.querySelectorAll("[data-auto-claude]")),
      automaticClaudeStatus: byId("automaticClaudeStatus"),
      dailyBudgetUsd: byId("dailyBudgetUsd"),
      singleCallBudgetUsd: byId("singleCallBudgetUsd"),
      systemOllama: byId("systemOllama"),
      systemClaude: byId("systemClaude"),
      systemObserver: byId("systemObserver"),
      systemCliOverride: byId("systemCliOverride"),
      systemSettingsSource: byId("systemSettingsSource"),
      systemModelModeSource: byId("systemModelModeSource"),
      systemAutomaticClaudeSource: byId("systemAutomaticClaudeSource"),
      systemApiKeySource: byId("systemApiKeySource")
    };
  }

  function buildSystemCommand(intent, parameters = {}) {
    return JSON.stringify({intent, parameters, source: "ui"});
  }

  function executeSystemCommand(intent, parameters = {}, callback = applyRealTelemetry) {
    if (!controller || typeof controller.executeSystemCommand !== "function") return;
    controller.executeSystemCommand(buildSystemCommand(intent, parameters), (payload) => {
      if (callback) callback(payload);
    });
  }

  function humanReasonLabel(code) {
    const value = String(code || "").trim();
    if (!value) return routingLabels.low_complexity;
    return routingLabels[value] || value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function modelLabel(provider, model) {
    const cleanModel = String(model || "").trim();
    const cleanProvider = String(provider || "").trim().toLowerCase();
    if (!cleanModel || cleanModel === "NONE") return "NONE";
    if (cleanProvider === "anthropic") return cleanModel.includes("haiku") ? "CLAUDE HAIKU" : "CLAUDE";
    if (cleanProvider === "ollama") return cleanModel.includes("llama") ? "LLAMA 3.1 8B" : cleanModel.toUpperCase();
    if (cleanProvider === "memory") return "NONE";
    if (cleanProvider === "local_tool") return "NONE";
    return cleanModel.toUpperCase();
  }

  function modeLabel(mode) {
    const value = String(mode || "local").toLowerCase();
    if (value === "claude") return "CLAUDE";
    if (value === "automatic") return "AUTOMATIC";
    return "LOCAL";
  }

  function money(value, digits = 4) {
    const number = Number(value || 0);
    if (!Number.isFinite(number) || number <= 0) return "$0";
    return `$${number.toFixed(digits)}`;
  }

  function latencyLabel(ms) {
    const value = Number(ms || 0);
    if (!Number.isFinite(value) || value <= 0) return "—";
    if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
    return `${Math.round(value)} ms`;
  }

  function compactTelemetryLabel(data) {
    const state = String(data.state || "").toLowerCase();
    const provider = String(data.provider || "").toLowerCase();
    const mode = String(data.mode || "local").toLowerCase();
    const reason = String(data.reason_code || "").toLowerCase();
    if (state === "error") return `ERROR · ${humanReasonLabel(data.provider_error_type || reason).toUpperCase()}`;
    if (reason.includes("memory") || provider === "memory") return `MEMORY · ${latencyLabel(data.latency_ms)} · 0 COST`;
    if (provider === "local_tool" || reason === "local_tool") return `LOCAL TOOL · ${latencyLabel(data.latency_ms)}`;
    if (state === "routing_automatic") return "ROUTING REQUEST";
    if (state === "thinking_cloud" || provider === "anthropic") return `${mode === "automatic" ? "AUTO" : "CLAUDE"} · CLAUDE · ${latencyLabel(data.latency_ms)}`;
    if (state === "thinking_local") return "ANALYSING LOCALLY";
    return `${modeLabel(mode)} · LLAMA · ${latencyLabel(data.latency_ms)}`;
  }

  function parseTelemetryPayload(payload) {
    if (!payload) return null;
    if (typeof payload === "object") return payload;
    try {
      return JSON.parse(String(payload));
    } catch (error) {
      console.error("[Echo UI JS] telemetry JSON invalido", error);
      return null;
    }
  }

  function updateTelemetryButtons() {
    document.querySelectorAll("[data-telemetry-mode]").forEach((button) => {
      button.classList.toggle("active", button.dataset.telemetryMode === telemetryMode);
    });
  }

  function applyTelemetryMock(stateName) {
    if (!telemetryDebugMode) return;
    const name = telemetryMocks[stateName] ? stateName : "idle_local";
    const data = telemetryMocks[name];
    const elements = telemetryElements();

    telemetryState = name;
    telemetryMode = data.behaviour;
    if (elements.stage) {
      elements.stage.dataset.telemetryMode = data.behaviour;
      elements.stage.dataset.telemetryState = name;
    }
    if (elements.compact) elements.compact.classList.add("visible");
    if (elements.compactText) elements.compactText.textContent = data.compact;
    if (elements.mode) elements.mode.textContent = data.mode;
    if (elements.model) elements.model.textContent = data.model;
    if (elements.routing) elements.routing.textContent = data.routing;
    if (elements.latency) elements.latency.textContent = data.latency;
    if (elements.tokens) elements.tokens.textContent = data.tokens;
    if (elements.cost) elements.cost.textContent = data.cost;
    if (elements.daily) elements.daily.textContent = data.daily;
    if (elements.note) elements.note.textContent = `DEMO · ${data.note}`;
    updateTelemetryButtons();
  }

  function applyRealTelemetry(payload) {
    const data = parseTelemetryPayload(payload);
    if (!data || telemetryDebugMode) return;
    latestRealTelemetry = data;
    if (String(data.state || "") === "error") {
      configurationError = data;
      renderConfigurationError(data);
      updateModelConfigState(data);
      updateSystemFields(data);
      return;
    }
    activeRuntimeState = data;
    configurationError = null;
    const elements = telemetryElements();
    const state = String(data.state || "idle_local");
    const mode = String(data.mode || "local").toLowerCase();
    const provider = String(data.provider || "").toLowerCase();
    const reasonCode = String(data.reason_code || "");
    const reason = String(data.reason_label || humanReasonLabel(reasonCode));

    telemetryMode = mode === "claude" ? "claude" : mode === "automatic" ? "automatic" : "local";
    telemetryState = state;
    if (elements.stage) {
      elements.stage.dataset.telemetryMode = telemetryMode;
      elements.stage.dataset.telemetryState = state;
    }
    if (elements.compact) elements.compact.classList.add("visible");
    if (elements.compactText) elements.compactText.textContent = compactTelemetryLabel(data);
    if (elements.mode) elements.mode.textContent = modeLabel(mode);
    if (elements.model) elements.model.textContent = Number(data.llm_calls || 0) > 0 ? modelLabel(provider, data.model) : "NONE";
    if (elements.routing) elements.routing.textContent = reason;
    if (elements.latency) elements.latency.textContent = latencyLabel(data.latency_ms);
    if (elements.tokens) {
      const inputTokens = data.input_tokens;
      const outputTokens = data.output_tokens;
      elements.tokens.textContent = inputTokens || outputTokens ? `${inputTokens || 0} → ${outputTokens || 0}` : "—";
    }
    if (elements.cost) elements.cost.textContent = money(data.estimated_cost_usd);
    if (elements.daily) elements.daily.textContent = `${money(data.daily_used_usd, 3)} / ${money(data.daily_budget_usd || 0, 2)}`;
    if (elements.source) elements.source.textContent = String(data.source || data.response_source || "NONE").toUpperCase();
    if (elements.fallback && elements.fallbackRow) {
      const fallback = String(data.fallback_reason || "");
      elements.fallback.textContent = fallback ? humanReasonLabel(fallback).toUpperCase() : "—";
      elements.fallbackRow.hidden = !fallback;
    }
    if (elements.note) {
      const fallback = data.fallback_reason ? `Fallback: ${humanReasonLabel(data.fallback_reason)}.` : "";
      const lock = data.mode_locked ? "Modo bloqueado por CLI." : "";
      elements.note.textContent = String(data.note || fallback || lock || reason || "");
    }
    renderConfigurationError(null);
    updateSystemFields(data);
    updateModelConfigState(data);
    if (elements.dailyBudgetUsd && data.daily_budget_usd !== undefined) elements.dailyBudgetUsd.value = Number(data.daily_budget_usd || 0).toFixed(2);
    if (elements.singleCallBudgetUsd && data.max_single_call_estimated_usd !== undefined) {
      elements.singleCallBudgetUsd.value = Number(data.max_single_call_estimated_usd || 0).toFixed(2);
    }
    updateTelemetryButtons();
  }

  function updateModelConfigState(data) {
    const elements = telemetryElements();
    const autoEnabled = Boolean(data.automatic_claude_enabled);
    const keyConfigured = Boolean(data.api_key_configured);
    const keySource = String(data.api_key_source || "none");
    const storageAvailable = Boolean(data.secret_storage_available);
    const paidCallsEnabled = Boolean(data.paid_calls_enabled);

    elements.autoClaudeButtons.forEach((button) => {
      const enabled = button.dataset.autoClaude === "true";
      button.classList.toggle("active", enabled === autoEnabled);
      button.setAttribute("aria-pressed", enabled === autoEnabled ? "true" : "false");
    });

    if (elements.automaticClaudeStatus) {
      let label = "Local model only";
      if (!storageAvailable && keySource !== "environment") label = "SECURE STORAGE UNAVAILABLE";
      else if (autoEnabled && !keyConfigured) label = "Cloud unavailable — API key required";
      else if (autoEnabled && keyConfigured && !paidCallsEnabled) label = "Paid calls disabled";
      else if (autoEnabled && keyConfigured && paidCallsEnabled) label = "Cloud routing available";
      elements.automaticClaudeStatus.textContent = label;
    }

    if (elements.keyPanel) elements.keyPanel.hidden = false;
    if (elements.keyInput) {
      elements.keyInput.value = "";
      elements.keyInput.hidden = !storageAvailable || keyConfigured;
      elements.keyInput.disabled = !storageAvailable || keySource === "environment";
    }
    if (elements.saveAnthropicKey) {
      elements.saveAnthropicKey.hidden = !storageAvailable || keyConfigured || keySource === "environment";
      elements.saveAnthropicKey.disabled = !storageAvailable || keyConfigured || keySource === "environment";
    }
    if (elements.removeAnthropicKey) {
      elements.removeAnthropicKey.hidden = !storageAvailable || !keyConfigured || keySource === "environment";
      elements.removeAnthropicKey.disabled = !storageAvailable || !keyConfigured || keySource === "environment";
    }
    if (elements.testAnthropicKey) {
      elements.testAnthropicKey.disabled = !storageAvailable && keySource !== "environment";
    }
    if (elements.keyStatus) {
      if (keySource === "environment") elements.keyStatus.textContent = "CONFIGURED BY ENVIRONMENT";
      else if (!storageAvailable) elements.keyStatus.textContent = "SECURE STORAGE UNAVAILABLE";
      else if (keyConfigured) elements.keyStatus.textContent = "CONFIGURED SECURELY";
      else elements.keyStatus.textContent = "API key not configured.";
    }
  }

  function renderConfigurationError(data) {
    const elements = telemetryElements();
    const hasError = Boolean(data && data.provider_error_type);
    if (elements.configError) {
      elements.configError.hidden = !hasError;
      elements.configError.textContent = hasError ? String(data.note || humanReasonLabel(data.provider_error_type)).toUpperCase() : "";
    }
    if (elements.configureClaude) elements.configureClaude.hidden = !(hasError && String(data.provider_error_type) === "missing_api_key");
    if (elements.note && hasError) elements.note.textContent = "O modo ativo não foi alterado.";
  }

  function updateSystemFields(data) {
    const elements = telemetryElements();
    if (elements.systemOllama) elements.systemOllama.textContent = "AVAILABLE";
    if (elements.systemClaude) elements.systemClaude.textContent = data.api_key_configured ? "CONFIGURED" : "NOT CONFIGURED";
    if (elements.systemObserver) elements.systemObserver.textContent = String(data.context_observer_state || "UNKNOWN").toUpperCase();
    if (elements.systemCliOverride) elements.systemCliOverride.textContent = data.mode_locked ? "YES" : "NO";
    if (elements.systemSettingsSource) elements.systemSettingsSource.textContent = String(data.settings_source || "settings.json");
    if (elements.systemModelModeSource) elements.systemModelModeSource.textContent = String(data.model_routing_mode_source || "default");
    if (elements.systemAutomaticClaudeSource) elements.systemAutomaticClaudeSource.textContent = String(data.automatic_claude_enabled_source || "default");
    if (elements.systemApiKeySource) elements.systemApiKeySource.textContent = String(data.api_key_source || "none");
  }

  function setTelemetryPanel(open) {
    const elements = telemetryElements();
    telemetryPanelOpen = Boolean(open);
    if (elements.panel) {
      elements.panel.classList.toggle("open", telemetryPanelOpen);
      elements.panel.setAttribute("aria-hidden", telemetryPanelOpen ? "false" : "true");
      if (telemetryPanelOpen) elements.panel.focus({preventScroll: true});
    }
    if (elements.stage) {
      elements.stage.dataset.telemetryPanel = telemetryPanelOpen ? "open" : "closed";
    }
    if (elements.compact) {
      elements.compact.setAttribute("aria-expanded", telemetryPanelOpen ? "true" : "false");
      if (!telemetryPanelOpen) elements.compact.focus({preventScroll: true});
    }
    if (window.echoFreeWorkspace && typeof window.echoFreeWorkspace.saveWorkspaceState === "function") {
      window.echoFreeWorkspace.saveWorkspaceState();
    }
  }

  function openPanelSection(sectionName) {
    const section = String(sectionName || "status").toLowerCase();
    document.querySelectorAll("[data-panel-section]").forEach((button) => {
      const active = button.dataset.panelSection === section;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", active ? "true" : "false");
    });
    document.querySelectorAll("[data-panel-page]").forEach((page) => {
      page.classList.toggle("active", page.dataset.panelPage === section);
    });
  }

  function runAutomaticRoutingMock() {
    if (!telemetryDebugMode) return;
    window.clearTimeout(telemetryRoutingTimer);
    applyTelemetryMock("routing_automatic");
    telemetryRoutingTimer = window.setTimeout(() => applyTelemetryMock("thinking_cloud"), 650);
    telemetryRoutingTimer = window.setTimeout(() => applyTelemetryMock("response_ready"), 1250);
  }

  function setTelemetryMode(mode) {
    const value = String(mode || "").trim().toLowerCase();
    if (!telemetryDebugMode) {
      executeSystemCommand("set_model_mode", {mode: value});
      return;
    }
    if (value === "local") {
      applyTelemetryMock("idle_local");
    } else if (value === "claude") {
      applyTelemetryMock("thinking_cloud");
    } else if (value === "automatic") {
      runAutomaticRoutingMock();
    }
  }

  const WORKSPACE_STORAGE_KEY = "echo_os.free_workspace.v1";
  const STAGE_WIDTH = 1300;
  const STAGE_HEIGHT = 812;
  const HEADER_SAFE_HEIGHT = 62;
  const SNAP_THRESHOLD = 70;
  const ECHO_ITEM_SIZE = {width: 360, height: 360};
  const PANEL_ITEM_SIZE = {width: 318, height: 350};

  const SNAP_ZONES = {
    echo: {
      center: {x: 650, y: 372},
      "top-center": {x: 650, y: 196},
      "upper-left": {x: 360, y: 220},
      "upper-right": {x: 940, y: 220},
      "center-left": {x: 360, y: 390},
      "center-right": {x: 940, y: 390}
    },
    telemetryPanel: {
      "upper-left": {x: 56, y: 92},
      "upper-right": {x: 926, y: 92},
      "center-left": {x: 56, y: 236},
      "center-right": {x: 926, y: 236}
    }
  };

  const SNAP_ZONE_ORDER = {
    echo: ["upper-left", "top-center", "upper-right", "center-right", "center", "center-left"],
    telemetryPanel: ["upper-left", "upper-right", "center-right", "center-left"]
  };

  function createWorkspaceItem({id, type, x, y, width, height, snapZone = null, locked = false, zIndex = 1}) {
    return {id, type, x, y, width, height, snapZone, locked, zIndex};
  }

  function createFreeWorkspaceController() {
    const stage = document.querySelector(".stage");
    const mind = byId("mind");
    const panel = byId("telemetryPanel");
    const panelDragHandle = document.querySelector('[data-workspace-drag-handle="telemetryPanel"]');
    const recenterButton = byId("recenterEcho");
    const resetButton = byId("resetWorkspace");
    const lockButton = byId("toggleWorkspaceLock");

    const defaults = {
      locked: false,
      nextZIndex: 12,
      telemetryPanelOpen: false,
      items: {
        echo: createWorkspaceItem({
          id: "echo",
          type: "echo",
          x: SNAP_ZONES.echo.center.x,
          y: SNAP_ZONES.echo.center.y,
          width: ECHO_ITEM_SIZE.width,
          height: ECHO_ITEM_SIZE.height,
          snapZone: "center",
          zIndex: 1
        }),
        telemetryPanel: createWorkspaceItem({
          id: "telemetryPanel",
          type: "panel",
          x: SNAP_ZONES.telemetryPanel["upper-left"].x,
          y: SNAP_ZONES.telemetryPanel["upper-left"].y,
          width: PANEL_ITEM_SIZE.width,
          height: PANEL_ITEM_SIZE.height,
          snapZone: "upper-left",
          zIndex: 9
        })
      }
    };

    let state = loadWorkspaceState();
    let activeDrag = null;
    let resizeFrame = 0;

    function cloneItem(item) {
      return {...item};
    }

    function loadWorkspaceState() {
      try {
        const raw = window.localStorage ? window.localStorage.getItem(WORKSPACE_STORAGE_KEY) : "";
        if (!raw) return {
          locked: defaults.locked,
          nextZIndex: defaults.nextZIndex,
          telemetryPanelOpen: defaults.telemetryPanelOpen,
          items: {
            echo: cloneItem(defaults.items.echo),
            telemetryPanel: cloneItem(defaults.items.telemetryPanel)
          }
        };
        const parsed = JSON.parse(raw);
        return {
          locked: Boolean(parsed.locked),
          nextZIndex: Number(parsed.nextZIndex) || defaults.nextZIndex,
          telemetryPanelOpen: Boolean(parsed.telemetryPanelOpen),
          items: {
            echo: {...cloneItem(defaults.items.echo), ...(parsed.items && parsed.items.echo ? parsed.items.echo : {})},
            telemetryPanel: {
              ...cloneItem(defaults.items.telemetryPanel),
              ...(parsed.items && parsed.items.telemetryPanel ? parsed.items.telemetryPanel : {})
            }
          }
        };
      } catch (error) {
        console.warn("[Echo UI JS] workspace state invalido", error);
        return {
          locked: defaults.locked,
          nextZIndex: defaults.nextZIndex,
          telemetryPanelOpen: defaults.telemetryPanelOpen,
          items: {
            echo: cloneItem(defaults.items.echo),
            telemetryPanel: cloneItem(defaults.items.telemetryPanel)
          }
        };
      }
    }

    function saveWorkspaceState() {
      if (!window.localStorage) return;
      const safeState = {
        locked: state.locked,
        nextZIndex: state.nextZIndex,
        telemetryPanelOpen,
        items: {
          echo: state.items.echo,
          telemetryPanel: state.items.telemetryPanel
        }
      };
      window.localStorage.setItem(WORKSPACE_STORAGE_KEY, JSON.stringify(safeState));
    }

    function stagePointFromEvent(event) {
      if (!stage) return {x: 0, y: 0};
      const rect = stage.getBoundingClientRect();
      const scale = rect.width ? STAGE_WIDTH / rect.width : 1;
      return {
        x: (event.clientX - rect.left) * scale,
        y: (event.clientY - rect.top) * scale
      };
    }

    function rectsIntersect(a, b) {
      return a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;
    }

    function itemRect(item) {
      if (item.type === "echo") {
        return {
          left: item.x - item.width / 2,
          top: item.y - item.height / 2,
          right: item.x + item.width / 2,
          bottom: item.y + item.height / 2,
          width: item.width,
          height: item.height
        };
      }
      return {
        left: item.x,
        top: item.y,
        right: item.x + item.width,
        bottom: item.y + item.height,
        width: item.width,
        height: item.height
      };
    }

    function inputSafeRect() {
      const form = byId("echoForm");
      if (!stage || !form) return {left: 350, top: 708, right: 950, bottom: 790};
      const stageRect = stage.getBoundingClientRect();
      const formRect = form.getBoundingClientRect();
      const scale = stageRect.width ? STAGE_WIDTH / stageRect.width : 1;
      return {
        left: (formRect.left - stageRect.left) * scale - 16,
        top: (formRect.top - stageRect.top) * scale - 14,
        right: (formRect.right - stageRect.left) * scale + 16,
        bottom: (formRect.bottom - stageRect.top) * scale + 14
      };
    }

    function clampNumber(value, min, max) {
      return Math.min(Math.max(value, min), max);
    }

    function clampItemToStage(item) {
      const inputRect = inputSafeRect();
      if (item.type === "echo") {
        item.x = clampNumber(item.x, item.width / 2, STAGE_WIDTH - item.width / 2);
        item.y = clampNumber(item.y, HEADER_SAFE_HEIGHT + item.height / 2, STAGE_HEIGHT - item.height / 2);
        if (rectsIntersect(itemRect(item), inputRect)) {
          item.y = Math.min(item.y, inputRect.top - item.height / 2 - 14);
          item.y = clampNumber(item.y, HEADER_SAFE_HEIGHT + item.height / 2, STAGE_HEIGHT - item.height / 2);
        }
      } else {
        item.x = clampNumber(item.x, 12, STAGE_WIDTH - item.width - 12);
        item.y = clampNumber(item.y, HEADER_SAFE_HEIGHT + 12, STAGE_HEIGHT - item.height - 12);
        if (rectsIntersect(itemRect(item), inputRect)) {
          item.y = Math.min(item.y, inputRect.top - item.height - 12);
          item.y = clampNumber(item.y, HEADER_SAFE_HEIGHT + 12, STAGE_HEIGHT - item.height - 12);
        }
      }
      return item;
    }

    function findNearestSnapZone(item) {
      const zones = SNAP_ZONES[item.id] || {};
      let nearest = null;
      let nearestDistance = Number.POSITIVE_INFINITY;
      for (const [name, point] of Object.entries(zones)) {
        const dx = item.x - point.x;
        const dy = item.y - point.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance < nearestDistance) {
          nearest = name;
          nearestDistance = distance;
        }
      }
      return nearestDistance <= SNAP_THRESHOLD ? nearest : null;
    }

    function snapItemIfClose(item) {
      const zoneName = findNearestSnapZone(item);
      if (!zoneName) {
        item.snapZone = null;
        return;
      }
      const point = SNAP_ZONES[item.id][zoneName];
      item.x = point.x;
      item.y = point.y;
      item.snapZone = zoneName;
      clampItemToStage(item);
    }

    function bringToFront(item) {
      state.nextZIndex += 1;
      item.zIndex = state.nextZIndex;
      applyWorkspaceItem(item, false);
    }

    function applyWorkspaceItem(item, immediate = false) {
      clampItemToStage(item);
      if (item.id === "echo" && window.echoEntity && typeof window.echoEntity.setCenter === "function") {
        window.echoEntity.setCenter(item.x, item.y, immediate);
        if (mind) {
          mind.style.zIndex = String(Math.min(3, Math.max(1, item.zIndex)));
          mind.classList.toggle("workspace-locked", state.locked || item.locked);
        }
      } else if (item.id === "telemetryPanel" && panel) {
        panel.style.setProperty("--workspace-x", `${item.x}px`);
        panel.style.setProperty("--workspace-y", `${item.y}px`);
        panel.style.zIndex = String(Math.max(9, item.zIndex));
        panel.classList.toggle("workspace-locked", state.locked || item.locked);
      }
    }

    function applyAllWorkspaceItems(immediate = false) {
      applyWorkspaceItem(state.items.echo, immediate);
      applyWorkspaceItem(state.items.telemetryPanel, immediate);
      updateWorkspaceStatus();
      if (lockButton) {
        lockButton.setAttribute("aria-pressed", state.locked ? "true" : "false");
        lockButton.textContent = state.locked ? "UNLOCK POSITION" : "LOCK POSITION";
      }
    }

    function updateWorkspaceStatus() {
      const echoSnap = byId("echoSnapStatus");
      const panelSnap = byId("panelSnapStatus");
      if (echoSnap) echoSnap.textContent = String(state.items.echo.snapZone || "FREE").toUpperCase();
      if (panelSnap) panelSnap.textContent = String(state.items.telemetryPanel.snapZone || "FREE").toUpperCase();
    }

    function startDrag(event, itemId) {
      if (state.locked) return;
      const item = state.items[itemId];
      if (!item || item.locked) return;
      const target = itemId === "echo" ? mind : panel;
      if (!target) return;
      const point = stagePointFromEvent(event);
      activeDrag = {
        itemId,
        pointerId: event.pointerId,
        startPoint: point,
        startItem: cloneItem(item)
      };
      bringToFront(item);
      target.setPointerCapture(event.pointerId);
      target.classList.add("workspace-dragging", "dragging");
      document.body.classList.add("workspace-dragging");
      event.preventDefault();
      event.stopPropagation();
    }

    function updateDrag(event) {
      if (!activeDrag || event.pointerId !== activeDrag.pointerId) return;
      const item = state.items[activeDrag.itemId];
      const point = stagePointFromEvent(event);
      item.x = activeDrag.startItem.x + point.x - activeDrag.startPoint.x;
      item.y = activeDrag.startItem.y + point.y - activeDrag.startPoint.y;
      item.snapZone = null;
      applyWorkspaceItem(item, false);
      updateWorkspaceStatus();
      event.preventDefault();
    }

    function finishDrag(event) {
      if (!activeDrag || event.pointerId !== activeDrag.pointerId) return;
      const item = state.items[activeDrag.itemId];
      const target = activeDrag.itemId === "echo" ? mind : panel;
      snapItemIfClose(item);
      applyWorkspaceItem(item, false);
      if (target) {
        target.classList.remove("workspace-dragging", "dragging");
        try {
          target.releasePointerCapture(event.pointerId);
        } catch (error) {
          console.warn("[Echo UI JS] pointer release falhou", error);
        }
      }
      document.body.classList.remove("workspace-dragging");
      activeDrag = null;
      updateWorkspaceStatus();
      saveWorkspaceState();
      event.preventDefault();
    }

    function cancelDrag() {
      if (!activeDrag) return;
      const item = state.items[activeDrag.itemId];
      const target = activeDrag.itemId === "echo" ? mind : panel;
      state.items[activeDrag.itemId] = activeDrag.startItem;
      applyWorkspaceItem(state.items[activeDrag.itemId], true);
      if (target) target.classList.remove("workspace-dragging", "dragging");
      document.body.classList.remove("workspace-dragging");
      activeDrag = null;
    }

    function moveItemToSnap(itemId, zoneName) {
      const item = state.items[itemId];
      const point = SNAP_ZONES[itemId] && SNAP_ZONES[itemId][zoneName];
      if (!item || !point) return;
      item.x = point.x;
      item.y = point.y;
      item.snapZone = zoneName;
      bringToFront(item);
      applyWorkspaceItem(item, false);
      updateWorkspaceStatus();
      saveWorkspaceState();
    }

    function moveItemByKeyboard(itemId, direction) {
      const order = SNAP_ZONE_ORDER[itemId] || [];
      const item = state.items[itemId];
      if (!item || !order.length || state.locked) return;
      const current = item.snapZone && order.includes(item.snapZone) ? order.indexOf(item.snapZone) : order.indexOf("center");
      const delta = direction === "ArrowRight" || direction === "ArrowDown" ? 1 : -1;
      const nextIndex = (Math.max(current, 0) + delta + order.length) % order.length;
      moveItemToSnap(itemId, order[nextIndex]);
    }

    function resetWorkspace() {
      state = {
        locked: false,
        nextZIndex: defaults.nextZIndex,
        telemetryPanelOpen,
        items: {
          echo: cloneItem(defaults.items.echo),
          telemetryPanel: cloneItem(defaults.items.telemetryPanel)
        }
      };
      applyAllWorkspaceItems(false);
      saveWorkspaceState();
    }

    function recenterEcho() {
      moveItemToSnap("echo", "center");
    }

    function toggleLock() {
      state.locked = !state.locked;
      applyAllWorkspaceItems(false);
      saveWorkspaceState();
    }

    function clampAfterResize() {
      window.cancelAnimationFrame(resizeFrame);
      resizeFrame = window.requestAnimationFrame(() => {
        applyAllWorkspaceItems(false);
        saveWorkspaceState();
      });
    }

    function bind() {
      if (mind) {
        mind.addEventListener("pointerdown", (event) => startDrag(event, "echo"));
        mind.addEventListener("pointermove", updateDrag);
        mind.addEventListener("pointerup", finishDrag);
        mind.addEventListener("pointercancel", cancelDrag);
        mind.addEventListener("dblclick", recenterEcho);
        mind.addEventListener("keydown", (event) => {
          if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) return;
          event.preventDefault();
          moveItemByKeyboard("echo", event.key);
        });
      }

      if (panel && panelDragHandle) {
        panelDragHandle.addEventListener("pointerdown", (event) => {
          if (event.target.closest("button")) return;
          startDrag(event, "telemetryPanel");
        });
        panel.addEventListener("pointermove", updateDrag);
        panel.addEventListener("pointerup", finishDrag);
        panel.addEventListener("pointercancel", cancelDrag);
      }

      if (recenterButton) recenterButton.addEventListener("click", recenterEcho);
      if (resetButton) resetButton.addEventListener("click", resetWorkspace);
      if (lockButton) lockButton.addEventListener("click", toggleLock);
      window.addEventListener("keydown", (event) => {
        if (event.key === "Escape") cancelDrag();
      });
    }

    return {
      initialize() {
        bind();
        applyAllWorkspaceItems(true);
        saveWorkspaceState();
      },
      clampAfterResize,
      saveWorkspaceState,
      restoreEchoPosition() {
        applyWorkspaceItem(state.items.echo, false);
      },
      resetWorkspace,
      recenterEcho,
      toggleLock,
      getInitialTelemetryPanelOpen() {
        return Boolean(state.telemetryPanelOpen);
      },
      items: state.items
    };
  }

  function applyEchoState(state) {
    const value = String(state || "idle").trim().toLowerCase();
    console.log("[Echo UI JS] applyEchoState:", value);

    if (window.echoEntity && typeof window.echoEntity.setState === "function") {
      window.echoEntity.setState(value);
    } else {
      console.error("[Echo UI JS] echoEntity.setState indisponivel");
    }

    updateStatusLabel(value);
    document.body.dataset.echoState = value;

    if (value === "idle") {
      const echoSays = byId("echoSays");
      if (echoSays) echoSays.classList.remove("visible");
    }
  }

  function focusInput() {
    window.setTimeout(() => {
      const input = byId("voiceIn");
      if (input && !input.disabled) input.focus();
    }, 0);
  }

  function speak(text) {
    const echoSays = byId("echoSays");
    if (!echoSays) return;
    echoSays.textContent = text;
    echoSays.classList.toggle("visible", Boolean(text));
  }

  function renderEchoResponse(text) {
    console.log("[Echo UI JS] renderEchoResponse:", text);

    const element = byId("echoResponse");
    if (!element) {
      console.error("[Echo UI JS] #echoResponse nao encontrado");
      return;
    }

    const value = String(text ?? "").trim();
    if (!value) {
      console.error("[Echo UI JS] resposta vazia");
      return;
    }

    replyFadeToken += 1;
    element.textContent = value;
    element.hidden = false;
    element.style.display = "block";
    element.style.visibility = "visible";
    element.style.opacity = "1";
    element.classList.remove("muted");
    element.classList.add("visible");
    updateResponseReadingMode(element, value);

    const responseTitle = byId("responseTitle");
    const bridgeReply = byId("bridgeReply");
    const workspaceHint = byId("workspaceHint");
    const clearButton = byId("clearButton");

    if (responseTitle) responseTitle.textContent = "Echo";
    if (bridgeReply) bridgeReply.textContent = value;
    if (workspaceHint) workspaceHint.classList.remove("visible");
    if (clearButton) clearButton.classList.add("visible");
    if (telemetryDebugMode && (telemetryMode === "automatic" || telemetryState === "thinking_cloud")) {
      applyTelemetryMock("response_ready");
    } else if (telemetryDebugMode) {
      applyTelemetryMock("idle_local");
    }

    const styles = getComputedStyle(element);
    console.log("[Echo UI JS] resposta aplicada:", {
      text: element.textContent,
      className: element.className,
      display: styles.display,
      visibility: styles.visibility,
      opacity: styles.opacity,
      zIndex: styles.zIndex
    });
  }

  function isLongEchoResponse(element, text) {
    const lineCount = String(text || "").split(/\r?\n/).length;
    const wordCount = String(text || "").trim().split(/\s+/).filter(Boolean).length;
    if (lineCount >= 8 || wordCount >= 120) return true;
    return element.scrollHeight > 230;
  }

  function updateResponseReadingMode(element, text) {
    const stage = document.querySelector(".stage");
    const reading = isLongEchoResponse(element, text);
    if (stage) stage.dataset.responseMode = reading ? "reading" : "short";
    element.dataset.responseMode = reading ? "reading" : "short";
    element.tabIndex = reading ? 0 : -1;
    if (reading) {
      element.setAttribute("role", "region");
      element.setAttribute("aria-label", "Resposta longa do Echo");
      if (window.echoEntity && typeof window.echoEntity.setCenter === "function") {
        window.echoEntity.setCenter(650, 120, false);
      }
    } else {
      element.removeAttribute("role");
      element.removeAttribute("aria-label");
      if (window.echoFreeWorkspace && typeof window.echoFreeWorkspace.restoreEchoPosition === "function") {
        window.echoFreeWorkspace.restoreEchoPosition();
      }
    }
  }

  function renderEchoError(message) {
    renderEchoResponse(message || "Não consegui responder a este pedido.");
  }

  function updateClock() {
    const clock = byId("topbarClock");
    const version = byId("appVersion");
    const now = new Date();
    const months = ["JAN", "FEV", "MAR", "ABR", "MAI", "JUN", "JUL", "AGO", "SET", "OUT", "NOV", "DEZ"];
    const hours = String(now.getHours()).padStart(2, "0");
    const minutes = String(now.getMinutes()).padStart(2, "0");
    const day = String(now.getDate()).padStart(2, "0");
    const value = `${hours}:${minutes} · ${day} ${months[now.getMonth()]} ${now.getFullYear()}`;
    if (clock) {
      clock.textContent = value;
      clock.setAttribute("datetime", now.toISOString());
    }
    if (version) version.textContent = `ECHO OS · ${APP_VERSION}`;
  }

  function startClock() {
    updateClock();
    window.setInterval(updateClock, 60000);
  }

  function createWorkspaceController() {
    const workspace = byId("researchWorkspace");
    const topic = byId("researchTopic");
    const status = byId("researchStatus");
    const summary = byId("researchSummary");
    const results = byId("researchResults");
    const stage = document.querySelector(".stage");

    function setMode(mode) {
      document.body.dataset.workspaceMode = mode;
      if (stage) stage.dataset.workspace = mode === "research" ? "research" : "conversation";
      console.log("[Echo UI JS] ui_workspace_mode=", mode);
    }

    function clearResults() {
      if (results) results.textContent = "";
      if (summary) summary.textContent = "";
    }

    function renderCards(items) {
      if (!results) return;
      results.textContent = "";
      const safeItems = Array.isArray(items) ? items.slice(0, 5) : [];
      for (const item of safeItems) {
        const card = document.createElement("article");
        card.className = "research-card";
        const title = document.createElement("strong");
        title.textContent = String(item.title || "Resultado");
        const snippet = document.createElement("p");
        snippet.textContent = String(item.snippet || item.body || "");
        const source = document.createElement("span");
        source.textContent = String(item.source || item.url || item.kind || "fonte");
        card.append(title, snippet, source);
        results.append(card);
      }
    }

    return {
      enterMode(mode, data = {}) {
        if (mode !== "research") {
          this.exitMode();
          return;
        }
        setMode("research");
        if (workspace) workspace.classList.add("visible");
        if (topic) topic.textContent = String(data.topic || "Pesquisa");
        if (status) status.textContent = "A preparar pesquisa.";
        clearResults();
      },
      updateResearch(data = {}) {
        setMode("research");
        if (workspace) workspace.classList.add("visible");
        if (topic && data.topic) topic.textContent = String(data.topic);
        if (status) status.textContent = String(data.statusText || data.message || "Resultados recebidos.");
        if (summary) summary.textContent = String(data.summary || "");
        renderCards(data.results);
      },
      failResearch(data = {}) {
        setMode("research");
        if (workspace) workspace.classList.add("visible");
        if (topic && data.topic) topic.textContent = String(data.topic);
        if (status) status.textContent = String(data.message || "Pesquisa indisponível.");
        if (summary) summary.textContent = String(data.summary || "");
        if (results) results.textContent = "";
      },
      exitMode() {
        setMode("conversation");
        if (workspace) workspace.classList.remove("visible");
      },
      clear() {
        clearResults();
        this.exitMode();
      },
      handleEvent(event) {
        if (!event || typeof event.type !== "string") return;
        if (event.type === "research_unavailable") {
          this.failResearch(event);
          return;
        }
        if (event.type === "conversation_cleared" || event.type === "topic_changed") {
          this.clear();
          return;
        }
        if (event.type === "research_started") {
          this.enterMode("research", event);
        } else if (event.type === "research_results_ready") {
          this.updateResearch({...event, statusText: "Resultados prontos."});
        } else if (event.type === "research_failed") {
          this.failResearch(event);
        } else if (event.type === "research_completed") {
          if (status) status.textContent = "Pesquisa concluída.";
        }
      }
    };
  }

  function handleUiEventPayload(payload) {
    console.log("[Echo UI JS] uiEvent recebido:", payload);
    let event = null;
    try {
      event = JSON.parse(String(payload || "{}"));
    } catch (error) {
      console.error("[Echo UI JS] uiEvent JSON invalido", error);
      return;
    }
    if (!event || typeof event.type !== "string") {
      console.error("[Echo UI JS] uiEvent sem type valido", event);
      return;
    }
    if (window.echoWorkspace) window.echoWorkspace.handleEvent(event);
  }

  function fadeCurrentReply() {
    const echoResponse = byId("echoResponse");
    const workspaceHint = byId("workspaceHint");
    const clearButton = byId("clearButton");
    if (!echoResponse) return;

    const token = replyFadeToken + 1;
    replyFadeToken = token;
    if (workspaceHint) workspaceHint.classList.remove("visible");
    if (clearButton) clearButton.classList.remove("visible");
    echoResponse.classList.add("muted");
    echoResponse.classList.remove("visible");
    echoResponse.style.opacity = "";
    echoResponse.style.visibility = "";
    echoResponse.style.display = "";
    speak("");
    window.setTimeout(() => {
      if (replyFadeToken !== token) return;
      echoResponse.textContent = "";
      echoResponse.classList.remove("muted");
    }, 220);
  }

  function sendMessage(text) {
    const input = byId("voiceIn");
    const message = String(text || "").trim();

    if (!message || requestActive) {
      focusInput();
      return;
    }
    if (!controllerReady || !controller) {
      applyEchoState("error");
      renderEchoError("A ligação ao Python ainda não está pronta.");
      focusInput();
      return;
    }

    if (input) input.value = "";
    fadeCurrentReply();
    if (telemetryDebugMode) {
      if (telemetryMode === "automatic") runAutomaticRoutingMock();
      else applyTelemetryMock("thinking_local");
    }
    requestActive = true;
    setInputEnabled(false);
    controller.submitMessage(message);
  }

  function connectControllerSignals(activeController) {
    console.log("[Echo UI JS] a ligar signals");

    activeController.stateChanged.connect((state) => {
      console.log("[Echo UI JS] stateChanged recebido:", state);
      applyEchoState(String(state));
    });

    activeController.responseReady.connect((text) => {
      console.log("[Echo UI JS] responseReady recebido:", text);
      renderEchoResponse(String(text));
    });

    activeController.errorOccurred.connect((message) => {
      console.log("[Echo UI JS] errorOccurred recebido:", message);
      renderEchoError(String(message));
      applyEchoState("error");
    });

    activeController.uiEvent.connect((payload) => {
      handleUiEventPayload(payload);
    });

    if (activeController.telemetryUpdated) {
      activeController.telemetryUpdated.connect((payload) => {
        console.log("[Echo UI JS] telemetryUpdated recebido:", payload);
        applyRealTelemetry(payload);
      });
    }

    if (activeController.modelModeChanged) {
      activeController.modelModeChanged.connect((payload) => {
        console.log("[Echo UI JS] modelModeChanged recebido:", payload);
        applyRealTelemetry(payload);
      });
    }

    activeController.requestStarted.connect((message) => {
      console.log("[Echo UI JS] requestStarted recebido:", message);
      requestActive = true;
      setInputEnabled(false);
    });

    activeController.requestFinished.connect(() => {
      console.log("[Echo UI JS] requestFinished recebido");
      requestActive = false;
      setInputEnabled(true);
      focusInput();
    });

    console.log("[Echo UI JS] signals ligados");
  }

  function initializeEchoChannel() {
    if (window.__echoChannelInitialized) {
      console.warn("[Echo UI JS] canal ja inicializado");
      return;
    }

    console.log("[Echo UI JS] QWebChannel disponivel:", typeof QWebChannel);
    console.log("[Echo UI JS] qt disponivel:", typeof qt);
    console.log(
      "[Echo UI JS] transport disponivel:",
      Boolean(window.qt && qt.webChannelTransport)
    );

    if (
      typeof QWebChannel === "undefined" ||
      !window.qt ||
      !qt.webChannelTransport
    ) {
      console.error("[Echo UI JS] QWebChannel indisponivel");
      return;
    }

    new QWebChannel(qt.webChannelTransport, (channel) => {
      window.__echoChannelInitialized = true;
      console.log("[Echo UI JS] objetos disponiveis:", Object.keys(channel.objects));

      controller = channel.objects.echoController;
      if (!controller) {
        console.error("[Echo UI JS] controller nao encontrado", Object.keys(channel.objects));
        return;
      }

      window.echoController = controller;
      controllerReady = true;
      console.log("[Echo UI JS] controller ligado", controller);

      connectControllerSignals(controller);
      if (typeof controller.getModelTelemetry === "function") {
        controller.getModelTelemetry((payload) => applyRealTelemetry(payload));
      }
      setInputEnabled(true);
      focusInput();
    });
  }

  function bindDomEvents() {
    const form = byId("echoForm");
    const clearButton = byId("clearButton");
    const closeResearch = byId("closeResearch");
    const telemetryCompact = byId("telemetryCompact");
    const closeTelemetry = byId("closeTelemetry");
    const telemetryPanel = byId("telemetryPanel");
    const autoClaudeButtons = Array.from(document.querySelectorAll("[data-auto-claude]"));
    const saveModelBudget = byId("saveModelBudget");
    const configureClaude = byId("configureClaude");
    const saveAnthropicKey = byId("saveAnthropicKey");
    const removeAnthropicKey = byId("removeAnthropicKey");
    const testAnthropicKey = byId("testAnthropicKey");

    if (form) {
      form.addEventListener("submit", (event) => {
        event.preventDefault();
        const input = byId("voiceIn");
        sendMessage(input ? input.value : "");
      });
    }

    if (clearButton) {
      clearButton.addEventListener("click", () => {
        const bridgeReply = byId("bridgeReply");
        const responseTitle = byId("responseTitle");
        const workspaceHint = byId("workspaceHint");
        const echoResponse = byId("echoResponse");

        if (bridgeReply) bridgeReply.textContent = "A resposta aparece aqui.";
        if (responseTitle) responseTitle.textContent = "Última resposta";
        if (workspaceHint) workspaceHint.classList.remove("visible");
        if (window.echoWorkspace) window.echoWorkspace.clear();
        if (echoResponse) {
          echoResponse.textContent = "";
          echoResponse.dataset.responseMode = "short";
          echoResponse.tabIndex = -1;
          echoResponse.hidden = false;
          echoResponse.style.display = "";
          echoResponse.style.visibility = "";
          echoResponse.style.opacity = "";
          echoResponse.classList.remove("visible", "muted");
        }
        const stage = document.querySelector(".stage");
        if (stage) stage.dataset.responseMode = "short";
        if (window.echoFreeWorkspace && typeof window.echoFreeWorkspace.restoreEchoPosition === "function") {
          window.echoFreeWorkspace.restoreEchoPosition();
        }
        clearButton.classList.remove("visible");
        speak("");
        if (controller && typeof controller.clearConversation === "function") controller.clearConversation();
        else if (controller) controller.setState("idle");
        else applyEchoState("idle");
        applyTelemetryMock("idle_local");
        setTelemetryPanel(false);
        focusInput();
      });
    }

    if (telemetryCompact) {
      telemetryCompact.addEventListener("click", (event) => {
        event.stopPropagation();
        setTelemetryPanel(!telemetryPanelOpen);
      });
    }

    if (closeTelemetry) {
      closeTelemetry.addEventListener("click", (event) => {
        event.stopPropagation();
        setTelemetryPanel(false);
        focusInput();
      });
    }

    if (telemetryPanel) {
      telemetryPanel.addEventListener("click", (event) => {
        const sectionButton = event.target.closest("button[data-panel-section]");
        if (sectionButton) {
          openPanelSection(sectionButton.dataset.panelSection);
          return;
        }
        const modeButton = event.target.closest("button[data-telemetry-mode]");
        if (!modeButton) return;
        setTelemetryMode(modeButton.dataset.telemetryMode);
      });
    }

    autoClaudeButtons.forEach((button) => {
      button.addEventListener("click", () => {
        const enabled = button.dataset.autoClaude === "true";
        if (enabled && button.dataset.confirmed !== "true") {
          button.dataset.confirmed = "true";
          const note = byId("telemetryNote");
          if (note) note.textContent = "Clica novamente em ON para confirmar Claude automático.";
          return;
        }
        autoClaudeButtons.forEach((item) => {
          if (item !== button) item.dataset.confirmed = "false";
        });
        executeSystemCommand("set_automatic_claude_enabled", {enabled});
      });
    });

    if (saveModelBudget) {
      saveModelBudget.addEventListener("click", () => {
        const daily = Number(byId("dailyBudgetUsd") ? byId("dailyBudgetUsd").value : 0);
        const single = Number(byId("singleCallBudgetUsd") ? byId("singleCallBudgetUsd").value : 0);
        executeSystemCommand("set_model_budget", {
          daily_budget_usd: Math.max(0, daily),
          max_single_call_estimated_usd: Math.max(0, single)
        });
      });
    }

    if (configureClaude) {
      configureClaude.addEventListener("click", () => {
        openPanelSection("models");
        const keyPanel = byId("anthropicKeyPanel");
        if (keyPanel) keyPanel.hidden = false;
        const keyInput = byId("anthropicApiKey");
        if (keyInput) keyInput.focus({preventScroll: true});
      });
    }

    if (saveAnthropicKey) {
      saveAnthropicKey.addEventListener("click", () => {
        const keyInput = byId("anthropicApiKey");
        executeSystemCommand("save_anthropic_key", {api_key: keyInput ? keyInput.value : ""}, (payload) => {
          if (keyInput) keyInput.value = "";
          applyRealTelemetry(payload);
        });
      });
    }

    if (removeAnthropicKey) {
      removeAnthropicKey.addEventListener("click", () => executeSystemCommand("remove_anthropic_key"));
    }

    if (testAnthropicKey) {
      testAnthropicKey.addEventListener("click", () => executeSystemCommand("test_anthropic_connection"));
    }

    document.addEventListener("click", (event) => {
      const target = event.target;
      if (!telemetryPanelOpen) return;
      if (telemetryPanel && telemetryPanel.contains(target)) return;
      if (telemetryCompact && telemetryCompact.contains(target)) return;
      setTelemetryPanel(false);
    });

    if (closeResearch) {
      closeResearch.addEventListener("click", () => {
        if (window.echoWorkspace) window.echoWorkspace.clear();
        focusInput();
      });
    }

    window.addEventListener("keydown", (event) => {
      if (telemetryPanelOpen && ["ArrowLeft", "ArrowRight"].includes(event.key) && document.activeElement && document.activeElement.matches("[data-panel-section]")) {
        const tabs = Array.from(document.querySelectorAll("[data-panel-section]"));
        const index = tabs.indexOf(document.activeElement);
        const delta = event.key === "ArrowRight" ? 1 : -1;
        const next = tabs[(index + delta + tabs.length) % tabs.length];
        if (next) {
          event.preventDefault();
          next.focus();
          openPanelSection(next.dataset.panelSection);
        }
        return;
      }
      if (event.key === "Escape" && telemetryPanelOpen) {
        event.preventDefault();
        const keyPanel = byId("anthropicKeyPanel");
        if (keyPanel && !keyPanel.hidden) {
          keyPanel.hidden = true;
          return;
        }
        setTelemetryPanel(false);
        focusInput();
      }
    });
  }

  window.addEventListener("DOMContentLoaded", () => {
    console.log("[Echo UI JS] DOMContentLoaded");
    console.log("[Echo UI JS] echoResponse no arranque:", byId("echoResponse"));
    telemetryDebugMode = new URLSearchParams(window.location.search).get("telemetryDebug") === "1";
    window.echoEntity = new window.EchoEntity(byId("mind"));
    window.echoWorkspace = createWorkspaceController();
    window.echoFreeWorkspace = createFreeWorkspaceController();
    window.echoTelemetryDemo = applyTelemetryMock;
    scaleStage();
    window.echoFreeWorkspace.initialize();
    startClock();
    applyEchoState("active");
    applyTelemetryMock("idle_local");
    setTelemetryPanel(window.echoFreeWorkspace.getInitialTelemetryPanelOpen());
    setInputEnabled(false);
    bindDomEvents();
    window.addEventListener("resize", () => {
      scaleStage();
      if (window.echoFreeWorkspace) window.echoFreeWorkspace.clampAfterResize();
    });
    initializeEchoChannel();
  });
})();

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
    local_tool: "Local tool",
    social_fast_path: "Social fast path",
    system_datetime: "System datetime",
    document_task: "Document task"
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

  let layoutFrame = 0;

  function chooseLayoutDensity(width, height) {
    if (width < 760 || height < 620) return "compact";
    if (width >= 2200) return "ultrawide";
    if (width >= 1500 && height >= 760) return "wide";
    return "standard";
  }

  const RESPONSE_LAYOUT_CONFIG = {
    inline: {
      maxRenderedRatio: 0.18,
      exitMaxRenderedRatio: 0.22,
      maxCharacters: 220,
      maxBlocks: 3
    },
    stacked: {
      maxRenderedRatio: 0.38,
      enterRenderedRatio: 0.20,
      exitMaxRenderedRatio: 0.34,
      maxCharacters: 760,
      maxBlocks: 7
    },
    focus: {
      enterRenderedRatio: 0.40,
      exitRenderedRatio: 0.34,
      forceCharacters: 1050,
      forceListItems: 8,
      forceBlocks: 9
    }
  };

  let activeResponseLayout = "inline";
  let responseLayoutFrame = 0;

  function responseSizeForElement(element) {
    if (!element || !String(element.textContent || "").trim()) return "none";
    const wordCount = String(element.textContent || "").trim().split(/\s+/).filter(Boolean).length;
    if (wordCount >= 120 || element.dataset.responseLayout === "focus") return "long";
    if (wordCount >= 36) return "medium";
    return "short";
  }

  function entityScaleForLayout(density, responseSize, height, responseLayout = "inline") {
    if (height < 620 || density === "compact") return "compact";
    if (responseLayout === "focus") return "compact";
    if (responseLayout === "stacked" || responseSize === "long") return "reduced";
    if (density === "ultrawide" && responseSize === "none") return "hero";
    return "standard";
  }

  function cssPixelNumber(element, propertyName, fallback) {
    if (!element) return fallback;
    const value = Number.parseFloat(getComputedStyle(element).getPropertyValue(propertyName));
    return Number.isFinite(value) && value > 0 ? value : fallback;
  }

  function updateAdaptiveLayout() {
    const scaler = byId("stageScaler");
    if (!scaler) {
      console.error("[Echo UI JS] #stageScaler nao encontrado");
      return;
    }
    const stage = document.querySelector(".stage");
    if (!stage) return;
    const rect = stage.getBoundingClientRect();
    const width = rect.width || window.innerWidth;
    const height = rect.height || window.innerHeight;
    const density = chooseLayoutDensity(width, height);
    const responseSize = responseSizeForElement(byId("echoResponse"));
    const responseLayout = stage.dataset.responseLayout || "inline";
    const entityScale = entityScaleForLayout(density, responseSize, height, responseLayout);

    stage.dataset.layoutDensity = density;
    stage.dataset.responseSize = responseSize;
    stage.dataset.responseLayout = responseLayout;
    stage.dataset.entityScale = entityScale;
    stage.style.setProperty("--stage-width", `${Math.round(width)}px`);
    stage.style.setProperty("--stage-height", `${Math.round(height)}px`);
    scaler.style.transform = "none";
    if (window.echoEntity && typeof window.echoEntity.resize === "function") {
      window.echoEntity.resize();
    }
  }

  function scheduleAdaptiveLayout() {
    window.cancelAnimationFrame(layoutFrame);
    layoutFrame = window.requestAnimationFrame(() => {
      updateAdaptiveLayout();
      if (window.echoFreeWorkspace) {
        window.echoFreeWorkspace.clampAfterResize({persist: false, restoreResponseLayout: true});
      } else {
        const response = byId("echoResponse");
        if (response && response.classList.contains("visible")) {
          applyResponseLayout(response, response.dataset.rawText || response.textContent || "", {measureAgain: false});
        }
      }
    });
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
    if (cleanProvider === "local_tool" || cleanProvider === "tool" || cleanProvider === "local") return "NONE";
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
    const provider = String(data.execution_provider || data.provider || "").toLowerCase();
    const mode = String(data.configured_model_mode || data.mode || "local").toLowerCase();
    const executionPath = String(data.execution_path || "").toLowerCase();
    const reason = String(data.reason_code || "").toLowerCase();
    if (state === "error") return `ERROR · ${humanReasonLabel(data.provider_error_type || reason).toUpperCase()}`;
    if (reason.includes("memory") || provider === "memory") return `MEMORY · ${latencyLabel(data.latency_ms)} · 0 COST`;
    if (provider === "tool" || provider === "local_tool" || reason === "local_tool" || executionPath === "document_task") {
      return `${modeLabel(mode)} · TOOL · ${latencyLabel(data.latency_ms)}`;
    }
    if (provider === "local" || executionPath === "system_datetime" || executionPath === "social_fast_path") {
      return `${modeLabel(mode)} · LOCAL · ${latencyLabel(data.latency_ms)}`;
    }
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
    const mode = String(data.configured_model_mode || data.mode || "local").toLowerCase();
    const provider = String(data.execution_provider || data.provider || "").toLowerCase();
    const executionModel = String(data.execution_model || data.model || "");
    const executionPath = String(data.execution_path || "");
    const reasonCode = String(data.reason_code || "");
    const reason = String(data.reason_label || humanReasonLabel(reasonCode || executionPath));

    telemetryMode = mode === "claude" ? "claude" : mode === "automatic" ? "automatic" : "local";
    telemetryState = state;
    if (elements.stage) {
      elements.stage.dataset.telemetryMode = telemetryMode;
      elements.stage.dataset.telemetryState = state;
    }
    if (elements.compact) elements.compact.classList.add("visible");
    if (elements.compactText) elements.compactText.textContent = compactTelemetryLabel(data);
    if (elements.mode) elements.mode.textContent = modeLabel(mode);
    if (elements.model) elements.model.textContent = modelLabel(provider, executionModel);
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

  function appendInlineMarkdown(parent, text) {
    const source = String(text || "");
    const tokenPattern = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*\n]+\*)/g;
    let lastIndex = 0;
    let match = null;

    while ((match = tokenPattern.exec(source)) !== null) {
      if (match.index > lastIndex) {
        parent.append(document.createTextNode(source.slice(lastIndex, match.index)));
      }

      const token = match[0];
      if (token.startsWith("`") && token.endsWith("`")) {
        const code = document.createElement("code");
        code.textContent = token.slice(1, -1);
        parent.append(code);
      } else if (token.startsWith("**") && token.endsWith("**")) {
        const strong = document.createElement("strong");
        strong.textContent = token.slice(2, -2);
        parent.append(strong);
      } else if (token.startsWith("*") && token.endsWith("*")) {
        const emphasis = document.createElement("em");
        emphasis.textContent = token.slice(1, -1);
        parent.append(emphasis);
      }
      lastIndex = tokenPattern.lastIndex;
    }

    if (lastIndex < source.length) {
      parent.append(document.createTextNode(source.slice(lastIndex)));
    }
  }

  function renderSafeMarkdown(container, text) {
    const source = String(text || "");
    const lines = source.split(/\r?\n/);
    let paragraphLines = [];
    let list = null;

    container.textContent = "";
    container.dataset.rawText = source;

    function flushParagraph() {
      if (!paragraphLines.length) return;
      const paragraph = document.createElement("p");
      paragraphLines.forEach((line, index) => {
        if (index > 0) paragraph.append(document.createElement("br"));
        appendInlineMarkdown(paragraph, line);
      });
      container.append(paragraph);
      paragraphLines = [];
    }

    function closeList() {
      if (!list) return;
      container.append(list);
      list = null;
    }

    for (const line of lines) {
      const trimmed = line.trim();
      const bullet = trimmed.match(/^[-*]\s+(.+)$/);
      if (!trimmed) {
        flushParagraph();
        closeList();
        continue;
      }
      if (bullet) {
        flushParagraph();
        if (!list) list = document.createElement("ul");
        const item = document.createElement("li");
        appendInlineMarkdown(item, bullet[1]);
        list.append(item);
        continue;
      }
      closeList();
      paragraphLines.push(line);
    }

    flushParagraph();
    closeList();
  }

  function classifyResponseContent(text) {
    const source = String(text || "");
    const lines = source.split(/\r?\n/);
    const codeBlockCount = (source.match(/```/g) || []).length >= 2 ? Math.floor((source.match(/```/g) || []).length / 2) : 0;
    const tableLineCount = lines.filter((line) => /^\s*\|.+\|\s*$/.test(line)).length;
    const listItemCount = lines.filter((line) => /^\s*[-*]\s+/.test(line)).length;
    const emailSignals = [
      /\bassunto\s*:/i,
      /\bol[áa]\s+[\wÀ-ÿ]/i,
      /\bcumprimentos\b/i,
      /\bsegue em anexo\b/i
    ].filter((pattern) => pattern.test(source)).length;

    if (codeBlockCount > 0) return "code";
    if (tableLineCount >= 2) return "table";
    if (emailSignals >= 2) return "email";
    if (listItemCount >= 3) return "list";
    if (source.length > 700 || lines.length >= 8) return "document";
    return listItemCount || tableLineCount ? "mixed" : "plain_text";
  }

  function collectResponseMetrics(element, text) {
    const stage = document.querySelector(".stage");
    const source = String(text || "");
    const lines = source.split(/\r?\n/);
    const stageRect = stage ? stage.getBoundingClientRect() : {width: STAGE_WIDTH, height: STAGE_HEIGHT};
    const responseRect = element ? element.getBoundingClientRect() : {height: 0, width: 0};
    const composer = byId("echoForm");
    const composerRect = composer ? composer.getBoundingClientRect() : {height: 0};
    const safeGap = cssPixelNumber(stage, "--safe-gap", 24);
    const headerHeight = cssPixelNumber(stage, "--header-height", HEADER_SAFE_HEIGHT);
    const renderedHeight = Math.max(element ? element.scrollHeight : 0, responseRect.height || 0);
    const availableStageHeight = Math.max(1, stageRect.height - headerHeight - composerRect.height - safeGap * 3);
    const paragraphCount = source.split(/\n\s*\n/).filter((block) => block.trim()).length || (source.trim() ? 1 : 0);
    const listItemCount = lines.filter((line) => /^\s*[-*]\s+/.test(line)).length;
    const codeBlockCount = (source.match(/```/g) || []).length >= 2 ? Math.floor((source.match(/```/g) || []).length / 2) : 0;
    const tableLineCount = lines.filter((line) => /^\s*\|.+\|\s*$/.test(line)).length;
    const blockCount = Math.max(paragraphCount, listItemCount + paragraphCount, lines.filter((line) => line.trim()).length);

    return {
      characterCount: source.length,
      blockCount,
      paragraphCount,
      listItemCount,
      codeBlockCount,
      tableLineCount,
      renderedHeight,
      availableStageHeight,
      availableStageWidth: stageRect.width || STAGE_WIDTH,
      renderedRatio: renderedHeight / availableStageHeight,
      contentType: classifyResponseContent(source)
    };
  }

  function determineResponseLayout(metrics, previousLayout = activeResponseLayout) {
    const inline = RESPONSE_LAYOUT_CONFIG.inline;
    const stacked = RESPONSE_LAYOUT_CONFIG.stacked;
    const focus = RESPONSE_LAYOUT_CONFIG.focus;
    const type = metrics.contentType;
    const forcedFocus = (
      metrics.characterCount >= focus.forceCharacters ||
      metrics.listItemCount >= focus.forceListItems ||
      metrics.blockCount >= focus.forceBlocks ||
      metrics.codeBlockCount > 0 ||
      metrics.tableLineCount >= 2 ||
      ["email", "document", "code", "table"].includes(type)
    );

    if (forcedFocus) return "focus";
    if (previousLayout === "focus" && metrics.renderedRatio > focus.exitRenderedRatio) return "focus";
    if (metrics.renderedRatio >= focus.enterRenderedRatio) return "focus";

    if (previousLayout === "stacked" && metrics.renderedRatio > inline.exitMaxRenderedRatio) return "stacked";
    if (
      metrics.renderedRatio > inline.maxRenderedRatio ||
      metrics.characterCount > inline.maxCharacters ||
      metrics.blockCount > inline.maxBlocks
    ) {
      return metrics.renderedRatio <= stacked.maxRenderedRatio &&
        metrics.characterCount <= stacked.maxCharacters &&
        metrics.blockCount <= stacked.maxBlocks ? "stacked" : "focus";
    }

    return "inline";
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

  function stageSize(stage) {
    const rect = stage ? stage.getBoundingClientRect() : null;
    return {
      width: Math.max(320, rect && rect.width ? rect.width : STAGE_WIDTH),
      height: Math.max(360, rect && rect.height ? rect.height : STAGE_HEIGHT)
    };
  }

  function workspaceDefaults(stage) {
    const size = stageSize(stage);
    const entityDiameter = cssPixelNumber(stage, "--entity-size-actual", ECHO_ITEM_SIZE.width);
    const entityRadius = Math.max(70, Math.min(220, entityDiameter / 2));
    const safeGap = cssPixelNumber(stage, "--safe-gap", 24);
    const headerHeight = cssPixelNumber(stage, "--header-height", HEADER_SAFE_HEIGHT);
    const panelWidth = cssPixelNumber(stage, "--telemetry-panel-width", PANEL_ITEM_SIZE.width);
    const centerX = size.width / 2;
    const topCenterY = headerHeight + entityRadius + safeGap;
    const topRightX = Math.max(size.width - entityRadius - safeGap * 1.4, centerX + entityRadius);
    const centerY = Math.max(headerHeight + entityRadius + safeGap, (size.height - 82) * 0.48);
    return {
      echo: {
        center: {x: centerX, y: centerY},
        "top-center": {x: centerX, y: topCenterY},
        "top-right": {x: Math.min(size.width - entityRadius - safeGap, topRightX), y: topCenterY},
        "upper-left": {x: Math.max(190, size.width * 0.28), y: topCenterY + 22},
        "upper-right": {x: Math.min(size.width - 190, size.width * 0.72), y: topCenterY + 22},
        "center-left": {x: Math.max(190, size.width * 0.28), y: centerY + 18},
        "center-right": {x: Math.min(size.width - 190, size.width * 0.72), y: centerY + 18}
      },
      telemetryPanel: {
        "upper-left": {x: 56, y: headerHeight + 30},
        "upper-right": {x: Math.max(12, size.width - panelWidth - 56), y: headerHeight + 30},
        "center-left": {x: 56, y: headerHeight + 174},
        "center-right": {x: Math.max(12, size.width - panelWidth - 56), y: headerHeight + 174}
      }
    };
  }

  function localRectFromElement(stage, element) {
    if (!stage || !element) return null;
    const stageRect = stage.getBoundingClientRect();
    const rect = element.getBoundingClientRect();
    if (!rect.width && !rect.height) return null;
    return {
      left: rect.left - stageRect.left,
      top: rect.top - stageRect.top,
      right: rect.right - stageRect.left,
      bottom: rect.bottom - stageRect.top,
      width: rect.width,
      height: rect.height
    };
  }

  function boxesIntersect(a, b) {
    if (!a || !b) return false;
    return a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;
  }

  function echoRectForPoint(point, radius) {
    return {
      left: point.x - radius,
      top: point.y - radius,
      right: point.x + radius,
      bottom: point.y + radius,
      width: radius * 2,
      height: radius * 2
    };
  }

  function responseAnchorForLayout(stage, layout) {
    const zones = workspaceDefaults(stage);
    if (layout === "focus") return "top-right";
    if (layout === "stacked") return "top-center";
    return "center";
  }

  function resolveSafeEchoAnchor(stage, requestedAnchor) {
    const zones = workspaceDefaults(stage);
    const point = zones.echo[requestedAnchor] || zones.echo.center;
    if (requestedAnchor !== "top-right" || !telemetryPanelOpen) return {anchor: requestedAnchor, point};

    const entityDiameter = cssPixelNumber(stage, "--entity-size-actual", ECHO_ITEM_SIZE.width);
    const entityRadius = Math.max(70, Math.min(220, entityDiameter / 2));
    const echoRect = echoRectForPoint(point, entityRadius);
    const panelRect = localRectFromElement(stage, byId("telemetryPanel"));
    if (!boxesIntersect(echoRect, panelRect)) return {anchor: requestedAnchor, point};
    return {anchor: "top-center", point: zones.echo["top-center"]};
  }

  function updateResponseLayoutBounds(stage, layout) {
    if (!stage) return;
    const size = stageSize(stage);
    const anchor = responseAnchorForLayout(stage, layout);
    let anchorResult = resolveSafeEchoAnchor(stage, anchor);
    const entityDiameter = cssPixelNumber(stage, "--entity-size-actual", ECHO_ITEM_SIZE.width);
    const entityRadius = Math.max(70, Math.min(220, entityDiameter / 2));
    const safeGap = cssPixelNumber(stage, "--safe-gap", 24);
    const headerHeight = cssPixelNumber(stage, "--header-height", HEADER_SAFE_HEIGHT);
    const inputHeight = cssPixelNumber(stage, "--input-height", 72);
    const minFocusResponseWidth = 540;
    const maxFocusResponseWidth = Math.min(980, Math.max(360, size.width - safeGap * 4));
    let focusResponseLeft = Math.max(safeGap * 2, 36);
    let focusResponseWidth = maxFocusResponseWidth;

    if (layout === "focus" && anchorResult.anchor === "top-right") {
      const echoRect = echoRectForPoint(anchorResult.point, entityRadius);
      const reservedRightEdge = echoRect.left - safeGap;
      const availableLeft = Math.max(safeGap * 2, 36);
      const availableWidth = reservedRightEdge - availableLeft;
      if (availableWidth < minFocusResponseWidth) {
        const zones = workspaceDefaults(stage);
        anchorResult = {anchor: "top-center", point: zones.echo["top-center"]};
      } else {
        focusResponseWidth = Math.min(maxFocusResponseWidth, availableWidth);
        focusResponseLeft = availableLeft + Math.max(0, (availableWidth - focusResponseWidth) / 2);
      }
    }

    const focusTop = headerHeight + safeGap * 1.7;
    const stackedTop = anchorResult.point.y + entityRadius + Math.max(14, safeGap * 0.7);
    const inlineBottom = inputHeight + safeGap + 34;
    const top = layout === "focus" && anchorResult.anchor === "top-right" ? focusTop : stackedTop;
    const bottomReserve = layout === "focus" ? inputHeight + safeGap * 3.1 : inputHeight + safeGap * 3.4;
    const maxHeight = Math.max(120, size.height - top - bottomReserve);

    stage.dataset.entityAnchor = anchorResult.anchor;
    stage.style.setProperty("--response-top", `${Math.round(top)}px`);
    stage.style.setProperty("--response-bottom", `${Math.round(inlineBottom)}px`);
    stage.style.setProperty("--response-max-height-current", `${Math.round(maxHeight)}px`);
    stage.style.setProperty("--focus-response-left", `${Math.round(focusResponseLeft)}px`);
    stage.style.setProperty("--focus-response-width", `${Math.round(focusResponseWidth)}px`);
  }

  function createFreeWorkspaceController() {
    const stage = document.querySelector(".stage");
    const mind = byId("mind");
    const panel = byId("telemetryPanel");
    const panelDragHandle = document.querySelector('[data-workspace-drag-handle="telemetryPanel"]');
    const recenterButton = byId("recenterEcho");
    const resetButton = byId("resetWorkspace");
    const lockButton = byId("toggleWorkspaceLock");
    const initialZones = workspaceDefaults(stage);

    const defaults = {
      locked: false,
      nextZIndex: 12,
      telemetryPanelOpen: false,
      items: {
        echo: createWorkspaceItem({
          id: "echo",
          type: "echo",
          x: initialZones.echo.center.x,
          y: initialZones.echo.center.y,
          width: ECHO_ITEM_SIZE.width,
          height: ECHO_ITEM_SIZE.height,
          snapZone: "center",
          zIndex: 1
        }),
        telemetryPanel: createWorkspaceItem({
          id: "telemetryPanel",
          type: "panel",
          x: initialZones.telemetryPanel["upper-left"].x,
          y: initialZones.telemetryPanel["upper-left"].y,
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

    function currentSnapZones() {
      return workspaceDefaults(stage);
    }

    function resizeWorkspaceItems() {
      const zones = currentSnapZones();
      for (const item of Object.values(state.items)) {
        if (!item || !item.snapZone || !zones[item.id] || !zones[item.id][item.snapZone]) continue;
        item.x = zones[item.id][item.snapZone].x;
        item.y = zones[item.id][item.snapZone].y;
      }
    }

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
      return {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top
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
      return {
        left: (formRect.left - stageRect.left) - 16,
        top: (formRect.top - stageRect.top) - 14,
        right: (formRect.right - stageRect.left) + 16,
        bottom: (formRect.bottom - stageRect.top) + 14
      };
    }

    function safeAreaRect() {
      const size = stageSize(stage);
      return {
        left: 12,
        top: HEADER_SAFE_HEIGHT + 12,
        right: size.width - 12,
        bottom: size.height - 12
      };
    }

    function clampNumber(value, min, max) {
      return Math.min(Math.max(value, min), max);
    }

    function clampItemToStage(item) {
      const inputRect = inputSafeRect();
      const safeRect = safeAreaRect();
      if (item.type === "echo") {
        item.x = clampNumber(item.x, safeRect.left + item.width / 2, safeRect.right - item.width / 2);
        item.y = clampNumber(item.y, safeRect.top + item.height / 2, safeRect.bottom - item.height / 2);
        if (rectsIntersect(itemRect(item), inputRect)) {
          item.y = Math.min(item.y, inputRect.top - item.height / 2 - 14);
          item.y = clampNumber(item.y, safeRect.top + item.height / 2, safeRect.bottom - item.height / 2);
        }
      } else {
        item.width = cssPixelNumber(stage, "--telemetry-panel-width", PANEL_ITEM_SIZE.width);
        item.x = clampNumber(item.x, safeRect.left, safeRect.right - item.width);
        item.y = clampNumber(item.y, safeRect.top, safeRect.bottom - item.height);
        if (rectsIntersect(itemRect(item), inputRect)) {
          item.y = Math.min(item.y, inputRect.top - item.height - 12);
          item.y = clampNumber(item.y, safeRect.top, safeRect.bottom - item.height);
        }
      }
      return item;
    }

    function findNearestSnapZone(item) {
      const zones = currentSnapZones()[item.id] || {};
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
      const point = currentSnapZones()[item.id][zoneName];
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
      const zones = currentSnapZones();
      const point = zones[itemId] && zones[itemId][zoneName];
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

    function clampAfterResize(options = {}) {
      window.cancelAnimationFrame(resizeFrame);
      resizeFrame = window.requestAnimationFrame(() => {
        resizeWorkspaceItems();
        applyAllWorkspaceItems(false);
        if (options.restoreResponseLayout) {
          const response = byId("echoResponse");
          if (response && response.classList.contains("visible")) {
            applyResponseLayout(response, response.dataset.rawText || response.textContent || "", {measureAgain: false});
          }
        }
        if (options.persist) saveWorkspaceState();
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
    renderSafeMarkdown(element, value);
    element.hidden = false;
    element.style.display = "block";
    element.style.visibility = "visible";
    element.style.opacity = "1";
    element.classList.remove("muted");
    element.classList.add("visible");
    updateResponseLayout(element, value);

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

  function applyResponseLayout(element, text, options = {}) {
    const stage = document.querySelector(".stage");
    if (!stage || !element) return;

    const metrics = collectResponseMetrics(element, text);
    const layout = determineResponseLayout(metrics, activeResponseLayout);
    activeResponseLayout = layout;
    const reading = layout !== "inline";
    const anchor = responseAnchorForLayout(stage, layout);

    stage.dataset.responseLayout = layout;
    stage.dataset.responseMode = reading ? "reading" : "short";
    stage.dataset.entityAnchor = anchor;
    element.dataset.responseLayout = layout;
    element.dataset.responseMode = reading ? "reading" : "short";
    element.dataset.contentType = metrics.contentType;
    element.tabIndex = reading ? 0 : -1;

    updateAdaptiveLayout();
    updateResponseLayoutBounds(stage, layout);

    if (reading) {
      element.setAttribute("role", "region");
      element.setAttribute("aria-label", "Resposta longa do Echo");
      if (layout === "focus") element.setAttribute("aria-label", "Área de leitura do Echo");
      if (window.echoEntity && typeof window.echoEntity.setCenter === "function") {
        const finalAnchor = stage.dataset.entityAnchor || responseAnchorForLayout(stage, layout);
        const anchorResult = resolveSafeEchoAnchor(stage, finalAnchor);
        stage.dataset.entityAnchor = anchorResult.anchor;
        window.echoEntity.setCenter(anchorResult.point.x, anchorResult.point.y, false);
      }
    } else {
      element.removeAttribute("role");
      element.removeAttribute("aria-label");
      stage.style.removeProperty("--response-top");
      stage.style.removeProperty("--response-bottom");
      stage.style.removeProperty("--response-max-height-current");
      stage.style.removeProperty("--focus-response-left");
      stage.style.removeProperty("--focus-response-width");
      if (window.echoFreeWorkspace && typeof window.echoFreeWorkspace.restoreEchoPosition === "function") {
        window.echoFreeWorkspace.restoreEchoPosition();
      }
    }

    if (options.measureAgain) {
      window.cancelAnimationFrame(responseLayoutFrame);
      responseLayoutFrame = window.requestAnimationFrame(() => {
        const nextMetrics = collectResponseMetrics(element, text);
        const nextLayout = determineResponseLayout(nextMetrics, activeResponseLayout);
        if (nextLayout !== activeResponseLayout) {
          applyResponseLayout(element, text, {measureAgain: false});
          return;
        }
        updateAdaptiveLayout();
        updateResponseLayoutBounds(stage, activeResponseLayout);
      });
    }
  }

  function updateResponseLayout(element, text) {
    applyResponseLayout(element, text, {measureAgain: true});
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
          echoResponse.dataset.responseLayout = "inline";
          echoResponse.dataset.contentType = "plain_text";
          echoResponse.tabIndex = -1;
          echoResponse.hidden = false;
          echoResponse.style.display = "";
          echoResponse.style.visibility = "";
          echoResponse.style.opacity = "";
          echoResponse.classList.remove("visible", "muted");
        }
        const stage = document.querySelector(".stage");
        activeResponseLayout = "inline";
        if (stage) {
          stage.dataset.responseMode = "short";
          stage.dataset.responseLayout = "inline";
          stage.dataset.entityAnchor = "center";
          stage.style.removeProperty("--response-top");
          stage.style.removeProperty("--response-bottom");
          stage.style.removeProperty("--response-max-height-current");
          stage.style.removeProperty("--focus-response-left");
          stage.style.removeProperty("--focus-response-width");
        }
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
    updateAdaptiveLayout();
    window.echoFreeWorkspace.initialize();
    startClock();
    applyEchoState("active");
    applyTelemetryMock("idle_local");
    setTelemetryPanel(window.echoFreeWorkspace.getInitialTelemetryPanelOpen());
    setInputEnabled(false);
    bindDomEvents();
    window.addEventListener("resize", () => {
      scheduleAdaptiveLayout();
    });
    initializeEchoChannel();
  });
})();

(function () {
  let controller = null;
  let controllerReady = false;
  let requestActive = false;
  let replyFadeToken = 0;
  let telemetryPanelOpen = false;
  let telemetryMode = "local";
  let telemetryState = "idle_local";
  let telemetryRoutingTimer = null;

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
    tool_result: "Local tool"
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
      note: byId("telemetryNote")
    };
  }

  function updateTelemetryButtons() {
    document.querySelectorAll("[data-telemetry-mode]").forEach((button) => {
      button.classList.toggle("active", button.dataset.telemetryMode === telemetryMode);
    });
  }

  function applyTelemetryMock(stateName) {
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
    if (elements.note) elements.note.textContent = data.note;
    updateTelemetryButtons();
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

  function runAutomaticRoutingMock() {
    window.clearTimeout(telemetryRoutingTimer);
    applyTelemetryMock("routing_automatic");
    telemetryRoutingTimer = window.setTimeout(() => applyTelemetryMock("thinking_cloud"), 650);
    telemetryRoutingTimer = window.setTimeout(() => applyTelemetryMock("response_ready"), 1250);
  }

  function setTelemetryMode(mode) {
    const value = String(mode || "").trim().toLowerCase();
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
      if (lockButton) {
        lockButton.setAttribute("aria-pressed", state.locked ? "true" : "false");
        lockButton.textContent = state.locked ? "UNLOCK POSITION" : "LOCK POSITION";
      }
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

    const responseTitle = byId("responseTitle");
    const bridgeReply = byId("bridgeReply");
    const workspaceHint = byId("workspaceHint");
    const clearButton = byId("clearButton");

    if (responseTitle) responseTitle.textContent = "Echo";
    if (bridgeReply) bridgeReply.textContent = value;
    if (workspaceHint) workspaceHint.classList.remove("visible");
    if (clearButton) clearButton.classList.add("visible");
    if (telemetryMode === "automatic" || telemetryState === "thinking_cloud") {
      applyTelemetryMock("response_ready");
    } else {
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
    if (telemetryMode === "automatic") runAutomaticRoutingMock();
    else applyTelemetryMock("thinking_local");
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
          echoResponse.hidden = false;
          echoResponse.style.display = "";
          echoResponse.style.visibility = "";
          echoResponse.style.opacity = "";
          echoResponse.classList.remove("visible", "muted");
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
        const modeButton = event.target.closest("button[data-telemetry-mode]");
        if (!modeButton) return;
        setTelemetryMode(modeButton.dataset.telemetryMode);
      });
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
      if (event.key === "Escape" && telemetryPanelOpen) {
        event.preventDefault();
        setTelemetryPanel(false);
        focusInput();
      }
    });
  }

  window.addEventListener("DOMContentLoaded", () => {
    console.log("[Echo UI JS] DOMContentLoaded");
    console.log("[Echo UI JS] echoResponse no arranque:", byId("echoResponse"));
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

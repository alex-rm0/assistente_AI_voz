(function () {
  let controller = null;
  let controllerReady = false;
  let requestActive = false;
  let replyFadeToken = 0;

  const stateLabels = {
    idle: ["ACTIVE · ECHO OS", "#5a5d6b"],
    thinking: ["THINKING", "#6ea8ff"],
    speaking: ["SPEAKING", "#8fd0c4"],
    error: ["ERROR", "#f08a8a"]
  };

  function byId(id) {
    return document.getElementById(id);
  }

  function resizeStage() {
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
    const suggestions = byId("suggestions");

    if (responseTitle) responseTitle.textContent = "Echo";
    if (bridgeReply) bridgeReply.textContent = value;
    if (workspaceHint) workspaceHint.classList.remove("visible");
    if (clearButton) clearButton.classList.add("visible");
    if (suggestions) suggestions.classList.add("hidden");

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
    const suggestions = byId("suggestions");
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
    if (suggestions) suggestions.classList.add("hidden");
    fadeCurrentReply();
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
      const suggestions = byId("suggestions");
      if (suggestions) suggestions.classList.add("hidden");
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
    const suggestions = byId("suggestions");
    const clearButton = byId("clearButton");
    const debugStates = document.querySelector(".debug-states");
    const closeResearch = byId("closeResearch");

    if (form) {
      form.addEventListener("submit", (event) => {
        event.preventDefault();
        const input = byId("voiceIn");
        sendMessage(input ? input.value : "");
      });
    }

    if (suggestions) {
      suggestions.addEventListener("click", (event) => {
        const button = event.target.closest("button[data-message]");
        if (!button) return;
        sendMessage(button.dataset.message);
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
        if (suggestions) suggestions.classList.remove("hidden");
        speak("");
        if (controller && typeof controller.clearConversation === "function") controller.clearConversation();
        else if (controller) controller.setState("idle");
        else applyEchoState("idle");
        focusInput();
      });
    }

    if (debugStates) {
      debugStates.addEventListener("click", (event) => {
        const button = event.target.closest("button[data-state]");
        if (!button) return;
        if (controller) controller.setState(button.dataset.state);
        else applyEchoState(button.dataset.state);
        if (button.dataset.state === "speaking") speak("Estou aqui.");
        if (button.dataset.state === "error") speak("Houve um erro na interface.");
      });
    }

    if (closeResearch) {
      closeResearch.addEventListener("click", () => {
        if (window.echoWorkspace) window.echoWorkspace.clear();
        focusInput();
      });
    }

    window.addEventListener("keydown", (event) => {
      const input = byId("voiceIn");
      if (["1", "2", "3", "4"].includes(event.key) && document.activeElement !== input) {
        const states = { "1": "idle", "2": "thinking", "3": "speaking", "4": "error" };
        if (controller) controller.setState(states[event.key]);
        else applyEchoState(states[event.key]);
      }
    });
  }

  window.addEventListener("DOMContentLoaded", () => {
    console.log("[Echo UI JS] DOMContentLoaded");
    console.log("[Echo UI JS] echoResponse no arranque:", byId("echoResponse"));
    window.echoEntity = new window.EchoEntity(byId("mind"));
    window.echoWorkspace = createWorkspaceController();
    resizeStage();
    applyEchoState("idle");
    setInputEnabled(false);
    bindDomEvents();
    window.addEventListener("resize", resizeStage);
    initializeEchoChannel();
  });
})();

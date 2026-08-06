(function () {
  // Single central control point for how each semantic state looks/moves.
  // Every per-state visual difference (speed, density/decay, glow, color,
  // ambient behaviour) lives here instead of being scattered across loop()/
  // draw() as ad-hoc ternaries -- see STATE_CONFIG below.
  //
  // driftScale multiplies the ambient wander of the entity's center point;
  // pulseChance/statePulse/nodeDecay/edgeDecay/spin keep their original
  // meaning (see loop()); sparkOnEnter is one of "focus" (spark a persistent
  // focus node, re-sparked periodically -- used by thinking/working),
  // "goal" (boost the goal node once -- listening/speaking) or "random"
  // (one-shot scatter spark -- error); accent is the [r,g,b] glow color.
  const STATE_CONFIG = {
    idle: {
      spin: 0.14, statePulse: 0.022, nodeDecay: 1.55, edgeDecay: 1.4,
      pulseChance: 2.2, driftScale: 1.0, ambientSparkRate: 0, periodicFocusSpark: false,
      accent: [70, 197, 255], outerAlpha: 0.20, coreAlpha: 0.50, sparkOnEnter: null
    },
    listening: {
      spin: 0.15, statePulse: 0.024, nodeDecay: 1.7, edgeDecay: 1.6,
      pulseChance: 1.8, driftScale: 0.85, ambientSparkRate: 0, periodicFocusSpark: false,
      accent: [102, 205, 255], outerAlpha: 0.21, coreAlpha: 0.52, sparkOnEnter: "goal"
    },
    thinking: {
      spin: 0.20, statePulse: 0.030, nodeDecay: 2.3, edgeDecay: 2.6,
      pulseChance: 1.3, driftScale: 0.4, ambientSparkRate: 0, periodicFocusSpark: true,
      accent: [70, 197, 255], outerAlpha: 0.24, coreAlpha: 0.58, sparkOnEnter: "focus"
    },
    reading: {
      spin: 0.12, statePulse: 0.016, nodeDecay: 1.4, edgeDecay: 1.3,
      pulseChance: 1.0, driftScale: 0.5, ambientSparkRate: 0, periodicFocusSpark: false,
      accent: [70, 197, 255], outerAlpha: 0.18, coreAlpha: 0.46, sparkOnEnter: null
    },
    working: {
      spin: 0.15, statePulse: 0.028, nodeDecay: 2.0, edgeDecay: 2.3,
      pulseChance: 1.5, driftScale: 0.35, ambientSparkRate: 0, periodicFocusSpark: true,
      accent: [70, 197, 255], outerAlpha: 0.23, coreAlpha: 0.55, sparkOnEnter: "focus"
    },
    speaking: {
      spin: 0.16, statePulse: 0.025, nodeDecay: 1.9, edgeDecay: 2.2,
      pulseChance: 0.8, driftScale: 1.1, ambientSparkRate: 0.8, periodicFocusSpark: false,
      accent: [143, 208, 196], outerAlpha: 0.20, coreAlpha: 0.54, sparkOnEnter: "goal"
    },
    error: {
      spin: 0.08, statePulse: 0.014, nodeDecay: 1.2, edgeDecay: 1.0,
      pulseChance: 0.6, driftScale: 0.15, ambientSparkRate: 0, periodicFocusSpark: false,
      accent: [240, 138, 138], outerAlpha: 0.16, coreAlpha: 0.50, sparkOnEnter: "random"
    }
  };

  // Spatial roles (compact/focus) are not cognitive states -- they combine
  // with whatever STATE_CONFIG mode is active by scaling intensity down a
  // little, so Echo draws less attention when the layout needs the space.
  const ROLE_INTENSITY = { normal: 1, compact: 0.85, focus: 0.75 };

  const TRANSITION_KEYS = [
    "spin", "statePulse", "nodeDecay", "edgeDecay", "pulseChance",
    "driftScale", "outerAlpha", "coreAlpha"
  ];
  // How quickly live parameters ease toward the target state's config --
  // higher = snappier. Kept state-independent and moderate so every
  // transition (idle->thinking, working->error, etc.) reads as a glide
  // rather than a jump, without ever feeling sluggish to respond.
  const TRANSITION_RATE = 2.4;

  class EchoEntity {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.dpr = Math.min(window.devicePixelRatio || 1, 2);
      this.nodes = [];
      this.edges = [];
      this.pulses = [];
      this.state = "idle";
      this.intensityBoost = 1;
      this.layoutRole = "normal";
      this.yaw = 0;
      this.pitch = 0.42;
      this.t = 0;
      this.last = performance.now();
      this.nextAmbient = 2.5;
      this.nextThinkingSpark = 0;
      this.thinkingFocus = null;
      this.cx = 0;
      this.cy = 0;
      this.cxTarget = 0;
      this.cyTarget = 0;
      this.customCenter = false;
      this.R = 160;

      // prefers-reduced-motion reaches only CSS transitions/animations by
      // default -- this canvas loop runs on its own requestAnimationFrame
      // clock, so it needs its own check. Motion (spin/pulse/drift/ambient
      // sparking) is reduced, not eliminated: Echo must keep reading as
      // alive, just calmer (section 15).
      this.motionMedia = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;
      this.reducedMotion = Boolean(this.motionMedia && this.motionMedia.matches);
      if (this.motionMedia) {
        const onChange = (event) => { this.reducedMotion = Boolean(event.matches); };
        if (typeof this.motionMedia.addEventListener === "function") this.motionMedia.addEventListener("change", onChange);
        else if (typeof this.motionMedia.addListener === "function") this.motionMedia.addListener(onChange);
      }

      this.liveParams = { ...STATE_CONFIG.idle };
      this.liveAccent = [...STATE_CONFIG.idle.accent];

      this.resize();
      this.buildMind();
      window.addEventListener("resize", () => this.resize());
      requestAnimationFrame((now) => this.loop(now));
    }

    resize() {
      const rect = this.canvas.getBoundingClientRect();
      this.w = rect.width || 1300;
      this.h = rect.height || 812;
      this.canvas.width = Math.round(this.w * this.dpr);
      this.canvas.height = Math.round(this.h * this.dpr);
      this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      this.R = this.responsiveRadius();
      if (!this.cxTarget) {
        this.cxTarget = this.w * 0.5;
        this.cyTarget = this.h * 0.5;
        this.cx = this.cxTarget;
        this.cy = this.cyTarget;
      }
    }

    responsiveRadius() {
      const stage = this.canvas.closest(".stage");
      const raw = stage ? getComputedStyle(stage).getPropertyValue("--entity-size-actual") : "";
      const size = Number.parseFloat(raw);
      if (!Number.isFinite(size) || size <= 0) return 160;
      return Math.max(70, Math.min(220, size / 2));
    }

    buildMind() {
      this.nodes = [];
      this.edges = [];
      this.addNode("goal", false);
      const types = ["memory", "project", "person", "place", "habit"];
      for (let i = 0; i < 108; i += 1) {
        const node = this.addNode(types[(Math.random() * types.length) | 0], Math.random() < 0.66);
        this.link(this.preferentialNode(), node);
        if (this.nodes.length > 6 && Math.random() < 0.36) this.linkNearest(node);
      }
      this.layout();
      for (const n of this.nodes) {
        n.pos = [...n.dir];
      }
      this.projectAll();
    }

    addNode(type, shell) {
      const isGoal = type === "goal";
      const n = {
        id: this.nodes.length,
        type,
        shell: isGoal ? false : shell,
        rFrac: isGoal ? 0.14 : shell ? 1 : 0.4 + Math.random() * 0.36,
        dir: [0, 1, 0],
        pos: [0, 1, 0],
        sx: undefined,
        sy: undefined,
        sz: 0,
        ss: 1,
        act: isGoal ? 0.8 : 0.25,
        grow: 1,
        lagK: 0.32 + Math.random() * 0.55
      };
      this.nodes.push(n);
      return n;
    }

    link(a, b, base) {
      if (a && b && a !== b) this.edges.push({ a, b, base: base || 0.5 + Math.random() * 0.4, act: 0.25, fade: 1 });
    }

    degree(node) {
      return this.edges.reduce((total, e) => total + (e.a === node || e.b === node ? 1 : 0), 0);
    }

    preferentialNode() {
      let total = 0;
      const weights = this.nodes.map((n) => {
        const w = this.degree(n) + 1;
        total += w;
        return w;
      });
      let roll = Math.random() * total;
      for (let i = 0; i < this.nodes.length; i += 1) {
        roll -= weights[i];
        if (roll <= 0) return this.nodes[i];
      }
      return this.nodes[0];
    }

    linkNearest(node) {
      let best = null;
      let bestDistance = Infinity;
      for (const other of this.nodes) {
        if (other === node) continue;
        const dx = other.dir[0] - node.dir[0];
        const dy = other.dir[1] - node.dir[1];
        const dz = other.dir[2] - node.dir[2];
        const d = dx * dx + dy * dy + dz * dz;
        if (d < bestDistance) {
          bestDistance = d;
          best = other;
        }
      }
      if (best) this.link(node, best, 0.32);
    }

    layout() {
      const fib = (i, total) => {
        const y = 1 - (2 * (i + 0.5)) / Math.max(1, total);
        const r = Math.sqrt(Math.max(0, 1 - y * y));
        const theta = i * 2.399963;
        return [Math.cos(theta) * r, y, Math.sin(theta) * r];
      };
      const shell = this.nodes.filter((n) => n.rFrac >= 0.99);
      const inner = this.nodes.filter((n) => n.rFrac < 0.99);
      shell.forEach((n, i) => { n.dir = fib(i, shell.length); });
      inner.forEach((n, i) => {
        const d = fib(i, inner.length);
        n.dir = [d[0] * n.rFrac, d[1] * n.rFrac, d[2] * n.rFrac];
      });
    }

    // Central entry point: translate a semantic state name into the
    // STATE_CONFIG entry that drives loop()/draw(); unknown state strings
    // fall back to "idle" rather than silently rendering nothing special.
    // intensityBoost lets a sub-state (e.g. rewrite regeneration) push the
    // same "working" mode a little harder without switching mode.
    setState(state, options = {}) {
      const next = String(state || "idle").trim().toLowerCase();
      this.state = STATE_CONFIG[next] ? next : "idle";
      this.intensityBoost = Number.isFinite(options.intensityBoost) ? options.intensityBoost : 1;
      const config = STATE_CONFIG[this.state];
      if (config.sparkOnEnter === "focus") {
        this.thinkingFocus = this.nodes[(Math.random() * this.nodes.length) | 0];
        this.nextThinkingSpark = 0;
        this.spark(this.thinkingFocus, 1);
      } else if (config.sparkOnEnter === "goal") {
        const goal = this.nodes[0];
        if (goal) goal.act = Math.max(goal.act, 0.72);
      } else if (config.sparkOnEnter === "random") {
        this.sparkRandom(1, 0);
      }
      if (!this.customCenter) {
        this.cxTarget = this.w * 0.5;
        this.cyTarget = this.h * 0.5;
      }
    }

    // Spatial role reported by the Adaptive Layout System (normal/compact/
    // focus) -- purely dampens intensity, never touches size/position
    // (those stay owned by CSS --entity-size-actual and setCenter/layout).
    setLayoutRole(role) {
      const next = String(role || "normal").trim().toLowerCase();
      this.layoutRole = Object.prototype.hasOwnProperty.call(ROLE_INTENSITY, next) ? next : "normal";
    }

    setCenter(x, y, immediate = false) {
      this.customCenter = true;
      this.cxTarget = Number.isFinite(x) ? x : this.w * 0.5;
      this.cyTarget = Number.isFinite(y) ? y : this.h * 0.5;
      if (immediate) {
        this.cx = this.cxTarget;
        this.cy = this.cyTarget;
      }
    }

    clearCustomCenter() {
      this.customCenter = false;
      this.cxTarget = this.w * 0.5;
      this.cyTarget = this.h * 0.5;
    }

    getCenter() {
      return { x: this.cxTarget || this.w * 0.5, y: this.cyTarget || this.h * 0.5 };
    }

    sparkRandom(count, depth = 1) {
      for (let i = 0; i < count; i += 1) {
        this.spark(this.nodes[(Math.random() * this.nodes.length) | 0], depth);
      }
    }

    spark(node, depth) {
      if (!node || depth < 0) return;
      node.act = Math.min(0.9, node.act + 0.45);
      for (const e of this.edges) {
        if (e.a !== node && e.b !== node) continue;
        e.act = Math.max(e.act, 0.55);
        const other = e.a === node ? e.b : e.a;
        this.pulses.push({ from: node, to: other, p: 0, sp: 0.95 + Math.random() * 0.45, s: 0.38 });
        if (depth > 0 && Math.random() < 0.22) this.spark(other, depth - 1);
      }
    }

    project(point) {
      const sy = Math.sin(this.yaw);
      const cyaw = Math.cos(this.yaw);
      let x = point[0] * cyaw - point[2] * sy;
      let z = point[0] * sy + point[2] * cyaw;
      const sp = Math.sin(this.pitch);
      const cp = Math.cos(this.pitch);
      const y = point[1] * cp - z * sp;
      z = point[1] * sp + z * cp;
      const f = 3.4;
      const per = f / (f - z);
      return { x: this.cx + x * this.R * per, y: this.cy + y * this.R * per, z, s: per };
    }

    projectAll(dt) {
      const frame = dt ? Math.min(1, dt * 60) : null;
      for (const n of this.nodes) {
        const p = this.project(n.pos);
        n.sz = p.z;
        n.ss = p.s;
        if (n.sx === undefined || frame === null) {
          n.sx = p.x;
          n.sy = p.y;
        } else {
          const k = Math.min(1, 0.12 * n.lagK * frame);
          n.sx += (p.x - n.sx) * k;
          n.sy += (p.y - n.sy) * k;
        }
      }
    }

    // Eases this.liveParams/this.liveAccent toward the current state's
    // target config (scaled by intensityBoost, layoutRole and reduced-
    // motion) so switching state glides rather than jumps -- see
    // TRANSITION_RATE/TRANSITION_KEYS above.
    stepLiveParams(dt) {
      const config = STATE_CONFIG[this.state] || STATE_CONFIG.idle;
      const roleScale = ROLE_INTENSITY[this.layoutRole] ?? 1;
      const motionScale = this.reducedMotion ? 0.4 : 1;
      const boost = this.intensityBoost || 1;
      const ease = Math.min(1, dt * TRANSITION_RATE);
      for (const key of TRANSITION_KEYS) {
        let target = config[key];
        if (key === "outerAlpha" || key === "coreAlpha" || key === "statePulse") target *= roleScale * boost;
        if (key === "spin" || key === "pulseChance" || key === "driftScale") target *= motionScale;
        this.liveParams[key] += (target - this.liveParams[key]) * ease;
      }
      for (let i = 0; i < 3; i += 1) {
        this.liveAccent[i] += (config.accent[i] - this.liveAccent[i]) * ease;
      }
    }

    loop(now) {
      const dt = Math.min(0.05, (now - this.last) / 1000);
      this.last = now;
      this.t += dt;
      this.stepLiveParams(dt);
      const config = STATE_CONFIG[this.state] || STATE_CONFIG.idle;
      const p = this.liveParams;

      this.yaw += dt * p.spin;
      const drift = p.driftScale;
      this.cx += (this.cxTarget + Math.sin(this.t * 0.12) * this.w * 0.006 * drift - this.cx) * Math.min(1, dt * 1.5);
      this.cy += (this.cyTarget + Math.cos(this.t * 0.1) * this.h * 0.006 * drift - this.cy) * Math.min(1, dt * 1.5);
      this.R = (52 + 11 * Math.sqrt(this.nodes.length)) * (1 + p.statePulse * Math.sin(this.t * 1.2));

      for (const n of this.nodes) {
        const k = Math.min(1, dt * 3);
        n.pos[0] += (n.dir[0] - n.pos[0]) * k;
        n.pos[1] += (n.dir[1] - n.pos[1]) * k;
        n.pos[2] += (n.dir[2] - n.pos[2]) * k;
        n.act *= Math.max(0, 1 - dt * p.nodeDecay);
      }
      for (const e of this.edges) e.act *= Math.max(0, 1 - dt * p.edgeDecay);

      this.nextAmbient -= dt;
      if (this.nextAmbient <= 0) {
        this.nextAmbient = 3.5 + Math.random() * 4;
        this.spark(this.nodes[(Math.random() * this.nodes.length) | 0], 1);
      }

      this.nextThinkingSpark -= dt;
      if (config.periodicFocusSpark && this.nextThinkingSpark <= 0) {
        const focus = this.thinkingFocus || this.nodes[(Math.random() * this.nodes.length) | 0];
        this.spark(focus, 1);
        this.nextThinkingSpark = 0.45 + Math.random() * 0.45;
      }
      if (config.ambientSparkRate > 0 && Math.random() < dt * config.ambientSparkRate * (this.reducedMotion ? 0.4 : 1)) {
        this.sparkRandom(1, 0);
      }

      const pulseChance = p.pulseChance * dt;
      if (Math.random() < pulseChance && this.edges.length) {
        const e = this.edges[(Math.random() * this.edges.length) | 0];
        this.pulses.push({ from: e.a, to: e.b, p: 0, sp: 0.55 + Math.random() * 0.7, s: 0.32 });
      }
      for (const pulse of this.pulses) pulse.p += dt * pulse.sp;
      this.pulses = this.pulses.filter((pulse) => pulse.p < 1);
      this.projectAll(dt);
      this.draw();
      requestAnimationFrame((next) => this.loop(next));
    }

    draw() {
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.w, this.h);
      const accent = this.liveAccent;
      const p = this.liveParams;
      ctx.globalCompositeOperation = "lighter";
      const outerAlpha = p.outerAlpha;
      const coreAlpha = p.coreAlpha;
      const outer = ctx.createRadialGradient(this.cx, this.cy - this.R * 0.15, 0, this.cx, this.cy, this.R * 1.3);
      outer.addColorStop(0, `rgba(${accent[0]},${accent[1]},${accent[2]},${outerAlpha})`);
      outer.addColorStop(0.6, "rgba(40,64,120,0.08)");
      outer.addColorStop(1, "rgba(30,48,92,0)");
      ctx.fillStyle = outer;
      ctx.beginPath();
      ctx.arc(this.cx, this.cy, this.R * 1.3, 0, Math.PI * 2);
      ctx.fill();
      const core = ctx.createRadialGradient(this.cx, this.cy, 0, this.cx, this.cy, this.R);
      core.addColorStop(0, `rgba(${accent[0]},${accent[1]},${accent[2]},${coreAlpha})`);
      core.addColorStop(0.45, `rgba(${accent[0]},${accent[1]},${accent[2]},0.14)`);
      core.addColorStop(1, `rgba(${accent[0]},${accent[1]},${accent[2]},0)`);
      ctx.fillStyle = core;
      ctx.beginPath();
      ctx.arc(this.cx, this.cy, this.R, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalCompositeOperation = "source-over";
      ctx.strokeStyle = "rgba(140,180,255,0.12)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(this.cx, this.cy, this.R, 0, Math.PI * 2);
      ctx.stroke();
      for (const e of this.edges) {
        const depth = ((e.a.sz + e.b.sz) / 2 + 1) / 2;
        const glow = Math.min(1, e.act);
        const alpha = (0.04 + 0.09 * e.base) * (0.35 + 0.65 * depth) + 0.16 * glow;
        ctx.strokeStyle = glow > 0.05 ? `rgba(170,205,255,${alpha})` : `rgba(140,160,205,${alpha})`;
        ctx.lineWidth = 0.7 + 0.35 * glow;
        ctx.beginPath();
        ctx.moveTo(e.a.sx, e.a.sy);
        ctx.lineTo(e.b.sx, e.b.sy);
        ctx.stroke();
      }
      ctx.globalCompositeOperation = "lighter";
      for (const pulse of this.pulses) {
        const x = pulse.from.sx + (pulse.to.sx - pulse.from.sx) * pulse.p;
        const y = pulse.from.sy + (pulse.to.sy - pulse.from.sy) * pulse.p;
        const rad = 4 * pulse.s;
        const g = ctx.createRadialGradient(x, y, 0, x, y, rad);
        g.addColorStop(0, `rgba(226,240,255,${0.55 * pulse.s})`);
        g.addColorStop(1, "rgba(226,240,255,0)");
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(x, y, rad, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalCompositeOperation = "source-over";
      const ordered = [...this.nodes].sort((a, b) => a.sz - b.sz);
      for (const n of ordered) {
        const depth = (n.sz + 1) / 2;
        const act = Math.min(1.3, n.act);
        const isGoal = n.type === "goal";
        const color = isGoal ? [170, 232, 255] : accent;
        const radius = (isGoal ? 3.3 : 2.15) * n.ss * (0.72 + 0.5 * depth) * (1 + act * 0.25);
        if (act > 0.06) {
          const glowRadius = radius + 6 * act;
          const g = ctx.createRadialGradient(n.sx, n.sy, 0, n.sx, n.sy, glowRadius);
          g.addColorStop(0, `rgba(${color[0]},${color[1]},${color[2]},${0.32 * act})`);
          g.addColorStop(1, `rgba(${color[0]},${color[1]},${color[2]},0)`);
          ctx.globalCompositeOperation = "lighter";
          ctx.fillStyle = g;
          ctx.beginPath();
          ctx.arc(n.sx, n.sy, glowRadius, 0, Math.PI * 2);
          ctx.fill();
          ctx.globalCompositeOperation = "source-over";
        }
        ctx.fillStyle = `rgba(${color[0]},${color[1]},${color[2]},${0.45 + 0.55 * depth})`;
        ctx.beginPath();
        ctx.arc(n.sx, n.sy, Math.max(0.7, radius), 0, Math.PI * 2);
        ctx.fill();
      }
      const heart = ctx.createRadialGradient(this.cx, this.cy, 0, this.cx, this.cy, 8);
      heart.addColorStop(0, "rgba(120,220,255,0.75)");
      heart.addColorStop(1, "rgba(120,220,255,0)");
      ctx.globalCompositeOperation = "lighter";
      ctx.fillStyle = heart;
      ctx.beginPath();
      ctx.arc(this.cx, this.cy, 8, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalCompositeOperation = "source-over";
      ctx.fillStyle = "rgba(210,244,255,0.95)";
      ctx.beginPath();
      ctx.arc(this.cx, this.cy, 2.7, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  window.EchoEntity = EchoEntity;
  window.ECHO_STATE_CONFIG = STATE_CONFIG;
})();

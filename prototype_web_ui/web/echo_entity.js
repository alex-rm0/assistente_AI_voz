(function () {
  class EchoEntity {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.dpr = Math.min(window.devicePixelRatio || 1, 2);
      this.nodes = [];
      this.edges = [];
      this.pulses = [];
      this.state = "idle";
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
      if (!this.cxTarget) {
        this.cxTarget = this.w * 0.5;
        this.cyTarget = this.h * 0.5;
        this.cx = this.cxTarget;
        this.cy = this.cyTarget;
      }
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

    setState(state) {
      this.state = state || "idle";
      if (this.state === "thinking") {
        this.thinkingFocus = this.nodes[(Math.random() * this.nodes.length) | 0];
        this.nextThinkingSpark = 0;
        this.spark(this.thinkingFocus, 1);
      } else if (this.state === "speaking") {
        const goal = this.nodes[0];
        if (goal) goal.act = Math.max(goal.act, 0.72);
      } else if (this.state === "error") {
        this.sparkRandom(1, 0);
      }
      if (!this.customCenter) {
        this.cxTarget = this.w * 0.5;
        this.cyTarget = this.h * 0.5;
      }
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

    loop(now) {
      const dt = Math.min(0.05, (now - this.last) / 1000);
      this.last = now;
      this.t += dt;
      const spin = this.state === "thinking" ? 0.17 : this.state === "speaking" ? 0.16 : 0.14;
      this.yaw += dt * spin;
      this.cx += (this.cxTarget + Math.sin(this.t * 0.12) * this.w * 0.006 - this.cx) * Math.min(1, dt * 1.5);
      this.cy += (this.cyTarget + Math.cos(this.t * 0.1) * this.h * 0.006 - this.cy) * Math.min(1, dt * 1.5);
      const statePulse = this.state === "thinking" ? 0.026 : this.state === "speaking" ? 0.025 : 0.022;
      this.R = (52 + 11 * Math.sqrt(this.nodes.length)) * (1 + statePulse * Math.sin(this.t * 1.2));
      const nodeDecay = this.state === "thinking" ? 2.1 : this.state === "speaking" ? 1.9 : 1.55;
      const edgeDecay = this.state === "thinking" ? 2.5 : this.state === "speaking" ? 2.2 : 1.4;
      for (const n of this.nodes) {
        const k = Math.min(1, dt * 3);
        n.pos[0] += (n.dir[0] - n.pos[0]) * k;
        n.pos[1] += (n.dir[1] - n.pos[1]) * k;
        n.pos[2] += (n.dir[2] - n.pos[2]) * k;
        n.act *= Math.max(0, 1 - dt * nodeDecay);
      }
      for (const e of this.edges) e.act *= Math.max(0, 1 - dt * edgeDecay);
      this.nextAmbient -= dt;
      if (this.nextAmbient <= 0) {
        this.nextAmbient = 3.5 + Math.random() * 4;
        this.spark(this.nodes[(Math.random() * this.nodes.length) | 0], 1);
      }
      this.nextThinkingSpark -= dt;
      if (this.state === "thinking" && this.nextThinkingSpark <= 0) {
        const focus = this.thinkingFocus || this.nodes[(Math.random() * this.nodes.length) | 0];
        this.spark(focus, 1);
        this.nextThinkingSpark = 0.45 + Math.random() * 0.45;
      }
      if (this.state === "speaking" && Math.random() < dt * 0.8) this.sparkRandom(1, 0);
      const pulseChance = this.state === "thinking" ? dt * 1.2 : this.state === "speaking" ? dt * 0.8 : dt * 2.2;
      if (Math.random() < pulseChance && this.edges.length) {
        const e = this.edges[(Math.random() * this.edges.length) | 0];
        this.pulses.push({ from: e.a, to: e.b, p: 0, sp: 0.55 + Math.random() * 0.7, s: 0.32 });
      }
      for (const p of this.pulses) p.p += dt * p.sp;
      this.pulses = this.pulses.filter((p) => p.p < 1);
      this.projectAll(dt);
      this.draw();
      requestAnimationFrame((next) => this.loop(next));
    }

    draw() {
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.w, this.h);
      const accent = this.state === "error" ? [240, 138, 138] : this.state === "speaking" ? [143, 208, 196] : [70, 197, 255];
      ctx.globalCompositeOperation = "lighter";
      const outerAlpha = this.state === "thinking" ? 0.22 : this.state === "error" ? 0.16 : 0.20;
      const coreAlpha = this.state === "thinking" ? 0.56 : this.state === "speaking" ? 0.54 : 0.50;
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
      for (const p of this.pulses) {
        const x = p.from.sx + (p.to.sx - p.from.sx) * p.p;
        const y = p.from.sy + (p.to.sy - p.from.sy) * p.p;
        const rad = 4 * p.s;
        const g = ctx.createRadialGradient(x, y, 0, x, y, rad);
        g.addColorStop(0, `rgba(226,240,255,${0.55 * p.s})`);
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
})();

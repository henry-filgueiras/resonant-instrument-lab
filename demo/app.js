// Static regime-oracle viewer, A/B comparison mode. Loads up to two
// summary.json files (file picker, paste box, or ?summaryA=URL / ?summaryB=URL
// query params — fetch only works when served, not file://) and renders each
// plus a detector-flip / stat-delta comparison card. Each slot also renders a
// small topology/body view from a sibling topology.json when available (fetched
// next to the summary URL, or loaded via the per-slot "+ topology.json" picker
// under file://). No framework, no build.

const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

const fmtFloat = (x, digits = 3) =>
  typeof x === "number" && Number.isFinite(x) ? x.toFixed(digits) : "—";

const fmtSeconds = (x, digits = 2) =>
  typeof x === "number" && Number.isFinite(x) ? `${x.toFixed(digits)} s` : "—";

const fmtSep = (x, digits = 2) =>
  x == null ? "inf" : fmtFloat(x, digits);

const signed = (d, digits = 3) =>
  `${d >= 0 ? "+" : ""}${d.toFixed(digits)}`;

function kvRow(k, v) {
  const row = el("div", "row");
  row.append(el("span", "k", k), el("span", "v mono", v));
  return row;
}

// --- per-slot state ---
const state = { A: null, B: null };
const topos = { A: null, B: null };

function getSlotEls(slotEl) {
  const q = (role) => slotEl.querySelector(`[data-role="${role}"]`);
  return {
    root: slotEl,
    file: q("file"),
    topoFile: q("topo-file"),
    pasteToggle: q("paste-toggle"),
    pastePanel: q("paste-panel"),
    pasteArea: q("paste-area"),
    pasteApply: q("paste-apply"),
    status: q("status"),
    empty: q("empty"),
    summary: q("summary"),
    metaName: q("meta-name"),
    metaDuration: q("meta-duration"),
    metaN: q("meta-n"),
    metaRate: q("meta-rate"),
    metaSchema: q("meta-schema"),
    metaPath: q("meta-path"),
    topoCard: q("topology-card"),
    topoStage: q("topology-stage"),
    topoFooter: q("topology-footer"),
    dynamicsCard: q("dynamics-card"),
    rStage: q("r-stage"),
    rasterStage: q("raster-stage"),
    dynamicsFooter: q("dynamics-footer"),
    detectors: q("detectors"),
    stats: q("stats"),
    rawJson: q("raw-json"),
  };
}

function setStatus(els, msg, kind) {
  els.status.textContent = msg || "";
  if (kind) els.status.dataset.kind = kind;
  else delete els.status.dataset.kind;
}

// --- topology ---

const SVG_NS = "http://www.w3.org/2000/svg";
const svg = (tag, attrs) => {
  const n = document.createElementNS(SVG_NS, tag);
  if (attrs) for (const k in attrs) n.setAttribute(k, attrs[k]);
  return n;
};

function inferTopologyUrl(summaryUrl) {
  if (typeof summaryUrl !== "string") return null;
  // Strip query/fragment, then replace the trailing "summary.json" basename.
  const [path] = summaryUrl.split(/[?#]/, 1);
  const m = path.match(/^(.*\/)?summary\.json$/);
  if (!m) return null;
  return (m[1] ?? "") + "topology.json";
}

// Map omega (in the topology's min..max range) to a subtle accent ramp.
// Green → teal → blue: lets two_cluster's frequency split read at a glance
// without turning into a rainbow. Single-frequency worlds get flat accent.
function nodeFill(t) {
  // t in [0, 1]; clamp for safety.
  const u = Math.max(0, Math.min(1, t));
  // HSL 160 (green) → 210 (blue), consistent lightness.
  const hue = 160 + u * 50;
  return `hsl(${hue.toFixed(1)}, 55%, 58%)`;
}

function renderTopology(slotKey) {
  const slotEl = document.querySelector(`.slot[data-slot="${slotKey}"]`);
  if (!slotEl) return;
  const els = getSlotEls(slotEl);
  const topo = topos[slotKey];

  els.topoStage.innerHTML = "";
  els.topoFooter.innerHTML = "";

  if (!topo || !Array.isArray(topo.nodes) || topo.nodes.length === 0) {
    els.topoCard.dataset.state = "missing";
    els.topoStage.append(el("p", "topology-missing", "topology unavailable"));
    return;
  }
  els.topoCard.dataset.state = "ok";

  // Geometry: 100×100 viewBox with an inner padded area so dots near the
  // edges don't clip and index labels have room.
  const VB = 100, PAD = 8;
  const inner = VB - 2 * PAD;
  const sx = (x) => PAD + x * inner;
  const sy = (y) => PAD + (1 - y) * inner; // flip: config y=0 is bottom

  const root = svg("svg", { viewBox: `0 0 ${VB} ${VB}`, role: "img" });
  root.setAttribute("aria-label", `topology: ${topo.nodes.length} nodes`);

  // Frame + faint centerlines.
  root.append(svg("rect", {
    class: "topo-frame",
    x: PAD, y: PAD, width: inner, height: inner, rx: 1,
  }));
  root.append(svg("line", {
    class: "topo-grid",
    x1: PAD + inner / 2, y1: PAD,
    x2: PAD + inner / 2, y2: PAD + inner,
  }));
  root.append(svg("line", {
    class: "topo-grid",
    x1: PAD,          y1: PAD + inner / 2,
    x2: PAD + inner,  y2: PAD + inner / 2,
  }));

  // Frequency range for subtle per-node tinting.
  const omegas = topo.nodes
    .map((n) => n.omega_0_hz)
    .filter((v) => typeof v === "number" && Number.isFinite(v));
  const wMin = omegas.length ? Math.min(...omegas) : 0;
  const wMax = omegas.length ? Math.max(...omegas) : 1;
  const wSpan = wMax - wMin > 1e-9 ? wMax - wMin : 1;

  for (const node of topo.nodes) {
    const p = node && node.pos;
    if (!Array.isArray(p) || p.length < 2) continue;
    const cx = sx(+p[0]);
    const cy = sy(+p[1]);
    const t = typeof node.omega_0_hz === "number"
      ? (node.omega_0_hz - wMin) / wSpan : 0.5;
    root.append(svg("circle", {
      class: "topo-node",
      cx, cy, r: 2.6,
      fill: nodeFill(t),
    }));
    const idx = Number.isInteger(node.index) ? node.index : "";
    const lbl = svg("text", {
      class: "topo-label",
      x: cx,
      y: cy - 4.2,
    });
    lbl.textContent = String(idx);
    root.append(lbl);
  }

  els.topoStage.append(root);

  // Footer: N • K0 • sigma • eta.
  const fmt = (x, d = 2) =>
    typeof x === "number" && Number.isFinite(x) ? x.toFixed(d) : "—";
  const coup = topo.coupling || {};
  const noise = topo.noise || {};
  const parts = [
    ["N", String(topo.N ?? topo.nodes.length)],
    ["K₀", fmt(coup.K0, 2)],
    ["σ", fmt(coup.sigma, 2)],
    ["η", fmt(noise.eta, 2)],
  ];
  for (const [k, v] of parts) {
    const span = el("span", null);
    span.append(el("span", "k", k), el("span", "v", v));
    els.topoFooter.append(span);
  }
}

// --- dynamics (r(t) sparkline + pulse raster) ---

// Mark-up scaffolding: every firing window in `windows_s` becomes a
// translucent accent band behind the sparkline / raster, so the reader
// sees at a glance *when* the named regime held.
function firingWindowBands(summary, durationS, viewW) {
  if (!summary || !durationS || durationS <= 0) return [];
  const dets = summary.detectors || {};
  const bands = [];
  for (const name of Object.keys(dets)) {
    const d = dets[name];
    if (!d || !d.fired || !Array.isArray(d.windows_s)) continue;
    for (const w of d.windows_s) {
      if (!Array.isArray(w) || w.length < 2) continue;
      const [s, e] = w;
      if (!Number.isFinite(s) || !Number.isFinite(e) || e <= s) continue;
      const x0 = Math.max(0, (s / durationS) * viewW);
      const x1 = Math.min(viewW, (e / durationS) * viewW);
      if (x1 > x0) bands.push({ x: x0, w: x1 - x0, name });
    }
  }
  return bands;
}

function renderRSeries(slotKey) {
  const slotEl = document.querySelector(`.slot[data-slot="${slotKey}"]`);
  if (!slotEl) return;
  const els = getSlotEls(slotEl);
  const summary = state[slotKey];
  els.rStage.innerHTML = "";
  if (!summary) return;
  const rs = summary.stats && summary.stats.r_series;
  const values = rs && Array.isArray(rs.values) ? rs.values : null;
  const fps = rs && typeof rs.fps === "number" && rs.fps > 0 ? rs.fps : null;
  if (!values || values.length < 2 || !fps) {
    els.rStage.append(el("p", "dynamics-missing", "r(t) unavailable"));
    return;
  }
  const duration = (values.length - 1) / fps;
  // Viewbox: 240×40 is the design aspect; stretched to the stage width.
  const VB_W = 240, VB_H = 40, PAD_L = 2, PAD_R = 2, PAD_T = 3, PAD_B = 3;
  const innerW = VB_W - PAD_L - PAD_R;
  const innerH = VB_H - PAD_T - PAD_B;
  const root = svg("svg", {
    viewBox: `0 0 ${VB_W} ${VB_H}`,
    preserveAspectRatio: "none",
    role: "img",
  });
  root.setAttribute("aria-label", `r(t) over ${duration.toFixed(2)} s`);

  // Firing-window bands — translucent vertical accents.
  for (const b of firingWindowBands(summary, duration, innerW)) {
    root.append(svg("rect", {
      class: "dyn-band",
      x: PAD_L + b.x,
      y: PAD_T,
      width: b.w,
      height: innerH,
    }));
  }

  // Reference line at r = 0.9 (the phase_locked threshold).
  const yLock = PAD_T + innerH * (1 - 0.9);
  root.append(svg("line", {
    class: "dyn-ref",
    x1: PAD_L, x2: PAD_L + innerW,
    y1: yLock, y2: yLock,
  }));

  // Polyline through the (sampled) r(t).
  const n = values.length;
  const pts = values.map((v, i) => {
    const x = PAD_L + (i / (n - 1)) * innerW;
    const clamped = Math.max(0, Math.min(1, v));
    const y = PAD_T + (1 - clamped) * innerH;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");
  root.append(svg("polyline", { class: "dyn-line", points: pts }));

  els.rStage.append(root);
}

function renderPulseRaster(slotKey) {
  const slotEl = document.querySelector(`.slot[data-slot="${slotKey}"]`);
  if (!slotEl) return;
  const els = getSlotEls(slotEl);
  const summary = state[slotKey];
  els.rasterStage.innerHTML = "";
  if (!summary) return;
  const raster = summary.stats && summary.stats.pulse_raster;
  if (!Array.isArray(raster) || raster.length === 0) {
    els.rasterStage.append(el("p", "dynamics-missing", "pulse raster unavailable"));
    return;
  }
  const durationS = (summary.meta && summary.meta.duration_s) || null;
  if (!durationS || durationS <= 0) {
    els.rasterStage.append(el("p", "dynamics-missing", "pulse raster unavailable"));
    return;
  }

  const N = raster.length;
  const ROW_H = 4.5;   // SVG units per node row
  const VB_W = 240;
  const VB_H = N * ROW_H;
  const PAD_L = 2, PAD_R = 2;
  const innerW = VB_W - PAD_L - PAD_R;

  const root = svg("svg", {
    viewBox: `0 0 ${VB_W} ${VB_H}`,
    preserveAspectRatio: "none",
    role: "img",
  });
  root.setAttribute("aria-label", `pulse raster, ${N} nodes over ${durationS.toFixed(2)} s`);

  // Firing-window bands, same as the r(t) sparkline.
  for (const b of firingWindowBands(summary, durationS, innerW)) {
    root.append(svg("rect", {
      class: "dyn-band",
      x: PAD_L + b.x,
      y: 0,
      width: b.w,
      height: VB_H,
    }));
  }

  // Per-node row separators (subtle) and pulse ticks.
  for (let i = 0; i < N; i++) {
    const yMid = (i + 0.5) * ROW_H;
    // faint baseline so empty rows still read as rows
    root.append(svg("line", {
      class: "dyn-row-base",
      x1: PAD_L, x2: PAD_L + innerW,
      y1: yMid, y2: yMid,
    }));
    const pulses = raster[i];
    if (!Array.isArray(pulses)) continue;
    for (const t of pulses) {
      if (!Number.isFinite(t)) continue;
      if (t < 0 || t > durationS) continue;
      const x = PAD_L + (t / durationS) * innerW;
      root.append(svg("line", {
        class: "dyn-tick",
        x1: x, x2: x,
        y1: yMid - ROW_H * 0.38,
        y2: yMid + ROW_H * 0.38,
      }));
    }
  }

  els.rasterStage.append(root);
}

function renderDynamicsFooter(slotKey) {
  const slotEl = document.querySelector(`.slot[data-slot="${slotKey}"]`);
  if (!slotEl) return;
  const els = getSlotEls(slotEl);
  const summary = state[slotKey];
  els.dynamicsFooter.innerHTML = "";
  if (!summary) return;
  const meta = summary.meta || {};
  const raster = summary.stats && Array.isArray(summary.stats.pulse_raster)
    ? summary.stats.pulse_raster : [];
  const totalPulses = raster.reduce(
    (acc, p) => acc + (Array.isArray(p) ? p.length : 0), 0
  );
  const parts = [
    ["window", typeof meta.duration_s === "number" ? `${meta.duration_s.toFixed(2)} s` : "—"],
    ["lock line", "r = 0.9"],
    ["pulses", String(totalPulses)],
  ];
  for (const [k, v] of parts) {
    const span = el("span", null);
    span.append(el("span", "k", k), el("span", "v", v));
    els.dynamicsFooter.append(span);
  }
}

function renderDynamics(slotKey) {
  renderRSeries(slotKey);
  renderPulseRaster(slotKey);
  renderDynamicsFooter(slotKey);
}

function topologiesMatch(a, b) {
  if (!a || !b || !Array.isArray(a.nodes) || !Array.isArray(b.nodes)) return null;
  if (a.nodes.length !== b.nodes.length) return false;
  const EPS = 1e-3;
  // Compare by index where possible; otherwise by order.
  const byIdx = (arr) => {
    const m = new Map();
    for (const n of arr) if (Number.isInteger(n?.index)) m.set(n.index, n);
    return m.size === arr.length ? m : null;
  };
  const ma = byIdx(a.nodes), mb = byIdx(b.nodes);
  const pairs = ma && mb
    ? [...ma.keys()].map((k) => [ma.get(k), mb.get(k)])
    : a.nodes.map((n, i) => [n, b.nodes[i]]);
  for (const [na, nb] of pairs) {
    if (!na || !nb) return false;
    const pa = na.pos, pb = nb.pos;
    if (!Array.isArray(pa) || !Array.isArray(pb)) return false;
    if (Math.abs(pa[0] - pb[0]) > EPS) return false;
    if (Math.abs(pa[1] - pb[1]) > EPS) return false;
  }
  return true;
}

function renderTopoBadge() {
  const badge = document.getElementById("topo-badge");
  if (!badge) return;
  const a = topos.A, b = topos.B;
  if (!a || !b) { badge.hidden = true; return; }
  const match = topologiesMatch(a, b);
  if (match === null) { badge.hidden = true; return; }
  badge.hidden = false;
  badge.dataset.same = match ? "true" : "false";
  badge.textContent = match ? "same topology" : "different topology";
}

function setTopology(slotKey, topo) {
  topos[slotKey] = topo && typeof topo === "object" ? topo : null;
  // If the summary card isn't visible yet, renderSummary will call us later.
  const slotEl = document.querySelector(`.slot[data-slot="${slotKey}"]`);
  if (slotEl && !slotEl.querySelector('[data-role="summary"]').hidden) {
    renderTopology(slotKey);
  }
  renderTopoBadge();
}

function renderDetectorCard(name, d) {
  const card = el("div", "detector-card");
  const fired = !!(d && d.fired);
  card.dataset.state = fired ? "fired" : "silent";
  card.append(el("div", "detector-name", name));
  card.append(el("div", "detector-verdict", fired ? "FIRED" : "silent"));
  const rows = el("div", "detector-rows");
  if (d) {
    rows.append(kvRow("confidence", fmtFloat(d.confidence, 3)));
    rows.append(kvRow("longest window", fmtSeconds(d.longest_window_s, 2)));
    if (Array.isArray(d.windows_s) && d.windows_s.length > 0) {
      const ws = d.windows_s
        .map(([s, e]) =>
          Number.isFinite(s) && Number.isFinite(e)
            ? `[${s.toFixed(2)}, ${e.toFixed(2)})`
            : "—"
        )
        .join("  ");
      rows.append(kvRow("windows", ws));
    }
  }
  card.append(rows);
  return card;
}

function renderSummary(slotKey, summary, sourceLabel) {
  const slotEl = document.querySelector(`.slot[data-slot="${slotKey}"]`);
  const els = getSlotEls(slotEl);
  if (!summary || typeof summary !== "object" || Array.isArray(summary)) {
    setStatus(els, "not a JSON object", "err");
    return;
  }
  state[slotKey] = summary;

  const ver = summary.schema_version;
  if (ver === 1) setStatus(els, `loaded: ${sourceLabel ?? ""}`.trim(), "ok");
  else setStatus(els, `schema_version=${ver ?? "?"} — best-effort`, "warn");

  const meta = summary.meta || {};
  els.metaName.textContent = meta.config_name || "(unnamed config)";
  els.metaDuration.textContent =
    typeof meta.duration_s === "number" ? `${meta.duration_s.toFixed(2)} s` : "—";
  els.metaN.textContent = meta.N ?? "—";
  els.metaRate.textContent =
    typeof meta.control_rate_hz === "number" ? `${meta.control_rate_hz} Hz` : "—";
  els.metaSchema.textContent = `v${ver ?? "?"}`;
  els.metaPath.textContent = meta.config_path || "";

  els.detectors.innerHTML = "";
  const detectors = summary.detectors || {};
  const names = Object.keys(detectors)
    .filter((k) => detectors[k] && typeof detectors[k] === "object")
    .sort();
  if (names.length === 0) {
    els.detectors.append(el("p", "empty-note", "(no detectors reported)"));
  } else {
    for (const n of names) els.detectors.append(renderDetectorCard(n, detectors[n]));
  }

  els.stats.innerHTML = "";
  const s = summary.stats || {};
  if ("mean_r" in s) els.stats.append(kvRow("mean r(t)", fmtFloat(s.mean_r)));
  if ("tail_1s_mean_r" in s) els.stats.append(kvRow("tail 1s r", fmtFloat(s.tail_1s_mean_r)));
  if ("tail_2way_velocity_separability" in s) {
    els.stats.append(kvRow("tail 2-way velocity sep.", fmtSep(s.tail_2way_velocity_separability)));
  }

  els.rawJson.textContent = JSON.stringify(summary, null, 2);
  els.empty.hidden = true;
  els.summary.hidden = false;

  renderTopology(slotKey);
  renderDynamics(slotKey);

  document.getElementById("empty-both").hidden = true;
  renderComparison();
  renderTopoBadge();
}

// --- comparison ---

// Human-readable sentence for a single detector state flip.
const FLIP_SENTENCE = {
  phase_locked:     { up: "Global lock emerged.",      down: "Lock collapsed." },
  drifting:         { up: "Drift appeared.",           down: "Drift disappeared." },
  phase_beating:    { up: "Phase beating appeared.",   down: "Phase beating faded." },
  flam:             { up: "A flam formed.",            down: "The flam dissolved." },
  dominant_cluster: { up: "A dominant cluster formed.", down: "The cluster dissolved." },
};

function takeawayForFlip({ name, to }) {
  const dir = to ? "up" : "down";
  const map = FLIP_SENTENCE[name];
  if (map) return map[dir];
  return `${name}: ${to ? "silent → FIRED" : "FIRED → silent"}.`;
}

function joinSentences(parts) {
  if (parts.length <= 1) return parts[0] ?? "";
  // lowercase the leading char of each continuation, drop its trailing "."
  const head = parts[0].replace(/\.$/, "");
  const tail = parts.slice(1).map((p) => {
    const t = p.replace(/\.$/, "");
    return t.charAt(0).toLowerCase() + t.slice(1);
  });
  return `${head}; ${tail.join("; ")}.`;
}

function buildTakeaway(flips, sA, sB) {
  if (flips.length > 0) return joinSentences(flips.map(takeawayForFlip));
  // no detector flip — fall back to the most legible stat delta
  const rA = sA?.tail_1s_mean_r, rB = sB?.tail_1s_mean_r;
  if (typeof rA === "number" && typeof rB === "number") {
    const dr = rB - rA;
    if (Math.abs(dr) >= 0.1) {
      return `Tail coherence ${dr > 0 ? "rose" : "fell"} ${Math.abs(dr).toFixed(2)}.`;
    }
  }
  const sepA = sA?.tail_2way_velocity_separability;
  const sepB = sB?.tail_2way_velocity_separability;
  if (typeof sepA === "number" && typeof sepB === "number" && Math.abs(sepB - sepA) >= 10) {
    return `2-way velocity separability ${sepB > sepA ? "grew" : "shrank"} sharply.`;
  }
  return "No detector flips; stats largely unchanged.";
}

function diffRow(label, change, dir) {
  const li = el("li", "diff-row");
  li.dataset.dir = dir;
  li.append(el("span", "diff-label mono", label));
  li.append(el("span", "diff-change mono", change));
  return li;
}

function firedLabel(d) {
  if (!d) return "absent";
  return d.fired ? "FIRED" : "silent";
}

function renderComparison() {
  const { A, B } = state;
  const container = document.getElementById("comparison");
  const list = document.getElementById("diff-list");
  const takeaway = document.getElementById("takeaway");
  if (!A || !B) {
    container.hidden = true;
    return;
  }

  const bullets = [];
  const flips = [];

  // -- detector flips / confidence changes --
  const detA = A.detectors || {};
  const detB = B.detectors || {};
  const detNames = Array.from(new Set([...Object.keys(detA), ...Object.keys(detB)]))
    .filter((n) => (detA[n] && typeof detA[n] === "object") || (detB[n] && typeof detB[n] === "object"))
    .sort();
  for (const n of detNames) {
    const a = detA[n], b = detB[n];
    const aFired = !!(a && a.fired);
    const bFired = !!(b && b.fired);
    if (!a || !b) {
      bullets.push(diffRow(n, `${firedLabel(a)} → ${firedLabel(b)}`, b ? "up" : "down"));
      if (a && !b) flips.push({ name: n, from: aFired, to: false });
      if (!a && b) flips.push({ name: n, from: false, to: bFired });
      continue;
    }
    if (aFired !== bFired) {
      bullets.push(diffRow(n, `${firedLabel(a)} → ${firedLabel(b)}`, bFired ? "up" : "down"));
      flips.push({ name: n, from: aFired, to: bFired });
    } else if (aFired && bFired) {
      const aConf = a.confidence, bConf = b.confidence;
      if (typeof aConf === "number" && typeof bConf === "number" && Math.abs(bConf - aConf) >= 0.001) {
        bullets.push(diffRow(
          n,
          `conf ${fmtFloat(aConf)} → ${fmtFloat(bConf)} (${signed(bConf - aConf)})`,
          "flat"
        ));
      }
    }
  }

  // -- stat deltas --
  const STAT_KEYS = [
    ["mean_r", "mean r(t)", 3, 0.001],
    ["tail_1s_mean_r", "tail 1s r", 3, 0.001],
    ["tail_2way_velocity_separability", "tail 2-way sep.", 2, 0.01],
  ];
  const sA = A.stats || {}, sB = B.stats || {};
  for (const [k, label, digits, tol] of STAT_KEYS) {
    const hasA = k in sA, hasB = k in sB;
    if (!hasA && !hasB) continue;
    const va = sA[k], vb = sB[k];
    // null on separability means "inf" — treat as not-a-number for delta
    const vaNum = typeof va === "number" && Number.isFinite(va);
    const vbNum = typeof vb === "number" && Number.isFinite(vb);
    if (!vaNum || !vbNum) {
      const show = (present, x) => !present ? "—" : (x == null ? "inf" : fmtFloat(x, digits));
      if (show(hasA, va) !== show(hasB, vb)) {
        bullets.push(diffRow(label, `${show(hasA, va)} → ${show(hasB, vb)}`, "flat"));
      }
      continue;
    }
    const d = vb - va;
    if (Math.abs(d) < tol) continue;
    bullets.push(diffRow(
      label,
      `${fmtFloat(va, digits)} → ${fmtFloat(vb, digits)} (${signed(d, digits)})`,
      d > 0 ? "up" : "down"
    ));
  }

  list.innerHTML = "";
  if (bullets.length === 0) {
    list.append(el("li", "diff-none", "(no changes)"));
  } else {
    for (const b of bullets) list.append(b);
  }
  takeaway.textContent = buildTakeaway(flips, sA, sB);
  container.hidden = false;
}

// --- loaders ---

function wireSlot(slotEl) {
  const slotKey = slotEl.dataset.slot;
  const els = getSlotEls(slotEl);
  els.file.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      const text = await file.text();
      renderSummary(slotKey, JSON.parse(text), file.name);
    } catch (err) {
      setStatus(els, `load failed: ${err.message}`, "err");
    }
  });
  if (els.topoFile) {
    els.topoFile.addEventListener("change", async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      try {
        setTopology(slotKey, JSON.parse(await file.text()));
      } catch (err) {
        setStatus(els, `topology load failed: ${err.message}`, "err");
      }
    });
  }
  els.pasteToggle.addEventListener("click", () => {
    els.pastePanel.hidden = !els.pastePanel.hidden;
    if (!els.pastePanel.hidden) els.pasteArea.focus();
  });
  els.pasteApply.addEventListener("click", () => {
    try {
      renderSummary(slotKey, JSON.parse(els.pasteArea.value), "pasted");
    } catch (err) {
      setStatus(els, `parse failed: ${err.message}`, "err");
    }
  });
}

for (const slotEl of document.querySelectorAll(".slot")) wireSlot(slotEl);

// query params: ?summaryA=... & ?summaryB=...  (legacy: ?summary=... → slot A)
// ?topologyA=... / ?topologyB=... override; otherwise infer a sibling
// "topology.json" next to the summary URL and fetch it best-effort. Slots
// load in parallel so each slot's summary+topology pair doesn't block the
// other (matters for snappy first paint and for headless capture).
(function loadFromQuery() {
  const qp = new URLSearchParams(location.search);
  const pairs = [
    ["A", qp.get("summaryA") || qp.get("summary"), qp.get("topologyA")],
    ["B", qp.get("summaryB"),                      qp.get("topologyB")],
  ];
  Promise.all(pairs.map(async ([key, url, topoOverride]) => {
    if (!url) return;
    const slotEl = document.querySelector(`.slot[data-slot="${key}"]`);
    const els = getSlotEls(slotEl);
    try {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      renderSummary(key, await resp.json(), url);
    } catch (err) {
      setStatus(els, `fetch failed: ${err.message}`, "err");
      return;
    }
    const topoUrl = topoOverride || inferTopologyUrl(url);
    if (!topoUrl) return;
    try {
      const tr = await fetch(topoUrl);
      if (!tr.ok) return; // missing sibling is fine — fail graceful
      setTopology(key, await tr.json());
    } catch (_) { /* network / parse: leave topology unavailable */ }
  }));
})();

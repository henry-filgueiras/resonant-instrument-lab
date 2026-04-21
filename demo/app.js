// Static regime-oracle viewer, A/B comparison mode. Loads up to two
// summary.json files (file picker, paste box, or ?summaryA=URL / ?summaryB=URL
// query params — fetch only works when served, not file://) and renders each
// plus a detector-flip / stat-delta comparison card. No framework, no build.

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

function getSlotEls(slotEl) {
  const q = (role) => slotEl.querySelector(`[data-role="${role}"]`);
  return {
    root: slotEl,
    file: q("file"),
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

  document.getElementById("empty-both").hidden = true;
  renderComparison();
}

// --- comparison ---

// Human-readable sentence for a single detector state flip.
const FLIP_SENTENCE = {
  phase_locked:  { up: "Global lock emerged.",     down: "Lock collapsed." },
  drifting:      { up: "Drift appeared.",          down: "Drift disappeared." },
  phase_beating: { up: "Phase beating appeared.",  down: "Phase beating faded." },
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
(async function loadFromQuery() {
  const qp = new URLSearchParams(location.search);
  const pairs = [
    ["A", qp.get("summaryA") || qp.get("summary")],
    ["B", qp.get("summaryB")],
  ];
  for (const [key, url] of pairs) {
    if (!url) continue;
    const slotEl = document.querySelector(`.slot[data-slot="${key}"]`);
    const els = getSlotEls(slotEl);
    try {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      renderSummary(key, await resp.json(), url);
    } catch (err) {
      setStatus(els, `fetch failed: ${err.message}`, "err");
    }
  }
})();

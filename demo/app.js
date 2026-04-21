// Static regime-oracle viewer. Reads a summary.json emitted by
// `scripts/run_sim.py --summary-json` and renders one compact card view.
// No framework, no build step. Loaders: file picker, paste box, and
// `?summary=URL` query param (fetch only works when served, not file://).

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

const els = {
  fileInput: $("file-input"),
  pasteToggle: $("paste-toggle"),
  pastePanel: $("paste-panel"),
  pasteArea: $("paste-area"),
  pasteApply: $("paste-apply"),
  status: $("status"),
  empty: $("empty"),
  summary: $("summary"),
  metaName: $("meta-name"),
  metaDuration: $("meta-duration"),
  metaN: $("meta-n"),
  metaRate: $("meta-rate"),
  metaSchema: $("meta-schema"),
  metaPath: $("meta-path"),
  detectors: $("detectors"),
  stats: $("stats"),
  rawJson: $("raw-json"),
};

function setStatus(msg, kind) {
  els.status.textContent = msg || "";
  if (kind) els.status.dataset.kind = kind;
  else delete els.status.dataset.kind;
}

function fmtFloat(x, digits = 3) {
  if (x == null || typeof x !== "number" || !Number.isFinite(x)) return "—";
  return x.toFixed(digits);
}

function fmtSeconds(x, digits = 2) {
  if (x == null || typeof x !== "number" || !Number.isFinite(x)) return "—";
  return `${x.toFixed(digits)} s`;
}

function kvRow(k, v) {
  const row = el("div", "row");
  row.append(el("span", "k", k), el("span", "v mono", v));
  return row;
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

function renderSummary(summary, sourceLabel) {
  if (!summary || typeof summary !== "object" || Array.isArray(summary)) {
    setStatus("not a JSON object", "err");
    return;
  }

  const ver = summary.schema_version;
  if (ver === 1) setStatus(`loaded: ${sourceLabel ?? ""}`.trim(), "ok");
  else setStatus(`schema_version=${ver ?? "?"} — rendering best-effort`, "warn");

  // meta
  const meta = summary.meta || {};
  els.metaName.textContent = meta.config_name || "(unnamed config)";
  els.metaDuration.textContent =
    typeof meta.duration_s === "number" ? `${meta.duration_s.toFixed(2)} s` : "—";
  els.metaN.textContent = meta.N ?? "—";
  els.metaRate.textContent =
    typeof meta.control_rate_hz === "number" ? `${meta.control_rate_hz} Hz` : "—";
  els.metaSchema.textContent = `v${ver ?? "?"}`;
  els.metaPath.textContent = meta.config_path || "";

  // detectors — sorted alphabetically for stable layout, skip non-objects
  els.detectors.innerHTML = "";
  const detectors = summary.detectors || {};
  const names = Object.keys(detectors)
    .filter((k) => detectors[k] && typeof detectors[k] === "object")
    .sort();
  if (names.length === 0) {
    const p = el("p", "empty", "(no detectors reported)");
    els.detectors.append(p);
  } else {
    for (const n of names) els.detectors.append(renderDetectorCard(n, detectors[n]));
  }

  // stats — render known keys, skip absent ones; null separability → "inf"
  els.stats.innerHTML = "";
  const s = summary.stats || {};
  if ("mean_r" in s) els.stats.append(kvRow("mean r(t)", fmtFloat(s.mean_r)));
  if ("tail_1s_mean_r" in s) els.stats.append(kvRow("tail 1s r", fmtFloat(s.tail_1s_mean_r)));
  if ("tail_2way_velocity_separability" in s) {
    const sep = s.tail_2way_velocity_separability;
    els.stats.append(
      kvRow(
        "tail 2-way velocity sep.",
        sep == null ? "inf" : fmtFloat(sep, 2)
      )
    );
  }

  // raw
  els.rawJson.textContent = JSON.stringify(summary, null, 2);

  els.empty.hidden = true;
  els.summary.hidden = false;
}

// --- loaders ---
els.fileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    const text = await file.text();
    renderSummary(JSON.parse(text), file.name);
  } catch (err) {
    setStatus(`load failed: ${err.message}`, "err");
  }
});

els.pasteToggle.addEventListener("click", () => {
  els.pastePanel.hidden = !els.pastePanel.hidden;
  if (!els.pastePanel.hidden) els.pasteArea.focus();
});

els.pasteApply.addEventListener("click", () => {
  try {
    renderSummary(JSON.parse(els.pasteArea.value), "pasted");
  } catch (err) {
    setStatus(`parse failed: ${err.message}`, "err");
  }
});

(async function loadFromQuery() {
  const url = new URLSearchParams(location.search).get("summary");
  if (!url) return;
  try {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    renderSummary(await resp.json(), url);
  } catch (err) {
    setStatus(`fetch failed: ${err.message}`, "err");
  }
})();

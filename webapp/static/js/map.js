// Drives the whole single-page explorer: renders the 2D UMAP, colors it by
// MoA or plate, and wires search + click selection to fetch a right-sidebar
// detail partial without ever navigating away from this page.

const TOP_N_MOAS = 10;
const MOA_PALETTE = [
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
  "#8c564b", "#e377c2", "#7f7c00", "#17becf", "#bcbd22",
];
const PLATE_PALETTE = [
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
  "#8c564b", "#e377c2", "#7f7c00", "#17becf", "#bcbd22",
  "#aec7e8", "#ffbb78", "#98df8a", "#ff9896",
];
const OTHER_ANNOTATED_COLOR = "#adb5bd";
const UNANNOTATED_COLOR = "#e9ecef";

let plotEl = null;
let umapData = null;
let selectedWellId = null;
let highlightTraceIndex = null;

function hoverText(data, i) {
  return `Plate: ${data.plate[i]}<br>MoA: ${data.moa[i]}`;
}

// "other" and "unannotated" are kept as two separate buckets rather than one
// vague "other" group: most of what lands here DOES have an annotated MoA,
// it's just too rare to get its own legend color. See the caption under the
// plot for the actual numbers.
function buildMoaTraces(data) {
  const counts = {};
  data.moa.forEach((m) => { counts[m] = (counts[m] || 0) + 1; });
  const topMoas = Object.keys(counts)
    .filter((m) => m !== "Unannotated")
    .sort((a, b) => counts[b] - counts[a])
    .slice(0, TOP_N_MOAS);

  const groups = {};
  topMoas.forEach((m) => { groups[m] = { x: [], y: [], text: [], customdata: [] }; });
  groups["other-annotated"] = { x: [], y: [], text: [], customdata: [] };
  groups["unannotated"] = { x: [], y: [], text: [], customdata: [] };

  for (let i = 0; i < data.well_id.length; i++) {
    const moa = data.moa[i];
    let key;
    if (topMoas.includes(moa)) key = moa;
    else if (moa === "Unannotated") key = "unannotated";
    else key = "other-annotated";
    const g = groups[key];
    g.x.push(data.x[i]);
    g.y.push(data.y[i]);
    g.customdata.push(data.well_id[i]);
    g.text.push(hoverText(data, i));
  }

  const traces = topMoas.map((m, i) => ({
    x: groups[m].x, y: groups[m].y, text: groups[m].text, customdata: groups[m].customdata,
    mode: "markers", type: "scattergl", name: m,
    marker: { size: 6, color: MOA_PALETTE[i % MOA_PALETTE.length] },
    hoverinfo: "text",
  }));
  traces.push({
    x: groups["other-annotated"].x, y: groups["other-annotated"].y,
    text: groups["other-annotated"].text, customdata: groups["other-annotated"].customdata,
    mode: "markers", type: "scattergl", name: "other annotated MoA (grouped for readability)",
    marker: { size: 5, color: OTHER_ANNOTATED_COLOR, opacity: 0.7 },
    hoverinfo: "text",
  });
  traces.push({
    x: groups["unannotated"].x, y: groups["unannotated"].y,
    text: groups["unannotated"].text, customdata: groups["unannotated"].customdata,
    mode: "markers", type: "scattergl", name: "no annotated MoA",
    marker: { size: 5, color: UNANNOTATED_COLOR, opacity: 0.6 },
    hoverinfo: "text",
  });
  return traces;
}

function buildPlateTraces(data) {
  const plates = Array.from(new Set(data.plate)).sort();
  const groups = {};
  plates.forEach((p) => { groups[p] = { x: [], y: [], text: [], customdata: [] }; });

  for (let i = 0; i < data.well_id.length; i++) {
    const g = groups[data.plate[i]];
    g.x.push(data.x[i]);
    g.y.push(data.y[i]);
    g.customdata.push(data.well_id[i]);
    g.text.push(hoverText(data, i));
  }

  return plates.map((p, i) => ({
    x: groups[p].x, y: groups[p].y, text: groups[p].text, customdata: groups[p].customdata,
    mode: "markers", type: "scattergl", name: p,
    marker: { size: 6, color: PLATE_PALETTE[i % PLATE_PALETTE.length] },
    hoverinfo: "text",
  }));
}

function buildHighlightTrace() {
  return {
    x: [], y: [], mode: "markers", type: "scattergl",
    name: "selected", showlegend: false, hoverinfo: "skip",
    marker: { size: 16, color: "rgba(0,0,0,0)", line: { color: "#000", width: 2 } },
  };
}

function defaultLayout() {
  return {
    margin: { t: 10, r: 10, b: 40, l: 40 },
    xaxis: { title: "UMAP 1" },
    yaxis: { title: "UMAP 2" },
    legend: { orientation: "h", y: -0.15 },
    dragmode: "pan",
    hovermode: "closest",
  };
}

// Mouse wheel zooms, left-drag pans, hover always on; the toolbar is trimmed
// to just "Reset axes" so normal navigation never needs it.
const PLOT_CONFIG = {
  responsive: true,
  scrollZoom: true,
  displaylogo: false,
  modeBarButtonsToRemove: [
    "zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d",
    "autoScale2d", "hoverClosestCartesian", "hoverCompareCartesian",
    "toggleSpikelines", "toImage",
  ],
};

function render(colorBy) {
  const traces = colorBy === "plate" ? buildPlateTraces(umapData) : buildMoaTraces(umapData);
  highlightTraceIndex = traces.length;
  traces.push(buildHighlightTrace());
  Plotly.react(plotEl, traces, plotEl.layout || defaultLayout(), PLOT_CONFIG);
  if (selectedWellId) updateHighlightPosition(selectedWellId);
}

function setColorBy(mode) {
  document.getElementById("color-by-moa").classList.toggle("active", mode === "moa");
  document.getElementById("color-by-plate").classList.toggle("active", mode === "plate");
  render(mode);
}

function updateHighlightPosition(wellId) {
  const idx = umapData.well_id.indexOf(wellId);
  if (idx === -1) return;
  Plotly.restyle(plotEl, { x: [[umapData.x[idx]]], y: [[umapData.y[idx]]] }, [highlightTraceIndex]);
}

function centerOn(x, y) {
  const layout = plotEl._fullLayout;
  if (!layout) return;
  const xr = layout.xaxis.range;
  const yr = layout.yaxis.range;
  const halfX = (xr[1] - xr[0]) / 2;
  const halfY = (yr[1] - yr[0]) / 2;
  Plotly.relayout(plotEl, {
    "xaxis.range": [x - halfX, x + halfX],
    "yaxis.range": [y - halfY, y + halfY],
  });
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s == null ? "" : s;
  return div.innerHTML;
}

function selectWell(wellId, opts) {
  opts = opts || {};
  selectedWellId = wellId;
  updateHighlightPosition(wellId);

  if (opts.center !== false) {
    const idx = umapData.well_id.indexOf(wellId);
    if (idx !== -1) centerOn(umapData.x[idx], umapData.y[idx]);
  }

  fetch(`/api/well/${encodeURIComponent(wellId)}`)
    .then((r) => r.text())
    .then((html) => { document.getElementById("sidebar-content").innerHTML = html; });
}

function renderDisambiguation(result) {
  const container = document.getElementById("sidebar-content");
  const items = result.matches.map((m) => `
    <li class="list-group-item list-group-item-action" role="button" data-well-id="${escapeHtml(m.well_id)}">
      <strong>${escapeHtml(m.label)}</strong>
      <div class="small text-muted">${escapeHtml(m.moa || "unannotated")}</div>
    </li>
  `).join("");
  container.innerHTML = `
    <p class="small text-muted">Multiple matches for &ldquo;${escapeHtml(result.query)}&rdquo; &mdash; pick one:</p>
    <ul class="list-group">${items}</ul>
    ${result.truncated ? '<p class="small text-muted mt-2">Showing a limited set of matches — try a more specific search.</p>' : ""}
  `;
  container.querySelectorAll("[data-well-id]").forEach((el) => {
    el.addEventListener("click", () => selectWell(el.dataset.wellId));
  });
}

function performSearch(query) {
  const msgEl = document.getElementById("search-message");
  msgEl.textContent = "";
  if (!query || !query.trim()) return;

  fetch(`/api/search?q=${encodeURIComponent(query)}`)
    .then((r) => r.json())
    .then((result) => {
      if (result.kind === "well") {
        selectWell(result.well_id);
      } else if (result.kind === "disambiguate") {
        renderDisambiguation(result);
      } else {
        msgEl.textContent = `No match found for "${result.query}".`;
      }
    });
}

const MIN_SIDEBAR_WIDTH = 200;
const MAX_SIDEBAR_WIDTH = 560;

// Draggable divider: `side` is which edge of `target` the handle sits on,
// so dragging computes width as the distance from that edge to the cursor.
function makeResizable(handle, target, side) {
  let dragging = false;

  handle.addEventListener("mousedown", (e) => {
    dragging = true;
    document.body.style.userSelect = "none";
    e.preventDefault();
  });

  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const rect = target.getBoundingClientRect();
    let width = side === "left" ? e.clientX - rect.left : rect.right - e.clientX;
    width = Math.max(MIN_SIDEBAR_WIDTH, Math.min(MAX_SIDEBAR_WIDTH, width));
    target.style.flex = `0 0 ${width}px`;
    if (plotEl) Plotly.Plots.resize(plotEl);
  });

  window.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false;
    document.body.style.userSelect = "";
  });
}

function initMap() {
  plotEl = document.getElementById("umap-plot");
  if (!plotEl) return;

  fetch("/api/umap")
    .then((r) => r.json())
    .then((data) => {
      umapData = data;
      render("moa");

      plotEl.on("plotly_click", (evt) => {
        const wellId = evt.points[0].customdata;
        if (wellId) selectWell(wellId, { center: false });
      });
    });

  document.getElementById("color-by-moa").addEventListener("click", () => setColorBy("moa"));
  document.getElementById("color-by-plate").addEventListener("click", () => setColorBy("plate"));

  const input = document.getElementById("search-input");
  document.getElementById("search-btn").addEventListener("click", () => performSearch(input.value));
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") performSearch(input.value);
  });

  document.querySelectorAll(".example-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      input.value = btn.dataset.q;
      performSearch(btn.dataset.q);
    });
  });

  makeResizable(document.getElementById("resize-handle-left"), document.getElementById("sidebar-left"), "left");
  makeResizable(document.getElementById("resize-handle-right"), document.getElementById("sidebar-right"), "right");
}

document.addEventListener("DOMContentLoaded", initMap);

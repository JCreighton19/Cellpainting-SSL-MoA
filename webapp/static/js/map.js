// Drives the whole single-page explorer: renders the 2D UMAP, colors it by
// MoA or plate, filters the map by search text, and wires point clicks to
// fetch a right-sidebar detail partial without ever navigating away from
// this page.

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

// Sub-1 opacity so overlapping points in dense clusters stay distinguishable
// instead of flattening into a single opaque blob.
const MARKER_OPACITY = 0.8;
const BACKGROUND_MARKER_OPACITY = 0.6;

// Used to de-emphasize points that don't match the active search filter.
const FILTERED_OUT_COLOR = "#e9ecef";
const FILTERED_OUT_OPACITY = 0.25;
// Used instead of the normal MoA/plate color for points that DO match an
// active search filter, so matches jump out regardless of colorBy mode.
const MATCH_HIGHLIGHT_COLOR = "#00e676";

// 3D UMAP was evaluated and intentionally deferred: the precomputed
// wells.parquet only carries umap_x/umap_y (see scripts/prepare_phase1_data.py,
// which fits UMAP with n_components=2), so a 3rd dimension needs an offline
// re-fit of the whole dataset. On the front end it would also mean swapping
// scattergl for Plotly's scatter3d (a different trace type with orbit-camera
// controls instead of 2D pan/zoom) and reworking centerOn()/updateHighlightPosition(),
// which both assume a 2D axis range. That's real restructuring, not a toggle
// away from where the app is today, so it isn't implemented here.

let plotEl = null;
let umapData = null;
let selectedWellId = null;
let highlightTraceIndex = null;
let currentColorBy = "moa";
// Set of well_id strings matching the active search filter, or null when no
// filter is active (i.e. every point renders in its normal color).
let matchSet = null;

function hoverText(data, i) {
  return `Plate: ${data.plate[i]}<br>MoA: ${data.moa[i]}`;
}

function emptyBucket() {
  return { x: [], y: [], text: [], customdata: [], color: [], opacity: [] };
}

function pushPoint(bucket, data, i, color, opacity) {
  bucket.x.push(data.x[i]);
  bucket.y.push(data.y[i]);
  bucket.text.push(hoverText(data, i));
  bucket.customdata.push(data.well_id[i]);
  bucket.color.push(color);
  bucket.opacity.push(opacity);
}

// Concatenates dim (filtered-out) points before bright (matching, or
// no-filter) points, so that once handed to Plotly as one trace's arrays,
// bright points are drawn last and therefore render on top of dim ones.
function mergeDimBright(dim, bright) {
  return {
    x: dim.x.concat(bright.x),
    y: dim.y.concat(bright.y),
    text: dim.text.concat(bright.text),
    customdata: dim.customdata.concat(bright.customdata),
    color: dim.color.concat(bright.color),
    opacity: dim.opacity.concat(bright.opacity),
  };
}

// "other" and "unannotated" are kept as two separate buckets rather than one
// vague "other" group: most of what lands here DOES have an annotated MoA,
// it's just too rare to get its own legend color. See the caption under the
// plot for the actual numbers.
//
// Trace order matters here beyond just this function's own dim/bright split:
// the "unannotated"/"other-annotated" traces are pushed before the top-MoA
// traces below so the more informative, colored traces always render above
// the muted background ones, filter active or not.
function buildMoaTraces(data) {
  const counts = {};
  data.moa.forEach((m) => { counts[m] = (counts[m] || 0) + 1; });
  const topMoas = Object.keys(counts)
    .filter((m) => m !== "Unannotated")
    .sort((a, b) => counts[b] - counts[a])
    .slice(0, TOP_N_MOAS);
  const topMoaColor = {};
  topMoas.forEach((m, i) => { topMoaColor[m] = MOA_PALETTE[i % MOA_PALETTE.length]; });

  const groups = {};
  topMoas.forEach((m) => { groups[m] = { dim: emptyBucket(), bright: emptyBucket() }; });
  groups["other-annotated"] = { dim: emptyBucket(), bright: emptyBucket() };
  groups["unannotated"] = { dim: emptyBucket(), bright: emptyBucket() };

  for (let i = 0; i < data.well_id.length; i++) {
    const moa = data.moa[i];
    let key, baseColor, baseOpacity;
    if (topMoas.includes(moa)) {
      key = moa;
      baseColor = topMoaColor[moa];
      baseOpacity = MARKER_OPACITY;
    } else if (moa === "Unannotated") {
      key = "unannotated";
      baseColor = UNANNOTATED_COLOR;
      baseOpacity = BACKGROUND_MARKER_OPACITY;
    } else {
      key = "other-annotated";
      baseColor = OTHER_ANNOTATED_COLOR;
      baseOpacity = BACKGROUND_MARKER_OPACITY;
    }

    const matches = !matchSet || matchSet.has(data.well_id[i]);
    const bucket = matches ? groups[key].bright : groups[key].dim;
    const color = matches ? (matchSet ? MATCH_HIGHLIGHT_COLOR : baseColor) : FILTERED_OUT_COLOR;
    pushPoint(bucket, data, i, color, matches ? baseOpacity : FILTERED_OUT_OPACITY);
  }

  const traces = [];
  [["unannotated", "no annotated MoA", 5], ["other-annotated", "other annotated MoA (grouped for readability)", 5]]
    .forEach(([key, name, size]) => {
      const g = mergeDimBright(groups[key].dim, groups[key].bright);
      traces.push({
        x: g.x, y: g.y, text: g.text, customdata: g.customdata,
        mode: "markers", type: "scattergl", name,
        marker: { size, color: g.color, opacity: g.opacity },
        hoverinfo: "text",
      });
    });
  topMoas.forEach((m) => {
    const g = mergeDimBright(groups[m].dim, groups[m].bright);
    traces.push({
      x: g.x, y: g.y, text: g.text, customdata: g.customdata,
      mode: "markers", type: "scattergl", name: m,
      marker: { size: 7, color: g.color, opacity: g.opacity },
      hoverinfo: "text",
    });
  });
  return traces;
}

function buildPlateTraces(data) {
  const plates = Array.from(new Set(data.plate)).sort();
  const plateColor = {};
  plates.forEach((p, i) => { plateColor[p] = PLATE_PALETTE[i % PLATE_PALETTE.length]; });

  const groups = {};
  plates.forEach((p) => { groups[p] = { dim: emptyBucket(), bright: emptyBucket() }; });

  for (let i = 0; i < data.well_id.length; i++) {
    const p = data.plate[i];
    const matches = !matchSet || matchSet.has(data.well_id[i]);
    const bucket = matches ? groups[p].bright : groups[p].dim;
    const color = matches ? (matchSet ? MATCH_HIGHLIGHT_COLOR : plateColor[p]) : FILTERED_OUT_COLOR;
    pushPoint(bucket, data, i, color, matches ? MARKER_OPACITY : FILTERED_OUT_OPACITY);
  }

  return plates.map((p) => {
    const g = mergeDimBright(groups[p].dim, groups[p].bright);
    return {
      x: g.x, y: g.y, text: g.text, customdata: g.customdata,
      mode: "markers", type: "scattergl", name: p,
      marker: { size: 6, color: g.color, opacity: g.opacity },
      hoverinfo: "text",
    };
  });
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

// Re-renders using the current colorBy mode and search filter (module state
// above), so any caller that changes either one just calls render() again.
function render() {
  const traces = currentColorBy === "plate" ? buildPlateTraces(umapData) : buildMoaTraces(umapData);
  highlightTraceIndex = traces.length;
  traces.push(buildHighlightTrace());
  Plotly.react(plotEl, traces, plotEl.layout || defaultLayout(), PLOT_CONFIG);
  if (selectedWellId) updateHighlightPosition(selectedWellId);
}

function setColorBy(mode) {
  currentColorBy = mode;
  document.getElementById("color-by-moa").classList.toggle("active", mode === "moa");
  document.getElementById("color-by-plate").classList.toggle("active", mode === "plate");
  render();
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

// Image / Attention / Overlay toggle for the right-sidebar thumbnail. The
// heatmap PNG is generated on demand by /api/attention/<well_id>.png (see
// routes.py) and lazily loaded the first time it's needed; Overlay reuses
// that same image, just at reduced opacity via CSS, rather than requesting
// or storing a separate pre-rendered overlay image.
function setImageView(view) {
  const container = document.getElementById("sidebar-content");
  const thumb = container.querySelector(".thumbnail-img");
  const attn = container.querySelector(".attention-img");

  container.querySelectorAll("[data-view]").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.view === view);
  });

  if (!attn) return; // this well has no attention map

  if (!attn.src) {
    attn.src = `/api/attention/${encodeURIComponent(attn.dataset.wellId)}.png`;
  }

  attn.classList.toggle("d-none", view === "image");
  attn.classList.toggle("overlay-mode", view === "overlay");
  if (thumb) thumb.classList.toggle("d-none", view === "attention");
}

// Plotly's scattergl only fires plotly_click on an actual marker (clicking
// empty canvas fires nothing to catch), so the simplest way to offer
// deselection is treating a second click on the already-selected point as
// "clear" rather than "re-select".
function clearSelection() {
  selectedWellId = null;
  Plotly.restyle(plotEl, { x: [[]], y: [[]] }, [highlightTraceIndex]);
  document.getElementById("sidebar-content").innerHTML =
    '<p class="text-muted small">Click a point on the map, or search, to explore a phenotype.</p>';
}

// Matches the same free-text query against every field shown in the map's
// hover/search UI. Kept case-insensitive and substring-based to mirror how
// the old server-side search felt, but resolved entirely client-side since
// umapData already holds everything needed.
function computeMatchSet(query) {
  const q = query.trim().toLowerCase();
  const matched = new Set();
  for (let i = 0; i < umapData.well_id.length; i++) {
    const haystack = [
      umapData.well_id[i], umapData.moa[i], umapData.broad_sample[i],
      umapData.pert_iname[i], umapData.plate[i],
    ].join(" ").toLowerCase();
    if (haystack.includes(q)) matched.add(umapData.well_id[i]);
  }
  return matched;
}

function applyFilter(query) {
  matchSet = computeMatchSet(query);
  render();
  const msgEl = document.getElementById("search-message");
  msgEl.textContent = matchSet.size
    ? `${matchSet.size.toLocaleString()} point${matchSet.size === 1 ? "" : "s"} match "${query}".`
    : `No points match "${query}".`;
}

function clearFilter() {
  if (!matchSet) return;
  matchSet = null;
  document.getElementById("search-message").textContent = "";
  render();
}

// Search now filters the map in place (matches turn bright green, everything
// else fades to gray) instead of jumping to and selecting a single result —
// this makes the spatial spread of matches visible, and leaves point
// selection entirely to clicking the map.
function performSearch(query) {
  const q = (query || "").trim();
  if (!q) {
    clearFilter();
    return;
  }
  applyFilter(q);
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
      render();

      plotEl.on("plotly_click", (evt) => {
        const wellId = evt.points[0].customdata;
        if (!wellId) return;
        if (wellId === selectedWellId) {
          clearSelection();
        } else {
          selectWell(wellId, { center: false });
        }
      });
    });

  document.getElementById("color-by-moa").addEventListener("click", () => setColorBy("moa"));
  document.getElementById("color-by-plate").addEventListener("click", () => setColorBy("plate"));

  // Delegated listener: _right_sidebar.html is swapped in via innerHTML on
  // every well selection, so the Image/Attention/Overlay buttons don't exist
  // yet at initMap() time -- bind once on the stable parent instead of
  // re-binding after every render.
  document.getElementById("sidebar-content").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-view]");
    if (btn) setImageView(btn.dataset.view);
  });

  const input = document.getElementById("search-input");
  document.getElementById("search-btn").addEventListener("click", () => performSearch(input.value));
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") performSearch(input.value);
  });
  // Clearing the box (e.g. selecting all + delete) restores original
  // coloring immediately, without needing a separate "clear filter" control.
  input.addEventListener("input", () => {
    if (!input.value.trim()) clearFilter();
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

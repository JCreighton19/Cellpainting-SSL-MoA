// Drives the whole single-page explorer: renders the 2D UMAP, colors it by
// MoA or plate, filters the map by search text, and wires point clicks to
// fetch a right-sidebar detail partial without ever navigating away from
// this page.

const TOP_N_MOAS = 20;
// tab10 (10 saturated hues, gray swapped for olive to stay distinct from the
// OTHER_ANNOTATED_COLOR/UNANNOTATED_COLOR grays below) + tab20's lighter
// variants of the same 10 hues. "Control Vehicle" is consistently this
// dataset's most common MoA and gets palette[0] (#1f77b4, blue).
const MOA_PALETTE = [
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
  "#8c564b", "#e377c2", "#7f7c00", "#17becf", "#bcbd22",
  "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
  "#c49c94", "#f7b6d2", "#dbdb8d", "#9edae5", "#393b79",
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
let neighborLinesTraceIndex = null;
let neighborMarkersTraceIndex = null;
// well_ids of the selected point's true nearest neighbors (full embedding
// space, from the sidebar's neighbor list), so render() can redraw them
// after a color-by/filter change rebuilds every trace from scratch.
let selectedNeighborIds = [];
let currentColorBy = "moa";
// Set of well_id strings matching the active search filter, or null when no
// filter is active (i.e. every point renders in its normal color).
let matchSet = null;
// Legend keys (MoA name, "other-annotated", "unannotated", or plate name)
// toggled off via the custom legend; persists across re-renders since
// buildMoaTraces/buildPlateTraces rebuild every trace from scratch each call.
let hiddenLegendKeys = new Set();
// Key of the single legend item currently isolated (every other key hidden),
// or null when none is isolated. Lets a second click on the same item tell
// "isolate this one" apart from "restore all" -- see renderLegend().
let isolatedLegendKey = null;
// Pending single-click timer, used to tell a real single click apart from
// the first half of a double-click -- see handleLegendClick().
let legendClickTimer = null;
const LEGEND_DBLCLICK_MS = 300;

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
// the muted background ones, filter active or not. When a filter IS active,
// matched points are pulled out of their color-group trace entirely and
// drawn in one shared trace pushed last (see matched/buildMatchedTrace
// below), so a match can never be hidden under a later group's gray points.
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
  const matched = emptyBucket();

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
    if (matchSet && matches) {
      pushPoint(matched, data, i, MATCH_HIGHLIGHT_COLOR, baseOpacity);
    } else if (matches) {
      pushPoint(groups[key].bright, data, i, baseColor, baseOpacity);
    } else {
      pushPoint(groups[key].dim, data, i, FILTERED_OUT_COLOR, FILTERED_OUT_OPACITY);
    }
  }

  const traces = [];
  const legendItems = [];
  [["unannotated", "no annotated MoA", 5, UNANNOTATED_COLOR], ["other-annotated", "other annotated MoA (grouped for readability)", 5, OTHER_ANNOTATED_COLOR]]
    .forEach(([key, name, size, swatchColor]) => {
      const g = mergeDimBright(groups[key].dim, groups[key].bright);
      traces.push({
        x: g.x, y: g.y, text: g.text, customdata: g.customdata,
        mode: "markers", type: "scattergl", name, showlegend: false,
        visible: hiddenLegendKeys.has(key) ? "legendonly" : true,
        marker: { size, color: g.color, opacity: g.opacity },
        hoverinfo: "text",
      });
      legendItems.push({ key, label: name, color: swatchColor });
    });
  topMoas.forEach((m) => {
    const g = mergeDimBright(groups[m].dim, groups[m].bright);
    traces.push({
      x: g.x, y: g.y, text: g.text, customdata: g.customdata,
      mode: "markers", type: "scattergl", name: m, showlegend: false,
      visible: hiddenLegendKeys.has(m) ? "legendonly" : true,
      marker: { size: 7, color: g.color, opacity: g.opacity },
      hoverinfo: "text",
    });
    legendItems.push({ key: m, label: m, color: topMoaColor[m] });
  });
  pushMatchedTrace(traces, matched);
  return { traces, legendItems };
}

// Shared trace for search-matched points, appended after every color-group
// trace so it always paints on top and a match is never obscured by a later
// group's gray (filtered-out) points. Omitted entirely when there's no
// active filter or nothing matched.
function pushMatchedTrace(traces, matched) {
  if (!matchSet || !matched.x.length) return;
  traces.push({
    x: matched.x, y: matched.y, text: matched.text, customdata: matched.customdata,
    mode: "markers", type: "scattergl", name: "search match", showlegend: false,
    marker: { size: 8, color: matched.color, opacity: matched.opacity },
    hoverinfo: "text",
  });
}

function buildPlateTraces(data) {
  const plates = Array.from(new Set(data.plate)).sort();
  const plateColor = {};
  plates.forEach((p, i) => { plateColor[p] = PLATE_PALETTE[i % PLATE_PALETTE.length]; });

  const groups = {};
  plates.forEach((p) => { groups[p] = { dim: emptyBucket(), bright: emptyBucket() }; });
  const matched = emptyBucket();

  for (let i = 0; i < data.well_id.length; i++) {
    const p = data.plate[i];
    const matches = !matchSet || matchSet.has(data.well_id[i]);
    if (matchSet && matches) {
      pushPoint(matched, data, i, MATCH_HIGHLIGHT_COLOR, MARKER_OPACITY);
    } else if (matches) {
      pushPoint(groups[p].bright, data, i, plateColor[p], MARKER_OPACITY);
    } else {
      pushPoint(groups[p].dim, data, i, FILTERED_OUT_COLOR, FILTERED_OUT_OPACITY);
    }
  }

  const traces = plates.map((p) => {
    const g = mergeDimBright(groups[p].dim, groups[p].bright);
    return {
      x: g.x, y: g.y, text: g.text, customdata: g.customdata,
      mode: "markers", type: "scattergl", name: p, showlegend: false,
      visible: hiddenLegendKeys.has(p) ? "legendonly" : true,
      marker: { size: 6, color: g.color, opacity: g.opacity },
      hoverinfo: "text",
    };
  });
  const legendItems = plates.map((p) => ({ key: p, label: p, color: plateColor[p] }));
  pushMatchedTrace(traces, matched);
  return { traces, legendItems };
}

function buildHighlightTrace() {
  return {
    x: [], y: [], mode: "markers", type: "scattergl",
    name: "selected", showlegend: false, hoverinfo: "skip",
    marker: { size: 16, color: "rgba(0,0,0,0)", line: { color: "#000", width: 2 } },
  };
}

// Two traces showing the selected point's TRUE nearest neighbors (computed
// server-side in the full embedding space, not 2D UMAP position) -- see
// updateNeighborHighlights(). Dotted connector lines make it obvious when a
// neighbor is visually far away on the map, which is the point: it makes the
// UMAP-vs-embedding-space discrepancy explained in "How to Read This Map"
// visible instead of just asserted.
function buildNeighborTraces() {
  return [
    {
      x: [], y: [], mode: "lines", type: "scattergl",
      name: "neighbor-lines", showlegend: false, hoverinfo: "skip",
      line: { color: "#6c757d", width: 1, dash: "dot" },
    },
    {
      x: [], y: [], mode: "markers", type: "scattergl",
      name: "true nearest neighbors", showlegend: false, hoverinfo: "skip",
      marker: { size: 13, color: "rgba(0,0,0,0)", line: { color: "#0d6efd", width: 2 } },
    },
  ];
}

function updateNeighborHighlights(wellId, neighborIds) {
  const idx = umapData.well_id.indexOf(wellId);
  if (idx === -1 || neighborLinesTraceIndex === null) return;
  const cx = umapData.x[idx];
  const cy = umapData.y[idx];
  const lineX = [], lineY = [], nx = [], ny = [];
  neighborIds.forEach((nid) => {
    const ni = umapData.well_id.indexOf(nid);
    if (ni === -1) return;
    lineX.push(cx, umapData.x[ni], null);
    lineY.push(cy, umapData.y[ni], null);
    nx.push(umapData.x[ni]);
    ny.push(umapData.y[ni]);
  });
  Plotly.restyle(plotEl, { x: [lineX], y: [lineY] }, [neighborLinesTraceIndex]);
  Plotly.restyle(plotEl, { x: [nx], y: [ny] }, [neighborMarkersTraceIndex]);
}

function clearNeighborHighlights() {
  selectedNeighborIds = [];
  if (neighborLinesTraceIndex === null) return;
  Plotly.restyle(plotEl, { x: [[]], y: [[]] }, [neighborLinesTraceIndex]);
  Plotly.restyle(plotEl, { x: [[]], y: [[]] }, [neighborMarkersTraceIndex]);
}

function defaultLayout() {
  return {
    margin: { t: 10, r: 10, b: 40, l: 40 },
    xaxis: { title: "UMAP 1" },
    yaxis: { title: "UMAP 2" },
    // Plotly's own legend is a single column and can't be reflowed into
    // multiple columns, so it's hidden here in favor of the custom
    // #map-legend built by renderLegend() below.
    showlegend: false,
    dragmode: "pan",
    hovermode: "closest",
  };
}

// Single click: isolate this key (hide every other), or restore all if it's
// already the isolated one.
function isolateLegendKey(key, items) {
  if (isolatedLegendKey === key) {
    isolatedLegendKey = null;
    hiddenLegendKeys.clear();
  } else {
    isolatedLegendKey = key;
    hiddenLegendKeys = new Set(items.map((i) => i.key).filter((k) => k !== key));
  }
  render();
}

// Double click: hide/show just this one key, independent of isolation.
function toggleLegendKeyHidden(key) {
  if (hiddenLegendKeys.has(key)) hiddenLegendKeys.delete(key);
  else hiddenLegendKeys.add(key);
  // An explicit hide/show of one key no longer matches the "exactly one key
  // visible" state isolate tracks, so drop it rather than leave it stale.
  isolatedLegendKey = null;
  render();
}

// A native "dblclick" listener alongside a "click" listener would fire click
// twice before dblclick on every double-click, isolating and un-isolating
// the item before the double-click handler even runs. Debouncing click by
// hand avoids that: a click is only treated as a single click if no second
// click follows within LEGEND_DBLCLICK_MS, otherwise it's a double-click.
function handleLegendClick(key, items) {
  if (legendClickTimer) {
    clearTimeout(legendClickTimer);
    legendClickTimer = null;
    toggleLegendKeyHidden(key);
    return;
  }
  legendClickTimer = setTimeout(() => {
    legendClickTimer = null;
    isolateLegendKey(key, items);
  }, LEGEND_DBLCLICK_MS);
}

// Custom legend so entries can wrap into multiple CSS columns (see
// .map-legend in style.css) instead of Plotly's single-column/single-row
// legend. Click isolates a color (hides every other); click it again to
// restore all. Double-click hides/shows just that one color. The caption
// above #map-legend (index.html) explains this to users, so no per-item
// title/tooltip is needed here.
function renderLegend(items) {
  const el = document.getElementById("map-legend");
  if (!el) return;
  el.innerHTML = "";
  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "legend-item" + (hiddenLegendKeys.has(item.key) ? " legend-item-hidden" : "");
    const swatch = document.createElement("span");
    swatch.className = "legend-swatch";
    swatch.style.backgroundColor = item.color;
    const label = document.createElement("span");
    label.className = "legend-label";
    label.textContent = item.label;
    row.append(swatch, label);
    row.addEventListener("click", () => handleLegendClick(item.key, items));
    el.appendChild(row);
  });
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
  const result = currentColorBy === "plate" ? buildPlateTraces(umapData) : buildMoaTraces(umapData);
  const traces = result.traces;
  highlightTraceIndex = traces.length;
  traces.push(buildHighlightTrace());
  neighborLinesTraceIndex = traces.length;
  const neighborTraces = buildNeighborTraces();
  traces.push(...neighborTraces);
  neighborMarkersTraceIndex = neighborLinesTraceIndex + 1;
  Plotly.react(plotEl, traces, plotEl.layout || defaultLayout(), PLOT_CONFIG);
  renderLegend(result.legendItems);
  // Plotly sizes the plot against #umap-plot's height at the moment
  // Plotly.react() runs, which is BEFORE renderLegend() above has given
  // #map-legend its real (non-zero) height -- so without this, the plot
  // claims the space the legend needs and the legend doesn't visually
  // appear until something else (e.g. dragging a sidebar) forces a resize.
  Plotly.Plots.resize(plotEl);
  if (selectedWellId) {
    updateHighlightPosition(selectedWellId);
    updateNeighborHighlights(selectedWellId, selectedNeighborIds);
  }
}

function setColorBy(mode) {
  currentColorBy = mode;
  document.getElementById("color-by-moa").classList.toggle("active", mode === "moa");
  document.getElementById("color-by-plate").classList.toggle("active", mode === "plate");
  // MoA and plate keys are different namespaces entirely, so any hidden/
  // isolated state from the previous mode wouldn't apply to the new one --
  // reset rather than leave a stale, invisible toggle behind.
  hiddenLegendKeys.clear();
  isolatedLegendKey = null;
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
    .then((html) => {
      document.getElementById("sidebar-content").innerHTML = html;
      const idsEl = document.getElementById("neighbor-well-ids");
      selectedNeighborIds = idsEl && idsEl.dataset.ids ? idsEl.dataset.ids.split(",") : [];
      updateNeighborHighlights(wellId, selectedNeighborIds);
    });
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
  if (thumb) {
    thumb.classList.toggle("d-none", view === "attention");
    thumb.classList.toggle("overlay-brighten", view === "overlay");
  }
}

// Two ways in: clicking the already-selected point again (handled in the
// plotly_click listener in initMap()), or clicking empty canvas (handled by
// the plain DOM click listener there, since plotly_click never fires for
// that case).
function clearSelection() {
  selectedWellId = null;
  Plotly.restyle(plotEl, { x: [[]], y: [[]] }, [highlightTraceIndex]);
  clearNeighborHighlights();
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
  document.getElementById("clear-filter-btn").classList.remove("d-none");
}

function clearFilter() {
  document.getElementById("clear-filter-btn").classList.add("d-none");
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

      let hitMarkerThisClick = false;
      plotEl.on("plotly_click", (evt) => {
        hitMarkerThisClick = true;
        const wellId = evt.points[0].customdata;
        if (!wellId) return;
        if (wellId === selectedWellId) {
          clearSelection();
        } else {
          selectWell(wellId, { center: false });
        }
      });

      // plotly_click only fires when a marker is actually hit, so it can't
      // tell us about a click on empty canvas -- a plain DOM listener on the
      // plot container catches those too. Deferred via setTimeout(0) so it
      // runs after plotly_click (same underlying click) has had a chance to
      // set hitMarkerThisClick first; skips the modebar so "Reset axes"
      // doesn't also clear the current selection.
      plotEl.addEventListener("click", (e) => {
        if (e.target.closest(".modebar")) return;
        setTimeout(() => {
          if (!hitMarkerThisClick && selectedWellId) clearSelection();
          hitMarkerThisClick = false;
        }, 0);
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
  // coloring immediately, in addition to the explicit "Clear filter" button.
  input.addEventListener("input", () => {
    if (!input.value.trim()) clearFilter();
  });
  document.getElementById("clear-filter-btn").addEventListener("click", () => {
    input.value = "";
    clearFilter();
  });

  makeResizable(document.getElementById("resize-handle-left"), document.getElementById("sidebar-left"), "left");
  makeResizable(document.getElementById("resize-handle-right"), document.getElementById("sidebar-right"), "right");
}

document.addEventListener("DOMContentLoaded", initMap);

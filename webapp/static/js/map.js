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

// Single shared marker size for every real data point (MoA/plate/background/
// search-match traces alike) -- previously these varied (5/6/7/8) by trace,
// which made some groups visually more prominent for no meaningful reason.
// Overlay traces (selection ring, neighbor ring) are a separate concept and
// keep their own larger sizes -- see buildHighlightTrace/buildNeighborTraces.
const POINT_SIZE = 6;

// Used to de-emphasize points that don't match the active search filter.
const FILTERED_OUT_COLOR = "#e9ecef";
const FILTERED_OUT_OPACITY = 0.25;
// Used instead of the normal MoA/plate color for points that DO match an
// active search filter, so matches jump out regardless of colorBy mode.
const MATCH_HIGHLIGHT_COLOR = "#00e676";

// 3D UMAP is a separate offline fit (n_components=3, see
// scripts/prepare_phase1_data.py), not a 3rd axis bolted onto the 2D layout
// -- wells.parquet carries both umap_x/umap_y (2D) and umap_x_3d/umap_y_3d/
// umap_z_3d (3D) side by side, and /api/umap returns both (x/y and x3d/y3d/z3d).
// The `is3D` flag below switches trace type (scattergl <-> scatter3d) and
// coordinate source; 2D remains the default and is otherwise untouched.
let is3D = false;

let plotEl = null;
let umapData = null;
let selectedWellId = null;
let highlightTraceIndex = null;
let neighborLinesTraceIndex = null;
let neighborMarkersTraceIndex = null;
// 3D's selected-point indicator: a plain DOM element positioned via manual
// camera projection (see startHighlight3D) rather than a Plotly trace, since
// even a single-point Plotly.restyle() on the 3D scene is expensive enough to
// freeze the UI (gl3d rebuilds every trace in the scene on any restyle).
let highlight3DEl = null;
let highlight3DRaf = null;
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
// Set of legend keys currently isolated (every key NOT in this set is
// hidden); empty when nothing is isolated (every key shown). A single click
// toggles that key's own membership in the set, so multiple colors can be
// isolated together at once instead of only ever one -- see
// isolateLegendKey() below.
let isolatedLegendKeys = new Set();
// Pending single-click timer, used to tell a real single click apart from
// the first half of a double-click -- see handleLegendClick().
let legendClickTimer = null;
const LEGEND_DBLCLICK_MS = 300;
// Thresholds for telling a real click apart from a 3D orbit-rotate drag that
// happens to end over empty canvas -- see the mousedown/click listeners in
// initMap().
const CLICK_MAX_DURATION_MS = 400;
const CLICK_MAX_MOVE_PX = 5;

function hoverText(data, i) {
  return `Plate: ${data.plate[i]}<br>MoA: ${data.moa[i]}`;
}

// Coordinate source for point i: the 2D UMAP (x/y) or the separate 3D UMAP
// fit (x3d/y3d/z3d), depending on the active view -- see `is3D` above.
function getCoords(data, i) {
  return is3D
    ? { x: data.x3d[i], y: data.y3d[i], z: data.z3d[i] }
    : { x: data.x[i], y: data.y[i], z: undefined };
}

function emptyBucket() {
  return { x: [], y: [], z: [], text: [], customdata: [], color: [], opacity: [] };
}

function pushPoint(bucket, data, i, color, opacity) {
  const c = getCoords(data, i);
  bucket.x.push(c.x);
  bucket.y.push(c.y);
  if (is3D) bucket.z.push(c.z);
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
    z: dim.z.concat(bright.z),
    text: dim.text.concat(bright.text),
    customdata: dim.customdata.concat(bright.customdata),
    color: dim.color.concat(bright.color),
    opacity: dim.opacity.concat(bright.opacity),
  };
}

// Trace type + optional z, shared by every point trace builder below so
// scattergl (2D) vs scatter3d (3D) only needs deciding in one place.
function makeTrace(g, { name, size, visible }) {
  const trace = {
    x: g.x, y: g.y, text: g.text, customdata: g.customdata,
    mode: "markers", type: is3D ? "scatter3d" : "scattergl", name, showlegend: false,
    visible,
    marker: { size, color: g.color, opacity: g.opacity },
    hoverinfo: "text",
  };
  if (is3D) trace.z = g.z;
  return trace;
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
// Cached the first time it's computed -- the underlying data (and therefore
// each MoA's frequency rank) never changes over the page's lifetime, and
// this is also what the sidebar's MoA badge calls (see getMoaColor below) to
// guarantee it always matches the map's own colors exactly, in whichever
// mode ("Color by" MoA or Plate) the map currently happens to be in.
let cachedTopMoaColor = null;

function computeTopMoaColor(data) {
  const counts = {};
  data.moa.forEach((m) => { counts[m] = (counts[m] || 0) + 1; });
  const topMoas = Object.keys(counts)
    .filter((m) => m !== "Unannotated")
    .sort((a, b) => counts[b] - counts[a])
    .slice(0, TOP_N_MOAS);
  const topMoaColor = {};
  topMoas.forEach((m, i) => { topMoaColor[m] = MOA_PALETTE[i % MOA_PALETTE.length]; });
  return topMoaColor;
}

// Looks up the exact color a MoA is drawn in on the map (or the same
// gray used for "other annotated" MoAs outside the top N), for the
// sidebar's MoA badge -- see applyMoaBadgeColor().
function getMoaColor(moa) {
  if (!umapData) return null;
  if (!cachedTopMoaColor) cachedTopMoaColor = computeTopMoaColor(umapData);
  return cachedTopMoaColor[moa] || OTHER_ANNOTATED_COLOR;
}

// Simple YIQ perceived-brightness check (the common heuristic for "does
// this background need light or dark text") -- picks black text for the
// lighter/brighter half of the MoA palette, white for the darker half.
function pickTextColor(hex) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  const brightness = (r * 299 + g * 587 + b * 114) / 1000;
  return brightness >= 128 ? "#000" : "#fff";
}

// Colors the sidebar's MoA badge (see partials/_right_sidebar.html) to match
// its dot color on the map, instead of a fixed color unrelated to the
// legend. No-op if this well has no MoA badge (control wells, etc.).
function applyMoaBadgeColor() {
  const badge = document.querySelector("#sidebar-content .moa-badge");
  if (!badge) return;
  const color = getMoaColor(badge.dataset.moa);
  if (!color) return;
  badge.style.backgroundColor = color;
  badge.style.color = pickTextColor(color);
}

function buildMoaTraces(data) {
  const topMoaColor = computeTopMoaColor(data);
  const topMoas = Object.keys(topMoaColor);

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
  [["unannotated", "no annotated MoA", UNANNOTATED_COLOR], ["other-annotated", "other annotated MoA (grouped for readability)", OTHER_ANNOTATED_COLOR]]
    .forEach(([key, name, swatchColor]) => {
      const g = mergeDimBright(groups[key].dim, groups[key].bright);
      traces.push(makeTrace(g, { name, size: POINT_SIZE, visible: hiddenLegendKeys.has(key) ? "legendonly" : true }));
      legendItems.push({ key, label: name, color: swatchColor });
    });
  topMoas.forEach((m) => {
    const g = mergeDimBright(groups[m].dim, groups[m].bright);
    traces.push(makeTrace(g, { name: m, size: POINT_SIZE, visible: hiddenLegendKeys.has(m) ? "legendonly" : true }));
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
  traces.push(makeTrace(matched, { name: "search match", size: POINT_SIZE, visible: true }));
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
    return makeTrace(g, { name: p, size: POINT_SIZE, visible: hiddenLegendKeys.has(p) ? "legendonly" : true });
  });
  const legendItems = plates.map((p) => ({ key: p, label: p, color: plateColor[p] }));
  pushMatchedTrace(traces, matched);
  return { traces, legendItems };
}

// Built in both 2D and 3D (unlike the neighbor traces below, which 3D skips
// entirely -- see render()): a single point restyled on click is cheap
// enough even under gl3d's whole-scene-rebuild-per-restyle behavior, unlike
// the neighbor lines/markers which scale with neighbor count on top of that.
// 2D only -- see render()'s is3D branch and startHighlight3D for 3D's
// restyle-free equivalent.
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
// visible instead of just asserted. 2D only -- see render().
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

// No-op in 3D (neighborLinesTraceIndex is null there -- see render()): gl3d
// rebuilds the ENTIRE scene's point buffers on any restyle to any trace
// bound to it (Scene.prototype.plot in plotly.js loops over every trace in
// the scene, not just the restyled ones), so restyling this on every click
// forced a full ~5,100-point scene rebuild each time. The sidebar's neighbor
// list/stats already show this information without needing an on-map
// overlay, so 3D just skips it instead of paying that cost.
function updateNeighborHighlights(wellId, neighborIds) {
  if (neighborLinesTraceIndex === null) return;
  const idx = umapData.well_id.indexOf(wellId);
  if (idx === -1) return;
  const lineX = [], lineY = [], nx = [], ny = [];
  neighborIds.forEach((nid) => {
    const ni = umapData.well_id.indexOf(nid);
    if (ni === -1) return;
    lineX.push(umapData.x[idx], umapData.x[ni], null);
    lineY.push(umapData.y[idx], umapData.y[ni], null);
    nx.push(umapData.x[ni]);
    ny.push(umapData.y[ni]);
  });
  // One restyle across both trace indices instead of two separate calls.
  Plotly.restyle(plotEl, { x: [lineX, nx], y: [lineY, ny] }, [neighborLinesTraceIndex, neighborMarkersTraceIndex]);
}

// UMAP axis titles don't carry meaningful units (see "How to Read This Map"),
// so the title is de-emphasized -- small, muted font instead of Plotly's bold
// default -- but otherwise 2D keeps its original, fully-interactive axes
// (grid, zeroline, tick labels all still Plotly's normal behavior as you pan/
// zoom); only tick marks (the little perpendicular dashes) are turned off,
// same as Plotly's own default for this axis type.
const AXIS_LINE_COLOR = "#ced4da";
function axisTitle(text) {
  return { text, font: { size: 11, color: "#868e96" } };
}
// Native scatter3d axis lines are drawn along whichever box edge is
// currently "back-facing" the camera, so they jump to a different edge as
// the view rotates -- showline: false here turns that off in favor of the
// fixed, origin-crossing lines added as their own traces (see
// buildOriginAxisTraces), which don't move.
const CLEAN_3D_AXIS = {
  showgrid: false, zeroline: false, showticklabels: false, ticks: "",
  showspikes: false,
  showline: false,
};

function default2DLayout() {
  return {
    margin: { t: 10, r: 10, b: 40, l: 40 },
    xaxis: { title: axisTitle("UMAP 1"), ticks: "" },
    yaxis: { title: axisTitle("UMAP 2"), ticks: "" },
    // Plotly's own legend is a single column and can't be reflowed into
    // multiple columns, so it's hidden here in favor of the custom
    // #map-legend built by renderLegend() below.
    showlegend: false,
    dragmode: "pan",
    hovermode: "closest",
  };
}

// Padded [lo, hi] extent of arr, extended to include 0 if the data doesn't
// already reach it (so an origin-crossing axis line always actually crosses).
// Shared by default3DLayout (explicit scene axis range) and
// buildOriginAxisTraces (axis line endpoints) so both agree on the same
// bounds -- see default3DLayout for why the range needs to be explicit.
function compute3DExtent(arr) {
  let lo = 0, hi = 0;
  for (const v of arr) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const pad = (hi - lo) * 0.05;
  return [lo - pad, hi + pad];
}

// scatter3d traces live in a "scene", not top-level xaxis/yaxis -- navigation
// is orbit/zoom (Plotly's built-in scatter3d camera controls) rather than the
// 2D pan/zoom above, so there's no dragmode to set here. showspikes: false
// on all three axes turns off the dashed projection lines Plotly otherwise
// draws from a hovered/selected point to each scene wall.
//
// Explicit range (instead of leaving it on Plotly's default autorange) matters
// for more than layout: without it, every Plotly.restyle() that moves the
// highlight/neighbor traces (i.e. every point click) makes gl3d recompute the
// scene's bounding box from all traces' data before it can redraw -- on top of
// gl3d's own full-scene redraw per restyle, that's real, avoidable cost on
// every single click. A fixed range (points never fall outside it) removes
// that recompute.
function default3DLayout() {
  return {
    margin: { t: 10, r: 10, b: 10, l: 10 },
    scene: {
      xaxis: { title: axisTitle("UMAP 1"), range: compute3DExtent(umapData.x3d), ...CLEAN_3D_AXIS },
      yaxis: { title: axisTitle("UMAP 2"), range: compute3DExtent(umapData.y3d), ...CLEAN_3D_AXIS },
      zaxis: { title: axisTitle("UMAP 3"), range: compute3DExtent(umapData.z3d), ...CLEAN_3D_AXIS },
    },
    showlegend: false,
    hovermode: "closest",
  };
}

// Fixed, view-independent stand-in for the native (moving) 3D axis lines:
// three line traces crossing at the data origin, each spanning that axis's
// full (padded) range -- see compute3DExtent. Plain data traces, so they
// don't move when the camera orbits, unlike scene.xaxis/yaxis/zaxis's own
// axis line.
function buildOriginAxisTraces(data) {
  const [xlo, xhi] = compute3DExtent(data.x3d);
  const [ylo, yhi] = compute3DExtent(data.y3d);
  const [zlo, zhi] = compute3DExtent(data.z3d);
  const axisLine = (x, y, z) => ({
    x, y, z, mode: "lines", type: "scatter3d",
    name: "axis-line", showlegend: false, hoverinfo: "skip",
    line: { color: AXIS_LINE_COLOR, width: 3 },
  });
  return [
    axisLine([xlo, xhi], [0, 0], [0, 0]),
    axisLine([0, 0], [ylo, yhi], [0, 0]),
    axisLine([0, 0], [0, 0], [zlo, zhi]),
  ];
}

function defaultLayout() {
  return is3D ? default3DLayout() : default2DLayout();
}

// Single click: toggle this key's membership in the isolated set. Clicking
// additional keys adds them to the set (isolating several colors together,
// not just one); clicking an already-isolated key removes it; emptying the
// set restores every key.
function isolateLegendKey(key, items) {
  if (isolatedLegendKeys.has(key)) {
    isolatedLegendKeys.delete(key);
  } else {
    isolatedLegendKeys.add(key);
  }
  if (isolatedLegendKeys.size === 0) {
    hiddenLegendKeys.clear();
  } else {
    hiddenLegendKeys = new Set(items.map((i) => i.key).filter((k) => !isolatedLegendKeys.has(k)));
  }
  render();
}

// Double click: hide/show just this one key, independent of isolation.
function toggleLegendKeyHidden(key) {
  if (hiddenLegendKeys.has(key)) hiddenLegendKeys.delete(key);
  else hiddenLegendKeys.add(key);
  // An explicit hide/show of one key no longer matches the isolated-set
  // state isolate tracks, so drop it rather than leave it stale.
  isolatedLegendKeys.clear();
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
// to just "Reset axes" so normal navigation never needs it. The 3D-only
// entries here (tableRotation, resetCameraLastSave3d) are harmless no-ops
// when the current plot is 2D -- Plotly silently ignores removal requests
// for buttons that don't apply to the active trace type.
const PLOT_CONFIG = {
  responsive: true,
  scrollZoom: true,
  displaylogo: false,
  modeBarButtonsToRemove: [
    "zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d",
    "autoScale2d", "hoverClosestCartesian", "hoverCompareCartesian",
    "toggleSpikelines", "toImage",
    "tableRotation", "resetCameraLastSave3d",
  ],
};

// Re-renders using the current colorBy mode and search filter (module state
// above), so any caller that changes either one just calls render() again.
// Pass resetLayout: true when switching 2D/3D -- scatter3d's "scene" layout
// and scattergl's "xaxis"/"yaxis" layout aren't interchangeable, so the old
// plotEl.layout (from the other mode) can't be reused there.
function render(opts) {
  opts = opts || {};
  const result = currentColorBy === "plate" ? buildPlateTraces(umapData) : buildMoaTraces(umapData);
  const traces = result.traces;
  // 3D skips the highlight-ring/neighbor-line overlay traces entirely --
  // see updateHighlightPosition/startHighlight3D and updateNeighborHighlights
  // for why (both restyle-free in 3D instead).
  if (is3D) {
    highlightTraceIndex = null;
    neighborLinesTraceIndex = null;
    neighborMarkersTraceIndex = null;
    traces.push(...buildOriginAxisTraces(umapData));
  } else {
    highlightTraceIndex = traces.length;
    traces.push(buildHighlightTrace());
    neighborLinesTraceIndex = traces.length;
    traces.push(...buildNeighborTraces());
    neighborMarkersTraceIndex = neighborLinesTraceIndex + 1;
  }
  const layout = opts.resetLayout ? defaultLayout() : (plotEl.layout || defaultLayout());
  Plotly.react(plotEl, traces, layout, PLOT_CONFIG);
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
  isolatedLegendKeys.clear();
  render();
}

// Toggles between the 2D UMAP (default) and the separate 3D UMAP fit. Colors,
// legend, search filter, and selection all carry over unchanged -- only the
// coordinate source, trace type, and layout (see render's resetLayout) differ.
function setDimension(is3DNext) {
  if (is3DNext === is3D) return;
  // Stop tracking regardless of which direction we're switching -- render()
  // below re-establishes the right one (a fresh 2D restyle, or a fresh
  // startHighlight3D loop) if there's still a selection.
  stopHighlight3D();
  is3D = is3DNext;
  render({ resetLayout: true });
}

function updateHighlightPosition(wellId) {
  if (is3D) {
    startHighlight3D(wellId);
    return;
  }
  if (highlightTraceIndex === null) return;
  const idx = umapData.well_id.indexOf(wellId);
  if (idx === -1) return;
  Plotly.restyle(plotEl, { x: [[umapData.x[idx]]], y: [[umapData.y[idx]]] }, [highlightTraceIndex]);
}

// Same model/view/projection transform gl3d itself uses on every point (see
// plotly.js's src/plots/gl3d/project.js -- not public API, so reimplemented
// here) to convert a raw (x, y, z) data coordinate to clip space. Reading
// these matrices costs nothing extra: Plotly already recomputes them every
// frame while orbiting, whether or not we look at them.
function projectPoint3D(cameraParams, v) {
  const xform = (m, p) => {
    const out = [0, 0, 0, 0];
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) out[j] += m[4 * i + j] * p[i];
    }
    return out;
  };
  return xform(cameraParams.projection, xform(cameraParams.view, xform(cameraParams.model, [v[0], v[1], v[2], 1])));
}

function get3DScene() {
  const fullLayout = plotEl && plotEl._fullLayout;
  return (fullLayout && fullLayout.scene && fullLayout.scene._scene) || null;
}

function ensureHighlight3DEl(container) {
  if (highlight3DEl && highlight3DEl.parentNode !== container) {
    highlight3DEl.parentNode.removeChild(highlight3DEl);
    highlight3DEl = null;
  }
  if (!highlight3DEl) {
    highlight3DEl = document.createElement("div");
    highlight3DEl.className = "umap-3d-highlight";
    container.appendChild(highlight3DEl);
  }
  return highlight3DEl;
}

function stopHighlight3D() {
  if (highlight3DRaf !== null) {
    cancelAnimationFrame(highlight3DRaf);
    highlight3DRaf = null;
  }
  if (highlight3DEl) highlight3DEl.style.display = "none";
}

// 3D's restyle-free stand-in for updateHighlightPosition: projects the
// selected point's data coordinate to screen pixels every animation frame
// (so it tracks the point through camera orbit/zoom) and positions a plain
// DOM dot there -- no Plotly.restyle()/react()/relayout() call involved, so
// it never triggers gl3d's whole-scene rebuild. Stops itself (see the guard
// below) once the selection changes or the scene becomes unavailable, e.g.
// after switching back to 2D.
function startHighlight3D(wellId) {
  if (highlight3DRaf !== null) cancelAnimationFrame(highlight3DRaf);

  function tick() {
    const scene = get3DScene();
    const idx = umapData.well_id.indexOf(wellId);
    if (!is3D || !scene || !scene.glplot || idx === -1 || wellId !== selectedWellId) {
      stopHighlight3D();
      return;
    }
    const container = scene.container;
    const el = ensureHighlight3DEl(container);
    // gl3d doesn't feed raw data values into the camera matrices -- every
    // vertex is pre-scaled by scene.dataScale first (confirmed directly in
    // plotly.js's scatter3d/convert.js: "xc = ... * scaleFactor[0]"), so the
    // point we project has to go through that same scaling or it lands
    // nowhere near the actual rendered point.
    const ds = scene.dataScale || [1, 1, 1];
    const p = projectPoint3D(scene.glplot.cameraParams, [
      umapData.x3d[idx] * ds[0],
      umapData.y3d[idx] * ds[1],
      umapData.z3d[idx] * ds[2],
    ]);
    const ndcX = p[3] > 0 ? p[0] / p[3] : NaN;
    const ndcY = p[3] > 0 ? p[1] / p[3] : NaN;
    // Behind the camera, or rotated outside the visible viewport (valid NDC
    // is -1..1 on each axis) -- hide rather than draw over the legend/
    // sidebars, which happened before since nothing clamped the projected
    // position to the plot area.
    if (p[3] <= 0 || ndcX < -1 || ndcX > 1 || ndcY < -1 || ndcY > 1) {
      el.style.display = "none";
    } else {
      el.style.display = "block";
      el.style.left = `${((ndcX + 1) / 2) * container.clientWidth}px`;
      el.style.top = `${((1 - ndcY) / 2) * container.clientHeight}px`;
    }
    highlight3DRaf = requestAnimationFrame(tick);
  }
  tick();
}

// 2D-only: scatter3d navigates via orbit/zoom camera controls rather than an
// xaxis/yaxis range, so there's no equivalent "pan to point" in 3D -- callers
// guard with `!is3D` (see selectWell) rather than this being a no-op here.
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

  if (opts.center !== false && !is3D) {
    const idx = umapData.well_id.indexOf(wellId);
    if (idx !== -1) centerOn(umapData.x[idx], umapData.y[idx]);
  }

  fetch(`/api/well/${encodeURIComponent(wellId)}`)
    .then((r) => r.text())
    .then((html) => {
      document.getElementById("sidebar-content").innerHTML = html;
      applyMoaBadgeColor();
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
  selectedNeighborIds = [];
  stopHighlight3D();
  // highlightTraceIndex/neighborLinesTraceIndex/neighborMarkersTraceIndex are
  // all null in 3D (see render()), so this is 2D-only and a no-op in 3D.
  if (highlightTraceIndex !== null) {
    // Single restyle across all 3 traces instead of separate calls.
    const idxs = [highlightTraceIndex, neighborLinesTraceIndex, neighborMarkersTraceIndex];
    Plotly.restyle(plotEl, { x: idxs.map(() => []), y: idxs.map(() => []) }, idxs);
  }
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

      // Orbiting the 3D view is a mousedown-drag-mouseup gesture on the same
      // canvas element, which browsers still fire a native "click" for on
      // release regardless of how much the mouse moved in between -- without
      // this, every rotate ended with the click handler below mistaking it
      // for a deselect-click on empty canvas. Recorded on mousedown (not by
      // plotly_click, which only fires for actual marker hits and wouldn't
      // catch a drag that ends over empty canvas).
      let mouseDownInfo = null;
      plotEl.addEventListener("mousedown", (e) => {
        mouseDownInfo = { x: e.clientX, y: e.clientY, time: Date.now() };
      });

      // plotly_click only fires when a marker is actually hit, so it can't
      // tell us about a click on empty canvas -- a plain DOM listener on the
      // plot container catches those too. Deferred via setTimeout(0) so it
      // runs after plotly_click (same underlying click) has had a chance to
      // set hitMarkerThisClick first; skips the modebar so "Reset axes"
      // doesn't also clear the current selection.
      plotEl.addEventListener("click", (e) => {
        if (e.target.closest(".modebar")) return;
        const down = mouseDownInfo;
        mouseDownInfo = null;
        // Only relevant in 3D (2D drags to pan, not rotate, and isn't
        // reported as having this problem) -- a real click stays put and
        // resolves quickly; a rotate drag moves the mouse and/or takes a while.
        const wasDrag = is3D && down && (
          Date.now() - down.time > CLICK_MAX_DURATION_MS ||
          Math.hypot(e.clientX - down.x, e.clientY - down.y) > CLICK_MAX_MOVE_PX
        );
        setTimeout(() => {
          if (!hitMarkerThisClick && selectedWellId && !wasDrag) clearSelection();
          hitMarkerThisClick = false;
        }, 0);
      });
    });

  document.getElementById("color-by-moa").addEventListener("click", () => setColorBy("moa"));
  document.getElementById("color-by-plate").addEventListener("click", () => setColorBy("plate"));

  document.getElementById("view-3d-toggle").addEventListener("change", (e) => setDimension(e.target.checked));

  // Delegated listener: _right_sidebar.html is swapped in via innerHTML on
  // every well selection, so the Image/Attention/Overlay buttons don't exist
  // yet at initMap() time -- bind once on the stable parent instead of
  // re-binding after every render.
  document.getElementById("sidebar-content").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-view]");
    if (btn) {
      setImageView(btn.dataset.view);
      return;
    }
    // Clicking a "Nearest Neighbors" row jumps to that well exactly as if
    // its point on the map had been clicked directly -- same selectWell()
    // call, so the sidebar swaps to it and the 2D view recenters on it.
    const neighborCard = e.target.closest(".neighbor-card-compact[data-well-id]");
    if (neighborCard) selectWell(neighborCard.dataset.wellId);
  });

  // Closes any open "?" help tooltip (.info-tooltip, e.g. Neighborhood
  // Summary / Neighborhood consistency) on a click outside it -- native
  // <details> has no built-in click-outside-to-close. This also covers
  // "click the other help button": that click's default action (opening
  // the clicked tooltip) runs after this listener, so the previously open
  // one is already closed by the time the other one opens. A single
  // document-level listener works for both current and future tooltips
  // without needing to rebind after _right_sidebar.html is swapped in.
  document.addEventListener("click", (e) => {
    document.querySelectorAll(".info-tooltip[open]").forEach((details) => {
      if (!details.contains(e.target)) details.removeAttribute("open");
    });
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

// Minimal Plotly 2D UMAP scatter. Fetches /api/umap, colors by the top
// mechanisms of action (everything else grouped as "other"), and navigates
// to a well's detail page on click.

const TOP_N_MOAS = 10;
const PALETTE = [
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
  "#8c564b", "#e377c2", "#7f7c00", "#17becf", "#bcbd22",
];
const OTHER_ANNOTATED_COLOR = "#adb5bd";
const UNANNOTATED_COLOR = "#e9ecef";

// "other" and "unannotated" are kept as two separate buckets rather than one
// vague "other" group: most of what lands here DOES have an annotated MoA,
// it's just too rare to get its own legend color. See the explanatory text
// under the plot on the home page for the actual numbers.
function buildTraces(data) {
  const counts = {};
  data.moa.forEach((m) => { counts[m] = (counts[m] || 0) + 1; });
  const topMoas = Object.keys(counts)
    .filter((m) => m !== "unannotated")
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
    else if (moa === "unannotated") key = "unannotated";
    else key = "other-annotated";
    const g = groups[key];
    g.x.push(data.x[i]);
    g.y.push(data.y[i]);
    g.customdata.push(data.well_id[i]);
    g.text.push(`Plate: ${data.plate[i]}<br>MoA: ${moa}`);
  }

  const traces = [];
  topMoas.forEach((m, i) => {
    const g = groups[m];
    traces.push({
      x: g.x, y: g.y, text: g.text, customdata: g.customdata,
      mode: "markers", type: "scattergl", name: m,
      marker: { size: 6, color: PALETTE[i % PALETTE.length] },
      hoverinfo: "text",
    });
  });
  const otherAnnotated = groups["other-annotated"];
  traces.push({
    x: otherAnnotated.x, y: otherAnnotated.y, text: otherAnnotated.text, customdata: otherAnnotated.customdata,
    mode: "markers", type: "scattergl", name: "other annotated MoA (grouped for readability)",
    marker: { size: 5, color: OTHER_ANNOTATED_COLOR, opacity: 0.7 },
    hoverinfo: "text",
  });
  const unannotated = groups["unannotated"];
  traces.push({
    x: unannotated.x, y: unannotated.y, text: unannotated.text, customdata: unannotated.customdata,
    mode: "markers", type: "scattergl", name: "no annotated MoA",
    marker: { size: 5, color: UNANNOTATED_COLOR, opacity: 0.6 },
    hoverinfo: "text",
  });

  return traces;
}

function initMap() {
  const el = document.getElementById("umap-plot");
  if (!el) return;

  fetch("/api/umap")
    .then((r) => r.json())
    .then((data) => {
      const traces = buildTraces(data);
      Plotly.newPlot(el, traces, {
        margin: { t: 10, r: 10, b: 40, l: 40 },
        xaxis: { title: "UMAP 1" },
        yaxis: { title: "UMAP 2" },
        legend: { orientation: "h", y: -0.15 },
      }, { responsive: true });

      el.on("plotly_click", (evt) => {
        const wellId = evt.points[0].customdata;
        if (wellId) window.location.href = `/well/${encodeURIComponent(wellId)}`;
      });
    });
}

document.addEventListener("DOMContentLoaded", initMap);

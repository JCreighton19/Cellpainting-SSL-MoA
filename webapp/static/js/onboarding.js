// First-run guided tour for the Cell Painting Embedding Explorer.
//
// Self-contained and additive: it never edits map.js or the DOM it builds,
// only reads from it (Plotly's public `graphDiv.data` convention -- see
// firstNonEmptyTrace below) and calls the handful of functions map.js
// already exposes on `window` for free -- top-level `function` declarations
// in a classic (non-module) <script> are own properties of the global
// object, so `centerOn`, `selectWell`, etc. are already callable here
// without map.js changing anything about how it works.
//
// Scope note: the spec asked for step 2 to spotlight individual dots
// directly. Real per-marker highlighting would mean projecting data
// coordinates to screen pixels for whatever Plotly build is loaded
// (scattergl draws to a WebGL canvas, so there's no per-point DOM element to
// measure) -- doable, but only via the same category of undocumented
// internal API map.js already leans on once, carefully, for the 3D
// highlight dot (see startHighlight3D's comment). Doing that a second time
// purely for a cosmetic tour step felt like unnecessary fragility, so step 2
// instead teaches "near vs. far" with a small static diagram in the tour
// card itself, without touching the map's current view. Steps 3 and 4
// spotlight real DOM elements
// (the legend, the sidebar) directly, and step 4 demonstrates a real
// selection via the existing `selectWell` (not a mock sidebar), so the user
// sees exactly what clicking a point looks like before they try it themselves.

(function () {
  const TOUR_STORAGE_KEY = "cpee_onboarding_completed_v1";

  function hasCompletedTour() {
    try {
      return localStorage.getItem(TOUR_STORAGE_KEY) === "1";
    } catch (e) {
      return false;
    }
  }

  function markTourCompleted() {
    try {
      localStorage.setItem(TOUR_STORAGE_KEY, "1");
    } catch (e) {
      /* localStorage unavailable (private mode, etc.) -- tour just replays next visit */
    }
  }

  // ---- Reading the live map (read-only; see file header) ----

  function getPlotEl() {
    return document.getElementById("umap-plot");
  }

  function plotHasData() {
    const el = getPlotEl();
    return !!(el && Array.isArray(el.data) && el.data.length);
  }

  function waitForPlotData(callback, attemptsLeft) {
    attemptsLeft = attemptsLeft === undefined ? 50 : attemptsLeft; // ~5s at 100ms
    if (plotHasData() || attemptsLeft <= 0) {
      callback();
      return;
    }
    setTimeout(() => waitForPlotData(callback, attemptsLeft - 1), 100);
  }

  // First trace carrying real points (helper traces like the selection ring
  // or neighbor lines are built with empty customdata -- see map.js's
  // buildHighlightTrace/buildNeighborTraces -- so this always finds a trace
  // of actual wells).
  function firstNonEmptyTrace() {
    const el = getPlotEl();
    if (!el || !Array.isArray(el.data)) return null;
    return el.data.find((t) => Array.isArray(t.customdata) && t.customdata.length > 0) || null;
  }

  // The last element rather than index 0: mergeDimBright (map.js) concatenates
  // filtered-out/dim points before matching/bright ones, so picking from the
  // end of a well-populated trace is a safe bet for a real, visible point.
  function pickAnchorPoint() {
    const trace = firstNonEmptyTrace();
    if (!trace) return null;
    const i = trace.customdata.length - 1;
    return { wellId: trace.customdata[i], x: trace.x[i], y: trace.y[i] };
  }

  function simulateSelect() {
    try {
      if (typeof window.selectWell !== "function") return;
      const anchor = pickAnchorPoint();
      if (!anchor) return;
      // { center: false } matches exactly what a real click passes in
      // map.js's plotly_click handler -- the view doesn't jump.
      window.selectWell(anchor.wellId, { center: false });
    } catch (e) {
      /* best-effort -- the slide's text still explains clicking either way */
    }
  }

  // ---- Small inline diagrams (near vs. far) ----
  //
  // Deliberately single-color: this is only about distance, and color is
  // introduced later, so mixing color into this diagram would muddle the
  // two ideas.

  function dot(cx, cy) {
    return (
      `<circle class="tour-dot-glow" cx="${cx}" cy="${cy}" r="13" fill="#0d6efd" opacity="0.2"></circle>` +
      `<circle cx="${cx}" cy="${cy}" r="7" fill="#0d6efd"></circle>`
    );
  }

  function dotDiagram(kind) {
    if (kind === "single") {
      return `<svg class="tour-dot-diagram" viewBox="0 0 200 70">${dot(100, 35)}</svg>`;
    }
    if (kind === "near") {
      return (
        `<svg class="tour-dot-diagram" viewBox="0 0 200 70">` +
        `<line x1="82" y1="35" x2="98" y2="29" stroke="#adb5bd" stroke-width="1.5" stroke-dasharray="3,3"></line>` +
        `${dot(80, 35)}${dot(100, 29)}</svg>`
      );
    }
    return (
      `<svg class="tour-dot-diagram" viewBox="0 0 200 70">` +
      `<line x1="35" y1="55" x2="165" y2="15" stroke="#adb5bd" stroke-width="1.5" stroke-dasharray="3,3"></line>` +
      `${dot(30, 55)}${dot(170, 15)}</svg>`
    );
  }

  // Combined bounding box of the 3 documentation buttons in the left
  // sidebar, used by the final slide to point at all 3 at once without
  // needing 3 separate spotlights. Deliberately excludes the "Restart
  // Tutorial" button just below them, even though it shares the same
  // sidebar section, since it isn't one of the "learn more" pages.
  function helpButtonsRect() {
    const els = ["how-to-read-dialog", "about-project-dialog", "technical-details-dialog"]
      .map((id) => document.querySelector(`[data-dialog-open="${id}"]`))
      .filter(Boolean);
    if (!els.length) return null;
    const rects = els.map((el) => el.getBoundingClientRect());
    const top = Math.min(...rects.map((r) => r.top));
    const left = Math.min(...rects.map((r) => r.left));
    const bottom = Math.max(...rects.map((r) => r.bottom));
    const right = Math.max(...rects.map((r) => r.right));
    return { top, left, bottom, right, width: right - left, height: bottom - top };
  }

  // Normalizes a spotlight/placement target, which may be a real element
  // (most slides) or a plain rect object like helpButtonsRect() above.
  function getRect(target) {
    if (!target) return null;
    return typeof target.getBoundingClientRect === "function" ? target.getBoundingClientRect() : target;
  }

  // ---- Slide content ----
  //
  // Flat list of slides the user clicks through one at a time -- nothing
  // advances on its own. `group` (0-4) maps each slide back to one of the 5
  // required screens for the progress dots: screens 2-4 are worth several
  // slides each (one idea per click), screens 1 and 5 are a single slide.

  const SLIDES = [
    {
      group: 0,
      target: () => null,
      html: () => `
        <h2 class="tour-title" id="tour-title">Welcome to the Cell Painting Embedding Explorer</h2>
        <p class="tour-body">This project tests whether AI can learn which drugs affect cells in similar ways just by looking at microscope images — figuring it out entirely on its own, without ever being told the right answers during training.</p>
        <p class="tour-body mb-0">Don't worry if you don't have a biology or machine learning background—we'll show you how to explore it in under a minute.</p>
      `,
    },
    {
      group: 1,
      target: () => getPlotEl(),
      html: () => `
        <h3 class="tour-title">What's a dot?</h3>
        ${dotDiagram("single")}
        <p class="tour-body mb-0">Every dot represents one experimental well, each of which is a sample containing thousands of cells treated with a specific drug.</p>
      `,
    },
    {
      group: 1,
      target: () => getPlotEl(),
      html: () => `
        <h3 class="tour-title">Nearby dots</h3>
        ${dotDiagram("near")}
        <p class="tour-body mb-0">Dots that sit close together look visually similar.</p>
      `,
    },
    {
      group: 1,
      target: () => getPlotEl(),
      html: () => `
        <h3 class="tour-title">Distant dots</h3>
        ${dotDiagram("far")}
        <p class="tour-body mb-0">Dots that sit far apart look different.</p>
      `,
    },
    {
      group: 2,
      target: () => document.getElementById("map-legend"),
      html: () => `
        <h3 class="tour-title">What do the colors mean?</h3>
        <p class="tour-body mb-0">By default, colors indicate the drug's mechanism of action, or MoA. Points treated with drugs that work in similar ways share the same color.</p>
      `,
    },
    {
      group: 2,
      target: () => document.getElementById("map-legend"),
      html: () => `
        <h3 class="tour-title">Clustering reveals learned biology</h3>
        <p class="tour-body mb-0">When dots of the same color cluster together, it suggests the AI found a meaningful biological pattern.</p>
      `,
    },
    {
      group: 3,
      target: () => getPlotEl(),
      html: () => `
        <h3 class="tour-title">Clicking a dot</h3>
        <p class="tour-body mb-0">To learn more about a point, simply click it to reveal its details.</p>
      `,
    },
    {
      group: 3,
      target: () => document.getElementById("sidebar-right"),
      onEnter: simulateSelect,
      html: () => `
        <h3 class="tour-title">Sample details</h3>
        <p class="tour-body mb-2">Clicking a point shows:</p>
        <ul class="tour-bullets mb-0">
          <li>its microscope image</li>
          <li>the compound tested</li>
          <li>its nearest neighbors</li>
          <li>experimental details</li>
        </ul>
      `,
    },
    {
      group: 4,
      target: () => helpButtonsRect(),
      html: () => `
        <h2 class="tour-title">You're ready to explore!</h2>
        <p class="tour-body mb-0">To learn more, explore the <strong>How to Read This Map</strong>, <strong>About the Project</strong>, and <strong>Technical Details</strong> sections in the left sidebar.</p>
      `,
    },
  ];

  function footerFor(index) {
    const isFirst = index === 0;
    const isLast = index === SLIDES.length - 1;
    return {
      back: !isFirst,
      skip: !isLast,
      nextLabel: isFirst ? "Begin Tour" : isLast ? "Begin Exploring" : "Next",
      primaryFinish: isLast,
    };
  }

  // ---- Tour engine ----

  let tourActive = false;
  let currentIndex = 0;
  let currentTargetFn = null;

  function fadeSwap(el, html, afterSwap) {
    el.style.opacity = "0";
    setTimeout(() => {
      el.innerHTML = html;
      if (afterSwap) afterSwap();
      el.style.opacity = "1";
    }, 150);
  }

  function buildFooterHtml(footer) {
    const backBtn = footer.back
      ? `<button type="button" class="btn btn-sm btn-outline-secondary" data-tour-action="back">Back</button>`
      : "";
    const nextAction = footer.primaryFinish ? "finish" : "next";
    const nextBtn = `<button type="button" class="btn btn-sm btn-primary" data-tour-action="${nextAction}">${footer.nextLabel}</button>`;
    const skipBtn = footer.skip
      ? `<button type="button" class="btn btn-sm btn-link text-muted" data-tour-action="skip">Skip Tour</button>`
      : "<span></span>";
    return `
      <div class="tour-footer">
        <div class="tour-footer-left">${skipBtn}</div>
        <div class="tour-footer-right">${backBtn}${nextBtn}</div>
      </div>
    `;
  }

  function buildDotsHtml(activeGroup) {
    let html = '<div class="tour-dots">';
    for (let g = 0; g < 5; g++) {
      html += `<span class="tour-dot${g === activeGroup ? " tour-dot-active" : ""}"></span>`;
    }
    return html + "</div>";
  }

  function buildLoadingHtml() {
    return `
      <div class="tour-content"><p class="tour-body text-muted mb-0">Loading the map…</p></div>
      <div class="tour-footer">
        <div class="tour-footer-left"><button type="button" class="btn btn-sm btn-link text-muted" data-tour-action="skip">Skip Tour</button></div>
        <div class="tour-footer-right"></div>
      </div>
    `;
  }

  function updateSpotlight(target) {
    const spot = document.getElementById("tour-spotlight");
    const overlay = document.getElementById("tour-overlay");
    const r = getRect(target);
    if (!r) {
      spot.classList.add("d-none");
      overlay.classList.add("tour-overlay-dim");
      return;
    }
    overlay.classList.remove("tour-overlay-dim");
    spot.classList.remove("d-none");
    const pad = 8;
    spot.style.top = `${r.top - pad}px`;
    spot.style.left = `${r.left - pad}px`;
    spot.style.width = `${r.width + pad * 2}px`;
    spot.style.height = `${r.height + pad * 2}px`;
  }

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(v, hi));
  }

  // Places the card on whichever side of the target has room for it
  // (right, left, below, above, in that preference order), so it never sits
  // on top of the very element it's explaining. Falls back to the side with
  // the most available space if the card doesn't fully fit anywhere -- e.g.
  // the plot, which can fill most of the viewport. Measures the card's own
  // current size (set just before this runs, once its real content for this
  // slide is in the DOM) rather than a guessed constant, since slide content
  // varies in height.
  function positionCardNear(target) {
    const card = document.getElementById("tour-card");
    card.style.right = "";
    card.style.bottom = "";

    const r = getRect(target);
    if (!r) {
      card.style.top = "50%";
      card.style.left = "50%";
      card.style.transform = "translate(-50%, -50%)";
      return;
    }

    card.style.transform = "none";
    const margin = 16;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const cw = card.offsetWidth || 380;
    const ch = card.offsetHeight || 220;

    const candidates = [
      { side: "right", space: vw - r.right, need: cw },
      { side: "left", space: r.left, need: cw },
      { side: "below", space: vh - r.bottom, need: ch },
      { side: "above", space: r.top, need: ch },
    ];
    let chosen = candidates.find((c) => c.space >= c.need + margin);
    if (!chosen) chosen = candidates.slice().sort((a, b) => b.space - a.space)[0];

    let top, left;
    if (chosen.side === "right") {
      top = clamp(r.top, margin, vh - ch - margin);
      left = clamp(r.right + margin, margin, vw - cw - margin);
    } else if (chosen.side === "left") {
      top = clamp(r.top, margin, vh - ch - margin);
      left = clamp(r.left - margin - cw, margin, vw - cw - margin);
    } else if (chosen.side === "below") {
      top = clamp(r.bottom + margin, margin, vh - ch - margin);
      left = clamp(r.left + r.width / 2 - cw / 2, margin, vw - cw - margin);
    } else {
      top = clamp(r.top - margin - ch, margin, vh - ch - margin);
      left = clamp(r.left + r.width / 2 - cw / 2, margin, vw - cw - margin);
    }
    card.style.top = `${top}px`;
    card.style.left = `${left}px`;
  }

  function renderSlide(index) {
    currentIndex = index;
    const slide = SLIDES[index];
    const targetEl = slide.target ? slide.target() : null;
    currentTargetFn = slide.target || null;

    const inner = document.getElementById("tour-card-inner");
    const html =
      `<div class="tour-content">${slide.html()}</div>` +
      buildDotsHtml(slide.group) +
      buildFooterHtml(footerFor(index));
    // Spotlight/card position are computed after the swap, inside the same
    // timeout, so they measure this slide's actual rendered size rather than
    // the outgoing slide's.
    fadeSwap(inner, html, () => {
      updateSpotlight(targetEl);
      positionCardNear(targetEl);
    });

    if (slide.onEnter) {
      try {
        slide.onEnter();
      } catch (e) {
        /* best-effort */
      }
    }
  }

  // Slides in groups 1-3 read real map state, so entering any of them waits
  // for the first render to finish rather than spotlighting an empty
  // legend/sidebar on a slow connection.
  function goSlide(index) {
    index = Math.max(0, Math.min(index, SLIDES.length - 1));
    const needsData = SLIDES[index].group >= 1 && SLIDES[index].group <= 3;
    if (needsData && !plotHasData()) {
      currentIndex = index;
      currentTargetFn = null;
      document.getElementById("tour-card-inner").innerHTML = buildLoadingHtml();
      updateSpotlight(null);
      positionCardNear(null);
      waitForPlotData(() => {
        if (currentIndex === index) renderSlide(index);
      });
      return;
    }
    renderSlide(index);
  }

  function handleNext() {
    goSlide(currentIndex + 1);
  }

  function handleBack() {
    goSlide(currentIndex - 1);
  }

  function onResize() {
    if (!tourActive) return;
    const targetEl = currentTargetFn ? currentTargetFn() : null;
    updateSpotlight(targetEl);
    positionCardNear(targetEl);
  }

  function onKeydown(e) {
    if (!tourActive) return;
    if (e.key === "Escape") endTour(true);
    else if (e.key === "Enter") handleNext();
  }

  function endTour(markComplete) {
    tourActive = false;
    document.getElementById("tour-root").classList.add("d-none");
    window.removeEventListener("resize", onResize);
    document.removeEventListener("keydown", onKeydown);
    if (markComplete) markTourCompleted();
  }

  function startTour(force) {
    if (tourActive) return;
    if (!force && hasCompletedTour()) return;
    tourActive = true;
    document.getElementById("tour-root").classList.remove("d-none");
    window.addEventListener("resize", onResize);
    document.addEventListener("keydown", onKeydown);
    goSlide(0);
    const card = document.getElementById("tour-card");
    if (card) card.focus();
  }

  document.getElementById("tour-root").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-tour-action]");
    if (!btn) return;
    const action = btn.dataset.tourAction;
    if (action === "next") handleNext();
    else if (action === "back") handleBack();
    else if (action === "skip") endTour(true);
    else if (action === "finish") endTour(true);
  });

  document.addEventListener("DOMContentLoaded", () => {
    const restartBtn = document.getElementById("restart-tutorial-btn");
    if (restartBtn) restartBtn.addEventListener("click", () => startTour(true));

    if (!hasCompletedTour()) {
      // Small delay so the tour's dim overlay doesn't flash in before the
      // rest of the page has painted.
      setTimeout(() => startTour(false), 500);
    }
  });

  window.CPEETour = { start: () => startTour(true) };
})();

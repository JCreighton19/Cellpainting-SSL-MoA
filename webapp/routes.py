import io
import json
from pathlib import Path
import matplotlib.cm as cm

import numpy as np
from flask import abort, jsonify, render_template, request, send_file
from PIL import Image

from similarity import compute_neighborhood_stats, generate_interpretation, title_case

ATTENTION_DIR = Path(__file__).resolve().parent / "static" / "attention"
ATTENTION_IMG_SIZE = 256

# generate_web_attention_maps.py now tiles each 1080x1080 site with evenly-
# spaced, slightly-overlapping crops covering the FULL image (previously
# floor(1080/224)=4 non-overlapping crops left the last ~17% of each
# dimension untiled). Coverage is complete, so the render size is just the
# full canvas -- kept as a named constant (not a bare ATTENTION_IMG_SIZE
# reference below) so a future coverage change only needs one number here.
_ATTENTION_SITE_SIZE = 1080
_ATTENTION_COVERED_PX = _ATTENTION_SITE_SIZE
ATTENTION_RENDER_SIZE = round(ATTENTION_IMG_SIZE * _ATTENTION_COVERED_PX / _ATTENTION_SITE_SIZE)


def _hot_colormap(norm):
    """norm: (H, W) float32 in [0,1] -> (H, W, 3) uint8.

    Hand-rolled black -> red -> yellow -> white ramp (matplotlib's "hot"
    colormap formula) so the app doesn't need a matplotlib dependency just
    for one small heatmap.
    """
    r = np.clip(norm * 3.0, 0, 1)
    g = np.clip(norm * 3.0 - 1.0, 0, 1)
    b = np.clip(norm * 3.0 - 2.0, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

# A few curated searches shown in the left sidebar. Picked from the moa/broad_sample
# columns actually present in wells.parquet.
EXAMPLE_QUERIES = [
    "HDAC inhibitor",
    "adrenergic receptor agonist",
    "dexamethasone",
    "amlodipine",
]

MAX_DISAMBIGUATION_WELLS = 30
DEFAULT_K = 5  # top-5 neighbors, shown in the right sidebar

MOA_DESCRIPTIONS_PATH = Path(__file__).resolve().parent / "moa_descriptions.json"
with open(MOA_DESCRIPTIONS_PATH) as _f:
    _MOA_DESCRIPTIONS = {k: v for k, v in json.load(_f).items() if not k.startswith("_")}

# Written by scripts/generate_web_channel_importance.py -- one combined JSON
# ({well_id: {channel_name: {"importance", "cosine_similarity"}}}), not one
# file per well (see that script's docstring). Optional: guarded rather than
# a bare open() since, unlike moa_descriptions.json, this artifact may not
# have been generated yet for a given model checkpoint.
CHANNEL_IMPORTANCE_PATH = Path(__file__).resolve().parent / "static" / "channel_importance.json"
if CHANNEL_IMPORTANCE_PATH.exists():
    with open(CHANNEL_IMPORTANCE_PATH) as _f:
        _CHANNEL_IMPORTANCE = json.load(_f)
else:
    _CHANNEL_IMPORTANCE = {}

# Order and (muted, readable-on-white) colors intentionally echo the
# thumbnail composite's channel coloring (scripts/generate_web_thumbnails.py's
# CHANNEL_COLORS), so the sidebar's channel-importance bars visually tie back
# to the thumbnail/attention image directly above them.
CHANNEL_DISPLAY = [
    ("Mito", "#dc3545"),
    ("AGP", "#d4a017"),
    ("RNA", "#e83e8c"),
    ("ER", "#28a745"),
    ("DNA", "#0d6efd"),
]


def build_channel_importance_bars(well_id):
    """Returns [{name, color, importance, pct}, ...] for the sidebar's
    channel-importance bar chart, with pct scaled relative to this well's
    most important channel -- or None if this well has no precomputed
    channel-ablation scores yet."""
    scores = _CHANNEL_IMPORTANCE.get(well_id)
    if not scores:
        return None
    max_importance = max(s["importance"] for s in scores.values()) or 1.0
    return [
        {
            "name": name,
            "color": color,
            "importance": scores[name]["importance"],
            "pct": 100.0 * scores[name]["importance"] / max_importance,
        }
        for name, color in CHANNEL_DISPLAY
        if name in scores
    ]


def describe_moa(moa_name: str):
    """Looks up curated educational text for a MoA label.

    Returns a single dict (not a list) — the sidebar shows exactly one concise
    description per point. Combined labels in the data ("A | B") are resolved
    by trying each atomic part in order and using the first one with a
    curated entry, since the dictionary only stores atomic MoA entries.
    Falls back to a None-filled placeholder (keyed on the first part) if no
    part is curated yet.
    """
    if not moa_name:
        return None
    parts = [p.strip() for p in moa_name.split("|")]
    for part in parts:
        entry = _MOA_DESCRIPTIONS.get(part)
        if entry:
            return {"name": part, **entry}
    return {"name": parts[0], "description": None, "pathway": None, "typically_affects": None, "sources": []}


def _compound_matches(df):
    return [
        {
            "compound_id": row["compound_id"],
            "compound_name": row["compound_name"],
            "dominant_moa": row["dominant_moa"],
            "n_wells": int(row["n_wells"]),
            "representative_well_id": row["well_ids"][0],
        }
        for _, row in df.iterrows()
    ]


def _well_matches(df):
    return [
        {
            "well_id": row["well_id"],
            "broad_sample": row["broad_sample"],
            "pert_iname": row["pert_iname"],
            "moa": row["moa"],
        }
        for _, row in df.iterrows()
    ]


def resolve_query(store, query: str):
    """Resolve a free-text query against wells + compounds.

    Returns one of:
      {"kind": "well", "well_id": ...}
      {"kind": "compound", "compound_id": ...}
      {"kind": "disambiguate", "compounds": [...], "wells": [...], "wells_truncated": bool}
      {"kind": "none"}

    Priority: exact well_id -> exact compound_id/name -> exact MoA (compound-level,
    resolves straight through if unambiguous, else disambiguates) -> exact
    plate -> substring match across compounds and wells, grouped.
    Plain substring matching throughout — no ranking/scoring.
    """
    q = query.strip()
    if not q:
        return {"kind": "none"}
    q_lower = q.lower()

    wells_df = store.wells
    compounds_df = store.compounds

    exact_well = wells_df[wells_df["well_id"].str.lower() == q_lower]
    if len(exact_well):
        return {"kind": "well", "well_id": exact_well.iloc[0]["well_id"]}

    exact_compound = compounds_df[
        (compounds_df["compound_id"].str.lower() == q_lower)
        | (compounds_df["compound_name"].fillna("").str.lower() == q_lower)
    ]
    if len(exact_compound):
        return {"kind": "compound", "compound_id": exact_compound.iloc[0]["compound_id"]}

    exact_moa = compounds_df[compounds_df["dominant_moa"].fillna("").str.lower() == q_lower]
    if len(exact_moa) == 1:
        return {"kind": "compound", "compound_id": exact_moa.iloc[0]["compound_id"]}
    if len(exact_moa) > 1:
        return {"kind": "disambiguate", "compounds": _compound_matches(exact_moa), "wells": [], "wells_truncated": False}

    exact_plate = wells_df[wells_df["plate"].str.lower() == q_lower]
    if len(exact_plate):
        return {"kind": "well", "well_id": exact_plate.iloc[0]["well_id"]}

    compound_mask = (
        compounds_df["compound_id"].fillna("").str.lower().str.contains(q_lower, regex=False)
        | compounds_df["compound_name"].fillna("").str.lower().str.contains(q_lower, regex=False)
        | compounds_df["dominant_moa"].fillna("").str.lower().str.contains(q_lower, regex=False)
    )
    matched_compounds = compounds_df[compound_mask]

    well_mask = (
        wells_df["well_id"].fillna("").str.lower().str.contains(q_lower, regex=False)
        | wells_df["broad_sample"].fillna("").str.lower().str.contains(q_lower, regex=False)
        | wells_df["pert_iname"].fillna("").str.lower().str.contains(q_lower, regex=False)
        | wells_df["moa"].fillna("").str.lower().str.contains(q_lower, regex=False)
        | wells_df["plate"].fillna("").str.lower().str.contains(q_lower, regex=False)
        | wells_df["well"].fillna("").str.lower().str.contains(q_lower, regex=False)
    )
    matched_wells = wells_df[well_mask]

    total = len(matched_compounds) + len(matched_wells)
    if total == 0:
        return {"kind": "none"}
    if total == 1:
        if len(matched_compounds):
            return {"kind": "compound", "compound_id": matched_compounds.iloc[0]["compound_id"]}
        return {"kind": "well", "well_id": matched_wells.iloc[0]["well_id"]}

    return {
        "kind": "disambiguate",
        "compounds": _compound_matches(matched_compounds),
        "wells": _well_matches(matched_wells.head(MAX_DISAMBIGUATION_WELLS)),
        "wells_truncated": len(matched_wells) > MAX_DISAMBIGUATION_WELLS,
    }


def register_routes(app, store, sim_index):
    def _search_context():
        # Powers the left sidebar's <datalist> search suggestions.
        return {
            "plates": sorted(store.wells["plate"].dropna().unique().tolist()),
            "moas": sorted(store.compounds["dominant_moa"].dropna().unique().tolist()),
            "compounds": sorted(store.compounds["compound_name"].dropna().unique().tolist()),
        }

    def _dataset_stats():
        # Powers the left sidebar's Dataset Overview section and the map's caption.
        wells = store.wells
        n_wells = len(wells)
        unannotated = int(wells["moa"].isna().sum())
        top10_moas = wells["moa"].value_counts().head(10)
        pct_unannotated = unannotated / n_wells
        pct_top10_coverage = top10_moas.sum() / n_wells
        return {
            "n_wells": n_wells,
            "n_compounds": len(store.compounds),
            "n_moas": int(wells["moa"].nunique()),
            "n_plates": int(wells["plate"].nunique()),
            "pct_unannotated": pct_unannotated,
            "pct_top10_coverage": pct_top10_coverage,
            "pct_other_annotated": 1 - pct_unannotated - pct_top10_coverage,
            "n_moas_in_legend": len(top10_moas),
        }

    @app.route("/")
    def home():
        # The entire application lives on this one page — search, map, and
        # detail panel all update in place via the JSON/partial APIs below.
        return render_template(
            "index.html", examples=EXAMPLE_QUERIES, stats=_dataset_stats(), **_search_context()
        )

    @app.route("/api/search")
    def api_search():
        q = request.args.get("q", "")
        result = resolve_query(store, q)

        if result["kind"] == "well":
            return jsonify({"kind": "well", "well_id": result["well_id"]})

        if result["kind"] == "compound":
            compound, _ = store.get_compound(result["compound_id"])
            return jsonify({"kind": "well", "well_id": compound["well_ids"][0]})

        if result["kind"] == "disambiguate":
            matches = [
                {
                    "label": title_case(c["compound_name"]) or c["compound_id"],
                    "moa": title_case(c["dominant_moa"]),
                    "well_id": c["representative_well_id"],
                }
                for c in result["compounds"]
            ] + [
                {
                    "label": title_case(w["pert_iname"]) or "Unannotated compound",
                    "moa": title_case(w["moa"]),
                    "well_id": w["well_id"],
                }
                for w in result["wells"]
            ]
            return jsonify({"kind": "disambiguate", "query": q, "matches": matches, "truncated": result["wells_truncated"]})

        return jsonify({"kind": "none", "query": q})

    @app.route("/api/well/<well_id>")
    def api_well(well_id):
        well, row_idx = store.get_well(well_id)
        if well is None:
            abort(404)

        k = request.args.get("k", default=DEFAULT_K, type=int)
        neighbor_idxs, scores = sim_index.search(row_idx, k=k)
        neighbors = []
        for i, score in zip(neighbor_idxs, scores):
            n = store.wells.iloc[int(i)]
            same_moa = bool(n["moa"] == well["moa"]) if well["moa"] else False
            neighbors.append({
                "well_id": n["well_id"],
                "broad_sample": n["broad_sample"],
                "pert_iname": n["pert_iname"],
                "moa": n["moa"],
                "similarity": float(score),
                "same_moa": same_moa,
            })

        stats = compute_neighborhood_stats(
            query_moa=well["moa"],
            neighbor_moas=[n["moa"] for n in neighbors],
            similarities=[n["similarity"] for n in neighbors],
        )
        interpretation = generate_interpretation("well", well["moa"], stats)
        moa_info = describe_moa(well["moa"])
        channel_importance_bars = build_channel_importance_bars(well_id)

        return render_template(
            "partials/_right_sidebar.html",
            well=well,
            neighbors=neighbors,
            stats=stats,
            interpretation=interpretation,
            moa_info=moa_info,
            channel_importance_bars=channel_importance_bars,
        )

    @app.route("/api/attention/<well_id>.png")
    def api_attention_png(well_id):
        # Validate well_id against the real dataset before touching the
        # filesystem (get_well does a dict lookup, not a filesystem read) --
        # avoids ever constructing a path from unvalidated user input.
        well, _ = store.get_well(well_id)
        if well is None or not well["attention_path"]:
            abort(404)

        npy_path = ATTENTION_DIR / f"{well_id}.npy"
        if not npy_path.exists():
            abort(404)

        arr = np.load(npy_path).astype(np.float32)
        # Percentile (not min/max) normalization: a few extreme attention
        # values would otherwise stretch the range so far that the rest of
        # the map reads as solid black.
        lo, hi = np.percentile(arr, [1, 99])
        norm = np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)
        rgba = cm.inferno(norm)
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        heat = Image.fromarray(rgb, mode="RGB").resize(
            (ATTENTION_RENDER_SIZE, ATTENTION_RENDER_SIZE), Image.BILINEAR
        ).convert("RGBA")

        # Composite at the same (0,0)-anchored scale as the thumbnail (see
        # ATTENTION_RENDER_SIZE above) instead of stretching over the full
        # canvas; the uncovered edge strip stays transparent so it doesn't
        # paint over thumbnail regions with no attention data.
        canvas = Image.new("RGBA", (ATTENTION_IMG_SIZE, ATTENTION_IMG_SIZE), (0, 0, 0, 0))
        canvas.paste(heat, (0, 0))

        buf = io.BytesIO()
        canvas.save(buf, "PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png", max_age=86400)

    @app.route("/api/umap")
    def api_umap():
        df = store.wells
        points = {
            "well_id": df["well_id"].tolist(),
            "x": df["umap_x"].tolist(),
            "y": df["umap_y"].tolist(),
            # Separate 3D UMAP fit (see scripts/prepare_phase1_data.py), additive
            # alongside x/y above -- powers the webapp's 2D/3D view toggle.
            "x3d": df["umap_x_3d"].tolist(),
            "y3d": df["umap_y_3d"].tolist(),
            "z3d": df["umap_z_3d"].tolist(),
            "moa": df["moa"].fillna("unannotated").apply(title_case).tolist(),
            "broad_sample": df["broad_sample"].fillna("unknown").tolist(),
            "plate": df["plate"].tolist(),
            # Compound name, so client-side search filtering can match e.g. "amlodipine".
            "pert_iname": df["pert_iname"].fillna("").tolist(),
        }
        return jsonify(points)

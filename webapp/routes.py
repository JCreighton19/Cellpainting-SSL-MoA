import json
from pathlib import Path

from flask import abort, jsonify, redirect, render_template, request, url_for

from similarity import generate_insight

# A few curated searches shown on the home page. Picked from the moa/broad_sample
# columns actually present in wells.parquet.
EXAMPLE_QUERIES = [
    "HDAC inhibitor",
    "adrenergic receptor agonist",
    "dexamethasone",
    "amlodipine",
]

MAX_DISAMBIGUATION_WELLS = 30

MOA_DESCRIPTIONS_PATH = Path(__file__).resolve().parent / "moa_descriptions.json"
with open(MOA_DESCRIPTIONS_PATH) as _f:
    _MOA_DESCRIPTIONS = {k: v for k, v in json.load(_f).items() if not k.startswith("_")}


def describe_moa(moa_name: str):
    """Looks up curated educational text for a MoA label.

    Combined labels in the data ("A | B") are split on " | " and each part is
    resolved independently, so the dictionary only needs atomic MoA entries.
    Returns a list of dicts (one per atomic part), each with description/
    pathway/typically_affects/sources, or None-filled fields as a fallback
    when no curated entry exists for that part yet.
    """
    if not moa_name:
        return []
    parts = [p.strip() for p in moa_name.split("|")]
    result = []
    for part in parts:
        entry = _MOA_DESCRIPTIONS.get(part)
        if entry:
            result.append({"name": part, **entry})
        else:
            result.append({
                "name": part,
                "description": None,
                "pathway": None,
                "typically_affects": None,
                "sources": [],
            })
    return result


def _neighbor_reason(query_moa, neighbor_moa, same_moa):
    """One-line, cautiously-worded explanation of why a neighbor showed up."""
    if same_moa:
        return "Shares the same annotated mechanism of action."
    if query_moa and neighbor_moa:
        return "Similar phenotype despite a different annotated mechanism of action."
    if query_moa and not neighbor_moa:
        return "Similar phenotype; this neighbor has no annotated mechanism of action."
    if neighbor_moa and not query_moa:
        return "Similar phenotype; this one has no annotated mechanism to compare against."
    return "Similar phenotype; neither has an annotated mechanism of action."


def _compound_matches(df):
    return [
        {
            "compound_id": row["compound_id"],
            "compound_name": row["compound_name"],
            "dominant_moa": row["dominant_moa"],
            "n_wells": int(row["n_wells"]),
        }
        for _, row in df.iterrows()
    ]


def _well_matches(df):
    return [
        {
            "well_id": row["well_id"],
            "plate": row["plate"],
            "well": row["well"],
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
    redirects straight through if unambiguous, else disambiguates) -> exact
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


def register_routes(app, store, sim_index, compound_sim_index):
    def _search_context():
        # Powers the home page's <datalist> search suggestions.
        return {
            "plates": sorted(store.wells["plate"].dropna().unique().tolist()),
            "moas": sorted(store.compounds["dominant_moa"].dropna().unique().tolist()),
            "compounds": sorted(store.compounds["compound_name"].dropna().unique().tolist()),
        }

    def _dataset_stats():
        # Powers the home page's Dataset Overview section and the map's explanatory text.
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
        return render_template(
            "index.html", examples=EXAMPLE_QUERIES, error=None, stats=_dataset_stats(), **_search_context()
        )

    @app.route("/search")
    def search():
        q = request.args.get("q", "")
        result = resolve_query(store, q)

        if result["kind"] == "well":
            return redirect(url_for("well_detail", well_id=result["well_id"]))
        if result["kind"] == "compound":
            return redirect(url_for("compound_detail", compound_id=result["compound_id"]))
        if result["kind"] == "disambiguate":
            return render_template(
                "search_results.html",
                query=q,
                compounds=result["compounds"],
                wells=result["wells"],
                wells_truncated=result["wells_truncated"],
            )
        return render_template(
            "index.html",
            examples=EXAMPLE_QUERIES,
            error=f"No match found for “{q}”." if q.strip() else "Enter a search term.",
            stats=_dataset_stats(),
            **_search_context(),
        )

    @app.route("/moa")
    def moa_detail():
        name = request.args.get("name", "").strip()
        if not name:
            abort(404)
        entries = describe_moa(name)
        compounds = store.compounds[store.compounds["dominant_moa"] == name]
        compound_list = _compound_matches(compounds)
        return render_template("moa_detail.html", moa_name=name, entries=entries, compounds=compound_list)

    @app.route("/well/<well_id>")
    def well_detail(well_id):
        well, row_idx = store.get_well(well_id)
        if well is None:
            abort(404)

        k = request.args.get("k", default=10, type=int)
        neighbor_idxs, scores = sim_index.search(row_idx, k=k)
        neighbors = []
        for i, score in zip(neighbor_idxs, scores):
            n = store.wells.iloc[int(i)]
            same_moa = bool(n["moa"] == well["moa"]) if well["moa"] else False
            neighbors.append({
                "well_id": n["well_id"],
                "plate": n["plate"],
                "well": n["well"],
                "broad_sample": n["broad_sample"],
                "pert_iname": n["pert_iname"],
                "moa": n["moa"],
                "thumbnail_path": n["thumbnail_path"],
                "similarity": float(score),
                "same_moa": same_moa,
                "reason": _neighbor_reason(well["moa"], n["moa"], same_moa),
            })

        insight = generate_insight(
            entity_label="well",
            query_moa=well["moa"],
            neighbor_moas=[n["moa"] for n in neighbors],
            similarities=[n["similarity"] for n in neighbors],
        )

        return render_template("well_detail.html", well=well, neighbors=neighbors, k=k, insight=insight)

    @app.route("/compound/<compound_id>")
    def compound_detail(compound_id):
        compound, row_idx = store.get_compound(compound_id)
        if compound is None:
            abort(404)

        k = request.args.get("k", default=10, type=int)
        neighbor_idxs, scores = compound_sim_index.search(row_idx, k=k)
        neighbors = []
        for i, score in zip(neighbor_idxs, scores):
            n = store.compounds.iloc[int(i)]
            same_moa = bool(n["dominant_moa"] == compound["dominant_moa"]) if compound["dominant_moa"] else False
            neighbors.append({
                "compound_id": n["compound_id"],
                "compound_name": n["compound_name"],
                "dominant_moa": n["dominant_moa"],
                "n_wells": int(n["n_wells"]),
                "thumbnail_path": n["thumbnail_path"],
                "similarity": float(score),
                "same_moa": same_moa,
                "reason": _neighbor_reason(compound["dominant_moa"], n["dominant_moa"], same_moa),
            })

        insight = generate_insight(
            entity_label="compound",
            query_moa=compound["dominant_moa"],
            neighbor_moas=[n["dominant_moa"] for n in neighbors],
            similarities=[n["similarity"] for n in neighbors],
        )

        return render_template("compound_detail.html", compound=compound, neighbors=neighbors, k=k, insight=insight)

    @app.route("/api/umap")
    def api_umap():
        df = store.wells
        points = {
            "well_id": df["well_id"].tolist(),
            "x": df["umap_x"].tolist(),
            "y": df["umap_y"].tolist(),
            "moa": df["moa"].fillna("unannotated").tolist(),
            "broad_sample": df["broad_sample"].fillna("unknown").tolist(),
            "plate": df["plate"].tolist(),
        }
        return jsonify(points)

"""One-time, startup-computed evaluation metrics for the left sidebar and the
Technical Details "Evaluation" section.

Deliberately computed live from whatever embeddings/wells are actually
loaded (store.wells / store.embeddings / sim_index) rather than hardcoded --
a hardcoded number would silently drift out of sync the next time the
webapp's data is regenerated for a different checkpoint (see
scripts/prepare_phase1_data.py). Mirrors the diagnostics in
analysis/replicate_correlation.py and notebooks/representation_evaluation.ipynb,
simplified to display-friendly numbers rather than a full statistical report.

Called once in app.py at startup (not per-request) since it does a
similarity search per well plus one SVD over the full embedding matrix --
cheap at this dataset's size (a few thousand wells, 384 dims) but not free.
"""
import numpy as np
from collections import Counter

MIN_WELLS_PER_MOA = 5
TOP_N_MOAS = 5
NEIGHBOR_K = 5
MIN_REPLICATE_PLATES = 2
N_RANDOM_PAIRS = 5000
RANDOM_SEED = 42


def compute_model_metrics(store, sim_index):
    """Returns a dict of display-ready evaluation numbers -- see the inline
    comments below for what each one means and how it's computed."""
    wells = store.wells
    embeddings = store.embeddings
    moa_values = wells["moa"].values
    plate_values = wells["plate"].values
    n = len(wells)

    # ---- Random-chance baselines: what a model with no learned signal at
    # all would already produce, given just how common each MoA/plate
    # already is in this dataset (sum of squared class frequencies --
    # equivalent to the probability two independent random wells share a
    # class). Every "vs. chance" comparison below uses one of these two. ----
    moa_counts = wells.loc[wells["moa"].notna(), "moa"].value_counts()
    chance_consistency = float((moa_counts / moa_counts.sum()).pow(2).sum())
    plate_counts = wells["plate"].value_counts()
    plate_chance_agreement = float((plate_counts / plate_counts.sum()).pow(2).sum())
    moa_majority_baseline = float(moa_counts.iloc[0] / moa_counts.sum()) if len(moa_counts) else 0.0

    # ---- Single pass over every well's true nearest neighbors (full
    # embedding space), computing 3 different things from the same search
    # so this doesn't pay for it 3 times:
    #   1. neighbor_consistency: mean fraction of a well's k neighbors that
    #      share its annotated MoA (soft, "how much overlap on average").
    #   2. moa_topk_accuracy: stricter classification-style accuracy --
    #      does the SINGLE MOST COMMON MoA among those k neighbors (majority
    #      vote) exactly match the well's own true MoA? Compared against
    #      moa_majority_baseline (always guessing the single most common
    #      MoA overall) rather than chance_consistency, since that's the
    #      correct baseline for a majority-vote classifier.
    #   3. plate_neighbor_agreement: mean fraction of a well's k neighbors
    #      that come from the SAME PLATE -- a direct, previously-unquantified
    #      measurement of the plate/batch effect described in Limitations.
    # ----
    fracs = []
    moa_fracs = {}
    moa_hits = 0
    moa_total = 0
    plate_fracs = []
    for idx in wells.index[wells["moa"].notna()]:
        moa = moa_values[idx]
        neighbor_idxs, _ = sim_index.search(idx, k=NEIGHBOR_K)
        if len(neighbor_idxs) == 0:
            continue
        neighbor_moas = moa_values[neighbor_idxs]
        frac = float(np.count_nonzero(neighbor_moas == moa)) / len(neighbor_idxs)
        fracs.append(frac)
        moa_fracs.setdefault(moa, []).append(frac)

        majority_moa = Counter(neighbor_moas).most_common(1)[0][0]
        moa_hits += int(majority_moa == moa)
        moa_total += 1

        plate_fracs.append(
            float(np.count_nonzero(plate_values[neighbor_idxs] == plate_values[idx])) / len(neighbor_idxs)
        )

    overall_consistency = float(np.mean(fracs)) if fracs else 0.0
    moa_topk_accuracy = (moa_hits / moa_total) if moa_total else 0.0
    plate_neighbor_agreement = float(np.mean(plate_fracs)) if plate_fracs else 0.0

    top_moas = sorted(
        (
            {"moa": moa, "consistency": float(np.mean(v)), "n_wells": len(v)}
            for moa, v in moa_fracs.items()
            if len(v) >= MIN_WELLS_PER_MOA
        ),
        key=lambda r: -r["consistency"],
    )[:TOP_N_MOAS]

    # ---- Cross-plate replicate enrichment: are same-compound wells imaged
    # on different plates more similar to each other than random pairs?
    # random_pair_mean_similarity is also a quick embedding-collapse check --
    # near 0 means unrelated wells don't look artificially alike; a value
    # close to 1 would mean every embedding looks nearly the same. ----
    treated = wells[wells["broad_sample"].notna() & ~wells["is_control"]]
    plate_counts_per_compound = treated.groupby("broad_sample")["plate"].nunique()
    rep_compounds = plate_counts_per_compound[plate_counts_per_compound >= MIN_REPLICATE_PLATES].index

    rep_sims = []
    for compound in rep_compounds:
        idxs = treated.index[treated["broad_sample"] == compound].to_numpy()
        plates = treated.loc[idxs, "plate"].to_numpy()
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                if plates[i] != plates[j]:
                    rep_sims.append(float(embeddings[idxs[i]] @ embeddings[idxs[j]]))

    rng = np.random.default_rng(RANDOM_SEED)
    all_compounds = wells["broad_sample"].values
    rand_sims = []
    attempts = 0
    while len(rand_sims) < N_RANDOM_PAIRS and attempts < N_RANDOM_PAIRS * 20:
        i, j = rng.integers(0, n, size=2)
        if all_compounds[i] != all_compounds[j]:
            rand_sims.append(float(embeddings[i] @ embeddings[j]))
        attempts += 1

    rand_mean = float(np.mean(rand_sims)) if rand_sims else 0.0
    rep_mean = float(np.mean(rep_sims)) if rep_sims else None
    enrichment = (rep_mean / rand_mean) if (rep_mean is not None and abs(rand_mean) > 1e-8) else None

    # ---- Embedding-space health: is the model actually using its full
    # 384-dimensional output, or did training collapse onto a handful of
    # directions? Entropy effective rank (Roy & Vetterli) rather than a raw
    # component count, since it degrades smoothly with concentration
    # instead of depending on an arbitrary variance-ratio cutoff. ----
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    variance = singular_values ** 2
    variance_ratio = variance / variance.sum()
    nonzero = variance_ratio[variance_ratio > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)))
    effective_rank = float(np.exp(entropy))

    return {
        "neighbor_consistency": overall_consistency,
        "chance_consistency": chance_consistency,
        "neighbor_enrichment": (
            overall_consistency / chance_consistency if chance_consistency > 0 else None
        ),
        "top_moas": top_moas,
        "replicate_enrichment": enrichment,
        "random_pair_mean_similarity": rand_mean,
        "moa_topk_accuracy": moa_topk_accuracy,
        "moa_majority_baseline": moa_majority_baseline,
        "plate_neighbor_agreement": plate_neighbor_agreement,
        "plate_chance_agreement": plate_chance_agreement,
        "plate_enrichment": (
            plate_neighbor_agreement / plate_chance_agreement if plate_chance_agreement > 0 else None
        ),
        "effective_rank": effective_rank,
        "embedding_dims": embeddings.shape[1],
    }

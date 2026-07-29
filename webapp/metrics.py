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


def compute_model_metrics(store):
    """Returns a dict of display-ready evaluation numbers -- see the inline
    comments below for what each one means and how it's computed."""
    wells = store.wells
    embeddings = store.embeddings
    # .to_numpy() (not .values) -- wells["moa"]/["plate"] can be Arrow-backed
    # pandas extension arrays depending on the pyarrow/pandas version, which
    # only support 1D fancy indexing; the batched neighbor lookup below
    # indexes with a 2D (batch, k) array, which needs a plain numpy array.
    moa_values = wells["moa"].to_numpy()
    plate_values = wells["plate"].to_numpy()
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
    #
    # Batched/vectorized rather than one sim_index.search() call per well --
    # at this dataset's size (thousands of wells), an O(n) dot-product +
    # argsort per well in a Python loop was the dominant cost of this whole
    # function (and, at startup, of the app becoming ready to serve traffic
    # at all). Each batch computes a (batch x n) similarity matrix in one
    # matmul and uses argpartition for an *unordered* top-k -- sufficient
    # since none of these 3 metrics depend on rank order within the k
    # neighbors, only on which k they are. Batched (not one n x n matmul)
    # so peak memory stays small instead of materializing the full
    # similarity matrix at once. self-similarity is excluded by forcing the
    # diagonal to -inf before the top-k selection, rather than requesting
    # k+1 and filtering afterward.
    # ----
    query_idxs = wells.index[wells["moa"].notna()].to_numpy()
    n_queries = len(query_idxs)

    fracs = np.empty(n_queries, dtype=np.float64)
    plate_fracs = np.empty(n_queries, dtype=np.float64)
    moa_fracs = {}
    moa_hits = 0

    BATCH_SIZE = 1024
    for start in range(0, n_queries, BATCH_SIZE):
        batch = query_idxs[start:start + BATCH_SIZE]
        sims = embeddings[batch] @ embeddings.T  # (batch, n)
        sims[np.arange(len(batch)), batch] = -np.inf  # exclude self
        neighbor_idxs = np.argpartition(-sims, NEIGHBOR_K - 1, axis=1)[:, :NEIGHBOR_K]
        # argpartition gives the correct top-k *set* but arbitrary order within
        # it; the majority-vote below breaks ties by first-seen order, so
        # restore descending-similarity order among just these k columns to
        # match the original sim_index.search() (argsort-based) ordering --
        # cheap, since it's only sorting k=NEIGHBOR_K elements per row.
        neighbor_sims = np.take_along_axis(sims, neighbor_idxs, axis=1)
        order = np.argsort(-neighbor_sims, axis=1)
        neighbor_idxs = np.take_along_axis(neighbor_idxs, order, axis=1)

        batch_moas = moa_values[batch]
        neighbor_moas = moa_values[neighbor_idxs]  # (batch, k)
        batch_fracs = (neighbor_moas == batch_moas[:, None]).mean(axis=1)
        fracs[start:start + len(batch)] = batch_fracs

        neighbor_plates = plate_values[neighbor_idxs]
        plate_fracs[start:start + len(batch)] = (
            neighbor_plates == plate_values[batch][:, None]
        ).mean(axis=1)

        for i, moa in enumerate(batch_moas):
            moa_fracs.setdefault(moa, []).append(batch_fracs[i])
            majority_moa = Counter(neighbor_moas[i]).most_common(1)[0][0]
            moa_hits += int(majority_moa == moa)

    overall_consistency = float(fracs.mean()) if n_queries else 0.0
    moa_topk_accuracy = (moa_hits / n_queries) if n_queries else 0.0
    plate_neighbor_agreement = float(plate_fracs.mean()) if n_queries else 0.0

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

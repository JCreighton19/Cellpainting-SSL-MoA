"""
Replicate correlation diagnostic for Cell Painting embeddings.

Measures whether same-compound cross-plate well embeddings are more similar
to each other than random well pairs — the primary indicator of phenotypically
meaningful representations.

Assumes embeddings are row-aligned with master_metadata_qc.parquet (same
ordering used by the notebook). If plates/wells .npy files are available
from extract_embeddings.py, pass them explicitly for guaranteed correctness.

Usage
-----
# Minimal (assumes metadata alignment):
python replicate_correlation.py \
    --embeddings /scratch/.../embeddings/053126_1033/embeddings_epoch_10.npy \
    --metadata   /scratch/.../data/processed/master_metadata_qc.parquet

# With explicit plate/well arrays (more robust):
python replicate_correlation.py \
    --embeddings /scratch/.../embeddings_epoch_10.npy \
    --metadata   /scratch/.../master_metadata_qc.parquet \
    --plates     /scratch/.../plates_epoch_10.npy \
    --wells      /scratch/.../wells_epoch_10.npy
"""

import argparse
import numpy as np
import pandas as pd
from scipy import stats


def build_well_embeddings(embeddings, metadata):
    """Mean-pool tile embeddings to well level. Returns (well_embs, well_df)."""
    metadata = metadata.copy().reset_index(drop=True)
    metadata["_idx"] = np.arange(len(metadata))
    metadata["well_key"] = metadata["plate"].astype(str) + "__" + metadata["well"].astype(str)

    rows, embs = [], []
    for key, group in metadata.groupby("well_key", sort=False):
        idx = group["_idx"].values
        embs.append(embeddings[idx].mean(axis=0))
        rows.append({
            "well_key": key,
            "plate": group["plate"].iloc[0],
            "well": group["well"].iloc[0],
            "broad_sample": group["broad_sample"].mode().iloc[0],
            "moa": group["moa"].mode().iloc[0],
            "n_tiles": len(group),
        })

    well_embs = np.stack(embs).astype(np.float32)
    # L2 normalize for cosine similarity via dot product
    norms = np.linalg.norm(well_embs, axis=1, keepdims=True)
    well_embs = well_embs / (norms + 1e-8)
    return well_embs, pd.DataFrame(rows).reset_index(drop=True)


def replicate_sims(well_embs, well_df, compounds):
    """Cosine similarities for all cross-plate pairs of each compound."""
    sims = []
    for compound in compounds:
        mask = (well_df["broad_sample"] == compound).values
        c_embs = well_embs[mask]
        c_plates = well_df.loc[mask, "plate"].values
        for i in range(len(c_embs)):
            for j in range(i + 1, len(c_embs)):
                if c_plates[i] != c_plates[j]:
                    sims.append(float(c_embs[i] @ c_embs[j]))
    return np.array(sims)


def random_baseline(well_embs, well_df, n_pairs, seed=42):
    """Cosine similarities for random different-compound well pairs."""
    rng = np.random.default_rng(seed)
    compounds = well_df["broad_sample"].values
    n = len(well_df)
    sims, attempts = [], 0
    while len(sims) < n_pairs and attempts < n_pairs * 20:
        i, j = rng.integers(0, n, size=2)
        if compounds[i] != compounds[j]:
            sims.append(float(well_embs[i] @ well_embs[j]))
        attempts += 1
    return np.array(sims)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, help="Path to embeddings .npy")
    parser.add_argument("--metadata",   required=True, help="Path to master_metadata_qc.parquet")
    parser.add_argument("--plates",     default=None,  help="plates .npy aligned with embeddings")
    parser.add_argument("--wells",      default=None,  help="wells .npy aligned with embeddings")
    parser.add_argument("--n_random",   type=int, default=10000, help="Random pairs for baseline")
    parser.add_argument("--min_plates", type=int, default=2,     help="Min plates per compound to include")
    args = parser.parse_args()

    # --- Load ---
    embeddings = np.load(args.embeddings).astype(np.float32)
    metadata   = pd.read_parquet(args.metadata)

    # Align: drop rows without MoA (same filter as notebook)
    mask_valid = metadata["moa"].notna()
    metadata   = metadata[mask_valid].reset_index(drop=True)
    embeddings = embeddings[mask_valid.values]

    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Embedding count ({len(embeddings)}) != metadata rows ({len(metadata)}). "
            "Pass --plates and --wells for explicit alignment."
        )

    # Override plate/well from saved arrays if provided
    if args.plates and args.wells:
        plates = np.load(args.plates)
        wells  = np.load(args.wells)
        if len(plates) != len(embeddings):
            raise ValueError(f"plates array length {len(plates)} != embeddings {len(embeddings)}")
        metadata = metadata.copy()
        metadata["plate"] = plates
        metadata["well"]  = wells
        print("Using explicit plate/well arrays.")

    print(f"Tiles      : {len(embeddings)}")
    print(f"Plates     : {metadata['plate'].nunique()}  {sorted(metadata['plate'].unique())}")
    print(f"broad_sample unique : {metadata['broad_sample'].nunique()}")
    print(f"MoA unique          : {metadata['moa'].nunique()}")
    print(f"broad_sample null   : {metadata['broad_sample'].isna().sum()}")

    # --- Well-level aggregation ---
    well_embs, well_df = build_well_embeddings(embeddings, metadata)
    print(f"\nWells      : {len(well_df)}")
    print(f"Tiles/well : mean={well_df['n_tiles'].mean():.1f}  "
          f"min={well_df['n_tiles'].min()}  max={well_df['n_tiles'].max()}")

    # --- Identify treated wells with cross-plate replicates ---
    # Exclude controls and unknown compounds
    control_moas = {"control vehicle", "unknown"}
    control_cpds = {"DMSO", "unknown", ""}
    treat_mask = (
        ~well_df["moa"].isin(control_moas) &
        ~well_df["broad_sample"].isin(control_cpds) &
        well_df["broad_sample"].notna()
    )
    treat_df   = well_df[treat_mask].copy().reset_index(drop=True)
    treat_embs = well_embs[treat_mask.values]

    cpd_plate_counts = treat_df.groupby("broad_sample")["plate"].nunique()
    rep_compounds    = cpd_plate_counts[cpd_plate_counts >= args.min_plates].index.tolist()

    print(f"\nTreated wells : {len(treat_df)}")
    print(f"Compounds with >= {args.min_plates} plates : {len(rep_compounds)}")

    if len(rep_compounds) < 2:
        print("\nERROR: Too few cross-plate replicates to evaluate.")
        print("Check that broad_sample is populated and multiple plates contain the same compounds.")
        return

    # --- Compute similarities ---
    print("\nComputing replicate similarities...", flush=True)
    rep_sims  = replicate_sims(treat_embs, treat_df, rep_compounds)

    print(f"Computing random baseline ({args.n_random} pairs)...", flush=True)
    rand_sims = random_baseline(treat_embs, treat_df, args.n_random)

    # --- Stats ---
    delta   = rep_sims.mean() - rand_sims.mean()
    z_score = delta / (rand_sims.std() + 1e-8)
    enrich  = rep_sims.mean() / (abs(rand_sims.mean()) + 1e-8)
    pct_above = (rep_sims > rand_sims.mean()).mean() * 100
    _, pval = stats.mannwhitneyu(rep_sims, rand_sims, alternative="greater")

    sep = "=" * 65
    print(f"\n{sep}")
    print("REPLICATE CORRELATION DIAGNOSTIC")
    print(sep)
    print(f"  Replicate pairs    : {len(rep_sims)}")
    print(f"  Random pairs       : {len(rand_sims)}")
    print()
    print(f"  Replicate sim      : mean={rep_sims.mean():.4f}  "
          f"std={rep_sims.std():.4f}  median={np.median(rep_sims):.4f}")
    print(f"  Random sim         : mean={rand_sims.mean():.4f}  "
          f"std={rand_sims.std():.4f}  median={np.median(rand_sims):.4f}")
    print()
    print(f"  Delta (rep - rand) : {delta:+.4f}")
    print(f"  Z-score            : {z_score:.2f}")
    print(f"  Enrichment ratio   : {enrich:.3f}x")
    print(f"  % replicates > rand mean : {pct_above:.1f}%")
    print(f"  Mann-Whitney p     : {pval:.3e}")

    print(f"\n{sep}")
    if pval > 0.05 or enrich < 1.02:
        verdict = "NO SIGNAL  — embeddings do not generalize across plates per compound."
        verdict += "\n           Root cause is upstream of model geometry / loss choice."
    elif enrich < 1.10:
        verdict = "WEAK SIGNAL — marginal cross-plate consistency. May reflect noise."
    elif enrich < 1.30:
        verdict = "MODERATE SIGNAL — detectable replicate structure."
    else:
        verdict = "STRONG SIGNAL — robust cross-plate phenotypic structure."
    print(f"  VERDICT: {verdict}")
    print(sep)

    # --- Per-MoA breakdown ---
    print("\nPer-MoA breakdown (compounds with cross-plate replicates):")
    print(f"  {'MoA':<42}  {'cpds':>5}  {'pairs':>6}  {'mean_sim':>9}  {'enrich':>8}")
    print("  " + "-" * 75)

    moa_rows = []
    rep_set  = set(rep_compounds)
    for moa, grp in treat_df[treat_df["broad_sample"].isin(rep_set)].groupby("moa"):
        cpds = [c for c in grp["broad_sample"].unique() if c in rep_set]
        sims = replicate_sims(treat_embs, treat_df, cpds)
        if len(sims):
            moa_rows.append((moa, len(cpds), len(sims), sims.mean(),
                             sims.mean() / (abs(rand_sims.mean()) + 1e-8)))

    for moa, n_cpd, n_pairs, m_sim, enr in sorted(moa_rows, key=lambda x: -x[4]):
        print(f"  {str(moa)[:42]:<42}  {n_cpd:>5}  {n_pairs:>6}  {m_sim:>9.4f}  {enr:>7.3f}x")

    # --- Tile vs well comparison ---
    print(f"\nTile-level MoA kNN (from embedding, if you want to compare):")
    print(f"  Run the notebook kNN cell on the same embeddings for comparison.")
    print(f"  Key diagnostic: well-level enrichment should be HIGHER than tile-level.")
    print(f"  If it is LOWER, the model is encoding within-well nuisance features.")


if __name__ == "__main__":
    main()

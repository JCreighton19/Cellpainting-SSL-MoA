"""
Phase 2 data prep: aggregate well-level embeddings (from prepare_phase1_data.py)
up to compound-level profiles.

Additive to Phase 1 — reads app_data/wells.parquet + well_embeddings.npy,
writes app_data/compounds.parquet + compound_embeddings.npy alongside them.
Does not modify or regenerate the well-level artifacts.

Usage:
    python scripts/prepare_phase2_compounds.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "app_data"


def l2_normalize(X, eps=1e-8):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    wells_path = args.data_dir / "wells.parquet"
    embs_path = args.data_dir / "well_embeddings.npy"
    print(f"Loading {wells_path}")
    wells = pd.read_parquet(wells_path)
    print(f"Loading {embs_path}")
    well_embs = np.load(embs_path)

    if len(wells) != len(well_embs):
        raise ValueError(
            f"wells.parquet has {len(wells)} rows but well_embeddings.npy has "
            f"{len(well_embs)} rows — these must come from the same build."
        )

    # broad_sample is None for control wells; groupby drops NaN keys by default,
    # so control wells are naturally excluded from compound-level aggregation.
    wells = wells.reset_index(drop=True).copy()
    wells["_idx"] = np.arange(len(wells))

    print("Mean-pooling wells -> compound-level profiles ...")
    rows, compound_embs = [], []
    for compound_id, grp in wells.groupby("broad_sample", sort=False):
        pooled = well_embs[grp["_idx"].values].mean(axis=0)
        compound_embs.append(pooled)

        moa_counts = grp["moa"].dropna().value_counts()
        dominant_moa = moa_counts.index[0] if len(moa_counts) else None

        name_counts = grp["pert_iname"].dropna().value_counts()
        compound_name = name_counts.index[0] if len(name_counts) else None

        rows.append({
            "compound_id": compound_id,
            "compound_name": compound_name,  # human-readable, e.g. "dexamethasone"
            "dominant_moa": dominant_moa,
            "n_wells": len(grp),
            "well_ids": grp["well_id"].tolist(),
            "thumbnail_path": None,  # same Phase 1 limitation: no rendered images yet
        })

    compound_embs = l2_normalize(np.stack(compound_embs)).astype(np.float32)
    compounds = pd.DataFrame(rows)
    print(f"Compounds: {len(compounds):,}  (mean {compounds['n_wells'].mean():.1f} wells/compound)")

    compounds_path = args.data_dir / "compounds.parquet"
    compound_embs_path = args.data_dir / "compound_embeddings.npy"
    compounds.to_parquet(compounds_path, index=False)
    np.save(compound_embs_path, compound_embs)
    print(f"Wrote {compounds_path} ({len(compounds)} rows)")
    print(f"Wrote {compound_embs_path} {compound_embs.shape}")


if __name__ == "__main__":
    main()

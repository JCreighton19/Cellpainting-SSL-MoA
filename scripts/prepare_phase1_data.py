"""
Phase 1 data prep: turn tile-level embeddings + master metadata into a
well-level parquet + embedding matrix the Flask app can load directly.

One-time (per checkpoint) offline step. Not part of the running app.

Usage:
    python scripts/prepare_phase1_data.py
    python scripts/prepare_phase1_data.py --emb embeddings/RUN/embeddings_epoch_N.npy
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from utils.postprocessing import postprocess  # noqa: E402

DEFAULT_EMB = REPO_ROOT / "embeddings" / "070226_135708" / "embeddings_epoch_200.npy"
DEFAULT_META = REPO_ROOT / "data" / "processed" / "master_metadata_qc.parquet"
DEFAULT_OUT = REPO_ROOT / "app_data"
# Populated by scripts/generate_web_thumbnails.py / generate_web_attention_maps.py;
# only referenced (not generated) here.
THUMBNAILS_DIR = REPO_ROOT / "webapp" / "static" / "thumbnails"
ATTENTION_DIR = REPO_ROOT / "webapp" / "static" / "attention"


def l2_normalize(X, eps=1e-8):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", type=Path, default=DEFAULT_EMB)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    print(f"Loading tile embeddings: {args.emb}")
    tile_embs = np.load(args.emb).astype(np.float32)
    print(f"Loading metadata: {args.meta}")
    meta = pd.read_parquet(args.meta)

    if len(tile_embs) != len(meta):
        raise ValueError(
            f"Row count mismatch: {len(tile_embs)} embeddings vs {len(meta)} metadata rows. "
            "These must come from the same dataset iteration (positional join)."
        )

    meta = meta.reset_index(drop=True).copy()
    meta["_idx"] = np.arange(len(meta))

    print("Mean-pooling tiles -> well-level profiles ...")
    rows, well_embs = [], []
    for (plate, well), grp in meta.groupby(["plate", "well"], sort=False):
        well_embs.append(tile_embs[grp["_idx"].values].mean(axis=0))
        bs = grp["broad_sample"].dropna()
        mo = grp["moa"].dropna()
        pi = grp["pert_iname"].dropna()  # human-readable compound name, e.g. "dexamethasone"
        thumb_path = THUMBNAILS_DIR / f"{plate}_{well}.webp"
        attn_path = ATTENTION_DIR / f"{plate}_{well}.npy"
        rows.append({
            "well_id": f"{plate}_{well}",
            "plate": plate,
            "well": well,
            "broad_sample": bs.iloc[0] if len(bs) else None,
            "pert_iname": pi.iloc[0] if len(pi) else None,
            "moa": mo.iloc[0] if len(mo) else None,
            "is_control": bool(grp["is_control"].max()),
            "control_type": grp["control_type"].dropna().iloc[0] if grp["control_type"].notna().any() else None,
            "n_tiles": len(grp),
            "thumbnail_path": f"thumbnails/{plate}_{well}.webp" if thumb_path.exists() else None,
            "attention_path": f"attention/{plate}_{well}.npy" if attn_path.exists() else None,
        })
    well_embs = np.stack(well_embs).astype(np.float32)
    wells = pd.DataFrame(rows)
    print(f"Wells: {len(wells):,}  (mean {wells['n_tiles'].mean():.1f} tiles/well)")

    print("Applying per-plate MAD + global sphering postprocessing (fit on negative-control wells) ...")
    ctrl_mask = (wells["control_type"] == "negcon").values
    print(f"  {ctrl_mask.sum()} negative-control wells used for fit")
    well_embs_post = postprocess(well_embs, ctrl_mask, wells["plate"].values)
    well_embs_post = l2_normalize(well_embs_post).astype(np.float32)

    print("Fitting 2D UMAP on well embeddings ...")
    import umap
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    coords = reducer.fit_transform(well_embs_post)
    wells["umap_x"] = coords[:, 0]
    wells["umap_y"] = coords[:, 1]

    args.out.mkdir(parents=True, exist_ok=True)
    wells_path = args.out / "wells.parquet"
    embs_path = args.out / "well_embeddings.npy"
    wells.to_parquet(wells_path, index=False)
    np.save(embs_path, well_embs_post)
    print(f"Wrote {wells_path} ({len(wells)} rows)")
    print(f"Wrote {embs_path} {well_embs_post.shape}")


if __name__ == "__main__":
    main()

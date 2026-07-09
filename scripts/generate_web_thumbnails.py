"""
One-time offline web artifact generation: builds a compressed RGB preview
image (256x256 WebP) per well for the Flask app's right-sidebar thumbnail.
NOT part of training or the DINO data pipeline -- run this by hand whenever
you want the webapp's thumbnails refreshed for a given embedding run.

Source images are the full site/FOV tensors already saved by
datasets/preprocess_dataset.py (5, H, W float32 in [0,1] -- NOT tiles), read
via master_metadata_qc.parquet's "pt_path" column. Channel order is fixed by
that script's `paths = [mito, agp, rna, er, dna]`, confirmed here rather than
assumed.

Representative site: each well typically has multiple sites/FOVs. We pick
the one whose (site-level) embedding is closest, by cosine similarity, to
that well's own mean embedding centroid -- i.e. the most "typical" image for
that well, using the same tile embeddings and (plate, well) grouping as
scripts/prepare_phase1_data.py.

Usage:
    python scripts/generate_web_thumbnails.py
    python scripts/generate_web_thumbnails.py --emb embeddings/RUN/embeddings_epoch_N.npy
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_EMB = REPO_ROOT / "embeddings" / "070226_135708" / "embeddings_epoch_200.npy"
DEFAULT_META = REPO_ROOT / "data" / "processed" / "master_metadata_qc.parquet"
DEFAULT_OUT = REPO_ROOT / "webapp" / "static" / "thumbnails"

THUMB_SIZE = 256
WEBP_QUALITY = 85
ESTIMATE_SAMPLE_SIZE = 20  # wells used to project total storage before the full run

# Channel order fixed by datasets/preprocess_dataset.py: paths = [mito, agp, rna, er, dna]
CHANNEL_ORDER = ["mito", "agp", "rna", "er", "dna"]
CHANNEL_COLORS = {
    "mito": (255, 0, 0),    # red
    "agp":  (255, 255, 0),  # yellow (Actin/Golgi/Plasma membrane stain)
    "rna":  (255, 0, 255),  # magenta
    "er":   (0, 255, 0),    # green
    "dna":  (0, 0, 255),    # blue
}
# (5, 3) matrix so a composite is one array contraction, not a per-channel loop.
_COLOR_MATRIX = np.array([CHANNEL_COLORS[c] for c in CHANNEL_ORDER], dtype=np.float32) / 255.0


def make_composite(image: np.ndarray) -> Image.Image:
    """image: (5, H, W) float32 in [0,1] -> PIL RGB image."""
    rgb = np.tensordot(image, _COLOR_MATRIX, axes=([0], [0]))  # (H, W, 3)
    rgb = np.clip(rgb, 0.0, 1.0)
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


def pick_representative_sites(tile_embs, meta):
    """Returns {(plate, well): row_idx} for the site closest (cosine) to
    each well's mean embedding centroid."""
    norm_embs = tile_embs / np.maximum(np.linalg.norm(tile_embs, axis=1, keepdims=True), 1e-8)
    reps = {}
    for (plate, well), grp in meta.groupby(["plate", "well"], sort=False):
        idxs = grp["_idx"].values
        centroid = norm_embs[idxs].mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-8)
        best_local = np.argmax(norm_embs[idxs] @ centroid)
        reps[(plate, well)] = idxs[best_local]
    return reps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", type=Path, default=DEFAULT_EMB)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    print(f"Loading site embeddings: {args.emb}")
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

    print("Selecting one representative site per well (closest to well centroid) ...")
    reps = pick_representative_sites(tile_embs, meta)
    wells = list(reps.items())
    n_wells = len(wells)
    print(f"Wells to generate: {n_wells:,}")

    args.out.mkdir(parents=True, exist_ok=True)

    sizes = []
    for i, ((plate, well), row_idx) in enumerate(wells):
        pt_path = meta.loc[row_idx, "pt_path"]
        sample = torch.load(pt_path, weights_only=False)
        img = make_composite(sample["image"].numpy())
        img = img.resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)

        out_path = args.out / f"{plate}_{well}.webp"
        img.save(out_path, "WEBP", quality=WEBP_QUALITY)
        sizes.append(out_path.stat().st_size)

        if (i + 1) == min(ESTIMATE_SAMPLE_SIZE, n_wells):
            avg_kb = np.mean(sizes) / 1024
            print(f"\n--- Storage estimate (from first {i + 1} wells) ---")
            print(f"  Thumbnails to generate : {n_wells:,}")
            print(f"  Average file size      : {avg_kb:.1f} KB")
            print(f"  Estimated total size   : {avg_kb * n_wells / 1024:.1f} MB")
            print(f"---------------------------------------------------\n")
        elif (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n_wells}")

    total_mb = sum(sizes) / 1024 / 1024
    print(f"\nDone. {n_wells:,} thumbnails written to {args.out}")
    print(f"Actual total size: {total_mb:.1f} MB ({np.mean(sizes) / 1024:.1f} KB/thumbnail avg)")


if __name__ == "__main__":
    main()

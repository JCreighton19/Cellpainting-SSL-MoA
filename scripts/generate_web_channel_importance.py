"""
One-time offline web artifact generation: computes per-well channel-ablation
importance scores (analysis/channel_ablation.py's compute_channel_importance)
for the Flask app's right-sidebar channel-importance display. NOT part of
training.

Unlike attention maps (one .npy file per well, since each is a real 2D
array), channel importance is just 5 channels x 2 floats per well -- small
enough that every well's result is written into ONE combined JSON file
rather than one file per well.

Reuses existing machinery rather than building a new pipeline:
  - analysis.extract_embeddings.get_checkpoints / load_model -- checkpoint
    discovery and model loading, identical to every other extraction script.
  - analysis.channel_ablation.compute_channel_importance -- the verified
    ablation algorithm itself (see analysis/channel_ablation.py).
  - scripts.generate_web_thumbnails.pick_representative_sites -- the SAME
    embedding-centroid site selection used for thumbnails and attention
    maps, so the channel-importance scores generated here correspond to the
    exact same image as the thumbnail already shown in the UI.

Usage:
    python scripts/generate_web_channel_importance.py --run_dir /path/to/checkpoints
    python scripts/generate_web_channel_importance.py --run_dir ... --epoch 200 --fill mean
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from analysis.extract_embeddings import get_checkpoints, load_model  # noqa: E402
from analysis.channel_ablation import compute_channel_importance  # noqa: E402
from generate_web_thumbnails import (  # noqa: E402
    SCRATCH_ROOT, DEFAULT_EMB, DEFAULT_META, pick_representative_sites,
)

DEFAULT_RUN_DIR = SCRATCH_ROOT / "checkpoints" / "070226_135708"
DEFAULT_OUT = REPO_ROOT / "webapp" / "static" / "channel_importance.json"


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--epoch", type=int, default=None,
                         help="Specific epoch checkpoint to use (default: latest)")
    parser.add_argument("--emb", type=Path, default=DEFAULT_EMB)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fill", choices=["zero", "mean"], default="zero")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    checkpoints = get_checkpoints(args.run_dir)
    if args.epoch is not None:
        matches = [(e, p) for (e, p) in checkpoints if e == args.epoch]
        if not matches:
            raise ValueError(f"No checkpoint found for epoch {args.epoch}")
        epoch, checkpoint_path = matches[0]
    else:
        epoch, checkpoint_path = sorted(checkpoints)[-1]
    print(f"Using checkpoint: {checkpoint_path} (epoch {epoch})")

    model = load_model(checkpoint_path, device)

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

    print("Selecting one representative site per well (same selection as thumbnails/attention) ...")
    reps = pick_representative_sites(tile_embs, meta)
    wells = list(reps.items())
    n_wells = len(wells)
    print(f"Wells to process: {n_wells:,}  (fill={args.fill})")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    for i, ((plate, well), row_idx) in enumerate(wells):
        pt_path = meta.loc[row_idx, "pt_path"]
        sample = torch.load(pt_path, weights_only=False)
        results[f"{plate}_{well}"] = compute_channel_importance(
            model, sample["image"], sample["otsu_threshold"], device, fill=args.fill
        )

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n_wells}")

    with open(args.out, "w") as f:
        json.dump(results, f)

    size_mb = args.out.stat().st_size / 1024 / 1024
    print(f"\nDone. {len(results):,} wells written to {args.out} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()

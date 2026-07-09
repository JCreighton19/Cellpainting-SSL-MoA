"""
One-time offline web artifact generation: builds a DINO attention map
(float16) per well, for the Flask app's right-sidebar "Attention" / "Overlay"
toggle. NOT part of training.

Saved at raw ViT patch-grid resolution (small -- a few KB per well), not
upsampled offline: webapp/routes.py resizes with PIL's bilinear
interpolation at render time, so smoothing happens in the Flask app rather
than being baked into a much larger stored array. (An earlier version of
this script upsampled each crop to CROP_SIZE x CROP_SIZE and stored that --
float16 at ~7.6GB total across all wells -- which fixed the visual
blockiness but was far too large to be a lightweight web artifact. Reverted
in favor of this patch-grid-resolution version.)

Reuses existing attention-extraction machinery rather than building a new
pipeline:
  - analysis.extract_attention_maps.patch_attention_to_expose_weights /
    extract_attention_for_crop -- the monkey-patched CLS->patch attention
    extraction for one 224x224 crop (last transformer block).
  - analysis.extract_embeddings.get_checkpoints / load_model -- checkpoint
    discovery and model loading, identical to every other extraction script.
  - scripts.generate_web_thumbnails.pick_representative_sites -- the SAME
    embedding-centroid site selection used for thumbnails, so the attention
    map generated here corresponds to the exact same image as the thumbnail
    already shown in the UI.

Unlike analysis/extract_attention_maps.py (which samples a few random FOVs
and a few random crops each, for notebook diagnostics), this script covers
every well's representative site, and tiles that FULL site into all its
non-overlapping foreground crops (same tiling as extract_embeddings.py's
embed_fov) rather than just one crop -- each crop's raw (grid, grid) CLS
attention patch is placed into a mosaic at that crop's position, so the
resulting map spatially lines up with the full-site thumbnail instead of
covering only a small sub-region of it.

Usage:
    python scripts/generate_web_attention_maps.py --run_dir /path/to/checkpoints
    python scripts/generate_web_attention_maps.py --run_dir ... --epoch 200
"""
import argparse
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
from analysis.extract_attention_maps import (  # noqa: E402
    patch_attention_to_expose_weights,
    extract_attention_for_crop,
)
from generate_web_thumbnails import (  # noqa: E402
    SCRATCH_ROOT, DEFAULT_EMB, DEFAULT_META, pick_representative_sites,
)

CROP_SIZE = 224
MIN_FG = 0.01  # paper: >=1% foreground in DNA channel -- matches extract_attention_maps.py
DEFAULT_RUN_DIR = SCRATCH_ROOT / "checkpoints" / "070226_135708"
DEFAULT_OUT = REPO_ROOT / "webapp" / "static" / "attention"
ESTIMATE_SAMPLE_SIZE = 20  # wells used to project total storage before the full run


@torch.no_grad()
def attention_mosaic_for_site(model, attn_block, image, otsu_thresh, device):
    """
    image: (5, H, W) float32 in [0,1], the full site tensor (same one used
    to build the thumbnail). Tiles it into non-overlapping CROP_SIZE crops
    (same unfold + Otsu foreground filter as extract_embeddings.embed_fov),
    runs each foreground crop through extract_attention_for_crop(), and
    places each crop's mean-over-heads CLS attention grid into a mosaic at
    that crop's position. Falls back to the single centre crop if no tile
    passes the foreground filter (same fallback as embed_fov).

    Returns a (n_h*grid, n_w*grid) float16 array of continuous, min-max
    normalized [0,1] attention values (resizing/smoothing happens in
    webapp/routes.py at render time, not here), or None if the image is
    smaller than one crop.
    """
    C, H, W = image.shape
    n_h, n_w = H // CROP_SIZE, W // CROP_SIZE
    if n_h == 0 or n_w == 0:
        return None

    crops = (image.unfold(1, CROP_SIZE, CROP_SIZE)
                  .unfold(2, CROP_SIZE, CROP_SIZE)
                  .permute(1, 2, 0, 3, 4)
                  .contiguous()
                  .view(n_h * n_w, C, CROP_SIZE, CROP_SIZE))
    fg = (crops[:, 4] > otsu_thresh).float().mean(dim=[1, 2]) >= MIN_FG
    positions = [(i // n_w, i % n_w) for i in range(n_h * n_w) if fg[i]]
    fg_crops = crops[fg]

    if len(fg_crops) == 0:
        r0, c0 = (H - CROP_SIZE) // 2, (W - CROP_SIZE) // 2
        fg_crops = image[:, r0:r0 + CROP_SIZE, c0:c0 + CROP_SIZE].unsqueeze(0)
        positions = [(n_h // 2, n_w // 2)]

    mosaic = None
    for crop, (r, c) in zip(fg_crops, positions):
        per_head = extract_attention_for_crop(model, crop, device)  # (heads, grid, grid)
        cls_attn = per_head.mean(axis=0)  # mean over heads -> (grid, grid)
        grid = cls_attn.shape[0]
        if mosaic is None:
            mosaic = np.zeros((n_h * grid, n_w * grid), dtype=np.float32)
        mosaic[r * grid:(r + 1) * grid, c * grid:(c + 1) * grid] = cls_attn

    attn = mosaic.astype(np.float32)
    attn -= attn.min()
    attn /= (attn.max() + 1e-8)
    return attn.astype(np.float16)


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
    patch_attention_to_expose_weights(model.vit.blocks[-1].attn)
    attn_block = model.vit.blocks[-1].attn

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

    print("Selecting one representative site per well (same selection as thumbnails) ...")
    reps = pick_representative_sites(tile_embs, meta)
    wells = list(reps.items())
    n_wells = len(wells)
    print(f"Wells to generate: {n_wells:,}")

    args.out.mkdir(parents=True, exist_ok=True)

    sizes = []
    for i, ((plate, well), row_idx) in enumerate(wells):
        pt_path = meta.loc[row_idx, "pt_path"]
        sample = torch.load(pt_path, weights_only=False)
        mosaic = attention_mosaic_for_site(
            model, attn_block, sample["image"], sample["otsu_threshold"], device
        )
        if mosaic is None:
            continue

        out_path = args.out / f"{plate}_{well}.npy"
        np.save(out_path, mosaic)
        sizes.append(out_path.stat().st_size)

        if (i + 1) == min(ESTIMATE_SAMPLE_SIZE, n_wells):
            avg_kb = np.mean(sizes) / 1024
            print(f"\n--- Storage estimate (from first {i + 1} wells) ---")
            print(f"  Maps to generate     : {n_wells:,}")
            print(f"  Attention grid shape : {mosaic.shape} (float16)")
            print(f"  Average file size    : {avg_kb:.2f} KB")
            print(f"  Estimated total size : {avg_kb * n_wells / 1024:.1f} MB")
            print(f"-----------------------------------------------------\n")
        elif (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n_wells}")

    total_mb = sum(sizes) / 1024 / 1024
    print(f"\nDone. {len(sizes):,} attention maps written to {args.out}")
    print(f"Actual total size: {total_mb:.1f} MB ({np.mean(sizes) / 1024:.2f} KB/map avg)")


if __name__ == "__main__":
    main()

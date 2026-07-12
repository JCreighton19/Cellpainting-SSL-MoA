"""
Post hoc channel-ablation interpretability: for a single Cell Painting FOV,
estimate how important each of the 5 fluorescence channels is to the
model's embedding by zeroing (or mean-filling) one channel at a time and
measuring how far the embedding moves from the unablated baseline.

Uses the trained checkpoint exactly as-is: no retraining, no fine-tuning,
no architecture change. Model stays in eval() mode with gradients disabled
throughout (torch.no_grad()).

Usage:
    python analysis/channel_ablation.py --image /path/to/well.pt --run_dir /path/to/checkpoints
    python analysis/channel_ablation.py --image /path/to/well.pt --checkpoint /path/to/dino_epoch_200.pt
    python analysis/channel_ablation.py --image /path/to/well.pt --run_dir ... --fill mean --out results.json
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from analysis.extract_embeddings import (
    CROP_SIZE,
    CROP_BATCH,
    get_checkpoints,
    load_model,
    select_foreground_crops,
    embed_crops,
)

# Tensor channel order is fixed by preprocess_dataset.py's
# `paths = [mito, agp, rna, er, dna]` -> np.stack(..., axis=0).
CHANNEL_NAMES = ["Mito", "AGP", "RNA", "ER", "DNA"]


@torch.no_grad()
def _forward_crops_batched(model, crops, device):
    """Sub-batched forward pass, mirroring embed_crops's VRAM-bounding
    pattern (analysis/extract_embeddings.py) -- returns RAW (unpooled,
    un-normalised) per-crop embeddings instead of a single pooled one, so
    multiple image variants can share one batched pass and be pooled
    separately afterward."""
    crops = crops.to(device)
    parts = [model(crops[i:i + CROP_BATCH]) for i in range(0, len(crops), CROP_BATCH)]
    return torch.cat(parts, dim=0)  # (K, D)


@torch.no_grad()
def compute_channel_importance(model, image, otsu_thresh, device, fill="zero"):
    """
    For one FOV, estimate each channel's importance to the embedding via
    ablation: zero (or mean-fill) one channel at a time, re-embed, and
    compare to the unablated baseline via cosine similarity.

    Foreground crops are selected ONCE from the ORIGINAL image and reused
    for every ablated variant, including the DNA ablation itself.
    select_foreground_crops picks crops based on the DNA channel, so
    re-selecting from a DNA-zeroed copy would degenerate to a single centre
    crop (see select_foreground_crops's fallback) instead of measuring the
    same locations used for every other channel -- that would make the DNA
    score measure "one center crop vs many pooled crops" rather than "same
    crops, DNA removed".

    Args:
        model: a loaded CellPaintingViT in eval() mode (see load_model).
        image: (5, H, W) tensor, already preprocessed -- as loaded from a
            data/tiles_qc/*.pt payload's "image" key.
        otsu_thresh: float, that same payload's "otsu_threshold".
        device: torch device.
        fill: "zero" (hard zero, the channel's low-intensity extreme after
            preprocess_dataset.py's percentile normalization) or "mean"
            (that image's own per-channel mean, avoiding the sharp
            synthetic edge a hard zero creates at every crop boundary).

    Returns:
        {channel_name: {"importance": float, "cosine_similarity": float}}
        importance = 1 - cosine_similarity(baseline_embedding, ablated_embedding)
    """
    if fill not in ("zero", "mean"):
        raise ValueError(f"fill must be 'zero' or 'mean', got {fill!r}")

    crops = select_foreground_crops(image, otsu_thresh)  # (K, 5, 224, 224)
    n_crops = crops.shape[0]

    baseline_emb = embed_crops(model, crops, device)  # (D,) pooled + L2-normalised

    fill_values = image.mean(dim=(1, 2)) if fill == "mean" else torch.zeros(5)

    # All 5 ablated variants share the identical K crop locations, so they
    # concatenate into one (5*K, 5, 224, 224) batch for a single round of
    # sub-batched model calls instead of 5 separate embed_crops() calls.
    ablated = crops.unsqueeze(0).repeat(5, 1, 1, 1, 1)  # (5, K, 5, 224, 224) -- copies, doesn't alias crops
    for c in range(5):
        ablated[c, :, c, :, :] = fill_values[c]
    ablated = ablated.view(5 * n_crops, 5, CROP_SIZE, CROP_SIZE)

    raw = _forward_crops_batched(model, ablated, device).view(5, n_crops, -1)  # (5, K, D)

    results = {}
    for c, name in enumerate(CHANNEL_NAMES):
        # Match embed_crops's exact pooling: normalise each crop embedding,
        # then mean over crops -- WITHOUT re-normalising after the mean
        # (embed_crops doesn't either), so this is directly comparable to
        # baseline_emb.
        ablated_emb = F.normalize(raw[c], dim=1).mean(dim=0)
        cos_sim = F.cosine_similarity(baseline_emb.unsqueeze(0), ablated_emb.unsqueeze(0)).item()
        results[name] = {
            "importance": 1.0 - cos_sim,
            "cosine_similarity": cos_sim,
        }
    return results


def print_table(results):
    print(f"{'Channel':<12} {'Cosine Similarity':>18} {'Importance':>14}")
    print("-" * 48)
    for name, scores in sorted(results.items(), key=lambda kv: -kv[1]["importance"]):
        print(f"{name:<12} {scores['cosine_similarity']:>18.2f} {scores['importance']:>14.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="Path to a data/tiles_qc/*.pt payload.")
    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--run_dir", help="Checkpoint dir; uses the latest dino_epoch_N.pt.")
    ckpt_group.add_argument("--checkpoint", help="Path to a specific dino_epoch_N.pt.")
    parser.add_argument("--fill", choices=["zero", "mean"], default="zero")
    parser.add_argument("--out", default=None, help="Output JSON path (default: <image_stem>_channel_importance.json next to --image).")
    args = parser.parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Using device:", device)

    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = get_checkpoints(args.run_dir)[-1][1]
    print("Checkpoint:", checkpoint_path)
    model = load_model(checkpoint_path, device)

    image_path = Path(args.image)
    payload = torch.load(image_path, weights_only=False)
    image = payload["image"]
    otsu_thresh = payload["otsu_threshold"]
    print(f"Image: {image_path}  (plate={payload.get('plate')}, well={payload.get('well')}, "
          f"site={payload.get('site')}, otsu_threshold={otsu_thresh:.4f})")

    results = compute_channel_importance(model, image, otsu_thresh, device, fill=args.fill)

    print()
    print_table(results)

    out_path = Path(args.out) if args.out else image_path.with_name(f"{image_path.stem}_channel_importance.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()

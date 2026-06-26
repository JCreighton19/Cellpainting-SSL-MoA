"""
Offline foreground-aware tile precomputation.

For each processed .pt image file, generates a bank of N diverse 224x224
foreground-aware tiles and saves them as a sidecar {stem}_tiles.pt file.

Sampling strategy:
  - Identical to the online foreground_crop used during training:
    rejection sampling against the per-image Otsu threshold, 1% DNA-channel
    foreground criterion, up to 10 attempts per crop.
  - Diversity filtering: a candidate tile is rejected if its IoU with any
    already-accepted tile exceeds --max_iou (default 0.5). This removes
    near-duplicate crops without altering the foreground-aware sampling
    distribution — the paper uses online random sampling (no diversity
    guarantee), so removing degenerate near-duplicates is conservative
    and consistent with the paper's intent.

Storage:
  - Tiles are saved as float32 by default. Pass --fp16 to halve disk usage
    (~3 decimal places of precision is sufficient for [0,1]-normalised images).
  - Output path is always {original_stem}_tiles.pt in the same directory,
    derived deterministically from the input path (no metadata changes needed).

Usage:
  python datasets/precompute_tiles.py [--processed_dir PATH] [--n_tiles 20]
                                      [--crop_size 224] [--max_iou 0.5]
                                      [--fp16] [--overwrite]

  If --processed_dir is omitted, defaults to $CP_OUTPUT_ROOT/data/tiles_qc.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Ensure project root is on path when run directly from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.foreground_crop import foreground_crop_single


def _iou(y1, x1, y2, x2, ts):
    """Intersection-over-Union between two square crops of side ts."""
    inter_h = max(0, ts - abs(y1 - y2))
    inter_w = max(0, ts - abs(x1 - x2))
    inter   = inter_h * inter_w
    return inter / (2 * ts * ts - inter)


def generate_tiles(image, otsu_threshold, n_tiles, crop_size, max_iou):
    """
    Sample up to n_tiles foreground-aware tiles from a single image.
    Candidates with IoU >= max_iou against any accepted tile are discarded.
    Falls back gracefully if the image cannot yield n_tiles diverse crops
    (e.g. very small foreground area).
    """
    tiles, positions = [], []
    budget = n_tiles * 30  # generous retry limit

    for _ in range(budget):
        if len(tiles) >= n_tiles:
            break
        crop, y0, x0 = foreground_crop_single(image, crop_size, otsu_threshold)
        if all(_iou(y0, x0, py, px, crop_size) < max_iou for py, px in positions):
            tiles.append(crop)
            positions.append((y0, x0))

    if not tiles:
        # Absolute fallback: accept one crop without diversity check
        crop, _, _ = foreground_crop_single(image, crop_size, otsu_threshold)
        tiles.append(crop)

    return torch.stack(tiles)  # (n_accepted, C, crop_size, crop_size)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute foreground-aware tile banks for DINO training."
    )
    parser.add_argument(
        "--processed_dir", default=None,
        help="Directory containing *.pt image files. "
             "Defaults to $CP_OUTPUT_ROOT/data/tiles_qc."
    )
    parser.add_argument("--n_tiles",   type=int,   default=20,
                        help="Target number of tiles per image (default: 20).")
    parser.add_argument("--crop_size", type=int,   default=224,
                        help="Tile spatial size in pixels (default: 224).")
    parser.add_argument("--max_iou",   type=float, default=0.5,
                        help="Maximum IoU allowed between any two tiles (default: 0.5).")
    parser.add_argument("--fp16",      action="store_true",
                        help="Store tiles as float16 to halve disk usage.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate tiles even if sidecar file already exists.")
    args = parser.parse_args()

    if args.processed_dir is None:
        root = os.environ.get("CP_OUTPUT_ROOT")
        if root is None:
            raise EnvironmentError("Set --processed_dir or export CP_OUTPUT_ROOT.")
        args.processed_dir = os.path.join(root, "data/tiles_qc")

    pt_files = sorted(
        p for p in Path(args.processed_dir).rglob("*.pt")
        if not p.stem.endswith("_tiles")
    )
    print(f"Found {len(pt_files)} image files in {args.processed_dir}")
    print(f"Config: n_tiles={args.n_tiles}, crop_size={args.crop_size}, "
          f"max_iou={args.max_iou}, fp16={args.fp16}")

    skipped = 0
    for pt_path in tqdm(pt_files, desc="Precomputing tiles"):
        out_path = pt_path.with_name(pt_path.stem + "_tiles.pt")

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        sample = torch.load(pt_path, weights_only=False)
        image  = sample["image"]                    # (C, H, W) float32
        thresh = sample["otsu_threshold"]
        if hasattr(thresh, "item"):
            thresh = thresh.item()

        tiles = generate_tiles(image, thresh, args.n_tiles, args.crop_size, args.max_iou)
        # tiles: (n_accepted, C, crop_size, crop_size) float32

        if args.fp16:
            tiles = tiles.half()

        torch.save(tiles, out_path)

    print(f"Done. Skipped {skipped} already-computed files.")


if __name__ == "__main__":
    main()

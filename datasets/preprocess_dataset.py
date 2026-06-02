import os
import torch
import tifffile as tiff
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

OUT_DIR = Path(os.path.join(
    os.environ["CP_OUTPUT_ROOT"],
    "data/tiles_qc"
))
OUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize(image, channel_stats, eps=1e-6):
    image = np.log1p(image)
    normed = np.empty_like(image, dtype=np.float32)
    for c in range(image.shape[0]):
        mean = channel_stats[c]["mean"]
        std  = channel_stats[c]["std"]
        normed[c] = (image[c] - mean)/(std + eps)

    return normed


def image_qc(image):
    # image: (C, H, W)

    flat = image.reshape(image.shape[0], -1)
    per_channel_std = np.std(flat, axis=1)
    per_channel_var = np.var(flat, axis=1)
    p1, p99 = np.percentile(flat, [1, 99])

    global_std = float(np.mean(per_channel_std))
    min_channel_std = float(np.min(per_channel_std))
    low_signal_frac = float((image < p1).mean())
    high_signal_frac = float((image > p99).mean())
    structure_score = float(np.mean(per_channel_var))

    return {
        "global_std": global_std,
        "min_channel_std": min_channel_std,
        "low_signal_frac": low_signal_frac,
        "high_signal_frac": high_signal_frac,
        "structure_score": structure_score
    }


def process_row(args):
    row, channel_stats = args
    plate, well, site, mito, agp, rna, er, dna, moa = row
    idx = f"{plate}_{well}_{site}"
    paths = [mito, agp, rna, er, dna]

    image = np.stack(
        [tiff.imread(p).astype(np.float32) for p in paths],
        axis=0
    )

    # QC filtering (note: thresholds somewhat arbitrarily chosen)
    qc = image_qc(image)

    if qc["min_channel_std"] < 0.01:
        print(f"Skipping dead/low-signal channel image: {idx}")
        return None

    if qc["low_signal_frac"] > 0.995:
        print(f"Skipping near-empty image: {idx}")
        return None

    if qc["high_signal_frac"] > 0.05:
        print(f"Skipping saturated image: {idx}")
        return None

    if qc["structure_score"] < 0.05:
        return None

    image = normalize(image, channel_stats)
    save_path = OUT_DIR / f"{idx}.pt"

    payload = { # save key metadata for self-contained debugging
        "image": torch.from_numpy(image),
        "plate": plate,
        "well": well,
        "site": site,
        "moa": moa
    }

    # verification check
    if "moa" not in payload:
        raise ValueError(f"Missing moa in payload for idx={idx}")

    torch.save(payload, save_path)

    return idx


def compute_channel_stats(rows):
    channel_sums = np.zeros(5, dtype=np.float64)
    channel_sq   = np.zeros(5, dtype=np.float64)
    n_pixels     = np.zeros(5, dtype=np.float64)
    print("Computing dataset normalization stats...")

    for i, row in enumerate(rows):
        paths = [
            row["mito_img_path"],
            row["agp_img_path"],
            row["rna_img_path"],
            row["er_img_path"],
            row["dna_img_path"],
        ]

        image = np.stack(
            [tiff.imread(p).astype(np.float32) for p in paths],
            axis=0
        )

        image = np.log1p(image)

        for c in range(5):
            x = image[c]
            channel_sums[c] += x.sum()
            channel_sq[c] += np.square(x).sum()
            n_pixels[c] += x.size

        if i % 1000 == 0:
            print(f"{i}/{len(rows)}")

    means = channel_sums / n_pixels
    vars_ = channel_sq / n_pixels - means**2
    stds = np.sqrt(vars_)
    stats = []

    for c in range(5):
        stats.append({"mean": means[c],"std": stds[c]})

    print("Means:", means)
    print("Stds:", stds)

    return stats


def main():
    """
    Do dataset-wide normalization
    """

    metadata_path = os.path.join(
        os.environ["CP_OUTPUT_ROOT"],
        "data/processed/master_metadata.parquet"
    )

    df = pd.read_parquet(metadata_path)
    df = df.reset_index().rename(columns={"index": "row_id"})
    rows = df.to_dict("records")
    channel_stats = compute_channel_stats(rows)

    saved = 0
    kept_indices = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        worker_inputs = [
            (
                (
                    row["plate"],
                    row["well"],
                    row["site"],
                    row["mito_img_path"],
                    row["agp_img_path"],
                    row["rna_img_path"],
                    row["er_img_path"],
                    row["dna_img_path"],
                    row.get("moa", "unknown"),
                ),
                channel_stats
            )
            for row in rows
        ]

        for i, result in enumerate(executor.map(process_row, worker_inputs)):
            if result is not None:
                saved += 1
                kept_indices.append(result)

            if i % 100 == 0:
                print(f"checked {i}/{len(rows)} | saved={saved}")

    filtered_df = df[df["row_id"].isin(kept_indices)].copy()
    filtered_df["pt_path"] = filtered_df.apply(
        lambda r: str(OUT_DIR / f"{r['plate']}_{r['well']}_{r['site']}.pt"),
        axis=1
    )
    filtered_df = filtered_df.drop(columns=[
        "mito_img_path",
        "agp_img_path",
        "rna_img_path",
        "er_img_path",
        "dna_img_path",
    ], errors="ignore")

    filtered_metadata_path = os.path.join(
        os.environ["CP_OUTPUT_ROOT"],
        "data/processed/master_metadata_qc.parquet"
    )

    filtered_df.to_parquet(filtered_metadata_path, index=False)

    print(f"Saved QC metadata → {filtered_metadata_path}")
    print(f"Finished pre-processing. {saved} rows saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
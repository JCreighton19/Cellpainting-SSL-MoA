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


def normalize(image):
    normed = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[0]):
        x = np.log1p(image[c])
        p1 = np.percentile(x, 1)
        p99 = np.percentile(x, 99)
        normed[c] = (x - p1) / (p99 - p1 + 1e-6)
    return normed


def image_qc(image):
    # image: (C, H, W)

    flat = image.reshape(image.shape[0], -1)
    per_channel_std = np.std(flat, axis=1)

    global_std = float(np.mean(per_channel_std))
    min_channel_std = float(np.min(per_channel_std))
    low_signal_frac = float((image < np.percentile(image, 1)).mean())
    high_signal_frac = float((image > np.percentile(image, 99)).mean())

    def structure_score(image):
        # measures how much structure exists vs flat/noise
        flat = image.reshape(image.shape[0], -1)

        per_channel_std = np.std(flat, axis=1)
        per_channel_mean = np.mean(flat, axis=1)

        return np.mean(per_channel_std / (per_channel_mean + 1e-6))

    structure_score = structure_score(image)

    return {
        "global_std": global_std,
        "min_channel_std": min_channel_std,
        "low_signal_frac": low_signal_frac,
        "high_signal_frac": high_signal_frac,
        "structure_score": structure_score
    }


def process_row(row):
    idx = row["index"]

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

    # QC filtering (note: thresholds somewhat arbitrarily chosen)
    qc_image = np.log1p(image)
    qc = image_qc(qc_image)

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

    image = normalize(image)
    moa = row.get("moa", "unknown")
    save_path = OUT_DIR / f"{idx}.pt"

    payload = {
        "image": torch.from_numpy(image),
        "plate": row["plate"],
        "well": row["well"],
        "site": row["site"],
        "moa": moa
    }

    # verification check
    if "moa" not in payload:
        raise ValueError(f"Missing moa in payload for idx={idx}")

    torch.save(payload, save_path)

    return idx


def main():
    metadata_path = os.path.join(
        os.environ["CP_OUTPUT_ROOT"],
        "data/processed/master_metadata.parquet"
    )

    df = pd.read_parquet(metadata_path)
    rows = df.reset_index().to_dict("records")

    saved = 0
    kept_indices = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        for i, result in enumerate(executor.map(process_row, rows)):
            if result is not None:
                saved += 1
                kept_indices.append(result)

            if i % 100 == 0:
                print(f"checked {i}/{len(rows)} | saved={saved}")

    filtered_df = df.loc[kept_indices].copy()
    filtered_metadata_path = os.path.join(
        os.environ["CP_OUTPUT_ROOT"],
        "data/processed/master_metadata_qc.parquet"
    )

    filtered_df.to_parquet(filtered_metadata_path, index=False)

    print(f"Saved QC metadata → {filtered_metadata_path}")
    print(f"Finished pre-processing. {saved} rows saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
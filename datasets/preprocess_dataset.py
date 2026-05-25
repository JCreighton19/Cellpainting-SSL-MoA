import os
import torch
import tifffile as tiff
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

OUT_DIR = Path("/scratch/creighton.jo/cellpainting/processed/tiles")
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
    mean = image.mean()
    std = image.std()
    near_zero = (image < 0.02).mean()

    return {
        "mean": float(mean),
        "std": float(std),
        "near_zero": float(near_zero)
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

    # QC filtering
    qc = image_qc(image)

    if qc["std"] < 0.01: # note: thresholds arbitrarily chosen
        print(f"Skipping low-std image: {idx}")
        return None

    if qc["near_zero"] > 0.98:
        print(f"Skipping near-empty image: {idx}")
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

    with ProcessPoolExecutor(max_workers=8) as executor:
        for i, result in enumerate(executor.map(process_row, rows)):
            if result is not None:
                saved += 1
            if i % 100 == 0:
                print(f"checked {i}/{len(rows)} | saved={saved}")

    print(f"Finished pre-processing. {len(rows)} rows saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
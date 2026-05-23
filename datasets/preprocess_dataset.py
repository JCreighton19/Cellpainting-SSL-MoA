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

def process_row(row):

    idx = row["index"]
    paths = [
        row.mito_img_path,
        row.agp_img_path,
        row.rna_img_path,
        row.er_img_path,
        row.dna_img_path,
    ]

    image = np.stack(
        [tiff.imread(p).astype(np.float32) for p in paths],
        axis=0
    )

    image = normalize(image)
    moa = row.moa if pd.notna(row.moa) else "unknown"
    save_path = OUT_DIR / f"{idx}.pt"

    torch.save({
        "image": torch.from_numpy(image),
        "plate": row.plate,
        "well": row.well,
        "site": row.site,
        "moa": moa
    }, save_path)

    return idx


def main():
    metadata_path = os.path.join(
        os.environ["CP_OUTPUT_ROOT"],
        "data/processed/master_metadata.parquet"
    )

    df = pd.read_parquet(metadata_path)
    rows = df.reset_index().to_dict("records")

    with ProcessPoolExecutor(max_workers=8) as executor:
        for i, _ in enumerate(executor.map(process_row, rows)):
            if i % 100 == 0:
                print(f"processed {i}/{len(rows)}")


if __name__ == "__main__":
    main()
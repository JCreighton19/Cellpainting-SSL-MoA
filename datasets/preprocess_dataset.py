import os
import torch
import tifffile as tiff
import numpy as np
import pandas as pd
from pathlib import Path

metadata_path = os.path.join(os.environ["CP_OUTPUT_ROOT"], "data/processed/master_metadata.parquet")
data_root = os.environ["CP_DATA_ROOT"]

out_dir = Path("/scratch/creighton.jo/cellpainting/processed_tiles")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(metadata_path)

def normalize(image):
    normed = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[0]):
        x = np.log1p(image[c])
        p1 = np.percentile(x, 1)
        p99 = np.percentile(x, 99)
        normed[c] = (x - p1) / (p99 - p1 + 1e-6)
    return normed

for i, row in df.iterrows():
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

    image = normalize(image)

    save_path = out_dir / f"{i}.pt"
    torch.save(torch.from_numpy(image), save_path)

    if i % 100 == 0:
        print(f"processed {i}/{len(df)}")
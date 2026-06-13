import os
import torch
import tifffile as tiff
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.filters import threshold_otsu
from concurrent.futures import ThreadPoolExecutor

OUT_DIR = Path(os.path.join(
    os.environ["CP_OUTPUT_ROOT"],
    "data/tiles_qc"
))
OUT_DIR.mkdir(parents=True, exist_ok=True)


# PAPER-FAITHFUL PREPROCESSING
def preprocess_image(image, eps=1e-6):
    image = image.astype(np.float32)
    out = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[0]):
        ch = image[c]
        lo = np.percentile(ch, 0.01)
        hi = np.percentile(ch, 99.9)
        ch = np.clip(ch, lo, hi)
        ch = (ch - lo) / (hi - lo + eps)
        out[c] = ch

    return out


def compute_otsu_threshold(dna_img):
    if np.std(dna_img) < 1e-6:
        return 0.0
    t = threshold_otsu(dna_img)
    return t


def process_row(args):
    row, _ = args
    plate, well, site, mito, agp, rna, er, dna, moa = row
    idx = f"{plate}_{well}_{site}"
    paths = [mito, agp, rna, er, dna]
    image = np.stack(
        [tiff.imread(p).astype(np.float32) for p in paths],
        axis=0
    )
    image = preprocess_image(image)
    otsu_threshold = compute_otsu_threshold(image[4])

    # Always keep image (no QC rejection)
    save_path = OUT_DIR / f"{idx}.pt"
    payload = {
        "image": torch.from_numpy(image),
        "otsu_threshold":otsu_threshold,
        "plate": plate,
        "well": well,
        "site": site,
        "moa": moa
    }

    torch.save(payload, save_path)

    return idx


def main():
    metadata_path = os.path.join(
        os.environ["CP_OUTPUT_ROOT"],
        "data/processed/master_metadata.parquet"
    )

    df = pd.read_parquet(metadata_path)
    rows = df.to_dict("records")
    with ThreadPoolExecutor(max_workers=16) as executor:
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
                None
            )
            for row in rows
        ]

        for i, result in enumerate(executor.map(process_row, worker_inputs)):
            if i % 100 == 0:
                print(f"processed {i}/{len(rows)}")

    filtered_df = df.copy()
    filtered_df["pt_path"] = (
        OUT_DIR.as_posix() + "/" +
        filtered_df["plate"].astype(str) + "_" +
        filtered_df["well"].astype(str) + "_" +
        filtered_df["site"].astype(str) + ".pt"
    )

    filtered_metadata_path = os.path.join(
        os.environ["CP_OUTPUT_ROOT"],
        "data/processed/master_metadata_qc.parquet"
    )

    filtered_df.to_parquet(filtered_metadata_path, index=False)
    print(f"Saved metadata → {filtered_metadata_path}")
    print(f"Finished preprocessing.")

if __name__ == "__main__":
    main()
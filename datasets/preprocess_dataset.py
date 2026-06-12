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


def preprocess_image(image, eps=1e-6):
    """
    Paper-faithful preprocessing (paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11811211):
    - per-channel percentile clipping (global behavior approximated per image)
    - scaling to [0, 1]
    """
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



def compute_otsu_mask(dna_img):
    if np.std(dna_img) < 1e-6:
        return np.zeros_like(dna_img, dtype=bool)
    t = threshold_otsu(dna_img)
    return dna_img > t


def image_qc(image):
    # image: (C, H, W)

    # compute std on RAW scale (pre-log stability check)
    per_channel_std = np.std(image, axis=(1, 2))
    raw = image  # keep reference in raw space for thresholding
    p1, p99 = np.percentile(raw, [1, 99])
    high_signal_frac = float((raw > p99).mean())

    global_std = float(np.mean(per_channel_std))
    min_channel_std = float(np.min(per_channel_std))
    dna_img = image[4]
    otsu_mask = compute_otsu_mask(dna_img)
    otsu_foreground_frac = float(otsu_mask.mean())

    return {
        "global_std": global_std,
        "min_channel_std": min_channel_std,
        "otsu_foreground_frac": otsu_foreground_frac,
        "high_signal_frac": high_signal_frac,
        "otsu_mask": otsu_mask
    }


def process_row(args):
    row, _ = args  # channel_stats removed
    plate, well, site, mito, agp, rna, er, dna, moa = row
    idx = f"{plate}_{well}_{site}"
    paths = [mito, agp, rna, er, dna]

    image = np.stack(
        [tiff.imread(p).astype(np.float32) for p in paths],
        axis=0
    )
    qc = image_qc(image)

    if qc["min_channel_std"] < 1e-2:
        print(f"Skipping dead/low-signal channel image: {idx}")
        return None

    if qc.get("otsu_foreground_frac", 0.0) < 0.03:
        print(f"Skipping near-empty image (Otsu): {idx}")
        return None

    if qc["high_signal_frac"] > 0.05:
        print(f"Skipping saturated image: {idx}")
        return None

    image = preprocess_image(image)
    otsu_mask = qc["otsu_mask"]
    save_path = OUT_DIR / f"{idx}.pt"

    payload = {
        "image": torch.from_numpy(image),
        "otsu_mask": torch.from_numpy(otsu_mask.astype(np.bool_)),
        "plate": plate,
        "well": well,
        "site": site,
        "moa": moa
    }

    torch.save(payload, save_path)
    sums = image.sum(axis=(1, 2))
    sq   = (image ** 2).sum(axis=(1, 2))
    n    = float(image.shape[1] * image.shape[2])
    return idx, (sums, sq, n)



def main():
    """
    Do dataset-wide normalization
    """
    metadata_path = os.path.join(
        os.environ["CP_OUTPUT_ROOT"],
        "data/processed/master_metadata.parquet"
    )

    df = pd.read_parquet(metadata_path)
    df["row_id"] = (
            df["plate"].astype(str) + "_" +
            df["well"].astype(str) + "_" +
            df["site"].astype(str)
    )
    assert df["row_id"].is_unique, "row_id is not unique — indexing bug risk"

    rows = df.to_dict("records")

    saved = 0
    kept_indices = []
    ch_sums = np.zeros(5, dtype=np.float64)
    ch_sq   = np.zeros(5, dtype=np.float64)
    ch_n    = 0.0

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
            if result is not None:
                idx, (s, q, n) = result
                saved += 1
                kept_indices.append(idx)
                ch_sums += s
                ch_sq   += q
                ch_n    += n

            if i % 100 == 0:
                print(f"checked {i}/{len(rows)} | saved={saved}")

    print("df rows:", len(df))
    print("kept_indices:", len(kept_indices))
    print("unique df row_id example:", df["row_id"].head())
    print("kept_indices example:", kept_indices[:5])
    print("matches:", df["row_id"].isin(kept_indices).sum())

    means = ch_sums / ch_n
    stds  = np.sqrt(np.maximum(ch_sq / ch_n - means**2, 0))
    print(f"Global channel means: {means.round(4)}")
    print(f"Global channel stds:  {stds.round(4)}")

    stats_path = os.path.join(os.environ["CP_OUTPUT_ROOT"], "data/processed/global_channel_stats.npy")
    np.save(stats_path, {"mean": means.astype(np.float32), "std": stds.astype(np.float32)})
    print(f"Saved global channel stats → {stats_path}")

    mean_t = torch.from_numpy(means.astype(np.float32)).view(5, 1, 1)
    std_t  = torch.from_numpy(stds.astype(np.float32)).view(5, 1, 1)
    print(f"Applying global z-score to {len(kept_indices)} files...")

    def _apply_zscore(idx):
        path = OUT_DIR / f"{idx}.pt"
        payload = torch.load(path, weights_only=False)
        payload["image"] = (payload["image"] - mean_t) / (std_t + 1e-6)
        torch.save(payload, path)

    with ThreadPoolExecutor(max_workers=16) as executor:
        for _ in executor.map(_apply_zscore, kept_indices):
            pass
    print("Z-score normalization complete.")

    filtered_df = df[df["row_id"].isin(kept_indices)].copy()
    filtered_df["pt_path"] = (
            OUT_DIR.as_posix() + "/" +
            filtered_df["plate"].astype(str) + "_" +
            filtered_df["well"].astype(str) + "_" +
            filtered_df["site"].astype(str) + ".pt"
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
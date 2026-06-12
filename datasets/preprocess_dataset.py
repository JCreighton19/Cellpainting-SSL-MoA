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


def normalize(image, channel_stats, eps=1e-6):
    means = np.array([s["mean"] for s in channel_stats], dtype=np.float32)[:, None, None]
    stds  = np.array([s["std"]  for s in channel_stats], dtype=np.float32)[:, None, None]
    return ((image - means) / (stds + eps)).astype(np.float32)


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
    row, channel_stats = args
    plate, well, site, mito, agp, rna, er, dna, moa = row
    idx = f"{plate}_{well}_{site}"
    paths = [mito, agp, rna, er, dna]

    image = np.stack(
        [tiff.imread(p).astype(np.float32) for p in paths],
        axis=0
    )
    qc = image_qc(image)  # RAW ONLY QC
    image_log = np.log1p(image)  # LOG ONLY FOR MODEL

    if qc["min_channel_std"] < 1e-2:
        print(f"Skipping dead/low-signal channel image: {idx}")
        return None

    if qc.get("otsu_foreground_frac", 0.0) < 0.03:
        print(f"Skipping near-empty image (Otsu): {idx}")
        return None

    if qc["high_signal_frac"] > 0.05:
        print(f"Skipping saturated image: {idx}")
        return None

    image = normalize(image_log, channel_stats)
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

    # verification check
    if "moa" not in payload:
        raise ValueError(f"Missing moa in payload for idx={idx}")

    torch.save(payload, save_path)

    return idx


def _stats_worker(paths):
    image = np.stack(
        [tiff.imread(p).astype(np.float32) for p in paths],
        axis=0
    )
    image = np.log1p(image)
    return (
        image.sum(axis=(1, 2)),
        np.square(image).sum(axis=(1, 2)),
        image.shape[1] * image.shape[2],
    )


def compute_per_plate_channel_stats(rows):
    from collections import defaultdict
    plates = defaultdict(list)
    for row in rows:
        plates[row["plate"]].append([
            row["mito_img_path"], row["agp_img_path"], row["rna_img_path"],
            row["er_img_path"],   row["dna_img_path"]
        ])

    plate_stats = {}
    for plate, path_lists in plates.items():
        sums  = np.zeros(5, dtype=np.float64)
        sq    = np.zeros(5, dtype=np.float64)
        n     = np.zeros(5, dtype=np.float64)
        total = len(path_lists)
        print(f"Computing stats for plate {plate} ({total} images)...")
        with ThreadPoolExecutor(max_workers=8) as executor:
            for s, q, pn in executor.map(_stats_worker, path_lists):
                sums += s
                sq   += q
                n    += pn
        n_safe = np.maximum(n, 1e-6)
        means = sums / n_safe
        stds = np.sqrt(np.maximum(sq / n_safe - means ** 2, 0))
        plate_stats[plate] = [{"mean": float(means[c]), "std": float(stds[c])} for c in range(5)]
        print(f"  means={means.round(4)}, stds={stds.round(4)}")

    return plate_stats


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

    # Create cache of channel stats
    stats_path = os.path.join(os.environ["CP_OUTPUT_ROOT"], "data/processed/channel_stats.npy")
    if os.path.exists(stats_path):
        plate_stats = np.load(stats_path, allow_pickle=True).item()
    else:
        plate_stats = compute_per_plate_channel_stats(rows)
        np.save(stats_path, plate_stats)

    saved = 0
    kept_indices = []

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
                plate_stats[row["plate"]]
            )
            for row in rows
        ]

        for i, result in enumerate(executor.map(process_row, worker_inputs)):
            if result is not None:
                saved += 1
                kept_indices.append(result)

            if i % 100 == 0:
                print(f"checked {i}/{len(rows)} | saved={saved}")

    print("df rows:", len(df))
    print("kept_indices:", len(kept_indices))
    print("unique df row_id example:", df["row_id"].head())
    print("kept_indices example:", kept_indices[:5])
    print("matches:", df["row_id"].isin(kept_indices).sum())

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
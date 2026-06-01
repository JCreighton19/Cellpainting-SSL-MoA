import pandas as pd
import numpy as np
import shutil
import os
from pathlib import Path

metadata = pd.read_parquet(
    "/scratch/creighton.jo/cellpainting/data/processed/master_metadata_qc.parquet"
)

OUTDIR = "/scratch/creighton.jo/cellpainting/debug_image_sample"
os.makedirs(OUTDIR, exist_ok=True)

CHANNELS = [
    "dna_img_path",
    "rna_img_path",
    "er_img_path",
    "agp_img_path",
    "mito_img_path"
]

# sample:
# 20 controls
# 20 treated
# across multiple plates

controls = metadata[
    (metadata["moa"] == "control vehicle")
].sample(
    20,
    random_state=42
)

treated = metadata[
    metadata["moa"] != "control vehicle"
].groupby(
    "moa",
    group_keys=False
).apply(
    lambda x: x.sample(
        min(1, len(x)),
        random_state=42
    )
)

sample_df = pd.concat([
    controls,
    treated.sample(
        min(30, len(treated)),
        random_state=42
    )
])

sample_df = sample_df.reset_index(drop=True)

records = []

for idx,row in sample_df.iterrows():

    sample_folder = Path(
        OUTDIR,
        f"sample_{idx}"
    )

    sample_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    rec = {}

    for col in CHANNELS:

        src = row[col]

        if os.path.exists(src):

            dst = sample_folder / Path(src).name

            shutil.copy2(
                src,
                dst
            )

            rec[col] = str(dst)

    rec["moa"] = row["moa"]
    rec["plate"] = row["plate"]
    rec["well"] = row["well"]

    records.append(rec)

pd.DataFrame(records).to_csv(
    f"{OUTDIR}/metadata.csv",
    index=False
)

print("saved to:", OUTDIR)
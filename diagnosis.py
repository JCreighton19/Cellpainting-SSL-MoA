from collections import defaultdict
import re
from pathlib import Path
import pandas as pd

SCRATCH_ROOT = Path("/scratch/creighton.jo/cellpainting")
IMAGE_ROOT = SCRATCH_ROOT / "data/raw/images"

# -----------------------
# 1. IMAGE SITES
# -----------------------
site_counts = defaultdict(set)

for p in IMAGE_ROOT.rglob("*.tiff"):
    m = re.search(r"(r\d{2}c\d{2})", p.name.lower())
    s = re.search(r"f(\d{2})p\d{2}", p.name.lower())

    if m and s:
        well = m.group(1)
        site = int(s.group(1))
        site_counts[well].add(site)

image_lengths = [len(v) for v in site_counts.values()]

print("\n=== IMAGE STATS ===")
print("min sites per well:", min(image_lengths))
print("max sites per well:", max(image_lengths))
print("unique patterns:", set(image_lengths))

# show example
first_well = list(site_counts.keys())[0]
print("\nexample well (images):", first_well)
print("sites:", sorted(site_counts[first_well]))

# -----------------------
# 2. LOAD DATA SITES
# -----------------------
LOAD_PATH = SCRATCH_ROOT / "data/raw/load_data_csv/BR00116991/load_data.csv"
df = pd.read_csv(LOAD_PATH)

df.columns = [c.lower() for c in df.columns]

print("\n=== LOAD_DATA STATS ===")
print("min site:", df["metadata_site"].min() if "metadata_site" in df.columns else df["site"].min())
print("max site:", df["metadata_site"].max() if "metadata_site" in df.columns else df["site"].max())
print("unique sites:", sorted(df["metadata_site"].unique())[:20] if "metadata_site" in df.columns else sorted(df["site"].unique())[:20])

print("\nrows:", len(df))
print("unique wells:", df["metadata_well"].nunique() if "metadata_well" in df.columns else df["well"].nunique())
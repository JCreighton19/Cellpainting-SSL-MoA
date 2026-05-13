# datasets/build_metadata_table.py

from pathlib import Path
import pandas as pd
import re

# CONFIG
PLATE = "BR00116991"
IMAGE_ROOT = Path(f"data/raw/images/{PLATE}")
LOAD_DATA_PATH = Path(f"data/raw/load_data_csv/{PLATE}/load_data.csv")
PLATEMAP_DIR = Path(f"data/raw/platemaps/{PLATE}")
COMPOUND_METADATA_PATH = Path(f"data/raw/compound_metadata/{PLATE}/compound_metadata.tsv")
OUTPUT_PATH = Path("data/processed/master_metadata.parquet")

# WELL NORMALIZATION
def rc_to_a01(well: str):
    """
    Convert r01c01 → A01
    """
    if well is None:
        return None
    match = re.match(r"r(\d{2})c(\d{2})", well.lower())
    if not match:
        return None

    row = int(match.group(1))
    col = int(match.group(2))
    row_letter = chr(ord("A") + row - 1)
    return f"{row_letter}{col:02d}"


def extract_rc_from_filename(name: str):
    """
    Extract r01c01 from filenames like:
    r01c01f01p01-ch1sk1fk1fl1.tiff
    """
    match = re.search(r"(r\d{2}c\d{2})", name.lower())
    return match.group(1) if match else None


# LOAD IMAGING INDEX
def load_imaging_index(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    df = df.rename(columns={
        "metadata_plate": "plate",
        "metadata_well": "well",
        "metadata_site": "site",
    })

    df["well"] = df["well"].apply(rc_to_a01)

    print(f"[load_data] rows={len(df)} wells={df['well'].nunique()}")
    return df


# LOAD PLATEMAPS
def load_plate_layouts(layout_dir: Path) -> pd.DataFrame:
    dfs = []
    for fp in layout_dir.glob("*.txt"):
        df = pd.read_csv(fp, sep="\t")
        # normalize column
        df = df.rename(columns={"well_position": "well"})
        # convert to A01
        df["well"] = df["well"].apply(rc_to_a01)
        df["plate"] = fp.stem
        dfs.append(df)

    if len(dfs) == 0:
        raise ValueError(f"No platemap files found in {layout_dir}")

    layout_df = pd.concat(dfs, ignore_index=True)
    print(f"[platemap] rows={len(layout_df)} wells={layout_df['well'].nunique()}")
    return layout_df


# COMPOUND METADATA
def load_compound_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    print(f"[compound] rows={len(df)}")
    return df


# IMAGE INDEX
def build_image_index(image_root: Path) -> pd.DataFrame:
    records = []
    for path in image_root.rglob("*.tiff"):
        rc = extract_rc_from_filename(path.name)
        if rc is None:
            continue

        well = rc_to_a01(rc)
        if well is None:
            continue
        records.append((well, str(path)))

    df = pd.DataFrame(records, columns=["well", "image_path"])
    df = df.groupby("well")["image_path"].apply(list).reset_index()
    print(f"[images] wells={df['well'].nunique()} images={len(records)}")

    return df


# MASTER MERGE
def build_master_metadata(load_df, layout_df, compound_df):
    merged = load_df.merge(layout_df, on=["plate", "well"], how="left")
    print(f"[merge] after platemap: {merged.shape}")
    missing = merged["broad_sample"].isna().mean() if "broad_sample" in merged else 1.0
    print(f"[merge] missing broad_sample: {missing:.3f}")

    if compound_df is not None and "broad_sample" in merged.columns:
        merged = merged.merge(compound_df, on="broad_sample", how="left")
    print(f"[merge] after compound: {merged.shape}")

    return merged


# IMAGE ATTACHMENT
def attach_image_paths(df, image_root: Path):
    img_df = build_image_index(image_root)
    merged = df.merge(img_df, on="well", how="left")
    missing = merged["image_paths"].isna().mean()
    print(f"[images] missing fraction: {missing:.3f}")

    return merged.rename(columns={"image_path": "image_paths"})


# VALIDATION
def validate(df):

    print("\n========== VALIDATION ==========\n")
    print("rows:", len(df))
    print("wells:", df["well"].nunique())

    if "broad_sample" in df.columns:
        miss = df["broad_sample"].isna().mean()
        print("missing broad_sample:", miss)
        if miss > 0.5:
            raise ValueError(
                "Platemap join failed: >50% missing broad_sample. "
                "Check well format consistency between load_data and platemap."
            )

    if "image_paths" in df.columns:
        print("missing images:", df["image_paths"].isna().mean())

    print("\npert_type:")
    if "pert_type" in df.columns:
        print(df["pert_type"].value_counts())


# MAIN
def main():

    print("\nLoading imaging index...")
    load_df = load_imaging_index(LOAD_DATA_PATH)

    print("\nLoading platemaps...")
    layout_df = load_plate_layouts(PLATEMAP_DIR)

    print("\nLoading compound metadata...")
    compound_df = load_compound_metadata(COMPOUND_METADATA_PATH)

    print("\nBuilding master table...")
    master = build_master_metadata(load_df, layout_df, compound_df)

    print("\nAttaching images...")
    master = attach_image_paths(master, IMAGE_ROOT)

    validate(master)
    print("\nSaving...")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
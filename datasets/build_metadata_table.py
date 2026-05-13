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

# EXTRACT WELL NAMES
def extract_well(name: str):
    """
    Extracts well like r01c01 from filenames like:
    r01c01f01p01-ch1sk1fk1fl1.tiff
    """
    match = re.search(r"(r\d{2}c\d{2})", name.lower())
    return match.group(1) if match else None


# LOAD IMAGING INDEX (CORE TABLE)
def load_imaging_index(load_data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(load_data_path)
    df.columns = [c.lower() for c in df.columns]

    rename_map = {
        "metadata_plate": "plate",
        "metadata_well": "well",
        "metadata_site": "site",
    }

    df = df.rename(columns=rename_map)

    if "plate" not in df.columns or "well" not in df.columns:
        raise ValueError(
            f"Missing plate/well after renaming. Columns are: {df.columns.tolist()}"
        )
    print(f"Loaded imaging index: {len(df)} rows")

    return df


# LOAD PLATEMAP
def load_plate_layouts(layout_dir: Path) -> pd.DataFrame:
    """
    Maps well → compound (broad_sample)
    """

    dfs = []
    for filepath in layout_dir.glob("*.txt"):
        df = pd.read_csv(filepath, sep="\t")
        df = df.rename(columns={
            "well_position": "well"
        })

        # plate id inferred from filename
        df["plate"] = filepath.stem
        dfs.append(df)

    layout_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded platemap rows: {len(layout_df)}")

    return layout_df


# LOAD COMPOUND METADATA
def load_compound_metadata(filepath: Path) -> pd.DataFrame:
    """
    Optional enrichment table (gene, SMILES, etc.)
    """

    df = pd.read_csv(filepath, sep="\t")
    print(f"Loaded compound metadata: {len(df)} rows")

    return df


# ATTACH IMAGE PATHS
def build_image_index(image_root: Path) -> pd.DataFrame:
    records = []
    for path in image_root.rglob("*"):
        if path.is_file():
            well = extract_well(path.name)
            if well:
                records.append((well, str(path)))

    df = pd.DataFrame(records, columns=["well", "image_path"])
    df = df.groupby("well")["image_path"].apply(list).reset_index()
    print(f"Indexed wells: {df['well'].nunique()}, images: {len(records)}")

    return df


# MASTER MERGE PIPELINE
def build_master_metadata(load_df, layout_df, compound_df):
    """
    Core joins:
    1. imaging index (load_data)
    2. platemap (well → compound)
    3. compound metadata enrichment
    """

    # 1. join imaging → platemap
    merged = load_df.merge(
        layout_df,
        how="left",
        on=["plate", "well"]
    )

    print(f"After platemap join: {merged.shape}")

    # 2. join compound metadata
    if compound_df is not None:
        if "broad_sample" in merged.columns and "broad_sample" in compound_df.columns:
            merged = merged.merge(
                compound_df,
                how="left",
                on="broad_sample"
            )

    print(f"After compound join: {merged.shape}")

    return merged


def attach_image_paths(load_df: pd.DataFrame, image_root: Path) -> pd.DataFrame:
    """
    Join precomputed image index
    """
    img_df = build_image_index(image_root)
    merged = load_df.merge(img_df, how="left", on="well")
    missing = merged["image_path"].isna().mean()
    print(f"Fraction missing images: {missing:.3f}")

    return merged.rename(columns={"image_path": "image_paths"})


# VALIDATION
def validate(df):

    print("\n================ VALIDATION ================\n")
    print("Rows:", len(df))
    print("Missing broad_sample:", df["broad_sample"].isna().mean())
    print("Unique compounds:", df["broad_sample"].nunique())
    print("Unique wells:", df["well"].nunique())
    print("Perturbation types:")
    if "pert_type" in df.columns:
        print(df["pert_type"].value_counts(dropna=False))
    print("\nMissing images:", df["image_paths"].isna().mean())


def main():
    print("\nLoading imaging index...")
    load_df = load_imaging_index(LOAD_DATA_PATH)

    print("\nLoading platemaps...")
    layout_df = load_plate_layouts(PLATEMAP_DIR)

    print("\nLoading compound metadata...")
    compound_df = load_compound_metadata(COMPOUND_METADATA_PATH)

    print("\nBuilding master table...")
    master = build_master_metadata(load_df, layout_df, compound_df)

    print("\nAttaching image paths...")
    master = attach_image_paths(master, IMAGE_ROOT)

    print("\nValidating...")
    validate(master)

    print("\nSaving parquet...")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
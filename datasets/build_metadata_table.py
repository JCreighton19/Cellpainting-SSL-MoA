# datasets/build_metadata_table.py

from pathlib import Path
import pandas as pd
import re

# CONFIG
SCRATCH_ROOT = Path("/scratch/creighton.jo/cellpainting")
IMAGE_ROOT = SCRATCH_ROOT / "data/raw/images"
LOAD_DATA_ROOT = SCRATCH_ROOT / "data/raw/load_data_csv"
COMPOUND_METADATA_PATH = (SCRATCH_ROOT / "data/raw/JUMP-Target-1_compound_metadata.tsv")
PLATEMAP_PATH = (SCRATCH_ROOT / "data/raw/platemaps/JUMP-Target-1_compound_platemap.txt")
OUTPUT_PATH = (SCRATCH_ROOT / "data/processed/master_metadata.parquet")
PLATES = sorted([
    p.name for p in IMAGE_ROOT.iterdir()
    if p.is_dir()
])

# WELL NORMALIZATION
def rc_to_a01(well: str):
    match = re.match(r"r(\d{2})c(\d{2})", well.lower())
    if not match:
        return None
    row = int(match.group(1))
    col = int(match.group(2))
    return f"{chr(ord('A') + row - 1)}{col:02d}"


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

    df["well"] = df["well"].str.upper().str.strip()

    print(f"[load_data] rows={len(df)} wells={df['well'].nunique()}")
    return df


# LOAD PLATEMAPS
def load_plate_layouts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={"well_position": "well"})
    df["well"] = df["well"].str.upper().str.strip()
    print(f"[platemap] rows={len(df)} wells={df['well'].nunique()}")
    return df


# COMPOUND METADATA
def load_compound_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    print(f"[compound] rows={len(df)}")
    return df


# IMAGE INDEX
def build_image_index(image_root: Path, plate: str):
    records = []

    for path in image_root.rglob("*.tiff"):
        rc = extract_rc_from_filename(path.name)
        if rc is None:
            continue
        well = rc_to_a01(rc)
        site_match = re.search(r"f(\d{2})p\d{2}", path.name.lower())
        if not site_match:
            continue
        site = int(site_match.group(1))
        rel_path = str(path.resolve())
        records.append((plate, well, site, rel_path))

    df = pd.DataFrame(records, columns=["plate", "well", "site", "image_path"])
    df = df.groupby(["plate", "well"])["image_path"].apply(lambda x: sorted(x)).reset_index()
    df = df.rename(columns={"image_path": "image_paths"})

    return df


# MASTER MERGE
def build_master_metadata(load_df, layout_df, compound_df):
    merged = load_df.merge(layout_df, on="well", how="left")
    print(f"[merge] after platemap: {merged.shape}")
    print("dup (plate,well):", merged.duplicated(["plate", "well"]).sum())
    print("row growth factor:", len(merged) / len(load_df))

    missing = merged["broad_sample"].isna().mean() if "broad_sample" in merged else 1.0
    print(f"[merge] missing broad_sample: {missing:.3f}")

    if compound_df is not None and "broad_sample" in merged.columns:
        merged = merged.merge(compound_df, on="broad_sample", how="left")
    print(f"[merge] after compound: {merged.shape}")
    print("dup (plate,well):", merged.duplicated(["plate", "well"]).sum())
    print("row growth factor:", len(merged) / len(load_df))

    return merged

# IMAGE ATTACHMENT
def attach_image_paths(df, image_root: Path, plate: str):
    img_df = build_image_index(image_root, plate)

    # Inner join to keep only rows that actually have downloaded images
    merged = df.merge(img_df, on=["plate", "well"], how="inner")
    print(f"[images] final rows with images: {len(merged)}")
    print(f"[images] wells with images: {merged['well'].nunique()}")

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
    print("\nLoading platemap...")
    layout_df = load_plate_layouts(PLATEMAP_PATH)
    print("\nLoading compound metadata...")
    compound_df = load_compound_metadata(COMPOUND_METADATA_PATH)
    compound_df = compound_df.drop_duplicates("broad_sample")
    all_master = []

    for plate in PLATES:
        print(f"\n==============================")
        print(f"PROCESSING {plate}")
        print(f"==============================")
        load_data_path = (
            LOAD_DATA_ROOT / plate / "load_data.csv"
        )
        image_root = IMAGE_ROOT / plate

        print("\nLoading imaging index...")
        load_df = load_imaging_index(load_data_path)

        # Ensure plate column exists
        load_df["plate"] = plate
        load_df = load_df.groupby(["plate", "well"], as_index=False).first()
        print("load_df shape:", load_df.shape)

        print("\nBuilding master table...")
        master = build_master_metadata(
            load_df,
            layout_df,
            compound_df
        )

        print("\nAttaching images...")
        master = attach_image_paths(
            master,
            image_root,
            plate
        )

        # Confirm no duplicates and validate
        dup_count = master.duplicated(["plate", "well"]).sum()
        if dup_count > 0:
            raise ValueError(
                f"Found {dup_count} duplicate (plate, well) rows"
            )
        validate(master)
        all_master.append(master)

    print("\nConcatenating all plates...")
    master_df = pd.concat(all_master, ignore_index=True)
    print("final shape:", master_df.shape)

    # Fill missing metadata
    for col in ["broad_sample", "gene", "control_type"]:
        if col in master_df.columns:
            master_df[col] = master_df[col].fillna("unknown")
    print("\nSaving...")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
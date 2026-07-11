# datasets/build_metadata_table.py

import json
from pathlib import Path
import pandas as pd
import re

# CONFIG
SCRATCH_ROOT = Path("/scratch/creighton.jo/cellpainting")
IMAGE_ROOT = SCRATCH_ROOT / "data/raw/images"
LOAD_DATA_ROOT = SCRATCH_ROOT / "data/raw/load_data_csv"
COMPOUND_METADATA_PATH = (SCRATCH_ROOT / "data/raw/JUMP-Target-1_compound_metadata.tsv")
# Compound platemap design is NOT the same across every Cell Painting Gallery
# dataset (only cpg0000-jump-pilot uses the single "JUMP-Target-1" design) --
# see resolve_plate_layout() below, which resolves the right platemap file
# per plate via barcode_platemap.csv instead of one hardcoded global file.
PLATEMAP_ROOT = SCRATCH_ROOT / "data/raw/platemaps"
# Written by scripts/download_compound_plates.py: {experiment: dataset}. Images/
# load_data_csv paths don't carry dataset at all (see ACQUISITIONS below), so
# this is the only place that mapping is recorded.
EXPERIMENT_DATASET_MANIFEST_PATH = SCRATCH_ROOT / "data/raw/experiment_datasets.json"
OUTPUT_PATH = (SCRATCH_ROOT / "data/processed/master_metadata.parquet")
MOA_PATH = SCRATCH_ROOT / "data/raw/repo-drug-annotation-20200324.txt"

# Raw images/load_data are organized as {experiment}/{acquisition_id}/..., since
# the same physical plate barcode can be re-imaged under multiple experiments/
# timepoints (e.g. CPJUMP1 Day1/Day4/2Weeks re-measurements of the same plate).
# acquisition_id (e.g. "BR00117006__2020-11-03T19_45_39-Measurement1") is the
# Cell Painting Gallery's own acquisition folder name and is globally unique;
# the bare barcode alone is not.
ACQUISITIONS = sorted(
    (experiment_dir.name, acquisition_dir.name)
    for experiment_dir in IMAGE_ROOT.iterdir() if experiment_dir.is_dir()
    for acquisition_dir in experiment_dir.iterdir() if acquisition_dir.is_dir()
)
CHANNEL_MAP = {
    "ch1": "mito_img_path",
    "ch2": "agp_img_path",
    "ch3": "rna_img_path",
    "ch4": "er_img_path",
    "ch5": "dna_img_path",
}

# IDENTIFIER PARSING
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


def assert_filesystem_safe(name: str, label: str):
    """acquisition_id becomes the `plate` column, which flows straight into
    .pt filenames (preprocess_dataset.py) and webapp asset filenames
    (thumbnails/attention maps). Fail fast if it isn't a safe path component."""
    if not _SAFE_ID_RE.match(name):
        raise ValueError(
            f"{label} {name!r} contains characters unsafe for use in filenames/paths."
        )


def parse_acquisition_id(acquisition_id: str):
    """
    "BR00117006__2020-11-03T19_45_39-Measurement1" -> (barcode="BR00117006", measurement=1)
    """
    match = re.match(r"^(?P<barcode>[^_]+)__.*-Measurement(?P<measurement>\d+)$", acquisition_id)
    if not match:
        raise ValueError(
            f"Could not parse barcode/measurement from acquisition id: {acquisition_id!r}. "
            "Expected '{barcode}__{timestamp}-Measurement{N}'."
        )
    return match.group("barcode"), int(match.group("measurement"))


def parse_timepoint(experiment: str):
    """Best-effort, human-readable timepoint label parsed from the experiment
    folder name. Returns None (not a guess) when no explicit timepoint token
    is present, e.g. the baseline "2020_11_04_CPJUMP1" experiment."""
    match = re.search(r"TimepointDay(\d+)", experiment, re.IGNORECASE)
    if match:
        return f"Day{match.group(1)}"
    match = re.search(r"(\d+)\s*Weeks?TimePoint", experiment, re.IGNORECASE)
    if match:
        return f"{match.group(1)}Weeks"
    return None


_BR_PLATE_SUFFIX_RE = re.compile(r"^(?P<base>BR\d+)(?P<suffix>[A-Za-z])$")


def strip_reimaging_suffix(barcode: str) -> str:
    """
    BR00116991A -> BR00116991 (drops a trailing re-imaging suffix letter)
    BR00116991F -> BR00116991
    BR00116992D -> BR00116992
    Only handles the "BR{digits}{letter}" plate pattern; any other barcode
    (including a bare "BR00116991" with no suffix) is returned unchanged.
    """
    match = _BR_PLATE_SUFFIX_RE.match(barcode)
    return match.group("base") if match else barcode


def resolve_load_data_path(experiment: str, acquisition_id: str, barcode: str) -> Path:
    """
    Most Cell Painting Gallery experiments (e.g. the original cpg0000-jump-pilot
    "2020_11_04_CPJUMP1") store load_data.csv per acquisition folder:
        load_data_csv/{experiment}/{acquisition_id}/load_data.csv

    Some experiments -- e.g. "2020_12_08_CPJUMP1_Bleaching" -- instead store it
    keyed by the BASE plate barcode (re-imaging suffix letter stripped), with
    no experiment folder at all:
        load_data_csv/{base_barcode}/load_data.csv
    even though the matching images still live under
        images/{experiment}/{acquisition_id}/...

    e.g. acquisition_id="BR00116991A__2020-11-11T18_34_27-Measurement1" ->
    base_barcode="BR00116991" -> data/raw/load_data_csv/BR00116991/load_data.csv

    Tries the acquisition-path form first (preserves existing behavior for
    experiments laid out that way), then falls back to the base-barcode form.
    This only affects WHERE load_data.csv is read from -- the `plate` column
    written into the metadata table stays the full acquisition_id either way.
    """
    acquisition_path = LOAD_DATA_ROOT / experiment / acquisition_id / "load_data.csv"
    if acquisition_path.exists():
        print(f"[load_data] Resolved load_data.csv via acquisition path: {acquisition_path}")
        return acquisition_path

    base_barcode = strip_reimaging_suffix(barcode)
    fallback_path = LOAD_DATA_ROOT / base_barcode / "load_data.csv"
    if fallback_path.exists():
        print(f"[load_data] Resolved load_data.csv via base barcode fallback: {base_barcode}")
        return fallback_path

    # Neither location exists -- return the acquisition-path form so the
    # existing "not found" handling/messaging in main() stays unchanged.
    return acquisition_path


def _selfcheck_base_barcode_resolution():
    """Standalone sanity check (not a full test suite) for the base-barcode
    fallback path construction used by resolve_load_data_path() above."""
    acquisition_id = "BR00116991A__2020-11-11T18_34_27-Measurement1"
    barcode, _ = parse_acquisition_id(acquisition_id)
    base_barcode = strip_reimaging_suffix(barcode)
    expected = LOAD_DATA_ROOT / "BR00116991" / "load_data.csv"
    actual = LOAD_DATA_ROOT / base_barcode / "load_data.csv"
    assert actual == expected, f"base barcode resolution check failed: {actual} != {expected}"


_selfcheck_base_barcode_resolution()


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
    df = df.dropna(subset=["plate", "well", "site"])
    print(f"[load_data] rows={len(df)} wells={df['well'].nunique()}")
    return df


# LOAD PLATEMAPS
def load_plate_layouts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={"well_position": "well"})
    df["well"] = df["well"].str.upper().str.strip()
    print(f"[platemap] rows={len(df)} wells={df['well'].nunique()}")
    return df


def load_experiment_dataset_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Experiment-to-dataset manifest not found at {path}. Run "
            "scripts/download_compound_plates.py for at least one experiment "
            "first (it writes this file), or add entries manually as "
            '{"experiment_name": "dataset_name"}.'
        )
    return json.loads(path.read_text())


def load_barcode_platemap(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[barcode_platemap] rows={len(df)} path={path}")
    return df


def resolve_plate_layout(dataset: str, experiment: str, barcode: str,
                          barcode_map_cache: dict, layout_cache: dict) -> pd.DataFrame:
    """Per-plate compound platemap resolution: barcode_platemap.csv (Assay_
    Plate_Barcode -> Plate_Map_Name) picks the platemap design for THIS
    plate, then that design's content file is loaded. Replaces a single
    hardcoded platemap file, which was only ever correct because every
    cpg0000-jump-pilot plate happens to share the same "JUMP-Target-1"
    design -- a different dataset (e.g. cpg0016) has a different platemap
    per plate, not one shared file. Both caches are keyed to avoid re-reading
    the same file for every plate in an experiment."""
    barcode_map_key = (dataset, experiment)
    if barcode_map_key not in barcode_map_cache:
        barcode_map_cache[barcode_map_key] = load_barcode_platemap(
            PLATEMAP_ROOT / dataset / experiment / "barcode_platemap.csv"
        )
    barcode_map = barcode_map_cache[barcode_map_key]

    matches = barcode_map.loc[barcode_map["Assay_Plate_Barcode"] == barcode, "Plate_Map_Name"]
    if matches.empty:
        raise ValueError(
            f"Barcode {barcode!r} (experiment={experiment}, dataset={dataset}) is not "
            f"present in {PLATEMAP_ROOT / dataset / experiment / 'barcode_platemap.csv'}. "
            "Every plate on disk is expected to have been selected by "
            "scripts/download_compound_plates.py, which only downloads plates already "
            "confirmed present in that file."
        )
    plate_map_name = matches.iloc[0]

    layout_key = (dataset, experiment, plate_map_name)
    if layout_key not in layout_cache:
        layout_cache[layout_key] = load_plate_layouts(
            PLATEMAP_ROOT / dataset / experiment / "platemap" / f"{plate_map_name}.txt"
        )
    return layout_cache[layout_key], plate_map_name


# COMPOUND METADATA
def load_compound_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    print(f"[compound] rows={len(df)}")
    return df

def load_moa(path: Path):
    df = pd.read_csv(
        path,
        sep="\t",
        skiprows=9
    )
    df.columns = [c.lower() for c in df.columns]
    print(f"[moa] rows={len(df)} cols={df.columns.tolist()}")
    return df

# IMAGE INDEX
def build_image_index(image_root: Path, plate: str):
    records = {}

    for path in image_root.rglob("*.tiff"):
        name = path.name.lower()
        rc = extract_rc_from_filename(name)
        if rc is None:
            continue

        well = rc_to_a01(rc)
        site_match = re.search(r"f(\d{2})p\d{2}", name)
        if not site_match:
            continue

        site = int(site_match.group(1))
        ch_match = re.search(r"ch0?([1-5])", name)
        if not ch_match:
            continue

        ch = f"ch{ch_match.group(1)}"
        if ch not in CHANNEL_MAP:
            continue

        key = (plate, well, site)
        if key not in records:
            records[key] = {
                "plate": plate,
                "well": well,
                "site": site,
                "dna_img_path": None,
                "agp_img_path": None,
                "mito_img_path": None,
                "er_img_path": None,
                "rna_img_path": None,
            }

        records[key][CHANNEL_MAP[ch]] = str(path.resolve())

    required_cols = [
        "dna_img_path",
        "agp_img_path",
        "mito_img_path",
        "er_img_path",
        "rna_img_path",
    ]

    for col in required_cols:
        for v in records.values():
            v.setdefault(col, None)

    df = pd.DataFrame(records.values())
    df = df.dropna(subset=[
        "dna_img_path",
        "agp_img_path",
        "mito_img_path",
        "er_img_path",
        "rna_img_path",
    ])

    return df


# MASTER MERGE
def build_master_metadata(load_df, layout_df, compound_df):
    merged = load_df.merge(layout_df, on="well", how="left")
    # Ensure no accidental image columns exist in metadata stage
    image_cols = [
        "url_origdna",
        "url_origagp",
        "url_origmito",
        "url_origer",
        "url_origrna",
    ]
    merged = merged.drop(columns=[c for c in image_cols if c in merged.columns], errors="ignore")

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

    # Remove any existing image columns before merge
    image_cols = [
        "url_origdna",
        "url_origagp",
        "url_origmito",
        "url_origer",
        "url_origrna",
    ]

    df = df.drop(columns=[c for c in image_cols if c in df.columns], errors="ignore")
    merged = df.merge(img_df, on=["plate", "well", "site"], how="inner")
    print(f"[images] final rows with images: {len(merged)}")
    print(f"[images] wells with images: {merged['well'].nunique()}")

    return merged


# VALIDATION
def validate(df):

    print("\n========== VALIDATION ==========\n")
    print("rows:", len(df))
    print("images:", len(df))
    print("wells:", df["well"].nunique())
    print("sites:", df["site"].nunique())

    if "broad_sample" in df.columns:
        miss = df["broad_sample"].isna().mean()
        print("missing broad_sample:", miss)
        if miss > 0.5:
            raise ValueError(
                "Platemap join failed: >50% missing broad_sample. "
                "Check well format consistency between load_data and platemap."
            )

    missing_cols = [
        "mito_img_path",
        "agp_img_path",
        "rna_img_path",
        "er_img_path",
        "dna_img_path"
    ]

    for col in missing_cols:
        if col not in df.columns:
            print(f"missing {col}: COLUMN NOT PRESENT")
        else:
            print(f"missing {col}:", df[col].isna().mean())

    print("\npert_type:")
    if "pert_type" in df.columns:
        print(df["pert_type"].value_counts())


# MAIN
def main():
    print("\nLoading experiment-to-dataset manifest...")
    experiment_dataset = load_experiment_dataset_manifest(EXPERIMENT_DATASET_MANIFEST_PATH)
    barcode_map_cache = {}
    layout_cache = {}
    print("\nLoading compound metadata...")
    compound_df = load_compound_metadata(COMPOUND_METADATA_PATH)
    compound_df = compound_df.drop_duplicates("broad_sample")
    moa_df = load_moa(MOA_PATH)
    compound_df = compound_df.merge(
        moa_df[["pert_iname", "moa"]],
        on="pert_iname",
        how="left"
    )
    print(compound_df.columns)
    print(compound_df["moa"].value_counts().head())
    print("missing MOA:", compound_df["moa"].isna().mean())
    all_master = []
    skipped_missing_load_data = []
    skipped_missing_manifest = []
    skipped_missing_platemap = []

    for experiment, acquisition_id in ACQUISITIONS:
        print(f"\n==============================")
        print(f"PROCESSING {experiment} / {acquisition_id}")
        print(f"==============================")
        assert_filesystem_safe(experiment, "experiment")
        assert_filesystem_safe(acquisition_id, "acquisition_id")
        barcode, measurement = parse_acquisition_id(acquisition_id)
        timepoint = parse_timepoint(experiment)

        # Both checks below are about INCOMPLETE download state, not data
        # corruption: the manifest and platemap files are written per-plate by
        # data/download_metadata.slurm (or scripts/download_compound_plates.py),
        # separately from images (data/download_images.slurm or aws s3 sync
        # directly) -- so it's expected that images can exist locally for an
        # experiment before its metadata-side download has completed. Skip and
        # report rather than crash the whole run, same as the missing
        # load_data.csv case above.
        if experiment not in experiment_dataset:
            print(f"WARNING: experiment {experiment!r} has no entry in "
                  f"{EXPERIMENT_DATASET_MANIFEST_PATH} -- skipping "
                  f"{experiment}/{acquisition_id}. The manifest is written "
                  "per-plate by data/download_metadata.slurm, so this means it "
                  "hasn't successfully processed ANY plate from this experiment "
                  "yet -- rerun it.")
            skipped_missing_manifest.append((experiment, acquisition_id))
            continue
        dataset = experiment_dataset[experiment]

        try:
            layout_df, plate_map_name = resolve_plate_layout(
                dataset, experiment, barcode, barcode_map_cache, layout_cache
            )
        except (ValueError, FileNotFoundError) as e:
            print(f"WARNING: could not resolve platemap for {experiment}/{acquisition_id} "
                  f"({e}) -- skipping. Usually means barcode_platemap.csv or the "
                  "platemap content file for this experiment/plate hasn't been "
                  "downloaded yet.")
            skipped_missing_platemap.append((experiment, acquisition_id))
            continue

        load_data_path = resolve_load_data_path(experiment, acquisition_id, barcode)
        image_root = IMAGE_ROOT / experiment / acquisition_id

        if not load_data_path.exists():
            # ACQUISITIONS is discovered purely from what's under IMAGE_ROOT, so
            # it assumes every acquisition with images also has a load_data.csv.
            # That's true for anything scripts/download_compound_plates.py or
            # data/download_metadata.slurm fetched (they always fetch both) --
            # it's NOT true for plates downloaded manually before that pipeline
            # existed (images-only). Skip rather than crash the whole run; see
            # the printed summary at the end for what got skipped.
            print(f"WARNING: {load_data_path} not found -- skipping "
                  f"{experiment}/{acquisition_id} (images present, load_data.csv "
                  "missing; likely an images-only manual download predating the "
                  "current pipeline).")
            skipped_missing_load_data.append((experiment, acquisition_id))
            continue

        print("\nLoading imaging index...")
        load_df = load_imaging_index(load_data_path)
        print("load_df shape:", load_df.shape)

        # `plate` is the globally-unique acquisition ID, not the bare barcode
        # load_imaging_index parsed out of load_data.csv's Metadata_Plate column
        # (that value is overwritten here). experiment/timepoint/measurement/
        # dataset/plate_map_name are explicit metadata-only columns; they never
        # enter file/dir naming.
        load_df["plate"] = acquisition_id
        load_df["barcode"] = barcode
        load_df["experiment"] = experiment
        load_df["timepoint"] = timepoint
        load_df["measurement"] = measurement
        load_df["dataset"] = dataset
        load_df["plate_map_name"] = plate_map_name

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
            acquisition_id
        )

        validate(master)
        print(master.groupby(["plate", "well", "site"]).size().describe())
        all_master.append(master)

    print("\nConcatenating all plates...")
    master_df = pd.concat(all_master, ignore_index=True)

    # Duplicate check must run on the fully concatenated table: two
    # experiments processed independently above can't detect a collision
    # against each other, only a global check after concat can.
    dup_count = master_df.duplicated(["plate", "well", "site"]).sum()
    if dup_count > 0:
        raise ValueError(
            f"Found {dup_count} duplicate (plate, well, site) rows across "
            "experiments. plate is expected to be the globally-unique "
            "acquisition ID -- check ACQUISITIONS discovery and load_data.csv "
            "contents for the offending plates."
        )

    master_df["is_control"] = (master_df["control_type"] == "negcon").astype(int)
    master_df["compound_count"] = (
        master_df.groupby("broad_sample", dropna=False)["broad_sample"]
        .transform("size")
    )
    master_df["moa_count"] = (
        master_df.groupby("moa", dropna=False)["moa"]
        .transform("size")
    )

    print("final shape:", master_df.shape)
    print("final cols:", master_df.columns)

    if skipped_missing_load_data:
        print(f"\n{len(skipped_missing_load_data)} acquisition(s) skipped (missing load_data.csv):")
        for experiment, acquisition_id in skipped_missing_load_data:
            print(f"  {experiment}/{acquisition_id}")
        print("Fetch load_data.csv for these and rerun to include them.")

    if skipped_missing_manifest:
        print(f"\n{len(skipped_missing_manifest)} acquisition(s) skipped (no experiment_datasets.json entry):")
        for experiment, acquisition_id in skipped_missing_manifest:
            print(f"  {experiment}/{acquisition_id}")
        print("Run data/download_metadata.slurm (or scripts/download_compound_plates.py) "
              "for these experiments and rerun to include them.")

    if skipped_missing_platemap:
        print(f"\n{len(skipped_missing_platemap)} acquisition(s) skipped (platemap not resolvable):")
        for experiment, acquisition_id in skipped_missing_platemap:
            print(f"  {experiment}/{acquisition_id}")
        print("Fetch barcode_platemap.csv / platemap content for these and rerun to include them.")

    print("\nSaving...")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
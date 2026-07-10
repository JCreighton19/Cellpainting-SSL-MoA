"""
Reproducible downloader for COMPOUND-perturbation Cell Painting Gallery plates.

Selects a fixed, diversity-maximizing set of TARGET_NUM_PLATES (default 24)
compound plates from analysis/outputs/compound_plates.csv -- the authoritative
inventory produced by scripts/scan_compound_plates.py -- instead of randomly
sampling a fixed count per experiment. For each selected plate, this script:
  1. Resolves the barcode to its full Gallery acquisition_id by listing S3
     (a barcode can only be resolved to an acquisition_id that actually
     exists in the bucket right now -- this IS the "confirm acquisition_id
     exists in S3" check, not a separate step).
  2. Verifies the acquisition's Images/ folder is non-empty and that the
     acquisition doesn't already exist locally under a different experiment.
  3. Prints a summary table (with a diversity breakdown) and, unless
     --dry-run, downloads images, load_data.csv, and any missing shared
     metadata/platemap resources.

Selection strategy (see select_plates() for the full rationale): candidates
are grouped by (dataset, experiment) and visited in a round robin that
interleaves datasets first, then experiments within each dataset -- so every
configured dataset, and every distinct experiment within it, contributes one
plate before any group contributes a second. Fully deterministic (no
randomness) so reruns always pick the same plates.

Local layout produced (see datasets/build_metadata_table.py, which now reads
platemaps from the same platemaps/{dataset}/{experiment}/ location this
script writes to):

    data/raw/
      images/{experiment}/{acquisition_id}/*.tiff
      load_data_csv/{experiment}/{acquisition_id}/load_data.csv
      platemaps/{dataset}/{experiment}/barcode_platemap.csv
      platemaps/{dataset}/{experiment}/platemap/{plate_map_name}.txt
      JUMP-Target-1_compound_metadata.tsv          (dataset-global, unchanged path)
      JUMP-Target-1_compound_metadata_targets.tsv  (dataset-global, unchanged path)
      experiment_datasets.json                     (experiment -> dataset manifest)

`plate` in the metadata table stays the full acquisition_id
("BR00117006__2020-11-03T19_45_39-Measurement1"), never the bare barcode --
see datasets/build_metadata_table.py for why (the same barcode can be
re-imaged at multiple timepoints/experiments).

Only cpg0000-jump-pilot is configured in DATASET_CONFIGS below. Candidates
from any other dataset present in compound_plates.csv (e.g. cpg0002-jump-scope)
are skipped with a warning until that dataset gets its own DATASET_CONFIGS
entry -- nothing else in this script assumes a specific dataset.

Usage:
    python scripts/download_compound_plates.py --dry-run
    python scripts/download_compound_plates.py
    python scripts/download_compound_plates.py --target-plates 24 --datasets cpg0000-jump-pilot
"""
import argparse
import csv
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

S3_BUCKET = "cellpainting-gallery"

# One entry per Cell Painting Gallery dataset. `source` is the
# data-generating-center folder under the dataset (cpg0000-jump-pilot only
# ever used source_4; cpg0016 spans multiple source_N folders with no single
# default, so it isn't guessed here -- add it once you know which source(s)
# you want). *_metadata_file are read from workspace/metadata/external_metadata/
# and are dataset-global (not experiment-scoped); target_metadata_file is
# optional per dataset.
DATASET_CONFIGS = {
    "cpg0000-jump-pilot": {
        "source": "source_4",
        "compound_metadata_file": "JUMP-Target-1_compound_metadata.tsv",
        "target_metadata_file": "JUMP-Target-1_compound_metadata_targets.tsv",
    },
    # "cpg0016": {
    #     "source": "source_?",   # fill in once known
    #     "compound_metadata_file": "...",
    #     "target_metadata_file": None,
    # },
}

# Not on the Cell Painting Gallery bucket at all (Broad Drug Repurposing Hub,
# broadinstitute.org/repurposing) -- can't be fetched with `aws s3`. Checked
# for presence only; see check_moa_annotation() below.
MOA_ANNOTATION_PATH = REPO_ROOT / "data" / "raw" / "repo-drug-annotation-20200324.txt"

# Authoritative compound-plate inventory produced by scripts/scan_compound_plates.py.
# Every row is already confirmed compound (Plate_Map_Name contains "compound") --
# no re-classification needed here.
COMPOUND_PLATES_CSV = REPO_ROOT / "analysis" / "outputs" / "compound_plates.csv"

# How many plates select_plates() should pick when --target-plates isn't given.
TARGET_NUM_PLATES = 24

# Plates already downloaded in the original single-experiment dataset -- never
# select or count these toward diversity coverage. Each entry is
# (dataset, experiment, barcode, acquisition_id_if_known). Scoped to
# (dataset, experiment), NOT barcode alone: 4 of these barcodes (BR00117010-13)
# also appear as separate, still-available plates under cpg0000-jump-pilot's
# sibling 2020_11_04_CPJUMP1_DL experiment in compound_plates.csv -- excluding
# by barcode alone would have wrongly dropped those too. The acquisition_id is
# only used as a post-resolution defense-in-depth check (see main()) -- it is
# NOT how filtering against the CSV happens, since the CSV has no
# acquisition_id column and resolve_acquisition_id() remains the source of
# truth for what a barcode actually resolves to.
EXCLUDED_ACQUISITIONS = [
    ("cpg0000-jump-pilot", "2020_11_04_CPJUMP1", "BR00116991", "BR00116991__2020-11-05T19_51_35-Measurement1"),
    ("cpg0000-jump-pilot", "2020_11_04_CPJUMP1", "BR00116992", "BR00116992__2020-11-05T21_31_31-Measurement1"),
    ("cpg0000-jump-pilot", "2020_11_04_CPJUMP1", "BR00116993", "BR00116993__2020-11-05T23_11_39-Measurement1"),
    ("cpg0000-jump-pilot", "2020_11_04_CPJUMP1", "BR00116994", "BR00116994__2020-11-06T00_59_44-Measurement1"),
    ("cpg0000-jump-pilot", "2020_11_04_CPJUMP1", "BR00116995", None),
    ("cpg0000-jump-pilot", "2020_11_04_CPJUMP1", "BR00117010", "BR00117010__2020-11-08T18_18_00-Measurement1"),
    ("cpg0000-jump-pilot", "2020_11_04_CPJUMP1", "BR00117011", "BR00117011__2020-11-08T19_57_47-Measurement1"),
    ("cpg0000-jump-pilot", "2020_11_04_CPJUMP1", "BR00117012", "BR00117012__2020-11-08T14_58_34-Measurement1"),
    ("cpg0000-jump-pilot", "2020_11_04_CPJUMP1", "BR00117013", "BR00117013__2020-11-08T16_38_19-Measurement1"),
]
EXCLUDED_DATASET_EXPERIMENT_BARCODES = {(d, e, b) for d, e, b, _ in EXCLUDED_ACQUISITIONS}
EXCLUDED_ACQUISITION_IDS = {a for _, _, _, a in EXCLUDED_ACQUISITIONS if a}


# --------------------------------------------------------------------------
# S3 helpers (shell out to `aws`, matching every other script in this repo --
# awscli is already a project dependency; boto3 is not).
# --------------------------------------------------------------------------

class S3Error(RuntimeError):
    pass


def s3_uri(*parts: str) -> str:
    return "s3://" + "/".join(p.strip("/") for p in (S3_BUCKET, *parts))


def _run_aws(args: list, capture=True) -> subprocess.CompletedProcess:
    cmd = ["aws", *args, "--no-sign-request"]
    logging.debug("RUN: %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=capture, text=True)


def s3_list(prefix: str) -> tuple:
    """`aws s3 ls {prefix}/` (prefix must resolve to a "directory"). Returns
    (dirs, files) -- names only, dirs without the trailing slash."""
    result = _run_aws(["s3", "ls", prefix.rstrip("/") + "/"])
    if result.returncode != 0:
        raise S3Error(f"s3 ls failed for {prefix}: {result.stderr.strip()}")
    dirs, files = [], []
    for line in result.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "PRE":
            dirs.append(parts[1].rstrip("/"))
        else:
            files.append(parts[-1])
    return dirs, files


# --------------------------------------------------------------------------
# Plate discovery / selection / validation
# --------------------------------------------------------------------------

@dataclass
class PlateSelection:
    dataset: str
    experiment: str
    barcode: str
    plate_map_name: str
    acquisition_id: Optional[str] = None
    status: str = "selected"
    reason: str = ""


def download_barcode_platemap(dataset: str, experiment: str, data_root: Path, force: bool) -> Path:
    """Fetches barcode_platemap.csv for one (dataset, experiment) group actually
    selected for download. Not used for plate discovery/classification anymore
    (that now comes from the pre-scanned compound_plates.csv) -- this is purely
    so datasets/build_metadata_table.py's resolve_plate_layout() has the file
    it needs locally. Called from the download phase in main(), so --dry-run
    never triggers it; the `force` skip-if-exists behavior matches every other
    single-file download in this script."""
    cfg = DATASET_CONFIGS[dataset]
    dest = data_root / "platemaps" / dataset / experiment / "barcode_platemap.csv"
    src = s3_uri(dataset, cfg["source"], "workspace", "metadata", "platemaps",
                 experiment, "barcode_platemap.csv")
    download_file(src, dest, force=force, dry_run=False)
    return dest


def read_compound_plates_csv(path: Path) -> list:
    """Loads analysis/outputs/compound_plates.csv -- the authoritative,
    pre-scanned inventory of confirmed compound plates (see
    scripts/scan_compound_plates.py). Every row is already known-compound;
    unlike the old live-S3 discovery this replaces, no further
    classification happens here."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/scan_compound_plates.py first to "
            "produce the compound-plate inventory this script selects from."
        )
    with open(path, newline="") as f:
        return [
            PlateSelection(
                dataset=row["dataset"].strip(),
                experiment=row["experiment"].strip(),
                barcode=row["plate_barcode"].strip(),
                plate_map_name=row["plate_map_name"].strip(),
            )
            for row in csv.DictReader(f)
        ]


def load_candidate_plates(csv_path: Path, dataset_filter=None, experiment_filter=None) -> list:
    """Reads compound_plates.csv and applies the filters that must happen
    BEFORE diversity selection: already-downloaded exclusions, datasets not
    yet configured in DATASET_CONFIGS (can't be downloaded without a `source`
    to build S3 paths from), and any explicit --datasets/--experiments
    narrowing. select_plates() only ever sees plates that are actually
    eligible to download."""
    all_rows = read_compound_plates_csv(csv_path)
    logging.info("Loaded %d candidate compound plate row(s) from %s", len(all_rows), csv_path)

    candidates = []
    n_excluded = 0
    unconfigured_datasets = set()
    for c in all_rows:
        if (c.dataset, c.experiment, c.barcode) in EXCLUDED_DATASET_EXPERIMENT_BARCODES:
            n_excluded += 1
            continue
        if c.dataset not in DATASET_CONFIGS:
            unconfigured_datasets.add(c.dataset)
            continue
        if dataset_filter and c.dataset not in dataset_filter:
            continue
        if experiment_filter and c.experiment not in experiment_filter:
            continue
        candidates.append(c)

    logging.info("Excluded %d already-downloaded plate(s).", n_excluded)
    if unconfigured_datasets:
        logging.warning(
            "Skipping plates from unconfigured dataset(s) %s -- add a DATASET_CONFIGS "
            "entry (source + metadata filenames) to include them in future selections.",
            sorted(unconfigured_datasets),
        )
    logging.info("%d candidate plate(s) eligible for selection.", len(candidates))
    return candidates


def select_plates(candidates: list, target_n: int) -> list:
    """Diversity-maximizing selection: candidates are grouped by
    (dataset, experiment), then visited in a round robin that interleaves
    datasets first and experiments within each dataset second -- e.g. with
    two datasets A (3 experiments) and B (65 experiments), the visiting
    order is [A/e0, B/e0, A/e1, B/e1, A/e2, B/e2, B/e3, B/e4, ...]. Because
    every group is visited once before any group is visited twice, the
    first `target_n` groups in that order are exactly what gets selected
    whenever target_n <= number of distinct groups (the common case) --
    every plate comes from a DIFFERENT experiment, which is the strongest
    possible reading of "avoid 24 near-identical plates from one experiment
    when more diverse options exist." If target_n exceeds the number of
    distinct groups, additional passes take a second (then third, ...)
    plate from each group in the same order, so every group is exhausted
    evenly rather than draining one group before touching the next.

    Interleaving by dataset (rather than a flat sort of every (dataset,
    experiment) pair together) matters once more datasets are configured:
    a flat sort would let one dataset's experiments fill most of target_n
    just because they happen to sort first alphabetically. Nesting
    guarantees every configured dataset contributes before any dataset's
    *second* experiment is used.

    Deterministic throughout -- group visiting order is a plain sort, and
    the barcode picked from within a group is always the lexicographically
    smallest one remaining -- so reruns always select the same plates.
    No randomness, so no seed is needed.
    """
    by_dataset = {}
    for c in candidates:
        by_dataset.setdefault(c.dataset, {}).setdefault(c.experiment, []).append(c)
    for experiments in by_dataset.values():
        for pool in experiments.values():
            pool.sort(key=lambda c: c.barcode)

    datasets = sorted(by_dataset)
    experiments_by_dataset = {d: sorted(by_dataset[d]) for d in datasets}

    group_order = []
    i = 0
    while True:
        added_this_round = False
        for d in datasets:
            exps = experiments_by_dataset[d]
            if i < len(exps):
                group_order.append((d, exps[i]))
                added_this_round = True
        if not added_this_round:
            break
        i += 1

    selected = []
    while len(selected) < target_n:
        progressed = False
        for dataset, experiment in group_order:
            if len(selected) >= target_n:
                break
            pool = by_dataset[dataset][experiment]
            if pool:
                selected.append(pool.pop(0))
                progressed = True
        if not progressed:
            break

    if len(selected) < target_n:
        logging.warning(
            "Only %d compound plate(s) available after exclusions/filters "
            "(requested %d) -- taking all of them.", len(selected), target_n,
        )
    return selected


def resolve_acquisition_id(dataset: str, experiment: str, barcode: str) -> str:
    """Confirms the barcode's acquisition exists in S3 by listing the
    experiment's images/ prefix and anchoring on "{barcode}__" -- a plain
    substring match (as in the old manual script) could false-match a
    different barcode that happens to contain this one."""
    cfg = DATASET_CONFIGS[dataset]
    prefix = s3_uri(dataset, cfg["source"], "images", experiment, "images")
    dirs, _ = s3_list(prefix)
    matches = [d for d in dirs if d.startswith(f"{barcode}__")]
    if not matches:
        raise S3Error(f"No acquisition found for barcode {barcode} under {prefix}/")
    if len(matches) > 1:
        raise S3Error(f"Ambiguous acquisitions for barcode {barcode} under {prefix}/: {matches}")
    return matches[0]


def verify_images_present(dataset: str, experiment: str, acquisition_id: str):
    cfg = DATASET_CONFIGS[dataset]
    prefix = s3_uri(dataset, cfg["source"], "images", experiment, "images",
                     acquisition_id, "Images")
    _, files = s3_list(prefix)
    if not files:
        raise S3Error(f"Acquisition {acquisition_id} has an empty Images/ folder at {prefix}/")


def check_local_collision(data_root: Path, experiment: str, acquisition_id: str):
    """acquisition_id must not already exist locally under a DIFFERENT
    experiment folder -- that would mean two experiments claim the same
    globally-unique acquisition, which should be impossible and would break
    the plate/well/site uniqueness invariant build_metadata_table.py relies
    on. Already existing under the SAME experiment is fine (already
    downloaded; handled as a normal skip-if-exists further down)."""
    images_root = data_root / "images"
    if not images_root.exists():
        return
    for exp_dir in images_root.iterdir():
        if exp_dir.name == experiment or not exp_dir.is_dir():
            continue
        if (exp_dir / acquisition_id).exists():
            raise S3Error(
                f"acquisition_id {acquisition_id} already exists locally under "
                f"experiment {exp_dir.name!r}, but is being resolved again under "
                f"{experiment!r}. This should never happen -- investigate before "
                f"downloading."
            )


# --------------------------------------------------------------------------
# Download primitives (restartable: single files use skip-if-exists +
# tmp-then-rename so a failed transfer never leaves a file that looks
# complete; images use `aws s3 sync`, which is inherently resumable/idempotent)
# --------------------------------------------------------------------------

def download_file(src_uri: str, dest: Path, force: bool, dry_run: bool) -> str:
    if dest.exists() and not force:
        logging.info("SKIP (exists): %s", dest)
        return "skipped"

    cmd = ["aws", "s3", "cp", src_uri, str(dest) + ".tmp", "--no-sign-request", "--only-show-errors"]
    logging.info("CMD: %s", " ".join(cmd))
    if dry_run:
        return "dry-run"

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_name(dest.name + ".tmp")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error("FAILED: %s -> %s\n%s", src_uri, dest, result.stderr.strip())
        tmp_dest.unlink(missing_ok=True)
        return "failed"
    tmp_dest.replace(dest)
    logging.info("OK: %s -> %s", src_uri, dest)
    return "downloaded"


def download_images(dataset: str, experiment: str, acquisition_id: str,
                     data_root: Path, dry_run: bool) -> str:
    cfg = DATASET_CONFIGS[dataset]
    src = s3_uri(dataset, cfg["source"], "images", experiment, "images",
                 acquisition_id, "Images") + "/"
    dest = data_root / "images" / experiment / acquisition_id
    cmd = ["aws", "s3", "sync", src, str(dest), "--no-sign-request", "--only-show-errors"]
    logging.info("CMD: %s", " ".join(cmd))
    if dry_run:
        return "dry-run"

    dest.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error("FAILED image sync: %s -> %s\n%s", src, dest, result.stderr.strip())
        return "failed"
    logging.info("OK image sync: %s -> %s", src, dest)
    return "downloaded"


def download_load_data_csv(dataset: str, experiment: str, barcode: str,
                            acquisition_id: str, data_root: Path,
                            force: bool, dry_run: bool) -> str:
    cfg = DATASET_CONFIGS[dataset]
    # S3 path is barcode-only; local path preserves acquisition_id since the
    # same barcode can occur at multiple timepoints/experiments.
    src = s3_uri(dataset, cfg["source"], "workspace", "load_data_csv", experiment,
                 barcode, "load_data.csv")
    dest = data_root / "load_data_csv" / experiment / acquisition_id / "load_data.csv"
    return download_file(src, dest, force=force, dry_run=dry_run)


def download_platemap_file(dataset: str, experiment: str, plate_map_name: str,
                            data_root: Path, force: bool, dry_run: bool) -> str:
    cfg = DATASET_CONFIGS[dataset]
    src = s3_uri(dataset, cfg["source"], "workspace", "metadata", "platemaps",
                 experiment, "platemap", f"{plate_map_name}.txt")
    dest = data_root / "platemaps" / dataset / experiment / "platemap" / f"{plate_map_name}.txt"
    return download_file(src, dest, force=force, dry_run=dry_run)


def download_dataset_metadata(dataset: str, data_root: Path, force: bool, dry_run: bool):
    cfg = DATASET_CONFIGS[dataset]
    for key in ("compound_metadata_file", "target_metadata_file"):
        filename = cfg.get(key)
        if not filename:
            logging.info("No %s configured for dataset %s -- skipping.", key, dataset)
            continue
        src = s3_uri(dataset, cfg["source"], "workspace", "metadata", "external_metadata", filename)
        # Flat, matching datasets/build_metadata_table.py's existing
        # (unchanged) COMPOUND_METADATA_PATH/MOA_PATH constants.
        dest = data_root / filename
        download_file(src, dest, force=force, dry_run=dry_run)


def check_moa_annotation():
    if MOA_ANNOTATION_PATH.exists():
        logging.info("MoA annotation file present: %s", MOA_ANNOTATION_PATH)
    else:
        logging.warning(
            "MoA annotation file missing: %s -- this is NOT on the Cell Painting "
            "Gallery bucket (it's from the Broad Drug Repurposing Hub, "
            "broadinstitute.org/repurposing) and cannot be fetched by this script. "
            "Download it manually.", MOA_ANNOTATION_PATH,
        )


def update_experiment_dataset_manifest(data_root: Path, experiment: str, dataset: str, dry_run: bool):
    """datasets/build_metadata_table.py needs to know which dataset each
    experiment came from (to find the right platemaps/{dataset}/{experiment}/
    directory) but the images/load_data_csv paths don't carry dataset at all
    -- so this small manifest is the only place that mapping is recorded."""
    path = data_root / "experiment_datasets.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    if manifest.get(experiment) not in (None, dataset):
        raise S3Error(
            f"experiment {experiment!r} is already mapped to dataset "
            f"{manifest[experiment]!r} in {path}, cannot also map it to {dataset!r}."
        )
    if manifest.get(experiment) == dataset:
        return
    manifest[experiment] = dataset
    logging.info("Manifest update: %s -> %s (%s)", experiment, dataset, path)
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_summary_table(selections: list):
    headers = ["dataset", "experiment", "barcode", "acquisition_id", "platemap_type"]
    rows = [
        [s.dataset, s.experiment, s.barcode, s.acquisition_id or "?", "compound"]
        for s in selections
    ]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(headers)]

    def fmt_row(cells):
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    print()
    print(fmt_row(headers))
    print(fmt_row(["-" * w for w in widths]))
    for r in rows:
        print(fmt_row(r))
    print()
    print(f"Selected: {len(rows)} plate(s).")

    # Diversity breakdown so it's easy to eyeball how spread-out the final
    # selection is, not just that acquisition IDs were resolved.
    group_counts = {}
    for s in selections:
        key = (s.dataset, s.experiment)
        group_counts[key] = group_counts.get(key, 0) + 1
    print(f"Diversity: {len(rows)} plate(s) drawn from {len(group_counts)} "
          f"distinct (dataset, experiment) group(s):")
    for (dataset, experiment), count in sorted(group_counts.items()):
        print(f"  {dataset} / {experiment}: {count}")
    print()


def setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"download_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    return log_path


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download COMPOUND-only Cell Painting Gallery plates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets", nargs="+", default=None,
                         help="Optional filter: only select plates from these dataset(s). "
                              f"Must be configured in DATASET_CONFIGS ({', '.join(DATASET_CONFIGS)}). "
                              "Default: no filter (every configured dataset present in "
                              "--compound-plates-csv is eligible).")
    parser.add_argument("--experiments", nargs="+", default=None,
                         help="Optional filter: only select plates from these experiment(s). "
                              "Default: no filter.")
    parser.add_argument("--target-plates", type=int, default=TARGET_NUM_PLATES,
                         help="Total number of compound plates to select, spread across as "
                              "many distinct (dataset, experiment) groups as possible.")
    parser.add_argument("--compound-plates-csv", type=Path, default=COMPOUND_PLATES_CSV,
                         help="Authoritative compound-plate inventory (see "
                              "scripts/scan_compound_plates.py) to select from.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Select, validate, and print the summary table -- no images, "
                              "load_data.csv, platemap content, or dataset metadata are "
                              "downloaded.")
    parser.add_argument("--force", action="store_true",
                         help="Redownload single-file resources (load_data.csv, platemaps, "
                              "metadata) even if already present. Image sync is always safe "
                              "to rerun and ignores this flag.")
    parser.add_argument("--data-root", default=None,
                         help="Defaults to $CP_OUTPUT_ROOT/data/raw.")
    parser.add_argument("--log-dir", default=None,
                         help="Defaults to $CP_OUTPUT_ROOT/logs.")
    return parser.parse_args()


def resolve_root(cli_value: Optional[str], env_suffix: str, default_name: str) -> Path:
    if cli_value:
        return Path(cli_value)
    import os
    root = os.environ.get("CP_OUTPUT_ROOT")
    if root:
        return Path(root) / env_suffix
    return REPO_ROOT / default_name


def main():
    args = parse_args()
    data_root = resolve_root(args.data_root, "data/raw", "data/raw")
    log_dir = resolve_root(args.log_dir, "logs", "logs")
    log_path = setup_logging(log_dir)

    if args.datasets:
        unknown = [d for d in args.datasets if d not in DATASET_CONFIGS]
        if unknown:
            logging.error(
                "Unconfigured dataset(s): %s. Add an entry to DATASET_CONFIGS in "
                "scripts/download_compound_plates.py first (source folder + metadata "
                "filenames) -- refusing to guess. Configured datasets: %s",
                unknown, list(DATASET_CONFIGS),
            )
            sys.exit(2)

    logging.info("Log file: %s", log_path)
    logging.info("Data root: %s", data_root)
    logging.info("Args: %s", vars(args))

    candidates = load_candidate_plates(
        args.compound_plates_csv,
        dataset_filter=set(args.datasets) if args.datasets else None,
        experiment_filter=set(args.experiments) if args.experiments else None,
    )
    diverse_selections = select_plates(candidates, args.target_plates)

    all_selections = []
    failures = 0
    for sel in diverse_selections:
        try:
            sel.acquisition_id = resolve_acquisition_id(sel.dataset, sel.experiment, sel.barcode)
            if sel.acquisition_id in EXCLUDED_ACQUISITION_IDS:
                # Defense-in-depth: load_candidate_plates() already excludes these
                # by (dataset, experiment, barcode) before selection, so this
                # should never actually trigger -- see EXCLUDED_ACQUISITIONS.
                raise S3Error(
                    f"acquisition {sel.acquisition_id} is in EXCLUDED_ACQUISITIONS "
                    "(already downloaded)"
                )
            verify_images_present(sel.dataset, sel.experiment, sel.acquisition_id)
            check_local_collision(data_root, sel.experiment, sel.acquisition_id)
        except S3Error as e:
            logging.error("SKIP %s/%s barcode=%s: %s", sel.dataset, sel.experiment, sel.barcode, e)
            failures += 1
            continue
        all_selections.append(sel)

    for dataset, experiment in sorted({(s.dataset, s.experiment) for s in all_selections}):
        update_experiment_dataset_manifest(data_root, experiment, dataset, args.dry_run)

    print_summary_table(all_selections)

    if args.dry_run:
        logging.info("Dry run complete -- nothing downloaded.")
        print(f"Full log: {log_path}")
        sys.exit(1 if failures else 0)

    # Download per-plate resources first, then dataset-level shared resources
    # (barcode_platemap.csv, platemap content files, compound/target metadata)
    # once per (dataset, experiment) or per dataset as appropriate.
    plate_map_names_by_experiment = {}
    for sel in all_selections:
        r_images = download_images(sel.dataset, sel.experiment, sel.acquisition_id, data_root, args.dry_run)
        r_ldc = download_load_data_csv(sel.dataset, sel.experiment, sel.barcode,
                                        sel.acquisition_id, data_root, args.force, args.dry_run)
        if "failed" in (r_images, r_ldc):
            failures += 1
        plate_map_names_by_experiment.setdefault((sel.dataset, sel.experiment), set()).add(sel.plate_map_name)

    for (dataset, experiment), plate_map_names in plate_map_names_by_experiment.items():
        # datasets/build_metadata_table.py needs this locally to resolve each
        # plate's platemap design; no longer fetched during selection (that now
        # reads analysis/outputs/compound_plates.csv instead).
        download_barcode_platemap(dataset, experiment, data_root, force=args.force)
        for plate_map_name in sorted(plate_map_names):
            r = download_platemap_file(dataset, experiment, plate_map_name, data_root, args.force, args.dry_run)
            if r == "failed":
                failures += 1

    for dataset in {sel.dataset for sel in all_selections}:
        download_dataset_metadata(dataset, data_root, args.force, args.dry_run)

    check_moa_annotation()

    logging.info("Done. %d failure(s). Full log: %s", failures, log_path)
    print(f"Full log: {log_path}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

"""
Reproducible downloader for COMPOUND-perturbation Cell Painting Gallery plates.

Replaces manual `aws s3 cp` runs. For each requested (dataset, experiment)
pair, this script:
  1. Downloads that experiment's barcode_platemap.csv (Assay_Plate_Barcode ->
     Plate_Map_Name) if not already present locally.
  2. Filters to barcodes whose Plate_Map_Name contains "compound" and does
     NOT contain "orf"/"crispr" -- ORF and CRISPR plates are never selected,
     regardless of --plates-per-experiment.
  3. Randomly samples (fixed --seed, so reruns are reproducible) up to
     --plates-per-experiment barcodes from the valid compound set.
  4. Resolves each barcode to its full Gallery acquisition_id by listing S3
     (a barcode can only be resolved to an acquisition_id that actually
     exists in the bucket right now -- this IS the "confirm acquisition_id
     exists in S3" check, not a separate step).
  5. Prints a summary table and, unless --dry-run, downloads images,
     load_data.csv, and any missing shared metadata/platemap resources.

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

Only cpg0000-jump-pilot is configured in DATASET_CONFIGS below. Adding a new
Cell Painting Gallery dataset (e.g. cpg0016) means adding one entry there --
nothing else in this script assumes a specific dataset.

Usage:
    python scripts/download_compound_plates.py \\
        --datasets cpg0000-jump-pilot \\
        --experiments 2020_11_04_CPJUMP1 \\
        --plates-per-experiment 5 \\
        --dry-run
"""
import argparse
import json
import logging
import random
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

COMPOUND_TOKEN = "compound"
REJECTED_TOKENS = ("orf", "crispr")


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


def classify_platemap(plate_map_name: str) -> str:
    name = plate_map_name.lower()
    if any(tok in name for tok in REJECTED_TOKENS):
        return "rejected"
    if COMPOUND_TOKEN in name:
        return "compound"
    return "unknown"


def download_barcode_platemap(dataset: str, experiment: str, data_root: Path, force: bool) -> Path:
    """Always actually fetched (skip-if-exists still applies), regardless of
    --dry-run: it's a few-KB inspection input needed to select plates and
    print the preview table at all, not part of "the download" --dry-run is
    meant to suppress (images/load_data_csv/platemap content/metadata)."""
    cfg = DATASET_CONFIGS[dataset]
    dest = data_root / "platemaps" / dataset / experiment / "barcode_platemap.csv"
    src = s3_uri(dataset, cfg["source"], "workspace", "metadata", "platemaps",
                 experiment, "barcode_platemap.csv")
    download_file(src, dest, force=force, dry_run=False)
    return dest


def discover_compound_barcodes(dataset: str, experiment: str, barcode_map_path: Path):
    """Returns (compound_barcode_to_platemap: dict, skip_counts: dict)."""
    import csv

    compound = {}
    skip_counts = {"rejected": 0, "unknown": 0}
    if not barcode_map_path.exists():
        logging.warning(
            "No barcode_platemap.csv for %s/%s (dry-run and file not fetched yet, "
            "or download failed) -- cannot select plates for this experiment.",
            dataset, experiment,
        )
        return compound, skip_counts

    with open(barcode_map_path, newline="") as f:
        for row in csv.DictReader(f):
            barcode = row["Assay_Plate_Barcode"].strip()
            plate_map_name = row["Plate_Map_Name"].strip()
            kind = classify_platemap(plate_map_name)
            if kind == "compound":
                compound[barcode] = plate_map_name
            else:
                skip_counts[kind] += 1
                logging.info("SKIP %s/%s barcode=%s plate_map=%s reason=%s",
                             dataset, experiment, barcode, plate_map_name, kind)
    return compound, skip_counts


def select_plates(dataset: str, experiment: str, compound_barcodes: dict,
                   n: int, seed: int) -> list:
    barcodes = sorted(compound_barcodes)
    if len(barcodes) <= n:
        if len(barcodes) < n:
            logging.warning(
                "%s/%s has only %d valid compound plates (requested %d) -- taking all of them.",
                dataset, experiment, len(barcodes), n,
            )
        chosen = barcodes
    else:
        chosen = random.Random(seed).sample(barcodes, n)
    return [
        PlateSelection(dataset, experiment, barcode, compound_barcodes[barcode])
        for barcode in sorted(chosen)
    ]


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

def print_summary_table(selections: list, skip_totals: dict):
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
    if skip_totals["rejected"] or skip_totals["unknown"]:
        print(f"Rejected (ORF/CRISPR): {skip_totals['rejected']}   "
              f"Unrecognized platemap type: {skip_totals['unknown']}")
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
    parser.add_argument("--datasets", nargs="+", required=True,
                         help=f"Cell Painting Gallery dataset name(s), e.g. cpg0000-jump-pilot. "
                              f"Must be configured in DATASET_CONFIGS ({', '.join(DATASET_CONFIGS)}).")
    parser.add_argument("--experiments", nargs="+", required=True,
                         help="Experiment/batch folder name(s), e.g. 2020_11_04_CPJUMP1. "
                              "Tried against every --datasets entry; an experiment that "
                              "doesn't exist under a given dataset is skipped with a warning.")
    parser.add_argument("--plates-per-experiment", type=int, required=True,
                         help="Number of valid compound plates to download per (dataset, experiment).")
    parser.add_argument("--dry-run", action="store_true",
                         help="Discover, validate, and print the summary table -- no images, "
                              "load_data.csv, platemap content, or dataset metadata are "
                              "downloaded. Exception: barcode_platemap.csv (a few KB) is still "
                              "fetched if missing, since plate selection can't be previewed "
                              "without it.")
    parser.add_argument("--force", action="store_true",
                         help="Redownload single-file resources (load_data.csv, platemaps, "
                              "metadata) even if already present. Image sync is always safe "
                              "to rerun and ignores this flag.")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for plate sampling (reproducible reruns).")
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

    all_selections = []
    skip_totals = {"rejected": 0, "unknown": 0}
    failures = 0

    for dataset in args.datasets:
        for experiment in args.experiments:
            logging.info("=== %s / %s ===", dataset, experiment)
            barcode_map_path = download_barcode_platemap(dataset, experiment, data_root, force=args.force)
            if not barcode_map_path.exists():
                logging.error(
                    "Could not fetch barcode_platemap.csv for %s/%s -- does this experiment "
                    "exist under this dataset? Skipping.", dataset, experiment,
                )
                failures += 1
                continue

            compound_barcodes, counts = discover_compound_barcodes(dataset, experiment, barcode_map_path)
            skip_totals["rejected"] += counts["rejected"]
            skip_totals["unknown"] += counts["unknown"]
            if not compound_barcodes:
                logging.warning("No valid compound plates found for %s/%s.", dataset, experiment)
                continue

            for sel in select_plates(dataset, experiment, compound_barcodes,
                                      args.plates_per_experiment, args.seed):
                try:
                    sel.acquisition_id = resolve_acquisition_id(dataset, experiment, sel.barcode)
                    verify_images_present(dataset, experiment, sel.acquisition_id)
                    check_local_collision(data_root, experiment, sel.acquisition_id)
                except S3Error as e:
                    logging.error("SKIP %s/%s barcode=%s: %s", dataset, experiment, sel.barcode, e)
                    failures += 1
                    continue
                all_selections.append(sel)

            update_experiment_dataset_manifest(data_root, experiment, dataset, args.dry_run)

    print_summary_table(all_selections, skip_totals)

    if args.dry_run:
        logging.info("Dry run complete -- nothing downloaded.")
        print(f"Full log: {log_path}")
        sys.exit(1 if failures else 0)

    # Download per-plate resources first, then dataset-level shared resources
    # (platemap content files, compound/target metadata) once per dataset.
    plate_map_names_by_experiment = {}
    for sel in all_selections:
        r_images = download_images(sel.dataset, sel.experiment, sel.acquisition_id, data_root, args.dry_run)
        r_ldc = download_load_data_csv(sel.dataset, sel.experiment, sel.barcode,
                                        sel.acquisition_id, data_root, args.force, args.dry_run)
        if "failed" in (r_images, r_ldc):
            failures += 1
        plate_map_names_by_experiment.setdefault((sel.dataset, sel.experiment), set()).add(sel.plate_map_name)

    for (dataset, experiment), plate_map_names in plate_map_names_by_experiment.items():
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

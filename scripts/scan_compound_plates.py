"""
Read-only Cell Painting Gallery inventory scan.

Determines which datasets/experiments in the Cell Painting Gallery contain
COMPOUND perturbation plates. Downloads NO images and saves NO per-plate
files permanently -- every barcode_platemap.csv is streamed straight into
memory via `aws s3 cp ... -` (stdout), never written to disk. The only
persistent output is the report this script writes itself (summary.txt +
two CSVs).

"source" (the data-generating-center folder, e.g. source_4) is NOT assumed.
Every top-level subfolder of a dataset -- plus the dataset root itself, for
datasets with no source_N level at all -- is probed for a
workspace/metadata/platemaps/ directory, since different Cell Painting
Gallery datasets are known to use different conventions (cpg0000-jump-pilot
uses a single source_4; cpg0016 spans many source_N folders; some datasets
may have no source level). See discover_platemap_roots() below.

A plate counts as "compound" if its Plate_Map_Name contains the substring
"compound" (case-insensitive) -- deliberately NOT restricted to the
"JUMP-Target-1_compound_platemap" name cpg0000-jump-pilot happens to use,
since other datasets are expected to use different naming.

Every dataset is scanned independently and defensively: a missing/broken
dataset (no platemaps directory, unreadable CSV, unexpected columns,
network error, or any other exception) is logged and skipped, never aborts
the rest of the run.

Usage:
    python scripts/scan_compound_plates.py
    python scripts/scan_compound_plates.py --datasets cpg0000-jump-pilot cpg0004-lincs
"""
import argparse
import csv
import io
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
S3_BUCKET = "cellpainting-gallery"

ALL_DATASETS = [
    "cpg0000-jump-pilot",
    "cpg0001-cellpainting-protocol",
    "cpg0002-jump-scope",
    "cpg0003-rosetta",
    "cpg0004-lincs",
    "cpg0005-gerry-bioactivity",
    "cpg0006-miami",
    "cpg0008-pki",
    "cpg0009-molglue",
    "cpg0010-caie-drugresponse",
    "cpg0011-lipocyteprofiler",
    "cpg0012-wawer-bioactivecompoundprofiling",
    "cpg0014-jump-adipocyte",
    "cpg0015-heterogeneity",
    "cpg0016-jump-assembled",
    "cpg0016-jump",
    "cpg0017-rohban-pathways",
    "cpg0018-singh-seedseq",
    "cpg0019-moshkov-deepprofiler",
    "cpg0020-varchamp",
    "cpg0021-periscope",
    "cpg0022-cmqtl",
    "cpg0023-mpi",
    "cpg0024-bortezomib",
    "cpg0025-dactyloscopy",
    "cpg0026-lacoste_haghighi-rare-diseases",
    "cpg0028-kelley-resistance",
    "cpg0029-chroma-pilot",
    "cpg0030-gustafsdottir-cellpainting",
    "cpg0031-caicedo-cmvip",
    "cpg0032-pooled-rare",
    "cpg0033-oasis-pilot",
    "cpg0034-arevalo-su-motive",
    "cpg0036-EU-OS-bioactives",
    "cpg0037-oasis",
    "cpg0038-tegtmeyer-neuropainting",
    "cpg0039-garcia-fossa-livecellpainting",
    "cpg0040-garcia-fossa-AgNP",
    "cpg0042-chandrasekaran-jump",
    "cpg0043-segmentation",
    "cpg0045-ncats-mito",
    "cpg0046-microrna",
    "cpg0047-amish",
]

COMPOUND_TOKEN = "compound"


# --------------------------------------------------------------------------
# S3 helpers. Self-contained (not imported from scripts/download_compound_plates.py)
# so this read-only scanner has zero dependency on the downloader's internals.
# --------------------------------------------------------------------------

class S3Error(RuntimeError):
    pass


def s3_uri(*parts) -> str:
    return "s3://" + "/".join(p.strip("/") for p in (S3_BUCKET, *parts) if p)


def _run_aws(args: list) -> subprocess.CompletedProcess:
    cmd = ["aws", *args, "--no-sign-request", "--only-show-errors"]
    return subprocess.run(cmd, capture_output=True, text=True)


def s3_list(prefix: str):
    """Returns (dirs, files) for one level under `prefix`. Raises S3Error on
    failure (nonexistent prefix, network error, etc.)."""
    result = _run_aws(["s3", "ls", prefix.rstrip("/") + "/"])
    if result.returncode != 0:
        raise S3Error(result.stderr.strip() or f"aws s3 ls failed for {prefix}")
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


def s3_list_safe(prefix: str):
    """Same as s3_list(), but swallows S3Error and returns ([], []) --
    for probing candidate paths that are expected to 404 most of the time."""
    try:
        return s3_list(prefix)
    except S3Error as e:
        logging.debug("s3 ls failed for %s: %s", prefix, e)
        return [], []


def read_barcode_platemap(uri: str):
    """Streams the CSV straight into memory -- never written to disk.
    Returns a list of (barcode, plate_map_name) tuples, or None if the file
    couldn't be read/parsed (logged, not raised, so the caller skips just
    this one experiment and keeps scanning)."""
    result = _run_aws(["s3", "cp", uri, "-"])
    if result.returncode != 0:
        logging.warning("Could not read %s: %s", uri, result.stderr.strip())
        return None

    try:
        reader = csv.DictReader(io.StringIO(result.stdout))
        fieldnames = reader.fieldnames or []
        if "Assay_Plate_Barcode" not in fieldnames or "Plate_Map_Name" not in fieldnames:
            logging.warning("Unexpected columns in %s: %s", uri, fieldnames)
            return None
        return [
            (row["Assay_Plate_Barcode"].strip(), row["Plate_Map_Name"].strip())
            for row in reader
            if row.get("Assay_Plate_Barcode")
        ]
    except Exception as e:
        logging.warning("Could not parse %s: %s", uri, e)
        return None


# --------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------

@dataclass
class ExperimentReport:
    dataset: str
    experiment: str
    num_plates: int = 0
    num_compound_plates: int = 0
    unique_plate_map_names: set = field(default_factory=set)


@dataclass
class CompoundPlateRow:
    dataset: str
    experiment: str
    plate_barcode: str
    plate_map_name: str


def discover_platemap_roots(dataset: str):
    """Yields (source, platemaps_uri, experiment_names) for every place under
    this dataset with a non-empty workspace/metadata/platemaps/ directory.
    `source` is "" for a dataset with no source_N level. Every top-level
    subfolder of the dataset is tried as a candidate source -- see module
    docstring for why this isn't hardcoded to source_4 or similar."""
    dataset_root = s3_uri(dataset)
    try:
        top_dirs, _ = s3_list(dataset_root)
    except S3Error as e:
        logging.error("Dataset %s: could not list %s (%s)", dataset, dataset_root, e)
        return

    for source in ["", *top_dirs]:
        platemaps_uri = s3_uri(dataset, source, "workspace", "metadata", "platemaps")
        experiment_names, _ = s3_list_safe(platemaps_uri)
        if experiment_names:
            yield source, platemaps_uri, experiment_names


def scan_dataset(dataset: str):
    """Returns (experiment_reports, compound_rows, has_platemap_metadata)."""
    logging.info("=== Scanning dataset: %s ===", dataset)

    reports_by_experiment = {}
    compound_rows = []
    found_any_platemaps_dir = False

    for source, platemaps_uri, experiment_names in discover_platemap_roots(dataset):
        found_any_platemaps_dir = True
        logging.info("  platemaps root: %s (%d experiment folder(s))",
                     platemaps_uri, len(experiment_names))

        for i, experiment in enumerate(sorted(experiment_names), 1):
            barcode_csv_uri = s3_uri(dataset, source, "workspace", "metadata",
                                      "platemaps", experiment, "barcode_platemap.csv")
            rows = read_barcode_platemap(barcode_csv_uri)
            if rows is None:
                logging.info("  [%d/%d] %s: could not read barcode_platemap.csv, skipping",
                             i, len(experiment_names), experiment)
                continue

            if experiment in reports_by_experiment:
                logging.warning(
                    "  Experiment name %r seen under multiple source prefixes in "
                    "dataset %s -- merging.", experiment, dataset
                )
                report = reports_by_experiment[experiment]
            else:
                report = ExperimentReport(dataset=dataset, experiment=experiment)
                reports_by_experiment[experiment] = report

            n_compound = 0
            for barcode, plate_map_name in rows:
                report.num_plates += 1
                report.unique_plate_map_names.add(plate_map_name)
                if COMPOUND_TOKEN in plate_map_name.lower():
                    n_compound += 1
                    report.num_compound_plates += 1
                    compound_rows.append(CompoundPlateRow(dataset, experiment, barcode, plate_map_name))

            logging.info("  [%d/%d] %s: %d plate(s), %d compound",
                         i, len(experiment_names), experiment, len(rows), n_compound)

    return list(reports_by_experiment.values()), compound_rows, found_any_platemaps_dir


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_dataset_summary(dataset: str, reports, has_platemap_metadata: bool, out):
    print("=========================================", file=out)
    print(f"Dataset: {dataset}", file=out)

    if not has_platemap_metadata:
        print("✗ No barcode_platemap metadata found.", file=out)
        print(file=out)
        return

    compound_reports = [r for r in reports if r.num_compound_plates > 0]
    if not compound_reports:
        print("✗ No compound perturbation plates found.", file=out)
        print(file=out)
        return

    total_compound_plates = sum(r.num_compound_plates for r in reports)
    print(f"✓ Compound experiments found: {len(compound_reports)}", file=out)
    print(f"✓ Compound plates found: {total_compound_plates}", file=out)
    print(file=out)
    print("Experiments:", file=out)
    for r in sorted(compound_reports, key=lambda r: r.experiment):
        print(f"{r.experiment} ({r.num_compound_plates} compound plates)", file=out)
    print(file=out)


def write_experiments_csv(path: Path, all_reports):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "experiment", "num_plates", "num_compound_plates",
                          "compound_found", "unique_plate_map_names"])
        for r in sorted(all_reports, key=lambda r: (r.dataset, r.experiment)):
            writer.writerow([
                r.dataset, r.experiment, r.num_plates, r.num_compound_plates,
                r.num_compound_plates > 0,
                "; ".join(sorted(r.unique_plate_map_names)),
            ])


def write_compound_plates_csv(path: Path, all_compound_rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "experiment", "plate_barcode", "plate_map_name"])
        for row in sorted(all_compound_rows, key=lambda r: (r.dataset, r.experiment, r.plate_barcode)):
            writer.writerow([row.dataset, row.experiment, row.plate_barcode, row.plate_map_name])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"gallery_scan_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    return log_path


def resolve_root(cli_value, env_suffix: str, default_name: str) -> Path:
    if cli_value:
        return Path(cli_value)
    root = os.environ.get("CP_OUTPUT_ROOT")
    if root:
        return Path(root) / env_suffix
    return REPO_ROOT / default_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read-only scan of the Cell Painting Gallery for compound-perturbation "
                     "plates. Downloads no images; saves nothing except its own report files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                         help="Datasets to scan (default: the full Cell Painting Gallery list).")
    parser.add_argument("--out-dir", default=None,
                         help="Defaults to $CP_OUTPUT_ROOT/gallery_scan/<timestamp>/.")
    parser.add_argument("--log-dir", default=None,
                         help="Defaults to $CP_OUTPUT_ROOT/logs.")
    return parser.parse_args()


def main():
    args = parse_args()
    log_path = setup_logging(resolve_root(args.log_dir, "logs", "logs"))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else resolve_root(
        None, f"gallery_scan/{ts}", f"gallery_scan/{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Log file: %s", log_path)
    logging.info("Output directory: %s", out_dir)
    logging.info("Datasets to scan (%d): %s", len(args.datasets), args.datasets)

    all_reports = []
    all_compound_rows = []
    dataset_has_metadata = {}

    for dataset in args.datasets:
        try:
            reports, compound_rows, has_platemap_metadata = scan_dataset(dataset)
        except Exception:
            logging.exception("Dataset %s: unexpected error, skipping.", dataset)
            dataset_has_metadata[dataset] = False
            continue

        all_reports.extend(reports)
        all_compound_rows.extend(compound_rows)
        dataset_has_metadata[dataset] = has_platemap_metadata

        n_compound_experiments = sum(1 for r in reports if r.num_compound_plates > 0)
        logging.info("Dataset %s complete: %d experiment(s) with metadata, %d with compound plates.",
                     dataset, len(reports), n_compound_experiments)

    experiments_csv = out_dir / "experiments.csv"
    compound_plates_csv = out_dir / "compound_plates.csv"
    write_experiments_csv(experiments_csv, all_reports)
    write_compound_plates_csv(compound_plates_csv, all_compound_rows)
    logging.info("Wrote %s (%d rows)", experiments_csv, len(all_reports))
    logging.info("Wrote %s (%d rows)", compound_plates_csv, len(all_compound_rows))

    summary_buf = io.StringIO()
    for dataset in args.datasets:
        reports = [r for r in all_reports if r.dataset == dataset]
        print_dataset_summary(dataset, reports, dataset_has_metadata.get(dataset, False), summary_buf)

    n_datasets_with_compound = sum(
        1 for d in args.datasets
        if any(r.dataset == d and r.num_compound_plates > 0 for r in all_reports)
    )
    print("=========================================", file=summary_buf)
    print(f"TOTAL: {n_datasets_with_compound}/{len(args.datasets)} dataset(s) have compound "
          f"plates; {len(all_compound_rows)} compound plate(s) found overall.", file=summary_buf)

    summary_text = summary_buf.getvalue()
    summary_path = out_dir / "summary.txt"
    summary_path.write_text(summary_text)

    print(summary_text)
    print(f"Summary: {summary_path}")
    print(f"Experiments CSV: {experiments_csv}")
    print(f"Compound plates CSV: {compound_plates_csv}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()

# Cell Painting DINO — Self-Supervised Morphological Profiling

A research pipeline that trains a self-supervised Vision Transformer (DINO) on
5-channel Cell Painting microscopy images from the JUMP Cell Painting
Consortium, with no compound or mechanism-of-action (MoA) labels used during
training — and an interactive web application, the **Cell Painting Embedding
Explorer**, for exploring the resulting embedding space.

> **Status: research proof of concept.** This project tests whether
> self-supervised representation learning can recover pharmacologically
> meaningful structure from Cell Painting images on its own. It is not a
> validated screening tool — see [Known Limitations](#known-limitations).

---

## Table of Contents

1. [Motivation](#motivation)
2. [Scientific Background](#scientific-background)
3. [Repository Structure](#repository-structure)
4. [Where the Data Lives: Repo vs. Cluster](#where-the-data-lives-repo-vs-cluster)
5. [End-to-End Pipeline](#end-to-end-pipeline)
   - [1. Data Acquisition](#1-data-acquisition)
   - [2. Metadata Construction](#2-metadata-construction)
   - [3. Image Preprocessing](#3-image-preprocessing)
   - [4. Dataset & Sampling](#4-dataset--sampling)
   - [5. Model & Training](#5-model--training)
   - [6. Embedding Extraction](#6-embedding-extraction)
   - [7. Post-Processing](#7-post-processing)
   - [8. Evaluation & Diagnostics](#8-evaluation--diagnostics)
   - [9. Web App Data Preparation](#9-web-app-data-preparation)
6. [The Web Application](#the-web-application)
7. [Model Results](#model-results)
8. [Environment & Setup](#environment--setup)
9. [Running the Pipeline End to End](#running-the-pipeline-end-to-end)
10. [Known Limitations](#known-limitations)
11. [Sources & References](#sources--references)
12. [Acknowledgements](#acknowledgements)

---

## Motivation

Cell Painting is a high-content microscopy assay that stains multiple
cellular compartments at once, producing rich, high-dimensional images of how
a cell "looks" after being treated with a compound. Turning those images into
a quantitative *morphological profile* is normally done with hand-engineered
feature extractors (e.g. CellProfiler) or with supervised deep networks
trained against labels — but large-scale labels (which mechanism of action a
compound affects, which pathway it perturbs) are expensive, incomplete, and
often simply don't exist for the majority of a compound library.

This project asks a narrower, more tractable question instead: **can a
self-supervised vision model, trained on nothing but the raw microscopy
images themselves, discover a representation in which phenotypically similar
wells end up close together — without ever being told what "similar" means?**
If so, the resulting embedding space becomes a general-purpose tool for
early-stage drug discovery workflows such as:

- **Hit prioritization** — surfacing novel compounds that look phenotypically
  similar to ones with a known mechanism.
- **Drug repurposing** — flagging existing drugs whose morphological profile
  suggests an unexplored therapeutic use.
- **Mechanism discovery** — generating hypotheses about an uncharacterized
  compound's mechanism by comparing it to better-understood compounds.
- **Prioritizing follow-up experiments** — deciding which hits are worth
  pursuing with more targeted, costly assays.

The approach follows [Kim et al. (2025)](#sources--references), who
demonstrated that self-supervised (DINO-style) vision transformers trained
directly on Cell Painting images produce more informative morphological
profiles than classical hand-engineered features. This repository is an
independent re-implementation and extension of that recipe on a subset of the
JUMP Cell Painting dataset, built end to end: raw image acquisition, DINO
pretraining, embedding extraction, statistical evaluation, and an interactive
web explorer for the resulting embedding space.

## Scientific Background

- **Cell Painting** stains six cellular components across five fluorescence
  channels (mitochondria, actin/Golgi/plasma membrane [AGP], RNA/nucleoli,
  endoplasmic reticulum, and DNA/nucleus), producing a detailed multi-channel
  view of how a compound changes cell morphology. See the [Broad Institute's
  Cell Painting overview](#sources--references).
- **JUMP Cell Painting Consortium** (`cpg0000-jump-pilot` on the public,
  no-sign-request "Cell Painting Gallery" S3 bucket) is the open dataset this
  project draws its images from. See the [JUMP Cell Painting
  Consortium](#sources--references).
- **DINO** (self-**DI**stillation with **NO** labels) is a self-supervised
  training recipe in which a "student" network is trained to match a
  slow-moving "teacher" (an exponential moving average of the student's own
  weights) across different augmented/cropped views of the same image — no
  labels involved. See [Caron et al. (2021)](#sources--references).
- **Vision Transformer (ViT)** is the backbone architecture DINO trains here:
  images are split into fixed-size patches, linearly embedded, and processed
  by a standard Transformer encoder. See [Dosovitskiy et al.
  (2020)](#sources--references).
- **UMAP** (Uniform Manifold Approximation and Projection) is the
  dimensionality-reduction method used to compress the model's 384-dimensional
  embeddings down to 2D/3D for visualization in the web app. See [McInnes et
  al. (2018)](#sources--references).

## Repository Structure

Only files tracked in this git repository are listed below. Large or
regenerable artifacts (raw images, processed image tensors, model
checkpoints, extracted tile embeddings, generated web thumbnails/attention
maps) are **not** committed — see
[Where the Data Lives](#where-the-data-lives-repo-vs-cluster) for where they
actually live.

```
JumpCP MoA/
├── analysis/                        # Post-training extraction & evaluation
│   ├── extract_embeddings.py        #   Tile-and-embed every FOV with a trained checkpoint
│   ├── extract_attention_maps.py    #   Pull CLS→patch attention maps for a few sample FOVs
│   ├── channel_ablation.py          #   Per-channel importance via ablation (zero/mean-fill)
│   ├── replicate_correlation.py     #   Cross-plate replicate-correlation diagnostic
│   ├── run_umap.py                  #   Early standalone UMAP + matplotlib scatter script
│   └── *.slurm                      #   SLURM submission wrappers for the above
│
├── app_data/                        # Committed snapshot that powers the live webapp
│   ├── wells.parquet                #   Well-level metadata + 2D/3D UMAP coordinates
│   ├── well_embeddings.npy          #   Post-processed, L2-normalized well embeddings
│   ├── compounds.parquet            #   Compound-level aggregation (mean-pooled over wells)
│   └── compound_embeddings.npy      #   Post-processed, L2-normalized compound embeddings
│
├── archive/                         # Deprecated/debug code, kept for reference only
│   ├── dataset.py                   #   Earlier CellPaintingDataset (pre-tile-caching design)
│   └── debug/
│       ├── diagnosis.py             #   Ad hoc join/coverage diagnostics over raw images
│       └── extract_debug_images.py  #   Dumps sample images for visual inspection
│
├── data/                            # Data-acquisition scripts + small tracked samples
│   ├── download_metadata.slurm      #   Fetches per-plate metadata (platemaps, load_data.csv)
│   ├── download_images.slurm        #   SLURM array job: syncs raw .tiff images from S3
│   └── raw/
│       ├── moa_overrides.tsv        #   Hand-curated MoA fallback for compounds Broad's file misses
│       ├── compound_metadata/…      #   One tracked example plate's compound metadata (sample)
│       └── platemaps/…              #   One tracked example plate's platemap (sample)
│
├── datasets/                        # Metadata joining, preprocessing, PyTorch Dataset/Sampler
│   ├── build_metadata_table.py      #   Joins imaging index + platemap + compound + MoA tables
│   ├── preprocess_dataset.py        #   Percentile-normalizes images, computes Otsu threshold
│   ├── precompute_tiles.py          #   Offline foreground-aware tile bank precomputation
│   ├── dataset.py                   #   CellPaintingDataset (PyTorch Dataset over .pt tiles)
│   ├── sampler.py                   #   MoASampler — MoA-/compound-aware sampling utilities
│   └── *.slurm                      #   SLURM wrappers for the above
│
├── env/
│   └── environment.yml              # Conda environment specification
│
├── models/
│   ├── config.py                    #   Training hyperparameters (CONFIG dict)
│   ├── dino/                        #   The model actually used for all reported results
│   │   ├── dino.py                  #     CellPaintingViT (5-channel ViT-S/14) + DINOHead
│   │   ├── dino_loss.py             #     DINOLoss (cross-entropy w/ centering + temp schedule)
│   │   ├── train_dino.py            #     Full training loop (multi-crop, EMA teacher, AMP)
│   │   └── train_dino.slurm         #     GPU SLURM job w/ auto-resume/requeue
│   └── scvg/                        #   Earlier experimental SupCon/VICReg-style variant
│       ├── scvg.py, scvg_loss.py, train_scvg.py, train_scvg.slurm
│
├── notebooks/                       # Exploratory analysis (Jupyter)
│   ├── eda.ipynb                    #   Early image/data-loading EDA
│   ├── metadata_eda.ipynb           #   Metadata parquet integrity checks (joins, coverage)
│   ├── train_dino.ipynb             #   Notebook version of the DINO training loop
│   ├── visualize_otsu.ipynb         #   Visual sanity checks of Otsu foreground thresholding
│   └── embedding_analysis.ipynb     #   Training-curve parsing + embedding sanity checks
│
├── scripts/                         # Gallery discovery/download + webapp asset generation
│   ├── scan_compound_plates.py      #   Read-only scan of the whole Cell Painting Gallery
│   ├── download_compound_plates.py  #   Deterministic, diversity-maximizing plate downloader
│   ├── prepare_phase1_data.py       #   Tile embeddings → well-level parquet + UMAP (2D & 3D)
│   ├── prepare_phase2_compounds.py  #   Well-level → compound-level aggregation
│   ├── generate_web_thumbnails.py   #   Renders per-well RGB composite thumbnails (WebP)
│   ├── generate_web_attention_maps.py   # Renders per-well attention-map mosaics
│   ├── generate_web_channel_importance.py # Per-well channel-ablation scores (JSON)
│   └── *.slurm                      #   SLURM wrappers for the above
│
├── utils/
│   ├── foreground_crop.py           #   Otsu-threshold rejection-sampling crop logic (shared)
│   └── postprocessing.py            #   MADScaler + SpheringTransform (paper's post-processing)
│
├── webapp/                          # Flask web application — the "Cell Painting Embedding Explorer"
│   ├── app.py                       #   App factory; wires DataStore + SimilarityIndex + routes
│   ├── routes.py                    #   All HTTP routes: search, well detail, UMAP JSON, attention PNGs
│   ├── data_store.py                #   Loads app_data/*.parquet + *.npy once at startup
│   ├── similarity.py                #   Cosine-similarity search (FAISS or numpy fallback) + copy logic
│   ├── moa_descriptions.json        #   191 curated, cited MoA descriptions for the sidebar
│   ├── requirements.txt             #   flask, pandas, numpy, pyarrow, Pillow
│   ├── static/
│   │   ├── css/style.css            #     All app styling (single stylesheet)
│   │   ├── js/map.js                #     Plotly UMAP rendering, search, legend, selection
│   │   ├── js/onboarding.js         #     First-run guided tour (see "The Web Application")
│   │   └── favicon.svg
│   └── templates/
│       ├── base.html                #     Page shell (navbar + Bootstrap/CSS includes)
│       ├── index.html               #     The single-page app + all info dialogs
│       └── partials/_right_sidebar.html  # Server-rendered well-detail partial (AJAX target)
│
└── literature/                      # (not tracked in git — see below) reference PDF(s)
```

A few structural notes:

- `models/__pycache__/*.pyc` is technically tracked in git history but is a
  stray compiled-bytecode artifact, not source — ignore it; it isn't
  meaningful project structure.
- `data/raw/` contains **example/sample metadata files for one plate only**
  (`BR00116991`) — small enough to check in for reference. The full raw
  dataset (24 plates' worth of images + metadata) is never committed; see the
  next section.
- `literature/` (containing the paper this project is based on) and the bulk
  of `data/`, `embeddings/`, and per-run `checkpoints/` exist on local/cluster
  disk but are **not tracked in git** — they're excluded from the structure
  tree above for that reason, but referenced throughout this document since
  they're central to how the pipeline actually runs.

## Where the Data Lives: Repo vs. Cluster

This is a research pipeline built around a SLURM-managed GPU cluster (paths
throughout the codebase reference `/scratch/creighton.jo/cellpainting` for
scratch space and `/shared/EL9/explorer/anaconda3/...` for the shared conda
install), with only small, derived artifacts checked into git. Concretely:

| Location | What lives there | In git? |
|---|---|---|
| `s3://cellpainting-gallery/cpg0000-jump-pilot/` | The full JUMP Cell Painting Consortium dataset (public, `--no-sign-request`) | No — external, public |
| `$CP_OUTPUT_ROOT/data/raw/` (cluster scratch) | Downloaded raw `.tiff` images, `load_data.csv`, platemaps, compound/MoA metadata for the ~24 selected plates | No (multi-hundred-GB) |
| `$CP_OUTPUT_ROOT/data/processed/` (cluster scratch) | `master_metadata.parquet` / `master_metadata_qc.parquet` — the joined, validated metadata table | No (regenerable from raw + code) |
| `$CP_OUTPUT_ROOT/data/tiles_qc/` (cluster scratch) | One preprocessed `.pt` tensor per (plate, well, site) — 5-channel, percentile-normalized image + Otsu threshold — plus optional precomputed tile-bank sidecar files | No (regenerable; large) |
| `$CP_OUTPUT_ROOT/checkpoints/<run_id>/` (cluster scratch) | DINO training checkpoints (`dino_epoch_N.pt`) + training logs | No (large; regenerable by re-running training) |
| `$CP_OUTPUT_ROOT/embeddings/<run_id>/` (cluster scratch) | Extracted per-FOV embeddings (`embeddings_epoch_N.npy`) + aligned `plates_epoch_N.npy` / `wells_epoch_N.npy`, plus attention-map exports | No (regenerable from a checkpoint) |
| `webapp/static/thumbnails/`, `webapp/static/attention/`, `webapp/static/channel_importance.json` | Rendered per-well thumbnails, attention mosaics, channel-importance scores that the running Flask app reads | No (generated by `scripts/generate_web_*.py`; large) |
| `literature/` | Reference paper(s) this project is built on | No |
| **`app_data/`** | The **one deployed data snapshot that is committed to git**: well- and compound-level parquet tables + their embedding matrices, small enough (a few MB total) to version directly | **Yes** |
| Everything else in the tree above | Source code, SLURM job scripts, environment spec, curated reference tables (MoA descriptions, MoA overrides) | **Yes** |

In short: **code and one lightweight, deployable data snapshot (`app_data/`)
live in git; every large or easily-regenerated artifact lives on the
cluster's scratch filesystem**, keyed off two environment variables
(`CP_OUTPUT_ROOT`, and `CP_DATA_ROOT` for the raw download scripts) that every
script/SLURM job in the pipeline reads from.

## End-to-End Pipeline

### 1. Data Acquisition

**Scripts:** `scripts/scan_compound_plates.py`, `scripts/download_compound_plates.py`,
`data/download_metadata.slurm`, `data/download_images.slurm`

The JUMP Cell Painting Gallery spans dozens of datasets (`cpg0000` through
`cpg0047`) and both compound- and genetic-perturbation (ORF/CRISPR) plates.
Acquisition happens in three stages:

1. **`scan_compound_plates.py`** performs a read-only inventory scan across
   every configured Cell Painting Gallery dataset, streaming each
   `barcode_platemap.csv` straight into memory (nothing is written to disk
   except the scan's own summary/CSV reports) and flagging any plate whose
   `Plate_Map_Name` contains "compound" — i.e. a chemical-perturbation plate,
   as opposed to an ORF/CRISPR genetic-perturbation plate. Output:
   `analysis/outputs/compound_plates.csv`, the authoritative inventory
   everything downstream selects from.
2. **`download_compound_plates.py`** selects a fixed, **deterministic,
   diversity-maximizing** set of plates (24 by default) from that inventory —
   a round-robin over `(dataset, experiment)` groups so that every available
   experiment contributes before any experiment contributes a second plate,
   rather than randomly sampling (which could concentrate on one experiment).
   Every barcode is independently re-confirmed as a real, non-empty S3
   acquisition before download. For this project, only `cpg0000-jump-pilot`
   (`source_4`) is configured.
3. **`data/download_metadata.slurm`** / **`data/download_images.slurm`** are
   the actual SLURM jobs used to fetch metadata and images (a fixed,
   hand-synced 24-plate list, `--array=0-23` for images so each plate
   downloads in parallel). Every transfer is atomic (download to `.tmp`, then
   rename) and safely resumable/idempotent, since interrupted cluster jobs are
   the normal case, not the exception.

The plates actually selected span three related JUMP-Pilot experiments —
`2020_11_04_CPJUMP1`, `2020_11_04_CPJUMP1_DL`, and
`2020_12_08_CPJUMP1_Bleaching` — all using the `JUMP-Target-1_compound_platemap`
plate design (~300 unique compounds, one fixed reference plate layout, imaged
across multiple replicate plates).

Two dataset-global reference files are also pulled directly from S3:
`JUMP-Target-1_compound_metadata.tsv` (compound ↔ InChIKey ↔ gene target ↔
SMILES) and `JUMP-Target-1_compound_metadata_targets.tsv`. Mechanism-of-action
labels themselves come from a file that is **not** on the Cell Painting
Gallery bucket at all — the Broad Institute's **Drug Repurposing Hub**
annotation file (`repo-drug-annotation-20200324.txt`) — supplemented by a
small, hand-curated fallback table, `data/raw/moa_overrides.tsv` (15 entries),
for compounds the Repurposing Hub file is missing or leaves blank, each
sourced individually (Selleck Chemicals, MedChemExpress, PMC/NCBI — see the
file itself for the per-compound citation).

### 2. Metadata Construction

**Script:** `datasets/build_metadata_table.py`

For every downloaded `(experiment, acquisition_id)` pair, this script:

1. Parses the acquisition ID (`BR00117006__2020-11-03T19_45_39-Measurement1`
   → barcode `BR00117006`, measurement `1`) and, where present, a
   human-readable timepoint (`Day1`, `2Weeks`, …) from the experiment folder
   name.
2. Resolves the correct **per-plate platemap** via that experiment's
   `barcode_platemap.csv` (a plate's design is *not* assumed — different
   Cell Painting Gallery datasets use different platemap conventions).
3. Loads that plate's imaging index (`load_data.csv`) and joins it against
   the platemap (well → compound) and the global compound metadata (compound
   → InChIKey/SMILES/target), then attaches the corresponding MoA (joined
   case-insensitively, since `pert_iname` casing differs slightly between
   the compound metadata and MoA annotation files) and the MoA-overrides
   fallback.
4. Attaches actual image file paths for all 5 channels per (plate, well,
   site) by indexing the raw `.tiff` filenames on disk (parsed well
   position, site, and channel directly from filenames like
   `r01c01f01p01-ch1sk1fk1fl1.tiff`).
5. Validates the join (checks for missing `broad_sample`, duplicate
   `(plate, well, site)` rows, image-path coverage) and raises rather than
   silently producing a broken table if more than half the rows fail to
   resolve a compound.

Output: `master_metadata.parquet` — one row per (plate, well, site, channel
set), the single source of truth every later stage reads from.

### 3. Image Preprocessing

**Script:** `datasets/preprocess_dataset.py`

For each row in the master metadata table, the 5 channel `.tiff` files
(mitochondria, AGP, RNA, ER, DNA — in that fixed order) are loaded and:

1. **Per-channel, per-image percentile normalization**: each channel is
   clipped to its own [0.01, 99.9] percentile range and rescaled to `[0, 1]`
   — the paper-faithful normalization scheme (see [Kim et al.
   (2025)](#sources--references)), applied independently per image rather
   than with dataset-wide statistics.
2. **Otsu thresholding** of the DNA channel (nucleus stain), used later as a
   cheap, unsupervised foreground/background split for cropping — if a crop
   contains too little DNA-channel signal above this threshold, it's
   considered background and rejected.

No images are dropped for quality at this stage — every row is preserved.
Each processed sample is saved as a `.pt` tensor payload (image + Otsu
threshold + plate/well/site/MoA) and indexed as `master_metadata_qc.parquet`
(the "QC" table `dataset.py` actually reads from). Optionally,
**`datasets/precompute_tiles.py`** pre-generates a bank of ~20 diverse
224×224 foreground-aware tiles per image offline (rejecting near-duplicate
crops via IoU) so training can sample from a fixed tile bank instead of doing
foreground-aware cropping online every step.

### 4. Dataset & Sampling

**Files:** `datasets/dataset.py`, `datasets/sampler.py`

`CellPaintingDataset` is the PyTorch `Dataset` everything trains against. It
can return either a full preprocessed image (`return_full_image=True`, used
during training so foreground cropping happens on-GPU, batched, in
`train_dino.py`) or a precomputed tile bank (`use_tiles=True`, reading the
`_tiles.pt` sidecar files from `precompute_tiles.py`). `MoASampler` builds
MoA-/compound-indexed lookup tables for MoA-balanced or compound-replicate
sampling strategies; as noted in the code, MoA-balanced sampling is currently
disabled in favor of uniform sampling (see
[Known Limitations](#known-limitations)).

### 5. Model & Training

**Files:** `models/dino/dino.py`, `models/dino/dino_loss.py`,
`models/dino/train_dino.py`, `models/config.py`

**Architecture (`CellPaintingViT`):** a `timm`-provided **ViT-S/14
(DINOv2-pretrained)** backbone with its patch-embedding `Conv2d` replaced to
accept **5 input channels** instead of 3 — the 3 pretrained RGB filters are
copied directly, and the 2 new channels are initialized from the *mean* of
those 3 pretrained filters (rather than random init), so the new channels
start from a sensible, pretraining-informed point. Only the `CLS` token is
used as the image representation (384-dimensional).

**Projection head (`DINOHead`):** the canonical DINO head — a 3-layer MLP
(384 → 2048 → 2048 → 256), L2-normalized bottleneck, then a **weight-normalized**
linear layer to a 20,000-dimensional output (`weight_g` frozen, matching the
paper's `norm_last_layer=True`).

**Loss (`DINOLoss`):** cross-entropy between a sharpened/centered teacher
distribution and a temperature-scaled student distribution, with:
- A **linear teacher-temperature warmup** (0.01 → 0.04 over 30 epochs, paper
  spec).
- An **EMA-centered** teacher output (`center_momentum` default 0.95, though
  training actually applies an "effective" momentum of 0.95 post-teacher-EMA
  update — see `train_dino.py`).

**Training loop (`train_dino.py`)** implements the full DINO multi-crop
recipe:
- **2 global crops** (224×224) + **6 local crops** (96×96, resized to 224 for
  the ViT) per image, selected via **Otsu-threshold rejection sampling**
  (`utils/foreground_crop.py`) requiring ≥1% DNA-channel foreground, batched
  and GPU-resident.
- A custom **augmentation** function (random H/W flips, additive intensity
  shift, gamma brightness change) applied independently to student and
  teacher views.
- **EMA teacher update** with momentum cosine-annealed 0.996 → 0.9998 over
  training.
- **Cosine LR schedule** with a 20-epoch linear warmup (paper spec).
- **Weight decay** linearly increased 0.04 → 0.4 over training (paper spec).
- Mixed-precision (`torch.autocast` + `GradScaler`), gradient accumulation
  (default 4 steps, effective batch size = 32 × 4 = 128), gradient clipping
  (max norm 3.0), and full checkpoint/optimizer/scheduler resume support —
  training runs on a preemptible SLURM GPU partition with `--requeue`, so
  interruption-safe resume is load-bearing, not a convenience feature.

Default hyperparameters (`models/config.py`):

| Key | Value |
|---|---|
| `lr` | `1e-4` |
| `n_epochs` | `200` |
| `batch_size` (per step) | `32` |
| `accum_steps` | `4` (effective batch = 128) |
| `weight_decay` | `0.04 → 0.4` (linear) |
| `num_workers` | `8` |

An earlier/experimental variant lives in `models/scvg/` (a SupCon/VICReg-style
DINO variant) — it is **not** the model used for any reported results in this
README; it's kept in the repo for reference.

### 6. Embedding Extraction

**Script:** `analysis/extract_embeddings.py`

Given a training run directory, this loads the (by default, latest) student
encoder checkpoint and, for **every field of view (FOV)** in the dataset:

1. Tiles the full-resolution image into non-overlapping 224×224 crops.
2. Keeps only crops with ≥1% DNA-channel foreground (same Otsu criterion as
   training) — falling back to a single center crop if none pass.
3. Embeds each surviving crop, L2-normalizes each, and **mean-pools** them
   into one embedding per FOV.

Output: `embeddings_epoch_N.npy` (FOV-level embeddings) aligned with
`plates_epoch_N.npy` / `wells_epoch_N.npy`. The same crop-selection logic is
reused (not reimplemented) by `analysis/channel_ablation.py` (per-channel
importance via ablation: zero/mean-fill one channel, re-embed, measure cosine
distance from baseline) and `analysis/extract_attention_maps.py` (extracts
last-block CLS→patch attention maps via a monkey-patched attention forward
pass that exposes softmax weights timm's fused SDPA path otherwise hides).

### 7. Post-Processing

**File:** `utils/postprocessing.py`

Following the paper's evaluation recipe, raw well-level embeddings are
post-processed before any similarity comparison:

1. **Per-plate MAD (median absolute deviation) normalization**, fit
   independently on each plate's own negative-control wells — removes
   plate/batch-level shift and scale before pooling across plates.
2. **Sphering (PCA whitening)**, fit once on the pooled, MAD-normalized
   controls across *all* plates — rebalances the feature space so no single
   learned direction dominates similarity comparisons.
3. **L2 normalization**, so cosine similarity reduces to a dot product.

This is what `scripts/prepare_phase1_data.py` (webapp data) and
`analysis/replicate_correlation.py --postprocess` (evaluation diagnostic)
both call.

### 8. Evaluation & Diagnostics

**Script:** `analysis/replicate_correlation.py`

The core sanity check for "did the model learn anything phenotypically
real": for every compound imaged on **2+ different plates**, compute the
cosine similarity between its cross-plate replicate wells, and compare that
distribution against a random-pair baseline (different compounds, any
plates). A Mann-Whitney U test and an enrichment ratio (mean replicate
similarity ÷ mean random similarity) quantify whether same-compound wells are
more self-similar than chance — the primary evidence of learned, generalizing
morphological structure rather than memorized per-image or per-plate noise.
See [Model Results](#model-results) for the actual numbers this pipeline
produced.

`notebooks/embedding_analysis.ipynb` additionally parses raw SLURM training
logs (loss/`cos_sim`/embedding-std curves) and runs UMAP + basic sanity
checks directly on extracted embeddings; `notebooks/metadata_eda.ipynb`
validates the metadata-table joins themselves (row growth, missing-value
rates, well/site coverage) before any model is even involved.

### 9. Web App Data Preparation

**Scripts:** `scripts/prepare_phase1_data.py`, `scripts/prepare_phase2_compounds.py`,
`scripts/generate_web_thumbnails.py`, `scripts/generate_web_attention_maps.py`,
`scripts/generate_web_channel_importance.py`

These are one-time (per checkpoint), offline steps that turn a trained
model's raw tile-level embeddings into everything the Flask app actually
reads at runtime — none of this happens live in the running app:

1. **`prepare_phase1_data.py`** — mean-pools tile embeddings up to one
   embedding per **well**, applies the post-processing above (fit on
   negative-control wells), and fits **two independent UMAP projections**
   (2D and 3D — the 3D view is a *separate* `n_components=3` fit, not a 3rd
   axis bolted onto the 2D layout) with `metric="cosine"`. Writes
   `app_data/wells.parquet` + `app_data/well_embeddings.npy`.
2. **`prepare_phase2_compounds.py`** — further mean-pools well embeddings up
   to one embedding per **compound** (control wells, which have no
   `broad_sample`, are naturally excluded via `groupby` dropping NaN keys).
   Writes `app_data/compounds.parquet` + `app_data/compound_embeddings.npy`.
3. **`generate_web_thumbnails.py`** — for each well, picks the single site
   whose own embedding is closest (cosine) to that well's mean-embedding
   centroid (i.e. the most "typical" image for that well), composites its 5
   channels into one false-color RGB image (Mito=red, AGP=yellow, RNA=magenta,
   ER=green, DNA=blue) and saves it as a 256×256 WebP.
4. **`generate_web_attention_maps.py`** — for that same representative site,
   tiles the full image, runs each foreground crop through the model with
   attention-weight extraction enabled, and assembles a spatially-aligned
   attention mosaic (kept at raw ViT patch-grid resolution; resizing/color
   mapping happens at request time in `routes.py`, not baked in ahead of
   time).
5. **`generate_web_channel_importance.py`** — computes per-well,
   per-channel ablation-importance scores (reusing
   `analysis/channel_ablation.py` directly) for the sidebar's channel bar
   chart, writing one combined JSON rather than one file per well.

## The Web Application

The **Cell Painting Embedding Explorer** (`webapp/`) is a single-page Flask
app: one HTML page, a Plotly-rendered 2D/3D UMAP scatter plot, and a
right-hand detail sidebar that updates via small JSON/HTML-partial API calls
— no client-side framework, no build step.

**Backend (`webapp/*.py`):**
- `app.py` — Flask app factory; loads `app_data/` once into memory via
  `DataStore` and builds a `SimilarityIndex` at startup (not per-request).
- `data_store.py` — thin wrapper around the well/compound parquet + `.npy`
  pairs, with O(1) `well_id`/`compound_id` lookup dicts.
- `similarity.py` — cosine-similarity nearest-neighbor search (uses
  [FAISS](https://github.com/facebookresearch/faiss)'s exact
  `IndexFlatIP` if installed, otherwise a plain NumPy dot-product + argsort —
  fine at this dataset's scale of a few thousand wells), plus the
  "Neighborhood Summary" statistics and the cautiously-worded, template-based
  interpretation text shown in the sidebar (no LLM — every number and every
  sentence is deterministic and derived directly from the stats it displays).
- `routes.py` — `/` (the page itself), `/api/search` (free-text resolution
  across wells/compounds/MoAs, with disambiguation when a query is
  ambiguous), `/api/well/<id>` (renders the right-sidebar partial),
  `/api/attention/<id>.png` (renders an on-demand heatmap PNG from a stored
  attention mosaic), and `/api/umap` (the full point cloud as JSON for
  Plotly).

**Frontend (`webapp/static/`, `webapp/templates/`):**
- `js/map.js` — builds/re-renders the Plotly 2D (`scattergl`) or 3D
  (`scatter3d`) trace set on every color-mode/search/dimension change, a
  custom multi-column legend (click to isolate, double-click to hide),
  client-side search filtering, and point-click → sidebar-fetch wiring.
- `js/onboarding.js` — a self-contained, first-run guided tour (localStorage-
  gated, replayable via "Restart Tutorial") that walks a new visitor through
  what a dot represents, what distance and color mean, and how to click a
  point — reusing the app's real selection/pan functions rather than faking
  any of it.
- `templates/index.html` — the entire page, plus three in-depth educational
  dialogs ("How to Read This Map," "About the Project," "Technical Details")
  covering everything from what an embedding is to the model's known
  limitations, each citing further reading.

## Model Results

These figures come from evaluating the checkpoint currently backing the
deployed `app_data/` snapshot (epoch 200 of run `070226_135708`) with
`analysis/replicate_correlation.py --postprocess` and the notebook-based
sanity checks in `notebooks/embedding_analysis.ipynb`, against the current
dataset (**5,102 wells, 306 compounds, 196 distinct MoA labels across 14
plates**, ~4.7% of wells unannotated).

| Diagnostic | Result | Interpretation |
|---|---|---|
| Cross-plate replicate similarity vs. random baseline (raw, pre-postprocess) | enrichment ≈ **1.14×** | Weak but present cross-plate consistency |
| Cross-plate replicate similarity vs. random baseline (**after** MAD + sphering) | enrichment ≈ **7.17×** | Post-processing substantially strengthens the signal |
| Replicate Z-score, raw → post-processed | **0.13 → 0.62** | Same direction: post-processing helps a lot |
| MoA-neighbor purity, raw → post-processed | **0.062 → 0.113** | Improves, but modest in absolute terms |
| Plate identity, linear-probe accuracy | **0.717** (chance ≈ 0.071) | A measurable, non-trivial batch/plate effect remains in the embedding |
| Compound-held-out MoA probe accuracy | **0.059** (chance ≈ 0.016, ~3.7×) | Above chance, but a long way from a reliable classifier |
| Compound nearest-neighbor retrieval@5 | **0.053** (chance ≈ 0.0036, ~15×) | The clearest positive signal — retrieval is well above chance |

**Reading these numbers honestly:** the model has learned *something* real —
same-compound replicates land measurably closer together than random pairs,
and that signal survives (and is strengthened by) standard batch-correction
post-processing. But the absolute effect sizes are modest, and a sizeable,
measurable plate/batch effect persists even after correction. Treat any
individual cluster or "this compound's neighbors share its MoA" observation
in the web app as a **hypothesis worth investigating further**, not as
independent biological confirmation — see [Known
Limitations](#known-limitations) and the app's own "Technical Details" and
"How to Read This Map" panels, which state these same caveats to end users
directly.

## Environment & Setup

The full dependency spec lives in `env/environment.yml` (conda) and
`webapp/requirements.txt` (the much lighter Flask app's own deps).

```bash
# Training / analysis environment
conda env create -f env/environment.yml
conda activate cellpainting

# Webapp only (much lighter — no PyTorch/CUDA needed to just browse results)
pip install -r webapp/requirements.txt
```

Key training-side dependencies: PyTorch + `torchvision`/`torchaudio` (CUDA
12.1 build), `timm` (for the pretrained DINOv2 ViT-S/14 backbone),
`umap-learn`, `scikit-image` (Otsu thresholding), `scikit-learn` (PCA
whitening), `tifffile`, and `awscli` (public S3 downloads,
`--no-sign-request`, no AWS account needed).

Every training/analysis script expects `CP_OUTPUT_ROOT` (and the raw-download
scripts additionally `CP_DATA_ROOT`) to be set to a scratch directory with
enough space for images, processed tiles, checkpoints, and embeddings — see
`models/dino/train_dino.slurm` for the canonical environment setup used on
the cluster this project was developed on.

## Running the Pipeline End to End

Roughly, in order (each stage's script/SLURM job is described in detail
above):

```bash
# 1. Discover which plates in the Gallery are compound-perturbation plates
python scripts/scan_compound_plates.py

# 2. Download a diverse, fixed set of compound plates (images + metadata)
python scripts/download_compound_plates.py --dry-run   # inspect selection first
python scripts/download_compound_plates.py              # then actually download
#    (or, equivalently: sbatch data/download_metadata.slurm && sbatch data/download_images.slurm)

# 3. Join everything into one metadata table
sbatch datasets/build_metadata_table.slurm

# 4. Normalize images + compute Otsu thresholds
sbatch datasets/preprocess.slurm
sbatch datasets/precompute_tiles.slurm   # optional: precomputed tile banks

# 5. Train DINO
sbatch models/dino/train_dino.slurm      # resumable; requeues itself on preemption

# 6. Extract embeddings from a trained checkpoint
sbatch analysis/embed.slurm              # analysis/extract_embeddings.py --run_dir ...

# 7. Evaluate
python analysis/replicate_correlation.py --embeddings ... --metadata ... --postprocess

# 8. Build the webapp's data snapshot from that checkpoint's embeddings
python scripts/prepare_phase1_data.py --emb embeddings/<run>/embeddings_epoch_N.npy
python scripts/prepare_phase2_compounds.py
sbatch scripts/generate_web_thumbnails.slurm
sbatch scripts/generate_web_attention_maps.slurm
sbatch scripts/generate_web_channel_importance.slurm

# 9. Run the web app
cd webapp && python app.py   # http://127.0.0.1:5050
```

## Known Limitations

Restated here from the web app's own "Technical Details" panel, since they
apply equally to this repository as a whole:

- **Single, small compound library.** The map/model is built on one fixed
  reference plate design (~300 compounds) imaged across replicate plates —
  not the broad chemical diversity of larger benchmark studies. Results here
  should not be read as a general test of the model's chemistry coverage.
- **A measurable plate/batch effect remains** even after MAD + sphering
  post-processing (plate-identity linear-probe accuracy ≈0.72 vs. ≈0.07
  chance) — some of what the embedding encodes is technical variation, not
  biology.
- **MoA-neighbor agreement is above chance but modest in absolute terms** —
  treat clustering patterns as suggestive, not diagnostic.
- **2D UMAP is for visualization only.** Similarity/neighbor calculations in
  the web app are performed in the full, post-processed embedding space, not
  from 2D coordinates — a point can look close on the map without being a
  true nearest neighbor, and vice versa (this is an expected property of any
  2D projection of high-dimensional data, not a bug).
- **Known gaps vs. the paper this project follows:** no dataset-wide
  channel mean/std normalization step after per-image percentile
  normalization; `ViT-S/14` (DINOv2) is used instead of the paper's `ViT-S/16`
  (DINOv1) backbone; MoA-balanced sampling (`MoASampler`) is implemented but
  currently disabled in favor of uniform sampling.
- **This is a proof of concept**, not a deployed or validated
  screening/discovery tool — see [Motivation](#motivation) for what it is
  trying to demonstrate instead.

## Sources & References

**Primary paper this project builds on** (PDF in `literature/`, not tracked
in git):

> Kim, V., Adaloglou, N., Osterland, M., Morelli, F. M., Halawa, M., König,
> T., Gnutt, D., & Marin Zapata, P. A. (2025). Self-supervision advances
> morphological profiling by unlocking powerful image representations.
> *Scientific Reports*, 15, 4876.
> [https://doi.org/10.1038/s41598-025-88825-4](https://doi.org/10.1038/s41598-025-88825-4)

**Model architecture & training recipe:**

> Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., &
> Joulin, A. (2021). Emerging Properties in Self-Supervised Vision
> Transformers. *ICCV 2021*.
> [arXiv:2104.14294](https://arxiv.org/abs/2104.14294)

> Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X.,
> Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S.,
> Uszkoreit, J., & Houlsby, N. (2020/2021). An Image is Worth 16x16 Words:
> Transformers for Image Recognition at Scale. *ICLR 2021*.
> [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

**Dimensionality reduction (used for the web app's UMAP view):**

> McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold
> Approximation and Projection for Dimension Reduction.
> [arXiv:1802.03426](https://arxiv.org/abs/1802.03426) ·
> [documentation](https://umap-learn.readthedocs.io/)

**Data source:**

> JUMP Cell Painting Consortium. Cell Painting Gallery, dataset
> `cpg0000-jump-pilot`.
> [https://jump-cellpainting.broadinstitute.org/](https://jump-cellpainting.broadinstitute.org/)

> Broad Institute. Morphological Profiling / Cell Painting overview.
> [https://www.broadinstitute.org/imaging/morphological-profiling](https://www.broadinstitute.org/imaging/morphological-profiling)

**Mechanism-of-action annotations:**

> Broad Institute Drug Repurposing Hub — `repo-drug-annotation-20200324.txt`.
> [https://www.broadinstitute.org/repurposing](https://www.broadinstitute.org/repurposing)
>
> Supplemented by 15 hand-curated entries in `data/raw/moa_overrides.tsv`,
> each individually sourced from Selleck Chemicals, MedChemExpress, or
> PMC/NCBI (see that file for the per-compound citation), and by the 191
> curated, individually-cited MoA descriptions in
> `webapp/moa_descriptions.json` (primarily Wikipedia and NCBI/PMC — see that
> file's own `sources` field per entry).

**Background reading linked from the web app itself:**

> TensorFlow. Introduction to word embeddings.
> [https://www.tensorflow.org/text/guide/word_embeddings](https://www.tensorflow.org/text/guide/word_embeddings)

> TensorFlow Embedding Projector (inspiration for the interactive embedding
> explorer UI). [https://projector.tensorflow.org/](https://projector.tensorflow.org/)

**Software:**

> `timm` (PyTorch Image Models) — pretrained ViT-S/14 (DINOv2) backbone.
> `umap-learn`, `scikit-learn`, `scikit-image`, `tifffile`, `Flask`, `Plotly.js`,
> and (optionally) [FAISS](https://github.com/facebookresearch/faiss) for the
> web app's nearest-neighbor search.

## Acknowledgements

Built on publicly available data from the JUMP Cell Painting Consortium and
methodology from Kim et al. (2025) and Caron et al. (2021). Training and
large-scale data processing were run on a SLURM-managed HPC cluster.

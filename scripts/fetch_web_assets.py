"""
Downloads and extracts the webapp's large per-well static assets
(thumbnails/, attention/) from external tarball URLs at deploy time.

These directories are intentionally NOT committed to git (~400MB across
~27,000 small files -- see the repo's deployment notes). Instead they're
uploaded as GitHub Release binary assets, and this script pulls them down
into place before the Flask app starts.

Idempotent: if a target directory already exists and is non-empty (e.g. in
local dev, where these are already populated on disk), it's left alone and
nothing is downloaded. Safe to run on every deploy.

Usage:
    THUMBNAILS_TAR_URL=... ATTENTION_TAR_URL=... python scripts/fetch_web_assets.py

Both env vars must point at a .tar.gz built the same way this repo's
dist/thumbnails.tar.gz / dist/attention.tar.gz were: the archive's top-level
entry is the directory name itself (thumbnails/ or attention/), so
extracting at webapp/static/ reproduces webapp/static/thumbnails/ or
webapp/static/attention/ directly -- e.g.:
    tar -czf thumbnails.tar.gz -C webapp/static thumbnails
"""
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "webapp" / "static"

TARGETS = {
    "thumbnails": os.environ.get("THUMBNAILS_TAR_URL"),
    "attention": os.environ.get("ATTENTION_TAR_URL"),
}


def fetch_and_extract(name: str, url: str):
    target_dir = STATIC_DIR / name
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"[{name}] {target_dir} already populated ({sum(1 for _ in target_dir.iterdir())} entries) -- skipping")
        return

    if not url:
        print(f"[{name}] no URL provided (set {'THUMBNAILS' if name == 'thumbnails' else 'ATTENTION'}_TAR_URL) -- skipping")
        return

    tmp_path = STATIC_DIR / f"_{name}.tar.gz"
    print(f"[{name}] downloading {url} ...")
    urllib.request.urlretrieve(url, tmp_path)
    size_mb = tmp_path.stat().st_size / 1024 / 1024
    print(f"[{name}] downloaded {size_mb:.1f} MB, extracting into {STATIC_DIR} ...")

    with tarfile.open(tmp_path) as tf:
        tf.extractall(STATIC_DIR)

    tmp_path.unlink()
    n_files = sum(1 for _ in target_dir.iterdir())
    print(f"[{name}] done -- {n_files} files in {target_dir}")


def main():
    for name, url in TARGETS.items():
        fetch_and_extract(name, url)


if __name__ == "__main__":
    main()

# sampler.py
import torch
from collections import defaultdict
import random

# datasets/sampler.py

import pandas as pd
from collections import defaultdict
from pathlib import Path
import random

class MoASampler:
    def __init__(self, processed_dir, metadata_path):
        self.moa_to_files      = defaultdict(list)
        self.compound_to_files = defaultdict(list)
        processed_dir = Path(processed_dir)
        print("Loading metadata...")
        df = pd.read_parquet(metadata_path)

        print("Building MoA and compound indices...")
        for idx, row in df.reset_index().iterrows():
            moa = row.get("moa", "unknown")
            if pd.isna(moa):
                moa = "unknown"

            compound = row.get("broad_sample", "unknown")
            if pd.isna(compound):
                compound = "unknown"

            file_path = processed_dir / f"{row['index']}.pt"
            if file_path.exists():
                self.moa_to_files[str(moa)].append(str(file_path))
                self.compound_to_files[str(compound)].append(str(file_path))

        self.moa_list = list(self.moa_to_files.keys())
        # exclude "unknown" — these have no specific compound identity
        self.compound_list = [c for c in self.compound_to_files if c != "unknown"]
        print(f"Loaded {len(self.moa_list)} MOAs | {len(self.compound_list)} compounds")

    def sample_moa(self):
        moa = random.choice(self.moa_list)
        file = random.choice(self.moa_to_files[moa])
        return file, moa

    def sample_moa_k(self, k):
        """Sample k files from the same randomly chosen MoA."""
        moa = random.choice(self.moa_list)
        files = random.choices(self.moa_to_files[moa], k=k)
        return files, moa

    def sample_compound_k(self, k):
        """Sample k files from the same randomly chosen compound (broad_sample)."""
        compound = random.choice(self.compound_list)
        files = random.choices(self.compound_to_files[compound], k=k)
        return files, compound
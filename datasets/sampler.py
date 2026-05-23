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
        self.moa_to_files = defaultdict(list)
        processed_dir = Path(processed_dir)
        print("Loading metadata...")
        df = pd.read_parquet(metadata_path)

        print("Building MOA index...")
        for idx, row in df.reset_index().iterrows():
            moa = row.get("moa", "unknown")
            if pd.isna(moa):
                moa = "unknown"

            file_path = processed_dir / f"{row['index']}.pt"
            if file_path.exists():
                self.moa_to_files[str(moa)].append(str(file_path))

        self.moa_list = list(self.moa_to_files.keys())
        print(f"Loaded {len(self.moa_list)} MOAs")

    def sample_moa(self):
        moa = random.choice(self.moa_list)
        file = random.choice(self.moa_to_files[moa])
        return file, moa
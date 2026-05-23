# sampler.py
from collections import defaultdict
import torch
from pathlib import Path
import random

class MoASampler:
    def __init__(self, processed_dir):
        self.moa_to_files = defaultdict(list)

        files = list(Path(processed_dir).rglob("*.pt"))

        for f in files:
            sample = torch.load(f)
            moa = sample.get("moa", "unknown")
            self.moa_to_files[moa].append(str(f))

        self.moa_list = list(self.moa_to_files.keys())

    def sample_moa(self):
        moa = random.choice(self.moa_list)
        file = random.choice(self.moa_to_files[moa])
        return file, moa
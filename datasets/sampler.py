# sampler.py
from collections import defaultdict
import torch
from pathlib import Path
import random

from collections import defaultdict
import pandas as pd
import random

class MoASampler:
    def __init__(self, processed_dir):
        self.moa_to_files = defaultdict(list)

        for f in Path(processed_dir).rglob("*.pt"):
            sample = torch.load(f)
            moa = sample["moa"]
            self.moa_to_files[moa].append(str(f))

        self.moa_list = list(self.moa_to_files.keys())

    def sample_moa(self):
        moa = random.choice(self.moa_list)
        file = random.choice(self.moa_to_files[moa])
        return file, moa
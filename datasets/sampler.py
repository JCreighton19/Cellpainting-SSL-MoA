# sampler.py
import torch
from collections import defaultdict
import random

class MoASampler:
    def __init__(self, files):
        self.moa_to_files = defaultdict(list)
        print(f"Scanning {len(files)} tile files...")
        skipped = 0

        for f in files:
            try:
                sample = torch.load(f)
                moa = sample.get("moa", "unknown")
                self.moa_to_files[str(moa)].append(str(f))

            except Exception as e:
                skipped += 1
                print(f"Error loading {f}: {e}")

        self.moa_list = list(self.moa_to_files.keys())

        print(f"Loaded {len(self.moa_list)} MOAs")
        print(f"Skipped {skipped} bad files")

    def sample_moa(self):
        moa = random.choice(self.moa_list)
        file = random.choice(self.moa_to_files[moa])
        return file, moa
import torch
from torch.utils.data import Dataset
import tifffile as tiff
import numpy as np
import random
from pathlib import Path

class CellPaintingDataset(Dataset):
    def __init__(self, processed_dir, tile_size=224):
        self.files = list(Path(processed_dir).glob("*.pt"))
        self.tile_size = tile_size

    def __len__(self):
        return len(self.files)

    def _augment(self, x):
        if random.random() < 0.5:
            x = torch.flip(x, dims=[1])
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])

        x = x + torch.randn_like(x) * 0.02
        if random.random() < 0.2:
            ch = random.randint(0, x.shape[0] - 1)
            x[ch] += torch.randn_like(x[ch]) * 0.05

        return x

    def __getitem__(self, idx):
        img = torch.load(self.files[idx])  # (C,H,W)

        C, H, W = img.shape
        ts = self.tile_size
        r = random.randint(0, H - ts)
        c = random.randint(0, W - ts)

        tile = img[:, r:r+ts, c:c+ts]
        tile = self._augment(tile)

        return {"image": tile}
import torch
from torch.utils.data import Dataset
import tifffile as tiff
import numpy as np
import pandas as pd
from pathlib import Path
import random
import os

class CellPaintingDataset(Dataset):
    def __init__(
            self,
            metadata_path,
            data_root,
            transform=None,
            channels=None,
            tile_size=224
        ):
        cp_root = os.environ.get("CP_DATA_ROOT", data_root)
        self.project_root = Path(cp_root).resolve()
        self.metadata = pd.read_parquet(metadata_path)
        self.transform = transform
        self.channels = channels if channels is not None else [1,2,3,4,5]
        self.tile_size = tile_size

        self._cache = {}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        for _ in range(10):
            row = self.metadata.iloc[idx]
            paths = [
                row["url_origdna"],
                row["url_origagp"],
                row["url_origmito"],
                row["url_origer"],
                row["url_origrna"],
            ]

            channels = []
            for p in paths:
                if p not in self._cache:
                    self._cache[p] = tiff.imread(str(p)).astype(np.float32)
                channels.append(self._cache[p])

            image = np.stack(channels, axis=0)

            image = self._normalize_channels(image)
            C, H, W = image.shape
            r = random.randint(0, H - self.tile_size)
            c = random.randint(0, W - self.tile_size)
            tile = image[:, r:r + self.tile_size, c:c + self.tile_size]
            if not self._is_informative(tile):
                continue

            tile = torch.from_numpy(tile).float()
            if self.transform:
                tile = self.transform(tile)

            return {
                "image": tile,
                "compound": row["pert_iname"],
                "broad_sample": row["broad_sample"],
                "plate": row["plate"],
                "well": row["well"],
                "site": row["site"],
                "row": r,
                "col": c
            }

        raise RuntimeError("Failed to sample informative tile")

    def _normalize_channels(self, image):
        normed = np.zeros_like(image)
        for c in range(image.shape[0]):
            x = np.log1p(image[c])
            p1 = np.percentile(x, 1)
            p99 = np.percentile(x, 99)
            x = (x - p1) / (p99 - p1 + 1e-6)

            normed[c] = x

        return normed

    def _is_informative(self, tile):
        signal = np.percentile(tile[4], 95) # channel 5=DNA
        variance = tile[4].var()

        return signal > 0.02 and variance > 0.005
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

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        for _ in range(10):
            row = self.metadata.iloc[random.randint(0, len(self.metadata) - 1)]
            paths = [
                row["mito_img_path"],
                row["agp_img_path"],
                row["rna_img_path"],
                row["er_img_path"],
                row["dna_img_path"],
            ]

            image = np.stack(
                [tiff.imread(str(p)).astype(np.float32) for p in paths],
                axis=0
            )

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
        dna = tile[-1]
        signal = np.percentile(dna, 95)
        variance = dna.var()
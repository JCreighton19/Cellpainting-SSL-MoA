import torch
from torch.utils.data import Dataset
import random
import torch.nn.functional as F
from pathlib import Path
import os
import pandas as pd

from datasets.sampler import MoASampler

class CellPaintingDataset(Dataset):
    def __init__(self, processed_dir, tile_size=224, random_crop=True, k_per_class=1, return_full_image=False):
        metadata_path = os.path.join(
            os.environ["CP_OUTPUT_ROOT"],
            "data/processed/master_metadata_qc.parquet"
        )
        self.metadata = pd.read_parquet(metadata_path)
        self.files = self.metadata["pt_path"].tolist()
        self.n_files = len(self.files)
        self.tile_size = tile_size
        self.random_crop = random_crop
        self.k_per_class = k_per_class
        self.return_full_image = return_full_image
        self.sampler = MoASampler(
            processed_dir=processed_dir,
            metadata_path=metadata_path
        )

    def __len__(self):
        return len(self.files)

    @staticmethod
    def sample_foreground_crop(img, tile_size):
        dna = img[4]
        H, W = dna.shape
        ts = tile_size

        small = F.avg_pool2d(
            dna.unsqueeze(0).unsqueeze(0),
            kernel_size=8
        ).squeeze()

        small = small - small.min()
        small = small / (small.max() + 1e-6)
        flat = small.flatten()
        idx = torch.multinomial(flat + 1e-6, 1).item()

        y = idx // small.shape[1]
        x = idx % small.shape[1]
        y = int(y * 8)
        x = int(x * 8)
        r = max(0, min(H - ts, y + random.randint(-ts // 2, ts // 2)))
        c = max(0, min(W - ts, x + random.randint(-ts // 2, ts // 2)))

        return img[:, r:r + ts, c:c + ts]


    def _crop_img(self, img):
        if self.random_crop:
            return self.sample_foreground_crop(img, self.tile_size)
        _, H, W = img.shape
        ts = self.tile_size
        r = (H - ts) // 2
        c = (W - ts) // 2
        return img[:, r:r + ts, c:c + ts]


    def __getitem__(self, idx):
        file = self.files[idx]
        sample = torch.load(file, weights_only=False)
        moa = sample.get("moa", None)
        img = sample["image"]

        return {
            "image": img if self.return_full_image else self._crop_img(img),
            "plate": sample["plate"],
            "well": sample["well"],
            "moa": moa,
            "otsu_mask": sample["otsu_mask"]
        }
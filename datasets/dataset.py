import torch
from torch.utils.data import Dataset
import random
import torch.nn.functional as F
from pathlib import Path
import os

from datasets.sampler import MoASampler

class CellPaintingDataset(Dataset):
    def __init__(self, processed_dir, tile_size=224,
                 augment=True, random_crop=True):
        self.files = list(Path(processed_dir).rglob("*.pt"))
        self.tile_size = tile_size
        self.augment = augment
        self.random_crop = random_crop
        self.sampler = MoASampler(
            processed_dir= processed_dir,
            metadata_path=os.path.join(
                os.environ["CP_OUTPUT_ROOT"],
                "data/processed/master_metadata.parquet"
            )
        )

    def __len__(self):
        return len(self.files)

    # Must be staticmethod or self-aware
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

    def _augment(self, x):
        # flips
        if random.random() < 0.5:
            x = torch.flip(x, dims=[1])
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])

        # mild intensity scaling
        if random.random() < 0.8:
            scale = torch.empty(1).uniform_(0.9, 1.1).item()
            x = x * scale

        # mild noise
        if random.random() < 0.5:
            noise = torch.randn_like(x) * 0.02
            x = x + noise

        return x.clamp(0, 1)

    def __getitem__(self, idx):
        file, moa = self.sampler.sample_moa()
        sample = torch.load(file)

        img = sample["image"]
        plate = sample["plate"]
        well = sample["well"]

        if self.random_crop:
            tile = self.sample_foreground_crop(img, self.tile_size)
        else:
            _, H, W = img.shape
            ts = self.tile_size

            r = (H - ts) // 2
            c = (W - ts) // 2

            tile = img[:, r:r + ts, c:c + ts]

        if self.augment:
            tile = self._augment(tile)

        return {
            "image": tile,
            "plate": plate,
            "well": well,
            "moa": moa
        }
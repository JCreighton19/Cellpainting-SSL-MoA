import torch
from torch.utils.data import Dataset
import random
import torch.nn.functional as F
from pathlib import Path
import os

from datasets.sampler import MoASampler

class CellPaintingDataset(Dataset):
    def __init__(self, processed_dir, tile_size=224, random_crop=True, k_per_class=1):
        self.files = list(Path(processed_dir).rglob("*.pt"))
        self.tile_size = tile_size
        self.random_crop = random_crop
        self.k_per_class = k_per_class

        self.sampler = MoASampler(
            processed_dir=processed_dir,
            metadata_path=os.path.join(
                os.environ["CP_OUTPUT_ROOT"],
                "data/processed/master_metadata.parquet"
            )
        )

    def __len__(self):
        if self.k_per_class > 1:
            return len(self.files) // self.k_per_class
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

    def _load_tile(self, file):
        return self._crop_img(torch.load(file, weights_only=False)["image"])

    def __getitem__(self, idx):
        if self.k_per_class > 1:
            files, moa = self.sampler.sample_moa_k(self.k_per_class)
            sample0 = torch.load(files[0], weights_only=False)
            tile0 = self._crop_img(sample0["image"])
            tiles = torch.stack([tile0] + [self._load_tile(f) for f in files[1:]])
            return {"image": tiles, "plate": sample0["plate"], "well": sample0["well"], "moa": moa}

        file, moa = self.sampler.sample_moa()
        sample = torch.load(file, weights_only=False)
        return {
            "image": self._crop_img(sample["image"]),
            "plate": sample["plate"],
            "well": sample["well"],
            "moa": moa
        }
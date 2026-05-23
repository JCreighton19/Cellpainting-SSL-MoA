import torch
from torch.utils.data import Dataset
import random
import torch.nn.functional as F
from pathlib import Path


class CellPaintingDataset(Dataset):
    def __init__(self, processed_dir, tile_size=224):
        self.files = list(Path(processed_dir).rglob("*.pt"))
        self.tile_size = tile_size

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
        if random.random() < 0.5:
            x = torch.flip(x, dims=[1])
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])

        # Only safe augmentation for now
        return x

    def __getitem__(self, idx):
        sample = torch.load(self.files[idx])

        img = sample["image"]
        plate = sample["plate"]
        well = sample["well"]

        tile = self.sample_foreground_crop(img, self.tile_size)
        tile = self._augment(tile)

        return {
            "image": tile,
            "plate": plate,
            "well": well
        }
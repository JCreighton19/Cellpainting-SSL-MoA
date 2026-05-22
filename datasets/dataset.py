import torch
from torch.utils.data import Dataset
from pathlib import Path

class CellPaintingDataset(Dataset):
    def __init__(self, processed_dir):
        self.files = list(Path(processed_dir).glob("*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = torch.load(self.files[idx])

        C, H, W = img.shape
        r = torch.randint(0, H - 224, (1,)).item()
        c = torch.randint(0, W - 224, (1,)).item()

        tile = img[:, r:r+224, c:c+224]

        return {"image": tile}
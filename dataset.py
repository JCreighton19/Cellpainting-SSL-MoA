import torch
from torch.utils.data import Dataset
import tifffile as tiff
import numpy as np
import pandas as pd
from pathlib import Path
import random

class CellPaintingDataset(Dataset):
    def __init__(
            self,
            metadata_path,
            data_root,
            transform=None,
            channels=None,
            tile_size=224
        ):
        self.project_root = Path(data_root).resolve()
        self.metadata = pd.read_parquet(metadata_path)
        self.transform = transform
        self.channels = channels if channels is not None else [1,2,3,4,5]
        self.tile_size = tile_size

        self._cache = {}  # used to reduce number of disk reads
        self.fields = [
            f for f in self._build_fields()
            if all(self._file_exists(p) for p in f["files"])
        ]

        if len(self.fields) == 0:
            raise ValueError("No valid fields found")

    def _file_exists(self, p):
        return Path(p).exists()

    def __len__(self):
        return 50000  # acts like "steps per epoch"

    def _sample_field(self):
        field_idx = random.randint(0, len(self.fields) - 1)
        print(f"[DEBUG] sample_field -> {field_idx}")
        return field_idx

    def _sample_coords(self, image):
        C, H, W = image.shape
        r = random.randint(0, H - self.tile_size)
        c = random.randint(0, W - self.tile_size)
        if H < self.tile_size or W < self.tile_size:
            raise ValueError(f"Image too small: {H}x{W}")
        return r, c

    def _build_fields(self):
        fields = []

        for _, row in self.metadata.iterrows():
            image_paths = row["image_paths"]
            selected_paths = [
                str(self.project_root / image_paths[ch - 1])
                for ch in self.channels
            ]

            fields.append({
                "files": selected_paths,
                "plate": row["plate"],
                "well": row["well"],
                "site": row["site"],
                "compound": row["pert_iname"],
                "broad_sample": row["broad_sample"],
                "gene": row["gene"],
                "control_type": row["control_type"]
            })

        print(f"Final valid fields: {len(fields)}")

        return fields


    def _load_field(self, field_idx):
        """Load and normalize a full field of view as (C, H, W)."""
        if field_idx in self._cache:
            return self._cache[field_idx]

        file_list = self.fields[field_idx]["files"]
        imgs = [tiff.imread(f) for f in file_list]
        if len(imgs) == 0:
            raise ValueError(f"No valid images for field {field_idx}")
        image = np.stack(imgs, axis=0).astype(np.float32)
        image = self._normalize_channels(image)
        print(f"[DEBUG] image shape {image.shape}")

        self._cache[field_idx] = image
        return image

    def __getitem__(self, idx):
        for _ in range(10):
            field_idx = self._sample_field()
            image = self._load_field(field_idx)
            if image is None:
                continue

            r, c = self._sample_coords(image)
            tile = image[:, r:r + self.tile_size, c:c + self.tile_size]
            if not self._is_informative(tile):
                continue

            tile = torch.from_numpy(tile).float()
            if self.transform is not None:
                tile = self.transform(tile)

            # Safety check
            if tile is None:
                print("[DEBUG] TRANSFORM RETURNED NONE")
                continue

            meta = self.fields[field_idx]

            print(f"[DEBUG] returning sample from field {field_idx}")

            return {
                "image": tile,
                "compound": meta["compound"],
                "broad_sample": meta["broad_sample"],
                "plate": meta["plate"],
                "well": meta["well"],
                "site": meta["site"],
                "row": r,
                "col": c
            }

        raise RuntimeError(f"Failed to sample valid tile after retries (idx={idx})")

    def _normalize_channels(self, image):
        normed = np.zeros_like(image)
        for c in range(image.shape[0]):
            channel = np.log1p(image[c])
            p1 = np.percentile(channel, 1)
            p99 = np.percentile(channel, 99)
            normed[c] = (channel - p1) / (p99 - p1 + 1e-6)
        return normed

    def _is_informative(self, tile):
        # DNA presence
        dna_signal = np.percentile(tile[4], 99)

        # overall structure
        total_signal = tile.mean()

        # texture (avoids flat gray tiles)
        variance = tile.var()

        return (dna_signal > 0.05) and (total_signal > 0.05) and (variance > 0.005)

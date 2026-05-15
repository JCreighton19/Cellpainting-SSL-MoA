import torch
from torch.utils.data import Dataset
import tifffile as tiff
import numpy as np
import pandas as pd
from pathlib import Path

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
        self.tiles = None # use lazy indexing to improve scalability

    def _file_exists(self, p):
        return Path(p).exists()

    def build_index(self):
        if self.tiles is None:
            self.tiles = self._index_tiles()

    def __len__(self):
        self.build_index()
        return len(self.tiles)

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

    def _index_tiles(self):
        sample_img = tiff.imread(self.fields[0]["files"][0])
        H, W = sample_img.shape

        rows = range(0, H - self.tile_size + 1, self.tile_size)
        cols = range(0, W - self.tile_size + 1, self.tile_size)

        tiles = []

        for field_idx in range(len(self.fields)):
            image = self._load_field(field_idx)  # load once per field

            for r in rows:
                for c in cols:
                    tile = image[:, r:r + self.tile_size, c:c + self.tile_size]

                    if self._is_informative(tile):
                        tiles.append((field_idx, r, c))

        return tiles


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

        self._cache[field_idx] = image
        return image


    def __getitem__(self, idx):
        self.build_index()
        field_idx, r, c = self.tiles[idx]
        image = self._load_field(field_idx)

        tile = image[:, r:r + self.tile_size, c:c + self.tile_size]
        tile = torch.from_numpy(tile)

        if self.transform:
            tile = self.transform(tile)

        meta = self.fields[field_idx]

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

    def _normalize_channels(self, image):
        normed = np.zeros_like(image)
        for c in range(image.shape[0]):
            channel = np.log1p(image[c])
            p1 = np.percentile(channel, 1)
            p99 = np.percentile(channel, 99)
            normed[c] = (channel - p1) / (p99 - p1 + 1e-6)
            normed[c] = np.clip(normed[c], 0, 1)
        return normed

    def _is_informative(self, tile):
        # DNA presence
        dna_signal = np.percentile(tile[4], 99)

        # overall structure
        total_signal = tile.mean()

        # texture (avoids flat gray tiles)
        variance = tile.var()

        return (dna_signal > 0.1) and (total_signal > 0.05) and (variance > 0.01)

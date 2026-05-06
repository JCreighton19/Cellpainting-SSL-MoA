import os
import re
import glob
import torch
from torch.utils.data import Dataset
import tifffile as tiff
import numpy as np


class CellPaintingDataset(Dataset):
    def __init__(self, data_dir, transform=None, channels=None, tile_size=224):
        self.data_dir = data_dir
        self.transform = transform
        self.channels = channels if channels is not None else [1,2,3,4,5]
        self.tile_size = tile_size

        self.files = glob.glob(os.path.join(data_dir, "*.tiff"))
        self._cache = {}  # used to reduce number of disk reads
        self.fields = self._group_files()   # list of field file-lists
        self.tiles = self._index_tiles()    # flat list of (field_idx, row, col)

    def _group_files(self):
        pattern = re.compile(r"(r\d+c\d+f\d+p\d+)")
        groups = {}
        for f in self.files:
            match = pattern.search(os.path.basename(f))
            if match:
                key = match.group(1)
                groups.setdefault(key, []).append(f)

        fields = []
        for key, file_list in groups.items():
            ch_map = {}
            for f in file_list:
                ch_match = re.search(r"ch(\d+)", f)
                if ch_match:
                    ch = int(ch_match.group(1))
                    ch_map[ch] = f

            ordered_files = [ch_map.get(ch) for ch in self.channels]

            # Debug print
            if None in ordered_files:
                print(f"Skipping field {key}, missing channels")

            if all(f is not None for f in ordered_files):
                fields.append(ordered_files)

        print(f"Final valid fields: {len(fields)}")
        return fields

    def _index_tiles(self):
        sample_img = tiff.imread(self.fields[0][0])
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

        file_list = self.fields[field_idx]
        imgs = [tiff.imread(f) for f in file_list]
        image = np.stack(imgs, axis=0).astype(np.float32)
        image = self._normalize_channels(image)

        self._cache[field_idx] = image
        return image

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        field_idx, r, c = self.tiles[idx]
        image = self._load_field(field_idx)

        tile = image[:, r:r + self.tile_size, c:c + self.tile_size]
        tile = torch.from_numpy(tile)

        if self.transform:
            tile = self.transform(tile)

        return tile

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
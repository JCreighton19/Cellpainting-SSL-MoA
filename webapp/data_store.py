"""Loads the precomputed well-level (Phase 1) and compound-level (Phase 2)
artifacts into memory once at app startup."""
from pathlib import Path

import numpy as np
import pandas as pd


class DataStore:
    def __init__(self, data_dir: Path):
        data_dir = Path(data_dir)
        self.wells = pd.read_parquet(data_dir / "wells.parquet").reset_index(drop=True)
        self.embeddings = np.load(data_dir / "well_embeddings.npy")

        if len(self.wells) != len(self.embeddings):
            raise ValueError(
                f"wells.parquet has {len(self.wells)} rows but well_embeddings.npy "
                f"has {len(self.embeddings)} rows — these must be built together."
            )

        self._id_to_row = {well_id: i for i, well_id in enumerate(self.wells["well_id"])}

        # Phase 2: compound-level aggregation, additive alongside the well-level data above.
        self.compounds = pd.read_parquet(data_dir / "compounds.parquet").reset_index(drop=True)
        self.compound_embeddings = np.load(data_dir / "compound_embeddings.npy")

        if len(self.compounds) != len(self.compound_embeddings):
            raise ValueError(
                f"compounds.parquet has {len(self.compounds)} rows but "
                f"compound_embeddings.npy has {len(self.compound_embeddings)} rows — "
                "these must be built together."
            )

        self._compound_id_to_row = {
            compound_id: i for i, compound_id in enumerate(self.compounds["compound_id"])
        }

    def get_well(self, well_id: str):
        """Returns (row: pandas.Series, row_idx: int) or (None, None) if not found."""
        row_idx = self._id_to_row.get(well_id)
        if row_idx is None:
            return None, None
        return self.wells.iloc[row_idx], row_idx

    def get_compound(self, compound_id: str):
        """Returns (row: pandas.Series, row_idx: int) or (None, None) if not found."""
        row_idx = self._compound_id_to_row.get(compound_id)
        if row_idx is None:
            return None, None
        return self.compounds.iloc[row_idx], row_idx

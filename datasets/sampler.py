import random
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path


class MoASampler:
    def __init__(self, processed_dir, metadata_path):
        self.moa_to_files      = defaultdict(list)
        self.compound_to_files = defaultdict(list)

        print("Loading metadata...")
        df = pd.read_parquet(metadata_path)

        print("Building indices...")
        _well_files   = {}
        _compound_moa = {}

        df = df[df["pt_path"].apply(lambda x: Path(x).exists())]

        for row in df.to_dict("records"):
            moa = row.get("moa")
            compound = row.get("broad_sample")
            moa_missing = pd.isna(moa)
            compound_missing = pd.isna(compound)

            plate = str(row.get("plate") or "")
            well  = str(row.get("well") or "")

            if not moa_missing:
                self.moa_to_files[moa].append(fp)

            if compound_missing:
                continue

            self.compound_to_files[compound].append(fp)
            _compound_moa.setdefault(compound, moa)

            if plate and well:
                key = (compound, plate, well)
                if key not in _well_files:
                    _well_files[key] = {"files": [], "moa": moa}
                _well_files[key]["files"].append(fp)

        self.compound_well_index = defaultdict(list)
        for (compound, plate, well), data in _well_files.items():
            self.compound_well_index[compound].append({
                "plate": plate, "well": well, "files": data["files"]
            })

        self.moa_list      = list(self.moa_to_files.keys())
        self.compound_list = list(self.compound_to_files.keys())
        self.replicate_compounds = [
            c for c, well_list in self.compound_well_index.items()
            if len({w["plate"] for w in well_list}) >= 2
            and _compound_moa.get(c) != "control vehicle"
        ]

        self.moa_keys = list(self.moa_to_files.keys())
        self.moa_weights = np.array([len(self.moa_to_files[m]) for m in self.moa_keys])
        self.moa_weights = self.moa_weights / self.moa_weights.sum()
        print(f"Loaded {len(self.moa_list)} MOAs | "
              f"{len(self.compound_list)} compounds | "
              f"{len(self.replicate_compounds)} replicate compounds (>=2 plates, non-control)")


    def sample_moa(self):
        moa = random.choices(self.moa_keys, weights=self.moa_weights, k=1)[0]
        return random.choice(self.moa_to_files[moa]), moa


    def sample_moa_k(self, k):
        moa = random.choice(self.moa_list)
        return random.choices(self.moa_to_files[moa], k=k), moa


    # def sample_compound_k(self, k):
    #     compound = random.choice(self.compound_list)
    #     return random.choices(self.compound_to_files[compound], k=k), compound


    # def sample_cross_plate_batch(self, n_compounds, n_tiles_per_well):
    #     compounds = random.sample(
    #         self.replicate_compounds,
    #         min(n_compounds, len(self.replicate_compounds))
    #     )
    #     result = []
    #     for cpd_idx, compound in enumerate(compounds):
    #         plate_to_wells = defaultdict(list)
    #         for w in self.compound_well_index[compound]:
    #             plate_to_wells[w["plate"]].append(w)
    #
    #         plates = list(plate_to_wells.keys())
    #         p1, p2 = random.sample(plates, 2)
    #         well1 = random.choice(plate_to_wells[p1])
    #         well2 = random.choice(plate_to_wells[p2])
    #
    #         result.append((cpd_idx, random.choices(well1["files"], k=n_tiles_per_well)))
    #         result.append((cpd_idx, random.choices(well2["files"], k=n_tiles_per_well)))

        return result

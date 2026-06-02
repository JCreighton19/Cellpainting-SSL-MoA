import random
import pandas as pd
from collections import defaultdict
from pathlib import Path


class MoASampler:
    def __init__(self, processed_dir, metadata_path):
        self.moa_to_files      = defaultdict(list)
        self.compound_to_files = defaultdict(list)
        processed_dir = Path(processed_dir)

        print("Loading metadata...")
        df = pd.read_parquet(metadata_path)

        print("Building indices...")
        _well_files   = {}   # (compound, plate, well) -> {"files": [], "moa": str}
        _compound_moa = {}   # compound -> moa (first seen, for control filtering)

        for idx, row in df.reset_index().iterrows():
            moa = str(row.get("moa") or "unknown")
            if moa == "nan":
                moa = "unknown"

            compound = str(row.get("broad_sample") or "unknown")
            if compound == "nan":
                compound = "unknown"

            plate = str(row.get("plate") or "")
            well  = str(row.get("well") or "")

            file_path = Path(row["pt_path"])
            if not file_path.exists():
                continue

            fp = str(file_path)
            self.moa_to_files[moa].append(fp)

            if compound == "unknown":
                continue

            self.compound_to_files[compound].append(fp)
            _compound_moa.setdefault(compound, moa)

            if plate and well:
                key = (compound, plate, well)
                if key not in _well_files:
                    _well_files[key] = {"files": [], "moa": moa}
                _well_files[key]["files"].append(fp)

        # compound -> list of well dicts
        self.compound_well_index = defaultdict(list)
        for (compound, plate, well), data in _well_files.items():
            self.compound_well_index[compound].append({
                "plate": plate, "well": well, "files": data["files"]
            })

        self.moa_list      = list(self.moa_to_files.keys())
        self.compound_list = list(self.compound_to_files.keys())

        # Compounds with wells on >=2 distinct plates, controls excluded
        self.replicate_compounds = [
            c for c, well_list in self.compound_well_index.items()
            if len({w["plate"] for w in well_list}) >= 2
            and _compound_moa.get(c, "") != "control vehicle"
        ]

        print(f"Loaded {len(self.moa_list)} MOAs | "
              f"{len(self.compound_list)} compounds | "
              f"{len(self.replicate_compounds)} replicate compounds (>=2 plates, non-control)")

    # ------------------------------------------------------------------
    # Existing methods — preserved for extract_embeddings.py / inference
    # ------------------------------------------------------------------

    def sample_moa(self):
        moa = random.choice(self.moa_list)
        return random.choice(self.moa_to_files[moa]), moa

    def sample_moa_k(self, k):
        moa = random.choice(self.moa_list)
        return random.choices(self.moa_to_files[moa], k=k), moa

    def sample_compound_k(self, k):
        compound = random.choice(self.compound_list)
        return random.choices(self.compound_to_files[compound], k=k), compound

    # ------------------------------------------------------------------
    # Well-level cross-plate sampling
    # ------------------------------------------------------------------

    def sample_cross_plate_batch(self, n_compounds, n_tiles_per_well):
        """
        Sample n_compounds compounds, each contributing 2 well entries from
        different plates.  Returns a flat list of (compound_idx, file_paths)
        pairs — 2 * n_compounds entries total, interleaved well1/well2.

        compound_idx is a contiguous integer label [0, n_compounds) used
        directly as the SupCon label: both entries sharing the same idx
        are the cross-plate positive pair.
        """
        compounds = random.sample(
            self.replicate_compounds,
            min(n_compounds, len(self.replicate_compounds))
        )
        result = []
        for cpd_idx, compound in enumerate(compounds):
            plate_to_wells = defaultdict(list)
            for w in self.compound_well_index[compound]:
                plate_to_wells[w["plate"]].append(w)

            plates = list(plate_to_wells.keys())
            p1, p2 = random.sample(plates, 2)
            well1 = random.choice(plate_to_wells[p1])
            well2 = random.choice(plate_to_wells[p2])

            # random.choices handles wells with fewer tiles than n_tiles_per_well
            result.append((cpd_idx, random.choices(well1["files"], k=n_tiles_per_well)))
            result.append((cpd_idx, random.choices(well2["files"], k=n_tiles_per_well)))

        return result

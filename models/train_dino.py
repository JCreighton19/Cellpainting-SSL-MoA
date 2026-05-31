import torch
import torch.nn as nn
import os
import sys
import random
import numpy as np
import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import torch.nn.functional as F

from datasets.sampler import MoASampler
from models.dino import CellPaintingViT


# -----------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------

def vicreg_loss(z1, z2, lambda_var=25.0, mu_cov=1.0):
    """Variance + covariance regularization on two views of backbone outputs."""
    N, D = z1.shape
    std1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    var_loss = F.relu(1.0 - std1).mean() + F.relu(1.0 - std2).mean()
    z1n = z1 - z1.mean(dim=0)
    z2n = z2 - z2.mean(dim=0)
    cov1 = (z1n.T @ z1n) / (N - 1)
    cov2 = (z2n.T @ z2n) / (N - 1)
    off1 = cov1.pow(2).sum() - cov1.diagonal().pow(2).sum()
    off2 = cov2.pow(2).sum() - cov2.diagonal().pow(2).sum()
    cov_loss = off1 / D + off2 / D
    return lambda_var * var_loss + mu_cov * cov_loss


def supcon_loss(features, labels, temperature=0.07):
    """SupCon loss. features: (N, D) L2-normalized. labels: (N,) integer tensor."""
    N = features.shape[0]
    sim = torch.matmul(features, features.T) / temperature
    eye = torch.eye(N, dtype=torch.bool, device=features.device)
    pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~eye
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()
    exp_sim = torch.exp(sim) * ~eye
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)
    n_pos = pos_mask.sum(dim=1).float()
    has_pos = n_pos > 0
    if not has_pos.any():
        return features.sum() * 0.0
    mean_log_prob = (log_prob * pos_mask).sum(dim=1) / n_pos.clamp(min=1)
    return -mean_log_prob[has_pos].mean()


# -----------------------------------------------------------------------
# Tile loading
# -----------------------------------------------------------------------

def load_tile(file_path, tile_size=224, augment=True):
    """Load a preprocessed .pt tile, apply foreground crop + optional augment."""
    sample = torch.load(file_path, weights_only=False)
    img = sample["image"]  # (5, H, W), already normalized float32

    # Foreground-biased crop via DNA channel (index 4)
    dna = img[4]
    H, W = dna.shape
    ts = tile_size
    small = F.avg_pool2d(dna.unsqueeze(0).unsqueeze(0), kernel_size=8).squeeze()
    small = (small - small.min()) / (small.max() + 1e-6)
    flat = small.flatten()
    idx = torch.multinomial(flat + 1e-6, 1).item()
    y = int((idx // small.shape[1]) * 8)
    x = int((idx % small.shape[1]) * 8)
    r = max(0, min(H - ts, y + random.randint(-ts // 2, ts // 2)))
    c = max(0, min(W - ts, x + random.randint(-ts // 2, ts // 2)))
    tile = img[:, r:r + ts, c:c + ts]

    if augment:
        k = random.randint(0, 3)
        if k > 0:
            tile = torch.rot90(tile, k, dims=[1, 2])
        if random.random() < 0.5:
            tile = tile.flip(1)
        if random.random() < 0.5:
            tile = tile.flip(2)
        tile = (tile * random.uniform(0.9, 1.1)).clamp(0, 1)

    return tile


# -----------------------------------------------------------------------
# Prefetcher — overlaps disk I/O with GPU computation
# -----------------------------------------------------------------------

class WellBatchPrefetcher:
    """
    Loads well batches in background using a thread pool, so disk I/O
    overlaps with GPU computation.  Replaces the DataLoader's async workers.

    A single daemon thread drives the I/O pool, continuously filling a
    small queue.  The training loop calls .get() to consume pre-loaded
    batches without blocking on disk.
    """

    def __init__(self, sampler, n_compounds, n_tiles_per_well,
                 n_workers=8, prefetch=2):
        self._sampler  = sampler
        self._n_cpd    = n_compounds
        self._n_tiles  = n_tiles_per_well
        self._q        = queue.Queue(maxsize=prefetch)
        self._pool     = ThreadPoolExecutor(max_workers=n_workers)
        self._running  = True
        threading.Thread(target=self._produce, daemon=True).start()

    @staticmethod
    def _load_well(args):
        cpd_idx, file_paths = args
        tiles = list(
            ThreadPoolExecutor(
                max_workers=len(file_paths)
            ).map(
                load_tile,
                file_paths
            )
        )
        return cpd_idx, torch.stack(tiles)

    def _produce(self):
        while self._running:
            spec   = self._sampler.sample_cross_plate_batch(self._n_cpd, self._n_tiles)
            loaded = list(self._pool.map(self._load_well, spec))
            self._q.put(loaded)

    def get(self):
        return self._q.get()

    def stop(self):
        self._running = False
        self._pool.shutdown(wait=False)


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    sys.stdout.reconfigure(line_buffering=True)

    run_dir = os.path.join(
        os.environ["CP_OUTPUT_ROOT"],
        "checkpoints",
        datetime.now().strftime("%m%d%y_%H%M")
    )
    os.makedirs(run_dir, exist_ok=True)

    device = (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    print("Using device:", device)

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    N_COMPOUNDS     = 8    # compounds per step
    TILES_PER_WELL  = 8    # tiles sampled per well (with replacement if needed)
    SUPCON_WEIGHT   = 1.0
    VICREG_WEIGHT   = 0.1
    STEPS_PER_EPOCH = 450
    n_epochs        = 30

    # ------------------------------------------------------------------
    # Sampler
    # ------------------------------------------------------------------
    data_dir = os.path.join(os.environ["CP_OUTPUT_ROOT"], "data/tiles_qc")
    sampler = MoASampler(
        processed_dir=data_dir,
        metadata_path=os.path.join(
            os.environ["CP_OUTPUT_ROOT"],
            "data/processed/master_metadata.parquet"
        )
    )

    if len(sampler.replicate_compounds) < N_COMPOUNDS:
        raise ValueError(
            f"Only {len(sampler.replicate_compounds)} replicate compounds found; "
            f"need at least {N_COMPOUNDS}."
        )

    print(f"\n=== Training Setup ===")
    print(f"Replicate compounds : {len(sampler.replicate_compounds)}")
    print(f"N_COMPOUNDS/step    : {N_COMPOUNDS}")
    print(f"TILES_PER_WELL      : {TILES_PER_WELL}")
    print(f"Tiles/step          : {N_COMPOUNDS * 2 * TILES_PER_WELL}  "
          f"({N_COMPOUNDS} compounds × 2 wells × {TILES_PER_WELL} tiles)")
    print(f"Well embs/step      : {N_COMPOUNDS * 2}  (SupCon batch size)")
    print(f"Steps/epoch         : {STEPS_PER_EPOCH}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    student_enc = CellPaintingViT(in_channels=5).to(device)

    class SupConHead(nn.Module):
        def __init__(self, dim=384, proj_dim=128):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.Linear(256, proj_dim)
            )

        def forward(self, x):
            return F.normalize(self.mlp(x), dim=1)

    supcon_head = SupConHead().to(device)

    optimizer = torch.optim.AdamW(
        list(student_enc.parameters()) + list(supcon_head.parameters()),
        lr=5e-5,
        weight_decay=0.04
    )

    trainable_params = tuple(
        list(student_enc.parameters()) + list(supcon_head.parameters())
    )

    # ------------------------------------------------------------------
    # Precompute fixed per-step values
    # ------------------------------------------------------------------
    # Labels are always [0,0,1,1,...,N-1,N-1]: each compound contributes
    # 2 wells (well-1 and well-2) in consecutive pairs.
    compound_labels = torch.arange(N_COMPOUNDS, device=device).repeat_interleave(2)

    # Start prefetcher — begins loading batches immediately in background
    prefetcher = WellBatchPrefetcher(
        sampler, N_COMPOUNDS, TILES_PER_WELL, n_workers=8, prefetch=2
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    losses = []

    try:
        for epoch in range(n_epochs):
            epoch_start = time.perf_counter()
            print(f"\nEpoch {epoch + 1}/{n_epochs} | Start: {datetime.now().strftime('%I:%M %p')}")

            student_enc.train()
            supcon_head.train()
            total_loss = 0.0

            for step in range(STEPS_PER_EPOCH):
                # Pre-loaded by background thread — typically returns immediately
                loaded = prefetcher.get()

                # Stack all wells into one tensor: (2*N*T, 5, H, W)
                all_tiles = torch.cat([tiles for _, tiles in loaded])
                if device == "cuda":
                    all_tiles = all_tiles.pin_memory().to(device, non_blocking=True)
                else:
                    all_tiles = all_tiles.to(device)

                # Single encoder forward pass for all tiles
                all_embs = student_enc(all_tiles)       # (2*N*T, D)
                D = all_embs.shape[-1]

                # Reshape: (2*N, T, D) — one row per well
                embs_3d = all_embs.view(2 * N_COMPOUNDS, TILES_PER_WELL, D)

                # VICReg: well-1 tiles vs well-2 tiles (two views of same compounds)
                view1 = embs_3d[0::2].reshape(-1, D)   # (N*T, D)
                view2 = embs_3d[1::2].reshape(-1, D)   # (N*T, D)
                vic_loss = vicreg_loss(view1, view2)

                # Mean-pool tiles within each well -> well embeddings (2*N, D)
                well_embs = embs_3d.mean(dim=1)

                # SupCon on well embeddings
                sc_z    = supcon_head(well_embs)
                sc_loss = supcon_loss(sc_z, compound_labels)

                loss = SUPCON_WEIGHT * sc_loss + VICREG_WEIGHT * vic_loss

                if step % 50 == 0:
                    print(f"  {step:>4}/{STEPS_PER_EPOCH}  "
                          f"sc={sc_loss.item():.4f}  "
                          f"vic={vic_loss.item():.4f}  "
                          f"loss={loss.item():.4f}")

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 3.0)
                optimizer.step()

                total_loss += loss.item()

            epoch_end = time.perf_counter()
            avg_loss  = total_loss / STEPS_PER_EPOCH
            losses.append(avg_loss)

            # "student_enc" key preserves compatibility with extract_embeddings.py
            torch.save({
                "student_enc": student_enc.state_dict(),
                "supcon_head": supcon_head.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "epoch":       epoch,
                "loss":        avg_loss,
            }, os.path.join(run_dir, f"dino_epoch_{epoch + 1}.pt"))

            print(
                f"Epoch {epoch + 1}/{n_epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Time: {(epoch_end - epoch_start) / 60:.2f} min"
            )

    finally:
        prefetcher.stop()

    print(f"\nFinished. Checkpoints at {run_dir}")


if __name__ == "__main__":
    main()

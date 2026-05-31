import torch
import torch.nn as nn
import os
import sys
import random
import numpy as np
import time
from datetime import datetime
import torch.nn.functional as F

from datasets.sampler import MoASampler
from models.dino import CellPaintingViT


# -----------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------

def vicreg_loss(z1, z2, lambda_var=25.0, mu_cov=1.0):
    """Variance + covariance regularization on two views of backbone outputs.
    Prevents dimensional collapse and decorrelates features.
    Applied to tile embeddings: z1 = well-1 tiles, z2 = well-2 tiles."""
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
        # Rotation (90° increments)
        k = random.randint(0, 3)
        if k > 0:
            tile = torch.rot90(tile, k, dims=[1, 2])
        # Flips
        if random.random() < 0.5:
            tile = tile.flip(1)
        if random.random() < 0.5:
            tile = tile.flip(2)
        # Mild intensity jitter — preserves inter-channel ratios
        tile = (tile * random.uniform(0.9, 1.1)).clamp(0, 1)

    return tile


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
        list(student_enc.parameters()) +
        list(supcon_head.parameters())
    )
    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    losses = []

    for epoch in range(n_epochs):
        epoch_start = time.perf_counter()
        print(f"\nEpoch {epoch + 1}/{n_epochs} | Start: {datetime.now().strftime('%I:%M %p')}")

        student_enc.train()
        supcon_head.train()
        total_loss = 0.0

        for step in range(STEPS_PER_EPOCH):
            # batch: [(cpd_idx, [files]), ...] — 2 * N_COMPOUNDS entries,
            # interleaved well1/well2 per compound.
            batch = sampler.sample_cross_plate_batch(N_COMPOUNDS, TILES_PER_WELL)

            # Forward: load all tiles, encode, collect per-well embedding lists
            all_tiles = [] # list of (T, 384) tensors — one entry per well
            well_sizes = []
            compound_labels = []

            for cpd_idx, file_paths in batch:
                tiles = torch.stack(
                    [load_tile(fp) for fp in file_paths]
                )
                all_tiles.append(tiles)
                well_sizes.append(len(file_paths))
                compound_labels.append(cpd_idx)

            # shape: (total_tiles,5,224,224)
            all_tiles = torch.cat(all_tiles, dim=0)
            if device == "cuda":
                all_tiles = all_tiles.pin_memory().to(
                    device,non_blocking=True
                )
            else:
                all_tiles = all_tiles.to(device)

            # Single encoder forward
            all_embs = student_enc(all_tiles)

            # Split embeddings back into wells
            well_tile_embs = []
            start = 0

            for size in well_sizes:
                end = start + size
                well_tile_embs.append(
                    all_embs[start:end]
                )
                start = end

            # VICReg on tile embeddings:
            # well_tile_embs[0::2] = well-1 tiles for each compound
            # well_tile_embs[1::2] = well-2 tiles for each compound
            # These are two "views" of the same set of compounds.
            view1 = torch.cat(well_tile_embs[0::2], dim=0)   # (N*T, 384)
            view2 = torch.cat(well_tile_embs[1::2], dim=0)   # (N*T, 384)
            vic_loss = vicreg_loss(view1, view2)

            # Well embeddings: mean-pool tiles within each well
            well_embs = torch.stack(
                [e.mean(dim=0) for e in well_tile_embs]
            )                                                  # (2*N, 384)

            # SupCon on well embeddings with compound labels
            labels   = torch.tensor(compound_labels, device=device)
            sc_z     = supcon_head(well_embs)
            sc_loss  = supcon_loss(sc_z, labels)

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

        # student_enc saved as "student_enc" for compatibility with extract_embeddings.py
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

    print(f"\nFinished. Checkpoints at {run_dir}")


if __name__ == "__main__":
    main()

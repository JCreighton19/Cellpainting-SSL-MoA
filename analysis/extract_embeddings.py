import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import re
import argparse

from datasets.dataset import CellPaintingDataset
from models.dino.dino import CellPaintingViT
from models.config import CONFIG

CROP_SIZE  = 224
CROP_BATCH = 64   # max crops per model forward pass (bounds VRAM)
MIN_FG     = 0.01 # paper: ≥1% foreground in DNA channel


def get_checkpoints(run_dir):
    files = os.listdir(run_dir)
    ckpts = []
    for f in files:
        if f.startswith("dino_epoch_") and f.endswith(".pt"):
            match = re.findall(r"\d+", f)
            if len(match) == 0:
                continue
            epoch = int(match[0])
            ckpts.append((epoch, f))

    if len(ckpts) == 0:
        raise ValueError("No checkpoints found")

    ckpts.sort()
    return [(e, os.path.join(run_dir, f)) for e, f in ckpts]


@torch.no_grad()
def embed_fov(model, img, otsu_thresh, device):
    """
    Tile a single FOV into non-overlapping CROP_SIZE×CROP_SIZE crops,
    keep those with ≥MIN_FG foreground (DNA channel > Otsu), embed each,
    and return the L2-normalised mean. Falls back to centre crop if none pass.

    img: (C, H, W) tensor, already on CPU (moved per-item to avoid OOM).
    """
    C, H, W = img.shape
    n_h, n_w = H // CROP_SIZE, W // CROP_SIZE

    if n_h == 0 or n_w == 0:
        # image smaller than crop size — use the full image
        crops = img.unsqueeze(0).to(device)
    else:
        # unfold into (n_h*n_w, C, CROP_SIZE, CROP_SIZE) — views until .contiguous()
        crops = (img.unfold(1, CROP_SIZE, CROP_SIZE)   # (C, n_h, W, crop)
                    .unfold(2, CROP_SIZE, CROP_SIZE)   # (C, n_h, n_w, crop, crop)
                    .permute(1, 2, 0, 3, 4)            # (n_h, n_w, C, crop, crop)
                    .contiguous()
                    .view(n_h * n_w, C, CROP_SIZE, CROP_SIZE))  # ~4 MB for 16 crops

        fg = (crops[:, 4] > otsu_thresh).float().mean(dim=[1, 2]) >= MIN_FG
        crops = crops[fg]

        if len(crops) == 0:
            r0 = (H - CROP_SIZE) // 2
            c0 = (W - CROP_SIZE) // 2
            crops = img[:, r0:r0 + CROP_SIZE, c0:c0 + CROP_SIZE].unsqueeze(0)

        crops = crops.to(device)

    # sub-batch through model to bound VRAM usage
    parts = [model(crops[i:i + CROP_BATCH]) for i in range(0, len(crops), CROP_BATCH)]
    z = F.normalize(torch.cat(parts, dim=0), dim=1)
    return z.mean(dim=0)  # (D,)


def load_model(checkpoint_path, device):
    model = CellPaintingViT(in_channels=5).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["student_enc"])
    model.eval()
    return model


def main():
    """
    Usage:
    1) DEFAULT (most common / sanity check)
       Runs ONLY the latest checkpoint:
        python extract_embeddings.py \
            --run_dir /path/to/checkpoints

    2) RUN SPECIFIC EPOCHS (debug / targeted comparison):
        python extract_embeddings.py \
            --run_dir /path/to/checkpoints \
            --epochs 1 5 10

    3) RUN ALL CHECKPOINTS (full training trajectory analysis):
        python extract_embeddings.py \
            --run_dir /path/to/checkpoints \
            --all
    """
    torch.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--epochs", nargs="+", type=int, default=None,
        help="Specific epochs to extract (e.g. 1 5 10)"
    )
    group.add_argument(
        "--all", action="store_true",
        help="Extract ALL checkpoints"
    )
    args = parser.parse_args()
    run_dir = args.run_dir

    if not os.path.exists(run_dir):
        raise ValueError("run_dir does not exist")

    run_name   = os.path.basename(os.path.normpath(run_dir))
    base_output = "/scratch/creighton.jo/cellpainting/embeddings"
    output_dir  = os.path.join(base_output, run_name)
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load full images — extraction tiles each FOV into multiple crops
    dataset = CellPaintingDataset(
        processed_dir=os.path.join(os.environ["CP_OUTPUT_ROOT"], "data/tiles_qc"),
        return_full_image=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=False,      # full images are large; skip pinning to save CPU RAM
        persistent_workers=True,
        prefetch_factor=2,     # reduced from training to limit CPU RAM pressure
    )

    checkpoints = get_checkpoints(run_dir)
    checkpoints = sorted(checkpoints, key=lambda x: x[0])

    if args.all:
        selected_checkpoints = checkpoints
    elif args.epochs is not None:
        selected_checkpoints = [(e, p) for (e, p) in checkpoints if e in args.epochs]
    else:
        selected_checkpoints = [checkpoints[-1]]

    print(f"Found {len(selected_checkpoints)} checkpoints to process")

    for epoch, checkpoint_path in selected_checkpoints:
        print(f"\n=== Processing epoch {epoch} ===")
        print(f"Checkpoint: {checkpoint_path}")
        model = load_model(checkpoint_path, device)

        embeddings, plates, wells = [], [], []

        with torch.no_grad():
            for step, batch in enumerate(loader):
                if step % 50 == 0:
                    print(f"Epoch {epoch} | Step {step}/{len(loader)}")

                images     = batch["image"]           # (B, C, H, W) full resolution, on CPU
                thresholds = batch["otsu_threshold"]  # (B,) per-FOV Otsu scalar

                fov_embs = torch.stack([
                    embed_fov(model, images[b], thresholds[b].item(), device)
                    for b in range(len(images))
                ])  # (B, D)

                embeddings.append(fov_embs.cpu().numpy())
                plates.extend(batch["plate"])
                wells.extend(batch["well"])

        embeddings = np.concatenate(embeddings, axis=0)
        plates     = np.array(plates)
        wells      = np.array(wells)

        print("Shape:", embeddings.shape)

        np.save(os.path.join(output_dir, f"embeddings_epoch_{epoch}.npy"), embeddings)
        np.save(os.path.join(output_dir, f"plates_epoch_{epoch}.npy"),     plates)
        np.save(os.path.join(output_dir, f"wells_epoch_{epoch}.npy"),      wells)

        print(f"Saved epoch {epoch} → {os.path.join(output_dir, f'embeddings_epoch_{epoch}')}.npy\n")


if __name__ == "__main__":
    main()

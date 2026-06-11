import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import re
import argparse

from datasets.dataset import CellPaintingDataset
from models.dino.dino import CellPaintingViT
from models.config import CONFIG


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


def foreground_multi_crop(images, crop_size, masks, offsets=[-32, 0, 32]):
    """
    Note: this function differs from foreground_crop in training loop
    in that it is deterministic - random foreground sampling, random
    jitter, and stochastic fallback have been removed.
    """
    B, C, H, W = images.shape
    ts = crop_size
    coords = masks.view(B, -1).float()
    fg_idx = coords.argmax(dim=1)
    ys = fg_idx // W
    xs = fg_idx % W
    crops = []

    for oy in offsets:
        for ox in offsets:
            r = (ys + oy - ts // 2).clamp(0, H - ts)
            c = (xs + ox - ts // 2).clamp(0, W - ts)
            crop = torch.stack([
                images[b, :, r[b]:r[b] + ts, c[b]:c[b] + ts]
                for b in range(B)
            ])
            crops.append(crop)

    return torch.cat(crops, dim=0)


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

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)

    # Optionally specify epochs
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--epochs",
        nargs="+",
        type=int,
        default=None,
        help="Specific epochs to extract (e.g. 1 5 10)"
    )

    group.add_argument(
        "--all",
        action="store_true",
        help="Extract ALL checkpoints"
    )

    args = parser.parse_args()
    run_dir = args.run_dir

    if not os.path.exists(run_dir):
        raise ValueError("run_dir does not exist")

    run_name = os.path.basename(os.path.normpath(run_dir))

    base_output = "/scratch/creighton.jo/cellpainting/embeddings"
    output_dir = os.path.join(base_output, run_name)
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Dataset (same for all checkpoints)
    dataset = CellPaintingDataset(
        processed_dir=os.path.join(
            os.environ["CP_OUTPUT_ROOT"],
            "data/tiles_qc"
        ),
        random_crop=False
    )

    loader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,  # defined shuffling in sampler
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    checkpoints = get_checkpoints(run_dir)
    checkpoints = sorted(checkpoints, key=lambda x: x[0])

    if args.all:
        selected_checkpoints = checkpoints
    elif args.epochs is not None:
        selected_checkpoints = [
            (e, p) for (e, p) in checkpoints if e in args.epochs
        ]
    else:
        # Default: latest checkpoint only
        selected_checkpoints = [checkpoints[-1]]

    print(f"Found {len(selected_checkpoints)} checkpoints to process")


    # Extract embeddings
    for epoch, checkpoint_path in selected_checkpoints:
        print(f"\n=== Processing epoch {epoch} ===")
        print(f"Checkpoint: {checkpoint_path}")
        model = load_model(checkpoint_path, device)
        embeddings = []
        plates = []
        wells = []

        with torch.no_grad():
            for step, batch in enumerate(loader):
                if step % 50 == 0:
                    print(f"Epoch {epoch} | Step {step}/{len(loader)}")
                x = batch["image"].to(device, non_blocking=True)
                m = batch["otsu_mask"].to(device, non_blocking=True)

                # multi-crop inference (KEY CHANGE)
                x_crops = foreground_multi_crop(x, crop_size=224, masks=m)
                with torch.no_grad():
                    z = model(x_crops)
                    z = torch.nn.functional.normalize(z, dim=1)

                # reshape: (num_crops, B, D)
                B = x.shape[0]
                n_crops = 9  # 3x3 offsets
                z = z.view(n_crops, B, -1).mean(dim=0)

                embeddings.append(z.cpu().numpy())
                plates.extend(batch["plate"])
                wells.extend(batch["well"])

        embeddings = np.concatenate(embeddings, axis=0)
        plates = np.array(plates)
        wells = np.array(wells)

        print("Shape:", embeddings.shape)

        # Save
        np.save(
            os.path.join(output_dir, f"embeddings_epoch_{epoch}.npy"),
            embeddings
        )
        np.save(
            os.path.join(output_dir, f"plates_epoch_{epoch}.npy"),
            plates
        )
        np.save(
            os.path.join(output_dir, f"wells_epoch_{epoch}.npy"),
            wells
        )
        embedding_file_name = f"embeddings_epoch_{epoch}"
        print(f"Saved epoch {epoch} at {os.path.join(output_dir, embedding_file_name)}.npy\n")

if __name__ == "__main__":
    main()
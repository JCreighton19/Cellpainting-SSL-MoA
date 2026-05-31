import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import re
import argparse

from datasets.dataset import CellPaintingDataset
from models.dino import CellPaintingViT


def get_latest_checkpoint(run_dir):
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
        raise ValueError(f"No valid checkpoints found in {run_dir}")

    ckpts.sort()
    return os.path.join(run_dir, ckpts[-1][1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    run_dir = args.run_dir

    # ---- Validate user inputted path ----
    if not os.path.exists(run_dir):
        raise ValueError(f"run_dir does not exist: {run_dir}")

    if not os.path.isdir(run_dir):
        raise ValueError(f"run_dir is not a directory: {run_dir}")

    parser.add_argument(
        "--checkpoint",
        default=None
    )

    if args.checkpoint is None:
        checkpoint_path = get_latest_checkpoint(run_dir)
    else:
        checkpoint_path = args.checkpoint

    run_name = os.path.basename(os.path.normpath(run_dir))
    base_output = "/scratch/creighton.jo/cellpainting/embeddings"
    output_dir = os.path.join(base_output, run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Run dir: {run_dir}")
    print(f"Checkpoint used: {checkpoint_path}")
    print(f"Output dir: {output_dir}")

    torch.save({
        "checkpoint_path": checkpoint_path,
        "run_dir": run_dir,
        "run_name": run_name
    }, os.path.join(output_dir, "run_info.pt"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load model
    print("Loading model...")
    model = CellPaintingViT(in_channels=5).to(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )
    model.load_state_dict(checkpoint["student_enc"])
    model.eval()

    # Dataset
    print("Loading dataset...")
    dataset = CellPaintingDataset(
        processed_dir=os.path.join(
            os.environ["CP_OUTPUT_ROOT"],
            "data/tiles_qc"
        ),
        random_crop=False
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print("Dataset size:", len(dataset))
    print("Batches:", len(loader))

    # Extract embeddings
    embeddings = []
    plates = []
    wells = []

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if step % 50 == 0:
                print(f"Step {step}/{len(loader)}")
            x = batch["image"].to(device, non_blocking=True)
            z = model(x)
            z = torch.nn.functional.normalize(z, dim=1)

            embeddings.append(z.cpu().numpy())
            plates.extend(batch["plate"])
            wells.extend(batch["well"])

    # Save outputs
    embeddings = np.concatenate(embeddings, axis=0)
    plates = np.array(plates)
    wells = np.array(wells)

    print("Final tile-level embeddings shape:", embeddings.shape)
    assert len(embeddings) == len(plates) == len(wells)

    np.save(
        os.path.join(output_dir, f"embeddings_{run_name}.npy"),
        embeddings
    )

    np.save(
        os.path.join(output_dir, f"plates_{run_name}.npy"),
        plates
    )

    np.save(
        os.path.join(output_dir, f"wells_{run_name}.npy"),
        wells
    )

    print("Saved outputs.")

if __name__ == "__main__":
    main()
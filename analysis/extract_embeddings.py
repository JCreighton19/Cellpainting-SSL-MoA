import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from dataset import CellPaintingDataset
from models.dino import CellPaintingViT


def main():
    output_dir = "/scratch/creighton.jo/cellpainting/embeddings"
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load model
    model = CellPaintingViT(in_channels=5).to(device)
    checkpoint_path = os.path.expanduser(
        "~/Cellpainting-SSL-MoA/checkpoints/dino_epoch_10.pt"
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )
    model.load_state_dict(checkpoint["student_enc"])
    model.eval()

    # Dataset
    BASE = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.join(
        BASE,
        "../data/processed/master_metadata.parquet"
    )

    data_root = os.path.join(BASE, "../data")
    dataset = CellPaintingDataset(
        metadata_path=metadata_path,
        data_root=data_root,
        channels=[1,2,3,4,5],
        tile_size=224
    )
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    print("Dataset size:", len(dataset))
    print("Batches:", len(loader))

    # Extract embeddings
    embeddings = []
    metadata = []
    #max_batches = 500 # limit

    with torch.no_grad():
        print("Starting extraction loop...")
        for step, batch in enumerate(loader):
            # if step >= max_batches:
            #     break
            if step % 50 == 0:
                print(f"Step {step}/{len(loader)}")
            x = batch["image"].to(device)
            z = model(x)
            z = torch.nn.functional.normalize(z, dim=1) # normalize
            embeddings.append(z.cpu().numpy())
            B = x.shape[0]
            for i in range(B):
                metadata.append({
                    "compound": batch["compound"][i],
                    "broad_sample": batch["broad_sample"][i],
                    "plate": batch["plate"][i],
                    "well": batch["well"][i],
                    "site": batch["site"][i],
                    "row": batch["row"][i],
                    "col": batch["col"][i],
                })

    # Save outputs
    embeddings = np.concatenate(embeddings, axis=0)
    metadata = pd.DataFrame(metadata)
    print("Embeddings shape:", embeddings.shape)
    print("Metadata shape:", metadata.shape)
    np.save(
        os.path.join(output_dir, "embeddings_epoch10.npy"),
        embeddings
    )
    metadata.to_parquet(
        os.path.join(output_dir, "metadata_epoch10.parquet"),
        index=False
    )
    print("Saved outputs.")

if __name__ == "__main__":
    main()
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import umap
import os

from dataset import CellPaintingDataset
from models.dino import CellPaintingViT

def main():
    os.makedirs("figures", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CellPaintingViT(in_channels=5).to(device)
    checkpoint = torch.load("checkpoints/dino_epoch_8.pt")
    model.load_state_dict(checkpoint["student_enc"])
    model.eval()

    dataset = CellPaintingDataset(
        data_dir="/scratch/creighton.jo/cellpainting/plate1",
        channels=[1,2,3,4,5],
        tile_size=224
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    embeddings = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z = model(batch)
            embeddings.append(z.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)

    reducer = umap.UMAP(n_components=2, random_state=42)
    emb_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=2)
    plt.title("Cell Painting DINO Embeddings (UMAP)")
    plt.savefig("figures/umap_embeddings.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()
import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import umap

from dataset import CellPaintingDataset
from models.dino import CellPaintingViT


def main():
    os.makedirs("figures", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load model
    model = CellPaintingViT(in_channels=5).to(device)
    checkpoint = torch.load("checkpoints/dino_epoch_8.pt", map_location=device)
    model.load_state_dict(checkpoint["student_enc"])
    model.eval()

    # Dataset
    dataset = CellPaintingDataset(
        data_dir="/scratch/creighton.jo/cellpainting/plate1",
        channels=[1,2,3,4,5],
        tile_size=224
    )

    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)

    # Extract embeddings (streaming)
    embeddings = []
    max_batches = None  # set to e.g. 100 for debugging

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = batch.to(device)
            z = model(batch)
            # move to CPU immediately to reduce GPU memory pressure
            embeddings.append(z.cpu())

    # concat safely (still OK now because tensors, not numpy lists)
    embeddings = torch.cat(embeddings, dim=0)

    print("Final embedding shape:", embeddings.shape)

    # save embeddings for reuse
    torch.save(embeddings, "figures/embeddings.pt")

    # UMAP
    embeddings_np = embeddings.numpy()
    reducer = umap.UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=15,
        min_dist=0.1
    )
    emb_2d = reducer.fit_transform(embeddings_np)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=2)
    plt.title("Cell Painting DINO Embeddings (UMAP)")
    plt.savefig("figures/umap_embeddings.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved: figures/umap_embeddings.png")


if __name__ == "__main__":
    main()
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap

def main():
    input_dir = "/scratch/creighton.jo/cellpainting/embeddings"
    output_dir = "/scratch/creighton.jo/cellpainting/figures"
    os.makedirs(output_dir, exist_ok=True)

    embeddings = np.load(
        os.path.join(input_dir, "embeddings_epoch10.npy")
    )

    metadata = pd.read_parquet(
        os.path.join(input_dir, "metadata_epoch10.parquet")
    )

    print("Embeddings:", embeddings.shape)
    print("Metadata:", metadata.shape)

    reducer = umap.UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=15,
        min_dist=0.1
    )

    emb_2d = reducer.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(8,6))
    plt.scatter(
        emb_2d[:,0],
        emb_2d[:,1],
        s=2
    )
    plt.title("DINO Cell Painting Embeddings")
    plt.savefig(
        os.path.join(output_dir, "umap_epoch10.png"),
        dpi=300,
        bbox_inches="tight"
    )
    print("Saved UMAP.")

if __name__ == "__main__":
    main()
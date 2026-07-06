import os
import argparse
import torch
import numpy as np

from datasets.dataset import CellPaintingDataset
from analysis.extract_embeddings import get_checkpoints, load_model

CROP_SIZE = 224
MIN_FG    = 0.01  # paper: >=1% foreground in DNA channel


def patch_attention_to_expose_weights(attn_module):
    """
    Monkey-patch a timm Attention module (in place) so its forward pass
    stores softmax attention weights on `attn_module.attn_map`.

    timm's fused SDPA path never materialises attention weights, so we
    force the manual path. This only rewires the module's `forward` at
    inference time in this script - dino.py / train_dino.py are untouched
    and training behavior is unaffected.
    """
    attn_module.fused_attn = False

    def forward(x, attn_mask=None, is_causal=False):
        B, N, C = x.shape
        qkv = attn_module.qkv(x).reshape(
            B, N, 3, attn_module.num_heads, attn_module.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = attn_module.q_norm(q), attn_module.k_norm(k)
        q = q * attn_module.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        attn_module.attn_map = attn.detach()  # (B, heads, N, N)
        attn = attn_module.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, attn_module.attn_dim)
        x = attn_module.norm(x)
        x = attn_module.proj(x)
        x = attn_module.proj_drop(x)
        return x

    attn_module.forward = forward


def select_foreground_crops(img, otsu_thresh, max_crops):
    """
    Same tiling + foreground-filter logic as extract_embeddings.embed_fov,
    but returns the crops themselves (for attention visualization) instead
    of pooled embeddings.
    """
    C, H, W = img.shape
    n_h, n_w = H // CROP_SIZE, W // CROP_SIZE

    if n_h == 0 or n_w == 0:
        return img.unsqueeze(0)

    crops = (img.unfold(1, CROP_SIZE, CROP_SIZE)
                .unfold(2, CROP_SIZE, CROP_SIZE)
                .permute(1, 2, 0, 3, 4)
                .contiguous()
                .view(n_h * n_w, C, CROP_SIZE, CROP_SIZE))

    fg = (crops[:, 4] > otsu_thresh).float().mean(dim=[1, 2]) >= MIN_FG
    crops = crops[fg]

    if len(crops) == 0:
        r0, c0 = (H - CROP_SIZE) // 2, (W - CROP_SIZE) // 2
        crops = img[:, r0:r0 + CROP_SIZE, c0:c0 + CROP_SIZE].unsqueeze(0)

    return crops[:max_crops]


@torch.no_grad()
def extract_attention_for_crop(model, crop, device):
    """
    Run one 224x224 crop through the model and return the CLS->patch
    attention map from the last transformer block: (heads, grid, grid).
    Assumes patch_attention_to_expose_weights() has already been applied
    to model.vit.blocks[-1].attn.
    """
    attn_block = model.vit.blocks[-1].attn
    x = crop.unsqueeze(0).to(device)
    model(x)  # triggers patched forward, populates attn_block.attn_map

    attn = attn_block.attn_map[0]                  # (heads, N, N)
    cls_attn = attn[:, 0, 1:]                       # CLS query -> patch keys
    grid = int(round(cls_attn.shape[-1] ** 0.5))    # 16 for 224/14
    cls_attn = cls_attn.reshape(attn.shape[0], grid, grid)
    return cls_attn.cpu().numpy()


def main():
    """
    Extract CLS-token attention maps from a trained checkpoint (typically
    the final one). Does not touch training - this is a read-only,
    post-hoc diagnostic, mirroring extract_embeddings.py's CLI.

    Usage:
        python -m analysis.extract_attention_maps --run_dir /path/to/checkpoints
        python -m analysis.extract_attention_maps --run_dir /path/to/checkpoints --epoch 150
    """
    torch.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--epoch", type=int, default=None,
                         help="Specific epoch checkpoint to use (default: latest)")
    parser.add_argument("--num_samples", type=int, default=8,
                         help="Number of FOVs to sample")
    parser.add_argument("--max_crops_per_fov", type=int, default=4,
                         help="Max foreground crops to extract attention for, per FOV")
    parser.add_argument("--output_dir", default=None,
                         help="Default: <CP_OUTPUT_ROOT>/attention_maps/<run_name>")
    args = parser.parse_args()

    if not os.path.exists(args.run_dir):
        raise ValueError("run_dir does not exist")

    run_name = os.path.basename(os.path.normpath(args.run_dir))
    output_dir = args.output_dir or os.path.join(
        os.environ["CP_OUTPUT_ROOT"], "attention_maps", run_name
    )
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    checkpoints = get_checkpoints(args.run_dir)
    if args.epoch is not None:
        matches = [(e, p) for (e, p) in checkpoints if e == args.epoch]
        if not matches:
            raise ValueError(f"No checkpoint found for epoch {args.epoch}")
        epoch, checkpoint_path = matches[0]
    else:
        epoch, checkpoint_path = sorted(checkpoints)[-1]

    print(f"Using checkpoint: {checkpoint_path} (epoch {epoch})")
    model = load_model(checkpoint_path, device)
    patch_attention_to_expose_weights(model.vit.blocks[-1].attn)

    dataset = CellPaintingDataset(
        processed_dir=os.path.join(os.environ["CP_OUTPUT_ROOT"], "data/tiles_qc"),
        return_full_image=True,
    )

    rng = np.random.default_rng(42)
    n_samples = min(args.num_samples, len(dataset))
    indices = rng.choice(len(dataset), size=n_samples, replace=False)

    attn_maps, crop_imgs, plates, wells = [], [], [], []

    for idx in indices:
        sample = dataset[int(idx)]
        crops = select_foreground_crops(
            sample["image"], sample["otsu_threshold"], args.max_crops_per_fov
        )
        for c in range(len(crops)):
            attn = extract_attention_for_crop(model, crops[c], device)
            attn_maps.append(attn)
            crop_imgs.append(crops[c].numpy().astype(np.float16))
            plates.append(sample["plate"])
            wells.append(sample["well"])

    attn_maps = np.stack(attn_maps)   # (N, heads, grid, grid)
    crop_imgs = np.stack(crop_imgs)   # (N, C, 224, 224)
    plates    = np.array(plates)
    wells     = np.array(wells)

    print("Attention maps shape:", attn_maps.shape)

    np.save(os.path.join(output_dir, f"attn_maps_epoch_{epoch}.npy"), attn_maps)
    np.save(os.path.join(output_dir, f"attn_crops_epoch_{epoch}.npy"), crop_imgs)
    np.save(os.path.join(output_dir, f"attn_plates_epoch_{epoch}.npy"), plates)
    np.save(os.path.join(output_dir, f"attn_wells_epoch_{epoch}.npy"), wells)

    print(f"Saved epoch {epoch} attention maps -> {output_dir}")


if __name__ == "__main__":
    main()

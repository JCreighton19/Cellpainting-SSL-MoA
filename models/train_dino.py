import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import random
import numpy as np
import time
from datetime import datetime
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF

from datasets.dataset import CellPaintingDataset
from models.dino_loss import DINOLoss
from models.dino import CellPaintingViT


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


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    sys.stdout.reconfigure(line_buffering=True)

    # Make checkpoints dir, if it does not yet exist
    run_dir = os.path.join(
        os.environ["CP_OUTPUT_ROOT"],
        "checkpoints",
        datetime.now().strftime("%m%d%y_%H%M")
    )
    os.makedirs(run_dir, exist_ok=True)

    # Device
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Using device:", device)

    N_CLASSES = 8    # MoA groups per batch
    K_PER_CLASS = 4  # tiles per group; effective batch size = 32
    SUPCON_WEIGHT = 0.2

    # Dataset
    data_dir = os.path.join(os.environ["CP_OUTPUT_ROOT"], "data/tiles_qc")
    dataset = CellPaintingDataset(
        processed_dir=data_dir,
        random_crop=True,
        k_per_class=K_PER_CLASS
    )
    debug_dir = os.path.join(run_dir, "aug_debug")
    os.makedirs(debug_dir, exist_ok=True)

    print("\n=== Dataset Summary ===")
    print(f"Dataset directory: {data_dir}")
    print(f"Dataset length: {len(dataset)}")

    def worker_init_fn(worker_id):
        seed = torch.initial_seed() % 2 ** 32
        random.seed(seed)
        np.random.seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=N_CLASSES,
        shuffle=False, # defined shuffling in sampler
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=worker_init_fn
    )

    def random_crop_batch(x, scale=(0.6, 1.0)):
        B, C, H, W = x.shape
        s = random.uniform(*scale)
        th, tw = int(H * s), int(W * s)
        if th == H and tw == W:
            return x
        tops  = [random.randint(0, H - th) for _ in range(B)]
        lefts = [random.randint(0, W - tw) for _ in range(B)]
        crops = torch.stack([x[i, :, tops[i]:tops[i] + th, lefts[i]:lefts[i] + tw] for i in range(B)])
        return F.interpolate(crops, size=(H, W), mode="bilinear", align_corners=False)

    def teacher_augment(x):
        B, C, H, W = x.shape

        # random 90° rotation per sample — 3 batch rot90s + index, no Python loop
        k = torch.randint(0, 4, (B,), device=x.device)
        all_rots = torch.stack([
            x,
            torch.rot90(x, 1, dims=[2, 3]),
            torch.rot90(x, 2, dims=[2, 3]),
            torch.rot90(x, 3, dims=[2, 3]),
        ])  # (4, B, C, H, W)
        x = all_rots[k, torch.arange(B, device=x.device)]

        # flips
        flip_h = torch.rand(B,device=x.device) < 0.5
        flip_w = torch.rand(B,device=x.device) < 0.5
        x = torch.where(
            flip_h[:, None, None, None],
            torch.flip(x, dims=[2]),
            x
        )
        x = torch.where(
            flip_w[:, None, None, None],
            torch.flip(x, dims=[3]),
            x
        )

        # Small intensity changes only
        intensity = torch.empty(
            B, 1, 1, 1,
            device=x.device
        ).uniform_(0.9, 1.1)
        x = x * intensity

        channel_scale = torch.empty(
            B, 5, 1, 1,
            device=x.device
        ).uniform_(0.95, 1.05)
        x = x * channel_scale

        return x.clamp(0, 1)


    def student_augment(x):
        B, C, H, W = x.shape

        # random 90° rotation per sample — 3 batch rot90s + index, no Python loop
        k = torch.randint(0, 4, (B,), device=x.device)
        all_rots = torch.stack([
            x,
            torch.rot90(x, 1, dims=[2, 3]),
            torch.rot90(x, 2, dims=[2, 3]),
            torch.rot90(x, 3, dims=[2, 3]),
        ])  # (4, B, C, H, W)
        x = all_rots[k, torch.arange(B, device=x.device)]

        # flips
        flip_h = torch.rand(B, device=x.device) < 0.5
        flip_w = torch.rand(B, device=x.device) < 0.5
        x = torch.where(
            flip_h[:, None, None, None],
            torch.flip(x, dims=[2]),
            x
        )
        x = torch.where(
            flip_w[:, None, None, None],
            torch.flip(x, dims=[3]),
            x
        )

        # stronger intensity perturbation
        intensity = torch.empty(
            B, 1, 1, 1,
            device=x.device
        ).uniform_(0.7, 1.3)
        x = x * intensity

        channel_scale = torch.empty(
            B, 5, 1, 1,
            device=x.device
        ).uniform_(0.85, 1.15)
        x = x * channel_scale

        # gaussian blur
        if random.random() < 0.5:
            x = TF.gaussian_blur(x,
                kernel_size=9,
                sigma=(0.5, 1.5)
            )

        # noise
        noise_mask = (
                torch.rand(
                    B, 1, 1, 1,
                    device=x.device
                ) < 0.5
        )

        x = x + noise_mask * torch.randn_like(x) * 0.02
        return x.clamp(0, 1)

    # Models
    student_enc = CellPaintingViT(in_channels=5).to(device)
    teacher_enc = CellPaintingViT(in_channels=5).to(device)
    teacher_enc.load_state_dict(student_enc.state_dict())

    for p in teacher_enc.parameters():
        p.requires_grad = False

    class DINOHead(nn.Module):
        def __init__(self, dim=384, proj_dim=8192):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(dim, 512),
                nn.GELU(),
                nn.Linear(512, proj_dim)
            )

        def forward(self, x):
            return self.mlp(x)

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

    student_head = DINOHead().to(device)
    teacher_head = DINOHead().to(device)
    teacher_head.load_state_dict(student_head.state_dict())

    supcon_head = SupConHead().to(device)

    teacher_enc.eval()
    teacher_head.eval()

    for p in teacher_head.parameters():
        p.requires_grad = False

    # Loss + optimizer
    dino_loss = DINOLoss().to(device)
    optimizer = torch.optim.AdamW(
        list(student_enc.parameters()) +
        list(student_head.parameters()) +
        list(supcon_head.parameters()),
        lr=5e-5,
        weight_decay=0.04
    )

    # Teacher update
    @torch.no_grad()
    def update_teacher(student_enc, teacher_enc,
                       student_head, teacher_head, momentum=0.995):
        for ps, pt in zip(student_enc.parameters(), teacher_enc.parameters()):
            pt.mul_(momentum).add_(ps * (1 - momentum))
        for ps, pt in zip(student_head.parameters(), teacher_head.parameters()):
            pt.mul_(momentum).add_(ps * (1 - momentum))

    # Precomputed constants reused every step
    moa_labels = torch.arange(N_CLASSES, device=device).repeat_interleave(K_PER_CLASS)  # (32,)
    sc_labels  = moa_labels.repeat(2)                                                     # (64,)
    trainable_params = (
        list(student_enc.parameters()) +
        list(student_head.parameters()) +
        list(supcon_head.parameters())
    )

    # Training loop
    n_epochs = 30
    losses = []

    VIS_CHANS = [0, 3, 4]  # Mito, ER, DNA
    def to_vis(x):
        x = x[:, VIS_CHANS]  # select channels
        return x.detach().cpu().float().clamp(0, 1)

    for epoch in range(n_epochs):
        epoch_start = time.perf_counter()
        epoch_start_dt = datetime.now()
        print(f"\nEpoch {epoch + 1}/{n_epochs} | Start time: {epoch_start_dt.strftime('%I:%M %p')}")

        student_enc.train()
        student_head.train()
        total_loss = 0

        cos_sims = []
        embed_stds = []
        embed_norms = []

        for step, batch in enumerate(loader):
            raw = batch["image"].to(device, non_blocking=True)  # (N_CLASSES, K_PER_CLASS, C, H, W)
            images = raw.view(N_CLASSES * K_PER_CLASS, *raw.shape[2:])  # (32, C, H, W)

            # create global views
            teacher_view1 = teacher_augment(
                random_crop_batch(images,scale=(0.8, 1.0))
            )

            teacher_view2 = teacher_augment(
                random_crop_batch(images,scale=(0.8, 1.0))
            )

            student_view1 = student_augment(
                random_crop_batch(images,scale=(0.6, 1.0))
            )

            student_view2 = student_augment(
                random_crop_batch(images,scale=(0.6, 1.0))
            )

            # Save augmented images for visual inspection/debugging
            if step % 200 == 0 and epoch == 0:
                n = 10
                orig = to_vis(images[:n])
                v1 = to_vis(student_view1[:n])
                v2 = to_vis(teacher_view1[:n])

                # interleave per sample: [orig, view1, view2]
                stacked = torch.stack([orig, v1, v2], dim=1)
                # shape: (n, 3, C, H, W)
                stacked = stacked.view(n * 3, *orig.shape[1:])
                # flatten into grid rows
                grid = make_grid(
                    stacked,
                    nrow=3  # 3 columns = (orig | view1 | view2)
                )

                save_image(grid, f"{debug_dir}/grid_e{epoch}_s{step}.png")

            # ENCODING — cache backbone to share across DINO and SupCon heads
            z1 = student_enc(student_view1)   # (B, 384)
            z2 = student_enc(student_view2)   # (B, 384)
            s1 = student_head(z1)
            s2 = student_head(z2)
            sc_z1 = supcon_head(z1)           # (B, 128) L2-normalized
            sc_z2 = supcon_head(z2)           # (B, 128) L2-normalized

            # Save augmented embeddings for inspection/debugging
            if step % 200 == 0 and epoch == 0:
                emb_dir = os.path.join(run_dir, "debug_embeddings")
                os.makedirs(emb_dir, exist_ok=True)

                np.save(
                    f"{emb_dir}/s1_e{epoch}_s{step}.npy",
                    s1.detach().cpu().numpy()
                )

                np.save(
                    f"{emb_dir}/s2_e{epoch}_s{step}.npy",
                    s2.detach().cpu().numpy()
                )

            with torch.no_grad():
                t1 = teacher_head(teacher_enc(teacher_view1))
                t2 = teacher_head(teacher_enc(teacher_view2))
                # collapse / diversity check
                all_s = torch.cat([s1, s2], dim=0)
                embed_std = all_s.std(dim=0).mean().item()
                embed_norm = all_s.norm(dim=1).mean().item()
                cos_sim = F.cosine_similarity(s1.detach(), t1.detach(), dim=1).mean().item()

            cos_sims.append(cos_sim)
            embed_stds.append(embed_std)
            embed_norms.append(embed_norm)

            dino_loss_val = (
                dino_loss(s1, t2, epoch) +
                dino_loss(s2, t1, epoch)
            ) / 2

            sc_features = torch.cat([sc_z1, sc_z2], dim=0)    # (2B, 128)
            sc_loss = supcon_loss(sc_features, sc_labels)

            loss = dino_loss_val + SUPCON_WEIGHT * sc_loss

            if step % 100 == 0:
                print(f"{step}/{len(loader)} steps "
                    f"loss={loss.item():.4f} "
                    f"dino={dino_loss_val.item():.4f} "
                    f"sc={sc_loss.item():.4f} "
                    f"std={embed_std:.4f} "
                    f"norm={embed_norm:.4f} "
                    f"cos_sim={cos_sim:.4f}"
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 3.0)
            optimizer.step()

            update_teacher(student_enc, teacher_enc,
                           student_head, teacher_head)

            total_loss += loss.item()

        epoch_end = time.perf_counter()
        epoch_time = epoch_end - epoch_start

        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)

        avg_cos = np.mean(cos_sims)
        avg_std = np.mean(embed_stds)
        avg_norm = np.mean(embed_norms)
        eps = 1e-6
        collapse_score = avg_std / (1.0 - avg_cos + eps)

        torch.save({
            "student_enc": student_enc.state_dict(),
            "student_head": student_head.state_dict(),
            "teacher_enc": teacher_enc.state_dict(),
            "supcon_head": supcon_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": avg_loss
        }, os.path.join(run_dir, f"dino_epoch_{epoch + 1}.pt"))

        print(
            f"Epoch {epoch + 1}/{n_epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"cos_sim: {avg_cos:.4f} | "
            f"std: {avg_std:.4f} | "
            f"norm: {avg_norm:.4f} | "
            f"collapse: {collapse_score:.4f} | "
            f"Time: {epoch_time / 60:.2f} min\n"
        )

    print(f"Finished training. Checkpoints saved at {run_dir}")

if __name__ == "__main__":
    main()
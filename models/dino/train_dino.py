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
from models.dino.dino_loss import DINOLoss
from models.dino.dino import CellPaintingViT
from models.config import CONFIG


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

    # Dataset
    data_dir = os.path.join(os.environ["CP_OUTPUT_ROOT"], "data/tiles_qc")
    dataset = CellPaintingDataset(
        processed_dir=data_dir,
        random_crop=True
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
        batch_size=CONFIG["batch_size"],
        shuffle=False, # defined shuffling in sampler
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=worker_init_fn
    )

    def random_crop_batch(x, scale=(0.6, 1.0)):
        B, C, H, W = x.shape

        out = []
        for i in range(B):
            s = random.uniform(*scale)
            th = int(H * s)
            tw = int(W * s)

            if th == H and tw == W:
                cropped = x[i]
            else:
                top = random.randint(0, H - th)
                left = random.randint(0, W - tw)
                cropped = x[i:i + 1, :, top:top + th, left:left + tw]
                cropped = torch.nn.functional.interpolate(
                    cropped,
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False
                )[0]

            out.append(cropped)
        return torch.stack(out)

    def teacher_augment(x):
        B, C, H, W = x.shape

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
        def __init__(self, dim=384, proj_dim=256):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(dim, 512),
                nn.GELU(),
                nn.Linear(512, proj_dim)
            )

        def forward(self, x):
            return self.mlp(x)

    student_head = DINOHead().to(device)
    teacher_head = DINOHead().to(device)
    teacher_head.load_state_dict(student_head.state_dict())

    teacher_enc.eval()
    teacher_head.eval()

    for p in teacher_head.parameters():
        p.requires_grad = False

    # Loss + optimizer
    dino_loss = DINOLoss().to(device)
    optimizer = torch.optim.AdamW(
        list(student_enc.parameters()) + list(student_head.parameters()),
        lr = CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )

    # Teacher update
    @torch.no_grad()
    def update_teacher(student_enc, teacher_enc,
                       student_head, teacher_head, momentum=0.995):
        with torch.no_grad():
            for ps, pt in zip(student_enc.parameters(), teacher_enc.parameters()):
                pt.mul_(momentum).add_(ps * (1 - momentum))

            for ps, pt in zip(student_head.parameters(), teacher_head.parameters()):
                pt.mul_(momentum).add_(ps * (1 - momentum))

    # Training loop
    n_epochs = CONFIG["n_epochs"]
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
            images = batch["image"].to(device, non_blocking=True)  # (B, C, H, W)

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

            # ENCODING
            s1 = student_head(student_enc(student_view1))
            s2 = student_head(student_enc(student_view2))

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

            with torch.no_grad():
                # collapse / diversity check
                all_s = torch.cat([s1, s2], dim=0)
                embed_std = all_s.std(dim=0).mean().item()
                embed_norm = all_s.norm(dim=1).mean().item()

                cos_sim = F.cosine_similarity(
                    s1.detach(),
                    t1.detach(),
                    dim=1
                ).mean().item()

            cos_sims.append(cos_sim)
            embed_stds.append(embed_std)
            embed_norms.append(embed_norm)

            loss = (
               dino_loss(s1, t2) +
               dino_loss(s2, t1)
            ) / 2

            if step % 100 == 0:
                print(f"{step}/{len(loader)} steps "
                    f"loss={loss.item():.4f} "
                    f"std={embed_std:.4f} "
                    f"norm={embed_norm:.4f} "
                    f"cos_sim={cos_sim:.4f}"
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(student_enc.parameters()) +
                list(student_head.parameters()),
                3.0
            )
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
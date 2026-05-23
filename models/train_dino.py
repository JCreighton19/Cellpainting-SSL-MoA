import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import sys
import random
import numpy as np
import time
from datetime import datetime

from datasets.dataset import CellPaintingDataset
from models.dino_loss import DINOLoss
from models.dino import CellPaintingViT


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
    data_dir = os.path.join(os.environ["CP_OUTPUT_ROOT"], "data/processed/tiles")
    dataset = CellPaintingDataset(
        processed_dir=data_dir,
        augment = True,
        random_crop = True
    )

    print("\n=== Dataset Summary ===")
    print(f"Dataset directory: {data_dir}")
    print(f"Dataset length: {len(dataset)}")

    def worker_init_fn(worker_id):
        seed = torch.initial_seed() % 2 ** 32
        random.seed(seed)
        np.random.seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False, # defined shuffling in sampler
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=worker_init_fn
    )

    import torch.nn.functional as F

    def batch_crop(x, scale_min, scale_max):
        B, C, H, W = x.shape
        device = x.device

        scales = torch.empty(B, device=device).uniform_(scale_min, scale_max)
        crop_sizes = (scales * H).long().clamp(1, H)

        # random centers
        cy = torch.randint(0, H, (B,), device=device)
        cx = torch.randint(0, W, (B,), device=device)

        # normalized coords grid
        theta = torch.zeros(B, 2, 3, device=device)

        for i in range(B):
            size = crop_sizes[i].item()

            y1 = (cy[i] - size / 2) / (H / 2)
            x1 = (cx[i] - size / 2) / (W / 2)

            theta[i, 0, 0] = size / H
            theta[i, 1, 1] = size / W
            theta[i, 0, 2] = x1
            theta[i, 1, 2] = y1

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        crops = F.grid_sample(x, grid, align_corners=False)

        return crops

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
        lr=5e-5,
        weight_decay=0.04
    )

    # Teacher update
    @torch.no_grad()
    def update_teacher(student_enc, teacher_enc,
                       student_head, teacher_head, momentum=0.99):
        with torch.no_grad():
            for ps, pt in zip(student_enc.parameters(), teacher_enc.parameters()):
                pt.mul_(momentum).add_(ps * (1 - momentum))

            for ps, pt in zip(student_head.parameters(), teacher_head.parameters()):
                pt.mul_(momentum).add_(ps * (1 - momentum))

    # Training loop
    n_epochs = 60
    losses = []

    for epoch in range(n_epochs):

        epoch_start = time.perf_counter()

        epoch_start_dt = datetime.now()
        print(f"\nEpoch {epoch + 1}/{n_epochs} | Start time: {epoch_start_dt.strftime('%I:%M %p')}")

        student_enc.train()
        student_head.train()
        total_loss = 0

        for step, batch in enumerate(loader):
            images = batch["image"].to(device, non_blocking=True)  # (B, C, H, W)

            # create multi-crop views ON GPU
            global_views_1 = batch_crop(images, 0.7, 1.0)
            global_views_2 = batch_crop(images, 0.7, 1.0)
            local_views = batch_crop(images, 0.3, 0.5)

            # ENCODING
            s1 = student_head(student_enc(global_views_1))
            s2 = student_head(student_enc(global_views_2))
            s3 = student_head(student_enc(local_views))

            with torch.no_grad():
                t1 = teacher_head(teacher_enc(global_views_1))
                t2 = teacher_head(teacher_enc(global_views_2))

            with torch.no_grad():
                # collapse / diversity check
                all_s = torch.cat([s1, s2, s3], dim=0)
                embed_std = all_s.std(dim=0).mean().item()
                embed_norm = all_s.norm(dim=1).mean().item()

                cos_sim = (
                      F.cosine_similarity(s1, s2, dim=1).mean().item() +
                      F.cosine_similarity(s1, s3, dim=1).mean().item()
                  ) / 2

            loss = (
               dino_loss(s1, t2) +
               dino_loss(s2, t1) +
               dino_loss(s3, t1) +
               dino_loss(s3, t2)
            ) / 4

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

        torch.save({
            "student_enc": student_enc.state_dict(),
            "student_head": student_head.state_dict(),
            "teacher_enc": teacher_enc.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": avg_loss
        }, os.path.join(run_dir, f"dino_epoch_{epoch + 1}.pt"))

        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {total_loss/len(loader):.4f} | Total Time: {epoch_time/60:.2f} min\n")

    print(f"Finished training. Checkpoints saved at {run_dir}")

if __name__ == "__main__":
    main()
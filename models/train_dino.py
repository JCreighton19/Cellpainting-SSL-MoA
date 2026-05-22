import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
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
    os.makedirs("checkpoints", exist_ok=True)

    # Device
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Using device:", device)

    # Dataset
    metadata_path = os.path.join(os.environ["CP_OUTPUT_ROOT"], "data/processed/master_metadata.parquet")
    data_root = os.environ["CP_DATA_ROOT"]
    dataset = CellPaintingDataset(
        metadata_path=metadata_path,
        data_root=data_root,
        channels=[1,2,3,4,5],
        tile_size=224
    )

    print("\n=== Dataset Summary ===")
    print("Compounds:", dataset.metadata["pert_iname"].nunique())
    print("Plates:", dataset.metadata["plate"].nunique())
    print("Genes:", dataset.metadata["gene"].nunique())
    print("Wells:", dataset.metadata["well"].nunique())

    def worker_init_fn(worker_id):
        seed = torch.initial_seed() % 2 ** 32
        random.seed(seed)
        np.random.seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_init_fn
    )

    # Augmentation
    def augment(x):
        # x: (C, H, W)

        # spatial flips
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2])
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[1])

        # intensity jitter
        x = x * (0.7 + 0.6 * torch.rand(1))

        # Gaussian noise
        x = x + 0.01 * torch.randn_like(x)

        # channel dropout
        if torch.rand(1).item() < 0.3:
            ch = torch.randint(0, x.shape[0], (1,)).item()
            x[ch] = 0

        # slight blur
        if torch.rand(1).item() < 0.5:
            x = transforms.functional.gaussian_blur(
                x,
                kernel_size=5
            )

        return x

    def global_crop(x):
        # stronger crop but still large view
        C, H, W = x.shape
        crop_size = random.randint(144, 224)

        r = random.randint(0, H - crop_size)
        c = random.randint(0, W - crop_size)

        x = x[:, r:r + crop_size, c:c + crop_size]

        x = F.interpolate(
            x.unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        return augment(x)

    def local_crop(x):
        # small crop = forces partial view learning
        C, H, W = x.shape
        crop_size = random.randint(64, 128)

        r = random.randint(0, H - crop_size)
        c = random.randint(0, W - crop_size)

        x = x[:, r:r + crop_size, c:c + crop_size]

        x = F.interpolate(
            x.unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        return augment(x)

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
                       student_head, teacher_head, momentum=0.999):
        with torch.no_grad():
            for ps, pt in zip(student_enc.parameters(), teacher_enc.parameters()):
                pt.mul_(momentum).add_(ps * (1 - momentum))

            for ps, pt in zip(student_head.parameters(), teacher_head.parameters()):
                pt.mul_(momentum).add_(ps * (1 - momentum))

    # Training loop
    n_epochs = 10
    losses = []

    for epoch in range(n_epochs):

        epoch_start = time.perf_counter()

        epoch_start_dt = datetime.now()
        print(f"Epoch {epoch + 1}/{n_epochs} | Start time: {epoch_start_dt.strftime('%I:%M %p')}")

        student_enc.train()
        student_head.train()
        total_loss = 0

        for step, batch in enumerate(loader):
            images = batch["image"]

            # GLOBAL / LOCAL VIEWS
            global_views_1 = torch.stack([global_crop(img) for img in images]).to(device)
            global_views_2 = torch.stack([global_crop(img) for img in images]).to(device)

            local_views = torch.stack([local_crop(img) for img in images]).to(device)

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
               dino_loss(s1, t1) +
               dino_loss(s1, t2) +
               dino_loss(s2, t1) +
               dino_loss(s2, t2) +
               dino_loss(s3, t1)
           ) / 5

            if step % 50 == 0:
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
        }, f"checkpoints/dino_epoch_{epoch + 1}.pt")

        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {total_loss/len(loader):.4f} | Total Time: {epoch_time/60:.2f} min\n")

if __name__ == "__main__":
    main()
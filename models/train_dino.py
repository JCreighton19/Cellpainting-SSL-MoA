import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys
import random
import numpy as np

from dataset import CellPaintingDataset
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
    BASE = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.join(BASE, "../data/processed/master_metadata.parquet")
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
        shuffle=True,
        num_workers=2
    )

    # Augmentation
    augment = transforms.Compose([
        # Start with simple augmentations
        #transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x))
        #transforms.RandomRotation(180),
        #transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
    ])

    def apply_augment(batch):
        return torch.stack([augment(img) for img in batch])

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
        lr=1e-4,
        weight_decay=0.04
    )

    # Teacher update
    @torch.no_grad()
    def update_teacher(student_enc, teacher_enc,
                       student_head, teacher_head, momentum=0.996):
        for ps, pt in zip(student_enc.parameters(), teacher_enc.parameters()):
            pt.data.mul_(momentum).add_((1 - momentum) * ps.data)

        for ps, pt in zip(student_head.parameters(), teacher_head.parameters()):
            pt.data.mul_(momentum).add_((1 - momentum) * ps.data)

    # Training loop
    n_epochs = 10
    losses = []

    for epoch in range(n_epochs):
        student_enc.train()
        student_head.train()
        total_loss = 0

        for step, batch in enumerate(loader):
            images = batch["image"]

            x1 = apply_augment(images)
            x2 = apply_augment(images)

            # Confirm shapes are as expected
            assert x1.ndim == 4
            assert x1.shape[1] == 5
            assert x1.shape[-1] == 224

            x1 = x1.to(device)
            x2 = x2.to(device)

            s1 = student_head(student_enc(x1))
            s2 = student_head(student_enc(x2))
            with torch.no_grad():
                embed_std = torch.cat([s1, s2]).std(dim=0).mean().item()
                embed_norm = torch.cat([s1, s2]).norm(dim=1).mean().item()

            with torch.no_grad():
                t1 = teacher_head(teacher_enc(x1))
                t2 = teacher_head(teacher_enc(x2))

            loss = (dino_loss(s1, t2) + dino_loss(s2, t1)) / 2

            if step % 20 == 0:
                print(
                    f"loss={loss.item():.4f} "
                    f"std={embed_std:.4f} "
                    f"norm={embed_norm:.4f} "
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

        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)

        torch.save({
            "student_enc": student_enc.state_dict(),
            "student_head": student_head.state_dict(),
            "teacher_enc": teacher_enc.state_dict(),
            "epoch": epoch,
            "loss": avg_loss
        }, f"checkpoints/dino_epoch_{epoch + 1}.pt")

        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {total_loss/len(loader):.4f}")

if __name__ == "__main__":
    main()
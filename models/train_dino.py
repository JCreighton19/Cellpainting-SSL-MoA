import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CellPaintingDataset
from models.dino_loss import DINOLoss
from models.dino import CellPaintingViT


def main():

    # Device
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Using device:", device)

    # Dataset
    dataset = CellPaintingDataset(
        "../data/plate1",
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
        transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
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
    n_epochs = 1

    for epoch in range(n_epochs):
        total_loss = 0

        for batch in loader:
            batch = batch.to(device)

            x1 = apply_augment(batch)
            x2 = apply_augment(batch)

            s1 = student_head(student_enc(x1))
            s2 = student_head(student_enc(x2))

            with torch.no_grad():
                t1 = teacher_head(teacher_enc(x1)).detach()
                t2 = teacher_head(teacher_enc(x2)).detach()

            loss = (dino_loss(s1, t2) + dino_loss(s2, t1)) / 2

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(student_enc.parameters()), 3.0
            )
            optimizer.step()

            update_teacher(student_enc, teacher_enc,
                           student_head, teacher_head)

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {total_loss/len(loader):.4f}")


if __name__ == "__main__":
    main()
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
from torchvision.utils import save_image
from torchvision.utils import make_grid

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
        random_crop=False
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
        batch_size=32,
        shuffle=False, # defined shuffling in sampler
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=worker_init_fn
    )

    import torch.nn.functional as F

    def shared_crop(x, scale_min=0.6, scale_max=1.0, out_size=224):
        B, C, H, W = x.shape
        device = x.device

        def make_crop():
            scale = torch.empty(1, device=device).uniform_(scale_min, scale_max).item()
            size = min(int(scale * H), H)

            cy = torch.randint(0, H, (B,), device=device)
            cx = torch.randint(0, W, (B,), device=device)

            out = []
            for i in range(B):
                y1 = max(0, min(H - size, cy[i].item() - size // 2))
                x1 = max(0, min(W - size, cx[i].item() - size // 2))

                patch = x[i:i + 1, :, y1:y1 + size, x1:x1 + size]
                patch = F.interpolate(patch, size=(out_size, out_size),
                                      mode="bilinear", align_corners=False)
                out.append(patch)

            return torch.cat(out, dim=0)

        return make_crop(), make_crop()

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

        cos_sims = []
        embed_stds = []
        embed_norms = []

        for step, batch in enumerate(loader):
            images = batch["image"].to(device, non_blocking=True)  # (B, C, H, W)

            # create conservative global views
            global_views_1, global_views_2 = shared_crop(images, 0.75, 0.9)

            # Save augmented images for visual inspection/debugging
            if step % 200 == 0 and epoch == 0:
                for c in range(images.shape[1]):
                    v1 = global_views_1[:4, c:c + 1]
                    v2 = global_views_2[:4, c:c + 1]

                    grid_c = make_grid(
                        torch.cat([
                            images[:4, c:c + 1].detach().cpu(),
                            v1.detach().cpu(),
                            v2.detach().cpu()
                        ], dim=0),
                        nrow=4
                    )

                    save_image(
                        grid_c,
                        f"{debug_dir}/grid_c{c}_e{epoch}_s{step}.png"
                    )

            # ENCODING
            s1 = student_head(student_enc(global_views_1))
            s2 = student_head(student_enc(global_views_2))

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
                t1 = teacher_head(teacher_enc(global_views_1))
                t2 = teacher_head(teacher_enc(global_views_2))

            with torch.no_grad():
                # collapse / diversity check
                all_s = torch.cat([s1, s2], dim=0)
                embed_std = all_s.std(dim=0).mean().item()
                embed_norm = all_s.norm(dim=1).mean().item()

                cos_sim = F.cosine_similarity(
                    s1, s2, dim=1
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
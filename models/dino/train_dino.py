import math
import torch
from torch.utils.data import DataLoader
import os
import sys
import random
import numpy as np
import time
from datetime import datetime
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from datasets.dataset import CellPaintingDataset
from models.dino.dino_loss import DINOLoss
from models.dino.dino import CellPaintingViT, DINOHead
from models.config import CONFIG
from utils.foreground_crop import foreground_crop


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    sys.stdout.reconfigure(line_buffering=True)

    # Make checkpoints dir, or resume an existing one
    resume_dir = os.environ.get("RESUME_DIR", "")
    if resume_dir:
        run_dir = resume_dir
    else:
        run_dir = os.path.join(
            os.environ["CP_OUTPUT_ROOT"],
            "checkpoints",
            datetime.now().strftime("%m%d%y_%H%M")
        )
    os.makedirs(run_dir, exist_ok=True)

    # Write run_dir so SLURM can requeue pointing at this directory
    current_run_file = os.path.join(os.environ["CP_OUTPUT_ROOT"], "checkpoints", "current_run")

    # Device
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Using device:", device)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # Dataset
    data_dir  = os.path.join(os.environ["CP_OUTPUT_ROOT"], "data/tiles_qc")
    use_tiles = os.environ.get("CP_USE_TILES", "0") == "1"
    dataset   = CellPaintingDataset(
        processed_dir=data_dir,
        return_full_image=not use_tiles,
        use_tiles=use_tiles,
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
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=worker_init_fn
    )


    def augment(x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Flip along H axis
        flip_h = torch.rand(B, device=x.device) < 0.5
        x = torch.where(
            flip_h[:, None, None, None],
            torch.flip(x, dims=[2]),
            x
        )

        # Flip along W axis
        flip_w = torch.rand(B, device=x.device) < 0.5
        x = torch.where(
            flip_w[:, None, None, None],
            torch.flip(x, dims=[3]),
            x
        )

        # Color step 1: additive intensity shift
        eps = torch.empty(B, device=x.device).uniform_(-0.3, 0.3)
        x = torch.clamp(x + eps[:, None, None, None], 0.0, 1.0)

        # Color step 2: gamma brightness change
        gamma = torch.empty(B, device=x.device).uniform_(0.5, 1.5)
        x = torch.clamp(x ** gamma[:, None, None, None], 0.0, 1.0)

        return x

    # Models
    student_enc = CellPaintingViT(in_channels=5).to(device)
    teacher_enc = CellPaintingViT(in_channels=5).to(device)
    teacher_enc.load_state_dict(student_enc.state_dict())
    for p in teacher_enc.parameters():
        p.requires_grad = False

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
    use_amp = (device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    _total_steps   = CONFIG["n_epochs"] * len(loader)
    _warmup_steps  = 20 * len(loader)
    _min_lr_ratio  = 1e-6 / CONFIG["lr"]

    def lr_lambda(step):
        if step < _warmup_steps:
            return step / max(1, _warmup_steps)
        progress = (step - _warmup_steps) / max(1, _total_steps - _warmup_steps)
        cosine   = 0.5 * (1 + math.cos(math.pi * progress))
        return _min_lr_ratio + (1 - _min_lr_ratio) * cosine


    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if not resume_dir:
        with open(current_run_file, "w") as f:
            f.write(run_dir)

    # Teacher update
    @torch.no_grad()
    def update_teacher(student_enc, teacher_enc,
                       student_head, teacher_head, momentum):
        for tp, sp in zip(
            list(teacher_enc.parameters()) + list(teacher_head.parameters()),
            list(student_enc.parameters()) + list(student_head.parameters())
        ):
            tp.data.mul_(momentum).add_(sp.data, alpha=1 - momentum)


    # Resume from latest checkpoint if available
    start_epoch = 0
    checkpoints = sorted([
        f for f in os.listdir(run_dir) if f.startswith("dino_epoch_") and f.endswith(".pt")
    ], key=lambda x: int(x.split("_")[-1].split(".")[0]))
    if checkpoints:
        ckpt_path = os.path.join(run_dir, checkpoints[-1])
        print(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        student_enc.load_state_dict(ckpt["student_enc"])
        try:
            student_head.load_state_dict(ckpt["student_head"])
            teacher_head.load_state_dict(ckpt["teacher_head"])
        except RuntimeError as e:
            print(f"Could not restore head weights (architecture mismatch): {e}")
            print("Starting heads from scratch — encoder weights preserved.")
        teacher_enc.load_state_dict(ckpt["teacher_enc"])
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except (ValueError, RuntimeError) as e:
            print(f"Could not restore optimizer state (architecture change?): {e}")
            print("Starting optimizer from scratch.")
        if "center" in ckpt:
            if ckpt["center"].shape == dino_loss.center.shape:
                dino_loss.center.copy_(ckpt["center"])
            else:
                print(f"Skipping center restore: shape {ckpt['center'].shape} != {dino_loss.center.shape}")
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    # Training loop
    n_epochs = CONFIG["n_epochs"]
    accum_steps = CONFIG.get("accum_steps", 4)
    m_min = 0.996
    m_max = 0.9998
    losses = []
    total_opt_steps  = n_epochs * max(len(loader) // accum_steps, 1)
    global_opt_step  = start_epoch * max(len(loader) // accum_steps, 1)

    for epoch in range(start_epoch, n_epochs):
        epoch_start = time.perf_counter()
        epoch_start_dt = datetime.now()
        print(f"\n====================="
              f"\nEpoch {epoch + 1}/{n_epochs} | Start time: {epoch_start_dt.strftime('%I:%M %p')}")

        student_enc.train()
        student_head.train()
        total_loss = 0
        progress = epoch / (n_epochs - 1)
        m = m_min + (m_max - m_min) * (0.5 * (1 - np.cos(np.pi * progress)))

        cos_sims = []
        embed_stds = []
        encoder_stds = []
        opt_step = 0

        for step, batch in enumerate(loader):
            if use_tiles:
                # --- TILE PATH ---
                # tiles: (B, N, 5, 224, 224) — precomputed foreground-aware crops
                tiles = batch["tiles"].to(device, non_blocking=True)
                print("tiles:", tiles.shape)
                B, N  = tiles.shape[:2]
                aB    = torch.arange(B, device=device)

                # 2 GLOBAL VIEWS: randomly sample one tile per image
                g1 = tiles[aB, torch.randint(0, N, (B,), device=device)]
                g2 = tiles[aB, torch.randint(0, N, (B,), device=device)]

                # 6 LOCAL VIEWS: directly sample precomputed tiles
                a6B = aB.unsqueeze(0).expand(6, -1).reshape(-1)
                tile_idx = torch.randint(0, tiles.shape[1], (6 * B,), device=device)
                local_crops = tiles[a6B, tile_idx]  # (6B, 5, 224, 224)

            else:
                # --- ORIGINAL PATH (full-image online foreground cropping) ---
                images     = batch["image"].to(device, non_blocking=True)
                thresholds = batch["otsu_threshold"].to(device, non_blocking=True)
                B          = images.shape[0]

                g1 = foreground_crop(images, 224, thresholds)
                g2 = foreground_crop(images, 224, thresholds)
                local_crops = torch.cat(
                    [foreground_crop(images, 96, thresholds) for _ in range(6)], dim=0
                )

            with torch.autocast(device_type="cuda", enabled=use_amp):
                g_t = augment(torch.cat([g1, g2], dim=0))
                g1_t, g2_t = g_t.chunk(2, dim=0)
                g_s = augment(torch.cat([g1, g2], dim=0))
                g1_s, g2_s = g_s.chunk(2, dim=0)

                # 6 LOCAL VIEWS (student): resize 96→224 for ViT, then augment
                local_resized = F.interpolate(local_crops, size=224, mode='bilinear', align_corners=False)
                locals_ = list(augment(local_resized).chunk(6, dim=0))

                # ENCODING
                with torch.no_grad():
                    t_both = teacher_head(teacher_enc(torch.cat([g1_t, g2_t], dim=0)))
                    t1, t2 = t_both.chunk(2, dim=0)

                s_enc = student_enc(torch.cat([g1_s, g2_s], dim=0))
                s_globals = student_head(s_enc)
                s_global, s_global_2 = s_globals.chunk(2, dim=0)
                s_local = list(student_head(student_enc(torch.cat(locals_, dim=0))).chunk(len(locals_), dim=0))

                with torch.no_grad():
                    all_s = torch.cat([s_global, s_global_2] + s_local, dim=0)
                    embed_std = all_s.std(dim=0).mean().item()
                    encoder_std = s_enc.detach().std(dim=0).mean().item()

                    cos_sim = 0.5 * (
                        F.cosine_similarity(s_global.detach(), t2, dim=1).mean() +
                        F.cosine_similarity(s_global_2.detach(), t1, dim=1).mean()
                    ).item()

                cos_sims.append(cos_sim)
                embed_stds.append(embed_std)
                encoder_stds.append(encoder_std)

                loss = 0
                # cross-global losses
                loss += dino_loss(s_global,   t2, epoch=epoch)
                loss += dino_loss(s_global_2, t1, epoch=epoch)

                # local losses
                for sl in s_local:
                    loss += dino_loss(sl, t1, epoch=epoch)
                    loss += dino_loss(sl, t2, epoch=epoch)

                loss = loss / (2 + 2 * len(locals_))

            if step % 100 == 0:
                print(f"{step}/{len(loader)} steps |"
                    f"loss={loss.item():.4f} |"
                    f"std={embed_std:.4f} |"
                    f"enc_std={encoder_std:.4f} |"
                    f"cos_sim={cos_sim:.4f}"
                )

            total_loss += loss.item()
            loss = loss / accum_steps
            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(student_enc.parameters()) +
                    list(student_head.parameters()),
                    3.0
                )

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_opt_step += 1
                wd = CONFIG["weight_decay"] + (
                    CONFIG.get("wd_end", CONFIG["weight_decay"]) - CONFIG["weight_decay"]
                ) * (global_opt_step / total_opt_steps)
                for param_group in optimizer.param_groups:
                    param_group["weight_decay"] = wd
                optimizer.zero_grad(set_to_none=True)
                update_teacher(student_enc, teacher_enc,
                               student_head, teacher_head, m)
                opt_step += 1

                # Recompute teacher AFTER EMA update before updating center
                effective_center_momentum = 0.95

                with torch.no_grad():
                    teacher_batch = torch.cat([t1, t2], dim=0).float()

                    # TEACHER ENTROPY METRICS
                    teacher_logits = (teacher_batch - dino_loss.center) / dino_loss.teacher_temp
                    teacher_probs = F.softmax(teacher_logits, dim=-1)
                    raw_teacher_probs = F.softmax(
                        teacher_batch - dino_loss.center,
                        dim=-1
                    )

                    entropy = -(teacher_probs * teacher_probs.log()).sum(dim=-1).mean()
                    max_prob = teacher_probs.max(dim=-1).values.mean()
                    effective_classes = entropy.exp()
                    if opt_step % 50 == 0:
                        print(
                            f"teacher entropy={entropy.item():.3f} | "
                            f"eff_classes={effective_classes.item():.1f} | "
                            f"top1={max_prob.item():.4f}"
                        )

                    # update center AFTER diagnostics
                    dino_loss.update_center(teacher_batch, effective_center_momentum)


        epoch_end = time.perf_counter()
        epoch_time = epoch_end - epoch_start

        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        avg_cos = np.mean(cos_sims)
        avg_std = np.mean(embed_stds)
        avg_enc_std = np.mean(encoder_stds)
        eps = 1e-6

        torch.save({
            "student_enc": student_enc.state_dict(),
            "student_head": student_head.state_dict(),
            "teacher_enc": teacher_enc.state_dict(),
            "teacher_head": teacher_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "center": dino_loss.center,
            "epoch": epoch,
            "loss": avg_loss
        }, os.path.join(run_dir, f"dino_epoch_{epoch + 1}.pt"))

        print(
            f"Epoch {epoch + 1}/{n_epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"std: {avg_std:.4f} | "
            f"enc_std: {avg_enc_std:.4f} | "
            f"cos_sim: {avg_cos:.4f} | "
            f"Time: {epoch_time / 60:.2f} min\n"
            f"====================="
        )

    print(f"Finished training. Checkpoints saved at {run_dir}")

if __name__ == "__main__":
    main()
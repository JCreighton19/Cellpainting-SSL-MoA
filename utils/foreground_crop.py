import torch


def foreground_crop(images, crop_size, otsu_thresholds):
    """
    Batch foreground-aware crop using Otsu rejection sampling.
    Identical to the inline version previously defined in train_dino.py.
    Works on CPU or GPU (all ops dispatched to images.device).
    """
    B, C, H, W = images.shape
    ts = crop_size
    dna = images[:, 4, :, :]  # DNA channel (B, H, W)

    max_attempts = 10
    y0_final = torch.zeros(B, dtype=torch.long, device=images.device)
    x0_final = torch.zeros(B, dtype=torch.long, device=images.device)
    accepted  = torch.zeros(B, dtype=torch.bool,  device=images.device)

    for _ in range(max_attempts):
        y0 = torch.randint(0, H - ts + 1, (B,), device=images.device)
        x0 = torch.randint(0, W - ts + 1, (B,), device=images.device)

        dna_crops = dna.unfold(1, ts, 1).unfold(2, ts, 1)
        dna_patch = dna_crops[torch.arange(B, device=images.device), y0, x0]

        thresh      = otsu_thresholds[:, None, None]
        fg_fraction = (dna_patch > thresh).float().mean(dim=[1, 2])
        valid       = fg_fraction >= 0.01  # paper threshold: 1% foreground

        newly_accepted = valid & ~accepted
        y0_final = torch.where(newly_accepted, y0, y0_final)
        x0_final = torch.where(newly_accepted, x0, x0_final)
        accepted  = accepted | newly_accepted

        if accepted.all():
            break

    patches = images.unfold(2, ts, 1).unfold(3, ts, 1)
    out = patches[torch.arange(B, device=images.device), :, y0_final, x0_final]
    return out


def foreground_crop_single(image, crop_size, otsu_threshold, max_attempts=10):
    """
    Single-image CPU foreground-aware crop using the same rejection-sampling logic.
    Returns (crop, y0, x0) so callers can perform IoU-based diversity filtering.
    """
    C, H, W = image.shape
    ts  = crop_size
    dna = image[4]

    y0_out = x0_out = 0
    for _ in range(max_attempts):
        y0 = torch.randint(0, H - ts + 1, (1,)).item()
        x0 = torch.randint(0, W - ts + 1, (1,)).item()
        fg = (dna[y0:y0 + ts, x0:x0 + ts] > otsu_threshold).float().mean().item()
        y0_out, x0_out = y0, x0
        if fg >= 0.01:
            break  # accepted — same criterion as batch version

    return image[:, y0_out:y0_out + ts, x0_out:x0_out + ts], y0_out, x0_out

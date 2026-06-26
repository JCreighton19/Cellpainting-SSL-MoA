# Basic config

CONFIG = {
    "lr": 1e-4,
    "n_epochs": 200,
    "batch_size": 32, # GPU limited to 32 without gradient accumulation
    "num_workers": 8,
    "weight_decay": 0.04,
    "wd_end": 0.4,    # paper: linearly increase WD 0.04 → 0.4 over training
    "accum_steps": 4  # effective batch size = batch size × accum_steps
}
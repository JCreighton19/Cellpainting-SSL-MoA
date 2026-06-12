# Basic config

CONFIG = {
    "lr": 5e-4,
    "n_epochs": 60,
    "batch_size": 32, # GPU limited to 32 without gradient accumulation
    "num_workers": 8,
    "weight_decay": 0.04,
    "accum_steps": 4 # effective batch size = batch size × accum_steps
}
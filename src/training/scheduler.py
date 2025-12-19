import torch

def get_scheduler(optimizer, config):
    if config["scheduler"]["type"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["scheduler"]["t_max"],
            eta_min=config["scheduler"]["min_lr"]
        )

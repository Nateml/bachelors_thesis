import torch


def get_scheduler(name: str):
    if name == "reduce_lr_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau

    elif name == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR

    elif name == "cosine_annealing_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts

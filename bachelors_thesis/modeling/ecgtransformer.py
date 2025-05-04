import torch.nn as nn
import torch
from omegaconf import DictConfig

class ECGTransformer(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        # ---------------------------------------
        # -------- INVIVIDUAL ENCODER -----------
        # ---------------------------------------

        

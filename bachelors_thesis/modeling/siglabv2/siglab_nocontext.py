

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from bachelors_thesis.registries.encoder_registry import get_encoder


class Classifier(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.input_size = cfg.classifier.in_size

        self.classifier = nn.Sequential(
            nn.Linear(self.input_size, cfg.classifier.layers[0].out_size),
            nn.ReLU(),
            *[
                layer
                for layer_cfg in cfg.classifier.layers[1:-1]
                for layer in (nn.Linear(layer_cfg.in_size, layer_cfg.out_size), nn.ReLU())
            ],
            # Final layer
            nn.Linear(cfg.classifier.layers[-1].in_size, cfg.num_classes)  # 6 classes for 6 precordial leads
        )

    def forward(self, z):
        """
        z: Tensor of shape (B, N, D),
            where B is the batch size and D is the feature dimension.
        """
        return self.classifier(z)

class SigLabNoContext(nn.Module):
    """Main SigLabDeepsets module"""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.feature_dim = cfg.feature_dim
        self.c = cfg.num_classes

        # --------1) Per-electrode encoder ---------------
        self.encoder = get_encoder(cfg.encoder.name)(cfg)

        # --------2) Classification head --------------
        self.classifier = Classifier(cfg)

    def forward(self, signals):
        """
        signals: Tensor of shape (B, N, T),
            where B is the batch size,
            N is the number of leads,
            and T is the length of the signal.
        Returns:
            lead_logits: tensor of shape (B, N, C)
            initial_logits: tensor of shape (B, N, C)
        """

        B, N, T = signals.shape

        # Process all electrode signals through the per-electrode local encoder
        features = self.encoder(signals)  # (B, N, D)
        _, _, D = features.shape

        logits = self.classifier(features)  # (B, N, C)

        #return final_logits, logits
        return logits

def loss_step(
        model,
        signals,
        lead_order,
        cfg
        ):

    if isinstance(signals, list):
        signals = torch.stack(signals)  # (B, N, T)

    # Forward pass
    logits = model(
        signals
    )

    B, N, C = logits.shape  # (B, N, C)

    targets = lead_order

    # Compute loss
    losses = [F.cross_entropy(logits[:,i], targets[:,i]) for i in range(N)]
    loss = torch.stack(losses).mean()

    # Compute accuracy
    preds = logits.argmax(-1)  # (B, N)
    correct = (preds == targets).float()
    acc = correct.mean()

    return loss, {
        "loss": loss.item(),
        "accuracy": acc.item()
    }
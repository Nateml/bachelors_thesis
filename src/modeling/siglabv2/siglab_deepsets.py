from omegaconf import DictConfig
from src.modeling.old.deepsets import DeepSetsContextEncoder
from src.registries.encoder_registry import get_encoder
import torch
import torch.nn as nn
import torch.nn.functional as F


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

class SigLabDeepsets(nn.Module):
    """Main SigLabDeepsets module"""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.feature_dim = cfg.feature_dim
        self.c = cfg.num_classes

        # --------1) Per-electrode encoder ---------------
        self.encoder = get_encoder(cfg.encoder.name)(cfg)

        # This used to be after the sab
        #self.init_head = Classifier(cfg)
        self.init_head = nn.Linear(cfg.feature_dim, cfg.num_classes)

        # --------2) Deepsets ---------
        self.deepsets = DeepSetsContextEncoder(
            input_size=cfg.feature_dim,
            output_size=cfg.feature_dim,
            hidden_dim=cfg.deepsets.hidden_dim,
            phi_layers=cfg.deepsets.phi_layers,
            rho_layers=cfg.deepsets.rho_layers
        )

        # --------4) Classification head --------------
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
        local_features = self.encoder(signals)  # (B, N, D)
        _, _, D = local_features.shape

        # Deepsets -> (B, D)
        context = self.deepsets(local_features, mask=None)

        # Concat the two representations
        features = torch.cat([local_features, context.unsqueeze(1).expand(-1, N, -1)], dim=-1)  # (B, N, 2*D)

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

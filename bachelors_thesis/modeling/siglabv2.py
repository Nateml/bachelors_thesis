"""
SigLab: A Deep Learning Model for ECG Signal Localization using labelled leads
"""

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from bachelors_thesis.modeling.encoders import CNNGRUEncoder
from bachelors_thesis.modeling.set_transformer.blocks import SetAttentionBlock


class LocalEncoder(nn.Module):
    """
    Shared 1D CNN encoder for each electrode signal.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()

        assert cfg.local_encoder.in_channels == 1, "Only 1 input channel is supported"

        # Construct layers from the configuration
        layers = []
        in_channels = cfg.local_encoder.in_channels
        for layer_cfg in cfg.local_encoder.layers:
            layers.append(nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=layer_cfg.out_channels,
                    kernel_size=layer_cfg.kernel_size,
                    stride=layer_cfg.stride,
                    padding=layer_cfg.padding
                ))
            layers.append(nn.ReLU())
            in_channels = layer_cfg.out_channels

        # Pooling Layer
        layers.append(nn.AdaptiveAvgPool1d(1))  # pooling layer
        # Flatten
        layers.append(nn.Flatten())
        # Fully Connected Layyer
        layers.append(nn.Linear(in_channels, cfg.feature_dim))
        layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        # x is of shape (B, num_electrodes, T):
        # B: batch size, num_electrodes: 6, T: length of the signal
        # Reshape to (B * num_electrodes, 1, T) for Conv1d
        # Each electrode signal is processed independently
        # and in parallel

        assert x.dim() == 3, "Input must be of shape (B, N, T)"

        B, N, T = x.shape
        # Flatten to (B * N, T) and add a channel dimension
        x = x.view(B * N, 1, T)

        out = self.encoder(x)  # (B * N, feature_dim)
        # Reshape back to (B, N, feature_dim)
        out = out.view(B, N, -1)

        return out

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
        z: Tensor of shape (B, D),
            where B is the batch size and D is the feature dimension.
        """
        return self.classifier(z)

class AttnBeliefUpdate(nn.Module):

    def __init__(self, cfg: DictConfig):
        super(AttnBeliefUpdate, self).__init__()

        # embed logits to the same dimension as the feature dimension
        self.logit_embed = nn.Linear(cfg.num_classes, cfg.feature_dim)

        # self-attention block
        self.sab = SetAttentionBlock(
            cfg.feature_dim,
            cfg.sab.num_heads,
            cfg.sab.ffn_expansion
        )

        # classification head
        self.head = nn.Linear(cfg.feature_dim, cfg.num_classes)

    def forward(self, logits, features):
        B, N, d = features.shape

        # 1) concatenate features and embedded logits
        logits = self.logit_embed(logits)  # (B, N, D)
        z = features + logits

        # 2) apply self-attention block to generate attention
        # enriched features
        h_new = self.sab(z)

        # 3) generate new logits by passing the enriched features
        # through the classification head
        logits_new = self.head(h_new)

        return h_new, logits_new


class SigLabV2(nn.Module):
    """Main SigLab module"""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.feature_dim = cfg.feature_dim
        self.c = cfg.num_classes

        # --------1) Per-electrode encoder ---------------
        self.encoder = CNNGRUEncoder(cfg)

        # --------2) Attention enriched features ---------
        self.sab = SetAttentionBlock(self.feature_dim, cfg.sab.num_heads, cfg.sab.ffn_expansion)

        self.init_head = Classifier(cfg)

        # --------3) L attention-refinement layers --------
        self.belief_layers = nn.ModuleList([
            AttnBeliefUpdate(cfg) for _ in range(cfg.num_belief_updates)
        ])

    def forward(self, signals):
        """
        signals: Tensor of shape (B, N, T),
            where B is the batch size,
            N is the number of leads,
            and T is the length of the signal.
        Returns:
            lead_logits: tensor of shape (B, N, N)
            where D is the feature dimension of the encoders
        """

        B, N, T = signals.shape

        # Process all electrode signals through the per-electrode local encoder
        features = self.encoder(signals)  # (B, N, D)
        _, _, D = features.shape

        # Set attention
        features = self.sab(features)  # (B, N, D)

        # Classification head for initial logits
        logits = self.init_head(features)  # (B, N, C)

        for layer in self.belief_layers:
            # Apply the belief update layer
            features, logits = layer(logits, features)

        return logits


def loss_step(
        model,
        batch,
        cfg
        ):

    if isinstance(batch, list):
        batch = torch.stack(batch)  # (B, N, T)

    # Forward pass
    logits = model(
        batch
    )

    B = batch.shape[0]  # batch size
    N = batch.shape[1]  # number of leads

    targets = torch.arange(N, device=logits[0].device).expand(B, -1)

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

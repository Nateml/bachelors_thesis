"""
SigLab: A Deep Learning Model for ECG Signal Localization using labelled leads
"""

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from bachelors_thesis.modeling.set_transformer.blocks import SetAttentionBlock, InducedSetAttentionBlock
from bachelors_thesis.registries.encoder_registry import get_encoder


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
        z: Tensor of shape (B, N, D),
            where B is the batch size and D is the feature dimension.
        """
        return self.classifier(z)

class AttnBeliefUpdateLogitsOnly(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # 1) Embed logits to a D‑dim space so attention has enough capacity
        self.logit_embed = nn.Sequential(
            nn.Linear(cfg.num_classes, cfg.feature_dim),
            nn.LayerNorm(cfg.feature_dim)
        )

        # 2) Multi‑head self‑attention operating *only* on the embedded logits
        self.sab = SetAttentionBlock(
            cfg.feature_dim,
            cfg.sab.num_heads,
            ffn_expansion = cfg.sab.ffn_expansion
        )

        # 3) Project back to class space and make a residual update
        self.delta_head = nn.Linear(cfg.feature_dim, cfg.num_classes)
        # self.delta_scale = cfg.belief.delta_scale  # e.g. 0.1
        self.delta_scale = 0.2

    def forward(self, logits, features):
        # ──1. embed logits────────────────────────────────────────
        z = self.logit_embed(logits)          # (B, N, D)

        # ──2. logits‑only attention──────────────────────────────
        z_ctx = self.sab(z)                   # (B, N, D)

        # ──3. residual ∆‑logits update───────────────────────────
        logits_delta = self.delta_head(z_ctx) # (B, N, C)
        logits_new   = logits + self.delta_scale * logits_delta

        # features are unchanged in this variant
        return features, logits_new


class AttnBeliefUpdate(nn.Module):

    def __init__(self, cfg: DictConfig):
        super(AttnBeliefUpdate, self).__init__()

        # embed logits to the same dimension as the feature dimension
        self.logit_embed = nn.Sequential(
            nn.Linear(cfg.num_classes, cfg.feature_dim),
            nn.LayerNorm(cfg.feature_dim)
        )

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
        h_new = self.sab(z)  # (B, N, D)

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
        self.encoder = get_encoder(cfg.encoder.name)(cfg)

        # This used to be after the sab
        #self.init_head = Classifier(cfg)
        self.init_head = nn.Linear(cfg.feature_dim, cfg.num_classes)

        # --------2) Attention enriched features ---------
        #self.sab = SetAttentionBlock(self.feature_dim, cfg.sab.num_heads, cfg.sab.ffn_expansion)
        #self.attention_blocks = nn.ModuleList([
        #    SetAttentionBlock(self.feature_dim, cfg.sab.num_heads, cfg.sab.ffn_expansion, mha_dropout=cfg.sab.mha_dropout)
        #    for _ in range(cfg.num_belief_updates)
        #])
        self.attention_blocks = nn.ModuleList([
            SetAttentionBlock(self.feature_dim, cfg.sab.num_heads, ffn_expansion=cfg.sab.ffn_expansion, mha_dropout=cfg.sab.mha_dropout)
            for _ in range(cfg.num_belief_updates)
        ])

        # --------3) L attention-refinement layers --------
        #self.belief_layers = nn.ModuleList([
            #AttnBeliefUpdate(cfg) for _ in range(cfg.num_belief_updates)
        #])

        # --------4) Classification head --------------
        self.classifier = Classifier(cfg)

        #self.alpha = nn.Parameter(torch.tensor(0.5))
        #self.alpha = 0.5

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

        # Set attention
        #features = self.sab(features)  # (B, N, D)

        # Classification head for initial logits
        #logits = self.init_head(features)  # (B, N, C)

        #new_logits = logits.clone()
        #for i, layer in enumerate(self.belief_layers):
            # Apply the belief update layer
            #features, logits = layer(logits, features)
            #features, new_logits = layer(new_logits, features)  # (B, N, C)

        for i, layer in enumerate(self.attention_blocks):
            # Apply the attention block to the features
            features = layer(features)

        logits = self.classifier(features)  # (B, N, C)

        #return final_logits, logits
        return logits

def focal_loss(
        logits,
        targets,
        alpha=0.25,
        gamma=2.0,
        reduction='mean'
    ):
    """
    Focal loss for multi-class classification.
    Args:
        logits: Tensor of shape (B, N, C), where B is the batch size,
                N is the number of leads, and C is the number of classes.
        targets: Tensor of shape (B, N), where B is the batch size and
                N is the number of leads.
        alpha: Weighting factor for the class imbalance.
        gamma: Focusing parameter to adjust the rate at which easy examples are down-weighted.
        reduction: Reduction method ('mean', 'sum', or 'none').
    Returns:
        loss: Computed focal loss.
    """
    BCE_losses = [F.cross_entropy(logits[:, i], targets[:, i], reduction='none') for i in range(logits.shape[1])]
    BCE_loss = torch.stack(BCE_losses, dim=1)  # (B, N)
    BCE_loss = BCE_loss.mean(dim=1)  # (B,)
    pt = torch.exp(-BCE_loss)  # Probability of true class
    loss = alpha * (1 - pt) ** gamma * BCE_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss.view(logits.size(0), -1).mean(dim=1)

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
    if cfg.loss.name == "cross-entropy":
        # Main loss
        # I am only going to use the precordial leads for the loss because
        # for this experiment I only care about precordial accuracy
        losses = [F.cross_entropy(logits[:,i], targets[:,i]) for i in range(6)]
        loss = torch.stack(losses).mean()

        # Loss for intermediate predictions
        #init_loss = F.cross_entropy(inter_logits, targets, reduction='mean')
        #init_losses = [F.cross_entropy(inter_logits[:,i], targets[:,i]) for i in range(N)]
        #init_loss = torch.stack(init_losses).mean()
        #loss = loss + 0.2 * init_loss

    elif cfg.loss.name == "focal-loss":
        loss = focal_loss(logits,
                        targets,
                        alpha=cfg.loss.alpha,
                        gamma=cfg.loss.gamma,
                        reduction=cfg.loss.reduction
                        )
        #init_loss = focal_loss(inter_logits,
        #                targets,
        #                alpha=cfg.loss.alpha,
        #                gamma=cfg.loss.gamma,
        #                reduction=cfg.loss.reduction
        #                )
        #loss = loss + 0.2 * init_loss

    else:
        raise ValueError(f"Unknown loss function: {cfg.loss.name}")

    # Compute accuracy
    preds = logits.argmax(-1)  # (B, N)
    correct = (preds == targets).float()
    acc = correct.mean()

    # Compute precordial accuracy
    precordial_leads = [0, 1, 2, 3, 4, 5]  # Assuming precordial leads are the first 6 leads
    precordial_logits = logits[:, precordial_leads]
    precordial_preds = precordial_logits.argmax(-1)  # (B, N)
    precordial_correct = (precordial_preds == targets[:, precordial_leads]).float()
    precordial_acc = precordial_correct.mean()

    return loss, {
        "loss": loss.item(),
        "accuracy": acc.item(),
        "precordial_accuracy": precordial_acc.item()
    }

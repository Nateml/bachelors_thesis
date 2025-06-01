from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modeling.old.deepsets import DeepSetsContextEncoder


class GruModel(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.feature_dim = cfg.feature_dim

        self.gru = nn.GRU(
            1,
            self.feature_dim,
            cfg.gru.num_layers,
            batch_first=True,
            bidirectional=cfg.gru.bidirectional,
            dropout=cfg.gru.dropout
            )

        if cfg.context_encoder.enabled:
            self.context_encoder = DeepSetsContextEncoder(
                input_size=self.feature_dim * (2 if cfg.gru.bidirectional else 1),
                output_size=self.feature_dim,
                hidden_dim=cfg.context_encoder.hidden_dim,
                phi_layers=cfg.context_encoder.phi.num_layers,
                rho_layers=cfg.context_encoder.rho.num_layers
            )

        self.classifier = Classifier(cfg)

    def forward(self, x):
        """
        Forward pass through the GRU model.

        Args
        ----
        x: torch.Tensor, shape (batch_size, seq_len, input_size)
            The input data.
        """

        B, N, T = x.shape

        # Transform leads to be part of the batch
        x = x.view(B * N, T, 1)

        # Forward pass through GRU
        out, _ = self.gru(x)  # out: (B * N, T, feature_dim) or (B * N, T, 2 * feature_dim) if bidirectional
        out = out[:, -1, :]  # Take the last output of the GRU
        # out: (B * N, feature_dim (* 2 if bidirectional))

        # Restore the lead dimension
        out = out.view(B, N, -1)  # (B, N, gru_out_dim)
        _, _, D = out.shape

        if self.context_encoder:
            # Create context masks
            features_expanded = out.repeat_interleave(N, dim=0)  # (B * N, N, feature_dim)
            target_idx = torch.arange(N, device=out.device).repeat(B)  # (B * N,)
            mask = torch.ones((B*N, N), dtype=torch.bool, device=out.device)  # (B * N, N)
            mask[torch.arange(B*N), target_idx] = False

            # Encode the context signals
            context = self.context_encoder(features_expanded, mask)  # (B, feature_dim)

            # Combine the two representations
            target_indices_expanded = target_idx.view(-1, 1, 1).expand(-1, 1, self.feature_dim)  # (B * N, 1, feature_dim)
            target_features = torch.gather(features_expanded, dim=1, index=target_indices_expanded).squeeze(1)  # (B * N, feature_dim)
            z = torch.cat([target_features, context], dim=-1)  # (B * N, 2 * feature_dim)

        else:
            # Convert to (B * N, feature_dim)
            z = out.view(B * N, D)

        # Pass through the classifier
        lead_logits = self.classifier(z)  # (B * N, num_classes)

        # Reshape back to (B, N, num_classes)
        lead_logits = lead_logits.view(B, N, -1)  # (B, N, num_classes)

        return lead_logits, z

class Classifier(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.input_size = cfg.feature_dim * (2 if cfg.context_encoder.enabled else 1)

        self.classifier = nn.Sequential(
            nn.Linear(self.input_size, cfg.classifier.layers[0].out_size),
            nn.ReLU(),
            *[
                layer
                for layer_cfg in cfg.classifier.layers[1:-1]
                for layer in (nn.Linear(layer_cfg.in_size, layer_cfg.out_size), nn.ReLU())
            ],
            # Final layer
            nn.Linear(cfg.classifier.layers[-1].in_size, cfg.classifier.num_classes)  # 6 classes for 6 precordial leads
        )

    def forward(self, z):
        """
        z: Tensor of shape (B, 2*D),
            where B is the batch size and D is the feature dimension.
        """
        return self.classifier(z)

def loss_step(
        model,
        batch,
        cfg
        ):
    
    if isinstance(batch, list):
        batch = torch.stack(batch)

    logits, z = model(batch)

    B = batch.shape[0]
    N = batch.shape[1]

    targets = torch.arange(N, device=batch.device).expand(B, -1)

    # Compute cross-entropy losses and average
    losses = [F.cross_entropy(logits[:,i], targets[:,i]) for i in range(N)]
    loss = torch.stack(losses).mean()

    # Compute accuracies
    preds = logits.argmax(dim=2)  # (B, N)
    correct = (preds == targets).float()
    acc = correct.mean().item()

    return loss, {
        "loss": loss.item(),
        "accuracy": acc
    }
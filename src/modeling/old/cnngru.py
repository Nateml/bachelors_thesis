from omegaconf import DictConfig
from src.modeling.old.deepsets import DeepSetsContextEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,  # To keep the same length
        )

        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        """
        Forward pass through the CNN block.

        Args
        ----
        x: torch.Tensor, shape (batch_size, in_channels, seq_len)
            The input data.
        """

        x = self.conv(x)  # Convolutional layer
        x = self.batchnorm(x)  # Batch normalization, speeds up training
        x = self.relu(x)  # Non-linearity
        x = self.maxpool(x)  # Downsample the output

        return x

class CNNGru(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.feature_dim = cfg.feature_dim

        self.cnn = nn.Sequential(
            *[CNNBlock(block.in_channels, block.out_channels, block.kernel_size) for block in cfg.cnn.blocks],
            nn.Dropout1d(cfg.cnn.dropout)
        )

        cnn_out_channels = cfg.cnn.blocks[-1].out_channels

        self.gru = nn.GRU(
            cnn_out_channels,
            self.feature_dim,
            cfg.gru.num_layers,
            batch_first=True,
            bidirectional=cfg.gru.bidirectional,
            dropout=cfg.gru.dropout
            )


        self.context_encoder = None
        if cfg.context_encoder.enabled:
            self.context_encoder = DeepSetsContextEncoder(
                input_size=self.feature_dim,
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
        x = x.view(B * N, 1, T)

        # CNN feature extraction
        x = self.cnn(x)  # (B*N, C, T')


        # Transform leads to be part of the batch
        x = x.permute(0, 2, 1)  # (B * N, T', C)

        # Forward pass through GRU
        out, _ = self.gru(x)  # out: (B * N, T', D)
        out = out[:, -1, :]  # Take the last output of the GRU -> (B * N, D)
        # out: (B * N, feature_dim)

        # Restore the lead dimension
        out = out.view(B, N, -1)  # (B, N, feature_dim)
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
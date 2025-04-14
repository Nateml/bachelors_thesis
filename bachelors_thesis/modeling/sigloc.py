
"""
SigLoc12: A Deep Learning Model for ECG Signal Localization
"""

import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        assert x.size(1) == 6, f"Input must have 6 electrodes. Input shape: {x.shape}"

        B, N, T = x.shape
        # Flatten to (B * N, T) and add a channel dimension
        x = x.view(B * N, 1, T)

        out = self.encoder(x)  # (B * N, feature_dim)
        # Reshape back to (B, N, feature_dim)
        out = out.view(B, N, -1)

        return out


class DeepSetsContextEncoder(nn.Module):
    """Encoder for context information using DeepSets architecture"""

    def __init__(self, cfg: DictConfig):
        super().__init__()

        hidden_dim = cfg.context_encoder.hidden_dim

        self.phi = nn.Sequential(
            nn.Linear(cfg.feature_dim, hidden_dim),
            nn.ReLU(),
            *[
                layer
                for _ in range(cfg.context_encoder.phi.num_layers -1)
                for layer in (nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            ]
        )

        self.rho = nn.Sequential(
            *[
                layer
                for _ in range(cfg.context_encoder.rho.num_layers - 1)
                for layer in (nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            ],
            nn.Linear(hidden_dim, cfg.feature_dim),
            nn.ReLU()
        )

    def forward(self, features, mask):
        """
        features: Tensor of shape (B, N, D),
            where B is the batch size,
            N is the number of electrodes per sample (6),
            and D is the feature dimension.
        mask: Tensor of shape (B, N),
            where N is the number of electrodes (6).
            Mask is a boolean tensor indicating which electrodes to include
            in the context encoding.

        Returns:
            context: (B, D)
                A context vector for each sample in the batch
        """
        B, N, D = features.shape

        assert mask.shape == (B, N), "Mask shape must match features shape"

        # Pass samples through the phi function
        # Flatten (B, N) to (B * N) and apply phi
        phi_in = features.view(B * N, D)  # (B * N, D)
        phi_out = self.phi(phi_in)  # (B * N, hidden_dim)

        # Reshape phi_out back to (B, N, hidden_dim)
        hidden_dim = phi_out.shape[-1]
        phi_out = phi_out.view(B, N, hidden_dim)  # (B, N, hidden_dim)

        # Mask phi_out to exclude unmasked electrodes
        # We have to expand the mask to (B, N, 1) so it can
        # broadcast over the hidden dimension
        phi_out_masked = phi_out * mask.unsqueeze(-1)  # (B, N, hidden_dim)

        # Any vectors that are masked out will now be zeroed out and therefore
        # won't contribute to the pooling operation

        # Sum pooling
        pooled = phi_out_masked.sum(dim=1)  # (B, hidden_dim)

        # Apply rho and return, we now have a context vector for each sample
        # in the batch
        return self.rho(pooled)


class Classifier(nn.Module):
    """Decodes the latent representation into a classification of the
        input signal (which lead is it from)"""

    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(cfg.feature_dim * 2, cfg.classifier.layers[0].out_size),
            nn.ReLU(),
            *[
                layer
                for layer_cfg in cfg.classifier.layers[1:-1]
                for layer in (nn.Linear(layer_cfg.in_size, layer_cfg.out_size), nn.ReLU())
            ],
            # Final layer
            nn.Linear(cfg.classifier.layers[-1].in_size, 6)  # 6 classes for 6 precordial leads
        )

    def forward(self, z):
        """
        z: Tensor of shape (B, 2*D),
            where B is the batch size and D is the feature dimension.
        """
        return self.classifier(z)


class SigLoc12(nn.Module):
    """Main SigLoc12 module"""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.feature_dim = cfg.feature_dim

        self.local_encoder = LocalEncoder(cfg)

        self.context_encoder = DeepSetsContextEncoder(cfg)

        self.classifier = Classifier(cfg)

    def forward(self, signals):
        """
        signals: Tensor of shape (B, N, T),
            where B is the batch size,
            N is the number of leads (6),
            and T is the length of the signal.
        Returns:
            embedding: tensor of shape (B, 2*D)
            lead_logits: tensor of shape (B, N, N)
            where D is the feature dimension of the encoders
        """

        B, N, T = signals.shape

        # Process all electrode signals through the per-electrode local encoder
        all_features = self.local_encoder(signals)  # (B, N, D)
        _, _, D = all_features.shape

        # Repeat x N times along batch dimension for each target index
        # Shape: (B*N, N, T)
        features_expanded = all_features.repeat_interleave(N, dim=0)

        # Create target indices 0...5 for each sample in the batch
        # (B*N, N)
        target_indices = torch.arange(N, device=signals.device).repeat(B)

        # Create context masks
        # (B*N, N)
        context_mask = torch.ones((B*N, N), dtype=torch.bool, device=signals.device)
        context_mask[torch.arange(B*N), target_indices] = False

        # Encode the context signals
        context = self.context_encoder(features_expanded, context_mask)  # (B, D)

        # Combine the two encodings
        target_indices_expanded = target_indices.view(-1, 1, 1).expand(-1, 1, D)
        target_features = torch.gather(features_expanded, dim=1, index=target_indices_expanded).squeeze(1)  # (B*N, D)
        z = torch.cat([target_features, context], dim=-1)  # (B, 2*D)

        # Classify the target signal using the combined encoding
        lead_logits = self.classifier(z)  # (B, N)

        # Reshape back to (B, N, N)
        lead_logits = lead_logits.view(B, N, N)  # (B, N, N)

        return z, lead_logits  # (B, 2*D), (B, N, N)

    def predict(self, input_, batch_size=64):
        """
        Runs prediction on the input signals (in evaluation mode).
        x: Tensor of shape (S, N, T),
            S is the number of samples,
            N is the number of leads (6),
            and T is the length of the signal.
        
        Returns:
            lead_logits: Tensor of shape (S, 6, 6),
                where 6 is the number of classes (leads).
                So the output is a 6x6 matrix of logits for each lead,
                for each sample.
        """

        # Set the model to evaluation mode
        self.eval()

        S, N, T = input_.shape
        assert N == 6, f"Input must have 6 electrodes. Input shape: {input_.shape}"

        # Create a dataset from the input
        class InferenceDataset(torch.utils.data.Dataset):
            def __init__(self, x):
                self.x = x

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                return self.x[idx]

        dataloader = torch.utils.data.DataLoader(
            InferenceDataset(input_),
            batch_size=64,
            shuffle=False
        )

        preds = np.zeros((S, N, N), dtype=np.float32)
        tracker = 0

        for x in dataloader:
            B = x.shape[0]  # batch size

            # Repeat x 6 times along batch dimension for each target index
            # Shape: (B*6, 6, T)
            x_expanded = x.repeat_interleave(N, dim=0)

            # Create target indices 0...5 for each sample in the batch
            # (B*6, 6)
            target_indices = torch.arange(N, device=x.device).repeat(B)

            # Create context masks
            context_mask = torch.ones((B*N, N), dtype=torch.bool, device=x.device)
            context_mask[torch.arange(B*N), target_indices] = False

            with torch.no_grad():
                # Forward pass through the model
                _, lead_logits = self(x_expanded, target_indices, context_mask)

            # Add the predictions to the output array
            preds[tracker:tracker + B, :, :] = lead_logits.view(B, N, N).cpu().numpy()
            tracker += B

        return torch.from_numpy(preds).float()


def loss_step(
        model,
        batch,
        cfg
        ):

    if isinstance(batch, list):
        batch = torch.stack(batch)  # (B, 6, T)

    # Forward pass: anchor + positive
    # logits is a tensor of shape (B, N, N) where N is the number of leads
    z, logits = model(
        batch
    )

    B = batch.shape[0]  # batch size
    N = batch.shape[1]  # number of leads

    targets = torch.arange(N, device=logits[0].device).expand(B, -1)

    # Compute cross-entropy losses and average
    losses = [F.cross_entropy(logits[:,i], targets[:,i]) for i in range(N)]
    loss = torch.stack(losses).mean()

    # Compute per-class accuracy and average
    accuracies = [(logits[i].argmax(dim=1) == targets[i]).float().mean() for i in range(6)]
    acc = torch.stack(accuracies).mean()

    return loss, {
        "loss": loss.item(),
        "accuracy": acc.item()
    }

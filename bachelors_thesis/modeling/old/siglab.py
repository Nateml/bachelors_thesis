"""
SigLab: A Deep Learning Model for ECG Signal Localization using labelled leads
"""

import numpy as np
from omegaconf import DictConfig
from scipy.optimize import linear_sum_assignment
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

    def forward(self, features):
        """
        features: Tensor of shape (B, N, D),
            where B is the batch size,
            N is the number of electrodes per sample,
            and D is the feature dimension.
        Returns:
            context: (B, D)
                A context vector for each sample in the batch
        """
        B, N, D = features.shape

        # Pass samples through the phi function
        # Flatten (B, N) to (B * N) and apply phi
        phi_in = features.view(B * N, D)  # (B * N, D)
        phi_out = self.phi(phi_in)  # (B * N, hidden_dim)

        # Reshape phi_out back to (B, N, hidden_dim)
        hidden_dim = phi_out.shape[-1]
        phi_out = phi_out.view(B, N, hidden_dim)  # (B, N, hidden_dim)

        # Sum pooling
        pooled = phi_out.sum(dim=1)  # (B, hidden_dim)

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
            nn.Linear(cfg.classifier.layers[-1].in_size, cfg.num_leads)
        )

    def forward(self, z):
        """
        z: Tensor of shape (B, N, 2*D),
            where B is the batch size,
            N is the number of leads,
            and D is the feature dimension.
        """

        # Flatten z to (B*N, 2*D)
        B, N, D2 = z.shape
        z = z.view(B*N, D2)

        # Classify, output is now (B*N, N)
        out = self.classifier(z)

        # Reshape back to (B, N, N)
        return out.view(B, N, -1)


class SigLab(nn.Module):
    """Main SigLab module"""

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
            N is the number of leads,
            and T is the length of the signal.
        Returns:
            lead_logits: tensor of shape (B, N, N)
            where D is the feature dimension of the encoders
        """

        B, N, T = signals.shape

        # Process all electrode signals through the per-electrode local encoder
        all_features = self.local_encoder(signals)  # (B, N, D)
        _, _, D = all_features.shape

        # Encode the context signals
        context = self.context_encoder(all_features)  # (B, D)
        context = context.unsqueeze(1).expand(-1, N, -1)  # (B, N, D)

        z = torch.cat([all_features, context], dim=-1)  # (B, N, 2*D)

        # Classify the target signal using the combined encoding
        lead_logits = self.classifier(z)  # (B, N, N)

        return lead_logits  # (B, N, N)

    def predict_leads(self, input_signals: torch.Tensor) -> np.ndarray:
        """
        Predict lead assignments using Hungarian matching.
        
        Args:
            model: Trained model (should be in eval mode)
            input_signals: Tensor of shape (B, 6, T)

        Returns:
            np.ndarray of shape (B, 6), each row is the predicted permutation of lead labels
        """
        self.eval()
        input_signals = input_signals.to(next(self.parameters()).device)

        with torch.no_grad():
            logits = self(input_signals)  # logits: (B, 6, 6)

        logits = logits.detach().cpu().numpy()
        B, N, _ = logits.shape
        predictions = np.zeros((B, N), dtype=np.int64)

        for b in range(B):
            cost = -logits[b]  # Maximize logits = minimize negative logits
            row_ind, col_ind = linear_sum_assignment(cost)
            assignment = np.zeros(N, dtype=int)
            assignment[row_ind] = col_ind
            predictions[b] = assignment

        return predictions  # shape: (B, 6)

    def predict(self, logits: torch.Tensor) -> np.ndarray:
        """
        Predict the lead labels from the logits using hungarian algorithm.
        logits: Tensor of shape (B, N, N)
            where B is the batch size,
            N is the number of leads.
        """

        B, N, _ = logits.shape
        predictions = np.zeros((B, N), dtype=np.int64)

        for b in range(B):
            cost = -logits[b].detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost)

            # build the assignment array
            pred = np.zeros(N, dtype=int)
            pred[row_ind] = col_ind
            predictions[b] = pred

        return predictions


def permutation_loss(logits, targets):
    """
    Uses Hungarian algorithm to compute the optimal assignment of the predictions.
    Is quite slow because it has to run per sample in the batch.
    logits: (B, N, N)
    targets: (B, N) - targets[i][j] = class index (0-(N-1)) for signal j in sample i

    Returns:
        - loss: average cross-entropy loss over the batch
        - accuracy: average accuracy over the batch
    """
    B, N, _ = logits.shape
    total_loss = 0.0
    total_acc = 0.0

    for b in range(B):
        # Step 1: Compute cost matrix
        cost = -1 * logits[b].detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)

        # Step 2: Compute the optimal assignment
        matched_logits = logits[b, row_ind]  # (N, N)
        matched_targets = targets[b, col_ind]  # (N,)

        # Step 3: Compute the loss
        print(matched_logits)
        print(matched_targets)
        total_loss += F.cross_entropy(matched_logits, matched_targets)

        # Step 4: Compute the accuracy
        predicted_labels = matched_logits.argmax(dim=1)
        correct = (predicted_labels == matched_targets).sum().item()
        acc = correct / N
        total_acc += acc

    return total_loss / B, total_acc / B

def loss_step(
        model,
        batch,
        cfg
        ):

    if isinstance(batch, list):
        batch = torch.stack(batch)  # (B, 6, T)

    # Forward pass: anchor + positive
    # logits is a tensor of shape (B, N, N) where N is the number of leads
    logits = model(
        batch
    )

    B = batch.shape[0]  # batch size
    N = batch.shape[1]  # number of leads

    targets = torch.arange(N, device=logits[0].device).expand(B, -1)

    # Compute permutation loss 
    loss, acc = permutation_loss(logits, targets)

    return loss, {
        "loss": loss.item(),
        "accuracy": acc
    }

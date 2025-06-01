"""
AURA12: A self-supervised learning framework for 12-lead ECG signals
Contains the AURA12 model, as well as the model-specific dataset class.
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


class AURA12(nn.Module):
    """Main AURA module"""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.feature_dim = cfg.feature_dim

        self.local_encoder = LocalEncoder(cfg)

        self.context_encoder = DeepSetsContextEncoder(cfg)

        self.classifier = Classifier(cfg)

    def forward(self, signals, target_idx, context_mask):
        """
        signals: Tensor of shape (B, N, T),
            where B is the batch size,
            N is the number of leads (6),
            and T is the length of the signal.
        target_idx: batched index of the signal in signals to embed and
            classify, shape (B,)
        context_mask: boolean tensor of shape (B, N) indicating which
            electrodes to include in the context encoding.
        Returns:
            embedding: tensor of shape (B, 2*D)
            lead_logits: tensor of shape (B, 6)
            where D is the feature dimension of the encoders
        """

        # Process all electrode signals through the per-electrode local encoder
        all_features = self.local_encoder(signals)  # (B, 6, D)

        # Create a mask to exclude the target electrode from
        # the context encoder
        # mask = torch.ones(6, dtype=torch.bool)  # all electrodes are masked
        # mask[target_idx] = False  # unmask the target electrode

        # Extract the target feature vector
        B, N, D = all_features.shape
        batch_indices = torch.arange(B, device=all_features.device)  # (B,)
        target_feature = all_features[batch_indices, target_idx]  # (B, D)

        # Encode the context signals
        context = self.context_encoder(all_features, context_mask)  # (B, D)

        # Combine the two encodings
        z = torch.cat([target_feature, context], dim=-1)  # (B, 2*D)

        # Classify the target signal using the combined encoding
        lead_logits = self.classifier(z)  # (B, 6)

        return z, lead_logits  # (B, 2*D), (B, 6)

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


def nt_xent_loss(z_i, z_j, temperature=0.1):
    """
    Computes NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
    z_i, z_j: positive pairs
    """

    assert z_i.shape == z_j.shape, "z_i and z_j must have the same shape"

    batch_size = z_i.shape[0]

    # Normalize embeddings to unit vectors
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    # Concatenate embeddings: 2B x D
    z = torch.cat([z_i, z_j], dim=0)

    # Compute cosine similarity matrix: 2B x 2B
    sim = torch.matmul(z, z.T)
    sim = sim / temperature

    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim.masked_fill_(mask, float('-inf'))

    # Positive pair indices: i <-> i + B
    positive_indices = torch.arange(batch_size, device=z.device)

    # Compute log-softmax over similarities
    logits = F.log_softmax(sim, dim=1)

    # Extract positive log-probs (z_i vs z_j)
    loss_i = -logits[positive_indices, positive_indices + batch_size]
    loss_j = -logits[positive_indices + batch_size, positive_indices]

    # Average loss over both halves of the batch
    loss = (loss_i + loss_j).mean()
    return loss


def loss_step(
        model,
        batch,
        cfg
        ):

    anchor_signals, anchor_idx, positive_signals, positive_idx, context_mask = batch
    
    # Forward pass: anchor + positive
    z_anchors, pred_anchors = model(
        anchor_signals, anchor_idx, context_mask 
    )
    z_positives, _ = model(
        positive_signals, positive_idx, context_mask
    )

    # Constrastive loss
    contrastive_loss = torch.tensor(0.0, device=z_anchors.device)
    loss_lambda = cfg.loss._lambda
    if (loss_lambda < 1):
        contrastive_loss = nt_xent_loss(
            z_anchors, z_positives, 
        )

    # Classification loss (cross-entropy)
    clf_loss = F.cross_entropy(pred_anchors, anchor_idx)

    # Total loss
    # (1 - lambda) * contrastive_loss + lambda * clf_loss
    total_loss = (1 - loss_lambda) * contrastive_loss + loss_lambda * clf_loss

    # Check classification accuracy on training set
    acc = (pred_anchors.argmax(dim=1) == anchor_idx).float().mean()

    return total_loss, {
        "loss": total_loss.item(),
        "contrastive_loss": contrastive_loss.item(),
        "classification_loss": clf_loss.item(),
        "accuracy": acc.item()
    }

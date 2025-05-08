
"""
SigLoc12: A Deep Learning Model for ECG Signal Localization
"""

import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN1D(nn.Module):
    """
    Simple 1D CNN for encoding ECG signals.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()

        # Construct layer
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

class InceptionBlock(nn.Module):
    """
    Performs multiple convolutions in parallel with different kernel sizes.
    Each branch has its own kernel size and output channels.
    The outputs of all branches are concatenated along the channel dimension.
    """
    def __init__(self, in_channels, bottleneck_channels, cfg: DictConfig):
        super().__init__()

        # Create the branches

        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            # kernel size 1
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=cfg.inception_block.branch1.out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            # Instance norm
            nn.BatchNorm1d(cfg.inception_block.branch1.out_channels),
            nn.ReLU()
        )

        # Branch 2: 5x1 convolution
        self.branch2 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm1d(bottleneck_channels),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=bottleneck_channels,
                out_channels=cfg.inception_block.branch2.out_channels,
                kernel_size=OmegaConf.select(cfg, "inception_block.branch2.kernel_size", default=5),
                stride=1,
                padding=OmegaConf.select(cfg, "inception_block.branch2.padding", default=2)
            ),
            nn.BatchNorm1d(cfg.inception_block.branch2.out_channels),
            nn.ReLU()
        )

        # Branch 3: 11x1 convolution
        self.branch3 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm1d(bottleneck_channels),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=bottleneck_channels,
                out_channels=cfg.inception_block.branch3.out_channels,
                kernel_size=OmegaConf.select(cfg, "inception_block.branch3.kernel_size", default=11),
                stride=1,
                padding=OmegaConf.select(cfg, "inception_block.branch3.padding", default=5)
            ),
            nn.BatchNorm1d(cfg.inception_block.branch3.out_channels),
            nn.ReLU(),
        )

        # Branch 4: Max pooling
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=OmegaConf.select(cfg, "inception_block.branch4.maxpool_size", default=3),
                         stride=1,
                         padding=1),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=cfg.inception_block.branch4.out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm1d(cfg.inception_block.branch4.out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # Parallel branches
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        # Concatenate along the channel dimension
        out = torch.cat([b1, b2, b3, b4], dim=1)

        return out


class InceptionEncoder(nn.Module):

    def __init__(self, cfg: DictConfig):
        super().__init__()

        layers = []
        current_channels = 1
        for i in range(cfg.inception_encoder.num_blocks):
            layers.append(InceptionBlock(
                in_channels=current_channels,
                bottleneck_channels=cfg.inception_encoder.bottleneck_channels,
                cfg=cfg
                ))
            
            current_channels = (
                cfg.inception_block.branch1.out_channels +
                cfg.inception_block.branch2.out_channels +
                cfg.inception_block.branch3.out_channels +
                cfg.inception_block.branch4.out_channels +
                cfg.inception_block.branch5.out_channels
            )

            # Small dropout layer
            layers.append(nn.Dropout(cfg.inception_encoder.dropout))

        # Pooling Layer
        layers.append(nn.AdaptiveAvgPool1d(1))  # pooling layer
        # Flatten
        layers.append(nn.Flatten())
        # Fully Connected Layyer
        layers.append(nn.Linear(current_channels, cfg.feature_dim))
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

        out = self.encoder(x)

        # Reshape back to (B, N, feature_dim)
        out = out.view(B, N, -1)

        return out

class InceptionEncoderWithRes(nn.Module):

    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg

        pre_skip_layers = []
        post_skip_layers = []
        current_channels = 1
        for i in range(cfg.inception_encoder.num_blocks):
            pre_skip_layers.append(InceptionBlock(
                in_channels=current_channels,
                bottleneck_channels=cfg.inception_encoder.bottleneck_channels,
                cfg=cfg
                ))
            
            current_channels = (
                cfg.inception_block.branch1.out_channels +
                cfg.inception_block.branch2.out_channels +
                cfg.inception_block.branch3.out_channels +
                cfg.inception_block.branch4.out_channels +
                cfg.inception_block.branch5.out_channels
            )

            # Small dropout layer
            if i < cfg.inception_encoder.num_blocks - 1:  # don't add dropout after the last block
                pre_skip_layers.append(nn.Dropout(cfg.inception_encoder.dropout))

        # Residual projection layer
        self.residual_proj = nn.Sequential(
            nn.Conv1d(
            in_channels=1,
            out_channels=current_channels,
            kernel_size=1,
            stride=1,
            padding=0
            ),
            nn.BatchNorm1d(current_channels),
            nn.ReLU()
        )
        
        # Pooling Layer
        post_skip_layers.append(nn.AdaptiveAvgPool1d(1))  # pooling layer
        # Flatten
        post_skip_layers.append(nn.Flatten())
        # Fully Connected Layyer
        post_skip_layers.append(nn.Linear(current_channels, cfg.feature_dim))
        post_skip_layers.append(nn.ReLU())

        # self.encoder = nn.Sequential(*layers)
        self.pre_skip = nn.Sequential(*pre_skip_layers)
        self.post_skip = nn.Sequential(*post_skip_layers)


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

        # Pass through the Inception blocks
        out = self.pre_skip(x)

        # Add skip connection
        # OmegaConf.select() returns None if the key is not found,
        # keeping the model backward compatible with older configs
        residual = self.residual_proj(x)  # (B * N, current_channels, T)

        out = out + residual  # (B * N, current_channels, T)

        # Pass through the post-skip layers
        out = self.post_skip(out)  # (B * N, feature_dim)

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
        self.cfg = cfg

        self.local_encoder = SimpleCNN1D(cfg)

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

        if self.cfg.context_encoder.enabled:
            # Repeat x N times along batch dimension for each target index
            # Shape: (B*N, N, T)
            features_expanded = all_features.repeat_interleave(N, dim=0)

            # Create target indices 0...5 for each sample in the batch
            # (B*N,)
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
        else:
            # Convert all_features to (B*N, D) for the classifier
            z = all_features.view(B*N, D)  # (B*N, D)

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

class SigLocNolan(nn.Module):
    """
    SigLoc12 with InceptionEncoder as the local encoder.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.feature_dim = cfg.feature_dim
        self.cfg = cfg

        if OmegaConf.select(self.cfg, "inception_encoder.skip_connection"):
            self.local_encoder = InceptionEncoderWithRes(cfg)
        else:
            self.local_encoder = InceptionEncoder(cfg)
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

        if self.cfg.context_encoder.enabled:
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
        else:
            # Convert all_features to (B*N, D) for the classifier
            z = all_features.view(B*N, D)

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
    if cfg.loss.name == "cross-entropy":
        losses = [F.cross_entropy(logits[:,i], targets[:,i]) for i in range(N)]
        loss = torch.stack(losses).mean()
    elif cfg.loss.name == "focal-loss":
        loss = focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction='mean')
    else:
        raise ValueError(f"Unknown loss function: {cfg.loss.name}")

    # Compute accuracies
    preds = logits.argmax(dim=2)  # (B, N)
    correct = (preds == targets).float()
    acc = correct.mean().item()

    return loss, {
        "loss": loss.item(),
        "accuracy": acc
    }

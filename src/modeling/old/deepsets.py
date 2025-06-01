import torch
import torch.nn as nn


class DeepSetsContextEncoder(nn.Module):
    """Encoder for context information using DeepSets architecture"""

    def __init__(self, input_size: int, output_size: int, hidden_dim: int, phi_layers: int, rho_layers: int):
        super().__init__()

        self.phi = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            *[
                layer
                for _ in range(phi_layers -1)
                for layer in (nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            ]
        )

        self.rho = nn.Sequential(
            *[
                layer
                for _ in range(rho_layers - 1)
                for layer in (nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            ],
            nn.Linear(hidden_dim, output_size),
            nn.ReLU()
        )

    def forward(self, features, mask):
        """
        features: Tensor of shape (B, N, D),
            where B is the batch size,
            N is the number of electrodes per sample,
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

        if mask is None:
            # If no mask is provided, create a mask of ones
            mask = torch.ones(B, N, dtype=torch.bool, device=features.device)

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
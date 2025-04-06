import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim import Adam


class AURA12Dataset(Dataset):
    def __init__(self, ecg_tensor, augment=True):
        """
        ecg_tensor: A tensor of shape (N, 6, T) where N is the number
                  of samples, 6 is the number of precordial leads,
                    and T is the length of the signal.
        """
        super().__init__()
        self.ecgs = ecg_tensor  # (N, 6, T)
        self.N = ecg_tensor.size(0)  # Number of ECGs
        self.T = ecg_tensor.size(2)  # Length of the signal (T)
        self.augment = augment 

        # Create the anchors
        self.anchor_pairs = []
        # Enumarate all leads for each ECG
        for ecg_idx in range(self.N):
            for lead_idx in range(6):
                self.anchor_pairs.append((ecg_idx, lead_idx))

    def __len__(self):
        return len(self.anchor_pairs)

    def __getitem__(self, idx):
        # Determine which ecg_idx, lead_idx to use as anchor
        ecg_idx, anchor_idx = self.anchor_pairs[idx]
        anchor_ecg = self.ecgs[ecg_idx]  # (6, T)

        # Sample positive ECG (from a different patient)
        valid_indices = list(range(self.N))
        valid_indices.remove(ecg_idx)  # Remove the anchor ECG index
        positive_ecg_idx = random.choice(valid_indices)
        positive_ecg = self.ecgs[positive_ecg_idx]  # (6, T)

        # Create the context mask
        # Mask all electrodes except the anchor electrode
        context_mask = torch.ones(6, dtype=torch.bool)
        context_mask[anchor_idx] = False

        # Augment the ECG
        if (self.augment and random.random() < 0.5):
            # 50% chance to augment the ECG
            anchor_ecg = self._augment(anchor_ecg)
            positive_ecg = self._augment(positive_ecg)

        return (
            anchor_ecg,
            anchor_idx,
            positive_ecg,
            anchor_idx,  # Same lead index as in the anchor ECG
            context_mask
        )

    def _augment(self, signals):
        """
        Augment the signal by randomly performing
        one of several augmentations:
        - Additive noise
        - Random cropping
        - Heart rate perturbation
        signals: Tensor of shape (6, T),
        """

        augmentations = [
            self.additive_noise,
            self.random_crop
        ]
        augmentation = random.choice(augmentations)

        noise_level = random.uniform(0.001, 0.015)
        return augmentation(signals, noise_level=noise_level)

    def additive_noise(self, signals, noise_level=0.01):
        """
        Additive Gaussian noise to the signal.
        signals: Tensor of shape (6, T),
        noise_level: Standard deviation of the noise.
        """
        noise = torch.randn_like(signals) * noise_level
        return signals + noise

    def random_crop(self, signals, **kwargs):
        """
        Randomly crop the signal to a smaller size.
        signals: Tensor of shape (6, T),
        crop_size: Size of the cropped signal.
        """
        _, T = signals.shape
        # Random crop size between 64 and T
        crop_size = random.randint(T//4, T)
        start = random.randint(0, T - crop_size)

        signals = signals[:, start:start + crop_size]

        # Pad to make sure the signal is of size T
        signals = F.pad(signals, (0, T - crop_size), mode='constant', value=0)

        return signals


class LocalEncoder(nn.Module):
    """
    Shared 1D CNN encoder for each electrode signal.
    """

    def __init__(self, feature_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=12, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=7, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Flatten(),  # Flatten the output
            nn.Linear(128, feature_dim),  # Fully connected layer
            nn.ReLU()
        )

    def forward(self, x):
        # x is of shape (B, num_electrodes, T):
        # B: batch size, num_electrodes: 6, T: length of the signal
        # Reshape to (B * num_electrodes, 1, T) for Conv1d
        # Each electrode signal is processed independently
        # and in parallel

        assert x.dim() == 3, "Input must be of shape (B, N, T)"
        assert x.size(1) == 6, "Input must have 6 electrodes"

        B, N, T = x.shape
        # Flatten to (B * N, T) and add a channel dimension
        x = x.view(B * N, 1, T)

        out = self.encoder(x)  # (B * N, feature_dim)
        # Reshape back to (B, N, feature_dim)
        out = out.view(B, N, -1)

        return out


class DeepSetsContextEncoder(nn.Module):
    """Encoder for context information using DeepSets architecture"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
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

    def __init__(self, input_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # 6 classes for 6 precordial leads
        )

    def forward(self, z):
        """
        z: Tensor of shape (B, 2*D),
            where B is the batch size and D is the feature dimension.
        """
        return self.classifier(z)


class AURA12(nn.Module):
    """Main AURA module"""

    def __init__(self, feature_dim=128):
        super().__init__()
        self.feature_dim = feature_dim

        self.local_encoder = LocalEncoder(
            feature_dim=feature_dim
            )

        self.context_encoder = DeepSetsContextEncoder(
            input_dim=feature_dim,
            hidden_dim=64,
            output_dim=feature_dim
            )

        self.classifier = Classifier(input_dim=2*feature_dim)

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


def train_loop(model,
               dataloader,
               optimizer,
               lambda_clf=1.0,
               device='cuda',
               temperature=0.1,
               contrastive_learning=True
               ):
    model.train()

    for batch_idx, (
        anchor_signals,
        anchor_idx,
        positive_signals,
        positive_idx,
        context_mask
    ) in enumerate(dataloader):
        # Each batch is a list of items
        # Each item:
        #   - anchor_signals: a list of all 6 precordial signals in the
        #                     12-lead ECG (each of shape (T,))
        #   - anchor_idx: the index of the signal in anchor_signals to be
        #                 used as the anchor
        #   - positive_signals: a list of precordial signals from
        #                       different patients
        #   - positive_idx: the index of the signal in positive_signals
        #                   which corresponds to the same lead as the
        #                   anchor signal 
        #   - anchor_label: the label of the anchor signal (0-5, one for
        #                   each precordial lead)
        #   - context_mask: a boolean tensor of shape (6,) indicating which
        #                   electrodes to include in the context encoding

        # Move tensors to device
        anchor_signals = anchor_signals.to(device)
        anchor_idx = anchor_idx.to(device)
        positive_signals = positive_signals.to(device)
        positive_idx = positive_idx.to(device)
        context_mask = context_mask.to(device)

        # Forward pass: anchor + positive
        z_anchors, pred_anchors = model(
            anchor_signals, anchor_idx, context_mask
        )
        z_positives, _ = model(
            positive_signals, positive_idx, context_mask
        )

        # Constrastive loss
        contrastive_loss = torch.tensor(0.0, device=device)
        if contrastive_learning:
            contrastive_loss = nt_xent_loss(
                z_anchors, z_positives, temperature
            )

        # Classification loss (cross-entropy)
        clf_loss = F.cross_entropy(pred_anchors, anchor_idx)

        # Total loss
        loss = contrastive_loss + lambda_clf * clf_loss

        # Backward + Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f"Batch {batch_idx}/{len(dataloader)}: "
                f"Loss: {loss.item():.4f} "
                f"(contrastive: {contrastive_loss.item():.4f}, "
                f"classification: {clf_loss.item():.4f})"
            )


def test_loop(model,
              dataloader,
              lambda_clf=1.0,
              device='cuda',
              temperature=0.1,
              contrastive_learning=True):
    model.eval()  # Set model to evaluation mode

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (
            anchor_signals,
            anchor_idx,
            positive_signals,
            positive_idx,
            context_mask
        ) in enumerate(dataloader):

            # Move tensors to device
            anchor_signals = anchor_signals.to(device)
            anchor_idx = anchor_idx.to(device)
            positive_signals = positive_signals.to(device)
            positive_idx = positive_idx.to(device)
            context_mask = context_mask.to(device)

            # Forward pass: anchor + positive
            z_anchors, pred_anchors = model(
                anchor_signals, anchor_idx, context_mask
            )
            z_positives, _ = model(
                positive_signals, positive_idx, context_mask
            )

            # Constrastive loss
            contrastive_loss = 0.0
            if contrastive_learning:
                contrastive_loss = nt_xent_loss(
                    z_anchors, z_positives, temperature
                )

            # Classification loss (cross-entropy)
            clf_loss = F.cross_entropy(pred_anchors, anchor_idx)

            # Total loss
            loss = contrastive_loss + lambda_clf * clf_loss
            total_loss += loss.item()

            preds = pred_anchors.argmax(dim=1)  # Get predicted classes (B,)
            correct = (preds == anchor_idx).sum().item()
            total_correct += correct
            total_samples += anchor_signals.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    return avg_loss, accuracy


def train(
    model,
    train_dataloader,
    epochs=10,
    temperature=0.1,
    lambda_clf=1.0,
    device='cuda',
    save_path=None,
    val_dataloader=None,
    learning_rate=1e-3,
    contrastive_learning=True
):
    """
    Trains the AURA12 model on the provided dataloader.

    model: AURA12 model instance
    train_dataloader: DataLoader for training data

    epochs: Number of training epochs
    temperature: Temperature parameter for NT-Xent loss
    lambda_clf: Weight for the classification loss, (total loss is calculated
        as contrastive_loss + lambda_clf * classification_loss)
    device: Device to use for training (e.g., 'cuda' or 'cpu')
    save_path: Path to save the model checkpoint (optional)
    val_dataloader: DataLoader for validation data (optional). If provided,
            the model will be evaluated on this data after each epoch and
            the results will be printed.

    Returns:
        model: Trained AURA12 model
    """
    model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}-------------------------------------")
        train_loop(model, train_dataloader,
                   optimizer, lambda_clf=lambda_clf,
                   device=device, temperature=temperature,
                   contrastive_learning=contrastive_learning)
        if val_dataloader is not None:
            test_loop(model, val_dataloader,
                      lambda_clf=lambda_clf, device=device,
                      temperature=temperature,
                      contrastive_learning=contrastive_learning)
        print("--------------------------------------------------")
        print("\n")

        # Save the model checkpoint
        if save_path is not None:
            torch.save(model.state_dict(), f"{save_path}_{epoch+1}.pth")
            print(f"Model checkpoint saved to {save_path}_{epoch+1}.pth")
            print("\n")

    print("Done! Hip hip hooray!")
    return model

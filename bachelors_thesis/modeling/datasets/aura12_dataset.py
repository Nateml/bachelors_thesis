import random

from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class AURA12Dataset(Dataset):
    def __init__(self, ecg_tensor, augment_cfg: DictConfig = None):
        """
        ecg_tensor: A tensor of shape (N, 6, T) where N is the number
                  of samples, 6 is the number of precordial leads,
                    and T is the length of the signal.
        augment_cfg: Configuration object containing params for data augmentation.
                    If None, no augmentation will be applied.
        """
        super().__init__()
        self.ecgs = ecg_tensor  # (N, 6, T)
        self.N = ecg_tensor.size(0)  # Number of ECGs
        self.T = ecg_tensor.size(2)  # Length of the signal (T)
        self.augment_cfg = augment_cfg
        self.random = random.Random(42)

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
        if (self.augment_cfg and self.augment_cfg.augment and self.random.random() < self.augment_cfg.augment_chance):
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

        aug_params = self.augment_cfg.augment_choice_strategy_params

        # Select the augmentations to use based on the configuration
        augmentations = []
        if 'noise' in aug_params.augment_types:
            augmentations.append(self.additive_noise)
        if 'random_crop' in aug_params.augment_types:
            augmentations.append(self.random_crop)

        assert len(augmentations) > 0, "No augmentations available"
        assert len(augmentations) == len(aug_params.augment_type_weights), \
            "Augmentation selection weights must match the number of augmentations"

        # Select augmentation from the available augmentations based on the configured strategy
        if (self.augment_cfg.augment_choice_strategy == 'random'):
            augmentation = random.choices(augmentations,
                                         weights=aug_params.augment_type_weights,
                                         k=1)[0]
        else:
            raise ValueError("Invalid augmentation choice strategy")

        # Pick a random noise level from the configured range
        noise_level = random.uniform(
            self.augment_cfg.augment_types.noise.noise_std_low,
            self.augment_cfg.augment_types.noise.noise_std_high
            )
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
        crop_size_low = self.augment_cfg.augment_types.random_crop.crop_size_low
        crop_size_high = self.augment_cfg.augment_types.random_crop.crop_size_high
        crop_size = random.randint(int(T * crop_size_low), int(T * crop_size_high))
        start = random.randint(0, T - crop_size)

        signals = signals[:, start:start + crop_size]

        # Pad to make sure the signal is of size T
        signals = F.pad(signals, (0, T - crop_size), mode='constant', value=0)

        return signals
import random

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


class SigLocDataset(Dataset):
    def __init__(self, ecg_tensor, augment_cfg=None, **kwargs):
        """
        ecg_tensor: A tensor of shape (N, E, T) where N is the number
                    of samples, E is the number of electrode signals,
                    and T is the length of each signal.
        """
        super().__init__()
        self.ecgs = ecg_tensor
        self.augment_cfg = augment_cfg
        self.random = random.Random(42)

    def __len__(self):
        # Number of samples 
        return self.ecgs.size(0) 
    
    def __getitem__(self, idx):
        # Get an ECG sample by index
        ecg = self.ecgs[idx]

        # Apply augmentation if specified
        if self.augment_cfg and self.augment_cfg.augment and self.random.random() < self.augment_cfg.augment_chance:
            ecg = self._augment(ecg)

        return ecg

    def _augment(self, ecg):
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
        return augmentation(ecg, noise_level=noise_level)

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
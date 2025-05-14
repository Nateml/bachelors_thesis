import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from bachelors_thesis.data.load_ptbdata_new import (
    ALL_LEADS,
    PRECORDIAL_LEADS,
)


class SigLocDataset(Dataset):
    def __init__(self, ecg_tensor, augment_cfg=None, shuffle_leads=False, filter_leads=PRECORDIAL_LEADS, **kwargs):
        """
        ecg_tensor: A tensor of shape (N, E, T) where N is the number
                    of samples, E is the number of electrode signals,
                    and T is the length of each signal.
        """
        super().__init__()
        self.ecgs = ecg_tensor
        self.filter_leads = filter_leads

        is_precordial_dataset = True if self.ecgs.shape[1] == len(PRECORDIAL_LEADS) else False
        is_full_dataset = True if self.ecgs.shape[1] == len(ALL_LEADS) else False

        # Filter only the leads we want
        if filter_leads is not None:
            if is_precordial_dataset:
                self.lead_indices = [PRECORDIAL_LEADS.index(lead) for lead in filter_leads]
            elif is_full_dataset:
                self.lead_indices = [ALL_LEADS.index(lead) for lead in filter_leads]
                # self.ecgs = [0 1 2 3 4 5 6 7 8 9 10 11]
                # self.ecgs = [0 1 2 6 7 8 9 10 11]
            else:
                raise ValueError("Invalid number of leads in the input tensor: " + str(self.ecgs.shape[1]))

        self.ecgs = self.ecgs[:, self.lead_indices, :]
        self.index_to_label = {i: lead for i, lead in enumerate(filter_leads)}

        self.label_order = np.arange(len(filter_leads))
        self.label_order_tensor = torch.tensor(self.label_order, dtype=torch.long)

        self.augment_cfg = augment_cfg
        self.random = random.Random(42)
        self.shuffle_leads = shuffle_leads

    def __len__(self):
        # Number of samples 
        return self.ecgs.size(0) 
    
    def __getitem__(self, idx):
        # Get an ECG sample by index
        ecg = self.ecgs[idx]

        # Apply augmentation if specified
        if self.augment_cfg and self.augment_cfg.augment and self.random.random() < self.augment_cfg.augment_chance:
            ecg = self._augment(ecg)

        # Shuffle along the lead dimension
        # Generate a random lead order for each sample in the batch
        lead_order = self.label_order.copy()
        if self.shuffle_leads:
            lead_order = self.random.sample(lead_order, len(lead_order))
            # Shuffle the leads in the ECG sample

        ecg = ecg[lead_order, :]

        # Convert lead_order to a tensor
        lead_order = torch.tensor(lead_order, dtype=torch.long)

        return ecg, self.label_order

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
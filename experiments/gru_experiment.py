import sys

import hydra
from loguru import logger
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
import wandb

from src.data.preprocessing import resample
from src.logger import init_logger
from src.modeling.train import train
from src.registries.dataset_registry import get_dataset
from src.registries.preprocessor_registry import get_preprocessor

# Enable cuDNN autotuner for performance optimization
# This is useful for convolutional networks
# It will test a few convolution algorithms and pick the fastest one
# Small overhead at sartup, but should improve performance
torch.backends.cudnn.benchmark = True

params = [
    {
        "hidden_size": 128,
        "num_layers": 1,
        "bidirectional": False,
        "dropout": 0.1
    },
    {
        "hidden_size": 128,
        "num_layers": 2,
        "bidirectional": False,
        "dropout": 0.1
    },
    {
        "hidden_size": 128,
        "num_layers": 3,
        "bidirectional": False,
        "dropout": 0.1
    },
    {
        "hidden_size": 128,
        "num_layers": 4,
        "bidirectional": True,
        "dropout": 0.1
    }
]

def _make_amp_objs(device: str, amp_dtype: str):
    use_amp = bool(amp_dtype)
    if not use_amp:
        return None, torch.enable_grad
    
    # GradScaler for mixed precision training
    need_scaler = (device == "cuda" and amp_dtype == "float16")
    scaler = GradScaler(device, enabled=need_scaler)

    # Autocast context manager
    def autocast_ctx():
        return autocast(device, dtype=getattr(torch, amp_dtype))

    return scaler, autocast_ctx

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """
    Entry point for training a model.
    """

    # Initialize the logger
    init_logger(log_level=cfg.run.log_level.upper(), log_file=cfg.run.log_file)
    logger.info(f"Running experiment: {cfg.run.experiment_name}")

    # AMP settings
    scaler, autocast_ctx = _make_amp_objs(cfg.run.device, OmegaConf.select(cfg, "run.amp_dtype"))
    logger.info(f"Using AMP: {scaler is not None}")

    # 1. Load the data
    data_path = cfg.dataset.path
    if cfg.dataset.only_precordial:
        data_path = data_path + "/precordial"
    else:
        data_path = data_path + "/all"
    train_data = np.load(data_path + "/train.npy")
    val_data = np.load(data_path + "/val.npy")

    # Initialize W&B
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.run.experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.wandb.tags,
            group=cfg.wandb.group
        )

        # Log dataset as a W&B artifact
        #dataset_artifact = wandb.Artifact(
        #    name=f"{cfg.dataset.name}_dataset",
        #    type="dataset",
        #    metadata={
        #        "description": cfg.dataset.description,
        #        "path": cfg.dataset.path,
        #        "sampling_rate": cfg.dataset.sampling_rate
        #    }
        #)
        #dataset_artifact.add_dir(cfg.dataset.path)
        #wandb.log_artifact(dataset_artifact)

    if OmegaConf.select(cfg, "dataset.resampled_rate"):
        logger.info(f"Resampling data to {cfg.dataset.resampled_rate} Hz...")
        train_data = resample(train_data, cfg.dataset.sampling_rate, cfg.dataset.resampled_rate)
        val_data = resample(val_data, cfg.dataset.sampling_rate, cfg.dataset.resampled_rate)

    # Preprocess the data
    for preprocessor in cfg.preprocessor_group.preprocessors:
        logger.info(f"Applying preprocessor: {preprocessor._preprocessor_.strip('_')}...")
        preprocessor_func = get_preprocessor(preprocessor._preprocessor_)
        train_data = preprocessor_func(train_data, sampling_rate=cfg.dataset.sampling_rate, **preprocessor)
        val_data = preprocessor_func(val_data, sampling_rate=cfg.dataset.sampling_rate, **preprocessor)

    # 2. Convert to torch tensors
    train_data = torch.from_numpy(train_data).float()
    val_data = torch.from_numpy(val_data).float()

    logger.info(f"Train data shape: {train_data.shape}")
    logger.info(f"Validation data shape: {val_data.shape}")

    # 3. Create the dataloaders
    torch_dataset = get_dataset(cfg)

    logger.info(f"Using dataset: {torch_dataset.__name__}")
    logger.info(f"Augmentation is set to {cfg.augment.augment}")

    # Reshape the data to (N, 6, T) from (N, T, 6)
    train_data = train_data.permute(0, 2, 1)
    val_data = val_data.permute(0, 2, 1)

    # Create the dataset
    train_dataset = torch_dataset(train_data, augment_cfg=cfg.augment)
    # Leave out the augmentation for validation
    val_dataset = torch_dataset(val_data, augment_cfg=None)
    
    # 4. Create the dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.run.batch_size,
        shuffle=True,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.run.batch_size,
        shuffle=False,
        pin_memory=True
    )

    # 5. Train the model
    train(
        cfg=cfg,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        scaler=scaler,
        autocast=autocast_ctx
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted.")
        sys.exit(0)
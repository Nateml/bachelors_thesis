import hydra
from loguru import logger
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
import wandb

from bachelors_thesis.logger import init_logger
from bachelors_thesis.modeling.train import train
from bachelors_thesis.registries.dataset_registry import get_dataset
from bachelors_thesis.registries.preprocessor_registry import get_preprocessor


@hydra.main(config_path="../../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """
    Entry point for training a model.
    """

    # Initialize the logger
    init_logger(log_level=cfg.run.log_level.upper(), log_file=cfg.run.log_file)
    logger.info(f"Running experiment: {cfg.run.experiment_name}")

    # 1. Load the data
    data_path = cfg.dataset.path
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
        dataset_artifact = wandb.Artifact(
            name=f"{cfg.dataset.name}_dataset",
            type="dataset",
            metadata={
                "description": cfg.dataset.description,
                "path": cfg.dataset.path,
                "sampling_rate": cfg.dataset.sampling_rate
            }
        )
        dataset_artifact.add_dir(cfg.dataset.path)
        wandb.log_artifact(dataset_artifact)

    # Preprocess the data
    for preprocessor in cfg.preprocessor_group.preprocessors:
        logger.info(f"Applying preprocessor: {preprocessor._preprocessor_.strip('_')}...")
        preprocessor_func = get_preprocessor(preprocessor._preprocessor_)
        train_data = preprocessor_func(train_data, sampling_rate=cfg.dataset.sampling_rate, **preprocessor)
        val_data = preprocessor_func(val_data, sampling_rate=cfg.dataset.sampling_rate, **preprocessor)

    # 2. Convert to torch tensors
    train_data = torch.from_numpy(train_data).float()
    val_data = torch.from_numpy(val_data).float()

    # 3. Create the dataloaders
    torch_dataset = get_dataset(cfg)

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
        val_dataloader=val_dataloader
    )

if __name__ == "__main__":
    main()
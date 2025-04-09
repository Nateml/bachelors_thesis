
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import wandb

from bachelors_thesis.modeling.dataset_registry import get_dataset
from bachelors_thesis.modeling.datasets.aura12_dataset import AURA12Dataset
from bachelors_thesis.modeling.model_registry import get_model
from bachelors_thesis.modeling.utils import log_to_wandb, save_checkpoint


class AverageMeter:
    def __init__(self):
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
    
    def average(self):
        return self.sum / self.count if self.count != 0 else 0

class AverageMeterDict: 
    def __init__(self):
        self.meters = {}
    
    def update(self, values: dict, n=1):
        for k, v in values.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter()
            self.meters[k].update(v, n)

    def average(self):
        return {k: meter.average() for k, meter in self.meters.items()}

def train_loop(model: nn.Module,
               optimizer: torch.optim.Optimizer,
               dataloader: DataLoader,
               loss_fn,
               cfg: DictConfig,
               device: str ='cuda'
               ):

    # Put the model in training mode
    model.train() 

    metrics = AverageMeterDict()

    # Enumerate over the dataloader
    for batch_idx, batch in enumerate(dataloader):
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
        #   - context_mask: a boolean tensor of shape (6,) indicating which
        #                   electrodes to include in the context encoding

        # Move tensors to device
        batch = [v.to(device) for v in batch]

        # Compute loss
        loss, step_metrics = loss_fn(model, batch, cfg)

        # Backward + Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics.update(step_metrics)

        if batch_idx % 100 == 0:
            print(
                f"Batch {batch_idx}/{len(dataloader)}: "
            )
            # Print the step metrics
            for k, v in step_metrics.items():
                print(f"{k}: {v:.4f}")

    # Return metrics
    return metrics.average()

def eval_loop(model: nn.Module,
              dataloader: DataLoader,
              loss_fn,
              cfg: DictConfig,
              device: str ='cuda'
              ):

    # Put the model in evaluation mode
    model.eval() 

    metrics = AverageMeterDict()

    # Enumerate over the dataloader
    for batch_idx, batch in enumerate(dataloader):
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
        batch = [v.to(device) for v in batch]

        # Compute loss
        _, step_metrics = loss_fn(model, batch, cfg)

        # Update metrics
        metrics.update(step_metrics)

    # Return metrics
    return metrics.average()

def train(
    cfg: DictConfig,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader
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
    assert cfg.model.model_name == "aura12", "Only AURA12 model is supported"
    assert cfg.loss.name == "dual_loss", "Only dual loss is supported"

    wandb.init(
        project=cfg.wandb.project,
        name=cfg.run.experiment_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=cfg.wandb.tags,
        group=cfg.wandb.group
    )

    # 1. Load model from registry
    model_type, loss_fn = get_model(cfg.model.model_name)
    model = model_type(cfg.model).to(cfg.run.device)

    # 2. Set up the optimizer
    assert cfg.optimizer.name == "adam", "Only Adam optimizer is supported"
    optimizer = Adam(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)

    best_val_loss = float('inf')

    for epoch in range(cfg.run.epochs):
        print(f"Epoch {epoch+1}/{cfg.run.epochs}-------------------------------------")

        train_results = train_loop(model, optimizer, train_dataloader, loss_fn, cfg)
        val_results = eval_loop(model, val_dataloader, loss_fn, cfg)

        log_to_wandb(train_results, val_results, epoch, cfg)

        print("--------------------------------------------------")
        print("\n")

        # Save the model checkpoint
        if cfg.run.checkpoint:
            best = val_results['loss'] < best_val_loss
            if best:
                best_val_loss = val_results['loss']

            save_checkpoint(
                model, cfg, epoch, val_results, best=best
            ) 


    wandb.finish()

    print("Done! Hip hip hooray!")
    return model

@hydra.main(config_path="../../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # 1. Load the data
    data_path = cfg.dataset.path
    train_data = np.load(data_path + "/train.npy")
    val_data = np.load(data_path + "/val.npy")

    # 2. Convert to torch tensors
    train_data = torch.from_numpy(train_data).float()
    val_data = torch.from_numpy(val_data).float()

    # 3. Create the dataloaders
    torch_dataset = get_dataset(cfg)
    if torch_dataset is AURA12Dataset:
        # Reshape the data to (N, 6, T) from (N, T, 6)
        train_data = train_data.permute(0, 2, 1)
        val_data = val_data.permute(0, 2, 1)
        # Create the dataset
        train_dataset = torch_dataset(train_data, cfg.augment)
        val_dataset = torch_dataset(val_data, cfg.augment)
    else:
        raise ValueError(f"Unknown dataset for: {cfg.model.model_name}. Dataset is of type {torch_dataset}.")
    
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
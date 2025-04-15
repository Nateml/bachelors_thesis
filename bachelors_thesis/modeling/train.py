from loguru import logger
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import wandb

from bachelors_thesis.logger import (
    create_progress_bar,
    log_epoch_summary,
    update_progress_bar,
)
from bachelors_thesis.modeling.utils import AverageMeterDict, log_to_wandb, save_checkpoint
from bachelors_thesis.registries.model_registry import get_model


def train_loop(model: nn.Module,
               optimizer: torch.optim.Optimizer,
               dataloader: DataLoader,
               loss_fn,
               cfg: DictConfig
               ):

    # Put the model in training mode
    model.train() 

    device = cfg.run.device

    metrics = AverageMeterDict()
    progress_bar = create_progress_bar(dataloader)

    # Enumerate over the dataloader
    for batch in dataloader:
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

        # Update the metrics and progress bar
        metrics.update(step_metrics)
        update_progress_bar(progress_bar, loss)

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
    for batch in dataloader:
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

    # 1. Load model from registry
    model_type, loss_fn = get_model(cfg.model.model_name)
    model = model_type(cfg.model).to(cfg.run.device)

    # 2. Set up the optimizer
    assert cfg.optimizer.name == "adam", "Only Adam optimizer is supported"
    optimizer = Adam(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)

    best_val_loss = float('inf')

    for epoch in range(cfg.run.epochs):
        logger.info(f"Epoch {epoch+1}/{cfg.run.epochs}--------------------------------------")

        train_results = train_loop(model, optimizer, train_dataloader, loss_fn, cfg)
        val_results = eval_loop(model, val_dataloader, loss_fn, cfg)

        log_epoch_summary(epoch, train_results, val_results)
        if cfg.wandb.enabled:
            log_to_wandb(train_results, val_results, epoch, cfg)

        logger.info("-----------------------------------\n")

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

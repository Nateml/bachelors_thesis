import copy
import sys

from loguru import logger
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import wandb

from bachelors_thesis.logger import (
    create_progress_bar,
    log_epoch_summary,
    update_progress_bar,
)
from bachelors_thesis.modeling.utils import AverageMeterDict, log_to_wandb, save_checkpoint
from bachelors_thesis.registries.model_registry import get_model
from bachelors_thesis.registries.scheduler_registry import get_scheduler


def train_loop(model: nn.Module,
               optimizer: torch.optim.Optimizer,
               dataloader: DataLoader,
               loss_fn,
               cfg: DictConfig,
               epoch: int,
               scaler: torch.cuda.amp.GradScaler = None,
               autocast: torch.cuda.amp.autocast = torch.enable_grad,
               scheduler: torch.optim.lr_scheduler._LRScheduler = None,
               ):

    # Put the model in training mode
    model.train() 

    device = cfg.run.device

    metrics = AverageMeterDict()
    progress_bar = create_progress_bar(dataloader)

    iters = len(dataloader)

    # Enumerate over the dataloader
    for i, (signals, lead_order) in enumerate(dataloader):

        # Move tensors to device
        signals = signals.to(device)
        lead_order = lead_order.to(device)

        # Compute loss
        with autocast():
            loss, step_metrics = loss_fn(model, signals, lead_order, cfg)

        # Backward + Optimizer step
        optimizer.zero_grad()

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Update the metrics and progress bar
        metrics.update(step_metrics)
        update_progress_bar(progress_bar, loss)

        # Update the learning rate scheduler if specified
        if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)\
        or isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step(epoch + i / iters)

    # Return metrics
    return metrics.average()

def eval_loop(model: nn.Module,
              dataloader: DataLoader,
              loss_fn,
              cfg: DictConfig,
              device: str ='cuda',
              scaler: torch.cuda.amp.GradScaler = None,
              autocast: torch.cuda.amp.autocast = torch.enable_grad
              ):

    # Put the model in evaluation mode
    model.eval() 

    metrics = AverageMeterDict()

    with torch.no_grad():
        # Enumerate over the dataloader
        for signals, lead_order in dataloader:
            # Move tensors to device
            signals = signals.to(device)
            lead_order = lead_order.to(device)

            # Compute loss
            with autocast():
                _, step_metrics = loss_fn(model, signals, lead_order, cfg)

            # Update metrics
            metrics.update(step_metrics)

    # Return metrics
    return metrics.average()

def train(
    cfg: DictConfig,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    scaler: torch.cuda.amp.GradScaler = None,
    autocast: torch.cuda.amp.autocast = torch.enable_grad
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

    try:
        # 1. Load model from registry
        model_type, loss_fn = get_model(cfg.model.model_name)
        model = model_type(cfg.model).to(cfg.run.device)

        # 2. Set up the optimizer
        assert cfg.optimizer.name == "adam", "Only Adam optimizer is supported"
        optimizer = AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)

        # Set up scheduler if specified
        if OmegaConf.select(cfg, "scheduler"):
            scheduler = get_scheduler(cfg.scheduler.name)(
                optimizer=optimizer,
                **cfg.scheduler.params
            )
        else:
            scheduler = None

        best_val_results = {
            "loss": float('inf'),
            "accuracy": 0.0,
        }
        epochs_no_improvement = 0
        patience = cfg.run.patience
        min_delta = cfg.run.min_delta

        for epoch in range(cfg.run.epochs):
            logger.info(f"Epoch {epoch+1}/{cfg.run.epochs}--------------------------------------")

            if scheduler:
                logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")

            train_results = train_loop(model, optimizer, train_dataloader, loss_fn, cfg, epoch, scaler=scaler, autocast=autocast, scheduler=scheduler)
            val_results = eval_loop(model, val_dataloader, loss_fn, cfg, scaler=scaler, autocast=autocast)

            # Update the learning rate scheduler if specified
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_results['loss'])

            is_best = val_results['loss'] + min_delta < best_val_results['loss']

            logger.info("-----------------------------------\n")

            # Check for early stopping
            if is_best:
                epochs_no_improvement = 0
                best_val_results = copy.deepcopy(val_results)

            else:
                epochs_no_improvement += 1

            # Logging
            log_epoch_summary(epoch, train_results, val_results)
            if cfg.wandb.enabled:
                log_to_wandb(train_results, val_results, best_val_results, epoch, cfg)

            # Save the model checkpoint
            if cfg.run.checkpoint:
                save_checkpoint(
                    model, cfg, epoch, val_results, best=is_best
                ) 

            if epochs_no_improvement >= patience:
                logger.info(f"Early stopping after epoch {epoch+1}/{cfg.run.epochs}")
                break

            # Free up memory
            torch.cuda.empty_cache()

        print("Done! Hip hip hooray!")
        return model
    finally:
        if cfg.wandb.enabled:
            if epoch:
                wandb.run.summary["stopped_epoch"] = epoch + 1
            wandb.finish()

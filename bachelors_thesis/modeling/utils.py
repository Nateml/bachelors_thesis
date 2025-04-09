from omegaconf import OmegaConf
import torch
import wandb


def log_to_wandb(train_metrics, val_metrics, epoch, cfg):
    # Prepend "train" or "val" to the metrics
    train_results = {f"train/{k}": v for k, v in train_metrics.items()}
    val_results = {f"val/{k}": v for k, v in val_metrics.items()}
    wandb.log({
        "epoch": epoch + 1,
        **train_results,
        **val_results,
        "model_config": OmegaConf.to_container(cfg, resolve=True),
    })

def save_checkpoint(model, cfg, epoch, metrics: dict, best: bool = False):
    checkpoint_path = f"{cfg.run.checkpoint_path}_latest.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}\n")

    # Save latest checkpoint to wandb
    artifact = wandb.Artifact(
        name=f"{cfg.run.experiment_name}_latest",
        type="model",
        metadata={
            "epoch": epoch + 1,
            **metrics,
            "model_config": OmegaConf.to_container(cfg, resolve=True),
        }
    )
    artifact.add_file(checkpoint_path)
    wandb.log_artifact(artifact, aliases=["latest"])

    if best:
        # Create a new artifact for the best model
        artifact = wandb.Artifact(
            name=f"{cfg.run.experiment_name}_best",
            type="model",
            metadata={
                "epoch": epoch + 1,
                **metrics,
                "model_config": OmegaConf.to_container(cfg, resolve=True),
            }
        )
        checkpoint_path = f"{cfg.run.checkpoint_path}_best.pth"
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact, aliases=["best"])
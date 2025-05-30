import pathlib

from omegaconf import OmegaConf
import torch
import wandb


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

def log_to_wandb(train_metrics, val_metrics, best_val_metrics, epoch, cfg):
    # Prepend "train" or "val" to the metrics
    # If bool is True, add "best" to the val metrics
    train_results = {f"train/{k}": v for k, v in train_metrics.items()}
    val_results = {f"val/{k}": v for k, v in val_metrics.items()}
    best_val_results = {f"val/best/{k}": v for k, v in best_val_metrics.items()}
    wandb.log({
        "epoch": epoch + 1,
        **train_results,
        **val_results,
        **best_val_results,
    })

def save_checkpoint(model, cfg, epoch, metrics: dict, best: bool = False):
    checkpoint_path = f"{cfg.run.checkpoint_path}.pth"
    # Create the directory if it doesn't exist
    path = pathlib.Path(cfg.run.checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save the model
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}\n")

    # Save latest checkpoint to wandb
    artifact = wandb.Artifact(
        name=f"{cfg.run.experiment_name}",
        type="model",
        metadata={
            "epoch": epoch + 1,
            **metrics
        }
    )
    artifact.add_file(checkpoint_path)
    aliases = ["latest"]
    if best:
        aliases.append("best")
    wandb.log_artifact(artifact, aliases=aliases)

def load_model_from_wandb(model_type, project_path: str, run_id: str, version: str, device: str = "cuda"):
    api = wandb.Api()

    run = api.run(f"{project_path}/{run_id}")
    config = dict(run.config)

    run_name = run.name
    artifact = api.artifact(f"{project_path}/{run_name}:{version}")
    artifact_path = artifact.download()

    cfg = OmegaConf.create(config)

    checkpoint = torch.load(artifact_path + f"/{run_name}.pth", map_location=device)

    model = model_type(cfg.model).to(device)
    model.load_state_dict(checkpoint)

    return model, cfg
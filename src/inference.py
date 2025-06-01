"""
Use this file to run inference on the model.
It should load the model, load (and preprocess) the data, and run inference.
The results can either be saved to a file, or the performance can be printed
to the console.
"""

from enum import Enum
import os
from pathlib import Path
from typing import Optional

from loguru import logger
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import typer

from src.data.load_ptbdata_new import ALL_LEADS, PRECORDIAL_LEADS
from src.evaluation import lead_level_accuracy, set_level_accuracy
from src.modeling.datasets.siglab_dataset import SigLabDataset
from src.run import lead_sets
from src.utils import apply_preprocessors, confusion_matrix, hungarian_predictions

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = Path(current_dir).resolve()

# Define the path to the model binary
model_dir = current_dir.parent / "models"

data_dir = current_dir.parent / "data" / "processed"

app = typer.Typer()

known_models = defaultdict(lambda: None)
known_models.update({
    "siglabv2": model_dir / "siglabv2" / "siglabv2.pth",
    "precordial": model_dir / "precordial" / "precordial.pth",
    "deepsets": model_dir / "deepsets" / "deepsets.pth",
    "nocontext": model_dir / "nocontext" / "nocontxt.pth"
    })

known_datasets = defaultdict(lambda: None)
known_datasets.update({
    "ptbxl100all": data_dir / "ptbxl100all",
    "ptbxl500all": data_dir / "ptbxl500all",
    "ptbxl100norm": data_dir / "ptbxl100norm",
    "ptbxl500norm": data_dir / "ptbxl500norm"
    })

class Split(str, Enum):
    train = "train"
    val = "val"
    test = "test"

@app.command()
def main(
    model: str = typer.Option(
        ...,
        help="Name of the model, or the path to the trained model checkpoint (e.g., 'model.pth)",
        prompt=True,
        case_sensitive=False,
    ),
    dataset: str = typer.Option(
        ...,
        help="Either the name of the dataset to use, e.g., 'ptbxl100all', or a path to a dataset directory." \
        "The dataset directory should contain a 'val.npy' file, a 'test.npy' file, and a 'train.npy' file" \
        "depending on the desired subset to use for inference. The data should be in the format expected by the SigLabDataset.",
        prompt=True,
        case_sensitive=False,
    ),
    split: Split = typer.Option(
        ...,
        help="Which split of the dataset to use for inference. Options are 'train', 'val', or 'test'.",
        prompt=True,
        case_sensitive=False,
    ),
    save_path: Optional[str] = typer.Option(
        None,
        help="Path to save the inference results. If not provided, results will not be saved.",
        prompt=False,
        case_sensitive=False,
    ),
    batch_size: int = typer.Option(
        32,
        help="Batch size for inference. Default is 16.",
        prompt=False,
        case_sensitive=False,
    )
):
    """
    Run inference on the specified model and dataset.
    """

    logger.info(f"Loading model from {model}...")

    model = known_models.get(model, model)

    # Check if the model file exists
    if not os.path.exists(model):
        logger.error(f"Model file {model} does not exist.")
        raise FileNotFoundError(f"Model file {model} does not exist.")

    model_dir = model.parent

    # Check for CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("CUDA available. Using GPU for inference.")
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available. Using CPU for inference.")

    # Load the model
    model_instance = torch.load(model, weights_only=False)
    model_instance.to(device)
    model_instance.eval()

    logger.info(f"Model {model} loaded successfully.")

    # Load the config
    config_path = model_dir / "config.yaml"
    if not config_path.exists():
        error_msg = f"Config file {config_path} does not exist. The model directory should contain a 'config.yaml' file."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Load the config 
    logger.info(f"Loading config from {config_path}...")
    cfg = OmegaConf.load(config_path)

    old_dataset = dataset
    dataset = known_datasets.get(dataset, None)

    # Load the dataset
    if dataset:
        dataset = Path(dataset)

        if OmegaConf.select(cfg, "dataset.only_precordial"):
            dataset = dataset / "precordial"
        else:
            dataset = dataset / "all"

        dataset = dataset.resolve()
    else:
        dataset = Path(old_dataset).resolve()

    logger.info(f"Loading dataset from {dataset}...")
    if not dataset.exists():
        error_msg = f"Dataset directory {dataset} does not exist."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Check if the split file exists
    split_file = dataset / f"{split.value}.npy"
    if not split_file.exists():
        error_msg = f"File {split_file} does not exist. The dataset directory should contain a '{split.value}.npy' file."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"Loading {split.value} split from {split_file}...")
    data = np.load(split_file)

    # Check that the data is the right shape
    if len(data.shape) != 3 or data.shape[-1] != cfg.model.num_classes:
        error_msg = f"Data shape {data.shape} is not compatible with the model. Expected shape (num_samples, {cfg.model.num_classes}, {cfg.model.num_classes})."
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("Data loaded succesfully.")

    # Apply preprocessors
    logger.info("Applying preprocessors...")
    data = apply_preprocessors(data, preprocessors=cfg.preprocessor_group.preprocessors, sampling_rate=cfg.dataset.sampling_rate)

    # Convert to torch tensors
    data = torch.from_numpy(data).float().to(device)
    data = data.permute(0, 2, 1)

    # Create the dataset
    lead_filter = lead_sets[OmegaConf.select(cfg, "run.leads", default="all")]
    dataset_instance = SigLabDataset(data, filter_leads=lead_filter)
    dataloader = DataLoader(dataset_instance, batch_size=batch_size, shuffle=False)

    if OmegaConf.select(cfg, "dataset.only_precordial") is None\
        or OmegaConf.select(cfg, "dataset.only_precordial"):

        data = data[:, [PRECORDIAL_LEADS.index(lead) for lead in lead_filter], :]
    else:
        data = data[:, [ALL_LEADS.index(lead) for lead in lead_filter], :]

    logger.info(f"Running inference on {len(dataset_instance)} samples with batch size {batch_size}...")
    c = cfg.model.num_classes
    logits = np.zeros((len(dataset_instance), c, c))
    targets = np.zeros((len(dataset_instance), c))

    for idx, (signals, lead_labels) in enumerate(tqdm(dataloader, desc="Inference Progress", unit="batch")):
        signals = signals.to(device)
        lead_labels = lead_labels.to(device)

        with torch.no_grad():
            these_logits = model_instance(signals)
            logits[(idx * batch_size):(idx * batch_size + batch_size)] = these_logits.cpu().numpy()
            targets[(idx * batch_size):(idx * batch_size + batch_size)] = lead_labels.cpu().numpy()

    predictions = logits.argmax(axis=-1)

    logger.info("Inference completed.")

    if save_path:
        save_path = Path(save_path).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving results to {save_path}...")

        np.savez(save_path, logits=logits, predictions=predictions, targets=targets)
        logger.info(f"Results saved to {save_path}.")

    # Calculate and print performance metrics
    report_str = report(logits, targets, leads=lead_filter)

    logger.info("Performance Report:")
    logger.info(report_str)


def report(logits, targets, leads):

    def print_cm(cm, leads):
        # Print each row, leading with the lead name
        out_str = "\t".join([""] + leads) + "\n"
        for i, lead in enumerate(leads):
            out_str += f"{lead}:\t" + "\t".join(f"{x}" for x in cm[i]) + "\n"

        return out_str

    def print_recall(recall, leads):
        out_str = ""
        for i, lead in enumerate(leads):
            out_str += f"{lead}: {recall[i]:.4f}, "
        return out_str
    
    preds = logits.argmax(axis=-1)
    hung_preds = hungarian_predictions(logits)

    lead_acc = lead_level_accuracy(predictions=preds, targets=targets)[0]
    set_acc = set_level_accuracy(predictions=preds, targets=targets)[0]

    hung_lead_acc = lead_level_accuracy(predictions=hung_preds, targets=targets)[0]
    hung_set_acc = set_level_accuracy(predictions=hung_preds, targets=targets)[0]

    cm = confusion_matrix(predictions=preds, targets=targets)

    # Calculate the recall for each lead
    recall = np.diag(cm) / cm.sum(axis=1)

    # ---------------------------------------------
    report_str = (
        f"\nLead Level Accuracy: {lead_acc:.4f}\n"
        f"Set Level Accuracy: {set_acc:.4f}\n"
        f"Hungarian Lead Level Accuracy: {hung_lead_acc:.4f}\n"
        f"Hungarian Set Level Accuracy: {hung_set_acc:.4f}\n"
        f"Confusion Matrix:\n{print_cm(cm, leads)}\n"
        f"Recall per lead: {print_recall(recall, leads)}\n"
    )

    return report_str


if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
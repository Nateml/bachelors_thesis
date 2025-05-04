"""
This script processes raw datasets. The script can be run from the command line and takes a dataset name as an argument.
This CLI users Typer, so it supports helpful features like listing the available datasets to process.
The script loads the selected dataset, processes it (saving interim data to the interim data directory), and saves the
final processed data to the processed data directory (as configured in config.py).
"""

from enum import Enum

from loguru import logger
import numpy as np
import typer

from bachelors_thesis.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from bachelors_thesis.data import load_ptbdata_new as load_ptbdata

# from bachelors_thesis.data import load_ptbdata
from bachelors_thesis.data import split_ptbxl

app = typer.Typer()


class Dataset(str, Enum):
    ptbxl100all = "ptbxl100all"
    ptbxl500all = "ptbxl500all"
    ptbxl100norm = "ptbxl100norm"
    ptbxl500norm = "ptbxl500norm"


@app.command()
def main(
    dataset: Dataset = typer.Option(
        ...,
        help="Which dataset to process",
        prompt=True,
        case_sensitive=False,
        ),
    only_precordial_leads: bool = typer.Option(
        True,
        help="Whether to only use precordial leads",
        prompt=True,
        case_sensitive=False
        ),
):
    if dataset in ["ptbxl100all", "ptbxl500all", "ptbxl100norm", "ptbxl500norm"]:
        sampling_rate = 100 if dataset.value.startswith("ptbxl100") else 500
        logger.info(f"Loading {dataset.value} dataset... This will take a while.")

        input_dir = str(RAW_DATA_DIR / "ptb-xl") + "/"

        X, meta = load_ptbdata.load_data(
            data_path=input_dir,
            sampling_rate=sampling_rate,
            limit=None,
            only_precordial_leads=only_precordial_leads,
            only_normal=dataset in ["ptbxl100norm", "ptbxl500norm"],
            )

        logger.info("Dataset loaded successfully.")

        # Save X to interim data as .npy

        interim_dir = INTERIM_DATA_DIR / dataset.value
        if only_precordial_leads:
            interim_dir = interim_dir / "precordial"
        else:
            interim_dir = interim_dir / "all"

        interim_dir.mkdir(parents=True, exist_ok=True)

        np.save(interim_dir / "X.npy", X)
        meta.to_csv(interim_dir / "meta.csv", index=True)

        logger.info(f"X and Y saved to {interim_dir}.")

        # Process for aura12
        logger.info("Processing dataset...")
        output_dir = PROCESSED_DATA_DIR / dataset.value
        output_dir = output_dir / "precordial" if only_precordial_leads else output_dir / "all"
        split_ptbxl.main(X, meta, output_dir=output_dir)


if __name__ == "__main__":
    app()

"""
This script splits PTB-XL data into training, validation, and test sets based on the strat_fold column in the metadata.
As per the PTB-XL creators' recommendations, the first 8 folds are used for training, the 9th fold for validation,
and the 10th fold for testing. This is because the records in folds 9 and 10 underwent at least one human validation so are of
parcticularly high quality.
The processed data is saved as .npy files in the PROCESSED_DATA_DIR directory.
The metadata is saved as .csv files for the training, validation, and test sets respectively. This metadata includes extra
information such as patient identifiers and biometric data, signal metadata, and ecg statements.
"""
from loguru import logger
from pathlib import Path
import numpy as np
import pandas as pd


def main(
        X: np.ndarray,
        meta: pd.DataFrame,
        output_dir: Path,
):
    data_train = X[np.where(meta.strat_fold < 9)]
    data_val = X[np.where(meta.strat_fold == 9)]
    data_test = X[np.where(meta.strat_fold == 10)]

    meta_train = meta.iloc[np.where(meta.strat_fold < 9)]
    meta_val = meta.iloc[np.where(meta.strat_fold == 9)]
    meta_test = meta.iloc[np.where(meta.strat_fold == 10)]

    assert len(data_train) == len(meta_train), "Training data and metadata lengths do not match."
    assert len(data_val) == len(meta_val), "Validation data and metadata lengths do not match."
    assert len(data_test) == len(meta_test), "Test data and metadata lengths do not match."

    # Save numpy arrays to .npy
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "train.npy", data_train)
    np.save(output_dir / "val.npy", data_val)
    np.save(output_dir / "test.npy", data_test)
    meta_train.to_csv(output_dir / "meta_train.csv", index=True)
    meta_val.to_csv(output_dir / "meta_val.csv", index=True)
    meta_test.to_csv(output_dir / "meta_test.csv", index=True)


    logger.success(f"PTB-XL data processed and saved to"
                   f"{output_dir}.")

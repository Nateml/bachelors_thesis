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
import numpy as np
import pandas as pd

from bachelors_thesis.config import PROCESSED_DATA_DIR


def main(
        X: np.ndarray,
        meta: pd.DataFrame,
        dataset: str,
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
    OUTPUT_DIR = PROCESSED_DATA_DIR / dataset
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "train.npy", data_train)
    np.save(OUTPUT_DIR / "val.npy", data_val)
    np.save(OUTPUT_DIR / "test.npy", data_test)
    meta_train.to_csv(OUTPUT_DIR / "meta_train.csv", index=True)
    meta_val.to_csv(OUTPUT_DIR / "meta_val.csv", index=True)
    meta_test.to_csv(OUTPUT_DIR / "meta_test.csv", index=True)


    logger.success(f"PTB-XL data processed and saved to"
                   f"{OUTPUT_DIR / dataset}.")

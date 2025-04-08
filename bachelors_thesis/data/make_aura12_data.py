"""
This script processes data for the AURA12 model. It is called by the dataset.py script.
It takes the raw data and splits it into training, validation, and test sets based on the strat_fold column in the Y dataframe.
As per the PTB-XL creators' recommendations, the first 8 folds are used for training, the 9th fold for validation,
and the 10th fold for testing. This is because the records in folds 9 and 10 underwent at least one human validation so are of
parcticularly high quality.
The processed data is saved as .npy files in the PROCESSED_DATA_DIR/aura12 directory.
The metadata is also saved as .npy files for the training, validation, and test sets respectively. This metadata includes extra
information such as patient identifiers and biometric data, signal metadata, and ecg statements.
"""
from loguru import logger
import numpy as np

from bachelors_thesis.config import PROCESSED_DATA_DIR


def main(
        X: np.ndarray,
        Y: np.ndarray,
        dataset: str,
):
    data_train = X[np.where(Y.strat_fold < 9)]
    data_val = X[np.where(Y.strat_fold == 9)]
    data_test = X[np.where(Y.strat_fold == 10)]

    meta_train = Y.iloc[np.where(Y.strat_fold < 9)]
    meta_val = Y.iloc[np.where(Y.strat_fold == 9)]
    meta_test = Y.iloc[np.where(Y.strat_fold == 10)]

    # Save numpy arrays to .npy
    OUTPUT_DIR = PROCESSED_DATA_DIR / dataset
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "train.npy", data_train)
    np.save(OUTPUT_DIR / "val.npy", data_val)
    np.save(OUTPUT_DIR / "test.npy", data_test)
    np.save(OUTPUT_DIR / "meta_train.npy", meta_train)
    np.save(OUTPUT_DIR / "meta_val.npy", meta_val)
    np.save(OUTPUT_DIR / "meta_test.npy", meta_test)

    logger.success(f"Data for AURA12 processed and saved to {
        OUTPUT_DIR / dataset
    }.")

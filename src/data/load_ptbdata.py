import ast
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import wfdb

PRECORDIAL_LEADS = [
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

LIMB_LEADS = [
    "I",
    "II",
    "III"
]

AUGMENTED_LEADS = [
    "aVR",
    "aVL",
    "aVF"
]

ALL_LEADS = LIMB_LEADS + AUGMENTED_LEADS + PRECORDIAL_LEADS

def load_data(
        data_path: str,
        sampling_rate: int = 100,
        limit: Optional[int] = None,
        only_precordial_leads: bool = False,
        only_normal: bool = True,
        n_workers: Optional[int] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load PTB‑XL recordings and metadata.

    Parameters
    ----------
    data_path : str
        Path to the **root** of the extracted PTB‑XL archive, e.g. ``"/datasets/PTB-XL"``.
    sampling_rate : {100, 500}, default ``500``
        Desired sampling rate of the loaded signals.
    limit : int, optional
        If given, load only the first ``limit`` records (after applying the other
        filters). Handy while developing.
    only_precordial_leads : bool, default ``False``
        If ``True``, keep only the precordial leads V1‑V6.
    only_normal : bool, default ``True``
        If ``True``, keep only recordings whose *diagnostic superclass* contains
        ``"NORM"``.
    n_workers : int, optional
        Number of worker threads used to read WFDB files in parallel. ``None``
        falls back to the default chosen by :pyclass:`concurrent.futures.ThreadPoolExecutor`.

    Returns
    -------
    X : np.ndarray of shape ``(n_records, n_samples, n_channels)``
        The ECG signals.
    Y : pandas.DataFrame
        Metadata rows aligned with **X** (same row order).
    """

    print("[load_data] --------------------------------------------------")
    print(f"[load_data] data_path                : {data_path}")
    print(f"[load_data] sampling_rate            : {sampling_rate} Hz")
    print(f"[load_data] only_precordial_leads    : {only_precordial_leads}")
    print(f"[load_data] only_normal              : {only_normal}")
    if limit is not None:
        print(f"[load_data] row limit                : {limit}")
    if n_workers is not None:
        print(f"[load_data] thread pool size         : {n_workers}\n")

    # --------------------------------------------
    # helpers
    # --------------------------------------------
    def _precordial_indices(header: Any) -> List[int]:
        """Return column indices of precordial leads inside *header*."""
        return [i for i, lead in enumerate(header.sig_name)
                if lead in {"V1", "V2", "V3", "V4", "V5", "V6"}]

    def lead_indices_in_order(header: Any, desired_order: List[str]) -> List[int]:
        name2idx = {name.lower(): i for i, name in enumerate(header.sig_name)}
        return [name2idx[lead.lower()] for lead in desired_order if lead.lower() in name2idx]
    
    def _load_one(path:str, desired_leads) -> np.ndarray:
        header = wfdb.rdheader(path)
        idx = lead_indices_in_order(header, desired_leads)
        sig, _ = wfdb.rdsamp(path, channels=idx)
        return sig.astype(np.float32)
    
    # --------------------------------------------
    # metadata
    # --------------------------------------------
    meta_path = os.path.join(data_path, "ptbxl_database.csv")
    print(f"[metadata] Reading {meta_path} ...", end=" ")
    meta = pd.read_csv(meta_path,
                       index_col="ecg_id")
    print("done (", len(meta), "rows )")
    meta.scp_codes = meta.scp_codes.apply(ast.literal_eval)

    # diagnostic aggregation table
    diag_path = os.path.join(data_path, "scp_statements.csv")
    print(f"[metadata] Reading diagnostics map {diag_path} ...", end=" ")
    diag_map = (
        pd.read_csv(diag_path, index_col=0)
        .query("diagnostic == 1")
    )
    print("done")

    print("[metadata] Aggregating diagnostic_superclass ...", end=" ")
    meta["diagnostic_superclass"] = meta.scp_codes.apply(
        lambda d: [diag_map.loc[k, "diagnostic_class"] for k in d
                   if k in diag_map.index]
    )
    print("done")

    if only_normal:
        before = len(meta)
        meta = meta[meta.diagnostic_superclass.map(lambda s: "NORM" in s)]
        print(f"[filter] only_normal=True -> {before} -> {len(meta)} rows")

    if limit is not None:
        before = len(meta)
        meta = meta.iloc[:limit]
        print(f"[filter] limit={limit} -> {before} -> {len(meta)} rows")

    # choose correct column containing the WFDB file path
    fname_col = "filename_hr" if sampling_rate == 500 else "filename_lr"
    filenames = meta[fname_col].to_numpy()
    full_paths = [os.path.join(data_path, fn) for fn in filenames]

    # Choose the leads to keep
    if only_precordial_leads:
        # Precordial leads
        desired_leads = PRECORDIAL_LEADS
    else:
        # All leads
        desired_leads = ALL_LEADS

    print(f"[filter] Keeping {len(desired_leads)} leads: {desired_leads}")

    # -----------------------------------------------------------------
    # read signal files in parallel (IO-bound -> threads are fine here)
    # -----------------------------------------------------------------
    print(f"[signals] Loading {len(full_paths)} WFDB files with ThreadPoolExecutor ..")
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        signals = list(pool.map(partial(_load_one, desired_leads=desired_leads),
                                full_paths))
    print("[signals] Finished loading waveforms.")

    X = np.stack(signals)  # (n_records, n_samples, n_channels)
    print(f"[signals] X.shape = {X.shape}\n")

    # reset the index to preserve one-to-one mapping with X
    return X, meta.reset_index(drop=False)
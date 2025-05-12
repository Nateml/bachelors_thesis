"""
Loads the PTB-XL dataset from the source directory. Most of this code has been taken from the PTB-XL example data loading script.
It is really slow right now, so I will look into optimizing it later.
Link to the source: https://physionet.org/content/ptb-xl/1.0.3/
"""

import ast

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
        data_path,
        sampling_rate,
        limit=None,
        only_precordial_leads=False,
        only_normal=True,
        ):

    def get_precordial_lead_indices(path):
        # Precordial leads
        precordial_leads = ["V1", "V2", "V3", "V4", "V5", "V6"]
        header = wfdb.rdheader(path)
        lead_indices = [i for i, lead in enumerate(header.sig_name)
                        if lead in precordial_leads]
        return lead_indices

    def lead_indices_in_order(header, desired_order):
        name2idx = {name: i for i, name in enumerate(header.sig_name)}
        return [name2idx[lead] for lead in desired_order if lead in name2idx]

    def load_raw_signals(file_list, path, desired_leads):
        signals = []
        for rec in file_list:
            header = wfdb.rdheader(path + rec)
            idx = lead_indices_in_order(header, desired_leads)
            sig, _ = wfdb.rdsamp(path + rec, channels=idx)
            signals.append(sig)

        return np.stack(signals)  # Shape: (N, T, L)

    # Function to load ECG signal data
    def load_raw_data(df, sampling_rate, path, lead_set):
        file_list = df.filename_lr.values if sampling_rate == 100 else df.filename_hr.values
        data = load_raw_signals(file_list, path, lead_set)
        return data

    # Load and convert annotation data
    print("Loading and converting annotation data...")
    Y = pd.read_csv(data_path + "/ptbxl_database.csv", index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    if limit:
        Y = Y.iloc[:limit]

    # Load scp_statements.csv for diagnostic aggregation
    print("Loading diagnostic aggregation data...")
    agg_df = pd.read_csv(data_path + "/scp_statements.csv", index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    # Function to aggregate diagnostic superclasses
    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    # Apply diagnostic superclass
    print("Applying diagnostic superclass...")
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

    if only_normal:
        Y = Y[Y.diagnostic_superclass.apply(lambda x: "NORM" in x)]

    # Get the desired leads
    # This is the order in which the channels will be stored in the .npy file
    # This order is important for the model to work properly
    if only_precordial_leads:
        lead_set = PRECORDIAL_LEADS
    else:
        lead_set = ALL_LEADS

    # Load raw signal data
    print("Loading raw signal data...")
    X = load_raw_data(Y, sampling_rate, data_path, lead_set=lead_set)

    return X, Y

import numpy as np
import pandas as pd
import wfdb
import ast


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

    # Function to load ECG signal data
    def load_raw_data(df, sampling_rate, path):
        if sampling_rate == 100:
            # Low resolution
            data = [wfdb.rdsamp(path+f,
                                channels=get_precordial_lead_indices(path+f)
                                if only_precordial_leads else None)
                    for f in df.filename_lr]
        else:
            # High resolution
            data = [wfdb.rdsamp(path+f,
                                channels=get_precordial_lead_indices(path+f)
                                if only_precordial_leads else None)
                    for f in df.filename_hr]
        data = np.array([signal for signal, _ in data])
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

    # Load raw signal data
    print("Loading raw signal data...")
    X = load_raw_data(Y, sampling_rate, data_path)

    return X, Y

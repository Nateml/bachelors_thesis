from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def lead_level_accuracy(predictions: np.ndarray = None, logits: np.ndarray = None, targets: np.ndarray = None):
    assert predictions is not None or logits is not None, "Either predictions or logits must be provided."
    assert predictions is None or logits is None, "Only one of predictions or logits can be provided."

    num_samples = logits.shape[0] if logits is not None else predictions.shape[0]
    num_classes = logits.shape[1] if logits is not None else predictions.shape[1]

    if targets is None:
        targets = np.broadcast_to(np.arange(num_classes), (num_samples, num_classes))

    assert targets.shape == (num_samples, num_classes), "Targets must have the same shape as the first two dimensions of logits."

    if logits is not None:
        # -------------------------------
        # Compute predictions from logits
        # -------------------------------
        assert logits.ndim == 3, "Logits must be 3D."
        #assert logits.shape[1] == logits.shape[2], "Last two dimensions of logits must be square."

        predictions: np.ndarray = logits.argmax(axis=2)  # (N, C)

    # ------------------------------------
    # Calculate accuracy using predictions
    # ------------------------------------
    assert predictions.ndim == 2, "Predictions must be 2D."

    # predictions is (N, C) and targets is (N, C)
    if targets is None:
        targets = np.broadcast_to(np.arange(num_classes), (num_samples, num_classes))

    assert targets.shape == (num_samples, num_classes), "Targets must have the same shape as the first two dimensions of logits."

    correct: np.ndarray = predictions == targets
    acc = correct.mean()
    return acc


def set_level_accuracy(predictions: np.ndarray = None, logits: np.ndarray = None, targets: np.ndarray = None):
    assert predictions is not None or logits is not None, "Either predictions or logits must be provided."
    assert predictions is None or logits is None, "Only one of predictions or logits can be provided."

    num_samples = logits.shape[0] if logits is not None else predictions.shape[0]
    num_classes = logits.shape[1] if logits is not None else predictions.shape[1]

    if targets is None:
        targets = np.broadcast_to(np.arange(num_classes), (num_samples, num_classes))

    assert targets.shape == (num_samples, num_classes), "Targets must have the same shape as the first two dimensions of logits."

    if predictions is None:
        # -------------------------------
        # Compute predictions from logits
        # -------------------------------
        assert logits.ndim == 3, "Logits must be 3D."
        #assert logits.shape[1] == logits.shape[2], "Last two dimensions of logits must be square."

        predictions: np.ndarray = logits.argmax(axis=2)  # (N, C)

    # ------------------------------------
    # Calculate accuracy using predictions
    # ------------------------------------
    assert predictions.ndim == 2, "Predictions must be 2D."

    # predictions is (N, C) and targets is (N, C)
    if targets is None:
        targets = np.broadcast_to(np.arange(num_classes), (num_samples, num_classes))

    assert targets.shape == (num_samples, num_classes), "Targets must have the same shape as the first two dimensions of logits."

    acc = np.mean(np.all(predictions == targets, axis=1))
    return acc

def pretty_code_density_plot(code_densities          : Counter,
                             miscl_code_densities    : Counter,
                             n                       : int  = 12,
                             n_misclassified         : int  = None,
                             n_total                 : int = None,
                             relative                : bool = True,
                             scp_statement_csv       : str | Path = "scp_statements.csv",
                             bar_thickness           : float = 0.45,
                             colours                 : tuple = ("#4C9BE8", "#F5A873")):
    """
    Horizontal grouped bar‑chart that compares the *n* most common
    mis‑classified SCP codes with their distribution in the full dataset.
    Bars are labelled by **code**; a description table sits underneath.
    -----------------------------------------------------------------------
    Parameters
    ----------
    code_densities        : Counter   – counts for the whole dataset
    miscl_code_densities  : Counter   – counts for the mis‑classified set
    n                     : int       – how many top mis‑classified codes
    n_misclassified       : int       – number of mis‑classified records
    n_total              : int        – number of records in the dataset
    relative              : bool      – plot frequencies (%) instead of raw counts
    scp_statement_csv     : str|Path  – maps Code → Description (PTB‑XL file)
    bar_thickness         : float     – vertical size of each bar
    colours               : tuple     – colours for the two groups
    """
    # ------------------------------------------------------------------ #
    # 1. Data wrangling                                                  #
    # ------------------------------------------------------------------ #
    top_codes = [c for c, _ in miscl_code_densities.most_common(n)]

    df = pd.DataFrame({
        "Misclassified" : [miscl_code_densities[c] for c in top_codes],
        "All records"   : [code_densities.get(c, 0) for c in top_codes],
    }, index=top_codes)

    if relative:
        assert n_misclassified is not None, "n_misclassified must be provided for relative frequencies."
        assert n_total is not None, "n_total must be provided for relative frequencies."
        df["Misclassified"] /= n_misclassified
        df["All records"]   /= n_total

    # pull human‑readable descriptions
    if Path(scp_statement_csv).exists():
        stmt = pd.read_csv(scp_statement_csv, index_col=0)["description"]
        descs = stmt.reindex(df.index).fillna("(n/a)")
    else:
        descs = pd.Series("(unknown)", index=df.index)

    # ------------------------------------------------------------------ #
    # 2. Plot                                                            #
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(11, 8), constrained_layout=True)
    gs  = fig.add_gridspec(2, 1, height_ratios=[4, 1.4], hspace=0.05)

    # --- a) horizontal bars ------------------------------------------ #
    ax = fig.add_subplot(gs[0])

    y = np.arange(len(df))
    ax.barh(y - bar_thickness/2, df["Misclassified"],
            height=bar_thickness, color=colours[0])
    ax.barh(y + bar_thickness/2, df["All records"],
            height=bar_thickness, color=colours[1])

    ax.set_yticks(y)
    ax.set_yticklabels(df.index)          # <- **SCP codes here**
    ax.invert_yaxis()
    ax.set_xlabel("Relative frequency" if relative else "Count")
    ax.set_title(f"Relative Frequency of SCP codes in misclassified ECGs vs. all ECGs (Top {n} codes)")

    fig.legend(labels=["Misclassified", "All records"],
               loc="center right",
               bbox_to_anchor=(0.93, 0.5),
               frameon=False)

    # some breathing room on the left so codes don't get clipped
    # plt.subplots_adjust(left=0.28)

    # --- b) description table ---------------------------------------- #
    tb_ax = fig.add_subplot(gs[1])
    tb_ax.axis("off")                     # table only, no axes
    tb_ax.set_title("SCP code descriptions", fontweight="bold", pad=42)

    cell_text = [[c, d] for c, d in zip(df.index, descs)]
    table = tb_ax.table(cellText=cell_text,
                        colLabels=["Code", "Description"],
                        loc="center",
                        cellLoc="left",
                        colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)                   # row‑height tweak

    plt.show()

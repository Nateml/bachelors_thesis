# Signals Are All You Need: A Deep Learning Architecture for ECG Lead Identification

## Abstract

Although electrocardiographic imaging (ECGI) provides detailed cardiac activation maps, its clinical accessibility remains limited — one key barrier being the accurate localisation of electrodes on the torso to construct a body surface potential map (BSPM). As a step towards eliminating this overhead, this study develops a deep learning model for automated 12-lead ECG lead identification from signal data alone. A permutation equivariant architecture is introduced that couples a branched convolutional network with a GRU encoder and utilises a series of Set Attention Blocks (SABs) to integrate inter-lead information.

Trained using the PTB-XL database, the model achieved an overall lead-level accuracy of 98.5% [95% CI 98.2-98.7], and demonstrated moderate robustness across cardiac conditions. Principal component analysis (PCA) showed that the model internalises the precordial V1-V6 arc as well as the horizontal and vertical electrical planes. Gradient-based attribution analysis revealed the Q and R waves to be the key morphological discriminators. Recall for lead aVF was the highest (99.8%), and lowest for lead V3 (96.7%). In general, recall for limb/augmented leads (99.2%) was higher than for precordial leads (97.5%).

To the authors' knowledge, this research presents the first model for signal-based lead identification in the 12-lead ECG, paving the way for continuous electrode localisation and truly imageless ECGI.

# Installation

The project dependencies are listed in `requirements.txt`. It is recommended to create and activate a virtual environment using `venv` or `virtualenv`, and then install the requirements like so:

```
> pip install -r requirements.txt
```

This **does not** contain PyTorch, which is required to use the models, due to different hardware requiring different versions (CPU vs CUDA). Please install PyTorch v2.6.0 by following the instructions [here](https://pytorch.org/get-started/previous-versions/).

To avoid installing unnecessary packages, the ones which are required to run the Jupyter notebooks are added seperately in the file `requirements-nb.txt`. If you plan on running the notebooks, just pip install from this file.

The code for this project was developed using Python 3.10.6. Different versions of Python may (or may not...) cause compatability issues.

## Data
This thesis uses the PTB-XL database. Since the dataset is large, it is not included in this GitHub repo. In order to train or run inference, download the database directly from [PhysioNet](https://physionet.org/content/ptb-xl/). Alternatively, the following command can be used: `wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/`.

Extract the files into `data/raw/ptb-xl/`. The directory should now look like this:

```
├── data/
    └── raw/
        └── ptb-xl/
            └── records100/
            └── records500/
            └── ptbxl_database.csv
            └── scp_statements.csv
            └── ...
```

Once the raw PTB-XL data is ready, the `dataset.py` script can be used to process the data:

```
> python src/dataset.py
```

The `dataset.py` script, by default, will process the 100Hz samples of the PTB-XL dataset. Additional options can be found by executing `> python src/dataset.py --help`.

The processed data should now be stored in `data/processed/ptbxl100all/all/`.

## Training 

If you would like to train a model, you can utilise the `run.py` script.

```
> python src/run.py
```

All the configuration options can be defined in the `config/config.yaml` file, as well as the other `.yaml` files in the `config` directory.

All the parameters in the configuration files can also be overriden via the command-line:

```
> python src/run.py run.exp_num="101" run.name="descriptive-name" run.device=cuda
```

If you do not have a device with CUDA capabilities, override the `run.device` argument with `cpu`. WARNING: training on CPU is *incredibly* slow.

The resulting model will be stored in the `checkpoints` directory.

## Inference

If you would just like to run inference with the model, then the `inference.py` script does the job.

```
> python src/inference.py
```

## Project Organization

```
├── data
│   ├── interim        <- Intermediate data
│   ├── processed      <- Processed data for the models to use
│   └── raw            <- Where the downloaded PTB-XL data goes
│
├── models             <- Trained and serialised models
│
├── checkpoints        <- Where the output of training a new model goes
│
├── notebooks          <- Jupyter notebooks.
│   ├── exploratory    <- "Exploratory" (messy) notebooks
│   └── reports        <- Cleaner notebooks containing some results
│
├── pyproject.toml     <- Python project configuration file with package metadata
│
├── requirements.txt   <- The requirements files for the core code
|
├── requirements-nb.txt <- Requirements for running the Jupyter notebooks
│
└── src                <- Source code
    │
    ├── dataset.py     <- Script to download and process PTB-XL data
    │
    ├── run.py         <- Entry script for training the model
    │
    └── modeling       <- Code related to the pytorch models
```

--------


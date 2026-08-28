# mlbiosig: Machine Learning Tool for Biosignature Interpretation on Icy Ocean Worlds

![CI](https://github.com/Gabriella8080/mlbiosig/actions/workflows/CI.yml/badge.svg)
![License](https://img.shields.io/badge/license-BSD--3-blue)


## Overview

`mlbiosig` an interpretable binary machine learning (ML) framework for distinguishing between biotic and abiotic pyrolysis-gas chromatography-mass spectrometry (py-GC-MS) samples, and identifying the mass-to-charge ratio (m/z) fragment ions driving model predictions. This work is motivated by the challenge of abiotic mimicry facing NASA's Europa Clipper (Pappalardo et al. 2024) and its MAss Spectrometer for Planetary EXploration (MASPEX) instrument (Waite et al. 2024).

This work transforms 154 py-GC-MS biotic and abiotic datasets into normalised py-MS features analogous to MASPEX's intermediate data products. Four classifier architectures (Logistic Regression, SVC, Random Forest, XGBoost) are trained and evaluated. Model predictions are interpreted using permutation feature importance, coefficient analysis, and SHapley Additive exPlanations (SHAP) to identify informative m/z fragment ions and determine abundance-dependent relationships between interacting features. 

The resulting `mlbiosig` Python package provides an automated implementation of the preprocessing, feature engineering, model training, evaluation, and interpretability frameworks developed across this study.

> **Accompanying Paper**: *From Mass Spectra to Biosignatures: Interpretable Machine Learning for Life Detection on Icy Ocean Worlds* - in preparation.


## Repository Structure

| Paths | Description |
| ----- | ------------|
| [`data/`](data/) | Mass-spectral CSV files extracted from ICOG py-GC-MS datasets (separated into [`biotic/`](data/biotic/) and [`abiotic/`](data/abiotic/) according to sample class). |
| [`docs/`](docs/) | Documentation covering [data preparation](docs/pygcms-preparation.md) and [sample inventory](docs/total-sample-breakdown.md). |
| [`mlbiosig/`](mlbiosig/) | `mlbiosig` Python package source code.  |
| [`notebooks/mlbiosig_demo/`](/irp-gmr24/notebooks/mlbiosig_demo/) | [Demonstration notebook](notebooks/mlbiosig_demo/01_mlbiosig_demo.ipynb) showing exemplar `mlbiosig` workflow. |
| [`notebooks/mlbiosig_dev/`](notebooks/mlbiosig_dev/) | Development notebooks documenting implementations, methodological choices, and experimental results. |
| [`tests`](tests/) | Unit tests validating `mlbiosig` functionalities and edge case performance.|
| [`environment.yml`](environment.yml) | Conda environment for `mlbiosig` specifying required dependencies.|


## Data Availability

The raw py-GC-MS `.D` datasets used in this work are proprietary to the ICOG laboratories and are therefore not made publicly available. The 154 mass-spectral CSV files extracted from these `.D` datasets and used for experiments are provided across [`data/biotic/`](data/biotic/) and [`data/abiotic/`](data/abiotic/). 

Extensive [documentation](docs/pygcms-preparation.md) describing the preparation of raw py-GC-MS data, extraction of mass-spectral measurements, and exporting of the resulting CSV files using Agilent MassHunter software is provided.

> **Sample Inventory**: A breakdown of all raw py-GC-MS `.D` samples used in this work can be found [here](docs/total-sample-breakdown.md). Ultimately, 140 biotic and 14 abiotic samples were included.


## `mlbiosig` Installation

To install `mlbiosig`, follow the instructions below:

```bash
conda env create -f environment.yml
conda activate biosignature-ml
pip install -e .
```

The package can then be imported using:

```bash
import mlbiosig
```

The recommended environment for using `mlbiosig` is the provided Conda [environment](environment.yml).


## Example Usage & Workflow

`mlbiosig` is organised into the four following modules:

* [`preprocess`](mlbiosig/preprocess.py): Processes mass-spectral CSV data per sample and constructs py-MS feature matrix.
* [`feature_engineering`](mlbiosig/feature_engineering.py): Performs feature cleaning, transformations, and engineering.
* [`classifiers`](mlbiosig/classifiers.py): Conducts automated hyperparameter tuning on fixed search ranges, and trains four classifier architectures.
* [`evaluate`](mlbiosig/evaluate.py): Evaluates model performance and investigates interpretability.

An example of using `mlbiosig` and the modules above is provided in the following [demonstration notebook](notebooks/mlbiosig_demo/01_mlbiosig_demo.ipynb). This includes instructions on loading and constructing the py-MS feature matrix, applying preprocessing and feature engineering, training and evaluating all four classifiers, applying interpretability frameworks, and a guide on collectively using these frameworks to investigate the m/z fragment ions driving biotic predictions.

## Development Notebooks

The development notebooks document the chronological implementation and methodological frameworks integrated into `mlbiosig`. This includes:

* Preprocessing ([Notebook 1](notebooks/mlbiosig_dev/01_preprocessing.ipynb))
* Feature engineering ([Notebook 2](notebooks/mlbiosig_dev/02_feature_engineering.ipynb))
* Balanced/Imbalanced ML classification training ([Notebook 3](notebooks/mlbiosig_dev/03_balanced_ml_classification.ipynb)/[4](notebooks/mlbiosig_dev/04_imbalanced_ml_classification.ipynb))
* Performance evaluation and interpretability ([Notebook 5](notebooks/mlbiosig_dev/05_evaluation.ipynb))

## Reproducibility

All code and processed mass-spectral CSV data required to reproduce the experimental results and analyses across this study are provided in this repository. 

Rerunning [notebooks](notebooks/) with the specified environment and data will reproduce reported results and unannotated figures subject to small numerical differences arising from the stochastic variation in hyperparameter tuning and classifier training. 

## Testing \& Validation

`mlbiosig` is validated using unit tests covering the package's functionality:

* [Preprocessing](tests/preprocess_test.py): Verifying sample datasets are provided for processing, along with sample-label requirements.
* [Model training](tests/classifiers_test.py): Checks hyperparameters are valid and stay within specified ranges, along with correct tree-based classifier parameters (`max_depth=None`) handling.
* [Interpretability functionality](tests/evaluate_test.py): Validates user model inputs for different interpretability tools. Coefficient analysis checks SVC uses `kernel=linear`. Hierarchical clustering verifies behaviour across different `t` values and expected distance matrix size from Ward's linkage output.

The entire test suite can be run using:
```bash
pytest
```


## License

The `mlbiosig` Python package source code is released under BSD 3-Clause License, with citation details provided in the [`CITATION.cff`](CITATION.cff) files. The laboratory-generated py-GC-MS `.D` datasets used are proprietary to the ICOG laboratories and are therefore not made publicly available.

## Bibliography

Pappalardo, R.T., Buratti, B.J., Korth, H. et al. (2024). *Science Overview of the Europa Clipper Mission.* Space Sci Rev 220, 40. https://doi.org/10.1007/s11214-024-01070-5

Waite, J.H., Burch, J.L., Brockwell, T.G. et al. (2024). *MASPEX-Europa: The Europa Clipper Neutral Gas Mass Spectrometer Investigation*. Space Sci Rev 220, 30. https://doi.org/10.1007/s11214-024-01061-6
# mlbiosig: Interpretable Machine Learning Tool for Biosignature Interpretation on Icy Ocean Worlds

![CI](https://github.com/Gabriella8080/mlbiosig/blob/main/.github/workflows/CI.yml/badge.svg)
![License](https://img.shields.io/badge/license-BSD--3-blue)


<i>Gabriella Rajpoot<sup>1</sup>, Solomon Hirsch<sup>1</sup>, Jonathan Watson<sup>1</sup>, Mark Sephton<sup>1</i>

<sup>1</sup>Royal School of Mines, Prince Consort Road, South Kensington, London SW7 2BP


## Overview

This independent research project's (IRP) repository developed an interpretable binary machine learning (ML) framework for distinguishing between biotic and abiotic pyrolysis-gas chromatography-mass spectrometry (py-GC-MS) samples, and identifying the mass-to-charge ratio (m/z) fragment ions driving model predictions. This work is motivated by the challenge of abiotic mimicry facing NASA's Europa Clipper (Pappalardo et al. 2024) and its MAss Spectrometer for Planetary EXploration (MASPEX) instrument (Waite et al. 2024). 

154 Imperial College Organic Geochemistry (ICOG) laboratory-generated py-GC-MS datasets, comprising 140 biotic and 14 abiotic samples, were transformed into normalised py-MS features analogous to MASPEX's intermediate data products. Four classifiers (Logistic Regression, Support Vector Classifier, Random Forest, XGBoost) were ultimately trained on 123 samples and tested on 31 samples, with performance evaluated using five-fold cross-validation F1, testing set F1, and ROC-AUC scores. Interpretability was explored using permutation importance, coefficient analysis, and SHapley Additive exPlanations (SHAP) to collectively identify the m/z fragment ions associated with biotic predictions, and determine the abundance relationships between interacting ions.

The resulting `mlbiosig` Python package provides an automated implementation of the preprocessing, feature engineering, model training, evaluation, and interpretability frameworks developed in this project.


## Repository Structure

| Paths | Description |
| ----- | ------------|
| [`data/`](data/) | Extracted mass-spectral CSV files from ICOG py-GC-MS datasets (separated into [`biotic/`](data/biotic/) and [`abiotic/`](data/abiotic/) corresponding to sample class). |
| [`docs/`](docs/) | Documentation covering [data preparation](docs/pygcms-preparation.md), [sample inventory](docs/total-sample-breakdown.md), [development notes](docs/development-notes/research-notes/), and [supervisor meeting slides](docs/development-notes/supervisor-meeting-slides/). |
| [`mlbiosig/`](mlbiosig/) | `mlbiosig` Python package source code.  |
| [`notebooks/mlbiosig_dev/`](notebooks/mlbiosig_dev/) | Development notebooks documenting implementations, methodological choices, and results presented in the [IRP final report](deliverables/gmr24-final-report.pdf). |
| [`notebooks/mlbiosig_demo/`](/irp-gmr24/notebooks/mlbiosig_demo/) | `mlbiosig` demonstration folder with [exemplar notebook](notebooks/mlbiosig_demo/01_mlbiosig_demo.ipynb). |
| [`tests`](tests/) | Unit tests checking `mlbiosig` validity and edge case performance.|
| [`environment.yml`](environment.yml) | Conda environment for `mlbiosig` containing required dependencies.|


## Data Availability

Raw py-GC-MS datasets are stored as `.D` files and are proprietary to the ICOG laboratories. The 154 mass-spectral CSV files used in this work are provided across [`data/biotic/`](data/biotic/) and [`data/abiotic/`](data/abiotic/), and were extracted from these `.D` datasets. The procedure for preparing raw py-GC-MS data, extracting mass-spectral measurements, and exporting these as CSV files using commercial *Agilent MassHunter* software is provided in extensive [documentation](docs/pygcms-preparation.md).

> **Sample Inventory**: A breakdown of all raw py-GC-MS `.D` samples used in this work can be found [here](docs/total-sample-breakdown.md). Ultimately, 140 biotic and 14 abiotic samples were included.


## `mlbiosig` Installation

`mlbiosig` is a modular framework and resulting Python package implemented from the notebooks developed in this IRP. It comprises of four modules:

* [`preprocess`](mlbiosig/preprocess.py): Processes mass-spectral CSV data per sample and constructs py-MS feature matrix.
* [`feature_engineering`](mlbiosig/feature_engineering.py): Performs feature cleaning, transformations, and engineering.
* [`classifiers`](mlbiosig/classifiers.py): Conducts automated hyperparameter tuning on fixed search ranges, and trains four classifier architectures.
* [`evaluate`](mlbiosig/evaluate.py): Evaluates model performance and investigates interpretability.

To install `mlbiosig`, follow the instructions below:

```bash
conda env create -f environment.yml
conda activate biosignature-ml
pip install -e .
```

## Example Workflow

An example of using `mlbiosig` is provided in the following [demonstration notebook](notebooks/mlbiosig_demo/01_mlbiosig_demo.ipynb). This includes loading and constructing the py-MS feature matrix, applying preprocessing and feature engineering, training and evaluating all four classifiers, applying interpretability frameworks, and a guide on collectively using these frameworks to investigate the m/z fragment ions driving biotic predictions.


## Testing \& Validation

`mlbiosig` is validated using unit tests covering:

* [Preprocessing](tests/preprocess_test.py): Verifying sample datasets are provided for processing, along with sample-label requirements.
* [Model training](tests/classifiers_test.py): Checks hyperparameters are valid and stay within specified ranges, along with correct tree-based classifier parameters (`max_depth=None`) handling.
* [Interpretability functionality](tests/evaluate_test.py): Validates user model inputs for different interpretability tools. Coefficient analysis checks SVC uses `kernel=linear`. Hierarchical clustering verifies behaviour across different `t` values and expected distance matrix size from Ward's linkage output.

The entire test suite can be run using:
```bash
pytest
```


## Development Documentation

Five development notebooks document the chronological implementation and methodological structure of `mlbiosig`. This includes preprocessing ([Notebook 1](notebooks/mlbiosig_dev/01_preprocessing.ipynb)), feature engineering ([Notebook 2](notebooks/mlbiosig_dev/02_feature_engineering.ipynb)), model training ([Notebook 3](notebooks/mlbiosig_dev/03_balanced_ml_classification.ipynb) & [4](notebooks/mlbiosig_dev/04_imbalanced_ml_classification.ipynb)), and evaluation and interpretability ([Notebook 5](notebooks/mlbiosig_dev/05_evaluation.ipynb)).

## Reproducibility

This repository provides all code, processed mass-spectral CSV data, development notebooks, environment specifications, and documentation required to reproduce the results and unannotated figures presented in the [IRP final report](deliverables/gmr24-final-report.pdf).

Some results may be subject to stochastic variation arising from hyperparameter tuning and classifier training. While rerunning notebooks might produce small numerical differences, the underlying workflow and analyses remain reproducible.

## License

The `mlbiosig` Python package source code is released under BSD 3-Clause License.

## Bibliography

Pappalardo, R.T., Buratti, B.J., Korth, H. et al. (2024). *Science Overview of the Europa Clipper Mission.* Space Sci Rev 220, 40. https://doi.org/10.1007/s11214-024-01070-5

Waite, J.H., Burch, J.L., Brockwell, T.G. et al. (2024). *MASPEX-Europa: The Europa Clipper Neutral Gas Mass Spectrometer Investigation*. Space Sci Rev 220, 30. https://doi.org/10.1007/s11214-024-01061-6
# Molecular Property Predictions

Unified notebook workflows for molecular property modeling with Chemprop and shared external plotting/utilities.

## Project Layout

- `1. Vapor Pressure/`
- `2. Oxidation Potential/`
- `3. pKa/`
- `4. Melting Point/`
- `5. Boiling Point/`
- `6. Exodia Prediction/`
- `2. Oxidation Potential/Scripts/` for OP source-preparation and model utilities
- `External Functions/`
  - `Plotting Functions/`
    - `dataprep_plotting.py`
    - `train_prediction_plotting.py`
- `Environment/`
  - `chemprop2.yml`

Each property folder contains:

- `*_DataPrep.ipynb` for curation/cleaning/splits
- `*_train_and_predict.ipynb` for training, inference, and diagnostics
- `Datasets/` for raw/curated/trainable inputs

## Shared Plotting Architecture

Plotting logic has been centralized so updates happen in one place:

- Dataprep plots: `External Functions/Plotting Functions/dataprep_plotting.py`
- Train/predict plots: `External Functions/Plotting Functions/train_prediction_plotting.py`

Notebooks import these shared modules via `External Functions` and `External Functions/Plotting Functions` path setup.

## Typical Workflow

1. Run a property `*_DataPrep.ipynb` notebook.
2. Confirm curated + trainable outputs under that property `Datasets/` path.
3. Run the matching `*_train_and_predict.ipynb` notebook.
4. Review generated metrics/predictions/figures.

## Environment Setup

A pinned environment file is included at:

- `Environment/chemprop2.yml`

Create and activate the environment with Conda:

```bash
conda env create -f Environment/chemprop2.yml
conda activate chemprop2
```

If the environment already exists and you want to update it:

```bash
conda env update -f Environment/chemprop2.yml --prune
```

## Dependencies

Primary dependencies are captured in `Environment/chemprop2.yml`.

## Notes

- `.gitignore` is configured to exclude generated model outputs, checkpoints, logs, and local environment artifacts.
- If you want to version model weights or output artifacts, remove the relevant ignore rules.

## Data availability

This publication draft omits raw, curated, and Chemprop-ready datasets, shared dataset copies, and trained model weights. Notebook outputs and source-specific inline measurement and molecule lists have been cleared. The workflows require separately obtained inputs placed in the documented `Datasets/` locations. Source-specific exclusions, corrections, and forced split selections were omitted from this draft, so these public notebooks alone will not reproduce the private training runs.

The empty data and model directories contain `.gitkeep` placeholders so their paths remain visible in a new checkout. The placeholders contain no dataset or model content.

# Student Performance MLOps

End-to-end MLOps pipeline for predicting student performance. The first stage
builds a reproducible machine learning baseline; later stages will add
experiment tracking, an inference API, deployment, and monitoring.

## Project structure

```text
data/raw/          Original dataset
data/processed/    Prepared data for training
notebooks/         Exploratory data analysis
src/data/          Data loading and preprocessing code
src/models/        Training and evaluation code
artifacts/models/  Saved model files
requirements.txt   Python dependencies
```

The raw dataset source and download instructions are in
[`data/raw/README.md`](data/raw/README.md).

Initial exploration is in [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).
The target, prediction time, and chosen inputs are in
[`docs/prediction_spec.md`](docs/prediction_spec.md).

## Local setup

Use a recent Python 3 version. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The `.venv` directory is excluded from Git by `.gitignore`.

## Train and compare models

After downloading the dataset, run from the project root:

```powershell
python -m src.models.train
python -m src.models.compare
python -m unittest discover -s tests
```

The first command makes a reproducible 80/20 split and fits a Ridge pipeline.
The comparison command uses 5-fold cross-validation on training rows to select
among a mean baseline, Ridge, and Random Forest. It then reports MAE, RMSE,
and R2 on the held-out test rows. See [`docs/model_evaluation.md`](docs/model_evaluation.md)
for results and limitations.

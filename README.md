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
[`data/raw/README.md`](data/raw/README.md). Baseline model implementation is a
later step in stage 1.

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

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

The folders are currently placeholders. Dataset selection and baseline model
implementation are separate steps in stage 1.

## Local setup

Use a recent Python 3 version. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The `.venv` directory is excluded from Git by `.gitignore`.

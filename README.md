# Student Performance MLOps

End-to-end MLOps project for predicting the final mathematics grade (`G3`,
0-20) from information available near the start of the course. This repository
currently contains the reproducible machine learning baseline. Experiment
tracking, an inference API, deployment, and monitoring are future stages.

## Project structure

```text
data/raw/          Original dataset
data/processed/    Prepared data for training
notebooks/         Exploratory data analysis
src/data/          Data loading and preprocessing code
src/models/        Training and evaluation code
artifacts/models/  Fitted preprocessing and model pipeline
requirements.txt   Python dependencies
```

The raw dataset source and download instructions are in
[`data/raw/README.md`](data/raw/README.md).

Initial exploration is in [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).
The target, prediction time, and chosen inputs are in
[`docs/prediction_spec.md`](docs/prediction_spec.md).

## Local setup

Use Python 3.10 or newer. From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The `.venv` directory is excluded from Git by `.gitignore`.

## Reproduce stage 1

Run these commands from the project root after installing dependencies:

```powershell
python src/data/download.py
python -m src.models.train
python -m src.models.compare
python -m unittest discover -s tests
```

The download command obtains the original UCI mathematics CSV (395 rows) and
checks its schema. The training command is a Ridge smoke check. The comparison
command splits the data 80/20 (`random_state=42`), selects among a mean
baseline, Ridge, and Random Forest using 5-fold cross-validation on training
rows, and evaluates the chosen model on 79 held-out rows. It saves the fitted
preprocessing-plus-model pipeline to
`artifacts/models/student_grade_pipeline.joblib` and verifies that loading it
back gives the same predictions. The final command runs the local tests.

For exploratory analysis, open `notebooks/01_eda.ipynb` with Jupyter. The
[prediction specification](docs/prediction_spec.md) explains the 12 selected
inputs and why `G1`, `G2`, and full-year absences are excluded.

## Current results and limits

Ridge had the lowest training cross-validation MAE (**3.243**). Its held-out
test MAE is **3.764** grade points, RMSE **4.603**, and R2 **-0.033**. The
negative R2 means this baseline did not demonstrate useful predictive quality
on the held-out students. The dataset is small and from two Portuguese schools;
the inputs must be checked for availability at prediction time before any real
use. See the [evaluation details](docs/model_evaluation.md) for the comparison
and limitations. This is a learning project, not a tool for student decisions.

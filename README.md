# Student Performance MLOps

End-to-end MLOps project for predicting the final mathematics grade (`G3`,
0-20) from information available near the start of the course. This repository
contains a reproducible machine learning baseline, MLflow experiment tracking,
and a prediction API. Deployment and monitoring are future stages.

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

## Stage 2: local MLflow UI

To start experiment tracking locally from the project root in **Git CMD**:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

Keep this server running. In a **second Git CMD** at the project root, activate
the same environment and run:

```cmd
.venv\Scripts\activate.bat
python -m src.models.compare --track
```

Open `http://127.0.0.1:5000` and select the
`student-performance-early-prediction` experiment. Each execution creates
three runs (Mean baseline, Ridge, Random Forest) with model parameters,
cross-validation MAE, and a fitted pipeline artifact. The chosen run also
records held-out test MAE, RMSE, and R2. Stop the server with `Ctrl+C`.
The local SQLite database and artifact directory are ignored by Git.
For the run comparison and selection rules, see
[`docs/mlflow_experiments.md`](docs/mlflow_experiments.md).

## Stage 3: prediction API

After installing dependencies and creating the model artifact, run from the
project root:

```cmd
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive request schema.
`GET /health` reports that the service started; `POST /predict` accepts exactly the
12 early-course features listed in the prediction specification and returns
`predicted_grade`. The pipeline loads once at startup. A missing or unreadable
model file prevents the service from starting. The numerical library versions
in `requirements.txt` match those used to create the checked-in `.joblib`
pipeline; reinstall dependencies after updating an existing environment.
For a request example and error behavior, see [`docs/api.md`](docs/api.md).

## Stage 4: Docker image

The Dockerfile packages the inference API and the checked-in pipeline using
only `requirements-api.txt`. From the project root, with Docker running:

```cmd
docker build -t student-performance-api:local .
docker run --rm -p 8000:8000 student-performance-api:local
```

For a background container on port 8001, use `scripts\run-docker.cmd`.
Build, request, logs, and stop commands are in [`docs/docker.md`](docs/docker.md).

## Stage 5: automated checks

GitHub Actions runs the data, model, tracking, and API tests on every push and
pull request using Python 3.13. Run the same tests locally with
`python -m unittest discover -s tests -v`. Docker image validation is a
separate next step.

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
command works without MLflow; use `--track` to send its runs to the server.
It splits the data 80/20 (`random_state=42`) and selects among a mean
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

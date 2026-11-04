# Comparing MLflow runs

Start the server and run `python -m src.models.compare --track` as described in
the [README](../README.md). Open `http://127.0.0.1:5000` and select the
`student-performance-early-prediction` experiment. One execution creates
three runs: `mean-baseline`, `ridge`, and `random-forest`.

## How to compare

1. Compare runs with the same `dataset_sha256` tag, `random_state_split=42`,
   `test_size=0.2`, and `cv_folds=5`. These values identify the data and split
   used for a fair comparison. Runs from a different dataset or evaluation
   setup should be reviewed separately.
2. Sort by `cv_mae` ascending. This metric is calculated only from training
   folds and is the model selection criterion. `cv_mae_std` shows how much the
   five fold scores varied.
3. Open the run with the lowest `cv_mae`. Its `selected_by_cv=true` tag and the
   run ID printed by the command identify the selected run. Inspect its
   `model/student_grade_pipeline.joblib` artifact. Every candidate run has a
   fitted pipeline artifact, but only the selected run has test metrics.
4. Read `test_mae`, `test_rmse`, and `test_r2` on the selected run as a final
   held-out evaluation. Do not change the winner based on these test metrics;
   doing so would use the test set for model selection.

For the current dataset, Ridge has the lowest CV MAE (3.243). Its test MAE is
3.764 and test R2 is -0.033. These results do not justify deploying it for
student-level decisions. Repeated executions with the same inputs create new
run IDs; use the run ID and dataset hash to trace a particular artifact.

The `.joblib` file is a **run artifact**. This project does not yet register
models in the MLflow Model Registry, assign model aliases, or promote a model
to production. The file in `artifacts/models/` is the local copy created by
the same comparison command.

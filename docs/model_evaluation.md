# Baseline model evaluation

Run `python -m src.models.compare` from the project root to reproduce these
numbers. The script uses the UCI mathematics file (395 rows), the 12 inputs
specified in [`prediction_spec.md`](prediction_spec.md), and `G3` as target.
It splits 80/20 with `random_state=42`: 316 training and 79 held-out rows.

The three candidate pipelines use the same preprocessing. Five shuffled folds
(`random_state=42`) on training rows select the model with lowest mean MAE:

| Model | CV MAE (mean +/- standard deviation) |
| --- | ---: |
| Mean prediction | 3.403 +/- 0.629 |
| Ridge (alpha=1) | **3.243 +/- 0.512** |
| Random Forest (200 trees, min leaf 3) | 3.444 +/- 0.416 |

Ridge was refit on all 316 training rows. On the 79 held-out rows it achieved
MAE **3.764**, RMSE **4.603**, and R2 **-0.033**. MAE and RMSE are in grade
points on the 0-20 scale. The negative R2 means this test result is worse than
predicting the test-set mean under that metric; it is not evidence of a useful
student-level prediction system.

The dataset is small, and this is one train/test split. The early prediction
feature choice excludes previous grades and full-year absences, so it answers a
harder question than a late-course model. CV selection did not inspect test
labels, but the test result still has sampling uncertainty. Do not tune models
against this test set repeatedly. A future iteration should use a larger,
representative dataset and validate inputs at the intended prediction time.

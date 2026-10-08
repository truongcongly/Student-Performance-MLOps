# Prediction definition

The baseline task is **regression**: estimate a student's final mathematics
grade `G3` on the 0-20 scale near the start of the course. This is a learning
exercise, not a validated decision system for students or schools.

## Input features

Use these 12 columns from `student-mat.csv`:

- Numeric or ordered: `Medu`, `Fedu`, `traveltime`, `studytime`, `failures`.
- Categorical (`yes`/`no`): `schoolsup`, `famsup`, `paid`, `activities`,
  `nursery`, `higher`, `internet`.

The baseline assumes these values are recorded before the prediction is made.
In a real deployment, verify this for each field and remove any field that is
only collected later. Do not use `G1`, `G2`, or the full-year `absences` count:
these are unavailable at the chosen prediction time. Other demographic and
personal fields are excluded from this first baseline to keep inputs limited.

## Initial data findings

The mathematics file has 395 rows and 33 columns, with no missing values or
duplicate rows. `G3` ranges from 0 to 20; its median is 11, and 38 students
have `G3 = 0`. `G1` and `G2` correlate strongly with `G3`, but their availability
depends on when prediction happens. These observations come from
[`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb); correlation does not
establish causation or model quality.

Training and held-out evaluation now follow this feature contract; see
[`model_evaluation.md`](model_evaluation.md) for the current results.

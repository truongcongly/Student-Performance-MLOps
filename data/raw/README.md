# Raw dataset

Source: [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student%2Bperformance),
Paulo Cortez, DOI: [10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T).
License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Run from the project root:

```powershell
python src/data/download.py
```

This downloads the original `student-mat.csv` (mathematics course) to this
directory. It has 395 student records. The CSV uses a semicolon (`;`) separator:

```python
import pandas as pd

students = pd.read_csv("data/raw/student-mat.csv", sep=";")
```

`G3` is the final grade (0-20) and the planned regression target. `G1` and
`G2` are earlier grades; including them makes prediction easier but changes
the real-world question to predicting near the end of the course. Decide
which features are available at prediction time before training a model.

Do not combine the mathematics and Portuguese files as if their rows were
independent students: some students appear in both.

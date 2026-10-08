"""Download and validate the UCI mathematics student performance dataset."""

import csv
import io
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


DATASET_URL = "https://archive.ics.uci.edu/static/public/320/student%2Bperformance.zip"
OUTPUT = Path(__file__).resolve().parents[2] / "data" / "raw" / "student-mat.csv"
REQUIRED_COLUMNS = {"school", "age", "studytime", "absences", "G1", "G2", "G3"}


def find_csv(archive_bytes: bytes) -> bytes:
    with ZipFile(io.BytesIO(archive_bytes)) as archive:
        for name in archive.namelist():
            if Path(name).name == "student-mat.csv":
                return archive.read(name)

        for name in archive.namelist():
            if name.lower().endswith(".zip"):
                try:
                    return find_csv(archive.read(name))
                except FileNotFoundError:
                    continue

    raise FileNotFoundError("student-mat.csv was not found in the UCI archive")


def main() -> None:
    with urlopen(DATASET_URL, timeout=30) as response:
        archive_bytes = response.read()

    data = find_csv(archive_bytes)

    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8")), delimiter=";"))
    if len(rows) != 395 or not REQUIRED_COLUMNS.issubset(rows[0]):
        raise ValueError("Unexpected UCI student-mat.csv schema or row count")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(data)
    print(f"Saved {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()

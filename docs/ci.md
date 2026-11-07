# Continuous integration checks

The `Python tests` GitHub Actions workflow runs on pushes and pull requests.
It has two jobs on Ubuntu:

1. `test` installs the API and test dependencies on Python 3.13, then runs
   `python -m unittest discover -s tests -v` for data, model, tracking, and API
   checks.
2. `docker` runs only after `test` passes. It builds the inference image,
   starts a container, checks `GET /health`, and sends the example JSON to
   `POST /predict`. It removes the container afterward.

Before merging a change, both jobs should pass. If `test` fails, inspect the
failing test and reproduce it locally. If `docker` fails, inspect the build
output or container logs shown by the workflow. Changes to the saved model or
dependencies need both checks because the API must still load the artifact.

Run the checks locally when Docker is available:

```cmd
python -m unittest discover -s tests -v
scripts\run-docker.cmd
curl.exe http://127.0.0.1:8001/health
docker stop student-performance-api
```

The workflow validates an image and API behavior. It does not publish an image,
deploy a service, or guarantee model accuracy. The held-out evaluation and its
limits are recorded in [`model_evaluation.md`](model_evaluation.md).

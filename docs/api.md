# Prediction API

Start the service from the project root:

```cmd
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

The saved pipeline must be present at
`artifacts/models/student_grade_pipeline.joblib`. The service loads it during
startup and will fail to start if it is missing or unreadable.

In a second Git CMD window, check health and request a prediction:

```cmd
curl.exe http://127.0.0.1:8000/health
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary "@docs/examples/student_request.json"
```

`GET /health` returns `{"status":"ok"}`. `POST /predict` returns a JSON
object such as `{"predicted_grade":11.1}`; the exact number depends on the
saved pipeline. The input consists of exactly the 12 fields in
[`student_request.json`](examples/student_request.json). Numeric fields must
be integers within the ranges shown at `http://127.0.0.1:8000/docs`; support
fields accept only `"yes"` or `"no"`.

Missing fields, extra fields, incorrect types, and values outside the allowed
ranges return HTTP 422 with validation details. The returned value is a model
estimate, not an observed grade or a decision about an individual student.

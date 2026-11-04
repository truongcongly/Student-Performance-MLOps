"""HTTP inference API for the saved student grade pipeline."""

from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from src.data.load import FEATURES
from src.models.compare import MODEL_PATH


YesNo = Literal["yes", "no"]


class StudentFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    Medu: int = Field(ge=0, le=4)
    Fedu: int = Field(ge=0, le=4)
    traveltime: int = Field(ge=1, le=4)
    studytime: int = Field(ge=1, le=4)
    failures: int = Field(ge=0, le=4)
    schoolsup: YesNo
    famsup: YesNo
    paid: YesNo
    activities: YesNo
    nursery: YesNo
    higher: YesNo
    internet: YesNo


class Prediction(BaseModel):
    predicted_grade: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(MODEL_PATH)
    yield
    del app.state.model


app = FastAPI(title="Student Performance Prediction API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(student: StudentFeatures, request: Request) -> Prediction:
    frame = pd.DataFrame([student.model_dump()], columns=list(FEATURES))
    grade = float(request.app.state.model.predict(frame)[0])
    return Prediction(predicted_grade=grade)

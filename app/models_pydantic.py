from pydantic import BaseModel, Field
from typing import Optional, List

class StudentInput(BaseModel):
    hours_study: float = Field(..., ge=0, le=24)
    attendance: float = Field(..., ge=0, le=100)
    sleep_hours: float = Field(..., ge=0, le=24)
    internet_hours: float = Field(..., ge=0, le=24)
    past_score: int = Field(..., ge=0, le=100)
    gender: str
    parent_education: str

class PredictionOut(BaseModel):
    prediction: str
    probabilities: Optional[List[float]] = None

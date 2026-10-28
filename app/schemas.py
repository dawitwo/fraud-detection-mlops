# app/schemas.py
from pydantic import BaseModel, Field
from typing import List

class Transaction(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="30 features: V1-V28, hour_of_day, scaled_amount",
    )

class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float
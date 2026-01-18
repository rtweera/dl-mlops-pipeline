"""
Pydantic schemas for API request and response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class OccupancyInput(BaseModel):
    """Input schema for occupancy prediction request."""
    
    datetime: str = Field(
        ...,
        description="Timestamp in format 'YYYY-MM-DD HH:MM:SS'",
        example="2015-02-04 17:51:00"
    )
    Temperature: float = Field(
        ...,
        description="Temperature in Celsius",
        example=23.18
    )
    Humidity: float = Field(
        ...,
        description="Relative humidity percentage",
        example=27.272
    )
    Light: float = Field(
        ...,
        description="Light intensity in Lux",
        example=426.0
    )
    CO2: float = Field(
        ...,
        description="CO2 concentration in ppm",
        example=721.25
    )
    HumidityRatio: float = Field(
        ...,
        description="Humidity ratio",
        example=0.00479
    )

    class Config:
        json_schema_extra = {
            "example": {
                "datetime": "2015-02-04 17:51:00",
                "Temperature": 23.18,
                "Humidity": 27.272,
                "Light": 426.0,
                "CO2": 721.25,
                "HumidityRatio": 0.00479
            }
        }


class OccupancyPrediction(BaseModel):
    """Output schema for occupancy prediction response."""
    
    prediction: int = Field(
        ...,
        description="Predicted occupancy (0: Not occupied, 1: Occupied)"
    )
    probability: Optional[float] = Field(
        None,
        description="Prediction probability/confidence",
        ge=0.0,
        le=1.0
    )
    timestamp: str = Field(
        ...,
        description="Server timestamp when prediction was made"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.95,
                "timestamp": "2026-01-19T12:00:00"
            }
        }


class HealthResponse(BaseModel):
    """Health check response schema."""
    
    status: str
    model_loaded: bool
    timestamp: str

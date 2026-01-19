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
    
    prediction: str = Field(
        ...,
        description="Predicted occupancy label (Person present / Person not present)"
    )
    probability: Optional[float] = Field(
        None,
        description="Prediction probability/confidence",
        ge=0.0,
        le=1.0
    )
    handling_time_ms: float = Field(
        ...,
        description="End-to-end request handling time in milliseconds",
        ge=0.0
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "Person present",
                "probability": 0.95,
                "handling_time_ms": 12.4
            }
        }


class HealthResponse(BaseModel):
    """Health check response schema."""
    
    status: str
    model_loaded: bool
    timestamp: str

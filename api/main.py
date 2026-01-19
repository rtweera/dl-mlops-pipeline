"""
FastAPI application for Occupancy Prediction Service.
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import logging
import time
from contextlib import asynccontextmanager

from schemas import OccupancyInput, OccupancyPrediction, HealthResponse
from model import get_predictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup: Load the model
    logger.info("Starting up the application...")
    try:
        predictor = get_predictor()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down the application...")


# Initialize FastAPI app
app = FastAPI(
    title="Occupancy Prediction API",
    description="Microservice for predicting room occupancy based on environmental sensors",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Occupancy Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs",
            "model_info": "/model/info"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Returns the status of the service and model.
    """
    try:
        predictor = get_predictor()
        model_loaded = predictor.is_loaded()
        
        return HealthResponse(
            status="healthy" if model_loaded else "unhealthy",
            model_loaded=model_loaded,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable"
        )


@app.post("/predict", response_model=OccupancyPrediction, tags=["Prediction"])
async def predict_occupancy(input_data: OccupancyInput):
    """
    Predict room occupancy based on sensor data.
    
    Args:
        input_data: Sensor readings including temperature, humidity, light, CO2, etc.
        
    Returns:
        Prediction result with occupancy status and confidence
    """
    try:
        start = time.perf_counter()
        # Get predictor instance
        predictor = get_predictor()
        
        # Convert Pydantic model to dict
        data_dict = input_data.model_dump()
        
        # Validate input
        predictor.validate_input(data_dict)
        
        # Make prediction
        prediction, probability = predictor.predict(data_dict)

        # Map numeric prediction to human-readable label
        label = "Person present" if prediction == 1 else "Person not present"
        
        logger.info(f"Prediction made: {label} (raw={prediction}) with probability {probability}")
        
        handling_time_ms = (time.perf_counter() - start) * 1000.0

        return OccupancyPrediction(
            prediction=label,
            probability=probability,
            handling_time_ms=handling_time_ms
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during prediction"
        )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """
    Get information about the loaded model.
    """
    try:
        predictor = get_predictor()
        
        if not predictor.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        return {
            "model_type": "LightGBM Classifier Pipeline",
            "model_loaded": True,
            "preprocessing_steps": [
                "CO2 BoxCox Transformation",
                "Light Discretization (KBins)",
                "Feature Engineering (time features, deltas, rates)",
                "SMOTE Oversampling"
            ],
            "model_path": str(predictor.model_path)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving model information"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

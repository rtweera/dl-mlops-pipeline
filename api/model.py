"""
Model loading and inference module.
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import logging

# Import transformers BEFORE loading the model so joblib can find the classes; else class not found error
from transformers import TransformCO2, DiscretizeLight, FeatureEngineer

logger = logging.getLogger(__name__)


class OccupancyPredictor:
    """
    Wrapper class for the occupancy prediction model.
    Handles model loading and inference.
    """
    
    def __init__(self, model_path: str = "./models/occupancy_model_pipeline.joblib"):
        """
        Initialize the predictor with the model path.
        
        Args:
            model_path: Path to the saved model pipeline
        """
        self.model_path = Path(model_path)
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained model pipeline from disk."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
            self.model = joblib.load(self.model_path)
            logger.info(f"Model successfully loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def predict(self, input_data: dict) -> Tuple[int, float]:
        """
        Make a prediction on the input data.
        
        Args:
            input_data: Dictionary containing the input features
            
        Returns:
            Tuple of (prediction, probability)
        """
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded")
        
        try:
            # Convert input to DataFrame
            # The model expects columns: Temperature, Humidity, Light, CO2, HumidityRatio, datetime
            df = pd.DataFrame([input_data])
            
            # Convert datetime string to datetime object
            df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S")
            
            # Make prediction
            prediction = self.model.predict(df)[0]
            
            # Get prediction probabilities
            try:
                probabilities = self.model.predict_proba(df)[0]
                probability = float(probabilities[prediction])
            except AttributeError:
                # If model doesn't support predict_proba
                probability = None
            
            return int(prediction), probability
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def validate_input(self, input_data: dict) -> bool:
        """
        Validate that input data contains all required features.
        
        Args:
            input_data: Dictionary containing input features
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        required_fields = ['datetime', 'Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
        missing_fields = [field for field in required_fields if field not in input_data]
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        return True


# Global predictor instance
predictor = None


def get_predictor() -> OccupancyPredictor:
    """
    Get or create the global predictor instance.
    This ensures we only load the model once. Singleton pattern.
    """
    global predictor
    if predictor is None:
        predictor = OccupancyPredictor()
    return predictor

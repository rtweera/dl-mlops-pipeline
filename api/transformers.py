"""
Custom transformer classes for the ML pipeline.
These must be defined here so joblib can find them when unpickling the saved model.
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import boxcox


def transform_co2(data, *, trained_lambda=None):
    """Transform CO2 using BoxCox to fit the best Normal curve possible."""
    lambda_ = None
    if trained_lambda is None:
        data["CO2"], lambda_ = boxcox(data["CO2"])
    else:
        data["CO2"] = boxcox(data["CO2"], lmbda=trained_lambda)
    return (data, lambda_)


def engineer_features(data):
    """Adds time series related features like delta, rates to dataframe."""
    # Time features
    data['hour'] = data['datetime'].dt.hour
    data['day_of_week'] = data['datetime'].dt.dayofweek
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)

    # Delta (i.e. Lag) features
    data["co2_delta"] = data["CO2"] - data["CO2"].shift(1)
    data["light_delta"] = data["Light"] - data["Light"].shift(1)
    data["hr_delta"] = data["HumidityRatio"] - data["HumidityRatio"].shift(1)
    data["temp_delta"] = data["Temperature"] - data["Temperature"].shift(1)

    # Rate features
    data["co2_rate"] = data["co2_delta"] / (data['datetime'].diff(periods=1).dt.total_seconds()/60)
    data["light_rate"] = data["light_delta"] / (data['datetime'].diff(periods=1).dt.total_seconds()/60)
    data["hr_rate"] = data["hr_delta"] / (data['datetime'].diff(periods=1).dt.total_seconds()/60)
    data["temp_rate"] = data["temp_delta"] / (data['datetime'].diff(periods=1).dt.total_seconds()/60)

    data.drop(columns=["datetime"], inplace=True)
    data.fillna(0, inplace=True)

    return data


class TransformCO2(BaseEstimator, TransformerMixin):
    """Custom transformer for CO2 BoxCox transformation."""
    
    def __init__(self):
        self.lambda_ = None

    def fit(self, X, y=None):
        X_copy = X.copy()
        X_copy, self.lambda_ = transform_co2(X_copy)
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy, _ = transform_co2(X_copy, trained_lambda=self.lambda_)
        return X_copy


class DiscretizeLight(BaseEstimator, TransformerMixin):
    """Custom transformer for Light discretization."""
    
    def __init__(self, discretizer):
        self.discretizer = discretizer

    def fit(self, X, y=None):
        X_copy = X.copy()
        self.discretizer.fit(X_copy["Light"].to_numpy().reshape(-1, 1))
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy["Light"] = self.discretizer.transform(X_copy["Light"].to_numpy().reshape(-1, 1))
        return X_copy


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for feature engineering."""
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        return engineer_features(X_copy)

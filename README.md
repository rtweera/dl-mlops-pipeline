# Occupancy Prediction Microservice

A containerized microservice for predicting room occupancy based on environmental sensor data using machine learning.

## Architecture

This project implements a microservices architecture with:

- **FastAPI Service**: REST API for occupancy predictions
- **LightGBM Model**: Trained ML pipeline for inference
- **Docker**: Containerized deployment

## Project Structure

```plaintext
dl-mlops-pipeline/
├── api/
│   ├── main.py              # FastAPI application
│   ├── model.py             # Model loading and inference
│   ├── schemas.py           # Pydantic models
│   ├── transformers.py      # Custom data transformers
│   ├── requirements.txt     # Python dependencies
│   ├── Dockerfile           # API service container
│   └── .dockerignore
├── models/
│   └── occupancy_model_pipeline.joblib  # Trained model (generated from notebook)
├── notebooks/
│   └── model-training.ipynb
├── data/
│   ├── Dataset-Link.md     # Dataset information
│   ├── dataset.txt
│   ├── dataset2.txt         
│   └── datatraining.txt
├── poetry.lock              # Poetry lock file
├── pyproject.toml           # Poetry project file
├── README.md                # Project documentation
└── docker-compose.yml       # Service orchestration
```

## Getting Started

### Prerequisites

- Docker and Docker Compose installed
- Poetry installed (for running the notebook dependencies)

### Step 1: Train and Save the Model

Run the Jupyter notebook to train the model and save it:

```plaintext
In traininig notebook, run up to the "Model Persistence" cells to save the model to models/
```

### Step 2: Build and Run with Docker Compose

```bash
# Build and start the service
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

The API will be available at `http://localhost:8000`

### Step 3: Access the API

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Model Info**: http://localhost:8000/model/info

## API Endpoints

### POST /predict

Predict room occupancy based on sensor readings.

**Request Body:**

```json
{
  "datetime": "2015-02-04 17:51:00",
  "Temperature": 23.18,
  "Humidity": 27.272,
  "Light": 426.0,
  "CO2": 721.25,
  "HumidityRatio": 0.00479
}
```

**Response:**

```json
{
  "prediction": 1,
  "probability": 0.95,
  "timestamp": "2026-01-19T12:00:00"
}
```

### GET /health

Check service health status.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-01-19T12:00:00"
}
```

## Testing the API

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "datetime": "2015-02-04 17:51:00",
    "Temperature": 23.18,
    "Humidity": 27.272,
    "Light": 426.0,
    "CO2": 721.25,
    "HumidityRatio": 0.00479
  }'
```

## Model Information

The trained model pipeline includes:

1. **CO2 BoxCox Transformation**: Normalizes CO2 distribution
2. **Light Discretization**: KBins discretization with 2 bins
3. **Feature Engineering**:
   - Time-based features (hour, day of week, cyclical encoding)
   - Delta features (lag-1 differences)
   - Rate features (change per minute)
4. **SMOTE Oversampling**: Handles class imbalance
5. **LightGBM Classifier**: Final prediction model

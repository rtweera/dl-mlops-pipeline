# Occupancy Prediction Microservice 🏢

A containerized microservice for predicting room occupancy based on environmental sensor data using machine learning.

## 🏗️ Architecture

This project implements a microservices architecture with:
- **FastAPI Service**: REST API for occupancy predictions
- **LightGBM Model**: Trained ML pipeline for inference
- **Docker**: Containerized deployment

## 📁 Project Structure

```
dl-mlops-pipeline/
├── api/
│   ├── main.py              # FastAPI application
│   ├── model.py             # Model loading and inference
│   ├── schemas.py           # Pydantic models
│   ├── requirements.txt     # Python dependencies
│   ├── Dockerfile           # API service container
│   └── .dockerignore
├── models/
│   └── occupancy_model_pipeline.joblib  # Trained model (generated from notebook)
├── notebooks/
│   └── 215565L_ML_project_notebook.ipynb
├── data/
└── docker-compose.yml       # Service orchestration
```

## 🚀 Getting Started

### Prerequisites

- Docker and Docker Compose installed
- Trained model file (`occupancy_model_pipeline.joblib`) in the `models/` directory

### Step 1: Train and Save the Model

Run the Jupyter notebook to train the model and save it:

```bash
# In your notebook, run the "Model Persistence" cells to save the model to models/
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

## 📡 API Endpoints

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

## 🧪 Testing the API

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

### Using Python

```python
import requests

# API endpoint
url = "http://localhost:8000/predict"

# Sample data
data = {
    "datetime": "2015-02-04 17:51:00",
    "Temperature": 23.18,
    "Humidity": 27.272,
    "Light": 426.0,
    "CO2": 721.25,
    "HumidityRatio": 0.00479
}

# Make request
response = requests.post(url, json=data)
print(response.json())
```

## 🛠️ Development

### Run Locally Without Docker

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f occupancy-api
```

### Stop Services

```bash
docker-compose down

# Remove volumes
docker-compose down -v
```

## 📊 Model Information

The trained model pipeline includes:
1. **CO2 BoxCox Transformation**: Normalizes CO2 distribution
2. **Light Discretization**: KBins discretization with 2 bins
3. **Feature Engineering**: 
   - Time-based features (hour, day of week, cyclical encoding)
   - Delta features (lag-1 differences)
   - Rate features (change per minute)
4. **SMOTE Oversampling**: Handles class imbalance
5. **LightGBM Classifier**: Final prediction model

## 🔒 Production Considerations

For production deployment, consider:
- [ ] Add authentication/authorization
- [ ] Configure CORS for specific origins
- [ ] Add rate limiting
- [ ] Implement logging aggregation
- [ ] Add monitoring (Prometheus/Grafana)
- [ ] Use environment variables for configuration
- [ ] Add model versioning
- [ ] Implement A/B testing capabilities
- [ ] Add input validation and sanitization
- [ ] Configure HTTPS/SSL

## 📝 License

See LICENSE file for details.

## 👥 Author

ML OPs Pipeline Project - 215565L

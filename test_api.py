"""
Example script to test the Occupancy Prediction API.
"""
import requests
import json
from datetime import datetime


def test_health_check(base_url: str = "http://localhost:8000"):
    """Test the health check endpoint."""
    print("Testing Health Check...")
    response = requests.get(f"{base_url}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    return response.status_code == 200


def test_prediction(base_url: str = "http://localhost:8000"):
    """Test the prediction endpoint with sample data."""
    print("Testing Prediction Endpoint...")
    
    # Sample data from the dataset
    sample_data = {
        "datetime": "2015-02-04 17:51:00",
        "Temperature": 23.18,
        "Humidity": 27.272,
        "Light": 426.0,
        "CO2": 721.25,
        "HumidityRatio": 0.00479
    }
    
    print(f"Input Data: {json.dumps(sample_data, indent=2)}")
    
    response = requests.post(f"{base_url}/predict", json=sample_data)
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {'Occupied' if result['prediction'] == 1 else 'Not Occupied'}")
        print(f"Probability: {result['probability']:.2%}")
        print(f"Response: {json.dumps(result, indent=2)}\n")
        return True
    else:
        print(f"Error: {response.text}\n")
        return False


def test_model_info(base_url: str = "http://localhost:8000"):
    """Test the model info endpoint."""
    print("Testing Model Info Endpoint...")
    response = requests.get(f"{base_url}/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    return response.status_code == 200


def test_multiple_predictions(base_url: str = "http://localhost:8000"):
    """Test multiple predictions with different scenarios."""
    print("Testing Multiple Predictions...")
    
    test_cases = [
        {
            "name": "Low occupancy scenario",
            "data": {
                "datetime": "2015-02-04 07:00:00",
                "Temperature": 20.5,
                "Humidity": 30.0,
                "Light": 0.0,
                "CO2": 400.0,
                "HumidityRatio": 0.004
            }
        },
        {
            "name": "High occupancy scenario",
            "data": {
                "datetime": "2015-02-04 14:00:00",
                "Temperature": 23.5,
                "Humidity": 28.0,
                "Light": 500.0,
                "CO2": 1200.0,
                "HumidityRatio": 0.005
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        response = requests.post(f"{base_url}/predict", json=test_case['data'])
        if response.status_code == 200:
            result = response.json()
            print(f"  Prediction: {'Occupied' if result['prediction'] == 1 else 'Not Occupied'}")
            print(f"  Probability: {result['probability']:.2%}")
        else:
            print(f"  Error: {response.status_code}")
    
    print()


if __name__ == "__main__":
    BASE_URL = "http://localhost:8000"
    
    print("=" * 60)
    print("Occupancy Prediction API - Test Suite")
    print("=" * 60)
    print()
    
    try:
        # Test health check
        if not test_health_check(BASE_URL):
            print("Health check failed. Is the service running?")
            exit(1)
        
        # Test model info
        test_model_info(BASE_URL)
        
        # Test prediction
        test_prediction(BASE_URL)
        
        # Test multiple predictions
        test_multiple_predictions(BASE_URL)
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to {BASE_URL}")
        print("Make sure the service is running with: docker-compose up")
    except Exception as e:
        print(f"Error: {str(e)}")

import os
import requests

BASE_URL = f"http://127.0.0.1:5001"

def test_predict(data):
    """Tests the /predict endpoint"""
    url = f"{BASE_URL}/predict"
    # Replace these keys/values with whatever your model expects
    
    response = requests.post(url, json=data)
    print(f"Predict Status: {response.status_code}")
    print(f"Prediction: {response.json()}")

def check_health():
    """Tests the /health endpoint"""
    url = f"{BASE_URL}/health"
    
    response = requests.get(url)
    print(f"Health Check Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response

def test_check_model():
    """Tests the /check_model endpoint"""
    url = f"{BASE_URL}/check_model"
    
    response = requests.get(url)
    print(f"Check Model Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response

def test_train_model():
    """Tests the /train endpoint"""
    url = f"{BASE_URL}/train"
    
    response = requests.post(url)
    print(f"Train Model Status: {response.status_code}")
    print(f"Response: {response.json()}")

if __name__ == "__main__":
    # First, check if the API is healthy
    # check_health()
    # response = test_check_model()
    # if('error' not in response.json()):
    sample_customer = {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }

    test_predict(sample_customer)

# else:
    # test_train_model()

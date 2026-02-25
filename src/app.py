"""FastAPI service for the Telco Customer Churn Prediction model.

This module provides REST endpoints to:
- Check if the API and a trained model are available
- Trigger model training (runs in background via Celery or similar)
- Make predictions given customer feature data

All endpoints interact with `ServeModel` to load/train the registered model
from the MLflow Model Registry.
"""

from fastapi import FastAPI, BackgroundTasks
from servings import ServeModel
import mlflow
from pydantic import BaseModel
from typing import Literal
from utils import Utils


# Initialize FastAPI application
app = FastAPI()
config = Utils().load_config()


@app.get("/health")
async def health():
    """Health check endpoint.
    
    Returns:
        dict: API status and brief usage instructions.
    """
    return {
        "api_status": "Working!",
        "message": (
            "Welcome to the Telco Customer Churn Prediction API. "
            "Use /train to train a model, /predict to get predictions and "
            "/check_model to check if a model is available for prediction"
        ),
    }


@app.get("/check_model")
async def check_model():
    """Check if a trained model is available for making predictions.
    
    Returns:
        dict: Model status (available or error message).
    """
    try:
        sm = ServeModel("ChurnPredictionModel")
    except Exception as e:
        return {
            "error": (
                "No model available to predict. "
                "Please train a model first (use /train api)."
            )
        }
    
    return {
        "model_status": "Model is available for prediction",
        "model_name": "ChurnPredictionModel"
        }


@app.post("/train")
async def train(BackgroundTasks: BackgroundTasks):
    """Trigger model training in the background.
    
    The training process is submitted as a background task and runs
    asynchronously without blocking the API response.
    
    Args:
        BackgroundTasks: FastAPI dependency for scheduling background work.
    
    Returns:
        dict: Training status and guidance on checking model availability.
    """
    sm = ServeModel("ChurnPredictionModel", train=True)

    # Schedule the training task to run in the background
    BackgroundTasks.add_task(sm.train_model)
    return {
        "model_status": "Model training started in background",
        "model_name": "ChurnPredictionModel",
        "message": (
            "Please check /check_model endpoint after some time "
            "to see if the model is ready for prediction"
        ),
    }


@app.post("/predict")
async def predict(input_data: dict):
    """Make a churn prediction given input customer features.
    
    Args:
        input_data (dict): Customer feature data to score with the model.
    
    Returns:
        dict: Input data and predicted churn status ('Yes' or 'No').
    """
    try:
        sm = ServeModel("ChurnPredictionModel")
        pred = sm(input_data)
    except Exception as e:
        try:
            loaded_model = mlflow.pyfunc.load_model(config["local_model_path"])
            pred = loaded_model.predict(input_data)
        except:
            return {
                "error": (
                    "No model available to predict. "
                    "Please train a model first (use /train api)."
                )
            }
    
    # Generate predictions using the served model
    print(pred)
    
    # Convert numeric predictions (0/1) to human-readable strings
    predictions = []
    for i in pred:
        if(i == 1):
            predictions.append("Yes")
        else:        
            predictions.append("No")
    
    return {
        "input_data": input_data,
        "prediction": predictions
        }

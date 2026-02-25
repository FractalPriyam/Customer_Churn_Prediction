"""Model serving utilities for the Telco churn prediction pipeline.

This module provides `ServeModel`, a convenience wrapper that:
- Loads production models from the MLflow Model Registry
- Handles inference by orchestrating validation and preprocessing
- Provides model training capability for automated retraining workflows
"""

from train import ModelTrain
import mlflow.pyfunc
import pandas as pd
from mlflow.tracking import MlflowClient
from data_ingestion import DataIngestion


class ServeModel:
    """Wrapper for loading and serving churn prediction models from MLflow.

    Args:
        model_name (str): Name of the registered model in MLflow Model Registry
        train (bool): If True, skip loading a model (used for training mode)

    Raises:
        ValueError: If no production model version is found for the given name
    """

    def __init__(self, model_name:str, train:bool=False):
        self.model_name = model_name
        client = MlflowClient()

        # Only load a model if not in training mode
        if not train:
            # Retrieve the latest production-stage model version
            latest_versions = client.get_latest_versions(model_name, stages=["Production"])
            if not latest_versions:
                raise ValueError(f"No production version found for model: {model_name}")
            else:
                self.model_version = latest_versions[0].version
                self.model_uri = f"models:/{self.model_name}/{self.model_version}"
                # Load the model using MLflow's generic Python model interface
                self.model = mlflow.pyfunc.load_model(self.model_uri)
                print(f"Loading model from: {self.model_uri}")

    def predict_data(self, df: pd.DataFrame) -> list:
        """Validate, preprocess, and generate predictions for a DataFrame.

        Args:
            df (pd.DataFrame): Raw input data to validate and score

        Returns:
            list: Predictions (churn likelihood or class) as a list
        """
        di = DataIngestion()
        # Validate input data against expected schema
        di.validate_data(df)
        # Apply preprocessing steps (encoding, scaling, etc.) in inference mode
        df = di.preprocess_data(df, inference=True)

        # Run inference using the loaded model
        prediction = self.model.predict(df)

        # Convert numpy array to list for JSON serialization
        return prediction.tolist()

    def train_model(self)-> None:
        """Trigger a complete model training and registration workflow.

        This method orchestrates the full ML pipeline: data loading, model
        tuning, evaluation, and registration in the MLflow Model Registry.
        Intended to be called asynchronously (e.g., as a background task).
        """
        mt = ModelTrain()
        mt()


    def __call__(self, data: dict)-> list:
        """Make a prediction via calling the instance directly (callable interface).

        Args:
            data (dict): A single record of customer features

        Returns:
            list: Prediction result(s)
        """
        # Wrap the dictionary in a list and convert to DataFrame
        df = pd.DataFrame([data])

        # Delegate to predict_data for validation and inference
        prediction = self.predict_data(df)

        return prediction


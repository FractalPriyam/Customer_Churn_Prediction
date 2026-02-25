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
from utils import Utils


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
        self.config = Utils().load_config()

        # Only load a model if not in training mode
        if not train:
            # Retrieve the latest production-stage model version
            try:
                latest_versions = client.get_latest_versions(model_name, stages=["Production"])
                self.model_version = latest_versions[0].version
                self.model_uri = f"models:/{self.model_name}/{self.model_version}"
                # Load the model using MLflow's generic Python model interface
                self.model = mlflow.sklearn.load_model(self.model_uri)
                print(f"Loading model from: {self.model_uri}")
            except Exception as e:
                try:
                    self.model = mlflow.sklearn.load_model(self.config["local_model_path"])
                    print(f"Loading model from local path: {self.config['local_model_path']}")
                except Exception as e:
                    raise ValueError(f"Error loading model from MLflow or local path. Error: {e}")

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
        proba = self.model.predict_proba(df)[:, 1]  # Assuming binary classification, get probability of positive class

        # Convert numpy array to list for JSON serialization
        return prediction, proba

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
        prediction, proba = self.predict_data(df)

        return prediction, proba


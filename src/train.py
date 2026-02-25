"""Training orchestration for the Telco churn models.

This module contains `ModelTrain`, a lightweight orchestration class that:
- ingests data via `DataIngestion`
- splits data for training and evaluation
- uses Optuna to tune hyperparameters for multiple model candidates
- logs runs and artifacts to MLflow and promotes registered models
"""

import warnings

warnings.filterwarnings("ignore")

from data_ingestion import DataIngestion
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from utils import Utils
from datetime import datetime
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import optuna
import pandas as pd
from mlflow.tracking import MlflowClient
import os


class ModelTrain:
    """Orchestrates model training, tuning and model registry promotion.

    Attributes:
        model_name (str): MLflow registered model name used for promotion
        config (dict): loaded configuration used to control trials and models
        X_train, X_test, y_train, y_test: placeholders populated by `get_data`
        scale_pos_weight (float): placeholder for class imbalance handling
    """

    def __init__(self):
        # Load pipeline configuration (models, hyperparameter spaces, optuna)
        self.config = Utils().load_config()
        
        # Friendly name used when registering/promoting models in MLflow
        self.model_name = self.config["model_name"]

        # Ensure MLflow experiment exists and is selected
        mlflow.set_experiment("Telco_Customer_Churn_v1")
        # Optionally enable autologging for sklearn: mlflow.sklearn.autolog()

        # Initialize placeholders for data splits
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        # Placeholder used when computing XGBoost's scale_pos_weight dynamically
        self.scale_pos_weight = 0.0

    def get_data(self):
        """Ingests data and performs a train/test split.

        Returns the full dataframe plus split components assigned to the
        instance for use by training functions.
        """
        di = DataIngestion()
        df = di()

        # Separate features and target
        y = df[["Churn"]]
        X = df.drop("Churn", axis=1)

        # Split the data into training and test sets (80/20)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return df, self.X_train, self.X_test, self.y_train, self.y_test

    def get_model_instance(self, model_name:str, trial=None, best_params=None, base_scale_pos_weight=1.0):
        """Factory method that returns a model instance configured by params.

        If an Optuna `trial` is provided this method will sample hyperparameters
        from the configured search space. The method supports `xgboost`,
        `random_forest`, and `logistic_regression` as model choices.
        """
        params = best_params if best_params else self.config[model_name]["params"].copy()
        space = self.config[model_name]["search_space"]

        # When running under Optuna, suggest values for each hyperparameter
        if trial:
            if model_name == "xgboost":
                params["n_estimators"] = trial.suggest_int("n_estimators", *space["n_estimators"])
                params["max_depth"] = trial.suggest_int("max_depth", *space["max_depth"])
                params["learning_rate"] = trial.suggest_float(
                    "learning_rate", *space["learning_rate"], log=True
                )
                params["subsample"] = trial.suggest_float("subsample", *space["subsample"])
                params["colsample_bytree"] = trial.suggest_float(
                    "colsample_bytree", *space["colsample_bytree"]
                )
                params["gamma"] = trial.suggest_float("gamma", *space["gamma"])

                # Dynamic handling of XGBoost's scale_pos_weight. The config
                # provides multipliers which are combined with a base value.
                mult_min, mult_max = space["scale_pos_weight_multipliers"]
                multiplier = trial.suggest_float("scale_pos_weight_multiplier", mult_min, mult_max)
                params["scale_pos_weight"] = base_scale_pos_weight * multiplier

            elif model_name == "random_forest":
                params["n_estimators"] = trial.suggest_int("n_estimators", *space["n_estimators"])
                params["max_depth"] = trial.suggest_int("max_depth", *space["max_depth"])
                params["min_samples_split"] = trial.suggest_int(
                    "min_samples_split", *space["min_samples_split"]
                )
                params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", *space["min_samples_leaf"])
                params["max_features"] = trial.suggest_categorical("max_features", space["max_features"])
                params["class_weight"] = trial.suggest_categorical("class_weight", space["class_weight"])
                params["criterion"] = trial.suggest_categorical("criterion", space["criterion"])

            elif model_name == "logistic_regression":
                # Tune regularization strength C
                params["C"] = trial.suggest_float("C", *space["C"], log=True)

                # Choose penalty type and handle elasticnet-specific l1_ratio
                penalty_choice = trial.suggest_categorical("penalty", space["penalty"])
                params["penalty"] = penalty_choice

                if penalty_choice == "elasticnet":
                    params["l1_ratio"] = trial.suggest_float("l1_ratio", *space["l1_ratio"])
                else:
                    # Remove l1_ratio when not applicable to avoid estimator errors
                    params.pop("l1_ratio", None)

        # Instantiate the requested model using the assembled params
        if model_name == "xgboost":
            return xgb.XGBClassifier(**params)
        if model_name == "random_forest":
            return RandomForestClassifier(**params)
        if model_name == "logistic_regression":
            return LogisticRegression(**params)

        raise ValueError(f"Unknown model type: {model_name}")

    def calculate_metrics(self, y_true, y_pred, y_probs):
        """Compute a small set of evaluation metrics for binary classification."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            # Note: roc_auc expects probability estimates; ensure correct input
            "roc_auc": roc_auc_score(y_true, y_pred),
        }

    def objective(self, trial, model_name, base_scale_pos_weight):
        """Optuna objective that trains a trial-specific model and logs results.

        The objective returns the ROC AUC (higher is better) which Optuna
        will use to compare trials.
        """
        # 1. Build the model using suggested params from the trial
        model = self.get_model_instance(model_name, trial, base_scale_pos_weight=base_scale_pos_weight)

        with mlflow.start_run(nested=True, run_name=f"Trial_{model_name}_{trial.number}") as run:
            # Attach metadata to the Optuna trial for later reference
            trial.set_user_attr("model_name", model_name)
            trial.set_user_attr("mlflow_run_id", run.info.run_id)

            # 2. Fit the model on training data
            model.fit(self.X_train, self.y_train)

            # 3. Predict on the test set
            preds = model.predict(self.X_test)
            probs = model.predict_proba(self.X_test)[:, 1]

            # 4. Compute metrics and log to MLflow
            metrics = self.calculate_metrics(self.y_test, preds, probs)

            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

            # Optuna maximizes the returned value (roc_auc)
            return metrics["roc_auc"]

    def train(self):
        """Coordinate Optuna studies for each configured model.

        Iterates through the `models_to_evaluate` list in the config and
        runs Optuna optimizations, keeping track of the best-performing
        model and its parameters.
        """
        best_overall_score = -1
        best_overall_params = None
        best_overall_model_name = None

        # 1. Retrieve the list of models to run from configuration
        models_to_run = self.config["models_to_evaluate"]
        print(f"Running Models: {models_to_run}")

        for model_name in models_to_run:
            print(f"\Running optuna for: {model_name.upper()}")
            self.current_model_name = model_name

            study = optuna.create_study(direction=self.config["optuna"]["direction"]) 
            study.optimize(
                lambda trial: self.objective(trial, model_name, self.scale_pos_weight),
                n_trials=self.config["optuna"]["n_trials"],
            )

            print(f"Best {model_name} AUC: {study.best_value:.4f}")

            if study.best_value > best_overall_score:
                best_overall_score = study.best_value
                best_overall_params = study.best_params
                best_overall_model_name = model_name

        return study

    def promote_model(self, client, study, registered_model_details):
        """Promote a registered model version to Staging/Production.

        Compares the new trial's AUC against the current production model
        (if any) and updates model stages accordingly.
        """
        client = MlflowClient()

        best_run_id = study.best_trial.user_attrs.get("mlflow_run_id")
        model_uri = f"runs:/{best_run_id}/model"
        model_name = study.best_trial.user_attrs.get("model_name")
        new_auc = study.best_trial.value

        print("Promoting the model to staging...")
        client.transition_model_version_stage(
            name=self.model_name,
            version=registered_model_details.version,
            stage="Staging",
            archive_existing_versions=True,
        )

        latest_versions = []
        try:
            latest_versions = client.get_latest_versions(self.model_name, stages=["Production"])
        except Exception as e:
            print(f"Error fetching latest model versions: {e}")

        if len(latest_versions) > 0:
            production_version = latest_versions[0]
            # 2. Get metrics for that production run
            prod_run = client.get_run(production_version.run_id)
            current_prod_auc = prod_run.data.metrics.get("roc_auc", 0.0)

            print(f"Current Production AUC: {current_prod_auc:.4f}")
            print(f"New Best Trial AUC: {new_auc:.4f}")

            # 3. Conditional Registration: only move to production if improved
            if new_auc > current_prod_auc:
                print("New model is better! moving to Production...")
                model_uri = f"runs:/{best_run_id}/model"
                client.transition_model_version_stage(
                    name=self.model_name,
                    version=registered_model_details.version,
                    stage="Production",
                    archive_existing_versions=True,
                )
            else:
                print("Current Production model is still superior. Skipping registration.")
        else:
            # If no production model exists, promote the current one
            print("No production model found. Registering new model as production.")
            client.transition_model_version_stage(
                name=self.model_name,
                version=registered_model_details.version,
                stage="Production",
                archive_existing_versions=True,
            )

        latest_versions = client.get_latest_versions(self.model_name, stages=["Production"])
        
        if latest_versions:
            production_version = latest_versions[0]
            run_id = production_version.run_id
            version_num = production_version.version
            
            # 2. Construct the URI for this specific version
            model_uri = f"models:/{self.model_name}/{version_num}"
            
            # 3. Export (Download) to a local folder for your Docker build
            export_path = "./exported_model"
            if not os.path.exists(export_path):
                os.makedirs(export_path)
                
            mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=export_path)
            print(f"Model version {version_num} exported to {export_path}")
        else:
            print("No version found in 'Production' stage!")

        return client

    def register_model(self, study):
        """Register the best trial's model in MLflow Model Registry."""
        client = MlflowClient()

        best_run_id = study.best_trial.user_attrs.get("mlflow_run_id")
        model_uri = f"runs:/{best_run_id}/model"
        model_name = self.model_name
        registered_model_details = mlflow.register_model(
            model_uri=f"runs:/{best_run_id}/model",
            name=model_name,
            tags={"source": "optuna_optimization", "metric": "roc_auc"},
        )
        print(f"Registered new model version: {registered_model_details.version} for {self.model_name}")
        return client, registered_model_details


    def __call__(self):
        # Note: `get_data` returns (df, X_train, X_test, y_train, y_test)
        df, X_train, y_train, X_test, y_test = self.get_data()
        study = self.train()

        print(f"\n=============================================")
        print(
            f"Best Model: {study.best_trial.user_attrs.get('model_name')} (AUC: {study.best_trial.value:.4f})"
        )
        print(f"=============================================")

        client, registered_model_details = self.register_model(study)
        self.promote_model(client, study, registered_model_details)


if __name__ == "__main__":
    mt = ModelTrain()
    mt()
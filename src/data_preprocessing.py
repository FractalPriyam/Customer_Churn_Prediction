"""Preprocessing utilities for the Telco Customer Churn dataset.

This module provides a small class `DataPreprocessing` that encapsulates
common column cleaning and encoding operations used before training or
inference. 
"""

import pandas as pd
import numpy as np
from utils import Utils


class DataPreprocessing:
    """Encapsulates preprocessing for training and inference.

    Args:
        df (pd.DataFrame): input dataframe to preprocess
        inference (bool): if True, skip transformations that require the
            target column (`Churn`) during inference
    """

    def __init__(self, df: pd.DataFrame, inference: bool=False):
        self.df = df
        self.inference = inference
        # Load project configuration if needed for future extensions
        self.config = Utils().load_config()

    def data_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply a sequence of cleaning and encoding steps to `df`.

        Operations performed (in order):
        - Map `Churn` to binary (unless `inference=True`).
        - Convert `TotalCharges` to numeric and impute missing values.
        - Drop `customerID` (identifier not useful for modeling).
        - Map binary yes/no fields to 1/0.
        - Map service availability fields to numeric flags.
        - Encode nominal categorical fields (`Contract`, `PaymentMethod`,
          `InternetService`) using category codes.

        The function mutates and returns the input `df` for convenience.
        """
        # If running inference, do not transform the target column
        if self.inference:
            pass
        else:
            # Convert target labels 'Yes'/'No' into binary 1/0
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

        # TotalCharges may contain empty strings; coerce to numeric and
        # replace non-numeric values with NaN, then impute with mean.
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

        # Drop unique identifier column; not used as a feature
        df.drop(columns=['customerID'], axis=1, inplace=True)

        # Simple binary mappings for boolean-like columns
        df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
        df["Partner"] = df["Partner"].map({"Yes": 1, "No": 0})
        df["Dependents"] = df["Dependents"].map({"Yes": 1, "No": 0})
        df["PhoneService"] = df["PhoneService"].map({"Yes": 1, "No": 0})
        df["PaperlessBilling"] = df["PaperlessBilling"].map({"Yes": 1, "No": 0})

        # Map multi-valued service fields. For entries like 'No internet service'
        # or 'No phone service' we map them to 0 (no feature) to keep numeric.
        df["OnlineSecurity"] = df["OnlineSecurity"].map(
            {'No': 0, 'Yes': 1, 'No internet service': 0}
        )
        df["OnlineBackup"] = df["OnlineBackup"].map(
            {'No': 0, 'Yes': 1, 'No internet service': 0}
        )
        df["TechSupport"] = df["TechSupport"].map(
            {'No': 0, 'Yes': 1, 'No internet service': 0}
        )
        df["StreamingTV"] = df["StreamingTV"].map(
            {'No': 0, 'Yes': 1, 'No internet service': 0}
        )
        df["StreamingMovies"] = df["StreamingMovies"].map(
            {'No': 0, 'Yes': 1, 'No internet service': 0}
        )
        df["DeviceProtection"] = df["DeviceProtection"].map(
            {'No': 0, 'Yes': 1, 'No internet service': 0}
        )
        df["MultipleLines"] = df["MultipleLines"].map(
            {'No': 0, 'Yes': 1, 'No phone service': 0}
        )

        # Encode nominal categories using pandas category codes. This yields
        # integer codes starting at 0; be cautious if ordering matters.
        df['Contract'] = df['Contract'].astype('category').cat.codes
        df['PaymentMethod'] = df['PaymentMethod'].astype('category').cat.codes
        df['InternetService'] = df['InternetService'].astype('category').cat.codes

        return df


    def __call__(self)-> pd.DataFrame:
        """Allow instances to be called to run preprocessing on their df."""
        df = self.data_preprocess(self.df)
        return df
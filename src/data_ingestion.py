import pandas as pd
from utils import Utils
from data_validation import ChurnValidation
from pydantic import ValidationError
from data_preprocessing import DataPreprocessing
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

'''
Data Ingestion pipeling. It works in 3 phases:
1. Read Raw Data: Reads the raw data from the specified location in config.yaml
2. Validate Data: Validates the data using pydantic models defined in data_validation.py
3. Preprocess Data: Preprocesses the data using the DataPreprocessing class defined in data_preprocessing.py
'''
class DataIngestion():
    def __init__(self):
        #Read config
        self.config = Utils().load_config()

    def read_raw_data(self) -> pd.DataFrame:
        # Read raw data from the specified location in config.yaml
        try:
            df = pd.read_csv(self.config["data"]["raw_data"])  
            print("Read raw data")
        except Exception as e:
            print(f"Error while reading raw data: {e}")
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> None:
        # Validate data using pydantic models defined in data_validation.py
        validated_data = []
        try:
            # Convert each row of the DataFrame to a dictionary and validate using ChurnValidation model
            validated_data = [ChurnValidation(**row) for row in df.to_dict(orient="records")]
            print("Data validated")
        except Exception as e:
            print(f"Error while validation data: {e}")
            raise ValidationError
        

    def preprocess_data(self, df: pd.DataFrame, inference: bool = False) -> pd.DataFrame:
        # Preprocess data using the DataPreprocessing class defined in data_preprocessing.py
        dp = DataPreprocessing(df, inference=inference)
        df = dp()
        print("Data preprocessed")
        return df
    
    def __call__(self):
        # Run the data ingestion pipeline

        # Read raw data
        df = self.read_raw_data()

        # Validate data
        self.validate_data(df) 

        # Preprocess data
        df = self.preprocess_data(df) 

        # Save the preprocessed data to the specified location in config.yaml
        df.to_csv(self.config["data"]["processed_data"])
        
        return df
from src.exception import CustomException
from src.logger import logging
import os
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

# Initialize logging
logger = logging.getLogger(__name__)

@dataclass
class PredictionConfig:
    model_path: str = os.path.join('artifacts', 'model.pkl')
    preprocessor_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class PredictionPipeline:
    def __init__(self):
        self.prediction_config = PredictionConfig()
    
    def load_model(self):
        """
        Load the trained model
        """
        try:
            logger.info(f"Loading model from {self.prediction_config.model_path}")
            with open(self.prediction_config.model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            raise CustomException(e, sys)
    
    def load_preprocessor(self):
        """
        Load the preprocessor
        """
        try:
            logger.info(f"Loading preprocessor from {self.prediction_config.preprocessor_path}")
            with open(self.prediction_config.preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            logger.info("Preprocessor loaded successfully")
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict(self, input_data: pd.DataFrame):
        """
        Make predictions on new data
        """
        logger.info("Starting prediction")
        
        try:
            # Load model and preprocessor
            model = self.load_model()
            # preprocessor = self.load_preprocessor()  # Uncomment if preprocessor was saved
            
            # Drop target column if present (trained on features only)
            if isinstance(input_data, pd.DataFrame) and 'target' in input_data.columns:
                logger.info("Dropping 'target' column from input data")
                input_data = input_data.drop(columns=['target'])

            # Make predictions
            logger.info("Making predictions")
            predictions = model.predict(input_data)

            
            logger.info(f"Predictions made successfully. Shape: {predictions.shape}")
            
            return predictions
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def batch_predict(self, file_path: str):
        """
        Make predictions on a batch of data from a CSV file
        """
        logger.info("Starting batch prediction")
        
        try:
            # Load input data
            logger.info(f"Loading data from {file_path}")
            input_data = pd.read_csv(file_path)
            
            logger.info(f"Input data shape: {input_data.shape}")
            
            # Make predictions
            predictions = self.predict(input_data)
            
            logger.info("Batch prediction completed successfully")
            
            return predictions
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Example usage
    logger.info("Prediction Pipeline")
    
    # Create sample data for testing
    from sklearn.datasets import make_regression
    
    X, _ = make_regression(n_samples=10, n_features=10, random_state=42)
    test_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    
    # Initialize pipeline
    prediction_pipeline = PredictionPipeline()
    
    # Run prediction (requires trained model)
    # predictions = prediction_pipeline.predict(test_df)
    
    logger.info("Prediction pipeline finished")

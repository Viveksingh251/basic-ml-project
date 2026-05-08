from src.exception import CustomException
from src.logger import logging
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Initialize logging
logger = logging.getLogger(__name__)

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')
    root_dir: str = os.path.join('artifacts')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self, dataframe: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """
        This method is responsible for ingesting data from a dataframe
        """
        logger.info("Entered the data ingestion method")
        try:
            # Create artifacts directory if not exists
            os.makedirs(self.ingestion_config.root_dir, exist_ok=True)

            
            logger.info("Reading the dataframe")
            df = dataframe
            
            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logger.info("Raw data saved successfully")
            
            # Train test split
            logger.info("Splitting data into train and test sets")
            train_set, test_set = train_test_split(df, test_size=test_size, random_state=random_state)
            
            # Save train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            
            logger.info("Data ingestion completed successfully")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a CSV file
        """
        try:
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            raise CustomException(e, sys)
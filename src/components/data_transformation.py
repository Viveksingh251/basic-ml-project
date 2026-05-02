from src.exception import CustomException
from src.logger import logging
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from dataclasses import dataclass

# Initialize logging
logger = logging.getLogger(__name__)

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    transformed_train_path: str = os.path.join('artifacts', 'train_transformed.csv')
    transformed_test_path: str = os.path.join('artifacts', 'test_transformed.csv')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='mean')
    
    def get_data_transformer_object(self):
        """
        Creates and returns a data transformer object for preprocessing
        """
        try:
            logger.info("Creating data transformer object")
            
            # Return pre-fitted scaler for transformation
            return self.scaler
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path: str, test_path: str, target_column: str = None):
        """
        This method is responsible for transforming data
        """
        logger.info("Entered the data transformation method")
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logger.info(f"Train data shape: {train_df.shape}")
            logger.info(f"Test data shape: {test_df.shape}")
            
            # Separate target column if specified
            if target_column:
                logger.info(f"Separating target column: {target_column}")
                target_train = train_df[target_column]
                target_test = test_df[target_column]
                train_df = train_df.drop(columns=[target_column])
                test_df = test_df.drop(columns=[target_column])
            
            # Identify numerical and categorical columns
            numerical_columns = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_columns = train_df.select_dtypes(include=['object']).columns.tolist()
            
            logger.info(f"Numerical columns: {numerical_columns}")
            logger.info(f"Categorical columns: {categorical_columns}")
            
            # Handle missing values in numerical columns
            if numerical_columns:
                logger.info("Imputing missing values in numerical columns")
                train_df[numerical_columns] = self.imputer.fit_transform(train_df[numerical_columns])
                test_df[numerical_columns] = self.imputer.transform(test_df[numerical_columns])
            
            # Handle categorical columns with label encoding
            if categorical_columns:
                logger.info("Encoding categorical columns")
                for col in categorical_columns:
                    self.label_encoder.fit(train_df[col])
                    train_df[col] = self.label_encoder.transform(train_df[col])
                    test_df[col] = self.label_encoder.transform(test_df[col])
            
            # Scale numerical columns
            if numerical_columns:
                logger.info("Scaling numerical columns")
                train_df[numerical_columns] = self.scaler.fit_transform(train_df[numerical_columns])
                test_df[numerical_columns] = self.scaler.transform(test_df[numerical_columns])
            
            # Save transformed data
            train_df.to_csv(self.transformation_config.transformed_train_path, index=False)
            test_df.to_csv(self.transformation_config.transformed_test_path, index=False)
            
            logger.info("Data transformation completed successfully")
            
            return (
                self.transformation_config.transformed_train_path,
                self.transformation_config.transformed_test_path,
                self.scaler,
                self.label_encoder
            )
        except Exception as e:
            raise CustomException(e, sys)

from src.exception import CustomException
from src.logger import logging
import os
import sys
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Initialize logging
logger = logging.getLogger(__name__)

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
    
    def run_pipeline(self, dataframe: pd.DataFrame, target_column: str, test_size: float = 0.2, model_type: str = 'regression'):
        """
        Run the complete training pipeline
        """
        logger.info("Starting training pipeline")
        
        try:
            # Step 1: Data Ingestion
            logger.info("Step 1: Data Ingestion")
            train_path, test_path = self.data_ingestion.initiate_data_ingestion(
                dataframe=dataframe,
                test_size=test_size,
                random_state=42
            )
            
            # Step 2: Data Transformation
            logger.info("Step 2: Data Transformation")
            transformed_train_path, transformed_test_path, scaler, label_encoder = self.data_transformation.initiate_data_transformation(
                train_path=train_path,
                test_path=test_path,
                target_column=target_column
            )
            
            # Step 3: Model Training
            logger.info("Step 3: Model Training")
            best_model = self.model_trainer.initiate_model_trainer(
                train_path=train_path,
                test_path=test_path,
                target_column=target_column,
                model_type=model_type,
            )

            
            logger.info("Training pipeline completed successfully")
            
            return best_model
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Example usage
    logger.info("Training Pipeline")
    
    # Create sample data for testing
    from sklearn.datasets import make_regression, make_classification
    
    # For regression
    X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['target'] = y
    
    # Initialize pipeline
    pipeline = TrainPipeline()
    
    # Run pipeline
    model = pipeline.run_pipeline(dataframe=df, target_column='target', test_size=0.2, model_type='regression')
    logger.info(f"Training pipeline finished. Best model: {type(model).__name__}")


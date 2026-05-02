from src.exception import CustomException
from src.logger import logging
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
from dataclasses import dataclass

# Initialize logging
logger = logging.getLogger(__name__)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
    model_report_path: str = os.path.join('artifacts', 'model_report.txt')

class ModelTrainer:
    def __init__(self):
        self.trainer_config = ModelTrainerConfig()
        self.models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'DecisionTree': DecisionTreeRegressor(),
            'RandomForest': RandomForestRegressor(),
            'GradientBoosting': GradientBoostingRegressor(),
        }
        self.classification_models = {
            'LogisticRegression': LogisticRegression(),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'RandomForestClassifier': RandomForestClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
        }
    
    def evaluate_models(self, X_train, X_test, y_train, y_test, model_type='regression'):
        """
        Evaluate multiple models and return the best one
        """
        try:
            logger.info("Starting model evaluation")
            
            models_to_use = self.models if model_type == 'regression' else self.classification_models
            
            report = {}
            for name, model in models_to_use.items():
                logger.info(f"Training {name}")
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                
                if model_type == 'regression':
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    report[name] = {
                        'MSE': mse,
                        'RMSE': rmse,
                        'MAE': mae,
                        'R2': r2
                    }
                    logger.info(f"{name} - MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R2: {r2}")
                else:
                    accuracy = model.score(X_test, y_test)
                    report[name] = {'Accuracy': accuracy}
                    logger.info(f"{name} - Accuracy: {accuracy}")
            
            return report
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_best_model(self, report, model_type='regression'):
        """
        Get the best model based on the evaluation report
        """
        try:
            logger.info("Finding best model")
            
            models_to_use = self.models if model_type == 'regression' else self.classification_models
            
            if model_type == 'regression':
                # For regression, higher R2 is better
                best_model_name = max(report, key=lambda x: report[x]['R2'])
            else:
                # For classification, higher accuracy is better
                best_model_name = max(report, key=lambda x: report[x]['Accuracy'])
            
            best_model = models_to_use[best_model_name]
            logger.info(f"Best model: {best_model_name}")
            
            return best_model, best_model_name
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_model_trainer(self, train_path: str, test_path: str, target_column: str, model_type: str = 'regression'):
        """
        This method is responsible for training the model
        """
        logger.info("Entered the model trainer method")
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logger.info(f"Train data shape: {train_df.shape}")
            logger.info(f"Test data shape: {test_df.shape}")
            
            # Separate target column
            y_train = train_df[target_column]
            y_test = test_df[target_column]
            X_train = train_df.drop(columns=[target_column])
            X_test = test_df.drop(columns=[target_column])
            
            logger.info(f"Features shape: {X_train.shape}, {X_test.shape}")
            logger.info(f"Target shape: {y_train.shape}, {y_test.shape}")
            
            # Evaluate models
            report = self.evaluate_models(X_train, X_test, y_train, y_test, model_type)
            
            # Get best model
            best_model, best_model_name = self.get_best_model(report, model_type)
            
            # Train best model on full training data
            logger.info(f"Training best model: {best_model_name}")
            best_model.fit(X_train, y_train)
            
            # Save the model
            os.makedirs(os.path.dirname(self.trainer_config.trained_model_file_path), exist_ok=True)
            with open(self.trainer_config.trained_model_file_path, 'wb') as f:
                pickle.dump(best_model, f)
            
            logger.info(f"Model saved successfully to {self.trainer_config.trained_model_file_path}")
            
            # Save model report
            with open(self.trainer_config.model_report_path, 'w') as f:
                f.write("Model Evaluation Report\n")
                f.write("=" * 50 + "\n\n")
                for model_name, metrics in report.items():
                    f.write(f"{model_name}:\n")
                    for metric, value in metrics.items():
                        f.write(f"  {metric}: {value}\n")
                    f.write("\n")
                f.write(f"\nBest Model: {best_model_name}\n")
            
            logger.info("Model training completed successfully")
            
            return best_model
        except Exception as e:
            raise CustomException(e, sys)

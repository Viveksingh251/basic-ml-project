import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import os
import sys
import pickle

# Initialize logging
logger = logging.getLogger(__name__)

def save_object(file_path: str, obj):
    """
    Save a Python object to a pickle file
    """
    try:
        logger.info(f"Saving object to {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        logger.info("Object saved successfully")
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path: str):
    """
    Load a Python object from a pickle file
    """
    try:
        logger.info(f"Loading object from {file_path}")
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        logger.info("Object loaded successfully")
        return obj
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, X_test, y_train, y_test, models: dict):
    """
    Evaluate multiple models and return their scores
    """
    try:
        logger.info("Starting model evaluation")
        
        report = {}
        for name, model in models.items():
            logger.info(f"Training {name}")
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
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
        
        return report
    except Exception as e:
        raise CustomException(e, sys)

def get_accuracy_score(y_true, y_pred):
    """
    Calculate accuracy score
    """
    try:
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_true, y_pred)
    except Exception as e:
        raise CustomException(e, sys)

def get_preprocessing_models():
    """
    Return dictionary of preprocessing models
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.impute import SimpleImputer
    
    return {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler(),
        'SimpleImputer': SimpleImputer(strategy='mean')
    }

def save_data(data: pd.DataFrame, file_path: str):
    """
    Save dataframe to CSV
    """
    try:
        logger.info(f"Saving data to {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data.to_csv(file_path, index=False)
        logger.info("Data saved successfully")
    except Exception as e:
        raise CustomException(e, sys)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataframe from CSV
    """
    try:
        logger.info(f"Loading data from {file_path}")
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {data.shape}")
        return data
    except Exception as e:
        raise CustomException(e, sys)

def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Drop specified columns from dataframe
    """
    try:
        logger.info(f"Dropping columns: {columns}")
        df = df.drop(columns=columns)
        logger.info(f"Columns dropped. New shape: {df.shape}")
        return df
    except Exception as e:
        raise CustomException(e, sys)

def get_feature_importance(model, feature_names):
    """
    Get feature importance from a trained model
    """
    try:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            return feature_importance
        else:
            logger.warning("Model does not have feature_importances_ attribute")
            return None
    except Exception as e:
        raise CustomException(e, sys)

# Import missing metrics for evaluate_models
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

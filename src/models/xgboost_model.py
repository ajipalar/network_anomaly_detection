"""
XGBoost model for network anomaly detection.
"""

import xgboost as xgb
import numpy as np
import joblib
import os
from typing import Dict, Optional


def create_xgboost_model(config: Dict) -> xgb.XGBClassifier:
    """
    Create XGBoost classifier from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized XGBClassifier
    """
    xgb_config = config.get('xgboost', {})
    
    model = xgb.XGBClassifier(
        n_estimators=xgb_config.get('n_estimators', 100),
        max_depth=xgb_config.get('max_depth', 6),
        learning_rate=xgb_config.get('learning_rate', 0.1),
        subsample=xgb_config.get('subsample', 1.0),
        colsample_bytree=xgb_config.get('colsample_bytree', 1.0),
        min_child_weight=xgb_config.get('min_child_weight', 1),
        gamma=xgb_config.get('gamma', 0),
        reg_alpha=xgb_config.get('reg_alpha', 0),
        reg_lambda=xgb_config.get('reg_lambda', 1),
        scale_pos_weight=xgb_config.get('scale_pos_weight', 1.0),
        random_state=config.get('random_state', 42),
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    return model


def save_xgboost_model(model: xgb.XGBClassifier, filepath: str, metadata: Optional[Dict] = None):
    """
    Save XGBoost model to file.
    
    Args:
        model: Trained XGBClassifier
        filepath: Path to save model
        metadata: Optional metadata to save alongside model
    """
    # Ensure directory exists
    dir_path = os.path.dirname(filepath)
    if dir_path:  # Only create directory if path has a directory component
        os.makedirs(dir_path, exist_ok=True)
    
    # Save model
    try:
        joblib.dump(model, filepath)
        print(f"XGBoost model saved successfully to: {filepath}")
    except Exception as e:
        print(f"Error saving XGBoost model to {filepath}: {e}")
        raise
    
    # Save metadata if provided
    if metadata:
        metadata_path = filepath.replace('.pkl', '_metadata.pkl')
        try:
            joblib.dump(metadata, metadata_path)
            print(f"Metadata saved to: {metadata_path}")
        except Exception as e:
            print(f"Warning: Could not save metadata to {metadata_path}: {e}")


def load_xgboost_model(filepath: str) -> tuple:
    """
    Load XGBoost model from file.
    
    Args:
        filepath: Path to saved model
        
    Returns:
        Tuple of (model, metadata) where metadata may be None
    """
    model = joblib.load(filepath)
    
    # Try to load metadata
    metadata_path = filepath.replace('.pkl', '_metadata.pkl')
    metadata = None
    if os.path.exists(metadata_path):
        metadata = joblib.load(metadata_path)
    
    return model, metadata


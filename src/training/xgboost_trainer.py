"""
XGBoost training utilities.
"""

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, Tuple, Optional
import os
from datetime import datetime

from ..models.xgboost_model import create_xgboost_model, save_xgboost_model


def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict,
    checkpoint_dir: str,
    model_name: Optional[str] = None
) -> Tuple[xgb.XGBClassifier, Dict]:
    """
    Train XGBoost model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Model configuration
        checkpoint_dir: Directory to save checkpoints
        model_name: Optional name for the model
        
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Create model
    model_config = config['model'].copy()
    model = create_xgboost_model(model_config)
    print(f"Created XGBoost model with config: {model_config.get('xgboost', {})}")
    
    # Train model
    print("Training XGBoost model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred, zero_division=0)
    val_recall = recall_score(y_val, y_val_pred, zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
    val_auc = roc_auc_score(y_val, y_val_proba)
    
    metrics = {
        'accuracy': val_accuracy,
        'precision': val_precision,
        'recall': val_recall,
        'f1': val_f1,
        'auc': val_auc
    }
    
    print(f"\nValidation Metrics:")
    print(f"  Accuracy:  {val_accuracy:.4f}")
    print(f"  Precision: {val_precision:.4f}")
    print(f"  Recall:    {val_recall:.4f}")
    print(f"  F1 Score:  {val_f1:.4f}")
    print(f"  AUC:       {val_auc:.4f}")
    
    # Save model
    if model_name is None:
        model_name = f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    best_model_path = os.path.join(checkpoint_dir, 'best_xgboost_model.pkl')
    # Normalize path to handle any relative/absolute path issues
    best_model_path = os.path.normpath(best_model_path)
    
    print(f"\nSaving model to: {best_model_path}")
    metadata = {
        'model_name': model_name,
        'metrics': metrics,
        'config': config
    }
    
    try:
        save_xgboost_model(model, best_model_path, metadata)
        
        # Verify file was created
        if os.path.exists(best_model_path):
            file_size = os.path.getsize(best_model_path)
            print(f"✓ Model file verified: {file_size} bytes")
        else:
            raise FileNotFoundError(f"Model file was not created at {best_model_path}")
    except Exception as e:
        print(f"✗ Error saving model: {e}")
        raise
    
    return model, metrics


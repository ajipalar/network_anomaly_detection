"""
K-fold cross validation utilities.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from typing import List, Tuple, Dict
import torch
from torch.utils.data import DataLoader

from ..data import NetworkAnomalyDataset, preprocess_data, create_dataloader
from ..models import create_model
from ..training import Trainer
from ..utils import get_device


def perform_kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    config: Dict,
    device: torch.device
) -> List[Dict]:
    """
    Perform K-fold cross validation.
    
    Args:
        X: Feature matrix
        y: Label vector
        config: Configuration dictionary
        device: PyTorch device
        
    Returns:
        List of fold results
    """
    cv_config = config['cross_validation']
    n_splits = cv_config['n_splits']
    shuffle = cv_config.get('shuffle', True)
    random_state = cv_config.get('random_state', 42)
    stratify = cv_config.get('stratify', True)
    
    # Choose CV strategy
    if stratify:
        kfold = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
        splits = kfold.split(X, y)
    else:
        kfold = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
        splits = kfold.split(X)
    
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx + 1}/{n_splits}")
        print(f"{'='*50}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create datasets
        train_dataset = NetworkAnomalyDataset(X_train, y_train)
        val_dataset = NetworkAnomalyDataset(X_val, y_val)
        
        # Create data loaders
        train_loader = create_dataloader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True
        )
        val_loader = create_dataloader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False
        )
        
        # Create model
        model_config = config['model'].copy()
        model_config['input_dim'] = X.shape[1]
        model = create_model(model_config)
        
        # Create trainer
        checkpoint_dir = f"{config['checkpoint']['save_dir']}/fold_{fold_idx + 1}"
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config['training'],
            device=device,
            checkpoint_dir=checkpoint_dir
        )
        
        # Train
        trainer.train(config['training']['num_epochs'])
        
        # Store results
        fold_results.append({
            'fold': fold_idx + 1,
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses,
            'best_val_loss': trainer.best_val_loss,
            'final_train_loss': trainer.train_losses[-1],
            'final_val_loss': trainer.val_losses[-1]
        })
    
    return fold_results


def print_cv_summary(fold_results: List[Dict]):
    """
    Print cross validation summary.
    
    Args:
        fold_results: List of fold results
    """
    print("\n" + "="*50)
    print("Cross Validation Summary")
    print("="*50)
    
    val_losses = [r['best_val_loss'] for r in fold_results]
    train_losses = [r['final_train_loss'] for r in fold_results]
    
    print(f"\nValidation Loss:")
    print(f"  Mean: {np.mean(val_losses):.4f}")
    print(f"  Std:  {np.std(val_losses):.4f}")
    print(f"  Min:  {np.min(val_losses):.4f}")
    print(f"  Max:  {np.max(val_losses):.4f}")
    
    print(f"\nTraining Loss:")
    print(f"  Mean: {np.mean(train_losses):.4f}")
    print(f"  Std:  {np.std(train_losses):.4f}")
    
    print("\nPer-fold results:")
    for result in fold_results:
        print(f"  Fold {result['fold']}: "
              f"Train={result['final_train_loss']:.4f}, "
              f"Val={result['final_val_loss']:.4f}")


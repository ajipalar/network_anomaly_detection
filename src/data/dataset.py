"""
Dataset loading and preprocessing utilities.
"""

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import torch


class NetworkAnomalyDataset(Dataset):
    """PyTorch Dataset for network anomaly detection."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Label vector (n_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def load_data(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load data from CSV file.
    
    Args:
        data_path: Path to CSV file
        
    Returns:
        Features DataFrame and labels Series
    """
    df = pd.read_csv(data_path)
    
    # Separate features and labels
    label_col = 'label'
    feature_cols = [col for col in df.columns if col != label_col]
    
    X = df[feature_cols]
    y = df[label_col]
    
    return X, y


def preprocess_data(
    X: pd.DataFrame,
    y: pd.Series,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = True
) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler]]:
    """
    Preprocess data: convert boolean columns and normalize.
    
    Args:
        X: Feature DataFrame
        y: Label Series
        scaler: Fitted scaler (for test/val sets)
        fit_scaler: Whether to fit the scaler
        
    Returns:
        Processed features, labels, and scaler
    """
    # Convert boolean columns to int
    X_processed = X.copy()
    bool_cols = X_processed.select_dtypes(include=['bool']).columns
    X_processed[bool_cols] = X_processed[bool_cols].astype(int)
    
    # Convert to numpy
    X_array = X_processed.values.astype(np.float32)
    y_array = y.values.astype(np.float32)
    
    # Normalize features
    if fit_scaler:
        scaler = StandardScaler()
        X_array = scaler.fit_transform(X_array)
    elif scaler is not None:
        X_array = scaler.transform(X_array)
    
    return X_array, y_array, scaler


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create PyTorch DataLoader.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


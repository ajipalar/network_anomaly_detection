"""
Utility to create initial train/test split.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


def create_train_test_split(
    data_path: str,
    train_path: str = 'data/processed/train.csv',
    test_path: str = 'data/processed/test.csv',
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Create train/test split and save to files if they don't exist.
    
    Args:
        data_path: Path to raw data CSV file
        train_path: Path to save training data
        test_path: Path to save test data
        test_size: Fraction of data to use for testing
        random_state: Random seed
        stratify: Whether to stratify split by label
        
    Returns:
        Tuple of (train_df, test_df) if files were created, (None, None) if files already exist
    """
    # Check if files already exist
    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"Train/test split already exists:")
        print(f"  Train: {train_path}")
        print(f"  Test: {test_path}")
        return None, None
    
    # Load raw data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")
    
    # Separate features and labels
    label_col = 'label'
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in data")
    
    X = df.drop(columns=[label_col])
    y = df[label_col]
    
    # Create train/test split
    print(f"Creating train/test split (test_size={test_size})...")
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )
    
    # Combine features and labels
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Create output directory
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    
    # Save to files
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved train data to {train_path} ({len(train_df)} samples)")
    print(f"Saved test data to {test_path} ({len(test_df)} samples)")
    
    return train_df, test_df


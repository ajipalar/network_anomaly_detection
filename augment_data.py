"""
Data augmentation script for network anomaly detection.

This script:
1. Holds out test data before augmentation
2. Adds Gaussian noise to numerical columns
3. Resamples positive class to make up ~30% of augmented data
4. Saves augmented training/validation data
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from typing import Tuple


def identify_numerical_columns(df: pd.DataFrame, label_col: str = 'label') -> list:
    """
    Identify numerical columns (excluding label and boolean columns).
    
    Args:
        df: DataFrame
        label_col: Name of label column
        
    Returns:
        List of numerical column names
    """
    # Exclude label column
    feature_cols = [col for col in df.columns if col != label_col]
    
    # Get numerical columns (float or int, but not boolean)
    numerical_cols = []
    for col in feature_cols:
        dtype = df[col].dtype
        if dtype in ['float64', 'float32', 'int64', 'int32']:
            # Check if it's actually boolean-like (only 0/1)
            unique_vals = df[col].unique()
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                continue  # Skip boolean-like columns
            numerical_cols.append(col)
    
    return numerical_cols


def add_gaussian_noise(
    X: pd.DataFrame,
    numerical_cols: list,
    noise_scale: float = 0.1
) -> pd.DataFrame:
    """
    Add Gaussian noise to numerical columns.
    
    Args:
        X: Feature DataFrame
        numerical_cols: List of numerical column names
        noise_scale: Standard deviation of noise as fraction of column std
        
    Returns:
        DataFrame with added noise
    """
    X_augmented = X.copy()
    
    for col in numerical_cols:
        if col not in X.columns:
            continue
        
        # Calculate noise scale based on column standard deviation
        col_std = X[col].std()
        if col_std > 0:
            noise_std = col_std * noise_scale
            noise = np.random.normal(0, noise_std, size=len(X))
            X_augmented[col] = X_augmented[col] + noise
        else:
            # If std is 0, use a small absolute noise
            noise = np.random.normal(0, 0.01, size=len(X))
            X_augmented[col] = X_augmented[col] + noise
    
    return X_augmented


def resample_to_target_ratio(
    X: pd.DataFrame,
    y: pd.Series,
    target_positive_ratio: float = 0.3,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Resample data to achieve target positive class ratio.
    
    Args:
        X: Feature DataFrame
        y: Label Series
        target_positive_ratio: Desired ratio of positive samples (0-1)
        random_state: Random seed
        
    Returns:
        Resampled X and y
    """
    np.random.seed(random_state)
    
    # Separate positive and negative samples
    positive_mask = y == 1.0
    negative_mask = y == 0.0
    
    X_positive = X[positive_mask].copy()
    y_positive = y[positive_mask].copy()
    X_negative = X[negative_mask].copy()
    y_negative = y[negative_mask].copy()
    
    n_positive = len(X_positive)
    n_negative = len(X_negative)
    
    print(f"Original distribution: {n_positive} positive, {n_negative} negative")
    print(f"Original positive ratio: {n_positive / (n_positive + n_negative):.3f}")
    
    # Calculate how many samples we need
    # target_positive_ratio = n_pos / (n_pos + n_neg)
    # Solving: n_neg = n_pos * (1 - target_positive_ratio) / target_positive_ratio
    # If we keep all positives, we need: n_neg_new = n_pos * (1 - target_positive_ratio) / target_positive_ratio
    
    if n_positive > 0:
        # Calculate target number of negative samples
        n_negative_target = int(n_positive * (1 - target_positive_ratio) / target_positive_ratio)
        
        # Resample negative class if needed
        if n_negative_target < n_negative:
            # Downsample negative class
            negative_indices = np.random.choice(
                n_negative,
                size=n_negative_target,
                replace=False
            )
            X_negative = X_negative.iloc[negative_indices].copy()
            y_negative = y_negative.iloc[negative_indices].copy()
        elif n_negative_target > n_negative:
            # Upsample negative class
            negative_indices = np.random.choice(
                n_negative,
                size=n_negative_target,
                replace=True
            )
            X_negative = X_negative.iloc[negative_indices].copy()
            y_negative = y_negative.iloc[negative_indices].copy()
        
        # Combine positive and negative samples
        X_resampled = pd.concat([X_positive, X_negative], ignore_index=True)
        y_resampled = pd.concat([y_positive, y_negative], ignore_index=True)
        
        # Shuffle
        shuffle_indices = np.random.permutation(len(X_resampled))
        X_resampled = X_resampled.iloc[shuffle_indices].reset_index(drop=True)
        y_resampled = y_resampled.iloc[shuffle_indices].reset_index(drop=True)
        
        n_pos_final = (y_resampled == 1.0).sum()
        n_neg_final = (y_resampled == 0.0).sum()
        print(f"Resampled distribution: {n_pos_final} positive, {n_neg_final} negative")
        print(f"Resampled positive ratio: {n_pos_final / (n_pos_final + n_neg_final):.3f}")
    else:
        print("Warning: No positive samples found, returning original data")
        X_resampled = X.copy()
        y_resampled = y.copy()
    
    return X_resampled, y_resampled


def augment_data(
    data_path: str = 'data/processed/train.csv',
    output_dir: str = "data/augmented",
    val_size: float = 0.1,
    noise_scale: float = 0.1,
    target_positive_ratio: float = 0.3,
    random_state: int = 42,
    num_augmentations: int = 10
) -> None:
    """
    Main augmentation function.
    
    Args:
        data_path: Path to training data CSV (default: data/processed/train.csv)
        output_dir: Directory to save augmented data
        val_size: Fraction of training data to use as validation
        noise_scale: Standard deviation of noise as fraction of column std
        target_positive_ratio: Target ratio of positive samples
        random_state: Random seed
        num_augmentations: Number of times to augment the data
    """
    print("=" * 60)
    print("Data Augmentation Script")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Separate features and labels
    label_col = 'label'
    feature_cols = [col for col in df.columns if col != label_col]
    X = df[feature_cols]
    y = df[label_col]
    
    # Identify numerical columns
    numerical_cols = identify_numerical_columns(df, label_col)
    print(f"\nIdentified {len(numerical_cols)} numerical columns:")
    print(f"  {numerical_cols}")
    
    # Note: Test data is already separated in data/processed/test.csv
    # We only augment the training data
    
    # STEP 1: Split train/val from training data
    print(f"\n{'='*60}")
    print("Step 1: Splitting train/validation data")
    print(f"{'='*60}")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_size,
        random_state=random_state,
        stratify=y
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # STEP 2: Augment training data
    print(f"\n{'='*60}")
    print("Step 2: Augmenting training data")
    print(f"{'='*60}")
    print(f"Noise scale: {noise_scale}")
    print(f"Target positive ratio: {target_positive_ratio}")
    print(f"Number of augmentations: {num_augmentations}")
    
    # Collect augmented samples
    X_train_augmented_list = [X_train]
    y_train_augmented_list = [y_train]
    
    for aug_idx in range(num_augmentations):
        print(f"\nAugmentation {aug_idx + 1}/{num_augmentations}...")
        
        # Add Gaussian noise
        X_train_noisy = add_gaussian_noise(
            X_train,
            numerical_cols,
            noise_scale=noise_scale
        )
        
        # Resample to target ratio
        X_train_resampled, y_train_resampled = resample_to_target_ratio(
            X_train_noisy,
            y_train,
            target_positive_ratio=target_positive_ratio,
            random_state=random_state + aug_idx
        )
        
        X_train_augmented_list.append(X_train_resampled)
        y_train_augmented_list.append(y_train_resampled)
    
    # Combine all augmented data
    X_train_augmented = pd.concat(X_train_augmented_list, ignore_index=True)
    y_train_augmented = pd.concat(y_train_augmented_list, ignore_index=True)
    
    # Final shuffle
    shuffle_indices = np.random.permutation(len(X_train_augmented))
    X_train_augmented = X_train_augmented.iloc[shuffle_indices].reset_index(drop=True)
    y_train_augmented = y_train_augmented.iloc[shuffle_indices].reset_index(drop=True)
    
    print(f"\nFinal augmented training data: {len(X_train_augmented)} samples")
    final_pos_ratio = (y_train_augmented == 1.0).sum() / len(y_train_augmented)
    print(f"Final positive ratio: {final_pos_ratio:.3f}")
    
    # Save augmented training data
    train_df = pd.concat([X_train_augmented, y_train_augmented], axis=1)
    train_path = os.path.join(output_dir, "train_data_augmented.csv")
    train_df.to_csv(train_path, index=False)
    print(f"Saved augmented training data to {train_path}")
    
    # Save validation data (no augmentation, but keep for consistency)
    val_df = pd.concat([X_val, y_val], axis=1)
    val_path = os.path.join(output_dir, "val_data.csv")
    val_df.to_csv(val_path, index=False)
    print(f"Saved validation data to {val_path}")
    
    print(f"\n{'='*60}")
    print("Augmentation complete!")
    print(f"{'='*60}")
    print(f"\nOutput files:")
    print(f"  - {train_path}")
    print(f"  - {val_path}")
    print(f"\nNote: Test data is in data/processed/test.csv (not augmented)")


def main():
    parser = argparse.ArgumentParser(description='Augment network anomaly detection data')
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/processed/train.csv',
        help='Path to training data CSV file (default: data/processed/train.csv)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/augmented',
        help='Directory to save augmented data'
    )
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.1,
        help='Fraction of training data to use as validation'
    )
    parser.add_argument(
        '--noise-scale',
        type=float,
        default=0.1,
        help='Standard deviation of noise as fraction of column std'
    )
    parser.add_argument(
        '--target-positive-ratio',
        type=float,
        default=0.3,
        help='Target ratio of positive samples in augmented data'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--num-augmentations',
        type=int,
        default=1,
        help='Number of times to augment the training data'
    )
    
    args = parser.parse_args()
    
    augment_data(
        data_path=args.data_path,
        output_dir=args.output_dir,
        val_size=args.val_size,
        noise_scale=args.noise_scale,
        target_positive_ratio=args.target_positive_ratio,
        random_state=args.random_state,
        num_augmentations=args.num_augmentations
    )


if __name__ == '__main__':
    main()


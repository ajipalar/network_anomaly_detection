"""
Main training script.
"""

import argparse
import os
from datetime import datetime
from typing import Tuple, Optional, Dict, Any
import numpy as np

import torch
from sklearn.model_selection import train_test_split

from src.data import load_data, preprocess_data, NetworkAnomalyDataset, create_dataloader, create_train_test_split
from src.models import create_model
from src.training import Trainer, train_xgboost_model
from src.utils import load_config, get_device
from src.validation import perform_kfold_cv, print_cv_summary

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Track if TensorBoard has been patched (module-level flag)
_tensorboard_patched = False


def safe_patch_tensorboard(root_logdir: str):
    """
    Safely patch TensorBoard to sync with wandb, only if not already patched.
    
    Args:
        root_logdir: Root directory for TensorBoard logs
    """
    global _tensorboard_patched
    
    # Early return if already patched or wandb not available
    if _tensorboard_patched or not WANDB_AVAILABLE or wandb is None:
        return
    
    # Only patch if wandb run is initialized
    if wandb.run is None:
        return
    
    try:
        # Check if SummaryWriter has been patched by looking for wandb-specific attributes
        from torch.utils.tensorboard import SummaryWriter
        
        # Check if already patched (either by us or another module)
        if hasattr(SummaryWriter, '_wandb_patched'):
            _tensorboard_patched = True
            return
        
        # Patch TensorBoard
        wandb.tensorboard.patch(root_logdir=root_logdir)
        # Mark as patched to prevent duplicate patches
        SummaryWriter._wandb_patched = True
        _tensorboard_patched = True
    except Exception as e:
        print(f"Warning: Could not patch TensorBoard: {e}")


def initialize_wandb(config: Dict[str, Any]) -> bool:
    """
    Initialize Weights & Biases if enabled in config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if wandb was initialized, False otherwise
    """
    wandb_config = config.get('logging', {})
    use_wandb = wandb_config.get('use_wandb', False) and WANDB_AVAILABLE
    
    if not use_wandb:
        return False
    
    project = wandb_config.get('wandb_project', 'network-anomaly-detection')
    entity = wandb_config.get('wandb_entity', None)
    run_name = wandb_config.get('wandb_run_name')
    if run_name is None:
        run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=config,
        job_type="training",
        sync_tensorboard=True
    )
    
    safe_patch_tensorboard(config['logging']['tensorboard_dir'])
    print(f"Initialized W&B run: {run_name}")
    print(f"TensorBoard logs will be synced to wandb")
    
    return True


def load_augmented_data(data_config: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load augmented training and validation data.
    
    Args:
        data_config: Data configuration dictionary
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val) if successful, None otherwise
    """
    aug_dir = data_config.get('augmented_data_dir', 'data/augmented')
    train_path = data_config.get('train_data_path') or os.path.join(aug_dir, 'train_data_augmented.csv')
    val_path = data_config.get('val_data_path') or os.path.join(aug_dir, 'val_data.csv')
    
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        print(f"Warning: Augmented data not found at {train_path} or {val_path}")
        return None
    
    print("Loading augmented data...")
    X_train, y_train = load_data(train_path)
    X_val, y_val = load_data(val_path)
    print(f"Loaded augmented training data: {len(X_train)} samples")
    print(f"Loaded validation data: {len(X_val)} samples")
    
    return X_train, y_train, X_val, y_val


def preprocess_train_val_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    """
    Preprocess training and validation data with a shared scaler.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        Tuple of (X_train_processed, y_train_processed, X_val_processed, y_val_processed, scaler)
    """
    print("Preprocessing training data...")
    X_train_processed, y_train_processed, scaler = preprocess_data(
        X_train, y_train,
        fit_scaler=True,
        scaler=None
    )
    
    print("Preprocessing validation data...")
    X_val_processed, y_val_processed, _ = preprocess_data(
        X_val, y_val,
        fit_scaler=False,
        scaler=scaler
    )
    
    return X_train_processed, y_train_processed, X_val_processed, y_val_processed, scaler


def split_data_for_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    data_config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split training data into train and validation sets for XGBoost.
    Note: Test set is already separated in data/processed/test.csv
    
    Args:
        X: Features (from training data)
        y: Labels (from training data)
        data_config: Data configuration dictionary
        
    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    print("Splitting training data into train/validation...")
    val_size = data_config['val_size']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_size,
        random_state=data_config['random_state'],
        stratify=y
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    return X_train, X_val, y_train, y_val


def split_data_for_pytorch(
    X: np.ndarray,
    y: np.ndarray,
    data_config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split training data into train and validation sets for PyTorch.
    Note: Test set is already separated in data/processed/test.csv
    
    Args:
        X: Features (from training data)
        y: Labels (from training data)
        data_config: Data configuration dictionary
        
    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    print("Splitting training data into train/validation...")
    val_size = data_config['val_size']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_size,
        random_state=data_config['random_state'],
        stratify=y
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    return X_train, X_val, y_train, y_val


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any],
    use_cv: bool
) -> None:
    """
    Train XGBoost model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Configuration dictionary
        use_cv: Whether cross-validation was requested (not yet supported for XGBoost)
    """
    if use_cv:
        print("XGBoost K-fold cross validation not yet implemented. Using standard train/val split.")
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    checkpoint_dir = config['checkpoint']['save_dir']
    model, metrics = train_xgboost_model(
        X_train,
        y_train,
        X_val,
        y_val,
        config,
        checkpoint_dir
    )
    
    print("XGBoost training completed!")


def create_pytorch_trainer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any],
    device: torch.device
) -> Trainer:
    """
    Create PyTorch trainer with model, data loaders, and configuration.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Configuration dictionary
        device: Device to run training on
        
    Returns:
        Configured Trainer instance
    """
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
    model_config['input_dim'] = X_train.shape[1]
    model = create_model(model_config)
    
    # Create trainer
    run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training'],
        device=device,
        checkpoint_dir=config['checkpoint']['save_dir'],
        tensorboard_dir=config['logging']['tensorboard_dir'],
        run_name=run_name
    )
    
    return trainer


def train_pytorch_standard(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any],
    device: torch.device,
    resume_checkpoint: Optional[str] = None
) -> None:
    """
    Train PyTorch model with standard train/val split.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Configuration dictionary
        device: Device to run training on
        resume_checkpoint: Optional path to checkpoint to resume from
    """
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    trainer = create_pytorch_trainer(X_train, y_train, X_val, y_val, config, device)
    
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        trainer.load_checkpoint(resume_checkpoint)
    
    print("Starting training...")
    trainer.train(config['training']['num_epochs'])
    print("Training completed!")


def train_pytorch_cv(
    X: np.ndarray,
    y: np.ndarray,
    config: Dict[str, Any],
    device: torch.device,
    is_augmented: bool = False
) -> None:
    """
    Train PyTorch model with K-fold cross-validation.
    
    Args:
        X: Features
        y: Labels
        config: Configuration dictionary
        device: Device to run training on
        is_augmented: Whether using augmented data (for warning message)
    """
    if is_augmented:
        print("Warning: Cross-validation with augmented data not fully supported.")
        print("Using augmented training data for CV...")
    
    print("Starting K-fold cross validation...")
    fold_results = perform_kfold_cv(X, y, config, device)
    print_cv_summary(fold_results)


def train_with_augmented_data(
    config: Dict[str, Any],
    device: torch.device,
    model_type: str,
    use_cv: bool,
    resume_checkpoint: Optional[str] = None
) -> bool:
    """
    Train model using augmented data.
    
    Args:
        config: Configuration dictionary
        device: Device to run training on
        model_type: Type of model ('pytorch' or 'xgboost')
        use_cv: Whether to use cross-validation
        resume_checkpoint: Optional path to checkpoint to resume from
        
    Returns:
        True if training was successful, False if augmented data not found
    """
    data_config = config['data']
    
    # Ensure train/test split exists (needed before augmentation)
    train_path = 'data/processed/train.csv'
    test_path = 'data/processed/test.csv'
    create_train_test_split(
        data_path=data_config['data_path'],
        train_path=train_path,
        test_path=test_path,
        test_size=data_config['test_size'],
        random_state=data_config['random_state'],
        stratify=True
    )
    
    data_result = load_augmented_data(data_config)
    
    if data_result is None:
        return False
    
    X_train, y_train, X_val, y_val = data_result
    X_train_processed, y_train_processed, X_val_processed, y_val_processed, _ = preprocess_train_val_data(
        X_train, y_train, X_val, y_val
    )
    
    if model_type == 'xgboost':
        train_xgboost(
            X_train_processed, y_train_processed,
            X_val_processed, y_val_processed,
            config, use_cv
        )
    elif use_cv:
        train_pytorch_cv(
            X_train_processed, y_train_processed,
            config, device, is_augmented=True
        )
    else:
        train_pytorch_standard(
            X_train_processed, y_train_processed,
            X_val_processed, y_val_processed,
            config, device, resume_checkpoint
        )
    
    return True


def train_with_original_data(
    config: Dict[str, Any],
    device: torch.device,
    model_type: str,
    use_cv: bool,
    resume_checkpoint: Optional[str] = None
) -> None:
    """
    Train model using original (non-augmented) data.
    
    Args:
        config: Configuration dictionary
        device: Device to run training on
        model_type: Type of model ('pytorch' or 'xgboost')
        use_cv: Whether to use cross-validation
        resume_checkpoint: Optional path to checkpoint to resume from
    """
    data_config = config['data']
    
    # Ensure train/test split exists
    train_path = 'data/processed/train.csv'
    test_path = 'data/processed/test.csv'
    create_train_test_split(
        data_path=data_config['data_path'],
        train_path=train_path,
        test_path=test_path,
        test_size=data_config['test_size'],
        random_state=data_config['random_state'],
        stratify=True
    )
    
    # Load training data
    print("Loading training data...")
    X, y = load_data(train_path)
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    
    print("Preprocessing data...")
    X_processed, y_processed, _ = preprocess_data(
        X, y,
        fit_scaler=True,
        scaler=None
    )
    
    if model_type == 'xgboost':
        # For XGBoost, split train into train/val (test is already separated)
        X_train, X_val, y_train, y_val = split_data_for_xgboost(
            X_processed, y_processed, data_config
        )
        train_xgboost(
            X_train, y_train,
            X_val, y_val,
            config, use_cv
        )
    elif use_cv:
        train_pytorch_cv(X_processed, y_processed, config, device)
    else:
        # For PyTorch standard training, split train into train/val (test is already separated)
        X_train, X_val, y_train, y_val = split_data_for_pytorch(
            X_processed, y_processed, data_config
        )
        train_pytorch_standard(
            X_train, y_train,
            X_val, y_val,
            config, device, resume_checkpoint
        )


def finish_wandb() -> None:
    """Finish wandb run and print summary."""
    if not WANDB_AVAILABLE or wandb is None:
        return
    
    wandb.finish()
    if wandb.run:
        print(f"View training results at: {wandb.run.url}")
        print(f"Project dashboard: https://wandb.ai/{wandb.run.entity or 'YOUR_USERNAME'}/{wandb.run.project}")
    else:
        print("W&B run completed")


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='Train network anomaly detection model')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--use-cv',
        action='store_true',
        help='Use K-fold cross validation'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize wandb if enabled
    initialize_wandb(config)
    
    # Get device
    device = get_device(
        use_cuda=config['device']['use_cuda'],
        cuda_device=config['device']['cuda_device']
    )
    print(f"Using device: {device}")
    
    # Determine model type and data source
    data_config = config['data']
    use_augmented = data_config.get('use_augmented_data', False)
    model_type = config['model'].get('type', 'pytorch').lower()
    
    # Train model
    if use_augmented:
        success = train_with_augmented_data(
            config, device, model_type, args.use_cv, args.resume
        )
        if not success:
            print("Falling back to original data loading...")
            train_with_original_data(
                config, device, model_type, args.use_cv, args.resume
            )
    else:
        train_with_original_data(
            config, device, model_type, args.use_cv, args.resume
        )
    
    # Finish wandb
    finish_wandb()


if __name__ == '__main__':
    main()

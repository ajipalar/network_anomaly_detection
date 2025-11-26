"""
Train final model using best weights from k-fold cross-validation.

This script:
1. Finds all fold checkpoints from k-fold CV
2. Identifies the best model (lowest validation loss)
3. Initializes a new model with those weights
4. Trains on full training data (train + val combined)
5. Saves as final model
"""

import argparse
import torch
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split

from src.data import load_data, preprocess_data, NetworkAnomalyDataset, create_dataloader
from src.models import create_model
from src.training import Trainer
from src.utils import load_config, get_device

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


def find_fold_checkpoints(checkpoint_dir: str) -> list:
    """
    Find all fold checkpoints and their validation losses.
    
    Args:
        checkpoint_dir: Base checkpoint directory
        
    Returns:
        List of dicts with 'fold', 'path', 'val_loss' for each fold
    """
    fold_checkpoints = []
    pattern = os.path.join(checkpoint_dir, "fold_*", "best_model.pt")
    
    for checkpoint_path in glob.glob(pattern):
        # Extract fold number from path
        fold_dir = os.path.dirname(checkpoint_path)
        fold_name = os.path.basename(fold_dir)
        try:
            fold_num = int(fold_name.split('_')[1])
            
            # Load checkpoint to get validation loss
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            fold_checkpoints.append({
                'fold': fold_num,
                'path': checkpoint_path,
                'val_loss': val_loss
            })
        except (ValueError, IndexError, Exception) as e:
            print(f"Warning: Could not load checkpoint {checkpoint_path}: {e}")
            continue
    
    # Sort by fold number
    fold_checkpoints.sort(key=lambda x: x['fold'])
    return fold_checkpoints


def find_best_fold(fold_checkpoints: list) -> dict:
    """
    Find the fold with the best (lowest) validation loss.
    
    Args:
        fold_checkpoints: List of fold checkpoint dicts
        
    Returns:
        Best fold checkpoint dict
    """
    if not fold_checkpoints:
        raise ValueError("No fold checkpoints found")
    
    best_fold = min(fold_checkpoints, key=lambda x: x['val_loss'])
    return best_fold


def main():
    parser = argparse.ArgumentParser(description='Train final model using best CV fold weights')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Checkpoint directory (default: from config)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for final model (default: checkpoints/final_model)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize wandb if enabled
    wandb_config = config.get('logging', {})
    use_wandb = wandb_config.get('use_wandb', False) and WANDB_AVAILABLE
    
    if use_wandb:
        project = wandb_config.get('wandb_project', 'network-anomaly-detection')
        entity = wandb_config.get('wandb_entity', None)
        from datetime import datetime
        run_name = f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=config,
            job_type="final_training",
            sync_tensorboard=True
        )
        safe_patch_tensorboard(config['logging']['tensorboard_dir'])
        print(f"Initialized W&B run: {run_name}")
        print(f"TensorBoard logs will be synced to wandb")
    
    # Get device
    device = get_device(
        use_cuda=config['device']['use_cuda'],
        cuda_device=config['device']['cuda_device']
    )
    print(f"Using device: {device}")
    
    # Find fold checkpoints
    checkpoint_dir = args.checkpoint_dir or config['checkpoint']['save_dir']
    print(f"\n{'='*60}")
    print("Finding fold checkpoints...")
    print(f"{'='*60}")
    
    fold_checkpoints = find_fold_checkpoints(checkpoint_dir)
    
    if not fold_checkpoints:
        raise ValueError(f"No fold checkpoints found in {checkpoint_dir}")
    
    print(f"Found {len(fold_checkpoints)} fold checkpoints:")
    for fold_info in fold_checkpoints:
        print(f"  Fold {fold_info['fold']}: val_loss={fold_info['val_loss']:.4f}, path={fold_info['path']}")
    
    # Find best fold
    best_fold = find_best_fold(fold_checkpoints)
    print(f"\nBest fold: Fold {best_fold['fold']} with validation loss {best_fold['val_loss']:.4f}")
    print(f"Loading weights from: {best_fold['path']}")
    
    # Load best model checkpoint
    best_checkpoint = torch.load(best_fold['path'], map_location=device)
    
    # Load data
    print(f"\n{'='*60}")
    print("Loading data...")
    print(f"{'='*60}")
    data_config = config['data']
    use_augmented = data_config.get('use_augmented_data', False)
    
    if use_augmented:
        # Load augmented data
        aug_dir = data_config.get('augmented_data_dir', 'data/augmented')
        train_path = data_config.get('train_data_path') or os.path.join(aug_dir, 'train_data_augmented.csv')
        val_path = data_config.get('val_data_path') or os.path.join(aug_dir, 'val_data.csv')
        
        if os.path.exists(train_path) and os.path.exists(val_path):
            print("Loading augmented data...")
            X_train, y_train = load_data(train_path)
            X_val, y_val = load_data(val_path)
            print(f"Loaded training data: {len(X_train)} samples")
            print(f"Loaded validation data: {len(X_val)} samples")
            
            # Combine train and val for final training
            import pandas as pd
            X_full = pd.concat([X_train, X_val], ignore_index=True)
            y_full = pd.concat([y_train, y_val], ignore_index=True)
            print(f"Combined training data: {len(X_full)} samples")
            
            # Preprocess
            print("Preprocessing data...")
            X_processed, y_processed, scaler = preprocess_data(
                X_full, y_full,
                fit_scaler=True,
                scaler=None
            )
        else:
            print(f"Warning: Augmented data not found, falling back to original data")
            use_augmented = False
    
    if not use_augmented:
        # Load original data
        X, y = load_data(data_config['data_path'])
        print(f"Loaded {len(X)} samples with {X.shape[1]} features")
        
        # Split: hold out test, combine train+val for final training
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=data_config['test_size'],
            random_state=data_config['random_state'],
            stratify=y
        )
        
        # Preprocess
        print("Preprocessing data...")
        X_processed, y_processed, scaler = preprocess_data(
            X_train_val, y_train_val,
            fit_scaler=True,
            scaler=None
        )
        print(f"Using {len(X_processed)} samples for final training (train+val combined)")
    
    # Create model
    print(f"\n{'='*60}")
    print("Creating model...")
    print(f"{'='*60}")
    model_config = config['model'].copy()
    model_config['input_dim'] = X_processed.shape[1]
    model = create_model(model_config)
    
    # Initialize with best fold weights
    print(f"Initializing model with weights from Fold {best_fold['fold']}...")
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model.to(device)
    
    # Split for final training (use a small portion for validation)
    # We'll use 10% of the combined data for validation during final training
    print(f"\n{'='*60}")
    print("Splitting data for final training...")
    print(f"{'='*60}")
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_processed, y_processed,
        test_size=0.1,
        random_state=data_config['random_state'],
        stratify=y_processed
    )
    print(f"Final training set: {len(X_train_final)} samples")
    print(f"Final validation set: {len(X_val_final)} samples")
    
    # Create datasets and loaders
    train_dataset = NetworkAnomalyDataset(X_train_final, y_train_final)
    val_dataset = NetworkAnomalyDataset(X_val_final, y_val_final)
    
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
    
    # Create trainer
    output_dir = args.output_dir or os.path.join(config['checkpoint']['save_dir'], 'final_model')
    os.makedirs(output_dir, exist_ok=True)
    
    from datetime import datetime
    run_name = f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training'],
        device=device,
        checkpoint_dir=output_dir,
        tensorboard_dir=config['logging']['tensorboard_dir'],
        run_name=run_name
    )
    
    # Train final model
    print(f"\n{'='*60}")
    print("Training final model...")
    print(f"{'='*60}")
    print(f"Starting from best fold weights (Fold {best_fold['fold']}, val_loss={best_fold['val_loss']:.4f})")
    print(f"Training for {config['training']['num_epochs']} epochs...")
    
    trainer.train(config['training']['num_epochs'])
    
    print(f"\n{'='*60}")
    print("Final model training complete!")
    print(f"{'='*60}")
    print(f"Final model saved to: {os.path.join(output_dir, 'best_model.pt')}")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
        if wandb.run:
            print(f"View training results at: {wandb.run.url}")
            print(f"Project dashboard: https://wandb.ai/{wandb.run.entity or 'YOUR_USERNAME'}/{wandb.run.project}")
        else:
            print("W&B run completed")


if __name__ == '__main__':
    main()


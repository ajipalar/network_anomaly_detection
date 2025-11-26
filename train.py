"""
Main training script.
"""

import argparse
import torch
from sklearn.model_selection import train_test_split

from src.data import load_data, preprocess_data, NetworkAnomalyDataset, create_dataloader
from src.models import create_model, create_xgboost_model
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


def main():
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
    wandb_config = config.get('logging', {})
    use_wandb = wandb_config.get('use_wandb', False) and WANDB_AVAILABLE
    
    if use_wandb:
        project = wandb_config.get('wandb_project', 'network-anomaly-detection')
        entity = wandb_config.get('wandb_entity', None)
        run_name = wandb_config.get('wandb_run_name')
        if run_name is None:
            from datetime import datetime
            run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=config,
            job_type="training",
            sync_tensorboard=True  # Sync TensorBoard logs to wandb
        )
        # Patch TensorBoard to automatically sync logs (only if not already patched)
        safe_patch_tensorboard(config['logging']['tensorboard_dir'])
        print(f"Initialized W&B run: {run_name}")
        print(f"TensorBoard logs will be synced to wandb")
    
    # Get device
    device = get_device(
        use_cuda=config['device']['use_cuda'],
        cuda_device=config['device']['cuda_device']
    )
    print(f"Using device: {device}")
    
    # Load data
    data_config = config['data']
    use_augmented = data_config.get('use_augmented_data', False)
    
    # Determine model type early
    model_type = config['model'].get('type', 'pytorch').lower()
    
    if use_augmented:
        # Load augmented data
        import os
        aug_dir = data_config.get('augmented_data_dir', 'data/augmented')
        train_path = data_config.get('train_data_path') or os.path.join(aug_dir, 'train_data_augmented.csv')
        val_path = data_config.get('val_data_path') or os.path.join(aug_dir, 'val_data.csv')
        
        if os.path.exists(train_path) and os.path.exists(val_path):
            print("Loading augmented data...")
            X_train, y_train = load_data(train_path)
            X_val, y_val = load_data(val_path)
            print(f"Loaded augmented training data: {len(X_train)} samples")
            print(f"Loaded validation data: {len(X_val)} samples")
            
            # Preprocess training data (fit scaler)
            print("Preprocessing training data...")
            X_train_processed, y_train_processed, scaler = preprocess_data(
                X_train, y_train,
                fit_scaler=True,
                scaler=None
            )
            
            # Preprocess validation data (use same scaler)
            print("Preprocessing validation data...")
            X_val_processed, y_val_processed, _ = preprocess_data(
                X_val, y_val,
                fit_scaler=False,
                scaler=scaler
            )
            
            if model_type == 'xgboost':
                # XGBoost training with augmented data
                if args.use_cv:
                    print("XGBoost K-fold cross validation not yet implemented. Using standard train/val split.")
                    args.use_cv = False
                
                print(f"Train: {len(X_train_processed)}, Val: {len(X_val_processed)}")
                
                # Train XGBoost model
                checkpoint_dir = config['checkpoint']['save_dir']
                model, metrics = train_xgboost_model(
                    X_train_processed,
                    y_train_processed,
                    X_val_processed,
                    y_val_processed,
                    config,
                    checkpoint_dir
                )
                
                print("XGBoost training completed!")
            elif args.use_cv:
                # PyTorch K-fold CV with augmented data
                print("Warning: Cross-validation with augmented data not fully supported.")
                print("Using augmented training data for CV...")
                # For CV, we'll use the augmented training data
                fold_results = perform_kfold_cv(
                    X_train_processed,
                    y_train_processed,
                    config,
                    device
                )
                print_cv_summary(fold_results)
            else:
                # PyTorch standard training with augmented data
                print(f"Train: {len(X_train_processed)}, Val: {len(X_val_processed)}")
                
                # Create datasets
                train_dataset = NetworkAnomalyDataset(X_train_processed, y_train_processed)
                val_dataset = NetworkAnomalyDataset(X_val_processed, y_val_processed)
                
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
                model_config['input_dim'] = X_train_processed.shape[1]
                model = create_model(model_config)
                
                # Create trainer
                from datetime import datetime
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
                
                # Resume from checkpoint if specified
                if args.resume:
                    print(f"Resuming from checkpoint: {args.resume}")
                    trainer.load_checkpoint(args.resume)
                
                # Train
                print("Starting training...")
                trainer.train(config['training']['num_epochs'])
                
                print("Training completed!")
        else:
            print(f"Warning: Augmented data not found at {train_path} or {val_path}")
            print("Falling back to original data loading...")
            use_augmented = False
    
    if not use_augmented:
        # Original data loading
        print("Loading data...")
        X, y = load_data(data_config['data_path'])
        print(f"Loaded {len(X)} samples with {X.shape[1]} features")
        
        # Preprocess data
        print("Preprocessing data...")
        X_processed, y_processed, scaler = preprocess_data(
            X, y,
            fit_scaler=True,
            scaler=None
        )
        
        if model_type == 'xgboost':
            # XGBoost training
            if args.use_cv:
                print("XGBoost K-fold cross validation not yet implemented. Using standard train/val split.")
                args.use_cv = False
            
            # Standard train/val/test split for XGBoost
            print("Splitting data...")
            X_train_processed, X_temp, y_train_processed, y_temp = train_test_split(
                X_processed,
                y_processed,
                test_size=data_config['test_size'],
                random_state=data_config['random_state']
            )
            
            val_size = data_config['val_size'] / (1 - data_config['test_size'])
            X_val_processed, X_test, y_val_processed, y_test = train_test_split(
                X_temp,
                y_temp,
                test_size=1 - val_size,
                random_state=data_config['random_state']
            )
            
            print(f"Train: {len(X_train_processed)}, Val: {len(X_val_processed)}, Test: {len(X_test)}")
            
            # Train XGBoost model
            checkpoint_dir = config['checkpoint']['save_dir']
            model, metrics = train_xgboost_model(
                X_train_processed,
                y_train_processed,
                X_val_processed,
                y_val_processed,
                config,
                checkpoint_dir
            )
            
            print("XGBoost training completed!")
            
        else:
            # PyTorch training
            if args.use_cv:
                # K-fold cross validation
                print("Starting K-fold cross validation...")
                fold_results = perform_kfold_cv(
                    X_processed,
                    y_processed,
                    config,
                    device
                )
                print_cv_summary(fold_results)
            else:
                # Standard train/val/test split
                print("Splitting data...")
                X_train_processed, X_temp, y_train_processed, y_temp = train_test_split(
                    X_processed,
                    y_processed,
                    test_size=data_config['test_size'],
                    random_state=data_config['random_state']
                )
                
                val_size = data_config['val_size'] / (1 - data_config['test_size'])
                X_val_processed, X_test, y_val_processed, y_test = train_test_split(
                    X_temp,
                    y_temp,
                    test_size=1 - val_size,
                    random_state=data_config['random_state']
                )
                
                print(f"Train: {len(X_train_processed)}, Val: {len(X_val_processed)}, Test: {len(X_test)}")
            
            # Create datasets
            train_dataset = NetworkAnomalyDataset(X_train_processed, y_train_processed)
            val_dataset = NetworkAnomalyDataset(X_val_processed, y_val_processed)
            
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
            model_config['input_dim'] = X_processed.shape[1]
            model = create_model(model_config)
            
            # Create trainer
            from datetime import datetime
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
            
            # Resume from checkpoint if specified
            if args.resume:
                print(f"Resuming from checkpoint: {args.resume}")
                trainer.load_checkpoint(args.resume)
            
            # Train
            print("Starting training...")
            trainer.train(config['training']['num_epochs'])
            
            print("Training completed!")
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()
            if wandb.run:
                print(f"View training results at: {wandb.run.url}")
                print(f"Project dashboard: https://wandb.ai/{wandb.run.entity or 'YOUR_USERNAME'}/{wandb.run.project}")
            else:
                print("W&B run completed")


if __name__ == '__main__':
    main()


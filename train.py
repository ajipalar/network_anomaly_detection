"""
Main training script.
"""

import argparse
import torch
from sklearn.model_selection import train_test_split

from src.data import load_data, preprocess_data, NetworkAnomalyDataset, create_dataloader
from src.models import create_model
from src.training import Trainer
from src.utils import load_config, get_device
from src.validation import perform_kfold_cv, print_cv_summary

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


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
            job_type="training"
        )
        print(f"Initialized W&B run: {run_name}")
    
    # Get device
    device = get_device(
        use_cuda=config['device']['use_cuda'],
        cuda_device=config['device']['cuda_device']
    )
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    X, y = load_data(config['data']['data_path'])
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    
    # Preprocess data
    print("Preprocessing data...")
    X_processed, y_processed, scaler = preprocess_data(
        X, y,
        fit_scaler=True,
        scaler=None
    )
    
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
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_processed,
            y_processed,
            test_size=config['data']['test_size'],
            random_state=config['data']['random_state']
        )
        
        val_size = config['data']['val_size'] / (1 - config['data']['test_size'])
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=1 - val_size,
            random_state=config['data']['random_state']
        )
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
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
            print(f"View training results at: {wandb.run.url if wandb.run else 'N/A'}")


if __name__ == '__main__':
    main()


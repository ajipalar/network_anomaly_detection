"""
Testing script for evaluating trained models.
"""

import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

from src.data import load_data, preprocess_data, NetworkAnomalyDataset, create_dataloader
from src.models import create_model
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


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }


def main():
    parser = argparse.ArgumentParser(description='Test network anomaly detection model')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (required unless --final-model is used)'
    )
    parser.add_argument(
        '--test-data',
        type=str,
        default=None,
        help='Path to test data (if different from config)'
    )
    parser.add_argument(
        '--final-model',
        action='store_true',
        help='Test the final model (checkpoints/final_model/best_model.pt) instead of specified checkpoint'
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
            run_name = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=config,
            job_type="testing",
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
    
    # Load test data (use original test data, not augmented)
    data_config = config['data']
    use_augmented = data_config.get('use_augmented_data', False)
    
    if use_augmented:
        # Try to load test data from augmented directory
        aug_dir = data_config.get('augmented_data_dir', 'data/augmented')
        test_path = args.test_data or data_config.get('test_data_path') or os.path.join(aug_dir, 'test_data.csv')
        
        if os.path.exists(test_path):
            print(f"Loading test data from augmented directory: {test_path}")
            X_test, y_test = load_data(test_path)
            print(f"Loaded {len(X_test)} test samples")
            
            # Load scaler from checkpoint if available, otherwise fit new one
            # For testing, we should use the same scaler as training
            # Try to load from checkpoint or use training data to fit scaler
            print("Loading scaler from training data...")
            train_path = data_config.get('train_data_path') or os.path.join(aug_dir, 'train_data_augmented.csv')
            if os.path.exists(train_path):
                X_train, _ = load_data(train_path)
                _, _, scaler = preprocess_data(X_train, pd.Series([0] * len(X_train)), fit_scaler=True, scaler=None)
            else:
                # Fallback: fit scaler on test data (not ideal but works)
                print("Warning: Could not find training data to fit scaler, fitting on test data")
                scaler = None
        else:
            print(f"Warning: Test data not found at {test_path}, using original data")
            use_augmented = False
    
    if not use_augmented:
        # Original data loading
        data_path = args.test_data or data_config['data_path']
        print(f"Loading data from {data_path}...")
        X_test, y_test = load_data(data_path)
        print(f"Loaded {len(X_test)} samples with {X_test.shape[1]} features")
        
        # For testing with original data, we need to split to get test set
        from sklearn.model_selection import train_test_split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_test, y_test,
            test_size=data_config['test_size'],
            random_state=data_config['random_state'],
            stratify=y_test
        )
        
        # Fit scaler on training portion (we'll only use test portion)
        _, _, scaler = preprocess_data(X_temp, y_temp, fit_scaler=True, scaler=None)
    
    # Preprocess test data with the scaler
    print("Preprocessing test data...")
    X_processed, y_processed, _ = preprocess_data(
        X_test, y_test,
        fit_scaler=False,
        scaler=scaler
    )
    
    # Create test dataset
    test_dataset = NetworkAnomalyDataset(X_processed, y_processed)
    test_loader = create_dataloader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    # Determine checkpoint path
    if args.final_model:
        # Use final model checkpoint
        checkpoint_path = os.path.join(config['checkpoint']['save_dir'], 'final_model', 'best_model.pt')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Final model not found at {checkpoint_path}. "
                f"Please run train_final_model.py first or specify --checkpoint."
            )
        print(f"Testing final model from: {checkpoint_path}")
    else:
        if args.checkpoint is None:
            raise ValueError("Must specify --checkpoint or use --final-model flag")
        checkpoint_path = args.checkpoint
        print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model_config = config['model'].copy()
    model_config['input_dim'] = X_processed.shape[1]
    model = create_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"Model loaded successfully")
    
    # Initialize TensorBoard writer for test metrics
    tensorboard_dir = config['logging']['tensorboard_dir']
    run_name = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join(tensorboard_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Evaluate
    print("Evaluating model...")
    results = evaluate_model(model, test_loader, device)
    
    # Log metrics to TensorBoard
    writer.add_scalar('Metrics/Accuracy', results['accuracy'], 0)
    writer.add_scalar('Metrics/Precision', results['precision'], 0)
    writer.add_scalar('Metrics/Recall', results['recall'], 0)
    writer.add_scalar('Metrics/F1_Score', results['f1'], 0)
    writer.add_scalar('Metrics/AUC', results['auc'], 0)
    
    # Log to wandb if available
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            'Metrics/Accuracy': results['accuracy'],
            'Metrics/Precision': results['precision'],
            'Metrics/Recall': results['recall'],
            'Metrics/F1_Score': results['f1'],
            'Metrics/AUC': results['auc']
        })
    
    # Log confusion matrix as image
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        results['confusion_matrix'],
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Normal', 'Anomaly'],
        yticklabels=['Normal', 'Anomaly']
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    writer.add_figure('ConfusionMatrix', plt.gcf(), 0)
    
    # Log to wandb if available
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"ConfusionMatrix": wandb.Image(plt.gcf())})
    
    plt.close()
    
    # Log ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(results['labels'], results['probabilities'])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {results["auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    writer.add_figure('ROC_Curve', plt.gcf(), 0)
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"ROC_Curve": wandb.Image(plt.gcf())})
    
    plt.close()
    
    # Log precision-recall curve
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(results['labels'], results['probabilities'])
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    writer.add_figure('PrecisionRecall_Curve', plt.gcf(), 0)
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"PrecisionRecall_Curve": wandb.Image(plt.gcf())})
    
    plt.close()
    
    # Close writer
    writer.close()
    
    # Print results
    print("\n" + "="*50)
    print("Test Results")
    print("="*50)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print(f"AUC:       {results['auc']:.4f}")
    print(f"\nTensorBoard logs saved to: {log_dir}")
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    print("\nClassification Report:")
    print(classification_report(
        results['labels'],
        results['predictions'],
        target_names=['Normal', 'Anomaly']
    ))
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
        if wandb.run:
            print(f"\nView test results at: {wandb.run.url}")
            print(f"Project dashboard: https://wandb.ai/{wandb.run.entity or 'YOUR_USERNAME'}/{wandb.run.project}")
        else:
            print("\nW&B run completed")


if __name__ == '__main__':
    main()


"""
Model assessment script with comprehensive reporting using Weights & Biases.
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from datetime import datetime
import os
import glob
from pathlib import Path
from typing import List, Dict, Tuple

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

from src.data import load_data, preprocess_data, NetworkAnomalyDataset, create_dataloader
from src.models import create_model
from src.utils import load_config, get_device


def plot_class_balance(y_train, y_test, y_val=None):
    """
    Create histogram of class balance.
    
    Args:
        y_train: Training labels
        y_test: Test labels
        y_val: Validation labels (optional)
        
    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(1, 2 if y_val is None else 3, figsize=(15, 5))
    
    datasets = [
        ("Training", y_train),
        ("Test", y_test)
    ]
    if y_val is not None:
        datasets.append(("Validation", y_val))
    
    for idx, (name, labels) in enumerate(datasets):
        ax = axes[idx] if len(datasets) > 1 else axes
        unique, counts = np.unique(labels, return_counts=True)
        bars = ax.bar(
            [f"Class {int(u)}" for u in unique],
            counts,
            color=['#3498db', '#e74c3c'],
            alpha=0.7,
            edgecolor='black'
        )
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold'
            )
        
        # Add percentage labels
        total = len(labels)
        for i, (u, count) in enumerate(zip(unique, counts)):
            percentage = (count / total) * 100
            ax.text(
                i,
                count + total * 0.01,
                f'{percentage:.1f}%',
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        ax.set_title(f'{name} Set Class Distribution', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xlabel('Class Label', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def predict_on_data(model, data_loader, device):
    """
    Run model predictions on a dataset.
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for the dataset
        device: Device to run on
        
    Returns:
        Tuple of (predictions, probabilities, true_labels)
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def plot_true_vs_predicted(y_true, y_pred, y_probs, dataset_name):
    """
    Create scatter plot of true vs predicted class labels.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities
        dataset_name: Name of the dataset
        
    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot: True vs Predicted
    ax1 = axes[0]
    scatter = ax1.scatter(
        y_true,
        y_pred,
        c=y_probs,
        cmap='RdYlBu',
        alpha=0.6,
        s=50,
        edgecolors='black',
        linewidths=0.5
    )
    ax1.set_xlabel('True Label', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Label', fontsize=12, fontweight='bold')
    ax1.set_title(f'{dataset_name}: True vs Predicted Labels', fontsize=14, fontweight='bold')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Prediction Probability')
    
    # Add confusion matrix annotations
    cm = confusion_matrix(y_true, y_pred)
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                    fontsize=14, fontweight='bold', color='white' if cm[i, j] > cm.max()/2 else 'black')
    
    # Scatter plot: True vs Probability
    ax2 = axes[1]
    colors = ['#3498db' if label == 0 else '#e74c3c' for label in y_true]
    ax2.scatter(
        y_true,
        y_probs,
        c=colors,
        alpha=0.6,
        s=50,
        edgecolors='black',
        linewidths=0.5
    )
    ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
    ax2.set_xlabel('True Label', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Prediction Probability', fontsize=12, fontweight='bold')
    ax2.set_title(f'{dataset_name}: True Label vs Prediction Probability', fontsize=14, fontweight='bold')
    ax2.set_xticks([0, 1])
    ax2.set_ylim([-0.05, 1.05])
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig


def calculate_metrics(y_true, y_pred, y_probs):
    """Calculate classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_probs),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }


def find_fold_checkpoints(checkpoint_dir: str) -> List[Tuple[int, str]]:
    """
    Find all fold checkpoints.
    
    Args:
        checkpoint_dir: Base checkpoint directory
        
    Returns:
        List of (fold_number, checkpoint_path) tuples, sorted by fold number
    """
    fold_checkpoints = []
    pattern = os.path.join(checkpoint_dir, "fold_*", "best_model.pt")
    
    for checkpoint_path in glob.glob(pattern):
        # Extract fold number from path
        fold_dir = os.path.dirname(checkpoint_path)
        fold_name = os.path.basename(fold_dir)
        try:
            fold_num = int(fold_name.split('_')[1])
            fold_checkpoints.append((fold_num, checkpoint_path))
        except (ValueError, IndexError):
            continue
    
    # Sort by fold number
    fold_checkpoints.sort(key=lambda x: x[0])
    return fold_checkpoints


def plot_fold_comparison(fold_metrics: List[Dict], metric_name: str):
    """
    Create bar plot comparing metrics across folds.
    
    Args:
        fold_metrics: List of metrics dictionaries for each fold
        metric_name: Name of metric to plot
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    folds = [f"Fold {m['fold']}" for m in fold_metrics]
    train_values = [m['train_metrics'][metric_name] for m in fold_metrics]
    test_values = [m['test_metrics'][metric_name] for m in fold_metrics]
    val_values = [m['val_metrics'][metric_name] for m in fold_metrics]
    
    x = np.arange(len(folds))
    width = 0.25
    
    bars1 = ax.bar(x - width, train_values, width, label='Train', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, val_values, width, label='Validation', color='#2ecc71', alpha=0.8)
    bars3 = ax.bar(x + width, test_values, width, label='Test', color='#e74c3c', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name.capitalize(), fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name.capitalize()} Comparison Across Folds', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def aggregate_fold_metrics(fold_metrics: List[Dict]) -> Dict:
    """
    Calculate aggregate statistics across folds.
    
    Args:
        fold_metrics: List of metrics dictionaries for each fold
        
    Returns:
        Dictionary with mean and std for each metric
    """
    metrics_to_aggregate = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    datasets = ['train', 'test', 'val']
    
    aggregated = {}
    for dataset in datasets:
        aggregated[dataset] = {}
        for metric in metrics_to_aggregate:
            values = [m[f'{dataset}_metrics'][metric] for m in fold_metrics]
            aggregated[dataset][metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description='Assess trained model and generate W&B report')
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
        help='Path to single model checkpoint (if not using CV folds)'
    )
    parser.add_argument(
        '--use-folds',
        action='store_true',
        help='Assess models from all CV folds (default: auto-detect)'
    )
    parser.add_argument(
        '--project',
        type=str,
        default=None,
        help='W&B project name (overrides config)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='W&B run name (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize wandb
    wandb_config = config.get('logging', {})
    use_wandb = wandb_config.get('use_wandb', True)
    
    # Get device
    device = get_device(
        use_cuda=config['device']['use_cuda'],
        cuda_device=config['device']['cuda_device']
    )
    print(f"Using device: {device}")
    
    # Determine if we should use fold checkpoints
    checkpoint_dir = config['checkpoint']['save_dir']
    fold_checkpoints = find_fold_checkpoints(checkpoint_dir)
    
    use_folds = args.use_folds or (len(fold_checkpoints) > 0 and args.checkpoint is None)
    
    if use_folds and len(fold_checkpoints) == 0:
        print("Warning: --use-folds specified but no fold checkpoints found. Falling back to single checkpoint.")
        use_folds = False
    
    if use_folds:
        print(f"Found {len(fold_checkpoints)} fold checkpoints:")
        for fold_num, path in fold_checkpoints:
            print(f"  Fold {fold_num}: {path}")
    else:
        if args.checkpoint is None:
            args.checkpoint = 'checkpoints/best_model.pt'
        print(f"Using single checkpoint: {args.checkpoint}")
    
    # Initialize wandb
    if use_wandb:
        project = args.project or wandb_config.get('wandb_project', 'network-anomaly-detection')
        entity = wandb_config.get('wandb_entity', None)
        run_name = args.name or wandb_config.get('wandb_run_name') or f"assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if use_folds:
            run_name = f"cv_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=config,
            job_type="assessment",
            sync_tensorboard=True  # Sync TensorBoard logs to wandb
        )
        # Patch TensorBoard to automatically sync logs (only if not already patched)
        safe_patch_tensorboard(config['logging']['tensorboard_dir'])
        print(f"Initialized W&B run: {run_name}")
        print(f"TensorBoard logs will be synced to wandb")
    
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
    
    # Split data (same way as training) - use consistent split for all folds
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
    
    # Plot class balance
    print("Generating class balance histogram...")
    balance_fig = plot_class_balance(y_train, y_test, y_val)
    if use_wandb:
        wandb.log({"Dataset/Class_Balance": wandb.Image(balance_fig)})
    plt.savefig('class_balance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create data loaders (same for all models)
    train_dataset = NetworkAnomalyDataset(X_train, y_train)
    test_dataset = NetworkAnomalyDataset(X_test, y_test)
    val_dataset = NetworkAnomalyDataset(X_val, y_val)
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    test_loader = create_dataloader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    model_config = config['model'].copy()
    model_config['input_dim'] = X_processed.shape[1]
    
    if use_folds:
        # Assess each fold model
        fold_metrics_list = []
        
        for fold_num, checkpoint_path in fold_checkpoints:
            print(f"\n{'='*70}")
            print(f"Assessing Fold {fold_num}")
            print(f"{'='*70}")
            print(f"Loading model from {checkpoint_path}...")
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model = create_model(model_config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            
            # Run predictions
            print(f"Running predictions for Fold {fold_num}...")
            train_preds, train_probs, train_labels = predict_on_data(model, train_loader, device)
            test_preds, test_probs, test_labels = predict_on_data(model, test_loader, device)
            val_preds, val_probs, val_labels = predict_on_data(model, val_loader, device)
            
            # Calculate metrics
            train_metrics = calculate_metrics(train_labels, train_preds, train_probs)
            test_metrics = calculate_metrics(test_labels, test_preds, test_probs)
            val_metrics = calculate_metrics(val_labels, val_preds, val_probs)
            
            fold_metrics_list.append({
                'fold': fold_num,
                'checkpoint_path': checkpoint_path,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'val_metrics': val_metrics,
                'train_preds': train_preds,
                'train_probs': train_probs,
                'train_labels': train_labels,
                'test_preds': test_preds,
                'test_probs': test_probs,
                'test_labels': test_labels,
                'val_preds': val_preds,
                'val_probs': val_probs,
                'val_labels': val_labels
            })
            
            # Create scatter plots for this fold
            train_scatter_fig = plot_true_vs_predicted(
                train_labels, train_preds, train_probs, f"Training (Fold {fold_num})"
            )
            test_scatter_fig = plot_true_vs_predicted(
                test_labels, test_preds, test_probs, f"Test (Fold {fold_num})"
            )
            val_scatter_fig = plot_true_vs_predicted(
                val_labels, val_preds, val_probs, f"Validation (Fold {fold_num})"
            )
            
            # Log to wandb
            if use_wandb:
                wandb.log({
                    f"Fold_{fold_num}/Predictions/Train_True_vs_Predicted": wandb.Image(train_scatter_fig),
                    f"Fold_{fold_num}/Predictions/Test_True_vs_Predicted": wandb.Image(test_scatter_fig),
                    f"Fold_{fold_num}/Predictions/Val_True_vs_Predicted": wandb.Image(val_scatter_fig)
                })
                
                # Log metrics per fold
                wandb.log({
                    f"Fold_{fold_num}/Metrics/Train_Accuracy": train_metrics['accuracy'],
                    f"Fold_{fold_num}/Metrics/Train_Precision": train_metrics['precision'],
                    f"Fold_{fold_num}/Metrics/Train_Recall": train_metrics['recall'],
                    f"Fold_{fold_num}/Metrics/Train_F1": train_metrics['f1'],
                    f"Fold_{fold_num}/Metrics/Train_AUC": train_metrics['auc'],
                    f"Fold_{fold_num}/Metrics/Test_Accuracy": test_metrics['accuracy'],
                    f"Fold_{fold_num}/Metrics/Test_Precision": test_metrics['precision'],
                    f"Fold_{fold_num}/Metrics/Test_Recall": test_metrics['recall'],
                    f"Fold_{fold_num}/Metrics/Test_F1": test_metrics['f1'],
                    f"Fold_{fold_num}/Metrics/Test_AUC": test_metrics['auc'],
                    f"Fold_{fold_num}/Metrics/Val_Accuracy": val_metrics['accuracy'],
                    f"Fold_{fold_num}/Metrics/Val_Precision": val_metrics['precision'],
                    f"Fold_{fold_num}/Metrics/Val_Recall": val_metrics['recall'],
                    f"Fold_{fold_num}/Metrics/Val_F1": val_metrics['f1'],
                    f"Fold_{fold_num}/Metrics/Val_AUC": val_metrics['auc']
                })
                
                # Log confusion matrices per fold
                for name, cm in [("Train", train_metrics['confusion_matrix']),
                                 ("Test", test_metrics['confusion_matrix']),
                                 ("Val", val_metrics['confusion_matrix'])]:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt='d',
                        cmap='Blues',
                        xticklabels=['Normal', 'Anomaly'],
                        yticklabels=['Normal', 'Anomaly'],
                        ax=ax
                    )
                    ax.set_title(f'Fold {fold_num} - {name} Set Confusion Matrix', fontsize=14, fontweight='bold')
                    ax.set_ylabel('True Label', fontsize=12)
                    ax.set_xlabel('Predicted Label', fontsize=12)
                    plt.tight_layout()
                    wandb.log({f"Fold_{fold_num}/Confusion_Matrix/{name}": wandb.Image(fig)})
                    plt.close()
            
            plt.close('all')
        
        # Aggregate results across folds
        print("\n" + "="*70)
        print("Aggregating Results Across Folds")
        print("="*70)
        
        aggregated = aggregate_fold_metrics(fold_metrics_list)
        
        # Create comparison plots
        print("Generating fold comparison plots...")
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            comparison_fig = plot_fold_comparison(fold_metrics_list, metric)
            if use_wandb:
                wandb.log({f"Comparison/{metric.capitalize()}_Across_Folds": wandb.Image(comparison_fig)})
            comparison_fig.savefig(f'fold_comparison_{metric}.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # Create summary table for all folds
        if use_wandb:
            summary_data = []
            for fold_data in fold_metrics_list:
                fold_num = fold_data['fold']
                summary_data.append([
                    f"Fold {fold_num} - Train",
                    fold_data['train_metrics']['accuracy'],
                    fold_data['train_metrics']['precision'],
                    fold_data['train_metrics']['recall'],
                    fold_data['train_metrics']['f1'],
                    fold_data['train_metrics']['auc']
                ])
                summary_data.append([
                    f"Fold {fold_num} - Test",
                    fold_data['test_metrics']['accuracy'],
                    fold_data['test_metrics']['precision'],
                    fold_data['test_metrics']['recall'],
                    fold_data['test_metrics']['f1'],
                    fold_data['test_metrics']['auc']
                ])
                summary_data.append([
                    f"Fold {fold_num} - Val",
                    fold_data['val_metrics']['accuracy'],
                    fold_data['val_metrics']['precision'],
                    fold_data['val_metrics']['recall'],
                    fold_data['val_metrics']['f1'],
                    fold_data['val_metrics']['auc']
                ])
            
            summary_table = wandb.Table(
                columns=["Dataset", "Accuracy", "Precision", "Recall", "F1", "AUC"],
                data=summary_data
            )
            wandb.log({"Metrics/All_Folds_Summary_Table": summary_table})
            
            # Log aggregated statistics
            for dataset in ['train', 'test', 'val']:
                for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                    stats = aggregated[dataset][metric]
                    wandb.log({
                        f"Aggregated/{dataset.capitalize()}_{metric}_mean": stats['mean'],
                        f"Aggregated/{dataset.capitalize()}_{metric}_std": stats['std'],
                        f"Aggregated/{dataset.capitalize()}_{metric}_min": stats['min'],
                        f"Aggregated/{dataset.capitalize()}_{metric}_max": stats['max']
                    })
        
        # Print aggregated results
        print("\nAggregated Metrics Across Folds:")
        for dataset in ['train', 'test', 'val']:
            print(f"\n{dataset.capitalize()} Set:")
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                stats = aggregated[dataset][metric]
                print(f"  {metric.capitalize()}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                      f"(min: {stats['min']:.4f}, max: {stats['max']:.4f})")
        
        # Print per-fold results
        print("\n" + "="*70)
        print("Per-Fold Results")
        print("="*70)
        for fold_data in fold_metrics_list:
            fold_num = fold_data['fold']
            print(f"\nFold {fold_num}:")
            for dataset_name, metrics in [("Training", fold_data['train_metrics']),
                                          ("Test", fold_data['test_metrics']),
                                          ("Validation", fold_data['val_metrics'])]:
                print(f"  {dataset_name}:")
                print(f"    Accuracy:  {metrics['accuracy']:.4f}")
                print(f"    Precision: {metrics['precision']:.4f}")
                print(f"    Recall:    {metrics['recall']:.4f}")
                print(f"    F1 Score:  {metrics['f1']:.4f}")
                print(f"    AUC:       {metrics['auc']:.4f}")
    
    else:
        # Single model assessment (original behavior)
        print(f"Loading model from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        model = create_model(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Run predictions
        print("Running predictions on training data...")
        train_preds, train_probs, train_labels = predict_on_data(model, train_loader, device)
        
        print("Running predictions on test data...")
        test_preds, test_probs, test_labels = predict_on_data(model, test_loader, device)
        
        print("Running predictions on validation data...")
        val_preds, val_probs, val_labels = predict_on_data(model, val_loader, device)
        
        # Calculate metrics
        print("Calculating metrics...")
        train_metrics = calculate_metrics(train_labels, train_preds, train_probs)
        test_metrics = calculate_metrics(test_labels, test_preds, test_probs)
        val_metrics = calculate_metrics(val_labels, val_preds, val_probs)
        
        # Create scatter plots
        print("Generating scatter plots...")
        train_scatter_fig = plot_true_vs_predicted(train_labels, train_preds, train_probs, "Training")
        test_scatter_fig = plot_true_vs_predicted(test_labels, test_preds, test_probs, "Test")
        val_scatter_fig = plot_true_vs_predicted(val_labels, val_preds, val_probs, "Validation")
        
        # Save plots
        train_scatter_fig.savefig('train_predictions.png', dpi=150, bbox_inches='tight')
        test_scatter_fig.savefig('test_predictions.png', dpi=150, bbox_inches='tight')
        val_scatter_fig.savefig('val_predictions.png', dpi=150, bbox_inches='tight')
        
        # Log to wandb
        if use_wandb:
            print("Logging to Weights & Biases...")
            
            # Log scatter plots
            wandb.log({
                "Predictions/Train_True_vs_Predicted": wandb.Image(train_scatter_fig),
                "Predictions/Test_True_vs_Predicted": wandb.Image(test_scatter_fig),
                "Predictions/Val_True_vs_Predicted": wandb.Image(val_scatter_fig)
            })
            
            # Log metrics
            wandb.log({
                "Metrics/Train_Accuracy": train_metrics['accuracy'],
                "Metrics/Train_Precision": train_metrics['precision'],
                "Metrics/Train_Recall": train_metrics['recall'],
                "Metrics/Train_F1": train_metrics['f1'],
                "Metrics/Train_AUC": train_metrics['auc'],
                "Metrics/Test_Accuracy": test_metrics['accuracy'],
                "Metrics/Test_Precision": test_metrics['precision'],
                "Metrics/Test_Recall": test_metrics['recall'],
                "Metrics/Test_F1": test_metrics['f1'],
                "Metrics/Test_AUC": test_metrics['auc'],
                "Metrics/Val_Accuracy": val_metrics['accuracy'],
                "Metrics/Val_Precision": val_metrics['precision'],
                "Metrics/Val_Recall": val_metrics['recall'],
                "Metrics/Val_F1": val_metrics['f1'],
                "Metrics/Val_AUC": val_metrics['auc']
            })
            
            # Log confusion matrices
            for name, cm in [("Train", train_metrics['confusion_matrix']),
                             ("Test", test_metrics['confusion_matrix']),
                             ("Val", val_metrics['confusion_matrix'])]:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=['Normal', 'Anomaly'],
                    yticklabels=['Normal', 'Anomaly'],
                    ax=ax
                )
                ax.set_title(f'{name} Set Confusion Matrix', fontsize=14, fontweight='bold')
                ax.set_ylabel('True Label', fontsize=12)
                ax.set_xlabel('Predicted Label', fontsize=12)
                plt.tight_layout()
                wandb.log({f"Confusion_Matrix/{name}": wandb.Image(fig)})
                plt.close()
            
            # Create summary table
            summary_table = wandb.Table(
                columns=["Dataset", "Accuracy", "Precision", "Recall", "F1", "AUC"],
                data=[
                    ["Train", train_metrics['accuracy'], train_metrics['precision'],
                     train_metrics['recall'], train_metrics['f1'], train_metrics['auc']],
                    ["Test", test_metrics['accuracy'], test_metrics['precision'],
                     test_metrics['recall'], test_metrics['f1'], test_metrics['auc']],
                    ["Validation", val_metrics['accuracy'], val_metrics['precision'],
                     val_metrics['recall'], val_metrics['f1'], val_metrics['auc']]
                ]
            )
            wandb.log({"Metrics/Summary_Table": summary_table})
        
        # Print results
        print("\n" + "="*70)
        print("Model Assessment Results")
        print("="*70)
        
        for name, metrics in [("Training", train_metrics),
                              ("Test", test_metrics),
                              ("Validation", val_metrics)]:
            print(f"\n{name} Set:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")
            print(f"  AUC:       {metrics['auc']:.4f}")
            print(f"  Confusion Matrix:")
            print(f"    {metrics['confusion_matrix']}")
    
    plt.close('all')
    
    if use_wandb:
        wandb.finish()
        if wandb.run:
            print(f"\nAssessment complete! View results at: {wandb.run.url}")
            print(f"Project dashboard: https://wandb.ai/{wandb.run.entity or 'YOUR_USERNAME'}/{wandb.run.project}")
        else:
            print("\nAssessment complete!")
    else:
        print("\nAssessment complete! (W&B logging disabled)")


if __name__ == '__main__':
    main()


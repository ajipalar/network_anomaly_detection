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
import wandb
from datetime import datetime

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
            probs = outputs.squeeze().cpu().numpy()
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
        default='checkpoints/best_model.pt',
        help='Path to model checkpoint'
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
    
    if use_wandb:
        project = args.project or wandb_config.get('wandb_project', 'network-anomaly-detection')
        entity = wandb_config.get('wandb_entity', None)
        run_name = args.name or wandb_config.get('wandb_run_name') or f"assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=config,
            job_type="assessment"
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
    
    # Split data (same way as training)
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
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model_config = config['model'].copy()
    model_config['input_dim'] = X_processed.shape[1]
    model = create_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Create data loaders
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
        print(f"\nAssessment complete! View results at: {wandb.run.url}")
    else:
        print("\nAssessment complete! (W&B logging disabled)")


if __name__ == '__main__':
    main()


"""
Testing script for evaluating trained models.
"""

import argparse
import torch
import numpy as np
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
            probs = outputs.squeeze().cpu().numpy()
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
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--test-data',
        type=str,
        default=None,
        help='Path to test data (if different from config)'
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
            job_type="testing"
        )
        print(f"Initialized W&B run: {run_name}")
    
    # Get device
    device = get_device(
        use_cuda=config['device']['use_cuda'],
        cuda_device=config['device']['cuda_device']
    )
    print(f"Using device: {device}")
    
    # Load data
    data_path = args.test_data or config['data']['data_path']
    print(f"Loading test data from {data_path}...")
    X, y = load_data(data_path)
    
    # Preprocess data (use scaler from training if available)
    # For now, we'll fit a new scaler (in production, load from training)
    X_processed, y_processed, _ = preprocess_data(X, y, fit_scaler=True)
    
    # Create test dataset
    test_dataset = NetworkAnomalyDataset(X_processed, y_processed)
    test_loader = create_dataloader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model_config = config['model'].copy()
    model_config['input_dim'] = X_processed.shape[1]
    model = create_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
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
        print(f"\nView test results at: {wandb.run.url if wandb.run else 'N/A'}")


if __name__ == '__main__':
    main()


# Weights & Biases Integration Guide

This project includes comprehensive Weights & Biases (wandb) integration for experiment tracking, visualization, and model assessment.

## Setup

### 1. Install wandb
```bash
pip install wandb
```

### 2. Login to wandb
```bash
wandb login
```
Enter your API key from https://wandb.ai/authorize

### 3. Configure in config.yaml
```yaml
logging:
  use_wandb: true
  wandb_project: "network-anomaly-detection"
  wandb_entity: "your-username"  # Optional, your wandb username/team
  wandb_run_name: null  # Auto-generated if null
```

## Features

### Training Logging

During training, the following metrics are automatically logged to wandb:

- **Loss/Train**: Training loss per epoch
- **Loss/Validation**: Validation loss per epoch
- **Train/BatchLoss**: Batch-level training loss
- **LearningRate**: Learning rate per epoch
- **Best/ValidationLoss**: Best validation loss achieved

### Testing Logging

During model evaluation (`test.py`), the following are logged:

- **Metrics/Accuracy**: Test accuracy
- **Metrics/Precision**: Test precision
- **Metrics/Recall**: Test recall
- **Metrics/F1_Score**: Test F1 score
- **Metrics/AUC**: Test ROC AUC
- **ConfusionMatrix**: Confusion matrix visualization
- **ROC_Curve**: ROC curve plot
- **PrecisionRecall_Curve**: Precision-Recall curve plot

### Model Assessment (`assess_model.py`)

The assessment script generates a comprehensive report with:

1. **Class Balance Histogram**
   - Shows distribution of positive/negative classes
   - Separate plots for train/test/validation sets
   - Percentage labels on bars

2. **Scatter Plots**
   - True vs Predicted labels (with probability coloring)
   - True vs Prediction Probability
   - Separate plots for train/test/validation sets
   - **Per-fold plots** when using cross-validation

3. **Metrics Summary**
   - Comprehensive metrics table comparing all datasets
   - Confusion matrices for each dataset
   - All metrics logged as wandb scalars

4. **Cross-Validation Assessment** (when using `--use-folds`)
   - Automatically detects and assesses all fold models
   - Per-fold metrics and visualizations
   - Fold comparison bar charts
   - Aggregated statistics (mean, std, min, max) across folds
   - Comprehensive summary table with all folds

## Usage

### Training with wandb
```bash
python train.py --config config.yaml
```
Metrics are automatically logged if `use_wandb: true` in config.

### Testing with wandb
```bash
python test.py --config config.yaml --checkpoint checkpoints/best_model.pt
```

### Model Assessment

**Single Model:**
```bash
python assess_model.py --config config.yaml --checkpoint checkpoints/best_model.pt
```

**All CV Folds (Auto-detects):**
```bash
python assess_model.py --config config.yaml --use-folds
```

The script automatically detects fold checkpoints (`checkpoints/fold_*/best_model.pt`) and assesses all of them if found.

### Custom Project/Name
```bash
python assess_model.py \
    --config config.yaml \
    --checkpoint checkpoints/best_model.pt \
    --project "my-custom-project" \
    --name "experiment-1"
```

## Viewing Results

1. **Web Interface**: 
   - Go to https://wandb.ai
   - Navigate to your project
   - View all runs, metrics, and visualizations

2. **Command Line**:
   ```bash
   wandb status  # Check current status
   wandb sync runs/  # Sync local runs
   ```

## Dual Logging (TensorBoard + wandb)

The project logs to both TensorBoard and wandb simultaneously:

- **TensorBoard**: Local visualization, detailed batch-level metrics
- **wandb**: Cloud-based tracking, collaboration, model versioning

You can use either or both:
- Set `use_wandb: false` to disable wandb (TensorBoard still works)
- TensorBoard always works regardless of wandb settings

## Best Practices

1. **Project Organization**:
   - Use descriptive project names
   - Group related experiments in the same project
   - Use run names to identify specific experiments

2. **Entity/Team**:
   - Set `wandb_entity` to share runs with your team
   - Leave as `null` to use your personal account

3. **Run Names**:
   - Auto-generated with timestamps by default
   - Override with `--name` flag for custom names
   - Use descriptive names: "baseline-model", "augmented-data", etc.

4. **Assessment Reports**:
   - Run `assess_model.py` after each training session
   - Compare reports across different model versions
   - Use wandb's comparison features to analyze improvements

## Troubleshooting

- **"wandb not found"**: Install with `pip install wandb`
- **"Not logged in"**: Run `wandb login`
- **"Project not found"**: Project is created automatically on first run
- **Disable wandb**: Set `use_wandb: false` in config.yaml

## Integration with Google Colab

wandb works seamlessly in Google Colab:

```python
# In colab_train.ipynb
!pip install wandb
!wandb login  # Enter your API key

# Then run training as normal
!python train.py --config config.yaml
```

Results will appear in your wandb dashboard automatically.


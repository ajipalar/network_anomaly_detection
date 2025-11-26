# TensorBoard Logging Guide

This project includes comprehensive TensorBoard logging for monitoring training and evaluation metrics.

## Overview

TensorBoard logs are automatically created during:
- **Training**: Real-time loss tracking, learning rate monitoring
- **Testing**: Complete evaluation metrics with visualizations
- **Cross-Validation**: Separate logs for each fold

## Log Locations

All TensorBoard logs are saved in the `runs/` directory (configurable in `config.yaml`):
```
runs/
├── train_20240101_120000/    # Training run
├── test_20240101_130000/      # Test evaluation
├── cv_fold_1_20240101_140000/ # CV fold 1
├── cv_fold_2_20240101_150000/ # CV fold 2
└── ...
```

## Training Metrics

During training, the following metrics are logged:

### Scalar Metrics
- **Train/BatchLoss**: Loss for each training batch (updated every batch)
- **Loss/Train**: Average training loss per epoch
- **Loss/Validation**: Validation loss per epoch
- **LearningRate**: Current learning rate per epoch
- **Best/ValidationLoss**: Best validation loss achieved (only when new best is found)

### Usage
```bash
# Start TensorBoard
tensorboard --logdir runs

# View specific run
tensorboard --logdir runs/train_20240101_120000
```

## Testing Metrics

During model evaluation, comprehensive metrics are logged:

### Scalar Metrics
- **Metrics/Accuracy**: Test accuracy
- **Metrics/Precision**: Test precision
- **Metrics/Recall**: Test recall
- **Metrics/F1_Score**: Test F1 score
- **Metrics/AUC**: Test ROC AUC score

### Visualizations
- **ConfusionMatrix**: Heatmap of confusion matrix
- **ROC_Curve**: Receiver Operating Characteristic curve
- **PrecisionRecall_Curve**: Precision-Recall curve

### Usage
```bash
# After running test.py
tensorboard --logdir runs

# Navigate to the test run in TensorBoard UI
```

## Cross-Validation Logging

When using K-fold cross validation, each fold gets its own TensorBoard log:
- Separate training/validation curves for each fold
- Easy comparison between folds
- Aggregate metrics across all folds

## Viewing Logs

### Local Machine

1. **Start TensorBoard:**
   ```bash
   tensorboard --logdir runs
   ```

2. **Open browser:**
   - Navigate to `http://localhost:6006`
   - Or use the URL shown in terminal

3. **View specific run:**
   ```bash
   tensorboard --logdir runs/train_20240101_120000
   ```

### Google Colab

1. **Load TensorBoard extension:**
   ```python
   %load_ext tensorboard
   ```

2. **Start TensorBoard:**
   ```python
   %tensorboard --logdir runs
   ```

3. **TensorBoard will appear inline in the notebook**

## Tips

1. **Compare Runs**: Use TensorBoard's comparison feature to overlay multiple runs
2. **Smoothing**: Adjust smoothing slider to reduce noise in loss curves
3. **Refresh**: TensorBoard auto-refreshes, but you can manually refresh
4. **Filtering**: Use the filter box to find specific metrics quickly

## Configuration

TensorBoard directory can be configured in `config.yaml`:
```yaml
logging:
  tensorboard_dir: "runs"  # Change this to customize log location
```

## Troubleshooting

- **No logs appearing**: Check that `runs/` directory exists and contains subdirectories
- **Can't connect**: Ensure TensorBoard is running and port 6006 is available
- **Outdated logs**: TensorBoard auto-refreshes, but you may need to reload the page
- **Too many runs**: Clean up old runs or use `--logdir_suffix` to filter

## Example Workflow

1. Train model:
   ```bash
   python train.py --config config.yaml
   ```

2. Start TensorBoard (in separate terminal):
   ```bash
   tensorboard --logdir runs
   ```

3. Monitor training in real-time at `http://localhost:6006`

4. After training, test model:
   ```bash
   python test.py --config config.yaml --checkpoint checkpoints/best_model.pt
   ```

5. View test metrics in TensorBoard (same URL, new run will appear)


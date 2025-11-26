# Network Anomaly Detection

A PyTorch-based supervised learning project for detecting network anomalies in embedded systems.

## Features

- **PyTorch Implementation**: Modern deep learning framework
- **K-fold Cross Validation**: Robust model evaluation
- **Model Checkpointing**: Save and resume training
- **Data Augmentation**: Gaussian noise augmentation with class resampling
- **Hugging Face Integration**: Upload models to Hugging Face Hub
- **Google Colab Support**: Ready-to-use Colab notebook
- **Comprehensive Testing**: Evaluation scripts with metrics

## Project Structure

```
network_anomaly_detection/
├── data/
│   └── raw/
│       └── embedded_system_network_security_dataset.csv
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py          # Data loading and preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   └── model.py           # Model definitions
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py         # Training utilities
│   ├── validation/
│   │   ├── __init__.py
│   │   └── cross_validation.py # K-fold CV
│   ├── utils/
│   │   ├── __init__.py
│   │   └── config.py          # Configuration management
│   └── huggingface/
│       ├── __init__.py
│       └── upload.py          # HF Hub integration
├── checkpoints/               # Model checkpoints (created during training)
├── logs/                      # Training logs (created during training)
├── config.yaml                # Configuration file
├── requirements.txt           # Python dependencies
├── train.py                   # Main training script
├── test.py                    # Testing script
├── assess_model.py           # Model assessment with W&B reporting
├── upload_to_hub.py          # Hugging Face upload script
├── colab_train.ipynb         # Google Colab notebook
└── README.md                  # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd network_anomaly_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to customize:
- Model architecture (hidden layers, dropout, etc.)
- Training hyperparameters (learning rate, batch size, epochs)
- Cross validation settings (number of folds)
- Checkpoint settings
- Hugging Face repository details
- Data augmentation settings

## Data Augmentation

The project includes a data augmentation pipeline that:
- Adds Gaussian noise to numerical features
- Resamples positive class to achieve ~30% ratio
- Holds out test data before augmentation
- Uses augmented data for training/validation, original test data for evaluation

### Running Augmentation

```bash
python augment_data.py --noise-scale 0.1 --target-positive-ratio 0.3
```

This will create:
- `data/augmented/train_data_augmented.csv` - Augmented training data
- `data/augmented/val_data.csv` - Validation data
- `data/augmented/test_data.csv` - Original test data (held out)

### Using Augmented Data

Set `use_augmented_data: true` in `config.yaml` to automatically use augmented data for training. The test script will automatically use the original test data for evaluation.

## Weights & Biases Integration

The project includes comprehensive Weights & Biases (wandb) logging:

- **Automatic logging** during training (loss, learning rate, metrics)
- **Model assessment** with comprehensive reports (`assess_model.py`)
- **Visualizations**: Class balance, confusion matrices, ROC curves, scatter plots
- **Dual logging**: Both TensorBoard and wandb for maximum flexibility
- **TensorBoard sync**: TensorBoard logs are automatically synced to wandb dashboard

### Dashboard

View live training metrics and model assessments:
- **Project Dashboard**: [https://wandb.ai/ajikarunia-university-of-california/network-anomaly-detection](https://wandb.ai/ajikarunia-university-of-california/network-anomaly-detection)
  *(Replace `YOUR_USERNAME` with your wandb username or team name from `config.yaml`)*

All TensorBoard logs are automatically synced to the wandb dashboard, so you can view both native wandb metrics and TensorBoard visualizations in one place.

### Setup

1. Install wandb: `pip install wandb`
2. Login: `wandb login` (get your API key from https://wandb.ai)
3. Configure in `config.yaml`:
   ```yaml
   logging:
     use_wandb: true
     wandb_project: "network-anomaly-detection"
     wandb_entity: "your-username"  # Optional
   ```

### Model Assessment

Generate comprehensive model assessment report:
```bash
python assess_model.py --config config.yaml --checkpoint checkpoints/best_model.pt
```

This creates:
- Class balance histogram
- Scatter plots (true vs predicted) for train/test/val
- Comprehensive metrics table
- All visualizations logged to wandb

## Usage

### Training

**Standard training (train/val/test split):**
```bash
python train.py --config config.yaml
```

**K-fold cross validation:**
```bash
python train.py --config config.yaml --use-cv
```

**Resume from checkpoint:**
```bash
python train.py --config config.yaml --resume checkpoints/checkpoint_epoch_50.pt
```

### Testing

```bash
python test.py --config config.yaml --checkpoint checkpoints/best_model.pt
```

### Model Assessment

**Single Model Assessment:**
```bash
python assess_model.py --config config.yaml --checkpoint checkpoints/best_model.pt
```

**Cross-Validation Fold Assessment (Auto-detects fold checkpoints):**
```bash
python assess_model.py --config config.yaml --use-folds
```

The assessment script automatically detects if fold checkpoints exist and assesses all folds. It generates:

- **Class balance histogram**: Distribution of positive/negative classes
- **Per-fold scatter plots**: True vs predicted labels for each fold
- **Fold comparison plots**: Bar charts comparing metrics across folds
- **Aggregated statistics**: Mean, std, min, max across all folds
- **Comprehensive metrics**: Accuracy, Precision, Recall, F1, AUC for each fold
- **All visualizations logged to Weights & Biases**

### Upload to Hugging Face

1. Get your Hugging Face token from https://huggingface.co/settings/tokens
2. Update `repo_id` in `config.yaml` or use `--repo-id` flag
3. Run:
```bash
python upload_to_hub.py --checkpoint checkpoints/best_model.pt --token YOUR_TOKEN
```

## Google Colab

1. Open `colab_train.ipynb` in Google Colab
2. Follow the cells sequentially:
   - Install dependencies
   - Upload data
   - Configure training
   - Train model
   - Test and upload

## Model Architecture

The default model is a feedforward neural network with:
- Input layer: 18 features
- Hidden layers: [128, 64, 32] neurons
- Output layer: 1 neuron (binary classification)
- Activation: ReLU
- Dropout: 0.3

## Checkpointing

Checkpoints are saved in the `checkpoints/` directory:
- `best_model.pt`: Best model based on validation loss
- `last_model.pt`: Final model after training
- `checkpoint_epoch_N.pt`: Checkpoints at regular intervals

## Metrics

The testing script evaluates:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Confusion Matrix

## TensorBoard Logging

The project includes comprehensive TensorBoard logging:

### Training Metrics
- **Train/BatchLoss**: Loss for each training batch
- **Loss/Train**: Average training loss per epoch
- **Loss/Validation**: Validation loss per epoch
- **LearningRate**: Learning rate per epoch
- **Best/ValidationLoss**: Best validation loss achieved

### Testing Metrics
- **Metrics/Accuracy**: Test accuracy
- **Metrics/Precision**: Test precision
- **Metrics/Recall**: Test recall
- **Metrics/F1_Score**: Test F1 score
- **Metrics/AUC**: Test ROC AUC
- **ConfusionMatrix**: Confusion matrix visualization
- **ROC_Curve**: ROC curve plot
- **PrecisionRecall_Curve**: Precision-Recall curve plot

### Viewing Logs

**Local:**
```bash
tensorboard --logdir runs
```

**Google Colab:**
```python
%load_ext tensorboard
%tensorboard --logdir runs
```

Then open the URL shown in the output (typically `http://localhost:6006`)

## License

Copyright 2025 Ajikarunia Palar

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Data

The [network traffic anomaly detection dataset](https://www.kaggle.com/datasets/ziya07/network-traffic-anomaly-detection-dataset?resource=download) was downloaded from kaggle.com on 11-25-25. 

## Contributing

[Add contribution guidelines here]

## Contact

[Add contact information here]


# Quick Start Guide

## Local Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Update configuration:**
   - Edit `config.yaml` to set your Hugging Face repository ID (if uploading)
   - Adjust model architecture and training hyperparameters as needed

3. **Train model:**
   ```bash
   # Standard training
   python train.py --config config.yaml
   
   # With K-fold cross validation
   python train.py --config config.yaml --use-cv
   ```

4. **Test model:**
   ```bash
   python test.py --config config.yaml --checkpoint checkpoints/best_model.pt
   ```

5. **Upload to Hugging Face:**
   ```bash
   python upload_to_hub.py --checkpoint checkpoints/best_model.pt --token YOUR_HF_TOKEN
   ```

## Google Colab Setup

1. **Open the notebook:**
   - Upload `colab_train.ipynb` to Google Colab
   - Or clone your repository in Colab

2. **Run cells sequentially:**
   - Install dependencies (Cell 1-2)
   - Upload data (Cell 3)
   - Check GPU (Cell 4)
   - Configure training (Cell 5)
   - Train model (Cell 6)
   - Test model (Cell 7)
   - Upload to Hugging Face (Cell 8-9)

3. **Monitor training:**
   - Use TensorBoard (Cell 10) to visualize training progress

## Key Features

- **Checkpointing**: Models are automatically saved during training
  - `checkpoints/best_model.pt`: Best model based on validation loss
  - `checkpoints/last_model.pt`: Final model after training
  - `checkpoints/checkpoint_epoch_N.pt`: Periodic checkpoints

- **Resume Training**: 
   ```bash
   python train.py --config config.yaml --resume checkpoints/checkpoint_epoch_50.pt
   ```

- **K-fold Cross Validation**: Robust evaluation with 5-fold CV
   ```bash
   python train.py --config config.yaml --use-cv
   ```

## Configuration Tips

- **Model Architecture**: Adjust `hidden_dims` in `config.yaml` to change network size
- **Training**: Modify `learning_rate`, `batch_size`, `num_epochs` as needed
- **Early Stopping**: Configure `early_stopping_patience` to prevent overfitting
- **Device**: Set `use_cuda: false` to force CPU training

## Troubleshooting

- **CUDA out of memory**: Reduce `batch_size` in config
- **Slow training**: Enable GPU in Colab (Runtime → Change runtime type → GPU)
- **Import errors**: Ensure all dependencies are installed (`pip install -r requirements.txt`)


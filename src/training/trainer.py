"""
Training utilities and trainer class.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Trainer:
    """Trainer class for model training and validation."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        checkpoint_dir: str,
        tensorboard_dir: Optional[str] = None,
        run_name: Optional[str] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            tensorboard_dir: Directory for TensorBoard logs
            run_name: Name for this training run (for TensorBoard)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # Initialize TensorBoard writer
        if tensorboard_dir is None:
            tensorboard_dir = "runs"
        
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.tensorboard_dir = tensorboard_dir
        self.run_name = run_name
        log_dir = os.path.join(tensorboard_dir, run_name)
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Initialize wandb if available and enabled
        self.use_wandb = False
        if WANDB_AVAILABLE:
            # Check if wandb is already initialized (from external script)
            if wandb.run is None:
                # Will be initialized externally
                self.use_wandb = False
            else:
                self.use_wandb = True
        
        # Loss function with class weights
        # Get class weights from config (default: positive=0.9, negative=0.1)
        class_weights = config.get('class_weights', {})
        positive_weight = class_weights.get('positive', 0.9)
        negative_weight = class_weights.get('negative', 0.1)
        
        # Calculate pos_weight for BCEWithLogitsLoss
        # pos_weight scales the positive class contribution
        # To achieve 90% positive and 10% negative contribution:
        # pos_weight = (positive_weight / negative_weight) = 0.9 / 0.1 = 9
        pos_weight = positive_weight / negative_weight
        
        # Create pos_weight tensor and move to device
        pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(device)
        
        # BCEWithLogitsLoss combines sigmoid + BCE loss for numerical stability
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        
        print(f"Using weighted BCEWithLogitsLoss: positive_weight={positive_weight}, negative_weight={negative_weight}, pos_weight={pos_weight:.2f}")
        
        # Optimizer
        optimizer_name = config.get('optimizer', 'adam').lower()
        lr = config.get('learning_rate', 0.001)
        weight_decay = config.get('weight_decay', 0.0001)
        
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Learning rate scheduler
        scheduler_type = config.get('scheduler', 'step').lower()
        if scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.get('scheduler_step_size', 30),
                gamma=config.get('scheduler_gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        else:
            self.scheduler = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        for batch_idx, (features, labels) in enumerate(pbar):
            features = features.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log batch loss to TensorBoard
            global_step = self.current_epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
            
            # Log to wandb if available
            if self.use_wandb and WANDB_AVAILABLE:
                wandb.log({'Train/BatchLoss': loss.item()}, step=global_step)
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for features, labels in self.val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(
        self,
        filepath: str,
        is_best: bool = False,
        is_last: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
        
        if is_last:
            last_path = os.path.join(self.checkpoint_dir, 'last_model.pt')
            torch.save(checkpoint, last_path)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        # Log previous losses to TensorBoard for continuity
        for epoch, (train_loss, val_loss) in enumerate(zip(self.train_losses, self.val_losses)):
            self.writer.add_scalar('Loss/Train', train_loss, epoch + 1)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch + 1)
            if self.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'Loss/Train': train_loss,
                    'Loss/Validation': val_loss
                }, step=epoch + 1)
    
    def train(self, num_epochs: int):
        """Train model for multiple epochs."""
        early_stopping_patience = self.config.get('early_stopping_patience', 10)
        early_stopping_min_delta = self.config.get('early_stopping_min_delta', 0.001)
        save_frequency = self.config.get('save_frequency', 10)
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Log metrics to TensorBoard
            self.writer.add_scalar('Loss/Train', train_loss, epoch + 1)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch + 1)
            
            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('LearningRate', current_lr, epoch + 1)
            
            # Log to wandb if available
            if self.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'Loss/Train': train_loss,
                    'Loss/Validation': val_loss,
                    'LearningRate': current_lr
                }, step=epoch + 1)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Checkpointing
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                patience_counter = 0
                # Log best validation loss
                self.writer.add_scalar('Best/ValidationLoss', val_loss, epoch + 1)
                if self.use_wandb and WANDB_AVAILABLE:
                    wandb.log({'Best/ValidationLoss': val_loss}, step=epoch + 1)
            else:
                patience_counter += 1
            
            # Save checkpoints
            if is_best or (epoch + 1) % save_frequency == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f'checkpoint_epoch_{epoch + 1}.pt'
                )
                self.save_checkpoint(
                    checkpoint_path,
                    is_best=is_best,
                    is_last=(epoch == num_epochs - 1)
                )
            
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Close TensorBoard writer
        self.writer.close()
        
        # TensorBoard logs are automatically synced to wandb via wandb.tensorboard.patch()
        # called during wandb.init() in train.py/test.py/assess_model.py


"""
Neural network models for network anomaly detection.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class NetworkAnomalyDetector(nn.Module):
    """
    Feedforward neural network for network anomaly detection.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 1,
        dropout: float = 0.3,
        activation: str = "relu"
    ):
        """
        Initialize model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (1 for binary classification)
            dropout: Dropout probability
            activation: Activation function ('relu', 'tanh', 'sigmoid')
        """
        super(NetworkAnomalyDetector, self).__init__()
        
        # Activation function mapping
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        act_fn = activations.get(activation.lower(), nn.ReLU())
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # For binary classification
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Output tensor (batch_size, output_dim)
        """
        return self.model(x)


def create_model(config: dict) -> NetworkAnomalyDetector:
    """
    Create model from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized model
    """
    model = NetworkAnomalyDetector(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        output_dim=config['output_dim'],
        dropout=config['dropout'],
        activation=config['activation']
    )
    return model


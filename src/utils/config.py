"""
Configuration management utilities.
"""

import yaml
from typing import Dict, Any
import os


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_device(use_cuda: bool = True, cuda_device: int = 0):
    """
    Get PyTorch device.
    
    Args:
        use_cuda: Whether to use CUDA
        cuda_device: CUDA device index
        
    Returns:
        PyTorch device
    """
    import torch
    
    if use_cuda and torch.cuda.is_available():
        return torch.device(f'cuda:{cuda_device}')
    else:
        return torch.device('cpu')


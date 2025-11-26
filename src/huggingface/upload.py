"""
Hugging Face model upload utilities.
"""

import torch
from huggingface_hub import HfApi, login, upload_file
from pathlib import Path
import os
from typing import Optional


def upload_model_to_hub(
    model_path: str,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False
):
    """
    Upload model to Hugging Face Hub.
    
    Args:
        model_path: Path to model checkpoint
        repo_id: Hugging Face repository ID (username/repo-name)
        token: Hugging Face token (if None, will use saved token)
        private: Whether repository should be private
    """
    # Login if token provided
    if token:
        login(token=token)
    
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload model file
    print(f"Uploading model to {repo_id}...")
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="pytorch_model.pt",
        repo_id=repo_id,
        repo_type="model"
    )
    
    print(f"Model uploaded successfully to https://huggingface.co/{repo_id}")


def save_model_for_hub(
    model: torch.nn.Module,
    save_path: str,
    metadata: Optional[dict] = None
):
    """
    Save model in format suitable for Hugging Face Hub.
    
    Args:
        model: PyTorch model
        save_path: Path to save model
        metadata: Additional metadata to save
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': metadata or {}
    }
    
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


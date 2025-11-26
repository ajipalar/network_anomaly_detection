"""
Script to upload trained model to Hugging Face Hub.
"""

import argparse
import os
from src.huggingface import upload_model_to_hub, save_model_for_hub
from src.models import create_model
from src.utils import load_config
import torch


def main():
    parser = argparse.ArgumentParser(description='Upload model to Hugging Face Hub')
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
        '--repo-id',
        type=str,
        default=None,
        help='Hugging Face repository ID (overrides config)'
    )
    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='Hugging Face token (if not set, uses saved token)'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Make repository private'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get repository ID
    repo_id = args.repo_id or config['huggingface']['repo_id']
    if repo_id == "your-username/network-anomaly-detector":
        print("ERROR: Please set your Hugging Face repository ID in config.yaml or --repo-id")
        print("Format: username/repository-name")
        return
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Create model
    model_config = config['model'].copy()
    # Get input_dim from checkpoint or config
    if 'config' in checkpoint and 'model' in checkpoint['config']:
        model_config['input_dim'] = checkpoint['config']['model']['input_dim']
    else:
        # Default or from config
        model_config['input_dim'] = config['model']['input_dim']
    
    model = create_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Save model for hub
    temp_model_path = 'model_for_hub.pt'
    metadata = {
        'model_config': model_config,
        'training_config': checkpoint.get('config', {}),
        'epoch': checkpoint.get('epoch', 0),
        'best_val_loss': checkpoint.get('best_val_loss', None)
    }
    save_model_for_hub(model, temp_model_path, metadata)
    
    # Upload to hub
    upload_model_to_hub(
        model_path=temp_model_path,
        repo_id=repo_id,
        token=args.token,
        private=args.private or config['huggingface']['private']
    )
    
    # Cleanup
    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)
    
    print("Done!")


if __name__ == '__main__':
    main()


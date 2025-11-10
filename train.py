# ============================================================================
# example_train.py
"""Simple example script for training a model."""

from pathlib import Path
from config import Config
from dataset import create_dataloaders
from models import create_model
from trainer import Trainer
import torch


def main():
    """Train a model with default settings."""
    
    # Create config
    config = Config()
    config.model_name = "gcn"  # or "gin", "pdglstm_0"
    config.max_epochs = 50
    config.batch_size = 4
    config.learning_rate = 0.001
    
    # Setup experiment directory
    exp_dir = Path(config.save_dir) / "experiments" / f"{config.model_name}_{config.model_num}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training {config.model_name} model...")
    print(f"Results will be saved to: {exp_dir}\n")
    
    # Load data
    print("Loading datasets...")
    dataloaders = create_dataloaders(config)
    print(f"✓ Loaded {len(dataloaders['train'].dataset)} training samples")
    print(f"✓ Loaded {len(dataloaders['valid'].dataset)} validation samples")
    print(f"✓ Loaded {len(dataloaders['test'].dataset)} test samples\n")
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created with {num_params:,} parameters\n")
    
    # Train
    trainer = Trainer(model, config, dataloaders, exp_dir)
    print("Starting training...")
    trainer.train()
    
    # Test
    print("\nRunning final evaluation on test set...")
    test_metrics = trainer.test()
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Best model saved to: {exp_dir / 'best_model.pth'}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
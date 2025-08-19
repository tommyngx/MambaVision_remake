"""
Main training script for MambaVision model
Complete pipeline from data loading to model training with visualization and logging
"""

import torch
import torch.nn as nn
import argparse
import wandb
import os
import matplotlib.pyplot as plt
from typing import Dict, Any

# Import our modules
from model import create_mambavision_tiny, create_mambavision_small
from dataset import create_synthetic_dataset, create_data_loaders
from training import Trainer, create_optimizer, create_scheduler


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train MambaVision on synthetic data')
    
    # Model parameters
    parser.add_argument('--model-size', type=str, default='tiny', choices=['tiny', 'small'],
                        help='Model size (default: tiny)')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='Number of classes (default: 10)')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size (default: 224)')
    
    # Dataset parameters
    parser.add_argument('--dataset-size', type=int, default=1000,
                        help='Total number of synthetic samples (default: 1000)')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Training split ratio (default: 0.8)')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Use data augmentation (default: True)')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs (default: 5)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step'],
                        help='Learning rate scheduler (default: cosine)')
    
    # System parameters
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda) (default: auto)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of data loading workers (default: 0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    # Logging and saving
    parser.add_argument('--use-wandb', action='store_true', default=False,
                        help='Use Weights & Biases logging (default: False)')
    parser.add_argument('--no-wandb', action='store_true', default=False,
                        help='Disable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='mambavision-training',
                        help='Weights & Biases project name (default: mambavision-training)')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints (default: ./checkpoints)')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Visualize model architecture (default: True)')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_arg: str) -> str:
    """Get device to use for training"""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    return device


def visualize_model(model: nn.Module, input_shape: tuple, save_path: str = 'model_architecture.png'):
    """Visualize model architecture using torchview"""
    try:
        from torchview import draw_graph
        
        print("Generating model visualization...")
        
        # Create model graph
        model_graph = draw_graph(
            model, 
            input_size=input_shape,
            save_graph=True,
            filename=save_path.replace('.png', ''),
            device='cpu'
        )
        
        print(f"Model architecture saved to: {save_path}")
        return True
        
    except ImportError:
        print("torchview not available. Install with: pip install torchview")
        return False
    except Exception as e:
        print(f"Error generating model visualization: {e}")
        return False


def plot_training_curves(metrics: Dict[str, Any], save_path: str = 'training_curves.png'):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = range(1, len(metrics['train_losses']) + 1)
    
    # Plot losses
    ax1.plot(epochs, metrics['train_losses'], 'b-', label='Train Loss')
    ax1.plot(epochs, metrics['val_losses'], 'r-', label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(epochs, metrics['val_accuracies'], 'g-', label='Val Accuracy')
    ax2.axhline(y=metrics['best_acc'], color='r', linestyle='--', label=f'Best: {metrics["best_acc"]:.2f}%')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {save_path}")


def main():
    """Main training function"""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Initialize wandb
    use_wandb = args.use_wandb and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"mambavision-{args.model_size}-{args.epochs}epochs"
        )
    
    print("=" * 60)
    print("MambaVision Training Pipeline")
    print("=" * 60)
    
    # Create model
    print(f"\n1. Creating {args.model_size} MambaVision model...")
    if args.model_size == 'tiny':
        model = create_mambavision_tiny(num_classes=args.num_classes, img_size=args.img_size)
    else:
        model = create_mambavision_small(num_classes=args.num_classes, img_size=args.img_size)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Visualize model
    if args.visualize:
        print(f"\n2. Visualizing model architecture...")
        visualize_model(
            model, 
            input_shape=(args.batch_size, 3, args.img_size, args.img_size),
            save_path='model_architecture.png'
        )
    
    # Create dataset
    print(f"\n3. Creating synthetic dataset...")
    train_dataset, val_dataset = create_synthetic_dataset(
        size=args.dataset_size,
        img_size=args.img_size,
        num_classes=args.num_classes,
        train_split=args.train_split,
        augment=args.augment,
        random_seed=args.seed
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device == 'cuda'
    )
    
    # Create loss function, optimizer, and scheduler
    print(f"\n4. Setting up training components...")
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, args.epochs, args.scheduler)
    
    print(f"Loss function: {criterion.__class__.__name__}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Scheduler: {scheduler.__class__.__name__}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_wandb=use_wandb,
        save_dir=args.save_dir
    )
    
    # Train model
    print(f"\n5. Starting training for {args.epochs} epochs...")
    print("=" * 60)
    
    training_metrics = trainer.train(
        num_epochs=args.epochs,
        resume_from=args.resume
    )
    
    print("=" * 60)
    print("Training completed!")
    
    # Plot training curves
    print(f"\n6. Generating training curves...")
    plot_training_curves(training_metrics, 'training_curves.png')
    
    # Final summary
    print(f"\n7. Final Results:")
    print(f"Best validation accuracy: {training_metrics['best_acc']:.2f}%")
    print(f"Final training loss: {training_metrics['train_losses'][-1]:.4f}")
    print(f"Final validation loss: {training_metrics['val_losses'][-1]:.4f}")
    
    # Log final metrics to wandb
    if use_wandb:
        wandb.log({
            'final/best_acc': training_metrics['best_acc'],
            'final/final_train_loss': training_metrics['train_losses'][-1],
            'final/final_val_loss': training_metrics['val_losses'][-1]
        })
        
        # Log training curves as images
        if os.path.exists('training_curves.png'):
            wandb.log({"training_curves": wandb.Image('training_curves.png')})
        
        if os.path.exists('model_architecture.png'):
            wandb.log({"model_architecture": wandb.Image('model_architecture.png')})
        
        wandb.finish()
    
    print(f"\nCheckpoints saved to: {args.save_dir}")
    print("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()

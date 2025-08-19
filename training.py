"""
Training utilities for MambaVision model
Includes training loop, evaluation, and metric tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import wandb


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[float]:
    """
    Computes the accuracy over the k top predictions
    
    Args:
        output: Model predictions (batch_size, num_classes)
        target: Ground truth labels (batch_size,)
        topk: Tuple of k values for top-k accuracy
    
    Returns:
        List of top-k accuracies
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


class Trainer:
    """
    Training class for MambaVision model
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_wandb: bool = True,
        save_dir: str = './checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_wandb = use_wandb
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        print(f"Training on device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        losses = AverageMeter()
        top1 = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f'Training Epoch {self.current_epoch + 1}')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Compute accuracy
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            
            # Update meters
            losses.update(loss.item(), images.size(0))
            top1.update(acc1, images.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc@1': f'{top1.avg:.2f}%'
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/batch_acc': acc1,
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': self.current_epoch,
                    'batch': batch_idx
                })
        
        return {
            'train_loss': losses.avg,
            'train_acc': top1.avg
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        losses = AverageMeter()
        top1 = AverageMeter()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validating')
            
            for images, targets in pbar:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                # Compute accuracy
                acc1 = accuracy(outputs, targets, topk=(1,))[0]
                
                # Update meters
                losses.update(loss.item(), images.size(0))
                top1.update(acc1, images.size(0))
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{losses.avg:.4f}',
                    'Acc@1': f'{top1.avg:.2f}%'
                })
        
        return {
            'val_loss': losses.avg,
            'val_acc': top1.avg
        }
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_acc': self.best_acc,
            'metrics': metrics
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with accuracy: {self.best_acc:.2f}%")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.current_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_acc = checkpoint['best_acc']
            
            print(f"Loaded checkpoint (epoch {self.current_epoch}, best_acc: {self.best_acc:.2f}%)")
        else:
            print(f"No checkpoint found at '{checkpoint_path}'")
    
    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from
        """
        if resume_from:
            self.load_checkpoint(resume_from)
        
        print(f"Starting training for {num_epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            
            # Check if best model
            is_best = val_metrics['val_acc'] > self.best_acc
            if is_best:
                self.best_acc = val_metrics['val_acc']
            
            # Save checkpoint
            self.save_checkpoint(metrics, is_best)
            
            # Store metrics
            self.train_losses.append(train_metrics['train_loss'])
            self.val_losses.append(val_metrics['val_loss'])
            self.val_accuracies.append(val_metrics['val_acc'])
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/epoch_loss': train_metrics['train_loss'],
                    'train/epoch_acc': train_metrics['train_acc'],
                    'val/epoch_loss': val_metrics['val_loss'],
                    'val/epoch_acc': val_metrics['val_acc'],
                    'val/best_acc': self.best_acc
                })
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs} Summary:")
            print(f"Train Loss: {train_metrics['train_loss']:.4f}, Train Acc: {train_metrics['train_acc']:.2f}%")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}, Val Acc: {val_metrics['val_acc']:.2f}%")
            print(f"Best Val Acc: {self.best_acc:.2f}%")
            print("-" * 50)
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print(f"Best validation accuracy: {self.best_acc:.2f}%")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_acc': self.best_acc
        }


def create_optimizer(model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4) -> optim.Optimizer:
    """Create optimizer for the model"""
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_scheduler(
    optimizer: optim.Optimizer, 
    num_epochs: int,
    scheduler_type: str = 'cosine'
) -> optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler"""
    if scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs//3, gamma=0.1)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


if __name__ == "__main__":
    # Test training utilities
    print("Testing training utilities...")
    
    # Create dummy data for testing
    dummy_output = torch.randn(32, 10)
    dummy_target = torch.randint(0, 10, (32,))
    
    # Test accuracy function
    acc = accuracy(dummy_output, dummy_target, topk=(1, 5))
    print(f"Top-1 accuracy: {acc[0]:.2f}%")
    print(f"Top-5 accuracy: {acc[1]:.2f}%")
    
    # Test AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i, 1)
    print(f"Average meter test: {meter.avg:.2f}")
    
    print("Training utilities test successful!")

"""
Dataset utilities for MambaVision training
Includes synthetic dataset generation using torchvision.datasets.FakeData
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple, Optional


def get_transforms(img_size: int = 224, augment: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get train and validation transforms
    
    Args:
        img_size: Target image size
        augment: Whether to apply data augmentation for training
    
    Returns:
        Tuple of (train_transform, val_transform)
    """
    
    # Normalization values (ImageNet standard)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, val_transform


def create_synthetic_dataset(
    size: int = 1000,
    img_size: int = 224,
    num_classes: int = 10,
    train_split: float = 0.8,
    augment: bool = True,
    random_seed: Optional[int] = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create synthetic dataset using torchvision.datasets.FakeData
    
    Args:
        size: Total number of samples
        img_size: Image size (height and width)
        num_classes: Number of classes
        train_split: Fraction of data to use for training
        augment: Whether to apply data augmentation
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    
    if random_seed is not None:
        torch.manual_seed(random_seed)
    
    # Get transforms
    train_transform, val_transform = get_transforms(img_size=img_size, augment=augment)
    
    # Create synthetic dataset
    print(f"Creating synthetic dataset with {size} samples...")
    print(f"Image size: {img_size}x{img_size}")
    print(f"Number of classes: {num_classes}")
    print(f"Train/Val split: {train_split:.1%}/{1-train_split:.1%}")
    
    # Create full dataset with training transforms
    full_dataset = datasets.FakeData(
        size=size,
        image_size=(3, img_size, img_size),
        num_classes=num_classes,
        transform=train_transform,
        random_offset=0
    )
    
    # Split into train and validation
    train_size = int(size * train_split)
    val_size = size - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed) if random_seed else None
    )
    
    # Apply different transforms to validation set
    val_dataset_with_transforms = datasets.FakeData(
        size=val_size,
        image_size=(3, img_size, img_size),
        num_classes=num_classes,
        transform=val_transform,
        random_offset=train_size  # Different offset for validation
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset_with_transforms)}")
    
    return train_dataset, val_dataset_with_transforms


def create_data_loaders(
    train_dataset,
    val_dataset,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training and validation
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print(f"Batch size: {batch_size}")
    
    return train_loader, val_loader


def get_sample_batch(data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get a sample batch from data loader for testing
    
    Args:
        data_loader: DataLoader to sample from
    
    Returns:
        Tuple of (images, labels)
    """
    for images, labels in data_loader:
        return images, labels


if __name__ == "__main__":
    # Test the dataset creation
    print("Testing synthetic dataset creation...")
    
    # Create synthetic dataset
    train_dataset, val_dataset = create_synthetic_dataset(
        size=1000,
        img_size=224,
        num_classes=10,
        train_split=0.8,
        augment=True
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        batch_size=32,
        num_workers=0
    )
    
    # Test a sample batch
    print("\nTesting sample batch...")
    images, labels = get_sample_batch(train_loader)
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image dtype: {images.dtype}")
    print(f"Labels dtype: {labels.dtype}")
    print(f"Image range: [{images.min().item():.3f}, {images.max().item():.3f}]")
    print(f"Unique labels: {torch.unique(labels).tolist()}")
    
    print("\nDataset creation successful!")

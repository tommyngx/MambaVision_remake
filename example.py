"""
Simple example script for testing MambaVision model
This script demonstrates basic usage without full training pipeline
"""

import torch
import torch.nn as nn
import time
from model import create_mambavision_tiny, create_mambavision_small
from dataset import create_synthetic_dataset, create_data_loaders


def test_model_creation():
    """Test model creation and basic functionality"""
    print("=" * 50)
    print("Testing Model Creation")
    print("=" * 50)
    
    # Test tiny model
    print("\n1. Creating tiny MambaVision model...")
    model_tiny = create_mambavision_tiny(num_classes=10)
    total_params_tiny = sum(p.numel() for p in model_tiny.parameters())
    print(f"Tiny model parameters: {total_params_tiny:,}")
    
    # Test small model  
    print("\n2. Creating small MambaVision model...")
    model_small = create_mambavision_small(num_classes=10)
    total_params_small = sum(p.numel() for p in model_small.parameters())
    print(f"Small model parameters: {total_params_small:,}")
    
    return model_tiny, model_small


def test_forward_pass(model, model_name="Model"):
    """Test forward pass with different input sizes"""
    print(f"\n3. Testing forward pass for {model_name}...")
    
    model.eval()
    
    # Test different batch sizes and input sizes
    test_cases = [
        (1, 3, 224, 224),   # Single image
        (4, 3, 224, 224),   # Small batch
        (8, 3, 224, 224),   # Medium batch
    ]
    
    for batch_size, channels, height, width in test_cases:
        with torch.no_grad():
            # Create random input
            x = torch.randn(batch_size, channels, height, width)
            
            # Time the forward pass
            start_time = time.time()
            outputs = model(x)
            end_time = time.time()
            
            print(f"  Input: {x.shape} -> Output: {outputs.shape}")
            print(f"  Forward pass time: {(end_time - start_time)*1000:.2f}ms")
            
            # Check output properties
            assert outputs.shape == (batch_size, 10), f"Expected output shape ({batch_size}, 10), got {outputs.shape}"
            assert not torch.isnan(outputs).any(), "Output contains NaN values"
            assert torch.isfinite(outputs).all(), "Output contains infinite values"


def test_dataset_creation():
    """Test synthetic dataset creation"""
    print("\n" + "=" * 50)
    print("Testing Dataset Creation")
    print("=" * 50)
    
    # Create synthetic dataset
    print("\n4. Creating synthetic dataset...")
    train_dataset, val_dataset = create_synthetic_dataset(
        size=100,  # Small dataset for quick testing
        img_size=224,
        num_classes=10,
        train_split=0.8,
        augment=True,
        random_seed=42
    )
    
    # Create data loaders
    print("\n5. Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        batch_size=8,
        num_workers=0
    )
    
    # Test data loading
    print("\n6. Testing data loading...")
    for i, (images, labels) in enumerate(train_loader):
        print(f"  Batch {i+1}: Images {images.shape}, Labels {labels.shape}")
        print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Label range: [{labels.min()}, {labels.max()}]")
        print(f"  Unique labels in batch: {torch.unique(labels).tolist()}")
        
        if i >= 2:  # Test first 3 batches
            break
    
    return train_loader, val_loader


def test_training_step(model, data_loader):
    """Test a single training step"""
    print("\n" + "=" * 50)
    print("Testing Training Step")
    print("=" * 50)
    
    print("\n7. Testing single training step...")
    
    # Set model to training mode
    model.train()
    
    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Get a batch of data
    images, labels = next(iter(data_loader))
    
    print(f"  Input batch shape: {images.shape}")
    print(f"  Target batch shape: {labels.shape}")
    
    # Forward pass
    start_time = time.time()
    outputs = model(images)
    loss = criterion(outputs, labels)
    forward_time = time.time() - start_time
    
    print(f"  Forward pass time: {forward_time*1000:.2f}ms")
    print(f"  Loss: {loss.item():.4f}")
    
    # Backward pass
    start_time = time.time()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    backward_time = time.time() - start_time
    
    print(f"  Backward pass time: {backward_time*1000:.2f}ms")
    
    # Calculate accuracy
    with torch.no_grad():
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).float().mean().item() * 100
        print(f"  Batch accuracy: {accuracy:.2f}%")
    
    return loss.item(), accuracy


def test_model_saving_loading(model, save_path="test_model.pth"):
    """Test model saving and loading"""
    print("\n" + "=" * 50)
    print("Testing Model Save/Load")
    print("=" * 50)
    
    print(f"\n8. Testing model save/load...")
    
    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"  Model saved to: {save_path}")
    
    # Create new model and load weights
    new_model = create_mambavision_tiny(num_classes=10)
    new_model.load_state_dict(torch.load(save_path, map_location='cpu'))
    print(f"  Model loaded successfully")
    
    # Test that models produce same output
    test_input = torch.randn(1, 3, 224, 224)
    
    model.eval()
    new_model.eval()
    
    with torch.no_grad():
        output1 = model(test_input)
        output2 = new_model(test_input)
        
        # Check if outputs are the same
        diff = torch.abs(output1 - output2).max().item()
        print(f"  Max difference between outputs: {diff:.10f}")
        assert diff < 1e-6, f"Models produce different outputs! Difference: {diff}"
        print(f"  ✅ Save/load test passed!")
    
    # Clean up
    import os
    os.remove(save_path)
    print(f"  Cleaned up test file: {save_path}")


def main():
    """Run all tests"""
    print("🧪 MambaVision Model Testing")
    print("This script tests the basic functionality of the MambaVision implementation")
    
    try:
        # Test model creation
        model_tiny, model_small = test_model_creation()
        
        # Test forward pass
        test_forward_pass(model_tiny, "Tiny")
        
        # Test dataset creation
        train_loader, val_loader = test_dataset_creation()
        
        # Test training step
        loss, accuracy = test_training_step(model_tiny, train_loader)
        
        # Test model saving/loading
        test_model_saving_loading(model_tiny)
        
        # Final summary
        print("\n" + "=" * 50)
        print("✅ All Tests Passed!")
        print("=" * 50)
        print(f"✅ Model creation: Success")
        print(f"✅ Forward pass: Success")
        print(f"✅ Dataset creation: Success")
        print(f"✅ Training step: Success (Loss: {loss:.4f}, Acc: {accuracy:.2f}%)")
        print(f"✅ Model save/load: Success")
        print("\n🎉 MambaVision implementation is working correctly!")
        print("\nNext steps:")
        print("- Run full training: python main.py")
        print("- Experiment with different model sizes")
        print("- Try different hyperparameters")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

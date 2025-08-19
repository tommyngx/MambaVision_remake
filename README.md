# MambaVision: A Hybrid Mamba-Transformer Vision Backbone

This repository contains a simplified implementation of the MambaVision model inspired by the paper ["MambaVision: A Hybrid Mamba‑Transformer Vision Backbone"](https://arxiv.org/abs/2407.08083). The implementation includes a complete training pipeline with synthetic data generation, model visualization, and Weights & Biases integration.

## 🚀 Features

- **Simplified MambaVision Architecture**: Hybrid model combining Mamba and Transformer blocks
- **Synthetic Dataset**: Uses `torchvision.datasets.FakeData` for quick testing
- **Complete Training Pipeline**: Full training loop with validation and checkpointing
- **Model Visualization**: Generate architecture diagrams using `torchview`
- **Weights & Biases Integration**: Comprehensive experiment tracking and logging
- **Flexible Configuration**: Command-line arguments for easy experimentation

## 📁 Project Structure

```
Licenta/
├── model.py           # MambaVision model architecture
├── dataset.py         # Synthetic dataset generation utilities
├── training.py        # Training utilities and trainer class
├── main.py           # Main training script
├── requirements.txt   # Python dependencies
├── example.py        # Simple example script
└── README.md         # This file
```

## 🛠️ Installation

1. **Clone the repository** (or create the project structure):
```bash
git clone <your-repo-url>
cd Licenta
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Optional: Set up Weights & Biases**:
```bash
wandb login
```

## 🔧 Model Architecture

The simplified MambaVision model includes:

### Key Components:
- **Patch Embedding**: Converts input images to token sequences
- **Mamba Blocks**: Simplified state-space models for efficient sequence modeling
- **Transformer Blocks**: Standard multi-head attention mechanisms
- **Hybrid Design**: Alternating between Mamba and Transformer blocks
- **Classification Head**: Final layers for prediction

### Model Variants:
- **Tiny**: 6 layers, 192 embedding dimensions (~1.2M parameters)
- **Small**: 8 layers, 384 embedding dimensions (~4.8M parameters)

## 🚀 Quick Start

### Basic Training
```bash
python main.py --model-size tiny --epochs 5 --batch-size 32
```

### Advanced Training with Custom Parameters
```bash
python main.py \
    --model-size small \
    --epochs 10 \
    --batch-size 64 \
    --lr 2e-4 \
    --dataset-size 2000 \
    --img-size 224 \
    --use-wandb \
    --visualize
```

### Training Options
```bash
# Model configuration
--model-size {tiny,small}     # Model size (default: tiny)
--num-classes 10              # Number of classes (default: 10)
--img-size 224               # Input image size (default: 224)

# Dataset configuration
--dataset-size 1000          # Total synthetic samples (default: 1000)
--train-split 0.8            # Train/val split ratio (default: 0.8)
--augment                    # Enable data augmentation (default: True)

# Training configuration
--batch-size 32              # Batch size (default: 32)
--epochs 5                   # Number of epochs (default: 5)
--lr 1e-3                   # Learning rate (default: 1e-3)
--weight-decay 1e-4         # Weight decay (default: 1e-4)
--scheduler {cosine,step}    # LR scheduler (default: cosine)

# System configuration
--device {auto,cpu,cuda}     # Device to use (default: auto)
--num-workers 0              # Data loading workers (default: 0)
--seed 42                   # Random seed (default: 42)

# Logging and visualization
--use-wandb                  # Enable W&B logging (default: True)
--wandb-project PROJECT     # W&B project name
--visualize                  # Generate model architecture diagram
--save-dir ./checkpoints    # Checkpoint directory
```

## 📊 Monitoring Training

### Weights & Biases Integration
The training script automatically logs:
- **Training/Validation Loss and Accuracy**
- **Learning Rate Schedules**
- **Model Architecture Visualization**
- **Training Curves**
- **Hyperparameters**

### Local Outputs
- `model_architecture.png`: Model architecture diagram
- `training_curves.png`: Loss and accuracy plots
- `checkpoints/`: Model checkpoints (latest and best)

## 🔬 Example Usage

### Simple Example Script
```python
import torch
from model import create_mambavision_tiny
from dataset import create_synthetic_dataset, create_data_loaders

# Create model
model = create_mambavision_tiny(num_classes=10)

# Create synthetic data
train_dataset, val_dataset = create_synthetic_dataset(
    size=1000, img_size=224, num_classes=10
)

# Create data loaders
train_loader, val_loader = create_data_loaders(
    train_dataset, val_dataset, batch_size=32
)

# Test forward pass
for images, labels in train_loader:
    outputs = model(images)
    print(f"Input shape: {images.shape}")
    print(f"Output shape: {outputs.shape}")
    break
```

### Model Inference
```python
import torch
from model import create_mambavision_tiny

# Load trained model
model = create_mambavision_tiny(num_classes=10)
checkpoint = torch.load('checkpoints/best_checkpoint.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

# Inference
model.eval()
with torch.no_grad():
    # Random input image (batch_size=1, channels=3, height=224, width=224)
    x = torch.randn(1, 3, 224, 224)
    predictions = model(x)
    predicted_class = torch.argmax(predictions, dim=1)
    print(f"Predicted class: {predicted_class.item()}")
```

## 📈 Default Hyperparameters

For quick testing, the following default values work well:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| Model Size | tiny | Balance between speed and capacity |
| Batch Size | 32 | Good balance for most GPUs |
| Learning Rate | 1e-3 | Works well with AdamW optimizer |
| Epochs | 5 | Quick testing, increase for real training |
| Dataset Size | 1000 | Small for fast iteration |
| Image Size | 224×224 | Standard vision model input size |
| Weight Decay | 1e-4 | Prevents overfitting |
| Scheduler | cosine | Smooth learning rate decay |

## 🔧 Customization

### Adding New Model Variants
```python
# In model.py
def create_mambavision_custom(num_classes: int = 10) -> MambaVision:
    return MambaVision(
        embed_dim=512,
        depth=10,
        num_heads=8,
        use_mamba_ratio=0.6,  # 60% Mamba, 40% Transformer
        # ... other parameters
    )
```

### Custom Loss Functions
```python
# In training script
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# or
criterion = nn.FocalLoss(alpha=1, gamma=2)  # If implemented
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce batch size: `--batch-size 16`
   - Use smaller model: `--model-size tiny`

2. **Slow training**:
   - Increase number of workers: `--num-workers 4`
   - Use GPU if available
   - Reduce dataset size for testing

3. **Import errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility

### Performance Tips
- Use `--num-workers > 0` for faster data loading
- Enable `torch.compile()` for PyTorch 2.0+ (add to model initialization)
- Use mixed precision training for larger models

## 📚 References

- **Original Paper**: [MambaVision: A Hybrid Mamba‑Transformer Vision Backbone](https://arxiv.org/abs/2407.08083)
- **Mamba**: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- **Vision Transformer**: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

## 📄 License

This project is for educational and research purposes. Please cite the original MambaVision paper if you use this code in your research.

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this implementation!

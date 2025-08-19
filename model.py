"""
Simplified MambaVision Model Architecture
Inspired by "MambaVision: A Hybrid Mamba‑Transformer Vision Backbone"
https://arxiv.org/abs/2407.08083
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MambaBlock(nn.Module):
    """
    Simplified Mamba block inspired by the paper.
    This is a simplified version focusing on the core concepts.
    """
    def __init__(self, dim: int, state_size: int = 16, conv_kernel: int = 4):
        super().__init__()
        self.dim = dim
        self.state_size = state_size
        
        # Input projection
        self.in_proj = nn.Linear(dim, dim * 2)
        
        # Convolution for local interaction
        # For kernel_size=4, we need padding=1 to keep same length
        # But let's use padding='same' to ensure exact same length
        self.conv1d = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=conv_kernel,
            padding='same',  # Automatically calculates padding for same output size
            groups=dim
        )
        
        # State space parameters
        self.x_proj = nn.Linear(dim, state_size)
        self.dt_proj = nn.Linear(dim, dim)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        
        # Activation
        self.act = nn.SiLU()
        
    def forward(self, x):
        """
        Args:
            x: (B, L, D) where B=batch, L=sequence length, D=dim
        """
        B, L, D = x.shape
        residual = x  # Store original input for residual connection
        
        # Input projection and split
        x_and_res = self.in_proj(x)  # (B, L, 2*D)
        x, res = x_and_res.split(self.dim, dim=-1)  # (B, L, D), (B, L, D)
        
        # Apply activation
        x = self.act(x)
        
        # 1D convolution (transpose for conv1d)
        x_conv = x.transpose(1, 2)  # (B, D, L)
        x_conv = self.conv1d(x_conv)  # (B, D, L)
        x = x_conv.transpose(1, 2)  # (B, L, D)
        
        # Simplified state space modeling
        # In the real Mamba, this would involve selective scan
        # Here we use a simplified approximation
        dt = self.dt_proj(x)  # (B, L, D)
        dt = F.softplus(dt)
        
        # Apply gate
        x = x * dt
        
        # Apply gating with res
        x = x * self.act(res)
        
        # Output projection
        x = self.out_proj(x)
        
        # Residual connection with original input
        x = x + residual
        
        return x


class TransformerBlock(nn.Module):
    """
    Standard Transformer block for hybrid architecture
    """
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual connection
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MambaVision(nn.Module):
    """
    Simplified MambaVision: A Hybrid Mamba-Transformer Vision Backbone
    
    This implementation includes:
    - Patch embedding for converting images to tokens
    - Alternating Mamba and Transformer blocks
    - Classification head
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        use_mamba_ratio: float = 0.5,  # Fraction of blocks that are Mamba vs Transformer
        dropout: float = 0.1,
        state_size: int = 16,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(dropout)
        
        # Hybrid blocks (alternating Mamba and Transformer)
        self.blocks = nn.ModuleList()
        num_mamba_blocks = int(depth * use_mamba_ratio)
        
        for i in range(depth):
            if i < num_mamba_blocks:
                # Use Mamba block
                block = MambaBlock(
                    dim=embed_dim,
                    state_size=state_size
                )
            else:
                # Use Transformer block
                block = TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout
                )
            self.blocks.append(block)
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights"""
        # Initialize positional embeddings
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize other weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x):
        """Forward pass through feature extraction layers"""
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply blocks
        for block in self.blocks:
            x = block(x)
        
        # Global average pooling
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling over sequence dimension
        
        return x
    
    def forward(self, x):
        """Forward pass"""
        x = self.forward_features(x)
        x = self.head(x)
        return x


def create_mambavision_tiny(num_classes: int = 10, img_size: int = 224) -> MambaVision:
    """Create a tiny MambaVision model for quick testing"""
    return MambaVision(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=192,
        depth=6,
        num_heads=6,
        use_mamba_ratio=0.5,
        dropout=0.1,
        state_size=16,
    )


def create_mambavision_small(num_classes: int = 10, img_size: int = 224) -> MambaVision:
    """Create a small MambaVision model"""
    return MambaVision(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=384,
        depth=8,
        num_heads=8,
        use_mamba_ratio=0.5,
        dropout=0.1,
        state_size=16,
    )


if __name__ == "__main__":
    # Test the model
    model = create_mambavision_tiny(num_classes=10)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created successfully!")
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

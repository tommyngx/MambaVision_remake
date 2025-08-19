"""
Debug script to understand the tensor dimension mismatch
"""

import torch
from model import create_mambavision_tiny

def debug_model():
    """Debug the model to understand tensor shapes"""
    print("Debugging MambaVision model...")
    
    model = create_mambavision_tiny(num_classes=10)
    x = torch.randn(1, 3, 224, 224)
    
    print(f"Input shape: {x.shape}")
    
    # Test patch embedding
    patch_embed = model.patch_embed
    x_patches = patch_embed(x)
    print(f"After patch embedding: {x_patches.shape}")
    
    # Test positional embedding
    pos_embed_shape = model.pos_embed.shape
    print(f"Positional embedding shape: {pos_embed_shape}")
    
    # Add positional embedding
    x_with_pos = x_patches + model.pos_embed
    print(f"After adding pos embedding: {x_with_pos.shape}")
    
    # Test first block
    first_block = model.blocks[0]
    print(f"First block type: {type(first_block)}")
    
    try:
        output = first_block(x_with_pos)
        print(f"First block output shape: {output.shape}")
    except Exception as e:
        print(f"Error in first block: {e}")
        
        # Debug inside MambaBlock
        print("\nDebugging MambaBlock internals...")
        B, L, D = x_with_pos.shape
        print(f"MambaBlock input: B={B}, L={L}, D={D}")
        
        # Input projection
        x_and_res = first_block.in_proj(x_with_pos)
        print(f"After in_proj: {x_and_res.shape}")
        
        x_split, res_split = x_and_res.split(first_block.dim, dim=-1)
        print(f"After split - x: {x_split.shape}, res: {res_split.shape}")
        
        # Apply activation
        x_act = first_block.act(x_split)
        print(f"After activation: {x_act.shape}")
        
        # 1D convolution
        x_conv_input = x_act.transpose(1, 2)
        print(f"Conv input (after transpose): {x_conv_input.shape}")
        
        x_conv_output = first_block.conv1d(x_conv_input)
        print(f"Conv output: {x_conv_output.shape}")
        
        x_conv = x_conv_output.transpose(1, 2)
        print(f"After transpose back: {x_conv.shape}")
        
        # dt projection
        dt = first_block.dt_proj(x_conv)
        print(f"dt shape: {dt.shape}")
        
        # Apply gate
        x_gated = x_conv * torch.nn.functional.softplus(dt)
        print(f"After gating: {x_gated.shape}")
        
        # The problematic line
        print(f"x_gated shape: {x_gated.shape}")
        print(f"res_split shape: {res_split.shape}")
        print(f"act(res_split) shape: {first_block.act(res_split).shape}")

if __name__ == "__main__":
    debug_model()

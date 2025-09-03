#!/usr/bin/env python3
"""
Test script to verify RWKV integration with Latent Thought Model
"""

import torch
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import LatentThoughtModel, LTMConfig
from config import get_config_dict

def test_rwkv_model():
    """Test the RWKV model with basic functionality"""
    
    print("Testing RWKV integration with Latent Thought Model...")
    
    # Create a small test configuration
    config = LTMConfig(
        dim=256,  # Small dimension for testing
        n_layers=4,  # Few layers for testing
        n_heads=8,  # Few heads for testing
        vocab_size=1000,  # Small vocab for testing
        max_seq_len=128,  # Small sequence length for testing
        use_rwkv=True,  # Enable RWKV
        use_rwkv8_ffn=True,  # Use RWKV-8 FFN
        rwkv_mode="rwkv8",  # Use RWKV-8 mode
        head_size=32,  # RWKV head size
        dropout=0.0,  # No dropout for testing
        use_liger=False,  # Disable LIGER for testing
    )
    
    print(f"Model configuration:")
    print(f"  - Dimension: {config.dim}")
    print(f"  - Layers: {config.n_layers}")
    print(f"  - Heads: {config.n_heads}")
    print(f"  - Vocabulary size: {config.vocab_size}")
    print(f"  - Max sequence length: {config.max_seq_len}")
    print(f"  - RWKV enabled: {config.use_rwkv}")
    print(f"  - RWKV-8 FFN: {config.use_rwkv8_ffn}")
    print(f"  - RWKV mode: {config.rwkv_mode}")
    print(f"  - Head size: {config.head_size}")
    
    # Create the model
    model = LatentThoughtModel(config)
    model.eval()
    
    print(f"\nModel created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 64
    z_len = config.max_z_len // config.n_layers
    
    # Create dummy input
    tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    z = torch.randn(batch_size, config.max_z_len, config.dim)
    
    print(f"\nTesting forward pass...")
    print(f"  - Input tokens shape: {tokens.shape}")
    print(f"  - Latent z shape: {z.shape}")
    
    try:
        # Test forward pass
        with torch.no_grad():
            logits = model(tokens, z)
        
        print(f"  - Output logits shape: {logits.shape}")
        print(f"  - Expected output shape: ({batch_size}, {seq_len}, {config.vocab_size})")
        
        # Check if output shape is correct
        expected_shape = (batch_size, seq_len, config.vocab_size)
        if logits.shape == expected_shape:
            print(f"✓ Forward pass successful! Output shape matches expected shape.")
        else:
            print(f"✗ Forward pass failed! Output shape {logits.shape} does not match expected shape {expected_shape}")
            return False
            
        # Test generation
        print(f"\nTesting generation...")
        prompt = torch.randint(0, config.vocab_size, (1, 10))
        z_prompt = torch.randn(1, config.max_z_len, config.dim)
        
        with torch.no_grad():
            generated = model.generate(prompt, z_prompt, max_new_tokens=5)
        
        print(f"  - Prompt shape: {prompt.shape}")
        print(f"  - Generated shape: {generated.shape}")
        print(f"  - Generated tokens: {generated[0][-5:].tolist()}")
        
        if generated.shape[1] == prompt.shape[1] + 5:
            print(f"✓ Generation successful! Generated {5} new tokens.")
        else:
            print(f"✗ Generation failed! Expected {prompt.shape[1] + 5} tokens, got {generated.shape[1]}")
            return False
            
        # Test ELBO computation
        print(f"\nTesting ELBO computation...")
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        mu = torch.randn(batch_size, config.max_z_len, config.dim)
        log_var = torch.randn(batch_size, config.max_z_len, config.dim)
        eps = torch.randn(batch_size, config.max_z_len, config.dim)
        
        with torch.no_grad():
            nelbo, perplexity, h_before_cross_attention, kl_mean, nlkhd_mean = model.elbo(
                tokens, mu, log_var, eps, targets
            )
        
        print(f"  - ELBO: {nelbo.item():.4f}")
        print(f"  - Perplexity: {perplexity.item():.4f}")
        print(f"  - KL divergence: {kl_mean.item():.4f}")
        print(f"  - Negative log likelihood: {nlkhd_mean.item():.4f}")
        
        print(f"✓ ELBO computation successful!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_config_integration():
    """Test that RWKV configuration is properly integrated"""
    
    print(f"\nTesting configuration integration...")
    
    # Get the global config
    config_dict = get_config_dict()
    
    # Check if RWKV parameters are present
    rwkv_params = ['use_rwkv', 'use_rwkv8_ffn', 'head_size', 'rwkv_mode']
    
    for param in rwkv_params:
        if param in config_dict:
            print(f"✓ Configuration parameter '{param}' found with value: {config_dict[param]}")
        else:
            print(f"✗ Configuration parameter '{param}' not found!")
            return False
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("RWKV Integration Test for Latent Thought Model")
    print("=" * 60)
    
    # Test configuration integration
    config_success = test_config_integration()
    
    # Test model functionality
    model_success = test_rwkv_model()
    
    print(f"\n" + "=" * 60)
    print("Test Results:")
    print(f"  Configuration Integration: {'✓ PASS' if config_success else '✗ FAIL'}")
    print(f"  Model Functionality: {'✓ PASS' if model_success else '✗ FAIL'}")
    
    if config_success and model_success:
        print(f"\n🎉 All tests passed! RWKV integration is working correctly.")
        sys.exit(0)
    else:
        print(f"\n❌ Some tests failed. Please check the output above for details.")
        sys.exit(1)
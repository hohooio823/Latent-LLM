"""
Test script for DiT prior integration with LTM.
This script verifies that the DiT prior is properly integrated and functional.
"""

import torch
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import LatentThoughtModel, LTMConfig
from dit_prior import DiTPrior, DiTConfig

def test_dit_prior():
    """Test the DiT prior model."""
    print("Testing DiT prior...")
    
    # Create DiT config
    dit_config = DiTConfig(
        z_dim=512,
        max_z_len=32,
        dit_layers=6,
        dit_heads=8,
        dit_dim=512,
        num_timesteps=1000,
        beta_schedule="linear"
    )
    
    # Create DiT prior
    dit_prior = DiTPrior(dit_config)
    print(f"DiT prior created with {sum(p.numel() for p in dit_prior.parameters()):,} parameters")
    
    # Test forward pass
    batch_size = 4
    z = torch.randn(batch_size, dit_config.max_z_len, dit_config.z_dim)
    timesteps = torch.randint(0, dit_config.num_timesteps, (batch_size,))
    
    with torch.no_grad():
        output = dit_prior(z, timesteps)
    
    print(f"DiT forward pass successful: output shape {output.shape}")
    
    # Test sampling
    with torch.no_grad():
        samples = dit_prior.sample(batch_size, z.device)
    
    print(f"DiT sampling successful: samples shape {samples.shape}")
    
    # Test diffusion loss
    loss = dit_prior.p_losses(z, timesteps)
    print(f"DiT loss computation successful: loss = {loss.item():.4f}")
    
    print("✓ DiT prior tests passed!")
    return True

def test_ltm_with_dit():
    """Test LTM with DiT prior integration."""
    print("\nTesting LTM with DiT prior...")
    
    # Create LTM config with DiT prior enabled
    config = LTMConfig(
        dim=512,
        n_layers=6,
        n_heads=8,
        vocab_size=50258,
        max_seq_len=256,
        max_z_len=96,
        use_dit_prior=True,
        dit_layers=6,
        dit_heads=8,
        dit_dim=512,
        dit_num_timesteps=1000
    )
    
    # Create LTM model
    model = LatentThoughtModel(config)
    print(f"LTM model created with DiT prior")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    z = torch.randn(batch_size, config.max_z_len, config.dim)
    
    with torch.no_grad():
        logits = model(tokens, z)
    
    print(f"LTM forward pass successful: logits shape {logits.shape}")
    
    # Test ELBO computation
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    mu = torch.randn(batch_size, config.max_z_len, config.dim)
    log_var = torch.randn(batch_size, config.max_z_len, config.dim)
    eps = torch.randn_like(log_var)
    
    with torch.no_grad():
        nelbo, ppl, h, kl_loss, nlkhd = model.elbo(tokens, mu, log_var, eps, targets)
    
    print(f"ELBO computation successful:")
    print(f"  - NELBO: {nelbo.item():.4f}")
    print(f"  - Perplexity: {ppl.item():.4f}")
    print(f"  - KL Loss: {kl_loss.item():.4f}")
    print(f"  - NLKHD: {nlkhd.item():.4f}")
    
    # Test prior sampling
    with torch.no_grad():
        prior_samples = model.sample_from_prior(batch_size, tokens.device)
    
    print(f"Prior sampling successful: samples shape {prior_samples.shape}")
    
    print("✓ LTM with DiT prior tests passed!")
    return True

def test_training_integration():
    """Test training integration components."""
    print("\nTesting training integration...")
    
    # Test configuration
    config = LTMConfig(
        dim=256,
        n_layers=3,
        n_heads=4,
        vocab_size=50258,
        max_seq_len=128,
        max_z_len=48,
        use_dit_prior=True,
        dit_layers=3,
        dit_heads=4,
        dit_dim=256,
        dit_num_timesteps=100
    )
    
    # Create LTM model
    model = LatentThoughtModel(config)
    
    # Test optimizer
    from optimizer import PosteriorOptimizer
    optimizer = PosteriorOptimizer(
        model,
        inference_method="adamVI",
        lr=0.1,
        num_steps=5,
        max_z_len=config.max_z_len,
        z_dim=config.dim,
        use_dit_prior=True
    )
    
    # Test data
    batch_size = 2
    seq_len = 64
    tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Test optimization step
    with torch.no_grad():
        data = [tokens, targets, None]
        ctx = torch.no.no_grad()
        
        z, ppl, kl_loss, nlkhd = optimizer.step(data, ctx)
        
        print(f"Optimization step successful:")
        print(f"  - Optimized z shape: {z.shape}")
        print(f"  - Perplexity: {ppl.item():.4f}")
        print(f"  - KL Loss: {kl_loss.item():.4f}")
        print(f"  - NLKHD: {nlkhd.item():.4f}")
    
    print("✓ Training integration tests passed!")
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("DiT Prior Integration Test Suite")
    print("=" * 60)
    
    try:
        # Run all tests
        test_dit_prior()
        test_ltm_with_dit()
        test_training_integration()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed successfully!")
        print("DiT prior integration is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
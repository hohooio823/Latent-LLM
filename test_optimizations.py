#!/usr/bin/env python3
"""
Test script to verify the optimizations are working correctly
"""
import torch
import time
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import apply_optimized_config
from optimizations import MemoryOptimizer, benchmark_model, OptimizedLatentThoughtModel
from model import LTMConfig

def test_optimized_config():
    """Test that optimized configuration is applied correctly"""
    print("Testing optimized configuration...")
    
    # Create a basic config
    class MockConfig:
        def __init__(self):
            self.num_workers = 0
            self.compile = False
            self.gradient_checkpointing = False
            self.use_optimized_rwkv = False
            self.use_flash_attention = False
            self.memory_optimization = False
    
    config = MockConfig()
    
    # Apply optimizations
    optimized_config = apply_optimized_config(config)
    
    # Verify optimizations are applied
    assert optimized_config.num_workers == 8, f"Expected 8 workers, got {optimized_config.num_workers}"
    assert optimized_config.compile == True, f"Expected compile=True, got {optimized_config.compile}"
    assert optimized_config.gradient_checkpointing == True, f"Expected gradient_checkpointing=True, got {optimized_config.gradient_checkpointing}"
    assert optimized_config.use_optimized_rwkv == True, f"Expected use_optimized_rwkv=True, got {optimized_config.use_optimized_rwkv}"
    
    print("✅ Optimized configuration test passed")

def test_memory_optimizations():
    """Test memory optimization utilities"""
    print("Testing memory optimizations...")
    
    # Test memory optimization functions
    try:
        MemoryOptimizer.optimize_memory_usage()
        MemoryOptimizer.enable_tf32()
        MemoryOptimizer.enable_memory_efficient_attention()
        print("✅ Memory optimization utilities test passed")
    except Exception as e:
        print(f"❌ Memory optimization utilities test failed: {e}")

def test_optimized_model():
    """Test that optimized model can be created and run"""
    print("Testing optimized model...")
    
    try:
        # Create model configuration
        config = LTMConfig(
            dim=256,
            n_layers=4,
            n_heads=8,
            n_kv_heads=8,
            vocab_size=1000,
            multiple_of=32,
            max_seq_len=128,
            dropout=0.1,
            use_liger=False,
            use_rwkv=True,
            use_rwkv8_ffn=True,
            head_size=32,
            rwkv_mode="rwkv8",
            use_optimized_rwkv=True,
            gradient_checkpointing=True,
            hidden_dim=224
        )
        
        # Create optimized model
        model = OptimizedLatentThoughtModel(config)
        model.eval()
        
        # Test forward pass
        batch_size = 2
        seq_len = 64
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        z = torch.randn(batch_size, config.max_z_len // config.n_layers, config.dim)
        
        with torch.no_grad():
            output = model(x, z)
        
        assert output.shape == (batch_size, seq_len, config.vocab_size), f"Expected output shape {(batch_size, seq_len, config.vocab_size)}, got {output.shape}"
        
        print("✅ Optimized model test passed")
        
    except Exception as e:
        print(f"❌ Optimized model test failed: {e}")
        import traceback
        traceback.print_exc()

def test_kv_cache():
    """Test KV cache functionality"""
    print("Testing KV cache...")
    
    try:
        # Create model configuration
        config = LTMConfig(
            dim=256,
            n_layers=2,
            n_heads=8,
            n_kv_heads=8,
            vocab_size=1000,
            multiple_of=32,
            max_seq_len=128,
            dropout=0.1,
            use_liger=False,
            use_rwkv=True,
            use_rwkv8_ffn=True,
            head_size=32,
            rwkv_mode="rwkv8",
            use_optimized_rwkv=True,
            gradient_checkpointing=True,
            hidden_dim=224
        )
        
        # Create optimized model
        model = OptimizedLatentThoughtModel(config)
        model.eval()
        
        # Test KV cache initialization
        model.initialize_kv_cache(batch_size=2, device='cpu')
        assert model.kv_cache is not None, "KV cache should be initialized"
        
        # Test generation with cache
        idx = torch.randint(0, config.vocab_size, (2, 10))
        z = torch.randn(2, config.max_z_len // config.n_layers, config.dim)
        
        with torch.no_grad():
            output = model.generate_with_cache(idx, z, max_new_tokens=5)
        
        assert output.shape == (2, 15), f"Expected output shape (2, 15), got {output.shape}"
        
        # Test cache clearing
        model.clear_kv_cache()
        assert model.kv_cache is None, "KV cache should be cleared"
        
        print("✅ KV cache test passed")
        
    except Exception as e:
        print(f"❌ KV cache test failed: {e}")
        import traceback
        traceback.print_exc()

def benchmark_performance():
    """Benchmark model performance with optimizations"""
    print("Benchmarking performance...")
    
    try:
        # Create model configuration
        config = LTMConfig(
            dim=256,
            n_layers=4,
            n_heads=8,
            n_kv_heads=8,
            vocab_size=1000,
            multiple_of=32,
            max_seq_len=128,
            dropout=0.1,
            use_liger=False,
            use_rwkv=True,
            use_rwkv8_ffn=True,
            head_size=32,
            rwkv_mode="rwkv8",
            use_optimized_rwkv=True,
            gradient_checkpointing=True,
            hidden_dim=224
        )
        
        # Create optimized model
        model = OptimizedLatentThoughtModel(config)
        model.eval()
        
        # Create test data
        batch_size = 4
        seq_len = 64
        input_data = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # Benchmark
        print("Running benchmark...")
        benchmark_model(model, input_data, num_runs=50)
        
        print("✅ Performance benchmark completed")
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        import traceback
        traceback.print_exc()

def test_rwkv_optimization():
    """Test RWKV optimization functionality"""
    print("Testing RWKV optimization...")
    
    try:
        from rwkv_attention_optimized import RWKVAttentionOptimized
        
        # Create mock args
        class MockArgs:
            def __init__(self):
                self.dim = 256
                self.n_heads = 8
                self.n_kv_heads = 8
                self.head_size = 32
                self.dropout = 0.1
                self.norm_eps = 1e-5
        
        args = MockArgs()
        
        # Create optimized attention
        attention = RWKVAttentionOptimized(args)
        attention.eval()
        
        # Test forward pass
        batch_size = 2
        seq_len = 32
        x = torch.randn(batch_size, seq_len, args.dim)
        
        with torch.no_grad():
            output = attention(x)
        
        assert output.shape == (batch_size, seq_len, args.dim), f"Expected output shape {(batch_size, seq_len, args.dim)}, got {output.shape}"
        
        print("✅ RWKV optimization test passed")
        
    except Exception as e:
        print(f"❌ RWKV optimization test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests"""
    print("Running optimization tests...")
    print("=" * 50)
    
    test_optimized_config()
    test_memory_optimizations()
    test_optimized_model()
    test_kv_cache()
    test_rwkv_optimization()
    benchmark_performance()
    
    print("=" * 50)
    print("All tests completed!")

if __name__ == "__main__":
    main()
"""
Automatic Batch Size Finder for Optimal GPU Utilization
Finds the largest batch size that fits in GPU memory
"""

import torch
import gc
from typing import Optional, Dict, Any, Callable
import time
import numpy as np


class AutoBatchSizeFinder:
    """
    Automatically finds optimal batch size for maximum GPU utilization
    """
    
    def __init__(self, 
                 model: torch.nn.Module,
                 device: str = 'cuda',
                 target_gpu_util: float = 0.5,  # More conservative: 50% for safety
                 memory_multiplier: float = 6.0,  # Increased multiplier for RWKV + DiT
                 verbose: bool = True):
        self.model = model
        self.device = device
        self.target_gpu_util = target_gpu_util
        self.memory_multiplier = memory_multiplier
        self.verbose = verbose
        
    def find_optimal_batch_size(self,
                              create_batch_fn: Callable,
                              seq_len: int = 128,
                              min_batch_size: int = 1,
                              max_batch_size: int = 512,
                              test_training: bool = True) -> Dict[str, Any]:
        """
        Find optimal batch size with proper memory accounting
        """
        
        if self.verbose:
            # Get GPU info
            gpu_props = torch.cuda.get_device_properties(0)
            total_memory_gb = gpu_props.total_memory / 1024**3
            print(f"Finding optimal batch size for seq_len={seq_len}")
            print(f"GPU: {gpu_props.name} ({total_memory_gb:.1f} GB)")
            print(f"Target memory usage: {int(self.target_gpu_util*100)}%")
            print("="*60)
        
        # Clear cache before starting
        torch.cuda.empty_cache()
        gc.collect()
        
        # Phase 1: Find max forward-only batch size
        if self.verbose:
            print("\nPhase 1: Testing forward pass memory...")
        
        max_forward_batch = self._binary_search_batch_size(
            create_batch_fn, seq_len, min_batch_size, max_batch_size, 
            test_backward=False
        )
        
        if self.verbose:
            print(f"\nMax forward-only batch size: {max_forward_batch}")
        
        # Phase 2: Estimate max training batch size
        if test_training:
            # Account for gradient storage, optimizer states, and activation checkpoints
            estimated_max_training = max(1, int(max_forward_batch / self.memory_multiplier))
            
            if self.verbose:
                print(f"Estimated max training batch size: {estimated_max_training}")
                print("\nPhase 2: Testing full training memory...")
            
            # Binary search for actual training batch size
            max_training_batch = self._binary_search_batch_size(
                create_batch_fn, seq_len, 
                min_batch_size, min(estimated_max_training * 2, max_forward_batch),
                test_backward=True
            )
        else:
            max_training_batch = max_forward_batch
        
        # Apply safety margin based on target GPU utilization
        safe_batch_size = max(1, int(max_training_batch * self.target_gpu_util))
        
        if self.verbose:
            print(f"\nApplying safety margin: {max_training_batch} → {safe_batch_size}")

        # Calculate gradient accumulation
        target_effective_batch = 32  # Reasonable effective batch size
        grad_accum_steps = max(1, target_effective_batch // safe_batch_size)
        
        # Test final configuration
        torch.cuda.empty_cache()
        gc.collect()
        
        success, throughput, memory_gb = self._test_batch_size(
            safe_batch_size, seq_len, create_batch_fn, 
            test_backward=test_training, num_iters=1
        )
        
        if not success:
            # If safe batch size fails, reduce further
            safe_batch_size = max(1, safe_batch_size // 2)
            success, throughput, memory_gb = self._test_batch_size(
                safe_batch_size, seq_len, create_batch_fn,
                test_backward=test_training, num_iters=1
            )
        
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        result = {
            'batch_size': safe_batch_size,
            'max_tested': max_training_batch,
            'throughput': throughput,
            'memory_gb': memory_gb,
            'gpu_memory_usage': memory_gb / total_memory_gb,
            'gradient_accumulation_steps': grad_accum_steps,
            'effective_batch_size': safe_batch_size * grad_accum_steps,
            'seq_len': seq_len
        }
        
        if self.verbose:
            print("\n" + "="*60)
            print("OPTIMAL SETTINGS:")
            print(f"  Batch size: {result['batch_size']}")
            print(f"  Max tested: {result['max_tested']}")
            print(f"  Throughput: {result['throughput']:,.0f} tokens/sec")
            print(f"  Memory usage: {result['memory_gb']:.1f} GB ({result['gpu_memory_usage']*100:.0f}%)")
            print(f"  Gradient accumulation: {result['gradient_accumulation_steps']} steps")
            print(f"  Effective batch size: {result['effective_batch_size']}")
            print("="*60)
        
        return result
    
    def _binary_search_batch_size(self, create_batch_fn, seq_len, 
                                 min_size, max_size, test_backward=False):
        """Binary search for maximum working batch size"""
        
        left, right = min_size, max_size
        max_working = min_size
        
        while left <= right:
            mid = (left + right) // 2
            
            if self.verbose:
                print(f"\nTesting batch_size={mid}...", end=' ', flush=True)
            
            try:
                success, _, memory_gb = self._test_batch_size(
                    mid, seq_len, create_batch_fn, 
                    test_backward=test_backward, num_iters=1
                )
                
                if success:
                    max_working = mid
                    if self.verbose:
                        print(f"✓ {memory_gb:.1f} GB")
                    left = mid + 1
                else:
                    if self.verbose:
                        print("✗ OOM")
                    right = mid - 1
                    
            except Exception as e:
                if self.verbose:
                    error_msg = str(e)
                    if 'shape' in error_msg:
                        print(f"✗ Error: {error_msg[:50]}...")
                    else:
                        print(f"✗ Error: {error_msg[:100]}...")
                right = mid - 1
        
        return max_working
    
    def _test_batch_size(self, batch_size: int, seq_len: int, 
                        create_batch_fn: Callable, 
                        test_backward: bool = False,
                        num_iters: int = 3):
        """Test if a batch size works"""
        
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            # Create batch
            batch = create_batch_fn(batch_size, seq_len)
            
            # Move to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
            elif isinstance(batch, (list, tuple)):
                batch = [x.to(self.device) if torch.is_tensor(x) else x 
                        for x in batch]
            else:
                batch = batch.to(self.device) if torch.is_tensor(batch) else batch
            
            # Test forward (and backward if training)
            if test_backward:
                # Simulate training memory usage
                self.model.train()
                
                for _ in range(2):  # Warmup
                    if isinstance(batch, dict):
                        output = self.model(**batch)
                    elif isinstance(batch, (list, tuple)):
                        output = self.model(*batch)
                    else:
                        output = self.model(batch)
                    
                    # Get loss (assuming model returns loss or logits)
                    if isinstance(output, tuple):
                        loss = output[0] if torch.is_tensor(output[0]) else output[1]
                    else:
                        loss = output
                    
                    if loss.dim() > 0:
                        loss = loss.mean()
                    
                    loss.backward()
                    
                    # Clear gradients but keep allocated memory
                    self.model.zero_grad(set_to_none=False)
            else:
                # Just test forward
                self.model.eval()
                with torch.no_grad():
                    for _ in range(2):  # Warmup
                        if isinstance(batch, dict):
                            _ = self.model(**batch)
                        elif isinstance(batch, (list, tuple)):
                            _ = self.model(*batch)
                        else:
                            _ = self.model(batch)
            
            # Measure throughput
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            for _ in range(num_iters):
                if test_backward:
                    if isinstance(batch, dict):
                        output = self.model(**batch)
                    elif isinstance(batch, (list, tuple)):
                        output = self.model(*batch)
                    else:
                        output = self.model(batch)
                    
                    if isinstance(output, tuple):
                        loss = output[0] if torch.is_tensor(output[0]) else output[1]
                    else:
                        loss = output
                    
                    if loss.dim() > 0:
                        loss = loss.mean()
                    
                    loss.backward()
                    self.model.zero_grad(set_to_none=False)
                else:
                    with torch.no_grad():
                        if isinstance(batch, dict):
                            _ = self.model(**batch)
                        elif isinstance(batch, (list, tuple)):
                            _ = self.model(*batch)
                        else:
                            _ = self.model(batch)
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            
            # Calculate throughput
            throughput = (batch_size * seq_len * num_iters) / elapsed if elapsed > 0 else 0
            
            # Get memory usage
            memory_gb = torch.cuda.memory_allocated() / 1024**3
            
            return True, throughput, memory_gb
            
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            error_str = str(e).lower()
            if 'out of memory' in error_str or 'oom' in error_str:
                torch.cuda.empty_cache()
                gc.collect()
                return False, 0, 0
            else:
                # Re-raise if not OOM
                raise e
        except Exception as e:
            # Handle other errors (like shape mismatches)
            torch.cuda.empty_cache()
            gc.collect()
            return False, 0, 0


def auto_adjust_batch_size(config):
    """
    Automatically adjust batch size in config for optimal GPU usage
    """
    print("🔍 Auto-detecting optimal batch size...")
    
    # Create simplified test model that mimics the memory footprint
    class TestModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            # Mimic the memory footprint of the actual model
            self.embedding = torch.nn.Embedding(config.vocab_size, config.dim)
            
            # Use more memory-intensive layers to better simulate RWKV + DiT
            self.layers = torch.nn.ModuleList([
                torch.nn.TransformerEncoderLayer(
                    d_model=config.dim,
                    nhead=config.n_heads,
                    dim_feedforward=config.dim * 4,
                    dropout=config.dropout,
                    batch_first=True
                ) for _ in range(config.n_layers)
            ])
            
            # Add cross-attention layers to simulate RWKV cross-attention
            self.cross_layers = torch.nn.ModuleList([
                torch.nn.Linear(config.dim, config.dim) 
                for _ in range(config.n_layers)
            ])
            
            self.output = torch.nn.Linear(config.dim, config.vocab_size)
            
            # Add extra layers to simulate DiT prior memory
            if config.use_dit_prior:
                self.extra_layers = torch.nn.ModuleList([
                    torch.nn.TransformerEncoderLayer(
                        d_model=config.dim,
                        nhead=8,
                        dim_feedforward=config.dim * 2,
                        dropout=config.dropout,
                        batch_first=True
                    ) for _ in range(config.n_prior_layers)
                ])
        
        def forward(self, input_ids, labels=None, **kwargs):
            x = self.embedding(input_ids)
            
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < len(self.cross_layers):
                    x = x + self.cross_layers[i](x)
            
            if hasattr(self, 'extra_layers'):
                for layer in self.extra_layers:
                    x = layer(x)
            
            logits = self.output(x)
            
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1)
                )
                return loss
            return logits
    
    # Initialize test model
    test_model = TestModel(config).cuda()
    num_params = sum(p.numel() for p in test_model.parameters())
    print(f"Test model initialized with {num_params:,} parameters")
    
    # Print memory multiplier info
    memory_multiplier = 6.0  # More conservative for RWKV + DiT
    print(f"Memory multiplier for training: {memory_multiplier}x")
    
    # Create batch function
    def create_batch(batch_size, seq_len):
        # Create dummy tensors matching expected shapes
        return {
            'input_ids': torch.randint(0, config.vocab_size, (batch_size, seq_len)),
            'labels': torch.randint(0, config.vocab_size, (batch_size, seq_len))
        }
    
    # Find optimal batch size with conservative settings
    finder = AutoBatchSizeFinder(
        test_model, 
        target_gpu_util=0.5,  # Only use 50% for safety
        memory_multiplier=memory_multiplier,
        verbose=True
    )
    
    result = finder.find_optimal_batch_size(
        create_batch_fn=create_batch,
        seq_len=config.max_seq_len,
        min_batch_size=1,
        max_batch_size=5120, 
        test_training=True
    )
    
    # Clean up
    del test_model
    torch.cuda.empty_cache()
    gc.collect()
    
    # Update config
    config.batch_size = result['batch_size']
    config.gradient_accumulation_steps = result['gradient_accumulation_steps']
    
    print(f"\n✅ Config updated:")
    print(f"   batch_size = {config.batch_size}")
    print(f"   gradient_accumulation_steps = {config.gradient_accumulation_steps}")
    print(f"   effective_batch_size = {config.batch_size * config.gradient_accumulation_steps}")
    
    return config
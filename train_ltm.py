"""
Main training script for Latent Thought Language Model
"""
import math
import os
import time
from contextlib import nullcontext
from functools import partial
import torch
import torch.nn.functional as F
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

# Import configuration
import config
from model import LatentThoughtModel, LTMConfig
from optimizer import PosteriorOptimizer
from owt import Task
from simcse_integration import create_simcse_integration, SentenceSimilarityEvaluator
from auto_batch_size import auto_adjust_batch_size

# Before training, auto-detect optimal batch size:
if config.auto_batch_size:  # Add this flag to config
    config = auto_adjust_batch_size(config)

# Memory optimization utilities
def optimize_memory_usage():
    """Optimize PyTorch memory usage"""
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def monitor_memory():
    """Monitor and log memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
    import psutil
    import os
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"RAM Usage - RSS: {memory_info.rss / 1024**2:.2f}MB, VMS: {memory_info.vms / 1024**2:.2f}MB")

def safe_dataloader_init(dataloader, max_retries=3):
    """Safely initialize DataLoader with retry logic and memory monitoring"""
    import time
    for attempt in range(max_retries):
        try:
            # Monitor memory before DataLoader initialization
            print(f"DataLoader attempt {attempt + 1} - Memory before:")
            monitor_memory()
            
            # Get first batch to test DataLoader
            first_batch = next(iter(dataloader))
            
            # Monitor memory after DataLoader initialization
            print(f"DataLoader attempt {attempt + 1} - Memory after:")
            monitor_memory()
            
            return True, None
        except Exception as e:
            print(f"DataLoader initialization attempt {attempt + 1} failed: {e}")
            # Force memory cleanup on failure
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            
            if attempt < max_retries - 1:
                print("Retrying in 3 seconds...")
                time.sleep(3)
                # Try reducing batch size or workers
                if hasattr(dataloader, 'batch_sampler'):
                    # This is a simplified approach - in practice you might want more sophisticated logic
                    pass
            else:
                return False, e
    return False, "Max retries exceeded"

def enable_tf32():
    """Enable TF32 precision for better performance on Ampere+ GPUs"""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def enable_flash_attention():
    """Enable Flash Attention if available"""
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        print("Flash Attention is available and will be used")
        return True
    else:
        print("Flash Attention not available, using standard attention")
        return False

def main():
    """Main training function."""
    
    # -----------------------------------------------------------------------------
    # Memory Optimization Setup
    # -----------------------------------------------------------------------------
    # Apply memory optimizations at the start
    optimize_memory_usage()
    enable_tf32()
    enable_flash_attention()
    
    # -----------------------------------------------------------------------------
    # Distributed Training Setup
    # -----------------------------------------------------------------------------
    
    # Check if this is a distributed data parallel (DDP) run
    ddp = int(os.environ.get("RANK", -1)) != -1
    print(f"Using DDP for training: {ddp}")
    
    # Local variables to track the current training state
    iter_num = 0
    best_val_loss = 1e9
    ddp_world_size = 1
    gradient_accumulation_steps = config.gradient_accumulation_steps
    device = config.device
    
    if ddp:
        # Initialize the distributed process group
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])  # Global rank of this process
        ddp_local_rank = int(os.environ["LOCAL_RANK"])  # Local rank on this node
        ddp_world_size = int(os.environ["WORLD_SIZE"])  # Total number of processes
        device = f"cuda:{ddp_local_rank}"
        print(f"DDP setup complete. Using device: {device}")
        torch.cuda.set_device(device)
        master_process = (ddp_rank == 0)  # Process responsible for logging and checkpoints
        seed_offset = ddp_rank  # Each process gets a different seed
        
        # Scale down gradient accumulation steps proportionally to world size
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
        print(f"Adjusted gradient accumulation steps: {gradient_accumulation_steps}")
    else:
        print("Single GPU training (no DDP)")
        master_process = True
        seed_offset = 0
    
    # Calculate tokens processed per iteration for logging
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * config.batch_size * config.max_seq_len
    if master_process:
        print(f"Tokens per iteration: {tokens_per_iter:,}")
        print(f"  = {gradient_accumulation_steps} accumulation steps * {ddp_world_size} processes * {config.batch_size} batch size * {config.max_seq_len} sequence length")
    
    # Create output directories on the master process
    if master_process:
        os.makedirs(config.out_dir, exist_ok=True)    
    # -----------------------------------------------------------------------------
    # Initialization and Setup
    # -----------------------------------------------------------------------------
    
    # Set random seed for reproducibility
    torch.manual_seed(1337 + seed_offset)
    
    # Enable TF32 precision for better performance on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Apply memory optimizations
    if config.memory_optimization:
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("Memory optimizations applied")
    
    # Device and precision setup
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {
        "float32": torch.float32, 
        "bfloat16": torch.bfloat16, 
        "float16": torch.float16
    }[config.dtype]
    
    # Context manager for mixed precision training
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )
    
    # -----------------------------------------------------------------------------
    # Data Loading Setup
    # -----------------------------------------------------------------------------
    
    # Set up data iterator with latent variables
    iter_batches = partial(
        Task.iter_batches_with_latents,
        batch_size=config.batch_size,
        max_seq_len=config.max_seq_len,
        max_z_len=config.max_z_len,
        z_dim=config.z_dim,
        device=device,
        num_workers=config.num_workers,
        auto_download=config.DATASET_AUTO_DOWNLOAD,
    )
    
    # -----------------------------------------------------------------------------
    # Model Initialization
    # -----------------------------------------------------------------------------
    
    # Define model architecture parameters
    model_args = dict(
        dim=config.dim,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads,
        vocab_size=config.vocab_size,
        multiple_of=config.multiple_of,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        n_prior_layers=config.n_prior_layers,
        n_cls_tokens=config.n_cls_tokens,
        window_size=config.window_size,
        use_liger=config.use_liger,
        max_z_len=config.max_z_len,
        use_z_pos_emb=True,  # Use positional embeddings for latent variables
        use_rwkv=config.use_rwkv,  # Use RWKV instead of transformer
        use_rwkv8_ffn=config.use_rwkv8_ffn,  # Use RWKV-8 feed-forward network
        head_size=config.head_size,  # RWKV head size
        rwkv_mode=config.rwkv_mode,  # RWKV mode: "rwkv7" or "rwkv8"
        use_optimized_rwkv=config.use_optimized_rwkv,
        gradient_checkpointing=config.gradient_checkpointing,
        # DiT prior parameters
        use_dit_prior=config.use_dit_prior,
        dit_layers=config.dit_layers,
        dit_heads=config.dit_heads,
        dit_dim=config.dit_dim,
        dit_multiple_of=config.dit_multiple_of,
        dit_num_timesteps=config.dit_num_timesteps,
        dit_beta_schedule=config.dit_beta_schedule,
        dit_beta_start=config.dit_beta_start,
        dit_beta_end=config.dit_beta_end,
        # Additional optimization parameters
        norm_eps=1e-5,  # Default normalization epsilon
        padding=False,  # Default padding setting
    )
    
    if config.init_from == "scratch":
        # Initialize a new model from scratch
        print("Initializing a new model from scratch")
        gptconf = LTMConfig(**model_args)
        model = LatentThoughtModel(gptconf)
        print(model)
    elif config.init_from == "resume":
        print(f"Resuming training from checkpoint: {config.ckpt_path}")
        # Resume training from a checkpoint
        checkpoint = torch.load(config.ckpt_path, map_location=device)
        
        # Use architecture parameters from checkpoint
        checkpoint_model_args = checkpoint["model_args"]
        for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len",
                  "use_dit_prior", "dit_layers", "dit_heads", "dit_dim", "dit_multiple_of",
                  "dit_num_timesteps", "dit_beta_schedule", "dit_beta_start", "dit_beta_end"]:
            if k in checkpoint_model_args:
                model_args[k] = checkpoint_model_args[k]
        
        # Create model with checkpoint configuration
        gptconf = LTMConfig(**model_args)
        model = LatentThoughtModel(gptconf)
        
        # Load model weights from checkpoint, handling DDP prefixes if present
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
        
        # Restore training state
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
    
    # Move model to appropriate device
    model.to(device)
    
    # -----------------------------------------------------------------------------
    # Optimizer and Precision Setup
    # -----------------------------------------------------------------------------
    
    # Set up gradient scaler for mixed precision training (no-op if not float16)
    scaler = torch.amp.GradScaler(enabled=(config.dtype == "float16"), device='cuda')
    
    # Initialize optimizer with weight decay
    optimizer = model.configure_optimizers(
        config.weight_decay, 
        config.learning_rate, 
        (config.beta1, config.beta2), 
        device_type
    )
    
    # Load optimizer state if resuming from checkpoint
    if config.init_from == "resume" and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # Free up memory
    
    # Compile model for performance if enabled (requires PyTorch 2.0+)
    if config.compile:
        print("Compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)
    
    # -----------------------------------------------------------------------------
    # Distributed Training Wrap-up
    # -----------------------------------------------------------------------------
    
    # Wrap model in DDP container for distributed training
    if ddp:
        # No special buffer ignores needed (freqs_cos/freqs_sin are floats)
        model = DDP(model, device_ids=[ddp_local_rank])
    
    # -----------------------------------------------------------------------------
    # Model and Posterior Optimizer Setup
    # -----------------------------------------------------------------------------
    
    print(f"Training configuration: steps={config.num_steps}, layers={config.n_layers}, "
          f"z_len={config.max_z_len}, dim={config.dim}, heads={config.n_heads}")
    
    # Get raw model by unwrapping DDP container if needed
    raw_model = model.module if ddp else model
    
    # Initialize posterior optimizers for training and evaluation
    posterior_optimizer = PosteriorOptimizer(
        model=raw_model,
        inference_method=config.inference_method,
        num_steps=config.num_steps,
        max_z_len=config.max_z_len,
        z_dim=config.z_dim,
        use_dit_prior=config.use_dit_prior,
        lr=config.fast_lr,
        eval_mode=False
    )
    
    posterior_optimizer_test = PosteriorOptimizer(
        model=raw_model,
        inference_method=config.inference_method,
        num_steps=config.num_steps,
        max_z_len=config.max_z_len,
        z_dim=config.z_dim,
        use_dit_prior=config.use_dit_prior,
        lr=config.fast_lr,
        eval_mode=True
    )
    
    # Initialize SimCSE integration if enabled
    simcse_module = None
    simcse_evaluator = None
    if config.use_simcse:
        print("Initializing SimCSE integration...")
        simcse_config = {
            'pooler_type': config.simcse_pooler_type,
            'temperature': config.simcse_temperature,
            'projection_dim': config.simcse_projection_dim,
            'use_projection_head': config.simcse_use_projection_head,
            'simcse_weight': config.simcse_weight
        }
        simcse_module = create_simcse_integration(raw_model, simcse_config)
        simcse_module.to(device)  # Move SimCSE module to the same device as the model
        simcse_evaluator = SentenceSimilarityEvaluator(simcse_module)
        print("SimCSE integration initialized successfully.")
    
    # -----------------------------------------------------------------------------
    # Training Utilities
    # -----------------------------------------------------------------------------
    
    def estimate_loss(lr=None):
        """
        Estimate loss on validation set with memory optimizations.
        """
        loss_out = {}
        ppl_out = {}
        kl_out = {}
        simcse_loss_out = {}
        
        # Use smaller batch size for validation to save memory
        eval_batch_size = max(1, config.batch_size // 4)  # Reduce batch size by 75%
        if eval_batch_size < 1:
            eval_batch_size = 1
    
        model.eval()  # Set model to evaluation mode
        for split in ["validation"]:
            batch_iter = iter_batches(split=split, batch_size=eval_batch_size)
            losses = torch.zeros(config.eval_iters)  # Track losses over evaluation iterations
            ppl_list = torch.zeros(config.eval_iters)  # Track perplexities
            kl_list = torch.zeros(config.eval_iters)  # Track KL divergences
            simcse_losses = torch.zeros(config.eval_iters) if config.use_simcse else None
            
            for k in range(config.eval_iters):
                # Get next batch
                X, Y, Z = next(batch_iter)
                
                # Optimize latent variables for this batch with fewer steps to save memory
                optimization_steps = min(config.num_steps, 8)  # Limit optimization steps
                
                Z, ppl, kl_avg, nlkhd = posterior_optimizer_test.step(
                    data=[X, Y, Z],
                    ctx=ctx,
                    scaler=scaler,
                    steps=optimization_steps,
                    lr=lr
                )
                
                # Forward pass with optimized latents
                with ctx:
                    if config.use_simcse and simcse_module is not None:
                        # Use SimCSE module for forward pass
                        results = simcse_module(
                            X, torch.ones_like(X).bool(), Y,
                            z=Z,
                            compute_contrastive=False
                        )
                        loss = results['loss']
                    else:
                        logits = model(X, Z, Y)
                        loss = raw_model.last_loss
                
                # Record metrics
                losses[k] = loss.item()
                ppl_list[k] = ppl.item()
                kl_list[k] = kl_avg.item()
                
                if config.use_simcse and simcse_losses is not None:
                    # Compute SimCSE loss on validation data
                    with torch.no_grad():
                        # Create a second batch for contrastive learning
                        try:
                            X2, Y2, Z2 = next(batch_iter)
                            simcse_result = simcse_module(
                                X, torch.ones_like(X).bool(), Y,
                                z=Z,
                                compute_contrastive=True,
                                contrastive_batch={'input_ids': X2, 'attention_mask': torch.ones_like(X2).bool(), 'z': Z2}
                            )
                            simcse_losses[k] = simcse_result['contrastive_loss'].item()
                        except StopIteration:
                            # If we run out of batches, skip SimCSE computation
                            simcse_losses[k] = 0.0
                
                # Aggressive memory cleanup to avoid OOM issues
                del X, Y, Z, loss, ppl, kl_avg, nlkhd
                if 'logits' in locals():
                    del logits
                if 'results' in locals():
                    del results
                if 'simcse_result' in locals():
                    del simcse_result
                
                # Force garbage collection and memory cleanup
                if config.memory_optimization:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Additional memory cleanup every few iterations
                if k % 10 == 0:
                    torch.cuda.empty_cache()
            
            # Compute average metrics
            loss_out[split] = losses.mean()
            ppl_out[split] = ppl_list.mean()
            kl_out[split] = kl_list.mean()
            
            if config.use_simcse and simcse_losses is not None:
                simcse_loss_out[split] = simcse_losses.mean()
        
        model.train()  # Set model back to training mode
        torch.cuda.empty_cache()
        return loss_out, ppl_out, kl_out, simcse_loss_out if config.use_simcse else None
    
    def get_lr(it):
        """
        Get learning rate for current iteration based on warmup and cosine decay.
        """
        # 1) Linear warmup for warmup_iters steps
        if it < config.warmup_iters:
            return config.learning_rate * it / config.warmup_iters
        # 2) If it > lr_decay_iters, return min learning rate
        if it > config.lr_decay_iters:
            return config.min_lr
        # 3) In between, use cosine decay down to min learning rate
        decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return config.min_lr + coeff * (config.learning_rate - config.min_lr)
    
    def fast_lr_linear_decay(epoch):
        """
        Linearly interpolate fast learning rate between initial and final values.
        """
        return config.initial_fast_lr + epoch / (config.lr_decay_iters - 1) * (config.final_fast_lr - config.initial_fast_lr)
    
    # -----------------------------------------------------------------------------
    # Main Training Loop
    # -----------------------------------------------------------------------------
    
    # Initialize training
    print("Initializing data loading...")
    train_batch_iter = iter_batches(split="train")
    
    # Test DataLoader with memory monitoring
    print("Testing DataLoader...")
    monitor_memory()
    
    try:
        X, Y, Z = next(train_batch_iter)
        print("DataLoader initialized successfully")
        monitor_memory()
    except Exception as e:
        print(f"Error initializing DataLoader: {e}")
        print("Attempting to reduce batch size and retry...")
        # Try with smaller batch size
        try:
            original_batch_size = config.batch_size
            config.batch_size = max(1, config.batch_size // 2)
            print(f"Reducing batch size to {config.batch_size}")
            train_batch_iter = iter_batches(split="train")
            X, Y, Z = next(train_batch_iter)
            print(f"DataLoader initialized successfully with reduced batch size: {config.batch_size}")
            monitor_memory()
        except Exception as e2:
            print(f"Failed to initialize DataLoader even with reduced batch size: {e2}")
            # Try with even smaller batch size
            try:
                config.batch_size = max(1, config.batch_size // 2)
                print(f"Further reducing batch size to {config.batch_size}")
                train_batch_iter = iter_batches(split="train")
                X, Y, Z = next(train_batch_iter)
                print(f"DataLoader initialized successfully with further reduced batch size: {config.batch_size}")
                monitor_memory()
            except Exception as e3:
                print(f"Failed to initialize DataLoader even with further reduced batch size: {e3}")
                # Try with minimal batch size
                try:
                    config.batch_size = 1
                    print(f"Using minimal batch size: {config.batch_size}")
                    train_batch_iter = iter_batches(split="train")
                    X, Y, Z = next(train_batch_iter)
                    print(f"DataLoader initialized successfully with minimal batch size: {config.batch_size}")
                    monitor_memory()
                except Exception as e4:
                    print(f"Failed to initialize DataLoader even with minimal batch size: {e4}")
                    print("Training cannot start due to DataLoader initialization failure.")
                    return
    
    t0 = time.time()
    local_iter_num = 0  # Iterations in current process
    running_mfu = -1.0  # Model flops utilization (efficiency metric)
    print(f"Starting training loop, max iterations: {config.max_iters}")
    
    while True:
        # Determine learning rates for this iteration
        lr = get_lr(iter_num) if config.decay_lr else config.learning_rate  # Model learning rate
        current_lr = fast_lr_linear_decay(iter_num)  # Latent variable learning rate
    
        # Update optimizer learning rates
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    
        # -----------------------------------------------------------------------------
        # Evaluation and Checkpointing
        # -----------------------------------------------------------------------------
        
        # Evaluate model and save checkpoint periodically
        if iter_num % config.eval_interval == 0 and master_process:
            losses, ppl_out, kl_out, simcse_losses = estimate_loss(current_lr)
            print(f"Step {iter_num}: val loss {losses['validation']:.4f}, val PPL {ppl_out['validation']:.4f}, val KL {kl_out['validation']:.4f}")
            if config.use_simcse and simcse_losses is not None:
                print(f"Step {iter_num}: SimCSE val loss {simcse_losses['validation']:.4f}")
    
            # Save checkpoint if validation loss improved or if always_save_checkpoint is True
            if losses["validation"] < best_val_loss or config.always_save_checkpoint:
                best_val_loss = losses["validation"]
                if iter_num > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config.get_config_dict(),
                        'rng_state': torch.random.get_rng_state()
                    }
                    ckpt_path = os.path.join(config.out_dir, f"ckpt_{iter_num}.pt")
                    print(f"Saving checkpoint to {ckpt_path}")
                    torch.save(checkpoint, ckpt_path)
    
        # -----------------------------------------------------------------------------
        # Forward and Backward Pass
        # -----------------------------------------------------------------------------
        
        # Forward and backward passes with gradient accumulation
        for micro_step in range(gradient_accumulation_steps):
            # For distributed training, only synchronize gradients on the last micro-step
            if ddp:
                model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
            
            # Optimize latent variables for current batch with reduced steps to save memory
            optimization_steps = min(config.num_steps, 8)  # Limit optimization steps
            Z, ppl, kl, _ = posterior_optimizer.step(
                data=[X, Y, Z],
                ctx=ctx,
                scaler=scaler,
                steps=optimization_steps,
                lr=current_lr
            )
            
            # Forward pass with optimized latents
            with ctx:
                if config.use_simcse and simcse_module is not None:
                    # Fetch a second view for contrastive learning
                    X2, Y2, Z2 = next(train_batch_iter)
                    
                    results = simcse_module(
                        X, torch.ones_like(X).bool(), Y,
                        z=Z,
                        compute_contrastive=True,
                        contrastive_batch={'input_ids': X2,
                                           'attention_mask': torch.ones_like(X2).bool(),
                                           'z': Z2}
                    )
                    loss = results['total_loss'] / gradient_accumulation_steps  # Scale loss for gradient accumulation
                    
                    # Reuse X2,Y2,Z2 as next batch to avoid double dataloader step
                    X, Y, Z = X2, Y2, Z2
                else:
                    logits = model(X, Z, Y)
                    loss = raw_model.last_loss
                    loss = loss / gradient_accumulation_steps  # Scale loss for gradient accumulation
                    
                    # Prefetch next batch asynchronously while GPU is busy
                    X, Y, Z = next(train_batch_iter)
            
            # Backward pass with gradient scaling for mixed precision
            scaler.scale(loss).backward()
            
            # Memory cleanup during training
            if micro_step % 2 == 0:  # Clean up every other micro step
                torch.cuda.empty_cache()
        
        # -----------------------------------------------------------------------------
        # Gradient Processing and Optimizer Step
        # -----------------------------------------------------------------------------
        
        # Apply gradient clipping if enabled
        if config.grad_clip != 0.0:
            scaler.unscale_(optimizer)  # Unscale gradients for clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
        # Update model parameters
        scaler.step(optimizer)
        scaler.update()
        
        # Clear gradients to free memory
        optimizer.zero_grad(set_to_none=True)
        
        # Apply memory optimization periodically
        if iter_num % 50 == 0 and config.memory_optimization:
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            # Monitor memory every 50 iterations
            if master_process:
                print(f"Memory usage at iteration {iter_num}:")
                monitor_memory()
                
        # Emergency memory cleanup if memory usage is too high
        if iter_num % 25 == 0 and config.memory_optimization:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                if allocated > 6.0:  # If GPU memory > 6GB
                    print(f"Emergency memory cleanup at iteration {iter_num} (allocated: {allocated:.2f}GB)")
                    torch.cuda.empty_cache()
                    gc.collect()
    
        # -----------------------------------------------------------------------------
        # Logging and Timing
        # -----------------------------------------------------------------------------
        
        # Calculate timing and log progress
        t1 = time.time()
        dt = t1 - t0  # Time for this iteration
        t0 = t1
        
        if iter_num % config.log_interval == 0 and master_process:
            # Get loss as float, scale up due to gradient accumulation
            lossf = loss.item() * gradient_accumulation_steps
            
            # Calculate model flops utilization (efficiency metric)
            if local_iter_num >= 5:  # Skip first few iterations for warm-up
                mfu = raw_model.estimate_mfu(config.batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                
            # Log training progress
            print(
                f"{iter_num} | loss {lossf:.4f} | ppl {ppl:.4f} | kl {kl:.4f} | "
                f"lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%"
            )
        
        # Update iteration counters
        iter_num += 1
        local_iter_num += 1
    
        # Check for termination condition
        if iter_num > config.max_iters:
            print(f"Training completed after {iter_num} iterations.")
            break
    
    # -----------------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------------
    
    # Clean up distributed training resources
    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
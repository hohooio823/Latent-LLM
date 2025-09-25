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
if getattr(config, "auto_batch_size", False):
    config = auto_adjust_batch_size(config)


def optimize_memory_usage():
    """Optimize PyTorch memory usage"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def monitor_memory():
    """Monitor and log memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {reserved:.2f}GB")
    import psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(
        f"RAM Usage - RSS: {memory_info.rss / 1024**2:.2f}MB, VMS: {memory_info.vms / 1024**2:.2f}MB"
    )


def enable_tf32():
    """Enable TF32 precision for better performance on Ampere+ GPUs"""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def enable_flash_attention():
    """Enable Flash Attention if available"""
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        print("Flash Attention is available and will be used")
        return True
    else:
        print("Flash Attention not available, using standard attention")
        return False


def main():
    """Main training function."""
    optimize_memory_usage()
    enable_tf32()
    enable_flash_attention()

    # -----------------------------------------------------------------------------
    # Distributed
    # -----------------------------------------------------------------------------
    ddp = int(os.environ.get("RANK", -1)) != -1
    print(f"Using DDP for training: {ddp}")

    iter_num = 0
    best_val_loss = 1e9
    ddp_world_size = 1
    gradient_accumulation_steps = config.gradient_accumulation_steps
    device = config.device

    if ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        print(f"DDP setup complete. Using device: {device}")
        torch.cuda.set_device(device)
        master_process = (ddp_rank == 0)
        seed_offset = ddp_rank
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
        print(f"Adjusted gradient accumulation steps: {gradient_accumulation_steps}")
    else:
        print("Single GPU training (no DDP)")
        master_process = True
        seed_offset = 0

    tokens_per_iter = (
        gradient_accumulation_steps
        * ddp_world_size
        * config.batch_size
        * config.max_seq_len
    )
    if master_process:
        print(f"Tokens per iteration: {tokens_per_iter:,}")
        print(
            f"  = {gradient_accumulation_steps} accumulation steps * {ddp_world_size} processes * {config.batch_size} batch size * {config.max_seq_len} sequence length"
        )

    if master_process:
        os.makedirs(config.out_dir, exist_ok=True)

    # -----------------------------------------------------------------------------
    # Init
    # -----------------------------------------------------------------------------
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
        config.dtype
    ]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # -----------------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------------
    iter_batches = partial(
        Task.iter_batches_with_latents,
        batch_size=config.batch_size,
        max_seq_len=config.max_seq_len,
        max_z_len=config.max_z_len,
        z_dim=config.z_dim,
        device=device,
        num_workers=getattr(config, "num_workers", 0),
        auto_download=getattr(config, "DATASET_AUTO_DOWNLOAD", False),
    )

    # -----------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------
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
        use_liger=getattr(config, "use_liger", True),
        max_z_len=config.max_z_len,
        use_z_pos_emb=True,
        use_rwkv=getattr(config, "use_rwkv", False),
        use_rwkv8_ffn=getattr(config, "use_rwkv8_ffn", False),
        head_size=getattr(config, "head_size", 64),
        rwkv_mode=getattr(config, "rwkv_mode", "rwkv8"),
        use_optimized_rwkv=getattr(config, "use_optimized_rwkv", False),
        gradient_checkpointing=getattr(config, "gradient_checkpointing", False),
        # DiT prior params
        use_dit_prior=getattr(config, "use_dit_prior", False),
        dit_layers=getattr(config, "dit_layers", 0),
        dit_heads=getattr(config, "dit_heads", 0),
        dit_dim=getattr(config, "dit_dim", 0),
        dit_multiple_of=getattr(config, "dit_multiple_of", 32),
        dit_num_timesteps=getattr(config, "dit_num_timesteps", 1000),
        dit_beta_schedule=getattr(config, "dit_beta_schedule", "linear"),
        dit_beta_start=getattr(config, "dit_beta_start", 0.0001),
        dit_beta_end=getattr(config, "dit_beta_end", 0.02),
        norm_eps=1e-5,
        padding=False,
    )

    print("Initializing a new model from scratch")
    gptconf = LTMConfig(**model_args)
    model = LatentThoughtModel(gptconf)
    model.to(device)
    if master_process:
        print(model)

    # -----------------------------------------------------------------------------
    # Optimizer & compile
    # -----------------------------------------------------------------------------
    scaler = torch.amp.GradScaler(enabled=(config.dtype == "float16"), device="cuda")
    optimizer = model.configure_optimizers(
        config.weight_decay, config.learning_rate, (config.beta1, config.beta2), device_type
    )

    if getattr(config, "compile", False):
        print("Compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model = model.module if ddp else model

    print(
        f"Training configuration: steps={config.num_steps}, layers={config.n_layers}, z_len={config.max_z_len}, dim={config.dim}, heads={config.n_heads}"
    )

    # Posterior optimizers
    posterior_optimizer = PosteriorOptimizer(
        model=raw_model,
        inference_method=config.inference_method,
        num_steps=config.num_steps,
        max_z_len=config.max_z_len,
        z_dim=config.z_dim,
        use_dit_prior=getattr(config, "use_dit_prior", False),
        lr=config.fast_lr,
        eval_mode=False,  # training VI uses training branch
    )

    posterior_optimizer_test = PosteriorOptimizer(
        model=raw_model,
        inference_method=config.inference_method,
        num_steps=config.num_steps,
        max_z_len=config.max_z_len,
        z_dim=config.z_dim,
        use_dit_prior=getattr(config, "use_dit_prior", False),
        lr=config.fast_lr,
        eval_mode=True,  # validation VI uses eval branch
    )

    # Optional SimCSE
    simcse_module = None
    simcse_evaluator = None
    if getattr(config, "use_simcse", False):
        print("Initializing SimCSE integration...")
        simcse_config = {
            "pooler_type": config.simcse_pooler_type,
            "temperature": config.simcse_temperature,
            "projection_dim": config.simcse_projection_dim,
            "use_projection_head": config.simcse_use_projection_head,
            "simcse_weight": config.simcse_weight,
        }
        simcse_module = create_simcse_integration(raw_model, simcse_config)
        simcse_module.to(device)
        simcse_evaluator = SentenceSimilarityEvaluator(simcse_module)
        print("SimCSE integration initialized successfully.")

    # -----------------------------------------------------------------------------
    # Eval utils
    # -----------------------------------------------------------------------------
    def estimate_loss(lr=None):
        loss_out = {}
        ppl_out = {}
        kl_out = {}
        simcse_loss_out = {}

        eval_batch_size = max(1, config.batch_size // 4)
        model.eval()
        for split in ["validation"]:
            batch_iter = iter_batches(split=split, batch_size=eval_batch_size)
            losses = torch.zeros(config.eval_iters)
            ppl_list = torch.zeros(config.eval_iters)
            kl_list = torch.zeros(config.eval_iters)
            simcse_losses = torch.zeros(config.eval_iters) if getattr(config, "use_simcse", False) else None

            for k in range(config.eval_iters):
                X, Y, Z = next(batch_iter)

                Z, ppl, kl_avg, nlkhd = posterior_optimizer_test.step(
                    data=[X, Y, Z], ctx=ctx, scaler=scaler, steps=config.num_steps, lr=lr
                )

                with ctx:
                    if getattr(config, "use_simcse", False) and simcse_module is not None:
                        results = simcse_module(
                            X, torch.ones_like(X).bool(), Y, z=Z, compute_contrastive=False
                        )
                        loss = results["loss"]
                    else:
                        _ = model(X, Z, Y)
                        loss = raw_model.last_loss

                losses[k] = loss.item()
                ppl_list[k] = ppl.item()
                kl_list[k] = kl_avg.item()

                if getattr(config, "use_simcse", False) and simcse_losses is not None:
                    with torch.no_grad():
                        try:
                            X2, Y2, Z2 = next(batch_iter)
                            simcse_result = simcse_module(
                                X,
                                torch.ones_like(X).bool(),
                                Y,
                                z=Z,
                                compute_contrastive=True,
                                contrastive_batch={
                                    "input_ids": X2,
                                    "attention_mask": torch.ones_like(X2).bool(),
                                    "z": Z2,
                                },
                            )
                            simcse_losses[k] = simcse_result["contrastive_loss"].item()
                        except StopIteration:
                            simcse_losses[k] = 0.0

                del X, Y, Z, loss, ppl, kl_avg, nlkhd
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

            loss_out[split] = losses.mean()
            ppl_out[split] = ppl_list.mean()
            kl_out[split] = kl_list.mean()

            if getattr(config, "use_simcse", False) and simcse_losses is not None:
                simcse_loss_out[split] = simcse_losses.mean()

        model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return loss_out, ppl_out, kl_out, (simcse_loss_out if getattr(config, "use_simcse", False) else None)

    def get_lr(it):
        if config.decay_lr:
            if it < config.warmup_iters:
                return config.learning_rate * it / config.warmup_iters
            if it > config.lr_decay_iters:
                return config.min_lr
            decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return config.min_lr + coeff * (config.learning_rate - config.min_lr)
        else:
            return config.learning_rate

    def fast_lr_linear_decay(epoch):
        return config.initial_fast_lr + epoch / (config.lr_decay_iters - 1) * (config.final_fast_lr - config.initial_fast_lr)

    # -----------------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------------
    print("Initializing data loading...")
    train_batch_iter = iter_batches(split="train")
    print("Testing DataLoader...")
    monitor_memory()
    X, Y, Z = next(train_batch_iter)
    print("DataLoader initialized successfully")
    monitor_memory()

    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0
    print(f"Starting training loop, max iterations: {config.max_iters}")

    while True:
        lr = get_lr(iter_num)
        current_lr = fast_lr_linear_decay(iter_num)

        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if iter_num % config.eval_interval == 0 and master_process:
            losses, ppl_out, kl_out, simcse_losses = estimate_loss(current_lr)
            print(
                f"Step {iter_num}: val loss {losses['validation']:.4f}, val PPL {ppl_out['validation']:.4f}, val KL {kl_out['validation']:.4f}"
            )
            if getattr(config, "use_simcse", False) and simcse_losses is not None:
                print(f"Step {iter_num}: SimCSE val loss {simcse_losses['validation']:.4f}")

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
                        "rng_state": torch.random.get_rng_state(),
                    }
                    ckpt_path = os.path.join(config.out_dir, f"ckpt_{iter_num}.pt")
                    print(f"Saving checkpoint to {ckpt_path}")
                    torch.save(checkpoint, ckpt_path)

        # Gradient accumulation
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1

            Z, ppl, kl, _ = posterior_optimizer.step(
                data=[X, Y, Z], ctx=ctx, scaler=scaler, steps=config.num_steps, lr=current_lr
            )

            with ctx:
                if getattr(config, "use_simcse", False) and simcse_module is not None:
                    X2, Y2, Z2 = next(train_batch_iter)
                    results = simcse_module(
                        X,
                        torch.ones_like(X).bool(),
                        Y,
                        z=Z,
                        compute_contrastive=True,
                        contrastive_batch={
                            "input_ids": X2,
                            "attention_mask": torch.ones_like(X2).bool(),
                            "z": Z2,
                        },
                    )
                    loss = results["total_loss"] / gradient_accumulation_steps
                    X, Y, Z = X2, Y2, Z2  # reuse second view as next batch
                else:
                    _ = model(X, Z, Y)
                    loss = raw_model.last_loss / gradient_accumulation_steps
                    X, Y, Z = next(train_batch_iter)

            scaler.scale(loss).backward()

            if micro_step % 2 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if config.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if iter_num % config.log_interval == 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(config.batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

            print(
                f"{iter_num} | loss {lossf:.4f} | ppl {ppl:.4f} | kl {kl:.4f} | lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%"
            )

        iter_num += 1
        local_iter_num += 1

        if iter_num > config.max_iters:
            print(f"Training completed after {iter_num} iterations.")
            break

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()

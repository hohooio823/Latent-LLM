import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial
import numpy as np
import threading
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from model import MultiLayerLatentPromptTransformer, ModelArgs
from optimizer import PosteriorOptimizer
from dataloader.owt_gpt2 import Task

# -----------------------------------------------------------------------------
# I/O
tag = "owt_mlpt_"
out_dir = "output/owt_liger_mlpt/" + tag + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
eval_interval = 1000
log_interval = 1
eval_iters = 100 # Default: 100
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume' 
ckpt_path = ''
num_steps = 16
inference_method = 'adamVI' 
cold_start = True # A cold start is different from a short run. It denotes starting from a “fixed” random beginning.
# wandb logging
wandb_log = False  
wandb_proj_name = "relm_owt_new"
# data
DATA_CACHE_DIR = "data_owt"
batch_size = 64  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_seq_len = 1024
vocab_size = 50258  
# model
dim = 768
n_layers = 12
n_heads = 12
n_kv_heads = 12
multiple_of = 32
dropout = 0.0
window_size = 256
# prior p(z)
initial_fast_lr = 0.3
final_fast_lr = 0.34
fast_lr = 0.34
n_prior_layers = 0
n_cls_tokens = 0
max_z_len = n_layers * 8
z_dim = dim
z_dir = out_dir + '/owt_gpt2_z' 
joint_training = "decoder" 
# adamw optimizer
gradient_accumulation_steps = 8  # used to simulate larger batch sizes
learning_rate = 4e-4  # max learning rate
max_iters = 60000  # total number of training iterations ~ 30B tokens
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 1000  # how many steps to warm up for
# system
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = "bfloat16"  # float32|bfloat16|float16
compile = False  # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# fixing some hyperparams to sensible defaults
lr_decay_iters = max_iters  # should be ~= max_iters per Chinchilla
min_lr = 4e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
print(f"using ddp to train {ddp}")
if ddp:
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    print(device)
    torch.cuda.set_device(device)
    master_process = ddp_rank == 1  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
    print("done")
else:
    print("not ddp, we are running on a single gpu, and one process")
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len
if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    print(f"breaks down as: {gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {batch_size} batch size * {max_seq_len} max seq len")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(z_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda"# if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# task-specific setup
iter_batches = partial(
    Task.iter_batches_with_latents,
    batch_size=batch_size,
    max_seq_len=max_seq_len,
    max_z_len=max_z_len,
    z_dim=z_dim,
    device=device,
    num_workers=0,
    z_dir=z_dir
)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    vocab_size=vocab_size,
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout,
    n_prior_layers=n_prior_layers,
    n_cls_tokens=n_cls_tokens,
    window_size=window_size,
    use_liger=True,
    max_z_len=max_z_len,
    use_z_pos_emb=True,
)  # start with model_args from command line
if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    gptconf = ModelArgs(**model_args)
    model = MultiLayerLatentPromptTransformer(gptconf)
    print(model)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ########################################
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = ModelArgs(**model_args)
    model = MultiLayerLatentPromptTransformer(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(enabled=(dtype == "float16"), device='cuda')
# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if init_from == "resume" and "optimizer" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    # Ignore the `freqs_cis` buffer so that DDP does not broadcast it at
    # construction time since NCCL does not support `ComplexFloat`
    prefix = "_orig_mod." if compile else ""
    model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
    model = DDP(model, device_ids=[ddp_local_rank])

print(f"steps {num_steps}, {n_layers} layers, z_len {max_z_len}, dim= {dim}, {n_heads} heads")
# Inference method
raw_model = model.module if ddp else model  # unwrap DDP container if needed
posterior_optimizer = PosteriorOptimizer(model = raw_model, inference_method=inference_method, num_steps=num_steps, max_z_len=max_z_len, z_dim=z_dim, lr=fast_lr, eval_mode = False)
posterior_optimizer_test = PosteriorOptimizer(model = raw_model, inference_method=inference_method, num_steps=num_steps, max_z_len=max_z_len, z_dim=z_dim, lr=fast_lr, eval_mode = True)

def estimate_loss(lr = None):
    loss_out = {}
    ppl_out = {}
    kl_out = {}

    model.eval()
    for split in ["val"]:
        batch_iter = iter_batches(split=split, batch_size=16)
        losses = torch.zeros(eval_iters)  # keep on CPU
        ppl_list = torch.zeros(eval_iters)  # keep on CPU
        kl_list = torch.zeros(eval_iters)  # keep on CPU
        for k in range(eval_iters):
            
            X, Y, Z = next(batch_iter)
            
            Z, ppl, kl_avg, nlkhd = posterior_optimizer_test.step(data=[X, Y, Z], ctx=ctx, scaler=scaler, steps=num_steps, lr=lr)
            with ctx:
                logits = model(X, Z, Y)
                loss = raw_model.last_loss
            losses[k] = loss.item()
            ppl_list[k] = ppl.item()
            kl_list[k] = kl_avg.item()
            del X, Y, Z, logits, loss, ppl, kl_avg, nlkhd  # Clear all intermediate variables
            torch.cuda.empty_cache() 
            
        loss_out[split] = losses.mean()
        ppl_out[split] = ppl_list.mean()
        kl_out[split] = kl_list.mean()
    model.train()
    torch.cuda.empty_cache()
    return loss_out, ppl_out, kl_out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    current_time = datetime.today().strftime('%m-%d-%H')
    wandb.init(
        name=f'mlpt_{current_time}_layers{n_layers}_dim{dim}_steps{num_steps}_z{max_z_len}_ws{window_size}_fastlrdecay_init{initial_fast_lr}final{final_fast_lr}',
        entity='ucla-dk-yh', 
        project=wandb_proj_name,
        group=f'steps{num_steps}_{inference_method}_cs',
        config = config
        )
    

# training loop
train_batch_iter = iter_batches(split="train")
X, Y, Z = next(train_batch_iter)
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process

running_mfu = -1.0

# current_shard = shard_id[0]
z_buffer = []
ix_buffer = []
    
# Define the linear decay function
def fast_lr_linear_decay(epoch):
    return initial_fast_lr + epoch / (lr_decay_iters - 1) * (final_fast_lr - initial_fast_lr)

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate

    # calculate lr
    current_lr = fast_lr_linear_decay(iter_num)

    for param_group in optimizer.param_groups:
        if joint_training=='both':
            lr_scale = param_group['lr_scale']
            param_group['lr'] = lr * lr_scale
        else:
            param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses, ppl_out, kl_out = estimate_loss(current_lr)
        print(f"step {iter_num}: val loss {losses['val']:.4f}, val PPL {ppl_out['val']:.4f}, val KL {kl_out['val']:.4f}")

        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "tokens": iter_num * tokens_per_iter,
                    "loss/loss_val": losses["val"],
                    "loss/ppl_val": ppl_out["val"],
                    "loss/kl_val": kl_out["val"],                        
                    "lr": lr,
                    "fast_lr": current_lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                }, step = iter_num
            )
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                    'rng_state': torch.random.get_rng_state()
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt_"+ str(iter_num) +".pt"))

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
        
        Z, ppl, kl, _ = posterior_optimizer.step(data=[X, Y, Z], ctx=ctx, scaler=scaler, steps=num_steps, lr=current_lr)
        with ctx:
            logits = model(X, Z, Y)
            loss = raw_model.last_loss
            loss = loss / gradient_accumulation_steps
            if not cold_start:
                z_buffer.append(Z)
                ix_buffer.append(ix)
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y, Z = next(train_batch_iter)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"{iter_num} | loss {lossf:.4f} | ppl {ppl:.4f} | kl {kl:.4f} | lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%"
        )
        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "tokens": iter_num * tokens_per_iter,
                    "loss/loss_train": lossf,
                    "loss/ppl_train": ppl,
                    "loss/kl_train": kl,
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                }, step = iter_num
            )
        
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

"""
Configuration file for Latent Thought LM training.
Contains all hyperparameters and settings.
"""
import os
from datetime import datetime

# -----------------------------------------------------------------------------
# Output and checkpointing settings
# -----------------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
out_dir = f"output/{timestamp}"
eval_interval = 1000  # Evaluate model every N iterations
log_interval = 1  # Log training progress every N iterations
eval_iters = 100  # Number of batches to use for evaluation
always_save_checkpoint = True  # If True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' (new model) or 'resume' (load from checkpoint)
ckpt_path = ''  # Path to checkpoint file when resuming training

# -----------------------------------------------------------------------------
# Dataset configuration
# -----------------------------------------------------------------------------
DATA_CACHE_DIR = "data_owt"  # Directory for caching dataset
batch_size = 64  # Micro-batch size (before gradient accumulation)
max_seq_len = 1024  # Maximum sequence length
vocab_size = 50258  # Vocabulary size 

# -----------------------------------------------------------------------------
# Model architecture
# -----------------------------------------------------------------------------
dim = 768  # Hidden dimension size
n_layers = 12  # Number of transformer layers
n_heads = 12  # Number of attention heads
n_kv_heads = 12  # Number of key/value heads
multiple_of = 32  # Hidden dimension is rounded to a multiple of this value
dropout = 0.0  # Dropout probability
window_size = 256  # Context window size

# RWKV configuration
use_rwkv = True  # Whether to use RWKV instead of transformer
use_rwkv8_ffn = True  # Whether to use RWKV-8 feed-forward network
head_size = 64  # RWKV head size
rwkv_mode = "rwkv8"  # RWKV mode: "rwkv7" or "rwkv8"

# -----------------------------------------------------------------------------
# Latent variable configuration
# -----------------------------------------------------------------------------
num_steps = 16  # Number of steps for posterior inference
inference_method = 'adamVI'  # Method used for posterior inference
initial_fast_lr = 0.3  # Initial learning rate for latent optimization
final_fast_lr = 0.34  # Final learning rate for latent optimization
fast_lr = 0.34  # Current learning rate for latent optimization
n_prior_layers = 0  # Number of layers in the prior network
n_cls_tokens = 0  # Number of classification tokens
max_z_len = n_layers * 8  # Maximum length of latent sequence
z_dim = dim  # Dimension of latent variables

# DiT prior configuration
use_dit_prior = True  # Whether to use DiT prior instead of Gaussian prior
dit_layers = 6  # Number of layers in DiT prior (reduced for efficiency)
dit_heads = 8  # Number of heads in DiT prior
dit_dim = 512  # Hidden dimension in DiT prior (reduced for efficiency)
dit_multiple_of = 32  # Multiple of for DiT hidden dimension
dit_num_timesteps = 1000  # Number of diffusion timesteps
dit_beta_schedule = "linear"  # Beta schedule for diffusion
dit_beta_start = 0.0001  # Starting beta value
dit_beta_end = 0.02  # Ending beta value

# -----------------------------------------------------------------------------
# SimCSE configuration
# -----------------------------------------------------------------------------
use_simcse = True  # Whether to use SimCSE contrastive learning
simcse_pooler_type = "cls"  # Pooling strategy: cls, cls_before_pooler, avg, avg_top2, avg_first_last
simcse_temperature = 0.05  # Temperature for softmax in contrastive loss
simcse_projection_dim = 256  # Dimension of projection head
simcse_use_projection_head = True  # Whether to use projection head
simcse_weight = 0.1  # Weight for SimCSE loss in total loss

# -----------------------------------------------------------------------------
# Optimizer settings
# -----------------------------------------------------------------------------
gradient_accumulation_steps = 8  # Number of steps to accumulate gradients (simulates larger batch)
learning_rate = 4e-4  # Maximum learning rate for model training
max_iters = 60000  # Total number of training iterations (~30B tokens)
weight_decay = 1e-1  # L2 regularization
beta1 = 0.9  # AdamW beta1 parameter
beta2 = 0.95  # AdamW beta2 parameter
grad_clip = 1.0  # Gradient clipping threshold (disable if 0.0)

# Learning rate schedule settings
decay_lr = True  # Whether to use learning rate decay
warmup_iters = 1000  # Number of warmup iterations
lr_decay_iters = max_iters  # Number of iterations over which to decay learning rate
min_lr = 4e-5  # Minimum learning rate (typically learning_rate/10)

# -----------------------------------------------------------------------------
# System and hardware settings
# -----------------------------------------------------------------------------
device = "cuda"  # Device to use: 'cpu', 'cuda', 'cuda:0', etc.
dtype = "bfloat16"  # Data type: float32, bfloat16, or float16
compile = False  # Whether to use PyTorch 2.0 compilation for speed

# Create a dictionary of all configuration parameters
def get_config_dict():
    """Return a dictionary containing all configuration parameters."""
    globals_dict = globals()
    return {k: v for k, v in globals_dict.items() 
            if not k.startswith("_") and 
               not callable(v) and 
               k not in ('os', 'datetime', 'timestamp')}
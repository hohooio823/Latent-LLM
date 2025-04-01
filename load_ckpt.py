"""
Load a trained Latent Thought Model from a checkpoint file.

This script sets up the environment and loads a pretrained LatentThoughtModel
from a saved checkpoint.
"""

import os
import sys
from contextlib import nullcontext
import torch
import numpy as np
from transformers import GPT2TokenizerFast
from model import LatentThoughtModel, LTMConfig

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Checkpoint path
checkpoint = f'{checkpoint}'

# Load tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.add_special_tokens({'bos_token': '<|beginoftext|>'})

# Device and precision settings
device_id = "cuda:0"
device = torch.device(device_id)
device = device_id  # Simplified reference
dtype = "float32"   # Precision: float32, bfloat16, or float16

# Random seed for reproducibility
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on matrix multiply
torch.backends.cudnn.allow_tf32 = True        # Allow TF32 on cuDNN
np.random.seed(seed)

# Set up autocast context manager for mixed precision
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {
    'float32': torch.float32, 
    'bfloat16': torch.bfloat16, 
    'float16': torch.float16
}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------------

# Load model checkpoint
checkpoint_dict = torch.load(checkpoint, map_location=device)
gptconf = LTMConfig(**checkpoint_dict['model_args'])
cfg = checkpoint_dict["config"]  # Additional configuration parameters

# Initialize model
model = LatentThoughtModel(gptconf)

# Clean up state dict (remove DDP prefixes if present)
state_dict = checkpoint_dict['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

# Load state dict and prepare model
model.load_state_dict(state_dict, strict=False)
model.eval()  # Set to evaluation mode
model.to(device)

print(f"Model loaded successfully from {ckpt_name}")
print(f"Model configuration: {gptconf.n_layers} layers, {gptconf.dim} hidden dim, {gptconf.max_z_len} max Z length")
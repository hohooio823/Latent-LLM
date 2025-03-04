"""
Sample from the trained model with PyTorch
"""
import os
import sys
import pickle
from contextlib import nullcontext
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname("/home/deqian/random_effect_LLM/sampling/zero-shot-reasoning-likelihood/data.ipynb"))))
import time 
from optimizer import PosteriorOptimizer
import numpy as np
import argparse
import yaml
from tokenizer import Tokenizer
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPTNeoXTokenizerFast
import logging

import os
import json
import logging
from datetime import datetime
import random
from zero_shot_utils import *

task_to_test = ['wsc', 'obqa', 'arc_easy'] # ['wsc', 'winogrande', 'siqa', 'piqa', 'obqa', 'hellaswag', 'arc_easy', 'arc_challenge']
process_functions = []
for task in task_to_test:
    if task in task_functions:
        process_functions.append(task_functions[task])
    else:
        raise ValueError(f"Task '{task}' not supported.")

checkpoints_to_check = ['output/owt_liger/owt_liger_mlpt_2024_11_13_08_17_57/ckpt_58000.pt',
                        'output/owt_liger/owt_liger_mlpt_2024_11_19_08_12_13/ckpt_58000.pt']

checkpoint = checkpoints_to_check[0]

checkpoint = f'../../{checkpoint}'
ckpt_name = f"logs/{checkpoint.split('/')[-2]}_{checkpoint.split('/')[-1].split('.')[0]}"

fast_lr = 0.3
posterior_steps = 15
max_z_len = None # None if want to use cfg['max_z_len']

logging.basicConfig(filename=f"{ckpt_name}_z{max_z_len}.log", level=logging.INFO, format="%(message)s")

if 'dclm' in checkpoint : 
    from model_old import ModelArgs, LatentPromptTransformerVIPostTraining, LatentPromptTransformerVI

    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
    special_tokens = {'bos_token': '<|beginoftext|>'}
    tokenizer.add_special_tokens(special_tokens)
    bos_token_id = tokenizer.bos_token_id
    use_liger = True
    use_z_pos_emb = True
elif 'mlpt' in checkpoint:
    from model import ModelArgs, MultiLayerLatentPromptTransformer 
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    use_liger = True
    use_z_pos_emb = True
    tokenizer.add_special_tokens({'bos_token': '<|beginoftext|>'})
    bos_token_id = tokenizer.bos_token_id
else:
    from model import ModelArgs, LatentPromptTransformerVIPostTraining, LatentPromptTransformerVI

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    use_liger = True
    use_z_pos_emb = True

# -----------------------------------------------------------------------------
device_id = "cuda"
device = torch.device(device_id)
device = device_id
dtype = "float32"
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
np.random.seed(seed)
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# init from a model saved in a specific directory
checkpoint_dict = torch.load(checkpoint, map_location=device)
gptconf = ModelArgs(**checkpoint_dict['model_args'])
cfg = checkpoint_dict["config"]
gptconf.use_liger = use_liger
gptconf.use_z_pos_emb = use_z_pos_emb

# model = LatentPromptTransformerVIPostTraining(gpt.conf)
model = LatentPromptTransformerVI(gptconf)
state_dict = checkpoint_dict['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict, strict=False)

model.eval()
model.to(device)
bos_token = tokenizer.bos_token

if max_z_len == None:
    max_z_len=cfg['max_z_len']
posterior_optimizer = PosteriorOptimizer(model = model, 
                                        inference_method='adamVIPPL', 
                                        num_steps=posterior_steps, 
                                        max_z_len=max_z_len, 
                                        z_dim=cfg['z_dim'],
                                        lr = fast_lr)

"""
Sample from the trained model with PyTorch
"""
import os
import sys
from contextlib import nullcontext
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPTNeoXTokenizerFast
import torch.nn.functional as F
import logging
from datetime import datetime
from utils_gen_ppl import calculate_sentence_entropy

batch_size = 64
max_new_tokens = 1024
import json

##load in json file

# retokenize because training used diff tokenizer 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer_gpt = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer_gpt.pad_token = tokenizer_gpt.eos_token
encoded_dict = tokenizer_gpt(
    responses,
    max_length=1024,
    padding='max_length',  # Pad sequences to the max_length
    truncation=True,       # Truncate sequences longer than max_length
    return_tensors='pt'    # Return PyTorch tensors
)
input_ids = encoded_dict['input_ids']  # Shape: (batch_size, 1024)
print(f"input_ids shape: {input_ids.shape}")

perplexity_batch_size = 8
with torch.no_grad():
    eval_model = GPT2LMHeadModel.from_pretrained("gpt2-xl").to(device).eval()
    batches = input_ids.shape[0] // perplexity_batch_size
    total_perplexity = 0
    for i in range(batches):
        s = input_ids[i * perplexity_batch_size:(i + 1) * perplexity_batch_size]
        s = torch.tensor(s, device=device)
        loss, logits = eval_model(s, labels=s)[:2]
        logits = logits.transpose(-1, -2)
        perplexity = F.cross_entropy(logits[..., :-1], s[..., 1:], reduction="none").mean(dim=-1).exp().mean()
        total_perplexity += perplexity
    total_perplexity /= batches

sample_entropies = calculate_sentence_entropy(input_ids)
entropy = np.mean(sample_entropies)

# print("ckpt: ", checkpoint)
print("gen perplexity: ", total_perplexity.item(), "entropy: ", entropy)


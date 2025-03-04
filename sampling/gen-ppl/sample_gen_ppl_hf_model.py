"""
Sample from the trained model with PyTorch
"""
import os
import sys
import math
import torch
import numpy as np
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from utils_gen_ppl import calculate_sentence_entropy
import time

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


model_name = "gpt2-medium"
print(f"model name: {model_name}")
device = torch.device("cuda")
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Generate samples
print("Generating samples...")

num_samples=64
max_length=1024

# ## Analytical
samples = []
start_time = time.time()
for i in range(num_samples):
    input_ids = torch.tensor([[tokenizer.bos_token_id]]).to(device)
    for _ in range(max_length - 1):
        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    print("sample ", i, input_ids[0])
    samples.append(input_ids[0])

end_time = time.time()

# Calculate the sampling speed
total_time = end_time - start_time
sampling_speed = total_time / num_samples
print(f"Sampling speed: {sampling_speed} seconds per sample")

samples = torch.stack(samples)


print("Evaluate perplexity...")
perplexity_batch_size = 8
with torch.no_grad():
    eval_model = GPT2LMHeadModel.from_pretrained("gpt2-xl").to(device).eval()
    batches = samples.shape[0] // perplexity_batch_size
    total_perplexity = 0
    for i in range(batches):
        s = samples[i * perplexity_batch_size:(i + 1) * perplexity_batch_size]
        loss, logits = eval_model(s, labels=s)[:2]
        logits = logits.transpose(-1, -2)
        perplexity = F.cross_entropy(logits[..., :-1], s[..., 1:], reduction="none").mean(dim=-1).exp().mean()
        total_perplexity += perplexity
    total_perplexity /= batches

print("gen perplexity: ", total_perplexity.item())

sample_entropies = calculate_sentence_entropy(samples)
entropy = np.mean(sample_entropies)

print("ckpt: ", model_name)
print("entropy: ", entropy)
print(f"number of samples {num_samples}, batch size: {perplexity_batch_size}")


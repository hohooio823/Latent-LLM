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
from model_gpt import Transformer, ModelArgs
import torch.nn.functional as F
import logging
from datetime import datetime
from utils_gen_ppl import calculate_sentence_entropy


for checkpoint in checkpoints_to_check:
    print("="*30)
    checkpoint = f'../../{checkpoint}'

    temperature = 1 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = None # retain only the top_k most likely tokens, clamp others to have 0 probability
    folder_name = f"0123-AR-temp{temperature}-topk{top_k}"

    os.makedirs(folder_name, exist_ok=True)

    ckpt_name = f"{folder_name}/{checkpoint.split('/')[-2]}_{checkpoint.split('/')[-1].split('.')[0]}"

    logging.basicConfig(filename=f"{ckpt_name}_{datetime.now().strftime('%m%d%H%M')}.log", level=logging.INFO, format="%(message)s", filemode='w')

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'bos_token': '<|beginoftext|>'})
    bos_token_id = tokenizer.bos_token_id

    # -----------------------------------------------------------------------------
    device_id = "cuda:0"
    device = torch.device(device_id)
    device = device_id
    dtype = "float32"
    seed = 1
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

    model = Transformer(gptconf)

    state_dict = checkpoint_dict['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    model.to(device)
    bos_token = tokenizer.bos_token

    ##################################################################

    batch_size = 64
    max_new_tokens = 1024
    start_ids = torch.full((batch_size, 1), tokenizer.bos_token_id, dtype=torch.long, device=device)
    with ctx:
        samples = model.generate(start_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    print(samples[:2])

    # retokenize because training used diff tokenizer 
    tokenizer_gpt = GPT2TokenizerFast.from_pretrained('gpt2')
    responses = [tokenizer.decode(sample, skip_special_tokens=True) for sample in samples]
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
        eval_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).eval()
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

    sample_entropies = calculate_sentence_entropy(samples)
    entropy = np.mean(sample_entropies)

    logging.info(f"ckpt: {checkpoint}")
    logging.info(f"number of samples {batch_size}, batch size: {perplexity_batch_size}")
    logging.info(f"topk: {top_k}, temperature: {temperature}, gen perplexity: {total_perplexity.item()}, entropy: {entropy}")

    print("ckpt: ", checkpoint)
    print(f"topk: {top_k}, temperature: {temperature}, number of samples {batch_size}, batch size: {perplexity_batch_size}")
    print("gen perplexity: ", total_perplexity.item(), "entropy: ", entropy)
    logging.info("="*20)
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
import logging
from datetime import datetime
from optimizer import PosteriorOptimizer
from datasets import load_dataset
from utils_cnn import *

    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_15_09_04_44/ckpt_10000.pt',
    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_15_09_04_44/ckpt_20000.pt',
    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_15_09_04_44/ckpt_30000.pt',

checkpoints_to_check = [
    'output/owt_mlpt_decIE/owt_mlpt_2025_01_12_02_11_49/ckpt_49000.pt',
    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_02_21_41_22/ckpt_59000.pt',
    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_15_09_04_44/ckpt_59000.pt',
    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_16_01_48_06/ckpt_28000.pt',
    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_15_09_04_44/ckpt_30000.pt',
]


# conda activate language
# cd sampling/zero-shot-reasoning-likelihood/
# conda activate language
# CUDA_VISIBLE_DEVICES=0 python zero_shot_sampling_qa.py

dataset = load_dataset("cnn_dailymail", "1.0.0", split="test")  

for checkpoint in checkpoints_to_check:
    print("="*30)
    checkpoint = f'../../{checkpoint}'
    posterior_steps = 16

    const_var = False
    max_new_tokens=100
    temperature=1
    top_k=1

    question_prompt = " TL;DR: "

    folder_name = f"0120-mlpt_{posterior_steps}steps_novar{const_var}_maxnew{max_new_tokens}_temp{temperature}_topk{top_k}"

    os.makedirs(folder_name, exist_ok=True)

    ckpt_name = f"{folder_name}/{checkpoint.split('/')[-2]}_{checkpoint.split('/')[-1].split('.')[0]}"
    max_z_len = None # None if want to use cfg['max_z_len']

    logging.basicConfig(filename=f"{ckpt_name}_steps{posterior_steps}z{max_z_len}_{datetime.now().strftime('%m%d%H%M%S')}.log", level=logging.INFO, format="%(message)s")

    from model import ModelArgs, LatentPromptTransformerVI, MultiLayerLatentPromptTransformer
    use_liger = True
    use_z_pos_emb = True    
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    if 'mlpt' in checkpoint:
        tokenizer.add_special_tokens({'bos_token': '<|beginoftext|>'})

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
    gptconf.use_liger = use_liger
    gptconf.use_z_pos_emb = use_z_pos_emb

    if 'mlpt' in checkpoint:
        model = MultiLayerLatentPromptTransformer(gptconf)
    else:
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
    eos_token = tokenizer.eos_token

    if max_z_len == None:
        max_z_len=cfg['max_z_len']
        
    fast_lr = cfg['fast_lr']

    posterior_optimizer = PosteriorOptimizer(model = model, 
                                            inference_method='adamVIPPL', 
                                            num_steps=posterior_steps, 
                                            max_z_len=max_z_len, 
                                            z_dim=cfg['z_dim'],
                                            lr = fast_lr,
                                            const_var = const_var)

    ##################################################################
    message_ckpt = f"Using checkpoint {checkpoint}, turn_off_var = {const_var}, max_z_len = {max_z_len}, fast_lr = {fast_lr}, posterior_steps = {posterior_steps}, use_liger={use_liger}, use_z_pos_emb={use_z_pos_emb}"
    logging.info("="*30)
    logging.info(message_ckpt)
    print("="*30)
    print(message_ckpt)

    generated_summaries = []
    reference_summaries = []

    for index, item in enumerate(dataset):  
        article_text = item['article']
        reference_summary = item['highlights']

        log_info = True if index % 20 == 0 else False
        
        question_specific_seed = np.random.randint(100000)
        torch.manual_seed(question_specific_seed)
        torch.cuda.manual_seed(question_specific_seed)

        question_prompt_tokens = tokenizer.encode(question_prompt, add_special_tokens=False)
        question_prompt_len = len(question_prompt_tokens)

        article_text = f"{bos_token}{article_text}"
        article_allowed_len = gptconf.max_seq_len - max_new_tokens - question_prompt_len
        article_tokens = tokenizer.encode(article_text, max_length=article_allowed_len, add_special_tokens=False, truncation=True)

        input_tokens = article_tokens
        question_input = (torch.tensor(input_tokens, dtype=torch.long, device=device)[None, ...])
        print("question_input.shape to infer z", question_input.shape)
        
        z1 = torch.randn(1, max_z_len,  cfg['z_dim']).to(device)
        z = z1 * 0.01
        with ctx:
            z, ppl, kl_loss, nlkhd = posterior_optimizer.step(data=[question_input[:, :-1], question_input[:, 1:], z], ctx=ctx)
            generation_tokens = article_tokens + question_prompt_tokens
            input_tensor = (torch.tensor(generation_tokens, dtype=torch.long, device=device)[None, ...])
            print("input_tensor.shape to generate", input_tensor.shape)

            y = model.generate(input_tensor, z=z, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
            generated_answer = tokenizer.decode(y[0][input_tensor.shape[1]:].tolist())
            
        generated_summary = split_sentences(generated_answer.strip())
        generated_summaries.append(generated_summary)
        reference_summaries.append(reference_summary)  
        rouge_scores = calculate_rouge(generated_summaries, reference_summaries)

        avg_rouge = np.mean(list(rouge_scores.values()))
        rouge_scores_formatted = {key: round(value, 3) for key, value in rouge_scores.items()}
        info = f"idx {index}, Current avg: {avg_rouge:.3f}, {rouge_scores_formatted}"    
        print(info)
        
        if log_info:
            inference_text = tokenizer.decode(input_tokens)
            generation_text = tokenizer.decode(generation_tokens)
            logging.info(info)
            logging.info(f"inference_text: {inference_text}")
            logging.info(f"generation_text: {generation_text}")
            logging.info(f"generated_summary: {generated_summary}")
            logging.info(f"reference_summary: {reference_summary}")
                        
    message = f"Final avg: {avg_rouge:.3f}, {rouge_scores}"
    logging.info(message)

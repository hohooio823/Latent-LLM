"""
Sample from the trained model with PyTorch
"""
import os
import sys
import pickle
from contextlib import nullcontext
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
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

task_to_test = ['arc_easy'] # ['wsc', 'winogrande', 'siqa', 'piqa', 'obqa', 'hellaswag', 'arc_easy', 'arc_challenge']
process_functions = []
for task in task_to_test:
    if task in task_functions:
        process_functions.append(task_functions[task])
    else:
        raise ValueError(f"Task '{task}' not supported.")

# checkpoints_to_check = ['output/owt_liger/owt_liger_2024_11_09_18_23_30/ckpt_55000.pt',
#                         'output/owt_liger/owt_liger_2024_11_06_00_30_53/ckpt_25000.pt',
#                         'output/owt_liger/owt_liger_2024_11_08_00_54_37/ckpt_26000.pt',
#                         'output/owt_liger/owt_liger_2024_11_09_07_33_53/ckpt_43000.pt']

# checkpoints_to_check = ['output/owt_liger/owt_liger_2024_11_08_00_54_37/ckpt_26000.pt',
#                         'output/owt_liger/owt_liger_2024_11_09_07_33_53/ckpt_43000.pt']

# checkpoints_to_check = ['output/owt_liger/owt_liger_mlpt_2024_11_10_08_09_46/ckpt_32000.pt',
#                         'output/owt_liger/owt_liger_mlpt_2024_11_12_01_47_28/ckpt_45000.pt']
# checkpoints_to_check = ['output/owt_liger/owt_liger_mlpt_2024_11_13_08_17_57/ckpt_58000.pt',
#                         'output/owt_liger/owt_liger_mlpt_2024_11_19_08_12_13/ckpt_58000.pt']

# checkpoints_to_check = ['output/owt_liger/owt_liger_mlpt_2024_11_12_01_47_28/ckpt_45000.pt',
#                         'output/owt_liger/owt_liger_mlpt_2024_11_19_08_12_13/ckpt_58000.pt']

# checkpoints_to_check = ['output/owt_liger/owt_liger_mlpt_2024_11_10_08_09_46/ckpt_32000.pt']
# checkpoints_to_check = ['output/owt_liger/owt_liger_mlpt_2024_11_13_08_17_57/ckpt_58000.pt']
# checkpoints_to_check = ['output/owt_liger/owt_liger_mlpt_2024_11_12_01_47_28/ckpt_45000.pt']
checkpoints_to_check = ['output/owt_liger/owt_liger_mlpt_2024_11_19_08_12_13/ckpt_58000.pt']                        

# checkpoints_to_check = ['output/owt_liger/owt_liger_2024_11_09_18_23_30/ckpt_55000.pt']
# checkpoints_to_check = ['output/owt_liger/owt_liger_2024_11_08_00_54_37/ckpt_26000.pt']

# checkpoints_to_check = ['output/owt_liger/owt_liger_2024_11_06_00_30_53/ckpt_25000.pt']
# checkpoints_to_check = ['output/owt_liger/owt_liger_2024_11_09_07_33_53/ckpt_43000.pt']
# checkpoints_to_check = ['output/owt_liger/owt_mlptencdec_2024_12_03_08_59_26/ckpt_46000.pt']


for checkpoint in checkpoints_to_check:
    print("="*30)
    checkpoint = f'../../{checkpoint}'
    ckpt_name = f"logs/batch_{checkpoint.split('/')[-2]}_{checkpoint.split('/')[-1].split('.')[0]}"
    
    posterior_steps = 15
    max_z_len = None # None if want to use cfg['max_z_len']
    batch_size = 8
    
    logging.basicConfig(filename=f"{ckpt_name}_z{max_z_len}_{datetime.now().strftime('%m%d%H%M')}.log", level=logging.INFO, format="%(message)s")

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

    if max_z_len == None:
        max_z_len=cfg['max_z_len']
        
    fast_lr = cfg['fast_lr']

    posterior_optimizer = PosteriorOptimizer(model = model, 
                                            inference_method='adamVIPPL', 
                                            num_steps=posterior_steps, 
                                            max_z_len=max_z_len, 
                                            z_dim=cfg['z_dim'],
                                            lr = fast_lr,
                                            reduce=False,
                                            padding = True)

    ##################################################################
    message_ckpt = f"Using checkpoint {checkpoint}, max_z_len = {max_z_len}, fast_lr = {fast_lr}, posterior_steps = {posterior_steps}, use_liger={use_liger}, use_z_pos_emb={use_z_pos_emb}"
    logging.info("="*30)
    logging.info(message_ckpt)
    logging.info("using adamw")
    
    all_messages = []
    for task_id, process_func in enumerate(process_functions):
        print("="*30)
        if task_to_test[task_id] != 'arc_easy' and task_to_test[task_id] != 'arc_challenge':
            # Note that arc_easy and arc_challenge have different number of options (3, 4, 5 in each question)
            dataset_lists = [process_func()]
        else: 
            dataset_lists = process_func()
        
        all_correct = 0
        all_tested = 0
        
        for subdataset_id, all_questions_list in enumerate(dataset_lists):
            correct = 0
            max_length = 1024
            logging.info(f"{task_to_test[task_id]}: max dataset token len = {max_length}")
            
            # Prepare input batch
            all_labels = [item['label'] for item in all_questions_list]        
            padded_inputs = []
            padded_targets = []

            for item in all_questions_list:
                sentences = [f"{bos_token}{s}".strip() for s in item['sentences']]
                tokenized_batch = [tokenizer.encode(sent, add_special_tokens=False)[:gptconf.max_seq_len] for sent in sentences]

                input_padded_batch = [tok + [50256] * (max_length - len(tok)) for tok in tokenized_batch]  # Pad inputs with 50256 (valid token ID)
                target_padded_batch = [tok + [-1] * (max_length - len(tok)) for tok in tokenized_batch]  # Pad targets with -1 since adamVIPPL uses torch.cross_entropy loss. The ignore_index is -1.

                padded_inputs.append(torch.stack([torch.tensor(tok, dtype=torch.long, device=device) for tok in input_padded_batch]))
                padded_targets.append(torch.stack([torch.tensor(tok, dtype=torch.long, device=device) for tok in target_padded_batch]))

            # Group multiple batches together
            grouped_batches = [padded_inputs[i:i + batch_size] for i in range(0, len(padded_inputs), batch_size)]
            grouped_targets = [padded_targets[i:i + batch_size] for i in range(0, len(padded_targets), batch_size)]
            grouped_labels = [all_labels[i:i + batch_size] for i in range(0, len(all_labels), batch_size)]
            
            logging.info(f"len(grouped_batches): {len(grouped_batches)}, len(grouped_batches[0]): {len(grouped_batches[0])} , grouped_batches[0][0].shape: {grouped_batches[0][0].shape}")
            group_seed = [np.random.randint(100000) for i in range(len(grouped_batches))]
            print(group_seed)
            num_options_per_question = len(all_questions_list[0]['sentences'])
            logging.info(f"number of options per question: {num_options_per_question}")
            
            total_tested = 0
            with ctx:            
                for group_index, (batch_group, target_group, label_group, batch_specific_seed) in enumerate(zip(grouped_batches, grouped_targets, grouped_labels, group_seed)):
                    x_input_batch = torch.cat([x[:, :-1] for x in batch_group], dim=0)
                    target_batch = torch.cat([t[:, 1:] for t in target_group], dim=0)
                    total_tested += len(label_group)
                    # Generate individual random values for batch_size, then repeat and reshape to match x_input_batch.shape[0]
                    z1_individual = torch.randn(len(target_group), max_z_len, cfg['z_dim']).to(device)
                    z1 = torch.repeat_interleave(z1_individual, repeats=num_options_per_question, dim=0)
                    z = z1 * 0.01
                    
                    torch.manual_seed(batch_specific_seed)
                    torch.cuda.manual_seed(batch_specific_seed)
                    loss, ppl, kl_loss, nlkhd = posterior_optimizer.step(data=[x_input_batch, target_batch, z], ctx=ctx, seed=batch_specific_seed)

                    loss_output = (nlkhd + kl_loss).cpu().detach().numpy()
                    nlkhd = nlkhd.cpu().detach().numpy()
                    kl_loss = kl_loss.cpu().detach().numpy()
                    generated_answers = [label[np.argmin(loss_output[i:i + len(batch_group[i])])] for i, label in enumerate(label_group)]

                    for i, label in enumerate(label_group):                        
                        loss_group = loss_output[i:i + len(batch_group[i])]
                        kl_loss_group = kl_loss[i:i + len(batch_group[i])]
                        nlkhd_group = nlkhd[i:i + len(batch_group[i])]

                        generated_answer = label[np.argmin(loss_group)] 
                        is_correct = generated_answer == all_questions_list[group_index*batch_size +i]['correct_index']
                        if is_correct:
                            correct += 1
                        msg = f"group {group_index}: loss_output: {np.round(loss_group, 2)}, kl_loss: {np.round(kl_loss_group, 2)}, nlkhd: {np.round(nlkhd_group, 2)}"
                        logging.info(msg)
                        print(msg)               

                        msg = f"generatedID: {generated_answer}, correctID: {all_questions_list[group_index*batch_size +i]['correct_index']}, current rate {correct/total_tested:.2f} ({correct}/{total_tested})"        
                        logging.info(msg)
                        print(msg)
                    # break

            all_tested += len(all_questions_list)
            all_correct += correct
            message = f"Evaluation for {task_to_test[task_id]} subset {subdataset_id+1}/{len(dataset_lists)}, correct rate {correct / len(all_questions_list):4f}, {correct}/{len(all_questions_list)}, total correct rate {all_correct / all_tested:.4f}, {all_correct}/{all_tested}"
            logging.info(message)
            all_messages.append(message)
            print(message)

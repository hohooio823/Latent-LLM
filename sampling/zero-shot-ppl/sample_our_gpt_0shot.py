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
import logging
from datetime import datetime
from utils_ppl_0shot_update import get_dataset
from torch.utils.data import DataLoader
import math
# checkpoints_to_check = ['output/gpt/test2024_11_14_05_04_29/ckpt_200000.pt',
#                         'output/gpt/test2024_11_15_01_54_10/ckpt_174000.pt',
#                         'output/gpt/test2024_11_16_05_40_48/ckpt_192000.pt',
#                         'output/gpt/test2024_11_18_06_21_25/ckpt_244000.pt']

# checkpoints_to_check = ['output/gpt/test2024_11_14_05_04_29/ckpt_54000.pt']
# checkpoints_to_check = ['output/gpt/test2024_11_14_05_04_29/ckpt_42000.pt']
# checkpoints_to_check = ['output/gpt/test2024_11_14_05_04_29/ckpt_26000.pt']
checkpoints_to_check = [
    'output/gpt/test2024_11_14_05_04_29/ckpt_200000.pt',
    # 'output/gpt/test2024_11_15_01_54_10/ckpt_174000.pt',
    # 'output/gpt/test2024_11_18_06_21_25/ckpt_244000.pt'
    ]
    
datasets_to_test = ['ptb', 'wikitext2', 'lambada', 'lm1b', 'ag_news', 'scientific_papers_pubmed', 'scientific_papers_arxiv']
# datasets_to_test = ['ptb', 'wikitext2', 'lambada']
# datasets_to_test = ['lm1b', 'ag_news']
# datasets_to_test = ['scientific_papers_arxiv']
# datasets_to_test = ['scientific_papers_pubmed']

for checkpoint in checkpoints_to_check:
    print("="*30)
    checkpoint = f'../../{checkpoint}'
    folder_name = f"0125-out_gpt"

    os.makedirs(folder_name, exist_ok=True)

    ckpt_name = f"{folder_name}/{checkpoint.split('/')[-2]}_{checkpoint.split('/')[-1].split('.')[0]}"
    logging.basicConfig(filename=f"{ckpt_name}_{datetime.now().strftime('%m%d%H%M%S')}.log", level=logging.INFO, format="%(message)s")

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'bos_token': '<|beginoftext|>'})
    bos_token_id = tokenizer.bos_token_id

    # -----------------------------------------------------------------------------
    device_id = "cuda:0"
    device = torch.device(device_id)
    device = device_id
    dtype = "float32"
    seed = 1024
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

    for dataset_name in datasets_to_test:
        mode = 'test' if dataset_name in ['lm1b', 'ag_news'] else 'validation'
        logging.info("="*30)
        logging.info(f"Evaluating on {dataset_name}, Checkpoint {checkpoint}")    

        device = torch.device(device_id)
        dataset = get_dataset(dataset_name, mode, cache_dir='data', block_size=1024)
        dataloader = DataLoader(dataset, batch_size= 1, sampler=None, shuffle=False, num_workers=4, pin_memory=True)
        print(dataloader.dataset)

        ppl_list = []
        for batch_id, batch in enumerate(dataloader):
            x = torch.tensor(batch['input_ids']).to(device)

            batch_size = x.shape[0]
            with ctx:
                logits = model(tokens = x[:, :-1], targets = x[:, 1:])
                loss = model.last_loss

                ppl = math.exp(loss.item())
                ppl_list.append(ppl)
                log_message = f"Data {batch_id}, ppl: {ppl:.3f}"
                print(log_message)
                # logging.info(log_message)

        print(f"Eval dataset: {dataset_name}, Checkpoint: {checkpoint}")
        print(f"Final ppl: {np.mean(ppl_list):.3f}")
        logging.info(f"Dataset {dataset_name}-{mode}, Final avg ppl: {np.mean(ppl_list):.3f}")

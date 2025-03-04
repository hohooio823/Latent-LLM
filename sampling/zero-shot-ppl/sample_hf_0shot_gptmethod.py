# import torch
# import torch.nn.functional as F
# import numpy as np
# from transformers import GPT2TokenizerFast, GPT2LMHeadModel
# from utils_ppl_0shot_update import get_dataset
# from torch.utils.data import DataLoader
# import math
# seed = 12
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)

# device = torch.device('cuda:0')
# datasets_to_test = ['ptb']# , 'wikitext2', 'lm1b', 'lambada','ag_news', 'scientific_papers_pubmed', 'scientific_papers_arxiv']

# # 1. Load the GPT model
# eval_model_name = "gpt2-medium"
# eval_model = GPT2LMHeadModel.from_pretrained(eval_model_name).to(device).eval()

# for dataset_name in datasets_to_test:
#     mode = 'test' if dataset_name in ['lm1b', 'ag_news'] else 'validation'
#     dataset = get_dataset(dataset_name, mode, cache_dir='data', block_size=1024)
#     dataloader = DataLoader(dataset, batch_size=1, sampler=None, shuffle=False, num_workers=4, pin_memory=True)
    
#     ppl_list = []

#     for batch_id, batch in enumerate(dataloader):
#         question_specific_seed = np.random.randint(100000)
#         torch.manual_seed(question_specific_seed)
#         torch.cuda.manual_seed(question_specific_seed)
#         x = torch.tensor(batch['input_ids'], dtype=torch.long, device=device)

#         with torch.no_grad():
#             outputs = eval_model(x, labels=x)  # Hugging Face auto-shifts
#             loss = outputs.loss  # average cross-entropy loss over all tokens
#             ppl = loss.exp()
#         ppl_list.append(ppl.item())
#         avg_ppl = np.mean(ppl_list)

#     print(f"Eval dataset: {dataset_name}, Final ppl: {np.mean(ppl_list):.3f}")

import torch
import torch.nn.functional as F
import numpy as np
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from utils_ppl_0shot_update import get_dataset
from torch.utils.data import DataLoader
import math
seed = 12
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

device = torch.device('cuda:0')
datasets_to_test = ['ptb']# , 'wikitext2', 'lm1b', 'lambada','ag_news', 'scientific_papers_pubmed', 'scientific_papers_arxiv']

# 1. Load the GPT model
eval_model_name = "openai-community/gpt2-large"
eval_model = GPT2LMHeadModel.from_pretrained(eval_model_name).to(device).eval()

for dataset_name in datasets_to_test:
    mode = 'test' if dataset_name in ['lm1b', 'ag_news'] else 'validation'
    dataset = get_dataset(dataset_name, mode, cache_dir='data', block_size=1024)
    dataloader = DataLoader(dataset, batch_size=1, sampler=None, shuffle=False, num_workers=4, pin_memory=True)
    
    all_tokens = 0
    all_loss = 0
    for batch_id, batch in enumerate(dataloader):
        question_specific_seed = np.random.randint(100000)
        torch.manual_seed(question_specific_seed)
        torch.cuda.manual_seed(question_specific_seed)
        x = torch.tensor(batch['input_ids'], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = eval_model(x, labels=x)  # Hugging Face auto-shifts
            loss = outputs.loss  # average cross-entropy loss over all tokens
            # ppl = loss.exp()
        
        all_tokens += x.size(1)
        all_loss += loss.item() * x.size(1)
        # avg_loss_list.append(loss.item())
        
        # ppl_list.append(ppl.item())
        # avg_ppl = np.mean(ppl_list)
    all_avg_loss = all_loss / all_tokens
    ppl = math.exp(all_avg_loss)

    print(f"Eval dataset: {dataset_name}, Final ppl: {ppl:.3f}")


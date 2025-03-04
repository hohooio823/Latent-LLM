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
from utils_ppl_0shot_update import get_dataset
from torch.utils.data import DataLoader
from optimizer import PosteriorOptimizer


checkpoints_to_check = [
    # 'output/owt_liger_mlpt/owt_mlpt_2025_01_08_18_53_24/ckpt_59000.pt', # 3layer 24z 64steps

    #### 'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_02_21_33_35/ckpt_59000.pt', # 12layer 24z 16steps
    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_02_21_33_35/ckpt_20000.pt', # 12layer 24z 16steps

    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_02_21_33_35/ckpt_30000.pt', # 12layer 24z 16steps

    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_02_21_33_35/ckpt_40000.pt', # 12layer 24z 16steps
    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_02_21_33_35/ckpt_50000.pt', # 12layer 24z 16steps
    # 'output/owt_mlpt_decIE/owt_mlpt_2025_01_18_21_11_37/ckpt_59000.pt', # 12layer 96z 16steps
    # 'output/owt_mlpt_decIE/owt_mlpt_2025_01_18_21_11_37/ckpt_30000.pt', # 12layer 96z 16steps
    # 'output/owt_mlpt_decIE/owt_mlpt_2025_01_18_21_11_37/ckpt_40000.pt', # 12layer 96z 16steps
    # 'output/owt_mlpt_decIE/owt_mlpt_2025_01_18_21_11_37/ckpt_50000.pt', # 12layer 96z 16steps
    # 'output/owt_mlpt_decIE/owt_mlpt_2025_01_16_07_13_04/ckpt_54000.pt', # 12layer 96z 16steps
    # 'output/owt_mlpt_decIE/owt_mlpt_2025_01_16_07_13_04/ckpt_30000.pt', # 12layer 96z 16steps
    # 'output/owt_mlpt_decIE/owt_mlpt_2025_01_16_07_13_04/ckpt_40000.pt', # 12layer 96z 16steps
    # 'output/owt_mlpt_decIE/owt_mlpt_2025_01_16_07_13_04/ckpt_50000.pt', # 12layer 96z 16steps
    # 'output/owt_liger_mlpt/owt_mlpt_2025_01_09_07_50_46/ckpt_28000.pt', # 12layer 96z 16steps
    # 'output/owt_liger_mlpt/owt_mlpt_2025_01_09_07_50_46/ckpt_20000.pt', # 12layer 96z 16steps
    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2024_12_30_23_04_55/ckpt_59000.pt', # 12layer 96z 16steps
    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2024_12_30_23_04_55/ckpt_30000.pt', # 12layer 96z 16steps
    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2024_12_30_23_04_55/ckpt_40000.pt', # 12layer 96z 16steps
    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2024_12_30_23_04_55/ckpt_50000.pt', # 12layer 96z 16steps

    # 'output/owt_liger_mlpt/owt_mlpt_2025_01_24_08_40_10/ckpt_5000.pt', # 3layer 96z 64steps
    # 'output/owt_liger_mlpt/owt_mlpt_2025_01_23_09_01_55/ckpt_4000.pt', # 12layer24z128steps

    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_13_07_31_49/ckpt_55000.pt', # 3layer 24z 32steps
    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_13_07_31_49/ckpt_10000.pt', # 3layer 24z 32steps
    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_13_07_31_49/ckpt_20000.pt', # 3layer 24z 32steps
    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_13_07_31_49/ckpt_30000.pt', # 3layer 24z 32steps
    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_13_07_31_49/ckpt_40000.pt', # 3layer 24z 32steps

    # 'output/owt_liger_mlpt/owt_mlpt_2025_01_26_09_27_54/ckpt_5000.pt', # 6layer 24z 16steps
    'output/owt_liger_mlpt/owt_mlpt_2025_01_26_09_27_54/ckpt_32000.pt'
]

# ['ptb', 'wikitext2', 'lambada', 'lm1b', 'ag_news', 'scientific_papers_pubmed', 'scientific_papers_arxiv']:
# datasets_to_test = ['ptb']#, 'wikitext2', 'lambada','lm1b', 'ag_news']
datasets_to_test = ['wikitext2', 'lambada','lm1b', 'ag_news']
# datasets_to_test = ['scientific_papers_pubmed']

for checkpoint in checkpoints_to_check:
    print("="*30)
    checkpoint = f'../../../../random_effect_LLM/{checkpoint}'
    folder_name = f"0223-batch1_mlpt"

    os.makedirs(folder_name, exist_ok=True)

    ckpt_name = f"{folder_name}/{checkpoint.split('/')[-2]}_{checkpoint.split('/')[-1].split('.')[0]}"
    max_z_len = None # None if want to use cfg['max_z_len']

    logging.basicConfig(filename=f"{ckpt_name}_z{max_z_len}_{datetime.now().strftime('%m%d%H%M%S')}.log", level=logging.INFO, format="%(message)s")

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
    # print(cfg)
    posterior_steps = cfg['num_steps']



    ##################################################################
    message_ckpt = f"Using checkpoint {checkpoint}, seed = {seed}, max_z_len = {max_z_len}, fast_lr = {fast_lr}, posterior_steps = {posterior_steps}, use_liger={use_liger}, use_z_pos_emb={use_z_pos_emb}"
    logging.info("="*30)
    logging.info(message_ckpt)

    posterior_optimizer = PosteriorOptimizer(model = model, 
                                            inference_method='adamVIPPL', 
                                            num_steps=posterior_steps, 
                                            max_z_len=max_z_len, 
                                            z_dim=cfg['z_dim'],
                                            lr = fast_lr)

    for dataset_name in datasets_to_test:
        mode = 'test' if dataset_name in ['lm1b', 'ag_news'] else 'validation'
        dataset = get_dataset(dataset_name, mode, cache_dir='data', block_size=1024)
        dataloader = DataLoader(dataset, batch_size= 1, sampler=None, shuffle=False, num_workers=4, pin_memory=True)
        # print(dataloader.dataset)

        ppl_list = []
        kl_loss_list = []    
        nlkhd_list = []

        for batch_id, batch in enumerate(dataloader):
            question_specific_seed = np.random.randint(100000)
            torch.manual_seed(question_specific_seed)
            torch.cuda.manual_seed(question_specific_seed)
            
            x = torch.tensor(batch['input_ids']).to(device)
            z1 = torch.randn(1, max_z_len,  cfg['z_dim']).to(device)
            z = z1 * 0.01
            with ctx:
                z, ppl, kl_loss, nlkhd = posterior_optimizer.step(data=[x[:, :-1], x[:, 1:], z], ctx=ctx, seed=question_specific_seed)

                ppl_list.append(ppl.cpu().numpy())
                kl_loss_list.append(kl_loss.cpu().numpy())
                nlkhd_list.append(nlkhd.cpu().numpy())

                avg_ppl = np.mean(ppl_list)
                avg_kl_loss = np.mean(kl_loss_list)
                avg_nlkhd = np.mean(nlkhd_list)

                log_message = f"Data {batch_id}, ppl: {ppl.cpu().numpy()}, kl loss: {kl_loss.cpu().numpy()}, nlkhd: {nlkhd.cpu().numpy()}, ppl avg: {avg_ppl}, kl loss avg: {avg_kl_loss}, nlkhd avg: {avg_nlkhd:}"
                # print(log_message)
                # logging.info(log_message)

        print(f"Eval dataset: {dataset_name}, Checkpoint: {checkpoint}")
        print(f"Final ppl: {np.mean(ppl_list):.3f}")
        # print(f"Final kl loss: {np.mean(kl_loss_list):.3f}")
        
        logging.info(f"Dataset {dataset_name}-{mode}, Final avg ppl: {np.mean(ppl_list):.3f}, kl loss: {np.mean(kl_loss_list):.3f}")

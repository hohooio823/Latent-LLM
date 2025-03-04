"""
Sample from the trained model with PyTorch
"""
import os
import sys
from contextlib import nullcontext
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import torch.nn.functional as F
import logging
from datetime import datetime
from utils_gen_ppl import calculate_sentence_entropy
import time

# 12 layers
# checkpoints_to_check = [
    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_02_21_41_22/ckpt_59000.pt',
    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2024_12_27_07_11_18/ckpt_19000.pt',

    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2024_12_29_06_19_23/ckpt_19000.pt',
    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2024_12_30_23_04_55/ckpt_59000.pt',

    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2024_12_30_23_06_26/ckpt_59000.pt',
    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_02_21_33_35/ckpt_59000.pt',

    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_02_21_41_22/ckpt_59000.pt',
    # 'output/owt_liger_mlpt/owt_mlpt_2025_01_09_07_50_46/ckpt_28000.pt',

    # 'output/owt_liger_mlpt/owt_mlpt_2025_01_23_09_01_55/ckpt_4000.pt',
    # 'output/owt_liger_mlpt/owt_mlpt_2025_01_23_09_03_25/ckpt_2000.pt',

    # 'output/owt_liger_mlpt/owt_mlpt_2025_01_24_20_36_17/ckpt_26000.pt',
    # 'output/owt_liger_mlpt/owt_mlpt_2025_01_27_08_51_16/ckpt_32000.pt',

    # 'output/owt_liger_mlpt/owt_mlpt_2025_01_27_09_09_32/ckpt_2000.pt',
    # 'output/owt_mlpt_decIE/owt_mlpt_2025_01_12_02_39_04/ckpt_59000.pt',

    # 'output/owt_mlpt_decIE/owt_mlpt_2025_01_16_07_13_04/ckpt_54000.pt',
    # 'output/owt_mlpt_decIE/owt_mlpt_2025_01_18_21_11_37/ckpt_59000.pt',
    # 'output/owt_mlpt_decIE/owt_mlpt_2025_01_18_21_13_46/ckpt_43000.pt',

    # 'output/owt_liger/owt_liger_2024_12_14_08_12_37/ckpt_59000.pt',
    # 'output/owt_liger/owt_liger_2024_11_09_18_23_30/ckpt_55000.pt',
    # 'output/owt_liger/owt_liger_2024_11_09_18_23_30/ckpt_44000.pt',

# ]


# 6 layers
checkpoints_to_check = [
    # 'output/owt_liger_mlpt/owt_mlpt_2024_12_26_09_40_35/ckpt_10000.pt',
    # 'output/owt_liger_mlpt/owt_mlpt_2024_12_26_09_40_35/ckpt_18000.pt',
    # 'output/owt_liger_mlpt/owt_mlpt_2024_12_27_09_27_55/ckpt_10000.pt',
    # 'output/owt_liger_mlpt/owt_mlpt_2024_12_27_09_27_55/ckpt_18000.pt',
    # 'output/owt_liger_mlpt/owt_mlpt_2024_12_27_09_35_31/ckpt_10000.pt',
    # 'output/owt_liger_mlpt/owt_mlpt_2024_12_28_08_45_36/ckpt_15000.pt',
    # 'output/owt_liger_mlpt/owt_mlpt_2024_12_28_08_45_36/ckpt_28000.pt',
    # 'output/owt_liger_mlpt/owt_mlpt_2024_12_28_08_51_09/ckpt_15000.pt',
    # 'output/owt_liger_mlpt/owt_mlpt_2024_12_29_05_50_00/ckpt_10000.pt',
    # 'output/owt_liger_mlpt/owt_mlpt_2024_12_29_05_50_00/ckpt_18000.pt',
    # 'output/owt_liger_mlpt/owt_mlpt_2024_12_29_05_50_50/ckpt_10000.pt',
    # 'output/owt_liger_mlpt/owt_mlpt_2024_12_29_05_50_50/ckpt_19000.pt',
    # 'output/owt_liger_mlpt/owt_mlpt_2024_12_31_05_05_25/ckpt_10000.pt',
    # 'output/owt_liger_mlpt/owt_mlpt_2024_12_31_05_05_25/ckpt_19000.pt',
    # 'output/owt_liger_mlpt/owt_mlpt_2024_12_31_20_22_00/ckpt_20000.pt',
    # 'output/owt_liger_mlpt/owt_mlpt_2024_12_31_20_22_00/ckpt_35000.pt',

    # 'output/owt_liger_mlpt/owt_mlpt_2025_01_24_08_44_40/ckpt_10000.pt',
    # 'output/owt_liger_mlpt/owt_mlpt_2025_01_24_08_45_28/ckpt_4000.pt',

    'output/owt_mlpt_decIE/owt_mlpt_2025_01_16_07_15_51/ckpt_58000.pt',
    'output/owt_mlpt_decIE/owt_mlpt_2025_01_16_07_15_51/ckpt_20000.pt',
    'output/owt_mlpt_decIE/owt_mlpt_2025_01_16_07_15_51/ckpt_40000.pt',

]

z_mult_to_check = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

for checkpoint in checkpoints_to_check:
    print("="*30)
    checkpoint = f'../../{checkpoint}'
    temperature = 1 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 1 # retain only the top_k most likely tokens, clamp others to have 0 probability
    # z_mult = 1 # multiply z by this value
    folder_name = f"0129_6layer"

    os.makedirs(folder_name, exist_ok=True)
    ckpt_name = f"{folder_name}/{checkpoint.split('/')[-2]}_{checkpoint.split('/')[-1].split('.')[0]}"
    logging.basicConfig(filename=f"{ckpt_name}_{datetime.now().strftime('%m%d%H%M')}.log", level=logging.INFO, format="%(message)s", filemode='w')

    for z_mult in z_mult_to_check:
        
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


        ##################################################################

        batch_size = 64
        max_new_tokens = 1024
        start_ids = torch.full((batch_size, 1), tokenizer.bos_token_id, dtype=torch.long, device=device)

        z = torch.randn(batch_size, cfg['max_z_len'],  cfg['z_dim']).to(device)
        z = z * z_mult
        # start_time = time.time()

        samples = model.generate(start_ids, z=z , max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
        
        # end_time = time.time()
        # total_time = end_time - start_time
        # # sampling_speed = total_time / num_samples
        # print(f"Sampling speed: {total_time} seconds per sample")
        
        # print(samples[:2])

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

        sample_entropies = calculate_sentence_entropy(samples)
        entropy = np.mean(sample_entropies)

        logging.info(f"ckpt: {checkpoint}")
        logging.info(f"number of samples {batch_size}, batch size: {perplexity_batch_size}, z_mult: {z_mult}, topk: {top_k}, temperature: {temperature}")
        logging.info(f"gen perplexity: {total_perplexity.item()}, entropy: {entropy}")
        logging.info(f"-"*20)

    
        print("ckpt: ", checkpoint)
        print(f"zmult: {z_mult}, topk: {top_k}, temperature: {temperature}, number of samples {batch_size}, batch size: {perplexity_batch_size}")
        print("gen perplexity: ", total_perplexity.item(), "entropy: ", entropy)
    
    logging.info("="*20)
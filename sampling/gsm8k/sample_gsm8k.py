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
from utils_gsm8k import load_gsm8k

all_q, all_a = load_gsm8k()

for checkpoint in checkpoints_to_check:
    print("="*30)
    checkpoint = f'../../{checkpoint}'

    const_var = False
    max_new_tokens=50
    temperature=1
    top_k=1
    beam_size = 5

    question_prompt = "Question: " # "Question:"  # "Q: "  # "Q:"
    answer_prompt = "Answer: " # "Answer:"  # "A: "  # "A:"
    join_by = "" # " "

    folder_name = f"0125-mlpt_novar{const_var}_maxnew{max_new_tokens}_temp{temperature}_topk{top_k}_beam{beam_size}"

    os.makedirs(folder_name, exist_ok=True)

    ckpt_name = f"{folder_name}/{checkpoint.split('/')[-2]}_{checkpoint.split('/')[-1].split('.')[0]}"
    max_z_len = None # None if want to use cfg['max_z_len']

    logging.basicConfig(filename=f"{ckpt_name}_z{max_z_len}_{datetime.now().strftime('%m%d%H%M%S')}.log", level=logging.INFO, format="%(message)s")
    logging.info(f"question_prompt (length {len(question_prompt)}): {repr(question_prompt)}")
    logging.info(f"answer_prompt (length {len(answer_prompt)}): {repr(answer_prompt)}")
    logging.info(f"join_by (length {len(join_by)}): {repr(join_by)}")


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
    posterior_steps = cfg['num_steps']

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

    all_messages = []
    all_correct = 0

    for index, question in enumerate(all_q):  # all_q, all_a
        log_info = True if index % 20 == 0 else False
        ground_truth_answer = all_a[index]
        
        question_specific_seed = np.random.randint(100000)
        torch.manual_seed(question_specific_seed)
        torch.cuda.manual_seed(question_specific_seed)
        
        question_text = f"{bos_token}{question_prompt}{question}"
        question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
        question_tokens = question_tokens[:gptconf.max_seq_len]
        question_input = (torch.tensor(question_tokens, dtype=torch.long, device=device)[None, ...])
        
        correct_count = 0
        all_generated_answers = []
        for i in range(beam_size): 
            z1 = torch.randn(1, max_z_len,  cfg['z_dim']).to(device)
            z = z1 * 0.01
            with ctx:
                z, ppl, kl_loss, nlkhd = posterior_optimizer.step(data=[question_input[:, :-1], question_input[:, 1:], z], ctx=ctx)
                prompt = f"{question_text}{join_by}{answer_prompt}"
                answer_tokens = tokenizer.encode(prompt, add_special_tokens=False)
                answer_input = (torch.tensor(answer_tokens, dtype=torch.long, device=device)[None, ...])

                y = model.generate(answer_input, z=z, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
                generated_answer = tokenizer.decode(y[0][answer_input.shape[1]:].tolist())
                all_generated_answers.append(generated_answer)
                if ground_truth_answer in generated_answer: 
                    correct_count += 1
                
        is_correct = True if correct_count > 0 else False
        if is_correct:
            all_correct += 1
                
        
        info = f"idx {index}, is_correct: {is_correct}, correct_count: {correct_count}, correct: {all_correct}/{index+1}, {all_correct / (index+1):.4f}"
        print('-'*30)
        print(info)
        print("**INPUT prompt:**", prompt)
        print(f"**ground_truth:** {ground_truth_answer}")
        for i, generated_answer in enumerate(all_generated_answers):
            print(f"beam {i}: {generated_answer}")

        if log_info:
            logging.info("-"*30)
            logging.info(info)
            logging.info(f"**INPUT question_text:** {prompt}")
            logging.info(f"**ground_truth:** {ground_truth_answer}")

            for i, generated_answer in enumerate(all_generated_answers):
                logging.info(f"beam {i}: {generated_answer}")
                

    message = f"Final correct rate {all_correct / len(all_q):4f}, {all_correct}/{len(all_q)}"
    logging.info(message)

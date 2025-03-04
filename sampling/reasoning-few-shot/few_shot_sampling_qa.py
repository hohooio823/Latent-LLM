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
from few_shot_utils_qa import *
from optimizer import PosteriorOptimizer

task_to_test =  ['hellaswag'] # ['wsc', 'winogrande', 'siqa', 'piqa', 'obqa', 'hellaswag', 'arc_easy', 'arc_challenge']
process_functions = []
for task in task_to_test:
    if task in task_functions:
        process_functions.append(task_functions[task])
    else:
        raise ValueError(f"Task '{task}' not supported.")

# checkpoints_to_check = [
#     # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_15_09_04_44/ckpt_10000.pt',
#     # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_15_09_04_44/ckpt_20000.pt',
#     'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_15_09_04_44/ckpt_30000.pt',
# ]

# checkpoints_to_check = [
    # 'output/owt_liger/owt_liger_mlpt_2024_11_19_08_12_13/ckpt_58000.pt',
    # 'output/owt_liger/owt_liger_mlpt_2024_11_13_08_17_57/ckpt_58000.pt',
    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_15_09_04_44/ckpt_30000.pt',
    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_15_09_04_44/ckpt_59000.pt',
    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_16_01_48_06/ckpt_28000.pt',
    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_17_01_58_29/ckpt_59000.pt',
    # 'output/owt_mlpt_decIE/owt_mlpt_2025_01_18_21_13_46/ckpt_26000.pt',
    # 'output/owt_mlpt_decIE/owt_mlpt_2025_01_18_21_11_37/ckpt_51000.pt',
    # 'output/owt_liger_mlpt/owt_mlpt_2025_01_09_07_50_46/ckpt_28000.pt',
    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_02_21_41_22/ckpt_59000.pt',
    # 'output/owt_mlpt_decIE/owt_mlpt_2025_01_12_02_39_04/ckpt_59000.pt',
# ]
checkpoints_to_check = [
    'output/owt_liger_mlpt/owt_mlpt_2025_01_08_18_53_24/ckpt_59000.pt', # 3layer 24z 64steps
    'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_02_21_33_35/ckpt_59000.pt', # 12layer 24z 16steps
    # 'output/owt_liger_mlpt/owt_mlpt_2025_01_24_08_40_10/ckpt_5000.pt', # 3layer 96z 64steps
    # 'output/owt_liger_mlpt/owt_mlpt_2025_01_23_09_01_55/ckpt_4000.pt', # 12layer24z128steps
    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_13_07_31_49/ckpt_55000.pt', # 3layer 24z 32steps
    # 'output/owt_liger_mlpt_mlhead/owt_mlpt_2025_01_15_09_04_44/ckpt_59000.pt',
]

data_train = process_hellaswag_train()


for checkpoint in checkpoints_to_check:
    print("="*30)
    checkpoint = f'../../{checkpoint}'
    const_var = True
    num_shots = 10

    folder_name = f"0126_{num_shots}shot_mlpt_novar{const_var}"

    os.makedirs(folder_name, exist_ok=True)

    ckpt_name = f"{folder_name}/{checkpoint.split('/')[-2]}_{checkpoint.split('/')[-1].split('.')[0]}"
    max_z_len = None # None if want to use cfg['max_z_len']

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
    print(cfg)
    posterior_steps = cfg['num_steps']

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
    logging.info("using adamw")
    
    all_messages = []
    for task_id, process_func in enumerate(process_functions):
        print("="*30)
        print(message_ckpt)

        all_correct = 0
        all_tested = 0
        
        correct = 0
        all_questions_list = process_func()
        
        for index, item in enumerate(all_questions_list):   
            log_info = True if index % 100 == 0 else False
            question = item['question']
            answers = item['answer']

            found_context = False
            while not found_context:
                question_specific_seed = np.random.randint(1000000)
                torch.manual_seed(question_specific_seed)
                torch.cuda.manual_seed(question_specific_seed)
                # --- construct few shot examples ---
                sampled_data, _ = construct_few_shot(data_train, num_shots=num_shots, seed=question_specific_seed)            
                sampled_data_formatted = [f"{bos_token}{item}{eos_token}" for item in sampled_data]
                in_context_prompt = "".join(sampled_data_formatted)

                # --- evaluate ---
                question_text = f"{in_context_prompt}{bos_token}{question}"
                question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
                question_tokens = question_tokens[:gptconf.max_seq_len]
                question_input = (torch.tensor(question_tokens, dtype=torch.long, device=device)[None, ...])

                candidate_seqs = []
                for i in range(len(answers)):
                    answer_text = answers[i]
                    answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)
                    answer_tokens = answer_tokens[:gptconf.max_seq_len]
                    answer_input = (torch.tensor(answer_tokens, dtype=torch.long, device=device)[None, ...])
                    candidate_seqs.append(answer_input)
            
                all_qa_pair_lens = [cand.shape[1] + question_input.shape[1] for cand in candidate_seqs]
                print(f"question_input len: {question_input.shape[1]}, all answers added shape: {all_qa_pair_lens}")
                if all([l <= gptconf.max_seq_len for l in all_qa_pair_lens]):
                    found_context = True

            z1 = torch.randn(1, max_z_len,  cfg['z_dim']).to(device)
            z = z1 * 0.01
            with ctx:
                z, ppl, kl_loss, nlkhd = posterior_optimizer.step(data=[question_input[:, :-1], question_input[:, 1:], z], ctx=ctx, seed=question_specific_seed)


            lkhds = model.evaluate_conditional(question_input, z, candidate_seqs)
            generated_answer = item['label'][np.argmax(lkhds)]
            is_correct = generated_answer == item['correct_index']
            if is_correct:
                correct += 1

            option_info = f"idx {index}, is_correct: {is_correct}, likelihoods: {np.round(lkhds,2)}, generated_answer: {generated_answer}, correct_index: {item['correct_index']}, correct: {correct}/{index+1}, {correct / (index+1):.4f}"
            print(option_info)
            if log_info:
                logging.info(option_info)
                logging.info(f"**INPUT TO INFERENCE**{question_text}")
                logging.info(f"**OPTIONS**{answers}")

        all_tested += len(all_questions_list)
        all_correct += correct
        message = f"Evaluation for {task_to_test[task_id]}, correct rate {correct / len(all_questions_list):4f}, {correct}/{len(all_questions_list)}, total correct rate {all_correct / all_tested:.4f}, {all_correct}/{all_tested}"
        logging.info(message)
        logging.info(f"Checkpoint: {checkpoint}, Steps: {posterior_steps}")
        logging.info("-"*20)
        all_messages.append(message)
        print(message)

    # logging.info(all_messages)

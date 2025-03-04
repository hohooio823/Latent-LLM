"""
Sample from the trained model with PyTorch
"""
import os
import pickle
from contextlib import nullcontext
import torch
from model import ModelArgs, LatentPromptTransformerVI, LatentPromptTransformer
from optimizer import PosteriorOptimizer
import numpy as np
import argparse
import yaml
from tokenizer import Tokenizer
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

import os
import json
import logging
from datetime import datetime


device_id = "cuda:0"
# parser = argparse.ArgumentParser(description='Evaluation settings')
# parser.add_argument('--dataset', type=str, choices=['lambada', 'ptb', 'wikitext2', 'wikitext103'])
# parser.add_argument('--ckpt', type = str, default = 'output/owt/relm_owt_vi_2024_09_04_17_07_56/ckpt_46000.pt') 
# parser.add_argument('--max_steps', type = int, default = 100)
# parser.add_argument('--save', type = int, default = 1)
# args = parser.parse_args()

checkpoint = "/home/deqian/random_effect_LLM/output/owt_liger/owt_liger_2024_11_09_07_33_53/ckpt_43000.pt"
# checkpoint = "output/owt/relm_owt_vi_2024_09_14_03_59_48/ckpt_10000.pt"
# checkpoint = 'output/owt/relm_owt_vi_2024_09_15_21_42_20/ckpt_12000.pt'
# checkpoint = "output/owt/relm_owt_vi_2024_09_14_03_59_48/ckpt_46000.pt"
# checkpoint = "output/owt/relm_owt_vi_2024_09_04_17_07_56/ckpt_46000.pt"
device = torch.device(device_id)

num_steps = 15
import json
with open('data_owt/owt_all_data/val00.json', "r") as f:
    all_data = json.load(f)

print("length of dataset: ", len(all_data))

# Set up logging
dataset_name = "validation"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
_, train_dataset, ckpt_filename, ckpt_id = checkpoint.split("/") # ['output', 'owt', 'relm_owt_vi_2024_09_04_17_07_56', 'ckpt_46000.pt']
ckpt_id = ckpt_id.split(".")[0]
log_file = os.path.join(log_dir, f"run_log_{dataset_name}_{ckpt_filename}_{ckpt_id}_{train_dataset}_{datetime.now().strftime('%m%d%H%M')}.log")
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

logging.info("Criteria: if current_ppl < 100 and ((current_ppl > (5 + last_ppl)) or (abs(last_ppl - current_ppl) < 0.1))")
logging.info(f"Trained on {train_dataset}, evaluating on {dataset_name}, max step {num_steps}")
logging.info(f"Checkpoint {checkpoint}")
print(checkpoint)
print("dataset: ", dataset_name)

num_samples = 1 # number of samples to draw
max_new_tokens = 1024 # number of tokens generated in each sample
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 1 # retain only the top_k most likely tokens, clamp others to have 0 probability
adam_lr = 0.1
logging.info(f"Adam lr {adam_lr}")

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
seed = 0
device = device_id
#dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
dtype = "float32"
use_posterior = True


# REPLAN parameters:
replan = 0 # replan interval; don't replan if replan=0
max_context = 256
const_var = False
if replan > 0:
    logging.info(f"----------- REPLAN interval {replan}, max_new_tokens {max_new_tokens}, max_context {max_context}, turn off variance = {const_var} -----------")

compile = True # use PyTorch 2.0 to compile the model to be faster
decoder_only = False # only use the decoder part of the model
# exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# init from a model saved in a specific directory
checkpoint_dict = torch.load(checkpoint, map_location=device)
gptconf = ModelArgs(**checkpoint_dict['model_args'])
cfg = checkpoint_dict["config"]

# model = LatentPromptTransformerVI(gptconf)
model = LatentPromptTransformer(gptconf)
state_dict = checkpoint_dict['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict, strict=False)

model.eval()
model.to(device)
if compile:
    print("Compiling the model...")
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# load the tokenizer
vocab_source = checkpoint_dict["config"].get("vocab_source", "llama2")
vocab_size = gptconf.vocab_size

enc = GPT2TokenizerFast.from_pretrained('gpt2')

ppl_list = []
kl_loss_list = []    
nlkhd_list = []
steps_list = []
double_check_info = {}
all_match_rate = []


for dataid, data in enumerate(all_data):  
    start = data['text']
    # start = "Question: Who wrote the book the origin of species? Answer: Charles Darwin. Question: Who is the founder of the ubuntu project? Answer: Mark Shuttleworth. Question: Who is the quarterback for the green bay packers? Answers: Aaron Rodgers. Question: Panda is a national animal of which country? Answer:"
    # print(start)
    # start = """One day, a little girl named Lily found a needle in her room. She knew it was difficult to play with it because it was sharp. Lily wanted to share the needle with her mom, so she could sew a button on her shirt. Lily went to her mom and said, "Mom, I found this needle. Can you share it with me and sew my shirt?" Her mom smiled and said, "Yes, Lily, we can share the needle and fix your shirt." Together, they shared the needle and sewed the button on Lily's shirt. It was not difficult for them because they were sharing and helping each other. After they finished, Lily thanked her mom for sharing the needle and fixing her shirt. They both felt happy because they had shared and worked together."""

    bos_token = tokenizer.bos_token

    if decoder_only:
        bos_token_id = enc.bos_id
        x = torch.tensor([bos_token_id], dtype=torch.long, device=device).unsqueeze(0)
    else:
        input_text = f"{bos_token} {start}".strip()
        start_ids = tokenizer.encode(input_text, add_special_tokens=False)
        start_ids = start_ids[:1024]
        # start_ids = enc.encode(start)
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    batch_size = x.shape[0]
    z1 = torch.randn(batch_size, cfg['max_z_len'],  cfg['z_dim']).to(device)
    # z2 = torch.randn(batch_size, cfg['max_z_len'],  cfg['z_dim']).to(device)
    # z3 = torch.randn(batch_size, cfg['max_z_len'],  cfg['z_dim']).to(device)
    # z4 = torch.randn(batch_size, cfg['max_z_len'],  cfg['z_dim']).to(device)
    # z5 = torch.randn(batch_size, cfg['max_z_len'],  cfg['z_dim']).to(device)
    a = 1.0
    # z = (a * z1 + (1-a) * z2) * 0.01
    z = z1 * 0.01
    posterior_optimizer = PosteriorOptimizer(model = model, 
                                            # inference_method=cfg['inference_method'],
                                            # inference_method='adamVIPPL', 
                                            inference_method='adam', 
                                            # num_steps=cfg['num_steps'], 
                                            num_steps=num_steps,
                                            max_z_len=cfg['max_z_len'], 
                                            z_dim=cfg['z_dim'],
                                            lr = adam_lr)

    # run generation
    # with torch.no_grad():

    with ctx:
        for k in range(num_samples):
            if replan > 0:
                y = x[:, 1:]
                z, ppl, kl_loss, nlkhd, info = posterior_optimizer.step(data=[x[:, :-1], y, z], ctx=ctx, scaler=None, return_all = True, early_stop = True, const_var = const_var)
                
                # reconstruction starts with len=1 x
                x = x[:, :1]
                all_generated = x
                for step in range(1, len(start_ids), replan):
                    x = x[:, -max_context:]
                    if step + replan > len(start_ids):
                        generate_length = len(start_ids) - step
                    else:
                        generate_length = replan
                    logging.info(f"         from {step} to {step+generate_length}, use x len {x.shape}, generate_length {generate_length}")
                    print(f"         from {step} to {step+generate_length}, use x len {x.shape}, generate_length {generate_length}")
                    y_step = model.generate(x, z, generate_length, temperature=temperature, top_k=top_k)
                    # y_new_generated = y_step[:, -generate_length:]
                    y_new_generated = y_step[:, -generate_length:]
                    x = torch.cat((x, y_new_generated), dim=1)
                    all_generated = torch.cat((all_generated, y_new_generated), dim=1)
                    
                    sequence_true_ids = torch.tensor(start_ids[step:(step+generate_length)], device=y.device)
                    sequence_matches = torch.eq(y_new_generated, sequence_true_ids)
                    sequence_percentage = (sequence_matches.sum().item() / generate_length) * 100
                    logging.info(f"         percentage: {sequence_percentage}")
                    print(f"         percentage: {sequence_percentage}")
                    # break 
                    # reset z
                    # z = torch.randn(batch_size,  cfg['z_dim']).to(device)
                    # if step==0:
                    #     print(start, enc.decode(y_new_generated[0].tolist()))
                    # else:
                    #     print(enc.decode(y_new_generated[0].tolist()))
                all_matches = torch.eq(all_generated, torch.tensor(start_ids, device=y.device))
                percentage = (all_matches.sum().item() / len(start_ids)) * 100
                
                actual_steps = info['steps']
                ppl_list.append(ppl.cpu().numpy())
                kl_loss_list.append(kl_loss.cpu().numpy())
                nlkhd_list.append(nlkhd.cpu().numpy())
                steps_list.append(actual_steps)
                all_match_rate.append(percentage)       
                
                log_message = f"Data {dataid}, match {percentage:.5f}%, avg match {np.mean(all_match_rate):.5f}%, ppl: {ppl:.5f}, ppl avg: {np.mean(ppl_list):.5f}, kl loss: {kl_loss:.5f}, nlkhd: {nlkhd:.5f}, kl loss avg: {np.mean(kl_loss_list):.5f}, nlkhd avg: {np.mean(nlkhd_list):.5f}, steps = {actual_steps} out of {num_steps}, avg {np.mean(steps_list):.2f},"                
                logging.info(log_message)
                logging.info(info['message'])  
                print(log_message)              
            else:
                if use_posterior:
                    y = x[:, 1:]
                    # z, ppl, kl_loss, nlkhd, info = posterior_optimizer.step(data=[x[:, :-1], y, z], ctx=ctx, scaler=None, return_all = True, early_stop = True)
                    z = posterior_optimizer.step(data=[x[:, :-1], y, z], ctx=ctx, scaler=scaler)
                y = model.generate(x[:, :1], z=z, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
                # print(enc.decode(y[0].tolist()))
                # print('---------------')
                matches = torch.eq(y[0][:len(start_ids)], torch.tensor(start_ids, device=y.device))
                percentage = (matches.sum().item() / len(start_ids)) * 100
                                
                # actual_steps = info['steps']
                # ppl_list.append(ppl.cpu().numpy())
                # kl_loss_list.append(kl_loss.cpu().numpy())
                # nlkhd_list.append(nlkhd.cpu().numpy())
                # steps_list.append(actual_steps)
                all_match_rate.append(percentage)                
                
                log_message = f"Data {dataid}, match {percentage:.5f}%, avg match {np.mean(all_match_rate):.5f}%,"                

                # log_message = f"Data {dataid}, match {percentage:.5f}%, avg match {np.mean(all_match_rate):.5f}%, ppl: {ppl:.5f}, ppl avg: {np.mean(ppl_list):.5f}, kl loss: {kl_loss:.5f}, nlkhd: {nlkhd:.5f}, kl loss avg: {np.mean(kl_loss_list):.5f}, nlkhd avg: {np.mean(nlkhd_list):.5f}, steps = {actual_steps} out of {num_steps}, avg {np.mean(steps_list):.2f},"                
                print(log_message)
                logging.info(percentage)
                # logging.info(info['message'])
                # if ppl.cpu().numpy() > 20:
                #     double_check_info[dataid] = info 

# print(f"Checkpoint: {checkpoint}")
# print(f"Eval dataset: {dataset_name}")
# print(f"Optimizer steps = {num_steps}")
# print(f"Final ppl: {np.mean(ppl_list):.3f}")
# print(f"Final kl loss: {np.mean(kl_loss_list):.3f}")
# print(f"Average steps: {np.mean(steps_list):.3f}/{num_steps}")
print(f"Average match rate: {np.mean(all_match_rate):.5f}")


# logging.info(f"Checkpoint: {checkpoint}")
# logging.info(f"Eval dataset: {dataset_name}")
# logging.info(f"Optimizer steps = {num_steps}")
# logging.info(f"Final ppl: {np.mean(ppl_list):.3f}")
# logging.info(f"Final kl loss: {np.mean(kl_loss_list):.3f}")
# logging.info(f"Average steps: {np.mean(steps_list):.3f}/{num_steps}")
logging.info(f"Average match rate: {np.mean(all_match_rate):.5f}")

            
final_results = {
    "checkpoint": checkpoint,
    # "optimizer_max_steps": num_steps,
    # "final_avg_steps": float(np.mean(steps_list)),
    # "final_avg_ppl": float(np.mean(ppl_list)),
    "final_avg_match": float(np.mean(all_match_rate)),
    # "final_avg_kl_loss": float(np.mean(kl_loss_list)),
    # "final_avg_nlkhd": float(np.mean(nlkhd_list)),
    
    # "steps_list": list(steps_list),
    # "ppl_list": list(ppl_list),
    # "kl_loss_list": list(kl_loss_list),
    # "nlkhd_list": list(nlkhd_list),
    "match_list": list(all_match_rate)
}
# print(final_results)
logging.info(f"Final Results: {final_results}")

# with open(os.path.join(log_dir, f"final_results_{dataset_name}_{ckpt_filename}_{train_dataset}_{datetime.now().strftime('%m%d%H%M')}.json"), 'w') as f:
#     json.dump(final_results, f, indent=2)
    


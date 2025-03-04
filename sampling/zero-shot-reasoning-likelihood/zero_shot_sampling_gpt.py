"""
Sample from the trained model with PyTorch
"""
import os
import sys
import pickle
from contextlib import nullcontext
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model import ModelArgs, LatentPromptTransformerVIPostTraining, LatentPromptTransformerVI
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

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

### gpt2 ###
model_name = 'gpt2-large'  
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

### gpt2-neo ###
# from transformers import AutoTokenizer, AutoModelForCausalLM
# model_name = "gpt-neo-125m"
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

#############################################################################
model.eval()  # Set the model to evaluation mode
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Function to calculate log-likelihood of a sentence
def calculate_sentence_log_likelihood(sentence):    
    inputs = tokenizer(sentence, return_tensors='pt')
    input_ids = inputs['input_ids'].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    
    # Extract the total log-likelihood
    n_tokens = input_ids.size(1) - 1  # Number of tokens over which loss is computed
    log_likelihood = -outputs.loss.item() * n_tokens
        
    return log_likelihood


def find_best_mc_option(options):
    log_likelihoods = []
    for option in options:
        likelihood = calculate_sentence_log_likelihood(option)
        log_likelihoods.append(likelihood)
        print(f"Log-likelihood for option: '{option}' -> {likelihood:.4f}")
    
    # Find the index of the maximum likelihood
    best_option_index = log_likelihoods.index(max(log_likelihoods))
    return best_option_index, options[best_option_index]


##########
logging.basicConfig(filename=f"{model_name}.log", level=logging.INFO, format="%(message)s")
message_ckpt = f"Using model {model_name}"
logging.info(message_ckpt)
logging.info("using mean loglikelihood for evaluation")

task_to_test = ['wsc', 'winogrande', 'siqa', 'piqa', 'obqa', 'hellaswag', 'arc_easy', 'arc_challenge'] # ['winogrande', 'siqa', 'piqa', 'hellaswag', 'arc_challenge'] # ['wsc', 'obqa', 'arc_easy'] # ['wsc', 'winogrande', 'siqa', 'piqa', 'obqa', 'hellaswag', 'arc_easy', 'arc_challenge']
process_functions = []
for task in task_to_test:
    if task in task_functions:
        process_functions.append(task_functions[task])
    else:
        raise ValueError(f"Task '{task}' not supported.")

all_messages = []
for task_id, process_func in enumerate(process_functions):
    correct = 0
    data_list = process_func()

    for index, item in enumerate(data_list):  
        sentences_raw = item['sentences']
        sentences = [f"{tokenizer.bos_token}{s}".strip() for s in item['sentences']]
        log_likelihoods = []
        for i, option in enumerate(sentences):
            likelihood = calculate_sentence_log_likelihood(option)
            log_likelihoods.append(likelihood)
            print(f"option {i}: {sentences[i]} -> loglkhd {likelihood:.4f}")
        
        # Find the index of the maximum likelihood
        best_option_index = log_likelihoods.index(max(log_likelihoods))
        generated_answer = item['label'][best_option_index]          
        
        is_correct = generated_answer == item['correct_index']
        if is_correct:
            correct += 1

        print(f"index {index}: correct: {is_correct}, lkhd: {(log_likelihoods)}, generatedID: {generated_answer}, correctID: {item['correct_index']}, current rate {correct/(index+1):.2f}")
        logging.info(f"index {index}: correct: {is_correct}, lkhd: {(log_likelihoods)}, generatedID: {generated_answer}, correctID: {item['correct_index']}, current rate {correct/(index+1):.2f}")
        print("---")
        break

    # print("-"*20)
    message = f"Evaluation for {task_to_test[task_id]}, correct rate {correct / len(data_list):4f}, {correct}/{len(data_list)}"
    logging.info(message)
    all_messages.append(message)
    print(message)
    break




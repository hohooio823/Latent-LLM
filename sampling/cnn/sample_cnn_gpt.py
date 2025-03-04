"""
Sample from the trained model with PyTorch
"""
import os
import sys
from contextlib import nullcontext
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPTNeoXTokenizerFast, GPT2Tokenizer
import logging
from datetime import datetime
from datasets import load_dataset
from utils_cnn import *


# Load the pre-trained GPT-2 medium model and tokenizer
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

### gpt2-neo ###
# from transformers import AutoTokenizer, AutoModelForCausalLM
# model_name = "gpt-neo-125m"
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
model.eval()  # Set the model to evaluation mode
model.to('cuda' if torch.cuda.is_available() else 'cpu')

dataset = load_dataset("cnn_dailymail", "1.0.0", split="test")  

folder_name = f"{model_name}"
os.makedirs(folder_name, exist_ok=True)

max_new_tokens = 100

logging.basicConfig(filename=f"{folder_name}/{model_name}_maxnew{max_new_tokens}_{datetime.now().strftime('%m%d%H%M')}.log", level=logging.INFO, format="%(message)s")

##################################################################
message_ckpt = f"Using model {model_name}"
logging.info(message_ckpt)

generated_summaries = []
reference_summaries = []

for index, item in enumerate(dataset):  
    article_text = item['article']
    reference_summary = item['highlights']

    log_info = True if index % 100 == 0 else False

    input_text = f"{article_text} TL;DR:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length = 1024-max_new_tokens).to(model.device)
    input_len = input_ids.shape[1]

    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,        # Maximum length of the generated text
        early_stopping=True    # Stop generation when an end-of-sequence token is generated
    )

    # Decode the generated summary
    summary = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
    generated_summary = split_sentences(summary.strip())

    generated_summaries.append(generated_summary)
    reference_summaries.append(reference_summary)
    rouge_scores = calculate_rouge(generated_summaries, reference_summaries)

    avg_rouge = np.mean(list(rouge_scores.values()))
    info = f"idx {index}, Current avg: {avg_rouge:.3f}, {rouge_scores}"    
    if log_info:
        logging.info(info)
        print(info)
        logging.info(f"input_text: {input_text}")
        logging.info("-" * 50)
        logging.info(f"generated_summary: {generated_summary}")
        logging.info("-" * 50)
        logging.info(f"reference_summary: {reference_summary}")


message = f"Final avg: {avg_rouge:.3f}, {rouge_scores}"
logging.info(message)

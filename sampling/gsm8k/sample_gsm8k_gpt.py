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
from utils_gsm8k import load_gsm8k
all_q, all_a = load_gsm8k()

# Load the pre-trained GPT-2 medium model and tokenizer
model_name = "gpt2-large"
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

folder_name = f"0122-{model_name}"
os.makedirs(folder_name, exist_ok=True)

max_new_tokens = 20
num_beams = 5


logging.basicConfig(filename=f"{folder_name}/{model_name}_maxnew{max_new_tokens}_beam{num_beams}_{datetime.now().strftime('%m%d%H%M')}.log", level=logging.INFO, format="%(message)s")

##################################################################
message_ckpt = f"Using model {model_name}"
logging.info(message_ckpt)

all_messages = []
all_correct = 0
question_prompt = "Question: "
answer_prompt = "Answer: "
logging.info(f"question_prompt (length {len(question_prompt)}): {repr(question_prompt)}")
logging.info(f"answer_prompt (length {len(answer_prompt)}): {repr(answer_prompt)}")

for index, question in enumerate(all_q):  # all_q, all_a
    log_info = True if index % 100 == 0 else False
    ground_truth_answer = all_a[index]

    # question_text = f"Question: {question}Answer: "
    question_text = f"{question_prompt}{question}{answer_prompt}"

    input_ids = tokenizer.encode(question_text, return_tensors="pt").to(model.device)

    correct_count = 0
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,       
        # max_length = 200,
        num_beams=num_beams,         
        num_return_sequences=num_beams, 
        early_stopping=True   
    )

    print(f"index {index}, question_text: {question_text}")
    generated_answers = []
    for seq in output:
        # decoded = tokenizer.decode(seq, skip_special_tokens=True)
        # print("all_decoded: ", decoded)
        # print("-" * 50)
        seq = seq[input_ids.shape[1]:]
        decoded = tokenizer.decode(seq, skip_special_tokens=True)
        generated_answers.append(decoded)
        print(f"generated_answer: {decoded}")
    
    for generated_answer in generated_answers: 
        if ground_truth_answer in generated_answer: 
            correct_count += 1
            
    is_correct = True if correct_count > 0 else False
    if is_correct:
        all_correct += 1
            
    # print(generated_answers)

    info = f"idx {index}, is_correct: {is_correct}, correct_count: {correct_count}, correct: {all_correct}/{index+1}, {all_correct / (index+1):.4f}"
    if log_info:
        logging.info(info)
        print(info)
        logging.info(f"**INPUT question_text:** {question_text}")
        logging.info(f"ground_truth: {ground_truth_answer}")
        print(f"ground_truth: {ground_truth_answer}")
        for i, generated_answer in enumerate(generated_answers):
            logging.info(f"beam {i}: {generated_answer}")
            print(f"beam {i}: {generated_answer}")
        logging.info("-"*30)
            

message = f"Final correct rate {all_correct / len(all_q):4f}, {all_correct}/{len(all_q)}"
logging.info(message)

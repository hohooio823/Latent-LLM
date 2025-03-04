from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from tqdm import tqdm
from utils_ppl import *

device = torch.device('cuda:0')
model_id = "openai-community/gpt2-medium"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
eos_token = tokenizer.bos_token

dataset_names = ["ptb", "wikitext2", "lm1b", "lambada"]
connect_by = "\n\n"
# connect_by = eos_token
print(f"connect_by: {connect_by}")

for dataset_name in dataset_names:
    dataset = load_data(dataset_name)
    keyword = "sentence" if dataset_name == "ptb" else "text"
    all_encodings = []
    for sequence in dataset[keyword]:
        encodings = tokenizer(sequence + connect_by, return_tensors="pt")
        # encodings = tokenizer(sequence + eos_token, return_tensors="pt")
        all_encodings.extend(encodings.input_ids[0].tolist())

    all_encodings = torch.tensor(all_encodings).view(1, -1)

    max_length = model.config.n_positions
    stride = 1024
    seq_len = all_encodings.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        input_ids = all_encodings[:, begin_loc:end_loc].to(device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            neg_log_likelihood = outputs.loss

        # Accumulate the total negative log-likelihood and the total number of tokens
        num_valid_tokens = input_ids.size(1)  # All tokens are valid
        nll_sum += neg_log_likelihood * num_valid_tokens
        n_tokens += num_valid_tokens

        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
    ppl = torch.exp(avg_nll)

    print(f"Dataset: {dataset_name}, model_id: {model_id}, final ppl: {ppl.item()}")

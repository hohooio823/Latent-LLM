import json 
import random 

def load_gsm8k():
    # Load the questions and answers from a JSON file
    json_file_path_q = "../../data_gsm8k/all_data_test/question/data00.json"
    json_file_path_a = "../../data_gsm8k/all_data_test/answer/data00.json"

    all_q = []
    all_a = []
    with open(json_file_path_q, "r") as f:
        for line in f:
            example = json.loads(line)  
            all_q.append(example)
    with open(json_file_path_a, "r") as f:
        for line in f:
            example = json.loads(line)  
            all_a.append(example)

    # Initialize variables for tracking results
    correct_count = 0
    total_count = len(all_q)
    print(f"Loaded {total_count} questions and answers, example: Q -> {all_q[0]}, A -> {all_a[0]}")
    return all_q, all_a


def load_gsm8k_train():
    with open("../../data_gsm8k/all_data_train/question/data00.json", "r") as f:
        all_q = [json.loads(line) for line in f]


    with open("../../data_gsm8k/all_data_train/answer/data00.json", "r") as f:
        all_a = [json.loads(line) for line in f]

    return all_q, all_a

def compose_qa_pairs(all_q, all_a, seed, num_samples):
    random.seed(seed)
    qa_pairs = list(zip(all_q, all_a))
    sampled_pairs = random.sample(qa_pairs, num_samples)
    return sampled_pairs



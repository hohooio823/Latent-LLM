import json
from datasets import load_dataset
import re
import datasets

def process_wsc():
    data_list = []

    with open('wsc273.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    blocks = []
    current_block = []

    for line in lines:
        line = line.strip()
        if line == '':
            if current_block:
                blocks.append(current_block)
                current_block = []
        else:
            current_block.append(line)

    # Append the last block if any
    if current_block:
        blocks.append(current_block)

    for block in blocks:
        if len(block) != 4:
            print(f"Unexpected block format: {block}")
            continue
        sentence_with_mask = block[0]
        mask_line = block[1]  # should be '[MASK]'
        options_line = block[2]
        correct_answer = block[3]
        
        options = options_line.split(',')
        options = [opt.strip() for opt in options]
        
        if correct_answer not in options:
            print(f"Correct answer not in options: {correct_answer}")
            continue
        
        correct_index = options.index(correct_answer)
        
        # Replace [MASK] with each option
        sentences = []
        for opt in options:
            sentence = sentence_with_mask.replace(' [MASK] ', opt)
            sentences.append(sentence)
        
        data_list.append({
            'sentences': sentences,
            'correct_index': correct_index,
            'label': [0, 1]
        })
        
    return data_list

# Prepare the winogrande
def process_winogrande():
    data_list = []
    file_path = 'winogrande-dev.jsonl'

    # Reading the JSONL file
    with open(file_path, 'r') as file:
        for line in file:
            # Parse each line as JSON and append to the list
            data = json.loads(line.strip())
            answer_to_num = {"1": 0, "2": 1}

            sentence1 = data["sentence"].replace('_', data["option1"])
            sentence2 = data["sentence"].replace('_', data["option2"])
            sentences = [sentence1, sentence2]
            
            data_list.append({
                'sentences': sentences,
                'correct_index': answer_to_num[data["answer"]],
                'label': [0, 1]
            })
    return data_list
      
# Prepare the siqa

def process_siqa():
    with open("siqa-dev-labels.lst", "r") as file:
        data = file.readlines()

    responses = [int(line.strip()) for line in data]

    data_list = []
    file_path = 'siqa-dev.jsonl'

    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            data = json.loads(line.strip())
            query = f"Q: {data['context']} {data['question']}\nA:"
            sentence1 = f"{query} {data['answerA']}"
            sentence2 = f"{query} {data['answerB']}"
            sentence3 = f"{query} {data['answerC']}"
            sentences = [sentence1, sentence2, sentence3]
            data_list.append({
                'sentences': sentences,
                'correct_index': responses[i],
                'label': [1,2,3]
            })
    return data_list
      
      
def process_piqa():

    with open("piqa-valid-labels.lst", "r") as file:
        data = file.readlines()

    responses = [int(line.strip()) for line in data]


    data_list = []
    with open('piqa-valid.jsonl', 'r') as file:
        for i, line in enumerate(file):
            data = json.loads(line.strip())
            query = f"Question: {data['goal']}\nAnswer:"
            sentence1 = f"{query} {data['sol1']}"
            sentence2 = f"{query} {data['sol2']}"

            data_list.append({
                'sentences': [sentence1, sentence2], 
                'correct_index': responses[i],
                'label': [0,1]
            })
    return data_list
            
def process_obqa():
    ds = load_dataset("allenai/openbookqa", "main")
    cleaned_ds = ds['test']
    data_list = []

    for index, item in enumerate(cleaned_ds):  
        sentences = []
        for i in range(len(item['choices']['label'])):
            st = f"{item['question_stem']} {item['choices']['text'][i]}"
            sentences.append(st)
        correct_index = item['answerKey']
        label = item['choices']['label']
    
        data_list.append({
            'sentences': sentences,
            'correct_index': correct_index,
            'label': label
        })
    return data_list

def process_hellaswag():
    # Prepare the hellaswag

    def preprocess(text):
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
        def _process_doc(doc):
            ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
            out_doc = {
                "query": preprocess(doc["activity_label"] + ": " + ctx),
                "choices": [preprocess(ending) for ending in doc["endings"]],
                "gold": int(doc["label"]),
            }
            return out_doc

        return dataset.map(_process_doc)

    ds = load_dataset("Rowan/hellaswag")
    test_ds = ds['validation']
    cleaned_ds = process_docs(test_ds)

    data_list = []
    for index, item in enumerate(cleaned_ds):  
        sentences = [item['query'] + ' ' + item['choices'][i] for i in range(4)]
        correct_index = item['gold']
        
        data_list.append({
            'sentences': sentences,
            'correct_index': correct_index,
            'label': list(range(4))
        })
        
    return data_list


def process_arc_easy():
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy")
    cleaned_ds = ds['test']
    print(f"Processing arc_easy, total number of examples: {len(cleaned_ds)}")

    data_list_4_options = []
    data_list_3_options = []
    data_list_5_options = []

    for index, item in enumerate(cleaned_ds):  
        sentences = []
        for i in range(len(item['choices']['label'])):
            st = f"Question: {item['question']}\nAnswer: {item['choices']['text'][i]}"
            sentences.append(st)
        
        correct_index = item['answerKey']
        label = item['choices']['label']
        
        if len(item['choices']['label']) == 4:
            data_list_4_options.append({
                'sentences': sentences,
                'correct_index': correct_index,
                'label': label
            })
        elif len(item['choices']['label']) == 3:
            data_list_3_options.append({
                'sentences': sentences,
                'correct_index': correct_index,
                'label': label
            })
        elif len(item['choices']['label']) == 5:
            data_list_5_options.append({
                'sentences': sentences,
                'correct_index': correct_index,
                'label': label
            })
        else:
            print(f"Unexpected number of options: {len(item['choices']['label'])}")

    return data_list_3_options, data_list_4_options, data_list_5_options


def process_arc_challenge():
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    cleaned_ds = ds['test']
    data_list_3_options = []
    data_list_4_options = []
    data_list_5_options = []

    print(f"Processing arc_challenge, total number of examples: {len(cleaned_ds)}")

    for index, item in enumerate(cleaned_ds):  
        sentences = []
        for i in range(len(item['choices']['label'])):
            st = f"Question: {item['question']}\nAnswer: {item['choices']['text'][i]}"
            sentences.append(st)
        
        correct_index = item['answerKey']
        label = item['choices']['label']
    
        if len(item['choices']['label']) == 4:
            data_list_4_options.append({
                'sentences': sentences,
                'correct_index': correct_index,
                'label': label
            })
        elif len(item['choices']['label']) == 3:
            data_list_3_options.append({
                'sentences': sentences,
                'correct_index': correct_index,
                'label': label
            })
        elif len(item['choices']['label']) == 5:
            data_list_5_options.append({
                'sentences': sentences,
                'correct_index': correct_index,
                'label': label
            })
        else:
            print(f"Unexpected number of options: {len(item['choices']['label'])}")

    return data_list_3_options, data_list_4_options, data_list_5_options


task_functions = {
    "wsc": process_wsc,
    "winogrande": process_winogrande,
    "siqa": process_siqa, 
    "piqa": process_piqa,
    "obqa": process_obqa,
    "hellaswag": process_hellaswag,
    "arc_easy": process_arc_easy,
    "arc_challenge": process_arc_challenge
}

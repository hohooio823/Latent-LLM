import json
import requests
import datasets
from datasets import load_dataset
from pathlib import Path

def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
        response = requests.get(url, stream=True)
        data_list = []

        # Process each line in the response content
        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                data_list.append(data)

        return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = datasets.Dataset.from_list(lambada_data)
    return dataset


def load_data(dataset_name):
    cache_dir = 'data'
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)  # Ensure cache directory exists

    if dataset_name == 'wikitext2':
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir = 'data', split='validation')  # ['text']
    elif dataset_name == 'ptb':
        dataset = load_dataset("ptb_text_only", cache_dir = 'data', trust_remote_code=True, split='validation') # ['sentence']
    elif dataset_name == 'lambada':
        dataset = get_lambada_test_dataset() # ['text']
        # dataset = load_dataset("cimec/lambada", cache_dir = 'data', split='test')
    elif dataset_name == 'lm1b':
        dataset = datasets.load_dataset('lm1b', cache_dir = 'data', trust_remote_code=True, split='test')  # ['text']
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    return dataset



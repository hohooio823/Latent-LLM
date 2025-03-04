import re
from transformers import GPT2TokenizerFast
from itertools import chain
import numpy as np
import torch
import os
import urllib.request
import zipfile
import datasets
import requests
import json

from torch.utils.data import DataLoader, DistributedSampler


def wt_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def ptb_detokenizer(x):
    x = x.replace(" 's", "'s")
    x = x.replace("s ' ", "s' ")
    x = x.replace(" n't", "n't")
    x = x.replace(" \n ", "\n")
    x = x.replace("\\/", "/")
    for _ in range(10):
        x = x.replace(" N ", " 1 ")
    x = x.replace("$ 1", "$1")
    x = x.replace("# 1", "#1")
    x = x.replace("<unk>", "?")
    return x


def lm1b_detokenizer(x):
    x = x.replace('http : / / ', 'http://')
    x = x.replace('https : / / ', 'https://')
    x = re.sub(r' \'(\w+)', r"'\1", x)
    x = re.sub(r' (\w+) \. ', r' \1. ', x)
    x = re.sub(r' (\w+) \.$', r' \1.', x)
    x = x.replace(' ? ', '? ')
    x = re.sub(r' \?$', '?', x)
    x = x.replace(' ! ', '! ')
    x = re.sub(r' \!$', '!', x)
    x = x.replace(' , ', ', ')
    x = x.replace(' : ', ': ')
    x = x.replace(' ; ', '; ')
    x = x.replace(' / ', '/')
    x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
    x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
    x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
    x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
    x = x.replace('$ ', '$')
    x = x.replace('£ ', '£')
    return x


def lambada_detokenizer(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return '\n' + text.strip()


def scientific_papers_detokenizer(x):
    x = wt_detokenizer(x)
    x = lm1b_detokenizer(x)
    return x


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


def _group_texts(examples, block_size, bos, eos):
    # Concatenate all texts.
    concatenated_examples = list(itertools.chain(* examples['input_ids']))
    total_length = len(concatenated_examples)
    # TODO(yair): look into not dropping the remainder but rather padding it.
    # We drop the small remainder, and if the total_length < block_size - 2
    # we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of
    # this drop, you can customize this part to your needs.
    new_block_size = block_size - 2  # [BOS] and [EOS] to be added
    total_length = (total_length // new_block_size) * new_block_size
    # Split by chunks of max_len.
    result = {}
    _values = []
    _attn_masks = []
    for i in range(0, total_length, new_block_size):
        _values.append(
            [bos]
            + concatenated_examples[i: i + new_block_size]
            + [eos])
        _attn_masks.append(torch.ones(block_size))
    result['input_ids'] = _values
    result['attention_mask'] = _attn_masks
    return result


def get_dataset(
    dataset_name, mode, cache_dir=None,
    block_size=1024, num_proc=8, streaming=False):
    filename = f'{dataset_name}_{mode}_bs{block_size}_wrapped.dat'
    cache_dir = ""
    _path = os.path.join(cache_dir, filename)

    if dataset_name == 'wikitext103':
        dataset = datasets.load_dataset(
            'wikitext',
            name='wikitext-103-raw-v1',
            cache_dir=cache_dir, trust_remote_code=True)
    elif dataset_name == 'wikitext2':
        dataset = datasets.load_dataset(
            'wikitext',
            name='wikitext-2-raw-v1',
            cache_dir=cache_dir, trust_remote_code=True)
    elif dataset_name == 'ptb':
        dataset = datasets.load_dataset(
            'ptb_text_only', cache_dir=cache_dir, trust_remote_code=True)
    elif dataset_name == 'lambada':
        dataset = get_lambada_test_dataset()
    elif dataset_name == 'openwebtext-train':
        dataset = datasets.load_dataset(
            'openwebtext',
            split='train[:-100000]',
            cache_dir=cache_dir,
            streaming=streaming)
    elif dataset_name == 'openwebtext-valid':
        dataset = datasets.load_dataset(
            'openwebtext',
            split='train[-100000:]',
            cache_dir=cache_dir,
            streaming=streaming)
    elif dataset_name == 'scientific_papers_arxiv':
        dataset = datasets.load_dataset(
            'scientific_papers', 'arxiv',
            trust_remote_code=True,
            cache_dir=cache_dir,
            streaming=streaming)
    elif dataset_name == 'scientific_papers_pubmed':
        dataset = datasets.load_dataset(
            'scientific_papers', 'pubmed',
            trust_remote_code=True,
            cache_dir=cache_dir,
            streaming=streaming)
    elif dataset_name == 'ag_news':
        dataset = datasets.load_dataset(
            'ag_news',
            cache_dir=cache_dir,
            streaming=streaming)
    else:
        dataset = datasets.load_dataset(dataset_name, cache_dir=cache_dir, trust_remote_code=True)

    if dataset_name in ['lambada', 'openwebtext-train',
                        'openwebtext-valid']:
        data = dataset
    else:
        data = dataset[mode]

    if dataset_name.startswith('wikitext'):
        detokenizer = wt_detokenizer
    elif dataset_name == 'ptb':
        detokenizer = ptb_detokenizer
    elif dataset_name == 'lm1b':
        detokenizer = lm1b_detokenizer
    elif dataset_name == 'lambada':
        detokenizer = lambada_detokenizer
    elif dataset_name.startswith('scientific_papers'):
        detokenizer = scientific_papers_detokenizer
    else:
        detokenizer = None

    def _apply_detokenizer(detokenizer):
        def detok(text):
            for i, t in enumerate(text, 0):
                text[i] = detokenizer(t)
            return text
        return detok
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    EOS = tokenizer.encode(tokenizer.eos_token)[0]

    def preprocess_and_tokenize(example):
        if dataset_name == 'ptb':
            text = example['sentence']
        elif 'scientific_papers' in dataset_name:
            text = example['article']
        else:
            text = example['text']

        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)

        tokenizer.padding_side = 'right'
        tokenizer.truncation_side = 'right'
        tokens = tokenizer(text,
                            add_special_tokens=False,
                            return_attention_mask=False,
                            return_token_type_ids=False)
        tokens = {'input_ids':
                    [t + [EOS] for t in tokens['input_ids']]}
        # Still missing BOS, but will be added in group_texts

        return tokens
    tokenized_dataset = data.map(
        preprocess_and_tokenize,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,
        desc='Tokenizing')
    if dataset_name == 'ptb':
        tokenized_dataset = tokenized_dataset.remove_columns(
            'sentence')
    elif 'scientific_papers' in dataset_name:
        tokenized_dataset = tokenized_dataset.remove_columns([
            'article', 'abstract', 'section_names'])
    elif dataset_name == 'ag_news':
        tokenized_dataset = tokenized_dataset.remove_columns(
            ['text', 'label'])
    else:
        tokenized_dataset = tokenized_dataset.remove_columns(
            'text')

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    chunked_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=num_proc, load_from_cache_file=True)
    chunked_dataset = chunked_dataset.with_format('torch')

    return chunked_dataset




def get_dataloaders(config):

    if config.data.valid in ['lm1b', 'ag_news']:
        validation_split = 'test'
    else:
        validation_split = 'validation'

    valid_set = get_dataset(
        config.data.valid, 
        mode=validation_split)


    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=config.eval.batch_size,
        num_workers=4,
        pin_memory=True,
    )
    return valid_loader


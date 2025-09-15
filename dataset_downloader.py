"""
Dataset downloading and preprocessing script for Latent Thought Language Model.
Uses HuggingFace datasets to download and preprocess OpenWebText and other datasets.
"""

import argparse
import os
import glob
import json
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import logging

from datasets import load_dataset, Dataset
from transformers import GPT2TokenizerFast

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# TOP-LEVEL WORKER FUNCTIONS FOR MULTIPROCESSING
# =============================================================================

def _filter_non_empty_text(example):
    """Top-level function for filtering. Replaces the lambda."""
    return len(example.get('text', '').strip()) > 0

def _worker_tokenize_and_wrap(texts, tokenizer_name, max_length):
    """Top-level helper function to concatenate and tokenize texts."""
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
    
    all_tokens = []
    current_tokens = []
    eos_token_id = tokenizer.eos_token_id

    for text in texts:
        tokens = tokenizer(text, truncation=False, padding=False, return_tensors=None)['input_ids']
        if current_tokens:
            tokens_with_eos = tokens + [eos_token_id]
        else:
            tokens_with_eos = tokens
        if len(current_tokens) + len(tokens_with_eos) <= max_length:
            current_tokens.extend(tokens_with_eos)
        else:
            if current_tokens:
                all_tokens.append(current_tokens)
            current_tokens = tokens_with_eos[:max_length]
    if current_tokens:
        all_tokens.append(current_tokens)
    return all_tokens


def _worker_process_batch(batch, tokenizer_name, max_length):
    """Top-level worker function that processes a batch of data."""
    results = []
    texts = batch.get('text', [])
    tokenized_segments = _worker_tokenize_and_wrap(texts, tokenizer_name, max_length)
    for tokens in tokenized_segments:
        results.append({'input_ids': tokens, 'attention_mask': [1] * len(tokens)})
    return results


class DatasetDownloader:
    """Handles downloading and preprocessing of datasets for LTM training."""

    def __init__(self, cache_dir="data_owt", tokenizer_name="gpt2", data_files_limit=None):
        self.cache_dir = cache_dir
        self.tokenizer_name = tokenizer_name
        self.data_files_limit = data_files_limit
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories for dataset storage."""
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "tok_gpt2"), exist_ok=True)

    def download_openwebtext(self, split="train", num_samples=None):
        """Download OpenWebText dataset from HuggingFace."""
        logger.info(f"Downloading OpenWebText dataset ({split} split)...")
        
        split_map = {
            "train": "train", "val": "validation", "validation": "validation", "test": "test"
        }
        hf_split = split_map.get(split, "train")

        datasets_to_try = [
            ("olivercareyncl/openwebtext", hf_split), ("allenai/c4", hf_split),
            ("wikitext", "wikitext-2-raw-v1"), ("pile", hf_split)
        ] if split == "train" else [
            ("wikitext", "wikitext-2-raw-v1"), ("allenai/c4", hf_split),
            ("olivercareyncl/openwebtext", hf_split)
        ]

        for dataset_name, dataset_split in datasets_to_try:
            try:
                current_split = hf_split if dataset_name != "wikitext" else split
                if current_split == "val": current_split = "validation"

                logger.info(f"Trying dataset: {dataset_name} (config: {dataset_split}, split: {current_split})")
                
                if dataset_name == "olivercareyncl/openwebtext" and self.data_files_limit is not None:
                    logger.info(f"Limiting download to {self.data_files_limit} data file(s).")
                    files_to_load = [f"split/chunk_{i}.json" for i in range(1, self.data_files_limit + 1)]
                    dataset = load_dataset(dataset_name, data_files=files_to_load, split=dataset_split)
                elif dataset_name in ["wikitext", "pile"]:
                    dataset = load_dataset(dataset_name, dataset_split, split=current_split)
                elif dataset_name == "allenai/c4":
                    dataset = load_dataset(dataset_name, "en", split=dataset_split)
                else:
                    dataset = load_dataset(dataset_name, split=dataset_split)

                if num_samples is not None:
                    dataset = dataset.select(range(min(num_samples, len(dataset))))

                logger.info(f"Successfully downloaded {len(dataset)} samples from {dataset_name}")
                return dataset
            except Exception as e:
                logger.error(f"Failed to load {dataset_name}: {e}")
                continue
        logger.error("All dataset loading attempts failed. Creating synthetic dataset.")
        return Dataset.from_dict({"text": ["Synthetic test document.", "Another synthetic text."]})

    def preprocess_dataset(self, dataset, split="train", num_workers=4, max_length=1024):
        """Preprocess dataset by tokenizing text."""
        logger.info(f"Preprocessing {split} split with {num_workers} workers...")
        dataset = dataset.filter(_filter_non_empty_text, num_proc=num_workers)
        process_func = partial(_worker_process_batch, tokenizer_name=self.tokenizer_name, max_length=max_length)
        if len(dataset) > 10000 and num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                num_batches = num_workers * 4
                batch_size = max(1, len(dataset) // num_batches)
                futures = [executor.submit(process_func, dataset[i:i + batch_size]) for i in range(0, len(dataset), batch_size)]
                results = [item for future in tqdm(futures, desc="Processing batches") for item in future.result()]
        else:
            logger.info("Dataset is small or num_workers=1, processing in a single thread.")
            results = process_func(dataset[:])
        logger.info(f"Preprocessed {len(results)} examples")
        return results

    def save_tokenized_data(self, tokenized_data, split="train"):
        """Save tokenized data to binary files."""
        logger.info(f"Saving tokenized data for {split} split...")
        all_tokens = [token for ex in tokenized_data for token in ex['input_ids']]
        tokens_array = np.array(all_tokens, dtype=np.uint16)
        output_file = os.path.join(self.cache_dir, "tok_gpt2", f"{split}{len(tokenized_data)}.bin")
        tokens_array.tofile(output_file)
        logger.info(f"Saved {len(tokens_array)} tokens to {output_file}")
        metadata = {'split': split, 'num_examples': len(tokenized_data), 'num_tokens': len(tokens_array)}
        with open(os.path.join(self.cache_dir, "tok_gpt2", f"{split}_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

    def download_and_preprocess(self, dataset_name="openwebtext", split="train", num_samples=None, num_workers=4, max_length=1024):
        """Complete pipeline: download, preprocess, and save dataset."""
        logger.info(f"Starting pipeline for {dataset_name} ({split})...")
        dataset = self.download_openwebtext(split, num_samples)
        if dataset is None: return False
        tokenized_data = self.preprocess_dataset(dataset, split, num_workers, max_length)
        self.save_tokenized_data(tokenized_data, split)
        logger.info("Pipeline completed successfully!")
        return True

def main():
    parser = argparse.ArgumentParser(description="Download and preprocess datasets for LTM.")
    parser.add_argument("--dataset", type=str, default="openwebtext")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--cache_dir", type=str, default="data_owt")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--data-files-limit", type=int, default=None)
    args = parser.parse_args()

    downloader = DatasetDownloader(
        cache_dir=args.cache_dir,
        tokenizer_name=args.tokenizer,
        data_files_limit=args.data_files_limit
    )
    downloader.download_and_preprocess(
        dataset_name=args.dataset, split=args.split,
        num_samples=args.num_samples, num_workers=args.num_workers,
        max_length=args.max_length
    )

if __name__ == "__main__":
    main()

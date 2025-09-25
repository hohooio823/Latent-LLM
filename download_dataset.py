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
from functools import partial
import logging

from datasets import load_dataset, Dataset
from transformers import GPT2TokenizerFast

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetDownloader:
    """Handles downloading and preprocessing of datasets for LTM training."""

    def __init__(self, cache_dir="data_owt", tokenizer_name="gpt2", data_files_limit=None):
        self.cache_dir = cache_dir
        self.tokenizer_name = tokenizer_name
        self.data_files_limit = data_files_limit
        self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories for dataset storage."""
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "tok_gpt2"), exist_ok=True)

    def download_dataset(self, dataset_name="openwebtext", split="train", num_samples=None):
        """
        Download a specified dataset from HuggingFace.
        Handles special cases like FineWeb/FineWeb-Edu and provides fallbacks.
        """
        logger.info(f"Attempting to download dataset: {dataset_name} ({split} split)...")
        
        dataset_loaded = False
        dataset = None
        
        try:
            # Handle Fineweb-Edu
            if "fineweb-edu" in dataset_name.lower():
                hf_path = "HuggingFaceFW/fineweb-edu"
                config_name = "sample/10BT"
                logger.info(f"Loading FineWeb-Edu dataset: {hf_path} (config: {config_name})")
                
                if self.data_files_limit is not None:
                    logger.info(f"Limiting download to the first {self.data_files_limit} data file(s).")
                    # --- FIX: Prepend the correct subdirectory path to the filenames ---
                    files_to_load = [f"{config_name}/{i:03d}_00000.parquet" for i in range(self.data_files_limit)]
                    # When providing explicit data_files, you don't need the 'name' argument.
                    dataset = load_dataset(hf_path, data_files=files_to_load, split="train")
                else:
                    dataset = load_dataset(hf_path, name=config_name, split="train")
                dataset_loaded = True

            # Handle original Fineweb
            elif "fineweb" in dataset_name.lower():
                hf_path = "HuggingFaceFW/fineweb"
                config_name = "sample-10BT"
                if "100b" in dataset_name.lower():
                    config_name = "sample-100BT"
                logger.info(f"Loading FineWeb dataset: {hf_path} (config: {config_name})")

                if self.data_files_limit is not None:
                    logger.info(f"Limiting download to the first {self.data_files_limit} data file(s).")
                    # Original Fineweb files are at the root of their config, so no prefix is needed.
                    files_to_load = [f"{i:03d}_00000.parquet" for i in range(self.data_files_limit)]
                    dataset = load_dataset(hf_path, name=config_name, data_files=files_to_load, split="train")
                else:
                    dataset = load_dataset(hf_path, name=config_name, split="train")
                dataset_loaded = True
            
            if dataset_loaded:
                if num_samples is not None:
                    logger.info(f"Selecting the first {num_samples} samples.")
                    dataset = dataset.select(range(min(num_samples, len(dataset))))
                logger.info(f"Successfully prepared {len(dataset)} samples from {dataset_name} for processing.")
                return dataset

        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")
            if "fineweb" in dataset_name.lower():
                logger.error("Creating a synthetic dataset as a fallback.")
                return Dataset.from_dict({"text": ["Synthetic test document.", "Another synthetic text."]})


        # --- Original fallback logic for OpenWebText and other datasets ---
        logger.info(f"Using fallback logic for OpenWebText/C4/etc...")
        split_map = { "train": "train", "val": "validation", "validation": "validation", "test": "test" }
        hf_split = split_map.get(split, "train")

        datasets_to_try = [
            ("olivercareyncl/openwebtext", hf_split), ("allenai/c4", hf_split),
            ("wikitext", "wikitext-2-raw-v1"), ("pile", hf_split)
        ] if split == "train" else [
            ("wikitext", "wikitext-2-raw-v1"), ("allenai/c4", hf_split), ("olivercareyncl/openwebtext", hf_split)
        ]

        for ds_name, ds_split in datasets_to_try:
            try:
                current_split = hf_split if ds_name != "wikitext" else split
                if current_split == "val": current_split = "validation"
                logger.info(f"Trying dataset: {ds_name} (config: {ds_split}, split: {current_split})")
                
                if ds_name == "olivercareyncl/openwebtext" and self.data_files_limit is not None:
                    files_to_load = [f"split/chunk_{i}.json" for i in range(1, self.data_files_limit + 1)]
                    dataset = load_dataset(ds_name, data_files=files_to_load, split=ds_split)
                elif ds_name in ["wikitext", "pile"]: dataset = load_dataset(ds_name, ds_split, split=current_split)
                elif ds_name == "allenai/c4": dataset = load_dataset(ds_name, "en", split=ds_split)
                else: dataset = load_dataset(ds_name, split=ds_split)

                if num_samples is not None:
                    dataset = dataset.select(range(min(num_samples, len(dataset))))
                logger.info(f"Successfully downloaded {len(dataset)} samples from {ds_name}")
                return dataset
            except Exception as e:
                logger.error(f"Failed to load {ds_name}: {e}")
                continue
        logger.error("All dataset loading attempts failed. Creating synthetic dataset.")
        return Dataset.from_dict({"text": ["Synthetic test document.", "Another synthetic text."]})
    
    def tokenize_function(self, examples, max_length):
        eos_token = self.tokenizer.eos_token
        # Robustness fix: filter out potential None values from the text list
        full_text = eos_token.join(filter(None, examples['text']))
        
        tokens = self.tokenizer(full_text, truncation=False, padding=False, return_attention_mask=False)['input_ids']
        
        result = {'input_ids': []}
        for i in range(0, len(tokens), max_length):
            chunk = tokens[i:i + max_length]
            if len(chunk) == max_length:
                result['input_ids'].append(chunk)
        return result

    def download_and_preprocess(self, dataset_name="openwebtext", split="train", num_samples=None, num_workers=4, max_length=1024):
        """Complete pipeline: download, preprocess, and save dataset using datasets.map for memory efficiency."""
        logger.info(f"Starting pipeline for {dataset_name} ({split})...")
        dataset = self.download_dataset(dataset_name, split, num_samples)
        if dataset is None or len(dataset) == 0:
            logger.error("Dataset could not be loaded. Aborting.")
            return False
        
        # Robustness fix: Ensure text exists and is not just whitespace
        dataset = dataset.filter(lambda example: example.get('text') and len(example.get('text', '').strip()) > 0, num_proc=num_workers)
        logger.info(f"Filtered to {len(dataset)} non-empty documents.")
        
        logger.info(f"Tokenizing and chunking with {num_workers} workers...")
        _tokenize_fn = partial(self.tokenize_function, max_length=max_length)
        
        tokenized_dataset = dataset.map(
            _tokenize_fn,
            batched=True,
            num_proc=num_workers,
            remove_columns=dataset.column_names
        )
        logger.info(f"Tokenization complete. Total examples after chunking: {len(tokenized_dataset)}")
        
        output_file = os.path.join(self.cache_dir, "tok_gpt2", f"{split}{len(tokenized_dataset)}.bin")
        metadata = {'split': split, 'num_examples': len(tokenized_dataset), 'num_tokens': 0}
        
        total_tokens = 0
        with open(output_file, 'wb') as f:
            for example in tqdm(tokenized_dataset, desc=f"Writing to {output_file}"):
                tokens = example['input_ids']
                tokens_array = np.array(tokens, dtype=np.uint16)
                f.write(tokens_array.tobytes())
                total_tokens += len(tokens_array)

        metadata['num_tokens'] = total_tokens
        logger.info(f"Saved {total_tokens:,} tokens to {output_file}")

        metadata_file = os.path.join(self.cache_dir, "tok_gpt2", f"{split}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info("Pipeline completed successfully!")
        return True

def main():
    parser = argparse.ArgumentParser(description="Download and preprocess datasets for LTM.")
    parser.add_argument("--dataset", type=str, default="openwebtext", help="Dataset name, e.g., openwebtext, fineweb-edu")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process from the source dataset")
    parser.add_argument("--cache_dir", type=str, default="data_owt")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--num_workers", type=int, default=max(1, os.cpu_count() - 2))
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--data-files-limit", type=int, default=None, help="Limit the number of source data files to load (for fineweb, fineweb-edu, openwebtext)")
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
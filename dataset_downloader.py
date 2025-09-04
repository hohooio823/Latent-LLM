"""
Dataset downloading and preprocessing script for Latent Thought Language Model.
Uses HuggingFace datasets to download and preprocess OpenWebText and other datasets.
"""

import argparse
import os
import glob
import json
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import logging

from datasets import load_dataset, Dataset
from transformers import GPT2TokenizerFast, AutoTokenizer
import sentencepiece as spm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetDownloader:
    """
    Handles downloading and preprocessing of datasets for LTM training.
    """
    
    def __init__(self, cache_dir="data_owt", tokenizer_name="gpt2"):
        """
        Initialize the dataset downloader.
        
        Args:
            cache_dir (str): Directory to cache dataset files
            tokenizer_name (str): Name of the tokenizer to use
        """
        self.cache_dir = cache_dir
        self.tokenizer_name = tokenizer_name
        self.tokenizer = None
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories for dataset storage."""
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "tok_gpt2"), exist_ok=True)
        
    def download_openwebtext(self, split="train", num_samples=None):
        """
        Download OpenWebText dataset from HuggingFace.
        
        Args:
            split (str): Dataset split to download
            num_samples (int): Number of samples to download (None for all)
            
        Returns:
            Dataset: HuggingFace dataset object
        """
        logger.info(f"Downloading OpenWebText dataset ({split} split)...")
        
        try:
            # Try to load from HuggingFace datasets
            dataset = load_dataset("openwebtext", split=split)
            
            if num_samples is not None:
                dataset = dataset.select(min(num_samples, len(dataset)))
                
            logger.info(f"Downloaded {len(dataset)} samples from OpenWebText")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to download OpenWebText: {e}")
            # Fallback to a smaller dataset for testing
            logger.info("Falling back to smaller dataset for testing...")
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            if num_samples is not None:
                dataset = dataset.select(min(num_samples, len(dataset)))
            return dataset
    
    def download_custom_dataset(self, dataset_name, split="train", num_samples=None):
        """
        Download a custom dataset from HuggingFace.
        
        Args:
            dataset_name (str): Name of the dataset on HuggingFace
            split (str): Dataset split to download
            num_samples (int): Number of samples to download (None for all)
            
        Returns:
            Dataset: HuggingFace dataset object
        """
        logger.info(f"Downloading custom dataset: {dataset_name} ({split} split)...")
        
        try:
            dataset = load_dataset(dataset_name, split=split)
            
            if num_samples is not None:
                dataset = dataset.select(min(num_samples, len(dataset)))
                
            logger.info(f"Downloaded {len(dataset)} samples from {dataset_name}")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
            return None
    
    def load_tokenizer(self):
        """Load the tokenizer for preprocessing."""
        if self.tokenizer is None:
            logger.info(f"Loading tokenizer: {self.tokenizer_name}")
            self.tokenizer = GPT2TokenizerFast.from_pretrained(self.tokenizer_name)
            # Add EOS token as specified in the paper
            if self.tokenizer.eos_token is None:
                self.tokenizer.add_special_tokens({'eos_token': '[EOS]'})
            # Set pad and bos tokens to eos token as per paper's approach
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.bos_token is None:
                self.tokenizer.bos_token = self.tokenizer.eos_token
        return self.tokenizer
    
    def tokenize_text(self, text, max_length=1024):
        """
        Tokenize text using the loaded tokenizer.
        
        Args:
            text (str): Text to tokenize
            max_length (int): Maximum sequence length
            
        Returns:
            dict: Tokenized data
        """
        tokenizer = self.load_tokenizer()
        
        # Tokenize the text
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )
        
        return tokens
    
    def preprocess_dataset(self, dataset, split="train", num_workers=4, max_length=1024):
        """
        Preprocess dataset by tokenizing text following the paper's approach.
        
        Args:
            dataset (Dataset): HuggingFace dataset
            split (str): Dataset split
            num_workers (int): Number of workers for parallel processing
            max_length (int): Maximum sequence length
            
        Returns:
            list: List of tokenized examples
        """
        logger.info(f"Preprocessing {split} split with {num_workers} workers...")
        
        # Filter out empty texts
        def filter_text(example):
            text = example.get('text', '')
            return len(text.strip()) > 0
        
        dataset = dataset.filter(filter_text)
        
        # Concatenate documents and wrap to max_length with EOS tokens between segments
        def concatenate_and_wrap(texts):
            """Concatenate texts and wrap to max_length with EOS tokens."""
            all_tokens = []
            current_tokens = []
            
            for text in texts:
                # Tokenize the text
                tokens = self.tokenize_text(text)['input_ids']
                
                # Add EOS token between documents as per paper
                if current_tokens:
                    tokens_with_eos = tokens + [self.tokenizer.eos_token_id]
                else:
                    tokens_with_eos = tokens
                
                # Check if adding this text would exceed max_length
                if len(current_tokens) + len(tokens_with_eos) <= max_length:
                    current_tokens.extend(tokens_with_eos)
                else:
                    # Save current segment if it has content
                    if current_tokens:
                        all_tokens.append(current_tokens)
                    
                    # Start new segment with current text
                    current_tokens = tokens_with_eos
            
            # Add the last segment
            if current_tokens:
                all_tokens.append(current_tokens)
            
            return all_tokens
        
        # Process documents in batches
        def process_batch(examples, batch_size=1000):
            results = []
            
            # Group texts by document (assuming each example is a document)
            texts = [ex.get('text', '') for ex in examples]
            
            # Concatenate and wrap
            tokenized_segments = concatenate_and_wrap(texts)
            
            # Create attention masks
            for tokens in tokenized_segments:
                attention_mask = [1] * len(tokens)
                results.append({
                    'input_ids': tokens,
                    'attention_mask': attention_mask
                })
            
            return results
        
        # Use parallel processing for large datasets
        if len(dataset) > 10000:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                batch_size = len(dataset) // num_workers + 1
                futures = []
                for i in range(0, len(dataset), batch_size):
                    batch = dataset[i:i + batch_size]
                    futures.append(executor.submit(process_batch, batch))
                
                results = []
                for future in tqdm(futures, desc="Processing batches"):
                    results.extend(future.result())
        else:
            results = process_batch(list(dataset))
        
        logger.info(f"Preprocessed {len(results)} examples")
        return results
    
    def save_tokenized_data(self, tokenized_data, split="train"):
        """
        Save tokenized data to binary files.
        
        Args:
            tokenized_data (list): List of tokenized examples
            split (str): Dataset split
        """
        logger.info(f"Saving tokenized data for {split} split...")
        
        # Combine all tokens into a single array
        all_tokens = []
        for example in tokenized_data:
            all_tokens.extend(example['input_ids'])
        
        # Convert to numpy array
        tokens_array = np.array(all_tokens, dtype=np.uint16)
        
        # Save to binary file
        output_file = os.path.join(self.cache_dir, "tok_gpt2", f"{split}{len(tokenized_data)}.bin")
        tokens_array.tofile(output_file)
        
        logger.info(f"Saved {len(tokens_array)} tokens to {output_file}")
        
        # Save metadata
        metadata = {
            'split': split,
            'num_examples': len(tokenized_data),
            'num_tokens': len(tokens_array),
            'file_path': output_file
        }
        
        metadata_file = os.path.join(self.cache_dir, "tok_gpt2", f"{split}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_file}")
    
    def download_and_preprocess(self, dataset_name="openwebtext", split="train",
                              num_samples=None, num_workers=4, max_length=1024):
        """
        Complete pipeline: download, preprocess, and save dataset.
        
        Args:
            dataset_name (str): Name of the dataset
            split (str): Dataset split
            num_samples (int): Number of samples to download
            num_workers (int): Number of workers for parallel processing
            max_length (int): Maximum sequence length for tokenization
        """
        logger.info(f"Starting dataset processing pipeline for {dataset_name} ({split} split)...")
        
        # Download dataset
        if dataset_name == "openwebtext":
            dataset = self.download_openwebtext(split, num_samples)
        else:
            dataset = self.download_custom_dataset(dataset_name, split, num_samples)
        
        if dataset is None:
            logger.error("Failed to download dataset")
            return False
        
        # Preprocess dataset
        tokenized_data = self.preprocess_dataset(dataset, split, num_workers, max_length)
        
        # Save tokenized data
        self.save_tokenized_data(tokenized_data, split)
        
        logger.info("Dataset processing pipeline completed successfully!")
        return True
    
    def verify_dataset(self, split="train"):
        """
        Verify that the dataset files exist and are valid.
        
        Args:
            split (str): Dataset split to verify
            
        Returns:
            bool: True if dataset is valid, False otherwise
        """
        logger.info(f"Verifying {split} dataset...")
        
        # Check for binary files
        bin_dir = os.path.join(self.cache_dir, "tok_gpt2")
        bin_files = glob.glob(os.path.join(bin_dir, f"{split}*.bin"))
        
        if not bin_files:
            logger.error(f"No binary files found for {split} split")
            return False
        
        # Check for metadata
        metadata_files = glob.glob(os.path.join(bin_dir, f"{split}_metadata.json"))
        
        if not metadata_files:
            logger.error(f"No metadata files found for {split} split")
            return False
        
        # Try to load a small portion of the data
        try:
            bin_file = bin_files[0]
            metadata_file = metadata_files[0]
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Test loading a small portion
            tokens = np.memmap(bin_file, dtype=np.uint16, mode="r")
            if len(tokens) == 0:
                logger.error("Binary file is empty")
                return False
            
            logger.info(f"Dataset verification successful: {metadata['num_examples']} examples, {metadata['num_tokens']} tokens")
            return True
            
        except Exception as e:
            logger.error(f"Dataset verification failed: {e}")
            return False

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Download and preprocess datasets for LTM training")
    parser.add_argument("--dataset", type=str, default="openwebtext", 
                       help="Dataset name (openwebtext, custom, etc.)")
    parser.add_argument("--split", type=str, default="train", 
                       help="Dataset split (train, val, test)")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to download (None for all)")
    parser.add_argument("--cache_dir", type=str, default="data_owt",
                       help="Directory to cache dataset files")
    parser.add_argument("--tokenizer", type=str, default="gpt2",
                       help="Tokenizer to use for preprocessing")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of workers for parallel processing")
    parser.add_argument("--max_length", type=int, default=1024,
                       help="Maximum sequence length for tokenization")
    parser.add_argument("--verify", action="store_true",
                       help="Verify dataset after processing")
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = DatasetDownloader(
        cache_dir=args.cache_dir,
        tokenizer_name=args.tokenizer
    )
    
    # Download and preprocess dataset
    success = downloader.download_and_preprocess(
        dataset_name=args.dataset,
        split=args.split,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
        max_length=args.max_length
    )
    
    if success and args.verify:
        downloader.verify_dataset(split=args.split)

if __name__ == "__main__":
    main()
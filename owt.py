"""
Download, preprocess and serve the OWT dataset as a DataLoader with latent variables.
"""

import argparse
import glob
import json
import os
import random
import re
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import requests
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm

from datasets import load_dataset
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from dataset_downloader import DatasetDownloader

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Base directory for caching dataset files
from config import DATA_CACHE_DIR  # use config path

# RNG seed parameters
DEFAULT_SEED = 42
RANK_SEED_OFFSET = 1337

# Latent variable initialization scale
LATENT_INIT_SCALE = 0.01

# -----------------------------------------------------------------------------
# Dataset Implementation
# -----------------------------------------------------------------------------

class PretokDatasetWithLatent(torch.utils.data.Dataset):
    """
    Loads pretokenized examples from disk and serves them as PyTorch tensors with latent variables.
    This class uses a regular Dataset interface for better multi-worker support.
    """

    def __init__(self, split, max_seq_len, max_z_len, z_dim, num_workers=0):
        """
        Initialize the dataset.
        
        Args:
            split (str): Dataset split to use ('train' or 'val')
            max_seq_len (int): Maximum sequence length for tokens
            max_z_len (int): Maximum length of latent vectors
            z_dim (int): Dimension of latent vectors
            num_workers (int): Number of DataLoader workers (for worker-specific setup)
        """
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.max_z_len = max_z_len
        self.z_dim = z_dim
        self.num_workers = num_workers
        
        # Find all available data shards
        bin_dir = os.path.join(DATA_CACHE_DIR, "tok_gpt2")
        shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))

        # Filter shards by split type
        if self.split == 'train':
            shard_filenames = [f for f in shard_filenames if 'train' in f]
        else:
            shard_filenames = [f for f in shard_filenames if 'val' in f]
            
        assert len(shard_filenames) > 0, f"No bin files found in {bin_dir}"
        
        # Build index of all examples across all shards
        self.examples = []
        self.shard_info = []
        
        print(f"Building dataset index for {split} split...")
        for shard in shard_filenames:
            # Load the shard data using memory mapping
            m = np.memmap(shard, dtype=np.uint16, mode="r")
            
            # Calculate number of complete batches in this shard
            num_batches = len(m) // self.max_seq_len
            num_batches -= 1  # drop the last partial batch
            assert num_batches > 0, "This shard is way too small. Please investigate."
            
            # Store shard information
            self.shard_info.append({
                'filename': shard,
                'num_batches': num_batches,
                'offset': len(self.examples)
            })
            
            # Add examples to index
            self.examples.extend([(shard, i) for i in range(num_batches)])
            
        print(f"Dataset index built with {len(self.examples)} examples")
        
        # Store latent variable parameters instead of generating all at once
        # This saves significant memory during dataset initialization
        self.latent_seed = DEFAULT_SEED + len(self.examples)
        self.latent_vars = None  # Generate on-demand

    def __len__(self):
        """Return the total number of examples in the dataset."""
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Get a single example by index.
        
        Args:
            idx (int): Index of the example to retrieve
            
        Returns:
            tuple: (x, y, z) where:
                - x is the input token sequence
                - y is the target token sequence (shifted by 1)
                - z is the latent variable vector
        """
        shard_info, batch_idx = self.examples[idx]
        
        # Load the shard data using memory mapping
        m = np.memmap(shard_info, dtype=np.uint16, mode="r")
        
        # Extract token sequence
        start = batch_idx * self.max_seq_len
        end = start + self.max_seq_len + 1
        
        # Convert to PyTorch tensor and move to appropriate data type
        chunk = torch.from_numpy((m[start:end]).astype(np.int64))
        
        # Create input and target sequences (shifted by one token)
        x = chunk[:-1]  # Input tokens
        y = chunk[1:]   # Target tokens (next token prediction)
        
        # Generate latent vector for this batch truly on-demand to save memory
        rng = np.random.RandomState(self.latent_seed + idx)
        z = rng.randn(self.z_dim * self.max_z_len).astype(np.float32) * LATENT_INIT_SCALE
        
        return x, y, z
    
    @staticmethod
    def get_shard_id(shard_file_name):
        # Extract shard number from filename (e.g., "train26.bin" → 26)
        match = re.search(r'(train|val)(\d+)\.bin', shard_file_name)
        if match:
            return int(match.group(2))
        return None


# -----------------------------------------------------------------------------
# Public Interface
# -----------------------------------------------------------------------------


class Task:
    """Task interface for working with the OWT dataset."""
    @staticmethod
    def ensure_dataset_available(split="train", dataset_name="openwebtext", 
                                cache_dir="data_owt", auto_download=True):
        """
        Ensure that the dataset is available for the given split.
        
        Args:
            split (str): Dataset split ('train' or 'val')
            dataset_name (str): Name of the dataset
            cache_dir (str): Directory where dataset is cached
            auto_download (bool): Whether to automatically download if missing
            
        Returns:
            bool: True if dataset is available, False otherwise
        """
        bin_dir = os.path.join(cache_dir, "tok_gpt2")
        bin_files = glob.glob(os.path.join(bin_dir, f"{split}*.bin"))
        
        if bin_files:
            return True
        
        if not auto_download:
            print(f"Dataset files not found for {split} split and auto_download=False")
            return False
        
        print(f"Dataset files not found for {split} split. Downloading...")
        
        try:
            # Initialize downloader
            downloader = DatasetDownloader(cache_dir=cache_dir)
            
            # Download and preprocess dataset
            success = downloader.download_and_preprocess(
                dataset_name=dataset_name,
                split=split,
                num_samples=None,  # Download all samples
                num_workers=4
            )
            
            if success:
                print(f"Successfully downloaded {split} dataset")
                return True
            else:
                print(f"Failed to download {split} dataset")
                return False
                
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False

    @staticmethod
    def iter_batches_with_latents(split, batch_size, max_seq_len, max_z_len, z_dim,
                                  device, num_workers=0, auto_download=True):
        """
        Create an iterable over batches of data with latent variables.
        
        Args:
            split (str): Dataset split ('train' or 'val')
            batch_size (int): Batch size
            max_seq_len (int): Maximum sequence length
            max_z_len (int): Maximum latent sequence length
            z_dim (int): Dimension of latent variables
            device (str): Device to load tensors to
            num_workers (int): Number of DataLoader workers
            auto_download (bool): Whether to auto-download dataset
            
        Yields:
            tuple: (x, y, z) containing batches of:
                - x: Input token tensors [batch_size, max_seq_len]
                - y: Target token tensors [batch_size, max_seq_len]
                - z: Latent variable tensors [batch_size, z_dim * max_z_len]
        """
        # Ensure dataset is available
        from config import DATA_CACHE_DIR
        if not Task.ensure_dataset_available(
            split=split,
            cache_dir=DATA_CACHE_DIR,
            auto_download=auto_download
        ):
            raise FileNotFoundError(f"Dataset files not found for {split} split and auto_download=False")
        
        # Create dataset instance
        ds = PretokDatasetWithLatent(
            split=split,
            max_seq_len=max_seq_len,
            max_z_len=max_z_len,
            z_dim=z_dim,
            num_workers=num_workers
        )
        
        # Create DataLoader with shuffle for training, no shuffle for validation
        shuffle = (split == "train")
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=True  # Drop last incomplete batch
        )
        
        # Process and yield batches
        for x, y, z in dl:
            # Move tensors to requested device
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z = z.to(device, non_blocking=True)
            
            # Yield the batch
            yield x, y, z

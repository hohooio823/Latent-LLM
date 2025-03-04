"""
Download, preprocess and serve the owt dataset as a DataLoader.
"""

import argparse
import glob
import json
import os
import random
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import requests
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm
import re

from datasets import load_dataset

DATA_CACHE_DIR = "/shared/jianwen/random_effect_LLM/data_owt"
    
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

def process_shard(args):
    shard_id, shard = args
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    special_tokens = {'bos_token': '<|beginoftext|>'}
    tokenizer.add_special_tokens(special_tokens)
    bos_token = tokenizer.bos_token 
    eos_token = tokenizer.eos_token
    with open(shard, "r") as f:
        data = json.load(f)
        
    all_tokens = []
    for example in tqdm(data, position=shard_id):
        text = example["text"]
        text = text.strip()  # get rid of leading/trailing whitespace

        input_text = f"{bos_token}{text}{eos_token}".strip()
        tokens = tokenizer.encode(input_text, add_special_tokens=False)
        
        all_tokens.extend(tokens)
        
    # convert to uint16 nparray
    all_tokens = np.array(all_tokens, dtype=np.uint16)

    # save .bin files into a new tok{N} directory
    bin_dir = os.path.join(DATA_CACHE_DIR, f"tok_gpt2")
    shard_basename = os.path.basename(shard)
    bin_basename = shard_basename.replace(".json", ".bin")
    tokenized_filename = os.path.join(bin_dir, bin_basename)
    # write the bytes
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())
    bos_token_count = (all_tokens == tokenizer.bos_token_id).sum()
    if bos_token_count > 0:
        avg_seq_len = all_tokens.size / bos_token_count
    else:
        avg_seq_len = 0  # Or some other fallback value
        print(f"Warning: No BOS tokens found in {tokenized_filename}")
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")


def pretokenize():
    # iterate the shards and tokenize all of them one by one
    data_dir = os.path.join(DATA_CACHE_DIR, "owt_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    print(shard_filenames)
    
    bin_dir = os.path.join(DATA_CACHE_DIR, f"tok_gpt2")
    os.makedirs(bin_dir, exist_ok=True)

    # process all the shards in a process pool
    fun = partial(process_shard)
    with ProcessPoolExecutor() as executor:
        executor.map(fun, enumerate(shard_filenames))
    print("Done.")

# vanilla owt dataset
class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""
    def __init__(self, split, max_seq_len):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        # orignial 42
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        current_shard = None
        print(f"Created a PretokDatasetWithLatent with rng seed {seed}")

        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok_gpt2")
        shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))

        # train/test split. let's use only shard 0 for test split, rest train
        # shard_filenames = shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        if self.split == 'train':
            shard_filenames = [f for f in shard_filenames if 'train' in f]
        else:
            shard_filenames = [f for f in shard_filenames if 'val' in f]
        assert len(shard_filenames)>0, f"No bin files found in {bin_dir}"
        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                if shard != current_shard:
                    current_shard = shard
                    print(f"Switching to new shard: {shard}")  # Notification of shard change

                # open the dataset for reading but keep it on disk with memmap
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1  # drop the last partial batch
                assert num_batches > 0, "this shard is way too small? investigate."

                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    # calling .astype will copy the data into a new numpy array, now in RAM
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    # yield x, y, z, ix, self.get_shard_id(shard)
                    yield x, y

# modified owt dataset to enable inputs with latents
class PretokDatasetWithLatent(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, max_z_len, z_dim, z_dir=None):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.max_z_len = max_z_len
        self.z_dim = z_dim
        self.z_dir = z_dir if z_dir else DATA_CACHE_DIR

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        # orignial 42
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        current_shard = None
        print(f"Created a PretokDatasetWithLatent with rng seed {seed}")

        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok_gpt2")
        shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))

        # train/test split. let's use only shard 0 for test split, rest train
        # shard_filenames = shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        if self.split == 'train':
            shard_filenames = [f for f in shard_filenames if 'train' in f]
        else:
            shard_filenames = [f for f in shard_filenames if 'val' in f]
        assert len(shard_filenames)>0, f"No bin files found in {bin_dir}"
        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                if shard != current_shard:
                    current_shard = shard
                    print(f"Switching to new shard: {shard}")  # Notification of shard change

                # open the dataset for reading but keep it on disk with memmap
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1  # drop the last partial batch
                assert num_batches > 0, "this shard is way too small? investigate."

                # deal with z shards
                z_file_name = os.path.basename(shard).replace('.bin', '_z.npy')
                z_file = os.path.join(self.z_dir, z_file_name)

                if self.split == 'val':
                    if not os.path.exists(z_file):
                        # z_matrix = np.memmap(z_file, dtype=np.float32, mode='w+', shape=(num_batches, self.z_dim * self.max_z_len))
                        z_matrix = np.random.randn(num_batches, self.z_dim * self.max_z_len).astype(np.float32) * 0.01
                        np.save(z_file, z_matrix)
                    else:
                        # z_matrix = np.memmap(z_file, dtype=np.float32, mode='r+', shape=(num_batches, self.z_dim * self.max_z_len))
                        z_matrix = np.load(z_file)
                else:
                    z_matrix = np.random.randn(num_batches, self.z_dim * self.max_z_len).astype(np.float32) * 0.01

                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    # calling .astype will copy the data into a new numpy array, now in RAM
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    z = z_matrix[ix]
                    # yield x, y, z, ix, self.get_shard_id(shard)
                    yield x, y, z
    
    @staticmethod
    def get_shard_id(shard_file_name):
        # Assuming the shard file name follows the format "data/tok4096/train26.bin"
        match = re.search(r'(train|val)(\d+)\.bin', shard_file_name)
        if match:
            return int(match.group(2))  # Convert the matched string to an integer
        return None  
# -----------------------------------------------------------------------------
# public interface functions

class Task:
    @staticmethod
    def iter_batches_with_latents(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDatasetWithLatent(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        # for x, y, z, ix, shard_id in dl:
        for x, y, z in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z = z.to(device, non_blocking=True)
            # yield x, y, z, ix, shard_id
            yield x, y, z

    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["pretokenize"])
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    if args.stage == "pretokenize":
        pretokenize()
    else:
        raise ValueError(f"Unknown stage {args.stage}")
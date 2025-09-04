#!/usr/bin/env python3
"""
Command-line script to download and preprocess datasets for Latent Thought Language Model.
"""

import argparse
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_downloader import DatasetDownloader
from config import DATA_CACHE_DIR, DATASET_NAME, DATASET_TOKENIZER, DATASET_NUM_WORKERS

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Download and preprocess datasets for LTM training")
    parser.add_argument("--dataset", type=str, default=DATASET_NAME,
                       help=f"Dataset name (default: {DATASET_NAME})")
    parser.add_argument("--split", type=str, default="train", 
                       help="Dataset split to download (train, val, test)")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to download (None for all)")
    parser.add_argument("--cache_dir", type=str, default=DATA_CACHE_DIR,
                       help=f"Directory to cache dataset files (default: {DATA_CACHE_DIR})")
    parser.add_argument("--tokenizer", type=str, default=DATASET_TOKENIZER,
                       help=f"Tokenizer to use for preprocessing (default: {DATASET_TOKENIZER})")
    parser.add_argument("--num_workers", type=int, default=DATASET_NUM_WORKERS,
                       help=f"Number of workers for parallel processing (default: {DATASET_NUM_WORKERS})")
    parser.add_argument("--verify", action="store_true",
                       help="Verify dataset after processing")
    parser.add_argument("--list_datasets", action="store_true",
                       help="List available datasets and exit")
    
    args = parser.parse_args()
    
    if args.list_datasets:
        print("Available datasets:")
        print("  - openwebtext: OpenWebText dataset")
        print("  - wikitext: WikiText dataset (good for testing)")
        print("  - c4: C4 dataset")
        print("  - pile: The Pile dataset")
        print("  - custom: Any dataset available on HuggingFace datasets hub")
        return
    
    print(f"Downloading dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Number of workers: {args.num_workers}")
    
    if args.num_samples:
        print(f"Number of samples: {args.num_samples}")
    
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
        num_workers=args.num_workers
    )
    
    if success:
        print("✅ Dataset downloaded and preprocessed successfully!")
        
        if args.verify:
            print("Verifying dataset...")
            if downloader.verify_dataset(split=args.split):
                print("✅ Dataset verification passed!")
            else:
                print("❌ Dataset verification failed!")
                sys.exit(1)
        
        print(f"Dataset is ready for training at: {args.cache_dir}/tok_gpt2/")
        
    else:
        print("❌ Failed to download or preprocess dataset")
        sys.exit(1)

if __name__ == "__main__":
    main()
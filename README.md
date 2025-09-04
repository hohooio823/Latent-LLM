# Latent Thought Language Model (LTM)

## Installation

```bash
git clone [address]
cd Latent-Thought-LM
conda env create -f env.yml
conda activate ltm
```

## Dataset Download

Before training, you need to download and preprocess the dataset:

```bash
# Download OpenWebText dataset (recommended)
python download_dataset.py

# Download a specific dataset
python download_dataset.py --dataset wikitext --split train

# Download with custom parameters
python download_dataset.py --dataset openwebtext --split train --num_samples 10000 --num_workers 8

# List available datasets
python download_dataset.py --list_datasets
```

## Training

```bash
python train_ltm.py
```

The training script will automatically check for the dataset and download it if `DATASET_AUTO_DOWNLOAD=True` in config.py.

## Model checkpoints

We will release the trained model checkpoints after the paper is accepted.


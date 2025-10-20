# In utils/dataset_utils/smos_dataset.py
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import os
import pytorch_lightning as pl
from utils import dist_util # Import dist_util
import numpy as np # Import numpy

TARGET_SIZE = 80 # Padded size

class SMOSDataset(Dataset):
    def __init__(self, data_path_corrupted, data_path_original, target_size=TARGET_SIZE):
        super().__init__()
        # print(f"Loading corrupted data from: {data_path_corrupted}")
        # self.data_corrupted = torch.load(data_path_corrupted)
        # print(f"Loading original data from: {data_path_original}")
        # self.data_original = torch.load(data_path_original)
        data_path_corrupted = "./data/D_tensor_all.pt"
        data_path_original = "./data/D_original_tensor_all.pt"
        print(f"Loading corrupted data from: {data_path_corrupted}")
        self.data_corrupted = torch.load(data_path_corrupted) # Expect shape [N, 2, 69, 69]
        print(f"Loading original data from: {data_path_original}")
        self.data_original = torch.load(data_path_original) # Expect shape [N, 2, 69, 69]
        self.target_size = target_size
        if self.data_corrupted.shape[0] != self.data_original.shape[0]:
            raise ValueError("Corrupted and original datasets have different number of samples!")

        print(f"Dataset loaded. Corrupted shape: {self.data_corrupted.shape}, Original shape: {self.data_original.shape}")
        print(f"Padding images to {target_size}x{target_size}")


    def __len__(self):
        return self.data_corrupted.shape[0]

    def _pad_tensor(self, tensor):
        # Pad 69x69 to target_size x target_size (e.g., 80x80)
        _, _, h, w = tensor.shape
        pad_h = self.target_size - h
        pad_w = self.target_size - w

        # Calculate padding amounts for top, bottom, left, right
        # We'll use asymmetric padding if needed, putting more padding on bottom/right
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Pad the tensor (only H and W dimensions)
        padded_tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        return padded_tensor

    def __getitem__(self, idx):
        corrupted_tensor = self.data_corrupted[idx] # Shape [2, 69, 69]
        original_tensor = self.data_original[idx]   # Shape [2, 69, 69]

        # Pad tensors
        padded_corrupted = self._pad_tensor(corrupted_tensor.unsqueeze(0)).squeeze(0) # Shape [2, 80, 80]
        padded_original = self._pad_tensor(original_tensor.unsqueeze(0)).squeeze(0) # Shape [2, 80, 80]

        # Note: Normalization might be needed depending on your data distribution.
        # If your data isn't roughly in [-1, 1], add normalization here.
        # Example (simple scaling if data is positive):
        # max_val = torch.max(torch.abs(padded_original)) # Or a fixed reasonable max value
        # if max_val > 0:
        #     padded_original = padded_original / max_val
        #     padded_corrupted = padded_corrupted / max_val # Use same scaling

        return {
            "target": padded_original.float(),    # D_original (Ground Truth)
            "condition": padded_corrupted.float() # D (Corrupted Input)
        }

def load_smos_data(
        data_path_corrupted, # Path to D_tensor_all.pt
        data_path_original,  # Path to D_original_tensor_all.pt
        batch_size,
        val_split_ratio=0.1, # Ratio for validation split
        is_distributed=False,
        is_train=False,
        num_workers=0,
        seed=42
):
    pl.seed_everything(seed)
    dataset = SMOSDataset(data_path_corrupted, data_path_original)

    # Split dataset into training and validation
    total_samples = len(dataset)
    val_size = int(total_samples * val_split_ratio)
    train_size = total_samples - val_size
    split_train, split_val = random_split(dataset, [train_size, val_size])

    target_split = split_train if is_train else split_val
    print(f"Using {'train' if is_train else 'validation'} split with {len(target_split)} samples.")

    data_sampler = None
    if is_distributed and is_train:
        data_sampler = DistributedSampler(target_split)

    loader = DataLoader(
        target_split,
        batch_size=batch_size,
        shuffle=(data_sampler is None) and is_train,
        sampler=data_sampler,
        num_workers=num_workers,
        drop_last=is_train,
        pin_memory=True,
    )
    # print a data from loader for verification
    for batch in loader:
        print(f"Sample batch - target shape: {batch['target'].shape}, condition shape: {batch['condition'].shape}")
        break
    if is_train:
        while True:
            yield from loader
    else:
         # For validation/testing, just iterate once
         yield from loader # Or simply return loader if not using as generator
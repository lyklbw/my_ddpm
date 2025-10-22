import torch

# load a checkpoint file and print every key's shape
def load_checkpoint_shapes(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    for key in checkpoint:
        print(f"Key: {key}, Shape: {checkpoint[key].shape}")

load_checkpoint_shapes("smos_model/ema_0.9999_010000.pt")
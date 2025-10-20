# read a local pt file
import torch

file_path = "./data/D_tensor_all.pt"  # Replace with your file path
data_tensor = torch.load(file_path)

 # Set a breakpoint here to inspect data_tensor
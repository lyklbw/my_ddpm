from models.guided_ddpm_unet import UNetModel
from utils.galaxy_data_utils.transform_util import *
import torch
import numpy as np

def posenc(x, L_embed=4):
  rets = [x]
  for i in range(0, L_embed):
    for fn in [torch.sin, torch.cos]:
      rets.append(fn(2.*3.14159265*(i+1) * x))
  return torch.cat(rets, dim=-1)
  
def calcB(m=1024, d=2, sigma=1.0):
    B = torch.randn(m, d)*sigma
    return B.cuda()
    
def fourierfeat_enc(x, B):
    feat = torch.cat([#torch.sum(x**2, -1, keepdims=True), ## new
                      x, ## new
                      torch.cos(2*3.14159265*(x @ B.T)),
                      torch.sin(2*3.14159265*(x @ B.T))], -1)
    return feat

class PE_Module(torch.nn.Module):
    def __init__(self, type, embed_L):
        super(PE_Module, self).__init__()

        self.embed_L= embed_L
        self.type=type

    def forward(self, x):
        if self.type == 'posenc':
            return posenc(x, L_embed=self.embed_L)

        elif self.type== 'fourier':
            return fourierfeat_enc(x, B=self.embed_L)

class VicUnetModel(UNetModel):
    def __init__(self, image_size, in_channels, *args, **kwargs):
        assert in_channels == 2, "mri image is considered"
        # we use in_channels * 2 because image_dir is also input.
        super().__init__(image_size, in_channels * 2, *args, **kwargs)
        # self.uv_dense = np.load("./data/uv_dense.npy")
        # self.uv_dense = torch.tensor(self.uv_dense)
        # self.B = torch.nn.Parameter(calcB(m=10, d=2, sigma=5.0), requires_grad=False)
        # self.pe_encoder = PE_Module(type='fourier', embed_L= self.B)


    # Inside class VicUnetModel(UNetModel):
    def forward(self, x, timesteps, **kwargs):
        """
        :param x: the [N x 2 x H x W] tensor of noisy inputs, x_t at time t (target D_original + noise).
        :param timesteps: a batch of timestep indices.
        :param kwargs: Should contain 'condition_input': the [N x 2 x H x W] tensor (condition D).
        :return: noise estimation [N x 2 x H x W]
        """
        if 'condition_input' not in kwargs:
            raise ValueError("Missing 'condition_input' in model keyword arguments")

        c = kwargs['condition_input']

        # Ensure shapes match
        assert x.shape == c.shape, f"Shape mismatch: x {x.shape}, c {c.shape}"
        assert x.shape[1] == 2, f"Input x should have 2 channels, got {x.shape[1]}"

        # Concatenate noisy target and condition along the channel dimension
        conditioned_x = th.cat([x, c], dim=1) # Shape: [N, 4, H, W]
        
        # The original code passed 'visibility' derived from uv_coords etc. to super().forward
        # This implies the base UNetModel expects some form of context injection beyond timestep embedding.
        # Let's check guided_ddpm_unet.py UNetModel forward signature again.
        # It's forward(self, x, timesteps, y=None).
        # The VICDDPM authors likely used 'y' to pass their 'visibility' context.
        # Since our condition 'c' is now part of the input channels ('conditioned_x'),
        # we *might* not need to pass anything extra as 'y'. Let's try passing None.
        # If training fails or performance is poor, we might need to investigate how
        # context 'y' is used in the base UNetModel's ResBlocks and potentially pass
        # some processed version of 'c' or even 'x' itself as 'y'.

        # Simple approach: Pass None for context 'y'
        output = super().forward(conditioned_x, timesteps, y=None)

        # Check output shape
        assert output.shape == x.shape, f"Output shape mismatch: output {output.shape}, expected {x.shape}"

        return output
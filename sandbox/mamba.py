import os
import sys

# This line needs to be added since some terminals will not recognize the current directory
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
os.chdir("/home/wentinn/workspace/KF_RNN")

import torch
from transformers import Mamba2Config, Mamba2Model, GPT2Config, GPT2Model
from mamba_ssm import Mamba, Mamba2

from infrastructure import utils


if __name__ == "__main__":
    batch, length = 2, 64

    d_embed = 256
    n_layer = 12
    n_head = 8
    mlp_ratio = 4
    d_inner = mlp_ratio * d_embed

    mamba2 = Mamba2Model(Mamba2Config(
        state_size=64,
        hidden_size=d_embed,
        num_hidden_layers=n_layer,
        num_heads=n_head,
        head_dim=int(2 * d_embed / n_head),
    )).to("cuda")

    gpt2 = GPT2Model(GPT2Config(
        n_positions=250,
        n_embd=d_embed,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=d_inner,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, use_cache=False,
    ))
    print(utils.model_size(mamba2), utils.model_size(gpt2))

    x = torch.randn((batch, length, d_embed)).to("cuda")

    print(gpt2.forward(inputs_embeds=x))
    print(mamba2.forward(inputs_embeds=x))
    # print(mamba.forward(inputs_embeds=x))

    # model = Mamba2(
    #     # This module uses roughly 3 * expand * d_model^2 parameters
    #     d_model=d_embed, # Model dimension d_model
    #     d_state=64,  # SSM state expansion factor
    #     d_conv=4,    # Local convolution width
    #     expand=2,    # Block expansion factor
    # ).to("cuda")
    # y = model(x)
    # print(x.shape, y.shape)





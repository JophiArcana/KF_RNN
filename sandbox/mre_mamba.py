import os
import sys

# This line needs to be added since some terminals will not recognize the current directory
os.chdir("/home/wentinn/workspace/KF_RNN/")
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import torch
from transformers import MambaModel, MambaConfig, Mamba2Config, Mamba2Model

from infrastructure import utils
from model.transformer import MultiMamba2Config, MultiMamba2Model


if __name__ == "__main__":
    torch.manual_seed(1212)
    x = torch.randn((1, 13, 256), device="cuda:0")

    # torch.manual_seed(1212)
    # c = Mamba2Config(hidden_size=256, num_hidden_layers=1, num_heads=8, head_dim=32, expand=1)
    # m = Mamba2Model(c).to("cuda:0").eval()

    # out = m.forward(inputs_embeds=x[:, :-1, :], use_cache=True)
    # _out = m.forward(inputs_embeds=x[:, -1:, :], use_cache=True, cache_params=out.cache_params, cache_position=torch.randn((4,)))


    torch.manual_seed(1212)
    c2 = MultiMamba2Config(hidden_size=256, num_hidden_layers=1, num_heads=8, head_dim=32, expand=1)
    m2 = MultiMamba2Model(c2).to("cuda:0").eval()

    out2 = m2.forward(inputs_embeds=x[:, :-1, :], use_cache=True)
    _out2 = m2.forward(inputs_embeds=x[:, -1:, :], use_cache=True, cache_params=out2.cache_params, cache_position=torch.randn((4,)))

    (out2.last_hidden_state.norm() ** 2).backward()
    # g = torch.autograd.grad(out2.last_hidden_state.norm() ** 2, m2.parameters(), allow_unused=True)
    # print(g)

    # print("Synchronous")
    # print(out.last_hidden_state == out2.last_hidden_state)

    # print("Inference")
    # print(_out.last_hidden_state == _out2.last_hidden_state)


    # print({k: v.shape for k, v in out.cache_params.conv_states.items()})
    # print({k: v.shape for k, v in out.cache_params.ssm_states.items()})

    # out2 = m.forward(inputs_embeds=x[:, :1, :], use_cache=True, cache_params=out.cache_params, cache_position=torch.arange(4))
    # states2 = out2.cache_params.ssm_states


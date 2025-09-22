import os
import sys

# This line needs to be added since some terminals will not recognize the current directory
os.chdir("/home/wentinn/Desktop/KF_RNN/")
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import torch
from model.transformer import Mamba2Config as _Mamba2Config, Mamba2Model as _Mamba2Model
from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer, Mamba2Config, Mamba2Model

from infrastructure import utils
from infrastructure.settings import *
from model.transformer.multi_mamba.modeling_multimamba2 import MultiMamba2Mixer, MultiMamba2Config


if __name__ == "__main__":
    SEED = 1212
    torch.manual_seed(SEED)
    x = torch.randn((5, 13, 256), device="cuda:0")
    kwargs = dict(hidden_size=256, num_hidden_layers=1, num_heads=8, head_dim=32, expand=1, conv_kernel=4)

    torch.manual_seed(SEED)
    c = Mamba2Config(**kwargs)
    m = Mamba2Model(config=c).to("cuda:0").eval()
    # m = Mamba2Mixer(c, 0).to("cuda:0").eval()

    out = m.forward(inputs_embeds=x[..., :-1, :], use_cache=True)
    _out = m.forward(inputs_embeds=x[..., -1:, :], cache_params=out.cache_params, use_cache=True, cache_position=torch.LongTensor([4]))
    # out = m.forward(inputs_embeds=x[:, :-1, :], use_cache=True)
    # _out = m.forward(inputs_embeds=x[:, -1:, :], use_cache=True, cache_params=out.cache_params, cache_position=torch.randn((4,)))
    # grad = torch.autograd.grad(out.norm() ** 2, m.parameters(), allow_unused=True)

    torch.manual_seed(SEED)
    c2 = _Mamba2Config(**kwargs)
    m2 = _Mamba2Model(config=c).to("cuda:0").eval()
    # c2 = MultiMamba2Config(hidden_size=256, num_hidden_layers=1, num_heads=8, head_dim=32, expand=1)
    # m2 = MultiMamba2Mixer(c2, 0).to("cuda:0").eval()

    out2 = m2.forward(inputs_embeds=x[..., :-1, :], use_cache=True)
    _out2 = m2.forward(inputs_embeds=x[..., -1:, :], cache_params=out2.cache_params, use_cache=True, cache_position=torch.LongTensor([4]))
    # out2 = m2.forward(inputs_embeds=x[:, :-1, :], use_cache=True)
    # _out2 = m2.forward(inputs_embeds=x[:, -1:, :], use_cache=True, cache_params=out2.cache_params, cache_position=torch.randn((4,)))
    # grad2 = torch.autograd.grad(out2.norm() ** 2, m2.parameters(), allow_unused=True)
    # print(sum(v.norm() ** 2 for v in g if v is not None))

    print("Synchronous")
    print(torch.abs(out.last_hidden_state - out2.last_hidden_state).max())
    print(torch.abs(_out.last_hidden_state - _out2.last_hidden_state).max())

    # print("Inference")
    # print(torch.abs(_out.last_hidden_state - _out2.last_hidden_state).max())

    # print("Grad")
    # print(sum(v.norm() ** 2 for v in grad if v is not None))
    # print(sum(v2.norm() ** 2 for v2 in grad2 if v2 is not None))
    # print([(v - v2).norm() ** 2 for (v, v2) in zip(grad, grad2) if v is not None])


    # print({k: v.shape for k, v in out.cache_params.conv_states.items()})
    # print({k: v.shape for k, v in out.cache_params.ssm_states.items()})

    # out2 = m.forward(inputs_embeds=x[:, :1, :], use_cache=True, cache_params=out.cache_params, cache_position=torch.arange(4))
    # states2 = out2.cache_params.ssm_states


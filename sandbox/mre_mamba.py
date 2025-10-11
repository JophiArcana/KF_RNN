import time

import torch
from model.transformer import AdaSyncSSMConfig, AdaSyncSSMModel
from model.transformer import ObservableMambaConfig, ObservableMambaModel
from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer, Mamba2Config, Mamba2Model

from infrastructure import utils
from infrastructure.settings import *
from model.transformer.multi_mamba.modeling_multimamba2 import MultiMamba2Mixer, MultiMamba2Config


if __name__ == "__main__":
    SEED = 1212 # torch.randint(0, 1000000, ()).item()
    torch.manual_seed(SEED)
    x = torch.randn((1, 10, 256), device="cuda:0")

    cms = [
        (Mamba2Config(hidden_size=256, num_hidden_layers=1, num_heads=8, head_dim=32, expand=1, conv_kernel=4, use_fast_conv_scan=True), Mamba2Model,),
        (ObservableMambaConfig(hidden_size=256, num_hidden_layers=1, num_heads=8, head_dim=32, expand=1, conv_kernel=4, use_fast_conv_scan=True), ObservableMambaModel,),
        (AdaSyncSSMConfig(hidden_size=256, state_size=128, num_hidden_layers=1, num_heads=8, head_dim=32, conv_kernel=4), AdaSyncSSMModel,),
    ]

    out_list, _out_list, grad_list = [], [], []
    for c, mc in cms:
        torch.manual_seed(SEED)
        m = mc(config=c).to("cuda:0").eval()

        start_t = time.perf_counter()
        out = m.forward(inputs_embeds=x[..., :-1, :], use_cache=True)
        _out = m.forward(inputs_embeds=x[..., -1:, :], cache_params=out.cache_params, use_cache=True, cache_position=torch.LongTensor([4]))
        end_t = time.perf_counter()
        print(end_t - start_t)
        # out = m.forward(inputs_embeds=x[:, :-1, :], use_cache=True)
        # _out = m.forward(inputs_embeds=x[:, -1:, :], use_cache=True, cache_params=out.cache_params, cache_position=torch.randn((4,)))
        grad = torch.autograd.grad(out.last_hidden_state.norm() ** 2, m.parameters(), allow_unused=True)

        out_list.append(out)
        _out_list.append(_out)
        grad_list.append(grad)

    print("Synchronous")
    for out, _out in zip(out_list, _out_list):
        print(out.last_hidden_state.norm(), _out.last_hidden_state.norm())

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


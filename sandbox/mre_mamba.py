import torch

from transformers import MambaModel, MambaConfig, Mamba2Config, Mamba2Model


if __name__ == "__main__":
    c = Mamba2Config(hidden_size=256, num_hidden_layers=3, num_heads=8, head_dim=32, expand=1)
    m = Mamba2Model(c).to("cuda:0")

    x = torch.randn((1, 13, 256), device="cuda:0")

    out = m.forward(inputs_embeds=x, use_cache=True)
    states = out.cache_params.ssm_states

    # print({k: v.shape for k, v in out.cache_params.conv_states.items()})
    # print({k: v.shape for k, v in out.cache_params.ssm_states.items()})

    # out2 = m.forward(inputs_embeds=x[:, :1, :], use_cache=True, cache_params=out.cache_params, cache_position=torch.arange(4))
    # states2 = out2.cache_params.ssm_states


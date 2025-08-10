from argparse import Namespace

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

from model.transformer.base import TransformerPredictor


class LlamaInContextPredictor(TransformerPredictor):
    def __init__(self, modelArgs: Namespace):
        self.config: LlamaConfig = modelArgs.llama
        TransformerPredictor.__init__(self, modelArgs, self.config.hidden_size)

        self.core = LlamaForCausalLM(self.config)


class LlamaAssociativeInContextPredictor(LlamaInContextPredictor):
    def __init__(self, modelArgs: Namespace):
        LlamaInContextPredictor.__init__(self, modelArgs)

        self.cls_token = nn.Parameter(torch.randn((self.config.hidden_size,)) / (self.config.hidden_size ** 0.5))
        self.layers = nn.Sequential(*self.core.model.layers)

    def forward(self, trace: dict[str, dict[str, torch.Tensor]], **kwargs) -> dict[str, dict[str, torch.Tensor]]:
        B, L = trace["environment"]["observation"].shape[:2]
        embd_dict = self.trace_to_embedding(trace)

        observation_embds = torch.cat([
            torch.zeros((B, 1, self.S_D)),                      # [B x 1 x S_D]
            embd_dict["environment"]["observation"][:, :-1]     # [B x (L - 1) x S_D]
        ], dim=-2)                                              # [B x L x S_D]
        action_embds = sum(embd_dict["controller"].values())    # [B x L x S_D]
        embds = observation_embds + action_embds                # [B x L x S_D]

        x = self.cls_token.expand((B, self.config.hidden_size,))
        attention_mask = trace["mask"].to(torch.float) if "mask" in trace else None
        out: list[torch.Tensor] = []
        for i, embd in enumerate(torch.unbind(embds, dim=-2)):
            next_x = self.core.forward(
                inputs_embeds=torch.stack((embd, x,), dim=-2),
                output_hidden_states=True,
            ).hidden_states[-1][..., -1, :]                     # [B x S_D]
            x = next_x if attention_mask is None else torch.where(attention_mask[..., i], next_x, x)
            out.append(x)
        out: torch.Tensor = torch.stack(out, dim=-2)            # [B x L x S_D]

        return self.embedding_to_output({"environment": out})





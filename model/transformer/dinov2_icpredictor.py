from argparse import Namespace

import torch
import torch.nn as nn
from transformers import Dinov2Model

from model.transformer.base import TransformerPredictor


class Dinov2InContextPredictor(TransformerPredictor):
    def __init__(self, modelArgs: Namespace):
        self.config = modelArgs.dinov2
        TransformerPredictor.__init__(self, modelArgs, self.config.hidden_size)

        self.core = Dinov2Model(self.config)


class Dinov2AssociativeInContextPredictor(Dinov2InContextPredictor):
    def __init__(self, modelArgs: Namespace):
        Dinov2InContextPredictor.__init__(self, modelArgs)

        self.cls_token = nn.Parameter(torch.randn((self.config.hidden_size,)) / (self.config.hidden_size ** 0.5))
        self.obs_token = nn.Parameter(torch.randn((self.config.hidden_size,)) / (self.config.hidden_size ** 0.5))

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

        import time
        start_t = time.perf_counter()

        for i, embd in enumerate(torch.unbind(embds, dim=-2)):
            next_x = self.core.encoder.forward(
                torch.stack((embd + self.obs_token, x,), dim=-2),
                return_dict=True,
            ).last_hidden_state[..., -1, :]                     # [B x S_D]
            x = next_x if attention_mask is None else torch.where(attention_mask[..., i, None].to(torch.bool), next_x, x)
            out.append(x)
        
        end_t = time.perf_counter()
        print(f"forward: {end_t - start_t}s")
        # raise Exception()
        
        out: torch.Tensor = torch.stack(out, dim=-2)            # [B x L x S_D]

        return self.embedding_to_output({"environment": out})





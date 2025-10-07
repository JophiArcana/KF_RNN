from argparse import Namespace

import torch

from model.transformer.base import TransformerPredictor
from .modeling_adasync import AdaSyncSSMConfig, AdaSyncSSMModel


class AdaSyncSSMInContextPredictor(TransformerPredictor):
    def __init__(self, modelArgs: Namespace):
        self.config: AdaSyncSSMConfig = modelArgs.adasync
        TransformerPredictor.__init__(self, modelArgs, AdaSyncSSMModel(self.config), self.config.hidden_size)

    def forward(self, trace: dict[str, dict[str, torch.Tensor]], **kwargs) -> dict[str, dict[str, torch.Tensor]]:
        B, L = trace["environment"]["observation"].shape[:2]
        embd_dict = self.trace_to_embedding(trace)

        observation_embds = torch.cat([
            torch.zeros((B, 1, self.S_D,)),                     # [B x 1 x S_D]
            embd_dict["environment"]["observation"][:, :-1],    # [B x (L - 1) x S_D]
        ], dim=-2)                                              # [B x L x S_D]
        action_embds = sum(embd_dict["controller"].values())    # [B x L x S_D]
        embds = observation_embds + action_embds                # [B x L x S_D]

        out = self.core.eval().forward(
            inputs_embeds=embds,
            output_hidden_states=True,
            attention_mask=None, # trace["mask"].to(torch.float) if "mask" in trace else None,
        ).hidden_states[-1]                                     # [B x L x S_D]

        return self.embedding_to_output({"environment": out})





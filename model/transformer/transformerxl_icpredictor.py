from argparse import Namespace

import torch
from transformers import TransfoXLConfig, TransfoXLModel

from model.transformer.base import TransformerPredictor, TransformerController


class TransformerXLInContextPredictor(TransformerPredictor):
    def __init__(self, modelArgs: Namespace):
        self.config: TransfoXLConfig = modelArgs.transformerxl
        TransformerPredictor.__init__(self, modelArgs, TransfoXLModel(self.config), self.config.d_model)


class TransformerXLInContextController(TransformerXLInContextPredictor, TransformerController):
    def forward(self, trace: dict[str, dict[str, torch.Tensor]], **kwargs) -> dict[str, dict[str, torch.Tensor]]:
        B, L = trace["environment"]["observation"].shape[:2]

        embd_dict = self.trace_to_embedding(trace)
        embds = torch.stack([
            torch.cat([
                torch.zeros((B, 1, self.S_D)),
                embd_dict["environment"]["observation"][:, :-1]
            ], dim=-2) + self.observation_bias,
            sum(embd_dict["controller"].values()) + self.input_bias
        ], dim=-2).flatten(-3, -2)

        out = self.core(inputs_embeds=embds).last_hidden_state  # [B x 2L x S_D]
        return self.embedding_to_output({
            "controller": out[:, ::2],                          # [B x L x S_D]
            "environment": out[:, 1::2]                         # [B x L x S_D]
        })                                                      # [B x L x ...]




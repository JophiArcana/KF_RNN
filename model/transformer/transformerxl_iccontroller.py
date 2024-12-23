from argparse import Namespace
from typing import *

import torch
import torch.nn as nn
from transformers import TransfoXLModel

from model.transformer.base import TransformerController


class TransformerXLInContextController(TransformerController):
    def __init__(self, modelArgs: Namespace):
        self.config = modelArgs.transformerxl
        TransformerController.__init__(self, modelArgs, self.config.d_model)

        self.core = TransfoXLModel(self.config)

    def forward(self, trace: Dict[str, Dict[str, torch.Tensor]], **kwargs) -> Dict[str, Dict[str, torch.Tensor]]:
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





from argparse import Namespace
from typing import *

import torch
from transformers import TransfoXLModel

from model.transformer.base import TransformerController


class TransformerXLInContextController(TransformerController):
    def __init__(self, modelArgs: Namespace):
        self.config = modelArgs.transformerxl
        TransformerController.__init__(self, modelArgs, self.config.d_model)

        self.core = TransfoXLModel(self.config)

    def forward(self, trace: Dict[str, Dict[str, torch.Tensor]], **kwargs) -> Dict[str, torch.Tensor]:
        B, L = trace["environment"]["observation"].shape[:2]

        embd_dict = self.trace_to_embedding(trace)
        embds = torch.stack([
            torch.cat([torch.zeros((B, 1, self.S_D)), embd_dict["observation_embd"][:, :-1]], dim=-2),
            embd_dict["input_embd"]
        ], dim=-2).flatten(-3, -2)

        out = self.core(inputs_embeds=embds).last_hidden_state  # [B x 2L x S_D]
        return self.embedding_to_output({
            "input_embd": out[:, ::2],                          # [B x L x S_D]
            "observation_embd": out[:, 1::2]                    # [B x L x S_D]
        })                                                      # [B x L x ...]





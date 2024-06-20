from argparse import Namespace
from typing import *

import torch
from transformers import TransfoXLModel

from model.transformer.base import TransformerController


class TransformerXLInContextController(TransformerController):
    def __init__(self, modelArgs: Namespace):
        self.config = modelArgs.transformerxl
        super().__init__(modelArgs, self.config.d_model)

        self.core = TransfoXLModel(self.config)

    def forward(self, trace: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        B, L = trace["observation"].shape[:2]

        embd_dict = self.trace_to_embedding(trace)

        embds = torch.zeros((B, 2 * L, self.S_D))               # [B x 2L x S_D]
        embds[:, 2::2] = embd_dict["observation_embd"][:, :-1]  # [B x (L - 1) x S_D]
        embds[:, 1::2] = embd_dict["input_embd"]                # [B x L x S_D]

        out = self.core(inputs_embeds=embds).last_hidden_state  # [B x 2L x S_D]
        return self.embedding_to_output({
            "input_embd": out[:, ::2],                          # [B x L x S_D]
            "observation_embd": out[:, 1::2]                    # [B x L x S_D]
        })                                                      # [B x L x ...]





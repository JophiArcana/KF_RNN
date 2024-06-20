from argparse import Namespace
from typing import *

import torch
from transformers import GPT2Model

from model.transformer.base import TransformerPredictor


class GPT2InContextPredictor(TransformerPredictor):
    def __init__(self, modelArgs: Namespace):
        self.config = modelArgs.gpt2
        self.n_positions = self.config.n_positions
        TransformerPredictor.__init__(self, modelArgs, self.config.n_embd)

        self.core = GPT2Model(self.config)

    def forward(self, trace: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        B, L = trace["observation"].shape[:2]
        assert L <= self.n_positions, f"Trace length must be at most the context length of the transformer but got {self.n_positions}."

        embd_dict = self.trace_to_embedding(trace)
        embds = torch.cat([
            torch.zeros((B, 1, self.S_D)),                      # [B x 1 x S_D]
            embd_dict["observation_embd"][:, :-1]               # [B x (L - 1) x S_D]
        ], dim=1)                                               # [B x L x S_D]
        if self.input_enabled:
            embds = embds + embd_dict["input_embd"]             # [B x L x S_D]

        out = self.core(inputs_embeds=embds).last_hidden_state  # [B x L x S_D]
        return self.embedding_to_output({
            "observation_embd": out                             # [B x L x S_D]
        })                                                      # [B x L x ...]





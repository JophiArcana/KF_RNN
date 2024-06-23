from argparse import Namespace
from typing import *

import torch
import torch.nn as nn

from model.base import Predictor, Controller


class TransformerPredictor(Predictor):
    def __init__(self, modelArgs: Namespace, S_D: int):
        Predictor.__init__(self, modelArgs)
        self.S_D = S_D

        self.observation_in = nn.Parameter(torch.zeros((self.S_D, self.O_D)))       # [S_D x O_D]
        nn.init.kaiming_normal_(self.observation_in)
        self.observation_out = nn.Parameter(torch.zeros((self.O_D, self.S_D)))      # [O_D x S_D]
        nn.init.kaiming_normal_(self.observation_out)
        if self.input_enabled:
            self.input_in = nn.Parameter(torch.zeros((self.S_D, self.I_D)))         # [S_D x I_D]
            nn.init.kaiming_normal_(self.input_in)
        else:
            self.register_buffer("input_in", torch.zeros((self.S_D, self.I_D)))

    def forward(self, trace: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        B, L = trace["observation"].shape[:2]
        # assert L <= self.n_positions, f"Trace length must be at most the context length of the transformer but got {self.n_positions}."

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

    def trace_to_embedding(self, trace: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        result = {"observation_embd": trace["observation"] @ self.observation_in.mT}
        if self.input_enabled:
            result["input_embd"] = trace["input"] @ self.input_in.mT
        return result

    def embedding_to_output(self, embedding: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {"observation_estimation": embedding["observation_embd"] @ self.observation_out.mT}


class TransformerController(Controller, TransformerPredictor):
    def __init__(self, modelArgs: Namespace, S_D: int):
        Controller.__init__(self, modelArgs)
        TransformerPredictor.__init__(self, modelArgs, S_D)

        self.input_out = nn.Parameter(torch.zeros((self.I_D, self.S_D)))
        nn.init.kaiming_normal_(self.input_out)

    def embedding_to_output(self, embedding: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        result = super().embedding_to_output(embedding)
        result["input_estimation"] = embedding["input_embd"] @ self.input_out.mT
        return result




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

        self.input_in = nn.ParameterDict({
            k: nn.Parameter(torch.zeros((self.S_D, d)))
            for k, d in vars(self.problem_shape.controller).items()
        })
        for v in self.input_in.values():
            nn.init.kaiming_normal_(v)

    def forward(self, trace: Dict[str, Dict[str, torch.Tensor]], **kwargs) -> Dict[str, Dict[str, torch.Tensor]]:
        B, L = trace["environment"]["observation"].shape[:2]
        # assert L <= self.n_positions, f"Trace length must be at most the context length of the transformer but got {self.n_positions}."

        embd_dict = self.trace_to_embedding(trace)

        observation_embds = torch.cat([
            torch.zeros((B, 1, self.S_D)),                      # [B x 1 x S_D]
            embd_dict["environment"]["observation"][:, :-1]     # [B x (L - 1) x S_D]
        ], dim=-2)                                              # [B x L x S_D]
        action_embds = sum(embd_dict["controller"].values())    # [B x L x S_D]
        embds = observation_embds + action_embds                # [B x L x S_D]

        out = self.core(inputs_embeds=embds).last_hidden_state  # [B x L x S_D]
        return self.embedding_to_output({"environment": out})

    def trace_to_embedding(self, trace: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "environment": {
                "observation": trace["environment"]["observation"] @ self.observation_in.mT
            },
            "controller": {
                k: v @ self.input_in[k].mT
                for k, v in trace["controller"].items()
            }
        }

    def embedding_to_output(self, embedding: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "environment": {"observation": embedding["environment"] @ self.observation_out.mT},
            "controller": {}
        }


class TransformerController(Controller, TransformerPredictor):
    def __init__(self, modelArgs: Namespace, S_D: int):
        Controller.__init__(self, modelArgs)
        TransformerPredictor.__init__(self, modelArgs, S_D)

        self.input_out = nn.ParameterDict({
            k: nn.Parameter(torch.zeros((d, self.S_D)))
            for k, d in vars(self.problem_shape.controller).items()
        })
        for v in self.input_out.values():
            nn.init.kaiming_normal_(v)

    def embedding_to_output(self, embedding: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        result = TransformerPredictor.embedding_to_output(self, embedding)
        result["controller"] = {
            k: embedding["controller"] @ v.mT
            for k, v in self.input_out.items()
        }
        return result





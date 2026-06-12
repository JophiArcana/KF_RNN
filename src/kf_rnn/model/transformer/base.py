from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from kf_rnn.model.base import Predictor, Controller
from kf_rnn.infrastructure.config.schema import controller_dims


class TransformerPredictor(Predictor):
    @dataclass
    class Config(Predictor.Config):
        adapter: bool = False
        bias: bool = True

    def __init__(self, modelArgs: "TransformerPredictor.Config", core: PreTrainedModel, S_D: int):
        Predictor.__init__(self, modelArgs)
        self.core = core
        self.S_D = S_D

        if modelArgs.adapter:
            self.observation_in = nn.Parameter(torch.zeros((self.S_D, self.O_D)))       # [S_D x O_D]
            nn.init.kaiming_normal_(self.observation_in)
            self.observation_out = nn.Parameter(torch.zeros((self.O_D, self.S_D)))      # [O_D x S_D]
            nn.init.kaiming_normal_(self.observation_out)
        else:
            assert self.S_D == self.O_D, f"State dimension and observation dimension must be identical when no adapter is used but got {self.S_D} and {self.O_D}."
            self.observation_in = torch.eye(self.S_D)
            self.observation_out = torch.eye(self.O_D)

        self.input_in = nn.ParameterDict({
            k: nn.Parameter(torch.zeros((self.S_D, d)))                             # [S_D x I_D?]
            for k, d in controller_dims(self.problem_shape).items()
        })
        for v in self.input_in.values():
            nn.init.kaiming_normal_(v)

        if modelArgs.bias:
            b = torch.randn((self.S_D,)) / (self.S_D ** 0.5)
            self.input_bias = nn.Parameter(b)
            self.observation_bias = nn.Parameter(-b)
        else:
            self.register_buffer("input_bias", torch.zeros((self.S_D,)))
            self.register_buffer("observation_bias", torch.zeros((self.S_D,)))

    def forward(self, trace: dict[str, dict[str, torch.Tensor]], **kwargs) -> dict[str, dict[str, torch.Tensor]]:
        B, L = trace["environment"]["observation"].shape[:2]
        embd_dict = self.trace_to_embedding(trace)

        observation_embds = torch.cat([
            torch.zeros((B, 1, self.S_D,)),                     # [B x 1 x S_D]
            embd_dict["environment"]["observation"][:, :-1],    # [B x (L - 1) x S_D]
        ], dim=-2)                                              # [B x L x S_D]
        action_embds = sum(embd_dict["controller"].values())    # [B x L x S_D]
        # input_bias/observation_bias are token-type markers (observation_bias == -input_bias).
        # They are only meaningful when observations and actions occupy distinct token positions
        # (see TransformerXLInContextController); in this summed-embedding scheme they would cancel,
        # so they are intentionally not added here.
        embds = observation_embds + action_embds                # [B x L x S_D]

        out = self.core.forward(
            inputs_embeds=embds,
            output_hidden_states=True,
            attention_mask=None, # trace["mask"].to(torch.float) if "mask" in trace else None,
        ).hidden_states[-1]                                     # [B x L x S_D]

        # The single per-timestep hidden state decodes into both the next observation and the
        # controller actions, so it is supplied under both keys; subclasses pick what they need.
        return self.embedding_to_output({"environment": out, "controller": out})

    def trace_to_embedding(self, trace: dict[str, dict[str, torch.Tensor]]) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "environment": {
                "observation": trace["environment"]["observation"] @ self.observation_in.mT
            },
            "controller": {
                k: v @ self.input_in[k].mT
                for k, v in trace["controller"].items()
            }
        }

    def embedding_to_output(self, embedding: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "environment": {"observation": embedding["environment"] @ self.observation_out.mT},
            "controller": {}
        }


class TransformerController(Controller, TransformerPredictor):
    def __init__(self, modelArgs: "TransformerController.Config", core: PreTrainedModel, S_D: int):
        TransformerPredictor.__init__(self, modelArgs, core, S_D)

        self.input_out = nn.ParameterDict({
            k: nn.Parameter(torch.zeros((d, self.S_D)))
            for k, d in controller_dims(self.problem_shape).items()
        })

    def embedding_to_output(self, embedding: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
        result = TransformerPredictor.embedding_to_output(self, embedding)
        result["controller"] = {
            k: embedding["controller"] @ v.mT
            for k, v in self.input_out.items()
        }
        return result





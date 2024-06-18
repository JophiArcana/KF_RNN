from argparse import Namespace
from typing import *

import torch
import torch.nn as nn
from transformers import GPT2Model

from model.base.filter import Filter


class GPT2InContextFilter(Filter):
    def __init__(self, modelArgs: Namespace):
        super().__init__(modelArgs)
        self.config = modelArgs.gpt2
        self.n_embd = self.config.n_embd
        self.n_positions = self.config.n_positions

        self.core = GPT2Model(self.config)

        self.observation_in = nn.Parameter(torch.zeros((self.n_embd, self.O_D)))    # [S_D x O_D]
        nn.init.kaiming_normal_(self.observation_in)

        if self.input_enabled:
            self.input_in = nn.Parameter(torch.zeros((self.n_embd, self.I_D)))      # [S_D x I_D]
            nn.init.kaiming_normal_(self.input_in)
        else:
            self.register_buffer("input_in", torch.zeros((self.n_embd, self.I_D)))

        self.out = nn.Parameter(torch.zeros((self.O_D, self.n_embd)))               # [O_D x S_D]
        nn.init.kaiming_normal_(self.out)

    def forward(self, trace: Dict[str, torch.Tensor], position: str = "absolute") -> Dict[str, torch.Tensor]:
        inputs, observations = trace['input'], trace['observation']                                     # [B x L x I_D], [B x L x O_D]
        B, L = inputs.shape[:2]

        observation_embds = observations @ self.observation_in.mT                                       # [B x L x S_D]
        pad = torch.zeros((B, 1, self.n_embd))

        embds = torch.cat([pad, observation_embds[:, :-1]], dim=1)                                      # [B x L x S_D]
        if self.input_enabled:
            input_embds = inputs @ self.input_in.mT                                                     # [B x L x S_D]
            embds = input_embds + embds                                                                 # [B x L x S_D]

        if position == "absolute":
            assert L <= self.n_positions, f"Trace length must be at most the context length of the transformer but got {self.n_positions}."
            # print("absolute position", embds.shape)

            out = self.core(inputs_embeds=embds)
            out_embds = out.last_hidden_state                                                           # [B x L x S_D]
        elif position == "relative":
            padded_embds = torch.cat([embds, pad], dim=1)                                               # [B x (L + 1) x S_D]

            indices = (torch.arange(L)[:, None] + torch.arange(-self.n_positions, 0) + 1).clamp_min(-1) # [L x C]
            attention_mask = (indices >= 0)                                                             # [L x C]

            batched_embds = padded_embds[:, indices]                                                    # [B x L x C x S_D]
            flattened_batched_embds = batched_embds.flatten(0, 1)                                       # [BL x C x S_D]
            flattened_attention_mask = attention_mask.expand(B, L, self.n_positions).flatten(0, 1)      # [BL x C]

            out = self.core(inputs_embeds=flattened_batched_embds, attention_mask=flattened_attention_mask)
            out_embds = out.last_hidden_state.unflatten(0, (B, L)).index_select(-2, torch.tensor([self.n_positions - 1])).squeeze(-2)   # [B x L x S_D]
        else:
            raise AssertionError(f"position must be in (absolute, relative) but got {position}.")
        result = out_embds @ self.out.mT
        return {'observation_estimation': result}                                                       # [B x L x O_D]





from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

from kf_rnn.model.transformer.base import TransformerPredictor


class GPT2InContextPredictor(TransformerPredictor):
    @dataclass
    class Config(TransformerPredictor.Config):
        gpt2: GPT2Config = None
    def __init__(self, modelArgs: "GPT2InContextPredictor.Config"):
        self.config: GPT2Config = modelArgs.gpt2
        TransformerPredictor.__init__(self, modelArgs, GPT2Model(self.config), self.config.n_embd)





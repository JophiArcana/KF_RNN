from argparse import Namespace

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

from model.transformer.base import TransformerPredictor


class GPT2InContextPredictor(TransformerPredictor):
    def __init__(self, modelArgs: Namespace):
        self.config: GPT2Config = modelArgs.gpt2
        TransformerPredictor.__init__(self, modelArgs, GPT2Model(self.config), self.config.n_embd)





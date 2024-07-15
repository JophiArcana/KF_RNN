from argparse import Namespace

from transformers import GPT2Model

from model.transformer.base import TransformerPredictor


class GPT2InContextPredictor(TransformerPredictor):
    def __init__(self, modelArgs: Namespace):
        self.config = modelArgs.gpt2
        self.n_positions = self.config.n_positions
        TransformerPredictor.__init__(self, modelArgs, self.config.n_embd)

        self.core = GPT2Model(self.config)





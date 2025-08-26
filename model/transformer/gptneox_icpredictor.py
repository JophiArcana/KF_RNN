from argparse import Namespace

from transformers import GPTNeoXConfig, GPTNeoXModel

from model.transformer.base import TransformerPredictor


class GPTNeoXInContextPredictor(TransformerPredictor):
    def __init__(self, modelArgs: Namespace):
        self.config: GPTNeoXConfig = modelArgs.gptneox
        TransformerPredictor.__init__(self, modelArgs, GPTNeoXModel(self.config), self.config.hidden_size)





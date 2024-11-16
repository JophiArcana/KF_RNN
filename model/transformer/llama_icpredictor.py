from argparse import Namespace

from transformers import LlamaModel, LlamaForCausalLM

from model.transformer.base import TransformerPredictor


class LlamaInContextPredictor(TransformerPredictor):
    def __init__(self, modelArgs: Namespace):
        self.config = modelArgs.llama
        TransformerPredictor.__init__(self, modelArgs, self.config.hidden_size)

        self.core = LlamaForCausalLM(self.config)





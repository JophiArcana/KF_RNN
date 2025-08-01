from argparse import Namespace

from transformers import XLNetModel

from model.transformer.base import TransformerPredictor


class XLNetInContextPredictor(TransformerPredictor):
    def __init__(self, modelArgs: Namespace):
        self.config = modelArgs.xlnet
        TransformerPredictor.__init__(self, modelArgs, self.config.d_model)

        self.core = XLNetModel(self.config)





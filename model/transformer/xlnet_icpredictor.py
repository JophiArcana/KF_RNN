from argparse import Namespace

from transformers import XLNetConfig, XLNetModel

from model.transformer.base import TransformerPredictor


class XLNetInContextPredictor(TransformerPredictor):
    def __init__(self, modelArgs: Namespace):
        self.config: XLNetConfig = modelArgs.xlnet
        TransformerPredictor.__init__(self, modelArgs, XLNetModel(self.config), self.config.d_model)





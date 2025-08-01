from argparse import Namespace

from transformers import TransfoXLModel

from model.transformer.base import TransformerPredictor


class TransformerXLInContextPredictor(TransformerPredictor):
    def __init__(self, modelArgs: Namespace):
        self.config = modelArgs.transformerxl
        TransformerPredictor.__init__(self, modelArgs, self.config.d_model)

        self.core = TransfoXLModel(self.config)





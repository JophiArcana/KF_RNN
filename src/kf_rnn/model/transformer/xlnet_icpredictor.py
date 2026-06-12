from dataclasses import dataclass

from transformers import XLNetConfig, XLNetModel

from kf_rnn.model.transformer.base import TransformerPredictor


class XLNetInContextPredictor(TransformerPredictor):
    @dataclass
    class Config(TransformerPredictor.Config):
        xlnet: XLNetConfig = None
    def __init__(self, modelArgs: "XLNetInContextPredictor.Config"):
        self.config: XLNetConfig = modelArgs.xlnet
        TransformerPredictor.__init__(self, modelArgs, XLNetModel(self.config), self.config.d_model)





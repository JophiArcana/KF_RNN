from dataclasses import dataclass

from transformers import GPTNeoXConfig, GPTNeoXModel

from kf_rnn.model.transformer.base import TransformerPredictor


class GPTNeoXInContextPredictor(TransformerPredictor):
    @dataclass
    class Config(TransformerPredictor.Config):
        gptneox: GPTNeoXConfig = None
    def __init__(self, modelArgs: "GPTNeoXInContextPredictor.Config"):
        self.config: GPTNeoXConfig = modelArgs.gptneox
        TransformerPredictor.__init__(self, modelArgs, GPTNeoXModel(self.config), self.config.hidden_size)





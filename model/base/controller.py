from argparse import Namespace

from model.base.predictor import Predictor


class Controller(Predictor):
    def __init__(self, modelArgs: Namespace):
        super().__init__(modelArgs)
        assert self.input_enabled, f"Input must be enabled for controller model but got {self.input_enabled}."





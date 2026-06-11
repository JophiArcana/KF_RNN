from typing import Sequence

from kf_rnn.model.sequential.rnn_predictor import RnnPredictor


class RnnInContextPredictor(RnnPredictor):
    """A Kalman-style predictor used purely in-context: it shares ``RnnPredictor``'s
    forward pass but is never trained (empty training recipe)."""

    def training_recipe(self) -> Sequence[str]:
        return []

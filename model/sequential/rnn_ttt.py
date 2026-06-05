from argparse import Namespace
from typing import Sequence

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
# from causal_conv1d.cpp_functions import causal_conv1d_fwd_function
from tensordict import TensorDict, NonTensorData

from infrastructure import utils
from model.sequential.rnn_predictor import RnnPredictor


class RnnInContextPredictor(RnnPredictor):
    def forward(self, trace: dict[str, dict[str, torch.Tensor]], mode: str = None) -> dict[str, dict[str, torch.Tensor]]:
        trace: TensorDict = self.trace_to_td(trace)
        actions, observations = trace["controller"], trace["environment", "observation"]

        state_estimation = self.sample_initial_as_observations(observations, (*trace.shape[:-1], self.S_D,))
        return self.forward_with_initial(state_estimation, actions, observations, mode)

    def training_recipe(self) -> Sequence[str]:
        return []

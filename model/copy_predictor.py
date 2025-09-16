from argparse import Namespace
from typing import *

import torch
import torch.nn as nn
from tensordict import TensorDict

from infrastructure import utils
from infrastructure.static import TrainFunc
from model.base import Predictor
from model.convolutional import ConvolutionalPredictor


class CopyPredictor(Predictor):
    def __init__(self, modelArgs: Namespace):
        nn.Module.__init__(self)

    @classmethod
    def _analytical_error_and_cache(cls,
                                    kfs: TensorDict,         # [B... x ...]
                                    systems: TensorDict,     # [B... x ...]
    ) -> Tuple[TensorDict, Namespace]:                       # [B...]
        K = utils.complex(systems["environment", "K"])                                                  # [B... x S_D x O_D]
        S_D, O_D = K.shape[-2:]

        shape = systems.shape if kfs is None else kfs.shape
        cnn_kfs = TensorDict({
            "observation_IR": torch.eye(O_D)[:, None, :],
        }, batch_size=()).expand(shape)
        return ConvolutionalPredictor._analytical_error_and_cache(cnn_kfs, systems)

    @classmethod
    def train_func_list(cls, default_train_func: TrainFunc) -> Sequence[TrainFunc]:
        return ()

    def forward(self, trace: Dict[str, Dict[str, torch.Tensor]], **kwargs) -> Dict[str, torch.Tensor]:
        trace = TensorDict(trace, batch_size=trace["environment"]["observation"].shape[:-1])
        valid_keys = [("environment", "observation",),] + [("controller", ac_name) for ac_name in trace["controller"].keys()]        
        result: TensorDict = TensorDict.cat((
            trace[..., -1:].apply(torch.zeros_like),
            trace[..., :-1],
        ), dim=-1)
        for k in [*result.keys(include_nested=True, leaves_only=True),]:
            if k not in valid_keys:
                result.del_(k)
        return result





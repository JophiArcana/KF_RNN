from argparse import Namespace
from typing import *

import torch
import torch.nn as nn
from tensordict import TensorDict

from infrastructure import utils


class SystemGroup(nn.Module):
    def __init__(self, input_enabled: bool):
        super().__init__()
        self.input_enabled = input_enabled

    def generate_dataset(self, batch_size: int, sequence_length: int) -> TensorDict[str, torch.Tensor]:
        raise NotImplementedError()

    def td(self) -> TensorDict[str, torch.Tensor]:
        return TensorDict(dict((
            *self.named_parameters(),
            *self.named_buffers()
        )), batch_size=self.group_shape)


class SystemDistribution(object):
    def __init__(self, system_type: type):
        self.system_type = system_type

    def sample_parameters(self, SHP: Namespace, shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def sample(self,
               SHP: Namespace,
               shape: Tuple[int, ...],
               params: TensorDict[str, torch.Tensor] | Dict[str, torch.Tensor] = None
    ) -> SystemGroup:
        if params is None:
            params = self.sample_parameters(SHP, shape)
        return utils.call_func_with_kwargs(self.system_type, (params, SHP.input_enabled), vars(SHP))





from argparse import Namespace
from typing import *

import torch
import torch.nn as nn
from tensordict import TensorDict


class SystemGroup(nn.Module):
    def __init__(self, params: Dict[str, torch.Tensor], input_enabled: bool):
        super().__init__()
        self.input_enabled = input_enabled

    def generate_dataset(self,
                         batch_size: int,       # B
                         seq_length: int        # L
    ) -> TensorDict[str, torch.Tensor]:
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

    def sample(self, SHP: Namespace, shape: Tuple[int, ...]) -> SystemGroup:
        print(self.system_type)
        return self.system_type(self.sample_parameters(SHP, shape), SHP.input_enabled)





from argparse import Namespace
from typing import *

import torch
import torch.nn as nn
from tensordict import TensorDict

from system.module_group import ModuleGroup


class ControllerGroup(ModuleGroup):
    def __init__(self, problem_shape: Namespace, group_shape: Tuple[int, ...]):
        ModuleGroup.__init__(self, group_shape)
        self.problem_shape = problem_shape

    def forward(self,
                history: TensorDict[str, torch.Tensor]  # [N... x B x L x ...]
    ) -> TensorDict[str, torch.Tensor]:                 # [N... x B x ...]
        raise NotImplementedError()

    def get_zero_knowledge_action(self, batch_size) -> TensorDict[str, torch.Tensor]:
        return TensorDict({
            k: torch.zeros((*self.group_shape, batch_size, d))
            for k, d in vars(self.problem_shape.controller).items()
        }, batch_size=(*self.group_shape, batch_size))  # [N... x B x ...]


class DefaultControllerGroup(ControllerGroup):
    def forward(self,
                history: TensorDict[str, torch.Tensor]  # [N... x B x L x ...]
    ) -> TensorDict[str, torch.Tensor]:                 # [N... x B x ...]
        return TensorDict({}, batch_size=history.shape[:-1])


class RandomControllerGroup(ControllerGroup):
    def forward(self,
                history: TensorDict[str, torch.Tensor]  # [N... x B x L x ...]
    ) -> TensorDict[str, torch.Tensor]:                 # [N... x B x ...]
        return TensorDict({
            k: torch.randn((*history.shape[:-1], d))
            for k, d in vars(self.problem_shape.controller).items()
        }, batch_size=history.shape[:-1])


class LinearControllerGroup(ControllerGroup):
    def __init__(self, problem_shape: Namespace, group_shape: Tuple[int, ...]):
        ControllerGroup.__init__(self, problem_shape, group_shape)
        self.L = nn.Module()

    def forward(self,
                history: TensorDict[str, torch.Tensor]      # [N... x B x L x ...]
    ) -> TensorDict[str, torch.Tensor]:                     # [N... x B x ...]
        state = history[..., -1]["environment", "state"]    # [N... x B x S_D]
        return TensorDict({
            k: state @ -getattr(self.L, k).mT
            for k in vars(self.problem_shape.controller)
        }, batch_size=history.shape[:-1])





from argparse import Namespace
from typing import *

import torch
import torch.nn as nn
from tensordict import TensorDict

from infrastructure import utils
from system.module_group import ModuleGroup
from model.base import Controller


class ControllerGroup(ModuleGroup):
    def __init__(self, problem_shape: Namespace, group_shape: Tuple[int, ...]):
        ModuleGroup.__init__(self, group_shape)
        self.problem_shape = problem_shape

    def act(self,
            history: TensorDict  # [N... x B x L x ...]
    ) -> TensorDict:             # [N... x B x ...]
        raise NotImplementedError()

    def get_zero_knowledge_action(self, batch_size) -> TensorDict:
        return TensorDict({
            k: torch.zeros((*self.group_shape, batch_size, d))
            for k, d in vars(self.problem_shape.controller).items()
        }, batch_size=(*self.group_shape, batch_size))  # [N... x B x ...]


class ZeroControllerGroup(ControllerGroup):
    def act(self,
            history: TensorDict  # [N... x B x L x ...]
    ) -> TensorDict:             # [N... x B x ...]
        return history["controller"][..., -1].apply(torch.zeros_like)


class LinearControllerGroup(ControllerGroup):
    def __init__(self, problem_shape: Namespace, group_shape: Tuple[int, ...]):
        ControllerGroup.__init__(self, problem_shape, group_shape)
        self.L = nn.Module()

    def act(self,
            history: TensorDict  # [N... x B x L x ...]
    ) -> TensorDict:             # [N... x B x ...]
        if len(vars(self.problem_shape.controller)) == 0:
            return TensorDict({}, batch_size=history.shape[:-1])
        else:
            state = history[..., -1]["environment", "target_state_estimation"]  # [N... x B x S_D]
            return TensorDict({
                k: state @ -getattr(self.L, k).mT
                for k in vars(self.problem_shape.controller)
            }, batch_size=history.shape[:-1])


class NNControllerGroup(ControllerGroup):
    def __init__(self,
                 problem_shape: Namespace,
                 reference_module: Controller,
                 ensembled_learned_controllers: TensorDict
    ):
        ControllerGroup.__init__(self, problem_shape, ensembled_learned_controllers.shape)
        self.reference_module = reference_module
        self.ensembled_learned_controllers = ensembled_learned_controllers

    def act(self,
            history: TensorDict  # [N... x B x L x ...]
    ) -> TensorDict:             # [N... x B x ...]
        return TensorDict(utils.run_module_arr(
            self.reference_module,
            self.ensembled_learned_controllers,
            torch.cat([history, history[..., -1:].apply(torch.zeros_like)], dim=-1)
        ), batch_size=(*history.shape[:-1], history.shape[-1] + 1))[..., -1]["controller"]





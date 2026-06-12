
import torch
import torch.nn as nn
from tensordict import TensorDict

import ecliseutils as eu
from kf_rnn.system.module_group import ModuleGroup
from kf_rnn.model.base import Controller
from kf_rnn.infrastructure.config.schema import ProblemShape, controller_dims


class ControllerGroup(ModuleGroup):
    def __init__(self, problem_shape: ProblemShape, group_shape: tuple[int, ...]):
        ModuleGroup.__init__(self, group_shape)
        self.problem_shape = problem_shape

    def act(self,
            history: TensorDict  # [N... x B x L x ...]
    ) -> TensorDict:             # [N... x B x ...]
        raise NotImplementedError()

    def get_zero_knowledge_action(self, batch_size) -> TensorDict:
        return TensorDict({
            k: torch.zeros((*self.group_shape, batch_size, d))
            for k, d in controller_dims(self.problem_shape).items()
        }, batch_size=(*self.group_shape, batch_size))  # [N... x B x ...]


class ZeroControllerGroup(ControllerGroup):
    def act(self,
            history: TensorDict  # [N... x B x L x ...]
    ) -> TensorDict:             # [N... x B x ...]
        return history["controller"][..., -1].apply(torch.zeros_like)


class LinearControllerGroup(ControllerGroup):
    def __init__(self, problem_shape: ProblemShape, group_shape: tuple[int, ...]):
        ControllerGroup.__init__(self, problem_shape, group_shape)
        self.L = nn.Module()

    def act(self,
            history: TensorDict  # [N... x B x L x ...]
    ) -> TensorDict:             # [N... x B x ...]
        if len(controller_dims(self.problem_shape)) == 0:
            return TensorDict({}, batch_size=history.shape[:-1])
        else:
            state = history[..., -1]["environment", "target_state_estimation"]  # [N... x B x S_D]
            return TensorDict({
                k: state @ -getattr(self.L, k).mT
                for k in controller_dims(self.problem_shape)
            }, batch_size=history.shape[:-1])


class NNControllerGroup(ControllerGroup):
    def __init__(self,
                 problem_shape: ProblemShape,
                 reference_module: Controller,
                 stacked_controllers: TensorDict
    ):
        ControllerGroup.__init__(self, problem_shape, stacked_controllers.shape)
        self.reference_module = reference_module
        self.stacked_controllers = stacked_controllers

    def act(self,
            history: TensorDict  # [N... x B x L x ...]
    ) -> TensorDict:             # [N... x B x ...]
        return TensorDict(eu.run_module_arr(
            (self.reference_module, self.stacked_controllers,),
            torch.cat([history, history[..., -1:].apply(torch.zeros_like)], dim=-1),
        ), batch_size=(*history.shape[:-1], history.shape[-1] + 1))[..., -1]["controller"]





from argparse import Namespace
from typing import *

import numpy as np
import torch
from tensordict import TensorDict

from infrastructure import utils
from system.controller import ControllerGroup, DefaultControllerGroup
from system.module_group import ModuleGroup


class SystemGroup(ModuleGroup):
    def __init__(self, problem_shape: Namespace, group_shape: Tuple[int, ...]):
        ModuleGroup.__init__(self, group_shape)
        self.problem_shape = problem_shape
        self.controller = DefaultControllerGroup(self.problem_shape, self.group_shape)

    def generate_dataset(self, batch_size: int, sequence_length: int) -> TensorDict[str, torch.Tensor]:
        return self.generate_dataset_with_controller_arr(utils.array_of(self.controller), batch_size, sequence_length)
        # with torch.set_grad_enabled(False):
        #     action = self.controller.get_zero_knowledge_action(batch_size).expand(*self.group_shape, batch_size)    # [N... x B x ...]
        #     state = self.sample_initial_state(batch_size).expand(*self.group_shape, batch_size)                     # [N... x B x ...]
        #
        #     def construct_timestep(
        #             ac: TensorDict[str, torch.Tensor],  # [N... x B x ...]
        #             st: TensorDict[str, torch.Tensor]   # [N... x B x ...]
        #     ) -> TensorDict[str, torch.Tensor]:         # [N... x B x 1 x ...]
        #         return TensorDict({
        #             "environment": st,
        #             "controller": ac
        #         }, batch_size=(*self.group_shape, batch_size)).unsqueeze(-1)
        #
        #     history = construct_timestep(action, state)
        #     for _ in range(sequence_length - 1):
        #         action = self.controller.forward(history)                                                           # [N... x B x ...]
        #         state = self.step(history["environment"][..., -1], action)                                          # [N... x B x ...]
        #         history = torch.cat([
        #             history,
        #             construct_timestep(action, state)
        #         ], dim=-1)
        #     return history

    def generate_dataset_with_controller_arr(self,
                                             controller_arr: np.ndarray[ControllerGroup],
                                             batch_size: int,
                                             sequence_length: int
    ) -> TensorDict[str, torch.Tensor]:
        with torch.set_grad_enabled(False):
            group_shape = torch.broadcast_shapes(
                self.group_shape,
                *(controller.group_shape for controller in controller_arr.ravel())
            )

            action = utils.stack_tensor_arr(utils.multi_map(
                lambda controller: controller.get_zero_knowledge_action(batch_size).expand(*group_shape, batch_size),
                controller_arr, dtype=TensorDict
            ))
            state = self.sample_initial_state(batch_size).expand(*controller_arr.shape, *group_shape, batch_size)

            def construct_timestep(
                    ac: TensorDict[str, torch.Tensor],  # [C... x N... x B x ...]
                    st: TensorDict[str, torch.Tensor]   # [C... x N... x B x ...]
            ) -> TensorDict[str, torch.Tensor]:         # [C... x N... x B x 1 x ...]
                return TensorDict({
                    "environment": st,
                    "controller": ac
                }, batch_size=(*controller_arr.shape, *group_shape, batch_size)).unsqueeze(-1)

            history = construct_timestep(action, state)
            for _ in range(sequence_length - 1):
                action_arr = np.empty_like(controller_arr)
                for idx, controller in utils.multi_enumerate(controller_arr):
                    action_arr[idx] = controller.forward(history[idx])
                action = utils.stack_tensor_arr(action_arr)                     # [C... x N... x B x ...]
                state = self.step(history["environment"][..., -1], action)      # [C... x N... x B x ...]
                history = torch.cat([
                    history,
                    construct_timestep(action, state)
                ], dim=-1)
            return history


    def sample_initial_state(self,
                             batch_size: int                # B
    ) -> TensorDict[str, torch.Tensor]:                     # [N... x B x ...]
        raise NotImplementedError()

    def step(self,
             state: TensorDict[str, torch.Tensor],          # [N... x B x ...]
             action: TensorDict[str, torch.Tensor]          # [N... x B x ...]
    ) -> TensorDict[str, torch.Tensor]:                     # [N... x B x ...]
        raise NotImplementedError()


class SystemDistribution(object):
    def __init__(self, system_type: type):
        self.system_type = system_type

    def sample_parameters(self, SHP: Namespace, shape: Tuple[int, ...]) -> TensorDict[str, torch.Tensor]:
        raise NotImplementedError()

    def sample(self,
               SHP: Namespace,
               shape: Tuple[int, ...]
    ) -> SystemGroup:
        return utils.call_func_with_kwargs(self.system_type, (SHP.problem_shape, self.sample_parameters(SHP, shape)), vars(SHP))







from argparse import Namespace

import numpy as np
import numpy.typing
import torch
from tensordict import TensorDict

from infrastructure import utils
from .environment import EnvironmentGroup
from .controller import ControllerGroup
from .module_group import ModuleGroup


class SystemGroup(ModuleGroup):
    def __init__(self,
                 hyperparameters: Namespace,
                 environment: EnvironmentGroup,
                 controller: ControllerGroup,
    ):
        ModuleGroup.__init__(self, utils.broadcast_shapes(
            environment.group_shape,
            controller.group_shape,
        ))
        self.hyperparameters = hyperparameters
        self.environment = environment
        self.controller = controller

    def generate_dataset(self, batch_size: int, sequence_length: int) -> TensorDict:
        return self.generate_dataset_with_controller_arr(utils.array_of(self.controller), batch_size, sequence_length)

    def generate_dataset_with_controller_arr(
        self,
        controller_arr: np.typing.ArrayLike,
        batch_size: int,
        sequence_length: int,
    ) -> TensorDict:
        controller_arr = np.array(controller_arr)
        group_shape = utils.broadcast_shapes(
            self.group_shape,
            *(controller.group_shape for controller in controller_arr.ravel()),
        )

        action = utils.stack_tensor_arr(utils.multi_map(
            lambda controller: controller.get_zero_knowledge_action(batch_size).expand(*group_shape, batch_size),
            controller_arr, dtype=TensorDict,
        ))
        state = self.environment.sample_initial_state(batch_size).expand(*controller_arr.shape, *group_shape, batch_size)

        def construct_timestep(
                ac: TensorDict,  # [C... x N... x B x ...]
                st: TensorDict,  # [C... x N... x B x ...]
        ) -> TensorDict:         # [C... x N... x B x 1 x ...]
            return TensorDict({
                "environment": st,
                "controller": ac,
            }, batch_size=(*controller_arr.shape, *group_shape, batch_size,))[..., None]

        # history = construct_timestep(action, state)
        # for _ in range(sequence_length - 1):
        #     action_arr = np.empty_like(controller_arr)
        #     for idx, controller in utils.multi_enumerate(controller_arr):
        #         action_arr[idx] = controller.act(history[idx])
        #     action = utils.stack_tensor_arr(action_arr)                                 # [C... x N... x B x ...]
        #     state = self.environment.step(history["environment"][..., -1], action)      # [C... x N... x B x ...]
        #     history = torch.cat([
        #         history,
        #         construct_timestep(action, state)
        #     ], dim=-1)
            
        #     utils.empty_cache()
        
        history = [construct_timestep(action, state)]
        for _ in range(sequence_length - 1):
            action_arr = np.empty_like(controller_arr)
            for idx, controller in utils.multi_enumerate(controller_arr):
                action_arr[idx] = controller.act(history[-1][idx])
            action = utils.stack_tensor_arr(action_arr)                                 # [C... x N... x B x ...]
            state = self.environment.step(history[-1]["environment"][..., 0], action)   # [C... x N... x B x ...]
            history.append(construct_timestep(action, state))
            
            utils.empty_cache()
        history = torch.cat(history, dim=-1)

        return history


class SystemDistribution(object):
    def __init__(self, system_type: type):
        self.system_type = system_type

    def sample_parameters(self, SHP: Namespace, shape: tuple[int, ...]) -> TensorDict:
        raise NotImplementedError()

    def sample(
        self,
        SHP: Namespace,
        shape: tuple[int, ...],
    ) -> SystemGroup:
        return self.system_type(SHP, self.sample_parameters(SHP, shape))





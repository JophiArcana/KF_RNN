from typing import *

import torch

from system.module_group import ModuleGroup
from system.simple.base import SystemGroup
from system.controller import ControllerGroup


class ActionableSystemGroup(SystemGroup):
    def __init__(self, group_shape: Tuple[int, ...], controller: ControllerGroup):
        ModuleGroup.__init__(self, torch.broadcast_shapes(
            group_shape,
            controller.group_shape
        ))
        self.controller = controller

    def set_controller(self, controller: ControllerGroup):
        self.controller = controller
        self.group_shape = torch.broadcast_shapes(self.group_shape, self.controller.group_shape)





import torch
import torch.nn as nn
from tensordict import TensorDict


class ModuleGroup(nn.Module):
    def __init__(self, group_shape: tuple[int, ...]):
        super().__init__()
        self.group_shape = (*map(int, group_shape),)

    def td(self) -> TensorDict:
        return TensorDict({
            (*k.split("."),): v
            for k, v in (
                *self.named_parameters(),
                *self.named_buffers()
            )
        }, batch_size=self.group_shape)





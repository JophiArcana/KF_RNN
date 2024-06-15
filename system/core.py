import torch
import torch.nn as nn
from tensordict import TensorDict


class SystemGroup(nn.Module):
    def __init__(self, input_enabled: bool):
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







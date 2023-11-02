
import torch
import time


if __name__ == '__main__':
    T = 4
    D = 2
    state_weights = torch.randint(-10, 11, (T, D, D))

    buffered_state_weights = torch.cat([state_weights, torch.zeros_like(state_weights)])
    lower_triangular_indices = torch.arange(T)[:, None] - torch.arange(T)

    blocked_lower_triangular_matrix = buffered_state_weights[lower_triangular_indices]
    lower_triangular_matrix = blocked_lower_triangular_matrix.permute(0, 2, 1, 3).reshape(T * D, T * D)

    print(torch.tensor([[1, 0], [2, 3]]).repeat_interleave(2, dim=0).repeat_interleave(2, dim=1))
    print(list(torch.randn(5, 3)))

from typing import *

import torch


"""
Manually implemented computation of the Riccati solution. Worse precision but parallelizes much faster.
"""
# def _solve_stril_equation(T: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
#     n = T.shape[-1]
#     batch_shape = T.shape[:-2]
#
#     T, E = T.view(-1, n, n), E.view(-1, n, n)                                                   # [B x N x N], [B x N x N]
#
#     stril_indices = torch.tril_indices(n, n, offset=-1)                                         # [2 x (N(N - 1) / 2)]
#     coefficients = torch.zeros((T.shape[0], (n * (n - 1)) // 2, n, n))                          # [B x (N(N - 1) / 2) x N x N]
#
#     for idx, (i, j) in enumerate(zip(*stril_indices)):
#         coefficients[:, idx, i, :i] = coefficients[:, idx, i, :i] + T[:, :i, j]                 # [B x i]
#         coefficients[:, idx, j:, j] = coefficients[:, idx, j:, j] - T[:, i, j:]                 # [B x (N - j)]
#     coefficients = coefficients[:, :, *stril_indices]                                           # [B x (N(N - 1) / 2) x (N(N - 1) / 2)]
#
#     indexed_result = (torch.inverse(coefficients) @ E[:, *stril_indices, None]).squeeze(-1)     # [B x (N(N - 1) / 2)]
#
#     result = torch.zeros_like(T)                                                                # [B x N x N]
#     result[:, *stril_indices] = indexed_result
#     return result.view(*batch_shape, n, n)


def _torch_schur(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    n = A.shape[-1]

    A_complex = torch.complex(A, torch.zeros_like(A))
    L, V = torch.linalg.eig(A_complex)                                  # [B... x N], [B... x N x N]
    order = torch.argsort(L.abs(), dim=-1)                      # [B... x N]
    sorted_L = torch.take_along_dim(L, order, dim=-1)           # [B... x N]

    P = torch.eye(n, dtype=V.dtype)[order].mT                   # [B... x N x N]
    sorted_V = V @ P                                            # [B... x N x N]

    Q, R = torch.linalg.qr(sorted_V)                            # [B... x N x N], [B... x N x N]
    D = torch.diagonal(R, dim1=-2, dim2=-1)                     # [B... x N]
    R = R / D.unsqueeze(-2)                                     # [B... x N x N] / [B... x 1 x N]

    T = R @ torch.diag_embed(sorted_L) @ torch.inverse(R)       # [B... x N x N]
    return T.real, Q.real

    # # TODO: Schur precision refinement
    # T, Qhat = T.real, Q.real
    # I = torch.eye(n)
    #
    # Q = 0.5 * (Qhat @ (3 * I - Qhat.mT @ Qhat))
    # That = Q.mT @ A @ Q
    #
    # n_iter = 1000
    # for _ in range(n_iter):
    #     E = torch.tril(That, diagonal=-1)
    #     T = That - E
    #
    #     L = _solve_stril_equation(T, E)
    #     W = L - L.mT
    #     Y = Q.H @ Q - I
    #
    #     W2 = W @ W
    #     Q = 0.5 * Q @ (2 * I - Y + W2 + W2 @ Y) @ (I + W)
    #     That = Q.mT @ A @ Q
    #
    # return That, Q


def solve_discrete_are(A: torch.Tensor, B: torch.Tensor, Q: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    batch_shape = A.shape[:-2]
    Q = 0.5 * (Q + Q.mT)
    R = 0.5 * (R + R.mT)

    m, n = B.shape[-2:]

    I = torch.eye(m).expand(*batch_shape, m, m)
    zeros = torch.zeros((*batch_shape, m, m))

    Z = torch.cat([
        torch.cat([A, zeros], dim=-1),
        torch.cat([zeros, zeros], dim=-1)
    ], dim=-2) + torch.cat([
        -B @ torch.inverse(R) @ B.mT, I
    ], dim=-2) @ torch.inverse(A.mT) @ torch.cat([
        -Q, I
    ], dim=-1)

    T, U = _torch_schur(Z)
    U_1 = U[..., :m]
    U11 = U_1[..., :m, :]
    U21 = U_1[..., m:, :]
    return U21 @ torch.inverse(U11)





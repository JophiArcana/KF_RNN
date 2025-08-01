from typing import *

import torch

from infrastructure import utils


"""
Manually implemented computation of the Riccati solution. Worse precision but parallelizes much faster.
"""


def _torch_schur(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    A_complex = torch.complex(A, torch.zeros_like(A))
    L, V = torch.linalg.eig(A_complex)                          # [B... x N], [B... x N x N]
    order = torch.argsort(L.abs(), dim=-1)                      # [B... x N]
    L = torch.take_along_dim(L, order, dim=-1)
    V = torch.vmap(torch.take_along_dim, in_dims=(-2, None), out_dims=(-2,))(V, order, dim=-1)

    Q, R = torch.linalg.qr(V)                                   # [B... x N x N], [B... x N x N]
    D = torch.diagonal(R, dim1=-2, dim2=-1)                     # [B... x N]
    R = R / D.unsqueeze(-2)                                     # [B... x N x N] / [B... x 1 x N]

    T = R @ torch.diag_embed(L) @ torch.inverse(R)              # [B... x N x N]
    return T.real, Q.real


def solve_discrete_are(A: torch.Tensor, B: torch.Tensor, Q: torch.Tensor, R: torch.Tensor, r_scale: float) -> torch.Tensor:
    bsz = B.shape[:-2]
    m, n = B.shape[-2:]

    Q = 0.5 * (Q + Q.mT)
    R = 0.5 * (R + R.mT)
    I = torch.eye(m).expand((*bsz, m, m))
    Z = torch.cat([-B @ torch.inverse(R * r_scale) @ B.mT, I], dim=-2) @ torch.inverse(A.mT) @ torch.cat([-Q, I], dim=-1)
    Z[..., :m, :m] = Z[..., :m, :m] + A

    T, U = _torch_schur(Z)
    U_1 = U[..., :m]
    U11 = U_1[..., :m, :]
    U21 = U_1[..., m:, :]
    P = U21 @ torch.inverse(U11)
    return (P + P.mT) / 2


def solve_discrete_are2(A: torch.Tensor, B: torch.Tensor, Q: torch.Tensor, R: torch.Tensor, r_scale: float) -> torch.Tensor:
    bsz = B.shape[:-2]
    m, n = B.shape[-2:]

    Q = 0.5 * (Q + Q.mT)
    R = 0.5 * (R + R.mT)
    I = torch.eye(m).expand((*bsz, m, m))
    Z = torch.cat([-B @ torch.inverse(R) @ B.mT, I * r_scale], dim=-2) @ torch.inverse(A.mT) @ torch.cat([-Q, I], dim=-1)
    Z[..., :m, :m] = Z[..., :m, :m] + (A * r_scale)
    # print("Z:")
    # print(Z)

    T, U = _torch_schur(Z)
    U_1 = U[..., :m]
    U11 = U_1[..., :m, :]
    U21 = U_1[..., m:, :]
    P = U21 @ torch.inverse(U11)
    return (P + P.mT) / 2


def solve_discrete_are_zero(A: torch.Tensor, B: torch.Tensor, Q: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    bsz = B.shape[:-2]
    m, n = B.shape[-2:]

    Q = 0.5 * (Q + Q.mT)
    R = 0.5 * (R + R.mT)
    I = torch.eye(m).expand((*bsz, m, m))
    Z = torch.cat([-B @ torch.inverse(R) @ B.mT, torch.zeros((*bsz, m, m))], dim=-2) @ torch.inverse(A.mT) @ torch.cat([-Q, I], dim=-1)
    # print("Z:")
    # print(Z)

    T, U = _torch_schur(Z)
    U_1 = U[..., :m]
    U11 = U_1[..., :m, :]
    U21 = U_1[..., m:, :]
    P = U21 @ torch.inverse(U11)
    return (P + P.mT) / 2


def check_discrete_are(P: torch.Tensor, A: torch.Tensor, B: torch.Tensor, Q: torch.tensor, R: torch.Tensor):
    assert torch.all(P == P.mT)
    PB = P @ B
    p = A.mT @ (P - PB @ torch.inverse(R + B.mT @ PB) @ PB.mT) @ A + Q
    print(P - p)


def check_discrete_are_zero(P: torch.Tensor, A: torch.Tensor, B: torch.Tensor, Q: torch.tensor):
    assert torch.all(P == P.mT)
    PB = P @ B
    p = A.mT @ (P - PB @ torch.inverse(B.mT @ PB) @ PB.mT) @ A + Q
    print(P - p)


if __name__ == "__main__":
    torch.manual_seed(1212)
    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(precision=5, sci_mode=False, linewidth=1000)

    m, n = 4, 2
    A = torch.randn((m, m))
    A *= (0.95 / torch.linalg.eigvals(A).abs().max())
    B = torch.randn((m, n))
    Q = torch.randn((m, m))
    Q = utils.sqrtm(Q @ Q.mT)
    R = torch.randn((n, n))
    R = utils.sqrtm(R @ R.mT)
    R2 = torch.randn((n, n))
    R2 = utils.sqrtm(R2 @ R2.mT)

    c = 0.00001
    # P = solve_discrete_are(A, B, Q, R, c)
    P = solve_discrete_are2(A, B, Q, R, c)
    # check_discrete_are(P, A, B, Q, R * c)
    # raise Exception()
    # P2 = solve_discrete_are2(A, B, Q, R2, c)
    P0 = solve_discrete_are_zero(A, B, Q, R)
    # P02 = solve_discrete_are_zero(A, B, Q, R2)

    # check_discrete_are(P, A, B, Q, R * c)
    # print("Cross")
    # check_discrete_are(P2, A, B, Q, R * c)
    # check_discrete_are(P, A, B, Q, R2 * c)
    # print()
    check_discrete_are(P0, A, B, Q, R * c)
    # check_discrete_are(P02, A, B, Q, R2 * c)
    # print(P2 - P)
    # check_discrete_are_zero(P, A, B, Q)
    # check_discrete_are_zero(P2, A, B, Q)









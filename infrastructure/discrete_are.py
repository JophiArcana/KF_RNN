from typing import Optional

import torch


"""
Manually implemented computation of the Riccati solution. Worse precision but parallelizes much faster.
"""
def _torch_schur(A: torch.Tensor, vectors_only: bool) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    A_complex = torch.complex(A, torch.zeros_like(A))
    L, V = torch.linalg.eig(A_complex)                          # [B... x N], [B... x N x N]
    order = torch.argsort(L.abs(), dim=-1)                      # [B... x N]
    L = torch.take_along_dim(L, order, dim=-1)
    V = torch.vmap(torch.take_along_dim, in_dims=(-2, None), out_dims=(-2,))(V, order, dim=-1)

    Q, R = torch.linalg.qr(V)                                   # [B... x N x N], [B... x N x N]
    if vectors_only:
        return Q.real
    else:
        D = torch.diagonal(R, dim1=-2, dim2=-1)                 # [B... x N]
        R = R / D.unsqueeze(-2)                                 # [B... x N x N] / [B... x 1 x N]

        T = R @ torch.diag_embed(L) @ torch.inverse(R)          # [B... x N x N]
        return T.real, Q.real


def solve_discrete_are(A: torch.Tensor, B: torch.Tensor, Q: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    bsz = B.shape[:-2]
    m, n = B.shape[-2:]

    Q = 0.5 * (Q + Q.mT)
    R = 0.5 * (R + R.mT)
    I = torch.eye(m).expand((*bsz, m, m))
    Z = torch.cat([-B @ torch.inverse(R) @ B.mT, I], dim=-2) @ torch.inverse(A.mT) @ torch.cat([-Q, I], dim=-1)
    Z[..., :m, :m] = Z[..., :m, :m] + A

    U = _torch_schur(Z, True)
    U_1 = U[..., :m]
    U11 = U_1[..., :m, :]
    U21 = U_1[..., m:, :]
    return U21 @ torch.inverse(U11)





import scipy as sc
import torch


def torch_schur(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


if __name__ == "__main__":
    M = torch.tensor([
        [1., 1., 0.,],
        [0., 1., 1.,],
        [0., 0., 1.,],
    ])

    L, V = torch_schur(M)
    print(L)
    print(V)

    sL, sV = sc.linalg.schur(M)
    print(sL)
    print(sV)




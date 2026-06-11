import scipy as sc
import torch

from kf_rnn.infrastructure.discrete_are import _torch_schur


def torch_schur(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _torch_schur(A, vectors_only=False)


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




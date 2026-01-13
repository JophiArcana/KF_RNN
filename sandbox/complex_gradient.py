import torch
import torch.nn as nn


if __name__ == "__main__":
    torch.set_printoptions(linewidth=500, sci_mode=False)
    torch.manual_seed(12122002)

    G = torch.randn((3, 7,), dtype=torch.complex64)
    A = nn.Parameter(torch.randn((3, 5,), dtype=torch.complex64))
    B = nn.Parameter(torch.randn((5, 7,), dtype=torch.complex64))
    D = nn.Parameter(torch.randn((5,), dtype=torch.complex64))

    g = torch.sum(G.real * (A @ B).real + G.imag * (A @ B).imag)

    # print(torch.autograd.grad(torch.sum(B.real + B.imag), B))

    # print(torch.autograd.grad(g, B))
    # print(A.mH @ G)

    # print(torch.autograd.grad(B.abs().norm() ** 2, B))
    # print(2 * B)

    # print(torch.autograd.grad((A @ B).abs().norm() ** 2, A))
    # print(2 * (A @ B @ B.H))

    print(torch.autograd.grad(torch.sum((D ** 7).real + (D ** 7).imag), D))
    print(7 * (D.conj() ** 6 * torch.complex(torch.tensor(1.), torch.tensor(1.))))



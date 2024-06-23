from typing import *

import scipy as sc
import torch
import torch.nn as nn

from infrastructure import utils


"""
SciPy and backward method of computing the Riccati solution. Better precision but unparallelizable.
"""
def V_pert(n):
    idx = torch.arange(n * n)
    return torch.eye(n * n)[idx // n + n * (idx % n)]


class Riccati(torch.autograd.Function):
    @staticmethod  # FORWARDS PASS
    def forward(ctx: Any, A: torch.Tensor, B: torch.Tensor, Q: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        if not (A.type() == B.type() and A.type() == Q.type() and A.type() == R.type()):
            raise Exception('A, B, Q, and R must be of the same type.')

        Q = 0.5 * (Q + Q.T)
        R = 0.5 * (R + R.T)
        P = torch.from_numpy(sc.linalg.solve_discrete_are(
            A.detach().cpu(),
            B.detach().cpu(),
            Q.detach().cpu(),
            R.detach().cpu()
        )).type(A.type())

        ctx.save_for_backward(P, A, B, Q, R)  # Save variables for backwards pass
        return P

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        def _kron(M: torch.Tensor) -> torch.Tensor:
            return torch.kron(M, M)

        grad_output = grad_output.T.flatten()[None]
        P, A, B, Q, R = ctx.saved_tensors
        n, m = B.shape

        # Computes derivatives using method detailed in paper
        M2 = (R + B.mT @ (PB := P @ B)).inverse()
        M1 = P - (PBM2BT := (PBM2 := PB @ M2) @ B.T) @ P
        I, In2 = torch.eye(n), torch.eye(n ** 2)

        LHS = _kron(P @ B) @ _kron(M2) @ _kron(B.T)
        LHS = LHS - torch.kron(I, PBM2BT) - torch.kron(PBM2BT, I) + In2
        LHS = In2 - _kron(A.T) @ LHS
        invLHS = torch.inverse(LHS)

        RHS = V_pert(n) + In2
        dA = invLHS @ RHS @ torch.kron(I, A.mT @ M1)
        dA = grad_output @ dA
        dA = dA.view(n, n).T

        RHS = torch.kron(torch.eye(m), B.mT @ P)
        RHS = (torch.eye(m ** 2) + V_pert(m)) @ RHS
        RHS = _kron(PB) @ _kron(M2) @ RHS
        RHS = RHS - (In2 + V_pert(n)) @ (torch.kron(PBM2, P))
        dB = invLHS @ _kron(A.T) @ RHS
        dB = grad_output @ dB
        dB = dB.view(m, n).T

        dQ = (grad_output @ invLHS).view(n, n)
        dQ = 0.5 * (dQ + dQ.T)

        RHS = _kron(A.T) @ _kron(PB) @ _kron(M2)
        dR = invLHS @ RHS
        dR = (grad_output @ dR).view(m, m)
        dR = 0.5 * (dR + dR.T)

        return dA, dB, dQ, dR


def initialize():
    A = nn.Parameter(utils.sample_stable_state_matrix(6))
    B = nn.Parameter(0.1 * torch.randn((6, 4)))
    Q = nn.Parameter(0.1 * torch.randn((6, 6)))
    R = nn.Parameter(0.1 * torch.randn((4, 4)))

    return A, B, Q, R


def test_gradients():
    torch.set_default_dtype(torch.float64)
    A, B, Q, R = initialize()

    torch.autograd.gradcheck(Riccati.apply, (A, B, Q, R), raise_exception=True)


def test_interface():
    torch.set_default_dtype(torch.float64)
    A, B, Q, R = initialize()

    P = Riccati.apply(A, B, Q, R)
    P = P.sum()
    P.backward()

    P = Riccati.apply(A, B, Q, R)
    P = P.sum()
    P.backward()


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





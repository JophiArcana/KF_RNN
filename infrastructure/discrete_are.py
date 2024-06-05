from typing import *

import scipy as sc
import torch
import torch.nn as nn

from infrastructure import utils


def V_pert(n):
    """ Form the V_{m,n} perturbation matrix as defined in the paper

    Args:
        n

    Returns:
        V_{m,n}
    """
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

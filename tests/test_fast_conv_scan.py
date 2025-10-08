import torch
import torch.nn as nn

from infrastructure.fast_conv_scan import conv_scan
from infrastructure.utils import Timer


if __name__ == "__main__":
    from torch.utils._pytree import tree_flatten

    torch.manual_seed(1212)
    torch.set_printoptions(linewidth=500, sci_mode=False, precision=6)
    torch.set_default_dtype(torch.float64)
    torch.set_default_device("cuda")

    bsz, l, c = (400,), 1000, 16
    # bsz, l, c = (), 11, 16
    logA = nn.Parameter(0.01 * torch.randn(bsz + (l,)))
    B = nn.Parameter(torch.randn(bsz + (l + 1,)))

    T = Timer()
    n_trials = 100

    T.reset()
    for _ in range(n_trials):
        out1 = conv_scan(logA, B, c)
        grad1 = tree_flatten(torch.autograd.grad(out1.sum(), (logA, B,)))[0]
    print("fast_conv_scan took {:.6f}s".format(T.reset() / n_trials))

    T.reset()
    for _ in range(n_trials):
        ssm_state = B[..., 0]
        states = [ssm_state]
        for i in range(l):
            ssm_state = torch.exp(logA[..., i]) * ssm_state + B[..., i + 1]
            states.append(ssm_state)
        out2 = torch.stack(states, dim=-1)
        grad2 = tree_flatten(torch.autograd.grad(out2.sum(), (logA, B,)))[0]
    print("naive took {:.6f}s".format(T.reset() / n_trials))

    print((out1 / out2 - 1).abs().max(), (out1 - out2).abs().max())


    print([t.norm() for t in grad1])
    print([t.norm() for t in grad2])

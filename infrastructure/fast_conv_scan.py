import einops
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from infrastructure import utils


import time
class Timer:
    def __init__(self):
        self.t = time.perf_counter()

    def reset(self):
        t = time.perf_counter()
        out = t - self.t
        self.t = t
        return out


def exp_segment_sum(x: torch.Tensor):
    """
    More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
    """
    *bsz, seqlen = x.shape
    # 1. expand input tensor to have an additional dimension and repeat along that dimension
    # [..., seqlen] -> [..., seqlen, seqlen]
    # x = x[..., None].expand(bsz + [seqlen, seqlen,])
    padded_x = torch.zeros(bsz + [seqlen + 1, seqlen + 1,], dtype=x.dtype)
    padded_x[..., 1:, :-1] = x[..., None]

    # 2. create a lower triangular mask with the diagonal set to 0 to 0 out elements above diag
    padded_x.tril_(diagonal=-1)

    # 3. compute actual cumsum
    segsum = padded_x.cumsum_(dim=-2)

    # 4. apply mask to keep only the lower triangular part of the cumulative sum result (incl diagonal this time)
    exp_segsum = segsum.exp_()
    exp_segsum.tril_(diagonal=0)
    return segsum


class ConvScanFn(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            A: torch.Tensor,        # float: [... x L]
            B: torch.Tensor,        # float: [... x (L + 1)]
            chunk_size: int,        # int: C
    ) -> torch.Tensor:              # float: [... x (L + 1)]
        with torch.no_grad():
            out = _conv_scan_fwd(A, B, chunk_size=chunk_size)
        ctx.save_for_backward(A, out)
        ctx.chunk_size = chunk_size
        return out

    @staticmethod
    def backward(
            ctx,
            dout: torch.Tensor,     # float: [... x (L + 1)]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.set_default_device(dout.device)
        A, out = ctx.saved_tensors
        with torch.no_grad():
            dA, dB = _conv_scan_bwd(dout, {"A": A, "out": out,}, ctx.chunk_size)
        return dA, dB, None


def _conv_scan_fwd(
        A: torch.Tensor,    # float: [... x L]
        B: torch.Tensor,    # float: [... x (L + 1)]
        chunk_size: int,    # int: C
) -> torch.Tensor:          # float: [... x (L + 1)]
    L, C = A.shape[-1], chunk_size

    if L < C:
        exp_A_ss = exp_segment_sum(A)                                                       # float: [... x (L + 1) x (L + 1)]
        out = einops.einsum(exp_A_ss, B, "... l1 l2, ... l2 -> ... l1")                     # float: [... x (L + 1)]
    else:
        t_dict_start = utils.get_tensors_in_memory()

        p = (L + C) // C * C
        padded_A = Fn.pad(A, (0, p - L,), mode="constant", value=0.0)                       # float: [... x ~L]
        exp_A_ss = exp_segment_sum(padded_A.unflatten(-1, (-1, C,))[..., :-1])              # float: [... x (~L // C) x C x C]

        B = Fn.pad(B, (0, p - L - 1,), mode="constant", value=0.0).unflatten(-1, (-1, C,))  # float: [... x (~L // C) x C]
        out_diag = einops.einsum(exp_A_ss, B, "... c1 c2, ... c2 -> ... c1")                # float: [... x (~L // C) x C]

        out_lr = padded_A[..., C - 1:-1].unflatten(-1, (-1, C,)).cumsum_(dim=-1)            # float: [... x (L // C) x C]
        c_ss = exp_segment_sum(out_lr[..., :-1, -1])                                        # float: [... x (L // C) x (L // C)]
        r_ss = out_diag[..., :L // C, -1]                                                   # float: [... x (L // C)]
        out_lr.exp_()

        out_lr *= (c_ss @ r_ss[..., None])
        out_diag[..., 1:, :] += out_lr

        out = out_diag.flatten(-2, -1)[..., :L + 1]                                         # float: [... x (L + 1)]

        t_dict_end = utils.get_tensors_in_memory()

        diff = {k: v.shape for k, v in t_dict_end.items() if k not in t_dict_start}
        local_names = {id(v): k for k, v in locals().items()}
        print(diff.values())
        print()



    return out


def _conv_scan_bwd(
        dout: torch.Tensor,     # float: [... x (L + 1)]
        cache: dict[str, torch.Tensor],
        chunk_size: int,        # int: C
) -> tuple[torch.Tensor, torch.Tensor]:     # float: [... x L], [... x (L + 1)]
    L, C = dout.shape[-1] - 1, chunk_size

    A = cache["A"]                                                                                  # float: [... x L]
    dB = _conv_scan_fwd(A.flip(dims=(-1,)), dout, chunk_size=chunk_size).flip(dims=(-1,))           # float: [... x (L + 1)]
    # print(dB.squeeze())
    #
    # exp_A_ss = cache["exp_A_ss"]            # float: [... x (L + 1) x (L + 1)] or [... x (~L // C) x C x C]
    out = cache["out"]                      # float: [... x (L + 1)]
    # if L < C:
    #     dB = einops.einsum(exp_A_ss, dout, "... l1 l2, ... l1 -> ... l2")                           # float: [... x (L + 1)]
    # else:
    #     p = (L + C) // C * C
    #     padded_A = cache["padded_A"]                                                                # float: [... x ~L]
    #
    #     dout = Fn.pad(dout, (0, p - L - 1,), mode="constant", value=0.0).unflatten(-1, (-1, C,))    # float: [... x (~L // C) x C]
    #     dB_diag = einops.einsum(exp_A_ss, dout, "... c1 c2, ... c1 -> ... c2")                      # float: [... x (~L // C) x C]
    #
    #     l_ss = padded_A[..., :-C].unflatten(-1, (-1, C,)).flip(dims=(-1,)).cumsum(dim=-1).flip(dims=(-1,))  # float: [... x (L // C) x C]
    #     c_ss = exp_segment_sum(l_ss[..., 1:, 0])                                                    # float: [... x (L // C) x (L // C)]
    #     r_ss = dB_diag[..., -(L // C):, 0]                                                          # float: [... x (L // C)]
    #
    #     dB_lr = einops.einsum(torch.exp(l_ss), c_ss, r_ss, "... l1 c, ... l2 l1, ... l2 -> ... l1 c")  # float: [... x (~L // C - 1) x C]
    #     dB_diag[..., :-1, :] += dB_lr
    #
    #     dB = dB_diag.flatten(-2, -1)[..., :L + 1]                                                   # float: [... x (L + 1)]
    #     print(dB.squeeze())
    #     raise Exception()

    exp_A = torch.exp(A)                                                                            # float: [... x L]
    dA = einops.einsum(dB[..., 1:], out[..., :-1], exp_A, "..., ..., ... -> ...")                   # float: [... x L]
    return dA, dB


def conv_scan(
        A: torch.Tensor,    # float: [B x L x H x D]
        B: torch.Tensor,    # float: [B x L x H x D]
        chunk_size: int,    # int: C
) -> torch.Tensor:          # float: [B x L x H x D]
    return ConvScanFn.apply(A, B, chunk_size)






if __name__ == "__main__":
    from torch.utils._pytree import tree_flatten

    torch.manual_seed(1212)
    torch.set_printoptions(linewidth=500, sci_mode=False, precision=6)
    torch.set_default_dtype(torch.float64)
    torch.set_default_device("cuda")

    # bsz, l, h, d, c = 3, 1000, 8, 32, 256
    bsz, l, c = (400,), 1000, 16
    # bsz, l, c = (), 11, 5
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








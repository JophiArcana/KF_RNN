import einops
import torch
import torch.nn as nn
import torch.nn.functional as Fn


import time
class Timer:
    def __init__(self):
        self.t = time.perf_counter()

    def reset(self):
        t = time.perf_counter()
        out = t - self.t
        self.t = t
        return out


def segment_sum(x):
    """
    More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
    """
    *bsz, seqlen = x.shape
    # 1. expand input tensor to have an additional dimension and repeat along that dimension
    # [..., seqlen] -> [..., seqlen, seqlen]
    # x = x[..., None].expand(bsz + [seqlen, seqlen,])
    padded_x = torch.zeros(bsz + [seqlen + 1, seqlen + 1,])
    padded_x[..., 1:, :-1] = x[..., None]

    # 2. create a lower triangular mask with the diagonal set to 0 to 0 out elements above diag
    padded_x.tril_(diagonal=-1)

    # 3. compute actual cumsum
    segsum = torch.cumsum(padded_x, dim=-2)

    # 4. apply mask to keep only the lower triangular part of the cumulative sum result (incl diagonal this time)
    mask = torch.triu(torch.full((seqlen + 1, seqlen + 1,), True), diagonal=1)
    segsum.masked_fill_(mask, -torch.inf)
    return segsum


def exp_segment_sum(x):
    """
    More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
    """
    *bsz, seqlen = x.shape
    # 1. expand input tensor to have an additional dimension and repeat along that dimension
    # [..., seqlen] -> [..., seqlen, seqlen]
    # x = x[..., None].expand(bsz + [seqlen, seqlen,])
    padded_x = torch.zeros(bsz + [seqlen + 1, seqlen + 1,])
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
            A: torch.Tensor,        # float: [B x L x H x D]
            B: torch.Tensor,        # float: [B x (L + 1) x H x D]
            chunk_size: int,        # int: C
    ) -> torch.Tensor:              # float: [B x L x H x D]
        with torch.no_grad():
            out, cache = _conv_scan_fwd(A, B, chunk_size)
        ctx.cache = cache
        ctx.chunk_size = chunk_size
        return out

    @staticmethod
    def backward(
            ctx,
            dout: torch.Tensor,     # float: [B x L x H x D]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            dA, dB = _conv_scan_bwd(dout, ctx.cache, ctx.chunk_size)
        return dA, dB, None







def _conv_scan_fwd(
        A: torch.Tensor,    # float: [B x L x H x D]
        B: torch.Tensor,    # float: [B x (L + 1) x H x D]
        chunk_size: int,    # int: C
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:      # float: [B x L x H x D]
    *bsz, L, H, D = torch.broadcast_shapes(A.shape, B[..., :-1, :, :].shape)
    C = chunk_size

    A = einops.rearrange(A, "... l h d -> ... h d l")   # float: [B x H x D x L]
    cache = {}
    if L < C:
        exp_A_ss = exp_segment_sum(A)                                                       # float: [B x H x D x (L + 1) x (L + 1)]
        out = einops.einsum(exp_A_ss, B, "... h d l1 l2, ... l2 h d -> ... l1 h d")[..., 1:, :, :]  # float: [B x L x H x D]
    else:
        p = (L + C) // C * C
        padded_A = Fn.pad(A, (0, p - L,), mode="constant", value=0.0)                       # float: [B x H x D x ~L]
        exp_A_ss = exp_segment_sum(padded_A.unflatten(-1, (-1, C,))[..., :-1])              # float: [B x H x D x (~L // C) x C x C]

        B = torch.cat((B, torch.zeros(bsz + [p - L - 1, H, D,]),), dim=-3)                  # float: [B x ~L x H x D]
        B = einops.rearrange(B, "... (l c) h d -> ... h d l c", c=C)                        # float: [B x H x D x (~L // C) x C]

        out_diag = einops.einsum(exp_A_ss, B, "... c1 c2, ... c2 -> ... c1")                # float: [B x H x D x (~L // C) x C]

        l_ss = padded_A[..., C - 1:-1].unflatten(-1, (-1, C,)).cumsum(dim=-1)              # float: [B x H x D x (L // C) x C]
        c_ss = exp_segment_sum(l_ss[..., :-1, -1])                                          # float: [B x H x D x (L // C) x (L // C)]
        r_ss = out_diag[..., :L // C, -1]                                                   # float: [B x H x D x (L // C)]

        out_lr = einops.einsum(torch.exp(l_ss), c_ss, r_ss, "... l1 c, ... l1 l2, ... l2 -> ... l1 c")  # float: [B x H x D x (~L // C - 1) x C]
        out_diag[..., 1:, :] += out_lr

        out = out_diag.flatten(-2, -1)[..., 1:L + 1]                                        # float: [B x H x D x L]
        out = einops.rearrange(out, "... h d l -> ... l h d")                               # float: [B x L x H x D]

        cache.update({"padded_A": padded_A,})

    cache.update({
        "exp_A_ss": exp_A_ss,
        "B": B,
    })
    return out, cache


def _conv_scan_bwd(
        dout: torch.Tensor,     # float: [B x L x H x D]
        cache: dict[str, torch.Tensor],
        chunk_size: int,        # int: C
) -> tuple[torch.Tensor, torch.Tensor]:   # float: [B x H x D], [B x L x H x D], [B x L x H x D]
    *bsz, L, H, D = dout.shape
    C = chunk_size

    exp_A_ss = cache["exp_A_ss"]            # float: [B x H x D x (L + 1) x (L + 1)] or [B x H x (~L // C) x C x C]
    B = cache["B"]                          # float: [B x (L + 1) x H x D] or [B x H x (~L // C) x C]
    if L < C:
        exp_A_ss = exp_A_ss[..., 1:, :]                                                     # float: [B x H x D x L x (L + 1)]
        dB = einops.einsum(exp_A_ss, dout, "... h d l1 l2, ... l1 h d -> ... l2 h d")       # float: [B x (L + 1) x H x D]

        dA_ss = einops.einsum(dout, B, exp_A_ss, "... l1 h d, ... l2 h d, ... h d l1 l2 -> ... h d l1 l2")   # float: [B x H x D x L x (L + 1)]
        dA_ss.cumsum_(dim=-1)
        dA_ss.tril_(diagonal=0)
        dA = einops.einsum(dA_ss[..., :-1], "... h d l1 l2 -> ... l2 h d")                  # float: [B x L x H x D]
    else:
        p = (L + C) // C * C
        padded_A = cache["padded_A"]                                                        # float: [B x H x D x ~L]
        dout = Fn.pad(dout, (0, 0, 0, 0, 1, p - L - 1,), mode="constant", value=0.0)        # float: [B x ~L x H x D]
        dout = einops.rearrange(dout, "... (l c) h d -> ... h d l c", c=C)                  # float: [B x H x D x (~L // C) x C]

        dB_diag = einops.einsum(exp_A_ss, dout, "... c1 c2, ... c1 -> ... c2")              # float: [B x H x D x (~L // C) x C]

        l_ss = padded_A[..., :-C].unflatten(-1, (-1, C,)).flip(dims=(-1,)).cumsum(dim=-1).flip(dims=(-1,))  # float: [B x H x D x (L // C) x C]
        c_ss = exp_segment_sum(l_ss[..., 1:, 0])                                            # float: [B x H x D x (L // C) x (L // C)]
        r_ss = dB_diag[..., -(L // C):, 0]                                                  # float: [B x H x D x (L // C)]

        dB_lr = einops.einsum(torch.exp(l_ss), c_ss, r_ss, "... l1 c, ... l2 l1, ... l2 -> ... l1 c")  # float: [B x H x D x (~L // C - 1) x C]
        dB_diag[..., :-1, :] += dB_lr

        dB = dB_diag.flatten(-2, -1)[..., :L + 1]                                           # float: [B x H x D x (L + 1)]
        dB = einops.rearrange(dB, "... h d l -> ... l h d")                                 # float: [B x (L + 1) x H x D]






        dA = torch.zeros(bsz + [L, H, D,])

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
    # torch.set_default_device("cuda")

    # bsz, l, h, d, c = 3, 1000, 8, 32, 256
    bsz, l, h, d, c = 1, 124, 3, 1, 17
    logA = nn.Parameter(0.01 * torch.randn((bsz, l, h, d,)))
    B = nn.Parameter(torch.randn((bsz, l + 1, h, d,)))

    T = Timer()
    n_trials = 100

    T.reset()
    for _ in range(n_trials):
        out1 = conv_scan(logA, B, c)
        grad1 = tree_flatten(torch.autograd.grad(out1.sum(), (logA, B,)))[0]
    print("fast_conv_scan took {:.6f}s".format(T.reset() / n_trials))

    T.reset()
    for _ in range(n_trials):
        ssm_state = B[:, 0, :, :]
        states = []
        for i in range(l):
            ssm_state = torch.exp(logA[:, i, :, :]) * ssm_state + B[:, i + 1, :, :]
            states.append(ssm_state)
        out2 = torch.stack(states, dim=1)
        grad2 = tree_flatten(torch.autograd.grad(out2.sum(), (logA, B,)))[0]
    print("naive took {:.6f}s".format(T.reset() / n_trials))

    print((out1 / out2 - 1).abs().max(), (out1 - out2).abs().max())


    print([t.norm() for t in grad1])
    print([t.norm() for t in grad2])








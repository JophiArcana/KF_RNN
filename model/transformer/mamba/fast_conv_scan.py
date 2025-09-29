import einops
import torch
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
    segsum = torch.cumsum(padded_x, dim=-2)

    # 4. apply mask to keep only the lower triangular part of the cumulative sum result (incl diagonal this time)
    exp_segsum = torch.exp(segsum)
    exp_segsum.tril_(diagonal=0)
    return exp_segsum



def conv_scan(
        x: torch.Tensor,    # float: [B x H x D]
        A: torch.Tensor,    # float: [B x L x H x D]
        B: torch.Tensor,    # float: [B x L x H x D]
        chunk_size: int,    # int: C
) -> torch.Tensor:          # float: [B x L x H x D]
    *bsz, L, H, D = torch.broadcast_shapes(A.shape, B.shape)
    C = chunk_size

    A = einops.rearrange(A, "... l h d -> ... h d l")  # float: [B x H x D x L]
    if L < C:
        exp_A_ss = exp_segment_sum(A)
        x = torch.cat((x[..., None, :, :], B,), dim=-3)                                     # float: [B x (L + 1) x H x D]
        out = einops.einsum(exp_A_ss, x, "... h d l1 l2, ... l2 h d -> ... l1 h d")[..., 1:, :, :]  # float: [B x L x H x D]

    else:
        p = (L + C) // C * C
        padded_A = Fn.pad(A, (0, p - L,), mode="constant", value=0.0)                       # float: [B x H x D x ~L]
        exp_A_ss = exp_segment_sum(padded_A.unflatten(-1, (-1, C,))[..., :-1])              # float: [B x H x D x (~L // C) x C x C]
        print(exp_A_ss.shape)

        padded_x = torch.cat((x[..., None, :, :], B, torch.zeros(bsz + [p - L - 1, H, D,]),), dim=-3)   # float: [B x ~L x H x D]
        x_diag = einops.rearrange(padded_x, "... (l c) h d -> ... h d l c", c=C)            # float: [B x H x D x (~L // C) x C]

        out_diag = einops.einsum(exp_A_ss, x_diag, "... c1 c2, ... c2 -> ... c1")           # float: [B x H x D x (~L // C) x C]
        out_diag = out_diag.flatten(-2, -1)[..., :L + 1]                                    # float: [B x H x D x (L + 1)]

        l_ss = torch.cumsum(padded_A[..., C - 1:-1].unflatten(-1, (-1, C,)), -1)            # float: [B x H x D x (L // C) x C]
        c_ss = exp_segment_sum(l_ss[..., :-1, -1])                                          # float: [B x H x D x (L // C) x (L // C)]
        r_ss = out_diag[..., C - 1:L:C]                                                     # float: [B x H x D x (L // C)]

        out_lr = einops.einsum(
            torch.exp(l_ss[..., :, None, :]) * c_ss[..., :, :, None], r_ss,
            "... l1 l2 c, ... l2 -> ... l1 c",
        ).flatten(-2, -1)[..., :L + 1 - C]                                                  # float: [B x H x D x (L + 1 - C)]
        out = out_diag + torch.cat((torch.zeros(bsz + [H, D, C,]), out_lr,), dim=-1)        # float: [B x H x D x (L + 1)]
        out = einops.rearrange(out[..., 1:], "... h d l -> ... l h d")

    return out








if __name__ == "__main__":
    torch.manual_seed(1212)
    torch.set_default_dtype(torch.float64)
    torch.set_default_device("cuda")

    bsz, l, h, d, c = 3, 1000, 8, 32, 256
    # bsz, l, h, d, c = 1, 11, 1, 1, 5
    x = torch.randn((bsz, h, d,))
    logA = 0.01 * torch.randn((bsz, l, h, d,))
    B = torch.randn((bsz, l, h, d,))

    T = Timer()
    n_trials = 100

    T.reset()
    for _ in range(n_trials):
        out1 = conv_scan(x, logA, B, c)
    print("fast_conv_scan took {:.6f}s".format(T.reset() / n_trials))

    T.reset()
    for _ in range(n_trials):
        ssm_state = x
        states = []
        for i in range(l):
            ssm_state = torch.exp(logA[:, i, :, :]) * ssm_state + B[:, i, :, :]
            states.append(ssm_state)
        out2 = torch.stack(states, dim=1)
    print("naive took {:.6f}s".format(T.reset() / n_trials))

    print((out1 / out2 - 1).abs().max(), (out1 - out2).abs().max())








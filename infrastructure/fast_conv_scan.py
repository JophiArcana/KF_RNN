import einops
import torch
import torch.nn.functional as Fn

from infrastructure import utils


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


def _conv_scan_fwd_base2(
        A: torch.Tensor,    # float: [... x L]
        B: torch.Tensor,    # float: [... x (L + 1)]
) -> torch.Tensor:          # float: [... x (L + 1)]
    exp_A_ss = exp_segment_sum(A)                                                       # float: [... x (L + 1) x (L + 1)]
    out = einops.einsum(exp_A_ss, B, "... l1 l2, ... l2 -> ... l1")                     # float: [... x (L + 1)]

    return out


def _conv_scan_fwd_base(
        A: torch.Tensor,    # float: [... x L]
        B: torch.Tensor,    # float: [... x (L + 1)]
) -> torch.Tensor:          # float: [... x (L + 1)]
    *bsz, L = A.shape
    A_, out = torch.ones_like(B), B.clone()     # float: [... x (L + 1)]
    torch.exp(A, out=A_[..., 1:])               # float: [... x (L + 1)]

    k = 1
    while k <= L:
        out[..., k:].addcmul_(A_[..., k:], out[..., :-k].clone())
        A_[..., k:] *= A_[..., :-k].clone()
        k <<= 1
    return out


def _conv_scan_fwd(
        A: torch.Tensor,    # float: [... x L]
        B: torch.Tensor,    # float: [... x (L + 1)]
        chunk_size: int,    # int: C
) -> torch.Tensor:          # float: [... x (L + 1)]
    L, C = A.shape[-1], chunk_size
    if True: # L < C:
        out = _conv_scan_fwd_base(A, B)                                                     # float: [... x (L + 1)]
    else:
        p = (L + C) // C * C
        padded_A = Fn.pad(A, (0, p - L,), mode="constant", value=0.0)                       # float: [... x ~L]

        out_diag = _conv_scan_fwd_base(
            padded_A.unflatten(-1, (-1, C,))[..., :-1],
            Fn.pad(B, (0, p - L - 1,), mode="constant", value=0.0).unflatten(-1, (-1, C,)), # float: [... x (~L // C) x C]
        )                                                                                   # float: [... x (~L // C) x C]

        out_lr = padded_A[..., C - 1:-1].unflatten(-1, (-1, C,)).cumsum_(dim=-1)            # float: [... x (L // C) x C]
        out_lr_r = _conv_scan_fwd_base(out_lr[..., :-1, -1], out_diag[..., :L // C, -1])    # float: [... x (L // C)]
        out_lr.exp_()
        out_lr *= out_lr_r[..., None]
        out_diag[..., 1:, :] += out_lr

        out = out_diag.flatten(-2, -1)[..., :L + 1]                                         # float: [... x (L + 1)]
    return out


def _conv_scan_bwd(
        dout: torch.Tensor,     # float: [... x (L + 1)]
        cache: dict[str, torch.Tensor],
        chunk_size: int,        # int: C
) -> tuple[torch.Tensor, torch.Tensor]:     # float: [... x L], [... x (L + 1)]
    A = cache["A"]                                                                          # float: [... x L]
    dB = _conv_scan_fwd(A.flip(dims=(-1,)), dout, chunk_size=chunk_size).flip(dims=(-1,))   # float: [... x (L + 1)]
    out = cache["out"]                      # float: [... x (L + 1)]

    exp_A = torch.exp(A)                                                                    # float: [... x L]
    dA = einops.einsum(dB[..., 1:], out[..., :-1], exp_A, "..., ..., ... -> ...")           # float: [... x L]
    return dA, dB


def conv_scan(
        A: torch.Tensor,    # float: [B x L x H x D]
        B: torch.Tensor,    # float: [B x L x H x D]
        chunk_size: int,    # int: C
) -> torch.Tensor:          # float: [B x L x H x D]
    return ConvScanFn.apply(A, B, chunk_size)





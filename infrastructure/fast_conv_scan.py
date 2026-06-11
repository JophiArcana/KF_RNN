import einops
import torch


class ConvScanFn(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            A: torch.Tensor,        # float: [... x L]
            B: torch.Tensor,        # float: [... x (L + 1)]
            chunk_size: int,        # int: C (accepted for API compatibility; unused)
    ) -> torch.Tensor:              # float: [... x (L + 1)]
        with torch.no_grad():
            out = _conv_scan_fwd(A, B)
        ctx.save_for_backward(A, out)
        return out

    @staticmethod
    def backward(
            ctx,
            dout: torch.Tensor,     # float: [... x (L + 1)]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        A, out = ctx.saved_tensors
        with torch.no_grad():
            dA, dB = _conv_scan_bwd(dout, {"A": A, "out": out,})
        return dA, dB, None


def _conv_scan_fwd(
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


def _conv_scan_bwd(
        dout: torch.Tensor,     # float: [... x (L + 1)]
        cache: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:     # float: [... x L], [... x (L + 1)]
    A = cache["A"]                                                          # float: [... x L]
    dB = _conv_scan_fwd(A.flip(dims=(-1,)), dout).flip(dims=(-1,))          # float: [... x (L + 1)]
    out = cache["out"]                      # float: [... x (L + 1)]

    exp_A = torch.exp(A)                                                    # float: [... x L]
    dA = einops.einsum(dB[..., 1:], out[..., :-1], exp_A, "..., ..., ... -> ...")   # float: [... x L]
    return dA, dB


def conv_scan(
        A: torch.Tensor,    # float: [B x L x H x D]
        B: torch.Tensor,    # float: [B x L x H x D]
        chunk_size: int,    # int: C (accepted for API compatibility; unused)
) -> torch.Tensor:          # float: [B x L x H x D]
    return ConvScanFn.apply(A, B, chunk_size)

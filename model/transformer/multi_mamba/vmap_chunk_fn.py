# Copyright (c) 2024, Tri Dao, Albert Gu.

"""We want triton==2.1.0 or 2.2.0 for this
"""

import math
from packaging import version

import einops
import torch
import torch.nn.functional as F
from mamba_ssm.ops.triton.ssd_bmm import (
    _bmm_chunk_fwd_kernel,
    _bmm_chunk_bwd_kernel,
)
from mamba_ssm.ops.triton.ssd_chunk_scan import (
    _chunk_scan_bwd_dstates_kernel,
    _chunk_scan_bwd_dc_kernel,
    _chunk_scan_bwd_ddAcs_stable_kernel,
)
from mamba_ssm.ops.triton.ssd_combined import _chunk_scan_chunk_state_bwd_dx_kernel

import triton
import triton.language as tl

TRITON_22 = version.parse(triton.__version__) >= version.parse('2.2.0')


def vmap_bmm_chunk_fwd(
        a_list: torch.Tensor,
        b: torch.Tensor,
        chunk_size: int,
        seq_idx: torch.Tensor = None,
        causal: bool = False,
        output_dtype: torch.dtype = None,
):
    """
    Argument:
        a: (vmap, batch, seqlen, k) or (vmap, batch, seqlen, ngroups, k)
        b: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
        seq_idx: (batch, seqlen) or None. out[i, j] for seq_idx[i] != seq_idx[j] will be zeroed out.
        causal: if True, then out[i, j] for i > j will be arbitrary, only out[i, j] for i <= j are
            guaranteed to be correct.
    Return:
        out: (batch, nchunks, chunk_size, chunk_size) or (batch, nchunks, ngroups, chunk_size, chunk_size)
    """
    # Check constraints.
    has_groups = (a_list.dim() == 5)
    if not has_groups:
        vmap, batch, seqlen, k = a_list.shape
    else:
        vmap, batch, seqlen, ngroups, k = a_list.shape
    assert b.shape == a_list.shape[1:]
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if a_list.stride(-1) != 1 and a_list.stride(1) != 1:
        a_list = a_list.contiguous()
    if b.stride(-1) != 1 and b.stride(1) != 1:
        b = b.contiguous()
    nchunks = math.ceil(seqlen / chunk_size)
    # Allocates output.
    out_dtype = a_list.dtype if output_dtype is None else output_dtype
    dot_dtype = (tl.bfloat16 if a_list.dtype == torch.bfloat16 or b.dtype == torch.bfloat16 else
                 (tl.float16 if a_list.dtype == torch.float16 or b.dtype == torch.float16 else tl.float32))
    grid = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(chunk_size, META['BLOCK_SIZE_N']),
                    batch, nchunks if not has_groups else nchunks * ngroups)
    
    out_list: list[torch.Tensor] = []
    with torch.cuda.device(a_list.device.index):
        for _a in a_list:
            _out = torch.empty((batch, nchunks, *(() if not has_groups else (ngroups,)), chunk_size, chunk_size), device=a_list.device, dtype=out_dtype)
            _bmm_chunk_fwd_kernel[grid](
                _a, b, _out, seq_idx,
                seqlen, chunk_size, k, ngroups if has_groups else 1,
                _a.stride(0), _a.stride(1), 0 if not has_groups else _a.stride(2), _a.stride(-1),
                b.stride(0), b.stride(1), 0 if not has_groups else b.stride(2), b.stride(-1),
                _out.stride(0), _out.stride(1), 0 if not has_groups else _out.stride(2), _out.stride(-2), _out.stride(-1),
                *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
                causal,
                dot_dtype,
                HAS_SEQ_IDX=seq_idx is not None,
            )
            out_list.append(_out)
    out = torch.stack(out_list, dim=0)
    return out


def vmap_chunk_scan_bwd_dstates(
        C_list: torch.Tensor,
        dA_cumsum: torch.Tensor,
        dout: torch.Tensor,
        seq_idx: torch.Tensor = None,
        dtype: torch.dtype = None,
):
    batch, seqlen, nheads, headdim = dout.shape
    _, _, nchunks, chunk_size = dA_cumsum.shape
    _, _, _, ngroups, dstate = C_list.shape
    assert nheads % ngroups == 0
    assert C_list.shape[1:] == (batch, seqlen, ngroups, dstate)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    dtype = C_list.dtype if dtype is None else dtype
    dstates = torch.zeros((batch, nchunks, nheads, headdim, dstate), device=C_list.device, dtype=dtype)
    grid_dstates = lambda META: (triton.cdiv(headdim, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']),
                            batch * nchunks, nheads)
    with torch.cuda.device(dout.device.index):
        for _C in C_list:
            _dstates = torch.empty((batch, nchunks, nheads, headdim, dstate), device=C_list.device, dtype=dtype)
            _chunk_scan_bwd_dstates_kernel[grid_dstates](
                dout, _C, _dstates, dA_cumsum, seq_idx,
                headdim, dstate, chunk_size,
                batch, seqlen, nchunks, nheads // ngroups,
                dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
                _C.stride(0), _C.stride(1), _C.stride(2), _C.stride(3),
                _dstates.stride(0), _dstates.stride(1), _dstates.stride(2), _dstates.stride(3), _dstates.stride(4),
                dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
                *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
                HAS_SEQ_IDX=seq_idx is not None,
            )
            dstates += _dstates
    return dstates


def vmap_chunk_scan_chunk_state_bwd_dx(
        x: torch.Tensor,
        dt: torch.Tensor,
        dA_cumsum: torch.Tensor,
        B: torch.Tensor,
        CB_list: torch.Tensor,
        dout: torch.Tensor,
        dstates: torch.Tensor,
        D: torch.Tensor = None,
        seq_idx: torch.Tensor = None,
        dx: torch.Tensor = None,
):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert CB_list.shape[1:] == (batch, nchunks, ngroups, chunk_size, chunk_size)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert dout.shape == x.shape
    assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
        assert D.stride(-1) == 1
        BLOCK_SIZE_min = 32
        dD = torch.zeros(triton.cdiv(chunk_size, BLOCK_SIZE_min), batch, nchunks, nheads,
                         headdim if D.dim() == 2 else 1, device=D.device, dtype=torch.float32)
    else:
        dD = None
    dD_strides = ((dD.stride(0), dD.stride(1), dD.stride(2), dD.stride(3), dD.stride(4))
                    if D is not None else (0, 0, 0, 0, 0))
    if dx is None:
        dx = torch.zeros_like(x)
    else:
        assert dx.shape == x.shape
    ddt = torch.zeros((batch, nheads, nchunks, chunk_size), device=dout.device, dtype=torch.float32)
    grid_dx = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(headdim, META['BLOCK_SIZE_N']),
                        batch * nchunks, nheads)
    with torch.cuda.device(dout.device.index):
        for _CB in CB_list:
            _dx = torch.empty_like(dx)
            _ddt = torch.empty_like(ddt)
            _dD = torch.empty_like(dD)

            _chunk_scan_chunk_state_bwd_dx_kernel[grid_dx](
                x, _CB, dout, dt, dA_cumsum, seq_idx, D, B, dstates, _dx, _ddt, _dD,
                chunk_size, headdim, dstate,
                batch, seqlen, nheads // ngroups,
                x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                _CB.stride(0), _CB.stride(1), _CB.stride(2), _CB.stride(-1), _CB.stride(-2),
                dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
                dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
                dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
                *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
                D.stride(0) if D is not None else 0,
                B.stride(0), B.stride(1), B.stride(2), B.stride(3),
                dstates.stride(0), dstates.stride(1), dstates.stride(2), dstates.stride(3), dstates.stride(4),
                _dx.stride(0), _dx.stride(1), _dx.stride(2), _dx.stride(3),
                _ddt.stride(0), _ddt.stride(2), _ddt.stride(1), _ddt.stride(3),
                dD_strides[1], dD_strides[2], dD_strides[3], dD_strides[0], dD_strides[4],
                D is not None,
                D.dim() == 2 if D is not None else True,
                HAS_SEQ_IDX=seq_idx is not None,
                BLOCK_SIZE_DSTATE=max(triton .next_power_of_2(dstate), 16),
                IS_TRITON_22=TRITON_22
            )
            dx += _dx
            ddt += _ddt
            dD += _dD

    if D is not None:
        BLOCK_SIZE_actual = _chunk_scan_chunk_state_bwd_dx_kernel.best_config.kwargs["BLOCK_SIZE_M"]
        n_valid_blocks = (chunk_size + BLOCK_SIZE_actual - 1) // BLOCK_SIZE_actual
        dD = dD[:n_valid_blocks].sum(dim=(0, 1, 2)).to(dtype=D.dtype)
        if D.dim() == 1:
            dD = einops.rearrange(dD, "h 1 -> h")
    return dx, ddt.to(dtype=dt.dtype), dD


def vmap_chunk_scan_bwd_dC(
        prev_states: torch.Tensor,
        dA_cumsum: torch.Tensor,
        dout: torch.Tensor,
        seq_idx: torch.Tensor = None,
        C_list: torch.Tensor = None,
        ngroups: int = 1,
):
    batch, nchunks, nheads, headdim, dstate = prev_states.shape
    _, seqlen, _, _ = dout.shape
    _, _, _, chunk_size = dA_cumsum.shape
    assert prev_states.shape == (batch, nchunks, nheads, headdim, dstate)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert dout.shape == (batch, seqlen, nheads, headdim)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if C_list is not None:
        assert C_list.shape[1:] == (batch, seqlen, ngroups, dstate)
        C_strides = (C_list.stride(1), C_list.stride(2), C_list.stride(3), C_list.stride(4))
        ddA_cumsum_prev = torch.empty(batch, nheads, nchunks, chunk_size, device=dout.device, dtype=torch.float32)
        ddA_cumsum_prev_strides = (ddA_cumsum_prev.stride(0), ddA_cumsum_prev.stride(2), ddA_cumsum_prev.stride(1), ddA_cumsum_prev.stride(3))
    else:
        C_strides = (0, 0, 0, 0)
        ddA_cumsum_prev = None
        ddA_cumsum_prev_strides = (0, 0, 0, 0)
    nheads_ngroups_ratio = nheads // ngroups
    sm_count = torch.cuda.get_device_properties(dout.device).multi_processor_count
    nheads_per_program = max(min(math.ceil(batch * nchunks * nheads / sm_count), nheads_ngroups_ratio), 1)
    nsplits = triton.cdiv(nheads_ngroups_ratio, nheads_per_program)
    grid_dc = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']),
                        batch * nchunks, nsplits * ngroups)
    
    dC_list: list[torch.Tensor] = []
    with torch.cuda.device(dout.device.index):
        for _C in C_list:
            _dC = torch.empty((batch, seqlen, nsplits, ngroups, dstate), device=dout.device, dtype=torch.float32)
            _chunk_scan_bwd_dc_kernel[grid_dc](
                dout, prev_states, _C, dA_cumsum, seq_idx, _dC, ddA_cumsum_prev,
                chunk_size, dstate, headdim,
                batch, seqlen, nheads, nheads_per_program, ngroups,
                dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
                prev_states.stride(0), prev_states.stride(1), prev_states.stride(2), prev_states.stride(3), prev_states.stride(4),
                *C_strides,
                dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
                *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
                _dC.stride(0), _dC.stride(1), _dC.stride(2), _dC.stride(3), _dC.stride(4),
                *ddA_cumsum_prev_strides,
                HAS_DDA_CS=ddA_cumsum_prev is not None,
                HAS_SEQ_IDX=seq_idx is not None,
                BLOCK_SIZE_K=max(triton.next_power_of_2(headdim), 16),
            )
            dC_list.append(_dC.sum(2))
    dC = torch.stack(dC_list, dim=0)
    return dC if C_list is None else (dC, ddA_cumsum_prev)


def vmap_chunk_scan_bwd_ddAcs_stable(
        x: torch.Tensor,
        dt: torch.Tensor,
        dA_cumsum: torch.Tensor,
        dout: torch.Tensor,
        CB_list: torch.Tensor,
):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dout.shape == x.shape
    assert dA_cumsum.shape == dt.shape
    ngroups = CB_list.shape[3]
    assert nheads % ngroups == 0
    assert CB_list.shape[1:] == (batch, nchunks, ngroups, chunk_size, chunk_size)
    BLOCK_SIZE_M_min = 32
    ddA_cumsum = torch.zeros((batch, nheads, nchunks, triton.cdiv(chunk_size, BLOCK_SIZE_M_min),
                              chunk_size), device=x.device, dtype=torch.float32)
    grid_ddtcs = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']), batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        for _CB in CB_list:
            _ddA_cumsum = torch.empty_like(ddA_cumsum)
            _chunk_scan_bwd_ddAcs_stable_kernel[grid_ddtcs](
                x, dout, dt, dA_cumsum, _CB, _ddA_cumsum,
                chunk_size, headdim,
                batch, seqlen, nheads // ngroups,
                x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
                dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
                dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
                _CB.stride(0), _CB.stride(1), _CB.stride(2), _CB.stride(3), _CB.stride(4),
                _ddA_cumsum.stride(0), _ddA_cumsum.stride(2), _ddA_cumsum.stride(1), _ddA_cumsum.stride(3), _ddA_cumsum.stride(4),
                BLOCK_SIZE_K=max(triton.next_power_of_2(headdim), 16),
            )
            ddA_cumsum += _ddA_cumsum
        
    BLOCK_SIZE_M_actual = _chunk_scan_bwd_ddAcs_stable_kernel.best_config.kwargs["BLOCK_SIZE_M"]
    n_valid_blocks = (chunk_size + BLOCK_SIZE_M_actual - 1) // BLOCK_SIZE_M_actual
    ddA_cumsum = ddA_cumsum[:, :, :, :n_valid_blocks].sum(dim=3)
    return ddA_cumsum





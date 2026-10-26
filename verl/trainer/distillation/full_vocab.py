# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact fixed-prefix reverse KL with token-blocked first-order head VJPs.

Only the current student receives gradients. Frozen teacher hidden states and its
head are consumed during forward; backward saves only accumulated dH/dW. The
caller applies global token normalization once, outside this module.
"""

import torch


def _project(hidden, weight):
    if hidden.is_cuda and hidden.dtype == torch.bfloat16:
        return torch.mm(hidden, weight.T, out_dtype=torch.float32)
    return hidden @ weight.T


def _add_product(accumulator, left, right):
    if left.is_cuda and left.dtype == torch.bfloat16:
        torch.addmm(accumulator, left, right, out_dtype=torch.float32, out=accumulator)
    else:
        accumulator.addmm_(left, right)


class _ReverseKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hs, ws, ht, wt, token_chunk):
        dtype = torch.float64 if hs.dtype == torch.float64 else torch.float32
        dh = torch.zeros_like(hs, dtype=dtype)
        dw = torch.zeros_like(ws, dtype=dtype)
        total = torch.zeros((), device=hs.device, dtype=dtype)
        for start in range(0, len(hs), token_chunk):
            stop = min(start + token_chunk, len(hs))
            sh = hs[start:stop]
            lp = _project(sh, ws).log_softmax(-1)
            # Normalize the complete teacher before restricting to student rows.
            lt = _project(ht[start:stop], wt).log_softmax(-1)[:, : len(ws)]
            p = lp.exp()
            gap = lp - lt
            kl = (p * gap).sum(-1)
            dz = (p * (gap - kl[:, None])).to(hs.dtype)
            _add_product(dh[start:stop], dz, ws)
            _add_product(dw, dz.T, sh)
            total += kl.sum()
        ctx.save_for_backward(dh.to(hs.dtype), dw.to(ws.dtype))
        return total

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream):
        dh, dw = ctx.saved_tensors
        return dh * upstream, dw * upstream, None, None, None


def full_reverse_kl(hs, ws, ht, wt, *, token_chunk=512):
    """Return the exact KL sum, including teacher-only vocabulary mass.

    Token IDs must be aligned. Student and teacher hidden widths may differ;
    teacher output support must cover every student output row. An empty local
    token shard retains both student gradient connections.
    """
    if any(x.ndim != 2 for x in (hs, ws, ht, wt)):
        raise ValueError("full reverse KL requires 2-D hidden states and heads")
    if len(hs) != len(ht) or hs.shape[1] != ws.shape[1] or ht.shape[1] != wt.shape[1]:
        raise ValueError("Incompatible hidden/head shapes")
    if not len(ws) or len(wt) < len(ws):
        raise ValueError("Teacher support must cover the nonempty student vocabulary")
    if type(token_chunk) is not int or token_chunk <= 0:
        raise ValueError("token_chunk must be a positive integer")
    if len(hs) == 0:
        return hs.sum() * 0 + ws.sum() * 0
    return _ReverseKL.apply(hs, ws, ht.detach(), wt.detach(), token_chunk)

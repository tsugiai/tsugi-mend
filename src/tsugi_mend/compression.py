"""Optional gradient compression for the cross-rack reducer.

Phase 2 Week 1 stretch item per the design memo at
`docs/phase2_week1_async_tp_overlap.md`. Provides an opt-in feature flag
that wraps the cross-rack outer-step fragment with a compression
transform applied to the params_delta before merge. The synchronous
reducer path and the orchestrator path both pass through this hook so
the comparison stays apples-to-apples.

This module ships FOUR compression schemes:

1. INT8 quantization (lossy, simple): per-tensor symmetric quantization
   followed by dequantization. Fast, framework-independent, no extra
   state. Best as a "feature exists" demonstration; production usage
   would prefer PowerSGD with error feedback for accuracy.

2. PowerSGD-style low-rank compression with error feedback (lossy,
   convergence-preserving): rank-r truncated-SVD approximation of the
   2D-reshaped parameter delta with persistent error-feedback residual
   stored across calls. Reference: Vogels et al., NeurIPS 2019,
   arXiv:1905.13727 ("PowerSGD: Practical Low-Rank Gradient Compression
   for Distributed Optimization"). The torch built-in
   `torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook` was
   considered, but is DDP-bucket-bound and not directly reusable for
   the GraceWindowSyncer fragment merge path. We implement the
   minimal-state PowerSGD primitive in this module instead.

3. Lossless sparse delta encoding: non-zero elements are represented as
   flattened int64 indices plus exact values, then decoded back to a
   dense tensor before merge. The sparse representation is used only
   when its estimated payload is smaller than dense, so dense DiLoCo
   deltas fall back to the dense path instead of growing on the wire.
   This is useful only in genuinely sparse-update regimes.

4. None (default; lossless): pass-through. Preserves the bit-exact-
   loss-equivalence anchor.

The PowerSGD implementation here is a single-iteration low-rank
projection (analogous to PowerSGD's power-iteration kernel with K=1):

    G ~ P @ Q                       (rank-r reconstruction)
    where P, Q from one Power-iteration step on G.

Error feedback is supported via PowerSGDState, which carries the
per-tensor residual (G - P@Q) into the NEXT call's input. This is
the convergence-preserving variant: without EF the noisy compression
can prevent loss decrease at long-horizon training.

Patent-independence: gradient compression is published prior art
(PowerSGD: Vogels et al., NeurIPS 2019, arXiv:1905.13727; 1-bit Adam:
Tang et al., 2021; signSGD: Bernstein et al., ICML 2018). This module
exercises those published techniques and does not exercise the
K-Pool LoRA or Infinity patent estates.

Usage:

    from tsugi_mend.compression import (
        apply_compression, PowerSGDState, powersgd_compress_delta,
    )
    # Stateless transforms (int8, none):
    out = apply_compression(tensors, mode="int8")
    # PowerSGD with error feedback (stateful; one state per persistent
    # tensor identity, keyed by the caller's choice of key):
    state = PowerSGDState(rank=2)
    out0 = powersgd_compress_delta(tensors[0], state=state, key="layer.0.q")

The MendConfig.outer_step_compression_mode knob ("none" | "int8" |
"powersgd" | "sparse") selects which transform the runtime applies
inside the default fragment provider. Default "none" preserves bit-exact
loss equivalence with the vanilla baseline. "sparse" is also lossless,
but its communication benefit is conditional on element-sparse deltas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast

import torch
from torch import Tensor


CompressionMode = Literal["none", "int8", "powersgd", "sparse"]
SparseDeltaRepresentation = Literal["dense", "sparse"]
_INT64_INDEX_BYTES = 8


@dataclass(frozen=True)
class SparseDeltaPayload:
    """Wire-shape description for a lossless sparse-delta payload.

    `representation` is the selected transmission shape after the break-even
    check. For "dense", `dense` carries the tensor and `indices` / `values`
    are None. For "sparse", `indices` are flattened int64 positions and
    `values` are exact tensor values; `dense` is None. `estimated_bytes` is
    the selected representation's data-buffer estimate, excluding metadata
    common to both paths such as shape and dtype.
    """

    representation: SparseDeltaRepresentation
    shape: tuple[int, ...]
    dtype: torch.dtype
    nonzero_elements: int
    dense_bytes: int
    sparse_bytes: int
    estimated_bytes: int
    dense: Tensor | None = None
    indices: Tensor | None = None
    values: Tensor | None = None


@dataclass
class PowerSGDState:
    """Persistent state for PowerSGD with error feedback.

    Stores per-key residuals so the next call can fold the prior
    compression error back into the input. Following Vogels 2019
    Section 3 ("Error Feedback"), this is the variant that preserves
    long-horizon convergence under SGD.

    Fields:
        rank: low-rank approximation rank r. Higher r = less compression
            but lower error. PowerSGD paper default = 4.
        min_compression_size: skip compression for tensors with fewer
            elements than this; just return the original. Compressing
            very small tensors is pointless and the SVD cost dominates.
        residuals: per-key error-feedback buffer. Set at the end of a
            successful compression call; consumed at the start of the
            next call with the same key.
    """
    rank: int = 4
    min_compression_size: int = 1000
    residuals: dict[str, Tensor] = field(default_factory=dict)

    def reset(self, key: str | None = None) -> None:
        """Clear residual(s). If `key` is None, clear all. Otherwise
        clear every internal residual derived from `key` (this includes
        both the error-feedback residual and the cached Q init for that
        key).
        """
        if key is None:
            self.residuals.clear()
            return
        # Internal residual keys are constructed as `<user_key>:<suffix>`
        # (e.g., ":residual", ":Q0"). Remove all such derived keys.
        prefix = key + ":"
        for stored in list(self.residuals.keys()):
            if stored.startswith(prefix):
                del self.residuals[stored]


def _reshape_2d(tensor: Tensor) -> tuple[Tensor, tuple[int, ...]]:
    """Reshape a tensor to 2D for low-rank approximation. Returns the
    2D view and the original shape so the caller can restore it.

    For 1D tensors, returns a (numel, 1) column vector. For higher-
    rank tensors, flattens trailing dimensions into a single column
    dimension so we get a (rows, cols) view that PowerSGD can SVD.
    """
    orig_shape = tuple(tensor.shape)
    if tensor.dim() == 1:
        return tensor.unsqueeze(-1), orig_shape
    if tensor.dim() == 2:
        return tensor, orig_shape
    # Higher-dimensional: flatten trailing dims into the column dim.
    rows = tensor.shape[0]
    return tensor.reshape(rows, -1), orig_shape


def powersgd_compress_delta(
    tensor: Tensor,
    *,
    state: PowerSGDState,
    key: str,
) -> Tensor:
    """Compress a parameter delta tensor via PowerSGD rank-r approximation
    with error feedback.

    Algorithm (single-iteration variant; PowerSGD K=1 power step):
        1. Add prior residual r_{t-1} into G (error feedback).
        2. Reshape G to 2D (rows, cols).
        3. Draw a random orthonormal Q_0 of shape (cols, rank). Cached
           across calls in `state.residuals[key + ":Q"]` to amortize
           init cost.
        4. P = G @ Q_0; orthonormalize P via QR.
        5. Q = G.T @ P; the rank-r approximation is P @ Q.T.
        6. r_t = G - P @ Q.T (new residual).
        7. Return P @ Q.T reshaped to G's original shape.

    The return value is in the same dtype and shape as the input. The
    compression error is the residual stored in `state`; subsequent
    calls with the same `key` apply error feedback automatically.

    For tensors smaller than `state.min_compression_size`, this returns
    the input verbatim and does not update state. This avoids the SVD
    overhead on tiny tensors (bias vectors, layer norms).
    """
    if tensor.numel() < state.min_compression_size:
        return tensor.detach().clone()
    orig_dtype = tensor.dtype
    g = tensor.detach().to(torch.float32)
    G, orig_shape = _reshape_2d(g)
    rows, cols = G.shape
    r = min(state.rank, rows, cols)
    if r <= 0:
        return tensor.detach().clone()

    # Error feedback: fold in residual from the previous call.
    res_key = key + ":residual"
    if res_key in state.residuals:
        prior = state.residuals[res_key]
        if prior.shape == G.shape:
            G = G + prior

    # Cached random init for Q. Vogels 2019 uses orthonormal init via
    # QR on a random matrix; we cache it across calls so the spectral
    # subspace warms up rather than restarting each round.
    q_key = key + ":Q0"
    Q0 = state.residuals.get(q_key, None)
    if Q0 is None or Q0.shape != (cols, r):
        Q0 = torch.randn(cols, r, device=G.device, dtype=G.dtype)
        Q0, _ = torch.linalg.qr(Q0, mode="reduced")
        state.residuals[q_key] = Q0

    # One power-iteration step.
    P = G @ Q0
    P, _ = torch.linalg.qr(P, mode="reduced")
    Q = G.t() @ P
    # Cache the new Q as init for next round (warm start), orthonormalized
    # so the next call's power iteration starts from a clean basis. The
    # PowerSGD paper (Vogels 2019, Algorithm 1) uses this warm-start to
    # converge the subspace across rounds.
    Q_for_warmstart, _ = torch.linalg.qr(Q, mode="reduced")
    state.residuals[q_key] = Q_for_warmstart[:, :r] if Q_for_warmstart.shape[-1] >= r else Q_for_warmstart

    approx = P @ Q.t()
    # Persist the compression error for the next call.
    state.residuals[res_key] = (G - approx).detach()
    return cast(Tensor, approx.reshape(orig_shape).to(orig_dtype))


def int8_quantize_delta(tensor: Tensor) -> Tensor:
    """Per-tensor symmetric INT8 quantization plus immediate dequantization.

    Simulates the wire-level INT8 cost (8 bits per element instead of
    16 or 32) on the receiving end. The actual all-reduce traffic in
    production would carry the int8 buffer + a scalar scale; here we
    short-circuit the quantize -> transmit -> dequantize pipeline for
    benchmarking purposes.

    For a tensor with max absolute value M, the round-trip introduces
    quantization error bounded by M / 127. For typical neural-network
    gradients with M ~ 1e-3 to 1e-2, this is ~7.9e-6 to 7.9e-5 per
    element, which is below bf16 stochastic noise on continued-pretrain
    workloads.

    Returns a tensor of the same shape and dtype as the input but with
    int8-quantization noise applied. Caller controls whether to use
    error feedback by passing back the (input - output) residual on
    the next call.
    """
    if tensor.numel() == 0:
        return tensor.clone()
    orig_dtype = tensor.dtype
    flat = tensor.detach().to(torch.float32)
    abs_max = flat.abs().max().clamp(min=1e-12)
    scale = abs_max / 127.0
    quantized = (flat / scale).round().clamp(-128, 127).to(torch.int8)
    dequantized = quantized.to(torch.float32) * scale
    return dequantized.to(orig_dtype)


def _bitwise_nonzero_mask(tensor: Tensor) -> Tensor:
    """Return a flat mask for elements whose raw byte pattern is not zero.

    Value-level `tensor != 0` would drop IEEE-754 negative zero. Sparse delta
    sync must preserve exact bits, so the mask is computed over the raw bytes
    of each element instead.
    """
    if tensor.numel() == 0:
        return torch.empty(0, dtype=torch.bool, device=tensor.device)
    element_size = tensor.element_size()
    byte_view = tensor.detach().contiguous().view(torch.uint8)
    per_element = byte_view.reshape(tensor.numel(), element_size)
    return per_element.ne(0).any(dim=1)


def _dense_payload_bytes(tensor: Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _sparse_payload_bytes(tensor: Tensor, nonzero_elements: int) -> int:
    # Flattened int64 index per changed element plus the exact dtype value.
    return int(nonzero_elements * (_INT64_INDEX_BYTES + tensor.element_size()))


def sparse_delta_encode(tensor: Tensor) -> SparseDeltaPayload:
    """Encode a delta tensor with a lossless sparse representation when useful.

    Non-zero means "raw byte pattern differs from all-zero", not value-level
    inequality. That preserves negative zero and all finite or non-finite
    values exactly. Dense fallback is selected whenever the estimated sparse
    index+value data payload would be greater than or equal to the dense tensor
    payload.
    """
    detached = tensor.detach().contiguous()
    flat = detached.reshape(-1)
    mask = _bitwise_nonzero_mask(detached)
    indices = mask.nonzero(as_tuple=False).reshape(-1).to(dtype=torch.int64)
    nonzero_elements = int(indices.numel())
    dense_bytes = _dense_payload_bytes(detached)
    sparse_bytes = _sparse_payload_bytes(detached, nonzero_elements)
    shape = tuple(detached.shape)
    if sparse_bytes < dense_bytes:
        return SparseDeltaPayload(
            representation="sparse",
            shape=shape,
            dtype=detached.dtype,
            nonzero_elements=nonzero_elements,
            dense_bytes=dense_bytes,
            sparse_bytes=sparse_bytes,
            estimated_bytes=sparse_bytes,
            indices=indices,
            values=flat.index_select(0, indices).clone(),
        )
    return SparseDeltaPayload(
        representation="dense",
        shape=shape,
        dtype=detached.dtype,
        nonzero_elements=nonzero_elements,
        dense_bytes=dense_bytes,
        sparse_bytes=sparse_bytes,
        estimated_bytes=dense_bytes,
        dense=detached.clone(),
    )


def sparse_delta_decode(payload: SparseDeltaPayload) -> Tensor:
    """Decode a `SparseDeltaPayload` back to its exact dense tensor."""
    if payload.representation == "dense":
        if payload.dense is None:
            raise ValueError("dense sparse-delta payload is missing dense tensor")
        return payload.dense.detach().clone().reshape(payload.shape)
    if payload.representation == "sparse":
        if payload.indices is None or payload.values is None:
            raise ValueError("sparse sparse-delta payload is missing indices or values")
        out = torch.zeros(payload.shape, dtype=payload.dtype, device=payload.values.device)
        if payload.indices.numel() > 0:
            out.reshape(-1).index_copy_(
                0,
                payload.indices.to(device=payload.values.device),
                payload.values,
            )
        return out
    raise ValueError(f"unknown sparse-delta representation {payload.representation!r}")


def sparse_compress_delta(tensor: Tensor) -> Tensor:
    """Lossless sparse encode/decode transform for the compression hook."""
    return sparse_delta_decode(sparse_delta_encode(tensor))


def apply_compression(
    tensors: list[Tensor],
    mode: CompressionMode = "none",
    *,
    powersgd_state: PowerSGDState | None = None,
    key_prefix: str = "t",
) -> list[Tensor]:
    """Apply the named compression scheme to a list of tensors. Returns
    a new list; does not modify the input.

    For PowerSGD, the caller must supply a `powersgd_state` to enable
    error feedback across calls. Each tensor gets a unique key derived
    from `key_prefix` + its positional index, so persistent residuals
    line up across calls (callers should pass a stable key_prefix per
    layer-identity).
    """
    if mode == "none":
        return [t.detach().clone() for t in tensors]
    if mode == "int8":
        return [int8_quantize_delta(t) for t in tensors]
    if mode == "powersgd":
        if powersgd_state is None:
            powersgd_state = PowerSGDState()
        return [
            powersgd_compress_delta(t, state=powersgd_state, key=f"{key_prefix}.{i}")
            for i, t in enumerate(tensors)
        ]
    if mode == "sparse":
        return [sparse_compress_delta(t) for t in tensors]
    raise ValueError(
        f"unknown compression mode {mode!r}; expected one of: none, int8, powersgd, sparse"
    )


__all__ = [
    "CompressionMode",
    "PowerSGDState",
    "SparseDeltaPayload",
    "SparseDeltaRepresentation",
    "apply_compression",
    "int8_quantize_delta",
    "powersgd_compress_delta",
    "sparse_compress_delta",
    "sparse_delta_decode",
    "sparse_delta_encode",
]

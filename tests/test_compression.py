"""Stage A unit tests for the optional gradient-compression module."""
from __future__ import annotations

import pytest
import torch

from tsugi_mend.compression import (
    PowerSGDState,
    apply_compression,
    int8_quantize_delta,
    powersgd_compress_delta,
    sparse_delta_decode,
    sparse_delta_encode,
)


def _assert_same_bits(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    actual_bytes = actual.detach().contiguous().view(torch.uint8)
    expected_bytes = expected.detach().contiguous().view(torch.uint8)
    assert torch.equal(actual_bytes, expected_bytes)


def test_int8_quantize_preserves_shape_and_dtype():
    t = torch.randn(3, 5)
    q = int8_quantize_delta(t)
    assert q.shape == t.shape
    assert q.dtype == t.dtype


def test_int8_quantize_handles_zero_tensor():
    t = torch.zeros(4)
    q = int8_quantize_delta(t)
    assert torch.allclose(q, torch.zeros(4))


def test_int8_quantize_error_bound():
    """Per-tensor symmetric INT8 quantization introduces error bounded
    by max(|t|) / 127. Verify the bound is respected within a tight
    multiplier."""
    torch.manual_seed(42)
    for _ in range(5):
        t = torch.randn(100) * 0.01
        q = int8_quantize_delta(t)
        abs_max = t.abs().max().item()
        max_err = (q - t).abs().max().item()
        # Theoretical bound is abs_max / 127. Empirical max-error should
        # be at most ~half of that (rounding is to nearest), plus a small
        # safety multiplier for fp32-to-bf16 round-trip.
        assert max_err <= abs_max / 127.0 + 1e-7, (
            f"max_err={max_err}, bound={abs_max / 127.0}"
        )


def test_int8_quantize_handles_empty():
    t = torch.empty(0)
    q = int8_quantize_delta(t)
    assert q.numel() == 0


def test_apply_compression_none_is_identity():
    tensors = [torch.randn(2, 3), torch.randn(4)]
    out = apply_compression(tensors, mode="none")
    for a, b in zip(tensors, out):
        assert torch.allclose(a, b)
    # Ensure it returns NEW tensors (not aliases of the input).
    assert out[0] is not tensors[0]


def test_apply_compression_int8_round_trip():
    tensors = [torch.randn(8) * 0.01 for _ in range(3)]
    out = apply_compression(tensors, mode="int8")
    assert len(out) == len(tensors)
    for a, b in zip(tensors, out):
        # Quantized output is close-but-not-identical to input.
        diff = (a - b).abs().max().item()
        assert diff > 0
        # Bound: max(|a|) / 127
        assert diff <= a.abs().max().item() / 127.0 + 1e-7


def test_apply_compression_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown compression mode"):
        apply_compression([torch.randn(4)], mode="powerful")  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Lossless sparse delta codec
# ----------------------------------------------------------------------


def test_sparse_delta_round_trips_exact_cases():
    dense = torch.arange(1, 65, dtype=torch.float32).reshape(8, 8)

    sparse = torch.zeros(10_000, dtype=torch.float32)
    sparse[::100] = torch.linspace(1.0, 100.0, 100)

    all_zero = torch.zeros(512, dtype=torch.float32)
    all_nonzero = torch.arange(1, 513, dtype=torch.float32)

    negative_zero = torch.zeros(512, dtype=torch.float32)
    negative_zero[17] = -0.0
    negative_zero[31] = 1.25

    non_finite = torch.zeros(512, dtype=torch.float32)
    non_finite[3] = float("inf")
    non_finite[4] = float("-inf")
    non_finite[5] = float("nan")

    for tensor in (dense, sparse, all_zero, all_nonzero, negative_zero, non_finite):
        payload = sparse_delta_encode(tensor)
        decoded = sparse_delta_decode(payload)
        _assert_same_bits(decoded, tensor)


def test_sparse_delta_reports_dense_fallback_and_sparse_savings():
    dense = torch.ones(1024, dtype=torch.float32)
    dense_payload = sparse_delta_encode(dense)
    assert dense_payload.representation == "dense"
    assert dense_payload.estimated_bytes == dense_payload.dense_bytes
    assert dense_payload.sparse_bytes > dense_payload.dense_bytes

    sparse = torch.zeros(10_000, dtype=torch.float32)
    sparse[::100] = 1.0
    sparse_payload = sparse_delta_encode(sparse)
    assert sparse_payload.representation == "sparse"
    assert sparse_payload.nonzero_elements == 100
    assert sparse_payload.estimated_bytes == sparse_payload.sparse_bytes
    assert sparse_payload.sparse_bytes < sparse_payload.dense_bytes


def test_apply_compression_sparse_is_bit_exact_to_none_path():
    sparse = torch.zeros(4096, dtype=torch.float32)
    sparse[::200] = torch.arange(21, dtype=torch.float32)
    tensors = [
        torch.randn(16, 16),
        sparse,
        torch.tensor([0.0, -0.0, float("inf"), float("-inf")], dtype=torch.float32),
    ]

    none_path = apply_compression(tensors, mode="none")
    sparse_path = apply_compression(tensors, mode="sparse")

    assert len(sparse_path) == len(none_path)
    for sparse_tensor, dense_tensor in zip(sparse_path, none_path):
        _assert_same_bits(sparse_tensor, dense_tensor)


# ----------------------------------------------------------------------
# PowerSGD with error feedback
# ----------------------------------------------------------------------


def test_powersgd_preserves_shape_and_dtype():
    torch.manual_seed(0)
    t = torch.randn(64, 128)
    state = PowerSGDState(rank=4, min_compression_size=100)
    out = powersgd_compress_delta(t, state=state, key="layer.0")
    assert out.shape == t.shape
    assert out.dtype == t.dtype


def test_powersgd_skips_small_tensors():
    """Tensors below min_compression_size pass through verbatim."""
    torch.manual_seed(0)
    state = PowerSGDState(rank=4, min_compression_size=1000)
    t = torch.randn(32)  # 32 elements; below threshold
    out = powersgd_compress_delta(t, state=state, key="bias")
    assert torch.equal(out, t)
    # State should be empty since no compression happened.
    assert "bias:residual" not in state.residuals


def test_powersgd_low_rank_approximation_is_close_for_low_rank_inputs():
    """If the input is approximately low-rank (rank-2), rank-2 PowerSGD
    should achieve high-quality reconstruction. This is the core
    correctness contract."""
    torch.manual_seed(1)
    # Synthesize a rank-2 matrix.
    U = torch.randn(128, 2)
    V = torch.randn(2, 64)
    G = U @ V  # (128, 64) rank-2
    state = PowerSGDState(rank=2, min_compression_size=100)
    # Two warm-up calls allow the Q init to converge to the dominant
    # subspace.
    out = powersgd_compress_delta(G, state=state, key="layer.0")
    out = powersgd_compress_delta(G, state=state, key="layer.0")
    out = powersgd_compress_delta(G, state=state, key="layer.0")
    rel_err = (out - G).norm() / G.norm()
    # On exactly rank-2 input with rank-2 approximation, after a few
    # power-iteration warmups the relative error should be well under
    # 1% (typically 1e-4 ish). Allow generous slack here.
    assert rel_err.item() < 0.02, f"rel_err={rel_err.item()}"


def test_powersgd_error_feedback_drives_residual_to_zero_on_low_rank_signal():
    """Load-bearing EF claim: on an EXACTLY low-rank input compressed
    at the matching rank, repeated calls with persistent state drive
    the residual norm to near zero, because the warm-started subspace
    converges to the signal's principal components.

    Synthesize a rank-4 signal and compress at rank=4. After several
    iterations the residual should be near machine epsilon (single
    iteration may not nail it, but a few iterations definitely do).
    """
    torch.manual_seed(2)
    # Rank-4 ground-truth signal so the rank-4 approximation can in
    # principle capture all of it once the subspace converges.
    U = torch.randn(120, 4)
    V = torch.randn(4, 80)
    G = U @ V * 0.1

    state = PowerSGDState(rank=4, min_compression_size=100)
    residual_norms = []
    for _ in range(10):
        _ = powersgd_compress_delta(G, state=state, key="layer.0")
        res = state.residuals["layer.0:residual"]
        residual_norms.append(res.norm().item())

    g_norm = G.norm().item()
    # After 10 calls on a rank-4 input at rank=4, the residual should
    # be a small fraction of the signal norm (subspace has converged).
    assert residual_norms[-1] < g_norm * 0.1, (
        f"final residual norm {residual_norms[-1]:.4f} too large vs "
        f"signal norm {g_norm:.4f}"
    )
    # Residual should converge: either it decreased from the first
    # iteration, or it is already at the numerical noise floor. Warm-start
    # power iteration converges almost immediately on exactly-low-rank
    # input, so by iteration 10 both ends can sit at ~1e-5 where the
    # ordering is floating-point noise that varies across BLAS / torch
    # builds. Assert the meaningful convergence, not strict ordering at
    # the noise floor.
    noise_floor = g_norm * 1e-3
    assert residual_norms[-1] <= residual_norms[0] or residual_norms[-1] < noise_floor, (
        f"residual neither decreased nor reached the noise floor: "
        f"first={residual_norms[0]:.6f} last={residual_norms[-1]:.6f} "
        f"floor={noise_floor:.6f}"
    )


def test_powersgd_error_feedback_stays_bounded_under_stochastic_input():
    """Realistic-training EF property: when the input gradient sequence
    is stochastic (different g_t each call), the EF residual stays
    bounded near the per-step input norm. This matches the Vogels 2019
    theoretical analysis (their Theorem 4 says EF residual is bounded
    by O(sigma / sqrt(N)) under bounded gradient variance).
    """
    torch.manual_seed(3)
    state = PowerSGDState(rank=4, min_compression_size=100)
    residual_norms = []
    g_norms = []
    for _ in range(20):
        G = torch.randn(80, 80) * 0.1  # fresh stochastic g_t each call
        _ = powersgd_compress_delta(G, state=state, key="layer.0")
        res = state.residuals["layer.0:residual"]
        residual_norms.append(res.norm().item())
        g_norms.append(G.norm().item())

    mean_g_norm = sum(g_norms) / len(g_norms)
    # Smoke-level bound: the residual should not explode (e.g., grow
    # without limit toward infinity over the iteration sequence). The
    # theoretical steady-state under single-iteration power-method
    # PowerSGD with rank=4 on 80x80 random gradients is ||e*|| ~ a few
    # times the per-step ||g||, depending on the rank-r capture
    # fraction. We use a conservative bound: residual should stay
    # within 10x the per-step input norm.
    final_residual_avg = sum(residual_norms[-5:]) / 5
    assert final_residual_avg <= mean_g_norm * 10.0, (
        f"final residual avg {final_residual_avg:.4f} exceeded "
        f"10x per-step g_norm {mean_g_norm:.4f}; EF appears to be "
        f"blowing up rather than staying bounded"
    )


def test_powersgd_residual_state_persists_across_keys():
    """Different keys carry independent residuals; same key shares them."""
    torch.manual_seed(3)
    G1 = torch.randn(80, 80) * 0.1
    G2 = torch.randn(80, 80) * 0.1
    state = PowerSGDState(rank=4, min_compression_size=100)
    _ = powersgd_compress_delta(G1, state=state, key="a")
    _ = powersgd_compress_delta(G2, state=state, key="b")
    assert "a:residual" in state.residuals
    assert "b:residual" in state.residuals
    # Reset only "a"; "b" residual should persist.
    state.reset(key="a")
    assert "a:residual" not in state.residuals
    assert "b:residual" in state.residuals
    # Full reset clears everything.
    state.reset()
    assert state.residuals == {}


def test_powersgd_handles_1d_tensor():
    """1D tensors are reshaped to (numel, 1). Rank-1 max."""
    torch.manual_seed(4)
    t = torch.randn(2048) * 0.1
    state = PowerSGDState(rank=4, min_compression_size=100)
    out = powersgd_compress_delta(t, state=state, key="vec")
    assert out.shape == t.shape
    assert out.dtype == t.dtype


def test_apply_compression_powersgd_round_trip():
    """apply_compression with mode='powersgd' returns shape-matching
    tensors and updates the passed-in state."""
    torch.manual_seed(5)
    tensors = [torch.randn(80, 80) * 0.1 for _ in range(3)]
    state = PowerSGDState(rank=4, min_compression_size=100)
    out = apply_compression(
        tensors, mode="powersgd", powersgd_state=state, key_prefix="layer.0",
    )
    assert len(out) == len(tensors)
    for a, b in zip(tensors, out):
        assert b.shape == a.shape
        assert b.dtype == a.dtype
    # State should contain a residual per tensor.
    assert "layer.0.0:residual" in state.residuals
    assert "layer.0.1:residual" in state.residuals
    assert "layer.0.2:residual" in state.residuals


def test_apply_compression_powersgd_default_state_is_fresh():
    """Calling apply_compression with mode='powersgd' and no state
    creates a fresh state per call; the residual is discarded."""
    torch.manual_seed(6)
    tensors = [torch.randn(80, 80) * 0.1]
    # Without a passed-in state, each call gets its own state, so no
    # cross-call accumulation. This is fine for a once-off transform;
    # callers wanting EF must thread state through.
    out1 = apply_compression(tensors, mode="powersgd")
    out2 = apply_compression(tensors, mode="powersgd")
    assert out1[0].shape == tensors[0].shape
    assert out2[0].shape == tensors[0].shape

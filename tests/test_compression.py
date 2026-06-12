"""Stage A unit tests for the optional gradient-compression module."""
from __future__ import annotations

import pytest
import torch

from tsugi_mend.compression import (
    PowerSGDState,
    Quant4State,
    _bitwise_nonzero_mask,
    apply_compression,
    int8_quantize_delta,
    powersgd_compress_delta,
    quant4_compress_delta,
    quant4_decode,
    quant4_encode,
    sparse_delta_decode,
    sparse_delta_encode,
)
from tsugi_mend.config import MendConfig


def _assert_same_bits(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    actual_bytes = actual.detach().contiguous().view(torch.uint8)
    expected_bytes = expected.detach().contiguous().view(torch.uint8)
    assert torch.equal(actual_bytes, expected_bytes)


def _reference_bitwise_nonzero_mask(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.numel() == 0:
        return torch.empty(0, dtype=torch.bool, device=tensor.device)
    element_size = tensor.element_size()
    byte_view = tensor.detach().contiguous().view(torch.uint8)
    per_element = byte_view.reshape(tensor.numel(), element_size)
    return per_element.ne(0).any(dim=1)


def _floating_edge_tensor(dtype: torch.dtype) -> torch.Tensor:
    finite_edges = torch.tensor(
        [0.0, -0.0, float("inf"), float("-inf"), float("nan")],
        dtype=dtype,
    )
    denormal = torch.nextafter(
        torch.tensor([0.0], dtype=dtype),
        torch.tensor([1.0], dtype=dtype),
    )
    return torch.cat([finite_edges, denormal])


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_bitwise_nonzero_mask_matches_byte_reference_float_edges(dtype):
    tensor = _floating_edge_tensor(dtype)
    mask = _bitwise_nonzero_mask(tensor)
    assert torch.equal(mask, _reference_bitwise_nonzero_mask(tensor))
    assert mask.tolist() == [False, True, True, True, True, True]

    decoded = sparse_delta_decode(sparse_delta_encode(tensor))
    _assert_same_bits(decoded, tensor)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.bfloat16,
        torch.float16,
        torch.int32,
        torch.int64,
    ],
)
def test_bitwise_nonzero_mask_matches_byte_reference_large_random(dtype):
    torch.manual_seed(13)
    if dtype.is_floating_point:
        tensor = torch.randn(1_000_003, dtype=torch.float32).to(dtype)
        tensor[::97] = 0.0
        tensor[1::997] = -0.0
    else:
        tensor = torch.randint(-1000, 1000, (1_000_003,), dtype=dtype)
        tensor[::97] = 0

    assert torch.equal(
        _bitwise_nonzero_mask(tensor),
        _reference_bitwise_nonzero_mask(tensor),
    )


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


# ----------------------------------------------------------------------
# quant4: symmetric 4-bit block-wise quantization with error feedback
# (LOSSY when enabled; strictly opt-in; default "none" stays bit-exact)
# ----------------------------------------------------------------------


def test_apply_compression_none_is_bit_exact_with_quant4_available():
    """Off-path proof: with quant4 merely AVAILABLE (mode registered,
    state constructed), mode='none' output is torch.equal-identical to
    the input. The default path is byte-for-byte untouched."""
    torch.manual_seed(7)
    _ = Quant4State()  # quant4 exists; it must not perturb the none path
    tensors = [
        torch.randn(64, 32),
        torch.randn(513) * 1e-4,
        torch.randn(16, 8).to(torch.bfloat16),
        torch.randn(16, 8).to(torch.float16),
    ]
    out = apply_compression(tensors, mode="none")
    assert len(out) == len(tensors)
    for a, b in zip(tensors, out):
        assert torch.equal(b, a)
        assert b is not a  # a new tensor, not an alias of the input
    # Special values, incl. NaN where torch.equal is unsuitable: compare
    # raw bytes instead.
    special = torch.tensor(
        [0.0, -0.0, float("inf"), float("-inf"), float("nan")],
        dtype=torch.float32,
    )
    special_out = apply_compression([special], mode="none")[0]
    _assert_same_bits(special_out, special)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(3, 5), (2, 3, 4), (257,), (128,), (1,)])
def test_quant4_round_trip_shape_and_dtype(dtype, shape):
    torch.manual_seed(8)
    t = (torch.randn(shape) * 0.01).to(dtype)
    payload = quant4_encode(t)
    out = quant4_decode(payload)
    assert out.shape == t.shape
    assert out.dtype == t.dtype


def test_quant4_payload_packs_two_codes_per_byte():
    torch.manual_seed(8)
    t = torch.randn(300)  # zero-pads to 384 = 3 blocks of 128
    payload = quant4_encode(t)
    assert payload.packed.dtype == torch.uint8
    assert payload.packed.numel() == 384 // 2
    assert payload.scales.numel() == 3
    assert payload.numel == 300
    assert payload.shape == (300,)


def test_quant4_error_bounded_by_block_step_size():
    """Round-trip error is bounded per block by the block's step size
    (round-to-nearest actually achieves half the step)."""
    torch.manual_seed(9)
    block_size = 128
    t = torch.randn(4, 256) * 0.01  # 1024 elements = 8 blocks of 128
    payload = quant4_encode(t, block_size=block_size)
    out = quant4_decode(payload)
    err_blocks = (out - t).reshape(-1, block_size).abs()
    step = payload.scales.unsqueeze(1)  # per-block step size
    assert bool((err_blocks <= step * 0.5 + 1e-7).all())
    # And a fortiori the brief-level bound: error <= the step size.
    assert bool((err_blocks <= step).all())


def test_quant4_nonfinite_policy_pinned():
    """Pinned policy: NaN/+inf/-inf encode as 0 and are excluded from the
    block scale, so they neither crash nor poison finite neighbors;
    -0.0 decodes as +0.0 (sign of zero not preserved; lossy mode)."""
    t = torch.zeros(256, dtype=torch.float32)
    t[0] = float("nan")
    t[1] = float("inf")
    t[2] = float("-inf")
    t[3] = -0.0
    t[4] = 0.5
    t[5] = -0.5
    payload = quant4_encode(t, block_size=128)
    out = quant4_decode(payload)
    assert torch.isfinite(out).all()
    assert out[0].item() == 0.0
    assert out[1].item() == 0.0
    assert out[2].item() == 0.0
    assert out[3].item() == 0.0
    assert not bool(torch.signbit(out[3]))  # -0.0 came back as +0.0
    # Finite neighbors keep the normal bound: block scale derives from
    # max |finite| = 0.5, not from inf.
    step = payload.scales[0].item()
    assert step == pytest.approx(0.5 / 7.0)
    assert out[4].item() == pytest.approx(0.5, abs=step)
    assert out[5].item() == pytest.approx(-0.5, abs=step)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_quant4_compress_delta_handles_nonfinite_without_crash(dtype):
    state = Quant4State()
    base = [0.0, -0.0, float("inf"), float("-inf"), float("nan"), 1.0, -1.0, 0.25]
    t = torch.tensor(base * 32, dtype=dtype)
    out = quant4_compress_delta(t, state=state, key="k")
    assert out.shape == t.shape
    assert out.dtype == t.dtype
    assert torch.isfinite(out.to(torch.float32)).all()
    # Non-finites are sanitized BEFORE the EF update, so the persistent
    # residual stays finite and cannot poison later steps.
    assert torch.isfinite(state.residuals["k"]).all()
    # A second step with the carried residual also stays finite.
    out2 = quant4_compress_delta(t, state=state, key="k")
    assert torch.isfinite(out2.to(torch.float32)).all()


def test_quant4_error_feedback_shrinks_mean_reconstruction_error():
    """EF contract, asserted directly and deterministically: over repeated
    steps on a FIXED delta, the mean of the EF-corrected outputs converges
    to the true delta (the carried residual telescopes, leaving an error
    of ||r_T|| / T), while without EF the mean error stays pinned at the
    one-shot quantization error."""
    torch.manual_seed(10)
    g = torch.randn(64, 32) * 0.01
    steps = 8

    ef_state = Quant4State()
    ef_sum = torch.zeros_like(g)
    for _ in range(steps):
        ef_sum += quant4_compress_delta(g, state=ef_state, key="layer.0")
    ef_mean_err = (ef_sum / steps - g).norm().item()

    no_ef_sum = torch.zeros_like(g)
    for _ in range(steps):
        # A fresh state per call discards the residual = no error feedback.
        no_ef_sum += quant4_compress_delta(g, state=Quant4State(), key="layer.0")
    no_ef_mean_err = (no_ef_sum / steps - g).norm().item()

    assert no_ef_mean_err > 0  # quantization is genuinely lossy on this input
    assert ef_mean_err < no_ef_mean_err
    # The EF mean error shrinks like ~1/steps; demand at least a 2x win so
    # the assertion is meaningful rather than a tie-break.
    assert ef_mean_err < no_ef_mean_err / 2


def test_quant4_state_is_bounded_one_residual_per_key():
    torch.manual_seed(11)
    state = Quant4State()
    g = torch.randn(256) * 0.1
    for _ in range(5):
        quant4_compress_delta(g, state=state, key="a")
        quant4_compress_delta(g, state=state, key="b")
    # Bounded: exactly one residual tensor per key, no per-step growth.
    assert set(state.residuals.keys()) == {"a", "b"}
    assert state.residuals["a"].shape == g.shape
    state.reset(key="a")
    assert set(state.residuals.keys()) == {"b"}
    state.reset()
    assert state.residuals == {}


def test_quant4_handles_empty_tensor():
    t = torch.empty(0)
    payload = quant4_encode(t)
    out = quant4_decode(payload)
    assert out.numel() == 0
    assert out.shape == t.shape
    state = Quant4State()
    out2 = quant4_compress_delta(t, state=state, key="e")
    assert out2.numel() == 0


def test_quant4_rejects_bad_block_size():
    with pytest.raises(ValueError, match="block_size"):
        quant4_encode(torch.randn(8), block_size=3)
    with pytest.raises(ValueError, match="block_size"):
        quant4_encode(torch.randn(8), block_size=0)


def test_apply_compression_quant4_round_trip_and_state():
    torch.manual_seed(12)
    tensors = [torch.randn(64, 32) * 0.01, torch.randn(300) * 0.01]
    state = Quant4State()
    out = apply_compression(
        tensors, mode="quant4", quant4_state=state, key_prefix="layer.0",
    )
    assert len(out) == len(tensors)
    for a, b in zip(tensors, out):
        assert b.shape == a.shape
        assert b.dtype == a.dtype
        assert (a - b).abs().max().item() > 0  # lossy: not bit-identical
    assert "layer.0.0" in state.residuals
    assert "layer.0.1" in state.residuals


def test_apply_compression_unknown_mode_message_lists_quant4():
    with pytest.raises(ValueError, match="none, int8, powersgd, sparse, quant4"):
        apply_compression([torch.randn(4)], mode="quant5")  # type: ignore[arg-type]


def test_mend_config_accepts_quant4_mode():
    cfg = MendConfig(outer_step_compression_mode="quant4")
    assert cfg.outer_step_compression_mode == "quant4"
    with pytest.raises(ValueError, match="quant4"):
        MendConfig(outer_step_compression_mode="quant3")

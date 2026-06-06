"""Unit tests for the benchmark harness metric / CI helpers.

These exercise the pure helpers in ``benchmarks/metrics.py`` (no torch, no
multiprocessing): the bit-exact loss check, the steady-state summary, and
the paired-bootstrap uplift CI. The end-to-end gloo paired run itself is
exercised by ``benchmarks/run_paired.py`` (documented in
``benchmarks/README.md``); it is intentionally NOT a pytest case because it
spawns worker processes and is timing-dependent.
"""
from __future__ import annotations

import os
import sys

import pytest

# Make the repo root importable so `import benchmarks.metrics` resolves when
# the package is not installed (mirrors tests/conftest.py's src/ insertion).
_ROOT = os.path.dirname(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.metrics import (  # noqa: E402
    aggregate_seeded_uplift,
    bit_exact_equal,
    bootstrap_uplift_ci,
    steady_state,
)


# ----------------------------- bit_exact_equal -----------------------------


def test_bit_exact_identical_passes():
    a = [1.0, 0.5, 0.25, 0.125]
    assert bit_exact_equal(a, list(a)) is True


def test_bit_exact_tiny_difference_fails():
    a = [1.0, 0.5, 0.25]
    b = [1.0, 0.5, 0.25 + 1e-16]  # smallest representable perturbation
    # A tolerance-based check would pass this; the bit-exact check must not.
    assert b[2] != a[2]
    assert bit_exact_equal(a, b) is False


def test_bit_exact_length_mismatch_fails():
    assert bit_exact_equal([1.0, 2.0], [1.0, 2.0, 3.0]) is False


def test_bit_exact_nan_fails():
    nan = float("nan")
    assert bit_exact_equal([nan], [nan]) is False


def test_bit_exact_signed_zero_distinguished():
    # +0.0 and -0.0 compare == under Python's `==`, but represent different
    # bit patterns; the check must treat them as a mismatch.
    assert bit_exact_equal([0.0], [-0.0]) is False


# ------------------------------ steady_state -------------------------------


def test_steady_state_drops_warmup_and_computes_tps():
    # 4 steps, drop 2 warmup -> steady = [10ms, 10ms]; tokens_per_step=100.
    summary = steady_state([999.0, 999.0, 10.0, 10.0], tokens_per_step=100, warmup_steps=2)
    assert summary.n_steps_total == 4
    assert summary.warmup_steps == 2
    assert summary.n_steps_steady == 2
    # 100 tokens / 0.010 s = 10000 tokens/s.
    assert summary.tokens_per_second == pytest.approx(10000.0)
    assert summary.mean_step_time_ms == pytest.approx(10.0)


def test_steady_state_percentiles_monotonic():
    times = [float(i) for i in range(1, 101)]  # 1..100 ms
    summary = steady_state(times, tokens_per_step=10, warmup_steps=0)
    assert summary.p50_step_time_ms <= summary.p95_step_time_ms
    assert summary.p95_step_time_ms <= summary.p99_step_time_ms
    # p50 of 1..100 (linear interp) sits at the midpoint ~50.5.
    assert summary.p50_step_time_ms == pytest.approx(50.5)


def test_steady_state_requires_a_steady_step():
    with pytest.raises(ValueError):
        steady_state([5.0, 5.0], tokens_per_step=1, warmup_steps=2)


def test_steady_state_rejects_nonpositive_tokens():
    with pytest.raises(ValueError):
        steady_state([5.0], tokens_per_step=0, warmup_steps=0)


# --------------------------- bootstrap_uplift_ci ---------------------------


def test_bootstrap_uplift_positive_when_sdk_faster():
    # SDK is uniformly ~25% faster (smaller step time) -> positive uplift,
    # tight CI well above zero.
    baseline = [10.0] * 50
    sdk = [8.0] * 50
    ci = bootstrap_uplift_ci(baseline, sdk, n_resamples=2000, seed=0)
    # (10/8 - 1) * 100 = +25%.
    assert ci.point_estimate_pct == pytest.approx(25.0)
    assert ci.ci_low_pct == pytest.approx(25.0)  # zero variance -> degenerate CI
    assert ci.ci_high_pct == pytest.approx(25.0)
    assert ci.ci_low_pct > 0.0


def test_bootstrap_uplift_zero_when_equal():
    series = [7.0, 8.0, 9.0, 6.0, 7.5]
    ci = bootstrap_uplift_ci(series, list(series), n_resamples=1000, seed=1)
    assert ci.point_estimate_pct == pytest.approx(0.0)
    assert ci.ci_low_pct == pytest.approx(0.0)
    assert ci.ci_high_pct == pytest.approx(0.0)


def test_bootstrap_ci_brackets_point_estimate():
    baseline = [12.0, 11.0, 13.0, 10.0, 14.0, 9.0, 15.0, 8.0]
    sdk = [9.0, 10.0, 8.0, 11.0, 7.0, 12.0, 9.5, 10.5]
    ci = bootstrap_uplift_ci(baseline, sdk, n_resamples=5000, seed=42)
    assert ci.ci_low_pct <= ci.point_estimate_pct <= ci.ci_high_pct
    assert ci.n_paired_steps == 8
    assert ci.n_resamples == 5000
    assert ci.confidence == 0.95


def test_bootstrap_is_deterministic_for_fixed_seed():
    baseline = [10.0, 12.0, 9.0, 11.0, 13.0]
    sdk = [8.0, 9.0, 10.0, 7.0, 9.5]
    a = bootstrap_uplift_ci(baseline, sdk, n_resamples=3000, seed=7)
    b = bootstrap_uplift_ci(baseline, sdk, n_resamples=3000, seed=7)
    assert a.ci_low_pct == b.ci_low_pct
    assert a.ci_high_pct == b.ci_high_pct


def test_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        bootstrap_uplift_ci([1.0, 2.0], [1.0], n_resamples=10)


def test_bootstrap_requires_two_paired_steps():
    with pytest.raises(ValueError):
        bootstrap_uplift_ci([1.0], [1.0], n_resamples=10)


# -------------------------- aggregate_seeded_uplift -------------------------


def test_seeded_uplift_drops_fastest_and_slowest_then_aggregates():
    summary = aggregate_seeded_uplift(
        [-50.0, 10.0, 20.0, 30.0, 100.0],
        n_resamples=500,
        seed=123,
    )
    assert summary.n_runs == 5
    assert summary.n_surviving_runs == 3
    assert summary.dropped_low_pct == pytest.approx(-50.0)
    assert summary.dropped_high_pct == pytest.approx(100.0)
    assert summary.surviving_uplifts_pct == (10.0, 20.0, 30.0)
    assert summary.mean_uplift_pct == pytest.approx(20.0)
    assert summary.sample_variance_pct2 == pytest.approx(100.0)
    assert summary.ci_low_pct <= summary.mean_uplift_pct <= summary.ci_high_pct


def test_seeded_uplift_bootstrap_is_deterministic_for_fixed_seed():
    values = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    a = aggregate_seeded_uplift(values, n_resamples=1000, seed=9)
    b = aggregate_seeded_uplift(values, n_resamples=1000, seed=9)
    assert a.ci_low_pct == b.ci_low_pct
    assert a.ci_high_pct == b.ci_high_pct


def test_seeded_uplift_requires_at_least_five_runs():
    with pytest.raises(ValueError, match="n>=5"):
        aggregate_seeded_uplift([1.0, 2.0, 3.0, 4.0], n_resamples=10)


def test_seeded_uplift_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        aggregate_seeded_uplift([1.0, 2.0, 3.0, 4.0, float("inf")], n_resamples=10)

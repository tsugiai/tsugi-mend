from __future__ import annotations

import pytest

from benchmarks.run_paired import (
    bootstrap_uplift_ci,
    check_bit_exact_losses,
    percentile,
    uplift_pct,
)


def test_percentile_interpolates() -> None:
    assert percentile([10.0, 20.0, 30.0], 0.5) == 20.0
    assert percentile([10.0, 20.0], 0.25) == 12.5


def test_bootstrap_uplift_ci_is_paired_and_deterministic() -> None:
    baseline = [100.0, 200.0, 300.0, 400.0]
    sdk = [110.0, 220.0, 330.0, 440.0]
    lo, hi = bootstrap_uplift_ci(baseline, sdk, resamples=256, seed=123)
    assert lo == pytest.approx(10.0)
    assert hi == pytest.approx(10.0)


def test_uplift_rejects_nonpositive_baseline() -> None:
    with pytest.raises(ValueError, match="positive"):
        uplift_pct(0.0, 1.0)


def test_bit_exact_loss_check_reports_mismatches() -> None:
    passed = check_bit_exact_losses([1.0, 2.0], [1.0, 2.0])
    assert passed.passed is True
    assert passed.max_abs_loss_delta == 0.0

    failed = check_bit_exact_losses([1.0, 2.0], [1.0, 2.25])
    assert failed.passed is False
    assert failed.max_abs_loss_delta == 0.25
    assert failed.mismatches == [
        {
            "step": 1,
            "baseline_loss": 2.0,
            "sdk_loss": 2.25,
            "abs_delta": 0.25,
        }
    ]

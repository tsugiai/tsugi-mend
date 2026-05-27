"""FailSlowDetector z-score detection tests."""
from __future__ import annotations


import pytest

from tsugi_mend.failslow import FailSlowDetector


def test_warmup_period_returns_not_slow():
    det = FailSlowDetector(window_steps=20, zscore_threshold=3.0, min_samples=10)
    # 9 samples: still in warmup
    for i in range(9):
        d = det.observe("rank_a", 100.0)
        assert d.is_slow is False
        assert d.reason == "warmup"
    # 10th sample crosses into detection territory but value is constant.
    d = det.observe("rank_a", 100.0)
    assert d.reason in {"degenerate_window", "ok"}
    assert d.is_slow is False


def test_constant_history_then_spike():
    det = FailSlowDetector(window_steps=20, zscore_threshold=3.0, min_samples=5)
    for _ in range(10):
        det.observe("rank_a", 100.0)
    # Spike at 200ms after a constant 100ms history. The sample is
    # appended to the window before the decision (so a single spike is
    # not silently smoothed away by future samples), so the resulting
    # z-score is finite but well over the 3.0 threshold.
    d = det.observe("rank_a", 200.0)
    assert d.is_slow is True
    assert d.z_score > 3.0
    assert d.reason == "slow"


def test_degenerate_window_still_constant_returns_not_slow():
    """If the full window is identical and we observe another identical
    sample, std is exactly zero and the sample is not above the mean. The
    decision must be not-slow with reason 'degenerate_window'."""
    det = FailSlowDetector(window_steps=20, zscore_threshold=3.0, min_samples=5)
    for _ in range(15):
        det.observe("rank_a", 100.0)
    d = det.observe("rank_a", 100.0)
    assert d.is_slow is False
    assert d.window_std_ms == 0.0
    assert d.reason == "degenerate_window"


def test_normal_jitter_below_threshold_is_not_slow():
    det = FailSlowDetector(window_steps=50, zscore_threshold=3.0, min_samples=10)
    # Build a window with mean=100, modest variance.
    import random
    random.seed(0)
    for _ in range(30):
        det.observe("rank_a", 100.0 + random.gauss(0, 5))
    # One sample at +2 sigma should not trigger.
    d = det.observe("rank_a", 110.0)
    assert d.is_slow is False
    assert d.z_score < 3.0


def test_large_outlier_triggers_slow():
    det = FailSlowDetector(window_steps=50, zscore_threshold=3.0, min_samples=10)
    for _ in range(30):
        det.observe("rank_a", 100.0)
        det.observe("rank_a", 110.0)  # gives non-zero std
    # 10sigma+ outlier
    d = det.observe("rank_a", 1000.0)
    assert d.is_slow is True
    assert d.z_score > 3.0


def test_per_rank_independence():
    det = FailSlowDetector(window_steps=20, zscore_threshold=3.0, min_samples=5)
    # rank_a builds a 100ms history.
    for _ in range(10):
        det.observe("rank_a", 100.0)
    # rank_b sees a first 1000ms sample; should be in warmup, not slow.
    d = det.observe("rank_b", 1000.0)
    assert d.is_slow is False
    assert d.reason == "warmup"


def test_reset_clears_window():
    det = FailSlowDetector(window_steps=20, zscore_threshold=3.0, min_samples=5)
    for _ in range(10):
        det.observe("rank_a", 100.0)
    det.reset("rank_a")
    d = det.observe("rank_a", 100.0)
    assert d.reason == "warmup"


def test_validation():
    with pytest.raises(ValueError, match="window_steps"):
        FailSlowDetector(window_steps=1, zscore_threshold=3.0, min_samples=5)
    with pytest.raises(ValueError, match="zscore_threshold"):
        FailSlowDetector(window_steps=20, zscore_threshold=0, min_samples=5)
    with pytest.raises(ValueError, match="min_samples"):
        FailSlowDetector(window_steps=20, zscore_threshold=3.0, min_samples=1)
    with pytest.raises(ValueError, match="cannot exceed"):
        FailSlowDetector(window_steps=5, zscore_threshold=3.0, min_samples=10)


def test_negative_step_time_rejected():
    det = FailSlowDetector(window_steps=20, zscore_threshold=3.0, min_samples=5)
    with pytest.raises(ValueError, match="step_time_ms"):
        det.observe("rank_a", -1.0)

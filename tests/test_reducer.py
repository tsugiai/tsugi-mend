"""Decoupled DiLoCo reducer tests: quorum, grace window, token-weighted merge."""
from __future__ import annotations

import pytest
import torch

from tsugi_mend.reducer import (
    GraceWindowSyncer,
    LearnerFragment,
    token_weighted_merge,
    uniform_merge,
)


def _mk_fragment(learner_id: str, round_id: int, values: list[float], tokens: int):
    """Build a fragment with one 1D tensor parameter."""
    return LearnerFragment(
        learner_id=learner_id,
        round_id=round_id,
        params_delta=[torch.tensor(values, dtype=torch.float32)],
        tokens_consumed=tokens,
    )


# ---- token-weighted merge --------------------------------------------------


def test_token_weighted_merge_equal_tokens_is_average():
    f1 = _mk_fragment("a", 0, [1.0, 2.0, 3.0], tokens=100)
    f2 = _mk_fragment("b", 0, [3.0, 2.0, 1.0], tokens=100)
    merged = token_weighted_merge([f1, f2])
    assert len(merged) == 1
    assert torch.allclose(merged[0], torch.tensor([2.0, 2.0, 2.0]))


def test_token_weighted_merge_biased_by_tokens():
    f1 = _mk_fragment("a", 0, [10.0], tokens=300)
    f2 = _mk_fragment("b", 0, [0.0], tokens=100)
    # merged = (300*10 + 100*0) / 400 = 7.5
    merged = token_weighted_merge([f1, f2])
    assert torch.allclose(merged[0], torch.tensor([7.5]))


def test_token_weighted_merge_rejects_empty():
    with pytest.raises(ValueError):
        token_weighted_merge([])


def test_token_weighted_merge_rejects_zero_tokens():
    f = _mk_fragment("a", 0, [1.0], tokens=0)
    with pytest.raises(ValueError, match="positive total tokens"):
        token_weighted_merge([f])


def test_token_weighted_merge_mismatched_param_count():
    f1 = LearnerFragment(
        learner_id="a",
        round_id=0,
        params_delta=[torch.tensor([1.0]), torch.tensor([2.0])],
        tokens_consumed=100,
    )
    f2 = _mk_fragment("b", 0, [3.0], tokens=100)
    with pytest.raises(ValueError, match="expected"):
        token_weighted_merge([f1, f2])


# ---- uniform merge ---------------------------------------------------------


def test_uniform_merge_ignores_token_counts():
    f1 = _mk_fragment("a", 0, [10.0], tokens=999999)
    f2 = _mk_fragment("b", 0, [0.0], tokens=1)
    merged = uniform_merge([f1, f2])
    assert torch.allclose(merged[0], torch.tensor([5.0]))


# ---- syncer state machine --------------------------------------------------


class FakeClock:
    """Deterministic monotonic clock for tests."""
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance_ms(self, ms: float) -> None:
        self.now += ms / 1000.0


def test_syncer_quorum_not_yet_satisfied_returns_none():
    clock = FakeClock()
    sync = GraceWindowSyncer(
        quorum_min_learners=3,
        grace_window_ms=1000,
        token_weighted=True,
        clock=clock,
    )
    sync.start_round(round_id=0)
    assert sync.submit(_mk_fragment("a", 0, [1.0], tokens=100)) is None
    assert sync.submit(_mk_fragment("b", 0, [2.0], tokens=100)) is None


def test_syncer_K_satisfied_then_grace_window_then_fire():
    clock = FakeClock()
    sync = GraceWindowSyncer(
        quorum_min_learners=2,
        grace_window_ms=500,
        token_weighted=True,
        clock=clock,
    )
    sync.start_round(round_id=7)
    assert sync.submit(_mk_fragment("a", 7, [1.0], tokens=100)) is None
    # K-th learner arrives; grace window starts.
    out = sync.submit(_mk_fragment("b", 7, [3.0], tokens=100))
    # Grace not yet elapsed; submit returns None (tick checks grace).
    assert out is None
    # Advance clock past grace window; next tick fires.
    clock.advance_ms(600)
    out = sync.tick()
    assert out is not None
    assert out.round_id == 7
    assert sorted(out.learners_merged) == ["a", "b"]
    assert out.learners_absent == []
    assert out.reason == "grace_expired"
    assert torch.allclose(out.merged_delta[0], torch.tensor([2.0]))


def test_syncer_third_learner_arrives_within_grace_is_included():
    clock = FakeClock()
    sync = GraceWindowSyncer(
        quorum_min_learners=2,
        grace_window_ms=500,
        token_weighted=True,
        clock=clock,
    )
    sync.start_round(round_id=1)
    sync.submit(_mk_fragment("a", 1, [1.0], tokens=100))
    sync.submit(_mk_fragment("b", 1, [3.0], tokens=100))
    # Within grace window.
    clock.advance_ms(200)
    sync.submit(_mk_fragment("c", 1, [5.0], tokens=100))
    # After grace window.
    clock.advance_ms(400)
    out = sync.tick()
    assert out is not None
    assert sorted(out.learners_merged) == ["a", "b", "c"]
    assert out.learners_absent == []
    assert out.reason == "grace_expired"
    # Equal-tokens average of [1, 3, 5] = 3.0
    assert torch.allclose(out.merged_delta[0], torch.tensor([3.0]))


def test_syncer_finalize_below_quorum_raises():
    sync = GraceWindowSyncer(quorum_min_learners=3, grace_window_ms=100)
    sync.start_round(round_id=0)
    sync.submit(_mk_fragment("a", 0, [1.0], tokens=100))
    with pytest.raises(RuntimeError, match="aborting round"):
        sync.finalize_on_timeout()


def test_syncer_failslow_exclusion_drops_submissions():
    clock = FakeClock()
    sync = GraceWindowSyncer(
        quorum_min_learners=2,
        grace_window_ms=100,
        clock=clock,
    )
    sync.start_round(round_id=5)
    sync.mark_failslow("slow_rank")
    # Slow rank's submission must be ignored.
    assert sync.submit(_mk_fragment("slow_rank", 5, [99.0], tokens=100)) is None
    # Two healthy ranks reach quorum.
    sync.submit(_mk_fragment("a", 5, [1.0], tokens=100))
    sync.submit(_mk_fragment("b", 5, [3.0], tokens=100))
    clock.advance_ms(200)
    out = sync.tick()
    assert out is not None
    assert "slow_rank" not in out.learners_merged
    assert "slow_rank" in out.learners_excluded
    assert out.learners_absent == []


def test_syncer_ignores_stale_round_fragments():
    sync = GraceWindowSyncer(quorum_min_learners=2, grace_window_ms=100)
    sync.start_round(round_id=10)
    # Fragment from round 9 is stale; reject.
    assert sync.submit(_mk_fragment("a", 9, [1.0], tokens=100)) is None
    assert sync.submit(_mk_fragment("a", 11, [1.0], tokens=100)) is None


def test_syncer_expected_three_learners_early_finalizes_all_present():
    clock = FakeClock()
    sync = GraceWindowSyncer(
        quorum_min_learners=2,
        grace_window_ms=1000,
        token_weighted=True,
        clock=clock,
    )
    sync.start_round(round_id=20, expected_learner_ids={"a", "b", "c"})

    assert sync.submit(_mk_fragment("a", 20, [1.0], tokens=100)) is None
    assert sync.submit(_mk_fragment("b", 20, [3.0], tokens=100)) is None
    clock.advance_ms(100)
    out = sync.submit(_mk_fragment("c", 20, [5.0], tokens=100))

    assert out is not None
    assert out.reason == "all_present"
    assert out.learners_absent == []
    assert out.learners_excluded == []
    assert sorted(out.learners_merged) == ["a", "b", "c"]
    assert torch.allclose(out.merged_delta[0], torch.tensor([3.0]))


def test_syncer_expected_four_learners_records_absent_after_grace():
    clock = FakeClock()
    sync = GraceWindowSyncer(
        quorum_min_learners=2,
        grace_window_ms=500,
        token_weighted=True,
        clock=clock,
    )
    sync.start_round(round_id=21, expected_learner_ids={"a", "b", "c", "d"})

    assert sync.submit(_mk_fragment("a", 21, [1.0], tokens=100)) is None
    assert sync.submit(_mk_fragment("b", 21, [3.0], tokens=100)) is None
    clock.advance_ms(600)
    out = sync.tick()

    assert out is not None
    assert out.reason == "grace_expired"
    assert out.learners_absent == ["c", "d"]
    assert out.learners_excluded == []
    assert sorted(out.learners_merged) == ["a", "b"]
    assert torch.allclose(out.merged_delta[0], torch.tensor([2.0]))


def test_syncer_expected_eight_learners_separates_failslow_from_absent():
    clock = FakeClock()
    expected = {f"r{i}" for i in range(8)}
    sync = GraceWindowSyncer(
        quorum_min_learners=5,
        grace_window_ms=250,
        token_weighted=True,
        clock=clock,
    )
    sync.start_round(round_id=22, expected_learner_ids=expected)
    sync.mark_failslow("r7")

    for idx in range(5):
        assert (
            sync.submit(_mk_fragment(f"r{idx}", 22, [float(idx)], tokens=100))
            is None
        )
    clock.advance_ms(300)
    out = sync.tick()

    assert out is not None
    assert out.reason == "grace_expired"
    assert out.learners_absent == ["r5", "r6"]
    assert out.learners_excluded == ["r7"]
    assert set(out.learners_absent).isdisjoint(out.learners_excluded)
    assert sorted(out.learners_merged) == ["r0", "r1", "r2", "r3", "r4"]
    assert torch.allclose(out.merged_delta[0], torch.tensor([2.0]))


def test_syncer_rejects_quorum_min_greater_than_total_learners():
    sync = GraceWindowSyncer(quorum_min_learners=4, grace_window_ms=100)
    with pytest.raises(ValueError, match="quorum_min_learners cannot exceed"):
        sync.start_round(round_id=23, expected_learner_ids={"a", "b", "c"})
    with pytest.raises(ValueError, match="quorum_min_learners cannot exceed"):
        sync.start_round(round_id=23, total_learners=3)


def test_syncer_expected_none_preserves_grace_reason_and_merge():
    clock = FakeClock()
    sync = GraceWindowSyncer(
        quorum_min_learners=2,
        grace_window_ms=500,
        token_weighted=True,
        clock=clock,
    )
    sync.start_round(round_id=24)
    assert sync.submit(_mk_fragment("a", 24, [1.0], tokens=100)) is None
    assert sync.submit(_mk_fragment("b", 24, [3.0], tokens=100)) is None
    clock.advance_ms(600)
    out = sync.tick()

    assert out is not None
    assert out.reason == "grace_expired"
    assert out.learners_absent == []
    assert sorted(out.learners_merged) == ["a", "b"]
    assert torch.allclose(out.merged_delta[0], torch.tensor([2.0]))


def test_syncer_simulated_merge_delay_is_applied():
    """Phase 2 Week 1 Day 4-7: simulated_merge_delay_ms must inject the
    requested wall-clock delay into _finalize. Used by the delay-sweep
    microbenchmark to model real-world cross-rack grace-window wait."""
    import time
    sync = GraceWindowSyncer(
        quorum_min_learners=1,
        grace_window_ms=0,
        simulated_merge_delay_ms=100,  # 100ms synthetic delay
    )
    sync.start_round(round_id=1)
    t0 = time.monotonic()
    out = sync.submit(_mk_fragment("a", 1, [1.0], tokens=100))
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    assert out is not None
    # Allow a 50ms tolerance for scheduling jitter.
    assert 90.0 <= elapsed_ms <= 250.0, (
        f"expected ~100ms delay; got {elapsed_ms:.1f}ms"
    )


def test_syncer_zero_delay_default_is_no_op():
    """The default simulated_merge_delay_ms=0 must not introduce any
    measurable delay."""
    import time
    sync = GraceWindowSyncer(quorum_min_learners=1, grace_window_ms=0)
    sync.start_round(round_id=1)
    t0 = time.monotonic()
    out = sync.submit(_mk_fragment("a", 1, [1.0], tokens=100))
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    assert out is not None
    # Default config should add ~zero overhead; allow 20ms for jitter.
    assert elapsed_ms < 20.0, f"unexpected delay: {elapsed_ms:.1f}ms"


# ----------------------------------------------------------------------
# Bimodal / long-tail simulated-delay distributions (Track D, 2026-05-23)
# ----------------------------------------------------------------------


def test_syncer_rejects_unknown_distribution():
    import pytest as _pytest
    with _pytest.raises(ValueError, match="simulated_merge_delay_distribution"):
        GraceWindowSyncer(
            quorum_min_learners=1,
            grace_window_ms=0,
            simulated_merge_delay_ms=100,
            simulated_merge_delay_distribution="bogus",
        )


def test_syncer_bimodal_distribution_95_5_falcon_anchored():
    """Bimodal (FALCON-CoV-anchored 2026-05-23): 95% rounds at 0.9*base,
    5% rounds at 2.9*base. Verify the sampled values cluster on those two
    exact discrete values and the 95/5 split holds within Wilson interval
    at n=1000."""
    import random as _random
    base = 2000.0
    short_val = 0.9 * base  # 1800
    long_val = 2.9 * base   # 5800
    sync = GraceWindowSyncer(
        quorum_min_learners=1,
        grace_window_ms=0,
        simulated_merge_delay_ms=int(base),
        simulated_merge_delay_distribution="bimodal",
    )
    sync.set_delay_rng(_random.Random(42))
    samples = [sync._sample_delay_ms() for _ in range(1000)]
    short = sum(1 for s in samples if s == short_val)
    long = sum(1 for s in samples if s == long_val)
    assert short + long == 1000, "bimodal must produce only short/long values"
    # 95/5 split within Wilson interval at n=1000.
    assert 0.92 <= short / 1000 <= 0.98, f"short fraction {short/1000:.3f} not near 0.95"
    assert 0.02 <= long / 1000 <= 0.08, f"long fraction {long/1000:.3f} not near 0.05"
    # Mass-weighted mean should be ~= base (the headline FALCON-anchored property).
    mean = sum(samples) / len(samples)
    assert 0.9 * base <= mean <= 1.1 * base, (
        f"bimodal mean {mean:.1f} should sit near base {base:.1f}; "
        f"the 95/5 split is calibrated so mean ~= base"
    )


def test_syncer_long_tail_distribution_falcon_cov_anchored():
    """Long-tail (FALCON-CoV-anchored 2026-05-23): log-normal with
    sigma=sqrt(log(1+0.29^2))~=0.285 and mu=log(base)-sigma^2/2. Median
    sits near base; coefficient of variation matches FALCON Table 2's
    inter-node RDMA CoV=0.29."""
    import math
    import random as _random
    import statistics as _stats
    base = 1000.0
    sync = GraceWindowSyncer(
        quorum_min_learners=1,
        grace_window_ms=0,
        simulated_merge_delay_ms=int(base),
        simulated_merge_delay_distribution="long_tail",
    )
    sync.set_delay_rng(_random.Random(42))
    samples = sorted(sync._sample_delay_ms() for _ in range(2000))
    median = samples[len(samples) // 2]
    # Median of log-normal with mu = log(base) - sigma^2/2 is exp(mu).
    # For sigma=0.285, exp(mu) = base * exp(-sigma^2/2) ~= 0.96 * base.
    assert 0.8 * base <= median <= 1.15 * base, (
        f"long-tail median {median:.1f} should sit near base {base:.1f} "
        f"(theoretical ~ {base * math.exp(-0.285**2 / 2):.1f})"
    )
    # Coefficient of variation ~ 0.29 (FALCON Table 2 inter-node RDMA).
    mean = _stats.mean(samples)
    stdev = _stats.stdev(samples)
    cov = stdev / mean
    assert 0.22 <= cov <= 0.36, (
        f"long-tail CoV {cov:.3f} not within tolerance band around FALCON's 0.29; "
        f"the long-tail re-tune anchors the distribution's variance to FALCON Table 2"
    )
    # Heavy right tail still exists: max of 2000 samples should exceed 1.4x base.
    assert samples[-1] > 1.4 * base, (
        f"long-tail max {samples[-1]:.1f} should still exceed 1.4x base "
        f"({1.4 * base:.1f}); the FALCON-CoV-anchored sigma~=0.285 has a "
        f"tighter but still right-skewed tail"
    )


def test_syncer_constant_distribution_is_default_behavior():
    """Constant distribution should match the prior simulated_merge_delay
    contract: always inject exactly base ms."""
    sync = GraceWindowSyncer(
        quorum_min_learners=1,
        grace_window_ms=0,
        simulated_merge_delay_ms=200,
        simulated_merge_delay_distribution="constant",
    )
    samples = [sync._sample_delay_ms() for _ in range(50)]
    assert all(s == 200.0 for s in samples), (
        f"constant distribution emitted non-base values: {set(samples)}"
    )

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


def test_syncer_ignores_stale_round_fragments():
    sync = GraceWindowSyncer(quorum_min_learners=2, grace_window_ms=100)
    sync.start_round(round_id=10)
    # Fragment from round 9 is stale; reject.
    assert sync.submit(_mk_fragment("a", 9, [1.0], tokens=100)) is None
    assert sync.submit(_mk_fragment("a", 11, [1.0], tokens=100)) is None


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


# ----------------------------------------------------------------------
# World-size-aware multi-rack reducer (ENG-4): expected-learner awareness,
# early-finalize ("all_present"), and absentee diagnostics. Parametrized
# over 3, 4, and 8 learners.
# ----------------------------------------------------------------------


# Each learner-count maps to (quorum_min, grace_window_ms) used by the
# scenario tests. Quorum is set below the total so absences/exclusions can
# still reach quorum.
_RACK_COUNTS = [3, 4, 8]


def _rack_ids(n: int) -> list[str]:
    return [f"rack-{i}" for i in range(n)]


@pytest.mark.parametrize("n", _RACK_COUNTS)
def test_multirack_all_present_early_finalizes(n: int):
    """(a) All expected learners present before the grace window expires
    -> early-finalize, reason 'all_present', learners_absent == []. The
    grace window is huge so we know finalize was NOT driven by grace."""
    clock = FakeClock()
    ids = _rack_ids(n)
    sync = GraceWindowSyncer(
        quorum_min_learners=n - 1,  # quorum below total
        grace_window_ms=1_000_000,  # effectively never expires
        token_weighted=True,
        clock=clock,
    )
    sync.start_round(round_id=3, expected_learner_ids=set(ids))
    out = None
    for i, lid in enumerate(ids):
        # Submit each within a tiny slice of the (enormous) grace window.
        clock.advance_ms(1.0)
        result = sync.submit(_mk_fragment(lid, 3, [float(i)], tokens=100))
        if i < n - 1:
            assert result is None, f"finalized too early at learner {lid}"
        else:
            out = result
    assert out is not None, "expected early-finalize on the last expected learner"
    assert out.reason == "all_present"
    assert out.learners_absent == []
    assert sorted(out.learners_merged) == sorted(ids)
    assert out.learners_excluded == []
    # Merged delta = uniform-token average of 0..n-1.
    expected_mean = sum(range(n)) / n
    assert torch.allclose(out.merged_delta[0], torch.tensor([expected_mean]))


@pytest.mark.parametrize("n", _RACK_COUNTS)
def test_multirack_some_absent_grace_expiry_lists_absentees(n: int):
    """(b) Quorum met but some expected learners never arrive -> the round
    finalizes on grace expiry, reason 'grace_expired', and the missing
    learners land in learners_absent."""
    clock = FakeClock()
    ids = _rack_ids(n)
    quorum = n - 1
    sync = GraceWindowSyncer(
        quorum_min_learners=quorum,
        grace_window_ms=500,
        token_weighted=True,
        clock=clock,
    )
    sync.start_round(round_id=4, expected_learner_ids=set(ids))
    # Exactly `quorum` learners arrive; the last (n - quorum) never do.
    present = ids[:quorum]
    absent = ids[quorum:]
    out = None
    for lid in present:
        out = sync.submit(_mk_fragment(lid, 4, [1.0], tokens=100))
    # Quorum met but not all expected present -> no early finalize yet.
    assert out is None, "should not early-finalize while expected learners pending"
    # Grace window expires.
    clock.advance_ms(600)
    out = sync.tick()
    assert out is not None
    assert out.reason == "grace_expired"
    assert sorted(out.learners_merged) == sorted(present)
    assert out.learners_absent == sorted(absent)
    assert out.learners_excluded == []


@pytest.mark.parametrize("n", _RACK_COUNTS)
def test_multirack_failslow_and_absent_are_disjoint(n: int):
    """(c) Fail-slow + absent interplay: one expected learner is fail-slow-
    excluded, one expected learner simply never arrives, the rest report.
    The excluded learner must appear in learners_excluded (not absent), the
    missing one in learners_absent (not excluded), and the two lists are
    disjoint. A fail-slow exclusion also counts as 'no longer pending', so
    if it leaves no other pending learners the round early-finalizes."""
    clock = FakeClock()
    ids = _rack_ids(n)
    # Need at least 3 distinct roles: one excluded, one absent, >=1 present.
    excluded_id = ids[0]
    absent_id = ids[1]
    present_ids = ids[2:]
    quorum = max(1, len(present_ids))  # present learners can reach quorum
    sync = GraceWindowSyncer(
        quorum_min_learners=quorum,
        grace_window_ms=500,
        token_weighted=True,
        clock=clock,
    )
    sync.start_round(round_id=5, expected_learner_ids=set(ids))
    sync.mark_failslow(excluded_id)
    # The fail-slow learner's late submission is ignored.
    assert sync.submit(_mk_fragment(excluded_id, 5, [99.0], tokens=100)) is None
    out = None
    for lid in present_ids:
        out = sync.submit(_mk_fragment(lid, 5, [2.0], tokens=100))
    # `absent_id` is the sole remaining pending expected learner, so the round
    # has NOT early-finalized yet (excluded does not count as present).
    assert out is None, "absent learner still pending; should not finalize early"
    # Grace expiry finalizes the round.
    clock.advance_ms(600)
    out = sync.tick()
    assert out is not None
    assert out.reason == "grace_expired"
    assert out.learners_excluded == [excluded_id]
    assert out.learners_absent == [absent_id]
    assert sorted(out.learners_merged) == sorted(present_ids)
    # Disjointness invariants.
    excluded_set = set(out.learners_excluded)
    absent_set = set(out.learners_absent)
    merged_set = set(out.learners_merged)
    assert excluded_set.isdisjoint(absent_set)
    assert excluded_set.isdisjoint(merged_set)
    assert absent_set.isdisjoint(merged_set)


@pytest.mark.parametrize("n", _RACK_COUNTS)
def test_multirack_failslow_last_pending_triggers_early_finalize(n: int):
    """(c, complement) When the only remaining pending expected learner is
    fail-slow-excluded after the rest have reported, a tick early-finalizes
    with reason 'all_present' and that learner shows up excluded, not absent."""
    clock = FakeClock()
    ids = _rack_ids(n)
    excluded_id = ids[-1]
    present_ids = ids[:-1]
    quorum = len(present_ids)
    sync = GraceWindowSyncer(
        quorum_min_learners=quorum,
        grace_window_ms=1_000_000,  # would never expire on its own
        token_weighted=True,
        clock=clock,
    )
    sync.start_round(round_id=6, expected_learner_ids=set(ids))
    out = None
    for lid in present_ids:
        clock.advance_ms(1.0)
        out = sync.submit(_mk_fragment(lid, 6, [1.0], tokens=100))
    # All-but-one present; the last is still pending -> no finalize.
    assert out is None
    # Operator/FALCON gives up on the straggler. Now nothing is pending.
    sync.mark_failslow(excluded_id)
    out = sync.tick()
    assert out is not None
    assert out.reason == "all_present"
    assert out.learners_excluded == [excluded_id]
    assert out.learners_absent == []
    assert sorted(out.learners_merged) == sorted(present_ids)


@pytest.mark.parametrize("n", _RACK_COUNTS)
def test_multirack_quorum_greater_than_total_raises(n: int):
    """(d) quorum_min > total must raise a clear ValueError when the total is
    known (either via expected_learner_ids or total_learners)."""
    ids = _rack_ids(n)
    sync = GraceWindowSyncer(quorum_min_learners=n + 1, grace_window_ms=100)
    with pytest.raises(ValueError, match="cannot exceed"):
        sync.start_round(round_id=0, expected_learner_ids=set(ids))
    # Same via total_learners alone.
    sync2 = GraceWindowSyncer(quorum_min_learners=n + 1, grace_window_ms=100)
    with pytest.raises(ValueError, match="cannot exceed"):
        sync2.start_round(round_id=0, total_learners=n)


def test_multirack_total_and_expected_must_agree():
    """(d, complement) Passing both total_learners and expected_learner_ids
    with disagreeing sizes raises a clear ValueError."""
    sync = GraceWindowSyncer(quorum_min_learners=1, grace_window_ms=100)
    with pytest.raises(ValueError, match="must equal"):
        sync.start_round(
            round_id=0,
            expected_learner_ids={"a", "b", "c"},
            total_learners=4,
        )


def test_multirack_total_learners_does_not_enable_early_finalize():
    """A bare total_learners count validates quorum but cannot name the
    expected ids, so early-finalize stays disabled and learners_absent stays
    empty (count alone is not expected-set awareness)."""
    clock = FakeClock()
    sync = GraceWindowSyncer(
        quorum_min_learners=2,
        grace_window_ms=500,
        clock=clock,
    )
    sync.start_round(round_id=8, total_learners=3)
    sync.submit(_mk_fragment("a", 8, [1.0], tokens=100))
    out = sync.submit(_mk_fragment("b", 8, [3.0], tokens=100))
    # Quorum met, but no expected set -> must wait out grace (no early finalize).
    assert out is None
    clock.advance_ms(600)
    out = sync.tick()
    assert out is not None
    assert out.reason == "grace_expired"
    assert out.learners_absent == []


# ---- regression guard: expected=None is byte-for-byte the old behavior -----


def test_multirack_regression_expected_none_identical_merge_and_reason():
    """(e) REGRESSION GUARD: with expected_learner_ids=None (the default), an
    existing-style round yields the IDENTICAL merged_delta and reason it does
    today, plus an empty learners_absent. Mirrors
    test_syncer_third_learner_arrives_within_grace_is_included exactly, then
    re-runs the same scenario WITH expected awareness to show the merge math
    is unchanged and only the reason/diagnostics differ."""
    # --- legacy path (no expected set) -------------------------------------
    clock = FakeClock()
    legacy = GraceWindowSyncer(
        quorum_min_learners=2,
        grace_window_ms=500,
        token_weighted=True,
        clock=clock,
    )
    legacy.start_round(round_id=1)  # no expected_learner_ids
    legacy.submit(_mk_fragment("a", 1, [1.0], tokens=100))
    legacy.submit(_mk_fragment("b", 1, [3.0], tokens=100))
    clock.advance_ms(200)
    legacy.submit(_mk_fragment("c", 1, [5.0], tokens=100))
    clock.advance_ms(400)
    out = legacy.tick()
    assert out is not None
    assert out.reason == "grace_expired"
    assert sorted(out.learners_merged) == ["a", "b", "c"]
    assert out.learners_absent == []  # new field defaults empty in legacy mode
    legacy_delta = out.merged_delta[0].clone()
    assert torch.allclose(legacy_delta, torch.tensor([3.0]))

    # --- expected-aware path, same arrivals --------------------------------
    clock2 = FakeClock()
    aware = GraceWindowSyncer(
        quorum_min_learners=2,
        grace_window_ms=500,
        token_weighted=True,
        clock=clock2,
    )
    aware.start_round(round_id=1, expected_learner_ids={"a", "b", "c"})
    aware.submit(_mk_fragment("a", 1, [1.0], tokens=100))
    aware.submit(_mk_fragment("b", 1, [3.0], tokens=100))
    clock2.advance_ms(200)
    out2 = aware.submit(_mk_fragment("c", 1, [5.0], tokens=100))
    # All three expected present + quorum met -> early-finalize immediately
    # (no need to advance to grace expiry).
    assert out2 is not None
    assert out2.reason == "all_present"
    assert out2.learners_absent == []
    assert sorted(out2.learners_merged) == ["a", "b", "c"]
    # The MERGE MATH is identical regardless of WHY/WHEN it fired.
    assert torch.allclose(out2.merged_delta[0], legacy_delta)


@pytest.mark.parametrize("token_weighted", [True, False])
def test_multirack_roster_vs_none_grace_expiry_bit_identical(token_weighted: bool):
    """Roster awareness must not perturb merged bits when it falls through to
    the same grace-expiry reason as the historical roster-None path.

    The uniform-merge case is the ENG-16 coverage gap; the token-weighted
    parameter keeps the proof symmetric with the existing reducer behavior.
    """
    fragments = [
        _mk_fragment("rack-0", 9, [1.5, -2.25, 0.125], tokens=137),
        _mk_fragment("rack-1", 9, [4.0, 8.0, -16.0], tokens=251),
        _mk_fragment("rack-2", 9, [-0.5, 3.0, 7.75], tokens=89),
    ]

    def run(expected_learner_ids: set[str] | None):
        clock = FakeClock()
        sync = GraceWindowSyncer(
            quorum_min_learners=3,
            grace_window_ms=500,
            token_weighted=token_weighted,
            clock=clock,
        )
        sync.start_round(round_id=9, expected_learner_ids=expected_learner_ids)
        for fragment in fragments:
            assert sync.submit(fragment) is None
        clock.advance_ms(600)
        out = sync.tick()
        assert out is not None
        return out

    roster_result = run({"rack-0", "rack-1", "rack-2", "ghost"})
    none_result = run(None)

    assert roster_result.reason == none_result.reason == "grace_expired"
    assert roster_result.learners_merged == none_result.learners_merged
    assert roster_result.learners_absent == ["ghost"]
    assert none_result.learners_absent == []
    assert len(roster_result.merged_delta) == len(none_result.merged_delta)
    for roster_tensor, none_tensor in zip(
        roster_result.merged_delta, none_result.merged_delta
    ):
        assert roster_tensor.dtype == none_tensor.dtype
        assert roster_tensor.shape == none_tensor.shape
        assert torch.equal(roster_tensor, none_tensor)


def test_multirack_regression_default_round_unchanged_token_weighted():
    """(e) A token-weighted legacy round (expected=None) still produces the
    exact token-weighted result, untouched by the new feature path."""
    clock = FakeClock()
    sync = GraceWindowSyncer(
        quorum_min_learners=2,
        grace_window_ms=500,
        token_weighted=True,
        clock=clock,
    )
    sync.start_round(round_id=2)
    sync.submit(_mk_fragment("a", 2, [10.0], tokens=300))
    sync.submit(_mk_fragment("b", 2, [0.0], tokens=100))
    clock.advance_ms(600)
    out = sync.tick()
    assert out is not None
    assert out.reason == "grace_expired"
    assert out.learners_absent == []
    # (300*10 + 100*0)/400 = 7.5
    assert torch.allclose(out.merged_delta[0], torch.tensor([7.5]))

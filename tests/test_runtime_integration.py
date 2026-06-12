"""End-to-end runtime integration test on a toy CPU model.

Stage A's most aggressive test: spin up mend_init / mend_shutdown on a tiny
LoRA-style nn.Module on CPU, run a handful of fake training steps, and
verify the diagnostics file was written and the schedule fires at the
expected boundaries.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

import pytest
import torch
import torch.nn as nn

from tsugi_mend import MendConfig, mend_init, mend_shutdown
from tsugi_mend.reducer import LearnerFragment
from tsugi_mend.runtime import get_runtime


class ToyLoraStyleModel(nn.Module):
    """A two-linear-layer toy model. Not actually LoRA-structured, but
    shaped like one for the runtime to operate on."""

    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(16, 64)
        self.down = nn.Linear(64, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.relu(self.up(x)))


def test_mend_init_and_shutdown_on_toy_model(tmp_path):
    model = ToyLoraStyleModel()
    config = MendConfig(
        # Reduce minimums so the validation passes on a 1-rack toy.
        quorum_min_learners=1,
        grace_window_ms=0,
        sync_period_steps=4,
        momentum_sync_period_steps=8,
        # No sideband peers => no async loop spawned.
        sideband_peers=(),
        diagnostics_dir=str(tmp_path / "diag"),
    )
    mend_init(model, config, rank_id="rack-0/rank-0")
    runtime = get_runtime(model)
    assert runtime.topology is not None
    # Drive a small training loop's worth of step_begin/step_end calls.
    for step in range(20):
        runtime.step_begin(step)
        # Pretend we ran a step
        time.sleep(0.001)
        runtime.step_end(step)
    mend_shutdown(model)
    # Diagnostics file should exist and parse.
    diag_dir = tmp_path / "diag"
    files = list(diag_dir.glob("max_sdk_pid*.jsonl"))
    assert files, "diagnostics file not written"
    events = []
    with open(files[0]) as f:
        for line in f:
            events.append(json.loads(line))
    event_names = {e["event"] for e in events}
    assert "mend_init" in event_names
    assert "mend_shutdown" in event_names


def test_mend_init_twice_raises():
    model = ToyLoraStyleModel()
    config = MendConfig(quorum_min_learners=1, grace_window_ms=0)
    mend_init(model, config)
    try:
        with pytest.raises(RuntimeError, match="already called"):
            mend_init(model, config)
    finally:
        mend_shutdown(model)


def test_mend_shutdown_idempotent():
    model = ToyLoraStyleModel()
    config = MendConfig(quorum_min_learners=1, grace_window_ms=0)
    mend_init(model, config)
    mend_shutdown(model)
    # Second shutdown is a silent no-op (not an error).
    mend_shutdown(model)


def test_schedule_for_uses_des_loc_cadence():
    model = ToyLoraStyleModel()
    config = MendConfig(
        quorum_min_learners=1,
        grace_window_ms=0,
        sync_period_steps=4,
        momentum_sync_period_steps=12,
    )
    mend_init(model, config)
    try:
        rt = get_runtime(model)
        assert rt.schedule_for(0).should_sync_params is True
        assert rt.schedule_for(0).should_sync_momenta is True
        assert rt.schedule_for(4).should_sync_params is True
        assert rt.schedule_for(4).should_sync_momenta is False
        assert rt.schedule_for(8).should_sync_params is True
        assert rt.schedule_for(8).should_sync_momenta is False
        assert rt.schedule_for(12).should_sync_params is True
        assert rt.schedule_for(12).should_sync_momenta is True
        assert rt.schedule_for(5).should_sync_params is False
    finally:
        mend_shutdown(model)


# ---------------------------------------------------------------------------
# Phase 2 Week 1: outer-step concurrent-orchestrator integration tests
# ---------------------------------------------------------------------------


def _build_fragment(learner_id: str, round_id: int, value: float) -> LearnerFragment:
    """Tiny fragment compatible with the ToyLoraStyleModel parameter shapes."""
    return LearnerFragment(
        learner_id=learner_id,
        round_id=round_id,
        params_delta=[torch.full((1,), value)],
        tokens_consumed=100,
    )


def test_outer_step_runtime_methods_succeed_when_concurrent_enabled(tmp_path):
    """When concurrent_outer_step=True, the runtime exposes the three
    outer-step methods and they orchestrate an async merge that
    completes within the grace window."""
    model = ToyLoraStyleModel()
    config = MendConfig(
        quorum_min_learners=2,
        grace_window_ms=20,
        sync_period_steps=4,
        momentum_sync_period_steps=8,
        concurrent_outer_step=True,
        sideband_peers=(),  # no sideband; runtime spawns its own loop
        diagnostics_dir=str(tmp_path / "diag"),
    )
    mend_init(model, config, rank_id="rack-0/rank-0")
    rt = get_runtime(model)
    try:
        assert rt.outer_step_in_flight() is False

        def provider() -> "asyncio.Queue[LearnerFragment]":
            queue: asyncio.Queue[LearnerFragment] = asyncio.Queue()
            async def drip() -> None:
                await queue.put(_build_fragment("rack-a", 7, value=1.0))
                await queue.put(_build_fragment("rack-b", 7, value=3.0))
            asyncio.get_event_loop().create_task(drip())
            return queue

        rt.outer_step_begin(round_id=7, fragment_provider=provider)
        # Should not block.
        assert rt.outer_step_in_flight() in (True, False)
        # Drain.
        deadline = time.monotonic() + 2.0
        result = None
        while time.monotonic() < deadline:
            result = rt.outer_step_collect()
            if result is not None:
                break
            time.sleep(0.005)
        assert result is not None, "outer-step did not complete within 2s"
        assert result.round_id == 7
        assert sorted(result.learners_merged) == ["rack-a", "rack-b"]
        # token-weighted-merge with equal weights: (1*100 + 3*100) / 200 = 2
        assert torch.allclose(result.merged_delta[0], torch.tensor([2.0]))
        assert rt.outer_step_in_flight() is False
    finally:
        mend_shutdown(model)

    # Verify diagnostics captured both events.
    diag_files = list((tmp_path / "diag").glob("max_sdk_pid*.jsonl"))
    assert diag_files, "diagnostics file not written"
    events = []
    with open(diag_files[0]) as f:
        for line in f:
            events.append(json.loads(line))
    event_names = [e["event"] for e in events]
    assert "outer_step_begin" in event_names
    assert "outer_step_collect" in event_names
    # Verify the mend_init event flags concurrent_outer_step_active=True.
    init_event = next(e for e in events if e["event"] == "mend_init")
    assert init_event["concurrent_outer_step_active"] is True


def test_outer_step_begin_raises_when_concurrent_disabled():
    """When concurrent_outer_step=False, calling outer_step_begin must
    raise a clear RuntimeError so callers know to use the synchronous
    GraceWindowSyncer path."""
    model = ToyLoraStyleModel()
    config = MendConfig(
        quorum_min_learners=1,
        grace_window_ms=0,
        concurrent_outer_step=False,
        sideband_peers=(),
    )
    mend_init(model, config)
    rt = get_runtime(model)
    try:
        with pytest.raises(RuntimeError, match="concurrent_outer_step is"):
            rt.outer_step_begin(
                round_id=1,
                fragment_provider=lambda: asyncio.Queue(),
            )
        # outer_step_collect should be a silent no-op (returns None).
        assert rt.outer_step_collect() is None
        # in_flight should always be False when orchestrator isn't allocated.
        assert rt.outer_step_in_flight() is False
    finally:
        mend_shutdown(model)


def test_outer_step_double_submit_raises_runtime_error(tmp_path):
    """Submitting a second outer-round while the first is PENDING must
    raise; the runtime delegates this to the orchestrator state machine."""
    model = ToyLoraStyleModel()
    config = MendConfig(
        quorum_min_learners=4,           # quorum we can't satisfy in this test
        grace_window_ms=200,
        concurrent_outer_step=True,
        sideband_peers=(),
        diagnostics_dir=str(tmp_path / "diag"),
    )
    mend_init(model, config)
    rt = get_runtime(model)
    try:
        def never_provider() -> "asyncio.Queue[LearnerFragment]":
            return asyncio.Queue()
        rt.outer_step_begin(round_id=1, fragment_provider=never_provider)
        with pytest.raises(RuntimeError, match="PENDING"):
            rt.outer_step_begin(round_id=2, fragment_provider=never_provider)
        # Drain the failed round (it will fail-fast on the deadline).
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                rt.outer_step_collect()
            except RuntimeError:
                # Expected: the asyncio task raised due to quorum not satisfied.
                break
            if not rt.outer_step_in_flight():
                break
            time.sleep(0.01)
    finally:
        mend_shutdown(model)


# ---------------------------------------------------------------------------
# Auto-tuner for sync_period_steps
# ---------------------------------------------------------------------------


def _drive_step_loop(runtime, n_steps: int, step_sleep_s: float) -> None:
    """Run n_steps of step_begin / step_end with a fixed simulated
    per-step compute time. Used by the auto-tuner tests."""
    for step in range(n_steps):
        runtime.step_begin(step)
        time.sleep(step_sleep_s)
        runtime.step_end(step)


def test_auto_tuner_off_preserves_static_sync_period(tmp_path):
    """When config.auto_tune_sync_period is False (default), the
    effective sync period must equal the static config value across
    the entire run, regardless of step time."""
    model = ToyLoraStyleModel()
    config = MendConfig(
        quorum_min_learners=1,
        grace_window_ms=2000,
        sync_period_steps=10,
        momentum_sync_period_steps=20,
        auto_tune_sync_period=False,  # default; restate for clarity
        sideband_peers=(),
        diagnostics_dir=str(tmp_path / "diag"),
    )
    mend_init(model, config, rank_id="rack-0/rank-0")
    rt = get_runtime(model)
    try:
        assert rt.effective_sync_period_steps() == 10
        _drive_step_loop(rt, n_steps=60, step_sleep_s=0.005)
        # Static value preserved even after the warmup window would have
        # crossed had auto-tune been on.
        assert rt.effective_sync_period_steps() == 10
    finally:
        mend_shutdown(model)


def test_auto_tuner_on_delay_bound_regime_sets_n_star(tmp_path):
    """With auto-tune ON and T_step << G (delay-bound), the effective N
    should equal ceil(G / T_step), bounded above by the static
    sync_period_steps."""
    model = ToyLoraStyleModel()
    # Keep G well above the static upper bound times T_step so timing
    # jitter on loaded runners cannot drop raw N* below the clamp.
    config = MendConfig(
        quorum_min_learners=1,
        grace_window_ms=10_000,
        sync_period_steps=15,
        momentum_sync_period_steps=60,
        auto_tune_sync_period=True,
        auto_tune_sync_period_warmup_steps=20,
        auto_tune_sync_period_min=2,
        sideband_peers=(),
        diagnostics_dir=str(tmp_path / "diag"),
    )
    mend_init(model, config, rank_id="rack-0/rank-0")
    rt = get_runtime(model)
    try:
        # Drive 25 steps at ~10ms each. The warmup boundary fires at
        # step 19 (0-indexed; warmup_steps=20 samples).
        _drive_step_loop(rt, n_steps=25, step_sleep_s=0.010)
        # Raw N* stays above 15 even on a heavily loaded runner, so the
        # effective value must be the static sync_period_steps upper bound.
        eff = rt.effective_sync_period_steps()
        assert eff == 15, (
            f"expected effective_sync_period=15 from the static upper-bound clamp; got {eff}"
        )
    finally:
        mend_shutdown(model)
    # Diagnostics should contain the auto-tune-decided event with raw N*
    # above 15 and effective clamped at 15.
    diag_files = list((tmp_path / "diag").glob("max_sdk_pid*.jsonl"))
    assert diag_files
    events = [json.loads(line) for line in open(diag_files[0])]
    decided = [e for e in events if e["event"] == "auto_tune_sync_period_decided"]
    assert len(decided) == 1
    assert decided[0]["effective_sync_period_steps"] == 15
    assert decided[0]["n_star_raw"] > 15  # the unclamped N* exceeds the upper bound
    assert decided[0]["grace_window_ms"] == 10_000


def test_auto_tuner_on_compute_bound_regime_picks_smaller_n(tmp_path):
    """With auto-tune ON and T_step ~= G (compute-bound), the auto-tuner
    should pick N* = ceil(G / T_step), which sits below the static
    upper bound."""
    model = ToyLoraStyleModel()
    # G=50ms, T_step ~= 10ms => N* = ceil(50/10) = 5. Static is 20, so
    # 5 < 20 and the auto-tuner picks N=5.
    config = MendConfig(
        quorum_min_learners=1,
        grace_window_ms=50,
        sync_period_steps=20,
        momentum_sync_period_steps=40,
        auto_tune_sync_period=True,
        auto_tune_sync_period_warmup_steps=20,
        auto_tune_sync_period_min=2,
        sideband_peers=(),
        diagnostics_dir=str(tmp_path / "diag"),
    )
    mend_init(model, config, rank_id="rack-0/rank-0")
    rt = get_runtime(model)
    try:
        _drive_step_loop(rt, n_steps=25, step_sleep_s=0.010)
        eff = rt.effective_sync_period_steps()
        # raw N* ~ 5; with timing jitter accept 3-9 inclusive (the
        # contract is "below the static upper bound and above the
        # min clamp", not the exact value).
        assert 3 <= eff <= 9, (
            f"expected effective_sync_period in [3, 9] for G=50ms / "
            f"T_step~=10ms compute-bound regime; got {eff}"
        )
        assert eff < 20, (
            f"effective N {eff} must be strictly below static upper bound 20 in this regime"
        )
    finally:
        mend_shutdown(model)


def test_auto_tuner_respects_min_clamp(tmp_path):
    """With T_step >> G (deeply compute-bound), raw N* would be 1; the
    auto_tune_sync_period_min clamp must keep effective N at the
    configured floor."""
    model = ToyLoraStyleModel()
    # G=10ms, T_step ~= 15ms => raw N* = ceil(10/15) = 1. Clamp min is
    # 4, so effective N = 4.
    config = MendConfig(
        quorum_min_learners=1,
        grace_window_ms=10,
        sync_period_steps=20,
        momentum_sync_period_steps=40,
        auto_tune_sync_period=True,
        auto_tune_sync_period_warmup_steps=20,
        auto_tune_sync_period_min=4,
        sideband_peers=(),
        diagnostics_dir=str(tmp_path / "diag"),
    )
    mend_init(model, config, rank_id="rack-0/rank-0")
    rt = get_runtime(model)
    try:
        _drive_step_loop(rt, n_steps=25, step_sleep_s=0.015)
        eff = rt.effective_sync_period_steps()
        assert eff == 4, (
            f"expected effective_sync_period=4 (clamped from raw N*=1); got {eff}"
        )
    finally:
        mend_shutdown(model)
    # Verify diagnostics record the raw and clamped values.
    diag_files = list((tmp_path / "diag").glob("max_sdk_pid*.jsonl"))
    assert diag_files
    events = [json.loads(line) for line in open(diag_files[0])]
    decided = next(e for e in events if e["event"] == "auto_tune_sync_period_decided")
    assert decided["min_clamp"] == 4
    assert decided["effective_sync_period_steps"] == 4
    assert decided["n_star_raw"] <= 2  # raw N* before clamp


def test_auto_tuner_min_above_max_rejected_at_config_init():
    """Config validation must reject auto_tune_sync_period_min >
    sync_period_steps."""
    with pytest.raises(ValueError, match="auto_tune_sync_period_min"):
        MendConfig(
            quorum_min_learners=1,
            grace_window_ms=100,
            sync_period_steps=4,
            auto_tune_sync_period_min=8,  # invalid; exceeds upper bound
        )


def test_full_training_loop_with_concurrent_outer_step(tmp_path):
    """Simulate a small training loop: inner steps + outer-step boundaries
    at sync_period_steps. The runtime should orchestrate concurrent merges
    without blocking the inner-step loop."""
    model = ToyLoraStyleModel()
    config = MendConfig(
        quorum_min_learners=1,           # easy to satisfy single-rack quorum
        grace_window_ms=5,
        sync_period_steps=5,
        momentum_sync_period_steps=10,
        concurrent_outer_step=True,
        sideband_peers=(),
        diagnostics_dir=str(tmp_path / "diag"),
    )
    mend_init(model, config, rank_id="rack-0/rank-0")
    rt = get_runtime(model)

    def make_provider(rid: int):
        def provider() -> "asyncio.Queue[LearnerFragment]":
            queue: asyncio.Queue[LearnerFragment] = asyncio.Queue()
            async def drip() -> None:
                await queue.put(_build_fragment("rack-0", rid, value=float(rid)))
            asyncio.get_event_loop().create_task(drip())
            return queue
        return provider

    collected_round_ids: list[int] = []
    try:
        for step in range(20):
            rt.step_begin(step)
            sched = rt.schedule_for(step)
            if sched.should_sync_params and not rt.outer_step_in_flight():
                rt.outer_step_begin(round_id=step, fragment_provider=make_provider(step))
            # Drain whatever's ready (non-blocking).
            collected = rt.outer_step_collect()
            if collected is not None:
                collected_round_ids.append(collected.round_id)
            time.sleep(0.002)  # simulate an inner step
            rt.step_end(step)

        # After the loop, collect any final in-flight rounds.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            collected = rt.outer_step_collect()
            if collected is not None:
                collected_round_ids.append(collected.round_id)
            if not rt.outer_step_in_flight():
                break
            time.sleep(0.01)
    finally:
        mend_shutdown(model)

    # We should have collected at least 2 outer-rounds across the 20 inner steps.
    # Exact count depends on timing; the contract is "more than 1, no deadlocks".
    assert len(collected_round_ids) >= 2, (
        f"only collected {len(collected_round_ids)} outer-rounds; "
        f"expected at least 2 across 20 inner-steps with sync_period=5"
    )


# ---------------------------------------------------------------------------
# ENG-5: world-size-aware multi-rack reducer wired LIVE through the runtime.
#
# These tests drive the REAL ConcurrentOuterStep / _MaxRuntime path (via
# mend_init / outer_step_begin / outer_step_collect) with a multi-learner
# provider whose fragments carry distinct learner_ids, and assert:
#   (a) early-finalize fires with reason == "all_present" once every expected
#       learner reports before the grace window elapses;
#   (b) MergeResult.learners_absent is populated when an expected learner never
#       arrives;
#   (c) the merged delta is bit-identical to the expected=None (no-roster) path
#       for the same arrivals -- early-finalize merges the same fragment set
#       grace-expiry would, just sooner.
#
# ROSTER-ID CONTRACT exercised here: the declared roster ids EQUAL the
# learner_id of the fragments the provider delivers ("rack-0", "rack-1", ...).
# CPU-only, asyncio/in-process; no GPU, no real NCCL.
# ---------------------------------------------------------------------------


def _multi_learner_provider_factory(
    fragments_spec, send_delay_s: float = 0.0
):
    """Build a fragment_provider that drips a list of LearnerFragments (with
    distinct learner_ids) into the orchestrator's queue.

    `fragments_spec` is a list of LearnerFragment. A fresh queue + drip task
    is created each time the returned provider is invoked, so the same factory
    can back several rounds. `send_delay_s` spaces out submissions; 0.0 means
    "deliver as fast as the loop allows".
    """

    def provider() -> "asyncio.Queue[LearnerFragment]":
        queue: asyncio.Queue[LearnerFragment] = asyncio.Queue()

        async def drip() -> None:
            for f in fragments_spec:
                if send_delay_s > 0:
                    await asyncio.sleep(send_delay_s)
                await queue.put(f)

        asyncio.get_event_loop().create_task(drip())
        return queue

    return provider


def _collect_with_deadline(rt, timeout_s: float = 3.0):
    """Poll rt.outer_step_collect() until it returns a MergeResult or the
    deadline elapses. Returns the result (or None on timeout)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = rt.outer_step_collect()
        if result is not None:
            return result
        time.sleep(0.002)
    return None


def test_eng5_early_finalize_fires_all_present_via_runtime(tmp_path):
    """(a) With the operator-declared roster wired through MendConfig, the
    real runtime path early-finalizes (reason == 'all_present') the moment
    all expected learners report -- well before the (deliberately long) grace
    window would have expired."""
    model = ToyLoraStyleModel()
    config = MendConfig(
        quorum_min_learners=3,
        # A long grace window: if early-finalize did NOT fire, the round would
        # take ~grace_window_ms to finalize. We assert it finalizes much
        # faster, which is only possible via the all_present early path.
        grace_window_ms=5_000,
        sync_period_steps=4,
        momentum_sync_period_steps=8,
        concurrent_outer_step=True,
        expected_learner_ids=("rack-0", "rack-1", "rack-2"),
        sideband_peers=(),
        diagnostics_dir=str(tmp_path / "diag"),
    )
    mend_init(model, config, rank_id="rack-0/rank-0")
    rt = get_runtime(model)
    try:
        frags = [
            _build_fragment("rack-0", round_id=11, value=1.0),
            _build_fragment("rack-1", round_id=11, value=2.0),
            _build_fragment("rack-2", round_id=11, value=3.0),
        ]
        provider = _multi_learner_provider_factory(frags)
        t0 = time.monotonic()
        rt.outer_step_begin(round_id=11, fragment_provider=provider)
        result = _collect_with_deadline(rt, timeout_s=3.0)
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        assert result is not None, "outer-step did not complete within 3s"
        assert result.round_id == 11
        assert result.reason == "all_present", (
            f"expected early-finalize reason 'all_present'; got {result.reason!r}"
        )
        assert sorted(result.learners_merged) == ["rack-0", "rack-1", "rack-2"]
        assert result.learners_absent == []
        # Must complete far faster than the 5s grace window (early-finalize).
        assert elapsed_ms < 2_000.0, (
            f"round took {elapsed_ms:.0f}ms; early-finalize should be far "
            f"under the 5000ms grace window"
        )
        assert rt.outer_step_in_flight() is False
    finally:
        mend_shutdown(model)

    # Diagnostics must record reason == 'all_present' and an empty absent list.
    diag_files = list((tmp_path / "diag").glob("max_sdk_pid*.jsonl"))
    assert diag_files, "diagnostics file not written"
    events = [json.loads(line) for line in open(diag_files[0])]
    collect_event = next(e for e in events if e["event"] == "outer_step_collect")
    assert collect_event["reason"] == "all_present"
    assert collect_event["learners_absent"] == []


def test_eng5_learners_absent_populated_when_one_never_arrives(tmp_path):
    """(b) An expected learner that never submits is reported in
    MergeResult.learners_absent, and (since not all expected arrive) the round
    finalizes via grace expiry rather than the all_present early path."""
    model = ToyLoraStyleModel()
    config = MendConfig(
        # Quorum reachable by the two arriving learners so the round can still
        # finalize after grace; the third is declared-but-absent.
        quorum_min_learners=2,
        grace_window_ms=30,
        sync_period_steps=4,
        momentum_sync_period_steps=8,
        concurrent_outer_step=True,
        expected_learner_ids=("rack-0", "rack-1", "rack-2"),
        sideband_peers=(),
        diagnostics_dir=str(tmp_path / "diag"),
    )
    mend_init(model, config, rank_id="rack-0/rank-0")
    rt = get_runtime(model)
    try:
        # Only rack-0 and rack-1 arrive; rack-2 is absent for round 12.
        frags = [
            _build_fragment("rack-0", round_id=12, value=1.0),
            _build_fragment("rack-1", round_id=12, value=2.0),
        ]
        provider = _multi_learner_provider_factory(frags)
        rt.outer_step_begin(round_id=12, fragment_provider=provider)
        result = _collect_with_deadline(rt, timeout_s=3.0)

        assert result is not None, "outer-step did not complete within 3s"
        assert result.round_id == 12
        assert sorted(result.learners_merged) == ["rack-0", "rack-1"]
        # rack-2 never arrived and was never fail-slow-excluded -> absent.
        assert result.learners_absent == ["rack-2"]
        # Not all expected present, so the early path did not fire.
        assert result.reason != "all_present"
        assert result.reason in ("grace_expired", "quorum_satisfied")
    finally:
        mend_shutdown(model)

    diag_files = list((tmp_path / "diag").glob("max_sdk_pid*.jsonl"))
    assert diag_files
    events = [json.loads(line) for line in open(diag_files[0])]
    collect_event = next(e for e in events if e["event"] == "outer_step_collect")
    assert collect_event["learners_absent"] == ["rack-2"]


def _run_one_round_collect(config, *, round_id, frags, tmp_path, rank_id="rack-0/rank-0"):
    """Spin up a full runtime with `config`, drive one outer round delivering
    `frags`, and return the collected MergeResult. Used by the bit-exact test
    to compare the roster vs no-roster paths under otherwise identical config."""
    model = ToyLoraStyleModel()
    mend_init(model, config, rank_id=rank_id)
    rt = get_runtime(model)
    try:
        provider = _multi_learner_provider_factory(frags)
        rt.outer_step_begin(round_id=round_id, fragment_provider=provider)
        result = _collect_with_deadline(rt, timeout_s=3.0)
        assert result is not None, "outer-step did not complete within 3s"
        return result
    finally:
        mend_shutdown(model)


def test_eng5_merged_delta_bit_identical_to_no_roster_path(tmp_path):
    """(c) BIT-EXACT: the merged delta produced via the wired roster path
    (early-finalize) is byte-for-byte identical to the merged delta produced by
    the expected=None path (quorum then full grace) for the same arrivals.
    Early-finalize merges the same fragment set grace-expiry would, just
    sooner, so the loss is unchanged."""
    # Identical fragment set for both runs (distinct learner_ids, distinct
    # token weights and values so the token-weighted merge is non-trivial).
    def make_frags(round_id):
        return [
            LearnerFragment(
                learner_id="rack-0",
                round_id=round_id,
                params_delta=[torch.tensor([1.5, -2.25, 0.125])],
                tokens_consumed=137,
            ),
            LearnerFragment(
                learner_id="rack-1",
                round_id=round_id,
                params_delta=[torch.tensor([4.0, 8.0, -16.0])],
                tokens_consumed=251,
            ),
            LearnerFragment(
                learner_id="rack-2",
                round_id=round_id,
                params_delta=[torch.tensor([-0.5, 3.0, 7.75])],
                tokens_consumed=89,
            ),
        ]

    base_kwargs = dict(
        quorum_min_learners=3,
        grace_window_ms=200,
        sync_period_steps=4,
        momentum_sync_period_steps=8,
        concurrent_outer_step=True,
        token_weighted_merge=True,
        sideband_peers=(),
    )

    # Path 1: roster declared -> world-size-aware early-finalize.
    config_roster = MendConfig(
        expected_learner_ids=("rack-0", "rack-1", "rack-2"),
        diagnostics_dir=str(tmp_path / "diag_roster"),
        **base_kwargs,
    )
    result_roster = _run_one_round_collect(
        config_roster, round_id=21, frags=make_frags(21), tmp_path=tmp_path
    )

    # Path 2: no roster (the historical default path), same arrivals.
    config_none = MendConfig(
        expected_learner_ids=None,
        diagnostics_dir=str(tmp_path / "diag_none"),
        **base_kwargs,
    )
    result_none = _run_one_round_collect(
        config_none, round_id=21, frags=make_frags(21), tmp_path=tmp_path
    )

    # The roster path early-finalized; the no-roster path waited out grace.
    assert result_roster.reason == "all_present"
    assert result_none.reason in ("grace_expired", "quorum_satisfied")
    # Same learners merged in both.
    assert result_roster.learners_merged == result_none.learners_merged

    # BIT-EXACT: every merged tensor identical (not merely allclose).
    assert len(result_roster.merged_delta) == len(result_none.merged_delta)
    for t_roster, t_none in zip(result_roster.merged_delta, result_none.merged_delta):
        assert t_roster.dtype == t_none.dtype
        assert t_roster.shape == t_none.shape
        assert torch.equal(t_roster, t_none), (
            f"roster-path merged delta {t_roster} differs from no-roster "
            f"path {t_none}; early-finalize must not change the merge math"
        )


def test_eng5_misdeclared_roster_falls_back_safely(tmp_path):
    """SAFE FALLBACK: a per-round roster the syncer cannot reconcile with the
    arriving fragments (here, a roster naming a learner that never arrives,
    smaller-than-quorum handled at config time) must NOT hang or change the
    merged result -- the round still finalizes (via grace) over the learners
    that did arrive. This exercises the per-round override seam on
    outer_step_begin and the syncer's own absent-but-finalize fallthrough."""
    model = ToyLoraStyleModel()
    config = MendConfig(
        quorum_min_learners=2,
        grace_window_ms=30,
        sync_period_steps=4,
        momentum_sync_period_steps=8,
        concurrent_outer_step=True,
        # Config roster left None; we override per-round below.
        sideband_peers=(),
        diagnostics_dir=str(tmp_path / "diag"),
    )
    mend_init(model, config, rank_id="rack-0/rank-0")
    rt = get_runtime(model)
    try:
        frags = [
            _build_fragment("rack-0", round_id=31, value=1.0),
            _build_fragment("rack-1", round_id=31, value=3.0),
        ]
        provider = _multi_learner_provider_factory(frags)
        # Declare a roster that names a never-arriving learner ("ghost").
        rt.outer_step_begin(
            round_id=31,
            fragment_provider=provider,
            expected_learner_ids=frozenset({"rack-0", "rack-1", "ghost"}),
        )
        result = _collect_with_deadline(rt, timeout_s=3.0)
        assert result is not None, "misdeclared roster caused a hang"
        assert sorted(result.learners_merged) == ["rack-0", "rack-1"]
        # The never-arriving declared learner is surfaced, round still merges.
        assert result.learners_absent == ["ghost"]
        # token-weighted merge over the two arrivals: (100*1 + 100*3)/200 = 2.
        assert torch.allclose(result.merged_delta[0], torch.tensor([2.0]))
    finally:
        mend_shutdown(model)


def test_eng5_per_round_undersized_roster_warns_once_and_matches_none(
    caplog, tmp_path
):
    """A per-round roster override smaller than quorum is ignored exactly
    once and falls back to the historical roster-None semantics."""

    def make_frags(round_id):
        return [
            _build_fragment("rack-0", round_id=round_id, value=1.0),
            _build_fragment("rack-1", round_id=round_id, value=3.0),
            _build_fragment("rack-2", round_id=round_id, value=5.0),
        ]

    base_kwargs = dict(
        quorum_min_learners=3,
        grace_window_ms=30,
        sync_period_steps=4,
        momentum_sync_period_steps=8,
        concurrent_outer_step=True,
        sideband_peers=(),
    )
    baseline = _run_one_round_collect(
        MendConfig(diagnostics_dir=str(tmp_path / "diag_none"), **base_kwargs),
        round_id=51,
        frags=make_frags(51),
        tmp_path=tmp_path,
    )

    caplog.clear()
    caplog.set_level(logging.WARNING, logger="tsugi_mend.concurrent")

    model = ToyLoraStyleModel()
    config = MendConfig(
        diagnostics_dir=str(tmp_path / "diag_undersized"),
        **base_kwargs,
    )
    mend_init(model, config, rank_id="rack-0/rank-0")
    rt = get_runtime(model)
    try:
        provider = _multi_learner_provider_factory(make_frags(52))
        rt.outer_step_begin(
            round_id=52,
            fragment_provider=provider,
            expected_learner_ids=frozenset({"rack-0", "rack-1"}),
        )
        result = _collect_with_deadline(rt, timeout_s=3.0)
    finally:
        mend_shutdown(model)

    assert result is not None, "undersized roster fallback caused a hang"
    assert result.reason == baseline.reason == "grace_expired"
    assert result.learners_absent == []
    assert result.learners_merged == baseline.learners_merged
    assert len(result.merged_delta) == len(baseline.merged_delta)
    for fallback_tensor, baseline_tensor in zip(
        result.merged_delta, baseline.merged_delta
    ):
        assert fallback_tensor.dtype == baseline_tensor.dtype
        assert fallback_tensor.shape == baseline_tensor.shape
        assert torch.equal(fallback_tensor, baseline_tensor)

    warnings = [
        record
        for record in caplog.records
        if record.name == "tsugi_mend.concurrent"
        and "declared roster of 2 learner(s) is smaller than quorum_min=3"
        in record.getMessage()
    ]
    assert len(warnings) == 1


def test_eng5_config_expected_learner_ids_validation():
    """MendConfig.expected_learner_ids validation: default None is accepted;
    a well-formed roster is accepted; empty-string, duplicate, and
    quorum-exceeds-roster cases are rejected at config init."""
    # Default None and a valid roster both construct fine.
    assert MendConfig(quorum_min_learners=1, grace_window_ms=0).expected_learner_ids is None
    ok = MendConfig(
        quorum_min_learners=2,
        grace_window_ms=0,
        expected_learner_ids=("rack-0", "rack-1", "rack-2"),
    )
    assert ok.expected_learner_ids == ("rack-0", "rack-1", "rack-2")

    # Empty-string entry rejected.
    with pytest.raises(ValueError, match="non-empty strings"):
        MendConfig(
            quorum_min_learners=1,
            grace_window_ms=0,
            expected_learner_ids=("rack-0", ""),
        )

    # Duplicate entries rejected.
    with pytest.raises(ValueError, match="duplicates"):
        MendConfig(
            quorum_min_learners=1,
            grace_window_ms=0,
            expected_learner_ids=("rack-0", "rack-0"),
        )

    # quorum_min_learners > len(roster) rejected (quorum unsatisfiable).
    with pytest.raises(ValueError, match="cannot exceed"):
        MendConfig(
            quorum_min_learners=4,
            grace_window_ms=0,
            expected_learner_ids=("rack-0", "rack-1"),
        )


def test_eng5_config_default_path_unchanged_byte_for_byte(tmp_path):
    """With expected_learner_ids unset (default None), the wired runtime path
    is the historical quorum-then-full-grace behavior: no early-finalize, an
    empty learners_absent, exactly as before ENG-5."""
    model = ToyLoraStyleModel()
    config = MendConfig(
        quorum_min_learners=2,
        grace_window_ms=20,
        sync_period_steps=4,
        momentum_sync_period_steps=8,
        concurrent_outer_step=True,
        # expected_learner_ids left at default None.
        sideband_peers=(),
        diagnostics_dir=str(tmp_path / "diag"),
    )
    assert config.expected_learner_ids is None
    mend_init(model, config, rank_id="rack-0/rank-0")
    rt = get_runtime(model)
    try:
        frags = [
            _build_fragment("rack-a", round_id=41, value=1.0),
            _build_fragment("rack-b", round_id=41, value=3.0),
        ]
        provider = _multi_learner_provider_factory(frags)
        rt.outer_step_begin(round_id=41, fragment_provider=provider)
        result = _collect_with_deadline(rt, timeout_s=3.0)
        assert result is not None
        # No roster -> never the early path; absent list stays empty.
        assert result.reason != "all_present"
        assert result.learners_absent == []
        assert sorted(result.learners_merged) == ["rack-a", "rack-b"]
    finally:
        mend_shutdown(model)


def test_eng12_underdeclared_roster_disables_in_place_via_runtime(tmp_path):
    """ENG-12: an unexpected learner disables roster awareness in place on
    the live runtime path and preserves the roster-None merged bits."""

    def make_frags(round_id):
        return [
            _build_fragment("rack-0", round_id=round_id, value=1.0),
            _build_fragment("rack-2", round_id=round_id, value=5.0),
            _build_fragment("rack-1", round_id=round_id, value=3.0),
            _build_fragment("rack-3", round_id=round_id, value=7.0),
        ]

    base_kwargs = dict(
        quorum_min_learners=2,
        grace_window_ms=30,
        sync_period_steps=4,
        momentum_sync_period_steps=8,
        concurrent_outer_step=True,
        sideband_peers=(),
    )
    config_none = MendConfig(
        diagnostics_dir=str(tmp_path / "diag_none"),
        **base_kwargs,
    )
    none_result = _run_one_round_collect(
        config_none,
        round_id=61,
        frags=make_frags(61),
        tmp_path=tmp_path,
    )

    # The under-declared roster rides in on the ENG-5 config knob; rack-2 /
    # rack-3 arrive outside it and must trip the in-place fallback.
    config_fallback = MendConfig(
        diagnostics_dir=str(tmp_path / "diag_fallback"),
        expected_learner_ids=("rack-0", "rack-1"),
        **base_kwargs,
    )
    fallback_result = _run_one_round_collect(
        config_fallback,
        round_id=62,
        frags=make_frags(62),
        tmp_path=tmp_path,
    )

    assert fallback_result.roster_fallback is True
    assert none_result.roster_fallback is False
    assert fallback_result.reason == none_result.reason == "grace_expired"
    assert fallback_result.learners_merged == none_result.learners_merged
    assert len(fallback_result.merged_delta) == len(none_result.merged_delta)
    for fallback_tensor, none_tensor in zip(
        fallback_result.merged_delta, none_result.merged_delta
    ):
        assert torch.equal(fallback_tensor, none_tensor)

    diag_files = list((tmp_path / "diag_fallback").glob("max_sdk_pid*.jsonl"))
    assert diag_files
    events = [json.loads(line) for line in open(diag_files[0])]
    collect_event = next(e for e in events if e["event"] == "outer_step_collect")
    assert collect_event["roster_fallback"] is True

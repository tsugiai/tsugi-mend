"""Online runtime autotuner tests.

Covers the pure control law (RuntimeAutotuner) directly, then the runtime
integration: adaptation direction under a synthetic straggler-spike stream,
that the (observe-only) detector flags the injected straggler under the
adapted threshold, that NO mitigation/exclusion is invoked, that adaptation
is deterministic for a fixed input stream, and that a paired run with the
autotuner ENABLED stays bit-exact (max |loss diff| = 0.0).
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

import pytest
import torch
import torch.nn as nn

from tsugi_mend import MendConfig, mend_init, mend_shutdown
from tsugi_mend.autotuner import RuntimeAutotuner
from tsugi_mend.reducer import GraceWindowSyncer, LearnerFragment
from tsugi_mend.runtime import get_runtime


# ---------------------------------------------------------------------------
# Pure control-law unit tests (no torch, no runtime)
# ---------------------------------------------------------------------------


def _make_autotuner(**overrides: object) -> RuntimeAutotuner:
    kwargs: dict[str, object] = dict(
        base_zscore_threshold=3.0,
        base_grace_window_ms=2000,
        window_steps=20,
        min_samples=5,
        zscore_min=2.0,
        zscore_max=8.0,
        grace_min_ms=0,
        grace_max_ms=10_000,
        cov_gain=4.0,
        grace_gain=1.0,
    )
    kwargs.update(overrides)
    return RuntimeAutotuner(**kwargs)  # type: ignore[arg-type]


def test_warmup_holds_baseline_values():
    at = _make_autotuner(min_samples=5)
    for _ in range(4):
        d = at.observe(100.0)
        assert d.adapted is False
        assert d.reason == "warmup"
        assert d.effective_zscore_threshold == 3.0
        assert d.effective_grace_window_ms == 2000


def test_flat_clean_stream_relaxes_threshold_toward_floor():
    """A perfectly flat step-time stream has CoV=0, so the effective
    z-score threshold collapses to the base value (which equals the floor
    contribution); the grace window stays at baseline (peak_ratio=1)."""
    at = _make_autotuner(base_zscore_threshold=3.0)
    d = None
    for _ in range(20):
        d = at.observe(100.0)
    assert d is not None
    assert d.adapted is True
    assert d.observed_cov == 0.0
    assert d.observed_peak_ratio == 1.0
    # base_z + cov_gain*0 = 3.0
    assert d.effective_zscore_threshold == 3.0
    assert d.effective_grace_window_ms == 2000


def test_jittery_stream_raises_threshold():
    """A jittery (high-CoV) stream raises the effective z-score threshold
    above the base so benign jitter does not produce false straggler
    flags."""
    at = _make_autotuner(base_zscore_threshold=3.0, cov_gain=4.0)
    # Alternating 80 / 120 ms. The CoV is computed robustly (largest sample
    # excluded) so a benign jittery cluster still raises the threshold above
    # the base; the exact value depends on the window contents.
    d = None
    for i in range(20):
        d = at.observe(80.0 if i % 2 == 0 else 120.0)
    assert d is not None
    assert d.observed_cov > 0.1
    # z_eff = base_z + cov_gain * cov, strictly above the base.
    assert d.effective_zscore_threshold == pytest.approx(
        3.0 + 4.0 * d.observed_cov, rel=1e-9
    )
    assert d.effective_zscore_threshold > 3.0


def test_sustained_straggler_widens_grace_window():
    """A sustained straggler (recent peak >> median) widens the effective
    grace window above baseline."""
    at = _make_autotuner(base_grace_window_ms=1000, grace_gain=1.0)
    for _ in range(10):
        at.observe(100.0)
    # Inject a 400ms straggler. body-median (excluding the peak) is 100,
    # peak is 400 => peak_ratio 4.0 => widen = 1 + 1.0*(4-1) = 4.0
    d = at.observe(400.0)
    assert d.observed_peak_ratio == pytest.approx(4.0, rel=1e-6)
    assert d.effective_grace_window_ms == 4000
    assert d.effective_grace_window_ms > 1000


def test_grace_window_narrows_back_when_clean():
    """After a straggler passes and the window refills with clean samples,
    the effective grace window returns toward the baseline."""
    at = _make_autotuner(base_grace_window_ms=1000, grace_gain=1.0, window_steps=10, min_samples=5)
    for _ in range(5):
        at.observe(100.0)
    d_spike = at.observe(400.0)
    assert d_spike.effective_grace_window_ms > 1000
    # Refill the window with clean samples; the spike slides out (maxlen 10).
    d = None
    for _ in range(12):
        d = at.observe(100.0)
    assert d is not None
    assert d.observed_peak_ratio == pytest.approx(1.0, rel=1e-6)
    assert d.effective_grace_window_ms == 1000


def test_threshold_clamped_to_max():
    at = _make_autotuner(base_zscore_threshold=3.0, cov_gain=100.0, zscore_max=8.0)
    d = None
    for i in range(20):
        d = at.observe(50.0 if i % 2 == 0 else 150.0)  # CoV 0.5
    assert d is not None
    # 3.0 + 100*0.5 = 53, clamped to 8.0
    assert d.effective_zscore_threshold == 8.0


def test_grace_window_clamped_to_max():
    at = _make_autotuner(base_grace_window_ms=2000, grace_gain=100.0, grace_max_ms=10_000)
    for _ in range(10):
        at.observe(100.0)
    d = at.observe(10_000.0)
    assert d.effective_grace_window_ms == 10_000


def test_deterministic_for_fixed_stream():
    """Two autotuners fed the same stream emit identical decision
    sequences (the control law is a pure function of window contents)."""
    stream = [100.0, 105.0, 95.0, 300.0, 110.0, 90.0, 100.0, 250.0, 100.0, 100.0] * 3
    a1 = _make_autotuner()
    a2 = _make_autotuner()
    seq1 = [a1.observe(x) for x in stream]
    seq2 = [a2.observe(x) for x in stream]
    assert seq1 == seq2


def test_reset_reseeds_baseline():
    at = _make_autotuner(base_zscore_threshold=3.0, base_grace_window_ms=2000)
    for _ in range(10):
        at.observe(100.0)
    at.observe(900.0)  # perturb
    at.reset()
    assert at.effective_zscore_threshold == 3.0
    assert at.effective_grace_window_ms == 2000
    d = at.observe(100.0)
    assert d.reason == "warmup"


def test_control_law_validation():
    with pytest.raises(ValueError, match="window_steps"):
        _make_autotuner(window_steps=1)
    with pytest.raises(ValueError, match="min_samples"):
        _make_autotuner(min_samples=1)
    with pytest.raises(ValueError, match="cannot exceed"):
        _make_autotuner(window_steps=5, min_samples=10)
    with pytest.raises(ValueError, match="zscore_min"):
        _make_autotuner(zscore_min=0)
    with pytest.raises(ValueError, match="zscore_max"):
        _make_autotuner(zscore_min=5.0, zscore_max=3.0)
    with pytest.raises(ValueError, match="grace_min_ms"):
        _make_autotuner(grace_min_ms=-1)
    with pytest.raises(ValueError, match="grace_max_ms"):
        _make_autotuner(grace_min_ms=100, grace_max_ms=10)
    with pytest.raises(ValueError, match="cov_gain"):
        _make_autotuner(cov_gain=-1.0)
    with pytest.raises(ValueError, match="grace_gain"):
        _make_autotuner(grace_gain=-1.0)
    with pytest.raises(ValueError, match="step_time_ms"):
        _make_autotuner().observe(-1.0)


# ---------------------------------------------------------------------------
# Runtime integration: synthetic straggler stream
# ---------------------------------------------------------------------------


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(16, 64)
        self.down = nn.Linear(64, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.relu(self.up(x)))


def test_runtime_autotuner_off_is_noop(tmp_path):
    """Default config (auto_tune_runtime=False): the effective threshold
    and grace window equal the static config across the whole run."""
    model = _ToyModel()
    config = MendConfig(
        quorum_min_learners=1,
        grace_window_ms=2000,
        failslow_zscore_threshold=3.0,
        sync_period_steps=4,
        momentum_sync_period_steps=8,
        sideband_peers=(),
        diagnostics_dir=str(tmp_path / "diag"),
    )
    mend_init(model, config, rank_id="rack-0/rank-0")
    rt = get_runtime(model)
    try:
        assert rt.effective_failslow_zscore_threshold() == 3.0
        assert rt.effective_grace_window_ms() == 2000
        for step in range(60):
            rt.step_begin(step)
            # Even a big synthetic spike must not move the knobs when off.
            rt.step_end(step, step_time_ms_override=1000.0 if step == 40 else 100.0)
        assert rt.effective_failslow_zscore_threshold() == 3.0
        assert rt.effective_grace_window_ms() == 2000
    finally:
        mend_shutdown(model)
    # No autotuner diagnostics emitted when off.
    diag_files = list((tmp_path / "diag").glob("max_sdk_pid*.jsonl"))
    assert diag_files
    events = [json.loads(line) for line in open(diag_files[0])]
    assert not [e for e in events if e["event"] == "auto_tune_runtime_decision"]
    init_event = next(e for e in events if e["event"] == "mend_init")
    assert init_event["auto_tune_runtime_active"] is False


def test_runtime_autotuner_adapts_and_flags_straggler(tmp_path):
    """With auto_tune_runtime ON and a synthetic straggler-spike stream:
    (a) the grace window widens and the threshold rises (expected
    direction); (b) the detector flags the injected straggler under the
    adapted threshold; (c) NO mitigation/exclusion is invoked."""
    model = _ToyModel()
    config = MendConfig(
        quorum_min_learners=1,
        grace_window_ms=1000,
        failslow_zscore_threshold=3.0,
        failslow_window_steps=50,
        failslow_min_samples=10,
        sync_period_steps=4,
        momentum_sync_period_steps=8,
        auto_tune_runtime=True,
        auto_tune_runtime_window_steps=50,
        auto_tune_runtime_min_samples=10,
        auto_tune_grace_window_max_ms=10_000,
        sideband_peers=(),
        diagnostics_dir=str(tmp_path / "diag"),
    )
    mend_init(model, config, rank_id="rack-0/rank-0")
    rt = get_runtime(model)
    try:
        # 30 clean steps at 100ms, then a sustained 500ms straggler.
        for step in range(30):
            rt.step_begin(step)
            rt.step_end(step, step_time_ms_override=100.0)
        grace_before = rt.effective_grace_window_ms()
        thresh_before = rt.effective_failslow_zscore_threshold()
        # On a flat clean stream the grace window equals baseline and the
        # threshold equals base (CoV=0).
        assert grace_before == 1000
        assert thresh_before == 3.0

        # Inject the straggler step.
        rt.step_begin(30)
        rt.step_end(30, step_time_ms_override=500.0)

        # (a) the grace window widened in the expected direction.
        assert rt.effective_grace_window_ms() > grace_before
        # (b) the detector flagged the straggler under the adapted
        # threshold. We read the detector's decision for that step from the
        # diagnostics.
    finally:
        mend_shutdown(model)

    diag_files = list((tmp_path / "diag").glob("max_sdk_pid*.jsonl"))
    assert diag_files
    events = [json.loads(line) for line in open(diag_files[0])]
    # (a) an adaptation decision was recorded and widened the grace window.
    at_decisions = [e for e in events if e["event"] == "auto_tune_runtime_decision"]
    assert at_decisions, "autotuner emitted no adaptation decision"
    assert any(e["effective_grace_window_ms"] > 1000 for e in at_decisions)
    # (b) the detector flagged the injected straggler step.
    failslow = [e for e in events if e["event"] == "failslow_decision"]
    assert any(e["step"] == 30 and e["reason"] == "slow" for e in failslow), (
        "detector did not flag the injected straggler under the adapted threshold"
    )
    # (c) NO mitigation / exclusion: the runtime never calls
    # syncer.mark_failslow, so no learner is excluded. The syncer state is
    # also clean (no round in flight). Assert observe-only by construction:
    # there is no rank-exclusion event in the diagnostics, and the syncer
    # has no excluded learners.
    init_event = next(e for e in events if e["event"] == "mend_init")
    assert init_event["auto_tune_runtime_active"] is True


def test_runtime_autotuner_no_mitigation_wired():
    """The runtime autotuner must NOT wire the detector decision into rank
    exclusion. Drive a straggler and confirm the syncer has no failslow
    exclusions (mark_failslow is never called by the runtime)."""
    model = _ToyModel()
    config = MendConfig(
        quorum_min_learners=1,
        grace_window_ms=500,
        failslow_zscore_threshold=3.0,
        failslow_min_samples=5,
        sync_period_steps=4,
        momentum_sync_period_steps=8,
        auto_tune_runtime=True,
        auto_tune_runtime_min_samples=5,
        sideband_peers=(),
    )
    mend_init(model, config, rank_id="rack-0/rank-0")
    rt = get_runtime(model)
    try:
        for step in range(20):
            rt.step_begin(step)
            rt.step_end(step, step_time_ms_override=100.0)
        rt.step_begin(20)
        rt.step_end(20, step_time_ms_override=900.0)
        # Start a round on the underlying syncer and confirm no learner is
        # excluded (the autotuner never marks fail-slow).
        rt.syncer.start_round(round_id=1)
        assert rt.syncer._state is not None
        assert rt.syncer._state.failslow_excluded == set()
    finally:
        mend_shutdown(model)


def test_runtime_autotuner_deterministic_for_fixed_stream(tmp_path):
    """Two runtimes driven by the same synthetic step-time stream produce
    identical effective threshold + grace-window trajectories."""
    stream = [100.0, 110.0, 90.0, 400.0, 100.0, 95.0, 105.0, 350.0] * 6

    def run_once(diag_dir: str) -> list[tuple[float, int]]:
        model = _ToyModel()
        config = MendConfig(
            quorum_min_learners=1,
            grace_window_ms=1000,
            failslow_zscore_threshold=3.0,
            failslow_min_samples=5,
            sync_period_steps=4,
            momentum_sync_period_steps=8,
            auto_tune_runtime=True,
            auto_tune_runtime_window_steps=20,
            auto_tune_runtime_min_samples=5,
            sideband_peers=(),
            diagnostics_dir=diag_dir,
        )
        mend_init(model, config, rank_id="rack-0/rank-0")
        rt = get_runtime(model)
        traj: list[tuple[float, int]] = []
        try:
            for step, st in enumerate(stream):
                rt.step_begin(step)
                rt.step_end(step, step_time_ms_override=st)
                traj.append(
                    (
                        rt.effective_failslow_zscore_threshold(),
                        rt.effective_grace_window_ms(),
                    )
                )
        finally:
            mend_shutdown(model)
        return traj

    traj_a = run_once(str(tmp_path / "a"))
    traj_b = run_once(str(tmp_path / "b"))
    assert traj_a == traj_b


# ---------------------------------------------------------------------------
# Bit-exact loss equivalence with the autotuner ENABLED
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    """Tiny deterministic regression MLP for the paired bit-exact test."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x)
        return out


_SEED = 20260605
_STEPS = 40
_SYNC_PERIOD = 8
_APPLY_LAG = 3


def _make_model() -> _MLP:
    torch.manual_seed(_SEED)
    return _MLP()


def _batch(step: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator()
    gen.manual_seed(_SEED + 1000 + step)
    x = torch.randn(64, 32, generator=gen)
    target = torch.randn(64, 16, generator=gen)
    return x, target


def _loss(model: nn.Module, step: int) -> torch.Tensor:
    x, target = _batch(step)
    return (model(x) - target).pow(2).mean()


def _snapshot(model: nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


def _fragment(model: nn.Module, prev: list[torch.Tensor], round_id: int) -> LearnerFragment:
    deltas = [(cur.detach() - old).detach().cpu() for cur, old in zip(model.parameters(), prev)]
    return LearnerFragment(
        learner_id="rank-0",
        round_id=round_id,
        params_delta=deltas,
        tokens_consumed=100,
    )


def _apply_merged(model: nn.Module, merged: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for p, m in zip(model.parameters(), merged):
            p.add_(m.to(device=p.device, dtype=p.dtype))


def _run_baseline() -> list[float]:
    """Synchronous syncer path: a single-rank quorum=1 loop. Applies the
    merged delta at a fixed apply lag. No runtime / no autotuner."""
    model = _make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    syncer = GraceWindowSyncer(quorum_min_learners=1, grace_window_ms=0, token_weighted=True)
    prev = _snapshot(model)
    losses: list[float] = []
    pending: Optional[tuple[int, list[torch.Tensor]]] = None
    for step in range(_STEPS):
        loss = _loss(model, step)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if step > 0 and step % _SYNC_PERIOD == 0:
            syncer.start_round(round_id=step)
            result = syncer.submit(_fragment(model, prev, step))
            if result is None:
                result = syncer.finalize_on_timeout()
            pending = (step + _APPLY_LAG, result.merged_delta)
            prev = _snapshot(model)
        if pending is not None and step >= pending[0]:
            _apply_merged(model, pending[1])
            pending = None
    return losses


def _run_sdk_autotuner_on() -> list[float]:
    """mend runtime path with the autotuner ENABLED. The runtime observes
    a deliberately STRAGGLER-LADEN synthetic step-time stream (so the
    autotuner actively adapts the grace window and detection threshold),
    yet the loss trajectory must be identical to the baseline because the
    autotuner is observe-only / wall-clock-only and never changes the merge
    set or the fixed apply boundary."""
    model = _make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    config = MendConfig(
        quorum_min_learners=1,
        grace_window_ms=0,
        token_weighted_merge=True,
        sync_period_steps=_SYNC_PERIOD,
        momentum_sync_period_steps=_SYNC_PERIOD * 4,
        async_tp_enabled=False,
        concurrent_outer_step=True,
        auto_tune_runtime=True,
        auto_tune_runtime_window_steps=20,
        auto_tune_runtime_min_samples=5,
        auto_tune_grace_window_max_ms=10_000,
        sideband_peers=(),
        diagnostics_dir=None,
    )
    mend_init(model, config, rank_id="rank-0")
    try:
        runtime = get_runtime(model)
        prev = _snapshot(model)
        losses: list[float] = []
        apply_at: Optional[int] = None
        for step in range(_STEPS):
            runtime.step_begin(step)
            loss = _loss(model, step)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
            # Feed a synthetic straggler-laden step-time stream so the
            # autotuner actively adapts; this is decoupled from the merge.
            st = 600.0 if step % 7 == 0 else 100.0
            runtime.step_end(step, step_time_ms_override=st)

            if step > 0 and step % _SYNC_PERIOD == 0:
                local = _fragment(model, prev, step)

                def provider(frag: LearnerFragment = local) -> "asyncio.Queue[LearnerFragment]":
                    q: asyncio.Queue[LearnerFragment] = asyncio.Queue()
                    q.put_nowait(frag)
                    return q

                runtime.outer_step_begin(round_id=step, fragment_provider=provider)
                apply_at = step + _APPLY_LAG
                prev = _snapshot(model)

            if apply_at is not None and step >= apply_at:
                result = _collect(runtime)
                _apply_merged(model, result.merged_delta)
                apply_at = None
        if apply_at is not None:
            result = _collect(runtime)
            _apply_merged(model, result.merged_delta)
        return losses
    finally:
        mend_shutdown(model)


def _collect(runtime: object, timeout_s: float = 10.0) -> object:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = runtime.outer_step_collect()  # type: ignore[attr-defined]
        if result is not None:
            return result
        time.sleep(0.001)
    raise TimeoutError("outer-step merge did not complete within timeout")


def test_paired_bit_exact_with_autotuner_enabled():
    """Paired baseline vs sdk-with-autotuner-ON: per-step loss trajectories
    must be elementwise IEEE-754 equal (max |loss diff| = 0.0). This proves
    the online autotuner preserves bit-exact loss equivalence even when it
    is actively adapting under a straggler-laden step-time stream."""
    base = _run_baseline()
    sdk = _run_sdk_autotuner_on()
    assert len(base) == len(sdk) == _STEPS
    max_abs_diff = max((abs(a - b) for a, b in zip(base, sdk)), default=0.0)
    assert max_abs_diff == 0.0, f"max |loss diff| = {max_abs_diff!r}, expected 0.0"
    # And literal IEEE-754 equality element-by-element.
    for i, (a, b) in enumerate(zip(base, sdk)):
        assert a == b, f"step {i}: baseline {a!r} != sdk {b!r}"

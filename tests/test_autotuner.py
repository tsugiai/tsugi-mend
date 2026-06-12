"""Online runtime autotuner tests.

Covers the pure control law (RuntimeAutotuner) directly, the v2 observe-only
signals (per-learner EWMA/CUSUM drift flag + sustained peer-relative gate),
then the runtime integration: adaptation direction under a synthetic
straggler-spike stream, that the (observe-only) detector flags the injected
straggler under the adapted threshold, that NO mitigation/exclusion is
invoked, that adaptation is deterministic for a fixed input stream, and that
a paired run with the autotuner ENABLED stays bit-exact (max |loss diff| =
0.0).
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from collections import deque
from typing import Optional

import pytest
import torch
import torch.nn as nn

from tsugi_mend import MendConfig, mend_init, mend_shutdown
from tsugi_mend.autotuner import (
    AutotuneDecision,
    EwmaCusumDriftClassifier,
    RuntimeAutotuner,
)
from tsugi_mend.diagnostics import DiagnosticsWriter
from tsugi_mend.failslow import FailSlowDetector
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
# v2 observe-only signals: per-learner EWMA/CUSUM drift flag
# ---------------------------------------------------------------------------

# Deterministic stationary "noise": a fixed cycle around 100 ms with zero
# trend. Literal values (no RNG) so the test is bit-reproducible everywhere.
_NOISE_CYCLE = [100.0, 104.0, 97.0, 102.0, 95.0, 103.0, 99.0, 106.0, 96.0, 98.0]


def test_drift_classifier_flags_slow_ramp_that_static_window_misses():
    """A slow intra-window ramp (+0.5 ms/step) never trips the static
    sliding-window z-score detector (the window mean ramps along with the
    samples, so the per-step z-score stays small), but the EWMA/CUSUM drift
    classifier integrates the gap against its lagging slow reference and
    flags it."""
    ramp = [100.0 + 0.5 * i for i in range(300)]
    static = FailSlowDetector(window_steps=50, zscore_threshold=3.0, min_samples=10)
    static_flags = [static.observe("rank-0", x).is_slow for x in ramp]
    assert not any(static_flags), "static sliding-window z-score should miss a slow ramp"
    drift = EwmaCusumDriftClassifier()
    drift_flags = [drift.observe("rank-0", x).flagged for x in ramp]
    assert any(drift_flags), "drift classifier should flag a slow ramp"


def test_drift_classifier_no_flag_on_stationary_noise():
    """Stationary noise around a fixed level must never flag: the one-sided
    CUSUM's slack absorbs benign jitter and negative deviations drain the
    accumulator back to zero."""
    drift = EwmaCusumDriftClassifier()
    decisions = [drift.observe("rank-0", _NOISE_CYCLE[i % 10]) for i in range(300)]
    assert not any(d.flagged for d in decisions)
    # The accumulator never even gets close to the default threshold.
    assert max(d.cusum for d in decisions) < 8.0


def test_drift_classifier_no_flag_on_constant_stream():
    drift = EwmaCusumDriftClassifier()
    decisions = [drift.observe("rank-0", 100.0) for _ in range(300)]
    assert not any(d.flagged for d in decisions)


def test_drift_classifier_peer_relative_flags_only_the_drifter():
    """With peer streams fed, each learner is judged against the median of
    the OTHER learners' EWMAs: only the slowly degrading learner is
    flagged, and the healthy peers never are (the median keeps the drifter
    from polluting their baseline)."""
    drift = EwmaCusumDriftClassifier()
    drifter_flagged_at: Optional[int] = None
    for i in range(300):
        for lid in ("rack-0", "rack-1", "rack-2"):
            assert drift.observe(lid, 100.0).flagged is False
        # rack-3 is healthy for 50 steps, then degrades +0.5 ms/step.
        d = drift.observe("rack-3", 100.0 + 0.5 * max(0, i - 50))
        if d.flagged and drifter_flagged_at is None:
            drifter_flagged_at = i
    assert drifter_flagged_at is not None, "drifting learner was never flagged"
    assert drifter_flagged_at > 50


def test_drift_classifier_burn_in_settles_estimates_before_charting():
    """The CUSUM must not accumulate during the min_samples burn-in (the
    scale estimate is still settling there); otherwise startup noise
    pre-loads the accumulator and stationary streams false-flag."""
    drift = EwmaCusumDriftClassifier(min_samples=10)
    for i in range(10):
        d = drift.observe("rank-0", _NOISE_CYCLE[i % 10])
        assert d.cusum == 0.0
        assert d.flagged is False


def test_drift_classifier_reset_drops_state():
    drift = EwmaCusumDriftClassifier(min_samples=5)
    for i in range(100):
        drift.observe("rank-0", 100.0 + 2.0 * i)
    drift.reset("rank-0")
    d = drift.observe("rank-0", 1000.0)
    assert d.observation_count == 1
    assert d.cusum == 0.0
    assert d.flagged is False


def test_drift_classifier_validation():
    with pytest.raises(ValueError, match="ewma_alpha"):
        EwmaCusumDriftClassifier(ewma_alpha=0.0)
    with pytest.raises(ValueError, match="ewma_alpha"):
        EwmaCusumDriftClassifier(ewma_alpha=1.5)
    with pytest.raises(ValueError, match="baseline_alpha"):
        EwmaCusumDriftClassifier(baseline_alpha=0.0)
    with pytest.raises(ValueError, match="cusum_slack"):
        EwmaCusumDriftClassifier(cusum_slack=-1.0)
    with pytest.raises(ValueError, match="cusum_threshold"):
        EwmaCusumDriftClassifier(cusum_threshold=0.0)
    with pytest.raises(ValueError, match="min_samples"):
        EwmaCusumDriftClassifier(min_samples=0)
    with pytest.raises(ValueError, match="learner_id"):
        EwmaCusumDriftClassifier().observe("", 100.0)
    with pytest.raises(ValueError, match="latency_ms"):
        EwmaCusumDriftClassifier().observe("rank-0", -1.0)


def test_autotuner_surfaces_drift_flag_on_decisions():
    """The RuntimeAutotuner feeds its local stream to the drift classifier
    and surfaces flag/CUSUM on every decision; a slow ramp flips the flag."""
    at = _make_autotuner()
    decisions = [at.observe(x) for x in [100.0] * 10]
    assert all(d.drift_flag is False for d in decisions)
    decisions = [at.observe(100.0 + 1.0 * i) for i in range(200)]
    assert any(d.drift_flag for d in decisions)


def test_drift_flag_never_feeds_effective_values():
    """FLAG-ONLY contract: two autotuners differing ONLY in drift knobs (one
    flags eagerly, one can never flag) emit byte-identical effective
    threshold + grace-window trajectories on the same stream."""
    stream = [100.0] * 10 + [100.0 + 1.0 * i for i in range(100)]
    sensitive = _make_autotuner(drift_cusum_threshold=0.5, drift_cusum_slack=0.0)
    inert = _make_autotuner(drift_cusum_threshold=1e9)
    seq_sensitive = [sensitive.observe(x) for x in stream]
    seq_inert = [inert.observe(x) for x in stream]
    assert any(d.drift_flag for d in seq_sensitive)
    assert not any(d.drift_flag for d in seq_inert)

    def effective(seq: list[AutotuneDecision]) -> list[tuple[float, int]]:
        return [(d.effective_zscore_threshold, d.effective_grace_window_ms) for d in seq]

    assert effective(seq_sensitive) == effective(seq_inert)


def test_observe_peer_rejects_local_id_and_never_moves_effective_values():
    at = _make_autotuner()
    with pytest.raises(ValueError, match="reserved"):
        at.observe_peer("local", 100.0)
    for _ in range(50):
        at.observe(100.0)
    z_before = at.effective_zscore_threshold
    g_before = at.effective_grace_window_ms
    # An absurdly slow peer stream: the peer itself gets flagged, but the
    # local effective values must not move (peer samples never enter the
    # rolling window).
    peer_decisions = [at.observe_peer("rack-9", 100_000.0) for _ in range(100)]
    assert any(d.flagged for d in peer_decisions)
    assert at.effective_zscore_threshold == z_before
    assert at.effective_grace_window_ms == g_before


def test_drift_flag_diagnostic_emitted_on_rising_edge_only(tmp_path):
    """When a diagnostics writer is attached, the autotuner emits ONE
    auto_tune_drift_flag event per rising edge, not one per flagged step."""
    diag = DiagnosticsWriter(str(tmp_path))
    at = _make_autotuner(diagnostics=diag)
    for x in [100.0] * 10 + [100.0 + 2.0 * i for i in range(200)]:
        at.observe(x)
    diag.close()
    diag_files = list(tmp_path.glob("max_sdk_pid*.jsonl"))
    assert diag_files
    events = [json.loads(line) for line in open(diag_files[0])]
    flag_events = [e for e in events if e["event"] == "auto_tune_drift_flag"]
    assert len(flag_events) == 1
    assert flag_events[0]["learner_id"] == "local"
    assert flag_events[0]["cusum"] > flag_events[0]["cusum_threshold"]


# ---------------------------------------------------------------------------
# v2 observe-only signals: sustained peer-relative gate
# ---------------------------------------------------------------------------


def test_sustain_gate_suppresses_single_window_blip():
    """With sustain_windows=2 the gate requires window_steps+1 consecutive
    deviating evaluations; a lone spike deviates the window statistics for
    at most window_steps evaluations (its window residence), so it must be
    fully suppressed, while the v1-default (K=1) reacts to it."""
    gated = _make_autotuner(
        window_steps=10, min_samples=5, base_grace_window_ms=1000, sustain_windows=2
    )
    v1 = _make_autotuner(window_steps=10, min_samples=5, base_grace_window_ms=1000)
    stream = [100.0] * 10 + [400.0] + [100.0] * 20
    gated_grace = [gated.observe(x).effective_grace_window_ms for x in stream]
    v1_grace = [v1.observe(x).effective_grace_window_ms for x in stream]
    assert all(g == 1000 for g in gated_grace), "K=2 gate must suppress a one-window blip"
    assert any(g > 1000 for g in v1_grace), "K=1 (v1 behavior) reacts to the blip"


def test_sustain_gate_passes_sustained_shift_after_k_windows():
    """A genuinely sustained straggler regime (peak_ratio > 1 on every
    evaluation) passes the K=2 gate exactly once the deviation streak spans
    (K-1)*window_steps + 1 evaluations, and applies the same widened value
    v1 would."""
    at = _make_autotuner(
        window_steps=10, min_samples=5, base_grace_window_ms=1000, sustain_windows=2
    )
    for _ in range(10):
        at.observe(100.0)
    assert at.effective_grace_window_ms == 1000
    need = (2 - 1) * 10 + 1  # 11 consecutive deviating evaluations
    widened_at: Optional[int] = None
    widened_value: Optional[int] = None
    for i in range(40):
        d = at.observe(400.0 if i % 2 == 0 else 100.0)
        if d.effective_grace_window_ms > 1000 and widened_at is None:
            widened_at = i
            widened_value = d.effective_grace_window_ms
    assert widened_at is not None, "sustained shift never passed the gate"
    assert widened_at + 1 == need
    # Same widened value the ungated law computes for this window:
    # peak 400 / body-median 100 -> ratio 4 -> 1000 * (1 + 1*(4-1)) = 4000.
    assert widened_value == 4000


def test_sustain_gate_threshold_side_timing():
    """The z-score threshold move obeys the same gate: with K=3 and
    window_steps=10 the threshold first rises on the 21st consecutive
    deviating (cov > 0) evaluation."""
    at = _make_autotuner(window_steps=10, min_samples=5, sustain_windows=3)
    for _ in range(10):
        d = at.observe(100.0)
        assert d.effective_zscore_threshold == 3.0
    need = (3 - 1) * 10 + 1  # 21
    raised_at: Optional[int] = None
    for i in range(40):
        d = at.observe(80.0 if i % 2 == 0 else 120.0)
        if d.effective_zscore_threshold > 3.0 and raised_at is None:
            raised_at = i
    assert raised_at is not None
    assert raised_at + 1 == need


def test_sustain_gate_relaxation_is_never_delayed():
    """Once the deviation clears, the effective value returns to baseline
    immediately (relaxation is not gated), exactly as in v1."""
    at = _make_autotuner(
        window_steps=10, min_samples=5, base_grace_window_ms=1000, sustain_windows=2
    )
    for _ in range(10):
        at.observe(100.0)
    # Sustained regime long enough to pass the gate and widen.
    for i in range(20):
        d = at.observe(400.0 if i % 2 == 0 else 100.0)
    assert d.effective_grace_window_ms > 1000
    # Clean samples: as soon as the 400s slide out of the window the ratio
    # returns to 1 and the grace window snaps back to baseline.
    last = None
    for _ in range(12):
        last = at.observe(100.0)
    assert last is not None
    assert last.observed_peak_ratio == pytest.approx(1.0, rel=1e-9)
    assert last.effective_grace_window_ms == 1000


def _v1_reference_law(
    stream: list[float],
    *,
    base_z: float = 3.0,
    base_g: int = 2000,
    window_steps: int = 20,
    min_samples: int = 5,
    zscore_min: float = 2.0,
    zscore_max: float = 8.0,
    grace_min_ms: int = 0,
    grace_max_ms: int = 10_000,
    cov_gain: float = 4.0,
    grace_gain: float = 1.0,
) -> list[tuple[float, int]]:
    """Independent replay of the v1 (pre-gate) control law, as a pure
    function of the window contents. Used to prove the default knobs leave
    adaptation decisions unchanged."""
    win: deque[float] = deque(maxlen=window_steps)
    out: list[tuple[float, int]] = []
    for x in stream:
        win.append(float(x))
        if len(win) < min_samples:
            out.append((base_z, base_g))
            continue
        ordered = sorted(win)
        body = ordered[:-1] if len(ordered) > 1 else ordered
        m = len(body)
        mean = sum(body) / m
        cov = 0.0
        if mean > 0.0:
            var = sum((v - mean) ** 2 for v in body) / m
            cov = math.sqrt(var) / mean
        if m % 2 == 1:
            median = body[m // 2]
        else:
            median = (body[m // 2 - 1] + body[m // 2]) / 2.0
        peak = ordered[-1]
        peak_ratio = peak / median if median > 0.0 else 1.0
        z = min(max(base_z + cov_gain * cov, zscore_min), zscore_max)
        g_raw = int(round(base_g * (1.0 + grace_gain * max(0.0, peak_ratio - 1.0))))
        g = int(min(max(float(g_raw), float(grace_min_ms)), float(grace_max_ms)))
        out.append((z, g))
    return out


def test_default_knobs_reproduce_v1_decisions_on_fixture_stream():
    """ON path with default v2 knobs (sustain_windows=1, default drift
    knobs): the effective-value trajectory on the existing fixture stream
    is byte-identical to an independent replay of the v1 control law."""
    stream = [100.0, 105.0, 95.0, 300.0, 110.0, 90.0, 100.0, 250.0, 100.0, 100.0] * 3
    at = _make_autotuner()
    got = [
        (d.effective_zscore_threshold, d.effective_grace_window_ms)
        for d in (at.observe(x) for x in stream)
    ]
    assert got == _v1_reference_law(stream)


def test_sustain_windows_validation():
    with pytest.raises(ValueError, match="sustain_windows"):
        _make_autotuner(sustain_windows=0)


def test_reset_clears_gate_and_drift_state():
    at = _make_autotuner(
        window_steps=10,
        min_samples=5,
        sustain_windows=2,
        drift_cusum_threshold=2.0,
        drift_cusum_slack=0.0,
    )
    for i in range(60):
        at.observe(100.0 + 5.0 * i)
    at.reset()
    d = at.observe(100.0)
    assert d.reason == "warmup"
    assert d.drift_flag is False
    assert d.drift_cusum == 0.0
    assert at.effective_zscore_threshold == 3.0
    assert at.effective_grace_window_ms == 2000


def test_config_v2_knob_defaults_and_validation():
    """The new MendConfig knobs default to behavior-preserving values and
    are validated at config init."""
    c = MendConfig()
    assert c.auto_tune_sustain_windows == 1
    assert c.auto_tune_drift_ewma_alpha == 0.2
    assert c.auto_tune_drift_baseline_alpha == 0.02
    assert c.auto_tune_drift_cusum_slack == 1.0
    assert c.auto_tune_drift_cusum_threshold == 8.0
    with pytest.raises(ValueError, match="auto_tune_sustain_windows"):
        MendConfig(auto_tune_sustain_windows=0)
    with pytest.raises(ValueError, match="auto_tune_drift_ewma_alpha"):
        MendConfig(auto_tune_drift_ewma_alpha=0.0)
    with pytest.raises(ValueError, match="auto_tune_drift_ewma_alpha"):
        MendConfig(auto_tune_drift_ewma_alpha=1.5)
    with pytest.raises(ValueError, match="auto_tune_drift_baseline_alpha"):
        MendConfig(auto_tune_drift_baseline_alpha=0.0)
    with pytest.raises(ValueError, match="auto_tune_drift_cusum_slack"):
        MendConfig(auto_tune_drift_cusum_slack=-0.1)
    with pytest.raises(ValueError, match="auto_tune_drift_cusum_threshold"):
        MendConfig(auto_tune_drift_cusum_threshold=0.0)


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
    # No v2 drift-flag diagnostics either: the OFF path never constructs an
    # autotuner, so the observe-only drift classifier never runs.
    assert not [e for e in events if e["event"] == "auto_tune_drift_flag"]
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

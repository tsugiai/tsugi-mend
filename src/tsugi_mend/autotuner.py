"""Online runtime autotuner: fail-slow detection sensitivity + grace window.

This module implements a small, deterministic control law that adapts two
operational knobs from the runtime's own observed per-rank step-time stream,
continuously (every step past a warmup window), not one-shot:

1. **Fail-slow detection sensitivity** -- the effective z-score threshold the
   `failslow.FailSlowDetector` uses. The control law tracks the live
   coefficient of variation (CoV) of recent step times and moves the
   threshold so the detector follows the cluster's actual jitter: a jittery
   cluster gets a higher threshold (fewer false straggler flags under benign
   jitter), a clean cluster gets a lower threshold toward a floor (a real
   fail-slow is still caught). This is the lightweight online
   performance-monitoring idea from Guard (arXiv:2605.17879).

2. **Grace-window wait** -- the effective wall-clock `grace_window_ms` the
   syncer waits after quorum. The control law widens the wait when a
   sustained straggler is observed (recent peak/median step-time ratio high)
   so the syncer can absorb a real laggard, and narrows it back toward the
   static baseline when the cluster is clean. This is the operational
   recovery-wait heuristic from "From Detection to Recovery"
   (arXiv:2605.09370).

3. **Per-learner EWMA/CUSUM drift flag (v2, observe-only)** -- an
   exponentially weighted moving average of each learner's step latency plus
   a one-sided CUSUM accumulator against the peer baseline (the median of
   the OTHER learners' EWMAs when peer streams are fed; the learner's own
   slow-EWMA reference when only a single local stream is observed). When
   the CUSUM crosses its threshold the learner is FLAGGED in the
   decision/diagnostic stream -- flag only, never an exclusion, never a
   cadence change. The design pattern is classical statistical process
   control (EWMA and one-sided CUSUM control charts); it catches a slow
   intra-window drift that a static sliding-window z-score misses, with
   O(1) state per learner and O(1) time per observation (the peer-baseline
   median costs O(P) over a fixed set of P learners).

4. **Sustained peer-relative gate (v2)** -- the adaptation moves in (1) and
   (2) only apply after their triggering condition (a deviation of the
   rolling window's peer-sample distribution: CoV > 0 for the sensitivity
   move, peak/median > 1 for the grace move) has held for
   ``(sustain_windows - 1) * window_steps + 1`` consecutive window
   evaluations, i.e. for ``sustain_windows`` consecutive windows' worth of
   evaluations. This adopts the detection half of the multi-window
   sustained-deviation pattern from Guard (arXiv:2605.17879); its
   mitigation half (job restart) is deliberately NOT adopted. A single
   anomalous sample can deviate the rolling-window statistics for at most
   ``window_steps`` consecutive evaluations (its window residence), so any
   ``sustain_windows >= 2`` provably suppresses single-sample blips while a
   genuinely sustained shift still passes the gate. The default
   ``sustain_windows = 1`` applies moves immediately and reproduces the v1
   control law byte-for-byte.

Numerical-safety contract (why this is bit-exact-safe by construction):

- The detection threshold is OBSERVE-ONLY. The detector's decision is a
  diagnostic flag; no mitigation/exclusion is wired off it here. Changing the
  threshold changes only which steps are FLAGGED, never any tensor value.
- The grace window is a WALL-CLOCK WAIT. In default (lossless) mode the
  syncer waits for the same fragments regardless of the wait length, and the
  merged delta is the token-weighted merge of the same fragment set applied
  at the same logical boundary. Changing the wait changes timing/overlap,
  never WHICH fragments merge or the apply boundary.
- The drift flag is FLAG-ONLY. It is reported on the decision (and, when a
  diagnostics writer is attached, as an ``auto_tune_drift_flag`` event); it
  feeds neither the effective threshold, nor the grace window, nor any
  exclusion, cadence, or tensor path.
- The sustain gate only changes WHEN the (already observe-only /
  wall-clock-only) effective values move; it introduces no new actuation
  path.
- This module deliberately does NOT touch the merge cadence
  (sync_period_steps / momentum cadence / apply lag). Adapting the merge
  cadence from measured step times would make a paired baseline vs sdk run
  choose different cadences (their step times differ because the sdk overlaps
  the merge) and would break bit-exact loss equivalence. The cadence is out
  of scope here on purpose.

The control law is pure Python (stdlib only), holds a bounded deque of recent
samples plus O(1) per-learner drift state, and is fully deterministic for a
fixed input stream so a run's adaptation is reproducible. It performs no I/O
of its own (the optional diagnostics writer appends JSONL lines only).
"""
from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from tsugi_mend.diagnostics import DiagnosticsWriter

# Defaults for the v2 observe-only signals. `config.MendConfig` imports these
# so the config surface and the control law cannot drift apart (the same
# convention the sideband defaults use).
DEFAULT_SUSTAIN_WINDOWS = 1
DEFAULT_DRIFT_EWMA_ALPHA = 0.2
DEFAULT_DRIFT_BASELINE_ALPHA = 0.02
DEFAULT_DRIFT_CUSUM_SLACK = 1.0
DEFAULT_DRIFT_CUSUM_THRESHOLD = 8.0
DEFAULT_DRIFT_MIN_SAMPLES = 10

# Learner id the RuntimeAutotuner uses for its own (local) step-time stream
# inside the drift classifier. Reserved: `observe_peer` rejects it.
_LOCAL_LEARNER_ID = "local"

# Robust-scale floor used by the CUSUM standardization, as an absolute
# epsilon plus a small fraction of the baseline level. Prevents division by
# zero on perfectly flat streams while keeping the floor proportionate to
# the latency scale.
_SCALE_FLOOR_ABS = 1e-9
_SCALE_FLOOR_REL = 1e-3


@dataclass(frozen=True)
class DriftDecision:
    """One per-learner drift-classifier decision (observe-only).

    Attributes:
        learner_id: the learner this observation belongs to.
        flagged: True when the one-sided CUSUM is above its threshold. The
            CUSUM only starts accumulating after a ``min_samples`` burn-in
            during which the level/scale estimates settle (classical SPC
            practice: estimate the in-control parameters first, then start
            the chart), so the flag cannot fire during burn-in. Flag only:
            nothing in this SDK feeds it into exclusion, cadence, or
            tensors.
        cusum: current value of the one-sided CUSUM accumulator (in robust
            standard-deviation units).
        ewma_ms: the learner's fast EWMA of step latency after this
            observation.
        baseline_ms: the peer baseline this observation was compared
            against (median of the other learners' EWMAs when peers exist,
            otherwise this learner's own slow EWMA reference).
        scale_ms: the robust scale (EW mean absolute deviation, floored)
            used to standardize this observation.
        observation_count: observations seen for this learner so far.
    """

    learner_id: str
    flagged: bool
    cusum: float
    ewma_ms: float
    baseline_ms: float
    scale_ms: float
    observation_count: int


@dataclass
class _LearnerDriftState:
    """O(1) per-learner drift state: two EWMAs, a scale, a CUSUM, a count."""

    ewma_ms: float
    slow_ewma_ms: float
    scale_ms: float
    cusum: float
    count: int


class EwmaCusumDriftClassifier:
    """Per-learner EWMA/CUSUM drift classifier (observe-only, flag-only).

    Classical statistical process control (EWMA and one-sided CUSUM control
    charts) applied to per-learner step latency:

    - A fast EWMA ``m_i`` summarizes each learner's current latency level.
    - A slow EWMA tracks the learner's own long-horizon reference level.
    - A robust scale ``s_i`` (EW mean absolute deviation of the one-step
      prediction residual) standardizes deviations.
    - A one-sided CUSUM accumulates standardized exceedances of the peer
      baseline beyond a slack allowance::

          C_i = max(0, C_i + (x - baseline) / s_i - cusum_slack)

      and the learner is FLAGGED while ``C_i > cusum_threshold``. The
      first ``min_samples`` observations are a burn-in that only settles
      the level/scale estimates; the CUSUM does not accumulate until the
      burn-in completes (classical SPC practice: estimate the in-control
      parameters first, then start the chart), so a freshly seeded scale
      cannot convert startup noise into decision pressure.

    Peer baseline: when observations from multiple learners are fed, the
    baseline for learner ``i`` is the median of the OTHER learners' fast
    EWMAs (median keeps a drifting learner from polluting its peers'
    baselines). With a single observed stream the baseline falls back to
    the learner's own slow EWMA, which lags a slow ramp enough for the
    CUSUM to integrate the gap -- this is what catches a drift that a
    static sliding-window z-score misses (the window mean ramps along with
    the samples, so the per-step z-score never trips). Note the
    single-stream fallback detects drift EPISODES (the slow reference
    eventually absorbs a completed level shift); the peer-relative form
    detects persistent deviation, because healthy peers do not absorb it.

    Complexity: O(1) state per learner; O(1) time per observation, plus an
    O(P) median over the fixed set of P learners when peers exist.

    FLAG-ONLY contract: decisions are diagnostics. This class is not wired
    to (and must never be wired to) learner exclusion, merge cadence, or
    any tensor path.
    """

    def __init__(
        self,
        *,
        ewma_alpha: float = DEFAULT_DRIFT_EWMA_ALPHA,
        baseline_alpha: float = DEFAULT_DRIFT_BASELINE_ALPHA,
        cusum_slack: float = DEFAULT_DRIFT_CUSUM_SLACK,
        cusum_threshold: float = DEFAULT_DRIFT_CUSUM_THRESHOLD,
        min_samples: int = DEFAULT_DRIFT_MIN_SAMPLES,
    ) -> None:
        if not (0.0 < ewma_alpha <= 1.0):
            raise ValueError(f"ewma_alpha must be in (0, 1]; got {ewma_alpha}")
        if not (0.0 < baseline_alpha <= 1.0):
            raise ValueError(f"baseline_alpha must be in (0, 1]; got {baseline_alpha}")
        if cusum_slack < 0:
            raise ValueError(f"cusum_slack must be >= 0; got {cusum_slack}")
        if cusum_threshold <= 0:
            raise ValueError(f"cusum_threshold must be > 0; got {cusum_threshold}")
        if min_samples < 1:
            raise ValueError(f"min_samples must be >= 1; got {min_samples}")
        self.ewma_alpha = float(ewma_alpha)
        self.baseline_alpha = float(baseline_alpha)
        self.cusum_slack = float(cusum_slack)
        self.cusum_threshold = float(cusum_threshold)
        self.min_samples = int(min_samples)
        self._states: dict[str, _LearnerDriftState] = {}

    def observe(self, learner_id: str, latency_ms: float) -> DriftDecision:
        """Feed one step latency for one learner; return the flag decision."""
        if not learner_id:
            raise ValueError("learner_id must be a non-empty string")
        if latency_ms < 0:
            raise ValueError(f"latency_ms must be >= 0; got {latency_ms}")
        x = float(latency_ms)
        state = self._states.get(learner_id)
        if state is None:
            # Seed both EWMAs at the first sample; no decision pressure yet.
            self._states[learner_id] = _LearnerDriftState(
                ewma_ms=x, slow_ewma_ms=x, scale_ms=0.0, cusum=0.0, count=1
            )
            return DriftDecision(
                learner_id=learner_id,
                flagged=False,
                cusum=0.0,
                ewma_ms=x,
                baseline_ms=x,
                scale_ms=0.0,
                observation_count=1,
            )

        # Baseline is computed BEFORE this sample updates any estimate, so
        # the sample is judged against the pre-existing reference.
        peer_baseline = self._peer_baseline(learner_id)
        baseline = peer_baseline if peer_baseline is not None else state.slow_ewma_ms
        scale = max(state.scale_ms, _SCALE_FLOOR_ABS + _SCALE_FLOOR_REL * abs(baseline))
        state.count += 1
        # Burn-in: the first min_samples observations only settle the
        # level/scale estimates; the chart starts accumulating afterwards.
        # (A freshly seeded scale sits at the floor, which would otherwise
        # turn ordinary startup noise into a huge standardized deviation
        # and falsely pre-load the accumulator.)
        if state.count > self.min_samples:
            deviation = (x - baseline) / scale
            state.cusum = max(0.0, state.cusum + deviation - self.cusum_slack)
        flagged = state.cusum > self.cusum_threshold

        # Update the level/scale estimates AFTER the decision. The scale
        # tracks the one-step prediction residual against the PRE-update
        # fast EWMA (standard EW-MAD recursion).
        residual = abs(x - state.ewma_ms)
        state.scale_ms += self.ewma_alpha * (residual - state.scale_ms)
        state.ewma_ms += self.ewma_alpha * (x - state.ewma_ms)
        state.slow_ewma_ms += self.baseline_alpha * (x - state.slow_ewma_ms)

        return DriftDecision(
            learner_id=learner_id,
            flagged=flagged,
            cusum=state.cusum,
            ewma_ms=state.ewma_ms,
            baseline_ms=baseline,
            scale_ms=scale,
            observation_count=state.count,
        )

    def _peer_baseline(self, learner_id: str) -> Optional[float]:
        """Median of the OTHER learners' fast EWMAs; None when no peers."""
        peers = [s.ewma_ms for lid, s in self._states.items() if lid != learner_id]
        if not peers:
            return None
        return float(statistics.median(peers))

    def reset(self, learner_id: str) -> None:
        """Drop one learner's drift state (e.g. after a confirmed node swap)."""
        self._states.pop(learner_id, None)

    def clear(self) -> None:
        """Drop all drift state."""
        self._states.clear()


@dataclass(frozen=True)
class AutotuneDecision:
    """One online autotuner decision, emitted after a step is observed.

    Attributes:
        adapted: True if the autotuner produced an updated effective value
            this step (i.e. it was past warmup with enough samples).
        effective_zscore_threshold: the z-score threshold the detector
            should use going forward (observe-only; numerically inert).
        effective_grace_window_ms: the wall-clock grace window the syncer
            should wait going forward.
        observed_cov: coefficient of variation of the recent step-time
            window (std / mean); 0.0 during warmup or for a flat window.
        observed_peak_ratio: recent max / recent median step time; 1.0 when
            flat or during warmup. The straggler-magnitude signal that
            drives the grace-window widening.
        window_size: number of samples currently in the rolling window.
        reason: "warmup" while the window is below min_samples; otherwise
            "adapted".
        drift_flag: True while the local stream's EWMA/CUSUM drift
            classifier is above its threshold (v2, observe-only). FLAG
            ONLY: never feeds exclusion, cadence, or tensors.
        drift_cusum: current value of the local stream's one-sided CUSUM
            accumulator (robust standard-deviation units).
    """

    adapted: bool
    effective_zscore_threshold: float
    effective_grace_window_ms: int
    observed_cov: float
    observed_peak_ratio: float
    window_size: int
    reason: str
    drift_flag: bool = False
    drift_cusum: float = 0.0


class RuntimeAutotuner:
    """Deterministic online control law for detection sensitivity + grace window.

    The runtime feeds each rank's per-step wall-clock duration (ms) into
    :meth:`observe`, which returns an :class:`AutotuneDecision`. The runtime
    then applies the effective values to the (observe-only) fail-slow
    detector and to the syncer's wall-clock grace window.

    Control law (all bounded, monotone-clamped, deterministic):

    - Detection threshold.  Let ``cov`` be the coefficient of variation of
      the rolling window.  The effective threshold is::

          z_eff = clamp(base_z + cov_gain * cov,
                        zscore_min, zscore_max)

      A higher live CoV (jittery cluster) raises ``z_eff`` so benign jitter
      does not trip a straggler flag; a near-flat window pulls ``z_eff`` down
      toward ``base_z`` (and never below ``zscore_min``), keeping the
      detector sharp enough to catch a real fail-slow.

    - Grace window.  Let ``peak_ratio`` be recent max / recent median.  The
      effective grace window is::

          g_eff = clamp(round(base_grace_ms * (1 + grace_gain * max(0, peak_ratio - 1))),
                        grace_min_ms, grace_max_ms)

      A sustained straggler (peak_ratio well above 1) widens the wall-clock
      wait so the syncer can absorb the laggard; a clean window (peak_ratio
      near 1) returns the wait toward the static baseline.

    - Sustained peer-relative gate (v2).  A deviation move (raising the
      threshold above base, widening the grace window above base) only
      applies once its triggering condition (``cov > 0`` for the threshold,
      ``peak_ratio > 1`` for the grace window -- both deviations of the
      window's peer-sample distribution) has held for
      ``(sustain_windows - 1) * window_steps + 1`` consecutive adapted
      evaluations, i.e. ``sustain_windows`` consecutive windows' worth.
      Until then the corresponding effective value holds at the (clamped)
      static baseline. Relaxation back toward the baseline is never gated:
      the moment the deviation condition clears, the effective value
      returns to baseline exactly as in v1. A single anomalous sample
      deviates the rolling-window statistics for at most ``window_steps``
      consecutive evaluations (its window residence), so any
      ``sustain_windows >= 2`` provably suppresses single-sample blips.
      The default ``sustain_windows = 1`` reproduces the v1 behavior
      byte-for-byte (a deviation applies on its first evaluation, and a
      non-deviating evaluation's raw value already equals the baseline
      hold value).

    - Drift flag (v2, observe-only).  Every observation is also fed to an
      :class:`EwmaCusumDriftClassifier` for the local stream; the resulting
      flag/CUSUM are surfaced on the decision (``drift_flag`` /
      ``drift_cusum``) and, when a diagnostics writer is attached, as an
      ``auto_tune_drift_flag`` JSONL event on the flag's rising edge. The
      flag never feeds the effective values, exclusion, cadence, or any
      tensor. Peer step-latency streams may be fed via
      :meth:`observe_peer` so the local learner is judged against a
      peer-median baseline instead of its own slow-EWMA reference.

    Both laws are pure functions of the current window contents plus the
    gate/drift state, so the sequence of decisions is fully reproducible
    for a fixed input stream.
    """

    def __init__(
        self,
        *,
        base_zscore_threshold: float,
        base_grace_window_ms: int,
        window_steps: int,
        min_samples: int,
        zscore_min: float,
        zscore_max: float,
        grace_min_ms: int,
        grace_max_ms: int,
        cov_gain: float,
        grace_gain: float,
        sustain_windows: int = DEFAULT_SUSTAIN_WINDOWS,
        drift_ewma_alpha: float = DEFAULT_DRIFT_EWMA_ALPHA,
        drift_baseline_alpha: float = DEFAULT_DRIFT_BASELINE_ALPHA,
        drift_cusum_slack: float = DEFAULT_DRIFT_CUSUM_SLACK,
        drift_cusum_threshold: float = DEFAULT_DRIFT_CUSUM_THRESHOLD,
        diagnostics: Optional[DiagnosticsWriter] = None,
    ) -> None:
        if window_steps < 2:
            raise ValueError(f"window_steps must be >= 2; got {window_steps}")
        if min_samples < 2:
            raise ValueError(f"min_samples must be >= 2; got {min_samples}")
        if min_samples > window_steps:
            raise ValueError(
                f"min_samples ({min_samples}) cannot exceed window_steps ({window_steps})"
            )
        if zscore_min <= 0:
            raise ValueError(f"zscore_min must be > 0; got {zscore_min}")
        if zscore_max < zscore_min:
            raise ValueError(
                f"zscore_max ({zscore_max}) must be >= zscore_min ({zscore_min})"
            )
        if grace_min_ms < 0:
            raise ValueError(f"grace_min_ms must be >= 0; got {grace_min_ms}")
        if grace_max_ms < grace_min_ms:
            raise ValueError(
                f"grace_max_ms ({grace_max_ms}) must be >= grace_min_ms ({grace_min_ms})"
            )
        if cov_gain < 0:
            raise ValueError(f"cov_gain must be >= 0; got {cov_gain}")
        if grace_gain < 0:
            raise ValueError(f"grace_gain must be >= 0; got {grace_gain}")
        if sustain_windows < 1:
            raise ValueError(f"sustain_windows must be >= 1; got {sustain_windows}")

        self.base_zscore_threshold = float(base_zscore_threshold)
        self.base_grace_window_ms = int(base_grace_window_ms)
        self.window_steps = window_steps
        self.min_samples = min_samples
        self.zscore_min = float(zscore_min)
        self.zscore_max = float(zscore_max)
        self.grace_min_ms = int(grace_min_ms)
        self.grace_max_ms = int(grace_max_ms)
        self.cov_gain = float(cov_gain)
        self.grace_gain = float(grace_gain)
        self.sustain_windows = int(sustain_windows)

        self._window: Deque[float] = deque(maxlen=window_steps)
        # Effective values seed at the static baseline so that, before the
        # window fills, the runtime behaves exactly as the un-tuned config.
        self._effective_zscore_threshold = self.base_zscore_threshold
        self._effective_grace_window_ms = self.base_grace_window_ms

        # Sustained peer-relative gate state. `need` is the consecutive
        # deviating-evaluation count a move must reach before it applies:
        # sustain_windows=1 -> 1 (apply immediately; v1 behavior),
        # sustain_windows=K -> (K-1)*window_steps + 1 (strictly longer than
        # a single sample's window residence for any K >= 2). The hold
        # values are what the v1 law emits on a deviation-free window, so
        # holding at them is byte-identical to v1 when no deviation is
        # being gated.
        self._sustain_need = (self.sustain_windows - 1) * self.window_steps + 1
        self._zscore_hold = _clamp(
            self.base_zscore_threshold, self.zscore_min, self.zscore_max
        )
        self._grace_hold = int(
            _clamp(
                float(self.base_grace_window_ms),
                float(self.grace_min_ms),
                float(self.grace_max_ms),
            )
        )
        self._zscore_streak = 0
        self._grace_streak = 0

        # Observe-only drift classifier for the local step-time stream
        # (plus any peer streams fed via observe_peer).
        self._drift = EwmaCusumDriftClassifier(
            ewma_alpha=drift_ewma_alpha,
            baseline_alpha=drift_baseline_alpha,
            cusum_slack=drift_cusum_slack,
            cusum_threshold=drift_cusum_threshold,
            min_samples=min_samples,
        )
        self._diagnostics = diagnostics
        self._drift_flag_last: dict[str, bool] = {}
        self._observation_count = 0

    @property
    def effective_zscore_threshold(self) -> float:
        return self._effective_zscore_threshold

    @property
    def effective_grace_window_ms(self) -> int:
        return self._effective_grace_window_ms

    def observe(self, step_time_ms: float) -> AutotuneDecision:
        """Feed one per-step wall-clock duration (ms); return the decision.

        During warmup (window below ``min_samples``) the effective values are
        held at the static baseline and ``adapted`` is False. The drift
        classifier observes every sample (it has its own burn-in) but is
        flag-only either way.
        """
        if step_time_ms < 0:
            raise ValueError(f"step_time_ms must be >= 0; got {step_time_ms}")
        self._observation_count += 1
        drift = self._drift.observe(_LOCAL_LEARNER_ID, float(step_time_ms))
        self._record_drift(drift)
        self._window.append(float(step_time_ms))
        n = len(self._window)
        if n < self.min_samples:
            return AutotuneDecision(
                adapted=False,
                effective_zscore_threshold=self._effective_zscore_threshold,
                effective_grace_window_ms=self._effective_grace_window_ms,
                observed_cov=0.0,
                observed_peak_ratio=1.0,
                window_size=n,
                reason="warmup",
                drift_flag=drift.flagged,
                drift_cusum=drift.cusum,
            )

        cov = self._coefficient_of_variation()
        peak_ratio = self._peak_ratio()

        z_raw = _clamp(
            self.base_zscore_threshold + self.cov_gain * cov,
            self.zscore_min,
            self.zscore_max,
        )
        widen = 1.0 + self.grace_gain * max(0.0, peak_ratio - 1.0)
        g_int = int(round(self.base_grace_window_ms * widen))
        g_raw = int(_clamp(float(g_int), float(self.grace_min_ms), float(self.grace_max_ms)))

        # Sustained peer-relative gate: count consecutive evaluations whose
        # window statistics deviate from the peer-sample distribution; a
        # deviation move applies only once the streak spans
        # `sustain_windows` windows' worth of evaluations. A non-deviating
        # evaluation resets the streak and (by construction) its raw value
        # already equals the baseline hold value, so relaxation toward
        # baseline is never delayed.
        self._zscore_streak = self._zscore_streak + 1 if cov > 0.0 else 0
        self._grace_streak = self._grace_streak + 1 if peak_ratio > 1.0 else 0
        z_eff = z_raw if self._zscore_streak >= self._sustain_need else self._zscore_hold
        g_eff = g_raw if self._grace_streak >= self._sustain_need else self._grace_hold

        self._effective_zscore_threshold = z_eff
        self._effective_grace_window_ms = g_eff
        return AutotuneDecision(
            adapted=True,
            effective_zscore_threshold=z_eff,
            effective_grace_window_ms=g_eff,
            observed_cov=cov,
            observed_peak_ratio=peak_ratio,
            window_size=n,
            reason="adapted",
            drift_flag=drift.flagged,
            drift_cusum=drift.cusum,
        )

    def observe_peer(self, learner_id: str, step_time_ms: float) -> DriftDecision:
        """Feed a PEER learner's step latency into the drift classifier only.

        Observe-only: peer observations never touch the rolling window, the
        effective threshold, or the grace window. They only sharpen the
        drift classifier's peer baseline (the local stream is then judged
        against the median of the peers' EWMAs instead of its own slow-EWMA
        reference) and can flag the peer itself. ``learner_id`` must not be
        the reserved local stream id.
        """
        if learner_id == _LOCAL_LEARNER_ID:
            raise ValueError(
                f"learner_id {_LOCAL_LEARNER_ID!r} is reserved for the local stream; "
                f"feed local samples through observe()"
            )
        drift = self._drift.observe(learner_id, float(step_time_ms))
        self._record_drift(drift)
        return drift

    def _record_drift(self, drift: DriftDecision) -> None:
        """Track the per-learner flag edge; emit the flag-only diagnostic on
        the rising edge (never on every step, to keep the stream readable)."""
        prev = self._drift_flag_last.get(drift.learner_id, False)
        self._drift_flag_last[drift.learner_id] = drift.flagged
        if drift.flagged and not prev and self._diagnostics is not None:
            self._diagnostics.emit(
                "auto_tune_drift_flag",
                learner_id=drift.learner_id,
                observation_index=self._observation_count,
                cusum=drift.cusum,
                cusum_threshold=self._drift.cusum_threshold,
                ewma_ms=drift.ewma_ms,
                baseline_ms=drift.baseline_ms,
            )

    def _coefficient_of_variation(self) -> float:
        # CoV of the BASELINE jitter, computed over the window with the
        # single largest sample removed. Excluding the top sample keeps a
        # lone transient straggler from inflating the steady-state jitter
        # estimate and thereby raising the detection threshold high enough
        # to mask the very spike it should catch. (Guard's online monitoring
        # tracks the cluster's benign jitter, not its outliers.) With
        # n >= min_samples >= 2 this leaves at least one sample.
        ordered = sorted(self._window)
        body = ordered[:-1] if len(ordered) > 1 else ordered
        m = len(body)
        mean = sum(body) / m
        if mean <= 0.0:
            return 0.0
        var = sum((x - mean) ** 2 for x in body) / m
        std = math.sqrt(var)
        return std / mean

    def _peak_ratio(self) -> float:
        ordered = sorted(self._window)
        n = len(ordered)
        # Median that ignores the largest sample so a single sustained
        # straggler does not inflate the "typical" baseline it is compared
        # against. With n >= min_samples >= 2 this leaves at least one
        # sample to take the median over.
        body = ordered[:-1] if n > 1 else ordered
        m = len(body)
        if m % 2 == 1:
            median = body[m // 2]
        else:
            median = (body[m // 2 - 1] + body[m // 2]) / 2.0
        peak = ordered[-1]
        if median <= 0.0:
            return 1.0
        return peak / median

    def reset(self) -> None:
        """Drop the rolling window, the gate streaks, and the drift state,
        and re-seed the effective values at the static baseline. Useful
        after an operator-confirmed node swap."""
        self._window.clear()
        self._effective_zscore_threshold = self.base_zscore_threshold
        self._effective_grace_window_ms = self.base_grace_window_ms
        self._zscore_streak = 0
        self._grace_streak = 0
        self._drift.clear()
        self._drift_flag_last.clear()
        self._observation_count = 0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


__all__ = [
    "AutotuneDecision",
    "DriftDecision",
    "EwmaCusumDriftClassifier",
    "RuntimeAutotuner",
    "DEFAULT_SUSTAIN_WINDOWS",
    "DEFAULT_DRIFT_EWMA_ALPHA",
    "DEFAULT_DRIFT_BASELINE_ALPHA",
    "DEFAULT_DRIFT_CUSUM_SLACK",
    "DEFAULT_DRIFT_CUSUM_THRESHOLD",
    "DEFAULT_DRIFT_MIN_SAMPLES",
]
